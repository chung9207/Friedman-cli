# Local Projection commands: estimate, iv, smooth, state, propensity

function register_lp_commands!()
    lp_estimate = LeafCommand("estimate", _lp_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("control-lags"; type=Int, default=4, description="Number of control lags"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate local projections (Jorda 2005)")

    lp_iv = LeafCommand("iv", _lp_iv;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index"),
            Option("instruments"; type=String, default="", description="Path to instruments CSV"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("control-lags"; type=Int, default=4, description="Number of control lags"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate LP-IV (Stock & Watson 2018)")

    lp_smooth = LeafCommand("smooth", _lp_smooth;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("knots"; type=Int, default=3, description="Number of B-spline knots"),
            Option("lambda"; type=Float64, default=0.0, description="Smoothing penalty (0=auto via CV)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate smooth LP (Barnichon & Brownlees 2019)")

    lp_state = LeafCommand("state", _lp_state;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index"),
            Option("state-var"; type=Int, default=nothing, description="State variable index"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("gamma"; type=Float64, default=1.5, description="Transition function steepness"),
            Option("method"; type=String, default="logistic", description="logistic|exponential|indicator"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate state-dependent LP (Auerbach & Gorodnichenko 2013)")

    lp_propensity = LeafCommand("propensity", _lp_propensity;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("treatment"; type=Int, default=1, description="Treatment variable index"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("score-method"; type=String, default="logit", description="logit|probit"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate propensity score LP (Angrist et al. 2018)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate"   => lp_estimate,
        "iv"         => lp_iv,
        "smooth"     => lp_smooth,
        "state"      => lp_state,
        "propensity" => lp_propensity,
    )
    return NodeCommand("lp", subcmds, "Local Projections (LP)")
end

function _lp_estimate(; data::String, shock::Int=1, horizons::Int=20,
                       control_lags::Int=4, vcov::String="newey_west",
                       output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating Local Projections: shock=$shock, horizons=$horizons, vcov=$vcov")
    println()

    model = estimate_lp(Y, shock, horizons;
        lags=control_lags,
        cov_type=Symbol(vcov))

    irf_result = lp_irf(model)

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"

    irf_df = DataFrame()
    irf_df.horizon = 0:horizons
    n_resp = size(irf_result.values, 2)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        irf_df[!, vname] = irf_result.values[:, vi]
        irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi]
        irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi]
    end

    output_result(irf_df; format=Symbol(format), output=output,
                  title="LP IRF to $shock_name shock")
end

function _lp_iv(; data::String, shock::Int=1, instruments::String="",
                 horizons::Int=20, control_lags::Int=4, vcov::String="newey_west",
                 output::String="", format::String="table")
    isempty(instruments) && error("LP-IV requires --instruments=<file.csv>")

    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    iv_df = load_data(instruments)
    Z = df_to_matrix(iv_df)

    println("Estimating LP-IV: shock=$shock, horizons=$horizons, instruments=$(size(Z, 2))")
    println()

    model = estimate_lp_iv(Y, shock, Z, horizons;
        lags=control_lags,
        cov_type=Symbol(vcov))

    irf_result = lp_iv_irf(model)

    # Weak instrument diagnostics
    wi = weak_instrument_test(model)
    println("First-stage F-statistic: $(round(wi.F_stat; digits=2))")
    if wi.F_stat < 10
        printstyled("⚠ Weak instruments (F < 10)\n"; color=:yellow)
    end
    println()

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"

    irf_df = DataFrame()
    irf_df.horizon = 0:horizons
    n_resp = size(irf_result.values, 2)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        irf_df[!, vname] = irf_result.values[:, vi]
        irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi]
        irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi]
    end

    output_result(irf_df; format=Symbol(format), output=output,
                  title="LP-IV IRF to $shock_name shock")
end

function _lp_smooth(; data::String, shock::Int=1, horizons::Int=20,
                     knots::Int=3, lambda::Float64=0.0,
                     output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    # Auto-select lambda via cross-validation if 0
    lam = if lambda == 0.0
        println("Cross-validating smoothing parameter...")
        cross_validate_lambda(Y, shock, horizons)
    else
        lambda
    end

    println("Estimating Smooth LP: shock=$shock, horizons=$horizons, knots=$knots, λ=$(round(lam; digits=4))")
    println()

    model = estimate_smooth_lp(Y, shock, horizons; n_knots=knots, lambda=lam)
    irf_result = smooth_lp_irf(model)

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"

    irf_df = DataFrame()
    irf_df.horizon = 0:horizons
    n_resp = size(irf_result.values, 2)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        irf_df[!, vname] = irf_result.values[:, vi]
        irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi]
        irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi]
    end

    output_result(irf_df; format=Symbol(format), output=output,
                  title="Smooth LP IRF to $shock_name shock")
end

function _lp_state(; data::String, shock::Int=1, state_var=nothing,
                    horizons::Int=20, gamma::Float64=1.5, method::String="logistic",
                    output::String="", format::String="table")
    isnothing(state_var) && error("state-dependent LP requires --state-var=<idx>")

    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating State-Dependent LP: shock=$shock, state=$state_var, γ=$gamma, method=$method")
    println()

    model = estimate_state_lp(Y, shock, state_var, horizons;
        gamma=gamma, method=Symbol(method))

    results = state_irf(model)

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"

    for (regime, label) in [(:expansion, "Expansion"), (:recession, "Recession")]
        irf_result = getfield(results, regime)
        irf_df = DataFrame()
        irf_df.horizon = 0:horizons
        n_resp = size(irf_result.values, 2)
        for vi in 1:n_resp
            vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
            irf_df[!, vname] = irf_result.values[:, vi]
        end
        output_result(irf_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(lowercase(label))."),
                      title="State LP IRF ($label) to $shock_name shock")
        println()
    end

    # Test regime difference
    diff_test = test_regime_difference(model)
    println("Regime Difference Test:")
    println("  Wald statistic: $(round(diff_test.wald_stat; digits=3))")
    println("  p-value: $(round(diff_test.p_value; digits=4))")
end

function _lp_propensity(; data::String, treatment::Int=1, horizons::Int=20,
                         score_method::String="logit",
                         output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating Propensity Score LP: treatment=$treatment, horizons=$horizons, method=$score_method")
    println()

    # Treatment is the specified column, covariates are all others
    treatment_vec = Y[:, treatment]
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    config = PropensityScoreConfig(; method=Symbol(score_method))
    model = estimate_propensity_lp(Y, treatment_vec, horizons;
        covariates=covariates, config=config)

    irf_result = propensity_irf(model)

    # Diagnostics
    diag = propensity_diagnostics(model)
    println("Propensity Score Diagnostics:")
    println("  Mean score: $(round(diag.mean_score; digits=4))")
    println("  Effective sample size: $(round(diag.effective_n; digits=1))")
    println()

    irf_df = DataFrame()
    irf_df.horizon = 0:horizons
    n_resp = size(irf_result.values, 2)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        irf_df[!, vname] = irf_result.values[:, vi]
        irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi]
        irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi]
    end

    treat_name = treatment <= length(varnames) ? varnames[treatment] : "treatment_$treatment"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Propensity Score LP: ATE of $treat_name")
end
