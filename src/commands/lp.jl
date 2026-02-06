# Local Projection commands: estimate, irf, fevd, hd, forecast (pipeline architecture)

function register_lp_commands!()
    lp_estimate = LeafCommand("estimate", _lp_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("method"; type=String, default="standard", description="standard|iv|smooth|state|propensity|robust"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("control-lags"; type=Int, default=4, description="Number of control lags"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("instruments"; type=String, default="", description="Path to instruments CSV (iv only)"),
            Option("knots"; type=Int, default=3, description="Number of B-spline knots (smooth only)"),
            Option("lambda"; type=Float64, default=0.0, description="Smoothing penalty, 0=auto CV (smooth only)"),
            Option("state-var"; type=Int, default=nothing, description="State variable index (state only)"),
            Option("gamma"; type=Float64, default=1.5, description="Transition steepness (state only)"),
            Option("transition"; type=String, default="logistic", description="logistic|exponential|indicator (state only)"),
            Option("treatment"; type=Int, default=1, description="Treatment variable index (propensity/robust only)"),
            Option("score-method"; type=String, default="logit", description="logit|probit (propensity/robust only)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate local projections (standard|iv|smooth|state|propensity|robust)")

    lp_irf = LeafCommand("irf", _lp_irf;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Single shock index (1-based)"),
            Option("shocks"; type=String, default="", description="Comma-separated shock indices (e.g. 1,2,3)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification (default: same as --lags)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("ci"; type=String, default="none", description="none|bootstrap"),
            Option("replications"; type=Int, default=200, description="Bootstrap replications"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for sign/narrative restrictions"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute structural LP impulse response functions")

    lp_fevd = LeafCommand("fevd", _lp_fevd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute forecast error variance decomposition via structural LP")

    lp_hd = LeafCommand("hd", _lp_hd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition via structural LP")

    lp_forecast = LeafCommand("forecast", _lp_forecast;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("shock-size"; type=Float64, default=1.0, description="Impulse shock size"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("ci-method"; type=String, default="analytical", description="analytical|bootstrap|none"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level"),
            Option("n-boot"; type=Int, default=500, description="Bootstrap replications"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute direct LP forecasts")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => lp_estimate,
        "irf"      => lp_irf,
        "fevd"     => lp_fevd,
        "hd"       => lp_hd,
        "forecast" => lp_forecast,
    )
    return NodeCommand("lp", subcmds, "Local Projections (LP)")
end

# ── Estimation handler ─────────────────────────────────────

function _lp_estimate(; data::String, method::String="standard", shock::Int=1,
                       horizons::Int=20, control_lags::Int=4, vcov::String="newey_west",
                       instruments::String="", knots::Int=3, lambda::Float64=0.0,
                       state_var=nothing, gamma::Float64=1.5, transition::String="logistic",
                       treatment::Int=1, score_method::String="logit",
                       output::String="", format::String="table")
    if method == "standard"
        _lp_estimate_standard(data, shock, horizons, control_lags, vcov, output, format)
    elseif method == "iv"
        _lp_estimate_iv(data, shock, horizons, control_lags, vcov, instruments, output, format)
    elseif method == "smooth"
        _lp_estimate_smooth(data, shock, horizons, knots, lambda, output, format)
    elseif method == "state"
        _lp_estimate_state(data, shock, horizons, state_var, gamma, transition, output, format)
    elseif method == "propensity"
        _lp_estimate_propensity(data, treatment, horizons, score_method, output, format)
    elseif method == "robust"
        _lp_estimate_robust(data, treatment, horizons, score_method, output, format)
    else
        error("unknown LP method: $method (expected standard|iv|smooth|state|propensity|robust)")
    end
end

function _lp_estimate_standard(data, shock, horizons, control_lags, vcov, output, format)
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

function _lp_estimate_iv(data, shock, horizons, control_lags, vcov, instruments, output, format)
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

function _lp_estimate_smooth(data, shock, horizons, knots, lambda, output, format)
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

function _lp_estimate_state(data, shock, horizons, state_var, gamma, transition, output, format)
    isnothing(state_var) && error("state-dependent LP requires --state-var=<idx>")

    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating State-Dependent LP: shock=$shock, state=$state_var, γ=$gamma, transition=$transition")
    println()

    state_vec = Y[:, state_var]
    model = estimate_state_lp(Y, shock, state_vec, horizons;
        gamma=gamma)

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
    jt = diff_test.joint_test
    println("Regime Difference Test:")
    println("  Avg t-statistic: $(round(jt.avg_t_stat; digits=3))")
    println("  p-value: $(round(jt.p_value; digits=4))")
end

function _lp_estimate_propensity(data, treatment, horizons, score_method, output, format)
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating Propensity Score LP: treatment=$treatment, horizons=$horizons, method=$score_method")
    println()

    # Treatment must be Bool — binarize via median split
    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = estimate_propensity_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    irf_result = propensity_irf(model)

    # Diagnostics
    diag = propensity_diagnostics(model)
    ps = diag.propensity_summary
    println("Propensity Score Diagnostics:")
    println("  Treated mean score: $(round(ps.treated.mean; digits=4))")
    println("  Control mean score: $(round(ps.control.mean; digits=4))")
    println("  Max weighted SMD: $(round(diag.balance.max_weighted; digits=4))")
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

function _lp_estimate_robust(data, treatment, horizons, score_method, output, format)
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Estimating Doubly Robust LP: treatment=$treatment, horizons=$horizons, method=$score_method")
    println()

    # Treatment must be Bool — binarize via median split
    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = doubly_robust_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    irf_result = propensity_irf(model)

    # Diagnostics
    diag = propensity_diagnostics(model)
    ps = diag.propensity_summary
    println("Doubly Robust Diagnostics:")
    println("  Treated mean score: $(round(ps.treated.mean; digits=4))")
    println("  Control mean score: $(round(ps.control.mean; digits=4))")
    println("  Max weighted SMD: $(round(diag.balance.max_weighted; digits=4))")
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
                  title="Doubly Robust LP: ATE of $treat_name")
end

# ── Post-estimation: IRF ─────────────────────────────────────

function _lp_irf(; data::String, shock::Int=1, shocks::String="",
                  horizons::Int=20, lags::Int=4, var_lags=nothing,
                  id::String="cholesky", ci::String="none",
                  replications::Int=200, conf_level::Float64=0.95,
                  vcov::String="newey_west", config::String="",
                  output::String="", format::String="table")
    # Multi-shock mode
    if !isempty(shocks)
        shock_indices = parse.(Int, split(shocks, ","))
    else
        shock_indices = [shock]
    end

    slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
        id, vcov, config; ci_type=Symbol(ci), reps=replications, conf_level=conf_level)

    n = size(Y, 2)
    irf_result = slp.irf

    println("Computing LP IRFs: horizons=$horizons, id=$id, ci=$ci")
    println()

    for shock_idx in shock_indices
        (shock_idx < 1 || shock_idx > n) && error("shock index $shock_idx out of range (data has $n variables)")
        shock_name = shock_idx <= length(varnames) ? varnames[shock_idx] : "shock_$shock_idx"

        irf_vals = irf_result.values  # H x n x n
        n_h = size(irf_vals, 1)

        irf_df = DataFrame()
        irf_df.horizon = 0:(n_h-1)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, vname] = irf_vals[:, vi, shock_idx]
        end

        if ci != "none" && !isnothing(irf_result.ci_lower)
            for (vi, vname) in enumerate(varnames)
                irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock_idx]
                irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock_idx]
            end
        end

        output_result(irf_df; format=Symbol(format),
                      output=isempty(output) ? "" : (length(shock_indices) > 1 ? replace(output, "." => "_$(shock_name).") : output),
                      title="LP IRF to $shock_name shock ($id identification)")
        println()
    end
end

# ── Post-estimation: FEVD ────────────────────────────────────

function _lp_fevd(; data::String, horizons::Int=20, lags::Int=4, var_lags=nothing,
                   id::String="cholesky", vcov::String="newey_west", config::String="",
                   output::String="", format::String="table")
    slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
        id, vcov, config)
    n = size(Y, 2)

    println("Computing LP FEVD: horizons=$horizons, id=$id")
    println()

    fevd_result = lp_fevd(slp, horizons)
    proportions = fevd_result.bias_corrected  # n_vars x n_shocks x H

    for vi in 1:n
        fevd_df = DataFrame()
        fevd_df.horizon = 1:horizons
        for si in 1:n
            shock_name = si <= length(varnames) ? varnames[si] : "shock_$si"
            fevd_df[!, shock_name] = proportions[vi, si, :]
        end
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        output_result(fevd_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(vname)."),
                      title="LP FEVD for $vname ($id identification)")
        println()
    end
end

# ── Post-estimation: HD ──────────────────────────────────────

function _lp_hd(; data::String, lags::Int=4, var_lags=nothing,
                 id::String="cholesky", vcov::String="newey_west", config::String="",
                 output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)
    vp = isnothing(var_lags) ? lags : var_lags
    hd_horizon = size(Y, 1) - vp

    method = get(ID_METHOD_MAP, id, :cholesky)
    check_func, narrative_check = _build_check_func(config)
    kwargs = Dict{Symbol,Any}(
        :method => method, :lags => lags, :var_lags => vp,
        :cov_type => Symbol(vcov),
    )
    if !isnothing(check_func);      kwargs[:check_func] = check_func; end
    if !isnothing(narrative_check);  kwargs[:narrative_check] = narrative_check; end

    slp = structural_lp(Y, hd_horizon; kwargs...)

    println("Computing LP Historical Decomposition: id=$id")
    println()

    hd_result = historical_decomposition(slp, hd_horizon)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        printstyled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        printstyled("Decomposition verification failed\n"; color=:yellow)
    end
    println()

    for vi in 1:n
        T_eff = hd_result.T_eff
        hd_df = DataFrame()
        hd_df.period = 1:T_eff
        hd_df.actual = hd_result.actual[:, vi]
        hd_df.initial = hd_result.initial_conditions[:, vi]

        for si in 1:n
            shock_name = si <= length(varnames) ? varnames[si] : "shock_$si"
            hd_df[!, "contrib_$shock_name"] = contribution(hd_result, vi, si)
        end

        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        output_result(hd_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(vname)."),
                      title="LP Historical Decomposition: $vname ($id identification)")
        println()
    end
end

# ── Post-estimation: Forecast ────────────────────────────────

function _lp_forecast(; data::String, shock::Int=1, horizons::Int=12,
                       shock_size::Float64=1.0, lags::Int=4,
                       vcov::String="newey_west",
                       ci_method::String="analytical", conf_level::Float64=0.95,
                       n_boot::Int=500,
                       output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Computing LP forecast: shock=$shock, horizons=$horizons, shock_size=$shock_size, ci=$ci_method")
    println()

    model = estimate_lp(Y, shock, horizons;
        lags=lags, cov_type=Symbol(vcov))

    shock_path = fill(shock_size, horizons)

    fc = forecast(model, shock_path;
        ci_method=Symbol(ci_method), conf_level=conf_level, n_boot=n_boot)

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    n_resp = size(fc.forecasts, 2)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        fc_df[!, vname] = fc.forecasts[:, vi]
        if ci_method != "none"
            fc_df[!, "$(vname)_lower"] = fc.ci_lower[:, vi]
            fc_df[!, "$(vname)_upper"] = fc.ci_upper[:, vi]
            fc_df[!, "$(vname)_se"] = fc.se[:, vi]
        end
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="LP Forecast (shock=$shock_name, h=$horizons, $(Int(round(conf_level*100)))% CI)")
end
