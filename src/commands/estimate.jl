# Estimate commands: var, bvar, lp, arima, gmm, static, dynamic, gdfm, arch, garch, egarch, gjr_garch, sv, fastica, ml

function register_estimate_commands!()
    est_var = LeafCommand("var", _estimate_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("trend"; type=String, default="constant", description="none|constant|trend|both"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a VAR(p) model")

    est_bvar = LeafCommand("bvar", _estimate_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("prior"; type=String, default="minnesota", description="Prior type: minnesota"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("method"; type=String, default="mean", description="mean|median (posterior extraction)"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a Bayesian VAR model (includes posterior extraction)")

    est_lp = LeafCommand("lp", _estimate_lp;
        args=[Argument("data"; description="Path to CSV data file")],
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

    est_arima = LeafCommand("arima", _estimate_arima;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto selection)"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("max-p"; type=Int, default=5, description="Max AR order for auto selection"),
            Option("max-d"; type=Int, default=2, description="Max differencing order for auto selection"),
            Option("max-q"; type=Int, default=5, description="Max MA order for auto selection"),
            Option("criterion"; type=String, default="bic", description="aic|bic (for auto selection)"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Estimate ARIMA(p,d,q) model (auto-selects order when --p omitted)")

    est_gmm = LeafCommand("gmm", _estimate_gmm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("config"; type=String, default="", description="TOML config for moment conditions and instruments"),
            Option("weighting"; short="w", type=String, default="twostep", description="identity|optimal|twostep|iterated"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a GMM model")

    est_static = LeafCommand("static", _estimate_static;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("criterion"; type=String, default="ic1", description="ic1|ic2|ic3 for auto selection"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate static factor model (PCA)")

    est_dynamic = LeafCommand("dynamic", _estimate_dynamic;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate dynamic factor model")

    est_gdfm = LeafCommand("gdfm", _estimate_gdfm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate generalized dynamic factor model")

    est_arch = LeafCommand("arch", _estimate_arch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate ARCH(q) model")

    est_garch = LeafCommand("garch", _estimate_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate GARCH(p,q) model")

    est_egarch = LeafCommand("egarch", _estimate_egarch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="EGARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate EGARCH(p,q) model")

    est_gjr_garch = LeafCommand("gjr_garch", _estimate_gjr_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate GJR-GARCH(p,q) model")

    est_sv = LeafCommand("sv", _estimate_sv;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate Stochastic Volatility model")

    est_fastica = LeafCommand("fastica", _estimate_fastica;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("method"; type=String, default="fastica", description="fastica|jade|sobi|dcov|hsic"),
            Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis (for FastICA)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="ICA-based non-Gaussian SVAR identification")

    est_ml = LeafCommand("ml", _estimate_ml;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("distribution"; short="d", type=String, default="student_t", description="student_t|skew_t|ghd|mixture_normal|pml|skew_normal"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Maximum likelihood non-Gaussian SVAR identification")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"       => est_var,
        "bvar"      => est_bvar,
        "lp"        => est_lp,
        "arima"     => est_arima,
        "gmm"       => est_gmm,
        "static"    => est_static,
        "dynamic"   => est_dynamic,
        "gdfm"      => est_gdfm,
        "arch"      => est_arch,
        "garch"     => est_garch,
        "egarch"    => est_egarch,
        "gjr_garch" => est_gjr_garch,
        "sv"        => est_sv,
        "fastica"   => est_fastica,
        "ml"        => est_ml,
    )
    return NodeCommand("estimate", subcmds, "Estimate econometric models")
end

# ── VAR ────────────────────────────────────────────────────

function _estimate_var(; data::String, lags=nothing, trend::String="constant",
                        output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    println("Estimating VAR($p) with $n variables: $(join(varnames, ", "))")
    println("Trend: $trend, Observations: $(size(Y, 1))")
    println()

    model = estimate_var(Y, p)
    report(model)

    coef_df = _build_var_coef_table(coef(model), varnames, p)
    output_result(coef_df; format=Symbol(format), output=output, title="VAR($p) Coefficients")

    println()
    output_model_criteria(model; format=format, title="Information Criteria")

    storage_save_auto!("var", serialize_model(model),
        Dict{String,Any}("command" => "estimate var", "data" => data, "lags" => p))
end

# ── BVAR ───────────────────────────────────────────────────

function _estimate_bvar(; data::String, lags::Int=4, prior::String="minnesota",
                         draws::Int=2000, sampler::String="nuts", method::String="mean",
                         config::String="", output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    p = lags

    println("Estimating Bayesian VAR($p) with $n variables: $(join(varnames, ", "))")
    println("Prior: $prior, Sampler: $sampler, Draws: $draws, Posterior: $method")
    println()

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? Symbol(prior) : :minnesota

    post = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_samples=draws,
        prior=prior_sym, hyper=prior_obj)

    model = if method == "median"
        posterior_median_model(post)
    else
        posterior_mean_model(post)
    end

    report(model)

    coef_df = _build_var_coef_table(coef(model), varnames, p)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="BVAR($p) Posterior $(titlecase(method)) Coefficients")

    println()
    output_model_criteria(model; format=format, title="Information Criteria (Posterior $(titlecase(method)))")

    storage_save_auto!("bvar", serialize_model(model),
        Dict{String,Any}("command" => "estimate bvar", "data" => data, "lags" => p,
                          "draws" => draws, "sampler" => sampler, "method" => method))
end

# ── LP ─────────────────────────────────────────────────────

function _estimate_lp(; data::String, method::String="standard", shock::Int=1,
                       horizons::Int=20, control_lags::Int=4, vcov::String="newey_west",
                       instruments::String="", knots::Int=3, lambda::Float64=0.0,
                       state_var=nothing, gamma::Float64=1.5, transition::String="logistic",
                       treatment::Int=1, score_method::String="logit",
                       output::String="", format::String="table")
    validate_method(method, ["standard", "iv", "smooth", "state", "propensity", "robust"], "LP method")
    if method == "standard"
        _estimate_lp_standard(data, shock, horizons, control_lags, vcov, output, format)
    elseif method == "iv"
        _estimate_lp_iv(data, shock, horizons, control_lags, vcov, instruments, output, format)
    elseif method == "smooth"
        _estimate_lp_smooth(data, shock, horizons, knots, lambda, output, format)
    elseif method == "state"
        _estimate_lp_state(data, shock, horizons, state_var, gamma, transition, output, format)
    elseif method == "propensity"
        _estimate_lp_propensity(data, treatment, horizons, score_method, output, format)
    elseif method == "robust"
        _estimate_lp_robust(data, treatment, horizons, score_method, output, format)
    end
end

function _estimate_lp_standard(data, shock, horizons, control_lags, vcov, output, format)
    Y, varnames = load_multivariate_data(data)

    println("Estimating Local Projections: shock=$shock, horizons=$horizons, vcov=$vcov")
    println()

    model = estimate_lp(Y, shock, horizons; lags=control_lags, cov_type=Symbol(vcov))
    irf_result = lp_irf(model)

    shock_name = _shock_name(varnames, shock)

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

function _estimate_lp_iv(data, shock, horizons, control_lags, vcov, instruments, output, format)
    isempty(instruments) && error("LP-IV requires --instruments=<file.csv>")

    Y, varnames = load_multivariate_data(data)
    Z, _ = load_multivariate_data(instruments)

    println("Estimating LP-IV: shock=$shock, horizons=$horizons, instruments=$(size(Z, 2))")
    println()

    model = estimate_lp_iv(Y, shock, Z, horizons; lags=control_lags, cov_type=Symbol(vcov))
    irf_result = lp_iv_irf(model)

    wi = weak_instrument_test(model)
    println("First-stage F-statistic: $(round(wi.F_stat; digits=2))")
    if wi.F_stat < 10
        printstyled("Warning: Weak instruments (F < 10)\n"; color=:yellow)
    end
    println()

    shock_name = _shock_name(varnames, shock)

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

function _estimate_lp_smooth(data, shock, horizons, knots, lambda, output, format)
    Y, varnames = load_multivariate_data(data)

    lam = if lambda == 0.0
        println("Cross-validating smoothing parameter...")
        cross_validate_lambda(Y, shock, horizons)
    else
        lambda
    end

    println("Estimating Smooth LP: shock=$shock, horizons=$horizons, knots=$knots, lambda=$(round(lam; digits=4))")
    println()

    model = estimate_smooth_lp(Y, shock, horizons; n_knots=knots, lambda=lam)
    irf_result = smooth_lp_irf(model)

    shock_name = _shock_name(varnames, shock)

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

function _estimate_lp_state(data, shock, horizons, state_var, gamma, transition, output, format)
    isnothing(state_var) && error("state-dependent LP requires --state-var=<idx>")

    Y, varnames = load_multivariate_data(data)

    println("Estimating State-Dependent LP: shock=$shock, state=$state_var, gamma=$gamma, transition=$transition")
    println()

    state_vec = Y[:, state_var]
    model = estimate_state_lp(Y, shock, state_vec, horizons; gamma=gamma)
    results = state_irf(model)

    shock_name = _shock_name(varnames, shock)

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

    diff_test = test_regime_difference(model)
    jt = diff_test.joint_test
    println("Regime Difference Test:")
    println("  Avg t-statistic: $(round(jt.avg_t_stat; digits=3))")
    println("  p-value: $(round(jt.p_value; digits=4))")
end

function _estimate_lp_propensity(data, treatment, horizons, score_method, output, format)
    Y, varnames = load_multivariate_data(data)

    println("Estimating Propensity Score LP: treatment=$treatment, horizons=$horizons, method=$score_method")
    println()

    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = estimate_propensity_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    irf_result = propensity_irf(model)

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

    treat_name = _var_name(varnames, treatment)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Propensity Score LP: ATE of $treat_name")
end

function _estimate_lp_robust(data, treatment, horizons, score_method, output, format)
    Y, varnames = load_multivariate_data(data)

    println("Estimating Doubly Robust LP: treatment=$treatment, horizons=$horizons, method=$score_method")
    println()

    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = doubly_robust_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    irf_result = propensity_irf(model)

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

    treat_name = _var_name(varnames, treatment)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Doubly Robust LP: ATE of $treat_name")
end

# ── ARIMA ──────────────────────────────────────────────────

function _estimate_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          max_p::Int=5, max_d::Int=2, max_q::Int=5,
                          criterion::String="bic", method::String="css_mle",
                          format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    method_sym = Symbol(method)
    safe_method = method_sym == :css_mle ? :mle : method_sym

    model = if isnothing(p)
        crit_sym = Symbol(lowercase(criterion))
        println("Auto ARIMA: variable=$vname, observations=$(length(y))")
        println("  Search: p=0:$max_p, d=0:$max_d, q=0:$max_q, criterion=$criterion, method=$method")
        println()
        m = auto_arima(y; max_p=max_p, max_q=max_q, max_d=max_d, criterion=crit_sym, method=safe_method)
        label = _model_label(ar_order(m), diff_order(m), ma_order(m))
        printstyled("Selected model: $label\n"; bold=true)
        println()
        m
    else
        label = _model_label(p, d, q)
        println("Estimating $label: variable=$vname, observations=$(length(y)), method=$method")
        println()
        _estimate_arima_model(y, p, d, q; method=method_sym)
    end

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    _arima_coef_table(model; format=format, output=output, title="$label Coefficients ($vname)")

    println()
    output_kv(Pair{String,Any}[
        "AIC" => round(aic(model); digits=4),
        "BIC" => round(bic(model); digits=4),
        "Log-likelihood" => round(loglikelihood(model); digits=4),
    ]; format=format, title="Information Criteria")
end

# ARIMA helpers (from old arima.jl)
function _estimate_arima_model(y::Vector{Float64}, p::Int, d::Int, q::Int; method::Symbol=:css_mle)
    if d == 0 && q == 0
        ar_method = method in (:ols, :mle) ? method : :mle
        return estimate_ar(y, p; method=ar_method)
    elseif d == 0 && p == 0
        return estimate_ma(y, q; method=method)
    elseif d == 0
        return estimate_arma(y, p, q; method=method)
    else
        return estimate_arima(y, p, d, q; method=method)
    end
end

function _model_label(p::Int, d::Int, q::Int)
    if d == 0 && q == 0
        return "AR($p)"
    elseif d == 0 && p == 0
        return "MA($q)"
    elseif d == 0
        return "ARMA($p,$q)"
    else
        return "ARIMA($p,$d,$q)"
    end
end

function _arima_coef_table(model; format::String="table", output::String="", title::String="Coefficients")
    c = coef(model)
    p_order = ar_order(model)
    q_order = ma_order(model)
    param_names = String[]
    for i in 1:p_order
        push!(param_names, "ar$i")
    end
    for i in 1:q_order
        push!(param_names, "ma$i")
    end
    n_named = p_order + q_order
    for i in (n_named+1):length(c)
        push!(param_names, "const$i")
    end

    coef_df = try
        se = stderror(model)
        DataFrame(parameter=param_names, estimate=round.(c; digits=6), std_error=round.(se; digits=6))
    catch
        DataFrame(parameter=param_names, estimate=round.(c; digits=6))
    end

    output_result(coef_df; format=Symbol(format), output=output, title=title)
end

# ── GMM ────────────────────────────────────────────────────

function _estimate_gmm(; data::String, config::String="",
                        weighting::String="twostep",
                        output::String="", format::String="table")
    isempty(config) && error("GMM requires a --config=<file.toml> specifying moment conditions and instruments")

    Y, varnames = load_multivariate_data(data)

    cfg = load_config(config)
    gmm_cfg = get_gmm(cfg)

    weighting_map = Dict("identity" => :identity, "optimal" => :optimal,
                         "twostep" => :two_step, "iterated" => :iterated)
    w = get(weighting_map, lowercase(weighting), :two_step)

    println("Estimating GMM: weighting=$weighting")
    println("  Moment conditions: $(length(gmm_cfg["moment_conditions"]))")
    println()

    moment_cols = gmm_cfg["moment_conditions"]
    shock_var = if !isempty(moment_cols)
        idx = findfirst(==(moment_cols[1]), varnames)
        isnothing(idx) ? 1 : idx
    else
        1
    end

    models = estimate_lp_gmm(Y, shock_var, 0; lags=4, weighting=w)

    if !isempty(models)
        model = models[1]
        summ = gmm_summary(model)
        jtest = j_test(model)
        println()
        println("Hansen's J-test for overidentification:")
        println("  J-statistic: $(round(jtest.J_stat; digits=4))")
        println("  p-value: $(round(jtest.p_value; digits=4))")
        println("  Degrees of freedom: $(jtest.df)")

        if jtest.p_value < 0.05
            printstyled("  -> Reject valid moment conditions at 5%\n"; color=:yellow)
        else
            printstyled("  -> Cannot reject valid moment conditions\n"; color=:green)
        end

        if !isempty(output)
            se = stderror(model)
            param_df = DataFrame(parameter=["theta$i" for i in 1:length(model.theta)],
                                 estimate=model.theta, std_error=se)
            output_result(param_df; format=Symbol(format), output=output, title="GMM Estimates")
        end
    end
end

# ── Factor Models ──────────────────────────────────────────

function _estimate_static(; data::String, nfactors=nothing, criterion::String="ic1",
                           output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        println("Selecting number of factors via Bai-Ng information criteria...")
        ic = ic_criteria(X, min(20, size(X, 2)))
        r_sym = Symbol("r_", uppercase(criterion))
        optimal_r = getfield(ic, r_sym)
        println("  $criterion suggests $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Estimating static factor model: $r factors, $(size(X, 2)) variables, $(size(X, 1)) observations")
    println()

    model = estimate_factors(X, r)

    scree = scree_plot_data(model)
    scree_df = DataFrame(component=scree.factors, eigenvalue=scree.explained_variance,
                         cumulative=scree.cumulative_variance)
    output_result(scree_df; format=Symbol(format), title="Scree Data (Eigenvalues & Variance Shares)")
    println()

    loadings = model.loadings
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format), output=output, title="Factor Loadings")
end

function _estimate_dynamic(; data::String, nfactors=nothing, factor_lags::Int=1,
                            method::String="twostep", output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        println("Selecting number of factors...")
        ic = ic_criteria(X, min(10, size(X, 2)))
        optimal_r = ic.r_IC1
        println("  Auto-selected $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Estimating dynamic factor model: $r factors, $factor_lags lags, method=$method")
    println()

    model = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))

    stable_result = is_stationary(model)
    stable = stable_result isa Bool ? stable_result : stable_result.is_stationary
    if stable
        printstyled("Factor VAR is stationary\n"; color=:green)
    else
        printstyled("Factor VAR is not stationary\n"; color=:yellow)
    end
    println()

    loadings = model.loadings
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format), output=output, title="Dynamic Factor Loadings")

    println()
    println("Factor VAR Companion Matrix eigenvalues:")
    comp = companion_matrix_factors(model)
    eig_moduli = abs.(eigvals(comp))
    for (i, ev) in enumerate(sort(eig_moduli; rev=true))
        println("  lambda$i = $(round(ev; digits=6))")
    end
end

function _estimate_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                         output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    q = if isnothing(dynamic_rank)
        println("Selecting dynamic rank...")
        ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
        q_opt = ic.q_ratio
        println("  Auto-selected $q_opt dynamic factors")
        q_opt
    else
        dynamic_rank
    end

    r = if isnothing(nfactors)
        println("Selecting static rank...")
        ic_static = ic_criteria(X, min(20, size(X, 2)))
        r_opt = ic_static.r_IC1
        println("  Auto-selected $r_opt static factors")
        r_opt
    else
        nfactors
    end

    println("Estimating GDFM: static rank=$r, dynamic rank=$q")
    println()

    model = estimate_gdfm(X, q; r=r)

    var_shares = common_variance_share(model)
    var_df = DataFrame(variable=varnames, common_variance_share=round.(var_shares; digits=4))
    output_result(var_df; format=Symbol(format), output=output,
                  title="GDFM Common Variance Shares")

    println()
    println("Average common variance share: $(round(mean(var_shares); digits=4))")
end

# ── Volatility Models (NEW) ───────────────────────────────

function _estimate_arch(; data::String, column::Int=1, q::Int=1,
                         output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    println("Estimating ARCH($q): variable=$vname, observations=$(length(y))")
    println()
    model = estimate_arch(y, q)
    param_names = ["mu"; "omega"; ["alpha$i" for i in 1:q]]
    _vol_estimate_output(model, vname, param_names, "ARCH($q)"; format=format, output=output)
    uc = unconditional_variance(model)
    println("Unconditional variance: $(round(uc; digits=4))")
end

function _estimate_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                          output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    println("Estimating GARCH($p,$q): variable=$vname, observations=$(length(y))")
    println()
    model = estimate_garch(y, p, q)
    param_names = ["mu"; "omega"; ["alpha$i" for i in 1:q]; ["beta$i" for i in 1:p]]
    _vol_estimate_output(model, vname, param_names, "GARCH($p,$q)"; format=format, output=output)
    hl = halflife(model)
    println("Half-life: $(round(hl; digits=2)) periods")
    uc = unconditional_variance(model)
    println("Unconditional variance: $(round(uc; digits=4))")
end

function _estimate_egarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                           output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    println("Estimating EGARCH($p,$q): variable=$vname, observations=$(length(y))")
    println()
    model = estimate_egarch(y, p, q)
    param_names = ["mu"; "omega"; ["alpha$i" for i in 1:q]; ["gamma$i" for i in 1:q]; ["beta$i" for i in 1:p]]
    _vol_estimate_output(model, vname, param_names, "EGARCH($p,$q)"; format=format, output=output)
end

function _estimate_gjr_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                              output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    println("Estimating GJR-GARCH($p,$q): variable=$vname, observations=$(length(y))")
    println()
    model = estimate_gjr_garch(y, p, q)
    param_names = ["mu"; "omega"; ["alpha$i" for i in 1:q]; ["gamma$i" for i in 1:q]; ["beta$i" for i in 1:p]]
    _vol_estimate_output(model, vname, param_names, "GJR-GARCH($p,$q)"; format=format, output=output)
    hl = halflife(model)
    println("Half-life: $(round(hl; digits=2)) periods")
end

function _estimate_sv(; data::String, column::Int=1, draws::Int=5000,
                       output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    println("Estimating Stochastic Volatility: variable=$vname, observations=$(length(y)), draws=$draws")
    println()
    model = estimate_sv(y; n_samples=draws)
    param_names = ["mu", "phi", "sigma_eta"]
    _vol_estimate_output(model, vname, param_names, "SV"; format=format, output=output)
end

# ── Non-Gaussian ICA ──────────────────────────────────────

function _estimate_fastica(; data::String, lags=nothing, method::String="fastica",
                             contrast::String="logcosh", output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    println("Non-Gaussian SVAR: method=$method, contrast=$contrast, VAR($p), $n variables")
    println()

    result = if method == "jade"
        identify_jade(model)
    elseif method == "sobi"
        identify_sobi(model)
    elseif method == "dcov"
        identify_dcov(model)
    elseif method == "hsic"
        identify_hsic(model)
    else
        identify_fastica(model; contrast=Symbol(contrast))
    end

    if hasproperty(result, :converged)
        if result.converged
            printstyled("Converged in $(result.iterations) iterations\n"; color=:green)
        else
            printstyled("Did not converge after $(result.iterations) iterations\n"; color=:yellow)
        end
    end
    println()

    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    println()

    shocks = result.shocks
    T_shocks = size(shocks, 1)
    n_show = min(T_shocks, 10)
    shock_df = DataFrame(shocks[1:n_show, :], ["shock_$i" for i in 1:n])
    insertcols!(shock_df, 1, :t => 1:n_show)
    output_result(shock_df; format=Symbol(format), output=output,
                  title="Structural Shocks (first $n_show observations)")
end

# ── Non-Gaussian ML ───────────────────────────────────────

function _estimate_ml(; data::String, lags=nothing, distribution::String="student_t",
                        output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    println("Non-Gaussian ML SVAR: distribution=$distribution, VAR($p), $n variables")
    println()

    result = if distribution == "mixture_normal"
        identify_mixture_normal(model)
    elseif distribution == "pml"
        identify_pml(model)
    elseif distribution == "skew_normal"
        identify_skew_normal(model)
    else
        identify_nongaussian_ml(model; distribution=Symbol(distribution))
    end

    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    println()

    output_kv(Pair{String,Any}[
        "Log-likelihood" => round(result.loglik; digits=4),
        "Log-likelihood (Gaussian)" => round(result.loglik_gaussian; digits=4),
        "AIC" => round(result.aic; digits=4),
        "BIC" => round(result.bic; digits=4),
        "Distribution" => string(result.distribution),
    ]; format=format, title="Model Fit")

    if !isempty(result.dist_params)
        println()
        println("Distribution parameters:")
        for (k, v) in pairs(result.dist_params)
            if v isa AbstractArray
                println("  $k = $(round.(v; digits=4))")
            else
                println("  $k = $(round(v; digits=4))")
            end
        end
    end

    if !isnothing(result.se) && length(result.se) > 0
        println()
        se_df = DataFrame(
            parameter=["B0[$i,$j]" for i in 1:n for j in 1:n],
            estimate=vec(result.B0),
            std_error=result.se[1:min(length(result.se), n*n)]
        )
        output_result(se_df; format=Symbol(format), output=output,
                      title="Parameter Estimates with Standard Errors")
    end
end

