# Forecast commands: var, bvar, lp, arima, static, dynamic, gdfm,
#                    arch, garch, egarch, gjr_garch, sv

function register_forecast_commands!()
    fc_var = LeafCommand("var", _forecast_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("confidence"; type=Float64, default=0.95, description="Confidence level for intervals"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute h-step ahead VAR forecasts")

    fc_bvar = LeafCommand("bvar", _forecast_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian h-step ahead forecasts with credible intervals")

    fc_lp = LeafCommand("lp", _forecast_lp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("shock-size"; type=Float64, default=1.0, description="Impulse shock size"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("ci-method"; type=String, default="analytical", description="analytical|bootstrap|none"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level"),
            Option("n-boot"; type=Int, default=500, description="Bootstrap replications"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute direct LP forecasts")

    fc_arima = LeafCommand("arima", _forecast_arima;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto selection)"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("max-p"; type=Int, default=5, description="Max AR order for auto selection"),
            Option("max-d"; type=Int, default=2, description="Max differencing order for auto selection"),
            Option("max-q"; type=Int, default=5, description="Max MA order for auto selection"),
            Option("criterion"; type=String, default="bic", description="aic|bic"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("confidence"; type=Float64, default=0.95, description="Confidence level"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="ARIMA forecast (auto-selects order when --p omitted)")

    fc_static = LeafCommand("static", _forecast_static;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("ci-method"; type=String, default="none", description="none|bootstrap|parametric"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level for intervals"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast observables using static factor model")

    fc_dynamic = LeafCommand("dynamic", _forecast_dynamic;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast observables using dynamic factor model")

    fc_gdfm = LeafCommand("gdfm", _forecast_gdfm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast observables using GDFM")

    fc_arch = LeafCommand("arch", _forecast_arch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast volatility using ARCH model")

    fc_garch = LeafCommand("garch", _forecast_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast volatility using GARCH model")

    fc_egarch = LeafCommand("egarch", _forecast_egarch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="EGARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast volatility using EGARCH model")

    fc_gjr_garch = LeafCommand("gjr_garch", _forecast_gjr_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast volatility using GJR-GARCH model")

    fc_sv = LeafCommand("sv", _forecast_sv;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast volatility using Stochastic Volatility model")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"       => fc_var,
        "bvar"      => fc_bvar,
        "lp"        => fc_lp,
        "arima"     => fc_arima,
        "static"    => fc_static,
        "dynamic"   => fc_dynamic,
        "gdfm"      => fc_gdfm,
        "arch"      => fc_arch,
        "garch"     => fc_garch,
        "egarch"    => fc_egarch,
        "gjr_garch" => fc_gjr_garch,
        "sv"        => fc_sv,
    )
    return NodeCommand("forecast", subcmds, "Forecasting")
end

# ── VAR Forecast ─────────────────────────────────────────

function _forecast_var(; data::String, lags=nothing, horizons::Int=12,
                        confidence::Float64=0.95, from_tag::String="",
                        output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing VAR($p) forecast: horizons=$horizons, confidence=$confidence")
    println()

    B = coef(model)
    forecasts = _var_forecast_point(B, Y, p, horizons)

    # Forecast error variance from MA(inf) representation
    Sigma = model.Sigma
    alpha = 1.0 - confidence
    z = quantile_normal(1.0 - alpha / 2.0)

    comp = companion_matrix(model)
    n_comp = size(comp, 1)
    J = zeros(n, n_comp)
    J[1:n, 1:n] = I(n)

    mse = zeros(horizons, n)
    Phi_power = Matrix{Float64}(I(n_comp))
    cumulative_mse = zeros(n, n)
    for h in 1:horizons
        if h > 1
            Phi_power = Phi_power * comp
        end
        Phi_h = J * Phi_power * J'
        cumulative_mse += Phi_h * Sigma * Phi_h'
        mse[h, :] = diag(cumulative_mse)
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = forecasts[:, vi]
        se = sqrt.(max.(mse[:, vi], 0.0))
        fc_df[!, "$(vname)_lower"] = forecasts[:, vi] .- z .* se
        fc_df[!, "$(vname)_upper"] = forecasts[:, vi] .+ z .* se
        fc_df[!, "$(vname)_se"] = se
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="VAR($p) Forecast (h=$horizons, $(Int(round(confidence*100)))% CI)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "var", "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "forecast var", "data" => data))
end

# Normal quantile without importing Distributions (Abramowitz & Stegun 26.2.23)
function quantile_normal(p::Float64)
    if p < 0.5
        return -quantile_normal(1.0 - p)
    end
    t = sqrt(-2.0 * log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t^2) / (1.0 + d1*t + d2*t^2 + d3*t^3)
end

# ── BVAR Forecast ────────────────────────────────────────

function _forecast_bvar(; data::String, lags::Int=4, horizons::Int=12,
                         draws::Int=2000, sampler::String="nuts",
                         config::String="", from_tag::String="",
                         output::String="", format::String="table")
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

    println("Computing Bayesian forecast: BVAR($p), horizons=$horizons")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    b_vecs, sigmas = MacroEconometricModels.extract_chain_parameters(post)
    n_draws = length(b_vecs)
    all_forecasts = zeros(n_draws, horizons, n)

    for d in 1:n_draws
        model_d = MacroEconometricModels.parameters_to_model(b_vecs[d], sigmas[d], p, n; data=Y)
        B_d = coef(model_d)
        all_forecasts[d, :, :] = _var_forecast_point(B_d, Y, p, horizons)
    end

    fc_mean = dropdims(mean(all_forecasts; dims=1); dims=1)
    fc_q16 = zeros(horizons, n)
    fc_q50 = zeros(horizons, n)
    fc_q84 = zeros(horizons, n)
    for h in 1:horizons
        for vi in 1:n
            sorted = sort(all_forecasts[:, h, vi])
            fc_q16[h, vi] = sorted[max(1, round(Int, 0.16 * n_draws))]
            fc_q50[h, vi] = sorted[max(1, round(Int, 0.50 * n_draws))]
            fc_q84[h, vi] = sorted[max(1, round(Int, 0.84 * n_draws))]
        end
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = fc_q50[:, vi]
        fc_df[!, "$(vname)_16pct"] = fc_q16[:, vi]
        fc_df[!, "$(vname)_84pct"] = fc_q84[:, vi]
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Bayesian VAR($p) Forecast (h=$horizons, 68% credible interval)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "bvar", "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "forecast bvar", "data" => data))
end

# ── LP Forecast ──────────────────────────────────────────

function _forecast_lp(; data::String, shock::Int=1, horizons::Int=12,
                       shock_size::Float64=1.0, lags::Int=4,
                       vcov::String="newey_west",
                       ci_method::String="analytical", conf_level::Float64=0.95,
                       n_boot::Int=500, from_tag::String="",
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

    storage_save_auto!("forecast", Dict{String,Any}("type" => "lp", "shock" => shock, "horizons" => horizons),
        Dict{String,Any}("command" => "forecast lp", "data" => data))
end

# ── ARIMA Forecast ───────────────────────────────────────

function _forecast_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                           max_p::Int=5, max_d::Int=2, max_q::Int=5,
                           criterion::String="bic", horizons::Int=12,
                           confidence::Float64=0.95, method::String="css_mle",
                           from_tag::String="",
                           format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    method_sym = Symbol(method)
    safe_method = method_sym == :css_mle ? :mle : method_sym

    model = if isnothing(p)
        crit_sym = Symbol(lowercase(criterion))
        println("Auto ARIMA forecast: variable=$vname, observations=$(length(y))")
        println("  Search: p=0:$max_p, d=0:$max_d, q=0:$max_q, criterion=$criterion")
        println()
        m = auto_arima(y; max_p=max_p, max_q=max_q, max_d=max_d, criterion=crit_sym, method=safe_method)
        label = _model_label(ar_order(m), diff_order(m), ma_order(m))
        printstyled("Selected model: $label\n"; bold=true)
        println()
        m
    else
        label = _model_label(p, d, q)
        println("$label forecast: variable=$vname, horizons=$horizons")
        println()
        _estimate_arima_model(y, p, d, q; method=method_sym)
    end

    fc = forecast(model, horizons; conf_level=confidence)

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    fc_df = DataFrame(
        horizon=1:horizons,
        forecast=round.(fc.forecast; digits=6),
        lower=round.(fc.ci_lower; digits=6),
        upper=round.(fc.ci_upper; digits=6),
        se=round.(fc.se; digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="$label Forecast for $vname (h=$horizons, $(Int(round(confidence*100)))% CI)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "arima", "model" => label,
        "horizons" => horizons),
        Dict{String,Any}("command" => "forecast arima", "data" => data))
end

# ── Factor Model Forecasts ───────────────────────────────

function _forecast_static(; data::String, nfactors=nothing, horizons::Int=12,
                            ci_method::String="none", conf_level::Float64=0.95,
                            from_tag::String="",
                            output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    r = if isnothing(nfactors)
        println("Selecting number of factors via Bai-Ng information criteria...")
        ic = ic_criteria(X, min(20, size(X, 2)))
        optimal_r = ic.r_IC1
        println("  IC1 suggests $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Forecasting with static factor model: $r factors, horizon=$horizons, CI=$ci_method")
    println()

    fm = estimate_factors(X, r)
    fc = forecast(fm, horizons; ci_method=Symbol(ci_method), conf_level=conf_level)

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = fc.observables[:, vi]
    end

    if ci_method != "none" && !isnothing(fc.observables_lower)
        for (vi, vname) in enumerate(varnames)
            fc_df[!, "$(vname)_lower"] = fc.observables_lower[:, vi]
            fc_df[!, "$(vname)_upper"] = fc.observables_upper[:, vi]
        end
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Static Factor Forecast (h=$horizons, $(length(varnames)) variables)")

    if !isnothing(fc.observables_se)
        println()
        avg_se = round.(mean(fc.observables_se; dims=1)[1, :]; digits=4)
        println("Average forecast standard errors:")
        for (vi, vname) in enumerate(varnames)
            println("  $vname: $(avg_se[vi])")
        end
    end

    storage_save_auto!("forecast", Dict{String,Any}("type" => "static", "horizons" => horizons,
        "n_factors" => r),
        Dict{String,Any}("command" => "forecast static", "data" => data))
end

function _forecast_dynamic(; data::String, nfactors=nothing, horizons::Int=12,
                             factor_lags::Int=1, method::String="twostep",
                             from_tag::String="",
                             output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    r = if isnothing(nfactors)
        println("Selecting number of factors...")
        ic = ic_criteria(X, min(10, size(X, 2)))
        optimal_r = ic.r_IC1
        println("  Auto-selected $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Forecasting with dynamic factor model: $r factors, $factor_lags lags, method=$method, horizon=$horizons")
    println()

    fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    fc = forecast(fm, horizons)

    # Reconstruct observables via loadings
    factor_fc = fc isa NamedTuple ? fc.factors : fc
    if factor_fc isa AbstractMatrix
        obs_fc = factor_fc * fm.loadings'
    else
        obs_fc = reshape(factor_fc, horizons, r) * fm.loadings'
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = obs_fc[:, vi]
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Forecast (h=$horizons, $(length(varnames)) variables)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "dynamic", "horizons" => horizons,
        "n_factors" => r),
        Dict{String,Any}("command" => "forecast dynamic", "data" => data))
end

function _forecast_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                          horizons::Int=12, from_tag::String="",
                          output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

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

    println("Forecasting with GDFM: static rank=$r, dynamic rank=$q, horizon=$horizons")
    println()

    fm = estimate_gdfm(X, q; r=r)

    # GDFM forecast via AR(1) extrapolation on common component factors
    common = fm.common_component  # T x N
    T_obs, N = size(common)

    F_pca = svd(common)
    factors = F_pca.U[:, 1:r] .* F_pca.S[1:r]'
    loadings = F_pca.V[:, 1:r]

    obs_fc = zeros(horizons, N)
    for fi in 1:r
        f = factors[:, fi]
        y_ar = f[2:end]
        x_ar = [ones(length(y_ar)) f[1:end-1]]
        beta = x_ar \ y_ar
        f_last = f[end]
        for h in 1:horizons
            f_next = beta[1] + beta[2] * f_last
            obs_fc[h, :] .+= f_next .* loadings[:, fi]
            f_last = f_next
        end
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizons
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = round.(obs_fc[:, vi]; digits=6)
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="GDFM Forecast (h=$horizons, $(length(varnames)) variables)")

    println()
    var_shares = common_variance_share(fm)
    println("Average common variance share: $(round(mean(var_shares); digits=4))")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "gdfm", "horizons" => horizons,
        "static_rank" => r, "dynamic_rank" => q),
        Dict{String,Any}("command" => "forecast gdfm", "data" => data))
end

# ── Volatility Model Forecasts (NEW) ────────────────────

function _forecast_arch(; data::String, column::Int=1, q::Int=1, horizons::Int=12,
                          from_tag::String="",
                          output::String="", format::String="table")
    y, vname = _extract_series(data, column)

    println("ARCH($q) Volatility Forecast: variable=$vname, horizons=$horizons")
    println()

    model = estimate_arch(y, q)
    fc = forecast(model, horizons)

    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="ARCH($q) Volatility Forecast ($vname, h=$horizons)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "arch", "horizons" => horizons),
        Dict{String,Any}("command" => "forecast arch", "data" => data))
end

function _forecast_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                           horizons::Int=12, from_tag::String="",
                           output::String="", format::String="table")
    y, vname = _extract_series(data, column)

    println("GARCH($p,$q) Volatility Forecast: variable=$vname, horizons=$horizons")
    println()

    model = estimate_garch(y, p, q)
    fc = forecast(model, horizons)

    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="GARCH($p,$q) Volatility Forecast ($vname, h=$horizons)")

    uc = unconditional_variance(model)
    println()
    println("Unconditional variance: $(round(uc; digits=4))")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "garch", "horizons" => horizons),
        Dict{String,Any}("command" => "forecast garch", "data" => data))
end

function _forecast_egarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                            horizons::Int=12, from_tag::String="",
                            output::String="", format::String="table")
    y, vname = _extract_series(data, column)

    println("EGARCH($p,$q) Volatility Forecast: variable=$vname, horizons=$horizons")
    println()

    model = estimate_egarch(y, p, q)
    fc = forecast(model, horizons)

    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="EGARCH($p,$q) Volatility Forecast ($vname, h=$horizons)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "egarch", "horizons" => horizons),
        Dict{String,Any}("command" => "forecast egarch", "data" => data))
end

function _forecast_gjr_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                               horizons::Int=12, from_tag::String="",
                               output::String="", format::String="table")
    y, vname = _extract_series(data, column)

    println("GJR-GARCH($p,$q) Volatility Forecast: variable=$vname, horizons=$horizons")
    println()

    model = estimate_gjr_garch(y, p, q)
    fc = forecast(model, horizons)

    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="GJR-GARCH($p,$q) Volatility Forecast ($vname, h=$horizons)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "gjr_garch", "horizons" => horizons),
        Dict{String,Any}("command" => "forecast gjr_garch", "data" => data))
end

function _forecast_sv(; data::String, column::Int=1, draws::Int=5000,
                        horizons::Int=12, from_tag::String="",
                        output::String="", format::String="table")
    y, vname = _extract_series(data, column)

    println("Stochastic Volatility Forecast: variable=$vname, horizons=$horizons, draws=$draws")
    println()

    model = estimate_sv(y; n_draws=draws)
    fc = forecast(model, horizons)

    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="SV Volatility Forecast ($vname, h=$horizons)")

    storage_save_auto!("forecast", Dict{String,Any}("type" => "sv", "horizons" => horizons),
        Dict{String,Any}("command" => "forecast sv", "data" => data))
end
