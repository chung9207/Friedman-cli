# VAR commands: estimate, lagselect, stability, irf, fevd, hd

function register_var_commands!()
    var_estimate = LeafCommand("estimate", _var_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("trend"; type=String, default="constant", description="none|constant|trend|both"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a VAR(p) model")

    var_lagselect = LeafCommand("lagselect", _var_lagselect;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("max-lags"; type=Int, default=12, description="Maximum lag order to test"),
            Option("criterion"; type=String, default="aic", description="aic|bic|hqc"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Select optimal lag order for VAR")

    var_stability = LeafCommand("stability", _var_stability;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Check VAR stationarity (eigenvalues of companion matrix)")

    var_irf = LeafCommand("irf", _var_irf;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias"),
            Option("ci"; type=String, default="bootstrap", description="none|bootstrap|theoretical"),
            Option("replications"; type=Int, default=1000, description="Bootstrap replications"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute frequentist impulse response functions")

    var_fevd = LeafCommand("fevd", _var_fevd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute forecast error variance decomposition")

    var_hd = LeafCommand("hd", _var_hd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition of shocks")

    var_forecast = LeafCommand("forecast", _var_forecast;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("confidence"; type=Float64, default=0.95, description="Confidence level for intervals"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute h-step ahead VAR forecasts")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate"  => var_estimate,
        "lagselect" => var_lagselect,
        "stability" => var_stability,
        "irf"       => var_irf,
        "fevd"      => var_fevd,
        "hd"        => var_hd,
        "forecast"  => var_forecast,
    )
    return NodeCommand("var", subcmds, "Vector Autoregression (VAR) models")
end

# ── Estimation handlers ──────────────────────────────────────

function _var_estimate(; data::String, lags=nothing, trend::String="constant",
                        output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
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
    MacroEconometricModels.summary(model)

    coef_mat = coef(model)
    n_rows = size(coef_mat, 1)

    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end

    coef_df = DataFrame(permutedims(coef_mat), row_names)
    insertcols!(coef_df, 1, :equation => varnames)

    output_result(coef_df; format=Symbol(format), output=output, title="VAR($p) Coefficients")

    println()
    output_kv([
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
        "Log-likelihood" => loglikelihood(model),
    ]; format=format, title="Information Criteria")
end

function _var_lagselect(; data::String, max_lags::Int=12, criterion::String="aic",
                         format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    n = size(Y, 2)

    max_p = min(max_lags, size(Y,1) ÷ (3*n))
    crit_sym = Symbol(lowercase(criterion))

    println("Lag order selection (max lags: $max_p, criterion: $criterion)")
    println()

    results = []
    for p in 1:max_p
        try
            m = estimate_var(Y, p)
            push!(results, (p=p, aic=m.aic, bic=m.bic, hqc=m.hqic))
        catch
            continue
        end
    end

    if isempty(results)
        error("could not estimate VAR for any lag order 1:$max_p")
    end

    res_df = DataFrame(results)
    rename!(res_df, :p => :lags, :aic => :AIC, :bic => :BIC, :hqc => :HQC)

    optimal = select_lag_order(Y, max_p; criterion=crit_sym)

    output_result(res_df; format=Symbol(format), output=output, title="Lag Order Selection")
    println()
    printstyled("Optimal lag order ($criterion): $optimal\n"; bold=true)

    if format == "json"
        output_kv(Pair{String,Any}["optimal_lag" => optimal, "criterion" => criterion];
                  format=format, title="Optimal Lag")
    end
end

function _var_stability(; data::String, lags=nothing, format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    model = estimate_var(Y, p)
    result = is_stationary(model)

    println("VAR($p) Stationarity Check")
    println()

    eigenvalues = result.eigenvalues
    moduli = abs.(eigenvalues)

    eig_df = DataFrame(
        index=1:length(eigenvalues),
        eigenvalue=string.(round.(eigenvalues; digits=6)),
        modulus=round.(moduli; digits=6)
    )

    output_result(eig_df; format=Symbol(format), output=output, title="Companion Matrix Eigenvalues")
    println()

    if result.is_stationary
        printstyled("VAR($p) is stable (all eigenvalues inside unit circle)\n"; color=:green, bold=true)
    else
        printstyled("VAR($p) is NOT stable (eigenvalue(s) outside unit circle)\n"; color=:red, bold=true)
    end
    println("  Max modulus: $(round(maximum(moduli); digits=6))")
end

# ── Post-estimation: IRF ─────────────────────────────────────

function _var_irf(; data::String, lags=nothing, shock::Int=1, horizons::Int=20,
                   id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                   config::String="", output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing IRFs: VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    println()

    # Arias identification handled separately
    if id == "arias"
        _var_irf_arias(model, config, horizons, varnames, shock; format=format, output=output)
        return
    end

    kwargs = _build_identification_kwargs(id, config)
    kwargs[:ci_type] = Symbol(ci)
    kwargs[:reps] = replications

    irf_result = irf(model, horizons; kwargs...)

    MacroEconometricModels.summary(irf_result)

    irf_vals = irf_result.values  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    if ci != "none" && !isnothing(irf_result.ci_lower)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock]
        end
    end

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock ($id identification)")
end

function _var_irf_arias(model::VARModel, config::String, horizons::Int,
                        varnames::Vector{String}, shock::Int; format::String="table", output::String="")
    isempty(config) && error("Arias identification requires a --config file with restrictions")
    cfg = load_config(config)
    id_cfg = get(cfg, "identification", Dict())

    zeros_list = get(id_cfg, "zero_restrictions", [])
    signs_list = get(id_cfg, "sign_restrictions", [])

    n = nvars(model)
    zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
    sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]

    restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
    result = identify_arias(model, restrictions, horizons)

    irf_vals = result.irf  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Arias et al. identification)")
end

# ── Post-estimation: FEVD ────────────────────────────────────

function _var_fevd(; data::String, lags=nothing, horizons::Int=20,
                    id::String="cholesky", config::String="",
                    output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing FEVD: VAR($p), horizons=$horizons, id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    fevd_result = fevd(model, horizons; kwargs...)

    MacroEconometricModels.summary(fevd_result)

    proportions = fevd_result.proportions  # n_vars x n_shocks x H

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
                      title="FEVD for $vname ($id identification)")
        println()
    end
end

# ── Post-estimation: HD ──────────────────────────────────────

function _var_hd(; data::String, lags=nothing, id::String="cholesky",
                  config::String="", output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing Historical Decomposition: VAR($p), id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    hd_result = historical_decomposition(model, size(Y, 1) - p; kwargs...)

    MacroEconometricModels.summary(hd_result)

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
                      title="Historical Decomposition: $vname ($id identification)")
        println()
    end
end

# ── Post-estimation: Forecast ──────────────────────────────────

function _var_forecast(; data::String, lags=nothing, horizons::Int=12,
                        confidence::Float64=0.95, output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing VAR($p) forecast: horizons=$horizons, confidence=$confidence")
    println()

    B = coef(model)
    forecasts = _var_forecast_point(B, Y, p, horizons)

    # Forecast error variance from MA(∞) representation for confidence intervals
    Sigma = model.Sigma
    alpha = 1.0 - confidence
    z = quantile_normal(1.0 - alpha / 2.0)

    # Compute cumulative forecast MSE via companion form
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
