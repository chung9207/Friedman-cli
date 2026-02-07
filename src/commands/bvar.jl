# BVAR commands: estimate, posterior, irf, fevd, hd

function register_bvar_commands!()
    bvar_estimate = LeafCommand("estimate", _bvar_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("prior"; type=String, default="minnesota", description="Prior type: minnesota"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a Bayesian VAR model")

    bvar_posterior = LeafCommand("posterior", _bvar_posterior;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("method"; type=String, default="mean", description="mean|median"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Extract posterior summary from Bayesian VAR")

    bvar_irf = LeafCommand("irf", _bvar_irf;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian impulse response functions with credible intervals")

    bvar_fevd = LeafCommand("fevd", _bvar_fevd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian forecast error variance decomposition")

    bvar_hd = LeafCommand("hd", _bvar_hd;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian historical decomposition")

    bvar_forecast = LeafCommand("forecast", _bvar_forecast;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian h-step ahead forecasts with credible intervals")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate"  => bvar_estimate,
        "posterior" => bvar_posterior,
        "irf"       => bvar_irf,
        "fevd"      => bvar_fevd,
        "hd"        => bvar_hd,
        "forecast"  => bvar_forecast,
    )
    return NodeCommand("bvar", subcmds, "Bayesian Vector Autoregression (BVAR)")
end

# ── Estimation handlers ──────────────────────────────────────

function _bvar_estimate(; data::String, lags::Int=4, prior::String="minnesota",
                         draws::Int=2000, sampler::String="nuts",
                         config::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)
    p = lags

    println("Estimating Bayesian VAR($p) with $n variables: $(join(varnames, ", "))")
    println("Prior: $prior, Sampler: $sampler, Draws: $draws")
    println()

    prior_obj = _build_prior(config, Y, p)

    prior_sym = isnothing(prior_obj) ? Symbol(prior) : :minnesota
    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler),
        n_samples=draws,
        prior=prior_sym,
        hyper=prior_obj)

    mean_model = posterior_mean_model(chain, p, n; data=Y)
    summary(mean_model)

    coef_mat = coef(mean_model)
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

    output_result(coef_df; format=Symbol(format), output=output, title="BVAR($p) Posterior Mean Coefficients")
end

function _bvar_posterior(; data::String, lags::Int=4, draws::Int=2000,
                          sampler::String="nuts", method::String="mean",
                          config::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)
    p = lags

    println("Estimating BVAR($p) and extracting posterior $method...")
    println()

    prior_obj = _build_prior(config, Y, p)

    prior_sym = isnothing(prior_obj) ? :normal : :minnesota
    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler),
        n_samples=draws,
        prior=prior_sym,
        hyper=prior_obj)

    model = if method == "median"
        posterior_median_model(chain, p, n; data=Y)
    else
        posterior_mean_model(chain, p, n; data=Y)
    end

    summary(model)

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

    output_result(coef_df; format=Symbol(format), output=output, title="BVAR($p) Posterior $(titlecase(method)) Coefficients")

    println()
    output_kv([
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
    ]; format=format, title="Information Criteria (Posterior $(titlecase(method)))")
end

# ── Post-estimation: Bayesian IRF ────────────────────────────

function _bvar_irf(; data::String, lags::Int=4, shock::Int=1, horizons::Int=20,
                    id::String="cholesky", draws::Int=2000, sampler::String="nuts",
                    config::String="", output::String="", format::String="table")
    chain, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    method = get(ID_METHOD_MAP, id, :cholesky)

    println("Computing Bayesian IRFs: BVAR($p), shock=$shock, horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :data => Y,
        :quantiles => [0.16, 0.5, 0.84],
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    birf = irf(chain, p, n, horizons; kwargs...)

    MacroEconometricModels.summary(birf)

    irf_mean = birf.mean
    n_h = size(irf_mean, 1)
    q_levels = birf.quantile_levels
    q_idx_lo = findfirst(==(0.16), q_levels)
    q_idx_med = findfirst(==(0.5), q_levels)
    q_idx_hi = findfirst(==(0.84), q_levels)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)

    for (vi, vname) in enumerate(varnames)
        if !isnothing(q_idx_med)
            irf_df[!, vname] = birf.quantiles[:, vi, shock, q_idx_med]
        else
            irf_df[!, vname] = irf_mean[:, vi, shock]
        end
        if !isnothing(q_idx_lo)
            irf_df[!, "$(vname)_16pct"] = birf.quantiles[:, vi, shock, q_idx_lo]
        end
        if !isnothing(q_idx_hi)
            irf_df[!, "$(vname)_84pct"] = birf.quantiles[:, vi, shock, q_idx_hi]
        end
    end

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Bayesian IRF to $shock_name shock ($id, 68% credible interval)")
end

# ── Post-estimation: Bayesian FEVD ───────────────────────────

function _bvar_fevd(; data::String, lags::Int=4, horizons::Int=20,
                     id::String="cholesky", draws::Int=2000, sampler::String="nuts",
                     config::String="", output::String="", format::String="table")
    chain, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

    println("Computing Bayesian FEVD: BVAR($p), horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    bfevd = fevd(chain, p, n, horizons;
        data=Y, quantiles=[0.16, 0.5, 0.84])

    MacroEconometricModels.summary(bfevd)

    mean_props = bfevd.mean

    for vi in 1:n
        fevd_df = DataFrame()
        fevd_df.horizon = 1:horizons
        for si in 1:n
            shock_name = si <= length(varnames) ? varnames[si] : "shock_$si"
            fevd_df[!, shock_name] = mean_props[vi, si, :]
        end
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        output_result(fevd_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(vname)."),
                      title="Bayesian FEVD for $vname ($id, posterior mean)")
        println()
    end
end

# ── Post-estimation: Bayesian HD ─────────────────────────────

function _bvar_hd(; data::String, lags::Int=4, id::String="cholesky",
                   draws::Int=2000, sampler::String="nuts",
                   config::String="", output::String="", format::String="table")
    chain, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    method = get(ID_METHOD_MAP, id, :cholesky)

    println("Computing Bayesian Historical Decomposition: BVAR($p), id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    horizon = size(Y, 1) - p

    bhd = historical_decomposition(chain, p, n, horizon;
        data=Y, method=method, quantiles=[0.16, 0.5, 0.84])

    MacroEconometricModels.summary(bhd)

    mean_contrib = bhd.mean
    initial_mean = bhd.initial_mean
    T_eff = size(mean_contrib, 1)

    for vi in 1:n
        hd_df = DataFrame()
        hd_df.period = 1:T_eff
        hd_df.initial = initial_mean[:, vi]

        for si in 1:n
            shock_name = si <= length(varnames) ? varnames[si] : "shock_$si"
            hd_df[!, "contrib_$shock_name"] = mean_contrib[:, vi, si]
        end

        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        output_result(hd_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(vname)."),
                      title="Bayesian HD: $vname ($id, posterior mean)")
        println()
    end
end

# ── Post-estimation: Bayesian Forecast ─────────────────────────

function _bvar_forecast(; data::String, lags::Int=4, horizons::Int=12,
                         draws::Int=2000, sampler::String="nuts",
                         config::String="", output::String="", format::String="table")
    chain, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

    println("Computing Bayesian forecast: BVAR($p), horizons=$horizons")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    # Extract posterior draws and forecast from each
    b_vecs, sigmas = MacroEconometricModels.extract_chain_parameters(chain)
    n_draws = length(b_vecs)
    all_forecasts = zeros(n_draws, horizons, n)

    for d in 1:n_draws
        model_d = MacroEconometricModels.parameters_to_model(b_vecs[d], sigmas[d], p, n; data=Y)
        B_d = coef(model_d)
        all_forecasts[d, :, :] = _var_forecast_point(B_d, Y, p, horizons)
    end

    # Compute posterior mean and quantiles
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
end
