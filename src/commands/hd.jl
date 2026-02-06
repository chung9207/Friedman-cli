# Historical Decomposition commands: compute

function register_hd_commands!()
    hd_compute = LeafCommand("compute", _hd_compute;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("draws"; type=Int, default=2000, description="MCMC draws (Bayesian mode)"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc (Bayesian mode)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        flags=[
            Flag("bayesian"; short="b", description="Use Bayesian estimation (BVAR + posterior HD)"),
        ],
        description="Compute historical decomposition of shocks")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "compute" => hd_compute,
    )
    return NodeCommand("hd", subcmds, "Historical Decomposition (HD)")
end

function _hd_compute(; data::String, lags=nothing, id::String="cholesky",
                      config::String="", draws::Int=2000, sampler::String="nuts",
                      bayesian::Bool=false, output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    # Bayesian HD via BVAR
    if bayesian
        _hd_bayesian(Y, p, n, varnames;
            id=id, config=config, draws=draws, sampler=sampler,
            format=format, output=output)
        return
    end

    id_method = Dict(
        "cholesky" => :cholesky,
        "sign" => :sign,
        "narrative" => :narrative,
        "longrun" => :long_run,
    )
    method = get(id_method, id, :cholesky)

    println("Computing Historical Decomposition: VAR($p), id=$id")
    println()

    model = estimate_var(Y, p)

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(:method => method)
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    hd_result = historical_decomposition(model, size(Y, 1) - p; kwargs...)

    MacroEconometricModels.summary(hd_result)

    # Verify decomposition
    is_valid = verify_decomposition(hd_result)
    if is_valid
        printstyled("✓ Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        printstyled("⚠ Decomposition verification failed\n"; color=:yellow)
    end
    println()

    # Output contributions for each variable
    for vi in 1:n
        T_eff = hd_result.T_eff
        hd_df = DataFrame()
        hd_df.period = 1:T_eff

        # Actual values
        hd_df.actual = hd_result.actual[:, vi]

        # Initial conditions
        hd_df.initial = hd_result.initial_conditions[:, vi]

        # Shock contributions
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

function _hd_bayesian(Y::Matrix{Float64}, p::Int, n::Int, varnames::Vector{String};
                      id::String="cholesky", config::String="",
                      draws::Int=2000, sampler::String="nuts",
                      format::String="table", output::String="")
    id_method = Dict(
        "cholesky" => :cholesky,
        "sign" => :sign,
        "narrative" => :narrative,
        "longrun" => :long_run,
    )
    method = get(id_method, id, :cholesky)

    println("Computing Bayesian Historical Decomposition: BVAR($p), id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? :normal : :minnesota

    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_samples=draws,
        prior=prior_sym, hyper=prior_obj)

    horizon = size(Y, 1) - p

    bhd = historical_decomposition(chain, p, n, horizon;
        data=Y, method=method, quantiles=[0.16, 0.5, 0.84])

    MacroEconometricModels.summary(bhd)

    # Output posterior mean contributions for each variable
    # bhd.mean: contributions (T_eff x n_vars x n_shocks)
    # bhd.initial_mean: initial conditions (T_eff x n_vars)
    mean_contrib = bhd.mean
    initial_mean = bhd.initial_mean
    T_eff = size(mean_contrib, 1)

    for vi in 1:n
        hd_df = DataFrame()
        hd_df.period = 1:T_eff

        # Initial conditions (posterior mean)
        hd_df.initial = initial_mean[:, vi]

        # Shock contributions (posterior mean)
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
