# FEVD commands: compute

function register_fevd_commands!()
    fevd_compute = LeafCommand("compute", _fevd_compute;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("draws"; type=Int, default=2000, description="MCMC draws (Bayesian mode)"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc (Bayesian mode)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        flags=[
            Flag("bayesian"; short="b", description="Use Bayesian estimation (BVAR + posterior FEVD)"),
        ],
        description="Compute forecast error variance decomposition")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "compute" => fevd_compute,
    )
    return NodeCommand("fevd", subcmds, "Forecast Error Variance Decomposition (FEVD)")
end

function _fevd_compute(; data::String, lags=nothing, horizons::Int=20,
                        id::String="cholesky", config::String="",
                        draws::Int=2000, sampler::String="nuts",
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

    # Bayesian FEVD via BVAR
    if bayesian
        _fevd_bayesian(Y, p, n, horizons, varnames;
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

    println("Computing FEVD: VAR($p), horizons=$horizons, id=$id")
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

    fevd_result = fevd(model, horizons; kwargs...)

    MacroEconometricModels.summary(fevd_result)

    # Output FEVD proportions for each variable
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

function _fevd_bayesian(Y::Matrix{Float64}, p::Int, n::Int, horizons::Int,
                        varnames::Vector{String};
                        id::String="cholesky", config::String="",
                        draws::Int=2000, sampler::String="nuts",
                        format::String="table", output::String="")
    println("Computing Bayesian FEVD: BVAR($p), horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? :normal : :minnesota

    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_samples=draws,
        prior=prior_sym, hyper=prior_obj)

    bfevd = fevd(chain, p, n, horizons;
        data=Y, quantiles=[0.16, 0.5, 0.84])

    MacroEconometricModels.summary(bfevd)

    # Output posterior mean FEVD proportions for each variable
    # bfevd.mean: n_vars x n_shocks x H
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
