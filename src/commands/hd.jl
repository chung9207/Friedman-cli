# Historical Decomposition commands: var, bvar, lp (action-first: friedman hd var ...)

function register_hd_commands!()
    hd_var = LeafCommand("var", _hd_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition of shocks")

    hd_bvar = LeafCommand("bvar", _hd_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian historical decomposition")

    hd_lp = LeafCommand("lp", _hd_lp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition via structural LP")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => hd_var,
        "bvar" => hd_bvar,
        "lp"   => hd_lp,
    )
    return NodeCommand("hd", subcmds, "Historical Decomposition")
end

# ── VAR HD ───────────────────────────────────────────────

function _hd_var(; data::String, lags=nothing, id::String="cholesky",
                  config::String="", from_tag::String="",
                  output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing Historical Decomposition: VAR($p), id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    hd_result = historical_decomposition(model, size(Y, 1) - p; kwargs...)

    report(hd_result)

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

    storage_save_auto!("hd", Dict{String,Any}("type" => "var", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd var", "data" => data))
end

# ── BVAR HD ──────────────────────────────────────────────

function _hd_bvar(; data::String, lags::Int=4, id::String="cholesky",
                   draws::Int=2000, sampler::String="nuts",
                   config::String="", from_tag::String="",
                   output::String="", format::String="table")
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    method = get(ID_METHOD_MAP, id, :cholesky)

    println("Computing Bayesian Historical Decomposition: BVAR($p), id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    horizon = size(Y, 1) - p

    bhd = historical_decomposition(post, horizon;
        method=method, quantiles=[0.16, 0.5, 0.84])

    report(bhd)

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

    storage_save_auto!("hd", Dict{String,Any}("type" => "bvar", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd bvar", "data" => data))
end

# ── LP HD ────────────────────────────────────────────────

function _hd_lp(; data::String, lags::Int=4, var_lags=nothing,
                 id::String="cholesky", vcov::String="newey_west", config::String="",
                 from_tag::String="",
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

    storage_save_auto!("hd", Dict{String,Any}("type" => "lp", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd lp", "data" => data))
end
