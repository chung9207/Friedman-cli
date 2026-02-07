# FEVD commands: var, bvar, lp (action-first: friedman fevd var ...)

function register_fevd_commands!()
    fevd_var = LeafCommand("var", _fevd_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute forecast error variance decomposition")

    fevd_bvar = LeafCommand("bvar", _fevd_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian forecast error variance decomposition")

    fevd_lp = LeafCommand("lp", _fevd_lp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute forecast error variance decomposition via structural LP")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => fevd_var,
        "bvar" => fevd_bvar,
        "lp"   => fevd_lp,
    )
    return NodeCommand("fevd", subcmds, "Forecast Error Variance Decomposition")
end

# ── VAR FEVD ─────────────────────────────────────────────

function _fevd_var(; data::String, lags=nothing, horizons::Int=20,
                    id::String="cholesky", config::String="",
                    from_tag::String="",
                    output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing FEVD: VAR($p), horizons=$horizons, id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    fevd_result = fevd(model, horizons; kwargs...)

    report(fevd_result)

    _output_fevd_tables(fevd_result.proportions, varnames, horizons;
                        id=id, title_prefix="FEVD", format=format, output=output)

    storage_save_auto!("fevd", Dict{String,Any}("type" => "var", "id" => id,
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "fevd var", "data" => data))
end

# ── BVAR FEVD ────────────────────────────────────────────

function _fevd_bvar(; data::String, lags::Int=4, horizons::Int=20,
                     id::String="cholesky", draws::Int=2000, sampler::String="nuts",
                     config::String="", from_tag::String="",
                     output::String="", format::String="table")
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

    println("Computing Bayesian FEVD: BVAR($p), horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    bfevd = fevd(post, horizons;
        quantiles=[0.16, 0.5, 0.84])

    report(bfevd)

    _output_fevd_tables(bfevd.mean, varnames, horizons;
                        id=id, title_prefix="Bayesian FEVD", format=format, output=output)

    storage_save_auto!("fevd", Dict{String,Any}("type" => "bvar", "id" => id,
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "fevd bvar", "data" => data))
end

# ── LP FEVD ──────────────────────────────────────────────

function _fevd_lp(; data::String, horizons::Int=20, lags::Int=4, var_lags=nothing,
                   id::String="cholesky", vcov::String="newey_west", config::String="",
                   from_tag::String="",
                   output::String="", format::String="table")
    slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
        id, vcov, config)
    n = size(Y, 2)

    println("Computing LP FEVD: horizons=$horizons, id=$id")
    println()

    fevd_result = lp_fevd(slp, horizons)

    _output_fevd_tables(fevd_result.bias_corrected, varnames, horizons;
                        id=id, title_prefix="LP FEVD", format=format, output=output)

    storage_save_auto!("fevd", Dict{String,Any}("type" => "lp", "id" => id,
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "fevd lp", "data" => data))
end
