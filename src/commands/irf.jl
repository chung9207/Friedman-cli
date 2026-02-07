# IRF commands: var, bvar, lp (action-first: friedman irf var ...)

function register_irf_commands!()
    irf_var = LeafCommand("var", _irf_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|fastica|jade|sobi|dcov|hsic|student_t|mixture_normal|pml|skew_normal|markov_switching|garch_id"),
            Option("ci"; type=String, default="bootstrap", description="none|bootstrap|theoretical"),
            Option("replications"; type=Int, default=1000, description="Bootstrap replications"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute frequentist impulse response functions")

    irf_bvar = LeafCommand("bvar", _irf_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian impulse response functions with credible intervals")

    irf_lp = LeafCommand("lp", _irf_lp;
        args=[Argument("data"; description="Path to CSV data file")],
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
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute structural LP impulse response functions")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => irf_var,
        "bvar" => irf_bvar,
        "lp"   => irf_lp,
    )
    return NodeCommand("irf", subcmds, "Impulse Response Functions")
end

# ── VAR IRF ──────────────────────────────────────────────

function _irf_var(; data::String, lags=nothing, shock::Int=1, horizons::Int=20,
                   id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                   config::String="", from_tag::String="",
                   output::String="", format::String="table")
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

    report(irf_result)

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

    # Auto-save
    storage_save_auto!("irf", Dict{String,Any}("type" => "var", "id" => id, "shock" => shock,
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "irf var", "data" => data))
end

function _var_irf_arias(model, config::String, horizons::Int,
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

    irf_vals = irf_mean(result)  # H x n x n
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

# ── BVAR IRF ─────────────────────────────────────────────

function _irf_bvar(; data::String, lags::Int=4, shock::Int=1, horizons::Int=20,
                    id::String="cholesky", draws::Int=2000, sampler::String="nuts",
                    config::String="", from_tag::String="",
                    output::String="", format::String="table")
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    method = get(ID_METHOD_MAP, id, :cholesky)

    println("Computing Bayesian IRFs: BVAR($p), shock=$shock, horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :quantiles => [0.16, 0.5, 0.84],
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    birf = irf(post, horizons; kwargs...)

    report(birf)

    irf_mean_vals = birf.mean
    n_h = size(irf_mean_vals, 1)
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
            irf_df[!, vname] = irf_mean_vals[:, vi, shock]
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

    storage_save_auto!("irf", Dict{String,Any}("type" => "bvar", "id" => id, "shock" => shock,
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "irf bvar", "data" => data))
end

# ── LP IRF ───────────────────────────────────────────────

function _irf_lp(; data::String, shock::Int=1, shocks::String="",
                  horizons::Int=20, lags::Int=4, var_lags=nothing,
                  id::String="cholesky", ci::String="none",
                  replications::Int=200, conf_level::Float64=0.95,
                  vcov::String="newey_west", config::String="",
                  from_tag::String="",
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

    storage_save_auto!("irf", Dict{String,Any}("type" => "lp", "id" => id,
        "shocks" => shock_indices, "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "irf lp", "data" => data))
end
