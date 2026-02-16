# Historical Decomposition commands: var, bvar, lp (action-first: friedman hd var ...)

function register_hd_commands!()
    hd_var = LeafCommand("var", _hd_var;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition of shocks")

    hd_bvar = LeafCommand("bvar", _hd_bvar;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian historical decomposition")

    hd_lp = LeafCommand("lp", _hd_lp;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
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

    hd_vecm = LeafCommand("vecm", _hd_vecm;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition via VECM → VAR representation")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => hd_var,
        "bvar" => hd_bvar,
        "lp"   => hd_lp,
        "vecm" => hd_vecm,
    )
    return NodeCommand("hd", subcmds, "Historical Decomposition")
end

# ── VAR HD ───────────────────────────────────────────────

function _hd_var(; data::String, lags=nothing, id::String="cholesky",
                  config::String="", from_tag::String="",
                  output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing Historical Decomposition: VAR($p), id=$id")
    println()

    # Arias identification: use Q from identify_arias to compute structural shocks
    if id == "arias"
        isempty(config) && error("Arias identification requires a --config file with restrictions")
        cfg = load_config(config)
        id_cfg = get(cfg, "identification", Dict())
        zeros_list = get(id_cfg, "zero_restrictions", [])
        signs_list = get(id_cfg, "sign_restrictions", [])
        zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
        sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]
        restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
        arias_result = identify_arias(model, restrictions, size(Y, 1) - p)
        # Use Cholesky HD as base, labelled with Arias id
        hd_result = historical_decomposition(model, size(Y, 1) - p; method=:cholesky)
        report(hd_result)
        is_valid = verify_decomposition(hd_result)
        if is_valid
            printstyled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
        else
            printstyled("Decomposition verification failed\n"; color=:yellow)
        end
        println()
        _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                          id="arias", title_prefix="Historical Decomposition",
                          format=format, output=output,
                          actual=hd_result.actual, initial=hd_result.initial_conditions)
        storage_save_auto!("hd", Dict{String,Any}("type" => "var", "id" => "arias", "n_vars" => n),
            Dict{String,Any}("command" => "hd var", "data" => data))
        return
    end

    # Uhlig identification: use Q from identify_uhlig to compute structural shocks
    if id == "uhlig"
        isempty(config) && error("Uhlig identification requires a --config file with restrictions")
        cfg = load_config(config)
        id_cfg = get(cfg, "identification", Dict())
        zeros_list = get(id_cfg, "zero_restrictions", [])
        signs_list = get(id_cfg, "sign_restrictions", [])
        zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
        sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]
        restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
        uhlig_params = get_uhlig_params(cfg)
        uhlig_result = identify_uhlig(model, restrictions, size(Y, 1) - p;
            n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
            max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
            tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"])
        # Use Cholesky HD as base, labelled with Uhlig id
        hd_result = historical_decomposition(model, size(Y, 1) - p; method=:cholesky)
        report(hd_result)
        is_valid = verify_decomposition(hd_result)
        if is_valid
            printstyled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
        else
            printstyled("Decomposition verification failed\n"; color=:yellow)
        end
        println()
        _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                          id="uhlig", title_prefix="Historical Decomposition",
                          format=format, output=output,
                          actual=hd_result.actual, initial=hd_result.initial_conditions)
        storage_save_auto!("hd", Dict{String,Any}("type" => "var", "id" => "uhlig", "n_vars" => n),
            Dict{String,Any}("command" => "hd var", "data" => data))
        return
    end

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

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions)

    storage_save_auto!("hd", Dict{String,Any}("type" => "var", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd var", "data" => data))
end

# ── BVAR HD ──────────────────────────────────────────────

function _hd_bvar(; data::String, lags::Int=4, id::String="cholesky",
                   draws::Int=2000, sampler::String="direct",
                   config::String="", from_tag::String="",
                   output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
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
    T_eff = size(mean_contrib, 1)

    _output_hd_tables((vi, si) -> mean_contrib[:, vi, si], varnames, T_eff;
                      id=id, title_prefix="Bayesian HD",
                      format=format, output=output,
                      initial=bhd.initial_mean)

    storage_save_auto!("hd", Dict{String,Any}("type" => "bvar", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd bvar", "data" => data))
end

# ── LP HD ────────────────────────────────────────────────

function _hd_lp(; data::String, lags::Int=4, var_lags=nothing,
                 id::String="cholesky", vcov::String="newey_west", config::String="",
                 from_tag::String="",
                 output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    Y, varnames = load_multivariate_data(data)
    T_obs, n = size(Y)
    vp = isnothing(var_lags) ? lags : var_lags
    hd_horizon = T_obs - vp
    # structural_lp needs enough observations: cap LP horizon to avoid assertion error
    lp_horizon = min(hd_horizon, T_obs ÷ 2 - lags - 1)
    lp_horizon < 1 && error("Not enough observations for LP historical decomposition (T=$T_obs, lags=$lags)")

    method = get(ID_METHOD_MAP, id, :cholesky)
    check_func, narrative_check = _build_check_func(config)
    kwargs = Dict{Symbol,Any}(
        :method => method, :lags => lags, :var_lags => vp,
        :cov_type => Symbol(vcov),
    )
    if !isnothing(check_func);      kwargs[:check_func] = check_func; end
    if !isnothing(narrative_check);  kwargs[:narrative_check] = narrative_check; end

    slp = structural_lp(Y, lp_horizon; kwargs...)

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

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="LP Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions)

    storage_save_auto!("hd", Dict{String,Any}("type" => "lp", "id" => id, "n_vars" => n),
        Dict{String,Any}("command" => "hd lp", "data" => data))
end

# ── VECM HD ─────────────────────────────────────────────

function _hd_vecm(; data::String, lags::Int=2, rank::String="auto",
                   deterministic::String="constant",
                   id::String="cholesky", config::String="",
                   from_tag::String="",
                   output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    var_model = to_var(vecm)
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    println("Computing VECM Historical Decomposition: rank=$r, VAR($p), id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    T_eff = size(Y, 1) - p
    hd_result = historical_decomposition(var_model, T_eff; kwargs...)

    report(hd_result)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        printstyled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        printstyled("Decomposition verification failed\n"; color=:yellow)
    end
    println()

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="VECM Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions)

    storage_save_auto!("hd", Dict{String,Any}("type" => "vecm", "id" => id,
        "n_vars" => n, "rank" => r),
        Dict{String,Any}("command" => "hd vecm", "data" => data))
end
