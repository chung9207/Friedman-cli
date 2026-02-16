# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# FEVD commands: var, bvar, lp (action-first: friedman fevd var ...)

function register_fevd_commands!()
    fevd_var = LeafCommand("var", _fevd_var;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute forecast error variance decomposition")

    fevd_bvar = LeafCommand("bvar", _fevd_bvar;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Bayesian forecast error variance decomposition")

    fevd_lp = LeafCommand("lp", _fevd_lp;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
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

    fevd_vecm = LeafCommand("vecm", _fevd_vecm;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("horizons"; short="h", type=Int, default=20, description="Forecast horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute FEVD via VECM → VAR representation")

    fevd_pvar = LeafCommand("pvar", _fevd_pvar;
        args=[Argument("data"; required=false, default="", description="Path to CSV panel data file")],
        options=[
            Option("id-col"; type=String, default="", description="Panel group identifier column"),
            Option("time-col"; type=String, default="", description="Time period column"),
            Option("lags"; short="p", type=Int, default=1, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=10, description="Forecast horizon"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute Panel VAR forecast error variance decomposition")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => fevd_var,
        "bvar" => fevd_bvar,
        "lp"   => fevd_lp,
        "vecm" => fevd_vecm,
        "pvar" => fevd_pvar,
    )
    return NodeCommand("fevd", subcmds, "Forecast Error Variance Decomposition")
end

# ── VAR FEVD ─────────────────────────────────────────────

function _fevd_var(; data::String, lags=nothing, horizons::Int=20,
                    id::String="cholesky", config::String="",
                    from_tag::String="",
                    output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing FEVD: VAR($p), horizons=$horizons, id=$id")
    println()

    # Arias identification: use identify_arias → irf_mean → compute FEVD from structural IRFs
    if id == "arias"
        isempty(config) && error("Arias identification requires a --config file with restrictions")
        cfg = load_config(config)
        id_cfg = get(cfg, "identification", Dict())
        zeros_list = get(id_cfg, "zero_restrictions", [])
        signs_list = get(id_cfg, "sign_restrictions", [])
        zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
        sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]
        restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
        arias_result = identify_arias(model, restrictions, horizons)
        irf_vals = irf_mean(arias_result)  # H x n x n
        n_h = size(irf_vals, 1)
        # Compute FEVD proportions from structural IRFs
        proportions = zeros(n, n, n_h)
        for h in 1:n_h
            total_var = zeros(n)
            for vi in 1:n
                for si in 1:n
                    cum_sq = sum(irf_vals[t, vi, si]^2 for t in 1:h)
                    proportions[vi, si, h] = cum_sq
                    total_var[vi] += cum_sq
                end
            end
            for vi in 1:n
                if total_var[vi] > 0
                    proportions[vi, :, h] ./= total_var[vi]
                end
            end
        end
        _output_fevd_tables(proportions, varnames, n_h;
                            id="arias", title_prefix="FEVD", format=format, output=output)
        storage_save_auto!("fevd", Dict{String,Any}("type" => "var", "id" => "arias",
            "horizons" => horizons, "n_vars" => n),
            Dict{String,Any}("command" => "fevd var", "data" => data))
        return
    end

    # Uhlig identification: use identify_uhlig → compute FEVD from structural IRFs
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
        uhlig_result = identify_uhlig(model, restrictions, horizons;
            n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
            max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
            tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"])
        irf_vals = uhlig_result.irf  # H x n x n
        n_h = size(irf_vals, 1)
        # Compute FEVD proportions from structural IRFs
        proportions = zeros(n, n, n_h)
        for h in 1:n_h
            total_var = zeros(n)
            for vi in 1:n
                for si in 1:n
                    cum_sq = sum(irf_vals[t, vi, si]^2 for t in 1:h)
                    proportions[vi, si, h] = cum_sq
                    total_var[vi] += cum_sq
                end
            end
            for vi in 1:n
                if total_var[vi] > 0
                    proportions[vi, :, h] ./= total_var[vi]
                end
            end
        end
        _output_fevd_tables(proportions, varnames, n_h;
                            id="uhlig", title_prefix="FEVD", format=format, output=output)
        storage_save_auto!("fevd", Dict{String,Any}("type" => "var", "id" => "uhlig",
            "horizons" => horizons, "n_vars" => n),
            Dict{String,Any}("command" => "fevd var", "data" => data))
        return
    end

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
                     id::String="cholesky", draws::Int=2000, sampler::String="direct",
                     config::String="", from_tag::String="",
                     output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
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
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
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

# ── VECM FEVD ───────────────────────────────────────────

function _fevd_vecm(; data::String, lags::Int=2, rank::String="auto",
                     deterministic::String="constant", horizons::Int=20,
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

    println("Computing VECM FEVD: rank=$r, VAR($p), horizons=$horizons, id=$id")
    println()

    kwargs = _build_identification_kwargs(id, config)
    fevd_result = fevd(var_model, horizons; kwargs...)

    report(fevd_result)

    _output_fevd_tables(fevd_result.proportions, varnames, horizons;
                        id=id, title_prefix="VECM FEVD", format=format, output=output)

    storage_save_auto!("fevd", Dict{String,Any}("type" => "vecm", "id" => id,
        "horizons" => horizons, "n_vars" => n, "rank" => r),
        Dict{String,Any}("command" => "fevd vecm", "data" => data))
end

# ── Panel VAR FEVD ─────────────────────────────────────────

function _fevd_pvar(; data::String, id_col::String="", time_col::String="",
                     lags::Int=1, horizons::Int=10,
                     from_tag::String="",
                     output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data_path, params = _resolve_from_tag(from_tag)
        data = data_path
        if isempty(id_col)
            id_col = get(params, "id_col", "")
        end
        if isempty(time_col)
            time_col = get(params, "time_col", "")
        end
        if lags == 1
            lags = get(params, "lags", lags)
        end
    end
    isempty(id_col) && error("Panel VAR FEVD requires --id-col")
    isempty(time_col) && error("Panel VAR FEVD requires --time-col")

    model, panel, varnames = _load_and_estimate_pvar(data, id_col, time_col, lags)
    n = length(varnames)

    println("Computing Panel VAR FEVD: horizons=$horizons")
    println()

    fevd_result = pvar_fevd(model, horizons)

    _output_fevd_tables(fevd_result.proportions, varnames, horizons;
                        id="cholesky", title_prefix="Panel VAR FEVD",
                        format=format, output=output)

    storage_save_auto!("fevd", Dict{String,Any}("type" => "pvar",
        "horizons" => horizons, "n_vars" => n),
        Dict{String,Any}("command" => "fevd pvar", "data" => data,
                          "id_col" => id_col, "time_col" => time_col, "lags" => lags))
end
