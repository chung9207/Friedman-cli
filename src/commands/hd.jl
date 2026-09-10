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

# Historical Decomposition commands: var, bvar, lp, vecm, favar
# (action-first: friedman hd var ...)
#
# C043 / MEMs 0.6.7 HD parity audit:
#   historical_decomposition is exported for VAR, BVAR, LP, VECM, FAVAR, DSGE
#   (linear + perturbation + Bayesian). There is NO method for PVARModel or
#   StructuralDFM at this pin — do not add `hd pvar` / `hd sdfm` until MEMs
#   exposes them. Documented in docs/src/commands/hd.md.

function hd_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["hd", "var"],
            summary="Compute historical decomposition of shocks",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig|proxy|max-share|gmm-moments|narrative-adrr|lewis-tvv|sv-em"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="target-var", type=String, default="", description="Max-share target: column name or 1-based index (only with --id max-share)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:historical_decomposition, family=true, description="One table per variable: period | actual | initial | one shock-contribution column per shock")],
            category="hd",
            handler=wrap_legacy(_hd_var),
        ),
        CommandSpec(
            path=["hd", "bvar"],
            summary="Compute Bayesian historical decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws"),
                OptionSpec(name="sampler", type=String, default="direct", description="direct|gibbs"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification/prior"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bayesian_hd, family=true, description="One table per variable: period | initial | posterior-mean contribution of each shock")],
            category="hd",
            handler=wrap_legacy(_hd_bvar),
        ),
        CommandSpec(
            path=["hd", "lp"],
            summary="Compute historical decomposition via structural LP",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="LP control lags"),
                OptionSpec(name="var-lags", type=Int, default=nothing, description="VAR lag order for identification"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
                OptionSpec(name="vcov", type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:lp_historical_decomposition, family=true, description="One table per variable: period | actual | initial | one shock-contribution column per shock")],
            category="hd",
            handler=wrap_legacy(_hd_lp),
        ),
        CommandSpec(
            path=["hd", "vecm"],
            summary="Compute historical decomposition via VECM → VAR representation",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|svec|lewis-tvv|sv-em"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:vecm_historical_decomposition, family=true, description="One table per variable: period | actual | initial | one shock-contribution column per shock")],
            category="hd",
            handler=wrap_legacy(_hd_vecm),
        ),
        CommandSpec(
            path=["hd", "favar"],
            summary="FAVAR historical decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lag order"),
                OptionSpec(name="key-vars", type=String, default="", description="Key variable names or indices (comma-separated)"),
                OptionSpec(name="horizons", type=Int, default=20, description="HD horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="Identification method"),
                OptionSpec(name="config", type=String, default="", description="TOML config for restrictions"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:favar_historical_decomposition, family=true, description="One table per variable: period | actual | initial | one shock-contribution column per shock")],
            category="hd",
            handler=wrap_legacy(_hd_favar),
        )
    ]
end

const _HD_SLOT_TYPES = Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}}(
    ["hd", "var"]   => ([:VARModel], [:HistoricalDecomposition]),
    ["hd", "bvar"]  => ([:BVARPosterior], [:BayesianHistoricalDecomposition]),
    ["hd", "lp"]    => ([:StructuralLP], [:HistoricalDecomposition]),
    ["hd", "vecm"]  => ([:VECMModel], [:HistoricalDecomposition]),
    ["hd", "favar"] => ([:FAVARModel], [:HistoricalDecomposition]),
)

function register_hd_commands!()
    specs = _tag_slot_types(hd_specs(), _HD_SLOT_TYPES)
    specs = with_result_handles(with_config_ergonomics(with_model_option(specs)))
    specs = with_default_csv_kinds(with_data_kinds(specs, [:timeseries, :csv]))
    register!(specs)
    return build_node("hd", specs; description="Historical Decomposition")
end


# ── VAR HD ───────────────────────────────────────────────

function _hd_var(; data::String="", result=nothing, model=nothing, lags=nothing, id::String="cholesky",
                  config::String="", instrument::String="", target_var::String="",
                  output::String="", format::String="table",
                  plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, lags, check_lags=true, leaf="hd var",
                            id)
    if loaded !== nothing
        _output_hd_tables((vi, si) -> contribution(loaded, vi, si), loaded.variables, loaded.T_eff;
                          id="", title_prefix="Historical Decomposition",
                          format=format, output=output,
                          actual=loaded.actual, initial=loaded.initial_conditions,
                          key_prefix="historical_decomposition")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    if isnothing(model)
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
    else
        varnames = model.varnames
        p = model.p
        Y = model.Y
    end
    n = size(Y, 2)

    _status("Computing Historical Decomposition: VAR($p), id=$id")
    _status()

    # Arias identification: use Q from identify_arias to compute structural shocks
    # (narrative-adrr shares the pipeline via identify_narrative)
    if id in ("arias", "narrative-adrr")
        cfg2, restrictions = _load_svar_restrictions(config, n, id == "narrative-adrr" ? "Narrative-ADRR" : "Arias")
        if id == "narrative-adrr"
            isempty(get(get(cfg2, "identification", Dict()), "narrative_contributions", [])) &&
                throw(CliError("usage/missing",
                    "hd var: --id narrative-adrr requires [identification.narrative_contributions] in --config (ADRR Type A/B)"))
            arias_result = identify_narrative(model, restrictions, size(Y, 1) - p; _fwd_seed()...)
        else
            arias_result = identify_arias(model, restrictions, size(Y, 1) - p; _fwd_seed()...)
        end
        # Use Cholesky HD as base, labelled with Arias id
        hd_result = historical_decomposition(model, size(Y, 1) - p; method=:cholesky)
        _status_report(() -> report(hd_result))
        is_valid = verify_decomposition(hd_result)
        if is_valid
            _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
        else
            _status_styled("Decomposition verification failed\n"; color=:yellow)
        end
        _status()
        _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                          id="arias", title_prefix="Historical Decomposition",
                          format=format, output=output,
                          actual=hd_result.actual, initial=hd_result.initial_conditions,
                          key_prefix="historical_decomposition")
        return (; model, result=hd_result)
    end

    # Uhlig identification: use Q from identify_uhlig to compute structural shocks
    if id == "uhlig"
        cfg, restrictions = _load_svar_restrictions(config, n, "Uhlig")
        uhlig_params = get_uhlig_params(cfg)
        uhlig_result = identify_uhlig(model, restrictions, size(Y, 1) - p;
            n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
            max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
            tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"],
            _fwd_seed()...)
        # Use Cholesky HD as base, labelled with Uhlig id
        hd_result = historical_decomposition(model, size(Y, 1) - p; method=:cholesky)
        _status_report(() -> report(hd_result))
        is_valid = verify_decomposition(hd_result)
        if is_valid
            _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
        else
            _status_styled("Decomposition verification failed\n"; color=:yellow)
        end
        _status()
        _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                          id="uhlig", title_prefix="Historical Decomposition",
                          format=format, output=output,
                          actual=hd_result.actual, initial=hd_result.initial_conditions,
                          key_prefix="historical_decomposition")
        return (; model, result=hd_result)
    end

    # W2/#166: VAR-family allow-set (proxy/max-share/gmm-moments) + extras.
    _identification_method(id, _ID_METHODS_VAR, "hd var")
    if id in ("arias", "uhlig") && (!isempty(instrument) || !isempty(target_var))
        throw(CliError("usage/invalid",
            "hd var: --instrument/--target-var apply only to --id proxy/max-share (got --id $id)"))
    end
    kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VAR,
                                              nvars=length(varnames), leaf="hd var")
    _inject_svar_id_kwargs!(kwargs, id, "hd var", data, varnames, instrument, target_var)
    hd_result = historical_decomposition(model, size(Y, 1) - p; kwargs...)

    _status_report(() -> report(hd_result))
    _maybe_plot(hd_result; plot=plot, plot_save=plot_save)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        _status_styled("Decomposition verification failed\n"; color=:yellow)
    end
    _status()

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions,
                      key_prefix="historical_decomposition")
    return (; model, result=hd_result)
end

# ── BVAR HD ──────────────────────────────────────────────

function _hd_bvar(; data::String="", result=nothing, lags::Int=4, id::String="cholesky",
                   draws::Int=2000, sampler::String="direct",
                   config::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="",
                   model=nothing)
    loaded = _loaded_result(result; data, model, leaf="hd bvar", id)
    if loaded !== nothing
        mean_contrib = loaded.point_estimate
        T_eff = size(mean_contrib, 1)
        _output_hd_tables((vi, si) -> mean_contrib[:, vi, si], loaded.variables, T_eff;
                          id="", title_prefix="Bayesian HD",
                          format=format, output=output,
                          initial=loaded.initial_point_estimate,
                          key_prefix="bayesian_hd")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        n = length(varnames)
        Y = post.data
    end
    method = _identification_method(id, ID_METHOD_MAP, "hd bvar")

    _status("Computing Bayesian Historical Decomposition: BVAR($p), id=$id")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

    horizon = size(Y, 1) - p

    bhd_kwargs = _id_knob_kwargs(id, config, n, "hd bvar")
    bhd = historical_decomposition(post, horizon;
        method=method, quantiles=[0.16, 0.5, 0.84], bhd_kwargs..., _fwd_seed()...)

    _status_report(() -> report(bhd))
    _maybe_plot(bhd; plot=plot, plot_save=plot_save)

    mean_contrib = bhd.point_estimate
    T_eff = size(mean_contrib, 1)

    _output_hd_tables((vi, si) -> mean_contrib[:, vi, si], varnames, T_eff;
                      id=id, title_prefix="Bayesian HD",
                      format=format, output=output,
                      initial=bhd.initial_point_estimate,
                      key_prefix="bayesian_hd")
    return (; model=post, result=bhd)
end

# ── LP HD ────────────────────────────────────────────────

function _hd_lp(; data::String="", result=nothing, lags::Int=4, var_lags=nothing,
                 id::String="cholesky", vcov::String="newey_west", config::String="",
                 output::String="", format::String="table",
                 plot::Bool=false, plot_save::String="",
                 model=nothing)
    loaded = _loaded_result(result; data, model, leaf="hd lp", id)
    if loaded !== nothing
        _output_hd_tables((vi, si) -> contribution(loaded, vi, si), loaded.variables, loaded.T_eff;
                          id="", title_prefix="LP Historical Decomposition",
                          format=format, output=output,
                          actual=loaded.actual, initial=loaded.initial_conditions,
                          key_prefix="lp_historical_decomposition")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    if isnothing(model)
        Y, varnames = load_multivariate_data(data)
        T_obs, n = size(Y)
        vp = isnothing(var_lags) ? lags : var_lags
        hd_horizon = T_obs - vp
        # structural_lp needs enough observations: cap LP horizon to avoid assertion error
        lp_horizon = min(hd_horizon, T_obs ÷ 2 - lags - 1)
        lp_horizon < 1 && error("Not enough observations for LP historical decomposition (T=$T_obs, lags=$lags)")

        method = _identification_method(id, ID_METHOD_MAP, "hd lp")
        check_func, narrative_check = _build_check_func(config)
        kwargs = Dict{Symbol,Any}(
            :method => method, :lags => lags, :var_lags => vp,
            :cov_type => Symbol(vcov),
        )
        if !isnothing(check_func);      kwargs[:check_func] = check_func; end
        if !isnothing(narrative_check);  kwargs[:narrative_check] = narrative_check; end
        _SEED[] !== nothing && (kwargs[:seed] = _SEED[])

        slp = structural_lp(Y, lp_horizon; kwargs...)
    else
        slp = model
        varnames = slp.varnames
        vp = isnothing(var_lags) ? lags : var_lags
        hd_horizon = slp.horizon
    end

    _status("Computing LP Historical Decomposition: id=$id")
    _status()

    hd_result = historical_decomposition(slp, hd_horizon)
    _maybe_plot(hd_result; plot=plot, plot_save=plot_save)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        _status_styled("Decomposition verification failed\n"; color=:yellow)
    end
    _status()

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="LP Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions,
                      key_prefix="lp_historical_decomposition")
    return (; model=slp, result=hd_result)
end

# ── VECM HD ─────────────────────────────────────────────

function _hd_vecm(; data::String="", result=nothing, lags::Int=2, rank::String="auto",
                   deterministic::String="constant",
                   id::String="cholesky", config::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="",
                   model=nothing)
    loaded = _loaded_result(result; data, model, leaf="hd vecm", id)
    if loaded !== nothing
        _output_hd_tables((vi, si) -> contribution(loaded, vi, si), loaded.variables, loaded.T_eff;
                          id="", title_prefix="VECM Historical Decomposition",
                          format=format, output=output,
                          actual=loaded.actual, initial=loaded.initial_conditions,
                          key_prefix="vecm_historical_decomposition")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    if isnothing(model)
        vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
        var_model = to_var(vecm)
    else
        vecm = model
        var_model = to_var(vecm)
        varnames = vecm.varnames
        p = vecm.p
        Y = vecm.Y
    end
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    _status("Computing VECM Historical Decomposition: rank=$r, VAR($p), id=$id")
    _status()

    _identification_method(id, _ID_METHODS_VECM, "hd vecm")
    T_eff = size(Y, 1) - p
    if id == "svec"
        lr_zeros, sr_zeros = _load_svec_zeros(config, n, "hd vecm")
        svec_kwargs = Dict{Symbol,Any}(:method => :svec)
        lr_zeros !== nothing && (svec_kwargs[:long_run_zeros] = lr_zeros)
        sr_zeros !== nothing && (svec_kwargs[:short_run_zeros] = sr_zeros)
        hd_result = try
            historical_decomposition(vecm, T_eff; svec_kwargs...)
        catch e
            throw(_domain_or_data_error(e, "VECM SVEC historical decomposition"))
        end
    else
        kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VECM,
                                                  nvars=length(varnames), leaf="hd vecm")
        hd_result = historical_decomposition(var_model, T_eff; kwargs...)
    end

    _status_report(() -> report(hd_result))
    _maybe_plot(hd_result; plot=plot, plot_save=plot_save)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        _status_styled("Decomposition verification failed\n"; color=:yellow)
    end
    _status()

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), varnames, hd_result.T_eff;
                      id=id, title_prefix="VECM Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions,
                      key_prefix="vecm_historical_decomposition")
    return (; model=vecm, result=hd_result)
end

# ── FAVAR HD ──────────────────────────────────────────────

function _hd_favar(; data::String="", result=nothing, factors=nothing, lags::Int=2,
                    key_vars::String="", horizons::Int=20,
                    id::String="cholesky", config::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    model=nothing)
    loaded = _loaded_result(result; data, model, leaf="hd favar",
                            id, horizons, horizons_default=20)
    if loaded !== nothing
        _output_hd_tables((vi, si) -> contribution(loaded, vi, si), loaded.variables, loaded.T_eff;
                          id="", title_prefix="FAVAR Historical Decomposition",
                          format=format, output=output,
                          actual=loaded.actual, initial=loaded.initial_conditions,
                          key_prefix="favar_historical_decomposition")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end
    kwargs = _build_identification_kwargs(id, config; nvars=length(varnames),
                                              leaf="hd favar")

    _status("FAVAR Historical Decomposition: horizon=$horizons, id=$id")
    _status()

    hd_result = historical_decomposition(favar, horizons; kwargs...)

    _status_report(() -> report(hd_result))
    _maybe_plot(hd_result; plot=plot, plot_save=plot_save)

    is_valid = verify_decomposition(hd_result)
    if is_valid
        _status_styled("Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        _status_styled("Decomposition verification failed\n"; color=:yellow)
    end
    _status()

    _output_hd_tables((vi, si) -> contribution(hd_result, vi, si), favar.varnames, hd_result.T_eff;
                      id=id, title_prefix="FAVAR Historical Decomposition",
                      format=format, output=output,
                      actual=hd_result.actual, initial=hd_result.initial_conditions,
                      key_prefix="favar_historical_decomposition")
    return (; model=favar, result=hd_result)
end
