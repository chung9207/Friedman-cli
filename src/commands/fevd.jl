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

# FEVD commands: var, bvar, lp, vecm, pvar, favar, sdfm (action-first: friedman fevd var ...)

function fevd_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["fevd", "var"],
            summary="Compute forecast error variance decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
                OptionSpec(name="horizons", type=Int, default=20, description="Forecast horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig|proxy|max-share|gmm-moments|narrative-adrr|lewis-tvv|sv-em"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="target-var", type=String, default="", description="Max-share target: column name or 1-based index (only with --id max-share)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser"),
                FlagSpec(name="generalized", description="Pesaran-Shin generalized FEVD (identification-free; shares do NOT sum to 1)"),
                FlagSpec(name="normalize", description="Rescale generalized shares to sum to 1 per variable")
            ],
            tables=[TableSpec(name=:fevd, description="Variance shares in tidy long form: horizon | variable | shock | value"),
                    TableSpec(name=:generalized_fevd, description="Pesaran-Shin generalized variance shares (--generalized); tidy long form"),
                    TableSpec(name=:fevd_by_variable, family=true, description="One wide table per variable (horizon | one column per shock) under --id arias|uhlig")],
            category="fevd",
            handler=wrap_legacy(_fevd_var),
        ),
        CommandSpec(
            path=["fevd", "bvar"],
            summary="Compute Bayesian forecast error variance decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
                OptionSpec(name="horizons", type=Int, default=20, description="Forecast horizon"),
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
            tables=[TableSpec(name=:bayesian_fevd, family=true, description="Posterior-mean variance shares, one wide table per variable (horizon | one column per shock)")],
            category="fevd",
            handler=wrap_legacy(_fevd_bvar),
        ),
        CommandSpec(
            path=["fevd", "lp"],
            summary="Compute forecast error variance decomposition via structural LP",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="horizons", type=Int, default=20, description="Forecast horizon"),
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
            tables=[TableSpec(name=:lp_fevd, family=true, description="Bias-corrected LP variance shares, one wide table per variable (horizon | one column per shock)")],
            category="fevd",
            handler=wrap_legacy(_fevd_lp),
        ),
        CommandSpec(
            path=["fevd", "vecm"],
            summary="Compute FEVD via VECM → VAR representation",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="horizons", type=Int, default=20, description="Forecast horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|svec|lewis-tvv|sv-em"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:vecm_fevd, description="Variance shares of the VECM's VAR representation, tidy long form: horizon | variable | shock | value")],
            category="fevd",
            handler=wrap_legacy(_fevd_vecm),
        ),
        CommandSpec(
            path=["fevd", "pvar"],
            summary="Compute Panel VAR forecast error variance decomposition",
            args=[ArgSpec(name="data", description="Path to CSV panel data file")],
            options=[
                OptionSpec(name="id-col", type=String, default="", description="Panel group identifier column"),
                OptionSpec(name="time-col", type=String, default="", description="Time period column"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Lag order"),
                OptionSpec(name="horizons", type=Int, default=10, description="Forecast horizon"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:panel_var_fevd, family=true, description="Panel VAR variance shares, one wide table per variable (horizon | one column per shock)")],
            category="fevd",
            handler=wrap_legacy(_fevd_pvar),
        ),
        CommandSpec(
            path=["fevd", "favar"],
            summary="FAVAR forecast error variance decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lag order"),
                OptionSpec(name="key-vars", type=String, default="", description="Key variable names or indices"),
                OptionSpec(name="horizons", type=Int, default=20, description="FEVD horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="Identification method"),
                OptionSpec(name="config", type=String, default="", description="TOML config for restrictions"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:favar_fevd, description="FAVAR variance shares, tidy long form: horizon | variable | shock | value")],
            category="fevd",
            handler=wrap_legacy(_fevd_favar),
        ),
        CommandSpec(
            path=["fevd", "sdfm"],
            summary="Structural DFM forecast error variance decomposition",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="q", type=Int, default=nothing, description="Number of dynamic factors (default: auto via --q-method)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|proxy (--id proxy requires --instrument)"),
                OptionSpec(name="q-method", type=String, default="hallin-liska", description="Auto factor selection: hallin-liska|bai-ng|amengual-watson", choices=["hallin-liska","bai-ng","amengual-watson"]),
                OptionSpec(name="method", type=String, default="fglr", description="Estimator: fglr|gdfm-var (gdfm-var is the legacy path)", choices=["fglr","gdfm-var"]),
                OptionSpec(name="spectral", type=String, default="lag-window", description="GDFM spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="var-lags", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="horizons", type=Int, default=20, description="FEVD horizon"),
                OptionSpec(name="config", type=String, default="", description="TOML config for sign restrictions"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:sdfm_fevd, description="Structural DFM variance shares in factor space, tidy long form: horizon | variable | shock | value")],
            category="fevd",
            handler=wrap_legacy(_fevd_sdfm),
        )
    ]
end

const _FEVD_SLOT_TYPES = Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}}(
    ["fevd", "var"]   => ([:VARModel], [:FEVD, :AriasSVARResult, :UhligSVARResult]),
    ["fevd", "bvar"]  => ([:BVARPosterior], [:BayesianFEVD]),
    ["fevd", "lp"]    => ([:StructuralLP], [:LPFEVD]),
    ["fevd", "vecm"]  => ([:VECMModel], [:FEVD]),
    ["fevd", "pvar"]  => ([:PVARModel], Symbol[]),
    ["fevd", "favar"] => ([:FAVARModel], [:FEVD]),
    ["fevd", "sdfm"]  => ([:StructuralDFM], [:FEVD]),
)

function register_fevd_commands!()
    specs = _tag_slot_types(fevd_specs(), _FEVD_SLOT_TYPES)
    specs = with_result_handles(with_config_ergonomics(with_model_option(specs)))
    specs = with_default_csv_kinds(with_data_kinds(specs, [:timeseries, :csv]))
    specs = [s.path == ["fevd", "pvar"] ? _copy_spec(s; data_kinds=[:panel, :csv]) : s for s in specs]
    register!(specs)
    return build_node("fevd", specs; description="Forecast Error Variance Decomposition")
end


# ── VAR FEVD ─────────────────────────────────────────────

function _fevd_var(; data::String="", result=nothing, model=nothing, lags=nothing, horizons::Int=20,
                    id::String="cholesky", config::String="",
                    instrument::String="", target_var::String="",
                    generalized::Bool=false, normalize::Bool=false,
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, lags, check_lags=true, leaf="fevd var",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="Forecast Error Variance Decomposition", key="fevd", plot, plot_save)
    if isnothing(model)
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
    else
        varnames = model.varnames
        p = model.p
    end
    n = length(varnames)

    _status("Computing FEVD: VAR($p), horizons=$horizons, id=$id")
    _status()

    # Arias identification: use identify_arias → irf_mean → compute FEVD from structural IRFs
    # (narrative-adrr shares the pipeline via identify_narrative)
    if id in ("arias", "narrative-adrr")
        cfg2, restrictions = _load_svar_restrictions(config, n, id == "narrative-adrr" ? "Narrative-ADRR" : "Arias")
        if id == "narrative-adrr"
            isempty(get(get(cfg2, "identification", Dict()), "narrative_contributions", [])) &&
                throw(CliError("usage/missing",
                    "fevd var: --id narrative-adrr requires [identification.narrative_contributions] in --config (ADRR Type A/B)"))
            arias_result = identify_narrative(model, restrictions, horizons; _fwd_seed()...)
        else
            arias_result = identify_arias(model, restrictions, horizons; _fwd_seed()...)
        end
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
                            id="arias", title_prefix="FEVD", format=format, output=output,
                            key_prefix="fevd_by_variable")
        return (; model, result=arias_result)
    end

    # Uhlig identification: use identify_uhlig → compute FEVD from structural IRFs
    if id == "uhlig"
        cfg, restrictions = _load_svar_restrictions(config, n, "Uhlig")
        uhlig_params = get_uhlig_params(cfg)
        uhlig_result = identify_uhlig(model, restrictions, horizons;
            n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
            max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
            tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"],
            _fwd_seed()...)
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
                            id="uhlig", title_prefix="FEVD", format=format, output=output,
                            key_prefix="fevd_by_variable")
        return (; model, result=uhlig_result)
    end

    # W8/#110 (MEMs#364): Pesaran-Shin generalized FEVD. It is NOT an identification
    # scheme -- it sidesteps identification entirely by shocking each variable under the
    # historical covariance -- so it is a separate flag rather than an --id value, and it
    # ignores --id.
    if generalized
        id == "cholesky" && !isempty(config) &&
            _status("--config ignored: generalized FEVD imposes no identification")
        id == "cholesky" || _status("--id $id ignored: generalized FEVD imposes no " *
                                    "identification")
        fevd_result = generalized_fevd(model, horizons; normalize=normalize)
        _status_report(() -> report(fevd_result))
        _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)
        # THE shares do NOT sum to 1 across shocks unless --normalize is given: the
        # generalized shocks are correlated, so their contributions overlap and double-count.
        # Saying so in the title keeps a reader from reading the rows as an orthogonal
        # decomposition.
        note = normalize ? "normalized to sum to 1" : "shares do NOT sum to 1 across shocks"
        output_result(long_table(fevd_result); format=Symbol(format), output=output,
                      title="Generalized FEVD (Pesaran-Shin, $note)", key="generalized_fevd")
        return (; model, result=fevd_result)
    end

    # W2/#166: VAR-family allow-set (proxy/max-share/gmm-moments) + extras.
    _identification_method(id, _ID_METHODS_VAR, "fevd var")
    if id in ("arias", "uhlig") && (!isempty(instrument) || !isempty(target_var))
        throw(CliError("usage/invalid",
            "fevd var: --instrument/--target-var apply only to --id proxy/max-share (got --id $id)"))
    end
    kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VAR,
                                              nvars=length(varnames), leaf="fevd var")
    _inject_svar_id_kwargs!(kwargs, id, "fevd var", data, varnames, instrument, target_var)
    fevd_result = fevd(model, horizons; kwargs...)

    _status_report(() -> report(fevd_result))

    _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)

    # C051: render via MEMs' uniform tidy long_table (horizon|variable|shock|value),
    # replacing the wide per-variable _output_fevd_tables. (Arias/Uhlig branches above
    # build proportions by hand with no FEVD result type, so they keep the wide helper.)
    output_result(long_table(fevd_result); format=Symbol(format), output=output,
                  title="FEVD ($id identification)", key="fevd")
    return (; model, result=fevd_result)
end

# ── BVAR FEVD ────────────────────────────────────────────

function _fevd_bvar(; data::String="", result=nothing, lags::Int=4, horizons::Int=20,
                     id::String="cholesky", draws::Int=2000, sampler::String="direct",
                     config::String="",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="",
                     model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd bvar",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="Bayesian FEVD", key="bayesian_fevd", plot, plot_save,
        key_prefix="bayesian_fevd")
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        n = length(varnames)
    end

    _status("Computing Bayesian FEVD: BVAR($p), horizons=$horizons, id=$id")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

    # W1/#186 fix (pre-existing silent-ignore): --id was validated nowhere
    # and threaded nowhere — every --id rendered cholesky numbers. Validate
    # against the base map and thread like the irf/hd bvar siblings.
    method = _identification_method(id, ID_METHOD_MAP, "fevd bvar")
    bfevd_kwargs = _id_knob_kwargs(id, config, n, "fevd bvar")
    bfevd = fevd(post, horizons;
        method=method, quantiles=[0.16, 0.5, 0.84], bfevd_kwargs...)

    _status_report(() -> report(bfevd))

    _maybe_plot(bfevd; plot=plot, plot_save=plot_save)

    # BayesianFEVD.point_estimate is (variable, shock, horizon) since MEMs 0.7.3 (#527
    # unified it with FEVD/LPFEVD) — exactly the order the shared renderer indexes.
    _output_fevd_tables(bfevd.point_estimate, varnames, horizons;
                        id=id, title_prefix="Bayesian FEVD", format=format, output=output,
                        key_prefix="bayesian_fevd")
    return (; model=post, result=bfevd)
end

# ── LP FEVD ──────────────────────────────────────────────

function _fevd_lp(; data::String="", result=nothing, horizons::Int=20, lags::Int=4, var_lags=nothing,
                   id::String="cholesky", vcov::String="newey_west", config::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="",
                   model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd lp",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="LP FEVD", key="lp_fevd", plot, plot_save, key_prefix="lp_fevd")
    if isnothing(model)
        slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
            id, vcov, config)
        n = size(Y, 2)
    else
        slp = model
        varnames = slp.varnames
        n = length(varnames)
    end

    _status("Computing LP FEVD: horizons=$horizons, id=$id")
    _status()

    fevd_result = lp_fevd(slp, horizons)

    _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)

    _output_fevd_tables(fevd_result.bias_corrected, varnames, horizons;
                        id=id, title_prefix="LP FEVD", format=format, output=output,
                        key_prefix="lp_fevd")
    return (; model=slp, result=fevd_result)
end

# ── VECM FEVD ───────────────────────────────────────────

function _fevd_vecm(; data::String="", result=nothing, lags::Int=2, rank::String="auto",
                     deterministic::String="constant", horizons::Int=20,
                     id::String="cholesky", config::String="",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="",
                     model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd vecm",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="Forecast Error Variance Decomposition", key="vecm_fevd", plot, plot_save)
    if isnothing(model)
        vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
        var_model = to_var(vecm)
    else
        vecm = model
        var_model = to_var(vecm)
        varnames = vecm.varnames
        p = vecm.p
    end
    n = length(varnames)
    r = cointegrating_rank(vecm)

    _status("Computing VECM FEVD: rank=$r, VAR($p), horizons=$horizons, id=$id")
    _status()

    _identification_method(id, _ID_METHODS_VECM, "fevd vecm")
    if id == "svec"
        lr_zeros, sr_zeros = _load_svec_zeros(config, n, "fevd vecm")
        svec_kwargs = Dict{Symbol,Any}(:method => :svec)
        lr_zeros !== nothing && (svec_kwargs[:long_run_zeros] = lr_zeros)
        sr_zeros !== nothing && (svec_kwargs[:short_run_zeros] = sr_zeros)
        fevd_result = try
            fevd(vecm, horizons; svec_kwargs...)
        catch e
            throw(_domain_or_data_error(e, "VECM SVEC FEVD"))
        end
    else
        kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VECM,
                                                  nvars=n, leaf="fevd vecm")
        fevd_result = fevd(var_model, horizons; kwargs...)
    end

    _status_report(() -> report(fevd_result))

    _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)

    # C051: tidy long_table (see fevd var).
    output_result(long_table(fevd_result); format=Symbol(format), output=output,
                  title="VECM FEVD ($id identification)", key="vecm_fevd")
    return (; model=vecm, result=fevd_result)
end

# ── Panel VAR FEVD ─────────────────────────────────────────

function _fevd_pvar(; data::String="", result=nothing, id_col::String="", time_col::String="",
                     lags::Int=1, horizons::Int=10,
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="",
                     model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd pvar",
                            horizons, horizons_default=10)
    loaded === nothing || return loaded
    if isnothing(model)
        model, panel, varnames = _load_and_estimate_pvar(data, id_col, time_col, lags)
    else
        varnames = model.varnames
    end
    n = length(varnames)

    _status("Computing Panel VAR FEVD: horizons=$horizons")
    _status()

    # MEMs 0.7.0 (C054): pvar_fevd now returns a raw (H+1)×n×n array indexed
    # [horizon, variable, shock] (was a struct with `.proportions`). Permute to
    # [variable, shock, horizon] and drop horizon 0 to keep the 1..H tables.
    fevd_arr = pvar_fevd(model, horizons)

    _maybe_plot(fevd_arr; plot=plot, plot_save=plot_save)

    proportions = permutedims(fevd_arr[2:end, :, :], (2, 3, 1))
    _output_fevd_tables(proportions, varnames, horizons;
                        id="cholesky", title_prefix="Panel VAR FEVD",
                        format=format, output=output, key_prefix="panel_var_fevd")
    return (; model, result=fevd_arr)
end

# ── FAVAR FEVD ─────────────────────────────────────────

function _fevd_favar(; data::String="", result=nothing, factors=nothing, lags::Int=2,
                      key_vars::String="", horizons::Int=20,
                      id::String="cholesky", config::String="",
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="",
                      model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd favar",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="Forecast Error Variance Decomposition", key="favar_fevd", plot, plot_save)
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end
    id_kwargs = _build_identification_kwargs(id, config; nvars=length(varnames),
                                                 leaf="fevd favar")

    _status("FAVAR FEVD: horizon=$horizons, id=$id")
    _status()

    result = fevd(favar, horizons; id_kwargs...)
    _maybe_plot(result; plot=plot, plot_save=plot_save)

    # C051: tidy long_table (horizon|variable|shock|value); fevd(favar,...) delegates to
    # fevd(to_var(favar),...) — the same FEVD type as fevd var.
    fevd_df = long_table(result)
    output_result(fevd_df; format=Symbol(format), output=output,
                  title="FAVAR FEVD ($id identification)", key="favar_fevd")
    return (; model=favar, result=result)
end

# ── Structural DFM FEVD ──────────────────────────────

function _fevd_sdfm(; data::String="", result=nothing, factors=nothing, id::String="cholesky",
                     var_lags::Int=1, horizons::Int=20,
                     config::String="", method::String="fglr",
                     spectral::String="lag-window", instrument::String="",
                     q_method::String="hallin-liska",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="",
                     model=nothing)
    loaded = _loaded_result(result; data, model, leaf="fevd sdfm",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_fevd_result(loaded; format, output,
        title="Forecast Error Variance Decomposition", key="sdfm_fevd", plot, plot_save)
    if isnothing(model)
        # W1/#165: shared estimation surface with `estimate sdfm`. FEVD uses the
        # identification stored at estimation (upstream #710).
        sdfm, _, _, q = _load_and_estimate_sdfm(data, factors, id, var_lags,
            horizons, config, method, spectral, instrument, q_method)
    else
        sdfm = model
    end

    _status("SDFM FEVD: id=$id, method=$method, horizon=$horizons")
    _status()

    result = fevd(sdfm, horizons)
    _maybe_plot(result; plot=plot, plot_save=plot_save)

    # C051: tidy long_table (horizon|variable|shock|value); fevd(sdfm,...) delegates to
    # fevd(sdfm.factor_var,...) — the same FEVD type as fevd var, in factor space.
    fevd_df = long_table(result)
    output_result(fevd_df; format=Symbol(format), output=output, title="SDFM FEVD")
    return (; model=sdfm, result=result)
end
