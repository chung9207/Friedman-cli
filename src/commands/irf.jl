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

# IRF commands: var, bvar, lp, vecm, pvar, favar, sdfm (action-first: friedman irf var ...)

function irf_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["irf", "var"],
            summary="Compute frequentist impulse response functions",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig|fastica|jade|sobi|dcov|hsic|student_t|mixture_normal|pml|skew_normal|markov_switching|garch_id|proxy|max-share|gmm-moments|narrative-adrr|lewis-tvv|sv-em"),
                OptionSpec(name="ci", type=String, default="bootstrap", description="none|bootstrap|theoretical"),
                OptionSpec(name="replications", type=Int, default=1000, description="Bootstrap replications"),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="target-var", type=String, default="", description="Max-share target: column name or 1-based index (only with --id max-share)"),
                OptionSpec(name="bootstrap", type=String, default="iid",
                           description="Bootstrap scheme (--ci bootstrap): iid|wild|block",
                           choices=["iid", "wild", "block"]),
                OptionSpec(name="block-length", type=Int, default=0,
                           description="Block length for --bootstrap block (0 = library default)"),
                OptionSpec(name="wild-dist", type=String, default="rademacher",
                           description="Wild-bootstrap multiplier: rademacher|mammen",
                           choices=["rademacher", "mammen"]),
                OptionSpec(name="bias-reps", type=Int, default=0,
                           description="Inner reps for --bias-correct (0 = same as --replications)"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="summary", type=String, default="none", description="Set-identified summary (only with --identified-set)", choices=["none", "median-target", "modal-model", "joint-band", "sup-t-band"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser"),
                FlagSpec(name="cumulative", description="Compute cumulative IRFs (for differenced data)"),
                FlagSpec(name="identified-set", description="Return full identified set for sign restrictions"),
                FlagSpec(name="stationary-only", description="Filter non-stationary bootstrap draws"),
                FlagSpec(name="bias-correct", description="Kilian (1998) bias-corrected bootstrap bands")
            ],
            # One IRF table per invocation — the tidy output is filtered to the single
            # --shock, so this is NOT a family.
            tables=[TableSpec(name=:irf, description="Responses to the selected shock: horizon | variable | shock | value | lower | upper (wide horizon-by-variable under --id arias|uhlig)"),
                    TableSpec(name=:irf_identified_set, description="Median and bounds over the sign-identified set (--identified-set with --id sign)"),
                    TableSpec(name=:arias_importance_sampling_diagnostics, description="Acceptance rate, draws, ESS and ESS fraction of the Arias importance sampler (--id arias)")],
            category="irf",
            handler=wrap_legacy(_irf_var),
        ),
        CommandSpec(
            path=["irf", "bvar"],
            summary="Compute Bayesian impulse response functions with credible intervals",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|robust-bayes"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws"),
                OptionSpec(name="sampler", type=String, default="direct", description="direct|gibbs"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification/prior"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser"),
                FlagSpec(name="cumulative", description="Compute cumulative IRFs (for differenced data)")
            ],
            tables=[TableSpec(name=:bayesian_irf, description="Posterior-mean responses to the selected shock with 68% credible bands: horizon | variable | shock | value | lower | upper"),
                    TableSpec(name=:robust_bayes_bands, description="Giacomini-Kitagawa robust bands for the selected shock: horizon | one lower/upper/robust_lower/robust_upper column per variable (--id robust-bayes)"),
                    TableSpec(name=:robust_bayes_diagnostics, description="Empty-set probability, informativeness and credibility level (--id robust-bayes)")],
            category="irf",
            handler=wrap_legacy(_irf_bvar),
        ),
        CommandSpec(
            path=["irf", "tvpvar"],
            summary="Date-specific IRF from a TVP-VAR-SV",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="date", type=Int, default=0, description="Date index in 1:T_eff to evaluate the IRF at (REQUIRED)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="Retained Gibbs draws"),
                OptionSpec(name="burnin", type=Int, default=1000, description="Burn-in sweeps discarded"),
                OptionSpec(name="thin", type=Int, default=1, description="Keep every k-th draw"),
                OptionSpec(name="n-train", type=Int, default=0, description="Training sample used to calibrate priors"),
                OptionSpec(name="k-q", type=Float64, default=0.01, description="Coefficient random-walk prior scale (> 0)"),
                OptionSpec(name="k-s", type=Float64, default=0.1, description="Covariance random-walk prior scale (> 0)"),
                OptionSpec(name="k-w", type=Float64, default=0.01, description="Log-volatility random-walk prior scale (> 0)"),
                OptionSpec(name="irf-draws", type=Int, default=500, description="Posterior draws used for the IRF bands"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-tvp", description="Hold coefficients constant"),
                   FlagSpec(name="no-sv", description="Hold volatilities constant"),
                   FlagSpec(name="no-stationary-only", description="Include explosive draws instead of discarding them")],
            tables=[TableSpec(name=:tvpvar_irf, description="Date-t responses to the selected shock with 68% credible bands: horizon | variable | shock | value | lower | upper")],
            category="irf",
            handler=wrap_legacy(_irf_tvpvar),
        ),
        CommandSpec(
            path=["irf", "lp"],
            summary="Compute structural LP impulse response functions",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="shock", type=Int, default=1, description="Single shock index (1-based)"),
                OptionSpec(name="shocks", type=String, default="", description="Comma-separated shock indices (e.g. 1,2,3)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="LP control lags"),
                OptionSpec(name="var-lags", type=Int, default=nothing, description="VAR lag order for identification (default: same as --lags)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
                OptionSpec(name="ci", type=String, default="none", description="none|bootstrap"),
                OptionSpec(name="replications", type=Int, default=200, description="Bootstrap replications"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="vcov", type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
                OptionSpec(name="config", type=String, default="", description="TOML config for sign/narrative restrictions"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser"),
                FlagSpec(name="cumulative", description="Compute cumulative IRFs (for differenced data)")
            ],
            tables=[TableSpec(name=:lp_irf, description="Local-projection responses to every selected shock in one tidy table: horizon | variable | shock | value | lower | upper")],
            category="irf",
            handler=wrap_legacy(_irf_lp),
        ),
        CommandSpec(
            path=["irf", "vecm"],
            summary="Compute impulse response functions via VECM → VAR representation",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|narrative|longrun|svec|lewis-tvv|sv-em"),
                OptionSpec(name="ci", type=String, default="bootstrap", description="none|bootstrap|theoretical"),
                OptionSpec(name="replications", type=Int, default=1000, description="Bootstrap replications"),
                OptionSpec(name="config", type=String, default="", description="TOML config for identification"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:vecm_irf, description="Responses of the VECM's VAR representation to the selected shock: horizon | variable | shock | value | lower | upper")],
            category="irf",
            handler=wrap_legacy(_irf_vecm),
        ),
        CommandSpec(
            path=["irf", "pvar"],
            summary="Compute Panel VAR impulse response functions (OIRF/GIRF)",
            args=[ArgSpec(name="data", description="Path to CSV panel data file")],
            options=[
                OptionSpec(name="id-col", type=String, default="", description="Panel group identifier column"),
                OptionSpec(name="time-col", type=String, default="", description="Time period column"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Lag order"),
                OptionSpec(name="horizons", type=Int, default=10, description="IRF horizon"),
                OptionSpec(name="irf-type", type=String, default="oirf", description="oirf|girf"),
                OptionSpec(name="boot-draws", type=Int, default=500, description="Bootstrap draws for CIs"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            # One table per shock. Orthogonalized and generalized responses are different
            # statistics, so each gets its own family (cf. fevd / generalized_fevd).
            tables=[TableSpec(name=:panel_var_oirf, family=true, description="Orthogonalized responses to one shock (--irf-type oirf): horizon | one column per variable, with bootstrap bands"),
                    TableSpec(name=:panel_var_girf, family=true, description="Generalized responses to one shock (--irf-type girf): horizon | one column per variable, with bootstrap bands")],
            category="irf",
            handler=wrap_legacy(_irf_pvar),
        ),
        CommandSpec(
            path=["irf", "favar"],
            summary="FAVAR impulse response functions",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lag order"),
                OptionSpec(name="key-vars", type=String, default="", description="Key variable names or indices (comma-separated)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="id", type=String, default="cholesky", description="Identification method"),
                OptionSpec(name="config", type=String, default="", description="TOML config for restrictions"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="panel-irf", description="Output panel-wide IRFs (N variables) instead of factor-level"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:favar_irf, description="FAVAR responses to every shock in one tidy table (panel-wide under --panel-irf): horizon | variable | shock | value | lower | upper")],
            category="irf",
            handler=wrap_legacy(_irf_favar),
        ),
        CommandSpec(
            path=["irf", "sdfm"],
            summary="Structural DFM impulse response functions (panel-wide)",
            args=[ArgSpec(name="data", description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="q", type=Int, default=nothing, description="Number of dynamic factors (default: auto via --q-method)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|proxy (--id proxy requires --instrument)"),
                OptionSpec(name="q-method", type=String, default="hallin-liska", description="Auto factor selection: hallin-liska|bai-ng|amengual-watson", choices=["hallin-liska","bai-ng","amengual-watson"]),
                OptionSpec(name="method", type=String, default="fglr", description="Estimator: fglr|gdfm-var (gdfm-var is the legacy path)", choices=["fglr","gdfm-var"]),
                OptionSpec(name="spectral", type=String, default="lag-window", description="GDFM spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="var-lags", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="horizons", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="config", type=String, default="", description="TOML config for sign restrictions"),
                OptionSpec(name="ci", type=String, default="none", description="Bands: none|bootstrap (residual bootstrap)", choices=["none","bootstrap"]),
                OptionSpec(name="reps", type=Int, default=200, description="Bootstrap replications (with --ci bootstrap)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:sdfm_irf, description="Panel-wide structural DFM responses to every shock: horizon | variable | shock | value | lower | upper")],
            category="irf",
            handler=wrap_legacy(_irf_sdfm),
        )
    ]
end

const _IRF_SLOT_TYPES = Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}}(
    ["irf", "var"]    => ([:VARModel],
                          [:ImpulseResponse, :AriasSVARResult, :UhligSVARResult, :SignIdentifiedSet]),
    ["irf", "bvar"]   => ([:BVARPosterior], [:BayesianImpulseResponse, :RobustBayesResult]),
    ["irf", "tvpvar"] => ([:TVPVARPosterior], [:BayesianImpulseResponse]),
    ["irf", "lp"]     => ([:StructuralLP], [:ImpulseResponse]),
    ["irf", "vecm"]   => ([:VECMModel], [:ImpulseResponse]),
    ["irf", "pvar"]   => ([:PVARModel], Symbol[]),
    ["irf", "favar"]  => ([:FAVARModel], [:ImpulseResponse]),
    ["irf", "sdfm"]   => ([:StructuralDFM], [:ImpulseResponse]),
)

function register_irf_commands!()
    specs = _tag_slot_types(irf_specs(), _IRF_SLOT_TYPES)
    specs = with_result_handles(with_config_ergonomics(with_model_option(specs)))
    specs = with_default_csv_kinds(with_data_kinds(specs, [:timeseries, :csv]))
    specs = [s.path == ["irf", "pvar"] ? _copy_spec(s; data_kinds=[:panel, :csv]) : s for s in specs]
    register!(specs)
    return build_node("irf", specs; description="Impulse Response Functions")
end


# ── VAR IRF ──────────────────────────────────────────────

function _irf_var(; data::String="", result=nothing, model=nothing, lags=nothing, shock::Int=1, horizons::Int=20,
                   id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                   config::String="", instrument::String="", target_var::String="",
                   bootstrap::String="iid", block_length::Int=0,
                   wild_dist::String="rademacher",
                   bias_correct::Bool=false, bias_reps::Int=0,
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="",
                   cumulative::Bool=false, identified_set::Bool=false,
                   stationary_only::Bool=false, summary::String="none")
    loaded = _loaded_result(result; data, model, lags, check_lags=true, leaf="irf var",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="irf", plot, plot_save, shock)
    if isnothing(model)
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
    else
        varnames = model.varnames
        p = model.p
    end
    n = length(varnames)

    _status("Computing IRFs: VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    _status()

    if id in ("arias", "uhlig") && (!isempty(instrument) || !isempty(target_var))
        throw(CliError("usage/invalid",
            "irf var: --instrument/--target-var apply only to --id proxy/max-share (got --id $id)"))
    end

    # Arias identification handled separately (narrative-adrr shares the Arias
    # pipeline via identify_narrative — same AriasSVARResult shape)
    if id in ("arias", "narrative-adrr")
        arias = _var_irf_arias(model, config, horizons, varnames, shock; format=format, output=output, estimator=id)
        return (; model, result=arias)
    end

    # Uhlig identification handled separately
    if id == "uhlig"
        uhlig = _var_irf_uhlig(model, config, horizons, varnames, shock; format=format, output=output)
        return (; model, result=uhlig)
    end

    # W2/#166: the VAR family admits proxy/max-share/gmm-moments beyond the base
    # map (validated here — _build_identification_kwargs below only knows base).
    _identification_method(id, _ID_METHODS_VAR, "irf var")
    if identified_set && id != "sign"
        throw(CliError("usage/invalid",
            "irf var: --identified-set applies only to --id sign (got --id $id)"))
    end

    if summary != "none" && !(identified_set && id == "sign")
        throw(CliError("usage/invalid",
            "irf var: --summary applies only to --identified-set with --id sign (got --summary $summary)"))
    end

    # Sign-identified set: return full draw set instead of point estimates
    if identified_set && id == "sign"
        check_func, _ = _build_check_func(config)
        isnothing(check_func) && error("--identified-set requires a --config file with sign restrictions")
        set = identify_sign(model, horizons, check_func; max_draws=replications, store_all=true,
                            _fwd_seed()...)
        # W2/#166 (#746): set-identified summaries operate on the held set.
        sum_label = ""
        if summary == "none"
            lower, upper = irf_bounds(set)
            med = irf_median(set)
        elseif summary in ("median-target", "modal-model")
            tgt = summary == "median-target" ? median_target(set) : modal_model(set)
            med, lower, upper = tgt.irf, nothing, nothing
            sum_label = " [$summary #$(tgt.index)]"
        else
            med = irf_median(set)
            lower, upper = summary == "joint-band" ? joint_band(set) : sup_t_band(set)
            sum_label = " [$summary]"
        end
        _status("Sign-Identified Set: $(set.n_accepted)/$(set.n_total) accepted ($(round(set.acceptance_rate*100; digits=1))%)")
        # Set arrays carry `horizons` rows, impact-first — label them 0..H-1
        # (build_irf_table's default), not 0..H.
        irf_df = build_irf_table(med, lower, upper, varnames, shock)
        shock_name = _shock_name(varnames, shock)
        output_result(irf_df; format=Symbol(format), output=output,
                      title="IRF Identified Set (sign, $shock_name shock)$sum_label",
                      key="irf_identified_set")
        return (; model, result=set)
    end

    # W8/#110 (MEMs#370): bootstrap scheme + Kilian (1998) bias correction. These only
    # bite under --ci bootstrap; say so rather than letting a flag look effective.
    boot = lowercase(strip(bootstrap))
    boot in ("iid", "wild", "block") || throw(CliError("usage/invalid-option",
        "invalid --bootstrap '$bootstrap'; must be iid, wild or block"))
    # Upstream `_wild_weights` implements ONLY these two — a :normal multiplier looks
    # plausible and is not supported, and reaches the estimator as an untyped
    # TaskFailedException from inside the threaded bootstrap loop (exit 1).
    wdist = lowercase(strip(wild_dist))
    wdist in ("rademacher", "mammen") || throw(CliError("usage/invalid-option",
        "invalid --wild-dist '$wild_dist'; must be rademacher or mammen"))
    block_length >= 0 || throw(CliError("usage/invalid",
        "--block-length must be ≥ 0 (got $block_length)"))
    bias_reps >= 0 || throw(CliError("usage/invalid",
        "--bias-reps must be ≥ 0 (got $bias_reps)"))
    if ci != "bootstrap" && (boot != "iid" || bias_correct || block_length > 0)
        _status("bootstrap options ignored: they apply to --ci bootstrap, not --ci $ci")
    end

    kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VAR,
                                              nvars=length(varnames), leaf="irf var")
    _inject_svar_id_kwargs!(kwargs, id, "irf var", data, varnames, instrument, target_var)
    kwargs[:ci_type] = Symbol(ci)
    kwargs[:reps] = replications
    if ci == "bootstrap"
        kwargs[:bootstrap] = Symbol(boot)
        kwargs[:block_length] = block_length
        kwargs[:wild_dist] = Symbol(wdist)
        kwargs[:bias_correct] = bias_correct
        kwargs[:bias_reps] = bias_reps
        if bias_correct
            # Kilian bootstrap-after-bootstrap: the inner loop re-estimates the VAR
            # bias_reps times PER OUTER DRAW, so the cost is multiplicative, not additive.
            _status("Kilian (1998) bias correction on: inner reps=" *
                    "$(bias_reps > 0 ? bias_reps : replications) per outer draw")
        end
    end
    if stationary_only
        kwargs[:stationary_only] = true
    end
    isnothing(_SEED[]) || (kwargs[:seed] = _SEED[])  # --seed → bootstrap/sign draws + ImpulseResponse manifest (C052/#243)

    # Upstream heteroskedasticity guards (lewis-tvv n≥2 / T_eff≥100,
    # sv-em MCEM) raise bare ArgumentError — wrap so degenerate input
    # is data/invalid (exit 3), not an untyped exit 1 (W1/#186).
    irf_result = try
        irf(model, horizons; kwargs...)
    catch e
        throw(_domain_or_data_error(e, "VAR IRF"))
    end

    if cumulative
        irf_result = cumulative_irf(irf_result)
        _status_styled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    _status_report(() -> report(irf_result))

    # C051: render via MEMs' uniform tidy long_table (horizon|variable|shock|value|lower|
    # upper), replacing the wide per-shock build_irf_table. Preserve the --shock selector
    # by filtering the tidy rows to the chosen structural shock.
    shock_name = irf_result.shocks[shock]
    irf_df = long_table(irf_result)
    irf_df = irf_df[irf_df.shock .== shock_name, :]
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock ($id identification)", key="irf")
    return (; model, result=irf_result)
end

function _var_irf_arias(model, config::String, horizons::Int,
                        varnames::Vector{String}, shock::Int; format::String="table", output::String="",
                        estimator::String="arias")
    cfg, restrictions = _load_svar_restrictions(config, nvars(model),
        estimator == "narrative-adrr" ? "Narrative-ADRR" : "Arias")
    if estimator == "narrative-adrr"
        id_cfg = get(cfg, "identification", Dict())
        isempty(get(id_cfg, "narrative_contributions", [])) && throw(CliError("usage/missing",
            "irf var: --id narrative-adrr requires [identification.narrative_contributions] in --config (ADRR Type A/B)"))
        result = identify_narrative(model, restrictions, horizons; _fwd_seed()...)
    else
        result = identify_arias(model, restrictions, horizons; _fwd_seed()...)
    end

    # W8/#110 (MEMs#372): the importance weights became operative in 0.7.2, so the summary
    # can now rest on far fewer EFFECTIVE draws than n_draws suggests. Under pure sign
    # restrictions the weights are uniform and ess_fraction == 1; with zero restrictions a
    # small fraction means a handful of draws carry most of the posterior mass. Reporting
    # only the IRF would hide that entirely, so the diagnostics get their own table.
    ess      = hasproperty(result, :ess) ? Float64(result.ess) : NaN
    ess_frac = hasproperty(result, :ess_fraction) ? Float64(result.ess_fraction) : NaN
    if isfinite(ess_frac) && ess_frac < 0.1
        _status_styled("  warning: Arias importance weights are DEGENERATE — effective " *
                       "sample $(round(ess; digits=1)) is $(round(100*ess_frac; digits=1))% " *
                       "of the draws; the weighted IRF rests on very few of them\n";
                       color=:yellow)
    end
    shock_name = _shock_name(varnames, shock)
    irf_df = build_irf_table(irf_mean(result), nothing, nothing, varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Arias et al. identification)", key="irf")
    output_kv(Pair{String,Any}[
        "acceptance_rate" => round(Float64(result.acceptance_rate); digits=6),
        "n_draws"         => length(result.weights),
        "ess"             => round(ess; digits=4),
        "ess_fraction"    => round(ess_frac; digits=6),
    ]; format=format, title="Arias Importance-Sampling Diagnostics")
    return result
end

function _var_irf_uhlig(model, config::String, horizons::Int,
                        varnames::Vector{String}, shock::Int; format::String="table", output::String="")
    cfg, restrictions = _load_svar_restrictions(config, nvars(model), "Uhlig")
    uhlig_params = get_uhlig_params(cfg)
    result = identify_uhlig(model, restrictions, horizons;
        n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
        max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
        tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"],
        _fwd_seed()...)
    _status("Uhlig identification: penalty=$(round(result.penalty; digits=6)), converged=$(result.converged)")
    for (si, sp) in enumerate(result.shock_penalties)
        _status("  Shock $si penalty: $(round(sp; digits=6))")
    end
    _status()
    irf_df = build_irf_table(result.irf, nothing, nothing, varnames, shock)
    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Uhlig identification)", key="irf")
    return result
end

# ── BVAR IRF ─────────────────────────────────────────────

function _irf_bvar(; data::String="", result=nothing, lags::Int=4, shock::Int=1, horizons::Int=20,
                    id::String="cholesky", draws::Int=2000, sampler::String="direct",
                    config::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    cumulative::Bool=false,
                    model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf bvar",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="bayesian_irf", plot, plot_save, shock)
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        n = length(varnames)
    end
    # Robust Bayes runs its own estimator (identify_robust_bayes on the posterior),
    # not the generic compute_Q path — which has no robust-bayes branch.
    if id == "robust-bayes"
        rb = _irf_bvar_robust_bayes(post, config, n, horizons, varnames, shock;
                                   format=format, output=output, plot=plot, plot_save=plot_save)
        return (; model=post, result=rb)
    end
    method = _identification_method(id, ID_METHOD_MAP, "irf bvar")

    _status("Computing Bayesian IRFs: BVAR($p), shock=$shock, horizons=$horizons, id=$id")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

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
    merge!(kwargs, _id_knob_kwargs(id, config, length(varnames), "irf bvar"))

    birf = irf(post, horizons; kwargs...)

    if cumulative
        birf = cumulative_irf(birf)
        _status_styled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(birf; plot=plot, plot_save=plot_save)

    _status_report(() -> report(birf))

    # C051: tidy long_table (horizon|variable|shock|value|lower|upper); value = posterior
    # mean, lower/upper = the outer credible quantiles (16/84pct). Preserve --shock by
    # filtering the tidy rows to the selected structural shock.
    shock_name = birf.shocks[shock]
    irf_df = long_table(birf)
    irf_df = irf_df[irf_df.shock .== shock_name, :]
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Bayesian IRF to $shock_name shock ($id, 68% credible interval)",
                  key="bayesian_irf")
    return (; model=post, result=birf)
end

function _irf_bvar_robust_bayes(post, config::String, n::Int, horizons::Int,
                                  varnames::Vector{String}, shock::Int;
                                  format::String="table", output::String="",
                                  plot::Bool=false, plot_save::String="")
    1 <= shock <= n || throw(CliError("usage/invalid",
        "irf bvar: --shock index $shock out of 1:$n"))
    _, restrictions = _load_svar_restrictions(config, n, "Robust-Bayes")
    result = identify_robust_bayes(post, restrictions, horizons)
    _status_report(() -> report(result))
    _status()

    # Arrays are (H, n_variables, n_shocks); horizons label 0..H-1 exactly like
    # the Arias wide table (build_irf_table's default).
    H = size(result.lower, 1)
    size(result.lower, 3) >= shock || throw(CliError("model/error",
        "irf bvar: robust-bayes result covers $(size(result.lower, 3)) shocks (asked $shock)"))
    shock_name = _shock_name(varnames, shock)
    band_df = DataFrame(horizon=collect(0:(H - 1)))
    for (vi, vname) in enumerate(varnames)
        band_df[!, "$(vname)_lower"] = result.lower[:, vi, shock]
        band_df[!, "$(vname)_upper"] = result.upper[:, vi, shock]
        band_df[!, "$(vname)_robust_lower"] = result.robust_lower[:, vi, shock]
        band_df[!, "$(vname)_robust_upper"] = result.robust_upper[:, vi, shock]
    end
    output_result(band_df; format=Symbol(format), output=output,
                  title="Robust Bayes bands to $shock_name shock (Giacomini-Kitagawa)",
                  key="robust_bayes_bands")
    _status()

    output_kv(Pair{String,Any}[
        "Empty-set probability" => round(Float64(result.empty_set_prob); digits=6),
        "Informativeness" => round(Float64(result.informativeness); digits=6),
        "Level" => round(Float64(result.level); digits=4),
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
        title="Robust Bayes Diagnostics", key="robust_bayes_diagnostics")

    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

# ── LP IRF ───────────────────────────────────────────────

function _irf_lp(; data::String="", result=nothing, shock::Int=1, shocks::String="",
                  horizons::Int=20, lags::Int=4, var_lags=nothing,
                  id::String="cholesky", ci::String="none",
                  replications::Int=200, conf_level::Float64=0.95,
                  vcov::String="newey_west", config::String="",
                  output::String="", format::String="table",
                  plot::Bool=false, plot_save::String="",
                  cumulative::Bool=false,
                  model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf lp",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="lp_irf", plot, plot_save, shock)
    # Multi-shock mode
    if !isempty(shocks)
        shock_indices = parse.(Int, split(shocks, ","))
    else
        shock_indices = [shock]
    end

    if isnothing(model)
        slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
            id, vcov, config; ci_type=Symbol(ci), reps=replications, conf_level=conf_level)
        n = size(Y, 2)
    else
        slp = model
        varnames = slp.varnames
        n = length(varnames)
    end
    irf_result = slp.irf

    if cumulative
        irf_result = cumulative_irf(irf_result)
        _status_styled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    _status("Computing LP IRFs: horizons=$horizons, id=$id, ci=$ci")
    _status()

    for shock_idx in shock_indices
        (shock_idx < 1 || shock_idx > n) && error("shock index $shock_idx out of range (data has $n variables)")
    end
    # C051: tidy long_table (horizon|variable|shock|value|lower|upper); slp.irf is a full
    # ImpulseResponse — Plagborg-Møller & Wolf (2021) stack LP responses into the same 3D
    # (horizon, variable, shock) array as a VAR IRF — same schema as irf var. --shocks may
    # name more than one shock (unlike irf var's single --shock), so filter to all
    # selected shocks in one tidy table instead of splitting into per-shock files. Index
    # into irf_result.shocks (not the CLI-loaded varnames — see irf var/vecm) since
    # structural_lp names shocks after its own internal VAR, not the CLI's column names.
    shock_names = [irf_result.shocks[s] for s in shock_indices]
    irf_df = long_table(irf_result)
    irf_df = irf_df[in.(irf_df.shock, Ref(shock_names)), :]
    title = length(shock_names) == 1 ?
        "LP IRF to $(shock_names[1]) shock ($id identification)" :
        "LP IRF to shocks $(join(shock_names, ", ")) ($id identification)"
    output_result(irf_df; format=Symbol(format), output=output, title=title, key="lp_irf")
    return (; model=slp, result=irf_result)
end

# ── VECM IRF ────────────────────────────────────────────

function _irf_vecm(; data::String="", result=nothing, lags::Int=2, rank::String="auto",
                    deterministic::String="constant",
                    shock::Int=1, horizons::Int=20,
                    id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                    config::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf vecm",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="vecm_irf", plot, plot_save, shock)
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

    _status("Computing VECM IRFs: rank=$r, VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    _status()

    _identification_method(id, _ID_METHODS_VECM, "irf vecm")
    if id == "svec"
        # Upstream (vecm/analysis.jl) re-identifies per draw, so bands would reuse
        # a frozen rotation — only point IRFs are available.
        ci == "none" || throw(CliError("usage/invalid",
            "irf vecm: --id svec supports only --ci none (got --ci $ci)"))
        lr_zeros, sr_zeros = _load_svec_zeros(config, n, "irf vecm")
        svec_kwargs = Dict{Symbol,Any}(:method => :svec)
        lr_zeros !== nothing && (svec_kwargs[:long_run_zeros] = lr_zeros)
        sr_zeros !== nothing && (svec_kwargs[:short_run_zeros] = sr_zeros)
        irf_result = try
            irf(vecm, horizons; svec_kwargs...)
        catch e
            throw(_domain_or_data_error(e, "VECM SVEC IRF"))
        end
    else
        kwargs = _build_identification_kwargs(id, config; methods=_ID_METHODS_VECM,
                                              nvars=n, leaf="irf vecm")
        kwargs[:ci_type] = Symbol(ci)
        kwargs[:reps] = replications
        isnothing(_SEED[]) || (kwargs[:seed] = _SEED[])  # --seed → bootstrap/sign draws + manifest (C052/#243)

        irf_result = irf(var_model, horizons; kwargs...)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    _status_report(() -> report(irf_result))

    # C051: tidy long_table filtered to the selected --shock (see irf var).
    shock_name = irf_result.shocks[shock]
    irf_df = long_table(irf_result)
    irf_df = irf_df[irf_df.shock .== shock_name, :]
    output_result(irf_df; format=Symbol(format), output=output,
                  title="VECM IRF to $shock_name shock ($id identification)", key="vecm_irf")
    return (; model=vecm, result=irf_result)
end

# ── Panel VAR IRF ──────────────────────────────────────────

function _irf_pvar(; data::String="", result=nothing, id_col::String="", time_col::String="",
                    lags::Int=1, horizons::Int=10,
                    irf_type::String="oirf", boot_draws::Int=500,
                    confidence::Float64=0.95,
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf pvar",
                            horizons, horizons_default=10)
    loaded === nothing || return loaded
    validate_method(irf_type, ["oirf", "girf"], "IRF type")

    if isnothing(model)
        model, panel, varnames = _load_and_estimate_pvar(data, id_col, time_col, lags)
    else
        varnames = model.varnames
    end
    n = length(varnames)

    _status("Computing Panel VAR IRFs: type=$irf_type, horizons=$horizons, bootstrap=$boot_draws")
    _status()

    # Compute IRFs with bootstrap CIs. MEMs 0.7.0 (C054) renamed kwargs
    # (n_boot→n_draws, conf_level→ci) and returns a NamedTuple
    # (irf, lower, upper, draws) instead of an ImpulseResponse struct.
    irf_result = pvar_bootstrap_irf(model, horizons;
        n_draws=boot_draws, ci=confidence, irf_type=Symbol(irf_type),
        _fwd_seed()...)

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    # Output per-shock IRF tables (W3/#138 family keys `panel_var_<oirf|girf>_<shock>`)
    key_prefix = irf_type == "girf" ? "panel_var_girf" : "panel_var_oirf"
    for shock in 1:n
        shock_name = _shock_name(varnames, shock)
        irf_df = build_irf_table(irf_result.irf, irf_result.lower, irf_result.upper,
                                 varnames, shock)
        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock_name),
                      title="Panel VAR $(uppercase(irf_type)) to $shock_name shock",
                      key=_table_key(key_prefix, shock_name))
        _status()
    end
    return (; model, result=irf_result)
end

# ── FAVAR IRF ──────────────────────────────────────────

function _irf_favar(; data::String="", result=nothing, factors=nothing, lags::Int=2,
                     key_vars::String="", horizons::Int=20,
                     id::String="cholesky", config::String="",
                     panel_irf::Bool=false,
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="",
                     model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf favar",
                            id, horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="favar_irf", plot, plot_save)
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end
    n = size(favar.Y, 2)

    id_kwargs = _build_identification_kwargs(id, config; nvars=n, leaf="irf favar")

    _status("FAVAR IRF: horizon=$horizons, id=$id" * (panel_irf ? ", panel-wide" : ""))
    _status()

    irf_result = irf(favar, horizons; id_kwargs...)

    if panel_irf
        irf_result = favar_panel_irf(favar, irf_result)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    # C051: tidy long_table (horizon|variable|shock|value|lower|upper); irf(favar,...)
    # delegates to irf(to_var(favar),...) — the same ImpulseResponse type as irf var —
    # so one tidy table covers every shock (no more per-shock output files).
    irf_df = long_table(irf_result)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="FAVAR IRF ($id identification)" * (panel_irf ? ", panel-wide" : ""),
                  key="favar_irf")
    return (; model=favar, result=irf_result)
end

# ── Structural DFM IRF ────────────────────────────────

function _irf_sdfm(; data::String="", result=nothing, factors=nothing, id::String="cholesky",
                    var_lags::Int=1, horizons::Int=40,
                    config::String="", method::String="fglr",
                    spectral::String="lag-window", instrument::String="",
                    q_method::String="hallin-liska",
                    ci::String="none", reps::Int=200,
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf sdfm",
                            id, horizons, horizons_default=40)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="sdfm_irf", plot, plot_save)
    ci in ("none", "bootstrap") || throw(CliError("usage/invalid",
        "irf sdfm: --ci must be none|bootstrap (got '$ci')"))
    reps >= 1 || throw(CliError("usage/invalid",
        "irf sdfm: --reps must be ≥ 1 (got $reps)"))
    if isnothing(model)
        # W1/#165: shared estimation surface with `estimate sdfm` (H=horizons —
        # post-0.9.1 the IRF is computed on demand, but the stored horizon still
        # bounds the mock clamp, so keep them equal).
        sdfm, _, _, q = _load_and_estimate_sdfm(data, factors, id, var_lags,
            horizons, config, method, spectral, instrument, q_method)
    else
        sdfm = model
    end

    _status("Structural DFM IRF: id=$id, method=$method, horizon=$horizons, ci=$ci")
    _status()

    # W1/#165: on-demand horizons (#717) + residual-bootstrap bands (#714/#716);
    # identification stays fixed at estimation (upstream rejects ident kwargs).
    irf_kwargs = Dict{Symbol,Any}(:ci_type => Symbol(ci), :reps => reps)
    isnothing(_SEED[]) || (irf_kwargs[:seed] = _SEED[])
    irf_result = irf(sdfm, horizons; irf_kwargs...)

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    # C051: tidy long_table (horizon|variable|shock|value|lower|upper); irf(sdfm,...)
    # returns a panel-wide ImpulseResponse directly (see MEMs favar/analysis.jl), same
    # schema as irf var — one tidy table covers every shock.
    irf_df = long_table(irf_result)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="SDFM IRF ($id identification)", key="sdfm_irf")
    return (; model=sdfm, result=irf_result)
end
