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

# Shared utilities for command handlers

# ── Data Loading Helpers ───────────────────────────────────

"""
    load_multivariate_data(data) → (Y::Matrix{Float64}, varnames::Vector{String})

Load CSV or a typed data handle, convert to a numeric matrix and extract variable names.
CSV path is bit-identical to `load_data` + `df_to_matrix`; a handle uses `to_matrix`/`varnames`.
"""
function load_multivariate_data(data::String)
    obj = resolve_data(data)
    if obj isa DataFrame
        df = obj
        vn = variable_names(df)
        # Guard missing cells as a typed data error BEFORE df_to_matrix's Matrix{Float64}
        # conversion (which throws an untyped ArgumentError → uncaught exit-1). Mirrors the
        # univariate `load_univariate_series` guard so every multivariate estimator surfaces
        # a `data/missing-values` (exit 3) instead of an internal error.
        for c in vn
            any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
                "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
        end
        return df_to_matrix(df), vn
    end
    Y = to_matrix(obj)
    vn = Vector{String}(varnames(obj))
    return Y, vn
end

"""
    load_univariate_series(data, column) → (y::Vector{Float64}, vname::String)

Load CSV or a typed data handle and extract a single numeric column by index.
"""
function load_univariate_series(data::String, column::Int)
    obj = resolve_data(data)
    if !(obj isa DataFrame)
        Y = to_matrix(obj)
        vn = Vector{String}(varnames(obj))
        (column < 1 || column > length(vn)) && throw(CliError("data/column-range",
            "column $column out of range (data has $(length(vn)) numeric column(s))";
            hint="--column is 1-based; pick 1..$(length(vn))"))
        return Vector{Float64}(Y[:, column]), vn[column]
    end
    df = obj
    varnames_ = variable_names(df)
    (column < 1 || column > length(varnames_)) && throw(CliError("data/column-range",
        "column $column out of range (data has $(length(varnames_)) numeric column(s))";
        hint="--column is 1-based; pick 1..$(length(varnames_))"))
    col = df[!, varnames_[column]]
    any(ismissing, col) && throw(CliError("data/missing-values",
        "column '$(varnames_[column])' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    return Vector{Float64}(col), varnames_[column]
end

"""
    _load_midas_data(data, column, hf_data, hf_column; m) → (y_lf, x_hf, ynm, xnm)

Load a mixed-frequency MIDAS dataset from two CSVs: the low-frequency target
(`data`, numeric column `column`) and the high-frequency indicator (`hf_data`,
numeric column `hf_column`). Both are read through the already-hardened
`load_univariate_series`, so a missing/non-numeric cell or an out-of-range column
surfaces a typed `data/missing-values`/`data/column-range` (never an untyped
exit-1 from a `Vector{Float64}` conversion — the standing shared-loader lesson;
this is the 5th hardened shared loader after univariate/multivariate/reg/panel).

The HF series must supply AT LEAST `m` observations per low-frequency period,
i.e. `length(x_hf) >= m * length(y_lf)`. The estimator's internal `_align_hf`
anchors the *last* HF observation to the *last* low-frequency period and works
backwards, so any *leading* ragged edge (extra early HF history) is dropped
automatically — the natural nowcasting layout (a long high-frequency indicator
against a shorter low-frequency target) is fully supported. Only a HF series
*shorter* than `m×LF` is rejected as a typed `data/shape` (exit 3), since
end-anchoring would then silently drop low-frequency target periods. A `K`
larger than the available aligned HF history surfaces via the estimator's own
typed error, mapped by `_midas_error`.
"""
function _load_midas_data(data::String, column::Int, hf_data::String, hf_column::Int; m::Int)
    y_lf, ynm = load_univariate_series(data, column)
    x_hf, xnm = load_univariate_series(hf_data, hf_column)
    nlf = length(y_lf); nhf = length(x_hf)
    nhf >= m * nlf || throw(CliError("data/shape",
        "high-frequency series '$xnm' has only $nhf observation(s) but the low-frequency target '$ynm' has $nlf and --m=$m needs at least m×LF = $(m * nlf) high-frequency observations (≥ $m aligned per low-frequency period)";
        hint="check --m, --hf-column, and that the HF file covers at least the target window"))
    nhf > m * nlf && _status("MIDAS alignment: dropping $(nhf - m * nlf) leading high-frequency observation(s) of '$xnm' (anchoring the last HF obs to the last low-frequency period)")
    return y_lf, x_hf, ynm, xnm
end

# ── Naming Helpers ─────────────────────────────────────────

"""Safe shock name: uses variable name if in range, else "shock_N"."""
_shock_name(varnames::Vector{String}, idx::Int) =
    idx <= length(varnames) ? varnames[idx] : "shock_$idx"

"""Safe variable name: uses variable name if in range, else "var_N"."""
_var_name(varnames::Vector{String}, idx::Int) =
    idx <= length(varnames) ? varnames[idx] : "var_$idx"

"""Generate per-variable output path by inserting suffix before extension."""
function _per_var_output_path(output::String, suffix::String)
    isempty(output) && return ""
    _validate_output_path(output)
    return replace(output, "." => "_$(suffix).")
end

# ── Output Helpers ─────────────────────────────────────────

"""Build a coefficient table DataFrame for VAR/BVAR models."""
function _build_var_coef_table(coef_mat::AbstractMatrix, varnames::Vector{String}, p::Int)
    n = length(varnames)
    n_rows = size(coef_mat, 1)
    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end
    coef_df = DataFrame(permutedims(coef_mat), row_names)
    insertcols!(coef_df, 1, :equation => varnames)
    return coef_df
end

"""Output AIC/BIC/HQC/loglik for a VAR-like model."""
function output_model_criteria(model; format::String="table", output::String="", title::String="Information Criteria")
    pairs = Pair{String,Any}[
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
    ]
    if hasproperty(model, :loglik) || hasmethod(loglikelihood, (typeof(model),))
        try
            push!(pairs, "Log-likelihood" => loglikelihood(model))
        catch; end
    end
    # W3/#138: one stable key across all callers — title variants ("… (posterior
    # mean)") stay human-facing only.
    output_kv(pairs; format=format, title=title, key="information_criteria")
end

"""Output per-variable FEVD tables.

W3/#138: `key_prefix` is the leaf's registry-declared family prefix; each
variable's table is addressed `<key_prefix>_<varslug>`. Empty prefix falls back
to the title slug (pre-W3 behavior; check_table_keys.jl flags any such leaf).
"""
function _output_fevd_tables(proportions::AbstractArray, varnames::Vector{String},
                              horizons::Int; id::String="cholesky",
                              title_prefix::String="FEVD",
                              format::String="table", output::String="",
                              key_prefix::String="")
    n = length(varnames)
    for vi in 1:n
        fevd_df = DataFrame()
        fevd_df.horizon = 1:horizons
        for si in 1:n
            fevd_df[!, _shock_name(varnames, si)] = proportions[vi, si, :]
        end
        vname = _var_name(varnames, vi)
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="$title_prefix for $vname ($id identification)",
                      key=isempty(key_prefix) ? "" : _table_key(key_prefix, vname))
        _status()
    end
end

"""Output per-variable HD tables (W3/#138: family keys `<key_prefix>_<varslug>`)."""
function _output_hd_tables(get_contrib::Function, varnames::Vector{String},
                            T_eff::Int; id::String="cholesky",
                            title_prefix::String="Historical Decomposition",
                            format::String="table", output::String="",
                            actual=nothing, initial=nothing,
                            key_prefix::String="")
    n = length(varnames)
    for vi in 1:n
        hd_df = DataFrame()
        hd_df.period = 1:T_eff
        if !isnothing(actual)
            hd_df.actual = actual[:, vi]
        end
        if !isnothing(initial)
            hd_df.initial = initial[:, vi]
        end
        for si in 1:n
            hd_df[!, "contrib_$(_shock_name(varnames, si))"] = get_contrib(vi, si)
        end
        vname = _var_name(varnames, vi)
        output_result(hd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="$title_prefix: $vname ($id identification)",
                      key=isempty(key_prefix) ? "" : _table_key(key_prefix, vname))
        _status()
    end
end

# ── Validation Helpers ─────────────────────────────────────

"""Validate that a method string is in the allowed set."""
function validate_method(method::String, allowed::Vector{String}, context::String)
    method in allowed || error("unknown $context: $method (expected $(join(allowed, "|")))")
    return method
end

# ── Test Helpers ───────────────────────────────────────────

"""Print colored p-value interpretation for hypothesis tests."""
function interpret_test_result(pvalue::Real, reject_msg::String, accept_msg::String; level::Float64=0.05)
    _status()
    if pvalue < level
        _status_styled("-> $reject_msg\n"; color=:yellow)
    else
        _status_styled("-> $accept_msg\n"; color=:green)
    end
end

"""
Map an untyped MEMs `ArgumentError`/`DomainError` from the long-memory family
(ARFIMA / GPH / local Whittle) to a typed `CliError`, so bad input surfaces as a
`data`-class exit (3) rather than an uncaught internal exit-1 (standing lesson).
Typed MEMs domain errors are already handled centrally by `_domain_error_class`.
"""
function _long_memory_error(e, what::String)
    e isa CliError && return e
    if e isa ArgumentError || e isa DomainError
        return CliError("data/invalid", "$what: $(sprint(showerror, e))";
            hint="need a longer numeric series (>= 8 obs) and valid p/q/bandwidth")
    end
    return CliError("model/error", "$what failed: $(sprint(showerror, e))")
end

"""Convert trend string to Symbol for test regression kwarg."""
function to_regression_symbol(trend::String)
    trend == "none" && return :none
    trend == "both" && return :both
    return Symbol(trend)
end

# ── Volatility Output Helpers ──────────────────────────────

"""Standard normal CDF approximation (Abramowitz & Stegun)."""
function _normal_cdf(x::Real)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * exp(-x * x / 2.0) * t *
        (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    x >= 0.0 ? 1.0 - p : p
end

"""Standard normal quantile (inverse CDF), Acklam's rational approximation.

Companion to `_normal_cdf` so CI half-widths can be formed without a Distributions
dependency. Accurate to ~1e-9 over the range, which is far finer than the 6-digit
rounding every renderer applies.
"""
function _normal_quantile(p::Real)
    (0 < p < 1) || throw(DomainError(p, "_normal_quantile requires 0 < p < 1"))
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p > 1 - plow
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    end
    q = p - 0.5; r = q * q
    return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) * q /
           (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
end

"""Shared volatility model estimation output: coefficients + persistence."""
function _vol_estimate_output(model, vname::String, param_names::Vector{String},
                               model_label::String; format::String="table", output::String="",
                               key::String="")
    c = coef(model)
    names = param_names[1:length(c)]
    coef_df = try
        # SVModel has no vcov/stderror — StatsAPI defaults cause infinite recursion
        model isa SVModel && throw(ErrorException("no SE for SV"))
        se = stderror(model)
        z = c ./ se
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        DataFrame(parameter=names, estimate=round.(c; digits=6),
                  std_error=round.(se; digits=6),
                  z_stat=round.(z; digits=3), p_value=round.(pv; digits=4))
    catch
        DataFrame(parameter=names, estimate=round.(c; digits=6))
    end
    output_result(coef_df; format=Symbol(format), output=output,
                  title="$model_label Coefficients ($vname)", key=key)
    _status()
    p_val = persistence(model)
    _status("Persistence: $(round(p_val; digits=4))")
end

"""Shared volatility forecast output: horizon/variance/volatility table."""
function _vol_forecast_output(fc, vname::String, model_label::String,
                               horizons::Int; format::String="table", output::String="",
                               key::String="")
    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )
    output_result(fc_df; format=Symbol(format), output=output,
                  title="$model_label Volatility Forecast ($vname, h=$horizons)", key=key)
end

# ── Volatility model table (F10: 5 rows × 4 verbs = 20 leaves) ──

"""
One row drives estimate / predict / residuals / forecast for a volatility model.
`order`: `:q_only` (ARCH), `:pq` (GARCH family), `:sv` (stochastic vol).
`post_est`: extras after estimate — `:uc`, `:halflife`, `:halflife_uc`, or `:none`.
`post_fc`: extras after forecast — `:uc` or `:none`.
"""
const VOL_MODELS = [
    (
        name = "arch",
        order = :q_only,
        estimate = (y; p=1, q=1, draws=5000, dist=:normal) -> estimate_arch(y, q),
        # W11/#113: estimate_arch takes NO `dist` kwarg upstream — Gaussian QMLE only.
        supports_dist = false,
        param_names = (p, q) -> String["mu"; "omega"; ["alpha$i" for i in 1:q]],
        label = (p, q) -> "ARCH($q)",
        post_est = :uc,
        post_fc = :none,
        predict_title = (p, q) -> "ARCH($q) Conditional Variance",
        resid_title = (p, q) -> "ARCH($q) Standardized Residuals",
    ),
    (
        name = "garch",
        order = :pq,
        estimate = (y; p=1, q=1, draws=5000, dist=:normal) -> estimate_garch(y, p, q; dist=dist),
        supports_dist = true,
        param_names = (p, q) -> String["mu"; "omega"; ["alpha$i" for i in 1:q]; ["beta$i" for i in 1:p]],
        label = (p, q) -> "GARCH($p,$q)",
        post_est = :halflife_uc,
        post_fc = :uc,
        predict_title = (p, q) -> "GARCH($p,$q) Conditional Variance",
        resid_title = (p, q) -> "GARCH($p,$q) Standardized Residuals",
    ),
    (
        name = "egarch",
        order = :pq,
        estimate = (y; p=1, q=1, draws=5000, dist=:normal) -> estimate_egarch(y, p, q; dist=dist),
        supports_dist = true,
        param_names = (p, q) -> String["mu"; "omega"; ["alpha$i" for i in 1:q]; ["gamma$i" for i in 1:q]; ["beta$i" for i in 1:p]],
        label = (p, q) -> "EGARCH($p,$q)",
        post_est = :none,
        post_fc = :none,
        predict_title = (p, q) -> "EGARCH($p,$q) Conditional Variance",
        resid_title = (p, q) -> "EGARCH($p,$q) Standardized Residuals",
    ),
    (
        name = "gjr_garch",
        order = :pq,
        estimate = (y; p=1, q=1, draws=5000, dist=:normal) -> estimate_gjr_garch(y, p, q; dist=dist),
        supports_dist = true,
        param_names = (p, q) -> String["mu"; "omega"; ["alpha$i" for i in 1:q]; ["gamma$i" for i in 1:q]; ["beta$i" for i in 1:p]],
        label = (p, q) -> "GJR-GARCH($p,$q)",
        post_est = :halflife,
        post_fc = :none,
        predict_title = (p, q) -> "GJR-GARCH($p,$q) Conditional Variance",
        resid_title = (p, q) -> "GJR-GARCH($p,$q) Standardized Residuals",
    ),
    (
        name = "sv",
        order = :sv,
        estimate = (y; p=1, q=1, draws=5000, dist=:normal) -> estimate_sv(y; n_samples=draws, _fwd_seed()...),
        # SV is a stochastic-volatility sampler, not a GARCH likelihood — no `dist`.
        supports_dist = false,
        param_names = (p, q) -> String["mu", "phi", "sigma_eta"],
        label = (p, q) -> "SV",
        post_est = :none,
        post_fc = :none,
        predict_title = (p, q) -> "SV Posterior Mean Volatility",
        resid_title = (p, q) -> "SV Standardized Residuals",
    ),
]

function _vol_resolve_model(vol, data::String, column::Int; p::Int=1, q::Int=1,
                             draws::Int=5000, dist::Symbol=:normal, model=nothing)
    if !isnothing(model)
        return model, "series"
    end
    y, vname = load_univariate_series(data, column)
    return vol.estimate(y; p=p, q=q, draws=draws, dist=dist), vname
end

"""Validate `--dist` for a volatility leaf (W11/#113).

Only garch/egarch/gjr-garch take a conditional distribution upstream; arch and sv do not,
and figarch/fiegarch accept the kwarg but reject anything except `:normal`. The option is
therefore declared ONLY where it can be honoured, and this is the matching runtime guard.
"""
function _vol_dist_symbol(vol, dist::String, leaf::String)
    d = Symbol(dist)
    d in (:normal, :student, :ged) || throw(CliError("usage/invalid",
        "$leaf: --dist must be normal, student or ged (got '$dist')"))
    (d === :normal || vol.supports_dist) || throw(CliError("usage/invalid",
        "$leaf: $(vol.label(1, 1)) is Gaussian-QMLE only upstream — --dist $dist is not available";
        hint="a conditional t/GED likelihood is available on garch, egarch and gjr-garch"))
    return d
end

function _vol_post_status(model, kind::Symbol)
    if kind === :halflife || kind === :halflife_uc
        hl = halflife(model)
        _status("Half-life: $(round(hl; digits=2)) periods")
    end
    if kind === :uc || kind === :halflife_uc
        uc = unconditional_variance(model)
        _status("Unconditional variance: $(round(uc; digits=4))")
    end
end

function _make_estimate_vol(vol)
    return function (; data::String, column::Int=1, p::Int=1, q::Int=1, draws::Int=5000,
                      dist::String="normal", output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="")
        dsym = _vol_dist_symbol(vol, dist, "estimate $(vol.name)")
        y, vname = load_univariate_series(data, column)
        label = vol.label(p, q)
        if vol.order === :sv
            _status("Estimating Stochastic Volatility: variable=$vname, observations=$(length(y)), draws=$draws")
        else
            _status("Estimating $label: variable=$vname, observations=$(length(y))")
        end
        _status()
        model = try
            vol.estimate(y; p=p, q=q, draws=draws, dist=dsym)
        catch e
            throw(_domain_or_data_error(e, "$label estimation"))
        end
        _maybe_plot(model; plot=plot, plot_save=plot_save)
        _vol_estimate_output(model, vname, vol.param_names(p, q), label; format=format, output=output,
                             key="$(vol.name)_coefficients")
        # W11/#113: the shape parameter (t degrees of freedom / GED shape) is estimated
        # JOINTLY but lives in `model.shape`, outside `coef(model)` — so it never reached
        # the coefficient table. Without this the user selects a fat-tailed likelihood and
        # cannot see what was actually fitted. Distinct output path so `--output` cannot
        # drop the coefficients above.
        if dsym !== :normal
            sh = hasproperty(model, :shape) ? Float64(model.shape) : NaN
            output_result(DataFrame(parameter=["shape"], estimate=[round(sh; digits=6)],
                                    distribution=[String(dsym)]);
                          format=Symbol(format),
                          output=_per_var_output_path(output, "shape"),
                          title="Conditional Distribution ($(dsym === :student ? "Student-t degrees of freedom" : "GED shape"))",
                          key="conditional_distribution")
        end
        _vol_post_status(model, vol.post_est)
        return model
    end
end

function _make_forecast_vol(vol)
    return function (; data::String="", result=nothing, column::Int=1, p::Int=1, q::Int=1, draws::Int=5000,
                      dist::String="normal", horizons::Int=12,
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="", model=nothing)
        loaded = _loaded_result(result; data, model, leaf="forecast $(replace(vol.name, '_' => '-'))")
        if loaded !== nothing
            h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
            _vol_forecast_output(loaded, "result", vol.label(p, q), h; format=format, output=output,
                                 key="$(vol.name)_volatility_forecast")
            _maybe_plot(loaded; plot=plot, plot_save=plot_save)
            return loaded
        end
        dsym = _vol_dist_symbol(vol, dist, "forecast $(vol.name)")
        m, vname = _vol_resolve_model(vol, data, column; p=p, q=q, draws=draws,
                                      dist=dsym, model=model)
        label = vol.label(p, q)
        if vol.order === :sv
            _status("Stochastic Volatility Forecast: variable=$vname, horizons=$horizons, draws=$draws")
        else
            _status("$label Volatility Forecast: variable=$vname, horizons=$horizons")
        end
        _status()
        fc = forecast(m, horizons)
        _maybe_plot(fc; plot=plot, plot_save=plot_save)
        _vol_forecast_output(fc, vname, label, horizons; format=format, output=output,
                             key="$(vol.name)_volatility_forecast")
        if vol.post_fc !== :none
            _status()
            _vol_post_status(m, vol.post_fc)
        end
        return (; model=m, result=fc)
    end
end

function _make_predict_vol(vol)
    return function (; data::String="", column::Int=1, p::Int=1, q::Int=1, draws::Int=5000,
                      output::String="", format::String="table", model=nothing)
        m, vname = _vol_resolve_model(vol, data, column; p=p, q=q, draws=draws, model=model)
        cond_var = predict(m)
        if vol.order === :sv
            _status("SV posterior mean volatility: variable=$vname, draws=$draws")
        else
            _status("$(vol.label(p, q)) conditional variance: variable=$vname")
        end
        _status()
        pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                            volatility=round.(sqrt.(cond_var); digits=6))
        output_result(pred_df; format=Symbol(format), output=output,
                      title="$(vol.predict_title(p, q)) ($vname)",
                      key="$(vol.name)_conditional_variance")
        return cond_var
    end
end

function _make_residuals_vol(vol)
    return function (; data::String="", column::Int=1, p::Int=1, q::Int=1, draws::Int=5000,
                      output::String="", format::String="table", model=nothing)
        m, vname = _vol_resolve_model(vol, data, column; p=p, q=q, draws=draws, model=model)
        resid = residuals(m)
        if vol.order === :sv
            _status("SV standardized residuals: variable=$vname, draws=$draws")
        else
            _status("$(vol.label(p, q)) standardized residuals: variable=$vname")
        end
        _status()
        res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
        output_result(res_df; format=Symbol(format), output=output,
                      title="$(vol.resid_title(p, q)) ($vname)",
                      key="$(vol.name)_standardized_residuals")
        return resid
    end
end

# Cached per-model handlers (built once at load)
const _VOL_ESTIMATE_HANDLERS = Dict(v.name => _make_estimate_vol(v) for v in VOL_MODELS)
const _VOL_FORECAST_HANDLERS = Dict(v.name => _make_forecast_vol(v) for v in VOL_MODELS)
const _VOL_PREDICT_HANDLERS = Dict(v.name => _make_predict_vol(v) for v in VOL_MODELS)
const _VOL_RESIDUALS_HANDLERS = Dict(v.name => _make_residuals_vol(v) for v in VOL_MODELS)

# Back-compat aliases used by tests / direct handler calls
const _estimate_arch = _VOL_ESTIMATE_HANDLERS["arch"]
const _estimate_garch = _VOL_ESTIMATE_HANDLERS["garch"]
const _estimate_egarch = _VOL_ESTIMATE_HANDLERS["egarch"]
const _estimate_gjr_garch = _VOL_ESTIMATE_HANDLERS["gjr_garch"]
const _estimate_sv = _VOL_ESTIMATE_HANDLERS["sv"]
const _forecast_arch = _VOL_FORECAST_HANDLERS["arch"]
const _forecast_garch = _VOL_FORECAST_HANDLERS["garch"]
const _forecast_egarch = _VOL_FORECAST_HANDLERS["egarch"]
const _forecast_gjr_garch = _VOL_FORECAST_HANDLERS["gjr_garch"]
const _forecast_sv = _VOL_FORECAST_HANDLERS["sv"]
const _predict_arch = _VOL_PREDICT_HANDLERS["arch"]
const _predict_garch = _VOL_PREDICT_HANDLERS["garch"]
const _predict_egarch = _VOL_PREDICT_HANDLERS["egarch"]
const _predict_gjr_garch = _VOL_PREDICT_HANDLERS["gjr_garch"]
const _predict_sv = _VOL_PREDICT_HANDLERS["sv"]
const _residuals_arch = _VOL_RESIDUALS_HANDLERS["arch"]
const _residuals_garch = _VOL_RESIDUALS_HANDLERS["garch"]
const _residuals_egarch = _VOL_RESIDUALS_HANDLERS["egarch"]
const _residuals_gjr_garch = _VOL_RESIDUALS_HANDLERS["gjr_garch"]
const _residuals_sv = _VOL_RESIDUALS_HANDLERS["sv"]

# ── IRF table assembly (F11) ───────────────────────────────

"""
    build_irf_table(irf_data, ci_lower, ci_upper, varnames, shock, horizons=nothing;
                    lower_suffix="_lower", upper_suffix="_upper", digits=nothing) -> DataFrame

Build a single-shock IRF table: `horizon` + one column per variable, optionally with CI bands.

`irf_data` is `H×n×S` (3D) or `H×n` (2D). When 3D, `shock` indexes the third dimension.
`horizons` defaults to `0:(H-1)`; pass an integer `Hmax` for `0:Hmax`, or any iterable of length H.
"""
function build_irf_table(irf_data::AbstractArray, ci_lower, ci_upper,
                         varnames::AbstractVector{<:AbstractString}, shock::Int,
                         horizons=nothing;
                         lower_suffix::String="_lower",
                         upper_suffix::String="_upper",
                         digits::Union{Nothing,Int}=nothing)
    nd = ndims(irf_data)
    nd == 2 || nd == 3 || error("irf_data must be 2D (H×n) or 3D (H×n×S), got ndims=$nd")
    n_h = size(irf_data, 1)
    n_v = length(varnames)

    slice = if nd == 3
        (arr, vi) -> arr[:, vi, shock]
    else
        (arr, vi) -> arr[:, vi]
    end

    hvals = if isnothing(horizons)
        0:(n_h - 1)
    elseif horizons isa Integer
        0:horizons
    else
        horizons
    end
    length(hvals) == n_h || error("horizons length $(length(hvals)) != IRF length $n_h")

    _maybe_round(x) = isnothing(digits) ? x : round.(x; digits=digits)

    irf_df = DataFrame(horizon=collect(hvals))
    for (vi, vname) in enumerate(varnames)
        vi > size(irf_data, 2) && break
        irf_df[!, String(vname)] = _maybe_round(slice(irf_data, vi))
    end
    if !isnothing(ci_lower)
        for (vi, vname) in enumerate(varnames)
            vi > size(ci_lower, 2) && break
            irf_df[!, "$(vname)$(lower_suffix)"] = _maybe_round(slice(ci_lower, vi))
        end
    end
    if !isnothing(ci_upper)
        for (vi, vname) in enumerate(varnames)
            vi > size(ci_upper, 2) && break
            irf_df[!, "$(vname)$(upper_suffix)"] = _maybe_round(slice(ci_upper, vi))
        end
    end
    return irf_df
end

# ── Constants ──────────────────────────────────────────────

"""
    ID_METHOD_MAP

Maps CLI identification method strings to MacroEconometricModels symbols.
"""
const ID_METHOD_MAP = Dict(
    "cholesky"          => :cholesky,
    "sign"              => :sign,
    "narrative"         => :narrative,
    "longrun"           => :long_run,
    "fastica"           => :fastica,
    "jade"              => :jade,
    "sobi"              => :sobi,
    "dcov"              => :dcov,
    "hsic"              => :hsic,
    "student_t"         => :student_t,
    "mixture_normal"    => :mixture_normal,
    "pml"               => :pml,
    "skew_normal"       => :skew_normal,
    "markov_switching"  => :markov_switching,
    "garch_id"          => :garch,
    "uhlig"             => :uhlig,
    "lewis-tvv"         => :lewis_tvv,
    "sv-em"             => :sv_em,
)

# W2/#166: method universes per estimator family. The base map above is shared
# with the LP/PVAR/FAVAR paths (whose upstream entry points accept exactly it);
# the VAR and VECM families each admit more. A per-leaf allow-set keeps a method
# valid on one family from silently degrading to :cholesky on another.
const _ID_METHODS_VAR = merge(ID_METHOD_MAP, Dict(
    "proxy"          => :proxy,
    "max-share"      => :max_share,
    "gmm-moments"    => :gmm_moments,
    "narrative-adrr" => :narrative_adrr,
))
const _ID_METHODS_VECM = merge(ID_METHOD_MAP, Dict(
    "svec" => :svec,
))

"""
    _identification_method(id, methods, leaf) → Symbol

Resolve a user `--id` against a family allow-set. Unknown ids used to fall back
to `:cholesky` silently (a mislabelled result with exit 0); now `usage/invalid`.
"""
function _identification_method(id::String, methods::Dict, leaf::String)
    haskey(methods, id) && return methods[id]
    valid = join(sort!(collect(keys(methods))), "|")
    throw(CliError("usage/invalid", "$leaf: --id must be one of $valid (got '$id')"))
end

"""
    _resolve_target_var(target, varnames, leaf) → Union{Int,String}

A `--target-var` may name a variable or give a 1-based index (upstream accepts
both); resolve names against the CSV columns up front so a miss is
`usage/invalid`, never an upstream `BoundsError` (exit 1).
"""
function _resolve_target_var(target::String, varnames::Vector{String}, leaf::String)
    maybe_int = tryparse(Int, target)
    if maybe_int !== nothing
        1 <= maybe_int <= length(varnames) || throw(CliError("usage/invalid",
            "$leaf: --target-var index $maybe_int out of 1:$(length(varnames))"))
        return maybe_int
    end
    target in varnames && return target
    throw(CliError("usage/invalid",
        "$leaf: --target-var '$target' not a column; available: $(join(varnames, ", "))"))
end

"""
    _inject_svar_id_kwargs!(kwargs, id, leaf, data, varnames, instrument_col, target_var)

W2/#166 extras for the VAR-family `--id` methods that need more than a TOML
config: proxy needs an instrument column (loaded from the CSV, so the data path
is required — instruments are not stored on models); max-share needs a target
variable (name or 1-based index). Both directions guarded: a missing extra is
`usage/missing`, an extra with the wrong id is `usage/invalid` (never a silent
no-op, never an upstream `BoundsError`).
"""
function _inject_svar_id_kwargs!(kwargs::Dict, id::String, leaf::String, data::String,
                                 varnames::Vector{String}, instrument_col::String,
                                 target_var::String)
    if id == "proxy"
        isempty(instrument_col) && throw(CliError("usage/missing",
            "$leaf: --id proxy requires --instrument <column>";
            hint="name a numeric, missing-free proxy column, e.g. --instrument mp_shock"))
        isempty(data) && throw(CliError("usage/missing",
            "$leaf: --id proxy requires --data (the instrument column lives in the CSV)"))
        kwargs[:instruments] = _load_instrument(data, instrument_col)
    elseif !isempty(instrument_col)
        throw(CliError("usage/invalid",
            "$leaf: --instrument applies only to --id proxy (got --id $id)"))
    end
    if id == "max-share"
        isempty(target_var) && throw(CliError("usage/missing",
            "$leaf: --id max-share requires --target-var <variable>";
            hint="a column name or 1-based index, e.g. --target-var gdp"))
        kwargs[:target] = _resolve_target_var(target_var, varnames, leaf)
    elseif !isempty(target_var)
        throw(CliError("usage/invalid",
            "$leaf: --target-var applies only to --id max-share (got --id $id)"))
    end
    return kwargs
end

"""
    _load_and_estimate_var(data, lags) -> (model, Y, varnames, p)

Load data from CSV, optionally auto-select lag order, and estimate a frequentist VAR.
"""
function _load_and_estimate_var(data::String, lags)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    # Forward the CSV column names (#119): without this the model renders y1..yn
    # and every downstream IRF/FEVD/forecast table loses the user's names.
    model = estimate_var(Y, p; varnames=varnames)
    return model, Y, varnames, p
end

"""
    _fwd_seed() → NamedTuple

Forward the global `--seed` as estimators' own `seed=` (W3/#167 — extends the
C052/#243 BVAR pattern to every `seed=`-accepting estimator at MEMs 0.9.3, where
it additionally records a `ReproManifest` for `reproduce`). Empty when `--seed`
was not given, so library defaults are untouched — including `seed::Int=<const>`
estimators (Krusell–Smith, spec tests), which cannot receive `nothing`.
Splat into estimator kwargs: `estimate_sv(y; n_samples=n, _fwd_seed()...)`.
"""
_fwd_seed() = _SEED[] === nothing ? NamedTuple() : (; seed=_SEED[])

"""
    _load_and_estimate_bvar(data, lags, config, draws, sampler) -> (post, Y, varnames, p, n)

Load data from CSV, build prior, and estimate a Bayesian VAR.
Returns a BVARPosterior (which carries p, n, data internally).
"""
function _load_and_estimate_bvar(data::String, lags::Int, config::String,
                                  draws::Int, sampler::String;
                                  hyperopt::String="glp")
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    p = lags

    hopt = lowercase(strip(hyperopt))
    hopt in ("glp", "grid") || throw(CliError("usage/invalid-option",
        "invalid --hyperopt '$hyperopt'; must be glp or grid"))

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? :normal : :minnesota

    # --hyperopt only bites when the hyperparameters are NOT pinned: upstream ignores it
    # entirely once `hyper` is supplied. Config therefore wins over the flag, and saying so
    # on stderr beats letting a user believe a flag took effect when it could not.
    if prior_obj !== nothing && hopt != "glp"
        _status("--hyperopt=$hopt ignored: [prior] in the config pins the hyperparameters")
    end

    # Forward --seed as the estimator's own seed (C052/#243): estimate_bvar seeds a
    # fresh Xoshiro(seed) and records it in the BVARPosterior ReproManifest,
    # so a saved posterior reproduces bit-for-bit. `nothing` → library default RNG.
    post = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_draws=draws,
        prior=prior_sym, hyper=prior_obj, hyperopt=Symbol(hopt),
        varnames=varnames, seed=_SEED[])

    return post, Y, varnames, p, n
end

"""
    _build_prior(config_path, Y, p) -> MinnesotaHyperparameters or nothing

Build a Minnesota prior from TOML config, or return nothing for default prior.
"""
function _build_prior(config_path::String, Y::AbstractMatrix, p::Int)
    if isempty(config_path)
        return nothing
    end
    cfg = load_config(config_path)
    prior_cfg = get_prior(cfg)

    if prior_cfg["type"] == "minnesota"
        if prior_cfg["optimize"]
            _status("Optimizing Minnesota prior hyperparameters...")
            return optimize_hyperparameters(Y, p)
        else
            # `omega` is a SCALAR weight on the residual-covariance prior, not the
            # per-variable AR scale. This used to pass a length-n Vector of AR residual
            # standard deviations, which real MEMs rejects outright
            # (`TypeError: in keyword argument omega, expected Real`) — an untyped exit 1 on
            # EVERY `--config` minnesota run across the whole BVAR family. The mock's
            # `omega::Vector{Float64}` accepted it, so T1/T2 stayed green; there was no T3
            # coverage of the config path. Per-variable σᵢ scaling is MEMs' own job inside
            # `gen_dummy_obs`, so there is nothing for the CLI to compute here.
            #
            # The config schema exposes only lambda1/2/3, so `mu` and `omega` keep upstream's
            # defaults rather than being invented from the data. NOTE (MEMs 0.7.3, #529):
            # `omega` is now the REPLICATION WEIGHT of the diag(σ̂) covariance dummy
            # (default moved 2.0 → 1.0), no longer an on/off switch — config-minnesota
            # results shifted at that bump for exactly this reason.
            return MinnesotaHyperparameters(;
                tau=get(prior_cfg, "lambda1", 0.2),
                decay=get(prior_cfg, "lambda3", 1.0),
                lambda=get(prior_cfg, "lambda2", 0.5),
            )
        end
    end
    return nothing
end

"""
    _require_rkeys(entry, keys, listname) → nothing

A TOML restriction entry missing a required key is `config/missing` (never a
`KeyError` exit 1).
"""
function _require_rkeys(entry, keys::Vector{String}, listname::String)
    for k in keys
        haskey(entry, k) || throw(CliError("config/missing",
            "identification.$listname entry $entry is missing required key '$k'"))
    end
end

function _parse_horizon_range(raw, what::String)
    (raw isa AbstractVector && length(raw) == 2) || throw(CliError("config/invalid",
        "$what must be a 2-element [lo, hi] range (got $raw)"))
    lo, hi = Int(raw[1]), Int(raw[2])
    (1 <= lo <= hi) || throw(CliError("config/invalid",
        "$what range must satisfy 1 ≤ lo ≤ hi (got [$lo, $hi])"))
    return lo:hi
end

"""
    _load_svar_restrictions(config_path, n, label) → (cfg, SVARRestrictions)

W2/#166: one builder for every declarative restriction kind, shared by the
arias/uhlig/narrative-adrr branches of irf/fevd/hd (which previously each
hand-rolled the zero/sign subset). Additive schema — old keys keep working:

- `zero_restrictions`: `{var, shock, horizon}`; `horizon = "long_run"` selects
  the long-run zero (#743).
- `sign_restrictions`: `{var, shock, sign, horizon}` or `{..., horizons=[lo,hi]}`
  for a horizon range (#743, expands to one restriction per horizon).
- `a0_zero_restrictions` / `a0_sign_restrictions`: `{equation, shock[, sign]}`.
- `elasticity_bounds`: `{numerator, denominator, shock, horizon, lower, upper}`
  (one-sided bounds allowed — a missing side defaults to ±Inf).
- `magnitude_bounds`: `{variable, shock, horizon, lower, upper}` (both required).
- `cumulative_restrictions`: `{variable, shock, sign, horizons=[lo,hi]}`.
- `narrative_shocks`: `{shock, dates=[...], sign}` (Antolín-Díaz / Rubio-Ramírez).
- `narrative_contributions`: `{variable, shock, window=[lo,hi], kind}` — ADRR
  Type A/B (`most_important`/`overwhelming`, default `most_important`).

Index/range/enum validation the builders perform stays upstream (`ArgumentError`
→ `data/invalid`); structural TOML problems (missing keys, malformed ranges)
are `config/*` here.
"""
function _load_svar_restrictions(config_path::String, n::Int, label::String)
    isempty(config_path) && throw(CliError("usage/missing",
        "$label identification requires a --config file with restrictions"))
    cfg = load_config(config_path)
    id_cfg = get(cfg, "identification", Dict())
    zero_restrs = Any[]
    sign_restrs = Any[]
    for r in get(id_cfg, "zero_restrictions", [])
        _require_rkeys(r, ["var", "shock"], "zero_restrictions")
        h = get(r, "horizon", 0)
        push!(zero_restrs, h == "long_run" ?
            zero_restriction(r["var"], r["shock"]; horizon=:long_run) :
            zero_restriction(r["var"], r["shock"]; horizon=Int(h)))
    end
    for r in get(id_cfg, "sign_restrictions", [])
        _require_rkeys(r, ["var", "shock", "sign"], "sign_restrictions")
        if haskey(r, "horizons")
            append!(sign_restrs, sign_restriction(r["var"], r["shock"], Symbol(r["sign"]);
                                                 horizons=_parse_horizon_range(r["horizons"], "sign_restrictions.horizons")))
        else
            push!(sign_restrs, sign_restriction(r["var"], r["shock"], Symbol(r["sign"]);
                                               horizon=Int(get(r, "horizon", 0))))
        end
    end
    for r in get(id_cfg, "a0_zero_restrictions", [])
        _require_rkeys(r, ["equation", "shock"], "a0_zero_restrictions")
        push!(zero_restrs, a0_zero_restriction(r["equation"], r["shock"]))
    end
    for r in get(id_cfg, "a0_sign_restrictions", [])
        _require_rkeys(r, ["equation", "shock", "sign"], "a0_sign_restrictions")
        push!(sign_restrs, a0_sign_restriction(r["equation"], r["shock"], Symbol(r["sign"])))
    end
    for r in get(id_cfg, "elasticity_bounds", [])
        _require_rkeys(r, ["numerator", "denominator", "shock"], "elasticity_bounds")
        push!(sign_restrs, elasticity_bound(r["numerator"], r["denominator"], r["shock"];
                                           horizon=Int(get(r, "horizon", 0)),
                                           lower=Float64(get(r, "lower", -Inf)),
                                           upper=Float64(get(r, "upper", Inf))))
    end
    for r in get(id_cfg, "magnitude_bounds", [])
        _require_rkeys(r, ["variable", "shock", "lower", "upper"], "magnitude_bounds")
        push!(sign_restrs, magnitude_bound(r["variable"], r["shock"];
                                          horizon=Int(get(r, "horizon", 0)),
                                          lower=Float64(r["lower"]), upper=Float64(r["upper"])))
    end
    for r in get(id_cfg, "cumulative_restrictions", [])
        _require_rkeys(r, ["variable", "shock", "sign", "horizons"], "cumulative_restrictions")
        push!(sign_restrs, cumulative_restriction(r["variable"], r["shock"], Symbol(r["sign"]);
                                                 horizons=_parse_horizon_range(r["horizons"], "cumulative_restrictions.horizons")))
    end
    for r in get(id_cfg, "narrative_shocks", [])
        _require_rkeys(r, ["shock", "dates", "sign"], "narrative_shocks")
        push!(sign_restrs, narrative_shock_restriction(r["shock"], collect(Int, r["dates"]), Symbol(r["sign"])))
    end
    for r in get(id_cfg, "narrative_contributions", [])
        _require_rkeys(r, ["variable", "shock", "window"], "narrative_contributions")
        push!(sign_restrs, narrative_contribution_restriction(r["variable"], r["shock"],
                                                             _parse_horizon_range(r["window"], "narrative_contributions.window");
                                                             kind=Symbol(get(r, "kind", "most_important"))))
    end
    return cfg, SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
end

"""
    _svar_toml_matrix(mat, n, what, leaf) -> Matrix{Float64}

Read an n×n AB-model pattern matrix from a TOML `[[...]]` array-of-arrays.
TOML `nan` decodes to `NaN`, which is upstream's free-parameter marker
(`_ab_is_free(x) = isnan(x)`); any fixed number is a calibrated entry.
Shape/cell problems are `usage/invalid`; value problems (non-square is
already excluded here) stay upstream (`ArgumentError` → `data/invalid`).
"""
function _svar_toml_matrix(mat, n::Int, what::String, leaf::String; table::String="svar")
    (mat isa Vector && length(mat) == n &&
     all(r -> r isa Vector && length(r) == n, mat)) ||
        throw(CliError("usage/invalid",
            "$leaf: [$table] $what must be a $n×$n matrix (n rows of n numbers; TOML `nan` = free parameter)"))
    M = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        v = mat[i][j]
        (v isa Real) || throw(CliError("usage/invalid",
            "$leaf: [$table] $what cell [$i,$j] must be a number or `nan` (got $(repr(v)))"))
        M[i, j] = Float64(v)
    end
    M
end

"""
    _load_svar_pattern(config_path, n, pattern, leaf) -> SVARPattern

W2/#166: build the AB-model pattern for `estimate svar`. `recursive` and
`blanchard-quah` need only n; the A/B/AB-model kinds read their matrices from
the `[svar]` TOML table (`A`, `B`, optional `long_run`) via `_svar_toml_matrix`.
"""
function _load_svar_pattern(config_path::String, n::Int, pattern::String, leaf::String)
    pattern == "recursive" && return recursive_pattern(n)
    pattern == "blanchard-quah" && return blanchard_quah_pattern(n)
    isempty(config_path) && throw(CliError("usage/missing",
        "$leaf: --pattern $pattern requires a --config file with [svar] matrices"))
    cfg = load_config(config_path)
    svar_cfg = get(cfg, "svar", Dict())
    if pattern == "a-model"
        haskey(svar_cfg, "A") || throw(CliError("usage/missing",
            "$leaf: --pattern a-model requires [svar] A in --config"))
        return a_model_pattern(_svar_toml_matrix(svar_cfg["A"], n, "A", leaf))
    elseif pattern == "b-model"
        haskey(svar_cfg, "B") || throw(CliError("usage/missing",
            "$leaf: --pattern b-model requires [svar] B in --config"))
        return b_model_pattern(_svar_toml_matrix(svar_cfg["B"], n, "B", leaf))
    elseif pattern == "ab-model"
        (haskey(svar_cfg, "A") && haskey(svar_cfg, "B")) || throw(CliError("usage/missing",
            "$leaf: --pattern ab-model requires [svar] A and B in --config"))
        A = _svar_toml_matrix(svar_cfg["A"], n, "A", leaf)
        B = _svar_toml_matrix(svar_cfg["B"], n, "B", leaf)
        lr = haskey(svar_cfg, "long_run") ?
            _svar_toml_matrix(svar_cfg["long_run"], n, "long_run", leaf) : nothing
        return ab_model_pattern(A, B; long_run=lr)
    end
    throw(CliError("usage/invalid", "$leaf: unknown --pattern $pattern"))
end

"""
    _load_svec_zeros(config_path, n, leaf) -> (long_run_zeros, short_run_zeros)

W2/#166: read the optional `[svec]` TOML matrices for `estimate svec`
(`long_run_zeros` / `short_run_zeros`, same n×n `nan`-means-free convention as
`_svar_toml_matrix`). Either key absent → `nothing`, which keeps upstream's
KPSW default for that side; no `--config` at all → `(nothing, nothing)`, i.e.
the fully default KPSW identification.
"""
function _load_svec_zeros(config_path::String, n::Int, leaf::String)
    isempty(config_path) && return nothing, nothing
    cfg = load_config(config_path)
    svec_cfg = get(cfg, "svec", Dict())
    lr = haskey(svec_cfg, "long_run_zeros") ?
        _svar_toml_matrix(svec_cfg["long_run_zeros"], n, "long_run_zeros", leaf; table="svec") : nothing
    sr = haskey(svec_cfg, "short_run_zeros") ?
        _svar_toml_matrix(svec_cfg["short_run_zeros"], n, "short_run_zeros", leaf; table="svec") : nothing
    return lr, sr
end

"""
    _build_check_func(config_path) -> (check_func, narrative_check)

Build sign restriction and narrative restriction check functions from TOML config.
Returns `(nothing, nothing)` if no config or no restrictions.
"""
function _build_check_func(config_path::String)
    if isempty(config_path)
        return nothing, nothing
    end
    cfg = load_config(config_path)
    id_cfg = get_identification(cfg)

    check_func = nothing
    narrative_check = nothing

    if haskey(id_cfg, "sign_matrix")
        sign_mat = id_cfg["sign_matrix"]
        horizons = get(id_cfg, "horizons", [0])
        check_func = function(irf_values)
            for h_idx in 1:length(horizons)
                h = horizons[h_idx] + 1  # 1-based indexing
                if h > size(irf_values, 1)
                    continue
                end
                for i in 1:size(sign_mat, 1)
                    for j in 1:size(sign_mat, 2)
                        s = sign_mat[i, j]
                        if s != 0
                            if s > 0 && irf_values[h, j, i] < 0
                                return false
                            elseif s < 0 && irf_values[h, j, i] > 0
                                return false
                            end
                        end
                    end
                end
            end
            return true
        end
    end

    if haskey(id_cfg, "narrative")
        narr = id_cfg["narrative"]
        shock_idx = narr["shock_index"]
        periods = narr["periods"]
        signs = narr["signs"]
        narrative_check = function(structural_shocks)
            for (t, s) in zip(periods, signs)
                if t > size(structural_shocks, 1)
                    continue
                end
                if s > 0 && structural_shocks[t, shock_idx] < 0
                    return false
                elseif s < 0 && structural_shocks[t, shock_idx] > 0
                    return false
                end
            end
            return true
        end
    end

    return check_func, narrative_check
end

"""
    _build_identification_kwargs(id, config) -> Dict{Symbol,Any}

Build the kwargs dict for irf/fevd/historical_decomposition calls
based on identification method and config file.
"""
function _build_identification_kwargs(id::String, config::String;
                                      methods::Dict=ID_METHOD_MAP,
                                      nvars::Union{Int,Nothing}=nothing,
                                      leaf::String="identification")
    # Unknown ids no longer degrade to :cholesky (W2/#166). Families admitting
    # more pass their own allow-set (VAR: _ID_METHODS_VAR; VECM: _ID_METHODS_VECM).
    method = _identification_method(id, methods, "identification")
    kwargs = Dict{Symbol,Any}(:method => method)

    check_func, narrative_check = _build_check_func(config)
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    if id in ("lewis-tvv", "sv-em")
        # Every call site passes its data width (hetero resolution needs n).
        nvars === nothing && error("$leaf: --id $id requires nvars (internal)")
        merge!(kwargs, _id_knob_kwargs(id, config, nvars, leaf))
    end

    return kwargs
end

"""
    _id_knob_kwargs(id, config, n, leaf) -> Dict{Symbol,Any}

Estimator knobs for the 0.9.6 statistical-ID methods, threaded into
irf/fevd/historical_decomposition calls (var/vecm/bvar/favar) whose
`compute_Q` branch forwards them to `identify_lewis_tvv` /
`identify_sv_svar` (kwarg names verified collision-free against
those path signatures at W0; LP leaves use upstream defaults —
structural_lp pins its own compute_Q allow-list). TOML-authored values
are validated in the `get_*_params` parsers (`config/invalid`);
`hetero_shocks` (1-based, TOML-authored) resolves against the
data width `n` here: out-of-range → `usage/invalid` (the
`--target-var` precedent — a column reference, not a config
shape). Empty means all shocks (upstream requires
`any(hetero)`); duplicates collapse (set semantics).
"""
function _id_knob_kwargs(id::String, config::String, n::Int, leaf::String)
    cfg = isempty(config) ? Dict{String,Any}() : load_config(config)
    out = Dict{Symbol,Any}()
    if id == "lewis-tvv"
        out[:weighting] = get_lewis_tvv_params(cfg)["weighting"]
    elseif id == "sv-em"
        sp = get_sv_svar_params(cfg)
        out[:maxiter] = sp["maxiter"]
        out[:gibbs_burn] = sp["gibbs_burn"]
        out[:gibbs_draws] = sp["gibbs_draws"]
        out[:init] = sp["init"]
        hs = sp["hetero_shocks"]
        for h in hs
            h <= n || throw(CliError("usage/invalid",
                "$leaf: --id sv-em hetero_shocks index $h out of 1:$n"))
        end
        out[:hetero] = isempty(hs) ? trues(n) : BitVector([i in hs for i in 1:n])
    end
    return out
end

"""
    _load_and_structural_lp(data, horizons, lags, var_lags, id, vcov, config;
                            ci_type=:none, reps=200, conf_level=0.95)

Load data, build identification kwargs, and compute structural LP.
Returns `(slp, Y, varnames)`.
"""
function _load_and_structural_lp(data::String, horizons::Int, lags::Int,
                                  var_lags, id::String, vcov::String,
                                  config::String;
                                  ci_type::Symbol=:none, reps::Int=200,
                                  conf_level::Float64=0.95)
    Y, varnames = load_multivariate_data(data)

    method = _identification_method(id, ID_METHOD_MAP, "structural lp")
    check_func, narrative_check = _build_check_func(config)

    vp = isnothing(var_lags) ? lags : var_lags

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :lags => lags,
        :var_lags => vp,
        :cov_type => Symbol(vcov),
        :ci_type => ci_type,
        :reps => reps,
        :conf_level => conf_level,
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    _SEED[] !== nothing && (kwargs[:seed] = _SEED[])
    # NOTE (W1/#186): no knob merge here — upstream structural_lp pins its
    # own compute_Q allow-list (lp/core.jl:440-443), so lewis/sv knobs have
    # no channel on LP leaves (methods run on upstream defaults, same as
    # the pre-existing uhlig-knob behavior on LP).
    slp = structural_lp(Y, horizons; kwargs...)
    return slp, Y, varnames
end

"""
    _load_and_estimate_vecm(data, lags, rank, deterministic, method, significance)
        -> (vecm, Y, varnames, p)

Load data from CSV and estimate a VECM. When rank=="auto", uses select_vecm_rank().
"""
function _load_and_estimate_vecm(data::String, lags::Int, rank::String,
                                  deterministic::String, method::String,
                                  significance::Float64)
    Y, varnames = load_multivariate_data(data)

    r = if rank == "auto"
        select_vecm_rank(Y, lags; significance=significance)
    else
        parse(Int, rank)
    end

    vecm = estimate_vecm(Y, lags; rank=r, deterministic=Symbol(deterministic),
                         method=Symbol(method), significance=significance,
                         varnames=varnames)
    return vecm, Y, varnames, lags
end

"""
    _var_forecast_point(B, Y, p, horizons) -> Matrix{Float64}

Iterate the VAR(p) equation h steps ahead to produce point forecasts.
B is the coefficient matrix (k × n), Y is the data matrix (T × n).
Returns a (horizons × n) matrix of forecast values.
"""
function _var_forecast_point(B::AbstractMatrix, Y::AbstractMatrix, p::Int, horizons::Int)
    T, n = size(Y)
    has_const = size(B, 1) > n * p

    forecasts = zeros(horizons, n)

    # lag_buf: [Y_T, Y_{T-1}, ..., Y_{T-p+1}] flattened to np vector
    lag_buf = zeros(n * p)
    for lag in 1:p
        lag_buf[(lag-1)*n+1:lag*n] = Y[T-lag+1, :]
    end

    for h in 1:horizons
        x = has_const ? vcat(lag_buf, 1.0) : lag_buf
        y_hat = B' * x
        forecasts[h, :] = y_hat

        # Shift lag buffer forward
        if p > 1
            lag_buf[n+1:end] = lag_buf[1:end-n]
        end
        lag_buf[1:n] = y_hat
    end

    return forecasts
end

# ── Panel VAR Helpers ─────────────────────────────────────

"""
    _parse_varlist(str) -> Vector{String}

Parse a comma-separated variable list string. Returns empty vector for empty input.
"""
function _parse_varlist(str::String)
    isempty(str) && return String[]
    return [strip(s) for s in split(str, ",") if !isempty(strip(s))]
end

"""
    _parse_lag_spec(s, flag; min=0) -> :auto | Int | Vector{Int}

Parse an ARDL/NARDL `--p`/`--q` lag spec: `"auto"` → `:auto`; a bare integer `≥ min` →
`Int`; a comma-separated list `"2,1,3"` → `Vector{Int}` (each `≥ min`). A bad token → typed
`usage/invalid` (never a raw parse throw). `min` is the smallest admissible value (`1` for an
AR order `--p`, `0` for a DL order `--q`). The per-regressor length check (`== k`) is done by
the caller, which knows the regressor count. (C062b)
"""
function _parse_lag_spec(s::AbstractString, flag::String; min::Int=0)
    s == "auto" && return :auto
    if occursin(',', s)
        toks = [strip(t) for t in split(s, ",") if !isempty(strip(t))]
        isempty(toks) && throw(CliError("usage/invalid", "$flag: empty list"))
        out = Int[]
        for t in toks
            v = tryparse(Int, t)
            (v === nothing || v < min) && throw(CliError("usage/invalid",
                "$flag: '$t' must be an integer ≥ $min"))
            push!(out, v)
        end
        return out
    end
    v = tryparse(Int, s)
    (v === nothing || v < min) && throw(CliError("usage/invalid",
        "$flag must be 'auto', an integer ≥ $min, or a comma-separated list of such; got '$s'"))
    return v
end

"""
    _parse_asym_spec(s) -> :all | Vector{Int}

Parse a NARDL `--asymmetric` spec: `"all"` → `:all` (split every regressor); else a
comma-separated list of 1-based regressor indices → `Vector{Int}` (non-empty, each `≥ 1`).
An empty result → typed `usage/invalid` (a fully-symmetric model → use `estimate ardl`). The
upper bound (`≤ k₀`) is checked by the caller, which knows the regressor count. (C062b)
"""
function _parse_asym_spec(s::AbstractString)
    s == "all" && return :all
    idxs = Int[]
    for t in split(s, ",")
        ts = strip(t)
        isempty(ts) && continue
        v = tryparse(Int, ts)
        (v === nothing || v < 1) && throw(CliError("usage/invalid",
            "--asymmetric: '$ts' must be a positive 1-based regressor index (or 'all')"))
        push!(idxs, v)
    end
    isempty(idxs) && throw(CliError("usage/invalid",
        "--asymmetric resolved to no indices; pass 'all' or 1-based indices like '1,3' " *
        "(a fully-symmetric model → use `estimate ardl`)"))
    return idxs
end

"""
    load_panel_data(data, id_col, time_col) -> PanelData

Load a panel CSV via xtset(), or return a PanelData handle as-is.
CSV still requires --id-col/--time-col; a `.jld2`/`.fmod`/`model://` PanelData
handle does not.
"""
function load_panel_data(data::String, id_col::String, time_col::String)
    if _is_handle_path(data)
        obj = load_model_dispatch(data)
        k = _data_kind_of(obj)
        k === :panel && return obj
        throw(CliError("data/wrong-kind",
            "$data is a $k handle ($(nameof(typeof(obj)))); this command expects PanelData";
            hint="data import --kind panel, or pass a panel CSV with --id-col/--time-col"))
    end
    isempty(id_col) && throw(CliError("usage/missing",
        "panel data requires --id-col to specify the group identifier column"))
    isempty(time_col) && throw(CliError("usage/missing",
        "panel data requires --time-col to specify the time period column"))
    df = load_data(data)
    id_col in names(df) || throw(CliError("data/missing-column", "id column '$id_col' not found in data (columns: $(join(names(df), ", ")))"))
    time_col in names(df) || throw(CliError("data/missing-column", "time column '$time_col' not found in data (columns: $(join(names(df), ", ")))"))
    # Get numeric variable columns excluding id and time (matches xtset's own
    # num_cols filter; passed explicitly so the panel keeps CLI-facing names).
    varnames = [n for n in variable_names(df) if n != id_col && n != time_col]
    isempty(varnames) && throw(CliError("data/invalid",
        "no numeric variable columns found after excluding id='$id_col'/time='$time_col'; a panel needs at least one numeric variable column"))
    # MEMs 0.7.0 xtset takes (df, group_col::Symbol, time_col::Symbol; ...) and
    # resolves group/time ID mapping internally (C054: the old Matrix/Vector
    # signature was removed upstream). It throws untyped ArgumentErrors on a bad panel
    # structure (duplicate (group,time) pairs, degenerate ids) — map those to a typed
    # data error so bad user input never surfaces as an internal exit-1 (benefits the whole
    # panel family: pvar/cips/hausman/pedroni/kao/westerlund).
    return try
        xtset(df, Symbol(id_col), Symbol(time_col); varnames=varnames)
    catch e
        e isa CliError && rethrow()
        e isa ArgumentError && throw(CliError("data/invalid", "invalid panel structure: $(e.msg)";
            hint="ensure (id, time) pairs are unique and variable columns are numeric"))
        rethrow()
    end
end

"""
    _load_panel_reg(data, id_col, time_col, dep, indep) → (pd, depsym, indepsyms, depc, indeps)

Load a long-format panel CSV and resolve `--dep`/`--indep` to `Symbol`s for the panel
estimators that take `(PanelData, dep, xs...)` — `estimate xtcointreg` (C062a) and, later,
`estimate pmg` / `test pmg-hausman` (C062c). Generalizes `_panel_coint_inputs` (test.jl):
`id`/`time` default to the first/second DATA column, `--dep` defaults to the first panel
variable, and `--indep` defaults to every other variable. Reuses the hardened
`load_panel_data` (typed duplicate-`(id,time)` / non-numeric / missing-column guards) and
validates every resolved name → `usage/invalid`, so bad input never reaches MEMs as an
untyped exit-1.
"""
function _load_panel_reg(data::String, id_col::String, time_col::String,
                         dep::String, indep::String)
    df = load_data(data)
    cols = names(df)
    length(cols) >= 3 || throw(CliError("usage/invalid",
        "panel cointegrating regression needs id, time, and variable column(s) (found $(length(cols)))"))
    id = isempty(id_col) ? cols[1] : id_col
    tc = isempty(time_col) ? cols[2] : time_col
    pd = load_panel_data(data, id, tc)          # typed: data/missing-column, data/invalid
    vars = pd.varnames                          # numeric cols minus id/time (non-empty — load_panel_data guards)
    depc = isempty(dep) ? vars[1] : dep
    depc in vars || throw(CliError("usage/invalid",
        "--dep '$depc' is not a panel variable (have: $(join(vars, ", ")))"))
    indeps = isempty(indep) ? filter(!=(depc), vars) : _parse_varlist(indep)
    isempty(indeps) && throw(CliError("usage/invalid", "need at least one regressor via --indep"))
    for v in indeps
        v in vars || throw(CliError("usage/invalid",
            "--indep '$v' is not a panel variable (have: $(join(vars, ", ")))"))
    end
    # Guard missing cells in the dep + regressor columns: `xtset`/`load_panel_data` silently
    # NaN-fills a blank cell (data/panel.jl `ismissing(v) ? NaN : …`), which would propagate to
    # NaN coefficients at exit 0 — inconsistent with `estimate cointreg`'s `_load_reg_data`,
    # which rejects the same input. Reject it here too (adversarial review C062a).
    for c in vcat([depc], indeps)
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    end
    return pd, Symbol(depc), Symbol.(indeps), depc, indeps
end

"""
    _load_and_estimate_pvar(data, id_col, time_col, lags; kwargs...) -> (model, panel, varnames)

Combined load + estimate for Panel VAR.
"""
function _load_and_estimate_pvar(data::String, id_col::String, time_col::String,
                                  lags::Int; method::String="gmm",
                                  transformation::String="fd", steps::String="twostep",
                                  system::Bool=false, collapse::Bool=false,
                                  dependent::String="", predet::String="", exog::String="",
                                  min_lag_endo::Int=2, max_lag_endo::Int=99)
    panel = load_panel_data(data, id_col, time_col)

    dep = _parse_varlist(dependent)
    pre = _parse_varlist(predet)
    exo = _parse_varlist(exog)

    # MEMs 0.7.0 renamed the variable-role kwargs (C054): dependent→dependent_vars,
    # predetermined→predet_vars, exogenous→exog_vars, system→system_instruments.
    # predet_vars/exog_vars are now plain String vectors (empty = none).
    model = if method == "feols"
        estimate_pvar_feols(panel, lags;
            dependent_vars=isempty(dep) ? nothing : dep,
            exog_vars=exo)
    else
        estimate_pvar(panel, lags;
            transformation=Symbol(transformation), steps=Symbol(steps),
            system_instruments=system, collapse=collapse,
            dependent_vars=isempty(dep) ? nothing : dep,
            predet_vars=pre,
            exog_vars=exo,
            min_lag_endo=min_lag_endo, max_lag_endo=max_lag_endo)
    end
    return model, panel, panel.varnames
end

"""
    _build_pvar_coef_table(model, varnames, p) -> DataFrame

Build a coefficient table for Panel VAR model with SE and p-values.
"""
function _build_pvar_coef_table(model, varnames::Vector{String}, p::Int)
    n = length(varnames)
    n_rows = size(model.Phi, 1)
    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end

    coef_df = DataFrame()
    for (vi, vname) in enumerate(varnames)
        coef_df[!, Symbol("$(vname)_coef")] = round.(model.Phi[1:length(row_names), vi]; digits=6)
        coef_df[!, Symbol("$(vname)_se")] = round.(model.se[1:length(row_names), vi]; digits=6)
        coef_df[!, Symbol("$(vname)_pval")] = round.(model.pvalues[1:length(row_names), vi]; digits=4)
    end
    insertcols!(coef_df, 1, :parameter => row_names)
    return coef_df
end

# ── FAVAR Helpers ─────────────────────────────────────────

"""
    _load_and_estimate_favar(data, factors, lags, key_vars, method, draws) → (favar, Y, varnames)
"""
function _load_and_estimate_favar(data::String, factors, lags::Int,
                                   key_vars::String, method::String, draws::Int)
    Y, varnames = load_multivariate_data(data)
    T_obs, n = size(Y)

    # Parse key variables (comma-separated names or indices)
    key_indices = Int[]
    if !isempty(key_vars)
        for kv in split(key_vars, ",")
            kv = strip(kv)
            idx = tryparse(Int, kv)
            if idx !== nothing
                push!(key_indices, idx)
            else
                found = findfirst(==(kv), varnames)
                found === nothing && error("key variable '$kv' not found in data columns: $varnames")
                push!(key_indices, found)
            end
        end
    end
    isempty(key_indices) && throw(CliError("usage/missing",
        "--key-vars is required for FAVAR";
        hint="comma-separated column names or indices, e.g. --key-vars y,x"))

    # Auto-select factors if not specified
    r = if factors === nothing
        auto_r = ic_criteria(Y, min(10, n - 1))
        _status_styled("  Auto-selected factors: $(auto_r.r_IC1) (IC1)\n"; color=:cyan)
        auto_r.r_IC1
    else
        factors
    end

    _status("Estimating FAVAR: $r factors, $lags lags, method=$method, $(length(key_indices)) key variables")

    # panel_varnames (W10/#131, MEMs#538): the key variables inside the augmented
    # VAR take their CSV column names, so irf/fevd/forecast favar label them
    # "infl"/"ffr" instead of the positional "X9"/"X10".
    favar = estimate_favar(Y, key_indices, r, lags;
                           method=Symbol(method),
                           n_draws=draws,
                           panel_varnames=varnames,
                           _fwd_seed()...)
    return favar, Y, varnames
end

const _SDFM_METHODS = Dict(
    "fglr" => :fglr,
    "gdfm-var" => :gdfm_var,
)

const _SDFM_Q_METHODS = Dict(
    "hallin-liska" => :hallin_liska,
    "bai-ng" => :bai_ng,
    "amengual-watson" => :amengual_watson,
)

const _GDFM_SPECTRAL = Dict(
    "lag-window" => :lag_window,
    "smoothed-periodogram" => :smoothed_periodogram,
)

"""
    _load_instrument(data, column) → Vector{Float64}

Load a proxy-instrument column for SDFM `identification=:proxy`: the column must
exist, be numeric, and hold no missings (upstream takes an `AbstractVector`, so a
missing cell would fail deep inside estimation as an untyped error).
"""
function _load_instrument(data::String, column::String)
    df = load_data(data)
    column in names(df) || throw(CliError("data/column-range",
        "instrument column '$column' not found; available: $(join(names(df), ", "))"))
    col = df[!, column]
    any(ismissing, col) && throw(CliError("data/missing-values",
        "instrument column '$column' contains missing values"))
    try
        return Vector{Float64}(col)
    catch
        throw(CliError("data/invalid", "instrument column '$column' is not numeric"))
    end
end

"""
    _load_and_estimate_sdfm(data, factors, id, var_lags, horizon, config, method,
                            spectral, instrument_col, q_method) → (sdfm, Y, varnames, q)

Shared Structural-DFM estimation for the `estimate`/`irf`/`fevd`/`forecast sdfm`
data paths (W1/#165): one implementation, one option surface.

- `factors === nothing` → upstream `:auto` q-selection via `q_method`
  (deterministic; replaces the legacy `ic_criteria_gdfm` auto path).
- `id == "proxy"` requires `--instrument`; `--instrument` with any other id is a
  `usage/invalid` no-op guard. `--q-method` with explicit `--factors` is ignored
  by upstream (selection never runs), so it is `usage/invalid` there too.
- `--seed` is forwarded as the estimator's own `seed=` (C052/#243 pattern).
"""
function _load_and_estimate_sdfm(data::String, factors, id::String, var_lags::Int,
                                 horizon::Int, config::String, method::String,
                                 spectral::String, instrument_col::String,
                                 q_method::String; bandwidth::Int=0,
                                 kernel::String="bartlett")
    haskey(_SDFM_METHODS, method) || throw(CliError("usage/invalid",
        "estimate sdfm: --method must be fglr|gdfm-var (got '$method')"))
    haskey(_GDFM_SPECTRAL, spectral) || throw(CliError("usage/invalid",
        "estimate sdfm: --spectral must be lag-window|smoothed-periodogram (got '$spectral')"))
    haskey(_SDFM_Q_METHODS, q_method) || throw(CliError("usage/invalid",
        "estimate sdfm: --q-method must be hallin-liska|bai-ng|amengual-watson (got '$q_method')"))
    if factors !== nothing && q_method != "hallin-liska"
        throw(CliError("usage/invalid",
            "estimate sdfm: --q-method '$q_method' applies only to automatic factor " *
            "selection (omit --factors to use it)"))
    end
    if !isempty(instrument_col) && id != "proxy"
        throw(CliError("usage/invalid",
            "estimate sdfm: --instrument applies only to --id proxy (got --id $id)"))
    end
    if id == "proxy" && isempty(instrument_col)
        throw(CliError("usage/missing",
            "estimate sdfm: --id proxy requires --instrument <column>";
            hint="name a numeric, missing-free proxy column, e.g. --instrument mp_shock"))
    end

    Y, varnames = load_multivariate_data(data)

    sign_check = nothing
    if id == "sign" && !isempty(config)
        sign_check, _ = _build_check_func(config)
    end
    instrument = isempty(instrument_col) ? nothing : _load_instrument(data, instrument_col)

    est_kw = (identification=Symbol(id), p=var_lags, H=horizon,
              method=_SDFM_METHODS[method], spectral=_GDFM_SPECTRAL[spectral],
              sign_check=sign_check, instrument=instrument, seed=_SEED[],
              bandwidth=bandwidth, kernel=Symbol(kernel), varnames=varnames)
    # Upstream rejects unknown `identification` symbols with a bare
    # ArgumentError — wrap so a bad --id is data/invalid (exit 3),
    # not an untyped exit 1 (same class as the W1/#186 VAR IRF wrap).
    sdfm, q = try
        if factors === nothing
            _status("Selecting dynamic factors (auto: $q_method)...")
            m = estimate_structural_dfm(Y, :auto; q_method=_SDFM_Q_METHODS[q_method], est_kw...)
            _status("  Auto-selected $(m.gdfm.q) dynamic factors")
            m, m.gdfm.q
        else
            estimate_structural_dfm(Y, factors; est_kw...), factors
        end
    catch e
        throw(_domain_or_data_error(e, "SDFM estimation"))
    end
    return sdfm, Y, varnames, q
end

# ── Panel/Matrix Loading Helper ──────────────────────────

"""
    _load_panel_or_matrix(data; id_col, time_col) → (result, is_panel)

Load data as PanelData if id_col/time_col are provided, else as Matrix.
"""
function _load_panel_or_matrix(data::String; id_col::String="", time_col::String="")
    if !isempty(id_col) && !isempty(time_col)
        pd = load_panel_data(data, id_col, time_col)
        _status_styled("  Panel: $(pd.n_groups) units, $(div(pd.T_obs, pd.n_groups)) periods\n"; color=:cyan)
        return pd, true
    else
        Y, varnames = load_multivariate_data(data)
        _status("  Matrix: $(size(Y, 1)) obs × $(size(Y, 2)) units")
        return Y, false
    end
end

# ── Result-handle re-render ────────────────────────────────

"""XOR + compute-flag checks for a loaded `--result`. Returns `nothing` to compute."""
function _loaded_result(result; data::String="", model=nothing, lags=nothing,
                        check_lags::Bool=false, leaf::String,
                        id=nothing, id_default::String="cholesky",
                        horizons=nothing, horizons_default::Union{Nothing,Int}=nothing)
    result === nothing && return nothing
    isempty(data) || throw(CliError("usage/invalid",
        "$leaf: --result cannot be combined with <data>"))
    model === nothing || throw(CliError("usage/invalid",
        "$leaf: --result cannot be combined with --model"))
    if check_lags
        lags === nothing || throw(CliError("usage/invalid",
            "$leaf: --lags does not apply with --result"))
    end
    if id !== nothing && id != id_default
        throw(CliError("usage/invalid",
            "$leaf: --id does not apply with --result"))
    end
    if horizons !== nothing && horizons_default !== nothing && horizons != horizons_default
        throw(CliError("usage/invalid",
            "$leaf: --horizons does not apply with --result"))
    end
    return result
end

function _result_varnames(result, n::Int)
    for f in (:varnames, :variables)
        hasproperty(result, f) || continue
        vn = getproperty(result, f)
        vn isa AbstractVector && length(vn) == n && return String[string(x) for x in vn]
    end
    return String["var_$i" for i in 1:n]
end

function _rerender_long_table(result; format::String="table", output::String="",
                              title::String="", key::String="",
                              plot::Bool=false, plot_save::String="")
    df = try
        long_table(result)
    catch e
        e isa MethodError && throw(CliError(
            "model/unsupported",
            "no long_table is defined for $(typeof(result))";
            hint="this result type cannot be re-rendered as a table; drop --result and recompute"))
        rethrow()
    end
    output_result(df; format=Symbol(format), output=output, title=title, key=key)
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_kv(result; format::String="table", output::String="",
                      title::String="", key::String="")
    pairs = Pair{String,Any}[]
    for n in propertynames(result)
        v = getproperty(result, n)
        if v isa Number || v isa AbstractString || v isa Bool || v isa Nothing
            push!(pairs, String(n) => v)
        end
    end
    output_kv(pairs; format=format, output=output, title=title, key=key)
    return result
end

function _fevd_proportions_from_irf(irf_vals::AbstractArray)
    n_h = size(irf_vals, 1)
    n = size(irf_vals, 2)
    proportions = zeros(n, n, n_h)
    for h in 1:n_h
        total_var = zeros(n)
        for vi in 1:n, si in 1:n
            cum_sq = sum(irf_vals[t, vi, si]^2 for t in 1:h)
            proportions[vi, si, h] = cum_sq
            total_var[vi] += cum_sq
        end
        for vi in 1:n
            total_var[vi] > 0 && (proportions[vi, :, h] ./= total_var[vi])
        end
    end
    return proportions, n_h
end

function _rerender_arias_irf(result; format::String="table", output::String="",
                             shock::Int=1, plot::Bool=false, plot_save::String="")
    irf_vals = irf_mean(result)
    n = size(irf_vals, 2)
    1 <= shock <= n || throw(CliError("usage/invalid",
        "shock index $shock out of 1:$n"))
    varnames = _result_varnames(result, n)
    shock_name = _shock_name(varnames, shock)
    irf_df = build_irf_table(irf_vals, nothing, nothing, varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Arias et al. identification)", key="irf")
    ess = hasproperty(result, :ess) ? Float64(result.ess) : NaN
    ess_frac = hasproperty(result, :ess_fraction) ? Float64(result.ess_fraction) : NaN
    output_kv(Pair{String,Any}[
        "acceptance_rate" => round(Float64(result.acceptance_rate); digits=6),
        "n_draws"         => length(result.weights),
        "ess"             => round(ess; digits=4),
        "ess_fraction"    => round(ess_frac; digits=6),
    ]; format=format, title="Arias Importance-Sampling Diagnostics")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_uhlig_irf(result; format::String="table", output::String="",
                             shock::Int=1, plot::Bool=false, plot_save::String="")
    n = size(result.irf, 2)
    1 <= shock <= n || throw(CliError("usage/invalid",
        "shock index $shock out of 1:$n"))
    varnames = _result_varnames(result, n)
    shock_name = _shock_name(varnames, shock)
    irf_df = build_irf_table(result.irf, nothing, nothing, varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Uhlig identification)", key="irf")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_identified_set(result; format::String="table", output::String="",
                                  shock::Int=1, plot::Bool=false, plot_save::String="")
    lower, upper = irf_bounds(result)
    med = irf_median(result)
    n = size(med, 2)
    1 <= shock <= n || throw(CliError("usage/invalid",
        "shock index $shock out of 1:$n"))
    varnames = _result_varnames(result, n)
    shock_name = _shock_name(varnames, shock)
    irf_df = build_irf_table(med, lower, upper, varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF Identified Set (sign, $shock_name shock)",
                  key="irf_identified_set")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_robust_bayes(result; format::String="table", output::String="",
                                shock::Int=1, plot::Bool=false, plot_save::String="")
    H = size(result.lower, 1)
    n = size(result.lower, 2)
    1 <= shock <= n || throw(CliError("usage/invalid",
        "shock index $shock out of 1:$n"))
    varnames = _result_varnames(result, n)
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
    output_kv(Pair{String,Any}[
        "Empty-set probability" => round(Float64(result.empty_set_prob); digits=6),
        "Informativeness" => round(Float64(result.informativeness); digits=6),
        "Level" => round(Float64(result.level); digits=4),
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
        title="Robust Bayes Diagnostics", key="robust_bayes_diagnostics")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_irf_result(result; format::String="table", output::String="",
                              title::String="", key::String="",
                              plot::Bool=false, plot_save::String="",
                              shock::Union{Nothing,Int}=nothing)
    tn = nameof(typeof(result))
    id_shock = something(shock, 1)
    tn === :AriasSVARResult && return _rerender_arias_irf(result; format, output, shock=id_shock, plot, plot_save)
    tn === :UhligSVARResult && return _rerender_uhlig_irf(result; format, output, shock=id_shock, plot, plot_save)
    tn === :SignIdentifiedSet && return _rerender_identified_set(result; format, output, shock=id_shock, plot, plot_save)
    tn === :RobustBayesResult && return _rerender_robust_bayes(result; format, output, shock=id_shock, plot, plot_save)
    df = try
        long_table(result)
    catch e
        e isa MethodError && throw(CliError(
            "model/unsupported",
            "no long_table is defined for $(typeof(result))";
            hint="this result type cannot be re-rendered as a table; drop --result and recompute"))
        rethrow()
    end
    if shock isa Int && hasproperty(result, :shocks) && "shock" in names(df)
        shocks = getproperty(result, :shocks)
        if shocks isa AbstractVector
            1 <= shock <= length(shocks) || throw(CliError("usage/invalid",
                "shock index $shock out of 1:$(length(shocks))"))
            shock_name = shocks[shock]
            df = df[df.shock .== shock_name, :]
        end
    end
    output_result(df; format=Symbol(format), output=output, title=title, key=key)
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _rerender_fevd_result(result; format::String="table", output::String="",
                               title::String="", key::String="",
                               plot::Bool=false, plot_save::String="",
                               key_prefix::String="")
    tn = nameof(typeof(result))
    if tn === :AriasSVARResult
        irf_vals = irf_mean(result)
        props, n_h = _fevd_proportions_from_irf(irf_vals)
        vn = _result_varnames(result, size(irf_vals, 2))
        _output_fevd_tables(props, vn, n_h; id="arias", title_prefix="FEVD",
                            format=format, output=output,
                            key_prefix=isempty(key_prefix) ? "fevd_by_variable" : key_prefix)
        _maybe_plot(result; plot=plot, plot_save=plot_save)
        return result
    elseif tn === :UhligSVARResult
        props, n_h = _fevd_proportions_from_irf(result.irf)
        vn = _result_varnames(result, size(result.irf, 2))
        _output_fevd_tables(props, vn, n_h; id="uhlig", title_prefix="FEVD",
                            format=format, output=output,
                            key_prefix=isempty(key_prefix) ? "fevd_by_variable" : key_prefix)
        _maybe_plot(result; plot=plot, plot_save=plot_save)
        return result
    elseif tn === :LPFEVD
        n = size(result.bias_corrected, 1)
        vn = _result_varnames(result, n)
        _output_fevd_tables(result.bias_corrected, vn, result.horizon;
                            id="", title_prefix="LP FEVD", format=format, output=output,
                            key_prefix=isempty(key_prefix) ? "lp_fevd" : key_prefix)
        _maybe_plot(result; plot=plot, plot_save=plot_save)
        return result
    elseif tn === :BayesianFEVD
        vn = _result_varnames(result, size(result.point_estimate, 1))
        H = hasproperty(result, :horizon) ? Int(result.horizon) : size(result.point_estimate, 3)
        _output_fevd_tables(result.point_estimate, vn, H;
                            id="", title_prefix="Bayesian FEVD", format=format, output=output,
                            key_prefix=isempty(key_prefix) ? "bayesian_fevd" : key_prefix)
        _maybe_plot(result; plot=plot, plot_save=plot_save)
        return result
    end
    return _rerender_long_table(result; format, output, title, key, plot, plot_save)
end

function _rerender_filter_result(result; format::String="table", output::String="",
                                 title::String="", key::String="",
                                 plot::Bool=false, plot_save::String="")
    tn = nameof(typeof(result))
    if tn === :X13FilterResult
        T = length(result.trend)
        tcol = collect(1:T)
        output_result(DataFrame(t=tcol, adjusted=round.(result.adjusted; digits=6));
                      format=Symbol(format), output=output, title="X-13 Seasonally Adjusted",
                      key="x_13_seasonally_adjusted")
        output_result(DataFrame(t=tcol, trend=round.(result.trend; digits=6));
                      format=Symbol(format), output=_per_var_output_path(output, "trend"),
                      title="X-13 Trend", key="x_13_trend")
        output_result(DataFrame(t=tcol, seasonal=round.(result.seasonal; digits=6));
                      format=Symbol(format), output=_per_var_output_path(output, "seasonal"),
                      title="X-13 Seasonal Factors", key="x_13_seasonal_factors")
        output_result(DataFrame(t=tcol, irregular=round.(result.irregular; digits=6));
                      format=Symbol(format), output=_per_var_output_path(output, "irregular"),
                      title="X-13 Irregular", key="x_13_irregular")
        order = result.arima_order
        order_str = order isa Tuple ? join(string.(order), ",") : string(order)
        output_kv(Pair{String,Any}[
            "method" => string(result.method),
            "frequency" => result.frequency,
            "transform" => string(result.transform),
            "arima_order" => order_str,
            "aic" => round(Float64(result.aic); digits=4),
            "sigma2" => round(Float64(result.sigma2); digits=6),
            "n_outliers" => Int(result.n_outliers),
            "T_obs" => Int(result.T_obs),
        ]; format=format, output=_per_var_output_path(output, "diagnostics"),
            title="X-13 Diagnostics", key="x_13_diagnostics")
        _maybe_plot(result; plot=plot, plot_save=plot_save)
        return result
    end
    t = collect(Float64, trend(result))
    c = collect(Float64, cycle(result))
    idx = collect(1:length(t))
    if hasproperty(result, :valid_range)
        vr = result.valid_range
        Tfull = hasproperty(result, :T_obs) ? Int(result.T_obs) : length(t)
        if length(t) == Tfull
            t = t[vr]
            c = c[vr]
        end
        idx = collect(vr)
    end
    result_df = DataFrame(t=idx, trend=round.(t; digits=6), cycle=round.(c; digits=6))
    output_result(result_df; format=Symbol(format), output=output, title=title, key=key)
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

_is_filter_result(result) = nameof(typeof(result)) === :X13FilterResult ||
    (applicable(trend, result) && applicable(cycle, result))

"""Accept `result=` from wrap_legacy; re-render without calling `handler`."""
function _with_result(handler, leaf::String; key::String="")
    return function (; result=nothing, kwargs...)
        data = get(kwargs, :data, "")
        data_s = data isa AbstractString ? String(data) : ""
        model = get(kwargs, :model, nothing)
        model_obj = model isa AbstractString ? nothing : model
        loaded = _loaded_result(result; data=data_s, model=model_obj, leaf=leaf)
        if loaded !== nothing
            fmt = string(get(kwargs, :format, "table"))
            out = string(get(kwargs, :output, ""))
            k = isempty(key) ? replace(leaf, r"[^A-Za-z0-9]+" => "_") : key
            plot = get(kwargs, :plot, false) === true
            plot_save = string(get(kwargs, :plot_save, ""))
            if _is_filter_result(loaded)
                _rerender_filter_result(loaded; format=fmt, output=out, title=leaf, key=k,
                                        plot=plot, plot_save=plot_save)
            elseif applicable(long_table, loaded)
                _rerender_long_table(loaded; format=fmt, output=out, title=leaf, key=k,
                                     plot=plot, plot_save=plot_save)
            else
                _rerender_kv(loaded; format=fmt, output=out, title=leaf, key=k)
                _maybe_plot(loaded; plot=plot, plot_save=plot_save)
            end
            return loaded
        end
        return handler(; kwargs...)
    end
end

# ── Plot Helpers ──────────────────────────────────────────

"""
    _maybe_plot(result; plot, plot_save, kwargs...)

Optionally plot a result using MacroEconometricModels' interactive D3.js plotting.
If `plot` is true, opens in browser. If `plot_save` is non-empty, saves to HTML file.
"""
function _maybe_plot(result; plot::Bool=false, plot_save::String="", kwargs...)
    !plot && isempty(plot_save) && return
    _validate_output_path(plot_save)
    # W5/#95 — defence in depth. The rule is still "only advertise --plot when a real
    # plot_result(::that type) EXISTS" (grep the real plotting/ before adding the flags;
    # the mock's generic fallback will not tell you). But an unguarded call here turns any
    # gap — a leaf whose result type lost its recipe at a bump, a plot kwarg the recipe
    # does not accept — into a raw MethodError, i.e. exit 1 "likely a bug", which blames
    # the CLI for a missing upstream method. Map that ONE case to a typed refusal and let
    # every other failure (save/display) propagate untouched so real bugs stay visible.
    p = try
        plot_result(result; kwargs...)
    catch e
        e isa MethodError && (e.f === plot_result) && throw(CliError(
            "model/unsupported",
            "no plot is defined for $(typeof(result))";
            hint="this result type has no plot_result recipe in MacroEconometricModels " *
                 "$(_mems_version_string()); drop --plot/--plot-save, or use --format json " *
                 "and plot the tables yourself"))
        rethrow()
    end
    if !isempty(plot_save)
        save_plot(p, plot_save)
        _status_styled("  Plot saved: $plot_save\n"; color=:green)
    end
    if plot
        display_plot(p)
        _status_styled("  Plot opened in browser\n"; color=:cyan)
    end
end

# ── DSGE Helpers ───────────────────────────────────────────

"""
    _dsge_call(f, args...; kwargs...)

Invoke `f` at the latest world age (`Base.invokelatest`). A representative-agent DSGE
spec loaded from a `.toml`/`.jl` file carries `@dsge`-generated residual functions that
are compiled at load time — i.e. in a *newer* world age than the running handler's
frame. Any MEMs call that **evaluates** those residual functions (steady state,
linearize, solve, DSGE estimation, perfect foresight, OccBin, Bayesian re-solves) must
go through this, or Julia throws a "method too new to be called from this world context"
`MethodError`. Calls that only consume an already-solved solution object (irf/fevd/
simulate/historical_decomposition on a `*Solution`) operate on numeric matrices and do
NOT need this. See the "RA DSGE loader" durable gotcha in CLAUDE.md.
"""
_dsge_call(f, args...; kwargs...) = Base.invokelatest(f, args...; kwargs...)

"""
    _dsge_sandbox() → Module

Fresh module for evaluating a runtime DSGE spec (a `.jl` file or a synthesized `@dsge`
block). The in-scope `MacroEconometricModels` object is injected as a const and its
exports are brought in via a *relative* `using .MacroEconometricModels` — so the bare
`@dsge`/`ModelSpec` names the macro expands to resolve against the injected object rather
than the load path. This is what makes the loader work identically under the real
package and the test mock (which shadows `MacroEconometricModels` in the test session).
"""
function _dsge_sandbox()
    mod = Module()
    Base.eval(mod, :(const MacroEconometricModels = $(MacroEconometricModels)))
    Base.eval(mod, :(using .MacroEconometricModels))
    return mod
end

"""
    _dsge_toml_block(dsge_cfg) → String

Synthesize an `@dsge begin … end` source block from the parsed `[model]` TOML fields
(`get_dsge`): `parameters` (name→value), `endogenous`, `exogenous`, `equations` (already
in `var[t]` form), and the optional `linear` flag. Real MEMs has no keyword RA-spec
constructor — specs are built by the `@dsge` macro, which parses the equations into
callable residual functions — so the TOML path must route through the macro too.
`E[t](...)` is not rewritten: upstream errors at expansion and the catch maps it to
`config/invalid`. Write leads as `x[t+1]` (`E_t x_{t+1}`).
"""
function _dsge_toml_block(dsge_cfg::Dict)
    params = dsge_cfg["parameters"]     # Dict name => value
    endog  = dsge_cfg["endogenous"]     # Vector{String}
    exog   = dsge_cfg["exogenous"]      # Vector{String}
    eqs    = dsge_cfg["equations"]      # Vector{String}
    is_linear = Bool(get(dsge_cfg, "linear", false))

    lines = String[]
    if !isempty(params)
        push!(lines, "    parameters: " * join(["$k = $v" for (k, v) in params], ", "))
    end
    push!(lines, "    endogenous: " * join(endog, ", "))
    isempty(exog) || push!(lines, "    exogenous: " * join(exog, ", "))
    is_linear && push!(lines, "    linear: true")
    util = strip(string(get(dsge_cfg, "utility", "")))
    beta = strip(string(get(dsge_cfg, "beta", "")))
    ctrls = get(dsge_cfg, "controls", String[])
    isempty(util) || push!(lines, "    utility: $util")
    isempty(beta) || push!(lines, "    beta: $beta")
    isempty(ctrls) || push!(lines, "    controls: " * join(ctrls, ", "))
    push!(lines, "")
    for eq in eqs
        push!(lines, "    " * eq)
    end
    return "@dsge begin\n" * join(lines, "\n") * "\nend"
end

# ── ModelSpec kind guards (MEMs 0.9.0: DSGESpec/HADSGESpec are gone) ──────────

_unwrap_loaderror(e) = e isa LoadError ? _unwrap_loaderror(e.error) : e

const _AGENT_KIND_FAMILY = Dict(
    "HouseholdSystem"            => ("heterogeneous-agent", "`dsge ha …`"),
    "DCEGMSystem"                => ("DCEGM", "`dsge dcegm …`"),
    "LifeCycleSystem"            => ("life-cycle", "`dsge lifecycle …`"),
    "ContinuousHouseholdSystem"  => ("continuous-time household", "`dsge ct …`"),
    "FirmSystem"                 => ("firm (Khan–Thomas)", "`dsge firm …`"),
    "IntermediarySystem"         => ("bank (Bewley)", "`dsge bank …`"),
)

_is_model_spec(x) = x isa MacroEconometricModels.ModelSpec

function _agent_kind_names(spec)
    String[string(nameof(typeof(v))) for v in values(spec.agents)]
end

_is_ra_spec(spec) = _is_model_spec(spec) && isempty(spec.agents)
_is_ha_spec(spec) = _is_model_spec(spec) &&
    MacroEconometricModels.has_kind(spec, MacroEconometricModels.HouseholdSystem)

function _ha_households(spec)
    collect(MacroEconometricModels.agents_of(spec, MacroEconometricModels.HouseholdSystem))
end

function _ha_model_symbol(spec)
    hh = _ha_households(spec)
    length(hh) == 1 || throw(CliError("model/unsupported",
        "this command supports exactly one household population; this spec has $(length(hh))";
        hint="multi-population HA solve is deferred (see `dsge ha` docs)"))
    return hh[1].model
end

function _wrong_command_for_kinds(spec, intended::String)
    names = unique(_agent_kind_names(spec))
    if isempty(names)
        return CliError("usage/wrong-command",
            "this is a representative-agent ModelSpec — use `dsge solve|irf|…`, not `$intended`",
            hint="e.g. friedman dsge solve <file>")
    end
    if length(names) == 1 && names[1] == "HouseholdSystem"
        return CliError("usage/wrong-command",
            "this is a heterogeneous-agent spec — use `dsge ha …`",
            hint="e.g. friedman dsge ha solve <file> --method reiter")
    end
    families = String[]
    for n in names
        pair = get(_AGENT_KIND_FAMILY, n, (n, "the matching `dsge` family command"))
        push!(families, pair[2])
    end
    return CliError("usage/wrong-command",
        "this spec has agent kind $(join(names, "/")) — use $(join(unique(families), " or ")), not `$intended`")
end

function _require_ra_spec(spec, intended::String="dsge solve")
    _is_model_spec(spec) || throw(CliError("config/invalid",
        "model did not evaluate to a ModelSpec (got $(typeof(spec)))"))
    _is_ra_spec(spec) || throw(_wrong_command_for_kinds(spec, intended))
    return spec
end

function _require_ha_spec(spec, intended::String="dsge ha")
    _is_model_spec(spec) || throw(CliError("config/invalid",
        "model did not evaluate to a ModelSpec (got $(typeof(spec)))"))
    _is_ha_spec(spec) || throw(_wrong_command_for_kinds(spec, intended))
    return spec
end

function _dsge_eval_invalid(e, msg::String; hint::String="")
    inner = _unwrap_loaderror(e)
    inner isa CliError && rethrow(inner)
    throw(CliError("config/invalid",
        msg * ": $(sprint(showerror, inner))"; hint=hint))
end

function _dsge_solve_error(e, label::String)
    e isa CliError && return e
    msg = sprint(showerror, e)
    if e isa ArgumentError && (occursin("#651", msg) ||
            occursin("multiple agent populations", msg) ||
            occursin("mixed agent kinds", msg) ||
            occursin("no solver for agent kind", msg))
        return CliError("model/unsupported", "$label: $msg")
    end
    return _domain_or_data_error(e, label)
end

"""
    _load_dsge_model(path) → ModelSpec

Load a representative-agent DSGE model from a `.toml` or `.jl` file.

- `.toml`: parse `[model]` (incl. optional `linear = true`) and build the spec by
  synthesizing an `@dsge` block from the equations ([`_dsge_toml_block`]).
- `.jl`: last expression must be a `ModelSpec` with no agent populations — typically
  an `@dsge begin … end` block. The sandbox pre-imports MEMs' exports.

Both paths compile the spec's residual functions at load time; every downstream MEMs
call that evaluates them must go through [`_dsge_call`] (world-age barrier).

An HA spec, or any other agent kind reachable via `to_spec`, is `usage/wrong-command`
(exit 2) — never silently remapped into an RA solver.
"""
function _load_dsge_model(path::String)
    _validate_input_path(path)
    isfile(path) || throw(CliError("data/file-not-found", "model file not found: $path"))
    ext = lowercase(splitext(path)[2])

    if ext == ".toml"
        config = load_config(path)
        dsge_cfg = get_dsge(config)

        isempty(dsge_cfg["endogenous"]) && throw(CliError("config/invalid",
            "TOML model must have [model] with endogenous variables"))
        isempty(dsge_cfg["equations"]) && throw(CliError("config/invalid",
            "TOML model must have [[model.equations]]"))

        is_linear = Bool(get(dsge_cfg, "linear", false))
        block = _dsge_toml_block(dsge_cfg)
        mod = _dsge_sandbox()
        spec = try
            include_string(mod, block)
        catch e
            _dsge_eval_invalid(e,
                "could not build a DSGE spec from the TOML model — check [[model.equations]]" *
                " and [model] parameters/endogenous/exogenous";
                hint="E[t](...) was removed; write the lead directly (x[t+1] is E_t x_{t+1})")
        end
        _require_ra_spec(spec, "dsge solve")

        lin_note = is_linear ? ", linear=true" : ""
        _status("Loaded DSGE model from TOML: $(length(dsge_cfg["endogenous"])) endogenous, $(length(dsge_cfg["exogenous"])) exogenous, $(length(dsge_cfg["equations"])) equations$lin_note")
        return spec

    elseif ext == ".jl"
        mod = _dsge_sandbox()
        result = try
            Base.include(mod, path)
        catch e
            e isa CliError && rethrow()
            _dsge_eval_invalid(e, "could not evaluate the DSGE model file '$path'";
                hint="the file should be an `@dsge begin … end` block; E[t](...) was removed — write x[t+1]")
        end
        _require_ra_spec(result, "dsge solve")
        spec = result
        lin_note = (hasproperty(spec, :linear) && spec.linear) ? ", linear=true" : ""
        _status("Loaded DSGE model from Julia file: $(spec.n_endog) endogenous, $(spec.n_exog) exogenous$lin_note")
        return spec

    else
        throw(CliError("usage/invalid-option",
            "unsupported model file extension '$ext' — use .toml or .jl"))
    end
end

# ── HA-DSGE Helpers (C040 / MEMs 0.6.7) ───────────────────

const _HA_BUILTIN_MODELS = (
    "krusell-smith" => :krusell_smith,
    "one-asset-hank" => :one_asset_hank,
    "two-asset-hank" => :two_asset_hank,
    "huggett" => :huggett,
    "endogenous-labor" => :endogenous_labor,
)

const _RA_METHOD_MAP = Dict(
    "gensys" => :gensys,
    "klein" => :klein,
    "perturbation" => :perturbation,
    "projection" => :projection,
    "pfi" => :pfi,
    "vfi" => :vfi,
    "blanchard-kahn" => :blanchard_kahn,
    "blanchard_kahn" => :blanchard_kahn,
)

const _RA_METHOD_CHOICES = ["gensys", "klein", "perturbation", "projection", "pfi",
                            "vfi", "blanchard-kahn"]

const _VFI_OPTIMIZER_CHOICES = ["auto", "grid1d", "fminbox-nm", "fminbox-lbfgs"]

const _VFI_OPTIMIZER_MAP = Dict("auto" => :auto, "grid1d" => :grid1d,
    "fminbox-nm" => :fminbox_nm, "fminbox-lbfgs" => :fminbox_lbfgs)

"""Parse `--smolyak-mu`: `""` (unset) → `nothing`, `"2"` → `2`, `"2,3"` → `[2, 3]`.

Upstream (`_smolyak_level_vector`) wants a scalar `μ ≥ 0` or a length-`nx`
vector with all entries `≥ 0`; the length-vs-`nx` check needs the solved
model, so a wrong-length vector surfaces as upstream `ArgumentError` →
`data/invalid`, while shape/negativity junk is `usage/invalid` here."""
function _parse_smolyak_mu(s::String)
    t = strip(s)
    isempty(t) && return nothing
    parts = [strip(p) for p in split(t, ",")]
    (any(isempty, parts) || any(p -> tryparse(Int, p) === nothing, parts)) &&
        throw(CliError("usage/invalid",
            "--smolyak-mu must be a non-negative integer or a comma-separated " *
            "list thereof (got '$s')"))
    vals = [tryparse(Int, p)::Int for p in parts]
    any(<(0), vals) && throw(CliError("usage/invalid",
        "--smolyak-mu entries must be ≥ 0 (got '$s')"))
    return length(vals) == 1 ? vals[1] : vals
end

"""Map CLI `--method` string to the MEMs `solve` symbol (`:blanchard_kahn` not hyphen)."""
function _parse_ra_method(method::String)
    key = lowercase(strip(method))
    haskey(_RA_METHOD_MAP, key) || throw(CliError("usage/invalid-option",
        "invalid --method '$method'; must be $(join(_RA_METHOD_CHOICES, "|"))"))
    return _RA_METHOD_MAP[key]
end

function _parse_hh_solver(hh_solver::String)
    s = lowercase(strip(hh_solver))
    s in ("egm", "vfi") || throw(CliError("usage/invalid-option",
        "invalid --hh-solver '$hh_solver'; must be egm|vfi"))
    return Symbol(s)
end

function _parse_ha_distribution(distribution::String)
    s = lowercase(strip(distribution))
    s in ("young", "winberry") || throw(CliError("usage/invalid-option",
        "invalid --distribution '$distribution'; must be young|winberry"))
    return Symbol(s)
end

const _HA_METHOD_MAP = Dict(
    "ssj" => :ssj,
    "reiter" => :reiter,
    "krusell-smith" => :krusell_smith,
    "krusell_smith" => :krusell_smith,
)

"""
    _parse_ha_method(method) → Symbol

Map CLI method string to MEMs symbol (`:ssj`, `:reiter`, `:krusell_smith`).
"""
function _parse_ha_method(method::String)
    key = lowercase(strip(method))
    haskey(_HA_METHOD_MAP, key) || throw(CliError("usage/invalid-option",
        "invalid --method '$method'; must be one of: ssj, reiter, krusell-smith"))
    return _HA_METHOD_MAP[key]
end

"""
    _ha_builtin_symbol(name) → Union{Symbol,Nothing}

Normalize CLI builtin model token (`huggett`, `:huggett`, `krusell-smith`) to MEMs Symbol.
"""
function _ha_builtin_symbol(name::String)
    s = lowercase(strip(name))
    startswith(s, ":") && (s = s[2:end])
    s = replace(s, "_" => "-")
    for (cli, sym) in _HA_BUILTIN_MODELS
        s == cli && return sym
    end
    return nothing
end

"""
    _load_ha_model(model) → ModelSpec

Load an HA-DSGE model from a builtin name (`huggett`, `krusell-smith`, …) or a `.jl`
file that evaluates to a `ModelSpec` carrying a `HouseholdSystem`.

The `.jl` path goes through [`_dsge_sandbox`]. At MEMs 0.9.0 the HA SSJ path evaluates
the spec's `NamedEquation` residual closures, so downstream `compute_steady_state` /
`solve` of a `.jl` spec must go through [`_dsge_call`] (world-age barrier).
"""
function _load_ha_model(model::String; distribution::String="young")
    isempty(strip(model)) && throw(CliError("usage/missing-arg",
        "HA model is required (builtin name or path to .jl ModelSpec)"))
    dist = _parse_ha_distribution(distribution)

    bsym = _ha_builtin_symbol(model)
    if bsym !== nothing
        _status("Loading HA-DSGE builtin :$bsym via load_ha_example (distribution=$dist)")
        return MacroEconometricModels.load_ha_example(bsym; distribution=dist)
    end

    _validate_input_path(model)
    isfile(model) || throw(CliError("data/file-not-found",
        "HA model file not found: $model (hint: use a builtin name like huggett, " *
        "or a .jl file evaluating to a heterogeneous-agent ModelSpec)"))
    ext = lowercase(splitext(model)[2])
    ext == ".jl" || throw(CliError("usage/invalid-option",
        "HA model file must be .jl (got '$ext'); builtins: " *
        join(first.( _HA_BUILTIN_MODELS), ", ")))

    mod = _dsge_sandbox()
    result = try
        Base.include(mod, model)
    catch e
        e isa CliError && rethrow()
        _dsge_eval_invalid(e, "could not evaluate the HA model file '$model'";
            hint="the file should be an `@dsge begin … end` block with heterogeneous:, " *
                 "idiosyncratic: and aggregation: declarations")
    end
    _require_ha_spec(result, "dsge ha")
    _status("Loaded HA ModelSpec from Julia file (model=$(_ha_model_symbol(result)))")
    return result
end

"""
    _solve_ha(spec; method=:ssj, kwargs...) → HADSGESolution | KrusellSmithSolution

Compute steady state (unless `ss` supplied) then solve with the HA method.
`.jl` specs go through [`_dsge_call`]: at 0.9.0 SSJ evaluates the aggregate
residual closures compiled at load time.
"""
function _solve_ha(spec::MacroEconometricModels.ModelSpec;
                   method::Symbol=:ssj,
                   ss=nothing,
                   n_reduced::Int=30,
                   T_horizon::Int=300,
                   kwargs...)
    _require_ha_spec(spec, "dsge ha")
    hh = get(kwargs, :hh_solver, :egm)
    if hh === :vfi && method === :krusell_smith
        throw(CliError("usage/invalid",
            "--hh-solver vfi is not implemented with --method krusell-smith " *
            "(the KS household problem has aggregate (K, z) in the state)";
            hint="use --hh-solver egm, or vfi on ssj/reiter/steady-state"))
    end
    try
        if ss === nothing
            _status("Computing HA steady state...")
            # Only pass steady-state kwargs (MEMs filters the rest inside solve,
            # but we call compute_steady_state separately and must not forward
            # n_reduced / T_horizon here).
            ss_keys = (:K_init, :r_bounds, :max_iter, :tol, :verbose, :price_fn, :clearing,
                       :hh_solver, :distribution)
            ss_kw = Dict{Symbol,Any}()
            for k in ss_keys
                haskey(kwargs, k) && (ss_kw[k] = kwargs[k])
            end
            ss = _dsge_call(compute_steady_state, spec; ss_kw...)
            if hasproperty(ss, :converged)
                _status_styled("  Steady state converged: $(ss.converged)\n";
                               color = ss.converged ? :green : :yellow)
            end
        end
        _status("Solving HA-DSGE with method=$method...")
        # Krusell–Smith owns its PLM-simulation RNG via seed= (MEMs#769), recorded
        # on KrusellSmithSolution.manifest. Other HA methods take no seed — only
        # forward here, never blindly into kwargs (ssj/reiter would reject it).
        ks_seed = (method === :krusell_smith && _SEED[] !== nothing) ? (; seed=_SEED[]) : NamedTuple()
        return _dsge_call(solve, spec; method=method, ss=ss,
                          n_reduced=n_reduced, T_horizon=T_horizon,
                          ks_seed..., kwargs...)
    catch e
        throw(_dsge_solve_error(e, "HA-DSGE solve"))
    end
end

"""Build the method-gated kwargs for `solve`. Exclusive knobs on the wrong method are
`usage/invalid` (not silently dropped). Defaults/sentinels mean "not passed"."""
function _ra_solve_extra(meth::Symbol; order::Int=1, degree::Int=5, grid::String="auto",
                         next_state::String="", howard_steps::Int=-1,
                         n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                         scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                         damping::Float64=0.0, anderson_m::Int=0,
                         optimizer::String="", smolyak_mu::String="")
    ns = lowercase(strip(next_state))
    os = lowercase(strip(optimizer))
    mu = _parse_smolyak_mu(smolyak_mu)
    vfi_exclusive = n_grid != 0 || n_choice != 0 || !isempty(os) || mu !== nothing
    pfi_exclusive = anderson_m != 0
    shared_knobs = howard_steps != -1 || n_quad != 0 || scale != 0.0 ||
                   tol != 0.0 || max_iter != 0 || damping != 0.0 || !isempty(ns)

    if meth === :vfi
        isempty(ns) || ns in ("auto", "linear", "residual") || throw(CliError("usage/invalid",
            "vfi --next-state must be auto|linear|residual (got '$next_state')"))
        pfi_exclusive && throw(CliError("usage/invalid",
            "--anderson-m is a PFI option; not valid with --method vfi"))
        g = lowercase(strip(grid))
        g in ("auto", "tensor", "smolyak") || throw(CliError("usage/invalid",
            "vfi supports --grid auto|tensor|smolyak only (got '$grid')"))
        n_grid == 0 || n_grid >= 3 || throw(CliError("usage/invalid",
            "--n-grid must be ≥ 3 (got $n_grid)"))
        n_choice == 0 || n_choice >= 3 || throw(CliError("usage/invalid",
            "--n-choice must be ≥ 3 (got $n_choice)"))
        # Dead-combo guards: upstream silently ignores the losing knob in each
        # pair, so accept-and-ignore would be a lie. The `auto` corners stay
        # allowed (resolution needs the solved model) and are documented.
        g == "smolyak" && n_grid != 0 && throw(CliError("usage/invalid",
            "--n-grid is a tensor-grid option; not valid with --grid smolyak";
            hint="use --smolyak-mu to control the sparse-grid level"))
        g == "smolyak" && degree != 5 && throw(CliError("usage/invalid",
            "--degree is a tensor-grid option; not valid with --grid smolyak";
            hint="the Smolyak grid sets its own degree from --smolyak-mu"))
        g == "tensor" && mu !== nothing && throw(CliError("usage/invalid",
            "--smolyak-mu is a Smolyak-grid option; not valid with --grid tensor"))
        os in ("fminbox-nm", "fminbox-lbfgs") && n_choice != 0 &&
            throw(CliError("usage/invalid",
                "--n-choice is a grid1d option; not valid with --optimizer $os"))
        # Parser `choices=` enforces this first; the map lookup below must
        # never KeyError on a direct call.
        !isempty(os) && !haskey(_VFI_OPTIMIZER_MAP, os) &&
            throw(CliError("usage/invalid",
                "--optimizer must be auto|grid1d|fminbox-nm|fminbox-lbfgs " *
                "(got '$optimizer')"))
    elseif meth === :pfi
        isempty(ns) || ns in ("linear", "policy", "nonlinear") || throw(CliError("usage/invalid",
            "pfi --next-state must be linear|policy|nonlinear (got '$next_state')"))
        vfi_exclusive && throw(CliError("usage/invalid",
            "--n-grid/--n-choice/--optimizer/--smolyak-mu are VFI options; " *
            "not valid with --method pfi"))
        anderson_m >= 0 || throw(CliError("usage/invalid",
            "--anderson-m must be ≥ 0 (got $anderson_m)"))
    elseif vfi_exclusive || pfi_exclusive || shared_knobs
        throw(CliError("usage/invalid",
            "VFI/PFI knobs require --method vfi or pfi (got $(meth))"))
    end
    howard_steps == -1 || howard_steps >= 0 || throw(CliError("usage/invalid",
        "--howard-steps must be ≥ 0 (got $howard_steps)"))
    n_quad == 0 || n_quad >= 1 || throw(CliError("usage/invalid",
        "--n-quad must be ≥ 1 (got $n_quad)"))
    scale == 0.0 || scale > 0 || throw(CliError("usage/invalid",
        "--scale must be > 0 (got $scale)"))
    tol == 0.0 || tol > 0 || throw(CliError("usage/invalid",
        "--tol must be > 0 (got $tol)"))
    max_iter == 0 || max_iter >= 1 || throw(CliError("usage/invalid",
        "--max-iter must be ≥ 1 (got $max_iter)"))
    damping == 0.0 || damping > 0 || throw(CliError("usage/invalid",
        "--damping must be > 0 (got $damping)"))

    extra = NamedTuple()
    if meth === :perturbation
        extra = (; extra..., order=order)
    elseif meth === :projection
        extra = (; extra..., degree=degree, grid=Symbol(grid))
    elseif meth === :pfi
        extra = (; extra..., degree=degree, grid=Symbol(grid))
        isempty(ns) || (extra = (; extra..., next_state=Symbol(ns)))
        howard_steps >= 0 && (extra = (; extra..., howard_steps=howard_steps))
        n_quad > 0 && (extra = (; extra..., n_quad=n_quad))
        scale > 0 && (extra = (; extra..., scale=scale))
        tol > 0 && (extra = (; extra..., tol=tol))
        max_iter > 0 && (extra = (; extra..., max_iter=max_iter))
        damping > 0 && (extra = (; extra..., damping=damping))
        anderson_m > 0 && (extra = (; extra..., anderson_m=anderson_m))
    elseif meth === :vfi
        # `:auto` passes through to upstream routing (nx ≤ 3 → tensor,
        # nx ≥ 4 → Smolyak) — the W1/#180 decision; `""` matches the default.
        g = lowercase(strip(grid))
        gsym = (g == "" || g == "auto") ? :auto : Symbol(g)
        extra = (; extra..., degree=degree, grid=gsym)
        isempty(ns) || (extra = (; extra..., next_state=Symbol(ns)))
        howard_steps >= 0 && (extra = (; extra..., howard_steps=howard_steps))
        n_grid >= 3 && (extra = (; extra..., n_grid=n_grid))
        n_choice >= 3 && (extra = (; extra..., n_choice=n_choice))
        !isempty(os) && (extra = (; extra..., optimizer=_VFI_OPTIMIZER_MAP[os]))
        mu !== nothing && (extra = (; extra..., smolyak_mu=mu))
        n_quad > 0 && (extra = (; extra..., n_quad=n_quad))
        scale > 0 && (extra = (; extra..., scale=scale))
        tol > 0 && (extra = (; extra..., tol=tol))
        max_iter > 0 && (extra = (; extra..., max_iter=max_iter))
        damping > 0 && (extra = (; extra..., damping=damping))
    end
    return extra
end

function _require_vfi_bellman(spec)
    u = hasproperty(spec, :bellman_utility) ? spec.bellman_utility : nothing
    b = hasproperty(spec, :bellman_beta) ? spec.bellman_beta : nothing
    (u !== nothing && b !== nothing) || throw(CliError("config/missing-key",
        "dsge solve --method vfi requires @dsge utility: and beta: " *
        "(or TOML [model] utility / beta)";
        hint="e.g. utility = \"log(C)\", beta = \"beta\", controls = [\"C\"]"))
    return nothing
end

"""
    _solve_dsge(spec; method="gensys", ...) → solution

Solve a representative-agent DSGE model: compute steady state → linearize → solve.
Per-method kwargs only: `order` for perturbation; `degree`/`grid` for projection/pfi/vfi;
VFI/PFI knobs only for the matching solver. `blanchard-kahn` maps to `:blanchard_kahn`.
An HA spec is refused (upstream would silently remap `:gensys` → `:ssj`).
"""
function _solve_dsge(spec::MacroEconometricModels.ModelSpec;
                     method::String="gensys", order::Int=1,
                     degree::Int=5, grid::String="auto",
                     constraint_solver::String="",
                     next_state::String="", howard_steps::Int=-1,
                     n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                     scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                     damping::Float64=0.0, anderson_m::Int=0,
                     optimizer::String="", smolyak_mu::String="")
    _require_ra_spec(spec, "dsge solve")
    meth = _parse_ra_method(method)
    extra = _ra_solve_extra(meth; order=order, degree=degree, grid=grid,
                            next_state=next_state, howard_steps=howard_steps,
                            n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                            scale=scale, tol=tol, max_iter=max_iter,
                            damping=damping, anderson_m=anderson_m,
                            optimizer=optimizer, smolyak_mu=smolyak_mu)
    meth === :vfi && _require_vfi_bellman(spec)
    # `:grid1d` maximizes one control; with explicitly declared controls the
    # count is known here, so reject up front (usage, not data). With default
    # (empty) controls the count resolves inside upstream, whose ArgumentError
    # maps to data/invalid — same as any other shape mismatch.
    if meth === :vfi && lowercase(strip(optimizer)) == "grid1d" &&
       hasproperty(spec, :bellman_controls) && length(spec.bellman_controls) > 1
        throw(CliError("usage/invalid",
            "--optimizer grid1d supports one continuous control (model " *
            "declares $(length(spec.bellman_controls)))";
            hint="use --optimizer auto (resolves to fminbox-nm) or fminbox-lbfgs"))
    end
    try
        _status("Computing steady state...")
        ss_kw = isempty(constraint_solver) ? (;) : (; solver=Symbol(constraint_solver))
        # World-age barrier: a runtime-loaded spec's @dsge residual fns are "too new" for
        # this frame — every MEMs call that evaluates them must go through _dsge_call.
        spec = _dsge_call(compute_steady_state, spec; ss_kw...)

        _status("Linearizing model...")
        _dsge_call(linearize, spec)

        _status("Solving with method=$method" *
                (meth === :perturbation ? ", order=$order" : "") *
                (meth in (:projection, :pfi, :vfi) ? ", degree=$degree, grid=$grid" : "") *
                "...")

        solve_kw = isempty(constraint_solver) ? NamedTuple() : (; solver=Symbol(constraint_solver))
        sol = _dsge_call(solve, spec; method=meth, extra..., solve_kw...)

        # Report diagnostics
        if sol isa MacroEconometricModels.DSGESolution ||
           sol isa MacroEconometricModels.PerturbationSolution
            det_status = is_determined(sol) ? "unique" : "indeterminate"
            stab_status = is_stable(sol) ? "stable" : "unstable"
            _status_styled("  Determinacy: $det_status\n"; color = is_determined(sol) ? :green : :red)
            _status_styled("  Stability: $stab_status\n"; color = is_stable(sol) ? :green : :red)
        end

        return sol
    catch e
        throw(_dsge_solve_error(e, "DSGE solve"))
    end
end

"""Load a one-column (or first-column) positive TFP path from CSV."""
function _load_positive_path(path::String; min_length::Int=2, name::String="Z_path")
    _validate_input_path(path)
    isfile(path) || throw(CliError("data/file-not-found", "$name file not found: $path"))
    df = load_data(path)
    mat = df_to_matrix(df)
    vec = mat[:, 1]
    length(vec) >= min_length || throw(CliError("data/shape",
        "$name has $(length(vec)) row(s); need ≥ $min_length"))
    all(>(0), vec) || throw(CliError("data/invalid",
        "$name values must all be positive"))
    return Float64.(vec)
end

"""
    _load_dsge_constraints(path; spec=nothing) → Vector{constraint}

Load OccBin and/or nonlinear constraints from a TOML file.
Bounds become `VariableBound`. `[[constraints.nonlinear]]` expr strings are
`Meta.parse`d then `parse_constraint`'d (Expr only upstream) → `OccBinConstraint`.
Nonlinear constraints require a loaded DSGE spec.
"""
function _load_dsge_constraints(path::String; spec=nothing)
    config = load_config(path)
    con_cfg = get_dsge_constraints(config)

    has_bounds = !isempty(get(con_cfg, "bounds", []))
    has_nonlinear = !isempty(get(con_cfg, "nonlinear", []))

    if has_nonlinear && spec === nothing
        throw(CliError("config/invalid",
            "nonlinear constraints require a loaded DSGE spec (pass spec keyword)"))
    end

    constraints = Any[]

    if has_bounds
        for b in con_cfg["bounds"]
            lo = get(b, "lower", nothing)
            hi = get(b, "upper", nothing)
            lo_arg = (lo === nothing || lo == -Inf) ? nothing : lo
            hi_arg = (hi === nothing || hi == Inf) ? nothing : hi
            (lo_arg === nothing && hi_arg === nothing) && throw(CliError("config/invalid",
                "constraints.bounds for '$(b["variable"])' needs a finite lower or upper"))
            c = variable_bound(Symbol(b["variable"]); lower=lo_arg, upper=hi_arg)
            push!(constraints, c)
        end
    end

    if has_nonlinear
        for nl in con_cfg["nonlinear"]
            raw = nl["expr"]
            expr = try
                raw isa Expr ? raw : Meta.parse(String(raw))
            catch e
                throw(CliError("config/invalid",
                    "could not parse nonlinear constraint expression $(repr(raw)): " *
                    sprint(showerror, e)))
            end
            expr isa Expr || throw(CliError("config/invalid",
                "nonlinear constraint did not parse to an Expr: $(repr(raw))"))
            # parse_constraint compiles against the spec's residual fns → world-age barrier
            c = try
                _dsge_call(parse_constraint, expr, spec)
            catch e
                e isa CliError && rethrow()
                throw(CliError("config/invalid",
                    "parse_constraint failed for $(repr(raw)): $(sprint(showerror, e))"))
            end
            push!(constraints, c)
        end
    end

    return constraints
end

"""Convert loaded constraints to 1 or 2 `OccBinConstraint`s (upstream's only shapes)."""
function _as_occbin_constraints(constraints, spec)
    out = MacroEconometricModels.OccBinConstraint[]
    for c in constraints
        if c isa MacroEconometricModels.OccBinConstraint
            push!(out, c)
        elseif c isa MacroEconometricModels.VariableBound
            var = c.var_name
            if c.lower !== nothing
                expr = Expr(:call, :(>=), Expr(:ref, var, :t), c.lower)
                bind = Expr(:(=), Expr(:ref, var, :t), c.lower)
                push!(out, MacroEconometricModels.OccBinConstraint{Float64}(
                    expr, var, Float64(c.lower), :geq, bind))
            end
            if c.upper !== nothing
                expr = Expr(:call, :(<=), Expr(:ref, var, :t), c.upper)
                bind = Expr(:(=), Expr(:ref, var, :t), c.upper)
                push!(out, MacroEconometricModels.OccBinConstraint{Float64}(
                    expr, var, Float64(c.upper), :leq, bind))
            end
        else
            throw(CliError("usage/invalid",
                "OccBin constraints must be variable bounds or parsed comparison Exprs " *
                "(got $(typeof(c)))"))
        end
    end
    n = length(out)
    n == 0 && throw(CliError("usage/invalid",
        "OccBin requires 1 or 2 constraints; none were usable"))
    n > 2 && throw(CliError("usage/invalid",
        "OccBin supports 1 or 2 constraints (got $n); split the file or drop extra bounds"))
    return out
end

function _occbin_solve_call(spec, cons; periods::Int)
    obs = _as_occbin_constraints(cons, spec)
    shock_path = zeros(Float64, periods, spec.n_exog)
    shock_path[1, 1] = 1.0
    if length(obs) == 1
        return _dsge_call(occbin_solve, spec, obs[1];
                          shock_path=shock_path, nperiods=periods)
    else
        return _dsge_call(occbin_solve, spec, obs[1], obs[2];
                          shock_path=shock_path, nperiods=periods)
    end
end

function _occbin_irf_call(spec, cons; shock_idx::Int, horizon::Int, magnitude::Real)
    obs = _as_occbin_constraints(cons, spec)
    if length(obs) == 1
        return _dsge_call(occbin_irf, spec, obs[1], shock_idx, horizon;
                          magnitude=magnitude)
    else
        return _dsge_call(occbin_irf, spec, obs[1], obs[2], shock_idx, horizon;
                          magnitude=magnitude)
    end
end

"""
    _load_panel_for_did(data, id_col, time_col) -> PanelData

Load panel CSV and print summary for DID/event study commands.
"""
function _load_panel_for_did(data::String, id_col::String, time_col::String)
    pd = load_panel_data(data, id_col, time_col)
    _status_styled("  Panel: $(pd.n_groups) groups, $(div(pd.T_obs, pd.n_groups)) periods, " *
                "$(pd.n_vars) variables"; color=:cyan)
    pd.balanced && _status_styled(" (balanced)"; color=:cyan)
    _status()
    return pd
end

# ── Regression Helpers ────────────────────────────────────

"""
    _load_reg_data(data, dep; weights_col="", clusters_col="", exclude_cols=String[])
        → (y, X, varnames)

Load CSV, split into dependent variable y and regressor matrix X.
If dep is empty, uses first numeric column as y.

`exclude_cols` drops further numeric columns from X without their having to be the
weights/clusters column. W10/#112 needs it for the Conley coordinate and time columns:
latitude/longitude are ordinary numeric CSV columns, so without this they would silently
enter the design matrix as regressors — a wrong point estimate, not an error.
"""
function _load_reg_data(data::String, dep::String; weights_col::String="", clusters_col::String="",
                        exclude_cols::Vector{String}=String[])
    df = load_data(data)
    numcols = variable_names(df)
    # Guard an all-non-numeric CSV before defaulting `--dep` to numcols[1] (else BoundsError →
    # untyped exit-1). Mirrors the same fix in `_load_xy_data`/`_load_iv_data` (adversarial review).
    isempty(numcols) && throw(CliError("data/invalid", "no numeric columns found in the data"))

    dep_col = isempty(dep) ? numcols[1] : dep
    # C067a: typed errors, not bare `error()` — a bad `--dep` or a degenerate column set
    # is user input (exit 3), not a CLI bug (exit 1). Benefits the whole cross-section reg
    # family (reg/iv/logit/probit/ologit/oprobit/mlogit/predict/residuals) that shares this.
    !isempty(dep) && !(dep_col in numcols) && throw(CliError("data/column-range",
        "dependent variable '$dep_col' not found in numeric columns: $(join(numcols, ", "))"))

    exclude = Set([dep_col])
    !isempty(weights_col) && push!(exclude, weights_col)
    !isempty(clusters_col) && push!(exclude, clusters_col)
    for c in exclude_cols
        isempty(c) || push!(exclude, c)
    end
    xcols = filter(c -> !(c in exclude), numcols)
    isempty(xcols) && throw(CliError("data/invalid",
        "no regressor columns remaining after excluding dep='$dep_col'"))

    # Guard missing cells BEFORE the Vector/Matrix{Float64} conversion (which throws an
    # untyped ArgumentError → exit-1). `_numeric_column_names` admits Union{Missing,…}
    # columns, so a single blank cell in dep or any regressor reaches here. Mirror the
    # univariate/multivariate loaders' typed guard for the whole reg family.
    for c in vcat([dep_col], xcols)
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    end
    y = Vector{Float64}(df[!, dep_col])
    X = Matrix{Float64}(df[!, xcols])
    return y, X, xcols
end

"""
    _load_iv_data(data, dep, endogenous, instruments)
        → (; y, X, Z, xcols, zcols, endog_idx, endog_names, inst_names, dep_col)

Shared IV/2SLS data loader for `estimate iv` and `test weak-instrument`, using the standard
(Stata `ivregress`) column partition. `--instruments` lists the **excluded** instruments
only; every other numeric column (besides `--dep` and `--endogenous`) is an exogenous
regressor (include a `const` column of ones for an intercept, matching `estimate reg`). So:

* `X` (regressors) = all numeric \\ ({dep} ∪ excluded-instruments) — exogenous + endogenous;
* `Z` (instruments) = all numeric \\ ({dep} ∪ endogenous) — exogenous + excluded instruments.

This keeps the excluded instruments OUT of the structural regressor matrix. The previous
`X = all-except-dep` layout let them leak into `X`, so MEMs raised an untyped
`Order condition violated (m < k)` on any real multi-instrument IV → uncaught exit-1 (C067b
fix; there was no T3 coverage for `estimate iv`).

All failure modes are TYPED CliErrors, never a bare `error()`: missing `--endogenous`/
`--instruments` → `usage/missing` (2); an unknown column, dep listed as endog/instr, a
column named as both endogenous and instrument, or an endogenous not among the regressors →
`data/column-range` (3); fewer excluded instruments than endogenous regressors (order
condition) or a degenerate regressor set → `data/invalid` (3); a missing cell in any
consumed column → `data/missing-values` (3), guarded BEFORE the `Matrix{Float64}`
conversion that would otherwise throw an untyped `ArgumentError`.
"""
function _load_iv_data(data::String, dep::String, endogenous::String, instruments::String;
                       clusters_col::String="")
    isempty(endogenous) && throw(CliError("usage/missing",
        "--endogenous is required (comma-separated endogenous regressor column names)"))
    isempty(instruments) && throw(CliError("usage/missing",
        "--instruments is required (comma-separated EXCLUDED instrument column names)"))

    df = load_data(data)
    numcols = variable_names(df)
    isempty(numcols) && throw(CliError("data/invalid", "no numeric columns found in the data"))

    dep_col = isempty(dep) ? numcols[1] : dep
    !isempty(dep) && !(dep_col in numcols) && throw(CliError("data/column-range",
        "dependent variable '$dep_col' not found in numeric columns: $(join(numcols, ", "))"))

    endog_names = String[strip(s) for s in split(endogenous, ",") if !isempty(strip(s))]
    inst_names  = String[strip(s) for s in split(instruments, ",") if !isempty(strip(s))]
    isempty(endog_names) && throw(CliError("usage/missing",
        "--endogenous is required (comma-separated endogenous regressor column names)"))
    isempty(inst_names) && throw(CliError("usage/missing",
        "--instruments is required (comma-separated EXCLUDED instrument column names)"))
    for nm in endog_names
        nm in numcols || throw(CliError("data/column-range",
            "endogenous variable '$nm' not found in numeric columns: $(join(numcols, ", "))"))
        nm == dep_col && throw(CliError("data/column-range",
            "endogenous variable '$nm' cannot be the dependent variable"))
    end
    for nm in inst_names
        nm in numcols || throw(CliError("data/column-range",
            "instrument '$nm' not found in numeric columns: $(join(numcols, ", "))"))
        nm == dep_col && throw(CliError("data/column-range",
            "instrument '$nm' cannot be the dependent variable"))
    end
    overlap = intersect(endog_names, inst_names)
    isempty(overlap) || throw(CliError("data/column-range",
        "column(s) $(join(overlap, ", ")) listed as both --endogenous and --instruments (an endogenous regressor cannot be its own excluded instrument)"))
    # Order condition (m ≥ k reduces to |excluded| ≥ |endogenous| here): a clearer message
    # than the wrapped MEMs `Order condition violated`.
    length(inst_names) >= length(endog_names) || throw(CliError("data/invalid",
        "under-identified: need at least as many excluded instruments ($(length(inst_names))) as endogenous regressors ($(length(endog_names)))"))

    # W10/#112: the cluster column is an ordinary numeric CSV column, so it must be kept out
    # of BOTH X and Z — otherwise a clustered AR test would silently regress on its own
    # cluster ids. `_load_clusters` reads it separately.
    if !isempty(clusters_col)
        clusters_col in names(df) || throw(CliError("data/column-range",
            "cluster column '$clusters_col' not found; available: $(join(names(df), ", "))"))
        clusters_col == dep_col && throw(CliError("data/column-range",
            "cluster column '$clusters_col' cannot be the dependent variable"))
        clusters_col in endog_names && throw(CliError("data/column-range",
            "cluster column '$clusters_col' cannot also be an endogenous regressor"))
        clusters_col in inst_names && throw(CliError("data/column-range",
            "cluster column '$clusters_col' cannot also be an excluded instrument"))
    end
    drop = c -> c == dep_col || (!isempty(clusters_col) && c == clusters_col)
    # X = exogenous + endogenous (exclude dep and the excluded instruments).
    xcols = filter(c -> !drop(c) && !(c in inst_names), numcols)
    isempty(xcols) && throw(CliError("data/invalid",
        "no regressor columns remaining after excluding dep='$dep_col' and the instruments"))
    # Z = exogenous + excluded instruments (exclude dep and the endogenous regressors).
    zcols = filter(c -> !drop(c) && !(c in endog_names), numcols)

    endog_idx = Int[]
    for nm in endog_names
        i = findfirst(==(nm), xcols)
        i === nothing && throw(CliError("data/column-range",
            "endogenous variable '$nm' is not among the regressors"))
        push!(endog_idx, i)
    end

    for c in unique(vcat([dep_col], xcols, zcols))
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    end

    y = Vector{Float64}(df[!, dep_col])
    X = Matrix{Float64}(df[!, xcols])
    Z = Matrix{Float64}(df[!, zcols])
    # Guard the degenerate more-instruments-than-observations regime up front: with m ≥ n the
    # first-stage F has non-positive df → MEMs returns a NaN diagnostic (and `robust_inv`
    # silently pinv's the rank-deficient Z'Z rather than throwing), yielding meaningless
    # estimates. A clear typed error beats a NaN verdict downstream (adversarial-review fix).
    size(Z, 2) < length(y) || throw(CliError("data/invalid",
        "too many instruments: the instrument set Z has $(size(Z, 2)) columns but only $(length(y)) observations (need m < n for an identified first stage)"))
    return (; y, X, Z, xcols, zcols, endog_idx, endog_names, inst_names, dep_col)
end

"""
    _load_xy_data(data, dep, indep) → (y, x, ynm, xnm)

Shared response/single-predictor loader for the nonparametric-regression leaves
(`estimate kernel-reg`, `estimate lowess`, C066). `--dep` (default: first numeric column)
is the response `y`; `--indep` (required) is a SINGLE predictor `x`. The 6th shared-loader
hardening: every failure is a TYPED `CliError`, never a bare `error()` (untyped exit-1).
Missing `--indep` → `usage/missing`; an unknown dep/indep column, or dep==indep →
`data/column-range`; a missing cell in either column → `data/missing-values`, guarded
BEFORE the `Vector{Float64}` conversion that would otherwise throw an untyped
`ArgumentError`. Returns `(Vector{Float64} y, Vector{Float64} x, dep_name, indep_name)`.
"""
function _load_xy_data(data::String, dep::String, indep::String)
    isempty(indep) && throw(CliError("usage/missing",
        "--indep is required (the single predictor column name)"))
    df = load_data(data)
    numcols = variable_names(df)
    isempty(numcols) && throw(CliError("data/invalid",
        "no numeric columns found in the data"))
    dep_col = isempty(dep) ? numcols[1] : dep
    !isempty(dep) && !(dep_col in numcols) && throw(CliError("data/column-range",
        "response variable '$dep_col' not found in numeric columns: $(join(numcols, ", "))"))
    indep in numcols || throw(CliError("data/column-range",
        "predictor '$indep' not found in numeric columns: $(join(numcols, ", "))"))
    indep == dep_col && throw(CliError("data/column-range",
        "predictor '$indep' cannot equal the response '$dep_col'"))
    for c in (dep_col, indep)
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    end
    y = Vector{Float64}(df[!, dep_col])
    x = Vector{Float64}(df[!, indep])
    return y, x, dep_col, indep
end

"""
    _parse_bandwidth(s, syms) → Union{Symbol,Real}

Parse a `--bw` argument for the nonparametric leaves (C066): a string matching one of the
allowed rule names in `syms` (e.g. `(:silverman, :sj)` or `(:cv, :rot)`) → that `Symbol`;
otherwise a positive number. Junk, a non-number, or a non-positive value → a typed
`usage/invalid` error (never a raw MEMs `ArgumentError`). Mirrors `_parse_penalty_lambda`.
"""
function _parse_bandwidth(s::AbstractString, syms::Tuple)
    for sym in syms
        s == string(sym) && return sym
    end
    v = tryparse(Float64, s)
    # Reject non-finite too: `Inf`/`NaN` slip past a bare `v <= 0` (both compare false), then
    # either produce a degenerate fit with a non-finite bandwidth or hit MEMs' `h > 0` with the
    # WRONG error class (a bad CLI arg must be usage/invalid, not data/invalid).
    (v === nothing || !isfinite(v) || v <= 0) && throw(CliError("usage/invalid",
        "--bw must be one of $(join(syms, '|')) or a positive number, got '$s'"))
    return v
end

"""Load cluster assignments from a CSV column, or return nothing."""
function _load_clusters(data::String, clusters_col::String)
    isempty(clusters_col) && return nothing
    df = load_data(data)
    clusters_col in names(df) || throw(CliError("data/column-range",
        "cluster column '$clusters_col' not found; available: $(join(names(df), ", "))"))
    col = df[!, clusters_col]
    any(ismissing, col) && throw(CliError("data/missing-values",
        "cluster column '$clusters_col' contains missing values; every observation must " *
        "belong to a cluster"))
    # DENSE-RANK to Int codes rather than `Vector{Int}(col)`. Cluster identity is all that
    # is ever used downstream (MEMs' `_cluster_vcov`/`wild_cluster_bootstrap` only call
    # `unique` and group by equality), so the partition — and hence every result — is
    # identical; but the old conversion raised an untyped `MethodError`/`InexactError` on a
    # String or non-integral Float cluster column, i.e. exit 1 on ordinary input like
    # `state = "CA"` or `firm_id = 3.0`. Ninth site in the shared-loader hardening class.
    levels = unique(col)
    code = Dict(v => i for (i, v) in enumerate(levels))
    length(levels) >= 2 || throw(CliError("data/invalid",
        "cluster column '$clusters_col' has only $(length(levels)) distinct value(s); " *
        "cluster-robust inference needs at least 2 clusters"))
    return Int[code[v] for v in col]
end

"""Load observation weights from a CSV column, or return nothing."""
function _load_weights(data::String, weights_col::String)
    isempty(weights_col) && return nothing
    df = load_data(data)
    weights_col in names(df) || throw(CliError("data/column-range",
        "weights column '$weights_col' not found; available: $(join(names(df), ", "))"))
    col = df[!, weights_col]
    any(ismissing, col) && throw(CliError("data/missing-values",
        "weights column '$weights_col' contains missing values"))
    w = try
        Vector{Float64}(col)
    catch
        throw(CliError("data/invalid", "weights column '$weights_col' is not numeric"))
    end
    # MEMs throws a bare `ArgumentError("All weights must be positive")`; catch it here so
    # a non-positive weight is typed user input rather than an internal exit-1.
    all(>(0.0), w) || throw(CliError("data/invalid",
        "weights column '$weights_col' must be strictly positive (found $(count(<=(0.0), w)) non-positive value(s))"))
    return w
end

"""
    _load_coords(data, lat_col, lon_col) → Matrix{Float64} (n × 2)

Conley (1999) spatial-HAC coordinate loader (W10/#112). Both columns are required and are
loaded as an `n × 2` `[lat lon]` matrix in the order `conley_se`/`estimate_reg` expect
(`metric=:haversine` reads column 1 as latitude, column 2 as longitude). Missing cells are
rejected BEFORE the `Matrix{Float64}` conversion that would otherwise throw untyped, and
out-of-range degrees are caught here because `:haversine` would silently return nonsense
distances rather than fail.
"""
function _load_coords(data::String, lat_col::String, lon_col::String, metric::String)
    df = load_data(data)
    for (role, c) in (("--lat", lat_col), ("--lon", lon_col))
        c in names(df) || throw(CliError("data/column-range",
            "$role column '$c' not found; available: $(join(names(df), ", "))"))
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "$role column '$c' contains missing values; coordinates must be fully observed"))
    end
    lat = try
        Vector{Float64}(df[!, lat_col])
    catch
        throw(CliError("data/invalid", "--lat column '$lat_col' is not numeric"))
    end
    lon = try
        Vector{Float64}(df[!, lon_col])
    catch
        throw(CliError("data/invalid", "--lon column '$lon_col' is not numeric"))
    end
    all(isfinite, lat) && all(isfinite, lon) || throw(CliError("data/invalid",
        "coordinate columns '$lat_col'/'$lon_col' contain non-finite values"))
    if metric == "haversine"
        all(v -> -90.0 <= v <= 90.0, lat) || throw(CliError("data/invalid",
            "--conley-metric haversine needs --lat '$lat_col' in degrees within [-90, 90]"))
        all(v -> -180.0 <= v <= 180.0, lon) || throw(CliError("data/invalid",
            "--conley-metric haversine needs --lon '$lon_col' in degrees within [-180, 180]"))
    end
    return hcat(lat, lon)
end

"""Build coefficient table DataFrame from a regression model."""
function _reg_coef_table(model, varnames::Vector{String})
    # C051: MEMs renders coefficient-bearing models (RegModel/Logit/Probit/IV, which the
    # CLI estimates with varnames=xcols) as a tidy coef table via Tables.jl — term|estimate|
    # std_error|stat|p_value|ci_lower|ci_upper. `varnames` is retained for the call sites
    # but the names now come from the model itself.
    DataFrame(model)
end

# --- Panel Regression Shared Helpers (v0.4.0) ---

"""Load panel CSV for panel regression. Returns PanelData."""
function _load_panel_for_preg(data::String, id_col::String, time_col::String)
    df = load_data(data)
    cols = names(df)
    id = isempty(id_col) ? cols[1] : id_col
    tc = isempty(time_col) ? cols[2] : time_col
    pd = load_panel_data(data, id, tc)
    _status_styled("  Panel: $(pd.n_groups) groups, $(pd.n_vars) variables"; color=:cyan)
    pd.balanced && _status_styled(" (balanced)"; color=:cyan)
    _status()
    return pd
end

"""Parse indep vars from comma-separated string. If empty, infer from all non-dep numeric cols."""
function _parse_indep_vars(pd, dep::String, indep_str::String)
    if isempty(indep_str)
        all_vars = pd.varnames
        return Symbol[Symbol(v) for v in all_vars if v != dep]
    else
        return Symbol[Symbol(strip(s)) for s in split(indep_str, ",")]
    end
end

"""Convert CLI option value with hyphens to MEMs Symbol with underscores."""
_to_sym(s::String) = Symbol(replace(s, "-" => "_"))

"""Build coefficient table from panel regression model."""
function _preg_coef_table(model, varnames::Vector{String})
    # C051: MEMs renders panel coefficient models (PanelReg/IV/Logit/Probit) as a tidy
    # coef table via Tables.jl (term|estimate|std_error|stat|p_value|ci_lower|ci_upper).
    # `varnames` retained for the call sites; names now come from the model.
    DataFrame(model)
end
