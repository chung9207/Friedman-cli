# Fitted values / residuals — collapsed family (C025 / P2-3 / F9)
# Surface unchanged: `predict <model>` and `residuals <model>` remain distinct paths.

# Model kinds shared by predict + residuals (23). Specs generated from this table.
# Per-kind extra options stay on the relevant leaves only.



# ── VAR Predict ─────────────────────────────────────────

function _predict_var(; data::String="", lags=nothing,
                       output::String="", format::String="table",
                       model=nothing)
    if isnothing(model)
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
    else
        varnames = model.varnames
        p = model.p
    end
    _status("Computing VAR($p) in-sample predictions: $(length(varnames)) variables")
    _status()

    fitted = predict(model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="VAR($p) In-Sample Predictions (T_eff=$T_eff)",
                  key="var_predictions")
end

# ── BVAR Predict ────────────────────────────────────────

function _predict_bvar(; data::String="", lags::Int=4, draws::Int=2000,
                        sampler::String="direct", config::String="",
                        output::String="", format::String="table",
                        model=nothing)
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        Y = post.Y
    end

    _status("Computing BVAR($p) in-sample predictions (posterior mean)")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

    var_model = posterior_mean_model(post; data=Y)
    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="BVAR($p) In-Sample Predictions (posterior mean, T_eff=$T_eff)",
                  key="bvar_predictions")
end

# ── ARIMA Predict ───────────────────────────────────────

function _predict_arima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          method::String="css_mle", auto::Bool=false,
                          output::String="", format::String="table",
                          model=nothing)
    # Bound BEFORE the branch: on the `--model <handle>` path there is no CSV to read a
    # column name from, and the title below is emitted on BOTH paths. Leaving it to the
    # branch made every `predict arima --model …` die with an UndefVarError (exit 1).
    vname = "model"
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        method_sym = Symbol(method)
        safe_method = method_sym == :css_mle ? :mle : method_sym

        model = if isnothing(p) || auto
            _status("Auto ARIMA predict: variable=$vname, observations=$(length(y))")
            _status()
            m = auto_arima(y; method=safe_method)
            label = _model_label(ar_order(m), diff_order(m), ma_order(m))
            _status_styled("Selected model: $label\n"; bold=true)
            _status()
            m
        else
            label = _model_label(p, d, q)
            _status("$label predict: variable=$vname")
            _status()
            _estimate_arima_model(y, p, d, q; method=method_sym)
        end
    end

    fitted = predict(model)

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    pred_df = DataFrame(
        t=1:length(fitted),
        fitted=round.(fitted; digits=6)
    )

    output_result(pred_df; format=Symbol(format), output=output,
                  title="$label In-Sample Predictions for $vname",
                  key="arima_predictions")
end

# ── VECM Predict ───────────────────────────────────────

function _predict_vecm(; data::String="", lags::Int=2, rank::String="auto",
                         deterministic::String="constant",
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    else
        vecm = model
        varnames = vecm.varnames
        p = vecm.p
    end
    r = cointegrating_rank(vecm)

    _status("Computing VECM in-sample predictions: rank=$r, lags=$p")
    _status()

    var_model = to_var(vecm)
    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="VECM In-Sample Predictions (rank=$r, T_eff=$T_eff)",
                  key="vecm_predictions")
end

# ── Static Factor Predict ─────────────────────────────

function _predict_static(; data::String="", nfactors=nothing,
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            ic = ic_criteria(X, min(20, size(X, 2)))
            ic.r_IC1
        else
            nfactors
        end

        fm = estimate_factors(X, r)
    else
        fm = model
        varnames = fm.varnames
        r = fm.r
    end
    fitted = predict(fm)
    T = size(fitted, 1)

    _status("Static factor model: $r factors, common component (T=$T)")
    _status()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="Static Factor Common Component ($r factors, T=$T)",
                  key="static_factor_common_component")
end

# ── Dynamic Factor Predict ────────────────────────────

function _predict_dynamic(; data::String="", nfactors=nothing, factor_lags::Int=1,
                            method::String="twostep",
                            output::String="", format::String="table",
                            model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            ic = ic_criteria(X, min(10, size(X, 2)))
            ic.r_IC1
        else
            nfactors
        end

        fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    else
        fm = model
        r = fm.r
        factor_lags = fm.p
        # DynamicFactorModel carries no varnames upstream — use real's own default naming
        varnames = ["Var $i" for i in 1:size(fm.X, 2)]
    end
    fitted = predict(fm)
    T = size(fitted, 1)

    _status("Dynamic factor model: $r factors, p=$factor_lags, common component (T=$T)")
    _status()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Common Component ($r factors, p=$factor_lags, T=$T)",
                  key="dynamic_factor_common_component")
end

# ── GDFM Predict ──────────────────────────────────────

function _predict_gdfm(; data::String="", nfactors=nothing, dynamic_rank=nothing,
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        q = if isnothing(dynamic_rank)
            ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
            ic.q_ratio
        else
            dynamic_rank
        end

        gm = estimate_gdfm(X, q)
    else
        gm = model
        q = gm.q
        # GeneralizedDynamicFactorModel carries no varnames upstream
        varnames = ["Var $i" for i in 1:size(gm.X, 2)]
    end
    fitted = predict(gm)
    T = size(fitted, 1)

    _status("GDFM: q=$q dynamic factors, common component (T=$T)")
    _status()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="GDFM Common Component (q=$q, T=$T)",
                  key="gdfm_common_component")
end

# Volatility predict handlers live in shared.jl (VOL_MODELS / _VOL_PREDICT_HANDLERS).

# ── FAVAR Predict ─────────────────────────────────────────

function _predict_favar(; data::String="", factors=nothing, lags::Int=2,
                         key_vars::String="",
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end
    var_model = to_var(favar)

    _status("FAVAR In-Sample Prediction")
    _status()

    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for v in 1:size(fitted, 2)
        vname = v <= length(favar.varnames) ? favar.varnames[v] : "var_$v"
        pred_df[!, vname] = round.(fitted[:, v]; digits=6)
    end
    output_result(pred_df; format=Symbol(format), output=output,
                  title="FAVAR In-Sample Predictions (T_eff=$T_eff)",
                  key="favar_predictions")
end

# ── Regression Predict ────────────────────────────────────

function _predict_reg(; data::String="", dep::String="", cov_type::String="hc1",
                       weights::String="", clusters::String="",
                       output::String="", format::String="table",
                       model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; weights_col=weights, clusters_col=clusters)
        w = _load_weights(data, weights)
        cl = _load_clusters(data, clusters)
        model = estimate_reg(y, X; cov_type=Symbol(cov_type), weights=w, varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
        wls_tag = isnothing(w) ? "OLS" : "WLS"
    else
        xcols = model.varnames
        dep_name = "y"
        wls_tag = "OLS"
    end
    _status("$wls_tag Fitted Values: $dep_name ~ $(join(xcols, " + "))")
    _status()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output, title="$wls_tag Fitted Values",
                  key="reg_fitted_values")
end

# ── Logit Predict ─────────────────────────────────────────

function _predict_logit(; data::String="", dep::String="", cov_type::String="hc1",
                         clusters::String="", threshold::Float64=0.5,
                         marginal_effects::Bool=false, odds_ratio::Bool=false,
                         classification_table::Bool=false,
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
        cl = _load_clusters(data, clusters)
        model = estimate_logit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    else
        dep_name = "y"
    end

    if marginal_effects
        me = MacroEconometricModels.marginal_effects(model)
        me_df = DataFrame(Variable=me.varnames, Effect=round.(me.effects; digits=6),
            SE=round.(me.se; digits=6), z=round.(me.z_stat; digits=4),
            p_value=round.(me.p_values; digits=4),
            CI_Lower=round.(me.ci_lower; digits=6), CI_Upper=round.(me.ci_upper; digits=6))
        output_result(me_df; format=Symbol(format), output=output,
                      title="Average Marginal Effects (Logit)")
    elseif odds_ratio
        or = MacroEconometricModels.odds_ratio(model)
        or_df = DataFrame(Variable=or.varnames, Odds_Ratio=round.(or.or; digits=6),
            CI_Lower=round.(or.ci_lower; digits=6), CI_Upper=round.(or.ci_upper; digits=6))
        output_result(or_df; format=Symbol(format), output=output,
                      title="Odds Ratios (Logit)")
    elseif classification_table
        ct = MacroEconometricModels.classification_table(model; threshold=threshold)
        _status("Classification Table (threshold=$threshold):")
        # Sort by KEY: the dict mixes scalars with a `confusion` matrix, so sorting
        # the pairs themselves compares a Matrix against a Float (exit 1).
        scalars = Pair{String,Any}[]
        confusion = nothing
        for k in sort(collect(keys(ct)))
            v = ct[k]
            if v isa AbstractMatrix
                confusion = v
            else
                push!(scalars, k => v isa AbstractFloat ? round(v; digits=6) : v)
            end
        end
        output_kv(scalars; format=format, output=output,
                  title="Classification Metrics (threshold=$threshold)",
                  key="classification_metrics")
        if confusion !== nothing
            conf_df = DataFrame(confusion, ["predicted_$j" for j in 0:size(confusion, 2) - 1];
                                makeunique=true)
            insertcols!(conf_df, 1, :actual => ["actual_$i" for i in 0:size(confusion, 1) - 1];
                        makeunique=true)
            output_result(conf_df; format=Symbol(format),
                          output=_per_var_output_path(output, "confusion"),
                          title="Confusion Matrix")
        end
    else
        _status("Logit Fitted Probabilities: $dep_name")
        _status()
        fitted = predict(model)
        pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
        output_result(pred_df; format=Symbol(format), output=output,
                      title="Logit Fitted Probabilities")
    end
end

# ── Probit Predict ────────────────────────────────────────

function _predict_probit(; data::String="", dep::String="", cov_type::String="hc1",
                          clusters::String="", threshold::Float64=0.5,
                          marginal_effects::Bool=false,
                          classification_table::Bool=false,
                          output::String="", format::String="table",
                          model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
        cl = _load_clusters(data, clusters)
        model = estimate_probit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    else
        dep_name = "y"
    end

    if marginal_effects
        me = MacroEconometricModels.marginal_effects(model)
        me_df = DataFrame(Variable=me.varnames, Effect=round.(me.effects; digits=6),
            SE=round.(me.se; digits=6), z=round.(me.z_stat; digits=4),
            p_value=round.(me.p_values; digits=4),
            CI_Lower=round.(me.ci_lower; digits=6), CI_Upper=round.(me.ci_upper; digits=6))
        output_result(me_df; format=Symbol(format), output=output,
                      title="Average Marginal Effects (Probit)")
    elseif classification_table
        ct = MacroEconometricModels.classification_table(model; threshold=threshold)
        _status("Classification Table (threshold=$threshold):")
        # Sort by KEY: the dict mixes scalars with a `confusion` matrix, so sorting
        # the pairs themselves compares a Matrix against a Float (exit 1).
        scalars = Pair{String,Any}[]
        confusion = nothing
        for k in sort(collect(keys(ct)))
            v = ct[k]
            if v isa AbstractMatrix
                confusion = v
            else
                push!(scalars, k => v isa AbstractFloat ? round(v; digits=6) : v)
            end
        end
        output_kv(scalars; format=format, output=output,
                  title="Classification Metrics (threshold=$threshold)",
                  key="classification_metrics")
        if confusion !== nothing
            conf_df = DataFrame(confusion, ["predicted_$j" for j in 0:size(confusion, 2) - 1];
                                makeunique=true)
            insertcols!(conf_df, 1, :actual => ["actual_$i" for i in 0:size(confusion, 1) - 1];
                        makeunique=true)
            output_result(conf_df; format=Symbol(format),
                          output=_per_var_output_path(output, "confusion"),
                          title="Confusion Matrix")
        end
    else
        _status("Probit Fitted Probabilities: $dep_name")
        _status()
        fitted = predict(model)
        pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
        output_result(pred_df; format=Symbol(format), output=output,
                      title="Probit Fitted Probabilities")
    end
end

# ── Panel Regression Predict ────────────────────────────

function _predict_preg(; data::String, dep::String="", indep::String="",
                        method::String="fe", cov_type::String="cluster",
                        id_col::String="", time_col::String="",
                        output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtreg(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Regression Fitted Values ($method): $dep")
    _status()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Regression Fitted Values ($method)",
                  key="panel_regression_fitted_values")
end

function _predict_piv(; data::String, dep::String="", exog::String="",
                       endog::String="", instruments::String="",
                       method::String="fe", cov_type::String="cluster",
                       id_col::String="", time_col::String="",
                       output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    isempty(endog) && error("--endog is required")
    pd = _load_panel_for_preg(data, id_col, time_col)

    exog_syms = isempty(exog) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(exog, ",")]
    endog_syms = Symbol[Symbol(strip(s)) for s in split(endog, ",")]
    inst_syms = isempty(instruments) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(instruments, ",")]

    model = estimate_xtiv(pd, Symbol(dep), exog_syms, endog_syms;
        instruments=inst_syms, model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel IV Fitted Values ($method): $dep")
    _status()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel IV Fitted Values ($method)",
                  key="panel_iv_fitted_values")
end

function _predict_plogit(; data::String, dep::String="", indep::String="",
                          method::String="pooled", cov_type::String="cluster",
                          id_col::String="", time_col::String="",
                          output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtlogit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Logit Fitted Probabilities ($method): $dep")
    _status()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Logit Fitted Probabilities ($method)",
                  key="panel_logit_fitted_probabilities")
end

function _predict_pprobit(; data::String, dep::String="", indep::String="",
                           method::String="pooled", cov_type::String="cluster",
                           id_col::String="", time_col::String="",
                           output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtprobit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Probit Fitted Probabilities ($method): $dep")
    _status()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Probit Fitted Probabilities ($method)",
                  key="panel_probit_fitted_probabilities")
end

# ── Ordered/Multinomial Predict ─────────────────────────

"""
    _choice_prob_table(probs, categories) → DataFrame

Tidy per-category predicted probabilities.

`predict` on an ordered/multinomial model returns an `n x n_categories` matrix, not a
vector — feeding it straight to `DataFrame` raised "adding AbstractArray other than
AbstractVector as a column" (exit 1) on every call (#85).
"""
function _choice_prob_table(probs::AbstractMatrix, categories)
    df = DataFrame(observation=1:size(probs, 1))
    labels = length(categories) == size(probs, 2) ? string.(categories) :
             string.(1:size(probs, 2))
    for (j, lab) in enumerate(labels)
        df[!, "prob_$lab"] = round.(probs[:, j]; digits=6)
    end
    return df
end

# A univariate result still renders as a single fitted-probability column.
_choice_prob_table(probs::AbstractVector, categories) =
    DataFrame(observation=1:length(probs), fitted_prob=round.(probs; digits=6))

"""
    _choice_me_table(me) → (DataFrame, has_se)

Tidy average marginal effects for the ordered/multinomial models (MEMs#550,
W10/#131): one row per variable × category, columns `variable|category|dydx[|se]`.

Both upstream shapes land here — the ordered models return a NamedTuple and
mlogit the exported `MultinomialMarginalEffects` struct — but the four fields
read identically. Notes: mlogit's `varnames` EXCLUDE intercept rows;
"SE unavailable" is spelled `nothing` on mlogit but a NaN-filled matrix on the
ordered models, so both are checked. Every variable's effects sum to ~0 across
categories (∂Σp/∂x = 0), base category included. Neither family carries
z/p/CI — deliberately NOT routed through the shared `MarginalEffects` renderer.
"""
function _choice_me_table(me)
    K, J = size(me.effects)
    labels = length(me.categories) == J ? string.(me.categories) : string.(1:J)
    df = DataFrame(variable=repeat(String.(me.varnames); inner=J),
                   category=repeat(labels; outer=K),
                   dydx=round.(vec(permutedims(me.effects)); digits=6))
    has_se = me.se !== nothing && !all(isnan, me.se)
    has_se && (df[!, :se] = round.(vec(permutedims(me.se)); digits=6))
    return df, has_se
end

"""Emit the AME table for a discrete-choice predict leaf (2nd table → distinct
output path, or `--output` would silently drop the probabilities). `key_stem` is the
caller's registry-declared table stem (W3/#138)."""
function _output_choice_me(model, model_label::String, key_stem::String;
                           format::String="table", output::String="")
    me = MacroEconometricModels.marginal_effects(model)
    me_df, has_se = _choice_me_table(me)
    has_se || _status_styled(
        "  Note: standard errors unavailable (model covariance missing)\n"; color=:yellow)
    _status()
    output_result(me_df; format=Symbol(format),
                  output=_per_var_output_path(output, "marginal_effects"),
                  title="$model_label Average Marginal Effects",
                  key="$(key_stem)_average_marginal_effects")
end

function _predict_ologit(; data::String="", dep::String="", cov_type::String="hc1",
                          clusters::String="", marginal_effects::Bool=false,
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    _status("Ordered Logit Predicted Probabilities: $dep_name")
    _status()

    fitted = predict(model)
    pred_df = _choice_prob_table(fitted, model.categories)
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Ordered Logit Predicted Probabilities")
    marginal_effects && _output_choice_me(model, "Ordered Logit", "ordered_logit";
                                          format=format, output=output)
end

function _predict_oprobit(; data::String="", dep::String="", cov_type::String="hc1",
                           clusters::String="", marginal_effects::Bool=false,
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    _status("Ordered Probit Predicted Probabilities: $dep_name")
    _status()

    fitted = predict(model)
    pred_df = _choice_prob_table(fitted, model.categories)
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Ordered Probit Predicted Probabilities")
    marginal_effects && _output_choice_me(model, "Ordered Probit", "ordered_probit";
                                          format=format, output=output)
end

function _predict_mlogit(; data::String="", dep::String="", cov_type::String="ols",
                          clusters::String="", marginal_effects::Bool=false,
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    _status("Multinomial Logit Predicted Probabilities: $dep_name")
    _status()

    fitted = predict(model)
    pred_df = _choice_prob_table(fitted, model.categories)
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Multinomial Logit Predicted Probabilities")
    marginal_effects && _output_choice_me(model, "Multinomial Logit", "multinomial_logit";
                                          format=format, output=output)
end



# ── VAR Residuals ───────────────────────────────────────

function _residuals_var(; data::String="", lags=nothing,
                          output::String="", format::String="table",
                          model=nothing)
    if isnothing(model)
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
    else
        varnames = model.varnames
        p = model.p
    end
    _status("Computing VAR($p) residuals: $(length(varnames)) variables")
    _status()

    resid = residuals(model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="VAR($p) Residuals (T_eff=$T_eff)",
                  key="var_residuals")
end

# ── BVAR Residuals ──────────────────────────────────────

function _residuals_bvar(; data::String="", lags::Int=4, draws::Int=2000,
                           sampler::String="direct", config::String="",
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        Y = post.Y
    end

    _status("Computing BVAR($p) residuals (posterior mean)")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

    var_model = posterior_mean_model(post; data=Y)
    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="BVAR($p) Residuals (posterior mean, T_eff=$T_eff)",
                  key="bvar_residuals")
end

# ── ARIMA Residuals ─────────────────────────────────────

function _residuals_arima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                            method::String="css_mle", auto::Bool=false,
                            output::String="", format::String="table",
                            model=nothing)
    # See _predict_arima: bound before the branch so the shared title below cannot hit an
    # UndefVarError on the `--model <handle>` path.
    vname = "model"
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        method_sym = Symbol(method)
        safe_method = method_sym == :css_mle ? :mle : method_sym

        model = if isnothing(p) || auto
            _status("Auto ARIMA residuals: variable=$vname, observations=$(length(y))")
            _status()
            m = auto_arima(y; method=safe_method)
            label = _model_label(ar_order(m), diff_order(m), ma_order(m))
            _status_styled("Selected model: $label\n"; bold=true)
            _status()
            m
        else
            label = _model_label(p, d, q)
            _status("$label residuals: variable=$vname")
            _status()
            _estimate_arima_model(y, p, d, q; method=method_sym)
        end
    end

    resid = residuals(model)

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    res_df = DataFrame(
        t=1:length(resid),
        residual=round.(resid; digits=6)
    )

    output_result(res_df; format=Symbol(format), output=output,
                  title="$label Residuals for $vname",
                  key="arima_residuals")
end

# ── VECM Residuals ─────────────────────────────────────

function _residuals_vecm(; data::String="", lags::Int=2, rank::String="auto",
                           deterministic::String="constant",
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    else
        vecm = model
        varnames = vecm.varnames
        p = vecm.p
    end
    r = cointegrating_rank(vecm)

    _status("Computing VECM residuals: rank=$r, lags=$p")
    _status()

    var_model = to_var(vecm)
    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="VECM Residuals (rank=$r, T_eff=$T_eff)",
                  key="vecm_residuals")
end

# ── Static Factor Residuals ───────────────────────────

function _residuals_static(; data::String="", nfactors=nothing,
                             output::String="", format::String="table",
                             model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            ic = ic_criteria(X, min(20, size(X, 2)))
            ic.r_IC1
        else
            nfactors
        end

        fm = estimate_factors(X, r)
    else
        fm = model
        varnames = fm.varnames
        r = fm.r
    end
    resid = residuals(fm)
    T = size(resid, 1)

    _status("Static factor model residuals: $r factors, idiosyncratic component (T=$T)")
    _status()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="Static Factor Idiosyncratic Component ($r factors, T=$T)",
                  key="static_factor_idiosyncratic_component")
end

# ── Dynamic Factor Residuals ──────────────────────────

function _residuals_dynamic(; data::String="", nfactors=nothing, factor_lags::Int=1,
                              method::String="twostep",
                              output::String="", format::String="table",
                              model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            ic = ic_criteria(X, min(10, size(X, 2)))
            ic.r_IC1
        else
            nfactors
        end

        fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    else
        fm = model
        r = fm.r
        factor_lags = fm.p
        # DynamicFactorModel carries no varnames upstream — use real's own default naming
        varnames = ["Var $i" for i in 1:size(fm.X, 2)]
    end
    resid = residuals(fm)
    T = size(resid, 1)

    _status("Dynamic factor model residuals: $r factors, p=$factor_lags, idiosyncratic component (T=$T)")
    _status()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Idiosyncratic Component ($r factors, p=$factor_lags, T=$T)",
                  key="dynamic_factor_idiosyncratic_component")
end

# ── GDFM Residuals ────────────────────────────────────

function _residuals_gdfm(; data::String="", nfactors=nothing, dynamic_rank=nothing,
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        q = if isnothing(dynamic_rank)
            ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
            ic.q_ratio
        else
            dynamic_rank
        end

        gm = estimate_gdfm(X, q)
    else
        gm = model
        q = gm.q
        # GeneralizedDynamicFactorModel carries no varnames upstream
        varnames = ["Var $i" for i in 1:size(gm.X, 2)]
    end
    resid = residuals(gm)
    T = size(resid, 1)

    _status("GDFM residuals: q=$q dynamic factors, idiosyncratic component (T=$T)")
    _status()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="GDFM Idiosyncratic Component (q=$q, T=$T)",
                  key="gdfm_idiosyncratic_component")
end

# Volatility residual handlers live in shared.jl (VOL_MODELS / _VOL_RESIDUALS_HANDLERS).

# ── FAVAR Residuals ───────────────────────────────────────

function _residuals_favar(; data::String="", factors=nothing, lags::Int=2,
                           key_vars::String="",
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end
    var_model = to_var(favar)

    _status("FAVAR Residuals")
    _status()

    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for v in 1:size(resid, 2)
        vname = v <= length(favar.varnames) ? favar.varnames[v] : "var_$v"
        res_df[!, vname] = round.(resid[:, v]; digits=6)
    end
    output_result(res_df; format=Symbol(format), output=output,
                  title="FAVAR Residuals (T_eff=$T_eff)",
                  key="favar_residuals")
end

# ── Regression Residuals ──────────────────────────────────

function _residuals_reg(; data::String="", dep::String="", cov_type::String="hc1",
                         weights::String="", clusters::String="",
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; weights_col=weights, clusters_col=clusters)
        w = _load_weights(data, weights)
        cl = _load_clusters(data, clusters)
        model = estimate_reg(y, X; cov_type=Symbol(cov_type), weights=w, varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
        wls_tag = isnothing(w) ? "OLS" : "WLS"
    else
        xcols = model.varnames
        dep_name = "y"
        wls_tag = "OLS"
    end
    _status("$wls_tag Residuals: $dep_name ~ $(join(xcols, " + "))")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output, title="$wls_tag Residuals",
                  key="reg_residuals")
end

# ── Logit Residuals ───────────────────────────────────────

function _residuals_logit(; data::String="", dep::String="", cov_type::String="hc1",
                           clusters::String="",
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
        cl = _load_clusters(data, clusters)
        model = estimate_logit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    else
        xcols = model.varnames
        dep_name = "y"
    end
    _status("Logit Residuals: $dep_name ~ $(join(xcols, " + "))")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output, title="Logit Residuals")
end

# ── Probit Residuals ──────────────────────────────────────

function _residuals_probit(; data::String="", dep::String="", cov_type::String="hc1",
                            clusters::String="",
                            output::String="", format::String="table",
                            model=nothing)
    if isnothing(model)
        y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
        cl = _load_clusters(data, clusters)
        model = estimate_probit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)
        dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    else
        xcols = model.varnames
        dep_name = "y"
    end
    _status("Probit Residuals: $dep_name ~ $(join(xcols, " + "))")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output, title="Probit Residuals")
end

# ── Panel Regression Residuals ──────────────────────────

function _residuals_preg(; data::String="", dep::String="", indep::String="",
                          method::String="fe", cov_type::String="cluster",
                          id_col::String="", time_col::String="",
                          output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtreg(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Regression Residuals ($method): $dep")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Regression Residuals ($method)",
                  key="panel_regression_residuals")
end

function _residuals_piv(; data::String="", dep::String="", exog::String="",
                         endog::String="", instruments::String="",
                         method::String="fe", cov_type::String="cluster",
                         id_col::String="", time_col::String="",
                         output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    isempty(endog) && error("--endog is required")
    pd = _load_panel_for_preg(data, id_col, time_col)

    exog_syms = isempty(exog) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(exog, ",")]
    endog_syms = Symbol[Symbol(strip(s)) for s in split(endog, ",")]
    inst_syms = isempty(instruments) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(instruments, ",")]

    model = estimate_xtiv(pd, Symbol(dep), exog_syms, endog_syms;
        instruments=inst_syms, model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel IV Residuals ($method): $dep")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel IV Residuals ($method)",
                  key="panel_iv_residuals")
end

function _residuals_plogit(; data::String="", dep::String="", indep::String="",
                            method::String="pooled", cov_type::String="cluster",
                            id_col::String="", time_col::String="",
                            output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtlogit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Logit Residuals ($method): $dep")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Logit Residuals ($method)",
                  key="panel_logit_residuals")
end

function _residuals_pprobit(; data::String="", dep::String="", indep::String="",
                             method::String="pooled", cov_type::String="cluster",
                             id_col::String="", time_col::String="",
                             output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model = estimate_xtprobit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    _status("Panel Probit Residuals ($method): $dep")
    _status()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Probit Residuals ($method)",
                  key="panel_probit_residuals")
end

# ── Ordered/Multinomial Residuals ───────────────────────

"""
    _choice_resid_table(resid, categories) → DataFrame

Per-category residual matrix (`n x J`) as one tidy column per category, mirroring
`_choice_prob_table`. An unordered/ordered response has J residuals per observation, so
this cannot collapse to a single `residual` column the way the binary models do.
"""
function _choice_resid_table(resid::AbstractMatrix, categories)
    df = DataFrame(observation=1:size(resid, 1))
    labels = length(categories) == size(resid, 2) ? string.(categories) :
             string.(1:size(resid, 2))
    for (j, lab) in enumerate(labels)
        df[!, "resid_$lab"] = round.(resid[:, j]; digits=6)
    end
    return df
end

"""
    _choice_residuals_output(model, label, key_stem, kind, generalized; format, output)

Shared renderer for `residuals ologit|oprobit|mlogit` (W4/#87, un-gated by MEMs#507).

`generalized=true` emits the length-`n` score residual (ordered models only — the flag is
not declared on mlogit). Otherwise emits the `n x J` per-category matrix selected by
`kind`. Every upstream call is wrapped: these are the raw MEMs entry points, and an
untyped exception here would surface as `internal/error` (exit 1) on ordinary bad input.

`key_stem` is the caller's registry-declared table stem (W3/#138): `--kind` selects a
different residual DEFINITION but the same table shape, so all three kinds share one
envelope key; the generalized residual is a different table (one length-n column) and
gets its own.
"""
function _choice_residuals_output(model, label::String, key_stem::String,
                                  kind::String, generalized::Bool;
                                  format::String, output::String)
    if generalized
        g = try
            generalized_residuals(model)
        catch e
            throw(CliError("model/error",
                "$label generalized residuals failed";
                hint=sprint(showerror, e)))
        end
        df = DataFrame(observation=1:length(g),
                       generalized_residual=round.(g; digits=6))
        return output_result(df; format=Symbol(format), output=output,
                             title="$label Generalized Residuals",
                             key="$(key_stem)_generalized_residuals")
    end
    R = try
        residuals(model; kind=Symbol(kind))
    catch e
        throw(CliError("model/error",
            "$label residuals (kind=$kind) failed";
            hint=sprint(showerror, e)))
    end
    df = R isa AbstractMatrix ? _choice_resid_table(R, model.categories) :
         DataFrame(observation=1:length(R), residual=round.(R; digits=6))
    return output_result(df; format=Symbol(format), output=output,
                         title="$label Residuals ($kind)",
                         key="$(key_stem)_residuals")
end

function _residuals_ologit(; data::String="", dep::String="", cov_type::String="hc1",
                            clusters::String="", kind::String="response",
                            generalized::Bool=false,
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    _status("Ordered Logit Residuals: $dep_name")
    _status()

    return _choice_residuals_output(model, "Ordered Logit", "ordered_logit", kind, generalized;
                                    format=format, output=output)
end

function _residuals_oprobit(; data::String="", dep::String="", cov_type::String="hc1",
                             clusters::String="", kind::String="response",
                             generalized::Bool=false,
                             output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    _status("Ordered Probit Residuals: $dep_name")
    _status()

    return _choice_residuals_output(model, "Ordered Probit", "ordered_probit", kind, generalized;
                                    format=format, output=output)
end

function _residuals_mlogit(; data::String="", dep::String="", cov_type::String="ols",
                            clusters::String="", kind::String="response",
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols)

    _status("Multinomial Logit Residuals: $dep_name")
    _status()

    # No `generalized` kwarg here on purpose — see _ORDERED_RESID_FLAGS: upstream defines
    # generalized_residuals for ordered models only, so the flag is not declared for mlogit
    # and must not be accepted here either (a declared option and its handler kwarg have to
    # agree in both directions).
    return _choice_residuals_output(model, "Multinomial Logit", "multinomial_logit", kind, false;
                                    format=format, output=output)
end


function _fitted_data_arg()
    return [ArgSpec(name="data", description="Path to CSV data file")]
end

# ── W3/#138: declared envelope table keys ────────────────────────────────────
#
# The `data` key of every `predict`/`residuals` leaf, keyed by model name. Keys are
# STATIC: the run-varying parts that used to reach the key through the title slug —
# lag order, cointegrating rank, T_eff, factor count, the SELECTED ARIMA/SARIMA/ARFIMA
# orders, the CSV column name, and the `--method`/`--probs`/`--kind`/`--threshold`
# option values — stay in the human-facing title and are kept out of the address an
# agent has to predict. None of these leaves loops over shocks or variables, so there
# are NO families here: every table is a singleton per invocation.
#
# Several tables are conditional (a flag or option selects the branch); all of them are
# declared, because the gate validates emitted ⊆ declared per leaf.

_vol_predict_tables(m::AbstractString) = [TableSpec(name=Symbol("$(m)_conditional_variance"),
    description="In-sample conditional variance and implied volatility, one row per period")]
_vol_resid_tables(m::AbstractString) = [TableSpec(name=Symbol("$(m)_standardized_residuals"),
    description="Standardized residuals, one row per period")]

const _PREDICT_TABLES = Dict{String,Vector{TableSpec}}(
    "var" => [TableSpec(name=:var_predictions,
        description="In-sample VAR fitted values, one column per variable")],
    "bvar" => [TableSpec(name=:bvar_predictions,
        description="In-sample BVAR fitted values at the posterior mean, one column per variable")],
    "arima" => [TableSpec(name=:arima_predictions,
        description="In-sample ARIMA fitted values, one row per period")],
    "vecm" => [TableSpec(name=:vecm_predictions,
        description="In-sample VECM fitted values (via the VAR representation), one column per variable")],
    "static" => [TableSpec(name=:static_factor_common_component,
        description="Common component of the static factor model, one column per observed series")],
    "dynamic" => [TableSpec(name=:dynamic_factor_common_component,
        description="Common component of the dynamic factor model, one column per observed series")],
    "gdfm" => [TableSpec(name=:gdfm_common_component,
        description="Common component of the generalized dynamic factor model, one column per series")],
    "arch" => _vol_predict_tables("arch"),
    "garch" => _vol_predict_tables("garch"),
    "egarch" => _vol_predict_tables("egarch"),
    "gjr-garch" => _vol_predict_tables("gjr_garch"),
    "sv" => [TableSpec(name=:sv_conditional_variance,
        description="Posterior-mean stochastic-volatility path (variance and volatility) per period")],
    "favar" => [TableSpec(name=:favar_predictions,
        description="In-sample FAVAR fitted values, one column per factor and observed variable")],
    "reg" => [TableSpec(name=:reg_fitted_values,
        description="OLS/WLS fitted values, one row per observation")],
    "logit" => [
        TableSpec(name=:logit_fitted_probabilities,
            description="Fitted success probabilities, one row per observation"),
        TableSpec(name=:average_marginal_effects_logit,
            description="Average marginal effects with SEs, z, p and CI (--marginal-effects)"),
        TableSpec(name=:odds_ratios_logit,
            description="Odds ratios with confidence bounds, one row per regressor (--odds-ratio)"),
        TableSpec(name=:classification_metrics,
            description="Accuracy/sensitivity/specificity at the chosen threshold (--classification-table)"),
        TableSpec(name=:confusion_matrix,
            description="Predicted-vs-actual counts at the chosen threshold (--classification-table)"),
    ],
    "probit" => [
        TableSpec(name=:probit_fitted_probabilities,
            description="Fitted success probabilities, one row per observation"),
        TableSpec(name=:average_marginal_effects_probit,
            description="Average marginal effects with SEs, z, p and CI (--marginal-effects)"),
        TableSpec(name=:classification_metrics,
            description="Accuracy/sensitivity/specificity at the chosen threshold (--classification-table)"),
        TableSpec(name=:confusion_matrix,
            description="Predicted-vs-actual counts at the chosen threshold (--classification-table)"),
    ],
    "preg" => [TableSpec(name=:panel_regression_fitted_values,
        description="Panel regression fitted values, one row per observation")],
    "piv" => [TableSpec(name=:panel_iv_fitted_values,
        description="Panel IV fitted values, one row per observation")],
    "plogit" => [TableSpec(name=:panel_logit_fitted_probabilities,
        description="Panel logit fitted probabilities, one row per observation")],
    "pprobit" => [TableSpec(name=:panel_probit_fitted_probabilities,
        description="Panel probit fitted probabilities, one row per observation")],
    "ologit" => [
        TableSpec(name=:ordered_logit_predicted_probabilities,
            description="Predicted probability of each ordered category, one row per observation"),
        TableSpec(name=:ordered_logit_average_marginal_effects,
            description="Average marginal effect per variable x category (--marginal-effects)"),
    ],
    "oprobit" => [
        TableSpec(name=:ordered_probit_predicted_probabilities,
            description="Predicted probability of each ordered category, one row per observation"),
        TableSpec(name=:ordered_probit_average_marginal_effects,
            description="Average marginal effect per variable x category (--marginal-effects)"),
    ],
    "mlogit" => [
        TableSpec(name=:multinomial_logit_predicted_probabilities,
            description="Predicted probability of each alternative, one row per observation"),
        TableSpec(name=:multinomial_logit_average_marginal_effects,
            description="Average marginal effect per variable x alternative (--marginal-effects)"),
    ],
    "sarima" => [TableSpec(name=:sarima_predictions,
        description="In-sample SARIMA fitted values, one row per period")],
    "poisson" => [TableSpec(name=:poisson_conditional_means,
        description="Fitted conditional means exp(x'b + offset), one row per observation")],
    "nbreg" => [TableSpec(name=:negative_binomial_conditional_means,
        description="Fitted conditional means exp(x'b + offset), one row per observation")],
    "ms-ar" => [TableSpec(name=:ms_ar_fitted_values,
        description="Regime-probability-weighted fitted values, one row per period")],
    "ms" => [TableSpec(name=:ms_regression_fitted_values,
        description="Regime-probability-weighted fitted values, one row per observation")],
    "statespace" => [TableSpec(name=:state_space_state_paths,
        description="Long state paths: one row per (period, state) with the filtered and/or smoothed level")],
    "sur" => [TableSpec(name=:sur_fitted_values_per_equation,
        description="Long per-equation fitted values: one row per (equation, observation)")],
    "3sls" => [TableSpec(name=Symbol("3sls_fitted_values_per_equation"),
        description="Long per-equation fitted values: one row per (equation, observation)")],
    "arfima" => [TableSpec(name=:arfima_predictions,
        description="In-sample ARFIMA fitted values, one row per period")],
    "igarch" => _vol_predict_tables("igarch"),
    "cgarch" => _vol_predict_tables("cgarch"),
    "aparch" => _vol_predict_tables("aparch"),
    "figarch" => _vol_predict_tables("figarch"),
    "fiegarch" => _vol_predict_tables("fiegarch"),
    "garch-midas" => _vol_predict_tables("garch_midas"),
)

const _RESIDUALS_TABLES = Dict{String,Vector{TableSpec}}(
    "var" => [TableSpec(name=:var_residuals,
        description="VAR residuals, one column per variable")],
    "bvar" => [TableSpec(name=:bvar_residuals,
        description="BVAR residuals at the posterior mean, one column per variable")],
    "arima" => [TableSpec(name=:arima_residuals,
        description="ARIMA residuals, one row per period")],
    "vecm" => [TableSpec(name=:vecm_residuals,
        description="VECM residuals (via the VAR representation), one column per variable")],
    "static" => [TableSpec(name=:static_factor_idiosyncratic_component,
        description="Idiosyncratic component of the static factor model, one column per series")],
    "dynamic" => [TableSpec(name=:dynamic_factor_idiosyncratic_component,
        description="Idiosyncratic component of the dynamic factor model, one column per series")],
    "gdfm" => [TableSpec(name=:gdfm_idiosyncratic_component,
        description="Idiosyncratic component of the generalized dynamic factor model, one column per series")],
    "arch" => _vol_resid_tables("arch"),
    "garch" => _vol_resid_tables("garch"),
    "egarch" => _vol_resid_tables("egarch"),
    "gjr-garch" => _vol_resid_tables("gjr_garch"),
    "sv" => _vol_resid_tables("sv"),
    "favar" => [TableSpec(name=:favar_residuals,
        description="FAVAR residuals, one column per factor and observed variable")],
    "reg" => [TableSpec(name=:reg_residuals,
        description="OLS/WLS residuals, one row per observation")],
    "logit" => [TableSpec(name=:logit_residuals,
        description="Response residuals y - p, one row per observation")],
    "probit" => [TableSpec(name=:probit_residuals,
        description="Response residuals y - p, one row per observation")],
    "preg" => [TableSpec(name=:panel_regression_residuals,
        description="Panel regression residuals, one row per observation")],
    "piv" => [TableSpec(name=:panel_iv_residuals,
        description="Panel IV residuals, one row per observation")],
    "plogit" => [TableSpec(name=:panel_logit_residuals,
        description="Panel logit residuals, one row per observation")],
    "pprobit" => [TableSpec(name=:panel_probit_residuals,
        description="Panel probit residuals, one row per observation")],
    # --kind picks a different residual DEFINITION but the same n x J shape, so the three
    # kinds share one key; --generalized is a different table (one length-n column).
    "ologit" => [
        TableSpec(name=:ordered_logit_residuals,
            description="Per-category residuals (response, pearson or deviance per --kind)"),
        TableSpec(name=:ordered_logit_generalized_residuals,
            description="Length-n Chesher-Irish score residual (--generalized)"),
    ],
    "oprobit" => [
        TableSpec(name=:ordered_probit_residuals,
            description="Per-category residuals (response, pearson or deviance per --kind)"),
        TableSpec(name=:ordered_probit_generalized_residuals,
            description="Length-n Chesher-Irish score residual (--generalized)"),
    ],
    "mlogit" => [TableSpec(name=:multinomial_logit_residuals,
        description="Per-alternative residuals (response, pearson or deviance per --kind)")],
    "setar" => [TableSpec(name=:setar_residuals,
        description="SETAR residuals, one row per effective period")],
    "star" => [TableSpec(name=:star_residuals,
        description="STAR residuals, one row per effective period")],
    "ms-ar" => [TableSpec(name=:ms_ar_residuals,
        description="MS-AR residuals (smoothed-probability weighted), one row per effective period")],
    "ms" => [TableSpec(name=:ms_regression_residuals,
        description="MS regression residuals (smoothed-probability weighted), one row per observation")],
    "sarima" => [TableSpec(name=:sarima_residuals,
        description="SARIMA residuals, one row per period")],
    "poisson" => [TableSpec(name=:poisson_residuals,
        description="Poisson residuals, one row per observation")],
    "nbreg" => [TableSpec(name=:negative_binomial_residuals,
        description="Negative binomial residuals, one row per observation")],
    # --standardized divides by sqrt(F_t); same columns, so one key covers both.
    "statespace" => [TableSpec(name=:state_space_innovations,
        description="Long one-step prediction errors: one row per (period, series), raw or standardized")],
    "sur" => [TableSpec(name=:sur_residuals_per_equation,
        description="Long per-equation residuals: one row per (equation, observation)")],
    "3sls" => [TableSpec(name=Symbol("3sls_residuals_per_equation"),
        description="Long per-equation residuals: one row per (equation, observation)")],
    "arfima" => [TableSpec(name=:arfima_residuals,
        description="ARFIMA residuals, one row per period")],
    "igarch" => _vol_resid_tables("igarch"),
    "cgarch" => _vol_resid_tables("cgarch"),
    "aparch" => _vol_resid_tables("aparch"),
    "figarch" => _vol_resid_tables("figarch"),
    "fiegarch" => _vol_resid_tables("fiegarch"),
    "garch-midas" => _vol_resid_tables("garch_midas"),
)

"""Registry table declarations for one fitted leaf. A missing entry is an internal
invariant failure (a leaf added without declaring its keys), not a user error."""
function _fitted_tables(verb::Symbol, name::String)
    d = verb === :predict ? _PREDICT_TABLES : _RESIDUALS_TABLES
    haskey(d, name) || error("no table declaration for `$verb $name` (add it to " *
                             "_PREDICT_TABLES/_RESIDUALS_TABLES in fitted.jl)")
    return copy(d[name])
end

# Extra option sets for discrete choice / special leaves
# The handlers take `marginal_effects`/`odds_ratio`/`classification_table` as `Bool`,
# so these MUST be flags. Declaring them as String options made the default `""`
# fail Bool conversion on every call ("non-boolean (String) used in boolean
# context" -> exit 1), breaking predict logit|probit|ologit|oprobit|mlogit
# outright; none had T3 coverage (#85).
const _LOGIT_EXTRA = [
    OptionSpec(name="threshold", type=Float64, default=0.5, description="Classification threshold"),
]
const _LOGIT_EXTRA_FLAGS = [
    FlagSpec(name="marginal-effects", description="Report average marginal effects"),
    FlagSpec(name="odds-ratio", description="Report odds ratios (logit)"),
    FlagSpec(name="classification-table", description="Report the classification table"),
]
# #85 removed `--marginal-effects`/`--category`/`--base-category` here because the
# handlers accepted none of them (MethodError → exit 1 on every call). W10/#131
# re-adds `--marginal-effects` WITH handler support, now that MEMs#550 (0.7.3)
# gives the ordered/multinomial models delta-method SEs. `--category`/
# `--base-category` stay out — upstream marginal_effects takes NO kwargs.
const _ORDERED_EXTRA = OptionSpec[]
const _MLOGIT_EXTRA = OptionSpec[]
const _CHOICE_ME_FLAGS = [
    FlagSpec(name="marginal-effects",
             description="Also report average marginal effects per category (delta-method SEs; no z/p — upstream reports none)"),
]

# W4/#87: MEMs 0.7.2 settled the ordered/multinomial residual question upstream
# (MEMs#507), which is what these three leaves were gated on. `residuals(m; kind=)`
# returns an n x J matrix (one column per category) for all three models.
const _CHOICE_RESID_EXTRA = [
    OptionSpec(name="kind", type=String, default="response",
               choices=["response", "pearson", "deviance"],
               description="Residual type: response (d-P, rows sum to zero) | pearson | deviance"),
]
# `generalized_residuals` is the length-n score residual (Chesher & Irish 1987) and
# upstream defines it for the ORDERED models ONLY. It is deliberately not advertised on
# mlogit: for an unordered response there is no meaningful length-n scalar residual, and
# upstream documents that the per-alternative `:response` residuals ARE the generalized
# ones. Advertising it there would be a flag whose handler call cannot succeed.
const _ORDERED_RESID_FLAGS = [
    FlagSpec(name="generalized",
             description="Length-n generalized (score) residual instead of the per-category matrix"),
]

"""Flags for a fitted leaf — `predict` exposes the discrete-choice reports, `residuals`
the ordered-model generalized residual."""
function _flags_for_kind(kind::Symbol, verb::Symbol)
    if verb === :residuals
        return (kind === :ologit || kind === :oprobit) ? _ORDERED_RESID_FLAGS : FlagSpec[]
    end
    verb === :predict || return FlagSpec[]
    kind === :logit && return _LOGIT_EXTRA_FLAGS
    kind === :probit && return filter(f -> f.name != "odds-ratio", _LOGIT_EXTRA_FLAGS)
    (kind === :ologit || kind === :oprobit || kind === :mlogit) && return _CHOICE_ME_FLAGS
    return FlagSpec[]
end

# kind => (handler_predict, handler_residuals, opts_builder_symbol)
const FITTED_MODEL_KINDS = [
    (; name="var",        pred=_predict_var,        res=_residuals_var,        kind=:var),
    (; name="bvar",       pred=_predict_bvar,       res=_residuals_bvar,       kind=:bvar),
    (; name="arima",      pred=_predict_arima,      res=_residuals_arima,      kind=:arima),
    (; name="vecm",       pred=_predict_vecm,       res=_residuals_vecm,       kind=:vecm),
    (; name="static",     pred=_predict_static,     res=_residuals_static,     kind=:factor_static),
    (; name="dynamic",    pred=_predict_dynamic,    res=_residuals_dynamic,    kind=:factor_dynamic),
    (; name="gdfm",       pred=_predict_gdfm,       res=_residuals_gdfm,       kind=:factor_gdfm),
    (; name="arch",       pred=_VOL_PREDICT_HANDLERS["arch"],       res=_VOL_RESIDUALS_HANDLERS["arch"],       kind=:vol),
    (; name="garch",      pred=_VOL_PREDICT_HANDLERS["garch"],      res=_VOL_RESIDUALS_HANDLERS["garch"],      kind=:vol),
    (; name="egarch",     pred=_VOL_PREDICT_HANDLERS["egarch"],     res=_VOL_RESIDUALS_HANDLERS["egarch"],     kind=:vol),
    # C044: kebab primary; snake alias applied in _specs_for_verb
    (; name="gjr-garch",  pred=_VOL_PREDICT_HANDLERS["gjr_garch"],  res=_VOL_RESIDUALS_HANDLERS["gjr_garch"],  kind=:vol),
    (; name="sv",         pred=_VOL_PREDICT_HANDLERS["sv"],         res=_VOL_RESIDUALS_HANDLERS["sv"],         kind=:vol),
    (; name="favar",      pred=_predict_favar,      res=_residuals_favar,      kind=:favar),
    (; name="reg",        pred=_predict_reg,        res=_residuals_reg,        kind=:reg),
    (; name="logit",      pred=_predict_logit,      res=_residuals_logit,      kind=:logit),
    (; name="probit",     pred=_predict_probit,     res=_residuals_probit,     kind=:probit),
    (; name="preg",       pred=_predict_preg,       res=_residuals_preg,       kind=:preg),
    (; name="piv",        pred=_predict_piv,        res=_residuals_piv,        kind=:preg),
    (; name="plogit",     pred=_predict_plogit,     res=_residuals_plogit,     kind=:preg),
    (; name="pprobit",    pred=_predict_pprobit,    res=_residuals_pprobit,    kind=:preg),
    (; name="ologit",     pred=_predict_ologit,     res=_residuals_ologit,     kind=:ologit),
    (; name="oprobit",    pred=_predict_oprobit,    res=_residuals_oprobit,    kind=:oprobit),
    (; name="mlogit",     pred=_predict_mlogit,     res=_residuals_mlogit,     kind=:mlogit),
]

function _opts_for_kind(kind::Symbol, verb::Symbol)
    if kind === :reg
        opts = copy(REG_OPTIONS)
        # weights for reg
        push!(opts, OptionSpec(name="weights", type=String, default="", description="Weights column"))
        return opts
    elseif kind === :preg
        return copy(PREG_OPTIONS)
    elseif kind === :vol || kind === :arima
        return [
            OptionSpec(name="column", short="c", type=Int, default=1, description="Column index"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :factor_static
        # #144: options MUST mirror the handler kwargs exactly (#85). The old shared
        # :factor block declared --lags, which NO factor handler accepts — the bound
        # default made every predict|residuals static|dynamic|gdfm invocation a
        # MethodError (exit 1), hidden by zero coverage on the whole sub-family.
        return [
            OptionSpec(name="nfactors", short="r", type=Int, default=nothing,
                       description="Number of factors (default: auto via IC)"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :factor_dynamic
        return [
            OptionSpec(name="nfactors", short="r", type=Int, default=nothing,
                       description="Number of factors (default: auto via IC)"),
            OptionSpec(name="factor-lags", short="p", type=Int, default=1,
                       description="Factor VAR lag order"),
            OptionSpec(name="method", type=String, default="twostep",
                       description="twostep|qml estimation method"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :factor_gdfm
        return [
            OptionSpec(name="nfactors", type=Int, default=nothing,
                       description="Number of static factors (unused when --dynamic-rank set)"),
            OptionSpec(name="dynamic-rank", short="q", type=Int, default=nothing,
                       description="Dynamic rank (default: auto)"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :logit
        return [REG_OPTIONS...; (verb === :predict ? _LOGIT_EXTRA : OptionSpec[])]
    elseif kind === :probit
        return [REG_OPTIONS...; (verb === :predict ? _LOGIT_EXTRA : OptionSpec[])]
    elseif kind === :ologit || kind === :oprobit
        return [REG_OPTIONS...; (verb === :predict ? _ORDERED_EXTRA : _CHOICE_RESID_EXTRA)]
    elseif kind === :mlogit
        return [REG_OPTIONS...; (verb === :predict ? _MLOGIT_EXTRA : _CHOICE_RESID_EXTRA)]
    elseif kind === :bvar
        return [
            OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
            OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws"),
            OptionSpec(name="sampler", type=String, default="direct", description="Sampler"),
            OptionSpec(name="config", type=String, default="", description="TOML prior config"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :favar
        return [
            OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors"),
            OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lags"),
            OptionSpec(name="key-vars", type=String, default="", description="Key variables"),
            OUTPUT_OPTIONS...,
        ]
    elseif kind === :vecm
        return [
            OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order"),
            OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank"),
            OUTPUT_OPTIONS...,
        ]
    else # :var default
        return [
            OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            OUTPUT_OPTIONS...,
        ]
    end
end

# C044 snake→kebab aliases for fitted leaves
const _FITTED_CLI_ALIASES = Dict("gjr-garch" => ["gjr_garch"])

function _specs_for_verb(verb::Symbol, title_prefix::String)
    specs = CommandSpec[]
    for m in FITTED_MODEL_KINDS
        handler = verb === :predict ? m.pred : m.res
        path0 = verb === :predict ? "predict" : "residuals"
        aliases = get(_FITTED_CLI_ALIASES, m.name, String[])
        push!(specs, CommandSpec(
            path=[path0, m.name],
            summary="$title_prefix ($(m.name))",
            args=_fitted_data_arg(),
            options=_opts_for_kind(m.kind, verb),
            flags=_flags_for_kind(m.kind, verb),
            tables=_fitted_tables(verb, m.name),
            category=path0,
            aliases=aliases,
            handler=handler,
        ))
    end
    return specs
end

function predict_specs()::Vector{CommandSpec}
    # The six C064a GARCH variants are NOT in VOL_MODELS (their option sets differ), so
    # _specs_for_verb cannot generate them — append their hand-written specs (#69).
    return vcat(_specs_for_verb(:predict, "In-sample fitted values"), CommandSpec[
        # W6/#108: SARIMA in-sample fitted values / residuals. Both come from the abstract
        # AbstractARIMAModel StatsAPI block, so no SARIMA-specific method is needed.
        CommandSpec(
            path=["predict", "sarima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[SARIMA_OPTIONS...,
                OUTPUT_OPTIONS...],
            flags=copy(SARIMA_FLAGS),
            tables=_fitted_tables(:predict, "sarima"),
            category="predict",
            handler=_predict_sarima,
        ),
        # W2/#107: count-data conditional means exp(x'b + offset). Upstream's 1-arg
        # `predict(m)` returns m.fitted; the (m, Xnew) out-of-sample form is out of scope
        # here, matching every other `predict` leaf. NO --plot: src/plotting/ has no
        # dispatch covering PoissonModel/NegBinModel on the 0.7.2 tag.
        CommandSpec(
            path=["predict", "poisson"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                OptionSpec(name="cov-type", type=String, default="robust",
                           choices=["robust", "mle", "hc0", "hc1", "hc2", "hc3", "cluster"],
                           description="robust (QMLE sandwich, default), mle, hc0-hc3, cluster"),
                OptionSpec(name="clusters", type=String, default="", description="Cluster variable column name"),
                OptionSpec(name="maxiter", type=Int, default=100, description="Maximum IRLS iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                OUTPUT_OPTIONS...],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "poisson"),
            category="predict",
            handler=_predict_poisson,
        ),
        CommandSpec(
            path=["predict", "nbreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                OptionSpec(name="maxiter", type=Int, default=1000, description="Maximum iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                OUTPUT_OPTIONS...],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "nbreg"),
            category="predict",
            handler=_predict_nbreg,
        ),
        # W3/#101: MS fitted values, un-gated by MEMs#510. `--probs` picks the regime
        # weighting; upstream warns `y - predict(m; probs=:filtered)` is NOT residuals(m)
        # (those are smoothed-weighted and use strictly more information), so these leaves
        # are not a restatement of `residuals ms|ms-ar`. Same OPPOSITE switching-variance
        # defaults as the residuals leaves — ms-ar FALSE, ms TRUE. Do not unify them.
        CommandSpec(
            path=["predict", "ms-ar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=1000, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="probs", type=String, default="smoothed", description="Regime weighting: smoothed or filtered", choices=["smoothed","filtered"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="switching-variance", description="Let σ² switch across regimes (default: off, Hamilton form)")],
            tables=_fitted_tables(:predict, "ms-ar"),
            category="predict",
            handler=_predict_ms_ar,
        ),
        CommandSpec(
            path=["predict", "ms"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="EM convergence tolerance (> 0)"),
                OptionSpec(name="probs", type=String, default="smoothed", description="Regime weighting: smoothed or filtered", choices=["smoothed","filtered"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-switching-variance", description="Force common σ² across regimes (default: σ² switches)")],
            tables=_fitted_tables(:predict, "ms"),
            category="predict",
            handler=_predict_ms,
        ),
        # #71: state-space state paths / innovations, read from the model's fields.
        CommandSpec(
            path=["predict", "statespace"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                # `--model` is reserved for the model HANDLE on predict/residuals (injected
                # by with_model_option, which does NOT dedupe), so the model TYPE is --kind
                # here — same value set as `estimate statespace --model`.
                OptionSpec(name="kind", type=String, default="local-level", description="State-space model", choices=["local-level","local-linear-trend"]),
                OptionSpec(name="init-mode", type=String, default="kappa", description="Diffuse initialisation", choices=["kappa","diffuse"]),
                OptionSpec(name="kappa", type=Float64, default=1e6, description="Large-kappa diffuse prior variance"),
                OptionSpec(name="state", type=String, default="both", description="Which state path to emit", choices=["filtered","smoothed","both"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "statespace"),
            category="predict",
            handler=_predict_statespace,
        ),
        # #68: SUR/3SLS carry PER-EQUATION fitted/residuals fields. Both mirror their
        # `estimate` sibling's options because the equation system lives in --config —
        # without it the model cannot be refit.
        CommandSpec(
            path=["predict", "sur"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML with [[equations]] blocks (required)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="iterate", description="Iterate the SUR feasible-GLS step to convergence"),
                   FlagSpec(name="no-intercept", description="Do not add an intercept to each equation")],
            tables=_fitted_tables(:predict, "sur"),
            category="predict",
            handler=_predict_sur,
        ),
        CommandSpec(
            path=["predict", "3sls"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML with [[equations]] and instruments (required)"),
                OptionSpec(name="instruments", type=String, default="common", description="Instrument mode", choices=["common","perequation"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-intercept", description="Do not add an intercept to each equation")],
            tables=_fitted_tables(:predict, "3sls"),
            category="predict",
            handler=_predict_3sls,
        ),
        # #73: ARFIMA mirrors `estimate arfima`'s full option set — the :arima kind in
        # FITTED_MODEL_KINDS supplies only --column, which would silently pin p=q=0.
        CommandSpec(
            path=["predict", "arfima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=0, description="AR order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="method", short="m", type=String, default="css", description="css|mle (fractional-integration estimator)", choices=["css","mle"]),
                OptionSpec(name="d0", type=Float64, default=nothing, description="Starting value for d (default: GPH pre-estimate)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "arfima"),
            category="predict",
            handler=_predict_arfima,
        ),
        CommandSpec(
            path=["predict", "igarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "igarch"),
            category="predict",
            handler=_predict_igarch,
        ),
        CommandSpec(
            path=["predict", "cgarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "cgarch"),
            category="predict",
            handler=_predict_cgarch,
        ),
        CommandSpec(
            path=["predict", "aparch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="fix-delta", type=Float64, default=nothing, description="Fix the power parameter delta"),
                OptionSpec(name="fix-gamma", type=Float64, default=nothing, description="Fix the asymmetry parameter gamma"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "aparch"),
            category="predict",
            handler=_predict_aparch,
        ),
        CommandSpec(
            path=["predict", "figarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "figarch"),
            category="predict",
            handler=_predict_figarch,
        ),
        CommandSpec(
            path=["predict", "fiegarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "fiegarch"),
            category="predict",
            handler=_predict_fiegarch,
        ),
        CommandSpec(
            path=["predict", "garch-midas"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="m-freq", type=Int, default=0, description="High-frequency observations per low-frequency block (required, ≥ 1)"),
                OptionSpec(name="k", type=Int, default=12, description="Number of MIDAS lags"),
                OptionSpec(name="rv", type=String, default="realized", description="Long-run driver", choices=["realized","macro"]),
                OptionSpec(name="span", type=String, default="fixed", description="Span", choices=["fixed","rolling"]),
                OptionSpec(name="config", type=String, default="", description="TOML with [garch_midas] x_lf (required for --rv macro)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:predict, "garch-midas"),
            category="predict",
            handler=_predict_garch_midas,
        ),
    ])
end

function residuals_specs()::Vector{CommandSpec}
    return vcat(_specs_for_verb(:residuals, "Model residuals"), CommandSpec[
        # #70 remainder: the nonlinear-TS models all define StatsAPI.residuals upstream
        # (ThresholdModel/STARModel/MSRegModel). NO matching `predict` — none of them EXPOSES
        # predict/fitted. For the MS models the fitted series is well defined (the
        # regime-probability-weighted conditional mean) and MEMs already computes it internally
        # to form these very residuals — it just doesn't store it. Filed as MEMs#510; adding a
        # leaf that recomputes it risks diverging from the definition upstream publishes.
        # Options mirror the `estimate` sibling MINUS the ones that drive only the attached
        # inference (SETAR's --reps/--ci-level/--het feed the Hansen bootstrap and threshold
        # CI, never the residuals); the refit passes linearity=false to skip that bootstrap.
        CommandSpec(
            path=["residuals", "setar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=String, default="1", description="Delay lag: an integer ≥ 1, or 'auto' (=1:p grid)"),
                OptionSpec(name="trim", type=Float64, default=0.15, description="Trimming fraction for the threshold grid (0 < trim < 0.5)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "setar"),
            category="residuals",
            handler=_residuals_setar,
        ),
        CommandSpec(
            path=["residuals", "star"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=Int, default=1, description="Delay lag for the self-exciting transition var (≥ 1)"),
                OptionSpec(name="type", type=String, default="auto", description="Transition shape: lstr1|lstr2|estr|auto", choices=["lstr1","lstr2","estr","auto"]),
                OptionSpec(name="n-gamma", type=Int, default=15, description="Grid points for the γ start values (≥ 2)"),
                OptionSpec(name="n-c", type=Int, default=15, description="Grid points for the c start values (≥ 2)"),
                OptionSpec(name="transition-col", type=Int, default=0, description="Column index of an external transition var s (0 = self-exciting y[t-d])"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "star"),
            category="residuals",
            handler=_residuals_star,
        ),
        CommandSpec(
            path=["residuals", "ms-ar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=1000, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            # NOTE the OPPOSITE default from `residuals ms` below — ms-ar's switching_variance
            # is FALSE upstream (Hamilton form), ms regression's is TRUE. Do not unify them.
            flags=[FlagSpec(name="switching-variance", description="Let σ² switch across regimes (default: off, Hamilton form)")],
            tables=_fitted_tables(:residuals, "ms-ar"),
            category="residuals",
            handler=_residuals_ms_ar,
        ),
        CommandSpec(
            path=["residuals", "ms"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="EM convergence tolerance (> 0)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-switching-variance", description="Force common σ² across regimes (default: σ² switches)")],
            tables=_fitted_tables(:residuals, "ms"),
            category="residuals",
            handler=_residuals_ms,
        ),
        # W6/#108: SARIMA residuals (abstract AbstractARIMAModel dispatch).
        CommandSpec(
            path=["residuals", "sarima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[SARIMA_OPTIONS...,
                OUTPUT_OPTIONS...],
            flags=copy(SARIMA_FLAGS),
            tables=_fitted_tables(:residuals, "sarima"),
            category="residuals",
            handler=_residuals_sarima,
        ),
        # W2/#107: count-data residuals. `residuals(m)` is a bare field accessor upstream
        # with NO `kind` kwarg, so no --kind is advertised here (a declared option the
        # handler cannot honour fails on every call — #85).
        CommandSpec(
            path=["residuals", "poisson"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                OptionSpec(name="cov-type", type=String, default="robust",
                           choices=["robust", "mle", "hc0", "hc1", "hc2", "hc3", "cluster"],
                           description="robust (QMLE sandwich, default), mle, hc0-hc3, cluster"),
                OptionSpec(name="clusters", type=String, default="", description="Cluster variable column name"),
                OptionSpec(name="maxiter", type=Int, default=100, description="Maximum IRLS iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                OUTPUT_OPTIONS...],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "poisson"),
            category="residuals",
            handler=_residuals_poisson,
        ),
        CommandSpec(
            path=["residuals", "nbreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                OptionSpec(name="maxiter", type=Int, default=1000, description="Maximum iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                OUTPUT_OPTIONS...],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "nbreg"),
            category="residuals",
            handler=_residuals_nbreg,
        ),
        # #71: state-space state paths / innovations, read from the model's fields.
        CommandSpec(
            path=["residuals", "statespace"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                # `--model` is reserved for the model HANDLE on predict/residuals (injected
                # by with_model_option, which does NOT dedupe), so the model TYPE is --kind
                # here — same value set as `estimate statespace --model`.
                OptionSpec(name="kind", type=String, default="local-level", description="State-space model", choices=["local-level","local-linear-trend"]),
                OptionSpec(name="init-mode", type=String, default="kappa", description="Diffuse initialisation", choices=["kappa","diffuse"]),
                OptionSpec(name="kappa", type=Float64, default=1e6, description="Large-kappa diffuse prior variance"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="standardized", description="Emit standardized innovations v_t/sqrt(F_t) instead of raw v_t")],
            tables=_fitted_tables(:residuals, "statespace"),
            category="residuals",
            handler=_residuals_statespace,
        ),
        # #68: SUR/3SLS carry PER-EQUATION fitted/residuals fields. Both mirror their
        # `estimate` sibling's options because the equation system lives in --config —
        # without it the model cannot be refit.
        CommandSpec(
            path=["residuals", "sur"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML with [[equations]] blocks (required)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="iterate", description="Iterate the SUR feasible-GLS step to convergence"),
                   FlagSpec(name="no-intercept", description="Do not add an intercept to each equation")],
            tables=_fitted_tables(:residuals, "sur"),
            category="residuals",
            handler=_residuals_sur,
        ),
        CommandSpec(
            path=["residuals", "3sls"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML with [[equations]] and instruments (required)"),
                OptionSpec(name="instruments", type=String, default="common", description="Instrument mode", choices=["common","perequation"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-intercept", description="Do not add an intercept to each equation")],
            tables=_fitted_tables(:residuals, "3sls"),
            category="residuals",
            handler=_residuals_3sls,
        ),
        # #73: ARFIMA mirrors `estimate arfima`'s full option set — the :arima kind in
        # FITTED_MODEL_KINDS supplies only --column, which would silently pin p=q=0.
        CommandSpec(
            path=["residuals", "arfima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=0, description="AR order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="method", short="m", type=String, default="css", description="css|mle (fractional-integration estimator)", choices=["css","mle"]),
                OptionSpec(name="d0", type=Float64, default=nothing, description="Starting value for d (default: GPH pre-estimate)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "arfima"),
            category="residuals",
            handler=_residuals_arfima,
        ),
        CommandSpec(
            path=["residuals", "igarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "igarch"),
            category="residuals",
            handler=_residuals_igarch,
        ),
        CommandSpec(
            path=["residuals", "cgarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "cgarch"),
            category="residuals",
            handler=_residuals_cgarch,
        ),
        CommandSpec(
            path=["residuals", "aparch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="fix-delta", type=Float64, default=nothing, description="Fix the power parameter delta"),
                OptionSpec(name="fix-gamma", type=Float64, default=nothing, description="Fix the asymmetry parameter gamma"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "aparch"),
            category="residuals",
            handler=_residuals_aparch,
        ),
        CommandSpec(
            path=["residuals", "figarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "figarch"),
            category="residuals",
            handler=_residuals_figarch,
        ),
        CommandSpec(
            path=["residuals", "fiegarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "fiegarch"),
            category="residuals",
            handler=_residuals_fiegarch,
        ),
        CommandSpec(
            path=["residuals", "garch-midas"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="m-freq", type=Int, default=0, description="High-frequency observations per low-frequency block (required, ≥ 1)"),
                OptionSpec(name="k", type=Int, default=12, description="Number of MIDAS lags"),
                OptionSpec(name="rv", type=String, default="realized", description="Long-run driver", choices=["realized","macro"]),
                OptionSpec(name="span", type=String, default="fixed", description="Span", choices=["fixed","rolling"]),
                OptionSpec(name="config", type=String, default="", description="TOML with [garch_midas] x_lf (required for --rv macro)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=_fitted_tables(:residuals, "garch-midas"),
            category="residuals",
            handler=_residuals_garch_midas,
        ),
    ])
end

function _overlay_estimator_data_kinds(specs::Vector{CommandSpec})
    out = CommandSpec[]
    for s in specs
        push!(out, _copy_spec(s; data_kinds=_data_kinds_for_estimator(s.path[end])))
    end
    return with_default_csv_kinds(out)
end

const _FITTED_MODEL_TYPES = Dict{String,Vector{Symbol}}(
    "var" => [:VARModel], "bvar" => [:BVARPosterior],
    "arima" => [:ARIMAModel, :ARMAModel, :ARModel, :MAModel],
    "vecm" => [:VECMModel], "static" => [:FactorModel],
    "dynamic" => [:DynamicFactorModel], "gdfm" => [:GeneralizedDynamicFactorModel],
    "arch" => [:ARCHModel], "garch" => [:GARCHModel], "egarch" => [:EGARCHModel],
    "gjr-garch" => [:GJRGARCHModel], "sv" => [:SVModel], "favar" => [:FAVARModel],
    "reg" => [:RegModel], "logit" => [:LogitModel], "probit" => [:ProbitModel],
    "preg" => [:PanelRegModel], "piv" => [:PanelIVModel],
    "plogit" => [:PanelLogitModel], "pprobit" => [:PanelProbitModel],
    "ologit" => [:OrderedLogitModel], "oprobit" => [:OrderedProbitModel],
    "mlogit" => [:MultinomialLogitModel], "sarima" => [:SARIMAModel],
    "poisson" => [:PoissonModel], "nbreg" => [:NegBinModel],
    "ms-ar" => [:MSRegModel], "ms" => [:MSRegModel],
    "statespace" => [:StateSpaceModel], "sur" => [:SURModel],
    "3sls" => [:ThreeSLSModel], "arfima" => [:ARFIMAModel],
    "igarch" => [:IGARCHModel], "cgarch" => [:CGARCHModel],
    "aparch" => [:APARCHModel], "figarch" => [:FIGARCHModel],
    "fiegarch" => [:FIEGARCHModel], "garch-midas" => [:GarchMidasModel],
    "setar" => [:ThresholdModel], "star" => [:STARModel],
)

function _wrap_fitted_specs(specs::Vector{CommandSpec})
    out = CommandSpec[]
    for s in specs
        leaf = join(s.path, " ")
        key = isempty(s.tables) ? "" : string(s.tables[1].name)
        mt = get(_FITTED_MODEL_TYPES, s.path[end], Symbol[])
        h = wrap_legacy(_with_result(s.handler, leaf; key=key))
        push!(out, _copy_spec(s; handler=h, model_types=mt))
    end
    return out
end

function register_predict_commands!()
    specs = _overlay_estimator_data_kinds(
        with_result_handles(with_config_ergonomics(with_model_option(
            _wrap_fitted_specs(predict_specs())))))
    register!(specs)
    return build_node("predict", specs; description="In-sample fitted values / predictions")
end

function register_residuals_commands!()
    specs = _overlay_estimator_data_kinds(
        with_result_handles(with_config_ergonomics(with_model_option(
            _wrap_fitted_specs(residuals_specs())))))
    register!(specs)
    return build_node("residuals", specs; description="Model residuals")
end
