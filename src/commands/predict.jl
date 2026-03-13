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

# Predict commands: in-sample fitted values for var, bvar, arima, vecm,
#                   static, dynamic, gdfm, arch, garch, egarch, gjr_garch, sv

function register_predict_commands!()
    pred_var = LeafCommand("var", _predict_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="In-sample fitted values from VAR model")

    pred_bvar = LeafCommand("bvar", _predict_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="In-sample fitted values from Bayesian VAR (posterior mean)")

    pred_arima = LeafCommand("arima", _predict_arima;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto selection)"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        flags=[Flag("auto"; short="a", description="Use auto ARIMA selection")],
        description="In-sample fitted values from ARIMA model")

    pred_vecm = LeafCommand("vecm", _predict_vecm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="In-sample fitted values from VECM (via VAR representation)")

    pred_static = LeafCommand("static", _predict_static;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Common component from static factor model (F * Λ')")

    pred_dynamic = LeafCommand("dynamic", _predict_dynamic;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Common component from dynamic factor model (F * Λ')")

    pred_gdfm = LeafCommand("gdfm", _predict_gdfm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Common component from generalized dynamic factor model")

    pred_arch = LeafCommand("arch", _predict_arch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Conditional variance from ARCH model")

    pred_garch = LeafCommand("garch", _predict_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Conditional variance from GARCH model")

    pred_egarch = LeafCommand("egarch", _predict_egarch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="EGARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Conditional variance from EGARCH model")

    pred_gjr_garch = LeafCommand("gjr_garch", _predict_gjr_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GJR-GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Conditional variance from GJR-GARCH model")

    pred_sv = LeafCommand("sv", _predict_sv;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Posterior mean volatility from stochastic volatility model")

    pred_favar = LeafCommand("favar", _predict_favar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("factors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("lags"; short="p", type=Int, default=2, description="VAR lag order"),
            Option("key-vars"; type=String, default="", description="Key variable names or indices (comma-separated)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="FAVAR in-sample fitted values")

    pred_reg = LeafCommand("reg", _predict_reg;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("weights"; type=String, default="", description="Weight column name (WLS)"),
        ],
        description="OLS/WLS in-sample fitted values")

    pred_logit = LeafCommand("logit", _predict_logit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("threshold"; type=Float64, default=0.5, description="Classification threshold"),
        ],
        flags=[
            Flag("marginal-effects"; description="Output marginal effects instead of fitted values"),
            Flag("odds-ratio"; description="Output odds ratios (logit only)"),
            Flag("classification-table"; description="Output classification table"),
        ],
        description="Logit in-sample fitted probabilities")

    pred_probit = LeafCommand("probit", _predict_probit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("threshold"; type=Float64, default=0.5, description="Classification threshold"),
        ],
        flags=[
            Flag("marginal-effects"; description="Output marginal effects instead of fitted values"),
            Flag("classification-table"; description="Output classification table"),
        ],
        description="Probit in-sample fitted probabilities")

    pred_preg = LeafCommand("preg", _predict_preg;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="In-sample fitted values from panel regression")

    pred_piv = LeafCommand("piv", _predict_piv;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[Option("dep"; type=String, default="", description="Dependent variable"),
                 Option("exog"; type=String, default="", description="Exogenous regressors (comma-separated)"),
                 Option("endog"; type=String, default="", description="Endogenous regressors (comma-separated)"),
                 Option("instruments"; type=String, default="", description="Instruments (comma-separated)"),
                 _PREG_COMMON_OPTIONS[3:4]..., _PREG_COMMON_OPTIONS[5:6]...,
                 _PREG_COMMON_OPTIONS[7:8]...],
        description="In-sample fitted values from panel IV regression")

    pred_plogit = LeafCommand("plogit", _predict_plogit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="In-sample fitted probabilities from panel logit")

    pred_pprobit = LeafCommand("pprobit", _predict_pprobit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="In-sample fitted probabilities from panel probit")

    pred_ologit = LeafCommand("ologit", _predict_ologit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Predicted probabilities from ordered logit")

    pred_oprobit = LeafCommand("oprobit", _predict_oprobit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Predicted probabilities from ordered probit")

    pred_mlogit = LeafCommand("mlogit", _predict_mlogit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Predicted probabilities from multinomial logit")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"       => pred_var,
        "bvar"      => pred_bvar,
        "arima"     => pred_arima,
        "vecm"      => pred_vecm,
        "static"    => pred_static,
        "dynamic"   => pred_dynamic,
        "gdfm"      => pred_gdfm,
        "arch"      => pred_arch,
        "garch"     => pred_garch,
        "egarch"    => pred_egarch,
        "gjr_garch" => pred_gjr_garch,
        "sv"        => pred_sv,
        "favar"     => pred_favar,
        "reg"       => pred_reg,
        "logit"     => pred_logit,
        "probit"    => pred_probit,
        "preg"      => pred_preg,
        "piv"       => pred_piv,
        "plogit"    => pred_plogit,
        "pprobit"   => pred_pprobit,
        "ologit"    => pred_ologit,
        "oprobit"   => pred_oprobit,
        "mlogit"    => pred_mlogit,
    )
    return NodeCommand("predict", subcmds, "In-sample predictions (fitted values)")
end

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
    n = size(Y, 2)

    println("Computing VAR($p) in-sample predictions: $(length(varnames)) variables")
    println()

    fitted = predict(model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="VAR($p) In-Sample Predictions (T_eff=$T_eff)")
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

    println("Computing BVAR($p) in-sample predictions (posterior mean)")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    var_model = posterior_mean_model(post; data=Y)
    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="BVAR($p) In-Sample Predictions (posterior mean, T_eff=$T_eff)")
end

# ── ARIMA Predict ───────────────────────────────────────

function _predict_arima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          method::String="css_mle", auto::Bool=false,
                          output::String="", format::String="table",
                          model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        method_sym = Symbol(method)
        safe_method = method_sym == :css_mle ? :mle : method_sym

        model = if isnothing(p) || auto
            println("Auto ARIMA predict: variable=$vname, observations=$(length(y))")
            println()
            m = auto_arima(y; method=safe_method)
            label = _model_label(ar_order(m), diff_order(m), ma_order(m))
            printstyled("Selected model: $label\n"; bold=true)
            println()
            m
        else
            label = _model_label(p, d, q)
            println("$label predict: variable=$vname")
            println()
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
                  title="$label In-Sample Predictions for $vname")
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
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    println("Computing VECM in-sample predictions: rank=$r, lags=$p")
    println()

    var_model = to_var(vecm)
    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="VECM In-Sample Predictions (rank=$r, T_eff=$T_eff)")
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
    end
    fitted = predict(fm)
    T = size(fitted, 1)

    println("Static factor model: $r factors, common component (T=$T)")
    println()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="Static Factor Common Component ($r factors, T=$T)")
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
        varnames = fm.varnames
    end
    fitted = predict(fm)
    T = size(fitted, 1)

    println("Dynamic factor model: $r factors, p=$factor_lags, common component (T=$T)")
    println()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Common Component ($r factors, p=$factor_lags, T=$T)")
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
        varnames = gm.varnames
    end
    fitted = predict(gm)
    T = size(fitted, 1)

    println("GDFM: q=$q dynamic factors, common component (T=$T)")
    println()

    pred_df = DataFrame()
    pred_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        pred_df[!, vname] = fitted[:, vi]
    end

    output_result(pred_df; format=Symbol(format), output=output,
                  title="GDFM Common Component (q=$q, T=$T)")
end

# ── ARCH Predict ──────────────────────────────────────

function _predict_arch(; data::String="", column::Int=1, q::Int=1,
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_arch(y, q)
    else
        vname = "series"
    end
    cond_var = predict(model)

    println("ARCH($q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="ARCH($q) Conditional Variance ($vname)")
end

# ── GARCH Predict ─────────────────────────────────────

function _predict_garch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                          output::String="", format::String="table",
                          model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_garch(y, p, q)
    else
        vname = "series"
    end
    cond_var = predict(model)

    println("GARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="GARCH($p,$q) Conditional Variance ($vname)")
end

# ── EGARCH Predict ────────────────────────────────────

function _predict_egarch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_egarch(y, p, q)
    else
        vname = "series"
    end
    cond_var = predict(model)

    println("EGARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="EGARCH($p,$q) Conditional Variance ($vname)")
end

# ── GJR-GARCH Predict ────────────────────────────────

function _predict_gjr_garch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                              output::String="", format::String="table",
                              model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_gjr_garch(y, p, q)
    else
        vname = "series"
    end
    cond_var = predict(model)

    println("GJR-GARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="GJR-GARCH($p,$q) Conditional Variance ($vname)")
end

# ── SV Predict ────────────────────────────────────────

function _predict_sv(; data::String="", column::Int=1, draws::Int=5000,
                       output::String="", format::String="table",
                       model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_sv(y; n_samples=draws)
    else
        vname = "series"
    end
    cond_var = predict(model)

    println("SV posterior mean volatility: variable=$vname, draws=$draws")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="SV Posterior Mean Volatility ($vname)")
end

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

    println("FAVAR In-Sample Prediction")
    println()

    fitted = predict(var_model)
    T_eff = size(fitted, 1)

    pred_df = DataFrame()
    pred_df.t = 1:T_eff
    for v in 1:size(fitted, 2)
        vname = v <= length(favar.varnames) ? favar.varnames[v] : "var_$v"
        pred_df[!, vname] = round.(fitted[:, v]; digits=6)
    end
    output_result(pred_df; format=Symbol(format), output=output,
                  title="FAVAR In-Sample Predictions (T_eff=$T_eff)")
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
    println("$wls_tag Fitted Values: $dep_name ~ $(join(xcols, " + "))")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output, title="$wls_tag Fitted Values")
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
        or_df = DataFrame(Variable=or.varnames, Odds_Ratio=round.(or.odds_ratio; digits=6),
            CI_Lower=round.(or.ci_lower; digits=6), CI_Upper=round.(or.ci_upper; digits=6))
        output_result(or_df; format=Symbol(format), output=output,
                      title="Odds Ratios (Logit)")
    elseif classification_table
        ct = MacroEconometricModels.classification_table(model; threshold=threshold)
        println("Classification Table (threshold=$threshold):")
        for (k, v) in sort(collect(ct))
            println("  $k: $v")
        end
    else
        println("Logit Fitted Probabilities: $dep_name")
        println()
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
        println("Classification Table (threshold=$threshold):")
        for (k, v) in sort(collect(ct))
            println("  $k: $v")
        end
    else
        println("Probit Fitted Probabilities: $dep_name")
        println()
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

    println("Panel Regression Fitted Values ($method): $dep")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Regression Fitted Values ($method)")
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

    println("Panel IV Fitted Values ($method): $dep")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel IV Fitted Values ($method)")
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

    println("Panel Logit Fitted Probabilities ($method): $dep")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Logit Fitted Probabilities ($method)")
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

    println("Panel Probit Fitted Probabilities ($method): $dep")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Panel Probit Fitted Probabilities ($method)")
end

# ── Ordered/Multinomial Predict ─────────────────────────

function _predict_ologit(; data::String="", dep::String="", cov_type::String="hc1",
                          clusters::String="",
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    println("Ordered Logit Predicted Probabilities: $dep_name")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Ordered Logit Predicted Probabilities")
end

function _predict_oprobit(; data::String="", dep::String="", cov_type::String="hc1",
                           clusters::String="",
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    println("Ordered Probit Predicted Probabilities: $dep_name")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Ordered Probit Predicted Probabilities")
end

function _predict_mlogit(; data::String="", dep::String="", cov_type::String="ols",
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols)

    println("Multinomial Logit Predicted Probabilities: $dep_name")
    println()

    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="Multinomial Logit Predicted Probabilities")
end
