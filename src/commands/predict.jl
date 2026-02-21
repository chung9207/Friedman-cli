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
    )
    return NodeCommand("predict", subcmds, "In-sample predictions (fitted values)")
end

# ── VAR Predict ─────────────────────────────────────────

function _predict_var(; data::String, lags=nothing,
                       output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
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

function _predict_bvar(; data::String, lags::Int=4, draws::Int=2000,
                        sampler::String="direct", config::String="",
                        output::String="", format::String="table")
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

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

function _predict_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          method::String="css_mle", auto::Bool=false,
                          output::String="", format::String="table")
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

function _predict_vecm(; data::String, lags::Int=2, rank::String="auto",
                         deterministic::String="constant",
                         output::String="", format::String="table")
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
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

function _predict_static(; data::String, nfactors=nothing,
                           output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        ic = ic_criteria(X, min(20, size(X, 2)))
        ic.r_IC1
    else
        nfactors
    end

    fm = estimate_factors(X, r)
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

function _predict_dynamic(; data::String, nfactors=nothing, factor_lags::Int=1,
                            method::String="twostep",
                            output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        ic = ic_criteria(X, min(10, size(X, 2)))
        ic.r_IC1
    else
        nfactors
    end

    fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
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

function _predict_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                         output::String="", format::String="table")
    X, varnames = load_multivariate_data(data)

    q = if isnothing(dynamic_rank)
        ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
        ic.q_ratio
    else
        dynamic_rank
    end

    gm = estimate_gdfm(X, q)
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

function _predict_arch(; data::String, column::Int=1, q::Int=1,
                         output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    model = estimate_arch(y, q)
    cond_var = predict(model)

    println("ARCH($q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="ARCH($q) Conditional Variance ($vname)")
end

# ── GARCH Predict ─────────────────────────────────────

function _predict_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                          output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    model = estimate_garch(y, p, q)
    cond_var = predict(model)

    println("GARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="GARCH($p,$q) Conditional Variance ($vname)")
end

# ── EGARCH Predict ────────────────────────────────────

function _predict_egarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                           output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    model = estimate_egarch(y, p, q)
    cond_var = predict(model)

    println("EGARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="EGARCH($p,$q) Conditional Variance ($vname)")
end

# ── GJR-GARCH Predict ────────────────────────────────

function _predict_gjr_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                              output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    model = estimate_gjr_garch(y, p, q)
    cond_var = predict(model)

    println("GJR-GARCH($p,$q) conditional variance: variable=$vname")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="GJR-GARCH($p,$q) Conditional Variance ($vname)")
end

# ── SV Predict ────────────────────────────────────────

function _predict_sv(; data::String, column::Int=1, draws::Int=5000,
                       output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    model = estimate_sv(y; n_samples=draws)
    cond_var = predict(model)

    println("SV posterior mean volatility: variable=$vname, draws=$draws")
    println()

    pred_df = DataFrame(t=1:length(cond_var), variance=round.(cond_var; digits=6),
                        volatility=round.(sqrt.(cond_var); digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="SV Posterior Mean Volatility ($vname)")
end
