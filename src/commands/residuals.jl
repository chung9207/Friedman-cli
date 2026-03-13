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

# Residuals commands: model residuals for var, bvar, arima, vecm,
#                     static, dynamic, gdfm, arch, garch, egarch, gjr_garch, sv

function register_residuals_commands!()
    res_var = LeafCommand("var", _residuals_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VAR model")

    res_bvar = LeafCommand("bvar", _residuals_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from Bayesian VAR (posterior mean)")

    res_arima = LeafCommand("arima", _residuals_arima;
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
        description="Residuals from ARIMA model")

    res_vecm = LeafCommand("vecm", _residuals_vecm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VECM (via VAR representation)")

    res_static = LeafCommand("static", _residuals_static;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from static factor model (X - FΛ')")

    res_dynamic = LeafCommand("dynamic", _residuals_dynamic;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from dynamic factor model (X - FΛ')")

    res_gdfm = LeafCommand("gdfm", _residuals_gdfm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from generalized dynamic factor model")

    res_arch = LeafCommand("arch", _residuals_arch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from ARCH model")

    res_garch = LeafCommand("garch", _residuals_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from GARCH model")

    res_egarch = LeafCommand("egarch", _residuals_egarch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="EGARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from EGARCH model")

    res_gjr_garch = LeafCommand("gjr_garch", _residuals_gjr_garch;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GJR-GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from GJR-GARCH model")

    res_sv = LeafCommand("sv", _residuals_sv;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from stochastic volatility model")

    res_favar = LeafCommand("favar", _residuals_favar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("factors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("lags"; short="p", type=Int, default=2, description="VAR lag order"),
            Option("key-vars"; type=String, default="", description="Key variable names or indices (comma-separated)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="FAVAR model residuals")

    res_reg = LeafCommand("reg", _residuals_reg;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("weights"; type=String, default="", description="Weight column name (WLS)"),
        ],
        description="OLS/WLS model residuals")

    res_logit = LeafCommand("logit", _residuals_logit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Logit model residuals (deviance residuals)")

    res_probit = LeafCommand("probit", _residuals_probit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Probit model residuals (deviance residuals)")

    res_preg = LeafCommand("preg", _residuals_preg;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="Residuals from panel regression")

    res_piv = LeafCommand("piv", _residuals_piv;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[Option("dep"; type=String, default="", description="Dependent variable"),
                 Option("exog"; type=String, default="", description="Exogenous regressors (comma-separated)"),
                 Option("endog"; type=String, default="", description="Endogenous regressors (comma-separated)"),
                 Option("instruments"; type=String, default="", description="Instruments (comma-separated)"),
                 _PREG_COMMON_OPTIONS[3:4]..., _PREG_COMMON_OPTIONS[5:6]...,
                 _PREG_COMMON_OPTIONS[7:8]...],
        description="Residuals from panel IV regression")

    res_plogit = LeafCommand("plogit", _residuals_plogit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="Residuals from panel logit")

    res_pprobit = LeafCommand("pprobit", _residuals_pprobit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[3:4]...,
                 _PREG_COMMON_OPTIONS[5:6]..., _PREG_COMMON_OPTIONS[7:8]...],
        description="Residuals from panel probit")

    res_ologit = LeafCommand("ologit", _residuals_ologit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Residuals from ordered logit")

    res_oprobit = LeafCommand("oprobit", _residuals_oprobit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Residuals from ordered probit")

    res_mlogit = LeafCommand("mlogit", _residuals_mlogit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Residuals from multinomial logit")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"       => res_var,
        "bvar"      => res_bvar,
        "arima"     => res_arima,
        "vecm"      => res_vecm,
        "static"    => res_static,
        "dynamic"   => res_dynamic,
        "gdfm"      => res_gdfm,
        "arch"      => res_arch,
        "garch"     => res_garch,
        "egarch"    => res_egarch,
        "gjr_garch" => res_gjr_garch,
        "sv"        => res_sv,
        "favar"     => res_favar,
        "reg"       => res_reg,
        "logit"     => res_logit,
        "probit"    => res_probit,
        "preg"      => res_preg,
        "piv"       => res_piv,
        "plogit"    => res_plogit,
        "pprobit"   => res_pprobit,
        "ologit"    => res_ologit,
        "oprobit"   => res_oprobit,
        "mlogit"    => res_mlogit,
    )
    return NodeCommand("residuals", subcmds, "Model residuals")
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
    n = size(Y, 2)

    println("Computing VAR($p) residuals: $(length(varnames)) variables")
    println()

    resid = residuals(model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="VAR($p) Residuals (T_eff=$T_eff)")
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

    println("Computing BVAR($p) residuals (posterior mean)")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    var_model = posterior_mean_model(post; data=Y)
    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="BVAR($p) Residuals (posterior mean, T_eff=$T_eff)")
end

# ── ARIMA Residuals ─────────────────────────────────────

function _residuals_arima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                            method::String="css_mle", auto::Bool=false,
                            output::String="", format::String="table",
                            model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        method_sym = Symbol(method)
        safe_method = method_sym == :css_mle ? :mle : method_sym

        model = if isnothing(p) || auto
            println("Auto ARIMA residuals: variable=$vname, observations=$(length(y))")
            println()
            m = auto_arima(y; method=safe_method)
            label = _model_label(ar_order(m), diff_order(m), ma_order(m))
            printstyled("Selected model: $label\n"; bold=true)
            println()
            m
        else
            label = _model_label(p, d, q)
            println("$label residuals: variable=$vname")
            println()
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
                  title="$label Residuals for $vname")
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
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    println("Computing VECM residuals: rank=$r, lags=$p")
    println()

    var_model = to_var(vecm)
    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="VECM Residuals (rank=$r, T_eff=$T_eff)")
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
    end
    resid = residuals(fm)
    T = size(resid, 1)

    println("Static factor model residuals: $r factors, idiosyncratic component (T=$T)")
    println()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="Static Factor Idiosyncratic Component ($r factors, T=$T)")
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
        varnames = fm.varnames
    end
    resid = residuals(fm)
    T = size(resid, 1)

    println("Dynamic factor model residuals: $r factors, p=$factor_lags, idiosyncratic component (T=$T)")
    println()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Idiosyncratic Component ($r factors, p=$factor_lags, T=$T)")
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
        varnames = gm.varnames
    end
    resid = residuals(gm)
    T = size(resid, 1)

    println("GDFM residuals: q=$q dynamic factors, idiosyncratic component (T=$T)")
    println()

    res_df = DataFrame()
    res_df.t = 1:T
    for (vi, vname) in enumerate(varnames)
        res_df[!, vname] = resid[:, vi]
    end

    output_result(res_df; format=Symbol(format), output=output,
                  title="GDFM Idiosyncratic Component (q=$q, T=$T)")
end

# ── ARCH Residuals ────────────────────────────────────

function _residuals_arch(; data::String="", column::Int=1, q::Int=1,
                           output::String="", format::String="table",
                           model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_arch(y, q)
    else
        vname = "series"
    end
    resid = residuals(model)

    println("ARCH($q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="ARCH($q) Standardized Residuals ($vname)")
end

# ── GARCH Residuals ───────────────────────────────────

function _residuals_garch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                            output::String="", format::String="table",
                            model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_garch(y, p, q)
    else
        vname = "series"
    end
    resid = residuals(model)

    println("GARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="GARCH($p,$q) Standardized Residuals ($vname)")
end

# ── EGARCH Residuals ──────────────────────────────────

function _residuals_egarch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                             output::String="", format::String="table",
                             model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_egarch(y, p, q)
    else
        vname = "series"
    end
    resid = residuals(model)

    println("EGARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="EGARCH($p,$q) Standardized Residuals ($vname)")
end

# ── GJR-GARCH Residuals ──────────────────────────────

function _residuals_gjr_garch(; data::String="", column::Int=1, p::Int=1, q::Int=1,
                                output::String="", format::String="table",
                                model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_gjr_garch(y, p, q)
    else
        vname = "series"
    end
    resid = residuals(model)

    println("GJR-GARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="GJR-GARCH($p,$q) Standardized Residuals ($vname)")
end

# ── SV Residuals ──────────────────────────────────────

function _residuals_sv(; data::String="", column::Int=1, draws::Int=5000,
                         output::String="", format::String="table",
                         model=nothing)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        model = estimate_sv(y; n_samples=draws)
    else
        vname = "series"
    end
    resid = residuals(model)

    println("SV standardized residuals: variable=$vname, draws=$draws")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="SV Standardized Residuals ($vname)")
end

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

    println("FAVAR Residuals")
    println()

    resid = residuals(var_model)
    T_eff = size(resid, 1)

    res_df = DataFrame()
    res_df.t = 1:T_eff
    for v in 1:size(resid, 2)
        vname = v <= length(favar.varnames) ? favar.varnames[v] : "var_$v"
        res_df[!, vname] = round.(resid[:, v]; digits=6)
    end
    output_result(res_df; format=Symbol(format), output=output,
                  title="FAVAR Residuals (T_eff=$T_eff)")
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
    println("$wls_tag Residuals: $dep_name ~ $(join(xcols, " + "))")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output, title="$wls_tag Residuals")
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
    println("Logit Residuals: $dep_name ~ $(join(xcols, " + "))")
    println()

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
    println("Probit Residuals: $dep_name ~ $(join(xcols, " + "))")
    println()

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

    println("Panel Regression Residuals ($method): $dep")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Regression Residuals ($method)")
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

    println("Panel IV Residuals ($method): $dep")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel IV Residuals ($method)")
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

    println("Panel Logit Residuals ($method): $dep")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Logit Residuals ($method)")
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

    println("Panel Probit Residuals ($method): $dep")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Panel Probit Residuals ($method)")
end

# ── Ordered/Multinomial Residuals ───────────────────────

function _residuals_ologit(; data::String="", dep::String="", cov_type::String="hc1",
                            clusters::String="",
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    println("Ordered Logit Residuals: $dep_name")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Ordered Logit Residuals")
end

function _residuals_oprobit(; data::String="", dep::String="", cov_type::String="hc1",
                             clusters::String="",
                             output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    println("Ordered Probit Residuals: $dep_name")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Ordered Probit Residuals")
end

function _residuals_mlogit(; data::String="", dep::String="", cov_type::String="ols",
                            clusters::String="",
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols)

    println("Multinomial Logit Residuals: $dep_name")
    println()

    resid = residuals(model)
    res_df = DataFrame(observation=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="Multinomial Logit Residuals")
end
