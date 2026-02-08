# Residuals commands: model residuals for var, bvar, arima, vecm,
#                     static, dynamic, gdfm, arch, garch, egarch, gjr_garch, sv

function register_residuals_commands!()
    res_var = LeafCommand("var", _residuals_var;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VAR model")

    res_bvar = LeafCommand("bvar", _residuals_bvar;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from Bayesian VAR (posterior mean)")

    res_arima = LeafCommand("arima", _residuals_arima;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto selection)"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        flags=[Flag("auto"; short="a", description="Use auto ARIMA selection")],
        description="Residuals from ARIMA model")

    res_vecm = LeafCommand("vecm", _residuals_vecm;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VECM (via VAR representation)")

    res_static = LeafCommand("static", _residuals_static;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from static factor model (X - FΛ')")

    res_dynamic = LeafCommand("dynamic", _residuals_dynamic;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from dynamic factor model (X - FΛ')")

    res_gdfm = LeafCommand("gdfm", _residuals_gdfm;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Idiosyncratic component from generalized dynamic factor model")

    res_arch = LeafCommand("arch", _residuals_arch;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from ARCH model")

    res_garch = LeafCommand("garch", _residuals_garch;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from GARCH model")

    res_egarch = LeafCommand("egarch", _residuals_egarch;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="EGARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from EGARCH model")

    res_gjr_garch = LeafCommand("gjr_garch", _residuals_gjr_garch;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="GJR-GARCH order"),
            Option("q"; type=Int, default=1, description="ARCH order"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from GJR-GARCH model")

    res_sv = LeafCommand("sv", _residuals_sv;
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Standardized residuals from stochastic volatility model")

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
    )
    return NodeCommand("residuals", subcmds, "Model residuals")
end

# ── VAR Residuals ───────────────────────────────────────

function _residuals_var(; data::String, lags=nothing, from_tag::String="",
                          output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "var", "lags" => p, "n_vars" => n),
        Dict{String,Any}("command" => "residuals var", "data" => data))
end

# ── BVAR Residuals ──────────────────────────────────────

function _residuals_bvar(; data::String, lags::Int=4, draws::Int=2000,
                           sampler::String="direct", config::String="",
                           from_tag::String="",
                           output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)

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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "bvar", "lags" => p, "n_vars" => n),
        Dict{String,Any}("command" => "residuals bvar", "data" => data))
end

# ── ARIMA Residuals ─────────────────────────────────────

function _residuals_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                            method::String="css_mle", auto::Bool=false,
                            from_tag::String="",
                            output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "arima", "model" => label),
        Dict{String,Any}("command" => "residuals arima", "data" => data))
end

# ── VECM Residuals ─────────────────────────────────────

function _residuals_vecm(; data::String, lags::Int=2, rank::String="auto",
                           deterministic::String="constant",
                           from_tag::String="",
                           output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "vecm", "rank" => r, "n_vars" => n),
        Dict{String,Any}("command" => "residuals vecm", "data" => data))
end

# ── Static Factor Residuals ───────────────────────────

function _residuals_static(; data::String, nfactors=nothing, from_tag::String="",
                             output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        ic = ic_criteria(X, min(20, size(X, 2)))
        ic.r_IC1
    else
        nfactors
    end

    fm = estimate_factors(X, r)
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "static", "nfactors" => r),
        Dict{String,Any}("command" => "residuals static", "data" => data))
end

# ── Dynamic Factor Residuals ──────────────────────────

function _residuals_dynamic(; data::String, nfactors=nothing, factor_lags::Int=1,
                              method::String="twostep", from_tag::String="",
                              output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        ic = ic_criteria(X, min(10, size(X, 2)))
        ic.r_IC1
    else
        nfactors
    end

    fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "dynamic", "nfactors" => r),
        Dict{String,Any}("command" => "residuals dynamic", "data" => data))
end

# ── GDFM Residuals ────────────────────────────────────

function _residuals_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                           from_tag::String="",
                           output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    X, varnames = load_multivariate_data(data)

    q = if isnothing(dynamic_rank)
        ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
        ic.q_ratio
    else
        dynamic_rank
    end

    gm = estimate_gdfm(X, q)
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

    storage_save_auto!("residuals", Dict{String,Any}("type" => "gdfm", "dynamic_rank" => q),
        Dict{String,Any}("command" => "residuals gdfm", "data" => data))
end

# ── ARCH Residuals ────────────────────────────────────

function _residuals_arch(; data::String, column::Int=1, q::Int=1, from_tag::String="",
                           output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    y, vname = load_univariate_series(data, column)
    model = estimate_arch(y, q)
    resid = residuals(model)

    println("ARCH($q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="ARCH($q) Standardized Residuals ($vname)")

    storage_save_auto!("residuals", Dict{String,Any}("type" => "arch", "q" => q),
        Dict{String,Any}("command" => "residuals arch", "data" => data))
end

# ── GARCH Residuals ───────────────────────────────────

function _residuals_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                            from_tag::String="",
                            output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    y, vname = load_univariate_series(data, column)
    model = estimate_garch(y, p, q)
    resid = residuals(model)

    println("GARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="GARCH($p,$q) Standardized Residuals ($vname)")

    storage_save_auto!("residuals", Dict{String,Any}("type" => "garch", "p" => p, "q" => q),
        Dict{String,Any}("command" => "residuals garch", "data" => data))
end

# ── EGARCH Residuals ──────────────────────────────────

function _residuals_egarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                             from_tag::String="",
                             output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    y, vname = load_univariate_series(data, column)
    model = estimate_egarch(y, p, q)
    resid = residuals(model)

    println("EGARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="EGARCH($p,$q) Standardized Residuals ($vname)")

    storage_save_auto!("residuals", Dict{String,Any}("type" => "egarch", "p" => p, "q" => q),
        Dict{String,Any}("command" => "residuals egarch", "data" => data))
end

# ── GJR-GARCH Residuals ──────────────────────────────

function _residuals_gjr_garch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                                from_tag::String="",
                                output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    y, vname = load_univariate_series(data, column)
    model = estimate_gjr_garch(y, p, q)
    resid = residuals(model)

    println("GJR-GARCH($p,$q) standardized residuals: variable=$vname")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="GJR-GARCH($p,$q) Standardized Residuals ($vname)")

    storage_save_auto!("residuals", Dict{String,Any}("type" => "gjr_garch", "p" => p, "q" => q),
        Dict{String,Any}("command" => "residuals gjr_garch", "data" => data))
end

# ── SV Residuals ──────────────────────────────────────

function _residuals_sv(; data::String, column::Int=1, draws::Int=5000,
                         from_tag::String="",
                         output::String="", format::String="table")
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
    y, vname = load_univariate_series(data, column)
    model = estimate_sv(y; n_samples=draws)
    resid = residuals(model)

    println("SV standardized residuals: variable=$vname, draws=$draws")
    println()

    res_df = DataFrame(t=1:length(resid), residual=round.(resid; digits=6))
    output_result(res_df; format=Symbol(format), output=output,
                  title="SV Standardized Residuals ($vname)")

    storage_save_auto!("residuals", Dict{String,Any}("type" => "sv"),
        Dict{String,Any}("command" => "residuals sv", "data" => data))
end
