# Residuals commands: model residuals for var, bvar, arima, vecm

function register_residuals_commands!()
    res_var = LeafCommand("var", _residuals_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VAR model")

    res_bvar = LeafCommand("bvar", _residuals_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
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
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
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
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Residuals from VECM (via VAR representation)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"   => res_var,
        "bvar"  => res_bvar,
        "arima" => res_arima,
        "vecm"  => res_vecm,
    )
    return NodeCommand("residuals", subcmds, "Model residuals")
end

# ── VAR Residuals ───────────────────────────────────────

function _residuals_var(; data::String, lags=nothing, from_tag::String="",
                          output::String="", format::String="table")
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
                           sampler::String="nuts", config::String="",
                           from_tag::String="",
                           output::String="", format::String="table")
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
