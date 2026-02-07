# Predict commands: in-sample fitted values for var, bvar, arima, vecm

function register_predict_commands!()
    pred_var = LeafCommand("var", _predict_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="In-sample fitted values from VAR model")

    pred_bvar = LeafCommand("bvar", _predict_bvar;
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
        description="In-sample fitted values from Bayesian VAR (posterior mean)")

    pred_arima = LeafCommand("arima", _predict_arima;
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
        description="In-sample fitted values from ARIMA model")

    pred_vecm = LeafCommand("vecm", _predict_vecm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="In-sample fitted values from VECM (via VAR representation)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"   => pred_var,
        "bvar"  => pred_bvar,
        "arima" => pred_arima,
        "vecm"  => pred_vecm,
    )
    return NodeCommand("predict", subcmds, "In-sample predictions (fitted values)")
end

# ── VAR Predict ─────────────────────────────────────────

function _predict_var(; data::String, lags=nothing, from_tag::String="",
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

    storage_save_auto!("predict", Dict{String,Any}("type" => "var", "lags" => p, "n_vars" => n),
        Dict{String,Any}("command" => "predict var", "data" => data))
end

# ── BVAR Predict ────────────────────────────────────────

function _predict_bvar(; data::String, lags::Int=4, draws::Int=2000,
                        sampler::String="nuts", config::String="",
                        from_tag::String="",
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

    storage_save_auto!("predict", Dict{String,Any}("type" => "bvar", "lags" => p, "n_vars" => n),
        Dict{String,Any}("command" => "predict bvar", "data" => data))
end

# ── ARIMA Predict ───────────────────────────────────────

function _predict_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          method::String="css_mle", auto::Bool=false,
                          from_tag::String="",
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

    storage_save_auto!("predict", Dict{String,Any}("type" => "arima", "model" => label),
        Dict{String,Any}("command" => "predict arima", "data" => data))
end

# ── VECM Predict ───────────────────────────────────────

function _predict_vecm(; data::String, lags::Int=2, rank::String="auto",
                         deterministic::String="constant",
                         from_tag::String="",
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

    storage_save_auto!("predict", Dict{String,Any}("type" => "vecm", "rank" => r, "n_vars" => n),
        Dict{String,Any}("command" => "predict vecm", "data" => data))
end
