# ARIMA commands: estimate, auto, forecast

function register_arima_commands!()
    arima_estimate = LeafCommand("estimate", _arima_estimate;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=1, description="AR order"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Estimate ARIMA(p,d,q) model")

    arima_auto = LeafCommand("auto", _arima_auto;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("max-p"; type=Int, default=5, description="Maximum AR order"),
            Option("max-d"; type=Int, default=2, description="Maximum differencing order"),
            Option("max-q"; type=Int, default=5, description="Maximum MA order"),
            Option("criterion"; type=String, default="bic", description="aic|bic"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Automatic ARIMA order selection")

    arima_forecast = LeafCommand("forecast", _arima_forecast;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto selection)"),
            Option("d"; type=Int, default=0, description="Differencing order"),
            Option("q"; type=Int, default=0, description="MA order"),
            Option("horizons"; short="h", type=Int, default=12, description="Forecast steps ahead"),
            Option("confidence"; type=Float64, default=0.95, description="Confidence interval level"),
            Option("method"; short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Estimate and forecast ARIMA model")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate"  => arima_estimate,
        "auto"      => arima_auto,
        "forecast"  => arima_forecast,
    )
    return NodeCommand("arima", subcmds, "ARIMA (AutoRegressive Integrated Moving Average) models")
end

# Estimate the appropriate model variant based on (p, d, q)
function _estimate_arima_model(y::Vector{Float64}, p::Int, d::Int, q::Int; method::Symbol=:css_mle)
    if d == 0 && q == 0
        return estimate_ar(y, p; method=method)
    elseif d == 0 && p == 0
        return estimate_ma(y, q; method=method)
    elseif d == 0
        return estimate_arma(y, p, q; method=method)
    else
        return estimate_arima(y, p, d, q; method=method)
    end
end

function _model_label(p::Int, d::Int, q::Int)
    if d == 0 && q == 0
        return "AR($p)"
    elseif d == 0 && p == 0
        return "MA($q)"
    elseif d == 0
        return "ARMA($p,$q)"
    else
        return "ARIMA($p,$d,$q)"
    end
end

function _arima_coef_table(model; format::String="table", output::String="", title::String="Coefficients")
    c = coef(model)
    se = stderror(model)

    # Build parameter names
    p_order = ar_order(model)
    q_order = ma_order(model)
    param_names = String[]
    for i in 1:p_order
        push!(param_names, "ar$i")
    end
    for i in 1:q_order
        push!(param_names, "ma$i")
    end
    # Add intercept/constant if present
    n_named = p_order + q_order
    for i in (n_named+1):length(c)
        push!(param_names, "const$i")
    end

    coef_df = DataFrame(
        parameter = param_names,
        estimate  = round.(c; digits=6),
        std_error = round.(se; digits=6),
    )

    output_result(coef_df; format=Symbol(format), output=output, title=title)
end

function _arima_estimate(; data::String, column::Int=1, p::Int=1, d::Int=0, q::Int=0,
                          method::String="css_mle", format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    method_sym = Symbol(method)
    label = _model_label(p, d, q)

    println("Estimating $label: variable=$vname, observations=$(length(y)), method=$method")
    println()

    model = _estimate_arima_model(y, p, d, q; method=method_sym)

    _arima_coef_table(model; format=format, output=output, title="$label Coefficients ($vname)")

    println()
    output_kv([
        "AIC" => Any(round(aic(model); digits=4)),
        "BIC" => Any(round(bic(model); digits=4)),
        "Log-likelihood" => Any(round(loglikelihood(model); digits=4)),
    ]; format=format, title="Information Criteria")
end

function _arima_auto(; data::String, column::Int=1, max_p::Int=5, max_d::Int=2, max_q::Int=5,
                      criterion::String="bic", method::String="css_mle",
                      format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    crit_sym = Symbol(lowercase(criterion))
    method_sym = Symbol(method)

    println("Auto ARIMA: variable=$vname, observations=$(length(y))")
    println("  Search: p=0:$max_p, d=0:$max_d, q=0:$max_q, criterion=$criterion, method=$method")
    println()

    model = auto_arima(y; max_p=max_p, max_q=max_q, max_d=max_d, criterion=crit_sym, method=method_sym)

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    printstyled("Selected model: $label\n"; bold=true)
    println()

    _arima_coef_table(model; format=format, output=output, title="$label Coefficients ($vname)")

    println()
    output_kv([
        "AIC" => Any(round(aic(model); digits=4)),
        "BIC" => Any(round(bic(model); digits=4)),
        "Log-likelihood" => Any(round(loglikelihood(model); digits=4)),
    ]; format=format, title="Information Criteria")
end

function _arima_forecast(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          horizons::Int=12, confidence::Float64=0.95,
                          method::String="css_mle", format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    method_sym = Symbol(method)

    # Estimate model: auto if p not specified, explicit otherwise
    model = if isnothing(p)
        println("Auto ARIMA + Forecast: variable=$vname, observations=$(length(y))")
        auto_arima(y; method=method_sym)
    else
        label = _model_label(p, d, q)
        println("$label Forecast: variable=$vname, observations=$(length(y)), method=$method")
        _estimate_arima_model(y, p, d, q; method=method_sym)
    end

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)
    println("  Model: $label, horizons=$horizons, confidence=$confidence")
    println()

    fc = forecast(model, horizons; conf_level=confidence)

    fc_df = DataFrame(
        horizon   = 1:horizons,
        forecast  = round.(fc.forecast; digits=6),
        ci_lower  = round.(fc.ci_lower; digits=6),
        ci_upper  = round.(fc.ci_upper; digits=6),
        se        = round.(fc.se; digits=6),
    )

    output_result(fc_df; format=Symbol(format), output=output,
                  title="$label Forecast — $vname ($(Int(round(confidence*100)))% CI)")
end
