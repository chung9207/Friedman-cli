# Unit Root & Cointegration Test commands: adf, kpss, pp, za, np, johansen

function register_test_commands!()
    test_adf = LeafCommand("adf", _test_adf;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test (1-based)"),
            Option("max-lags"; type=Int, default=nothing, description="Max lags (default: auto via AIC)"),
            Option("trend"; type=String, default="constant", description="none|constant|trend|both"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Augmented Dickey-Fuller unit root test")

    test_kpss = LeafCommand("kpss", _test_kpss;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test"),
            Option("trend"; type=String, default="constant", description="constant|trend"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="KPSS stationarity test")

    test_pp = LeafCommand("pp", _test_pp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test"),
            Option("trend"; type=String, default="constant", description="none|constant|trend"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Phillips-Perron unit root test")

    test_za = LeafCommand("za", _test_za;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test"),
            Option("trend"; type=String, default="both", description="intercept|trend|both"),
            Option("trim"; type=Float64, default=0.15, description="Trimming proportion"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Zivot-Andrews unit root test with structural break")

    test_np = LeafCommand("np", _test_np;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test"),
            Option("trend"; type=String, default="constant", description="constant|trend"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Ng-Perron unit root test")

    test_johansen = LeafCommand("johansen", _test_johansen;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order"),
            Option("trend"; type=String, default="constant", description="none|constant|trend"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Johansen cointegration test")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "adf"      => test_adf,
        "kpss"     => test_kpss,
        "pp"       => test_pp,
        "za"       => test_za,
        "np"       => test_np,
        "johansen" => test_johansen,
    )
    return NodeCommand("test", subcmds, "Unit Root & Cointegration Tests")
end

# Helper to extract a single series from data
function _extract_series(data::String, column::Int)
    df = load_data(data)
    varnames = variable_names(df)
    column > length(varnames) && error("column $column out of range (data has $(length(varnames)) numeric columns)")
    y = Vector{Float64}(df[!, varnames[column]])
    return y, varnames[column]
end

function _test_adf(; data::String, column::Int=1, max_lags=nothing,
                    trend::String="constant", format::String="table", output::String="")
    y, vname = _extract_series(data, column)

    lags_arg = isnothing(max_lags) ? :aic : max_lags
    regression = Symbol(trend == "none" ? :none : trend == "both" ? :both : trend)

    println("ADF Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = adf_test(y; lags=lags_arg, regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "Lags" => result.lags,
        "p-value" => round(result.pvalue; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="ADF Test: $vname")

    println()
    if result.pvalue < 0.05
        printstyled("→ Reject H₀ (unit root) at 5% level — series appears stationary\n"; color=:green)
    else
        printstyled("→ Cannot reject H₀ (unit root) at 5% level — series appears non-stationary\n"; color=:yellow)
    end
end

function _test_kpss(; data::String, column::Int=1, trend::String="constant",
                     format::String="table", output::String="")
    y, vname = _extract_series(data, column)

    regression = Symbol(trend)

    println("KPSS Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = kpss_test(y; regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="KPSS Test: $vname")

    println()
    # KPSS: H₀ is stationarity
    if hasproperty(result, :pvalue) && result.pvalue < 0.05
        printstyled("→ Reject H₀ (stationarity) at 5% — series appears non-stationary\n"; color=:yellow)
    else
        printstyled("→ Cannot reject H₀ (stationarity) — series appears stationary\n"; color=:green)
    end
end

function _test_pp(; data::String, column::Int=1, trend::String="constant",
                   format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    regression = Symbol(trend == "none" ? :none : trend)

    println("Phillips-Perron Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = pp_test(y; regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="Phillips-Perron Test: $vname")

    println()
    if result.pvalue < 0.05
        printstyled("→ Reject H₀ (unit root) at 5% — series appears stationary\n"; color=:green)
    else
        printstyled("→ Cannot reject H₀ (unit root) at 5% — series appears non-stationary\n"; color=:yellow)
    end
end

function _test_za(; data::String, column::Int=1, trend::String="both",
                   trim::Float64=0.15, format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    regression = Symbol(trend)

    println("Zivot-Andrews Test: variable=$vname, observations=$(length(y)), model=$trend")
    println()

    result = za_test(y; regression=regression, trim=trim)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "Break date" => result.break_index,
    ]

    output_kv(pairs; format=format, output=output, title="Zivot-Andrews Test: $vname")

    println()
    println("Estimated structural break at observation $(result.break_index)")
end

function _test_np(; data::String, column::Int=1, trend::String="constant",
                   format::String="table", output::String="")
    y, vname = _extract_series(data, column)
    regression = Symbol(trend)

    println("Ng-Perron Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = ngperron_test(y; regression=regression)

    pairs = Pair{String,Any}[
        "MZa statistic" => round(result.MZa; digits=4),
        "MZt statistic" => round(result.MZt; digits=4),
        "MSB statistic" => round(result.MSB; digits=4),
        "MPT statistic" => round(result.MPT; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="Ng-Perron Test: $vname")
end

function _test_johansen(; data::String, lags::Int=2, trend::String="constant",
                         format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    det = Symbol(trend == "none" ? :none : trend)

    println("Johansen Cointegration Test: $(size(Y, 2)) variables, lags=$lags, trend=$trend")
    println()

    result = johansen_test(Y, lags; deterministic=det)

    # Trace test
    trace_df = DataFrame(
        rank=0:(length(result.trace_stats)-1),
        trace_stat=round.(result.trace_stats; digits=4),
        p_value=round.(result.trace_pvalues; digits=4),
        reject=[p < 0.05 ? "yes" : "no" for p in result.trace_pvalues]
    )
    output_result(trace_df; format=Symbol(format), title="Johansen Trace Test")
    println()

    # Max eigenvalue test
    maxeig_df = DataFrame(
        rank=0:(length(result.max_eigen_stats)-1),
        max_stat=round.(result.max_eigen_stats; digits=4),
        p_value=round.(result.max_eigen_pvalues; digits=4),
        reject=[p < 0.05 ? "yes" : "no" for p in result.max_eigen_pvalues]
    )
    output_result(maxeig_df; format=Symbol(format),
                  output=output, title="Johansen Max Eigenvalue Test")

    println()
    # Determine cointegration rank
    rank = 0
    for i in 1:length(result.trace_pvalues)
        if result.trace_pvalues[i] < 0.05
            rank = i
        else
            break
        end
    end
    printstyled("Estimated cointegration rank: $rank\n"; bold=true)
end
