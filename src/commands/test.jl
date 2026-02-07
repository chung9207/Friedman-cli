# Test commands: adf, kpss, pp, za, np, johansen, normality, identifiability,
#                heteroskedasticity, arch_lm, ljung_box, var (lagselect, stability)

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

    test_normality = LeafCommand("normality", _test_normality;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Normality test suite for VAR residuals")

    test_identifiability = LeafCommand("identifiability", _test_identifiability;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("test"; short="t", type=String, default="all", description="strength|gaussianity|independence|overidentification|all"),
            Option("method"; type=String, default="fastica", description="fastica|jade|sobi|dcov|hsic (for gaussianity/independence/overidentification tests)"),
            Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis (for FastICA)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Test identifiability conditions for non-Gaussian SVAR")

    test_heteroskedasticity = LeafCommand("heteroskedasticity", _test_heteroskedasticity;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("method"; type=String, default="markov", description="markov|garch|smooth_transition|external"),
            Option("config"; type=String, default="", description="TOML config (for transition/regime variables)"),
            Option("regimes"; type=Int, default=2, description="Number of regimes"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Heteroskedasticity-based SVAR identification")

    test_arch_lm = LeafCommand("arch_lm", _test_arch_lm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test (1-based)"),
            Option("lags"; short="p", type=Int, default=4, description="Number of lags for ARCH-LM test"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="ARCH-LM test for conditional heteroskedasticity")

    test_ljung_box = LeafCommand("ljung_box", _test_ljung_box;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index to test (1-based)"),
            Option("lags"; short="p", type=Int, default=10, description="Number of lags for Ljung-Box test"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Ljung-Box test on squared residuals")

    # VAR-specific tests (lagselect, stability) as a nested NodeCommand
    var_lagselect = LeafCommand("lagselect", _test_var_lagselect;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("max-lags"; type=Int, default=12, description="Maximum lag order to test"),
            Option("criterion"; type=String, default="aic", description="aic|bic|hqc"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Select optimal lag order for VAR")

    var_stability = LeafCommand("stability", _test_var_stability;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Check VAR stationarity (eigenvalues of companion matrix)")

    var_node = NodeCommand("var",
        Dict{String,Union{NodeCommand,LeafCommand}}(
            "lagselect" => var_lagselect,
            "stability" => var_stability,
        ),
        "VAR model diagnostic tests")

    test_granger = LeafCommand("granger", _test_granger;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("cause"; type=Int, default=1, description="Cause variable index (1-based)"),
            Option("effect"; type=Int, default=2, description="Effect variable index (1-based)"),
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="VECM Granger causality test (short-run, long-run, strong)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "adf"                => test_adf,
        "kpss"               => test_kpss,
        "pp"                 => test_pp,
        "za"                 => test_za,
        "np"                 => test_np,
        "johansen"           => test_johansen,
        "normality"          => test_normality,
        "identifiability"    => test_identifiability,
        "heteroskedasticity" => test_heteroskedasticity,
        "arch_lm"            => test_arch_lm,
        "ljung_box"          => test_ljung_box,
        "var"                => var_node,
        "granger"            => test_granger,
    )
    return NodeCommand("test", subcmds, "Statistical tests (unit root, cointegration, diagnostics)")
end

# ── Unit Root Tests ──────────────────────────────────────

function _test_adf(; data::String, column::Int=1, max_lags=nothing,
                    trend::String="constant", format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)

    lags_arg = isnothing(max_lags) ? :aic : max_lags
    regression = to_regression_symbol(trend)

    println("ADF Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = adf_test(y; lags=lags_arg, regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "Lags" => result.lags,
        "p-value" => round(result.pvalue; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="ADF Test: $vname")

    interpret_test_result(result.pvalue,
        "Reject H0 (unit root) at 5% level -- series appears stationary",
        "Cannot reject H0 (unit root) at 5% level -- series appears non-stationary")
end

function _test_kpss(; data::String, column::Int=1, trend::String="constant",
                     format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)

    regression = to_regression_symbol(trend)

    println("KPSS Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = kpss_test(y; regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="KPSS Test: $vname")

    # KPSS: reversed interpretation (H0 = stationarity)
    pval = hasproperty(result, :pvalue) ? result.pvalue : 1.0
    println()
    if pval < 0.05
        printstyled("-> Reject H0 (stationarity) at 5% -- series appears non-stationary\n"; color=:yellow)
    else
        printstyled("-> Cannot reject H0 (stationarity) -- series appears stationary\n"; color=:green)
    end
end

function _test_pp(; data::String, column::Int=1, trend::String="constant",
                   format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    regression = to_regression_symbol(trend)

    println("Phillips-Perron Test: variable=$vname, observations=$(length(y)), trend=$trend")
    println()

    result = pp_test(y; regression=regression)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
    ]

    output_kv(pairs; format=format, output=output, title="Phillips-Perron Test: $vname")

    interpret_test_result(result.pvalue,
        "Reject H0 (unit root) at 5% -- series appears stationary",
        "Cannot reject H0 (unit root) at 5% -- series appears non-stationary")
end

function _test_za(; data::String, column::Int=1, trend::String="both",
                   trim::Float64=0.15, format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    regression = to_regression_symbol(trend)

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
    y, vname = load_univariate_series(data, column)
    regression = to_regression_symbol(trend)

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

# ── Cointegration ────────────────────────────────────────

function _test_johansen(; data::String, lags::Int=2, trend::String="constant",
                         format::String="table", output::String="")
    Y, varnames = load_multivariate_data(data)
    det = to_regression_symbol(trend)

    println("Johansen Cointegration Test: $(size(Y, 2)) variables, lags=$lags, trend=$trend")
    println()

    result = johansen_test(Y, lags; deterministic=det)

    trace_df = DataFrame(
        rank=0:(length(result.trace_stats)-1),
        trace_stat=round.(result.trace_stats; digits=4),
        p_value=round.(result.trace_pvalues; digits=4),
        reject=[p < 0.05 ? "yes" : "no" for p in result.trace_pvalues]
    )
    output_result(trace_df; format=Symbol(format), title="Johansen Trace Test")
    println()

    maxeig_df = DataFrame(
        rank=0:(length(result.max_eigen_stats)-1),
        max_stat=round.(result.max_eigen_stats; digits=4),
        p_value=round.(result.max_eigen_pvalues; digits=4),
        reject=[p < 0.05 ? "yes" : "no" for p in result.max_eigen_pvalues]
    )
    output_result(maxeig_df; format=Symbol(format),
                  output=output, title="Johansen Max Eigenvalue Test")

    println()
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

# ── Normality Test Suite ─────────────────────────────────

function _test_normality(; data::String, lags=nothing,
                           output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    println("Normality Test Suite: VAR($p), $n variables")
    println()

    suite = normality_test_suite(model)

    test_df = DataFrame(
        test=String[],
        statistic=Float64[],
        p_value=Float64[],
        df=Int[]
    )
    for r in suite.results
        push!(test_df, (
            test=string(r.test_name),
            statistic=round(r.statistic; digits=4),
            p_value=round(r.pvalue; digits=4),
            df=r.df
        ))
    end

    output_result(test_df; format=Symbol(format), output=output,
                  title="Normality Tests for VAR Residuals")

    println()
    n_reject = count(r -> r.pvalue < 0.05, suite.results)
    if n_reject > 0
        printstyled("$n_reject of $(length(suite.results)) tests reject normality at 5%\n"; color=:yellow)
        printstyled("Non-Gaussian identification methods may be applicable\n"; color=:green)
    else
        printstyled("No tests reject normality at 5% -- Gaussian assumption appears valid\n"; color=:green)
    end
end

# ── Identifiability Tests ────────────────────────────────

function _test_identifiability(; data::String, lags=nothing, test::String="all",
                                  method::String="fastica", contrast::String="logcosh",
                                  output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    println("Identifiability Tests: VAR($p), $n variables")
    println()

    results_df = DataFrame(
        test=String[],
        statistic=Float64[],
        p_value=Float64[],
        conclusion=String[]
    )

    run_strength = test == "all" || test == "strength"
    run_gaussianity = test == "all" || test == "gaussianity"
    run_independence = test == "all" || test == "independence"
    run_overid = test == "all" || test == "overidentification"
    run_comparison = test == "all"

    if run_strength
        str_result = test_identification_strength(model)
        push!(results_df, (
            test="Identification Strength",
            statistic=round(str_result.statistic; digits=4),
            p_value=round(str_result.pvalue; digits=4),
            conclusion=str_result.pvalue < 0.05 ? "Strong identification" : "Weak identification"
        ))
    end

    ica_result = nothing
    if run_gaussianity || run_independence || run_overid
        ica_result = if method == "jade"
            identify_jade(model)
        elseif method == "sobi"
            identify_sobi(model)
        elseif method == "dcov"
            identify_dcov(model)
        elseif method == "hsic"
            identify_hsic(model)
        else
            identify_fastica(model; contrast=Symbol(contrast))
        end
    end

    if run_gaussianity && !isnothing(ica_result)
        gauss_result = test_shock_gaussianity(ica_result)
        push!(results_df, (
            test="Shock Gaussianity",
            statistic=round(gauss_result.statistic; digits=4),
            p_value=round(gauss_result.pvalue; digits=4),
            conclusion=gauss_result.pvalue < 0.05 ? "Reject Gaussianity" : "Cannot reject Gaussianity"
        ))
    end

    if run_independence && !isnothing(ica_result)
        indep_result = test_shock_independence(ica_result)
        push!(results_df, (
            test="Shock Independence",
            statistic=round(indep_result.statistic; digits=4),
            p_value=round(indep_result.pvalue; digits=4),
            conclusion=indep_result.pvalue < 0.05 ? "Reject independence" : "Cannot reject independence"
        ))
    end

    if run_overid && !isnothing(ica_result)
        overid_result = test_overidentification(ica_result)
        push!(results_df, (
            test="Overidentification",
            statistic=round(overid_result.statistic; digits=4),
            p_value=round(overid_result.pvalue; digits=4),
            conclusion=overid_result.pvalue < 0.05 ? "Reject overidentification" : "Cannot reject overidentification"
        ))
    end

    if run_comparison
        comp_result = test_gaussian_vs_nongaussian(model)
        push!(results_df, (
            test="Gaussian vs Non-Gaussian",
            statistic=round(comp_result.statistic; digits=4),
            p_value=round(comp_result.pvalue; digits=4),
            conclusion=comp_result.pvalue < 0.05 ? "Non-Gaussian preferred" : "No significant difference"
        ))
    end

    output_result(results_df; format=Symbol(format), output=output,
                  title="Identifiability Test Results")

    println()
    n_reject = count(row -> row.p_value < 0.05, eachrow(results_df))
    if n_reject > 0
        printstyled("$n_reject of $(nrow(results_df)) tests significant at 5%\n"; color=:green)
    else
        printstyled("No tests significant at 5%\n"; color=:yellow)
    end
end

# ── Heteroskedasticity-Based Identification ──────────────

function _test_heteroskedasticity(; data::String, lags=nothing, method::String="markov",
                                     config::String="", regimes::Int=2,
                                     output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)
    df = load_data(data)

    println("Heteroskedasticity SVAR: method=$method, regimes=$regimes, VAR($p), $n variables")
    println()

    result = if method == "garch"
        identify_garch(model)
    elseif method == "smooth_transition"
        if isempty(config)
            error("smooth_transition requires --config specifying [nongaussian] transition_variable")
        end
        cfg = load_config(config)
        ng_cfg = get_nongaussian(cfg)
        tv_name = ng_cfg["transition_variable"]
        tv_idx = findfirst(==(tv_name), names(df))
        isnothing(tv_idx) && error("transition variable '$tv_name' not found in data")
        transition_var = Vector{Float64}(df[!, tv_name])
        identify_smooth_transition(model, transition_var)
    elseif method == "external"
        if isempty(config)
            error("external requires --config specifying [nongaussian] regime_variable")
        end
        cfg = load_config(config)
        ng_cfg = get_nongaussian(cfg)
        rv_name = ng_cfg["regime_variable"]
        rv_idx = findfirst(==(rv_name), names(df))
        isnothing(rv_idx) && error("regime variable '$rv_name' not found in data")
        regime_indicator = Vector{Float64}(df[!, rv_name])
        identify_external_volatility(model, regime_indicator; regimes=regimes)
    else
        identify_markov_switching(model; n_regimes=regimes)
    end

    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), output=output,
                  title="Structural Impact Matrix (B0) -- $method identification")
end

# ── ARCH-LM Test ─────────────────────────────────────────

function _test_arch_lm(; data::String, column::Int=1, lags::Int=4,
                         format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)

    println("ARCH-LM Test: variable=$vname, observations=$(length(y)), lags=$lags")
    println()

    result = arch_lm_test(y, lags)

    pairs = Pair{String,Any}[
        "LM statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
        "Lags" => lags,
    ]

    output_kv(pairs; format=format, output=output, title="ARCH-LM Test: $vname")

    interpret_test_result(result.pvalue,
        "Reject H0 (no ARCH effects) at 5% -- ARCH effects detected",
        "Cannot reject H0 (no ARCH effects) at 5%")
end

# ── Ljung-Box Squared Test ───────────────────────────────

function _test_ljung_box(; data::String, column::Int=1, lags::Int=10,
                           format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)

    println("Ljung-Box Squared Residuals Test: variable=$vname, observations=$(length(y)), lags=$lags")
    println()

    result = ljung_box_squared(y, lags)

    pairs = Pair{String,Any}[
        "Q statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
        "Lags" => lags,
    ]

    output_kv(pairs; format=format, output=output, title="Ljung-Box Squared Test: $vname")

    interpret_test_result(result.pvalue,
        "Reject H0 (no serial correlation in squared residuals) at 5%",
        "Cannot reject H0 at 5% -- no significant ARCH effects")
end

# ── VAR Lag Selection ────────────────────────────────────

function _test_var_lagselect(; data::String, max_lags::Int=12, criterion::String="aic",
                               format::String="table", output::String="")
    Y, _ = load_multivariate_data(data)
    n = size(Y, 2)

    max_p = min(max_lags, size(Y,1) ÷ (3*n))
    crit_sym = Symbol(lowercase(criterion))

    println("Lag order selection (max lags: $max_p, criterion: $criterion)")
    println()

    results = []
    for p in 1:max_p
        try
            m = estimate_var(Y, p)
            push!(results, (p=p, aic=m.aic, bic=m.bic, hqc=m.hqic))
        catch
            continue
        end
    end

    if isempty(results)
        error("could not estimate VAR for any lag order 1:$max_p")
    end

    res_df = DataFrame(results)
    rename!(res_df, :p => :lags, :aic => :AIC, :bic => :BIC, :hqc => :HQC)

    optimal = select_lag_order(Y, max_p; criterion=crit_sym)

    output_result(res_df; format=Symbol(format), output=output, title="Lag Order Selection")
    println()
    printstyled("Optimal lag order ($criterion): $optimal\n"; bold=true)

    if format == "json"
        output_kv(Pair{String,Any}["optimal_lag" => optimal, "criterion" => criterion];
                  format=format, title="Optimal Lag")
    end
end

# ── VAR Stability Check ─────────────────────────────────

function _test_var_stability(; data::String, lags=nothing, format::String="table", output::String="")
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    model = estimate_var(Y, p)
    result = is_stationary(model)

    println("VAR($p) Stationarity Check")
    println()

    eigenvalues = result.eigenvalues
    moduli = abs.(eigenvalues)

    eig_df = DataFrame(
        index=1:length(eigenvalues),
        eigenvalue=string.(round.(eigenvalues; digits=6)),
        modulus=round.(moduli; digits=6)
    )

    output_result(eig_df; format=Symbol(format), output=output, title="Companion Matrix Eigenvalues")
    println()

    if result.is_stationary
        printstyled("VAR($p) is stable (all eigenvalues inside unit circle)\n"; color=:green, bold=true)
    else
        printstyled("VAR($p) is NOT stable (eigenvalue(s) outside unit circle)\n"; color=:red, bold=true)
    end
    println("  Max modulus: $(round(maximum(moduli); digits=6))")
end

# ── VECM Granger Causality Test ────────────────────────

function _test_granger(; data::String, cause::Int=1, effect::Int=2,
                         lags::Int=2, rank::String="auto",
                         deterministic::String="constant",
                         format::String="table", output::String="")
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    cause_name = _var_name(varnames, cause)
    effect_name = _var_name(varnames, effect)

    println("VECM Granger Causality Test: $cause_name → $effect_name")
    println("VECM($(p-1)), rank=$r, $n variables")
    println()

    result = granger_causality_vecm(vecm, cause, effect)

    test_df = DataFrame(
        test=["Short-run", "Long-run", "Strong (joint)"],
        statistic=round.([result.short_run_stat, result.long_run_stat, result.strong_stat]; digits=4),
        df=[result.short_run_df, result.long_run_df, result.strong_df],
        p_value=round.([result.short_run_pvalue, result.long_run_pvalue, result.strong_pvalue]; digits=4)
    )

    output_result(test_df; format=Symbol(format), output=output,
                  title="Granger Causality: $cause_name → $effect_name")

    interpret_test_result(result.strong_pvalue,
        "Reject H0: $cause_name Granger-causes $effect_name (joint short+long-run)",
        "Cannot reject H0: no Granger causality from $cause_name to $effect_name")
end
