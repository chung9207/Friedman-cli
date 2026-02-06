# Non-Gaussian SVAR commands: fastica, ml, heteroskedasticity, normality, identifiability

function register_nongaussian_commands!()
    ng_fastica = LeafCommand("fastica", _nongaussian_fastica;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("method"; type=String, default="fastica", description="fastica|infomax|jade|sobi|dcov|hsic"),
            Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis (for FastICA)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="ICA-based non-Gaussian SVAR identification")

    ng_ml = LeafCommand("ml", _nongaussian_ml;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("distribution"; short="d", type=String, default="student_t", description="student_t|skew_t|ghd|mixture_normal|pml|skew_normal"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Maximum likelihood non-Gaussian SVAR identification")

    ng_heteroskedasticity = LeafCommand("heteroskedasticity", _nongaussian_heteroskedasticity;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("method"; type=String, default="markov", description="markov|garch|smooth_transition|external"),
            Option("config"; type=String, default="", description="TOML config (for transition/regime variables)"),
            Option("regimes"; type=Int, default=2, description="Number of regimes"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Heteroskedasticity-based SVAR identification")

    ng_normality = LeafCommand("normality", _nongaussian_normality;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Normality test suite for VAR residuals")

    ng_identifiability = LeafCommand("identifiability", _nongaussian_identifiability;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("test"; short="t", type=String, default="all", description="strength|gaussianity|independence|overidentification|all"),
            Option("method"; type=String, default="fastica", description="fastica|infomax|jade|sobi|dcov|hsic (for gaussianity/independence/overidentification tests)"),
            Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis (for FastICA)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Test identifiability conditions for non-Gaussian SVAR")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "fastica"            => ng_fastica,
        "ml"                 => ng_ml,
        "heteroskedasticity" => ng_heteroskedasticity,
        "normality"          => ng_normality,
        "identifiability"    => ng_identifiability,
    )
    return NodeCommand("nongaussian", subcmds,
        "Non-Gaussian SVAR Identification (ICA, ML, heteroskedasticity)")
end

# ── Helper: estimate VAR with auto lag selection ──

function _ng_estimate_var(data::String, lags)
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y, 1) ÷ (3 * n)); criterion=:aic)
    else
        lags
    end

    model = estimate_var(Y, p)
    return model, Y, varnames, p
end

# ── Handlers ──

function _nongaussian_fastica(; data::String, lags=nothing, method::String="fastica",
                                contrast::String="logcosh", output::String="", format::String="table")
    model, Y, varnames, p = _ng_estimate_var(data, lags)
    n = length(varnames)

    println("Non-Gaussian SVAR: method=$method, contrast=$contrast, VAR($p), $n variables")
    println()

    result = if method == "infomax"
        identify_infomax(model)
    elseif method == "jade"
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

    # Convergence info
    if hasproperty(result, :converged)
        if result.converged
            printstyled("Converged in $(result.iterations) iterations\n"; color=:green)
        else
            printstyled("Did not converge after $(result.iterations) iterations\n"; color=:yellow)
        end
    end
    println()

    # B0 matrix
    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    println()

    # Structural shocks summary (first few rows)
    shocks = result.shocks
    T_shocks = size(shocks, 1)
    n_show = min(T_shocks, 10)
    shock_df = DataFrame(shocks[1:n_show, :], ["shock_$i" for i in 1:n])
    insertcols!(shock_df, 1, :t => 1:n_show)
    output_result(shock_df; format=Symbol(format), output=output,
                  title="Structural Shocks (first $n_show observations)")
end

function _nongaussian_ml(; data::String, lags=nothing, distribution::String="student_t",
                           output::String="", format::String="table")
    model, Y, varnames, p = _ng_estimate_var(data, lags)
    n = length(varnames)

    println("Non-Gaussian ML SVAR: distribution=$distribution, VAR($p), $n variables")
    println()

    result = if distribution == "mixture_normal"
        identify_mixture_normal(model)
    elseif distribution == "pml"
        identify_pml(model)
    elseif distribution == "skew_normal"
        identify_skew_normal(model)
    else
        identify_nongaussian_ml(model; distribution=Symbol(distribution))
    end

    # B0 matrix
    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    println()

    # Model fit
    output_kv(Pair{String,Any}[
        "Log-likelihood" => round(result.loglik; digits=4),
        "Log-likelihood (Gaussian)" => round(result.loglik_gaussian; digits=4),
        "AIC" => round(result.aic; digits=4),
        "BIC" => round(result.bic; digits=4),
        "Distribution" => string(result.distribution),
    ]; format=format, title="Model Fit")

    # Distribution parameters
    if !isempty(result.dist_params)
        println()
        println("Distribution parameters:")
        for (k, v) in pairs(result.dist_params)
            if v isa AbstractArray
                println("  $k = $(round.(v; digits=4))")
            else
                println("  $k = $(round(v; digits=4))")
            end
        end
    end

    # Standard errors
    if !isnothing(result.se) && length(result.se) > 0
        println()
        se_df = DataFrame(
            parameter=["B0[$i,$j]" for i in 1:n for j in 1:n],
            estimate=vec(result.B0),
            std_error=result.se[1:min(length(result.se), n*n)]
        )
        output_result(se_df; format=Symbol(format), output=output,
                      title="Parameter Estimates with Standard Errors")
    end
end

function _nongaussian_heteroskedasticity(; data::String, lags=nothing, method::String="markov",
                                           config::String="", regimes::Int=2,
                                           output::String="", format::String="table")
    model, Y, varnames, p = _ng_estimate_var(data, lags)
    n = length(varnames)
    df = load_data(data)

    println("Heteroskedasticity SVAR: method=$method, regimes=$regimes, VAR($p), $n variables")
    println()

    result = if method == "garch"
        identify_garch(model)
    elseif method == "smooth_transition"
        # Get transition variable from config or data
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
        # Default: markov switching
        identify_markov_switching(model; n_regimes=regimes)
    end

    # B0 matrix
    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), output=output,
                  title="Structural Impact Matrix (B0) — $method identification")
end

function _nongaussian_normality(; data::String, lags=nothing,
                                  output::String="", format::String="table")
    model, Y, varnames, p = _ng_estimate_var(data, lags)
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
        printstyled("No tests reject normality at 5% — Gaussian assumption appears valid\n"; color=:green)
    end
end

function _nongaussian_identifiability(; data::String, lags=nothing, test::String="all",
                                        method::String="fastica", contrast::String="logcosh",
                                        output::String="", format::String="table")
    model, Y, varnames, p = _ng_estimate_var(data, lags)
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

    # For gaussianity, independence, and overidentification tests, we need an ICA result
    ica_result = nothing
    if run_gaussianity || run_independence || run_overid
        ica_result = if method == "infomax"
            identify_infomax(model)
        elseif method == "jade"
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
