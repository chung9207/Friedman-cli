# Factor Model commands: static, dynamic, gdfm

function register_factor_commands!()
    factor_static = LeafCommand("static", _factor_static;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("criterion"; type=String, default="ic1", description="ic1|ic2|ic3 for auto selection"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate static factor model (PCA)")

    factor_dynamic = LeafCommand("dynamic", _factor_dynamic;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order"),
            Option("method"; type=String, default="twostep", description="twostep|em"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate dynamic factor model")

    factor_gdfm = LeafCommand("gdfm", _factor_gdfm;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate generalized dynamic factor model")

    factor_forecast = LeafCommand("forecast", _factor_forecast;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("horizon"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("ci-method"; type=String, default="none", description="none|bootstrap|parametric"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level for intervals"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast observables using static factor model")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "static"   => factor_static,
        "dynamic"  => factor_dynamic,
        "gdfm"     => factor_gdfm,
        "forecast" => factor_forecast,
    )
    return NodeCommand("factor", subcmds, "Factor Models (static, dynamic, GDFM, forecast)")
end

function _factor_static(; data::String, nfactors=nothing, criterion::String="ic1",
                         output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    # Auto-select number of factors
    r = if isnothing(nfactors)
        println("Selecting number of factors via Bai-Ng information criteria...")
        ic = ic_criteria(X, min(20, size(X, 2)))
        r_sym = Symbol("r_", uppercase(criterion))
        optimal_r = getfield(ic, r_sym)
        println("  $criterion suggests $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Estimating static factor model: $r factors, $(size(X, 2)) variables, $(size(X, 1)) observations")
    println()

    model = estimate_factors(X, r)

    # Scree plot data
    scree = scree_plot_data(model)
    scree_df = DataFrame(
        component=scree.factors,
        eigenvalue=scree.explained_variance,
        cumulative=scree.cumulative_variance
    )
    output_result(scree_df; format=Symbol(format), title="Scree Data (Eigenvalues & Variance Shares)")
    println()

    # Factor loadings
    loadings = model.loadings  # N x r
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format),
                  output=output, title="Factor Loadings")
end

function _factor_dynamic(; data::String, nfactors=nothing, factor_lags::Int=1,
                          method::String="twostep", output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    r = if isnothing(nfactors)
        println("Selecting number of factors...")
        ic = ic_criteria(X, min(10, size(X, 2)))
        optimal_r = ic.r_IC1
        println("  Auto-selected $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Estimating dynamic factor model: $r factors, $factor_lags lags, method=$method")
    println()

    model = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))

    stable_result = is_stationary(model)
    stable = stable_result isa Bool ? stable_result : stable_result.is_stationary
    if stable
        printstyled("✓ Factor VAR is stationary\n"; color=:green)
    else
        printstyled("⚠ Factor VAR is not stationary\n"; color=:yellow)
    end
    println()

    # Factor loadings
    loadings = model.loadings
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format),
                  output=output, title="Dynamic Factor Loadings")

    # Factor VAR coefficients
    println()
    println("Factor VAR Companion Matrix eigenvalues:")
    comp = companion_matrix_factors(model)
    eig_moduli = abs.(eigvals(comp))
    for (i, ev) in enumerate(sort(eig_moduli; rev=true))
        println("  λ$i = $(round(ev; digits=6))")
    end
end

function _factor_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                       output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    q = if isnothing(dynamic_rank)
        println("Selecting dynamic rank...")
        ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
        q_opt = ic.q_ratio
        println("  Auto-selected $q_opt dynamic factors")
        q_opt
    else
        dynamic_rank
    end

    r = if isnothing(nfactors)
        println("Selecting static rank...")
        ic_static = ic_criteria(X, min(20, size(X, 2)))
        r_opt = ic_static.r_IC1
        println("  Auto-selected $r_opt static factors")
        r_opt
    else
        nfactors
    end

    println("Estimating GDFM: static rank=$r, dynamic rank=$q")
    println()

    model = estimate_gdfm(X, q; r=r)

    # Variance shares
    var_shares = common_variance_share(model)
    var_df = DataFrame(variable=varnames, common_variance_share=round.(var_shares; digits=4))
    output_result(var_df; format=Symbol(format), output=output,
                  title="GDFM Common Variance Shares")

    println()
    println("Average common variance share: $(round(mean(var_shares); digits=4))")
end

function _factor_forecast(; data::String, nfactors=nothing, horizon::Int=12,
                            ci_method::String="none", conf_level::Float64=0.95,
                            output::String="", format::String="table")
    df = load_data(data)
    X = df_to_matrix(df)
    varnames = variable_names(df)

    r = if isnothing(nfactors)
        println("Selecting number of factors via Bai-Ng information criteria...")
        ic = ic_criteria(X, min(20, size(X, 2)))
        optimal_r = ic.r_IC1
        println("  IC1 suggests $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Forecasting with static factor model: $r factors, horizon=$horizon, CI=$ci_method")
    println()

    model = estimate_factors(X, r)
    fc = forecast(model, horizon; ci_method=Symbol(ci_method), conf_level=conf_level)

    # Build forecast table for observables
    fc_df = DataFrame()
    fc_df.horizon = 1:horizon

    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = fc.observables[:, vi]
    end

    # Add confidence intervals if available
    if ci_method != "none" && !isnothing(fc.observables_lower)
        for (vi, vname) in enumerate(varnames)
            fc_df[!, "$(vname)_lower"] = fc.observables_lower[:, vi]
            fc_df[!, "$(vname)_upper"] = fc.observables_upper[:, vi]
        end
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Factor Model Forecast (h=$horizon, $(length(varnames)) variables)")

    # Standard errors if available
    if !isnothing(fc.observables_se)
        println()
        avg_se = round.(mean(fc.observables_se; dims=1)[1, :]; digits=4)
        println("Average forecast standard errors:")
        for (vi, vname) in enumerate(varnames)
            println("  $vname: $(avg_se[vi])")
        end
    end
end
