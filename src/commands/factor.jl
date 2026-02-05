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

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "static"  => factor_static,
        "dynamic" => factor_dynamic,
        "gdfm"    => factor_gdfm,
    )
    return NodeCommand("factor", subcmds, "Factor Models (static, dynamic, GDFM)")
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
        crit_sym = Symbol(lowercase(criterion))
        optimal_r = getfield(ic, crit_sym)
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
        component=1:length(scree.eigenvalues),
        eigenvalue=scree.eigenvalues,
        variance_share=scree.variance_shares,
        cumulative=scree.cumulative_shares
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
        ic = ic_criteria_dynamic(X, min(10, size(X, 2)), 4; method=:twostep)
        optimal_r = ic.r_opt
        println("  Auto-selected $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    println("Estimating dynamic factor model: $r factors, $factor_lags lags, method=$method")
    println()

    model = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))

    stable = is_stationary(model)
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
        q_opt = ic.q_opt
        println("  Auto-selected $q_opt dynamic factors")
        q_opt
    else
        dynamic_rank
    end

    r = if isnothing(nfactors)
        println("Selecting static rank...")
        ic = ic_criteria_gdfm(X, q)
        r_opt = haskey(ic, :r_opt) ? ic.r_opt : 0
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
