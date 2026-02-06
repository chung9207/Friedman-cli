# VAR commands: estimate, lagselect, stability

function register_var_commands!()
    var_estimate = LeafCommand("estimate", _var_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("trend"; type=String, default="constant", description="none|constant|trend|both"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a VAR(p) model")

    var_lagselect = LeafCommand("lagselect", _var_lagselect;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("max-lags"; type=Int, default=12, description="Maximum lag order to test"),
            Option("criterion"; type=String, default="aic", description="aic|bic|hqc"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Select optimal lag order for VAR")

    var_stability = LeafCommand("stability", _var_stability;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Check VAR stationarity (eigenvalues of companion matrix)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => var_estimate,
        "lagselect" => var_lagselect,
        "stability" => var_stability,
    )
    return NodeCommand("var", subcmds, "Vector Autoregression (VAR) models")
end

function _var_estimate(; data::String, lags=nothing, trend::String="constant",
                        output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)

    # Determine lag order
    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    println("Estimating VAR($p) with $n variables: $(join(varnames, ", "))")
    println("Trend: $trend, Observations: $(size(Y, 1))")
    println()

    model = estimate_var(Y, p)
    MacroEconometricModels.summary(model)

    # Output coefficient matrix
    # coef(model) returns (n_regressors x n_vars) — transpose so rows=equations
    coef_mat = coef(model)
    n_rows = size(coef_mat, 1)

    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end

    coef_df = DataFrame(permutedims(coef_mat), row_names)
    insertcols!(coef_df, 1, :equation => varnames)

    output_result(coef_df; format=Symbol(format), output=output, title="VAR($p) Coefficients")

    # Print information criteria
    println()
    output_kv([
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
        "Log-likelihood" => loglikelihood(model),
    ]; format=format, title="Information Criteria")
end

function _var_lagselect(; data::String, max_lags::Int=12, criterion::String="aic",
                         format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    n = size(Y, 2)

    max_p = min(max_lags, size(Y,1) ÷ (3*n))
    crit_sym = Symbol(lowercase(criterion))

    println("Lag order selection (max lags: $max_p, criterion: $criterion)")
    println()

    # Compute all criteria for each lag
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

    # Mark optimal
    optimal = select_lag_order(Y, max_p; criterion=crit_sym)

    output_result(res_df; format=Symbol(format), output=output, title="Lag Order Selection")
    println()
    printstyled("Optimal lag order ($criterion): $optimal\n"; bold=true)

    if format == "json"
        output_kv(Pair{String,Any}["optimal_lag" => optimal, "criterion" => criterion];
                  format=format, title="Optimal Lag")
    end
end

function _var_stability(; data::String, lags=nothing, format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
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
        printstyled("✓ VAR($p) is stable (all eigenvalues inside unit circle)\n"; color=:green, bold=true)
    else
        printstyled("✗ VAR($p) is NOT stable (eigenvalue(s) outside unit circle)\n"; color=:red, bold=true)
    end
    println("  Max modulus: $(round(maximum(moduli); digits=6))")
end
