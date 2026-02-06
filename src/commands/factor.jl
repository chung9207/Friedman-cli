# Factor Model commands: estimate (static|dynamic|gdfm), forecast

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

    estimate_node = NodeCommand("estimate",
        Dict{String,Union{NodeCommand,LeafCommand}}(
            "static"  => factor_static,
            "dynamic" => factor_dynamic,
            "gdfm"    => factor_gdfm,
        ),
        "Estimate factor models (static PCA, dynamic, GDFM)")

    factor_forecast = LeafCommand("forecast", _factor_forecast;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("model"; short="m", type=String, default="static", description="static|dynamic|gdfm"),
            Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
            Option("horizon"; short="h", type=Int, default=12, description="Forecast horizon"),
            Option("factor-lags"; short="p", type=Int, default=1, description="Factor VAR lag order (dynamic only)"),
            Option("method"; type=String, default="twostep", description="twostep|em (dynamic only)"),
            Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank (gdfm only)"),
            Option("ci-method"; type=String, default="none", description="none|bootstrap|parametric"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level for intervals"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Forecast observables using factor model (static, dynamic, or GDFM)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => estimate_node,
        "forecast" => factor_forecast,
    )
    return NodeCommand("factor", subcmds, "Factor Models (estimation and forecasting)")
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

function _factor_forecast(; data::String, model::String="static", nfactors=nothing,
                            horizon::Int=12, factor_lags::Int=1, method::String="twostep",
                            dynamic_rank=nothing, ci_method::String="none",
                            conf_level::Float64=0.95, output::String="", format::String="table")
    model in ("static", "dynamic", "gdfm") || error("Unknown model type: $model (expected static|dynamic|gdfm)")

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

    if model == "static"
        _factor_forecast_static(X, varnames, r, horizon, ci_method, conf_level, output, format)
    elseif model == "dynamic"
        _factor_forecast_dynamic(X, varnames, r, horizon, factor_lags, method, output, format)
    else  # gdfm
        q = if isnothing(dynamic_rank)
            println("Selecting dynamic rank...")
            ic_g = ic_criteria_gdfm(X, min(5, size(X, 2)))
            q_opt = ic_g.q_ratio
            println("  Auto-selected $q_opt dynamic factors")
            q_opt
        else
            dynamic_rank
        end
        _factor_forecast_gdfm(X, varnames, r, q, horizon, output, format)
    end
end

function _factor_forecast_static(X, varnames, r, horizon, ci_method, conf_level, output, format)
    println("Forecasting with static factor model: $r factors, horizon=$horizon, CI=$ci_method")
    println()

    fm = estimate_factors(X, r)
    fc = forecast(fm, horizon; ci_method=Symbol(ci_method), conf_level=conf_level)

    fc_df = DataFrame()
    fc_df.horizon = 1:horizon
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = fc.observables[:, vi]
    end

    if ci_method != "none" && !isnothing(fc.observables_lower)
        for (vi, vname) in enumerate(varnames)
            fc_df[!, "$(vname)_lower"] = fc.observables_lower[:, vi]
            fc_df[!, "$(vname)_upper"] = fc.observables_upper[:, vi]
        end
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Static Factor Forecast (h=$horizon, $(length(varnames)) variables)")

    if !isnothing(fc.observables_se)
        println()
        avg_se = round.(mean(fc.observables_se; dims=1)[1, :]; digits=4)
        println("Average forecast standard errors:")
        for (vi, vname) in enumerate(varnames)
            println("  $vname: $(avg_se[vi])")
        end
    end
end

function _factor_forecast_dynamic(X, varnames, r, horizon, factor_lags, method, output, format)
    println("Forecasting with dynamic factor model: $r factors, $factor_lags lags, method=$method, horizon=$horizon")
    println()

    fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    fc = forecast(fm, horizon)

    # Dynamic forecast returns factor forecasts; reconstruct observables via loadings
    factor_fc = fc isa NamedTuple ? fc.factors : fc
    if factor_fc isa AbstractMatrix
        obs_fc = factor_fc * fm.loadings'
    else
        obs_fc = reshape(factor_fc, horizon, r) * fm.loadings'
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizon
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = obs_fc[:, vi]
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Dynamic Factor Forecast (h=$horizon, $(length(varnames)) variables)")
end

function _factor_forecast_gdfm(X, varnames, r, q, horizon, output, format)
    println("Forecasting with GDFM: static rank=$r, dynamic rank=$q, horizon=$horizon")
    println()

    fm = estimate_gdfm(X, q; r=r)

    # GDFM forecast: project common component forward using spectral structure
    # Use the common component's last values + AR(1) extrapolation on factors
    common = fm.common_component  # T x N
    T_obs, N = size(common)

    # Extract static factors from common component via PCA
    F_pca = svd(common)
    factors = F_pca.U[:, 1:r] .* F_pca.S[1:r]'
    loadings = F_pca.V[:, 1:r]

    # Simple AR(1) forecast on each factor
    obs_fc = zeros(horizon, N)
    for fi in 1:r
        f = factors[:, fi]
        # OLS AR(1): f_t = a + b * f_{t-1}
        y_ar = f[2:end]
        x_ar = [ones(length(y_ar)) f[1:end-1]]
        beta = x_ar \ y_ar
        f_last = f[end]
        for h in 1:horizon
            f_next = beta[1] + beta[2] * f_last
            obs_fc[h, :] .+= f_next .* loadings[:, fi]
            f_last = f_next
        end
    end

    fc_df = DataFrame()
    fc_df.horizon = 1:horizon
    for (vi, vname) in enumerate(varnames)
        fc_df[!, vname] = round.(obs_fc[:, vi]; digits=6)
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="GDFM Forecast (h=$horizon, $(length(varnames)) variables)")

    println()
    var_shares = common_variance_share(fm)
    println("Average common variance share: $(round(mean(var_shares); digits=4))")
end
