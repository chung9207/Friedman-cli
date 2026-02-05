# GMM commands: estimate

function register_gmm_commands!()
    gmm_estimate = LeafCommand("estimate", _gmm_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("config"; type=String, default="", description="TOML config for moment conditions and instruments"),
            Option("weighting"; short="w", type=String, default="twostep", description="identity|optimal|twostep|iterated"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a GMM model")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => gmm_estimate,
    )
    return NodeCommand("gmm", subcmds, "Generalized Method of Moments (GMM)")
end

function _gmm_estimate(; data::String, config::String="",
                        weighting::String="twostep",
                        output::String="", format::String="table")
    isempty(config) && error("GMM requires a --config=<file.toml> specifying moment conditions and instruments")

    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    cfg = load_config(config)
    gmm_cfg = get_gmm(cfg)

    weighting_map = Dict(
        "identity" => :identity,
        "optimal" => :optimal,
        "twostep" => :two_step,
        "iterated" => :iterated,
    )
    w = get(weighting_map, lowercase(weighting), :two_step)

    println("Estimating GMM: weighting=$weighting")
    println("  Moment conditions: $(length(gmm_cfg["moment_conditions"]))")
    println()

    # Determine shock variable from config
    moment_cols = gmm_cfg["moment_conditions"]
    shock_var = if !isempty(moment_cols)
        idx = findfirst(==(moment_cols[1]), varnames)
        isnothing(idx) ? 1 : idx
    else
        1
    end

    # Use LP-GMM estimation
    models = estimate_lp_gmm(Y, shock_var, 0; lags=4, weighting=w)

    if !isempty(models)
        model = models[1]

        # gmm_summary now returns a NamedTuple
        summ = gmm_summary(model)

        # J-test
        jtest = j_test(model)
        println()
        println("Hansen's J-test for overidentification:")
        println("  J-statistic: $(round(jtest.J_stat; digits=4))")
        println("  p-value: $(round(jtest.p_value; digits=4))")
        println("  Degrees of freedom: $(jtest.df)")

        if jtest.p_value < 0.05
            printstyled("  → Reject valid moment conditions at 5%\n"; color=:yellow)
        else
            printstyled("  → Cannot reject valid moment conditions\n"; color=:green)
        end

        # Export parameters
        if !isempty(output)
            se = stderror(model)
            param_df = DataFrame(
                parameter=["θ$i" for i in 1:length(model.theta)],
                estimate=model.theta,
                std_error=se
            )
            output_result(param_df; format=Symbol(format), output=output, title="GMM Estimates")
        end
    end
end
