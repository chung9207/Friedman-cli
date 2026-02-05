# BVAR commands: estimate, posterior

function register_bvar_commands!()
    bvar_estimate = LeafCommand("estimate", _bvar_estimate;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("prior"; type=String, default="minnesota", description="Prior type: minnesota"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Estimate a Bayesian VAR model")

    bvar_posterior = LeafCommand("posterior", _bvar_posterior;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="nuts", description="nuts|hmc|smc"),
            Option("method"; type=String, default="mean", description="mean|median"),
            Option("config"; type=String, default="", description="TOML config for prior hyperparameters"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Extract posterior summary from Bayesian VAR")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => bvar_estimate,
        "posterior" => bvar_posterior,
    )
    return NodeCommand("bvar", subcmds, "Bayesian Vector Autoregression (BVAR)")
end

function _build_prior(config_path::String, Y::AbstractMatrix, p::Int)
    if isempty(config_path)
        return nothing
    end
    cfg = load_config(config_path)
    prior_cfg = get_prior(cfg)

    if prior_cfg["type"] == "minnesota"
        if prior_cfg["optimize"]
            println("Optimizing Minnesota prior hyperparameters...")
            return optimize_hyperparameters(Y, p)
        else
            sigma_ar = ones(size(Y, 2))  # will be estimated from AR(1) residuals
            for i in 1:size(Y, 2)
                y = Y[:, i]
                if length(y) > 2
                    X = y[1:end-1]
                    y_dep = y[2:end]
                    b = X \ y_dep
                    resid = y_dep .- X .* b
                    sigma_ar[i] = sqrt(sum(resid .^ 2) / (length(resid) - 1))
                end
            end
            return MinnesotaHyperparameters(;
                tau=prior_cfg["lambda1"],
                decay=prior_cfg["lambda3"],
                lambda=prior_cfg["lambda2"],
                sigma=sigma_ar
            )
        end
    end
    return nothing
end

function _bvar_estimate(; data::String, lags::Int=4, prior::String="minnesota",
                         draws::Int=2000, sampler::String="nuts",
                         config::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)
    p = lags

    println("Estimating Bayesian VAR($p) with $n variables: $(join(varnames, ", "))")
    println("Prior: $prior, Sampler: $sampler, Draws: $draws")
    println()

    prior_obj = _build_prior(config, Y, p)

    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler),
        nsamples=draws,
        prior=prior_obj)

    # Extract posterior mean model
    mean_model = posterior_mean_model(chain, p, n; data=Y)
    summary(mean_model)

    # Output coefficients
    coef_mat = coef(mean_model)
    n_cols = size(coef_mat, 2)
    col_names = String[]
    for lag in 1:p
        for v in varnames
            push!(col_names, "$(v)_L$(lag)")
        end
    end
    if n_cols > n * p
        push!(col_names, "const")
    end

    coef_df = DataFrame(coef_mat, col_names)
    insertcols!(coef_df, 1, :equation => varnames)

    output_result(coef_df; format=Symbol(format), output=output, title="BVAR($p) Posterior Mean Coefficients")
end

function _bvar_posterior(; data::String, lags::Int=4, draws::Int=2000,
                          sampler::String="nuts", method::String="mean",
                          config::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)
    p = lags

    println("Estimating BVAR($p) and extracting posterior $method...")
    println()

    prior_obj = _build_prior(config, Y, p)

    chain = estimate_bvar(Y, p;
        sampler=Symbol(sampler),
        nsamples=draws,
        prior=prior_obj)

    model = if method == "median"
        posterior_median_model(chain, p, n; data=Y)
    else
        posterior_mean_model(chain, p, n; data=Y)
    end

    summary(model)

    coef_mat = coef(model)
    n_cols = size(coef_mat, 2)
    col_names = String[]
    for lag in 1:p
        for v in varnames
            push!(col_names, "$(v)_L$(lag)")
        end
    end
    if n_cols > n * p
        push!(col_names, "const")
    end

    coef_df = DataFrame(coef_mat, col_names)
    insertcols!(coef_df, 1, :equation => varnames)

    output_result(coef_df; format=Symbol(format), output=output, title="BVAR($p) Posterior $(titlecase(method)) Coefficients")

    println()
    output_kv([
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
    ]; format=format, title="Information Criteria (Posterior $(titlecase(method)))")
end
