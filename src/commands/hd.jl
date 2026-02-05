# Historical Decomposition commands: compute

function register_hd_commands!()
    hd_compute = LeafCommand("compute", _hd_compute;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute historical decomposition of shocks")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "compute" => hd_compute,
    )
    return NodeCommand("hd", subcmds, "Historical Decomposition (HD)")
end

function _hd_compute(; data::String, lags=nothing, id::String="cholesky",
                      config::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    id_method = Dict(
        "cholesky" => :cholesky,
        "sign" => :sign,
        "narrative" => :narrative,
        "longrun" => :long_run,
    )
    method = get(id_method, id, :cholesky)

    println("Computing Historical Decomposition: VAR($p), id=$id")
    println()

    model = estimate_var(Y, p)

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(:method => method)
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    hd_result = historical_decomposition(model, size(Y, 1) - p; kwargs...)

    summary(hd_result)

    # Verify decomposition
    is_valid = verify_decomposition(hd_result)
    if is_valid
        printstyled("✓ Decomposition verified (contributions sum to actual values)\n"; color=:green)
    else
        printstyled("⚠ Decomposition verification failed\n"; color=:yellow)
    end
    println()

    # Output contributions for each variable
    for vi in 1:n
        T_eff = hd_result.T_eff
        hd_df = DataFrame()
        hd_df.period = 1:T_eff

        # Actual values
        hd_df.actual = hd_result.actual[:, vi]

        # Initial conditions
        hd_df.initial = hd_result.initial_conditions[:, vi]

        # Shock contributions
        for si in 1:n
            shock_name = si <= length(varnames) ? varnames[si] : "shock_$si"
            hd_df[!, "contrib_$shock_name"] = contribution(hd_result, vi, si)
        end

        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        output_result(hd_df; format=Symbol(format),
                      output=isempty(output) ? "" : replace(output, "." => "_$(vname)."),
                      title="Historical Decomposition: $vname ($id identification)")
        println()
    end
end
