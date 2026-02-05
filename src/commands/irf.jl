# IRF commands: compute

function register_irf_commands!()
    irf_compute = LeafCommand("compute", _irf_compute;
        args=[
            Argument("data"; description="Path to CSV data file"),
        ],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias"),
            Option("ci"; type=String, default="bootstrap", description="none|bootstrap|theoretical"),
            Option("replications"; type=Int, default=1000, description="Bootstrap replications"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Compute impulse response functions")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "compute" => irf_compute,
    )
    return NodeCommand("irf", subcmds, "Impulse Response Functions (IRF)")
end

function _build_check_func(config_path::String)
    if isempty(config_path)
        return nothing, nothing
    end
    cfg = load_config(config_path)
    id_cfg = get_identification(cfg)

    check_func = nothing
    narrative_check = nothing

    if haskey(id_cfg, "sign_matrix")
        sign_mat = id_cfg["sign_matrix"]
        horizons = get(id_cfg, "horizons", [0])
        check_func = function(irf_values)
            for h_idx in 1:length(horizons)
                h = horizons[h_idx] + 1  # 1-based indexing
                if h > size(irf_values, 1)
                    continue
                end
                for i in 1:size(sign_mat, 1)
                    for j in 1:size(sign_mat, 2)
                        s = sign_mat[i, j]
                        if s != 0
                            if s > 0 && irf_values[h, j, i] < 0
                                return false
                            elseif s < 0 && irf_values[h, j, i] > 0
                                return false
                            end
                        end
                    end
                end
            end
            return true
        end
    end

    if haskey(id_cfg, "narrative")
        narr = id_cfg["narrative"]
        shock_idx = narr["shock_index"]
        periods = narr["periods"]
        signs = narr["signs"]
        narrative_check = function(structural_shocks)
            for (t, s) in zip(periods, signs)
                if t > size(structural_shocks, 1)
                    continue
                end
                if s > 0 && structural_shocks[t, shock_idx] < 0
                    return false
                elseif s < 0 && structural_shocks[t, shock_idx] > 0
                    return false
                end
            end
            return true
        end
    end

    return check_func, narrative_check
end

function _irf_compute(; data::String, lags=nothing, shock::Int=1, horizons::Int=20,
                       id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
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

    # Map identification method
    id_method = Dict(
        "cholesky" => :cholesky,
        "sign" => :sign,
        "narrative" => :narrative,
        "longrun" => :long_run,
    )

    println("Computing IRFs: VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    println()

    model = estimate_var(Y, p)

    # Handle Arias identification separately
    if id == "arias"
        _irf_arias(model, config, horizons, varnames, shock; format=format, output=output)
        return
    end

    method = get(id_method, id, :cholesky)
    ci_type = Symbol(ci)

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :ci_type => ci_type,
        :reps => replications,
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    irf_result = irf(model, horizons; kwargs...)

    summary(irf_result)

    # Output IRF for the specified shock
    irf_vals = irf_result.values  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    # Add confidence intervals if available
    if ci != "none" && !isnothing(irf_result.ci_lower)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock]
        end
    end

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock ($id identification)")
end

function _irf_arias(model::VARModel, config::String, horizons::Int,
                    varnames::Vector{String}, shock::Int; format::String="table", output::String="")
    isempty(config) && error("Arias identification requires a --config file with restrictions")
    cfg = load_config(config)
    id_cfg = get(cfg, "identification", Dict())

    # Build restrictions from config
    zeros_list = get(id_cfg, "zero_restrictions", [])
    signs_list = get(id_cfg, "sign_restrictions", [])

    n = nvars(model)
    zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
    sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]

    restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
    result = identify_arias(model, restrictions, horizons)

    # Extract IRF from Arias result
    irf_vals = result.irf  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    shock_name = shock <= length(varnames) ? varnames[shock] : "shock_$shock"
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Arias et al. identification)")
end
