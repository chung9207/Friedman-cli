# List commands: models, results

function register_list_commands!()
    list_models = LeafCommand("models", _list_models;
        args=Argument[],
        options=[
            Option("type"; short="t", type=String, default="", description="Filter by type (var|bvar|lp|arima|arch|garch|...)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="List stored models and results")

    list_results = LeafCommand("results", _list_results;
        args=Argument[],
        options=[
            Option("type"; short="t", type=String, default="", description="Filter by type (irf|fevd|hd|forecast)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="List stored analysis results")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "models"  => list_models,
        "results" => list_results,
    )
    return NodeCommand("list", subcmds, "List stored models and results")
end

function _list_models(; type::String="", format::String="table", output::String="")
    model_types = ["var", "bvar", "lp", "arima", "gmm", "static", "dynamic", "gdfm",
                   "arch", "garch", "egarch", "gjr_garch", "sv", "fastica", "ml"]

    entries = if isempty(type)
        vcat([storage_list(; type_filter=t) for t in model_types]...)
    else
        storage_list(; type_filter=type)
    end

    if isempty(entries)
        println("No stored models found.")
        return
    end

    list_df = DataFrame(
        tag=String[],
        type=String[],
        timestamp=String[],
        info=String[]
    )
    for entry in entries
        tag = get(entry, "tag", "")
        etype = get(entry, "type", "")
        ts = get(entry, "timestamp", "")
        meta = get(entry, "meta", Dict{String,Any}())
        info = get(meta, "command", "")
        push!(list_df, (tag=tag, type=etype, timestamp=ts, info=info))
    end

    output_result(list_df; format=Symbol(format), output=output, title="Stored Models")
end

function _list_results(; type::String="", format::String="table", output::String="")
    result_types = ["irf", "fevd", "hd", "forecast"]

    entries = if isempty(type)
        vcat([storage_list(; type_filter=t) for t in result_types]...)
    else
        storage_list(; type_filter=type)
    end

    if isempty(entries)
        println("No stored results found.")
        return
    end

    list_df = DataFrame(
        tag=String[],
        type=String[],
        timestamp=String[],
        info=String[]
    )
    for entry in entries
        tag = get(entry, "tag", "")
        etype = get(entry, "type", "")
        ts = get(entry, "timestamp", "")
        meta = get(entry, "meta", Dict{String,Any}())
        info = get(meta, "command", "")
        push!(list_df, (tag=tag, type=etype, timestamp=ts, info=info))
    end

    output_result(list_df; format=Symbol(format), output=output, title="Stored Results")
end
