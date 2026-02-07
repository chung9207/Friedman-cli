# Project management commands

function register_project_commands!()
    project_list = LeafCommand("list", _project_list;
        args=Argument[],
        options=[
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="List all registered projects")

    # Also add a "show" leaf for explicit `friedman project show`
    project_show = LeafCommand("show", _project_show;
        args=Argument[],
        options=Option[],
        description="Show current project info")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "list" => project_list,
        "show" => project_show,
    )
    return NodeCommand("project", subcmds, "Project management")
end

function _project_show(; )
    dir = pwd()
    name = basename(dir)

    println("Current project: $name")
    println("  Path: $dir")

    path = joinpath(dir, ".friedmanlog.bson")
    if isfile(path)
        entries = storage_list()
        println("  Stored entries: $(length(entries))")

        type_counts = Dict{String,Int}()
        for e in entries
            t = get(e, "type", "unknown")
            type_counts[t] = get(type_counts, t, 0) + 1
        end
        if !isempty(type_counts)
            println("  By type:")
            for (t, c) in sort(collect(type_counts))
                println("    $t: $c")
            end
        end
    else
        println("  No storage file found (run a command to initialize)")
    end
end

function _project_list(; format::String="table", output::String="")
    projects = load_projects()

    if isempty(projects)
        println("No registered projects.")
        return
    end

    proj_df = DataFrame(
        name=String[],
        path=String[]
    )
    for p in projects
        push!(proj_df, (name=get(p, "name", ""), path=get(p, "path", "")))
    end

    output_result(proj_df; format=Symbol(format), output=output, title="Registered Projects")
end
