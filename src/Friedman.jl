module Friedman

using CSV, DataFrames, PrettyTables, JSON3, TOML, BSON, Dates
using MacroEconometricModels
using LinearAlgebra: eigvals, diag, I, svd
using Statistics: mean, median

# CLI engine
include("cli/types.jl")
include("cli/parser.jl")
include("cli/help.jl")
include("cli/dispatch.jl")

# IO and config
include("io.jl")
include("config.jl")

# Storage and settings
include("storage.jl")
include("settings.jl")

# Shared utilities (must come before command files)
include("commands/shared.jl")

# Commands (action-first hierarchy)
include("commands/estimate.jl")
include("commands/test.jl")
include("commands/irf.jl")
include("commands/fevd.jl")
include("commands/hd.jl")
include("commands/forecast.jl")
include("commands/list.jl")
include("commands/rename.jl")
include("commands/project.jl")

const FRIEDMAN_VERSION = v"0.1.4"

"""
    build_app() -> Entry

Construct the full CLI command tree.
"""
function build_app()
    root_cmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => register_estimate_commands!(),
        "test"     => register_test_commands!(),
        "irf"      => register_irf_commands!(),
        "fevd"     => register_fevd_commands!(),
        "hd"       => register_hd_commands!(),
        "forecast" => register_forecast_commands!(),
        "list"     => register_list_commands!(),
        "rename"   => register_rename_commands!(),
        "project"  => register_project_commands!(),
    )

    root = NodeCommand("friedman", root_cmds,
        "A macroeconometric analysis toolkit powered by MacroEconometricModels.jl")

    return Entry("friedman", root; version=FRIEDMAN_VERSION)
end

"""
    main(args=ARGS)

Entry point: build the CLI app and dispatch on the given arguments.
Handles stored tag resolution before dispatch.
"""
function main(args::Vector{String}=ARGS)
    # Initialize global settings on first run
    init_settings!()

    # Pre-dispatch: resolve stored tags for irf/fevd/hd/forecast
    args = resolve_stored_tags(args)

    # Handle bare "project" with no subcommand
    if length(args) == 1 && args[1] == "project"
        _project_show()
        return
    end

    app = build_app()
    dispatch(app, args)
end

export main, build_app

end # module Friedman
