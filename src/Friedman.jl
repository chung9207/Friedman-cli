module Friedman

using CSV, DataFrames, PrettyTables, JSON3, TOML, BSON, Dates
using MacroEconometricModels
using LinearAlgebra: eigvals, diag, I, svd
using Statistics: mean, median, var

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
include("commands/predict.jl")
include("commands/residuals.jl")
include("commands/filter.jl")
include("commands/data.jl")
include("commands/list.jl")
include("commands/rename.jl")
include("commands/project.jl")

const FRIEDMAN_VERSION = v"0.2.1"

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
        "forecast"  => register_forecast_commands!(),
        "predict"   => register_predict_commands!(),
        "residuals" => register_residuals_commands!(),
        "filter"    => register_filter_commands!(),
        "data"      => register_data_commands!(),
        "list"      => register_list_commands!(),
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

    # Pre-dispatch: resolve stored tags for irf/fevd/hd/forecast/predict/residuals
    args = resolve_stored_tags(args)

    # Handle bare "project" with no subcommand → default to "show"
    if length(args) == 1 && args[1] == "project"
        push!(args, "show")
    end

    app = build_app()
    try
        dispatch(app, args)
    catch e
        if e isa ParseError || e isa DispatchError
            printstyled(stderr, "Error: "; bold=true, color=:red)
            println(stderr, e.message)
            exit(1)
        else
            rethrow()
        end
    end
end

export main, build_app

end # module Friedman
