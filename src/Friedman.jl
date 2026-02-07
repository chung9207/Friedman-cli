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

# Old model-first commands that map to new action-first syntax
const DEPRECATION_MAP = Dict(
    "var" => Dict(
        "estimate"  => ["estimate", "var"],
        "lagselect" => ["test", "var", "lagselect"],
        "stability" => ["test", "var", "stability"],
        "irf"       => ["irf", "var"],
        "fevd"      => ["fevd", "var"],
        "hd"        => ["hd", "var"],
        "forecast"  => ["forecast", "var"],
    ),
    "bvar" => Dict(
        "estimate"  => ["estimate", "bvar"],
        "posterior" => ["estimate", "bvar"],
        "irf"       => ["irf", "bvar"],
        "fevd"      => ["fevd", "bvar"],
        "hd"        => ["hd", "bvar"],
        "forecast"  => ["forecast", "bvar"],
    ),
    "lp" => Dict(
        "estimate"  => ["estimate", "lp"],
        "irf"       => ["irf", "lp"],
        "fevd"      => ["fevd", "lp"],
        "hd"        => ["hd", "lp"],
        "forecast"  => ["forecast", "lp"],
    ),
    "factor" => Dict(
        "estimate"  => ["estimate"],  # factor estimate static -> estimate static
        "forecast"  => ["forecast"],  # factor forecast -> forecast static
    ),
    "arima" => Dict(
        "estimate"  => ["estimate", "arima"],
        "forecast"  => ["forecast", "arima"],
    ),
    "gmm" => Dict(
        "estimate"  => ["estimate", "gmm"],
    ),
    "nongaussian" => Dict(
        "fastica"            => ["estimate", "fastica"],
        "ml"                 => ["estimate", "ml"],
        "heteroskedasticity" => ["test", "heteroskedasticity"],
        "normality"          => ["test", "normality"],
        "identifiability"    => ["test", "identifiability"],
    ),
)

"""
    _rewrite_deprecated_args(args) -> Vector{String}

If args use the old model-first syntax (e.g. `var irf data.csv`),
print a deprecation warning and rewrite to new action-first syntax.
"""
function _rewrite_deprecated_args(args::Vector{String})
    isempty(args) && return args

    cmd = args[1]
    !haskey(DEPRECATION_MAP, cmd) && return args

    subcmd_map = DEPRECATION_MAP[cmd]
    if length(args) >= 2 && haskey(subcmd_map, args[2])
        new_prefix = subcmd_map[args[2]]
        # Special cases for factor: pass through the third arg (static/dynamic/gdfm)
        rest = args[3:end]

        new_args = vcat(new_prefix, rest)

        old_syntax = join(args[1:min(2,length(args))], " ")
        new_syntax = join(new_prefix, " ")
        printstyled("Warning: "; color=:yellow, bold=true)
        printstyled("'friedman $old_syntax' is deprecated. Use 'friedman $new_syntax' instead.\n"; color=:yellow)
        println()

        return new_args
    end

    return args
end

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
Handles deprecated syntax rewriting and stored tag resolution.
"""
function main(args::Vector{String}=ARGS)
    # Initialize global settings on first run
    init_settings!()

    # Deprecation layer: rewrite old model-first syntax
    args = _rewrite_deprecated_args(args)

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
