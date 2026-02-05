module Friedman

using CSV, DataFrames, PrettyTables, JSON3, TOML
using MacroEconometricModels
using LinearAlgebra: eigvals
using Statistics: mean

# CLI engine
include("cli/types.jl")
include("cli/parser.jl")
include("cli/help.jl")
include("cli/dispatch.jl")

# IO and config
include("io.jl")
include("config.jl")

# Commands
include("commands/var.jl")
include("commands/bvar.jl")
include("commands/irf.jl")
include("commands/fevd.jl")
include("commands/hd.jl")
include("commands/lp.jl")
include("commands/factor.jl")
include("commands/test_cmd.jl")
include("commands/gmm.jl")
include("commands/arima.jl")

const FRIEDMAN_VERSION = v"0.1.1"

"""
    build_app() → Entry

Construct the full CLI command tree.
"""
function build_app()
    root_cmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"    => register_var_commands!(),
        "bvar"   => register_bvar_commands!(),
        "irf"    => register_irf_commands!(),
        "fevd"   => register_fevd_commands!(),
        "hd"     => register_hd_commands!(),
        "lp"     => register_lp_commands!(),
        "factor" => register_factor_commands!(),
        "test"   => register_test_commands!(),
        "gmm"    => register_gmm_commands!(),
        "arima"  => register_arima_commands!(),
    )

    root = NodeCommand("friedman", root_cmds,
        "A macroeconometric analysis toolkit powered by MacroEconometricModels.jl")

    return Entry("friedman", root; version=FRIEDMAN_VERSION)
end

"""
    main(args=ARGS)

Entry point: build the CLI app and dispatch on the given arguments.
"""
function main(args::Vector{String}=ARGS)
    app = build_app()
    dispatch(app, args)
end

export main, build_app

end # module Friedman
