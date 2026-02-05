# CLI type system adapted from Comonicon.jl AST (see COMONICON_LICENSE)

"""
    Argument(name, type, required, default, description)

A positional command-line argument.
"""
struct Argument
    name::String
    type::Type
    required::Bool
    default::Any
    description::String
end

Argument(name::String; type::Type=String, required::Bool=true, default=nothing, description::String="") =
    Argument(name, type, required, default, description)

"""
    Option(name, short, type, default, description)

A named command-line option (e.g. `--lags=2` or `-l 2`).
"""
struct Option
    name::String
    short::String
    type::Type
    default::Any
    description::String
end

Option(name::String; short::String="", type::Type=String, default=nothing, description::String="") =
    Option(name, short, type, default, description)

"""
    Flag(name, short, description)

A boolean command-line flag (e.g. `--verbose` or `-v`).
"""
struct Flag
    name::String
    short::String
    description::String
end

Flag(name::String; short::String="", description::String="") =
    Flag(name, short, description)

"""
    LeafCommand(name, handler, args, options, flags, description)

A terminal command that executes a handler function.
"""
struct LeafCommand
    name::String
    handler::Function
    args::Vector{Argument}
    options::Vector{Option}
    flags::Vector{Flag}
    description::String
end

LeafCommand(name::String, handler::Function;
    args::Vector{Argument}=Argument[],
    options::Vector{Option}=Option[],
    flags::Vector{Flag}=Flag[],
    description::String="") =
    LeafCommand(name, handler, args, options, flags, description)

"""
    NodeCommand(name, subcmds, description)

A command group that dispatches to subcommands.
"""
struct NodeCommand
    name::String
    subcmds::Dict{String,Union{NodeCommand,LeafCommand}}
    description::String
end

NodeCommand(name::String; subcmds::Dict{String,Union{NodeCommand,LeafCommand}}=Dict{String,Union{NodeCommand,LeafCommand}}(), description::String="") =
    NodeCommand(name, subcmds, description)

"""
    Entry(name, root, version)

The top-level CLI entry point.
"""
struct Entry
    name::String
    root::NodeCommand
    version::VersionNumber
end

Entry(name::String, root::NodeCommand; version::VersionNumber=v"0.1.0") =
    Entry(name, root, version)
