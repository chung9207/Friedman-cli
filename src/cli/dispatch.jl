# Subcommand dispatch engine

"""
    _wants_help(args) → Bool

Check if help is requested anywhere in the argument list.
"""
_wants_help(args::Vector{String}) = "--help" in args || "-h" in args

"""
    dispatch(entry, args)

Main dispatch: walk the command tree from `entry` using `args`, then execute the matched leaf.
"""
function dispatch(entry::Entry, args::Vector{String}=ARGS)
    # Handle --version at top level
    if "--version" in args || "-V" in args
        println(entry.name, " v", entry.version)
        return
    end

    # Handle --help at top level only (not when a subcommand follows)
    if isempty(args) || args[1] in ("--help", "-h")
        print_help(stdout, entry)
        return
    end

    dispatch_node(entry.root, args; prog=entry.name)
end

"""
    dispatch_node(node, args; prog)

Walk into a NodeCommand, matching the first token as a subcommand name.
"""
function dispatch_node(node::NodeCommand, args::Vector{String}; prog::String=node.name)
    if isempty(args)
        print_help(stdout, node; prog=prog)
        return
    end

    subcmd_name = args[1]
    rest = args[2:end]

    # If first arg is a known subcommand, recurse into it (carries --help through)
    if haskey(node.subcmds, subcmd_name)
        subcmd = node.subcmds[subcmd_name]
        subprog = prog * " " * subcmd_name
        if subcmd isa NodeCommand
            dispatch_node(subcmd, rest; prog=subprog)
        else
            dispatch_leaf(subcmd, rest; prog=subprog)
        end
        return
    end

    # First arg isn't a subcommand — show help if requested, otherwise error
    if _wants_help(args)
        print_help(stdout, node; prog=prog)
        return
    end

    throw(DispatchError("$prog: unknown command '$subcmd_name'"))
end

"""
    dispatch_leaf(leaf, args; prog)

Parse arguments for a LeafCommand and call its handler.
"""
function dispatch_leaf(leaf::LeafCommand, args::Vector{String}; prog::String=leaf.name)
    # Handle --help
    if _wants_help(args)
        print_help(stdout, leaf; prog=prog)
        return
    end

    try
        parsed = tokenize(args)
        bound = bind_args(parsed, leaf)
        leaf.handler(; bound...)
    catch e
        if e isa ParseError
            throw(ParseError("$prog: $(e.message)"))
        else
            rethrow()
        end
    end
end
