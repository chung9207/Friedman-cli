# Subcommand dispatch engine

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
    if isempty(args) || args[1] in ("--help", "-h")
        print_help(stdout, node; prog=prog)
        return
    end

    subcmd_name = args[1]
    rest = args[2:end]

    if !haskey(node.subcmds, subcmd_name)
        printstyled(stderr, "Error: "; bold=true, color=:red)
        println(stderr, "unknown command '$subcmd_name'")
        println(stderr)
        print_help(stderr, node; prog=prog)
        exit(1)
    end

    subcmd = node.subcmds[subcmd_name]
    subprog = prog * " " * subcmd_name

    if subcmd isa NodeCommand
        dispatch_node(subcmd, rest; prog=subprog)
    else
        dispatch_leaf(subcmd, rest; prog=subprog)
    end
end

"""
    dispatch_leaf(leaf, args; prog)

Parse arguments for a LeafCommand and call its handler.
"""
function dispatch_leaf(leaf::LeafCommand, args::Vector{String}; prog::String=leaf.name)
    # Handle --help
    if "--help" in args || "-h" in args
        print_help(stdout, leaf; prog=prog)
        return
    end

    try
        parsed = tokenize(args)
        bound = bind_args(parsed, leaf)
        leaf.handler(; bound...)
    catch e
        if e isa ParseError
            printstyled(stderr, "Error: "; bold=true, color=:red)
            println(stderr, e.message)
            println(stderr)
            print_help(stderr, leaf; prog=prog)
            exit(1)
        else
            rethrow()
        end
    end
end
