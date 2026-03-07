# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# REPL / interactive session mode

using REPL
using REPL.LineEdit

"""
    Session

Mutable state for the interactive REPL session.
"""
mutable struct Session
    data_path::String
    df::Union{DataFrame,Nothing}
    Y::Union{Matrix{Float64},Nothing}
    varnames::Vector{String}
    results::Dict{Symbol,Any}
    last_model::Symbol
end

Session() = Session("", nothing, nothing, String[], Dict{Symbol,Any}(), :none)

function session_load_data!(s::Session, path::String)
    df = load_data(path)
    Y = df_to_matrix(df)
    vnames = variable_names(df)
    s.data_path = path
    s.df = df
    s.Y = Y
    s.varnames = vnames
    s.results = Dict{Symbol,Any}()
    s.last_model = :none
    return s
end

function session_clear!(s::Session)
    s.data_path = ""
    s.df = nothing
    s.Y = nothing
    s.varnames = String[]
    s.results = Dict{Symbol,Any}()
    s.last_model = :none
    return s
end

function session_store_result!(s::Session, model_type::Symbol, result)
    s.results[model_type] = result
    s.last_model = model_type
    return s
end

session_has_data(s::Session) = !isempty(s.data_path)

session_get_result(s::Session, model_type::Symbol) = get(s.results, model_type, nothing)

const BUILTIN_DATASETS = Dict(
    "fred-md" => :fred_md, "fred-qd" => :fred_qd,
    "pwt" => :pwt, "mpdta" => :mpdta, "ddcg" => :ddcg,
)

function parse_data_source(source::String)
    if startswith(source, ":")
        name = source[2:end]
        haskey(BUILTIN_DATASETS, name) || error("unknown built-in dataset ':$name'. Available: $(join(keys(BUILTIN_DATASETS), ", "))")
        return (:builtin, BUILTIN_DATASETS[name])
    else
        return (:file, source)
    end
end

function session_load_builtin!(s::Session, name::Symbol)
    ts = load_example(name)
    df = DataFrame(ts.data, ts.varnames)
    Y = Matrix{Float64}(ts.data)
    s.data_path = ":$(replace(string(name), "_" => "-"))"
    s.df = df
    s.Y = Y
    s.varnames = ts.varnames
    s.results = Dict{Symbol,Any}()
    s.last_model = :none
    return s
end

"""
    inject_session_data(session, args) → args

If session has data loaded and the command args don't already include a data file,
inject the session data path after the subcommand token and before any options.
"""
function inject_session_data(s::Session, args::Vector{String})
    session_has_data(s) || return args
    length(args) < 2 && return args

    cmd_depth = _command_depth(args)
    positionals_start = cmd_depth + 1

    # Check if there's already a positional arg (non-option) after the subcommand
    has_positional = false
    for i in positionals_start:length(args)
        arg = args[i]
        startswith(arg, "-") && break
        has_positional = true
        break
    end

    has_positional && return args

    new_args = copy(args)
    insert!(new_args, positionals_start, s.data_path)
    return new_args
end

"""
    _command_depth(args) → Int

Count how many leading tokens are command/subcommand names (not options or data files).
Returns 2 for "estimate var", 3 for "dsge bayes estimate", etc.
"""
function _command_depth(args::Vector{String})
    depth = 0
    for arg in args
        startswith(arg, "-") && break
        (endswith(arg, ".csv") || endswith(arg, ".toml") || endswith(arg, ".jl") || contains(arg, "/") || contains(arg, "\\")) && break
        depth += 1
        depth >= 4 && break
    end
    return depth
end

const DOWNSTREAM_ACTIONS = Set(["irf", "fevd", "hd", "forecast", "predict", "residuals"])

is_downstream_command(args::Vector{String}) =
    !isempty(args) && args[1] in DOWNSTREAM_ACTIONS

function detect_model_type(args::Vector{String})
    length(args) >= 2 || return :none
    return Symbol(args[2])
end

function is_estimate_command(args::Vector{String})
    !isempty(args) && args[1] == "estimate"
end

"""
    repl_dispatch(session, app, args)

Dispatch a command within the REPL. Handles REPL-specific commands
(data use/current/clear, exit/quit), injects session data, captures
estimation results. Never calls exit().
"""
function repl_dispatch(s::Session, app::Entry, args::Vector{String})
    isempty(args) && return

    # REPL-only commands
    if args[1] == "exit" || args[1] == "quit"
        throw(InterruptException())
    end

    # data use / data current / data clear
    if length(args) >= 2 && args[1] == "data"
        if args[2] == "use" && length(args) >= 3
            source = args[3]
            kind, val = parse_data_source(source)
            if kind == :builtin
                session_load_builtin!(s, val)
            else
                session_load_data!(s, val)
            end
            printstyled("✓ "; color=:green)
            println("Loaded $(s.data_path) ($(size(s.Y, 1))×$(size(s.Y, 2)), vars: $(join(s.varnames, ", ")))")
            return
        elseif args[2] == "current"
            if session_has_data(s)
                println("$(s.data_path) ($(size(s.Y, 1))×$(size(s.Y, 2)))")
                if !isempty(s.results)
                    println("Cached results: $(join(keys(s.results), ", "))")
                end
            else
                println("No data loaded")
            end
            return
        elseif args[2] == "clear"
            session_clear!(s)
            printstyled("✓ "; color=:green)
            println("Data and results cleared")
            return
        end
    end

    # Inject session data if needed
    args = inject_session_data(s, args)

    # Check if downstream command can use cached model
    extra_kw = Dict{Symbol,Any}()
    if is_downstream_command(args)
        model_type = detect_model_type(args)
        cached = session_get_result(s, model_type)
        if !isnothing(cached)
            extra_kw[:model] = cached
        end
    end

    # Dispatch and capture result
    result = dispatch(app, args; extra_kw...)

    # Cache estimation results
    if is_estimate_command(args) && !isnothing(result)
        model_type = detect_model_type(args)
        if model_type != :none
            session_store_result!(s, model_type, result)
        end
    end
end

"""
    start_repl()

Launch the interactive REPL with a `friedman>` prompt.
"""
function start_repl()
    app = build_app()
    s = SESSION
    session_clear!(s)

    printstyled("Friedman REPL v$(FRIEDMAN_VERSION)\n"; bold=true)
    println("Type commands as you would on the command line. Type 'exit' to quit.")
    println()

    _repl_readline_loop(app, s)
end

function _repl_readline_loop(app::Entry, s::Session)
    while true
        try
            printstyled("friedman> "; color=:blue, bold=true)
            line = readline(stdin)
            isempty(strip(line)) && continue

            args = _split_repl_line(line)
            try
                repl_dispatch(s, app, args)
            catch e
                if e isa InterruptException
                    println("Goodbye!")
                    return
                elseif e isa ParseError || e isa DispatchError
                    printstyled(stderr, "Error: "; bold=true, color=:red)
                    println(stderr, e.message)
                else
                    printstyled(stderr, "Error: "; bold=true, color=:red)
                    println(stderr, sprint(showerror, e))
                end
            end
        catch e
            if e isa EOFError || e isa InterruptException
                println("\nGoodbye!")
                return
            end
            rethrow()
        end
    end
end

"""
    _split_repl_line(line) → Vector{String}

Split a REPL input line into tokens, respecting quoted strings.
"""
function _split_repl_line(line::String)
    tokens = String[]
    i = 1
    while i <= length(line)
        while i <= length(line) && isspace(line[i])
            i += 1
        end
        i > length(line) && break

        if line[i] == '"'
            j = findnext('"', line, i + 1)
            if isnothing(j)
                push!(tokens, line[i+1:end])
                break
            end
            push!(tokens, line[i+1:j-1])
            i = j + 1
        else
            j = findnext(isspace, line, i)
            if isnothing(j)
                push!(tokens, line[i:end])
                break
            end
            push!(tokens, line[i:j-1])
            i = j
        end
    end
    return tokens
end

"""
    complete_command(app, partial_line) → Vector{String}

Return completion candidates for the current partial input line.
"""
function complete_command(app::Entry, partial::String)
    tokens = _split_repl_line(partial)
    isempty(tokens) && return sort(collect(keys(app.root.subcmds)))

    node = app.root
    for (i, tok) in enumerate(tokens[1:end-1])
        if node isa NodeCommand && haskey(node.subcmds, tok)
            sub = node.subcmds[tok]
            if sub isa NodeCommand
                node = sub
            else
                return _complete_leaf_options(sub, tokens[end])
            end
        else
            return String[]
        end
    end

    prefix = tokens[end]

    if node isa NodeCommand
        if startswith(prefix, "-")
            return String[]
        end
        return sort([k for k in keys(node.subcmds) if startswith(k, prefix)])
    end

    return String[]
end

function _complete_leaf_options(leaf::LeafCommand, prefix::String)
    startswith(prefix, "-") || return String[]
    options = ["--" * o.name for o in leaf.options]
    flags = ["--" * f.name for f in leaf.flags]
    all_opts = vcat(options, flags)
    return sort([o for o in all_opts if startswith(o, prefix)])
end

struct FriedmanCompletionProvider <: LineEdit.CompletionProvider
    app::Entry
end

function LineEdit.complete_line(c::FriedmanCompletionProvider, state)
    partial = String(LineEdit.buffer(state))
    completions = complete_command(c.app, partial)
    tokens = _split_repl_line(partial)
    last_token = isempty(tokens) ? "" : tokens[end]
    return completions, last_token, !isempty(completions)
end

const SESSION = Session()
