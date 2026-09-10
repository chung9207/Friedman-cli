# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# REPL / interactive session mode


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
    # `~` has no shell to expand it inside the REPL; store the expanded path so
    # every downstream command that receives it via inject_session_data resolves.
    # Stem `macro` injects `macro.jld2` when that file exists (resolve_stem).
    path = startswith(path, ":") ? path : resolve_stem(_expanduser(path); slot=:data)
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

"""
    parse_data_source(source) → (:builtin, Symbol) | (:file, String)

Classify a `data use` argument. A leading `:` marks a bundled example dataset,
resolved through the shared `parse_dataset_name` so the REPL accepts exactly the
names `data list`/`data load` advertise (either separator spelling).
"""
function parse_data_source(source::String)
    startswith(source, ":") && return (:builtin, parse_dataset_name(source))
    return (:file, source)
end

"""
    session_load_builtin!(session, name)

Load a bundled example dataset into the session. Delegates to `session_load_data!`
via the canonical `:name` reference so builtins and files take one code path —
in particular panel datasets keep their `group`/`time` id columns.
"""
function session_load_builtin!(s::Session, name::Symbol)
    return session_load_data!(s, ":$(name)")
end

"""
    _walk_command_path(app, args) → (node_or_leaf, depth)

Consume leading tokens while they match subcommand names on the registry tree.
Returns the node/leaf reached and the number of tokens consumed as the command path.
The first non-matching non-option token is a positional (data/model path) — not sniffed by extension.
"""
function _walk_command_path(app::Entry, args::Vector{String})
    node = app.root
    depth = 0
    for arg in args
        startswith(arg, "-") && break
        if node isa NodeCommand
            sub = get(node.subcmds, arg, nothing)
            isnothing(sub) && break
            depth += 1
            if sub isa LeafCommand
                return (sub, depth)
            end
            node = sub
        else
            break
        end
    end
    return (node, depth)
end

"""
    inject_session_data(session, args, app) → args

If session has data loaded and the leaf has no positional yet, inject the session
data path after the command path and before options.

Tree-walks `app.root` (no `.csv`/extension sniffing) so extensionless paths and
`:builtin` datasets are treated as positionals correctly (F27).
"""
function inject_session_data(s::Session, args::Vector{String}, app::Entry)
    session_has_data(s) || return args
    isempty(args) && return args

    node_or_leaf, depth = _walk_command_path(app, args)
    depth == 0 && return args
    # Incomplete path still on a NodeCommand — nothing to inject yet
    node_or_leaf isa LeafCommand || return args

    positionals_start = depth + 1
    for i in positionals_start:length(args)
        startswith(args[i], "-") && break
        return args  # already has a positional
    end

    new_args = copy(args)
    insert!(new_args, positionals_start, s.data_path)
    return new_args
end

# Actions that accept a cached in-memory model handle from Session.results (F15 twin of .fmod)
const MODEL_CACHE_ACTIONS = Set(["irf", "fevd", "hd", "forecast", "predict", "residuals"])

is_downstream_command(args::Vector{String}) =
    !isempty(args) && args[1] in MODEL_CACHE_ACTIONS

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

    # Inject session data if needed (tree-walk on registry)
    args = inject_session_data(s, args, app)

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
    app = APP
    s = SESSION
    session_clear!(s)
    # F63: build completion candidates once at session start
    idx = _reset_completion_index!(app)
    try
        _init_completion_provider()
    catch
        # REPL stdlib not available (e.g. compiled sysimage); tab completion disabled
    end
    # Keep index on provider if LineEdit path is active (rebuild after init)
    _COMPLETION_INDEX[] = idx

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

# ── Completion index (F63: precomputed once per session, not per keystroke) ──

"""
Sorted candidate vectors for tab completion, built once from the registry tree.
"""
struct CompletionIndex
    children::Dict{Vector{String},Vector{String}}  # path → sorted subcommand names
    leaf_opts::Dict{Vector{String},Vector{String}} # path → sorted --option/--flag names
end

function build_completion_index(app::Entry)::CompletionIndex
    children = Dict{Vector{String},Vector{String}}()
    leaf_opts = Dict{Vector{String},Vector{String}}()
    function walk(node::NodeCommand, path::Vector{String})
        kids = sort!(collect(keys(node.subcmds)))
        children[copy(path)] = kids
        for name in kids
            sub = node.subcmds[name]
            p = vcat(path, name)
            if sub isa NodeCommand
                walk(sub, p)
            else
                opts = String["--" * o.name for o in sub.options]
                append!(opts, ["--" * f.name for f in sub.flags])
                sort!(opts)
                leaf_opts[p] = opts
            end
        end
    end
    walk(app.root, String[])
    return CompletionIndex(children, leaf_opts)
end

# Session-scoped index; rebuilt in start_repl / when APP changes
const _COMPLETION_INDEX = Ref{Union{Nothing,CompletionIndex}}(nothing)

function _completion_index(app::Entry)::CompletionIndex
    idx = _COMPLETION_INDEX[]
    if isnothing(idx)
        idx = build_completion_index(app)
        _COMPLETION_INDEX[] = idx
    end
    return idx
end

function _reset_completion_index!(app::Entry)
    _COMPLETION_INDEX[] = build_completion_index(app)
    return _COMPLETION_INDEX[]
end

"""
    complete_command(app, partial_line; index=nothing) → Vector{String}

Return completion candidates for the current partial input line.
Uses a precomputed `CompletionIndex` (F63) when available.
"""
function complete_command(app::Entry, partial::String; index::Union{Nothing,CompletionIndex}=nothing)
    idx = isnothing(index) ? _completion_index(app) : index
    tokens = _split_repl_line(partial)
    isempty(tokens) && return get(idx.children, String[], String[])

    path = String[]
    for tok in tokens[1:end-1]
        startswith(tok, "-") && return String[]
        if haskey(idx.children, path) && tok in idx.children[path]
            push!(path, tok)
            if haskey(idx.leaf_opts, path)
                return _filter_prefix(idx.leaf_opts[path], tokens[end])
            end
        else
            return String[]
        end
    end

    prefix = tokens[end]
    if startswith(prefix, "-")
        # Completing options for a leaf reached by path (no trailing partial subcmd)
        return haskey(idx.leaf_opts, path) ? _filter_prefix(idx.leaf_opts[path], prefix) : String[]
    end
    kids = get(idx.children, path, String[])
    return _filter_prefix(kids, prefix)
end

_filter_prefix(candidates::Vector{String}, prefix::String) =
    isempty(prefix) ? candidates : filter(c -> startswith(c, prefix), candidates)

function _complete_leaf_options(leaf::LeafCommand, prefix::String)
    startswith(prefix, "-") || return String[]
    options = ["--" * o.name for o in leaf.options]
    flags = ["--" * f.name for f in leaf.flags]
    all_opts = sort(vcat(options, flags))
    return _filter_prefix(all_opts, prefix)
end

# FriedmanCompletionProvider is defined at runtime when REPL is loaded
# to avoid requiring REPL as a compile-time dependency
function _init_completion_provider()
    REPLmod = Base.require(Base.PkgId(Base.UUID("3fa0cd96-eef1-5676-8a61-b3b8758bbdc5"), "REPL"))
    LineEditmod = getfield(REPLmod, :LineEdit)
    CompletionProvider = getfield(LineEditmod, :CompletionProvider)

    @eval begin
        const _LineEdit = $LineEditmod

        struct FriedmanCompletionProvider <: $CompletionProvider
            app::Entry
            index::CompletionIndex
        end

        function $_LineEdit.complete_line(c::FriedmanCompletionProvider, state)
            partial = String($_LineEdit.buffer(state))
            completions = complete_command(c.app, partial; index=c.index)
            tokens = _split_repl_line(partial)
            last_token = isempty(tokens) ? "" : tokens[end]
            return completions, last_token, !isempty(completions)
        end
    end
end

const SESSION = Session()
