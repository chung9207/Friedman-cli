# Machine-readable self-description (P1-5; F6 · W5/#140 closes #63)
# Output is raw JSON (not an Envelope) — agents parse it as the schema document.
#
# Since W5/#140 the document is fully machine-actionable: every leaf doc carries
# a draft-07 `input_schema` over the invocation surface (shared with the MCP
# `inputSchema` in W7) and the registry-declared `tables` (the same set the W3
# key drift gate enforces); the root doc carries the `contract` block (embedded
# envelope schema + exit-code taxonomy) and, under `--docs`, the agent guide.

# ── Precompile-time embeds ────────────────────────────────────
# Baked in as consts (the FRIEDMAN_VERSION pattern): a runtime relative read
# breaks in the installed sysimage, where the source tree is not present.
const _AGENT_GUIDE_PATH = joinpath(@__DIR__, "agent_guide.md")
include_dependency(_AGENT_GUIDE_PATH)
const _AGENT_GUIDE_MD = read(_AGENT_GUIDE_PATH, String)

const _ENVELOPE_SCHEMA_PATH = normpath(joinpath(@__DIR__, "..", "..", "schema", "envelope-v1.json"))
include_dependency(_ENVELOPE_SCHEMA_PATH)
const _ENVELOPE_SCHEMA_STR = read(_ENVELOPE_SCHEMA_PATH, String)

function _type_str(T)
    return string(T)
end

function _default_json(x)
    return _json_safe(x)
end

# ── draft-07 input schema over the invocation surface ─────────
# One dialect everywhere (envelope schema, in-repo validator, CI validator are
# all draft-07); 2020-12 buys nothing used here.

"""JSON Schema type name for a declared CLI surface type."""
function _json_type(T::Type)
    T === String  && return "string"
    T === Int     && return "integer"
    T === Float64 && return "number"
    T === Bool    && return "boolean"
    return "string"  # unreached today (String/Int/Float64 are the full set); safe fallback
end

"""Last-wins CommandSpec for `path` (`register!` appends)."""
function _spec_for_path(path::Vector{String})
    i = findlast(s -> s.path == path, REGISTRY)
    return i === nothing ? nothing : REGISTRY[i]
end

function _option_spec(spec::CommandSpec, name::String)
    i = findfirst(o -> o.name == name, spec.options)
    return i === nothing ? nothing : spec.options[i]
end

"""`x-handle` annotation for a data slot or `OptionSpec.handle` option, or `nothing`."""
function _x_handle_dict(spec::CommandSpec, name::String; is_arg::Bool=false)
    if spec.path == ["show"] && is_arg
        return Dict{String,Any}(
            "role" => "any",
            "kinds" => String[],
            "types" => String[],
        )
    end
    if name == "data"
        return Dict{String,Any}(
            "role" => "data",
            "kinds" => String.(spec.data_kinds),
            "types" => String[],
        )
    end
    is_arg && return nothing
    ospec = _option_spec(spec, name)
    (ospec === nothing || !ospec.handle) && return nothing
    if name == "model"
        return Dict{String,Any}(
            "role" => "model",
            "kinds" => String[],
            "types" => String.(spec.model_types),
        )
    elseif name == "result"
        return Dict{String,Any}(
            "role" => "result",
            "kinds" => String[],
            "types" => String.(spec.result_types),
        )
    end
    return nothing
end

"""
    _input_schema(leaf, path) → Dict

Draft-07 object schema over `leaf`'s invocation surface. Property names are the
CLI's kebab-case spellings; each property carries an `x-cli` annotation
(`kind` = argument|option|flag, `position` for positionals, `long`/`short`
spellings) so an agent can reconstruct the exact argv from a validated object.
Shared with the MCP `inputSchema` (W7). Data slots and `OptionSpec.handle`
options also carry `x-handle` (`role`/`kinds`/`types`).
"""
function _input_schema(leaf::LeafCommand, path::Vector{String})
    spec = _spec_for_path(path)
    props = Dict{String,Any}()
    required = String[]
    for (i, a) in enumerate(leaf.args)
        p = Dict{String,Any}(
            "type" => _json_type(a.type),
            "x-cli" => Dict{String,Any}("kind" => "argument", "position" => i),
        )
        isempty(a.description) || (p["description"] = a.description)
        a.default === nothing || (p["default"] = _default_json(a.default))
        if spec !== nothing
            xh = _x_handle_dict(spec, a.name; is_arg=true)
            xh !== nothing && (p["x-handle"] = xh)
        end
        props[a.name] = p
        a.required && push!(required, a.name)
    end
    for o in leaf.options
        xcli = Dict{String,Any}("kind" => "option", "long" => "--" * o.name)
        isempty(o.short) || (xcli["short"] = "-" * o.short)
        p = Dict{String,Any}("type" => _json_type(o.type), "x-cli" => xcli)
        isempty(o.description) || (p["description"] = o.description)
        o.default === nothing || (p["default"] = _default_json(o.default))
        o.choices === nothing || (p["enum"] = o.choices)
        if spec !== nothing
            xh = _x_handle_dict(spec, o.name)
            xh !== nothing && (p["x-handle"] = xh)
        end
        props[o.name] = p
    end
    for f in leaf.flags
        xcli = Dict{String,Any}("kind" => "flag", "long" => "--" * f.name)
        isempty(f.short) || (xcli["short"] = "-" * f.short)
        p = Dict{String,Any}("type" => "boolean", "default" => false, "x-cli" => xcli)
        isempty(f.description) || (p["description"] = f.description)
        props[f.name] = p
    end
    return Dict{String,Any}(
        "\$schema" => "http://json-schema.org/draft-07/schema#",
        "title" => join(vcat(["friedman"], path), " "),
        "type" => "object",
        "properties" => props,
        "required" => required,
        "additionalProperties" => false,
    )
end

# ── Registry-declared result tables (W3/#138 keys) ────────────

"""
    _registry_tables(path, leaf) → Vector{Dict}

The leaf's registry-declared TableSpecs — the SAME declaration set the W3 key
drift gate enforces, so these are exactly the envelope `data` keys the leaf can
emit (`family: true` → keys are `<name>_<variable-slug>`). Lookup is last-wins
over REGISTRY (matching the adapter's dedup); a hidden-alias path resolves via
the leaf's primary name. Leaves outside the registry (only `schema` itself)
return an empty list.
"""
function _registry_tables(path::Vector{String}, leaf::LeafCommand)
    primary = isempty(path) ? path : vcat(path[1:end-1], [leaf.name])
    key = join(primary, " ")
    hit = nothing
    for spec in REGISTRY
        join(spec.path, " ") == key && (hit = spec)
    end
    hit === nothing && return Dict{String,Any}[]
    return [Dict{String,Any}(
                "name" => String(t.name),
                "description" => t.description,
                "family" => t.family,
            ) for t in hit.tables]
end

# ── Contract block (root doc) ─────────────────────────────────

const _EXIT_CODE_TAXONOMY = [
    (0, "ok",       "success"),
    (2, "usage",    "CLI usage error (unknown command/option, bad value)"),
    (3, "data",     "input data error (missing file, bad CSV, bad artifact)"),
    (4, "config",   "config/TOML error"),
    (5, "model",    "model/domain failure (typed codes where recognized)"),
    (6, "env",      "environment failure (network, model-version)"),
    (1, "internal", "unexpected internal error — report as a bug"),
]

function _contract_doc()
    return Dict{String,Any}(
        "envelope_schema" => JSON3.read(_ENVELOPE_SCHEMA_STR),
        "exit_codes" => [Dict{String,Any}(
                             "exit_code" => c, "class" => cl, "meaning" => m)
                         for (c, cl, m) in _EXIT_CODE_TAXONOMY],
    )
end

# ── Document builders ─────────────────────────────────────────

function _schema_arg(a::Argument)
    return Dict{String,Any}(
        "name" => a.name,
        "type" => _type_str(a.type),
        "required" => a.required,
        "default" => _default_json(a.default),
        "description" => a.description,
    )
end

function _schema_opt(o::Option)
    return Dict{String,Any}(
        "name" => o.name,
        "short" => o.short,
        "type" => _type_str(o.type),
        "default" => _default_json(o.default),
        "choices" => o.choices,
        "description" => o.description,
    )
end

function _schema_flag(f::Flag)
    return Dict{String,Any}(
        "name" => f.name,
        "short" => f.short,
        "description" => f.description,
    )
end

function _schema_leaf(leaf::LeafCommand, path::Vector{String})
    return Dict{String,Any}(
        "path" => path,
        "name" => leaf.name,
        "description" => leaf.description,
        "args" => [_schema_arg(a) for a in leaf.args],
        "options" => [_schema_opt(o) for o in leaf.options],
        "flags" => [_schema_flag(f) for f in leaf.flags],
        # W5/#140 (additive): machine-actionable invocation + result contract
        "input_schema" => _input_schema(leaf, path),
        "tables" => _registry_tables(path, leaf),
    )
end

function _schema_node(node::NodeCommand, path::Vector{String})
    cmds = Any[]
    for name in sort!(collect(keys(node.subcmds)))
        sub = node.subcmds[name]
        # Hide snake_case aliases from machine schema (C044); primary path only
        is_hidden_alias(name, sub) && continue
        sp = vcat(path, [name])
        if sub isa LeafCommand
            push!(cmds, Dict{String,Any}(
                "name" => name,
                "kind" => "leaf",
                "description" => sub.description,
            ))
        else
            push!(cmds, Dict{String,Any}(
                "name" => name,
                "kind" => "node",
                "description" => sub.description,
                "commands" => _schema_node(sub, sp)["commands"],
            ))
        end
    end
    doc = Dict{String,Any}(
        "name" => node.name,
        "path" => path,
        "description" => node.description,
        "commands" => cmds,
    )
    if isempty(path)
        # Root doc only: the full output contract (W5/#140)
        doc["contract"] = _contract_doc()
    end
    return doc
end

function _resolve_schema_path(root::NodeCommand, parts::Vector{String})
    isempty(parts) && return root, String[]
    node = root
    path = String[]
    for (i, p) in enumerate(parts)
        haskey(node.subcmds, p) || throw(CliError("usage/unknown-command",
            "schema: unknown command path segment '$p' under $(join(path, " "))"))
        sub = node.subcmds[p]
        push!(path, p)
        if sub isa LeafCommand
            i < length(parts) && throw(CliError("usage/unknown-command",
                "schema: '$p' is a leaf; extra path $(join(parts[i+1:end], " "))"))
            return sub, path
        end
        node = sub
    end
    return node, path
end

function _schema_cmd(; path_parts::Vector{String}=String[], format::String="json",
                       output::String="", docs::Bool=false, kwargs...)
    # path_parts injected by dispatch_schema (all positional tokens after `schema`)
    target, resolved = _resolve_schema_path(APP.root, path_parts)
    doc = if target isa LeafCommand
        _schema_leaf(target, resolved)
    else
        _schema_node(target, resolved)
    end
    # --docs: embed the agent guide verbatim (root or any command path)
    docs && (doc["docs"] = _AGENT_GUIDE_MD)
    # Raw JSON — bypass envelope (schema document is the payload)
    json_str = JSON3.write(doc)
    if isempty(output)
        println(json_str)
    else
        _validate_output_path(output)
        open(output, "w") do io
            write(io, json_str)
            println(io)
        end
        _status("Results written to $output")
    end
    return nothing
end

"""Dispatch `schema [path…] [--docs] [--output=…]` without fixed arity (path is varargs)."""
function dispatch_schema(args::Vector{String}; prog::String="friedman schema")
    leaf = register_schema_command!()
    if _wants_help(args)
        print_help(stdout, leaf; prog=prog)
        return
    end
    # Split path tokens from options. FLAGS never consume the next token —
    # the old splitter ate the following path segment for ANY dash token, so
    # `schema --docs estimate var` consumed `estimate` as --docs's value (D-6).
    flag_tokens = Set{String}()
    for f in leaf.flags
        push!(flag_tokens, "--" * f.name)
        isempty(f.short) || push!(flag_tokens, "-" * f.short)
    end
    path_parts = String[]
    opt_tokens = String[]
    i = 1
    while i <= length(args)
        tok = args[i]
        if startswith(tok, "-")
            push!(opt_tokens, tok)
            # consume option value if present (never for a known flag)
            if !(tok in flag_tokens) && !contains(tok, "=") &&
               i + 1 <= length(args) && _looks_like_value(args[i+1])
                push!(opt_tokens, args[i+1])
                i += 1
            end
        else
            push!(path_parts, tok)
        end
        i += 1
    end
    parsed = tokenize(opt_tokens)
    bound = bind_args(parsed, leaf)
    return _schema_cmd(; path_parts=path_parts, format=bound.format,
                         output=bound.output, docs=bound.docs)
end

function register_schema_command!()
    return LeafCommand("schema", _schema_cmd;
        args=Argument[],
        options=[
            Option("format"; short="f", type=String, default="json",
                description="Always json (raw schema document)",
                choices=["json"]),
            Option("output"; short="o", type=String, default="",
                description="Write schema JSON to file"),
        ],
        flags=[
            Flag("docs";
                description="Embed the agent guide as a `docs` markdown string"),
        ],
        description="Machine-readable CLI self-description (raw JSON, no envelope)")
end
