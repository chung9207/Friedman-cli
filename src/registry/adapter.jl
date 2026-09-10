# CommandSpec → LeafCommand / NodeCommand adapter (P2-1)

# Names/shorts the pre-dispatch layer owns (#117): `--help`/`-h` fire before
# tokenization on every leaf (`_wants_help`), and `--version`/`-V`, `--warranty`,
# `--conditions` are leading-only globals. A spec claiming one would be dead or
# shadowed on every invocation — refuse at registration (build_app), not in
# production. 48 leaves shipped an unreachable `-h` horizon short this way.
const _RESERVED_OPTION_NAMES = ("help", "version", "warranty", "conditions")
const _RESERVED_SHORTS = ("h", "V")

function _check_reserved(kind::String, name::String, short::String)
    name in _RESERVED_OPTION_NAMES && error(
        "registry: $kind '--$name' collides with a pre-dispatch global; " *
        "pick another name (e.g. '--$name-file')")
    short in _RESERVED_SHORTS && error(
        "registry: $kind '--$name' claims reserved short '-$short' " *
        "(help/version fire before tokenization, so the short can never bind); drop it")
    return nothing
end

function _to_argument(a::ArgSpec)
    return Argument(a.name; type=a.type, required=a.required,
                    default=a.default, description=a.description)
end

function _to_option(o::OptionSpec)
    _check_reserved("option", o.name, o.short)
    return Option(o.name; short=o.short, type=o.type, default=o.default,
                  description=o.description, choices=o.choices)
end

function _to_flag(f::FlagSpec)
    _check_reserved("flag", f.name, f.short)
    return Flag(f.name; short=f.short, description=f.description)
end

"""True when `v` is a nonempty string (handle path) or a loaded object."""
_slot_set(v) = v isa AbstractString ? !isempty(v) : v !== nothing

function _kw_str(kwargs::Dict{Symbol,Any}, key::Symbol)
    v = get(kwargs, key, "")
    return v isa AbstractString ? String(v) : ""
end

function _load_typed_handle(path::String, allowed::Vector{Symbol}, code::String; slot::String)
    if !startswith(path, ":") && !startswith(path, "model://")
        _validate_input_path(path)
    end
    obj = load_model_dispatch(path)
    n = nameof(typeof(obj))
    if n ∉ allowed
        other = code == "model/wrong-kind" ? "--result" : "--model"
        throw(CliError(code,
            "$path is a $n handle; this command accepts $(join(allowed, ", "))";
            hint="this is a $n; pass it as $other, not --$slot"))
    end
    return obj
end

function _nt_save_slot(nt, key::Symbol, path::String, flag::String)
    haskey(nt, key) || throw(CliError("model/no-result",
        "cannot $flag: handler NamedTuple has no $key field";
        hint="return (; model, result) or omit $flag"))
    obj = getfield(nt, key)
    isnothing(obj) && throw(CliError("model/no-result",
        "cannot $flag: handler returned nothing";
        hint="only estimate/solve commands produce savable models"))
    save_model_dispatch(path, obj)
    return nothing
end

function _save_bare(path::String, obj, flag::String)
    isnothing(obj) && throw(CliError("model/no-result",
        "cannot $flag: handler returned nothing";
        hint="only estimate/solve commands produce savable models"))
    save_model_dispatch(path, obj)
    return nothing
end

"""
    wrap_legacy(handler) → (ctx::CmdContext) -> Any

Adapt a legacy kwargs handler `_foo(; data, lags, ...)` to the CmdContext style.

Handle I/O (C029 + typed-handles):
- `--model` / `--result` stems → load, type-check, inject as objects
- `--save-model` / `--save-result` → persist the handler return (or NamedTuple fields)
"""
function wrap_legacy(handler::Function)
    return function (ctx::CmdContext)
        kwargs = Dict{Symbol,Any}()
        for (k, v) in ctx.args
            kwargs[k] = v
        end
        for (k, v) in ctx.opts
            kwargs[k] = v
        end
        for (k, v) in ctx.flags
            kwargs[k] = v
        end
        # format/output may live only in opts already; ensure present
        kwargs[:format] = string(ctx.fmt)
        kwargs[:output] = ctx.output

        # --result XOR --model XOR compute-from-data (producing leaves only).
        result_str = _kw_str(kwargs, :result)
        if !isempty(ctx.spec.result_types) && !isempty(result_str)
            if _slot_set(get(kwargs, :model, "")) || _slot_set(get(kwargs, :data, ""))
                throw(CliError("usage/invalid",
                    "--result cannot be combined with --model or a data path";
                    hint="omit --result to compute, or pass only --result to re-render"))
            end
        end

        # Stem-resolve + type-check the data slot. data= stays a String.
        if haskey(kwargs, :data) && kwargs[:data] isa AbstractString && !isempty(kwargs[:data])
            resolved = resolve_stem(String(kwargs[:data]); slot=:data)
            kwargs[:data] = resolved
            kinds = ctx.spec.data_kinds
            if !isempty(kinds) && _is_handle_path(resolved)
                # Confine filesystem handles on the resolved path; skip :example and model://.
                if !startswith(resolved, ":") && !startswith(resolved, "model://")
                    _validate_input_path(resolved)
                end
                obj = load_model_dispatch(resolved)
                k = _data_kind_of(obj)
                if k ∉ kinds
                    throw(CliError("data/wrong-kind",
                        "$resolved is a $k handle ($(nameof(typeof(obj)))); this command accepts $(join(kinds, ", "))";
                        hint="data import --kind timeseries, or pick a leaf that accepts $k"))
                end
            elseif !isempty(kinds) && !_is_handle_path(resolved) && !startswith(resolved, ":")
                :csv ∉ kinds && throw(CliError("data/wrong-kind",
                    "$resolved is CSV; this command does not accept :csv";
                    hint="data import first"))
            end
        end

        # --save-model / --save-result are never handler kwargs
        save_path = resolve_save_path(string(get(kwargs, :save_model, "")))
        save_result_path = resolve_save_path(string(get(kwargs, :save_result, "")))
        delete!(kwargs, :save_model)
        delete!(kwargs, :save_result)

        # --result STEM → loaded object when this leaf declares result_types.
        if haskey(kwargs, :result)
            rp = kwargs[:result]
            if rp isa AbstractString
                if isempty(rp)
                    delete!(kwargs, :result)
                elseif !isempty(ctx.spec.result_types)
                    resolved = resolve_stem(String(rp); slot=:result)
                    kwargs[:result] = _load_typed_handle(resolved, ctx.spec.result_types,
                        "data/wrong-result"; slot="result")
                    get!(kwargs, :data, "")
                end
            end
        end

        # --model PATH → loaded object; empty → drop so handler default applies.
        # `.jld2` (native, C052) and `.fmod` (interim, C029) paths are model
        # handles; `model://` (W7/#142) is the in-memory serve-session handle.
        # Builtin names and .jl/.toml model files (e.g. `dsge ha huggett`,
        # `dsge solve rbc.toml`) must pass through as strings (C040).
        # When model_types is nonempty, stem-resolve (no CSV fallback) first so
        # `--model var` finds var.jld2; then type-check the loaded object.
        if haskey(kwargs, :model)
            mp = kwargs[:model]
            if mp isa AbstractString
                if isempty(mp)
                    delete!(kwargs, :model)
                else
                    mp = String(mp)
                    if !isempty(ctx.spec.model_types)
                        mp = resolve_stem(mp; slot=:result)
                        kwargs[:model] = mp
                    end
                    if endswith(lowercase(mp), ".fmod") ||
                       endswith(lowercase(mp), ".jld2") ||
                       startswith(mp, "model://")
                        if !isempty(ctx.spec.model_types)
                            kwargs[:model] = _load_typed_handle(mp, ctx.spec.model_types,
                                "model/wrong-kind"; slot="model")
                        else
                            kwargs[:model] = load_model_dispatch(mp)
                        end
                        # allow missing data positional when handle supplies the model
                        get!(kwargs, :data, "")
                    end
                end
            end
        end

        # Config ergonomics (C030): merge file < config-json < --set; --strict
        config_json = string(get(kwargs, :config_json, ""))
        set_raw = get(kwargs, :set, String[])
        set_vals = set_raw isa AbstractString ?
            (isempty(set_raw) ? String[] : String[String(set_raw)]) :
            String[String(s) for s in set_raw]
        strict = get(kwargs, :strict, false) === true
        delete!(kwargs, :config_json)
        delete!(kwargs, :set)
        delete!(kwargs, :strict)
        prev_strict = _CONFIG_STRICT[]
        _CONFIG_STRICT[] = strict
        try
            config_path = string(get(kwargs, :config, ""))
            if !isempty(config_path) || !isempty(config_json) || !isempty(set_vals)
                merged = merge_config(config_path; config_json=config_json,
                                      set=set_vals, strict=strict)
                kwargs[:config] = write_merged_config_toml(merged)
            end

            result = handler(; kwargs...)

            nt_split = result isa NamedTuple &&
                (haskey(result, :model) || haskey(result, :result))
            if nt_split
                isempty(save_path) || _nt_save_slot(result, :model, save_path, "--save-model")
                isempty(save_result_path) || _nt_save_slot(result, :result, save_result_path, "--save-result")
            elseif !isempty(save_path) && !isempty(save_result_path)
                throw(CliError("usage/invalid",
                    "handler returned a single object; use a NamedTuple (; model, result)";
                    hint="pass only --save-model or only --save-result, or return (; model, result)"))
            else
                isempty(save_path) || _save_bare(save_path, result, "--save-model")
                isempty(save_result_path) || _save_bare(save_result_path, result, "--save-result")
            end
            return result
        finally
            _CONFIG_STRICT[] = prev_strict
        end
    end
end

"""
    to_leaf(spec::CommandSpec) → LeafCommand

Bridge a declarative CommandSpec to the existing LeafCommand engine.
"""
function to_leaf(spec::CommandSpec)
    leaf_name = isempty(spec.path) ? "command" : spec.path[end]
    args = [_to_argument(a) for a in spec.args]
    options = [_to_option(o) for o in spec.options]
    flags = [_to_flag(f) for f in spec.flags]

    function wrapper(; kwargs...)
        # Partition bound kwargs into args / opts / flags
        arg_names = Set(Symbol(a.name) for a in spec.args)
        # option names use underscore form from bind_args
        opt_map = Dict{Symbol,String}()  # bound_key => option name
        for o in spec.options
            opt_map[Symbol(replace(o.name, "-" => "_"))] = o.name
        end
        flag_keys = Set(Symbol(replace(f.name, "-" => "_")) for f in spec.flags)

        a = Dict{Symbol,Any}()
        o = Dict{Symbol,Any}()
        fl = Dict{Symbol,Bool}()
        for (k, v) in pairs(kwargs)
            if k in arg_names
                a[k] = v
            elseif k in flag_keys
                fl[k] = v === true
            elseif haskey(opt_map, k)
                o[k] = v
            else
                o[k] = v  # extras (e.g. format from globals)
            end
        end
        fmt = Symbol(get(o, :format, get(kwargs, :format, "table")))
        output = string(get(o, :output, get(kwargs, :output, "")))
        env = envelope_active() ? _ENVELOPE[] : Envelope(command=join(spec.path, " "))
        status_fn = (parts...) -> _status(parts...)
        ctx = CmdContext(a, o, fl, fmt, output, env, status_fn, spec)
        return spec.handler(ctx)
    end

    return LeafCommand(leaf_name, wrapper;
        args=args, options=options, flags=flags, description=spec.summary)
end

"""
    _alias_leaf(leaf, alias, canonical) → LeafCommand

Hidden snake_case alias for a kebab primary (C044 / F16).
Registered under `alias` in the subcmds dict; `leaf.name` stays `canonical` so
help/schema can hide aliases where `subcmds` key ≠ leaf.name.
Prints a one-line stderr deprecation on use (not suppressed into stdout).
"""
function _alias_leaf(leaf::LeafCommand, alias::String, canonical::String)
    inner = leaf.handler
    function wrapper(; kwargs...)
        printstyled(stderr, "warning: '$alias' is deprecated; use '$canonical' (removed in v1.0)\n";
                    color=:yellow)
        return inner(; kwargs...)
    end
    return LeafCommand(canonical, wrapper;
        args=leaf.args, options=leaf.options, flags=leaf.flags,
        description=leaf.description)
end

"""Register primary leaf plus any CommandSpec.aliases under a subcmds dict."""
function _register_leaf!(cmds::Dict{String,Union{NodeCommand,LeafCommand}},
                         primary::String, leaf::LeafCommand, aliases::Vector{String})
    cmds[primary] = leaf
    for alias in aliases
        alias == primary && continue
        haskey(cmds, alias) && error("alias '$alias' collides with existing subcommand")
        cmds[alias] = _alias_leaf(leaf, alias, primary)
    end
    return cmds
end

"""
    build_node(name, specs; description="") → NodeCommand

Build a node from CommandSpecs sharing path prefix `name`.
Supports depth-2 (`["estimate","var"]`) and depth-3 (`["dsge","bayes","irf"]`) paths.
"""
function build_node(name::String, specs::Vector{CommandSpec}; description::String="")
    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}()
    # Group depth-3 specs by middle segment
    nested = Dict{String,Vector{CommandSpec}}()
    for spec in specs
        length(spec.path) >= 2 || error("spec path must be [node, leaf, ...]: $(spec.path)")
        spec.path[1] == name || error("spec path[1]=$(spec.path[1]) != node $name")
        if length(spec.path) == 2
            leaf = to_leaf(spec)
            _register_leaf!(subcmds, spec.path[2], leaf, spec.aliases)
        elseif length(spec.path) == 3
            mid = spec.path[2]
            push!(get!(nested, mid, CommandSpec[]), spec)
        else
            error("registry paths deeper than 3 not yet implemented: $(spec.path)")
        end
    end
    for (mid, nspecs) in nested
        # child leaves: path[3] becomes leaf name under mid node
        child_cmds = Dict{String,Union{NodeCommand,LeafCommand}}()
        for spec in nspecs
            leaf = to_leaf(spec)
            _register_leaf!(child_cmds, spec.path[3], leaf, spec.aliases)
        end
        # description from first child category or mid name
        subcmds[mid] = NodeCommand(mid, child_cmds, mid)
    end
    return NodeCommand(name, subcmds, description)
end
