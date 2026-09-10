# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Runtime argument parser inspired by Comonicon.jl codegen (see COMONICON_LICENSE)

struct ParseError <: Exception
    message::String
end

Base.showerror(io::IO, e::ParseError) = print(io, "ParseError: ", e.message)

struct DispatchError <: Exception
    message::String
end

Base.showerror(io::IO, e::DispatchError) = print(io, "DispatchError: ", e.message)

struct ParsedArgs
    positional::Vector{String}
    options::Dict{String,String}
    flags::Set{String}
    multi::Dict{String,Vector{String}}  # repeatable options e.g. --set (C030)
end
ParsedArgs(pos, opts, flags) = ParsedArgs(pos, opts, flags, Dict{String,Vector{String}}())

"""True if `t` looks like an option value (not a flag), including negative numbers (F2)."""
_looks_like_value(t::AbstractString) = !startswith(t, "-") || occursin(r"^-(\.?\d)", t)

# Option names that may be repeated (`--set a=1 --set b=2`)
const _MULTI_OPTIONS = Set(["set"])

"""
    tokenize(tokens) → ParsedArgs

Parse raw CLI tokens into positional args, options, and flags.
Handles: `--option=value`, `--option value`, `-o value`, `--flag`, `-f`, positional args.
`--` stops option parsing (everything after is positional).
Negative numeric values (`-0.5`, `-3`) bind as option values (F2).
"""
function tokenize(tokens::Vector{String})
    positional = String[]
    options = Dict{String,String}()
    flags = Set{String}()
    multi = Dict{String,Vector{String}}()
    i = 1
    stop_parsing = false
    while i <= length(tokens)
        tok = tokens[i]
        if stop_parsing
            push!(positional, tok)
            i += 1
        elseif tok == "--"
            stop_parsing = true
            i += 1
        elseif startswith(tok, "--")
            body = tok[3:end]
            if contains(body, '=')
                k, v = split(body, '='; limit=2)
                if k in _MULTI_OPTIONS
                    push!(get!(multi, k, String[]), v)
                else
                    options[k] = v
                end
            elseif i + 1 <= length(tokens) && _looks_like_value(tokens[i+1])
                if body in _MULTI_OPTIONS
                    push!(get!(multi, body, String[]), tokens[i+1])
                else
                    options[body] = tokens[i+1]
                end
                i += 1
            else
                # Treat as flag
                push!(flags, body)
            end
            i += 1
        elseif startswith(tok, "-") && length(tok) > 1
            short = tok[2:end]
            if length(short) == 1
                # Single short option
                if i + 1 <= length(tokens) && _looks_like_value(tokens[i+1])
                    options[short] = tokens[i+1]
                    i += 1
                else
                    push!(flags, short)
                end
            else
                # Bundled short flags: -abc → flags a, b, c
                for ch in short
                    push!(flags, string(ch))
                end
            end
            i += 1
        else
            push!(positional, tok)
            i += 1
        end
    end
    return ParsedArgs(positional, options, flags, multi)
end

"""
    resolve_option(parsed, opt) → value or default

Look up an Option by its long name or short alias and convert to the target type.
"""
function resolve_option(parsed::ParsedArgs, opt::Option)
    raw = get(parsed.options, opt.name, nothing)
    if isnothing(raw) && !isempty(opt.short)
        raw = get(parsed.options, opt.short, nothing)
    end
    isnothing(raw) && return opt.default
    if opt.choices !== nothing && !(raw in opt.choices)
        throw(ParseError("option --$(opt.name) must be one of $(join(opt.choices, "|")), got '$raw'"))
    end
    return convert_value(opt.type, raw, opt.name)
end

"""
    resolve_flag(parsed, flag) → Bool

Check if a Flag was set (by long name or short alias).
"""
function resolve_flag(parsed::ParsedArgs, flag::Flag)
    flag.name in parsed.flags && return true
    !isempty(flag.short) && flag.short in parsed.flags && return true
    return false
end

"""
    convert_value(T, raw, name) → T

Convert a raw string to the target type.
"""
function convert_value(::Type{T}, raw::String, name::String) where T <: Integer
    v = tryparse(T, raw)
    isnothing(v) && throw(ParseError("option --$name expects an integer, got '$raw'"))
    return v
end

function convert_value(::Type{T}, raw::String, name::String) where T <: AbstractFloat
    v = tryparse(T, raw)
    isnothing(v) && throw(ParseError("option --$name expects a number, got '$raw'"))
    return v
end

function convert_value(::Type{String}, raw::String, ::String)
    return raw
end

function convert_value(::Type{Symbol}, raw::String, ::String)
    return Symbol(raw)
end

function convert_value(::Type{Bool}, raw::String, name::String)
    s = lowercase(raw)
    s in ("true", "1", "yes", "y") && return true
    s in ("false", "0", "no", "n") && return false
    throw(ParseError("option --$name expects a boolean (true/false/1/0/yes/no), got '$raw'"))
end

function convert_value(::Type{T}, raw::String, name::String) where T
    if !hasmethod(tryparse, Tuple{Type{T}, String})
        throw(ParseError("option --$name: cannot convert '$raw' to $T"))
    end
    v = tryparse(T, raw)
    isnothing(v) && throw(ParseError("option --$name: cannot convert '$raw' to $T"))
    return v
end

"""Levenshtein distance (F1 suggestions)."""
function _levenshtein(a::AbstractString, b::AbstractString)
    m, n = length(a), length(b)
    d = zeros(Int, m + 1, n + 1)
    for i in 0:m; d[i+1, 1] = i; end
    for j in 0:n; d[1, j+1] = j; end
    for j in 1:n
        for i in 1:m
            cost = a[i] == b[j] ? 0 : 1
            d[i+1, j+1] = min(d[i, j+1] + 1, d[i+1, j] + 1, d[i, j] + cost)
        end
    end
    return d[m+1, n+1]
end

"""Nearest known option within distance 2, else `nothing`."""
function _nearest(word::AbstractString, cands)
    best = nothing
    best_d = 3
    for c in cands
        d = _levenshtein(word, c)
        if d < best_d
            best_d = d
            best = c
        end
    end
    return best_d <= 2 ? best : nothing
end

"""
    bind_args(parsed, cmd) → NamedTuple

Bind parsed tokens to a LeafCommand's declared arguments, options, and flags.
Unknown options throw ParseError with a did-you-mean hint (F1).
"""
function bind_args(parsed::ParsedArgs, cmd::LeafCommand)
    # Bind positional arguments
    pos_values = Dict{Symbol,Any}()
    # --model PATH.fmod makes <data> optional (C029): model handle skips re-estimation
    model_set = haskey(parsed.options, "model") && !isempty(parsed.options["model"])
    result_set = haskey(parsed.options, "result") && !isempty(parsed.options["result"])

    for (idx, arg) in enumerate(cmd.args)
        if idx <= length(parsed.positional)
            pos_values[Symbol(arg.name)] = convert_value(arg.type, parsed.positional[idx], arg.name)
        elseif arg.required
            if (model_set || result_set) && arg.name == "data"
                pos_values[Symbol(arg.name)] = something(arg.default, "")
            else
                throw(ParseError("missing required argument: <$(arg.name)>"))
            end
        else
            pos_values[Symbol(arg.name)] = arg.default
        end
    end

    # Check for excess positional args
    if length(parsed.positional) > length(cmd.args)
        extras = parsed.positional[length(cmd.args)+1:end]
        throw(ParseError("unexpected arguments: $(join(extras, ", "))"))
    end

    # Unknown option / flag detection (F1)
    known = Set{String}()
    for o in cmd.options
        push!(known, o.name)
        isempty(o.short) || push!(known, o.short)
    end
    for f in cmd.flags
        push!(known, f.name)
        isempty(f.short) || push!(known, f.short)
    end
    push!(known, "help"); push!(known, "h")
    for k in keys(parsed.options)
        if !(k in known)
            sugg = _nearest(k, known)
            hint = sugg === nothing ? "" : " — did you mean --$sugg?"
            throw(ParseError("unknown option --$k$hint"))
        end
    end
    for k in parsed.flags
        if !(k in known)
            sugg = _nearest(k, known)
            hint = sugg === nothing ? "" : " — did you mean --$sugg?"
            throw(ParseError("unknown option --$k$hint"))
        end
    end

    # Bind options
    opt_values = Dict{Symbol,Any}()
    for opt in cmd.options
        key = Symbol(replace(opt.name, "-" => "_"))
        if opt.name in _MULTI_OPTIONS
            # Repeatable: prefer multi-vector; fall back to single options entry
            vals = get(parsed.multi, opt.name, String[])
            if isempty(vals)
                single = get(parsed.options, opt.name, nothing)
                vals = isnothing(single) ? String[] : String[single]
            end
            opt_values[key] = vals
        else
            opt_values[key] = resolve_option(parsed, opt)
        end
    end

    # Bind flags
    flag_values = Dict{Symbol,Any}()
    for flag in cmd.flags
        flag_values[Symbol(replace(flag.name, "-" => "_"))] = resolve_flag(parsed, flag)
    end

    return (; pos_values..., opt_values..., flag_values...)
end
