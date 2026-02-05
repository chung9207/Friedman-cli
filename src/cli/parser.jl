# Runtime argument parser inspired by Comonicon.jl codegen (see COMONICON_LICENSE)

struct ParseError <: Exception
    message::String
end

Base.showerror(io::IO, e::ParseError) = print(io, "ParseError: ", e.message)

struct ParsedArgs
    positional::Vector{String}
    options::Dict{String,String}
    flags::Set{String}
end

"""
    tokenize(tokens) → ParsedArgs

Parse raw CLI tokens into positional args, options, and flags.
Handles: `--option=value`, `--option value`, `-o value`, `--flag`, `-f`, positional args.
`--` stops option parsing (everything after is positional).
"""
function tokenize(tokens::Vector{String})
    positional = String[]
    options = Dict{String,String}()
    flags = Set{String}()
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
                options[k] = v
            elseif i + 1 <= length(tokens) && !startswith(tokens[i+1], "-")
                options[body] = tokens[i+1]
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
                if i + 1 <= length(tokens) && !startswith(tokens[i+1], "-")
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
    return ParsedArgs(positional, options, flags)
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

"""
    bind_args(parsed, cmd) → (positional_values, option_dict, flag_dict)

Bind parsed tokens to a LeafCommand's declared arguments, options, and flags.
Returns a NamedTuple of all bound values.
"""
function bind_args(parsed::ParsedArgs, cmd::LeafCommand)
    # Bind positional arguments
    pos_values = Dict{Symbol,Any}()
    for (idx, arg) in enumerate(cmd.args)
        if idx <= length(parsed.positional)
            pos_values[Symbol(arg.name)] = convert_value(arg.type, parsed.positional[idx], arg.name)
        elseif arg.required
            throw(ParseError("missing required argument: <$(arg.name)>"))
        else
            pos_values[Symbol(arg.name)] = arg.default
        end
    end

    # Check for excess positional args
    if length(parsed.positional) > length(cmd.args)
        extras = parsed.positional[length(cmd.args)+1:end]
        throw(ParseError("unexpected arguments: $(join(extras, ", "))"))
    end

    # Bind options
    opt_values = Dict{Symbol,Any}()
    for opt in cmd.options
        opt_values[Symbol(replace(opt.name, "-" => "_"))] = resolve_option(parsed, opt)
    end

    # Bind flags
    flag_values = Dict{Symbol,Any}()
    for flag in cmd.flags
        flag_values[Symbol(replace(flag.name, "-" => "_"))] = resolve_flag(parsed, flag)
    end

    return (; pos_values..., opt_values..., flag_values...)
end
