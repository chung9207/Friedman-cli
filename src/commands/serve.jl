# MCP server over stdio (C057/#61, W7/#142) — a PROJECTION of the registry,
# not a second implementation. Minimal JSON-RPC 2.0 (JSON3, no external MCP
# dep): initialize · tools/list (every registry leaf; inputSchema from the W5
# generator) · tools/call (argv reconstruction → run_cli with format=json
# forced; the result is the envelope JSON verbatim as text content, isError on
# nonzero exit class). Serial request handling — handlers are not
# thread-audited, and MCP clients serialize by default.
#
# stdout discipline: the JSON-RPC channel owns stdout. Every tool call runs
# under redirect_stdout to a TEMPFILE (Julia 1.12: redirect_stdout(IOBuffer)
# does not work) while responses are written to the `output` IO captured at
# loop start — so a redirected global stdout never swallows a response. MEMs
# logging and all CLI status already target stderr.

const _MCP_PROTOCOL_VERSION = "2024-11-05"

"""Enumerate MCP tools: every registry leaf except `serve` itself (a serve
inside serve would deadlock on stdin). Tool name = path joined with `_`
(`estimate_var`, `dsge_bayes_estimate`); hidden snake aliases are excluded the
same way `schema` excludes them."""
function _mcp_tools()
    tools = Vector{Tuple{String,LeafCommand,Vector{String}}}()
    function walk(node::NodeCommand, path::Vector{String})
        for name in sort!(collect(keys(node.subcmds)))
            sub = node.subcmds[name]
            is_hidden_alias(name, sub) && continue
            isempty(path) && name == "serve" && continue
            p = vcat(path, [name])
            if sub isa LeafCommand
                push!(tools, (join(p, "_"), sub, p))
            else
                walk(sub, p)
            end
        end
    end
    walk(APP.root, String[])
    return tools
end

"""
    _mcp_argv(leaf, path, arguments) → Vector{String}

Reconstruct an exact argv from a tools/call `arguments` object (the inverse of
the `x-cli` annotations): positionals in declared order (stopping at the first
absent one — later positionals cannot follow a gap), then options as
`--name value`, then flags when `true`. Unknown keys are appended as options so
the strict parser answers with its typed did-you-mean usage error. `--format
json` is forced last (when the leaf declares `--format`) so every result — and
every failure — is exactly one envelope.
"""
function _mcp_argv(leaf::LeafCommand, path::Vector{String}, arguments)
    argv = copy(path)
    consumed = Set{Symbol}()
    for a in leaf.args
        k = Symbol(a.name)
        haskey(arguments, k) || break
        push!(argv, string(arguments[k]))
        push!(consumed, k)
    end
    has_format = false
    for o in leaf.options
        o.name == "format" && (has_format = true)
        k = Symbol(o.name)
        (k in consumed || !haskey(arguments, k)) && continue
        o.name == "format" || push!(argv, "--" * o.name, string(arguments[k]))
        push!(consumed, k)
    end
    for f in leaf.flags
        k = Symbol(f.name)
        (k in consumed || !haskey(arguments, k)) && continue
        arguments[k] === true && push!(argv, "--" * f.name)
        push!(consumed, k)
    end
    # Unknown keys → let the strict parser reject them with a suggestion
    for (k, v) in pairs(arguments)
        k in consumed && continue
        v === true ? push!(argv, "--" * string(k)) :
                     push!(argv, "--" * string(k), string(v))
    end
    has_format && append!(argv, ["--format", "json"])
    return argv
end

"""Run one tool call through the full CLI pipeline (`run_cli`) with global
stdout redirected to a tempfile — the envelope (or raw JSON for `schema`) comes
back verbatim as the text content; a nonzero exit class sets `isError`."""
function _mcp_call(leaf::LeafCommand, path::Vector{String}, arguments)
    argv = _mcp_argv(leaf, path, arguments)
    out_path, out_io = mktemp()
    code = try
        redirect_stdout(out_io) do
            run_cli(argv)
        end
    finally
        try; close(out_io); catch; end
    end
    text = try
        read(out_path, String)
    finally
        try; rm(out_path; force=true); catch; end
    end
    return Dict{String,Any}(
        "content" => [Dict{String,Any}("type" => "text", "text" => String(strip(text)))],
        "isError" => Int(code) != 0,
    )
end

_rpc_send(io::IO, msg) = (write(io, JSON3.write(msg)); write(io, '\n'); flush(io))
_rpc_result(id, result) = Dict{String,Any}("jsonrpc" => "2.0", "id" => id, "result" => result)
_rpc_error(id, code::Int, msg::String) = Dict{String,Any}(
    "jsonrpc" => "2.0", "id" => id,
    "error" => Dict{String,Any}("code" => code, "message" => msg))

"""
    _serve_loop(input, output)

Line-delimited JSON-RPC 2.0 loop. `output` is captured up front, so the
per-call global-stdout redirection can never swallow a response. The session
`model://` store lives exactly as long as the loop (`_SERVE_MODEL_STORE`).
"""
function _serve_loop(input::IO, output::IO)
    _SERVE_MODEL_STORE[] = Dict{String,Any}()
    tools = _mcp_tools()
    by_name = Dict{String,Tuple{LeafCommand,Vector{String}}}(
        n => (l, p) for (n, l, p) in tools)
    try
        for line in eachline(input)
            s = strip(line)
            isempty(s) && continue
            req = try
                JSON3.read(s)
            catch
                _rpc_send(output, _rpc_error(nothing, -32700, "parse error"))
                continue
            end
            id = get(req, :id, nothing)
            method = String(get(req, :method, ""))
            if method == "initialize"
                _rpc_send(output, _rpc_result(id, Dict{String,Any}(
                    "protocolVersion" => _MCP_PROTOCOL_VERSION,
                    "capabilities" => Dict{String,Any}("tools" => Dict{String,Any}()),
                    "serverInfo" => Dict{String,Any}(
                        "name" => "friedman", "version" => string(FRIEDMAN_VERSION)))))
            elseif startswith(method, "notifications/")
                # notifications carry no id and get no response
            elseif method == "ping"
                _rpc_send(output, _rpc_result(id, Dict{String,Any}()))
            elseif method == "tools/list"
                _rpc_send(output, _rpc_result(id, Dict{String,Any}(
                    "tools" => [Dict{String,Any}(
                                    "name" => n,
                                    "description" => l.description,
                                    "inputSchema" => _input_schema(l, p))
                                for (n, l, p) in tools])))
            elseif method == "tools/call"
                params = get(req, :params, nothing)
                name = params === nothing ? "" : String(get(params, :name, ""))
                entry = get(by_name, name, nothing)
                if entry === nothing
                    _rpc_send(output, _rpc_error(id, -32602, "unknown tool: $name"))
                else
                    raw_args = params === nothing ? nothing : get(params, :arguments, nothing)
                    arguments = raw_args === nothing ? Dict{Symbol,Any}() : raw_args
                    _rpc_send(output, _rpc_result(id, _mcp_call(entry[1], entry[2], arguments)))
                end
            else
                id === nothing ||
                    _rpc_send(output, _rpc_error(id, -32601, "method not found: $method"))
            end
        end
    finally
        _SERVE_MODEL_STORE[] = nothing
    end
    return nothing
end

function _serve(; mcp::Bool=false, format::String="table", output::String="", kwargs...)
    mcp || throw(CliError("usage/missing",
        "serve requires --mcp (the only supported mode)";
        hint="friedman serve --mcp speaks JSON-RPC 2.0 / Model Context Protocol on stdio"))
    _status("friedman MCP server on stdio (JSON-RPC 2.0, protocol $_MCP_PROTOCOL_VERSION); " *
            "one line per message; Ctrl-D to stop")
    _serve_loop(stdin, stdout)
    return nothing
end

function serve_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["serve"],
            summary="Serve every command as a Model Context Protocol tool over stdio (--mcp)",
            args=ArgSpec[],
            options=OptionSpec[],
            flags=[FlagSpec(name="mcp",
                            description="MCP server: JSON-RPC 2.0 on stdio; tools/list mirrors the registry, tools/call returns the JSON envelope verbatim")],
            # No envelope tables: stdout is the JSON-RPC channel (gate-exempt).
            tables=TableSpec[],
            category="serve",
            handler=wrap_legacy(_serve),
        ),
    ]
end

function register_serve_commands!()
    specs = with_default_csv_kinds(serve_specs())
    register!(specs)
    return to_leaf(specs[1])
end
