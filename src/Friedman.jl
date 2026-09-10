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

module Friedman

using CSV, DataFrames, PrettyTables, JSON3, TOML
using MacroEconometricModels
# Load-bearing (C052): loading JLD2 activates MacroEconometricModelsJLD2Ext, which
# backs MEMs `save_model`/`load_model` (native `.jld2` handles). Unlike FFTW — pulled in
# transitively — a bare JLD2 dep does NOT auto-load at runtime, so import it explicitly.
import JLD2
using LinearAlgebra: eigvals, diag, I, svd
using Statistics: mean, median, var, quantile, std
using Random
using Serialization
using Logging: ConsoleLogger, with_logger, Info, Warn

# CLI engine
include("cli/types.jl")
include("cli/parser.jl")
include("cli/help.jl")
include("cli/dispatch.jl")

# Errors before io (loaders throw CliError)
include("output/errors.jl")
# IO and config
include("io.jl")
include("output/envelope.jl")
include("output/render.jl")
include("config.jl")

# Shared utilities (must come before command files)
include("commands/shared.jl")

# Model handles (.fmod) — after io/errors (uses CliError, _status)
include("model_handle.jl")
include("handles.jl")

# Declarative registry (P2-1) — before command files that emit CommandSpecs
include("registry/spec.jl")
include("registry/adapter.jl")

# Commands (action-first hierarchy)
include("commands/estimate.jl")
include("commands/test.jl")
include("commands/irf.jl")
include("commands/fevd.jl")
include("commands/hd.jl")
include("commands/forecast.jl")
include("commands/fitted.jl")  # predict + residuals collapsed (C025)
include("commands/filter.jl")
include("commands/data.jl")
include("commands/io.jl")           # input-output analysis (C049)
include("commands/nowcast.jl")
include("commands/dsge.jl")
include("commands/did.jl")
include("commands/multipliers.jl")  # multipliers nardl — new top-level (C062b)
include("commands/policy.jl")       # policy counterfactuals — new top-level (W4/#126)
include("commands/spectral.jl")
include("commands/schema.jl")
include("commands/model.jl")       # model info (C029)
include("commands/completions.jl") # completions bash|zsh|fish (C029)
include("commands/serve.jl")       # serve --mcp (C057/#61, W7/#142)
include("commands/show.jl")        # show HANDLE (typed-handles Wave 2)

# REPL (interactive session)
include("repl.jl")

# Single source of truth: Project.toml (F3). Resolved at precompile time;
# include_dependency invalidates the precompile cache when the version bumps.
Base.include_dependency(joinpath(dirname(@__DIR__), "Project.toml"))
const FRIEDMAN_VERSION = let proj = TOML.parsefile(joinpath(dirname(@__DIR__), "Project.toml"))
    VersionNumber(proj["version"])
end

"""
    build_app() -> Entry

Construct the full CLI command tree.
"""
function build_app()
    root_cmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "estimate" => register_estimate_commands!(),
        "test"     => register_test_commands!(),
        "irf"      => register_irf_commands!(),
        "fevd"     => register_fevd_commands!(),
        "hd"       => register_hd_commands!(),
        "forecast"  => register_forecast_commands!(),
        "predict"   => register_predict_commands!(),
        "residuals" => register_residuals_commands!(),
        "filter"    => register_filter_commands!(),
        "data"      => register_data_commands!(),
        "io"        => register_io_commands!(),
        "nowcast"   => register_nowcast_commands!(),
        "dsge"      => register_dsge_commands!(),
        "did"       => register_did_commands!(),
        "multipliers" => register_multipliers_commands!(),
        "policy"    => register_policy_commands!(),
        "spectral"  => register_spectral_commands!(),
        "schema"    => register_schema_command!(),
        "model"     => register_model_commands!(),
        "completions" => register_completions_commands!(),
        "serve"     => register_serve_commands!(),
        "show"      => register_show_commands!(),
    )

    root = NodeCommand("friedman", root_cmds,
        "A macroeconometric analysis toolkit powered by MacroEconometricModels.jl")

    return Entry("friedman", root; version=FRIEDMAN_VERSION)
end

"""Memoized command tree, built once at precompile time (F57)."""
const APP = build_app()

"""
    _mems_logger(io=stderr) → ConsoleLogger

Logger for MacroEconometricModels diagnostics (#348 / C050). Emits `@info` and
above by default; under `--quiet` (`_QUIET[]`) the threshold rises to `@warn`, so
info-level chatter is dropped while warnings and errors are always surfaced.
`@debug` stays below threshold unless the user opts in via `JULIA_DEBUG`. `io` is
injectable so tests can capture the stream.
"""
_mems_logger(io::IO=stderr) = ConsoleLogger(io, _QUIET[] ? Warn : Info)

"""Best-effort `command` for a net-emitted error envelope: resolve the raw
argv's non-dash tokens against the registry tree, stopping at the first token
that is not a subcommand (so data paths/values never leak into `command`)."""
function _net_command(args::Vector{String})
    node = APP.root
    path = String["friedman"]
    for tok in args
        startswith(tok, "-") && continue
        if node isa NodeCommand && haskey(node.subcmds, tok)
            push!(path, tok)
            sub = node.subcmds[tok]
            sub isa LeafCommand && break
            node = sub
        else
            break
        end
    end
    return join(path, " ")
end

"""
    _emit_error_net(args, err, t0, wants_json)

W2/#137 usage-error net: when the raw argv asked for JSON but no envelope has
been rendered — usage/parse/pre-dispatch failures throw BEFORE dispatch_leaf's
envelope exists — emit ONE minimal schema-valid error envelope on stdout, so
`--format json` never yields empty stdout on failure. dispatch_leaf's
render-then-rethrow path sets `_ENVELOPE_EMITTED`, so handler errors are never
double-emitted. stderr text and exit codes are unchanged. The net itself must
never throw (it runs inside the error path), so it swallows its own failures.
"""
function _emit_error_net(args::Vector{String}, err::CliError, t0::UInt64, wants_json::Bool)
    (wants_json && !_ENVELOPE_EMITTED[]) || return nothing
    try
        env = Envelope(command=_net_command(args))
        env.meta = Dict{String,Any}(
            "cli_version" => string(FRIEDMAN_VERSION),
            "julia"       => string(VERSION),
            "seed"        => _SEED[],
            "argv"        => copy(args),
            "elapsed_ms"  => (time_ns() - t0) / 1e6,
        )
        try
            env.meta["mems_version"] = string(pkgversion(MacroEconometricModels))
        catch
            env.meta["mems_version"] = "unknown"
        end
        set_error!(env, err.code, err.message; hint=err.hint)
        render(env, :json, stdout)
        _ENVELOPE_EMITTED[] = true
    catch
        # the net is best-effort by construction
    end
    return nothing
end

"""
    run_cli(args)::Cint

Single entry shared by `main` (dev) and `julia_main` (compiled). Exit codes
(P1-4): 0 ok · 2 usage · 3 data · 4 config · 5 model · 6 env · 1 internal.
"""
function run_cli(args::Vector{String})::Cint
    # Launch REPL if "repl" is the first argument
    if !isempty(args) && args[1] == "repl"
        start_repl()
        return Cint(0)
    end

    app = APP
    # W2/#137: decide the error-net format from the RAW argv, before anything
    # that can throw a usage error runs — tokenize/bind_args, and even
    # _extract_global_flags! (a bad --seed throws inside it).
    _ENVELOPE_EMITTED[] = false
    wants_json = _argv_wants_json(args)
    t0 = time_ns()
    try
        remaining = _extract_global_flags!(copy(args))
        _LAST_ARGV[] = copy(args)
        # Route MEMs @info/@warn/@error to stderr (#348 / C050); --quiet drops
        # @info, keeps @warn. @debug stays off unless JULIA_DEBUG is set.
        with_logger(_mems_logger()) do
            dispatch(app, remaining)
        end
        return Cint(0)
    catch e
        if e isa CliError
            _status_styled("Error: "; bold=true, color=:red)
            println(stderr, sprint(showerror, e))
            _emit_error_net(args, e, t0, wants_json)
            return Cint(exit_class(e))
        elseif e isa ParseError || e isa DispatchError
            _status_styled("Error: "; bold=true, color=:red)
            println(stderr, e.message)
            code = e isa DispatchError ? "usage/unknown-command" : "usage/parse"
            _emit_error_net(args, CliError(code, e.message), t0, wants_json)
            return Cint(2)  # usage
        else
            # MEMs domain error (MacroModelError hierarchy) → typed exit (C050)
            mapped = _domain_error_class(e)
            if mapped !== nothing
                _status_styled("Error: "; bold=true, color=:red)
                println(stderr, sprint(showerror, mapped))
                _emit_error_net(args, mapped, t0, wants_json)
                return Cint(exit_class(mapped))
            end
            _status_styled("Error: "; bold=true, color=:red)
            println(stderr, sprint(showerror, e))
            println(stderr, "this is likely a bug — please report")
            _emit_error_net(args, CliError("internal/error", sprint(showerror, e)), t0, wants_json)
            return Cint(1)
        end
    finally
        _QUIET[] = false
    end
end

"""
    main(args=ARGS)

Entry point: dispatch and exit non-zero on failure.
"""
function main(args::Vector{String}=ARGS)
    code = run_cli(args)
    code == 0 || exit(Int(code))
    return nothing
end

"""
    julia_main()::Cint

Entry point for PackageCompiler standalone executables.
"""
julia_main()::Cint = run_cli(ARGS)

export main, build_app, julia_main, run_cli

end # module Friedman
