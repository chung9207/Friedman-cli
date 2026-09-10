#!/usr/bin/env julia
# W3/#138 drift gate: envelope `data` keys must be registry-declared.
#
# Usage (from repo root):
#   julia --project test/tools/check_table_keys.jl [dump_dir ...]
#
# Always validates every golden envelope under test/golden/; each extra
# argument is a directory of raw envelope dumps (the FRIEDMAN_T3_DUMP_ENVELOPES
# export from a T3 run). For every envelope with a non-empty `data`:
#   - resolve `command` (leading "friedman " stripped) to its registry leaf;
#   - every `data` key must EQUAL a declared TableSpec name, or — when the
#     declaration carries `family=true` — start with "<name>_" (the family
#     prefix rule: `<declared>_<variable-slug>`).
# Also enforces declaration hygiene: every leaf except `completions *` must
# declare at least one table, and declared names must be [a-z0-9_]+.
#
# Exit 1 on any violation. This is the gate that makes TableSpec normative —
# before W3 the declarations were never read at runtime and had drifted on
# nearly every leaf.

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using CSV, DataFrames, JSON3, PrettyTables, TOML, Random
using LinearAlgebra: eigvals, diag, I, svd, diagm
using Statistics: mean, median, var, quantile
using Dates

const ROOT = dirname(dirname(@__DIR__))

include(joinpath(ROOT, "test", "mocks.jl"))
using .MacroEconometricModels

if !@isdefined(tf_unicode_rounded)
    const tf_unicode_rounded = text_table_borders__unicode_rounded
end

include(joinpath(ROOT, "src", "output", "errors.jl"))
include(joinpath(ROOT, "src", "io.jl"))
include(joinpath(ROOT, "src", "output", "envelope.jl"))
include(joinpath(ROOT, "src", "output", "render.jl"))
include(joinpath(ROOT, "src", "config.jl"))
const FRIEDMAN_VERSION = VersionNumber(TOML.parsefile(joinpath(ROOT, "Project.toml"))["version"])

include(joinpath(ROOT, "src", "cli", "types.jl"))
include(joinpath(ROOT, "src", "cli", "parser.jl"))
include(joinpath(ROOT, "src", "cli", "help.jl"))
include(joinpath(ROOT, "src", "cli", "dispatch.jl"))
include(joinpath(ROOT, "src", "commands", "shared.jl"))
include(joinpath(ROOT, "src", "model_handle.jl"))
include(joinpath(ROOT, "src", "handles.jl"))
include(joinpath(ROOT, "src", "registry", "spec.jl"))
include(joinpath(ROOT, "src", "registry", "adapter.jl"))
include(joinpath(ROOT, "src", "commands", "estimate.jl"))
include(joinpath(ROOT, "src", "commands", "test.jl"))
include(joinpath(ROOT, "src", "commands", "irf.jl"))
include(joinpath(ROOT, "src", "commands", "fevd.jl"))
include(joinpath(ROOT, "src", "commands", "hd.jl"))
include(joinpath(ROOT, "src", "commands", "forecast.jl"))
include(joinpath(ROOT, "src", "commands", "fitted.jl"))
include(joinpath(ROOT, "src", "commands", "filter.jl"))
include(joinpath(ROOT, "src", "commands", "data.jl"))
include(joinpath(ROOT, "src", "commands", "io.jl"))
include(joinpath(ROOT, "src", "commands", "nowcast.jl"))
include(joinpath(ROOT, "src", "commands", "dsge.jl"))
include(joinpath(ROOT, "src", "commands", "did.jl"))
include(joinpath(ROOT, "src", "commands", "multipliers.jl"))
include(joinpath(ROOT, "src", "commands", "policy.jl"))
include(joinpath(ROOT, "src", "commands", "spectral.jl"))
include(joinpath(ROOT, "src", "commands", "schema.jl"))
include(joinpath(ROOT, "src", "commands", "model.jl"))
include(joinpath(ROOT, "src", "commands", "completions.jl"))
include(joinpath(ROOT, "src", "commands", "serve.jl"))
include(joinpath(ROOT, "src", "commands", "show.jl"))

# Populate REGISTRY (register! runs inside each register function)
register_estimate_commands!()
register_test_commands!()
register_irf_commands!()
register_fevd_commands!()
register_hd_commands!()
register_forecast_commands!()
register_predict_commands!()
register_residuals_commands!()
register_filter_commands!()
register_data_commands!()
register_io_commands!()
register_nowcast_commands!()
register_dsge_commands!()
register_did_commands!()
register_multipliers_commands!()
register_policy_commands!()
register_spectral_commands!()
register_model_commands!()
register_completions_commands!()
register_serve_commands!()
register_show_commands!()

# Dedup by path (last wins — matches generate_cli_reference.jl)
const SPECS = Dict{String,CommandSpec}()
for spec in REGISTRY
    SPECS[join(spec.path, " ")] = spec
end

violations = String[]

# ── Declaration hygiene ──────────────────────────────────────
for (path, spec) in sort!(collect(SPECS); by=first)
    if isempty(spec.tables)
        # Documented no-table leaves (W3/#138): completions emit shell scripts;
        # data load/export/fix/transform write CSV (or a handle) directly and
        # data validate reports on stderr only; serve owns stdout as a JSON-RPC
        # channel (W7/#142). (estimate sdfm left this list at #147 — it now
        # emits a summary.) An explicitly empty declaration on these means
        # "emits nothing", not "forgot".
        startswith(path, "completions") && continue
        path in ("data load", "data export", "data fix", "data transform",
                 "data validate", "serve") && continue
        push!(violations, "$path: declares NO tables (every envelope-emitting leaf must declare its tables)")
        continue
    end
    for t in spec.tables
        occursin(r"^[a-z0-9_]+$", string(t.name)) ||
            push!(violations, "$path: declared table name '$(t.name)' is not a lowercase slug")
        isempty(t.description) &&
            push!(violations, "$path: table '$(t.name)' has an empty description")
    end
end

# ── Envelope validation ──────────────────────────────────────
function check_envelope(file::String)
    doc = try
        JSON3.read(read(file, String))
    catch
        return  # not JSON — not this gate's business
    end
    (doc isa JSON3.Object && haskey(doc, :schema_version) && haskey(doc, :data)) || return
    data = doc.data
    isempty(data) && return  # error envelopes carry empty data
    cmd = replace(string(doc.command), r"^friedman\s+" => "")
    spec = get(SPECS, cmd, nothing)
    if spec === nothing
        push!(violations, "$(basename(file)): command '$cmd' resolves to no registry leaf but carries data")
        return
    end
    for key in string.(keys(data))
        ok = any(spec.tables) do t
            n = string(t.name)
            key == n || (t.family && startswith(key, n * "_"))
        end
        ok || push!(violations,
            "$(basename(file)): $cmd emits '$key' — not declared (declared: " *
            join([string(t.name, t.family ? "_*" : "") for t in spec.tables], ", ") * ")")
    end
end

n_checked = 0
for f in readdir(joinpath(ROOT, "test", "golden"); join=true)
    endswith(f, ".json") || continue
    # render.envelope.json is the renderer-shape unit golden (synthetic
    # add_table! keys, not a leaf capture) — outside this gate's contract
    basename(f) == "render.envelope.json" && continue
    check_envelope(f); global n_checked += 1
end
for dir in ARGS
    isdir(dir) || (println(stderr, "warn: dump dir '$dir' not found"); continue)
    for f in readdir(dir; join=true)
        endswith(f, ".json") || continue
        check_envelope(f); global n_checked += 1
    end
end

println("checked $(length(SPECS)) leaf declarations + $n_checked envelope file(s)")
if isempty(violations)
    println("PASS: all envelope data keys are registry-declared")
else
    println(stderr, "=== check_table_keys: $(length(violations)) violation(s) ===")
    for v in violations
        println(stderr, "  ", v)
    end
    exit(1)
end
