#!/usr/bin/env julia
# Drift gate: every leaf with a data arg/option must declare nonempty data_kinds.
#
# Usage (from repo root):
#   julia --project test/tools/check_handle_kinds.jl
#
# Completions, serve, schema, and show (path slot, not data) are exempt. A data slot with
# empty data_kinds is a declaration bug — Task 7's with_default_csv_kinds safety
# net is what this gate holds.
#
# Also fail when a handle=true `--model` has empty model_types, or a handle=true
# `--result` has empty result_types. Key off `handle`, not the option name:
# FCEVAL_RESULT_OPTION (handle=false) and string `--model` (data validate,
# test adf --model=level) stay exempt.
#
# Exit 1 on any violation.

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

violations = String[]
const EXEMPT_NO_DATA = Set(["serve", "schema", "show"])

for spec in REGISTRY
    path = join(spec.path, " ")
    startswith(path, "completions") && continue
    path in EXEMPT_NO_DATA && continue
    has_data = any(a -> a.name == "data", spec.args) || any(o -> o.name == "data", spec.options)
    if has_data && isempty(spec.data_kinds)
        push!(violations, "$path: has a data slot but empty data_kinds")
    end
    if any(o -> o.handle && o.name == "model", spec.options) && isempty(spec.model_types)
        push!(violations, "$path: handle --model but empty model_types")
    end
    if any(o -> o.handle && o.name == "result", spec.options) && isempty(spec.result_types)
        push!(violations, "$path: handle --result but empty result_types")
    end
end
isempty(violations) || (println.(violations); exit(1))
println("check_handle_kinds: ok")
