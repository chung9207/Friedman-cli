# Top-level `model` commands — inspect .jld2 (native, C052) and .fmod (interim) handles

function _model_info(; path::String="", data::String="",
                      output::String="", format::String="table")
    # positional may bind as `path` or (legacy) first free; accept either
    p = !isempty(path) ? path : data
    isempty(p) && throw(CliError("usage/missing-arg", "model info requires a .jld2 or .fmod path"))
    info = endswith(lowercase(p), ".jld2") ? _native_model_info(p) : model_handle_info(p)
    fields = ["path", "magic", "model_type", "cli_version", "mems_version",
              "runtime_cli", "runtime_mems", "dimensions"]
    values = [
        info.path,
        info.magic,
        info.model_type,
        info.cli_version,
        info.mems_version,
        info.runtime_cli,
        info.runtime_mems,
        string(info.dimensions),
    ]
    # W3/#167: the native header carries a free-form note (bundle label / vintage).
    # `.fmod` handles have no note — the row is native-only, never empty filler.
    if :note in propertynames(info) && !isempty(info.note)
        push!(fields, "note")
        push!(values, info.note)
    end
    rows = DataFrame(field=fields, value=values)
    output_result(rows; format=Symbol(format), output=output, title="Model Handle Info")
    return info
end

"""
    _model_reproduce(; model, format, output)

Verify a saved handle against a fresh re-run: `reproduce(model)` re-estimates
from the `ReproManifest` seed recorded by `--seed` and compares field-by-field
(W3/#167, MEMs#769/#786). `matched=true` is bit-for-bit equality;
`matched=missing` (no recorded seed, or a type upstream never implemented
`reproduce` for — MEMs 0.9.3 answers those with a universal missing-verdict
fallback instead of throwing) is an honest "cannot verify", not a pass.
Exit is 0 in all these cases; only an unloadable handle errors.
Works on any loadable handle — `.jld2`, `.fmod`, or a `model://` session handle.
"""
function _model_reproduce(; path::String="", data::String="",
                            output::String="", format::String="table")
    # NB: the positional is `path`, not `model` — wrap_legacy reserves the
    # `model=` kwarg for its --model handle injection (a `.jld2` value there
    # would arrive as a loaded object and break the String binding).
    p = !isempty(path) ? path : data
    isempty(p) && throw(CliError("usage/missing-arg",
        "model reproduce requires a handle path (.jld2, .fmod, or model://)"))
    obj = load_model_dispatch(p)
    rep = try
        MacroEconometricModels.reproduce(obj)
    catch e
        # Defense-in-depth only: MEMs 0.9.3's universal reproduce(x) fallback
        # means this branch is unreachable there (unsupported types report a
        # missing verdict above). It guards a future upstream fallback removal.
        e isa MethodError && (e.f === MacroEconometricModels.reproduce) && throw(CliError(
            "model/unsupported",
            "reproduce is not defined for $(typeof(obj))";
            hint="only randomized results estimated with seed= carry a ReproManifest " *
                 "that reproduce can re-run — re-estimate with --seed"))
        rethrow()
    end
    verdict = rep.matched === missing ? "unverifiable (no recorded seed)" : string(rep.matched)
    summary = DataFrame(
        field=["handle", "model_type", "matched", "seed", "threads", "note"],
        value=[
            p,
            string(nameof(typeof(obj))),
            verdict,
            rep.seed === nothing ? "none recorded" : string(rep.seed),
            "$(rep.threads_captured) → $(rep.threads_current)",
            isempty(rep.note) ? "(none)" : rep.note,
        ],
    )
    output_result(summary; format=Symbol(format), output=output,
                  title="Reproduce Verification",
                  key="model_reproduce_summary")
    if !isempty(rep.fields)
        diffs = DataFrame(
            field=[string(d.name) for d in rep.fields],
            matched=[d.matched for d in rep.fields],
            max_abs_diff=[d.max_abs_diff for d in rep.fields],
        )
        output_result(diffs; format=Symbol(format), output=output,
                      title="Reproduce Field Comparison",
                      key="model_reproduce_fields")
    end
    return rep
end

function model_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["model", "info"],
            summary="Inspect a model handle (.jld2 native or .fmod interim): type, versions, dimensions",
            args=[ArgSpec(name="path", type=String, required=true, default=nothing,
                          description="Path to .jld2 or .fmod handle")],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           choices=["table", "csv", "json"], description="table|csv|json"),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:model_handle_info,
                              description="Handle path, magic, model type, writing and runtime versions, dimensions")],
            category="model",
            handler=wrap_legacy(_model_info),
        ),
        CommandSpec(
            path=["model", "reproduce"],
            summary="Verify a saved handle by re-running its estimator from the recorded seed",
            args=[ArgSpec(name="path", type=String, required=true, default=nothing,
                          description="Handle path (.jld2, .fmod, or model:// session handle)")],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           choices=["table", "csv", "json"], description="table|csv|json"),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:model_reproduce_summary,
                          description="Handle, model type, match verdict, recorded seed, thread counts, note"),
                TableSpec(name=:model_reproduce_fields,
                          description="Per-field re-run comparison (present when the report has field diffs)"),
            ],
            category="model",
            handler=wrap_legacy(_model_reproduce),
        ),
    ]
end

function register_model_commands!()
    specs = with_default_csv_kinds(model_specs())
    register!(specs)
    return build_node("model", specs; description="Model handles: inspect .jld2 (native) and .fmod (interim) files")
end
