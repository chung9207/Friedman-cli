using Documenter
using Friedman

# Agent guide single source (W5/#140): src/commands/agent_guide.md is baked into
# the binary and served by `friedman schema --docs`; the site renders the SAME
# file. Copied here at build time — docs/src/agent-guide.md is generated, never
# hand-edited (it is untracked; edit src/commands/agent_guide.md).
cp(normpath(joinpath(@__DIR__, "..", "src", "commands", "agent_guide.md")),
   joinpath(@__DIR__, "src", "agent-guide.md"); force=true)

makedocs(;
    modules = [Friedman],
    sitename = "Friedman-cli",
    repo = Remotes.GitHub("FriedmanJP", "Friedman-cli"),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Agent Guide" => "agent-guide.md",
        "CLI Reference" => [
            "Overview" => "commands/overview.md",
            # Generated option/arg tables (docs/generate_cli_reference.jl)
            "Generated: estimate" => "commands/generated/estimate.md",
            "Generated: test" => "commands/generated/test.md",
            "Generated: irf" => "commands/generated/irf.md",
            "Generated: fevd" => "commands/generated/fevd.md",
            "Generated: hd" => "commands/generated/hd.md",
            "Generated: forecast" => "commands/generated/forecast.md",
            "Generated: predict" => "commands/generated/predict.md",
            "Generated: residuals" => "commands/generated/residuals.md",
            "Generated: filter" => "commands/generated/filter.md",
            "Generated: data" => "commands/generated/data.md",
            "Generated: io" => "commands/generated/io.md",
            "Generated: nowcast" => "commands/generated/nowcast.md",
            "Generated: dsge" => "commands/generated/dsge.md",
            "Generated: did" => "commands/generated/did.md",
            "Generated: multipliers" => "commands/generated/multipliers.md",
            "Generated: policy" => "commands/generated/policy.md",
            "Generated: spectral" => "commands/generated/spectral.md",
            "Generated: serve" => "commands/generated/serve.md",
            "Generated: show" => "commands/generated/show.md",
            # Workflow guides (hand-written)
            "estimate (guide)" => "commands/estimate.md",
            "test (guide)" => "commands/test.md",
            "irf (guide)" => "commands/irf.md",
            "fevd (guide)" => "commands/fevd.md",
            "hd (guide)" => "commands/hd.md",
            "forecast (guide)" => "commands/forecast.md",
            "predict & residuals (guide)" => "commands/predict_residuals.md",
            "filter (guide)" => "commands/filter.md",
            "data (guide)" => "commands/data.md",
            "io (guide)" => "commands/io.md",
            "nowcast (guide)" => "commands/nowcast.md",
            "dsge (guide)" => "commands/dsge.md",
            "HA-DSGE workflow" => "commands/ha-dsge.md",
            "Not wrapped (v0.11.0)" => "commands/not-wrapped.md",
            "did (guide)" => "commands/did.md",
            "multipliers (guide)" => "commands/multipliers.md",
            "policy (guide)" => "commands/policy.md",
            "favar & sdfm (guide)" => "commands/favar.md",
            "structural breaks (guide)" => "commands/structural-breaks.md",
            "panel unit root (guide)" => "commands/panel-unit-root.md",
            "spectral (guide)" => "commands/spectral.md",
            "panel regression (guide)" => "commands/panel-regression.md",
            "ordered & multinomial (guide)" => "commands/ordered-multinomial.md",
        ],
        "Interactive REPL" => "repl.md",
        "Configuration" => "configuration.md",
        "API Reference" => "api.md",
        "Architecture" => "architecture.md",
        "Documentation rules" => "docrule.md",
    ],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://friedmanjp.github.io/Friedman-cli",
        edit_link = "master",
        # The generated CLI reference is one page per top-level command, and the
        # biggest (estimate: 66 leaves, test: 65) render well past Documenter's
        # 200 KiB default. These are lookup tables an agent greps, not prose —
        # raise the ceiling instead of fragmenting the reference across pages.
        size_threshold = 500 * 2^10,
        size_threshold_warn = 300 * 2^10,
    ),
    # docs_block: internal API surface is large; missing @docs bindings must not fail CI
    warnonly = [:missing_docs, :docs_block],
)

deploydocs(;
    repo = "github.com/FriedmanJP/Friedman-cli.git",
    devbranch = "master",
    push_preview = true,
)
