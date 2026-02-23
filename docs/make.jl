using Documenter
using Friedman

makedocs(;
    modules = [Friedman],
    sitename = "Friedman-cli",
    repo = Remotes.GitHub("chung9207", "Friedman-cli"),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "CLI Reference" => [
            "Overview" => "commands/overview.md",
            "estimate" => "commands/estimate.md",
            "test" => "commands/test.md",
            "irf" => "commands/irf.md",
            "fevd" => "commands/fevd.md",
            "hd" => "commands/hd.md",
            "forecast" => "commands/forecast.md",
            "predict & residuals" => "commands/predict_residuals.md",
            "filter" => "commands/filter.md",
            "data" => "commands/data.md",
            "nowcast" => "commands/nowcast.md",
        ],
        "Configuration" => "configuration.md",
        "API Reference" => "api.md",
        "Architecture" => "architecture.md",
    ],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://chung9207.github.io/Friedman-cli",
        edit_link = "master",
    ),
    warnonly = [:missing_docs],
)

deploydocs(;
    repo = "github.com/chung9207/Friedman-cli.git",
    devbranch = "master",
    push_preview = true,
)
