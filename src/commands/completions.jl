# `friedman completions bash|zsh|fish` — print shell completion scripts (C029)

const _COMPLETION_FILES = Dict(
    "bash" => "friedman.bash",
    "zsh"  => "friedman.zsh",
    "fish" => "_friedman.fish",
)

function _completions_root()::String
    # completions/ next to Project.toml (package root)
    return joinpath(dirname(dirname(@__DIR__)), "completions")
end

function _print_completions(; shell::String="bash", output::String="", format::String="table")
    haskey(_COMPLETION_FILES, shell) || throw(CliError(
        "usage/bad-shell",
        "unknown shell '$shell' (want bash|zsh|fish)",
    ))
    path = joinpath(_completions_root(), _COMPLETION_FILES[shell])
    isfile(path) || throw(CliError(
        "env/completions-missing",
        "completion file not found: $path",
        hint="run: julia --project docs/generate_cli_reference.jl",
    ))
    text = read(path, String)
    if !isempty(output)
        open(output, "w") do io
            write(io, text)
        end
        _status("Completions written to $output")
    else
        # completions are a script, not a table — print to stdout (not envelope)
        print(text)
    end
    return nothing
end

function completions_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["completions", "bash"],
            summary="Print bash completion script",
            args=ArgSpec[],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Write script to file instead of stdout"),
            ],
            flags=FlagSpec[],
            tables=TableSpec[],
            category="completions",
            handler=wrap_legacy((; kwargs...) -> _print_completions(; shell="bash", kwargs...)),
        ),
        CommandSpec(
            path=["completions", "zsh"],
            summary="Print zsh completion script",
            args=ArgSpec[],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Write script to file instead of stdout"),
            ],
            flags=FlagSpec[],
            tables=TableSpec[],
            category="completions",
            handler=wrap_legacy((; kwargs...) -> _print_completions(; shell="zsh", kwargs...)),
        ),
        CommandSpec(
            path=["completions", "fish"],
            summary="Print fish completion script",
            args=ArgSpec[],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Write script to file instead of stdout"),
            ],
            flags=FlagSpec[],
            tables=TableSpec[],
            category="completions",
            handler=wrap_legacy((; kwargs...) -> _print_completions(; shell="fish", kwargs...)),
        ),
    ]
end

function register_completions_commands!()
    specs = with_default_csv_kinds(completions_specs())
    register!(specs)
    return build_node("completions", specs; description="Shell completion scripts (bash|zsh|fish)")
end
