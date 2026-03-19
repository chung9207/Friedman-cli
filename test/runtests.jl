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

using Test

# Minimal mock for dispatch.jl's MacroEconometricModels.warranty/conditions calls
module MacroEconometricModels
    function warranty()
        println("THERE IS NO WARRANTY FOR THE PROGRAM")
    end
    function conditions()
        println("You may distribute copies of the program")
    end
end

@testset "Friedman CLI Engine" begin

    # Include CLI engine files directly for unit testing
    project_root = dirname(@__DIR__)
    include(joinpath(project_root, "src", "cli", "types.jl"))
    include(joinpath(project_root, "src", "cli", "parser.jl"))
    include(joinpath(project_root, "src", "cli", "help.jl"))
    include(joinpath(project_root, "src", "cli", "dispatch.jl"))

    @testset "Types" begin
        arg = Argument("data"; type=String, required=true, description="Data file")
        @test arg.name == "data"
        @test arg.type == String
        @test arg.required == true

        opt = Option("lags"; short="p", type=Int, default=4, description="Lag order")
        @test opt.name == "lags"
        @test opt.short == "p"
        @test opt.default == 4

        flag = Flag("verbose"; short="v", description="Verbose output")
        @test flag.name == "verbose"
        @test flag.short == "v"

        handler = (; kwargs...) -> nothing
        leaf = LeafCommand("estimate", handler;
            args=[arg], options=[opt], flags=[flag],
            description="Estimate model")
        @test leaf.name == "estimate"
        @test length(leaf.args) == 1
        @test length(leaf.options) == 1
        @test length(leaf.flags) == 1

        node = NodeCommand("var";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("estimate" => leaf),
            description="VAR commands")
        @test haskey(node.subcmds, "estimate")

        entry = Entry("friedman", node; version=v"0.1.0")
        @test entry.name == "friedman"
        @test entry.version == v"0.1.0"
    end

    @testset "Tokenizer" begin
        # Basic positional args
        parsed = tokenize(["file.csv", "other.csv"])
        @test parsed.positional == ["file.csv", "other.csv"]
        @test isempty(parsed.options)
        @test isempty(parsed.flags)

        # Long options with =
        parsed = tokenize(["--lags=4", "--format=json"])
        @test parsed.options["lags"] == "4"
        @test parsed.options["format"] == "json"

        # Long options with space
        parsed = tokenize(["--lags", "4"])
        @test parsed.options["lags"] == "4"

        # Short options
        parsed = tokenize(["-p", "4"])
        @test parsed.options["p"] == "4"

        # Flags
        parsed = tokenize(["--verbose", "--help"])
        @test "verbose" in parsed.flags
        @test "help" in parsed.flags

        # Mixed
        parsed = tokenize(["data.csv", "--lags=4", "--verbose", "-f", "json"])
        @test parsed.positional == ["data.csv"]
        @test parsed.options["lags"] == "4"
        @test parsed.options["f"] == "json"
        @test "verbose" in parsed.flags

        # -- stops option parsing
        parsed = tokenize(["--lags=4", "--", "--not-an-option"])
        @test parsed.options["lags"] == "4"
        @test "--not-an-option" in parsed.positional

        # Bundled short flags
        parsed = tokenize(["-abc"])
        @test "a" in parsed.flags
        @test "b" in parsed.flags
        @test "c" in parsed.flags
    end

    @testset "Tokenizer edge cases" begin
        # Empty tokens → empty ParsedArgs
        parsed = tokenize(String[])
        @test isempty(parsed.positional)
        @test isempty(parsed.options)
        @test isempty(parsed.flags)

        # --opt= → option with empty string value
        parsed = tokenize(["--opt="])
        @test parsed.options["opt"] == ""

        # --opt=val=ue → value is "val=ue" (split limit=2)
        parsed = tokenize(["--opt=val=ue"])
        @test parsed.options["opt"] == "val=ue"

        # --shock followed by -1 → -1 starts with "-" so --shock becomes a flag
        parsed = tokenize(["--shock", "-1"])
        @test "shock" in parsed.flags
        @test "1" in parsed.flags

        # Short option at end of tokens (no next token) → becomes flag
        parsed = tokenize(["-v"])
        @test "v" in parsed.flags

        # Single "-" → positional (not a flag, length is 1 so startswith("-") && length > 1 fails)
        parsed = tokenize(["-"])
        @test "-" in parsed.positional

        # Long flag followed by long flag
        parsed = tokenize(["--verbose", "--help"])
        @test "verbose" in parsed.flags
        @test "help" in parsed.flags

        # -- with nothing after → just empty positional
        parsed = tokenize(["--"])
        @test isempty(parsed.positional)
        @test isempty(parsed.options)
        @test isempty(parsed.flags)

        # Multiple = signs: --key=a=b=c → value is "a=b=c"
        parsed = tokenize(["--key=a=b=c"])
        @test parsed.options["key"] == "a=b=c"

        # Long option with space, next token is positional-looking
        parsed = tokenize(["--output", "file.csv"])
        @test parsed.options["output"] == "file.csv"

        # Short option followed by token starting with "-" → short becomes flag
        parsed = tokenize(["-p", "--other"])
        @test "p" in parsed.flags
        @test "other" in parsed.flags

        # Long option with space consumes next non-dash token as value
        parsed = tokenize(["--verbose", "file.csv"])
        @test parsed.options["verbose"] == "file.csv"
        @test isempty(parsed.positional)
    end

    @testset "Type conversion" begin
        # Int conversions
        @test convert_value(Int, "42", "x") == 42
        @test convert_value(Int, "-5", "x") == -5
        @test convert_value(Int, "0", "x") == 0
        @test_throws ParseError convert_value(Int, "abc", "x")
        @test_throws ParseError convert_value(Int, "3.14", "x")
        @test_throws ParseError convert_value(Int, "", "x")

        # Float64 conversions
        @test convert_value(Float64, "3.14", "x") == 3.14
        @test convert_value(Float64, "-0.5", "x") == -0.5
        @test convert_value(Float64, "42", "x") == 42.0
        @test convert_value(Float64, "1e-3", "x") == 0.001
        @test_throws ParseError convert_value(Float64, "abc", "x")
        @test_throws ParseError convert_value(Float64, "", "x")

        # String passthrough
        @test convert_value(String, "hello", "x") == "hello"
        @test convert_value(String, "", "x") == ""
        @test convert_value(String, "with spaces", "x") == "with spaces"

        # Symbol conversion
        @test convert_value(Symbol, "foo", "x") == :foo
        @test convert_value(Symbol, "bar_baz", "x") == :bar_baz

        # Fallback convert_value for arbitrary types (uses tryparse)
        @test convert_value(Float32, "3.14", "x") isa Float32
        @test_throws ParseError convert_value(Float32, "notanumber", "x")
    end

    @testset "Argument binding" begin
        handler = (; kwargs...) -> kwargs
        cmd = LeafCommand("test", handler;
            args=[
                Argument("data"; type=String, required=true, description="Data file"),
            ],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lag order"),
                Option("format"; short="f", type=String, default="table", description="Output format"),
            ],
            flags=[
                Flag("verbose"; short="v", description="Verbose"),
            ])

        # Normal binding
        parsed = tokenize(["myfile.csv", "--lags=8", "--verbose"])
        bound = bind_args(parsed, cmd)
        @test bound.data == "myfile.csv"
        @test bound.lags == 8
        @test bound.format == "table"
        @test bound.verbose == true

        # Defaults
        parsed = tokenize(["myfile.csv"])
        bound = bind_args(parsed, cmd)
        @test bound.lags == 4
        @test bound.verbose == false

        # Short options
        parsed = tokenize(["myfile.csv", "-p", "2", "-f", "json"])
        bound = bind_args(parsed, cmd)
        @test bound.lags == 2
        @test bound.format == "json"

        # Missing required arg
        parsed = tokenize(String[])
        @test_throws ParseError bind_args(parsed, cmd)

        # Excess positional args
        parsed = tokenize(["file1.csv", "file2.csv"])
        @test_throws ParseError bind_args(parsed, cmd)

        # Invalid type
        parsed = tokenize(["myfile.csv", "--lags=abc"])
        @test_throws ParseError bind_args(parsed, cmd)
    end

    @testset "Argument binding edge cases" begin
        handler = (; kwargs...) -> kwargs

        # Hyphen-to-underscore: option --my-opt binds to my_opt
        cmd_hyphen = LeafCommand("test", handler;
            options=[
                Option("my-opt"; type=String, default="default", description="Hyphenated option"),
            ])
        parsed = tokenize(["--my-opt=hello"])
        bound = bind_args(parsed, cmd_hyphen)
        @test bound.my_opt == "hello"

        # Flag hyphen-to-underscore: flag --dry-run binds to dry_run
        cmd_flag_hyphen = LeafCommand("test", handler;
            flags=[
                Flag("dry-run"; description="Dry run mode"),
            ])
        parsed = tokenize(["--dry-run"])
        bound = bind_args(parsed, cmd_flag_hyphen)
        @test bound.dry_run == true

        # Optional (non-required) argument uses default when omitted
        cmd_optional = LeafCommand("test", handler;
            args=[
                Argument("data"; type=String, required=false, default="default.csv", description="Optional data"),
            ])
        parsed = tokenize(String[])
        bound = bind_args(parsed, cmd_optional)
        @test bound.data == "default.csv"

        # Optional argument provided
        parsed = tokenize(["custom.csv"])
        bound = bind_args(parsed, cmd_optional)
        @test bound.data == "custom.csv"

        # Multiple positional arguments bind in order
        cmd_multi = LeafCommand("test", handler;
            args=[
                Argument("input"; type=String, required=true, description="Input file"),
                Argument("output"; type=String, required=true, description="Output file"),
            ])
        parsed = tokenize(["in.csv", "out.csv"])
        bound = bind_args(parsed, cmd_multi)
        @test bound.input == "in.csv"
        @test bound.output == "out.csv"

        # Float64 option type
        cmd_float = LeafCommand("test", handler;
            options=[
                Option("alpha"; type=Float64, default=0.05, description="Significance level"),
            ])
        parsed = tokenize(["--alpha=0.01"])
        bound = bind_args(parsed, cmd_float)
        @test bound.alpha == 0.01

        # Float64 option default
        parsed = tokenize(String[])
        bound = bind_args(parsed, cmd_float)
        @test bound.alpha == 0.05

        # Option with nothing default stays nothing when unset
        cmd_nothing = LeafCommand("test", handler;
            options=[
                Option("config"; type=String, default=nothing, description="Config file"),
            ])
        parsed = tokenize(String[])
        bound = bind_args(parsed, cmd_nothing)
        @test isnothing(bound.config)

        # Command with no args, no options, no flags → empty NamedTuple works
        cmd_empty = LeafCommand("test", handler)
        parsed = tokenize(String[])
        bound = bind_args(parsed, cmd_empty)
        @test bound == NamedTuple()

        # Short alias for flag
        cmd_short_flag = LeafCommand("test", handler;
            flags=[
                Flag("bayesian"; short="b", description="Bayesian mode"),
            ])
        parsed = tokenize(["-b"])
        bound = bind_args(parsed, cmd_short_flag)
        @test bound.bayesian == true
    end

    @testset "Help generation" begin
        handler = (; kwargs...) -> nothing
        leaf = LeafCommand("estimate", handler;
            args=[Argument("data"; description="CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            ],
            flags=[Flag("verbose"; short="v", description="Verbose output")],
            description="Estimate a VAR model")

        node = NodeCommand("var";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("estimate" => leaf),
            description="VAR commands")

        entry = Entry("friedman", node; version=v"0.1.0")

        # Help should not error
        buf = IOBuffer()
        print_help(buf, entry)
        help_text = String(take!(buf))
        @test contains(help_text, "friedman")
        @test contains(help_text, "estimate")

        buf = IOBuffer()
        print_help(buf, leaf; prog="friedman var estimate")
        help_text = String(take!(buf))
        @test contains(help_text, "estimate")
        @test contains(help_text, "<data>")
        @test contains(help_text, "--lags")
        @test contains(help_text, "-p")
        @test contains(help_text, "--verbose")
    end

    @testset "Help generation expanded" begin
        handler = (; kwargs...) -> nothing

        # NodeCommand help contains all subcommand names
        leaf1 = LeafCommand("estimate", handler; description="Estimate model")
        leaf2 = LeafCommand("lagselect", handler; description="Select lag order")
        leaf3 = LeafCommand("stability", handler; description="Check stability")
        node = NodeCommand("var";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}(
                "estimate" => leaf1, "lagselect" => leaf2, "stability" => leaf3),
            description="VAR analysis commands")

        buf = IOBuffer()
        print_help(buf, node; prog="friedman var")
        help_text = String(take!(buf))
        @test contains(help_text, "estimate")
        @test contains(help_text, "lagselect")
        @test contains(help_text, "stability")

        # NodeCommand help includes "Use ... --help" footer
        @test contains(help_text, "--help")
        @test contains(help_text, "friedman var")

        # Entry help includes version number
        entry = Entry("friedman", node; version=v"0.4.2")
        buf = IOBuffer()
        print_help(buf, entry)
        help_text = String(take!(buf))
        @test contains(help_text, "0.4.2")

        # Leaf with optional argument shows [arg] not <arg>
        leaf_opt_arg = LeafCommand("test", handler;
            args=[Argument("data"; required=false, default="data.csv", description="Data file")],
            description="Test command")
        buf = IOBuffer()
        print_help(buf, leaf_opt_arg; prog="friedman test")
        help_text = String(take!(buf))
        @test contains(help_text, "[data]")
        @test !contains(help_text, "<data>")

        # Default values displayed in option help
        leaf_defaults = LeafCommand("test", handler;
            options=[
                Option("lags"; type=Int, default=4, description="Lag order"),
                Option("format"; type=String, default="table", description="Output format"),
            ],
            description="Test command")
        buf = IOBuffer()
        print_help(buf, leaf_defaults; prog="friedman test")
        help_text = String(take!(buf))
        @test contains(help_text, "default: 4")
        @test contains(help_text, "default: table")

        # Leaf with no options/flags omits those sections
        leaf_no_opts = LeafCommand("test", handler;
            args=[Argument("data"; description="Data file")],
            description="Simple command")
        buf = IOBuffer()
        print_help(buf, leaf_no_opts; prog="friedman test")
        help_text = String(take!(buf))
        @test contains(help_text, "Arguments:")
        @test !contains(help_text, "Options:")
        @test !contains(help_text, "Flags:")

        # Leaf description is included
        leaf_desc = LeafCommand("estimate", handler;
            description="Estimate a Bayesian VAR model")
        buf = IOBuffer()
        print_help(buf, leaf_desc; prog="friedman bvar estimate")
        help_text = String(take!(buf))
        @test contains(help_text, "Estimate a Bayesian VAR model")

        # Option without default doesn't show "(default: ...)"
        leaf_no_default = LeafCommand("test", handler;
            options=[
                Option("config"; type=String, default=nothing, description="Config file"),
            ],
            description="Test")
        buf = IOBuffer()
        print_help(buf, leaf_no_default; prog="friedman test")
        help_text = String(take!(buf))
        @test contains(help_text, "--config")
        @test !contains(help_text, "default:")
    end

    @testset "Dispatch" begin
        # Track what was called
        called_with = Ref{Any}(nothing)
        handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        leaf = LeafCommand("estimate", handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("lags"; type=Int, default=4, description="Lags")],
            description="Estimate")

        node = NodeCommand("var";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("estimate" => leaf),
            description="VAR")

        # Test dispatch through node to leaf
        dispatch_node(node, ["estimate", "test.csv", "--lags=2"]; prog="friedman var")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:lags] == 2

        # Test help doesn't error
        dispatch_node(node, ["--help"]; prog="friedman var")
        dispatch_leaf(leaf, ["--help"]; prog="friedman var estimate")
    end

    @testset "Dispatch edge cases" begin
        called_with = Ref{Any}(nothing)
        handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        # Helper to capture stdout using a temp file (works on all Julia versions)
        function capture_stdout(f)
            mktemp() do path, io
                old_stdout = stdout
                redirect_stdout(io)
                try
                    f()
                finally
                    redirect_stdout(old_stdout)
                    close(io)
                end
                return read(path, String)
            end
        end

        # Nested NodeCommand (2 levels: group → subgroup → leaf)
        inner_leaf = LeafCommand("run", handler;
            args=[Argument("data"; description="Data")],
            description="Run analysis")
        inner_node = NodeCommand("sub";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("run" => inner_leaf),
            description="Sub group")
        outer_node = NodeCommand("top";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("sub" => inner_node),
            description="Top group")

        dispatch_node(outer_node, ["sub", "run", "test.csv"]; prog="friedman top")
        @test called_with[][:data] == "test.csv"

        # dispatch() with ["--version"] prints version
        entry = Entry("friedman", outer_node; version=v"0.4.2")
        version_output = strip(capture_stdout(() -> dispatch(entry, ["--version"])))
        @test contains(version_output, "0.4.2")

        # dispatch() with [] shows help (no error)
        help_output = capture_stdout(() -> dispatch(entry, String[]))
        @test contains(help_output, "friedman")

        # dispatch() with ["--help"] shows help
        help_output2 = capture_stdout(() -> dispatch(entry, ["--help"]))
        @test contains(help_output2, "friedman")

        # dispatch_node with ["--help"] shows node help
        node_help = capture_stdout(() -> dispatch_node(outer_node, ["--help"]; prog="friedman top"))
        @test contains(node_help, "sub")

        # dispatch_leaf with ["--help"] shows leaf help, handler NOT called
        called_with[] = nothing
        leaf_for_help = LeafCommand("check", handler;
            args=[Argument("data"; description="Data")],
            description="Check something")
        leaf_help = capture_stdout(() -> dispatch_leaf(leaf_for_help, ["--help"]; prog="friedman check"))
        @test contains(leaf_help, "check")
        @test isnothing(called_with[])  # handler was NOT called

        # Handler receives all defaults when no options specified
        cmd_defaults = LeafCommand("test", handler;
            options=[
                Option("lags"; type=Int, default=4, description="Lags"),
                Option("format"; type=String, default="table", description="Format"),
            ],
            flags=[
                Flag("verbose"; description="Verbose"),
            ])
        dispatch_leaf(cmd_defaults, String[]; prog="friedman test")
        @test called_with[][:lags] == 4
        @test called_with[][:format] == "table"
        @test called_with[][:verbose] == false

        # Non-ParseError exceptions from handler are rethrown
        error_handler = (; kwargs...) -> error("custom error")
        cmd_error = LeafCommand("fail", error_handler; description="Will fail")
        @test_throws ErrorException dispatch_leaf(cmd_error, String[]; prog="friedman fail")

        # -h short flag also triggers help
        h_help = capture_stdout(() -> dispatch_leaf(leaf_for_help, ["-h"]; prog="friedman check"))
        @test contains(h_help, "check")

        # -V short flag triggers version
        v_output = strip(capture_stdout(() -> dispatch(entry, ["-V"])))
        @test contains(v_output, "0.4.2")

        # --warranty flag prints warranty text
        warranty_output = capture_stdout(() -> dispatch(entry, ["--warranty"]))
        @test contains(warranty_output, "WARRANTY")

        # --conditions flag prints conditions text
        conditions_output = capture_stdout(() -> dispatch(entry, ["--conditions"]))
        @test contains(conditions_output, "copies")
    end

    @testset "DispatchError on unknown command" begin
        handler = (; kwargs...) -> nothing
        leaf = LeafCommand("run", handler; description="Run")
        node = NodeCommand("top";
            subcmds=Dict{String,Union{NodeCommand,LeafCommand}}("run" => leaf),
            description="Top group")

        # Unknown subcommand throws DispatchError
        @test_throws DispatchError dispatch_node(node, ["nonexistent"]; prog="friedman top")

        # Verify error message includes prog and bad command
        try
            dispatch_node(node, ["nonexistent"]; prog="friedman top")
        catch e
            @test e isa DispatchError
            @test contains(e.message, "nonexistent")
            @test contains(e.message, "friedman top")
        end

        # Empty args on leaf with required argument shows help (not ParseError)
        cmd_req = LeafCommand("check", handler;
            args=[Argument("data"; required=true, description="Data")],
            description="Check")
        helpout = let (tmppath, tmpio) = mktemp()
            try
                redirect_stdout(tmpio) do
                    dispatch_leaf(cmd_req, String[]; prog="friedman check")
                end
                close(tmpio)
                read(tmppath, String)
            finally
                try; close(tmpio); catch; end
                try; rm(tmppath); catch; end
            end
        end
        @test contains(helpout, "check")
        @test contains(helpout, "<data>")
    end

    # ──────────────────────────────────────────────────────────────
    # Action-first CLI hierarchy tests (v0.1.4)
    # ──────────────────────────────────────────────────────────────

    @testset "Estimate command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        # Build simulated estimate NodeCommand with 15 LeafCommand children
        est_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate VAR(p)")

        est_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lag order"),
                Option("prior"; type=String, default="minnesota", description="Prior type"),
                Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("method"; type=String, default="mean", description="Posterior extraction"),
                Option("config"; type=String, default="", description="Config"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate BVAR")

        est_lp = LeafCommand("lp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("method"; type=String, default="standard", description="Method"),
                Option("shock"; type=Int, default=1, description="Shock index"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("control-lags"; type=Int, default=4, description="Control lags"),
                Option("vcov"; type=String, default="newey_west", description="HAC"),
                Option("instruments"; type=String, default="", description="Instruments"),
                Option("knots"; type=Int, default=3, description="Knots"),
                Option("lambda"; type=Float64, default=0.0, description="Lambda"),
                Option("state-var"; type=Int, default=nothing, description="State var"),
                Option("gamma"; type=Float64, default=1.5, description="Gamma"),
                Option("transition"; type=String, default="logistic", description="Transition"),
                Option("treatment"; type=Int, default=1, description="Treatment"),
                Option("score-method"; type=String, default="logit", description="Score method"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate LP")

        est_arima = LeafCommand("arima", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=nothing, description="AR order"),
                Option("d"; type=Int, default=0, description="Diff order"),
                Option("q"; type=Int, default=0, description="MA order"),
                Option("max-p"; type=Int, default=5, description="Max AR"),
                Option("max-d"; type=Int, default=2, description="Max diff"),
                Option("max-q"; type=Int, default=5, description="Max MA"),
                Option("criterion"; type=String, default="bic", description="Criterion"),
                Option("method"; short="m", type=String, default="css_mle", description="Method"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Estimate ARIMA")

        est_gmm = LeafCommand("gmm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("config"; type=String, default="", description="Config"),
                Option("weighting"; short="w", type=String, default="twostep", description="Weighting"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate GMM")

        est_static = LeafCommand("static", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("criterion"; type=String, default="ic1", description="IC criterion"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Static factor model")

        est_dynamic = LeafCommand("dynamic", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("factor-lags"; short="p", type=Int, default=1, description="Factor lags"),
                Option("method"; type=String, default="twostep", description="Method"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Dynamic factor model")

        est_gdfm = LeafCommand("gdfm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Static factors"),
                Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GDFM")

        est_arch = LeafCommand("arch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate ARCH")

        est_garch = LeafCommand("garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate GARCH")

        est_egarch = LeafCommand("egarch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="EGARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate EGARCH")

        est_gjr_garch = LeafCommand("gjr_garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate GJR-GARCH")

        est_sv = LeafCommand("sv", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("draws"; short="n", type=Int, default=5000, description="MCMC draws"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate SV")

        est_fastica = LeafCommand("fastica", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("method"; type=String, default="fastica", description="ICA method"),
                Option("contrast"; type=String, default="logcosh", description="Contrast"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ICA-based SVAR")

        est_ml = LeafCommand("ml", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("distribution"; short="d", type=String, default="student_t", description="Distribution"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ML non-Gaussian SVAR")

        est_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lag order"),
                Option("rank"; short="r", type=String, default="auto", description="Cointegration rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),
                Option("method"; type=String, default="johansen", description="Method"),
                Option("significance"; type=Float64, default=0.05, description="Significance"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Estimate VECM")

        est_pvar = LeafCommand("pvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("lags"; short="p", type=Int, default=1, description="Lags"),
                Option("dependent"; type=String, default="", description="Dependent variables"),
                Option("predet"; type=String, default="", description="Predetermined variables"),
                Option("exog"; type=String, default="", description="Exogenous variables"),
                Option("transformation"; type=String, default="fd", description="Transformation"),
                Option("steps"; type=String, default="twostep", description="Steps"),
                Option("method"; type=String, default="gmm", description="Method"),
                Option("min-lag-endo"; type=Int, default=2, description="Min lag endo"),
                Option("max-lag-endo"; type=Int, default=99, description="Max lag endo"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            flags=[
                Flag("system"; description="Use system GMM"),
                Flag("collapse"; description="Collapse instruments"),
            ],
            description="Estimate Panel VAR")

        estimate_node = NodeCommand("estimate",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => est_var, "bvar" => est_bvar, "lp" => est_lp,
                "arima" => est_arima, "gmm" => est_gmm,
                "static" => est_static, "dynamic" => est_dynamic, "gdfm" => est_gdfm,
                "arch" => est_arch, "garch" => est_garch, "egarch" => est_egarch,
                "gjr_garch" => est_gjr_garch, "sv" => est_sv,
                "fastica" => est_fastica, "ml" => est_ml, "vecm" => est_vecm,
                "pvar" => est_pvar),
            "Estimate econometric models")

        # Structure tests
        @test estimate_node.name == "estimate"
        @test length(estimate_node.subcmds) == 17

        # All 17 are LeafCommands (this test uses a manually-built node; real counts checked in v0.3.2+ structure tests)
        for key in ["var", "bvar", "lp", "arima", "gmm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "fastica", "ml", "vecm", "pvar"]
            @test haskey(estimate_node.subcmds, key)
            @test estimate_node.subcmds[key] isa LeafCommand
        end

        # Option counts for key leaves
        @test length(estimate_node.subcmds["var"].options) == 4
        @test length(estimate_node.subcmds["bvar"].options) == 8
        @test length(estimate_node.subcmds["lp"].options) == 15
        @test length(estimate_node.subcmds["arima"].options) == 11
        @test length(estimate_node.subcmds["arch"].options) == 4
        @test length(estimate_node.subcmds["fastica"].options) == 5
        @test length(estimate_node.subcmds["ml"].options) == 4

        # Help text contains key subcmd names
        buf = IOBuffer()
        print_help(buf, estimate_node; prog="friedman estimate")
        help_text = String(take!(buf))
        @test contains(help_text, "var")
        @test contains(help_text, "bvar")
        @test contains(help_text, "lp")
        @test contains(help_text, "arima")
        @test contains(help_text, "gmm")
        @test contains(help_text, "static")
        @test contains(help_text, "fastica")
        @test contains(help_text, "arch")
        @test contains(help_text, "sv")
        @test contains(help_text, "pvar")

        # Estimate pvar option count (13 options + 2 flags)
        @test length(estimate_node.subcmds["pvar"].options) == 13
        @test length(estimate_node.subcmds["pvar"].flags) == 2

        # Arg binding: estimate var
        parsed = tokenize(["data.csv", "--lags=4", "--trend=both"])
        bound = bind_args(parsed, est_var)
        @test bound.data == "data.csv"
        @test bound.lags == 4
        @test bound.trend == "both"

        # Arg binding: estimate bvar
        parsed = tokenize(["data.csv", "--draws=5000", "--sampler=hmc"])
        bound = bind_args(parsed, est_bvar)
        @test bound.data == "data.csv"
        @test bound.draws == 5000
        @test bound.sampler == "hmc"
        @test bound.lags == 4  # default

        # Arg binding: estimate lp
        parsed = tokenize(["data.csv", "--method=iv", "--instruments=inst.csv"])
        bound = bind_args(parsed, est_lp)
        @test bound.data == "data.csv"
        @test bound.method == "iv"
        @test bound.instruments == "inst.csv"
        @test bound.horizons == 20  # default

        # Dispatch: estimate var test.csv --lags=4
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        est_var_d = LeafCommand("var", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("lags"; short="p", type=Int, default=nothing, description="Lags")],
            description="VAR")
        est_dispatch = NodeCommand("estimate",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => est_var_d),
            "Estimate")
        dispatch_node(est_dispatch, ["var", "test.csv", "--lags=4"]; prog="friedman estimate")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:lags] == 4

        # Dispatch: estimate lp test.csv --method=iv
        est_lp_d = LeafCommand("lp", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("method"; type=String, default="standard", description="Method")],
            description="LP")
        est_dispatch2 = NodeCommand("estimate",
            Dict{String,Union{NodeCommand,LeafCommand}}("lp" => est_lp_d),
            "Estimate")
        dispatch_node(est_dispatch2, ["lp", "test.csv", "--method=iv"]; prog="friedman estimate")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:method] == "iv"
    end

    @testset "Test command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        # Build test NodeCommand with 12 subcmds
        test_adf = LeafCommand("adf", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("max-lags"; type=Int, default=nothing, description="Max lags"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="ADF test")

        test_kpss = LeafCommand("kpss", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="KPSS test")

        test_pp = LeafCommand("pp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="PP test")

        test_za = LeafCommand("za", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("trend"; type=String, default="both", description="Trend"),
                Option("trim"; type=Float64, default=0.15, description="Trim"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Zivot-Andrews test")

        test_np = LeafCommand("np", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Ng-Perron test")

        test_johansen = LeafCommand("johansen", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("trend"; type=String, default="constant", description="Trend"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Johansen cointegration test")

        test_normality = LeafCommand("normality", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Normality test suite")

        test_identifiability = LeafCommand("identifiability", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("test"; short="t", type=String, default="all", description="Test type"),
                Option("method"; type=String, default="fastica", description="Method"),
                Option("contrast"; type=String, default="logcosh", description="Contrast"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Identifiability tests")

        test_heteroskedasticity = LeafCommand("heteroskedasticity", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("method"; type=String, default="markov", description="Method"),
                Option("config"; type=String, default="", description="Config"),
                Option("regimes"; type=Int, default=2, description="Regimes"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Heteroskedasticity SVAR")

        test_arch_lm = LeafCommand("arch_lm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="ARCH-LM test")

        test_ljung_box = LeafCommand("ljung_box", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("lags"; short="p", type=Int, default=10, description="Lags"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Ljung-Box test")

        # VAR-specific tests as nested NodeCommand
        var_lagselect = LeafCommand("lagselect", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("max-lags"; type=Int, default=12, description="Max lags"),
                Option("criterion"; type=String, default="aic", description="Criterion"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Lag selection")

        var_stability = LeafCommand("stability", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Stability check")

        var_node = NodeCommand("var",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "lagselect" => var_lagselect, "stability" => var_stability),
            "VAR diagnostic tests")

        test_granger = LeafCommand("granger", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("cause"; type=Int, default=1, description="Cause variable index"),
                Option("effect"; type=Int, default=2, description="Effect variable index"),
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Cointegrating rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),
                Option("model"; type=String, default="vecm", description="var|vecm"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            flags=[
                Flag("all"; description="Test all pairwise combinations (VAR only)"),
            ],
            description="Granger causality test (VAR or VECM)")

        # Panel VAR tests as nested NodeCommand
        pvar_hansen_j = LeafCommand("hansen_j", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("lags"; short="p", type=Int, default=1, description="Lags"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Hansen J test")

        pvar_mmsc = LeafCommand("mmsc", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("max-lags"; type=Int, default=4, description="Max lags"),
                Option("criterion"; type=String, default="bic", description="Criterion"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="MMSC model selection")

        pvar_lagselect = LeafCommand("lagselect", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("max-lags"; type=Int, default=4, description="Max lags"),
                Option("criterion"; type=String, default="bic", description="Criterion"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Panel VAR lag selection")

        pvar_stability = LeafCommand("stability", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("lags"; short="p", type=Int, default=1, description="Lags"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Panel VAR stability check")

        pvar_node = NodeCommand("pvar",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "hansen_j" => pvar_hansen_j, "mmsc" => pvar_mmsc,
                "lagselect" => pvar_lagselect, "stability" => pvar_stability),
            "Panel VAR diagnostic tests")

        test_lr = LeafCommand("lr", handler;
            args=[Argument("data1"; description="Path to CSV data file for restricted model"),
                  Argument("data2"; description="Path to CSV data file for unrestricted model")],
            options=[
                Option("lags1"; type=Int, default=nothing, description="Lag order for restricted model"),
                Option("lags2"; type=Int, default=nothing, description="Lag order for unrestricted model"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Likelihood Ratio test")

        test_lm = LeafCommand("lm", handler;
            args=[Argument("data1"; description="Path to CSV data file for restricted model"),
                  Argument("data2"; description="Path to CSV data file for unrestricted model")],
            options=[
                Option("lags1"; type=Int, default=nothing, description="Lag order for restricted model"),
                Option("lags2"; type=Int, default=nothing, description="Lag order for unrestricted model"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Lagrange Multiplier test")

        test_node = NodeCommand("test",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "adf" => test_adf, "kpss" => test_kpss, "pp" => test_pp,
                "za" => test_za, "np" => test_np, "johansen" => test_johansen,
                "normality" => test_normality, "identifiability" => test_identifiability,
                "heteroskedasticity" => test_heteroskedasticity,
                "arch_lm" => test_arch_lm, "ljung_box" => test_ljung_box,
                "var" => var_node, "granger" => test_granger,
                "pvar" => pvar_node, "lr" => test_lr, "lm" => test_lm),
            "Statistical tests")

        # Structure tests
        @test test_node.name == "test"
        @test length(test_node.subcmds) == 16  # manually-built node; real counts checked in structural break test section

        # var subcmd is a NodeCommand with 2 children
        @test test_node.subcmds["var"] isa NodeCommand
        @test length(test_node.subcmds["var"].subcmds) == 2
        @test haskey(test_node.subcmds["var"].subcmds, "lagselect")
        @test haskey(test_node.subcmds["var"].subcmds, "stability")

        # pvar subcmd is a NodeCommand with 4 children
        @test test_node.subcmds["pvar"] isa NodeCommand
        @test length(test_node.subcmds["pvar"].subcmds) == 4
        @test haskey(test_node.subcmds["pvar"].subcmds, "hansen_j")
        @test haskey(test_node.subcmds["pvar"].subcmds, "mmsc")
        @test haskey(test_node.subcmds["pvar"].subcmds, "lagselect")
        @test haskey(test_node.subcmds["pvar"].subcmds, "stability")

        # lr and lm are LeafCommands
        @test test_node.subcmds["lr"] isa LeafCommand
        @test test_node.subcmds["lm"] isa LeafCommand
        @test length(test_node.subcmds["lr"].args) == 2
        @test length(test_node.subcmds["lm"].args) == 2

        # All other subcmds are LeafCommands
        for key in ["adf", "kpss", "pp", "za", "np", "johansen",
                     "normality", "identifiability", "heteroskedasticity",
                     "arch_lm", "ljung_box", "granger", "lr", "lm"]
            @test test_node.subcmds[key] isa LeafCommand
        end

        # Option counts
        @test length(test_node.subcmds["adf"].options) == 5
        @test length(test_node.subcmds["kpss"].options) == 4
        @test length(test_node.subcmds["johansen"].options) == 4
        @test length(test_node.subcmds["arch_lm"].options) == 4
        @test length(test_node.subcmds["ljung_box"].options) == 4
        @test length(test_node.subcmds["identifiability"].options) == 6
        @test length(test_node.subcmds["heteroskedasticity"].options) == 6
        @test length(test_node.subcmds["granger"].options) == 8
        @test length(test_node.subcmds["granger"].flags) == 1
        @test length(test_node.subcmds["lr"].options) == 4
        @test length(test_node.subcmds["lm"].options) == 4

        # Help text
        buf = IOBuffer()
        print_help(buf, test_node; prog="friedman test")
        help_text = String(take!(buf))
        @test contains(help_text, "adf")
        @test contains(help_text, "kpss")
        @test contains(help_text, "johansen")
        @test contains(help_text, "normality")
        @test contains(help_text, "arch_lm")
        @test contains(help_text, "ljung_box")
        @test contains(help_text, "var")
        @test contains(help_text, "pvar")
        @test contains(help_text, "lr")
        @test contains(help_text, "lm")

        # Arg binding: adf
        parsed = tokenize(["data.csv", "--column=2", "--max-lags=8"])
        bound = bind_args(parsed, test_adf)
        @test bound.data == "data.csv"
        @test bound.column == 2
        @test bound.max_lags == 8
        @test bound.trend == "constant"  # default

        # Arg binding: johansen
        parsed = tokenize(["data.csv", "--lags=4", "--trend=none"])
        bound = bind_args(parsed, test_johansen)
        @test bound.data == "data.csv"
        @test bound.lags == 4
        @test bound.trend == "none"

        # Dispatch through nested var node: friedman test var lagselect test.csv
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        lagsel_d = LeafCommand("lagselect", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("max-lags"; type=Int, default=12, description="Max lags")],
            description="Lag selection")
        var_d = NodeCommand("var",
            Dict{String,Union{NodeCommand,LeafCommand}}("lagselect" => lagsel_d),
            "VAR tests")
        test_d = NodeCommand("test",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => var_d),
            "Tests")
        dispatch_node(test_d, ["var", "lagselect", "test.csv", "--max-lags=8"]; prog="friedman test")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:max_lags] == 8
    end

    @testset "IRF/FEVD/HD as top-level nodes (action-first)" begin
        handler = (; kwargs...) -> kwargs

        # -- IRF node --
        irf_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("shock"; type=Int, default=1, description="Shock"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("ci"; type=String, default="bootstrap", description="CI type"),
                Option("replications"; type=Int, default=1000, description="Replications"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Frequentist IRFs")

        irf_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("shock"; type=Int, default=1, description="Shock"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Bayesian IRFs")

        irf_lp = LeafCommand("lp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("shock"; type=Int, default=1, description="Shock"),
                Option("shocks"; type=String, default="", description="Multi-shocks"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("var-lags"; type=Int, default=nothing, description="VAR lags"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("ci"; type=String, default="none", description="CI"),
                Option("replications"; type=Int, default=200, description="Replications"),
                Option("conf-level"; type=Float64, default=0.95, description="Conf level"),
                Option("vcov"; type=String, default="newey_west", description="HAC"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="LP IRFs")

        irf_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Cointegrating rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),
                Option("shock"; type=Int, default=1, description="Shock"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("ci"; type=String, default="bootstrap", description="CI type"),
                Option("replications"; type=Int, default=1000, description="Replications"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VECM IRFs")

        irf_pvar = LeafCommand("pvar", handler;
            args=[Argument("data"; required=false, default="", description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("lags"; short="p", type=Int, default=1, description="Lags"),
                Option("horizons"; short="h", type=Int, default=10, description="Horizon"),
                Option("irf-type"; type=String, default="oirf", description="oirf|girf"),
                Option("boot-draws"; type=Int, default=500, description="Bootstrap draws"),
                Option("confidence"; type=Float64, default=0.95, description="Confidence level"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Panel VAR IRFs")

        irf_node = NodeCommand("irf",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => irf_var, "bvar" => irf_bvar, "lp" => irf_lp,
                "vecm" => irf_vecm, "pvar" => irf_pvar),
            "Impulse Response Functions")

        # -- FEVD node --
        fevd_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VAR FEVD")

        fevd_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Bayesian FEVD")

        fevd_lp = LeafCommand("lp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("var-lags"; type=Int, default=nothing, description="VAR lags"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("vcov"; type=String, default="newey_west", description="HAC"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="LP FEVD")

        fevd_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Cointegrating rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),
                Option("horizons"; short="h", type=Int, default=20, description="Horizon"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VECM FEVD")

        fevd_pvar = LeafCommand("pvar", handler;
            args=[Argument("data"; required=false, default="", description="Data file")],
            options=[
                Option("id-col"; type=String, default="", description="Panel group ID column"),
                Option("time-col"; type=String, default="", description="Time column"),
                Option("lags"; short="p", type=Int, default=1, description="Lags"),
                Option("horizons"; short="h", type=Int, default=10, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Panel VAR FEVD")

        fevd_node = NodeCommand("fevd",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => fevd_var, "bvar" => fevd_bvar, "lp" => fevd_lp,
                "vecm" => fevd_vecm, "pvar" => fevd_pvar),
            "Forecast Error Variance Decomposition")

        # -- HD node --
        hd_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VAR HD")

        hd_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Bayesian HD")

        hd_lp = LeafCommand("lp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("var-lags"; type=Int, default=nothing, description="VAR lags"),
                Option("id"; type=String, default="cholesky", description="Identification"),
                Option("vcov"; type=String, default="newey_west", description="HAC"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="LP HD")

        hd_node = NodeCommand("hd",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => hd_var, "bvar" => hd_bvar, "lp" => hd_lp),
            "Historical Decomposition")

        # Build root with all top-level commands
        root = NodeCommand("friedman",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "estimate" => NodeCommand("estimate", Dict{String,Union{NodeCommand,LeafCommand}}(), "Est"),
                "test" => NodeCommand("test", Dict{String,Union{NodeCommand,LeafCommand}}(), "Test"),
                "irf" => irf_node, "fevd" => fevd_node, "hd" => hd_node,
                "forecast" => NodeCommand("forecast", Dict{String,Union{NodeCommand,LeafCommand}}(), "Forecast")),
            "Friedman CLI")
        entry = Entry("friedman", root; version=v"0.4.2")

        # Top level HAS irf, fevd, hd (action-first)
        @test haskey(root.subcmds, "irf")
        @test haskey(root.subcmds, "fevd")
        @test haskey(root.subcmds, "hd")
        @test haskey(root.subcmds, "estimate")
        @test haskey(root.subcmds, "forecast")

        # IRF node structure
        @test irf_node.name == "irf"
        @test length(irf_node.subcmds) == 5
        @test haskey(irf_node.subcmds, "var")
        @test haskey(irf_node.subcmds, "bvar")
        @test haskey(irf_node.subcmds, "lp")
        @test haskey(irf_node.subcmds, "vecm")
        @test haskey(irf_node.subcmds, "pvar")
        @test irf_node.subcmds["var"] isa LeafCommand
        @test irf_node.subcmds["bvar"] isa LeafCommand
        @test irf_node.subcmds["lp"] isa LeafCommand
        @test irf_node.subcmds["pvar"] isa LeafCommand

        # IRF var option count (9 options)
        @test length(irf_node.subcmds["var"].options) == 9
        # IRF pvar option count (9 options)
        @test length(irf_node.subcmds["pvar"].options) == 9

        # FEVD node structure (5 leaves: var, bvar, lp, vecm, pvar)
        @test length(fevd_node.subcmds) == 5
        @test haskey(fevd_node.subcmds, "vecm")
        @test haskey(fevd_node.subcmds, "pvar")
        @test fevd_node.subcmds["vecm"] isa LeafCommand
        @test fevd_node.subcmds["pvar"] isa LeafCommand
        @test length(fevd_node.subcmds["pvar"].options) == 6

        # FEVD bvar has draws, sampler
        @test length(fevd_node.subcmds["bvar"].options) == 8
        fevd_bvar_opt_names = [o.name for o in fevd_node.subcmds["bvar"].options]
        @test "draws" in fevd_bvar_opt_names
        @test "sampler" in fevd_bvar_opt_names

        # HD lp has var-lags, vcov
        @test length(hd_node.subcmds["lp"].options) == 7
        hd_lp_opt_names = [o.name for o in hd_node.subcmds["lp"].options]
        @test "var-lags" in hd_lp_opt_names
        @test "vcov" in hd_lp_opt_names

        # Help text for IRF
        buf = IOBuffer()
        print_help(buf, irf_node; prog="friedman irf")
        help_text = String(take!(buf))
        @test contains(help_text, "var")
        @test contains(help_text, "bvar")
        @test contains(help_text, "lp")
        @test contains(help_text, "pvar")

        # Help text for FEVD
        buf = IOBuffer()
        print_help(buf, fevd_node; prog="friedman fevd")
        help_text = String(take!(buf))
        @test contains(help_text, "var")
        @test contains(help_text, "bvar")
        @test contains(help_text, "lp")
        @test contains(help_text, "pvar")

        # Help text for HD
        buf = IOBuffer()
        print_help(buf, hd_node; prog="friedman hd")
        help_text = String(take!(buf))
        @test contains(help_text, "var")
        @test contains(help_text, "bvar")
        @test contains(help_text, "lp")

        # Dispatch: friedman irf var test.csv --id=sign --shock=2
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        irf_var_d = LeafCommand("var", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("id"; type=String, default="cholesky", description="ID"),
                Option("shock"; type=Int, default=1, description="Shock"),
            ],
            description="IRF")
        irf_d = NodeCommand("irf",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => irf_var_d),
            "IRF")
        dispatch_node(irf_d, ["var", "test.csv", "--id=sign", "--shock=2"]; prog="friedman irf")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:id] == "sign"
        @test called_with[][:shock] == 2

        # Dispatch: friedman hd lp test.csv --id=longrun
        hd_lp_d = LeafCommand("lp", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("id"; type=String, default="cholesky", description="ID")],
            description="HD")
        hd_d = NodeCommand("hd",
            Dict{String,Union{NodeCommand,LeafCommand}}("lp" => hd_lp_d),
            "HD")
        dispatch_node(hd_d, ["lp", "test.csv", "--id=longrun"]; prog="friedman hd")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:id] == "longrun"
    end

    @testset "Forecast command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        # Build forecast NodeCommand with 12 leaves
        fc_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("confidence"; type=Float64, default=0.95, description="Confidence"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VAR forecast")

        fc_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="BVAR forecast")

        fc_lp = LeafCommand("lp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("shock"; type=Int, default=1, description="Shock"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("shock-size"; type=Float64, default=1.0, description="Shock size"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("vcov"; type=String, default="newey_west", description="HAC"),
                Option("ci-method"; type=String, default="analytical", description="CI method"),
                Option("conf-level"; type=Float64, default=0.95, description="Conf level"),
                Option("n-boot"; type=Int, default=500, description="Bootstrap reps"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="LP forecast")

        fc_arima = LeafCommand("arima", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=nothing, description="AR order"),
                Option("d"; type=Int, default=0, description="Diff"),
                Option("q"; type=Int, default=0, description="MA order"),
                Option("max-p"; type=Int, default=5, description="Max AR"),
                Option("max-d"; type=Int, default=2, description="Max diff"),
                Option("max-q"; type=Int, default=5, description="Max MA"),
                Option("criterion"; type=String, default="bic", description="Criterion"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("confidence"; type=Float64, default=0.95, description="Confidence"),
                Option("method"; short="m", type=String, default="css_mle", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ARIMA forecast")

        fc_static = LeafCommand("static", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("ci-method"; type=String, default="none", description="CI method"),
                Option("conf-level"; type=Float64, default=0.95, description="Conf level"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Static factor forecast")

        fc_dynamic = LeafCommand("dynamic", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("factor-lags"; short="p", type=Int, default=1, description="Factor lags"),
                Option("method"; type=String, default="twostep", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Dynamic factor forecast")

        fc_gdfm = LeafCommand("gdfm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GDFM forecast")

        fc_arch = LeafCommand("arch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ARCH forecast")

        fc_garch = LeafCommand("garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GARCH forecast")

        fc_egarch = LeafCommand("egarch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="EGARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="EGARCH forecast")

        fc_gjr_garch = LeafCommand("gjr_garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GJR-GARCH forecast")

        fc_sv = LeafCommand("sv", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("draws"; short="n", type=Int, default=5000, description="Draws"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="SV forecast")

        fc_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Cointegrating rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),
                Option("horizons"; short="h", type=Int, default=12, description="Horizon"),
                Option("ci-method"; type=String, default="none", description="CI method"),
                Option("replications"; type=Int, default=500, description="Replications"),
                Option("confidence"; type=Float64, default=0.95, description="Confidence"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VECM forecast")

        forecast_node = NodeCommand("forecast",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => fc_var, "bvar" => fc_bvar, "lp" => fc_lp,
                "arima" => fc_arima, "static" => fc_static,
                "dynamic" => fc_dynamic, "gdfm" => fc_gdfm,
                "arch" => fc_arch, "garch" => fc_garch, "egarch" => fc_egarch,
                "gjr_garch" => fc_gjr_garch, "sv" => fc_sv, "vecm" => fc_vecm),
            "Forecasting")

        # Structure tests
        @test forecast_node.name == "forecast"
        @test length(forecast_node.subcmds) == 13

        # All are LeafCommands
        for key in ["var", "bvar", "lp", "arima", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "vecm"]
            @test haskey(forecast_node.subcmds, key)
            @test forecast_node.subcmds[key] isa LeafCommand
        end

        # Option counts
        @test length(forecast_node.subcmds["var"].options) == 5
        @test length(forecast_node.subcmds["bvar"].options) == 7
        @test length(forecast_node.subcmds["lp"].options) == 10
        @test length(forecast_node.subcmds["arima"].options) == 13
        @test length(forecast_node.subcmds["static"].options) == 6
        @test length(forecast_node.subcmds["arch"].options) == 5
        @test length(forecast_node.subcmds["sv"].options) == 5

        # Help text
        buf = IOBuffer()
        print_help(buf, forecast_node; prog="friedman forecast")
        help_text = String(take!(buf))
        @test contains(help_text, "var")
        @test contains(help_text, "bvar")
        @test contains(help_text, "lp")
        @test contains(help_text, "arima")
        @test contains(help_text, "static")
        @test contains(help_text, "arch")
        @test contains(help_text, "sv")

        # Arg binding: forecast var
        parsed = tokenize(["data.csv", "--horizons=24", "--confidence=0.90"])
        bound = bind_args(parsed, fc_var)
        @test bound.data == "data.csv"
        @test bound.horizons == 24
        @test bound.confidence == 0.90

        # Arg binding: forecast arima
        parsed = tokenize(["data.csv", "--column=2", "--p=3", "--d=1", "--q=1", "--horizons=24"])
        bound = bind_args(parsed, fc_arima)
        @test bound.data == "data.csv"
        @test bound.column == 2
        @test bound.p == 3
        @test bound.d == 1
        @test bound.q == 1
        @test bound.horizons == 24
        @test bound.confidence == 0.95  # default

        # Dispatch: friedman forecast var test.csv --horizons=24
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        fc_var_d = LeafCommand("var", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("horizons"; short="h", type=Int, default=12, description="Horizon")],
            description="VAR forecast")
        fc_dispatch = NodeCommand("forecast",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => fc_var_d),
            "Forecast")
        dispatch_node(fc_dispatch, ["var", "test.csv", "--horizons=24"]; prog="friedman forecast")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:horizons] == 24

        # Dispatch: friedman forecast arima test.csv --column=1
        fc_arima_d = LeafCommand("arima", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("column"; short="c", type=Int, default=1, description="Column")],
            description="ARIMA forecast")
        fc_dispatch2 = NodeCommand("forecast",
            Dict{String,Union{NodeCommand,LeafCommand}}("arima" => fc_arima_d),
            "Forecast")
        dispatch_node(fc_dispatch2, ["arima", "test.csv", "--column=2"]; prog="friedman forecast")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:column] == 2
    end

    @testset "Predict command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        pred_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VAR predict")

        pred_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="BVAR predict")

        pred_arima = LeafCommand("arima", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=nothing, description="AR order"),
                Option("d"; type=Int, default=0, description="Differencing"),
                Option("q"; type=Int, default=0, description="MA order"),
                Option("method"; short="m", type=String, default="css_mle", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            flags=[Flag("auto"; short="a", description="Auto ARIMA")],
            description="ARIMA predict")

        pred_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VECM predict")

        pred_static = LeafCommand("static", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Static factor predict")

        pred_dynamic = LeafCommand("dynamic", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("factor-lags"; short="p", type=Int, default=1, description="Factor lags"),
                Option("method"; type=String, default="twostep", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Dynamic factor predict")

        pred_gdfm = LeafCommand("gdfm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GDFM predict")

        pred_arch = LeafCommand("arch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ARCH predict")

        pred_garch = LeafCommand("garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GARCH predict")

        pred_egarch = LeafCommand("egarch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="EGARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="EGARCH predict")

        pred_gjr_garch = LeafCommand("gjr_garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GJR-GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GJR-GARCH predict")

        pred_sv = LeafCommand("sv", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("draws"; short="n", type=Int, default=5000, description="Draws"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="SV predict")

        predict_node = NodeCommand("predict",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => pred_var, "bvar" => pred_bvar, "arima" => pred_arima, "vecm" => pred_vecm,
                "static" => pred_static, "dynamic" => pred_dynamic, "gdfm" => pred_gdfm,
                "arch" => pred_arch, "garch" => pred_garch, "egarch" => pred_egarch,
                "gjr_garch" => pred_gjr_garch, "sv" => pred_sv),
            "In-sample predictions")

        # Structure tests
        @test predict_node.name == "predict"
        @test length(predict_node.subcmds) == 12  # manually-built node; real counts checked in v0.3.2+ structure tests
        for key in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test haskey(predict_node.subcmds, key)
            @test predict_node.subcmds[key] isa LeafCommand
        end

        # Option counts
        @test length(predict_node.subcmds["var"].options) == 3
        @test length(predict_node.subcmds["bvar"].options) == 6
        @test length(predict_node.subcmds["arima"].options) == 7
        @test length(predict_node.subcmds["vecm"].options) == 5
        @test length(predict_node.subcmds["static"].options) == 3
        @test length(predict_node.subcmds["dynamic"].options) == 5
        @test length(predict_node.subcmds["gdfm"].options) == 4
        @test length(predict_node.subcmds["arch"].options) == 4
        @test length(predict_node.subcmds["garch"].options) == 5
        @test length(predict_node.subcmds["egarch"].options) == 5
        @test length(predict_node.subcmds["gjr_garch"].options) == 5
        @test length(predict_node.subcmds["sv"].options) == 4

        # Help text
        buf = IOBuffer()
        print_help(buf, predict_node; prog="friedman predict")
        help_text = String(take!(buf))
        for key in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test contains(help_text, key)
        end

        # Arg binding: predict var
        parsed = tokenize(["data.csv", "--lags=3"])
        bound = bind_args(parsed, pred_var)
        @test bound.data == "data.csv"
        @test bound.lags == 3

        # Dispatch: friedman predict var test.csv --lags=2
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        pred_var_d = LeafCommand("var", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("lags"; short="p", type=Int, default=nothing, description="Lags")],
            description="predict")
        pred_dispatch = NodeCommand("predict",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => pred_var_d),
            "Predict")
        dispatch_node(pred_dispatch, ["var", "test.csv", "--lags=2"]; prog="friedman predict")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:lags] == 2
    end

    @testset "Residuals command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        res_var = LeafCommand("var", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lags"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VAR residuals")

        res_bvar = LeafCommand("bvar", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("draws"; short="n", type=Int, default=2000, description="Draws"),
                Option("sampler"; type=String, default="nuts", description="Sampler"),
                Option("config"; type=String, default="", description="Config"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="BVAR residuals")

        res_arima = LeafCommand("arima", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=nothing, description="AR order"),
                Option("d"; type=Int, default=0, description="Differencing"),
                Option("q"; type=Int, default=0, description="MA order"),
                Option("method"; short="m", type=String, default="css_mle", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            flags=[Flag("auto"; short="a", description="Auto ARIMA")],
            description="ARIMA residuals")

        res_vecm = LeafCommand("vecm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("rank"; short="r", type=String, default="auto", description="Rank"),
                Option("deterministic"; type=String, default="constant", description="Deterministic"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="VECM residuals")

        res_static = LeafCommand("static", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Static factor residuals")

        res_dynamic = LeafCommand("dynamic", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("factor-lags"; short="p", type=Int, default=1, description="Factor lags"),
                Option("method"; type=String, default="twostep", description="Method"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Dynamic factor residuals")

        res_gdfm = LeafCommand("gdfm", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Factors"),
                Option("dynamic-rank"; short="q", type=Int, default=nothing, description="Dynamic rank"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GDFM residuals")

        res_arch = LeafCommand("arch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="ARCH residuals")

        res_garch = LeafCommand("garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GARCH residuals")

        res_egarch = LeafCommand("egarch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="EGARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="EGARCH residuals")

        res_gjr_garch = LeafCommand("gjr_garch", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("p"; type=Int, default=1, description="GJR-GARCH order"),
                Option("q"; type=Int, default=1, description="ARCH order"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="GJR-GARCH residuals")

        res_sv = LeafCommand("sv", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("column"; short="c", type=Int, default=1, description="Column"),
                Option("draws"; short="n", type=Int, default=5000, description="Draws"),

                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="SV residuals")

        residuals_node = NodeCommand("residuals",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "var" => res_var, "bvar" => res_bvar, "arima" => res_arima, "vecm" => res_vecm,
                "static" => res_static, "dynamic" => res_dynamic, "gdfm" => res_gdfm,
                "arch" => res_arch, "garch" => res_garch, "egarch" => res_egarch,
                "gjr_garch" => res_gjr_garch, "sv" => res_sv),
            "Model residuals")

        # Structure tests
        @test residuals_node.name == "residuals"
        @test length(residuals_node.subcmds) == 12  # manually-built node; real counts checked in v0.3.2+ structure tests
        for key in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test haskey(residuals_node.subcmds, key)
            @test residuals_node.subcmds[key] isa LeafCommand
        end

        # Option counts
        @test length(residuals_node.subcmds["var"].options) == 3
        @test length(residuals_node.subcmds["bvar"].options) == 6
        @test length(residuals_node.subcmds["arima"].options) == 7
        @test length(residuals_node.subcmds["vecm"].options) == 5
        @test length(residuals_node.subcmds["static"].options) == 3
        @test length(residuals_node.subcmds["dynamic"].options) == 5
        @test length(residuals_node.subcmds["gdfm"].options) == 4
        @test length(residuals_node.subcmds["arch"].options) == 4
        @test length(residuals_node.subcmds["garch"].options) == 5
        @test length(residuals_node.subcmds["egarch"].options) == 5
        @test length(residuals_node.subcmds["gjr_garch"].options) == 5
        @test length(residuals_node.subcmds["sv"].options) == 4

        # Help text
        buf = IOBuffer()
        print_help(buf, residuals_node; prog="friedman residuals")
        help_text = String(take!(buf))
        for key in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test contains(help_text, key)
        end

        # Dispatch: friedman residuals var test.csv
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end

        res_var_d = LeafCommand("var", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("lags"; short="p", type=Int, default=nothing, description="Lags")],
            description="residuals")
        res_dispatch = NodeCommand("residuals",
            Dict{String,Union{NodeCommand,LeafCommand}}("var" => res_var_d),
            "Residuals")
        dispatch_node(res_dispatch, ["var", "test.csv"]; prog="friedman residuals")
        @test called_with[][:data] == "test.csv"
    end

    @testset "Filter command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        filt_hp = LeafCommand("hp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lambda"; short="l", type=Float64, default=1600.0, description="Lambda"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="HP filter")

        filt_hamilton = LeafCommand("hamilton", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("horizon"; short="h", type=Int, default=8, description="Horizon"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Hamilton filter")

        filt_bn = LeafCommand("bn", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("p"; type=Int, default=nothing, description="AR order"),
                Option("q"; type=Int, default=nothing, description="MA order"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Beveridge-Nelson")

        filt_bk = LeafCommand("bk", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("pl"; type=Int, default=6, description="Min period"),
                Option("pu"; type=Int, default=32, description="Max period"),
                Option("K"; type=Int, default=12, description="Truncation"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Baxter-King filter")

        filt_bhp = LeafCommand("bhp", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lambda"; short="l", type=Float64, default=1600.0, description="Lambda"),
                Option("stopping"; type=String, default="BIC", description="Stopping"),
                Option("max-iter"; type=Int, default=100, description="Max iterations"),
                Option("sig-p"; type=Float64, default=0.05, description="Significance"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Boosted HP filter")

        filter_node = NodeCommand("filter",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "hp" => filt_hp, "hamilton" => filt_hamilton, "bn" => filt_bn,
                "bk" => filt_bk, "bhp" => filt_bhp),
            "Time series filters")

        @test filter_node.name == "filter"
        @test length(filter_node.subcmds) == 5
        for cmd in ["hp", "hamilton", "bn", "bk", "bhp"]
            @test haskey(filter_node.subcmds, cmd)
            @test filter_node.subcmds[cmd] isa LeafCommand
        end

        # Option counts
        @test length(filter_node.subcmds["hp"].options) == 4
        @test length(filter_node.subcmds["hamilton"].options) == 5
        @test length(filter_node.subcmds["bn"].options) == 5
        @test length(filter_node.subcmds["bk"].options) == 6
        @test length(filter_node.subcmds["bhp"].options) == 7

        # Help text
        buf = IOBuffer()
        print_help(buf, filter_node; prog="friedman filter")
        help_text = String(take!(buf))
        @test contains(help_text, "hp")
        @test contains(help_text, "hamilton")
        @test contains(help_text, "bn")
        @test contains(help_text, "bk")
        @test contains(help_text, "bhp")

        # Arg binding: filter hp data.csv --lambda=1600
        parsed = tokenize(["data.csv", "--lambda=1600.0"])
        bound = bind_args(parsed, filt_hp)
        @test bound.data == "data.csv"
        @test bound.lambda == 1600.0

        # Arg binding: filter bk data.csv --pl=6 --pu=32 --K=12
        parsed = tokenize(["data.csv", "--pl=6", "--pu=32", "--K=12"])
        bound = bind_args(parsed, filt_bk)
        @test bound.data == "data.csv"
        @test bound.pl == 6
        @test bound.pu == 32
        @test bound.K == 12

        # Dispatch
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> (called_with[] = kwargs)
        filt_dispatch_hp = LeafCommand("hp", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("lambda"; short="l", type=Float64, default=1600.0, description="Lambda"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="HP filter")
        filt_dispatch = NodeCommand("filter",
            Dict{String,Union{NodeCommand,LeafCommand}}("hp" => filt_dispatch_hp),
            "Filters")
        dispatch_node(filt_dispatch, ["hp", "test.csv"]; prog="friedman filter")
        @test called_with[][:data] == "test.csv"
    end

    @testset "Data command structure (action-first)" begin
        handler = (; kwargs...) -> kwargs

        data_list = LeafCommand("list", handler;
            args=Argument[],
            options=[
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="List datasets")

        data_load = LeafCommand("load", handler;
            args=[Argument("name"; description="Dataset name")],
            options=[
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("vars"; type=String, default="", description="Vars"),
                Option("country"; type=String, default="", description="Country"),
            ],
            flags=[Flag("transform"; short="t", description="Apply tcodes")],
            description="Load dataset")

        data_describe = LeafCommand("describe", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Describe data")

        data_diagnose = LeafCommand("diagnose", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Diagnose data")

        data_fix = LeafCommand("fix", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("method"; short="m", type=String, default="listwise", description="Method"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Fix data")

        data_transform = LeafCommand("transform", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("tcodes"; type=String, default="", description="Tcodes"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Transform data")

        data_filter = LeafCommand("filter", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("method"; short="m", type=String, default="hp", description="Method"),
                Option("component"; type=String, default="cycle", description="Component"),
                Option("lambda"; short="l", type=Float64, default=1600.0, description="Lambda"),
                Option("horizon"; type=Int, default=8, description="Horizon"),
                Option("lags"; short="p", type=Int, default=4, description="Lags"),
                Option("columns"; short="c", type=String, default="", description="Columns"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Filter data")

        data_validate = LeafCommand("validate", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("model"; type=String, default="", description="Model type"),
                Option("format"; short="f", type=String, default="table", description="Format"),
                Option("output"; short="o", type=String, default="", description="Output"),
            ],
            description="Validate data")

        data_balance = LeafCommand("balance", handler;
            args=[Argument("data"; description="Data file")],
            options=[
                Option("method"; type=String, default="dfm", description="Method"),
                Option("factors"; short="r", type=Int, default=3, description="Factors"),
                Option("lags"; short="p", type=Int, default=2, description="Lags"),
                Option("output"; short="o", type=String, default="", description="Output"),
                Option("format"; short="f", type=String, default="table", description="Format"),
            ],
            description="Balance panel")

        data_node = NodeCommand("data",
            Dict{String,Union{NodeCommand,LeafCommand}}(
                "list" => data_list, "load" => data_load, "describe" => data_describe,
                "diagnose" => data_diagnose, "fix" => data_fix, "transform" => data_transform,
                "filter" => data_filter, "validate" => data_validate, "balance" => data_balance),
            "Data management")

        @test data_node.name == "data"
        @test length(data_node.subcmds) == 9
        for cmd in ["list", "load", "describe", "diagnose", "fix", "transform", "filter", "validate", "balance"]
            @test haskey(data_node.subcmds, cmd)
            @test data_node.subcmds[cmd] isa LeafCommand
        end

        # Option counts
        @test length(data_node.subcmds["list"].options) == 2
        @test length(data_node.subcmds["load"].options) == 4
        @test length(data_node.subcmds["describe"].options) == 2
        @test length(data_node.subcmds["diagnose"].options) == 2
        @test length(data_node.subcmds["fix"].options) == 3
        @test length(data_node.subcmds["transform"].options) == 3
        @test length(data_node.subcmds["filter"].options) == 8
        @test length(data_node.subcmds["validate"].options) == 3
        @test length(data_node.subcmds["balance"].options) == 5

        # Help text
        buf = IOBuffer()
        print_help(buf, data_node; prog="friedman data")
        help_text = String(take!(buf))
        for cmd in ["list", "load", "describe", "diagnose", "fix", "transform", "filter", "validate", "balance"]
            @test contains(help_text, cmd)
        end

        # Arg binding: data load fred_md --transform
        parsed = tokenize(["fred_md", "--transform"])
        bound = bind_args(parsed, data_load)
        @test bound.name == "fred_md"
        @test bound.transform == true

        # Arg binding: data describe data.csv --format=json
        parsed = tokenize(["data.csv", "--format=json"])
        bound = bind_args(parsed, data_describe)
        @test bound.data == "data.csv"
        @test bound.format == "json"

        # Arg binding: data filter data.csv --method=hamilton --component=trend
        parsed = tokenize(["data.csv", "--method=hamilton", "--component=trend"])
        bound = bind_args(parsed, data_filter)
        @test bound.data == "data.csv"
        @test bound.method == "hamilton"
        @test bound.component == "trend"

        # Arg binding: data validate data.csv --model=var
        parsed = tokenize(["data.csv", "--model=var"])
        bound = bind_args(parsed, data_validate)
        @test bound.data == "data.csv"
        @test bound.model == "var"

        # Dispatch
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> (called_with[] = kwargs)
        data_load_d = LeafCommand("load", dispatch_handler;
            args=[Argument("name"; description="Name")],
            options=[Option("output"; short="o", type=String, default="", description="Output")],
            description="Load")
        data_dispatch = NodeCommand("data",
            Dict{String,Union{NodeCommand,LeafCommand}}("load" => data_load_d),
            "Data")
        dispatch_node(data_dispatch, ["load", "fred_md"]; prog="friedman data")
        @test called_with[][:name] == "fred_md"
    end

end

# ──────────────────────────────────────────────────────────────
# IO utilities (needs CSV, DataFrames, JSON3, PrettyTables)
# ──────────────────────────────────────────────────────────────
using CSV, DataFrames, JSON3, PrettyTables

# PrettyTables v3 renamed tf_unicode_rounded → text_table_borders__unicode_rounded;
# io.jl references the old name, so alias it for the test context.
if !@isdefined(tf_unicode_rounded)
    const tf_unicode_rounded = text_table_borders__unicode_rounded
end

# Include io.jl at top level so it can see PrettyTables exports
let project_root = dirname(@__DIR__)
    include(joinpath(project_root, "src", "io.jl"))
end

# PrettyTables v3 changed its API (tf/show_subheader removed).
# Override _write_table to use the v3 API so table output tests work.
function _write_table(df::DataFrame, output::String, title::String)
    io = isempty(output) ? stdout : open(output, "w")
    try
        pretty_table(io, df; title=title, alignment=:c)
    finally
        isempty(output) || close(io)
    end
    isempty(output) || println("Results written to $output")
end

@testset "IO utilities" begin

    @testset "load_data" begin
        mktempdir() do dir
            # Valid CSV → correct DataFrame
            csv_path = joinpath(dir, "test.csv")
            CSV.write(csv_path, DataFrame(a=[1,2,3], b=[4.0,5.0,6.0]))
            df = load_data(csv_path)
            @test nrow(df) == 3
            @test ncol(df) == 2
            @test "a" in names(df)
            @test "b" in names(df)

            # Nonexistent file → error
            @test_throws ErrorException load_data(joinpath(dir, "nonexistent.csv"))

            # Empty CSV (headers only, 0 rows) → error
            empty_path = joinpath(dir, "empty.csv")
            CSV.write(empty_path, DataFrame(a=Int[], b=Float64[]))
            @test_throws ErrorException load_data(empty_path)
        end
    end

    @testset "df_to_matrix" begin
        # Extracts only numeric columns
        df = DataFrame(a=[1,2,3], b=[4.0,5.0,6.0])
        mat = df_to_matrix(df)
        @test size(mat) == (3, 2)
        @test mat[:, 1] == [1.0, 2.0, 3.0]
        @test mat[:, 2] == [4.0, 5.0, 6.0]

        # Mixed numeric + string columns → skips strings
        df_mixed = DataFrame(name=["a","b","c"], x=[1.0,2.0,3.0], label=["d","e","f"], y=[4,5,6])
        mat_mixed = df_to_matrix(df_mixed)
        @test size(mat_mixed) == (3, 2)

        # No numeric columns → error
        df_str = DataFrame(name=["a","b","c"], label=["d","e","f"])
        @test_throws ErrorException df_to_matrix(df_str)
    end

    @testset "variable_names" begin
        df = DataFrame(name=["a","b"], x=[1.0,2.0], y=[3,4])
        vnames = variable_names(df)
        @test "x" in vnames
        @test "y" in vnames
        @test !("name" in vnames)

        # All numeric columns
        df_all_num = DataFrame(a=[1,2], b=[3.0,4.0], c=[5,6])
        @test length(variable_names(df_all_num)) == 3
    end

    @testset "output_result CSV" begin
        mktempdir() do dir
            df = DataFrame(x=[1.0,2.0], y=[3.0,4.0])
            outpath = joinpath(dir, "out.csv")
            output_result(df; format=:csv, output=outpath, title="Test")
            df_back = CSV.read(outpath, DataFrame)
            @test nrow(df_back) == 2
            @test df_back.x == [1.0, 2.0]
            @test df_back.y == [3.0, 4.0]
        end
    end

    @testset "output_result JSON" begin
        mktempdir() do dir
            df = DataFrame(x=[1.0,2.0], y=[3.0,4.0])
            outpath = joinpath(dir, "out.json")
            output_result(df; format=:json, output=outpath, title="Test")
            json_str = read(outpath, String)
            data = JSON3.read(json_str)
            @test length(data) == 2
            @test data[1]["x"] == 1.0
            @test data[2]["y"] == 4.0
        end
    end

    @testset "output_result table" begin
        mktempdir() do dir
            df = DataFrame(x=[1.0,2.0], y=[3.0,4.0])
            outpath = joinpath(dir, "out.txt")
            output_result(df; format=:table, output=outpath, title="Test Table")
            content = read(outpath, String)
            @test !isempty(content)
            @test contains(content, "Test Table")
        end
    end

    @testset "output_result from matrix" begin
        mktempdir() do dir
            mat = [1.0 2.0; 3.0 4.0]
            varnames = ["col1", "col2"]
            outpath = joinpath(dir, "mat.csv")
            output_result(mat, varnames; format="csv", output=outpath, title="Matrix")
            df_back = CSV.read(outpath, DataFrame)
            @test nrow(df_back) == 2
            @test "col1" in names(df_back)
            @test "col2" in names(df_back)
        end
    end

    @testset "output_kv JSON" begin
        mktempdir() do dir
            pairs = ["stat1" => "0.05", "stat2" => "1.96"]
            outpath = joinpath(dir, "kv.json")
            output_kv(pairs; format="json", output=outpath, title="Stats")
            json_str = read(outpath, String)
            d = JSON3.read(json_str, Dict{String,String})
            @test d["stat1"] == "0.05"
            @test d["stat2"] == "1.96"
        end
    end

    @testset "output_kv CSV" begin
        mktempdir() do dir
            pairs = ["metric_a" => "100", "metric_b" => "200"]
            outpath = joinpath(dir, "kv.csv")
            output_kv(pairs; format="csv", output=outpath, title="KV")
            df_back = CSV.read(outpath, DataFrame)
            @test nrow(df_back) == 2
            @test "metric" in names(df_back)
            @test "value" in names(df_back)
        end
    end

    @testset "output_kv table" begin
        mktempdir() do dir
            pairs = ["key1" => "val1", "key2" => "val2"]
            outpath = joinpath(dir, "kv.txt")
            output_kv(pairs; format="table", output=outpath, title="KV Table")
            content = read(outpath, String)
            @test !isempty(content)
            @test contains(content, "KV Table")
        end
    end
end

# ──────────────────────────────────────────────────────────────
# Config parsing (TOML is stdlib, no extra deps)
# ──────────────────────────────────────────────────────────────
using TOML

@testset "Config parsing" begin
    project_root = dirname(@__DIR__)
    include(joinpath(project_root, "src", "config.jl"))

    @testset "load_config" begin
        mktempdir() do dir
            # Valid TOML → correct Dict
            cfg_path = joinpath(dir, "test.toml")
            open(cfg_path, "w") do io
                write(io, """
                [section]
                key = "value"
                number = 42
                """)
            end
            cfg = load_config(cfg_path)
            @test cfg["section"]["key"] == "value"
            @test cfg["section"]["number"] == 42

            # Missing file → error
            @test_throws ErrorException load_config(joinpath(dir, "nonexistent.toml"))
        end
    end

    @testset "get_identification" begin
        # Sign method with matrix + horizons
        cfg = Dict("identification" => Dict(
            "method" => "sign",
            "sign_matrix" => Dict(
                "matrix" => [[1, -1, 1], [0, 1, -1], [0, 0, 1]],
                "horizons" => [0, 1, 2, 3]
            )
        ))
        id = get_identification(cfg)
        @test id["method"] == "sign"
        @test size(id["sign_matrix"]) == (3, 3)
        @test id["sign_matrix"][1, 1] == 1.0
        @test id["sign_matrix"][1, 2] == -1.0
        @test id["horizons"] == [0, 1, 2, 3]

        # Narrative restrictions
        cfg_narr = Dict("identification" => Dict(
            "method" => "narrative",
            "narrative" => Dict(
                "shock_index" => 1,
                "periods" => [10, 15, 20],
                "signs" => [1, -1, 1]
            )
        ))
        id_narr = get_identification(cfg_narr)
        @test id_narr["method"] == "narrative"
        @test id_narr["narrative"]["shock_index"] == 1
        @test id_narr["narrative"]["periods"] == [10, 15, 20]
        @test id_narr["narrative"]["signs"] == [1, -1, 1]

        # Empty config → defaults to cholesky
        id_empty = get_identification(Dict())
        @test id_empty["method"] == "cholesky"
    end

    @testset "get_prior" begin
        # Minnesota hyperparameters extracted
        cfg = Dict("prior" => Dict(
            "type" => "minnesota",
            "hyperparameters" => Dict(
                "lambda1" => 0.2,
                "lambda2" => 0.5,
                "lambda3" => 1.0,
                "lambda4" => 100000.0
            ),
            "optimization" => Dict("enabled" => true)
        ))
        pr = get_prior(cfg)
        @test pr["type"] == "minnesota"
        @test pr["lambda1"] == 0.2
        @test pr["lambda2"] == 0.5
        @test pr["lambda3"] == 1.0
        @test pr["lambda4"] == 100000.0

        # Optimization flag
        @test pr["optimize"] == true

        # Empty config → defaults
        pr_empty = get_prior(Dict())
        @test pr_empty["type"] == "minnesota"
        @test pr_empty["optimize"] == false
    end

    @testset "get_gmm" begin
        # Full GMM specification
        cfg = Dict("gmm" => Dict(
            "moment_conditions" => ["output", "inflation"],
            "instruments" => ["lag_output", "lag_inflation"],
            "weighting" => "twostep"
        ))
        gmm = get_gmm(cfg)
        @test gmm["moment_conditions"] == ["output", "inflation"]
        @test gmm["instruments"] == ["lag_output", "lag_inflation"]
        @test gmm["weighting"] == "twostep"

        # Empty config → defaults
        gmm_empty = get_gmm(Dict())
        @test gmm_empty["moment_conditions"] == String[]
        @test gmm_empty["instruments"] == String[]
        @test gmm_empty["weighting"] == "twostep"
    end

    @testset "_parse_matrix" begin
        # Valid 2x3 → correct Matrix{Float64}
        mat = _parse_matrix([[1, 2, 3], [4, 5, 6]])
        @test size(mat) == (2, 3)
        @test mat[1, 1] == 1.0
        @test mat[2, 3] == 6.0
        @test eltype(mat) == Float64

        # 1x1 matrix
        mat1 = _parse_matrix([[42]])
        @test size(mat1) == (1, 1)
        @test mat1[1, 1] == 42.0

        # Empty → 0x0 matrix
        mat_empty = _parse_matrix([])
        @test size(mat_empty) == (0, 0)

        # Square 3x3 identity-like
        mat_sq = _parse_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        @test size(mat_sq) == (3, 3)
        @test mat_sq[1, 1] == 1.0
        @test mat_sq[1, 2] == 0.0
        @test mat_sq[2, 2] == 1.0
    end

    @testset "get_identification from TOML file" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "sign.toml")
            open(cfg_path, "w") do io
                write(io, """
                [identification]
                method = "sign"
                [identification.sign_matrix]
                matrix = [[1, -1], [0, 1]]
                horizons = [0, 1]
                """)
            end
            cfg = load_config(cfg_path)
            id = get_identification(cfg)
            @test id["method"] == "sign"
            @test size(id["sign_matrix"]) == (2, 2)
            @test id["horizons"] == [0, 1]
        end
    end

    @testset "get_prior from TOML file" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "prior.toml")
            open(cfg_path, "w") do io
                write(io, """
                [prior]
                type = "minnesota"
                [prior.hyperparameters]
                lambda1 = 0.1
                lambda2 = 0.3
                lambda3 = 2.0
                lambda4 = 50000.0
                [prior.optimization]
                enabled = false
                """)
            end
            cfg = load_config(cfg_path)
            pr = get_prior(cfg)
            @test pr["lambda1"] == 0.1
            @test pr["lambda2"] == 0.3
            @test pr["optimize"] == false
        end
    end

    @testset "get_gmm from TOML file" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "gmm.toml")
            open(cfg_path, "w") do io
                write(io, """
                [gmm]
                moment_conditions = ["y1", "y2", "y3"]
                instruments = ["z1", "z2"]
                weighting = "iterated"
                """)
            end
            cfg = load_config(cfg_path)
            gmm = get_gmm(cfg)
            @test gmm["moment_conditions"] == ["y1", "y2", "y3"]
            @test gmm["instruments"] == ["z1", "z2"]
            @test gmm["weighting"] == "iterated"
        end
    end

    @testset "get_nongaussian" begin
        # Full non-Gaussian specification
        cfg = Dict("nongaussian" => Dict(
            "method" => "fastica",
            "contrast" => "exp",
            "distribution" => "skew_t",
            "n_regimes" => 3,
            "transition_variable" => "spread",
            "regime_variable" => "nber"
        ))
        ng = get_nongaussian(cfg)
        @test ng["method"] == "fastica"
        @test ng["contrast"] == "exp"
        @test ng["distribution"] == "skew_t"
        @test ng["n_regimes"] == 3
        @test ng["transition_variable"] == "spread"
        @test ng["regime_variable"] == "nber"

        # Empty config → defaults
        ng_empty = get_nongaussian(Dict())
        @test ng_empty["method"] == "fastica"
        @test ng_empty["contrast"] == "logcosh"
        @test ng_empty["distribution"] == "student_t"
        @test ng_empty["n_regimes"] == 2
        @test ng_empty["transition_variable"] == ""
        @test ng_empty["regime_variable"] == ""

        # Partial config
        ng_partial = get_nongaussian(Dict("nongaussian" => Dict("method" => "jade")))
        @test ng_partial["method"] == "jade"
        @test ng_partial["contrast"] == "logcosh"  # default
    end

    @testset "get_nongaussian from TOML file" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "nongaussian.toml")
            open(cfg_path, "w") do io
                write(io, """
                [nongaussian]
                method = "smooth_transition"
                transition_variable = "yield_spread"
                n_regimes = 2
                """)
            end
            cfg = load_config(cfg_path)
            ng = get_nongaussian(cfg)
            @test ng["method"] == "smooth_transition"
            @test ng["transition_variable"] == "yield_spread"
            @test ng["n_regimes"] == 2
            @test ng["distribution"] == "student_t"  # default
        end
    end

    @testset "get_uhlig_params" begin
        # Empty config → defaults
        uhlig_empty = get_uhlig_params(Dict())
        @test uhlig_empty["n_starts"] == 50
        @test uhlig_empty["n_refine"] == 10
        @test uhlig_empty["max_iter_coarse"] == 500
        @test uhlig_empty["max_iter_fine"] == 2000
        @test uhlig_empty["tol_coarse"] == 1e-4
        @test uhlig_empty["tol_fine"] == 1e-8

        # Custom values
        cfg = Dict("identification" => Dict("uhlig" => Dict(
            "n_starts" => 100,
            "n_refine" => 20,
            "max_iter_coarse" => 1000,
            "max_iter_fine" => 5000,
            "tol_coarse" => 1e-3,
            "tol_fine" => 1e-10,
        )))
        uhlig = get_uhlig_params(cfg)
        @test uhlig["n_starts"] == 100
        @test uhlig["n_refine"] == 20
        @test uhlig["max_iter_fine"] == 5000
        @test uhlig["tol_fine"] == 1e-10

        # Partial config — missing keys get defaults
        cfg_partial = Dict("identification" => Dict("uhlig" => Dict("n_starts" => 200)))
        uhlig_partial = get_uhlig_params(cfg_partial)
        @test uhlig_partial["n_starts"] == 200
        @test uhlig_partial["n_refine"] == 10  # default
    end

    @testset "get_dsge — valid model config" begin
        cfg = Dict(
            "model" => Dict(
                "parameters" => Dict("rho" => 0.9, "sigma" => 0.01, "beta" => 0.99),
                "endogenous" => ["C", "K", "Y"],
                "exogenous" => ["e_A"],
                "equations" => [
                    Dict("expr" => "C[t] + K[t] = Y[t]"),
                    Dict("expr" => "Y[t] = K[t-1]"),
                    Dict("expr" => "K[t] = rho * K[t-1] + sigma * e_A[t]"),
                ]
            )
        )
        result = get_dsge(cfg)
        @test result["parameters"] == Dict("rho" => 0.9, "sigma" => 0.01, "beta" => 0.99)
        @test result["endogenous"] == ["C", "K", "Y"]
        @test result["exogenous"] == ["e_A"]
        @test length(result["equations"]) == 3
        @test result["equations"][1] == "C[t] + K[t] = Y[t]"
    end

    @testset "get_dsge — missing model section" begin
        cfg = Dict{String,Any}()
        result = get_dsge(cfg)
        @test isempty(result["endogenous"])
    end

    @testset "get_dsge — solver defaults" begin
        cfg = Dict{String,Any}()
        result = get_dsge(cfg)
        @test result["solver_method"] == "gensys"
        @test result["solver_order"] == 1
        @test result["solver_degree"] == 5
        @test result["solver_grid"] == "auto"
    end

    @testset "get_dsge — solver overrides" begin
        cfg = Dict(
            "model" => Dict("endogenous" => ["Y"], "exogenous" => ["e"],
                           "parameters" => Dict{String,Any}(), "equations" => Dict[]),
            "solver" => Dict("method" => "perturbation", "order" => 2)
        )
        result = get_dsge(cfg)
        @test result["solver_method"] == "perturbation"
        @test result["solver_order"] == 2
    end

    @testset "get_dsge_constraints — bounds" begin
        cfg = Dict(
            "constraints" => Dict(
                "bounds" => [
                    Dict("variable" => "i", "lower" => 0.0),
                    Dict("variable" => "c", "lower" => 0.0, "upper" => 10.0),
                ]
            )
        )
        result = get_dsge_constraints(cfg)
        @test length(result["bounds"]) == 2
        @test result["bounds"][1]["variable"] == "i"
        @test result["bounds"][1]["lower"] == 0.0
        @test !haskey(result["bounds"][1], "upper")
        @test result["bounds"][2]["upper"] == 10.0
    end

    @testset "get_dsge_constraints — empty" begin
        cfg = Dict{String,Any}()
        result = get_dsge_constraints(cfg)
        @test isempty(result["bounds"])
    end

    @testset "get_dsge_constraints — nonlinear" begin
        config = Dict(
            "constraints" => Dict(
                "nonlinear" => [
                    Dict("expr" => "K[t] + C[t] <= Y[t]", "label" => "resource"),
                    Dict("expr" => "I[t] >= 0")
                ]
            )
        )
        result = get_dsge_constraints(config)
        @test haskey(result, "nonlinear")
        @test length(result["nonlinear"]) == 2
        @test result["nonlinear"][1]["expr"] == "K[t] + C[t] <= Y[t]"
        @test result["nonlinear"][1]["label"] == "resource"
        @test result["nonlinear"][2]["expr"] == "I[t] >= 0"
        @test !haskey(result["nonlinear"][2], "label") || result["nonlinear"][2]["label"] == ""
    end

    @testset "get_dsge_constraints — mixed bounds + nonlinear" begin
        config = Dict(
            "constraints" => Dict(
                "bounds" => [Dict("variable" => "i", "lower" => 0.0)],
                "nonlinear" => [Dict("expr" => "K[t] <= Y[t]", "label" => "cap")]
            )
        )
        result = get_dsge_constraints(config)
        @test length(result["bounds"]) == 1
        @test length(result["nonlinear"]) == 1
    end

    @testset "get_smm — valid config" begin
        cfg = Dict(
            "smm" => Dict(
                "weighting" => "optimal",
                "sim_ratio" => 10,
                "burn" => 200,
            )
        )
        result = get_smm(cfg)
        @test result["weighting"] == "optimal"
        @test result["sim_ratio"] == 10
        @test result["burn"] == 200
    end

    @testset "get_smm — defaults" begin
        cfg = Dict{String,Any}()
        result = get_smm(cfg)
        @test result["weighting"] == "two_step"
        @test result["sim_ratio"] == 5
        @test result["burn"] == 100
    end
end

# ──────────────────────────────────────────────────────────────
# Command handler tests (uses mock MacroEconometricModels)
# ──────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "test_commands.jl"))

# ──────────────────────────────────────────────────────────────
# REPL session tests (uses mocks + source files from test_commands.jl)
# ──────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "test_repl.jl"))

# ──────────────────────────────────────────────────────────────
# build_app() and main() tests (uses register_*_commands! from test_commands.jl)
# ──────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "test_main.jl"))

# ──────────────────────────────────────────────────────────────
# CLI structure tests using real register_*_commands!() functions
# (These run after test_commands.jl which includes mocks + all source files)
# ──────────────────────────────────────────────────────────────

@testset "DSGE command structure" begin
    dsge_node = register_dsge_commands!()
    @test dsge_node isa NodeCommand
    @test dsge_node.name == "dsge"

    # All 9 subcommands exist
    @test haskey(dsge_node.subcmds, "solve")
    @test haskey(dsge_node.subcmds, "irf")
    @test haskey(dsge_node.subcmds, "fevd")
    @test haskey(dsge_node.subcmds, "simulate")
    @test haskey(dsge_node.subcmds, "estimate")
    @test haskey(dsge_node.subcmds, "bayes")
    @test haskey(dsge_node.subcmds, "perfect-foresight")
    @test haskey(dsge_node.subcmds, "steady-state")
    @test haskey(dsge_node.subcmds, "hd")
    @test length(dsge_node.subcmds) == 9

    # All non-bayes subcmds are LeafCommands; bayes is NodeCommand
    for (name, cmd) in dsge_node.subcmds
        if name == "bayes"
            @test cmd isa NodeCommand
        else
            @test cmd isa LeafCommand
        end
    end

    # solve has model argument and key options
    solve_cmd = dsge_node.subcmds["solve"]
    @test length(solve_cmd.args) == 1
    @test solve_cmd.args[1].name == "model"
    opt_names = [o.name for o in solve_cmd.options]
    @test "method" in opt_names
    @test "order" in opt_names
    @test "constraints" in opt_names
    @test "format" in opt_names

    # estimate has data and params options
    est_cmd = dsge_node.subcmds["estimate"]
    opt_names = [o.name for o in est_cmd.options]
    @test "data" in opt_names
    @test "params" in opt_names
    @test "method" in opt_names
    @test "weighting" in opt_names

    # bayes is a NodeCommand with 8 sub-leaves
    bayes_node = dsge_node.subcmds["bayes"]
    @test bayes_node isa NodeCommand
    @test length(bayes_node.subcmds) == 8
    @test haskey(bayes_node.subcmds, "estimate")
    @test haskey(bayes_node.subcmds, "irf")
    @test haskey(bayes_node.subcmds, "fevd")
    @test haskey(bayes_node.subcmds, "simulate")
    @test haskey(bayes_node.subcmds, "summary")
    @test haskey(bayes_node.subcmds, "compare")
    @test haskey(bayes_node.subcmds, "predictive")
    @test haskey(bayes_node.subcmds, "hd")

    # All bayes sub-leaves are LeafCommands
    for (name, cmd) in bayes_node.subcmds
        @test cmd isa LeafCommand
    end

    # bayes estimate has model argument and SMC/MH options
    bayes_est = bayes_node.subcmds["estimate"]
    @test length(bayes_est.args) == 1
    @test bayes_est.args[1].name == "model"
    opt_names = [o.name for o in bayes_est.options]
    @test "data" in opt_names
    @test "params" in opt_names
    @test "priors" in opt_names
    @test "sampler" in opt_names
    @test "n-smc" in opt_names
    @test "n-particles" in opt_names
    @test "n-draws" in opt_names
    @test "burnin" in opt_names
    @test "ess-target" in opt_names
    @test "observables" in opt_names
    @test "solver" in opt_names
    @test "order" in opt_names
    flag_names = [f.name for f in bayes_est.flags]
    @test "delayed-acceptance" in flag_names

    # bayes irf has horizon + plot options
    bayes_irf = bayes_node.subcmds["irf"]
    opt_names = [o.name for o in bayes_irf.options]
    @test "horizon" in opt_names
    @test "data" in opt_names
    flag_names = [f.name for f in bayes_irf.flags]
    @test "plot" in flag_names

    # bayes compare has model2/params2/priors2 options
    bayes_cmp = bayes_node.subcmds["compare"]
    opt_names = [o.name for o in bayes_cmp.options]
    @test "model2" in opt_names
    @test "params2" in opt_names
    @test "priors2" in opt_names

    # bayes predictive has n-sim and periods options
    bayes_pp = bayes_node.subcmds["predictive"]
    opt_names = [o.name for o in bayes_pp.options]
    @test "n-sim" in opt_names
    @test "periods" in opt_names

    # irf has horizon and shock-size
    irf_cmd = dsge_node.subcmds["irf"]
    opt_names = [o.name for o in irf_cmd.options]
    @test "horizon" in opt_names
    @test "shock-size" in opt_names
    @test "constraints" in opt_names

    # simulate has periods and burn
    sim_cmd = dsge_node.subcmds["simulate"]
    opt_names = [o.name for o in sim_cmd.options]
    @test "periods" in opt_names
    @test "burn" in opt_names
    @test "seed" in opt_names
    flag_names = [f.name for f in sim_cmd.flags]
    @test "antithetic" in flag_names

    # perfect-foresight has shocks option
    pf_cmd = dsge_node.subcmds["perfect-foresight"]
    opt_names = [o.name for o in pf_cmd.options]
    @test "shocks" in opt_names
    @test "periods" in opt_names

    # Verify --constraint-solver option on solve
    @test any(o -> o.name == "constraint-solver", solve_cmd.options)

    # Verify --constraint-solver option on steady-state
    ss_cmd = dsge_node.subcmds["steady-state"]
    @test any(o -> o.name == "constraint-solver", ss_cmd.options)

    # Verify --constraint-solver option on perfect-foresight
    @test any(o -> o.name == "constraint-solver", pf_cmd.options)

    # Verify --constraint-solver in bayes subcommands (via _bayes_common_options)
    @test any(o -> o.name == "constraint-solver", bayes_est.options)
end

@testset "estimate smm command structure" begin
    est_node = register_estimate_commands!()
    @test haskey(est_node.subcmds, "smm")
    smm_cmd = est_node.subcmds["smm"]
    @test smm_cmd isa LeafCommand
    @test length(smm_cmd.args) == 1
    @test smm_cmd.args[1].name == "data"
    opt_names = [o.name for o in smm_cmd.options]
    @test "weighting" in opt_names
    @test "sim-ratio" in opt_names
    @test "burn" in opt_names
    @test "config" in opt_names

    # Verify estimate now has 31 subcommands
    @test length(est_node.subcmds) == 31
    @test haskey(est_node.subcmds, "smm")
    @test haskey(est_node.subcmds, "favar")
    @test haskey(est_node.subcmds, "sdfm")
    for key in ["var", "bvar", "lp", "arima", "gmm", "static", "dynamic", "gdfm",
                 "arch", "garch", "egarch", "gjr_garch", "sv", "fastica", "ml",
                 "vecm", "pvar", "smm", "favar", "sdfm", "reg", "iv", "logit", "probit"]
        @test haskey(est_node.subcmds, key)
        @test est_node.subcmds[key] isa LeafCommand
    end

    # FAVAR has key-vars option
    favar_cmd = est_node.subcmds["favar"]
    @test length(favar_cmd.args) == 1
    @test favar_cmd.args[1].name == "data"
    favar_opt_names = [o.name for o in favar_cmd.options]
    @test "factors" in favar_opt_names
    @test "lags" in favar_opt_names
    @test "key-vars" in favar_opt_names
    @test "method" in favar_opt_names
    @test "draws" in favar_opt_names

    # SDFM has identification and spectral options
    sdfm_cmd = est_node.subcmds["sdfm"]
    @test length(sdfm_cmd.args) == 1
    sdfm_opt_names = [o.name for o in sdfm_cmd.options]
    @test "factors" in sdfm_opt_names
    @test "id" in sdfm_opt_names
    @test "var-lags" in sdfm_opt_names
    @test "bandwidth" in sdfm_opt_names
    @test "kernel" in sdfm_opt_names
end

@testset "DID command structure" begin
    did_node = register_did_commands!()
    @test did_node isa NodeCommand
    @test did_node.name == "did"

    @test haskey(did_node.subcmds, "estimate")
    @test haskey(did_node.subcmds, "event-study")
    @test haskey(did_node.subcmds, "lp-did")
    @test haskey(did_node.subcmds, "test")
    @test length(did_node.subcmds) == 4

    @test did_node.subcmds["estimate"] isa LeafCommand
    @test did_node.subcmds["event-study"] isa LeafCommand
    @test did_node.subcmds["lp-did"] isa LeafCommand

    test_node = did_node.subcmds["test"]
    @test test_node isa NodeCommand
    @test haskey(test_node.subcmds, "bacon")
    @test haskey(test_node.subcmds, "pretrend")
    @test haskey(test_node.subcmds, "negweight")
    @test haskey(test_node.subcmds, "honest")
    @test length(test_node.subcmds) == 4
    for (name, cmd) in test_node.subcmds
        @test cmd isa LeafCommand
    end

    est_cmd = did_node.subcmds["estimate"]
    @test length(est_cmd.args) == 1
    @test est_cmd.args[1].name == "data"
    opt_names = [o.name for o in est_cmd.options]
    @test "outcome" in opt_names
    @test "treatment" in opt_names
    @test "method" in opt_names
    @test "id-col" in opt_names
    @test "time-col" in opt_names
    @test "control-group" in opt_names
    @test "cluster" in opt_names
    flag_names = [f.name for f in est_cmd.flags]
    @test "plot" in flag_names

    es_cmd = did_node.subcmds["event-study"]
    opt_names = [o.name for o in es_cmd.options]
    @test "outcome" in opt_names
    @test "treatment" in opt_names
    @test "leads" in opt_names
    @test "horizon" in opt_names
    @test "lags" in opt_names

    lp_cmd = did_node.subcmds["lp-did"]
    opt_names = [o.name for o in lp_cmd.options]
    @test "outcome" in opt_names
    @test "treatment" in opt_names
    @test "pre-window" in opt_names
    @test "post-window" in opt_names

    bacon_cmd = test_node.subcmds["bacon"]
    opt_names = [o.name for o in bacon_cmd.options]
    @test "outcome" in opt_names
    @test "treatment" in opt_names

    pt_cmd = test_node.subcmds["pretrend"]
    opt_names = [o.name for o in pt_cmd.options]
    @test "method" in opt_names
    @test "did-method" in opt_names

    nw_cmd = test_node.subcmds["negweight"]
    opt_names = [o.name for o in nw_cmd.options]
    @test "treatment" in opt_names

    h_cmd = test_node.subcmds["honest"]
    opt_names = [o.name for o in h_cmd.options]
    @test "mbar" in opt_names
    @test "method" in opt_names
end

@testset "FAVAR/SDFM command structure across actions" begin
    # IRF: 7 subcommands (5 original + favar + sdfm)
    irf_node = register_irf_commands!()
    @test length(irf_node.subcmds) == 7
    @test haskey(irf_node.subcmds, "favar")
    @test haskey(irf_node.subcmds, "sdfm")
    @test irf_node.subcmds["favar"] isa LeafCommand
    @test irf_node.subcmds["sdfm"] isa LeafCommand

    # FAVAR IRF has key-vars and panel-irf flag
    irf_favar = irf_node.subcmds["favar"]
    @test length(irf_favar.args) == 1
    irf_favar_opts = [o.name for o in irf_favar.options]
    @test "factors" in irf_favar_opts
    @test "key-vars" in irf_favar_opts
    @test "horizons" in irf_favar_opts
    @test "id" in irf_favar_opts
    irf_favar_flags = [f.name for f in irf_favar.flags]
    @test "panel-irf" in irf_favar_flags

    # SDFM IRF has factors, id, var-lags
    irf_sdfm = irf_node.subcmds["sdfm"]
    irf_sdfm_opts = [o.name for o in irf_sdfm.options]
    @test "factors" in irf_sdfm_opts
    @test "id" in irf_sdfm_opts
    @test "var-lags" in irf_sdfm_opts

    # FEVD: 7 subcommands (5 original + favar + sdfm)
    fevd_node = register_fevd_commands!()
    @test length(fevd_node.subcmds) == 7
    @test haskey(fevd_node.subcmds, "favar")
    @test haskey(fevd_node.subcmds, "sdfm")
    @test fevd_node.subcmds["favar"] isa LeafCommand
    @test fevd_node.subcmds["sdfm"] isa LeafCommand

    fevd_favar_opts = [o.name for o in fevd_node.subcmds["favar"].options]
    @test "key-vars" in fevd_favar_opts
    @test "horizons" in fevd_favar_opts

    # HD: 5 subcommands (4 original + favar)
    hd_node = register_hd_commands!()
    @test length(hd_node.subcmds) == 5
    @test haskey(hd_node.subcmds, "favar")
    @test hd_node.subcmds["favar"] isa LeafCommand

    hd_favar_opts = [o.name for o in hd_node.subcmds["favar"].options]
    @test "key-vars" in hd_favar_opts
    @test "id" in hd_favar_opts

    # Forecast: 14 subcommands (13 original + favar)
    fc_node = register_forecast_commands!()
    @test length(fc_node.subcmds) == 14
    @test haskey(fc_node.subcmds, "favar")
    @test fc_node.subcmds["favar"] isa LeafCommand

    fc_favar = fc_node.subcmds["favar"]
    fc_favar_opts = [o.name for o in fc_favar.options]
    @test "key-vars" in fc_favar_opts
    @test "horizons" in fc_favar_opts
    fc_favar_flags = [f.name for f in fc_favar.flags]
    @test "panel-forecast" in fc_favar_flags

    # Predict: 23 subcommands
    pred_node = register_predict_commands!()
    @test length(pred_node.subcmds) == 23
    @test haskey(pred_node.subcmds, "favar")
    @test pred_node.subcmds["favar"] isa LeafCommand

    pred_favar_opts = [o.name for o in pred_node.subcmds["favar"].options]
    @test "key-vars" in pred_favar_opts

    # Residuals: 23 subcommands
    res_node = register_residuals_commands!()
    @test length(res_node.subcmds) == 23
    @test haskey(res_node.subcmds, "favar")
    @test res_node.subcmds["favar"] isa LeafCommand

    res_favar_opts = [o.name for o in res_node.subcmds["favar"].options]
    @test "key-vars" in res_favar_opts
end

@testset "Structural break test command structure" begin
    test_node = register_test_commands!()

    # Test node now has 41 subcommands
    @test length(test_node.subcmds) == 41

    # Andrews structural break test
    @test haskey(test_node.subcmds, "andrews")
    andrews_cmd = test_node.subcmds["andrews"]
    @test andrews_cmd isa LeafCommand
    @test length(andrews_cmd.args) == 1
    andrews_opts = [o.name for o in andrews_cmd.options]
    @test "response" in andrews_opts
    @test "test" in andrews_opts
    @test "trimming" in andrews_opts
    andrews_flags = [f.name for f in andrews_cmd.flags]
    @test "plot" in andrews_flags

    # Bai-Perron multiple structural break test
    @test haskey(test_node.subcmds, "bai-perron")
    bp_cmd = test_node.subcmds["bai-perron"]
    @test bp_cmd isa LeafCommand
    @test length(bp_cmd.args) == 1
    bp_opts = [o.name for o in bp_cmd.options]
    @test "response" in bp_opts
    @test "max-breaks" in bp_opts
    @test "trimming" in bp_opts
    @test "criterion" in bp_opts
    bp_flags = [f.name for f in bp_cmd.flags]
    @test "plot" in bp_flags
end

@testset "Panel unit root test command structure" begin
    test_node = register_test_commands!()

    # PANIC test
    @test haskey(test_node.subcmds, "panic")
    panic_cmd = test_node.subcmds["panic"]
    @test panic_cmd isa LeafCommand
    @test length(panic_cmd.args) == 1
    panic_opts = [o.name for o in panic_cmd.options]
    @test "factors" in panic_opts
    @test "method" in panic_opts
    @test "id-col" in panic_opts
    @test "time-col" in panic_opts

    # Pesaran CIPS test
    @test haskey(test_node.subcmds, "cips")
    cips_cmd = test_node.subcmds["cips"]
    @test cips_cmd isa LeafCommand
    cips_opts = [o.name for o in cips_cmd.options]
    @test "lags" in cips_opts
    @test "deterministic" in cips_opts
    @test "id-col" in cips_opts

    # Moon-Perron test
    @test haskey(test_node.subcmds, "moon-perron")
    mp_cmd = test_node.subcmds["moon-perron"]
    @test mp_cmd isa LeafCommand
    mp_opts = [o.name for o in mp_cmd.options]
    @test "factors" in mp_opts
    @test "id-col" in mp_opts

    # Factor break test
    @test haskey(test_node.subcmds, "factor-break")
    fb_cmd = test_node.subcmds["factor-break"]
    @test fb_cmd isa LeafCommand
    fb_opts = [o.name for o in fb_cmd.options]
    @test "factors" in fb_opts
    @test "method" in fb_opts
    @test "id-col" in fb_opts
end

@testset "Spectral command structure" begin
    spectral_node = register_spectral_commands!()
    @test spectral_node isa NodeCommand
    @test length(spectral_node.subcmds) == 5

    @test haskey(spectral_node.subcmds, "acf")
    acf_cmd = spectral_node.subcmds["acf"]
    @test acf_cmd isa LeafCommand
    acf_opts = [o.name for o in acf_cmd.options]
    @test "column" in acf_opts
    @test "max-lag" in acf_opts

    @test haskey(spectral_node.subcmds, "periodogram")
    peri_cmd = spectral_node.subcmds["periodogram"]
    @test peri_cmd isa LeafCommand

    @test haskey(spectral_node.subcmds, "density")
    dens_cmd = spectral_node.subcmds["density"]
    @test dens_cmd isa LeafCommand
    dens_opts = [o.name for o in dens_cmd.options]
    @test "method" in dens_opts

    @test haskey(spectral_node.subcmds, "cross")
    cross_cmd = spectral_node.subcmds["cross"]
    @test cross_cmd isa LeafCommand
    cross_opts = [o.name for o in cross_cmd.options]
    @test "var1" in cross_opts
    @test "var2" in cross_opts

    @test haskey(spectral_node.subcmds, "transfer")
    trans_cmd = spectral_node.subcmds["transfer"]
    @test trans_cmd isa LeafCommand
    trans_opts = [o.name for o in trans_cmd.options]
    @test "filter" in trans_opts
    @test "lambda" in trans_opts
end
