using Test

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
        entry = Entry("friedman", node; version=v"0.1.2")
        buf = IOBuffer()
        print_help(buf, entry)
        help_text = String(take!(buf))
        @test contains(help_text, "0.1.2")

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
        entry = Entry("friedman", outer_node; version=v"0.1.2")
        version_output = strip(capture_stdout(() -> dispatch(entry, ["--version"])))
        @test contains(version_output, "0.1.2")

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
        @test contains(v_output, "0.1.2")
    end

    @testset "Non-Gaussian SVAR command structure" begin
        handler = (; kwargs...) -> kwargs

        # Simulate the structure that register_nongaussian_commands!() creates
        ng_fastica = LeafCommand("fastica", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("method"; type=String, default="fastica", description="fastica|infomax|jade"),
                Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="ICA-based non-Gaussian SVAR identification")

        ng_ml = LeafCommand("ml", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("distribution"; short="d", type=String, default="student_t", description="student_t|skew_t|ghd"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="Maximum likelihood non-Gaussian SVAR identification")

        ng_heteroskedasticity = LeafCommand("heteroskedasticity", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("method"; type=String, default="markov", description="markov|garch|smooth_transition|external"),
                Option("config"; type=String, default="", description="TOML config"),
                Option("regimes"; type=Int, default=2, description="Number of regimes"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="Heteroskedasticity-based SVAR identification")

        ng_normality = LeafCommand("normality", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="Normality test suite for VAR residuals")

        ng_identifiability = LeafCommand("identifiability", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("lags"; short="p", type=Int, default=nothing, description="Lag order"),
                Option("test"; short="t", type=String, default="all", description="strength|gaussianity|independence|all"),
                Option("method"; type=String, default="fastica", description="fastica|infomax|jade"),
                Option("contrast"; type=String, default="logcosh", description="logcosh|exp|kurtosis"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="Test identifiability conditions")

        subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
            "fastica" => ng_fastica,
            "ml" => ng_ml,
            "heteroskedasticity" => ng_heteroskedasticity,
            "normality" => ng_normality,
            "identifiability" => ng_identifiability,
        )
        node = NodeCommand("nongaussian", subcmds,
            "Non-Gaussian SVAR Identification")

        # Verify structure
        @test node.name == "nongaussian"
        @test length(node.subcmds) == 5
        @test haskey(node.subcmds, "fastica")
        @test haskey(node.subcmds, "ml")
        @test haskey(node.subcmds, "heteroskedasticity")
        @test haskey(node.subcmds, "normality")
        @test haskey(node.subcmds, "identifiability")

        # Verify leaf details
        @test node.subcmds["fastica"].name == "fastica"
        @test length(node.subcmds["fastica"].args) == 1
        @test length(node.subcmds["fastica"].options) == 5

        @test node.subcmds["ml"].name == "ml"
        @test length(node.subcmds["ml"].options) == 4

        @test node.subcmds["heteroskedasticity"].name == "heteroskedasticity"
        @test length(node.subcmds["heteroskedasticity"].options) == 6

        @test node.subcmds["normality"].name == "normality"
        @test length(node.subcmds["normality"].options) == 3

        @test node.subcmds["identifiability"].name == "identifiability"
        @test length(node.subcmds["identifiability"].options) == 6

        # Help text
        buf = IOBuffer()
        print_help(buf, node; prog="friedman nongaussian")
        help_text = String(take!(buf))
        @test contains(help_text, "fastica")
        @test contains(help_text, "ml")
        @test contains(help_text, "heteroskedasticity")
        @test contains(help_text, "normality")
        @test contains(help_text, "identifiability")

        # Leaf help text
        buf = IOBuffer()
        print_help(buf, ng_fastica; prog="friedman nongaussian fastica")
        help_text = String(take!(buf))
        @test contains(help_text, "--method")
        @test contains(help_text, "--contrast")
        @test contains(help_text, "logcosh")

        buf = IOBuffer()
        print_help(buf, ng_ml; prog="friedman nongaussian ml")
        help_text = String(take!(buf))
        @test contains(help_text, "--distribution")
        @test contains(help_text, "student_t")

        # Dispatch to handler via node
        called_with = Ref{Any}(nothing)
        dispatch_handler = (; kwargs...) -> begin called_with[] = Dict(kwargs) end
        test_leaf = LeafCommand("fastica", dispatch_handler;
            args=[Argument("data"; description="Data file")],
            options=[Option("method"; type=String, default="fastica", description="Method")],
            description="Test fastica")
        dispatch_leaf(test_leaf, ["test.csv", "--method=jade"]; prog="friedman nongaussian fastica")
        @test called_with[][:data] == "test.csv"
        @test called_with[][:method] == "jade"
    end

    @testset "Factor forecast command structure" begin
        handler = (; kwargs...) -> kwargs

        factor_forecast = LeafCommand("forecast", handler;
            args=[Argument("data"; description="Path to CSV data file")],
            options=[
                Option("nfactors"; short="r", type=Int, default=nothing, description="Number of factors"),
                Option("horizon"; short="h", type=Int, default=12, description="Forecast horizon"),
                Option("ci-method"; type=String, default="none", description="none|bootstrap|parametric"),
                Option("conf-level"; type=Float64, default=0.95, description="Confidence level"),
                Option("output"; short="o", type=String, default="", description="Export results to file"),
                Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            ],
            description="Forecast observables using static factor model")

        # Verify structure
        @test factor_forecast.name == "forecast"
        @test length(factor_forecast.args) == 1
        @test length(factor_forecast.options) == 6

        # Include in a factor node with 4 subcommands
        factor_static = LeafCommand("static", handler; description="Static factors")
        factor_dynamic = LeafCommand("dynamic", handler; description="Dynamic factors")
        factor_gdfm = LeafCommand("gdfm", handler; description="GDFM")

        subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
            "static" => factor_static, "dynamic" => factor_dynamic,
            "gdfm" => factor_gdfm, "forecast" => factor_forecast)
        factor_node = NodeCommand("factor", subcmds, "Factor Models")

        @test length(factor_node.subcmds) == 4
        @test haskey(factor_node.subcmds, "forecast")

        # Help text
        buf = IOBuffer()
        print_help(buf, factor_node; prog="friedman factor")
        help_text = String(take!(buf))
        @test contains(help_text, "forecast")
        @test contains(help_text, "static")

        # Leaf help
        buf = IOBuffer()
        print_help(buf, factor_forecast; prog="friedman factor forecast")
        help_text = String(take!(buf))
        @test contains(help_text, "--horizon")
        @test contains(help_text, "--ci-method")
        @test contains(help_text, "--conf-level")
        @test contains(help_text, "default: 12")
        @test contains(help_text, "default: 0.95")

        # Arg binding
        parsed = tokenize(["mydata.csv", "--horizon=24", "--ci-method=bootstrap"])
        bound = bind_args(parsed, factor_forecast)
        @test bound.data == "mydata.csv"
        @test bound.horizon == 24
        @test bound.ci_method == "bootstrap"
        @test bound.conf_level == 0.95
        @test isnothing(bound.nfactors)
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
end
