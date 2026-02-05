using Test

# Test the CLI engine independently (no MacroEconometricModels needed)
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
end
