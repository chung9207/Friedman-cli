using Test
using CSV, DataFrames

# Set up minimal Friedman context for testing repl.jl
module Friedman
    using CSV, DataFrames
    # Minimal stubs matching io.jl functions
    function load_data(path::String)
        isfile(path) || error("file not found: $path")
        df = CSV.read(path, DataFrame)
        nrow(df) == 0 && error("empty dataset: $path")
        return df
    end
    function df_to_matrix(df::DataFrame)
        numeric_cols = [n for n in names(df) if eltype(df[!, n]) <: Union{Number, Missing}]
        isempty(numeric_cols) && error("no numeric columns found")
        return Matrix{Float64}(df[!, numeric_cols])
    end
    variable_names(df::DataFrame) = [n for n in names(df) if eltype(df[!, n]) <: Union{Number, Missing}]

    function load_example(name::Symbol)
        if name == :fred_md
            data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
            return (data=data, varnames=["INDPRO", "CPI", "FEDFUNDS"])
        elseif name == :fred_qd
            data = [1.0 2.0; 3.0 4.0]
            return (data=data, varnames=["GDP", "PCE"])
        elseif name == :pwt
            data = ones(5, 3)
            return (data=data, varnames=["rgdpe", "pop", "emp"])
        elseif name == :mpdta
            data = ones(4, 2)
            return (data=data, varnames=["lemp", "lpop"])
        elseif name == :ddcg
            data = ones(3, 2)
            return (data=data, varnames=["y", "d"])
        else
            error("unknown dataset: $name")
        end
    end

    # Minimal stubs for REPL loop compilation
    struct ParseError <: Exception; message::String; end
    struct DispatchError <: Exception; message::String; end
    struct Argument; end
    struct Option; end
    struct Flag; end
    struct LeafCommand; end
    struct NodeCommand; end
    struct Entry; name::String; root::Any; version::VersionNumber; end
    const FRIEDMAN_VERSION = v"0.3.4"
    function dispatch(app, args)
        # Return a mock result for estimate commands
        if !isempty(args) && args[1] == "estimate" && length(args) >= 2
            return "mock_$(args[2])_model"
        end
        return nothing
    end
    function build_app() Entry("friedman", nothing, v"0.3.4") end

    include(joinpath(@__DIR__, "..", "src", "repl.jl"))
end

@testset "REPL Session" begin
    @testset "Session initialization" begin
        s = Friedman.Session()
        @test s.data_path == ""
        @test isnothing(s.df)
        @test isnothing(s.Y)
        @test isempty(s.varnames)
        @test isempty(s.results)
        @test s.last_model == :none
    end

    @testset "session_load_data!" begin
        s = Friedman.Session()
        tmpfile = tempname() * ".csv"
        open(tmpfile, "w") do io
            println(io, "x,y,z")
            println(io, "1.0,2.0,3.0")
            println(io, "4.0,5.0,6.0")
            println(io, "7.0,8.0,9.0")
        end
        Friedman.session_load_data!(s, tmpfile)
        @test s.data_path == tmpfile
        @test !isnothing(s.df)
        @test size(s.Y) == (3, 3)
        @test s.varnames == ["x", "y", "z"]
        @test isempty(s.results)
        rm(tmpfile; force=true)
    end

    @testset "session_load_data! clears results" begin
        s = Friedman.Session()
        s.results[:var] = "fake_model"
        s.last_model = :var
        tmpfile = tempname() * ".csv"
        open(tmpfile, "w") do io
            println(io, "a,b")
            println(io, "1.0,2.0")
            println(io, "3.0,4.0")
        end
        Friedman.session_load_data!(s, tmpfile)
        @test isempty(s.results)
        @test s.last_model == :none
        rm(tmpfile; force=true)
    end

    @testset "session_clear!" begin
        s = Friedman.Session()
        s.data_path = "test.csv"
        s.results[:var] = "fake"
        s.last_model = :var
        Friedman.session_clear!(s)
        @test s.data_path == ""
        @test isnothing(s.df)
        @test isempty(s.results)
        @test s.last_model == :none
    end

    @testset "session_store_result!" begin
        s = Friedman.Session()
        Friedman.session_store_result!(s, :var, "var_model")
        @test s.results[:var] == "var_model"
        @test s.last_model == :var
        Friedman.session_store_result!(s, :bvar, "bvar_model")
        @test s.results[:bvar] == "bvar_model"
        @test s.last_model == :bvar
        @test s.results[:var] == "var_model"
        Friedman.session_store_result!(s, :var, "var_model_v2")
        @test s.results[:var] == "var_model_v2"
        @test s.last_model == :var
    end

    @testset "session_has_data" begin
        s = Friedman.Session()
        @test !Friedman.session_has_data(s)
        s.data_path = "test.csv"
        @test Friedman.session_has_data(s)
    end

    @testset "session_get_result" begin
        s = Friedman.Session()
        @test isnothing(Friedman.session_get_result(s, :var))
        Friedman.session_store_result!(s, :var, "model")
        @test Friedman.session_get_result(s, :var) == "model"
        @test isnothing(Friedman.session_get_result(s, :bvar))
    end

    @testset "parse_data_source" begin
        @test Friedman.parse_data_source(":fred-md") == (:builtin, :fred_md)
        @test Friedman.parse_data_source(":fred-qd") == (:builtin, :fred_qd)
        @test Friedman.parse_data_source(":pwt") == (:builtin, :pwt)
        @test Friedman.parse_data_source(":mpdta") == (:builtin, :mpdta)
        @test Friedman.parse_data_source(":ddcg") == (:builtin, :ddcg)
        @test Friedman.parse_data_source("myfile.csv") == (:file, "myfile.csv")
        @test_throws ErrorException Friedman.parse_data_source(":nonexistent")
    end

    @testset "session_load_builtin!" begin
        s = Friedman.Session()
        Friedman.session_load_builtin!(s, :fred_md)
        @test s.data_path == ":fred-md"
        @test !isnothing(s.df)
        @test !isnothing(s.Y)
        @test size(s.Y) == (3, 3)
        @test s.varnames == ["INDPRO", "CPI", "FEDFUNDS"]
        @test isempty(s.results)
        @test s.last_model == :none
    end
end

@testset "REPL dispatch wrapper" begin
    @testset "inject_session_data" begin
        s = Friedman.Session()
        s.data_path = "/tmp/test.csv"

        # Args with no data positional — inject before options
        args = ["estimate", "var", "--lags", "4"]
        result = Friedman.inject_session_data(s, args)
        @test result == ["estimate", "var", "/tmp/test.csv", "--lags", "4"]

        # Args already have data (a .csv path) — don't inject
        args2 = ["estimate", "var", "mydata.csv", "--lags", "4"]
        result2 = Friedman.inject_session_data(s, args2)
        @test result2 == args2

        # No session data — return unchanged
        s2 = Friedman.Session()
        result3 = Friedman.inject_session_data(s2, args)
        @test result3 == args

        # Deep nesting: dsge bayes estimate
        s3 = Friedman.Session()
        s3.data_path = "/tmp/test.csv"
        args4 = ["dsge", "bayes", "estimate", "--draws", "1000"]
        result4 = Friedman.inject_session_data(s3, args4)
        @test result4 == ["dsge", "bayes", "estimate", "/tmp/test.csv", "--draws", "1000"]
    end

    @testset "detect_model_type" begin
        @test Friedman.detect_model_type(["estimate", "var"]) == :var
        @test Friedman.detect_model_type(["estimate", "bvar"]) == :bvar
        @test Friedman.detect_model_type(["irf", "var"]) == :var
        @test Friedman.detect_model_type(["test", "adf"]) == :adf
        @test Friedman.detect_model_type(["data", "use"]) == :use
        @test Friedman.detect_model_type(["data"]) == :none
    end

    @testset "is_estimate_command" begin
        @test Friedman.is_estimate_command(["estimate", "var", "d.csv"])
        @test !Friedman.is_estimate_command(["irf", "var", "d.csv"])
        @test !Friedman.is_estimate_command(["data", "use", "d.csv"])
    end
end

@testset "REPL line splitting" begin
    @test Friedman._split_repl_line("estimate var data.csv --lags 4") == ["estimate", "var", "data.csv", "--lags", "4"]
    @test Friedman._split_repl_line("  estimate   var  ") == ["estimate", "var"]
    @test Friedman._split_repl_line("data use \"my file.csv\"") == ["data", "use", "my file.csv"]
    @test Friedman._split_repl_line("") == String[]
    @test Friedman._split_repl_line("   ") == String[]
end

@testset "repl_dispatch" begin
    @testset "data use (file)" begin
        s = Friedman.Session()
        tmpfile = tempname() * ".csv"
        open(tmpfile, "w") do io
            println(io, "x,y")
            println(io, "1.0,2.0")
            println(io, "3.0,4.0")
        end
        Friedman.session_load_data!(s, tmpfile)
        @test s.data_path == tmpfile
        @test s.varnames == ["x", "y"]
        rm(tmpfile; force=true)
    end

    @testset "data current with no data" begin
        s = Friedman.Session()
        @test !Friedman.session_has_data(s)
    end

    @testset "data clear" begin
        s = Friedman.Session()
        s.data_path = "test.csv"
        s.results[:var] = "model"
        Friedman.session_clear!(s)
        @test !Friedman.session_has_data(s)
        @test isempty(s.results)
    end

    @testset "exit/quit throws InterruptException" begin
        app = Friedman.build_app()
        s = Friedman.Session()
        @test_throws InterruptException Friedman.repl_dispatch(s, app, ["exit"])
        @test_throws InterruptException Friedman.repl_dispatch(s, app, ["quit"])
    end

    @testset "empty args is no-op" begin
        app = Friedman.build_app()
        s = Friedman.Session()
        # Should return without error
        Friedman.repl_dispatch(s, app, String[])
    end

    @testset "data use loads file via repl_dispatch" begin
        s = Friedman.Session()
        app = Friedman.build_app()
        tmpfile = tempname() * ".csv"
        open(tmpfile, "w") do io
            println(io, "a,b")
            println(io, "1.0,2.0")
            println(io, "3.0,4.0")
        end
        Friedman.repl_dispatch(s, app, ["data", "use", tmpfile])
        @test s.data_path == tmpfile
        @test s.varnames == ["a", "b"]
        @test size(s.Y) == (2, 2)
        rm(tmpfile; force=true)
    end

    @testset "data use loads builtin via repl_dispatch" begin
        s = Friedman.Session()
        app = Friedman.build_app()
        Friedman.repl_dispatch(s, app, ["data", "use", ":fred-md"])
        @test s.data_path == ":fred-md"
        @test s.varnames == ["INDPRO", "CPI", "FEDFUNDS"]
    end

    @testset "data current output" begin
        s = Friedman.Session()
        app = Friedman.build_app()
        # No data — prints "No data loaded"
        output = let (tmppath, tmpio) = mktemp()
            try
                redirect_stdout(tmpio) do
                    Friedman.repl_dispatch(s, app, ["data", "current"])
                end
                close(tmpio)
                read(tmppath, String)
            finally
                try; close(tmpio); catch; end
                try; rm(tmppath; force=true); catch; end
            end
        end
        @test contains(output, "No data loaded")

        # With data
        Friedman.session_load_builtin!(s, :fred_md)
        output2 = let (tmppath, tmpio) = mktemp()
            try
                redirect_stdout(tmpio) do
                    Friedman.repl_dispatch(s, app, ["data", "current"])
                end
                close(tmpio)
                read(tmppath, String)
            finally
                try; close(tmpio); catch; end
                try; rm(tmppath; force=true); catch; end
            end
        end
        @test contains(output2, ":fred-md")
        @test contains(output2, "3×3")
    end

    @testset "data clear via repl_dispatch" begin
        s = Friedman.Session()
        app = Friedman.build_app()
        Friedman.session_load_builtin!(s, :fred_md)
        @test Friedman.session_has_data(s)
        Friedman.repl_dispatch(s, app, ["data", "clear"])
        @test !Friedman.session_has_data(s)
    end

    @testset "functions are defined" begin
        @test hasmethod(Friedman.repl_dispatch, Tuple{Friedman.Session, Friedman.Entry, Vector{String}})
        @test hasmethod(Friedman.start_repl, Tuple{})
        @test hasmethod(Friedman._split_repl_line, Tuple{String})
    end

    @testset "REPL result capture" begin
        s = Friedman.Session()
        s.data_path = "/tmp/test.csv"  # pretend data is loaded
        app = Friedman.build_app()

        # Simulate estimate var dispatch
        Friedman.repl_dispatch(s, app, ["estimate", "var", "data.csv"])
        @test Friedman.session_get_result(s, :var) == "mock_var_model"
        @test s.last_model == :var

        # Non-estimate command should not cache
        Friedman.repl_dispatch(s, app, ["irf", "var", "data.csv"])
        @test s.last_model == :var  # unchanged
    end
end
