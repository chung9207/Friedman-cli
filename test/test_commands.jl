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

# Comprehensive handler tests for the action-first command structure
# Uses mock MacroEconometricModels from test/mocks.jl to test all command handlers
# without requiring the actual MacroEconometricModels package.

using Test
using CSV, DataFrames, JSON3, PrettyTables, TOML
using Dates
using LinearAlgebra: eigvals, diag, I, svd, diagm
using Statistics: mean, median, var

# ─── Setup: Mock module + source includes ──────────────────────

project_root = dirname(@__DIR__)

# Load mock MacroEconometricModels module
include(joinpath(project_root, "test", "mocks.jl"))
using .MacroEconometricModels

# PrettyTables v3 compat
if !@isdefined(tf_unicode_rounded)
    const tf_unicode_rounded = text_table_borders__unicode_rounded
end

# Include io.jl and config.jl (needed by command handlers)
include(joinpath(project_root, "src", "io.jl"))
include(joinpath(project_root, "src", "config.jl"))

# Override _write_table for PrettyTables v3 (tf/show_subheader kwargs removed)
function _write_table(df::DataFrame, output::String, title::String)
    io = isempty(output) ? stdout : open(output, "w")
    try
        pretty_table(io, df; title=title, alignment=:c)
    finally
        isempty(output) || close(io)
    end
    isempty(output) || println("Results written to $output")
end

# Include CLI types (needed for LeafCommand, NodeCommand, etc.)
include(joinpath(project_root, "src", "cli", "types.jl"))

# Include command files in dependency order
include(joinpath(project_root, "src", "commands", "shared.jl"))
include(joinpath(project_root, "src", "commands", "estimate.jl"))
include(joinpath(project_root, "src", "commands", "test.jl"))
include(joinpath(project_root, "src", "commands", "irf.jl"))
include(joinpath(project_root, "src", "commands", "fevd.jl"))
include(joinpath(project_root, "src", "commands", "hd.jl"))
include(joinpath(project_root, "src", "commands", "forecast.jl"))
include(joinpath(project_root, "src", "commands", "predict.jl"))
include(joinpath(project_root, "src", "commands", "residuals.jl"))
include(joinpath(project_root, "src", "commands", "filter.jl"))
include(joinpath(project_root, "src", "commands", "data.jl"))

# ─── Test Helpers ───────────────────────────────────────────────

"""Create a temp CSV file with synthetic multivariate data."""
function _make_csv(dir; T=100, n=3, colnames=nothing)
    cols = isnothing(colnames) ? ["var$i" for i in 1:n] : colnames
    data = Dict{String,Vector{Float64}}()
    for (i, name) in enumerate(cols)
        data[name] = randn(T) .+ Float64(i)
    end
    path = joinpath(dir, "data.csv")
    CSV.write(path, DataFrame(data))
    return path
end

"""Create a temp instruments CSV file."""
function _make_instruments_csv(dir; T=100, n_inst=2)
    data = Dict{String,Vector{Float64}}()
    for i in 1:n_inst
        data["z$i"] = randn(T)
    end
    path = joinpath(dir, "instruments.csv")
    CSV.write(path, DataFrame(data))
    return path
end

"""Capture stdout output from a function call, returning the string."""
function _capture(f)
    path, io = mktemp()
    try
        redirect_stdout(io) do
            f()
        end
        close(io)
        return read(path, String)
    finally
        try; close(io); catch; end
        try; rm(path; force=true); catch; end
    end
end

"""Create a temp CSV file with synthetic panel data (group + time columns)."""
function _make_panel_csv(dir; G=5, T_per=20, n=3, colnames=nothing)
    cols = isnothing(colnames) ? ["var$i" for i in 1:n] : colnames
    rows = G * T_per
    data = Dict{String,Vector}()
    data["group"] = repeat(1:G, inner=T_per)
    data["time"] = repeat(1:T_per, outer=G)
    for (i, name) in enumerate(cols)
        data[name] = randn(rows) .+ Float64(i)
    end
    path = joinpath(dir, "panel.csv")
    CSV.write(path, DataFrame(data))
    return path
end

"""Create a TOML config for prior settings."""
function _make_prior_config(dir; optimize=false)
    path = joinpath(dir, "prior.toml")
    open(path, "w") do io
        write(io, """
        [prior]
        type = "minnesota"
        [prior.hyperparameters]
        lambda1 = 0.2
        lambda2 = 0.5
        lambda3 = 1.0
        lambda4 = 100000.0
        [prior.optimization]
        enabled = $optimize
        """)
    end
    return path
end

"""Create a TOML config for sign identification."""
function _make_sign_config(dir)
    path = joinpath(dir, "sign.toml")
    open(path, "w") do io
        write(io, """
        [identification]
        method = "sign"
        [identification.sign_matrix]
        matrix = [[1, -1, 1], [0, 1, -1], [0, 0, 1]]
        horizons = [0, 1, 2]
        """)
    end
    return path
end

"""Create a TOML config for narrative identification."""
function _make_narrative_config(dir)
    path = joinpath(dir, "narrative.toml")
    open(path, "w") do io
        write(io, """
        [identification]
        method = "narrative"
        [identification.sign_matrix]
        matrix = [[1, -1, 1], [0, 1, -1], [0, 0, 1]]
        horizons = [0]
        [identification.narrative]
        shock_index = 1
        periods = [10, 15]
        signs = [1, -1]
        """)
    end
    return path
end

"""Create a TOML config for Arias identification."""
function _make_arias_config(dir)
    path = joinpath(dir, "arias.toml")
    open(path, "w") do io
        write(io, """
        [[identification.zero_restrictions]]
        var = 1
        shock = 1
        horizon = 0
        [[identification.sign_restrictions]]
        var = 2
        shock = 1
        sign = "positive"
        horizon = 0
        """)
    end
    return path
end

"""Create a TOML config for Uhlig identification."""
function _make_uhlig_config(dir)
    path = joinpath(dir, "uhlig.toml")
    open(path, "w") do io
        write(io, """
        [[identification.zero_restrictions]]
        var = 1
        shock = 1
        horizon = 0
        [[identification.sign_restrictions]]
        var = 2
        shock = 1
        sign = "positive"
        horizon = 0
        [identification.uhlig]
        n_starts = 50
        n_refine = 10
        """)
    end
    return path
end

"""Create a TOML config for GMM."""
function _make_gmm_config(dir; colnames=["var1","var2","var3"])
    path = joinpath(dir, "gmm.toml")
    open(path, "w") do io
        write(io, """
        [gmm]
        moment_conditions = ["$(colnames[1])", "$(colnames[2])"]
        instruments = ["lag_$(colnames[1])", "lag_$(colnames[2])"]
        weighting = "twostep"
        """)
    end
    return path
end

"""Create a TOML config for nongaussian smooth_transition."""
function _make_ng_smooth_config(dir; transition_var="var2")
    path = joinpath(dir, "ng_smooth.toml")
    open(path, "w") do io
        write(io, """
        [nongaussian]
        method = "smooth_transition"
        transition_variable = "$transition_var"
        """)
    end
    return path
end

"""Create a TOML config for nongaussian external volatility."""
function _make_ng_external_config(dir; regime_var="var3")
    path = joinpath(dir, "ng_external.toml")
    open(path, "w") do io
        write(io, """
        [nongaussian]
        method = "external"
        regime_variable = "$regime_var"
        """)
    end
    return path
end

# ─── Tests ─────────────────────────────────────────────────────

@testset "Command Handlers" begin

# ═══════════════════════════════════════════════════════════════
# Shared utilities (shared.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Shared utilities" begin

    @testset "ID_METHOD_MAP" begin
        @test length(ID_METHOD_MAP) == 16
        @test ID_METHOD_MAP["cholesky"] == :cholesky
        @test ID_METHOD_MAP["sign"] == :sign
        @test ID_METHOD_MAP["narrative"] == :narrative
        @test ID_METHOD_MAP["longrun"] == :long_run
        @test ID_METHOD_MAP["fastica"] == :fastica
        @test ID_METHOD_MAP["jade"] == :jade
        @test ID_METHOD_MAP["sobi"] == :sobi
        @test ID_METHOD_MAP["dcov"] == :dcov
        @test ID_METHOD_MAP["hsic"] == :hsic
        @test ID_METHOD_MAP["student_t"] == :student_t
        @test ID_METHOD_MAP["mixture_normal"] == :mixture_normal
        @test ID_METHOD_MAP["pml"] == :pml
        @test ID_METHOD_MAP["skew_normal"] == :skew_normal
        @test ID_METHOD_MAP["markov_switching"] == :markov_switching
        @test ID_METHOD_MAP["garch_id"] == :garch
        @test ID_METHOD_MAP["uhlig"] == :uhlig
    end

    @testset "_load_and_estimate_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)

            # Auto lag selection (lags=nothing)
            out = _capture() do
                model, Y, varnames, p = _load_and_estimate_var(csv, nothing)
                @test model isa VARModel
                @test size(Y) == (100, 3)
                @test length(varnames) == 3
                @test p isa Int
                @test p >= 1
            end

            # Explicit lag
            out = _capture() do
                model, Y, varnames, p = _load_and_estimate_var(csv, 3)
                @test p == 3
            end
        end
    end

    @testset "_load_and_estimate_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_prior_config(dir; optimize=false)

            out = _capture() do
                post, Y, varnames, p, n = _load_and_estimate_bvar(csv, 2, cfg, 500, "direct")
                @test post isa BVARPosterior
                @test size(Y) == (100, 3)
                @test n == 3
                @test p == 2
            end

            # With empty config (no prior)
            out = _capture() do
                post, Y, varnames, p, n = _load_and_estimate_bvar(csv, 2, "", 500, "hmc")
                @test post isa BVARPosterior
            end
        end
    end

    @testset "_build_prior" begin
        mktempdir() do dir
            Y = ones(100, 3) .+ randn(100, 3) * 0.1

            # Empty config -> nothing
            @test isnothing(_build_prior("", Y, 2))

            # Minnesota with optimization
            cfg_opt = _make_prior_config(dir; optimize=true)
            out = _capture() do
                prior = _build_prior(cfg_opt, Y, 2)
                @test prior isa MinnesotaHyperparameters
            end

            # Minnesota without optimization
            cfg_no = _make_prior_config(dir; optimize=false)
            prior = _build_prior(cfg_no, Y, 2)
            @test prior isa MinnesotaHyperparameters
            @test prior.tau == 0.2
            @test prior.lambda == 0.5
            @test prior.decay == 1.0
        end
    end

    @testset "_build_check_func" begin
        mktempdir() do dir
            # Empty config -> (nothing, nothing)
            cf, nc = _build_check_func("")
            @test isnothing(cf)
            @test isnothing(nc)

            # Sign restrictions
            sign_cfg = _make_sign_config(dir)
            cf, nc = _build_check_func(sign_cfg)
            @test cf isa Function
            @test isnothing(nc)
            mock_irf = ones(3, 3, 3) * 0.1
            @test cf(mock_irf) isa Bool

            # Narrative restrictions
            narr_cfg = _make_narrative_config(dir)
            cf2, nc2 = _build_check_func(narr_cfg)
            @test nc2 isa Function
            mock_shocks = ones(20, 3) * 0.1
            @test nc2(mock_shocks) isa Bool
        end
    end

    @testset "_build_identification_kwargs" begin
        # Cholesky (default, no config)
        kwargs = _build_identification_kwargs("cholesky", "")
        @test kwargs[:method] == :cholesky
        @test !haskey(kwargs, :check_func)
        @test !haskey(kwargs, :narrative_check)

        # Unknown method -> falls back to :cholesky
        kwargs2 = _build_identification_kwargs("unknown", "")
        @test kwargs2[:method] == :cholesky

        # Sign with config
        mktempdir() do dir
            sign_cfg = _make_sign_config(dir)
            kwargs3 = _build_identification_kwargs("sign", sign_cfg)
            @test kwargs3[:method] == :sign
            @test haskey(kwargs3, :check_func)
        end
    end

    @testset "_load_and_structural_lp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)

            out = _capture() do
                slp, Y, varnames = _load_and_structural_lp(csv, 20, 4, nothing, "cholesky", "newey_west", "")
                @test slp isa StructuralLP
                @test size(Y) == (100, 3)
                @test length(varnames) == 3
            end

            # With explicit var_lags
            out = _capture() do
                slp, Y, varnames = _load_and_structural_lp(csv, 20, 4, 6, "cholesky", "newey_west", "")
                @test slp isa StructuralLP
            end

            # With CI
            out = _capture() do
                slp, Y, varnames = _load_and_structural_lp(csv, 20, 4, nothing, "cholesky", "newey_west", "";
                    ci_type=:bootstrap, reps=100)
                @test slp isa StructuralLP
            end
        end
    end

    @testset "_var_forecast_point" begin
        n = 3; p = 2; T = 50; horizons = 5
        Y = randn(T, n)
        k = n * p + 1
        B = zeros(k, n)
        for i in 1:n
            B[i, i] = 0.3
        end
        B[end, :] .= 0.01

        fc = _var_forecast_point(B, Y, p, horizons)
        @test size(fc) == (horizons, n)
        @test all(isfinite, fc)

        # Without constant
        B_nc = zeros(n * p, n)
        for i in 1:n
            B_nc[i, i] = 0.3
        end
        fc2 = _var_forecast_point(B_nc, Y, p, horizons)
        @test size(fc2) == (horizons, n)

        # Single lag
        B_1 = zeros(n + 1, n)
        B_1[1, 1] = 0.5
        B_1[end, :] .= 0.01
        fc3 = _var_forecast_point(B_1, Y, 1, horizons)
        @test size(fc3) == (horizons, n)
    end

    @testset "quantile_normal" begin
        # Symmetry: q(p) == -q(1-p)
        @test quantile_normal(0.975) ≈ -quantile_normal(0.025) atol=1e-4
        # Known approximate values
        @test quantile_normal(0.5) ≈ 0.0 atol=0.01
        @test quantile_normal(0.975) ≈ 1.96 atol=0.02
        @test quantile_normal(0.995) ≈ 2.576 atol=0.02
        # Edge: p < 0.5 triggers recursive branch
        @test quantile_normal(0.025) < 0
    end

    @testset "load_univariate_series — valid column" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            y, vname = load_univariate_series(csv, 1)
            @test length(y) == 50
            @test vname isa String
        end
    end

    @testset "load_univariate_series — out of range" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            @test_throws ErrorException load_univariate_series(csv, 10)
        end
    end

    @testset "load_multivariate_data" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            Y, varnames = load_multivariate_data(csv)
            @test size(Y) == (50, 3)
            @test length(varnames) == 3
        end
    end

    @testset "_shock_name / _var_name" begin
        varnames = ["gdp", "inf", "rate"]
        @test _shock_name(varnames, 1) == "gdp"
        @test _shock_name(varnames, 3) == "rate"
        @test _shock_name(varnames, 5) == "shock_5"
        @test _var_name(varnames, 2) == "inf"
        @test _var_name(varnames, 10) == "var_10"
    end

    @testset "_per_var_output_path" begin
        @test _per_var_output_path("", "gdp") == ""
        @test _per_var_output_path("results.csv", "gdp") == "results_gdp.csv"
        @test _per_var_output_path("out.json", "inf") == "out_inf.json"
    end

    @testset "validate_method" begin
        @test validate_method("cholesky", ["cholesky", "sign"], "id") == "cholesky"
        @test_throws ErrorException validate_method("unknown", ["cholesky", "sign"], "id")
    end

    @testset "interpret_test_result" begin
        # Significant result
        out = _capture() do
            interpret_test_result(0.01, "Reject H0", "Fail to reject H0")
        end
        @test contains(out, "Reject H0")

        # Non-significant result
        out = _capture() do
            interpret_test_result(0.10, "Reject H0", "Fail to reject H0")
        end
        @test contains(out, "Fail to reject H0")
    end

    @testset "to_regression_symbol" begin
        @test to_regression_symbol("constant") == :constant
        @test to_regression_symbol("none") == :none
        @test to_regression_symbol("both") == :both
        @test to_regression_symbol("trend") == :trend
    end

    @testset "_build_var_coef_table" begin
        coef_mat = [0.5 0.1; 0.2 0.3; 0.01 0.02]
        varnames = ["y1", "y2"]
        df = _build_var_coef_table(coef_mat, varnames, 1)
        @test size(df, 1) == 2
        @test "equation" in names(df)
        @test "y1_L1" in names(df)
        @test "const" in names(df)
    end

    @testset "_vol_forecast_output" begin
        fc = (forecast = [1.0, 2.0, 3.0],)
        out = _capture() do
            _vol_forecast_output(fc, "ret", "GARCH(1,1)", 3; format="table", output="")
        end
        @test contains(out, "GARCH(1,1)")
    end

end  # Shared utilities

# ═══════════════════════════════════════════════════════════════
# Estimate handlers (estimate.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Estimate handlers" begin

    @testset "register_estimate_commands!" begin
        node = register_estimate_commands!()
        @test node isa NodeCommand
        @test node.name == "estimate"
        @test length(node.subcmds) == 17
        for cmd in ["var", "bvar", "lp", "arima", "gmm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "fastica", "ml", "vecm", "pvar"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_estimate_var — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=nothing, format="table")
                end
            end
            @test occursin("Estimating VAR(", out)
            @test occursin("Coefficients", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_estimate_var — explicit lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=3, format="table")
                end
            end
            @test occursin("VAR(3)", out)
        end
    end

    @testset "_estimate_var — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=2, format="json")
                end
            end
            @test occursin("AIC", out)
        end
    end

    @testset "_estimate_var — csv output to file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "coefs.csv")
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=2, output=outfile, format="csv")
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test nrow(result_df) > 0
        end
    end

    @testset "_estimate_bvar — mean" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_bvar(; data=csv, lags=2, prior="minnesota", draws=100,
                                    sampler="direct", method="mean", config="", format="table")
                end
            end
            @test occursin("Bayesian VAR(2)", out) || occursin("BVAR(2)", out)
            @test occursin("Mean", out) || occursin("mean", out)
        end
    end

    @testset "_estimate_bvar — median" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_bvar(; data=csv, lags=2, prior="minnesota", draws=100,
                                    sampler="direct", method="median", config="", format="table")
                end
            end
            @test occursin("Median", out) || occursin("median", out)
        end
    end

    @testset "_estimate_bvar — with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_prior_config(dir; optimize=false)
            out = cd(dir) do
                _capture() do
                    _estimate_bvar(; data=csv, lags=2, prior="minnesota", draws=100,
                                    sampler="direct", method="mean", config=cfg, format="table")
                end
            end
            @test occursin("BVAR(2)", out)
        end
    end

    @testset "_estimate_lp — standard" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="standard", shock=1, horizons=10,
                                  control_lags=4, vcov="newey_west", format="table")
                end
            end
            @test occursin("Local Projections", out)
            @test occursin("LP Coefficients", out)
            @test occursin("Estimation Summary", out)

        end
    end

    @testset "_estimate_lp — iv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            iv_csv = _make_instruments_csv(dir; T=100, n_inst=2)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="iv", shock=1, horizons=10,
                                  control_lags=4, vcov="newey_west", instruments=iv_csv,
                                  format="table")
                end
            end
            @test occursin("LP-IV", out)
            @test occursin("F-statistic", out)
            @test occursin("LP-IV Coefficients", out)

        end
    end

    @testset "_estimate_lp — iv missing instruments error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="iv", shock=1, horizons=10,
                                  control_lags=4, vcov="newey_west", instruments="",
                                  format="table")
                end
            end
        end
    end

    @testset "_estimate_lp — smooth with auto lambda" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="smooth", shock=1, horizons=10,
                                  knots=3, lambda=0.0, format="table")
                end
            end
            @test occursin("Smooth LP", out)
            @test occursin("Cross-validating", out)
            @test occursin("Smooth LP Coefficients", out)

        end
    end

    @testset "_estimate_lp — smooth with explicit lambda" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="smooth", shock=1, horizons=10,
                                  knots=3, lambda=0.5, format="table")
                end
            end
            @test occursin("Smooth LP", out)
            @test !occursin("Cross-validating", out)
            @test occursin("Smooth LP Coefficients", out)

        end
    end

    @testset "_estimate_lp — state" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="state", shock=1, horizons=10,
                                  state_var=2, gamma=1.5, transition="logistic", format="table")
                end
            end
            @test occursin("State-Dependent LP", out)
            @test occursin("Expansion", out)
            @test occursin("Recession", out)
            @test occursin("State LP Coefficients", out)
            @test occursin("Regime Difference Test", out)

        end
    end

    @testset "_estimate_lp — state missing state_var error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="state", shock=1, horizons=10,
                                  state_var=nothing, gamma=1.5, format="table")
                end
            end
        end
    end

    @testset "_estimate_lp — propensity" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="propensity", treatment=1, horizons=10,
                                  score_method="logit", format="table")
                end
            end
            @test occursin("Propensity Score LP", out)
            @test occursin("Diagnostics", out)
            @test occursin("ATE Estimates", out)

        end
    end

    @testset "_estimate_lp — robust" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="robust", treatment=1, horizons=10,
                                  score_method="logit", format="table")
                end
            end
            @test occursin("Doubly Robust LP", out)
            @test occursin("Diagnostics", out)
            @test occursin("ATE Estimates", out)

        end
    end

    @testset "_estimate_lp — unknown method error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _estimate_lp(; data=csv, method="invalid", format="table")
                end
            end
        end
    end

    @testset "_estimate_arima — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arima(; data=csv, column=1, p=nothing, d=0, q=0,
                                     max_p=3, max_d=1, max_q=3, criterion="bic",
                                     method="css_mle", format="table")
                end
            end
            @test occursin("Auto ARIMA", out)
            @test occursin("Selected model", out)
            @test occursin("Coefficients", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_estimate_arima — explicit AR" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arima(; data=csv, column=1, p=2, d=0, q=0,
                                     method="ols", format="table")
                end
            end
            @test occursin("AR(2)", out)
        end
    end

    @testset "_estimate_arima — explicit ARIMA" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arima(; data=csv, column=1, p=1, d=1, q=1,
                                     method="css_mle", format="table")
                end
            end
            @test occursin("ARIMA(1,1,1)", out)
        end
    end

    @testset "_estimate_arima — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arima(; data=csv, column=1, p=2, d=0, q=0,
                                     method="ols", format="json")
                end
            end
            @test occursin("AIC", out)
        end
    end

    @testset "_estimate_gmm — missing config error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _estimate_gmm(; data=csv, config="", weighting="twostep", format="table")
                end
            end
        end
    end

    @testset "_estimate_gmm — with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            out = cd(dir) do
                _capture() do
                    _estimate_gmm(; data=csv, config=cfg, weighting="twostep", format="table")
                end
            end
            @test occursin("GMM", out) || occursin("Estimating GMM", out)
            @test occursin("J-test", out) || occursin("Hansen", out)
        end
    end

    @testset "_estimate_gmm — different weightings" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            for w in ["identity", "optimal", "twostep", "iterated"]
                out = cd(dir) do
                    _capture() do
                        _estimate_gmm(; data=csv, config=cfg, weighting=w, format="table")
                    end
                end
                @test occursin("GMM", out) || occursin("Estimating GMM", out)
            end
        end
    end

    @testset "_estimate_static — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_static(; data=csv, nfactors=nothing, criterion="ic1", format="table")
                end
            end
            @test occursin("static factor model", out)
            @test occursin("Scree Data", out)
            @test occursin("Factor Loadings", out)
        end
    end

    @testset "_estimate_static — explicit factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_static(; data=csv, nfactors=3, format="table")
                end
            end
            @test occursin("3 factors", out)
        end
    end

    @testset "_estimate_dynamic — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_dynamic(; data=csv, nfactors=nothing, factor_lags=1,
                                       method="twostep", format="table")
                end
            end
            @test occursin("dynamic factor model", out)
            @test occursin("stationary", out) || occursin("Stationary", out) || occursin("not stationary", out)
        end
    end

    @testset "_estimate_dynamic — explicit factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_dynamic(; data=csv, nfactors=2, factor_lags=2,
                                       method="twostep", format="table")
                end
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_estimate_gdfm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_gdfm(; data=csv, nfactors=nothing, dynamic_rank=nothing, format="table")
                end
            end
            @test occursin("GDFM", out)
            @test occursin("Common Variance Shares", out)
            @test occursin("Average common variance share", out)
        end
    end

    @testset "_estimate_gdfm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _estimate_gdfm(; data=csv, nfactors=3, dynamic_rank=2, format="table")
                end
            end
            @test occursin("static rank=3", out)
            @test occursin("dynamic rank=2", out)
        end
    end

    @testset "_estimate_arch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arch(; data=csv, column=1, q=1, format="table")
                end
            end
            @test occursin("ARCH(1)", out)
            @test occursin("Persistence", out)
            @test occursin("Unconditional variance", out)
        end
    end

    @testset "_estimate_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GARCH(1,1)", out)
            @test occursin("Persistence", out)
            @test occursin("Half-life", out)
            @test occursin("Unconditional variance", out)
        end
    end

    @testset "_estimate_egarch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_egarch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("EGARCH(1,1)", out)
            @test occursin("Persistence", out)
        end
    end

    @testset "_estimate_gjr_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_gjr_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GJR-GARCH(1,1)", out)
            @test occursin("Persistence", out)
            @test occursin("Half-life", out)
        end
    end

    @testset "_estimate_sv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_sv(; data=csv, column=1, draws=100, format="table")
                end
            end
            @test occursin("Stochastic Volatility", out)
            @test occursin("Persistence", out)
        end
    end

    @testset "_estimate_fastica — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_fastica(; data=csv, lags=2, method="fastica",
                                        contrast="logcosh", format="table")
                end
            end
            @test occursin("Non-Gaussian SVAR", out)
            @test occursin("method=fastica", out)
            @test occursin("Structural Impact Matrix", out)
            @test occursin("Structural Shocks", out)
            @test occursin("Converged", out) || occursin("converged", out)
        end
    end

    @testset "_estimate_fastica — all 5 ICA methods" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for method in ["fastica", "jade", "sobi", "dcov", "hsic"]
                out = cd(dir) do
                    _capture() do
                        _estimate_fastica(; data=csv, lags=2, method=method, format="table")
                    end
                end
                @test occursin("method=$method", out)
                @test occursin("Structural Impact Matrix", out)
            end
        end
    end

    @testset "_estimate_fastica — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_fastica(; data=csv, lags=nothing, method="fastica", format="table")
                end
            end
            @test occursin("Non-Gaussian SVAR", out)
        end
    end

    @testset "_estimate_ml — student_t" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_ml(; data=csv, lags=2, distribution="student_t", format="table")
                end
            end
            @test occursin("Non-Gaussian ML SVAR", out)
            @test occursin("distribution=student_t", out)
            @test occursin("Structural Impact Matrix", out)
            @test occursin("Log-likelihood", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_estimate_ml — mixture_normal" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_ml(; data=csv, lags=2, distribution="mixture_normal", format="table")
                end
            end
            @test occursin("mixture_normal", out)
        end
    end

    @testset "_estimate_ml — pml" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_ml(; data=csv, lags=2, distribution="pml", format="table")
                end
            end
            @test occursin("pml", out)
        end
    end

    @testset "_estimate_ml — skew_normal" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_ml(; data=csv, lags=2, distribution="skew_normal", format="table")
                end
            end
            @test occursin("skew_normal", out)
        end
    end

    @testset "_estimate_ml — dist_params and se output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_ml(; data=csv, lags=2, distribution="student_t", format="table")
                end
            end
            # Mock has non-empty dist_params and se
            @test occursin("Distribution parameters", out)
            @test occursin("Parameter Estimates with Standard Errors", out) ||
                  occursin("std_error", out)
        end
    end

    @testset "_model_label" begin
        @test _model_label(2, 0, 0) == "AR(2)"
        @test _model_label(0, 0, 3) == "MA(3)"
        @test _model_label(2, 0, 1) == "ARMA(2,1)"
        @test _model_label(1, 1, 1) == "ARIMA(1,1,1)"
        @test _model_label(3, 2, 0) == "ARIMA(3,2,0)"
    end

    @testset "_estimate_arima_model — AR" begin
        y = randn(100)
        model = _estimate_arima_model(y, 2, 0, 0; method=:ols)
        @test model isa ARModel
        @test ar_order(model) == 2
        @test ma_order(model) == 0
        @test diff_order(model) == 0
    end

    @testset "_estimate_arima_model — MA" begin
        y = randn(100)
        model = _estimate_arima_model(y, 0, 0, 2; method=:css_mle)
        @test model isa MAModel
        @test ma_order(model) == 2
    end

    @testset "_estimate_arima_model — ARMA" begin
        y = randn(100)
        model = _estimate_arima_model(y, 2, 0, 1; method=:css_mle)
        @test model isa ARMAModel
        @test ar_order(model) == 2
        @test ma_order(model) == 1
    end

    @testset "_estimate_arima_model — ARIMA" begin
        y = randn(100)
        model = _estimate_arima_model(y, 1, 1, 1; method=:css_mle)
        @test model isa ARIMAModel
        @test diff_order(model) == 1
    end

    @testset "_estimate_arima_model — AR method normalization" begin
        y = randn(100)
        # css_mle is not valid for AR, should normalize to :mle
        model = _estimate_arima_model(y, 2, 0, 0; method=:css_mle)
        @test model isa ARModel
    end

    @testset "_arima_coef_table" begin
        model = estimate_arma(randn(100), 2, 1)
        out = _capture() do
            _arima_coef_table(model; format="table", title="Test Coefs")
        end
        @test occursin("Test Coefs", out)
        @test occursin("ar1", out) || occursin("ar", out)
    end

    @testset "_arima_coef_table — includes SE and z-stat" begin
        model = estimate_ar(randn(100), 2)
        out = _capture() do
            _arima_coef_table(model; format="table", title="AR Coefs")
        end
        @test occursin("z_stat", out) || occursin("std_error", out)
    end

    @testset "_estimate_arch — includes SE columns" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arch(; data=csv, column=1, q=1, format="table")
                end
            end
            @test occursin("std_error", out) || occursin("z_stat", out)
        end
    end

end  # Estimate handlers

# ═══════════════════════════════════════════════════════════════
# Test handlers (test.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Test handlers" begin

    @testset "register_test_commands!" begin
        node = register_test_commands!()
        @test node isa NodeCommand
        @test node.name == "test"
        @test length(node.subcmds) == 16
        for cmd in ["adf", "kpss", "pp", "za", "np", "johansen",
                     "normality", "identifiability", "heteroskedasticity",
                     "arch_lm", "ljung_box", "var", "granger", "pvar", "lr", "lm"]
            @test haskey(node.subcmds, cmd)
        end
        # VAR is a nested NodeCommand with lagselect and stability
        var_node = node.subcmds["var"]
        @test var_node isa NodeCommand
        @test haskey(var_node.subcmds, "lagselect")
        @test haskey(var_node.subcmds, "stability")
        # PVAR is a nested NodeCommand with 4 children
        pvar_node = node.subcmds["pvar"]
        @test pvar_node isa NodeCommand
        @test length(pvar_node.subcmds) == 4
        @test haskey(pvar_node.subcmds, "hansen_j")
        @test haskey(pvar_node.subcmds, "mmsc")
        @test haskey(pvar_node.subcmds, "lagselect")
        @test haskey(pvar_node.subcmds, "stability")
        # LR and LM are LeafCommands with 2 positional args
        @test node.subcmds["lr"] isa LeafCommand
        @test node.subcmds["lm"] isa LeafCommand
        @test length(node.subcmds["lr"].args) == 2
        @test length(node.subcmds["lm"].args) == 2
    end

    @testset "_test_adf — reject" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_adf(; data=csv, column=1, max_lags=nothing, trend="constant", format="table")
            end
            @test occursin("ADF Test", out)
            @test occursin("Reject", out) || occursin("stationary", out)
        end
    end

    @testset "_test_adf — explicit lags and different trends" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for trend in ["none", "constant", "trend", "both"]
                out = _capture() do
                    _test_adf(; data=csv, column=1, max_lags=4, trend=trend, format="table")
                end
                @test occursin("ADF Test", out)
            end
        end
    end

    @testset "_test_kpss" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_kpss(; data=csv, column=1, trend="constant", format="table")
            end
            @test occursin("KPSS Test", out)
            @test occursin("stationary", out) || occursin("Cannot reject", out)
        end
    end

    @testset "_test_pp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_pp(; data=csv, column=1, trend="constant", format="table")
            end
            @test occursin("Phillips-Perron", out)
            @test occursin("Reject", out) || occursin("stationary", out)
        end
    end

    @testset "_test_pp — none trend" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_pp(; data=csv, column=1, trend="none", format="table")
            end
            @test occursin("Phillips-Perron", out)
        end
    end

    @testset "_test_za" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_za(; data=csv, column=1, trend="both", trim=0.15, format="table")
            end
            @test occursin("Zivot-Andrews", out)
            @test occursin("Break date", out) || occursin("structural break", out)
        end
    end

    @testset "_test_np" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_np(; data=csv, column=1, trend="constant", format="table")
            end
            @test occursin("Ng-Perron", out)
            @test occursin("MZa", out)
            @test occursin("MZt", out)
            @test occursin("MSB", out)
            @test occursin("MPT", out)
        end
    end

    @testset "_test_johansen" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_johansen(; data=csv, lags=2, trend="constant", format="table")
            end
            @test occursin("Johansen", out)
            @test occursin("Trace Test", out)
            @test occursin("Max Eigenvalue", out)
            @test occursin("cointegration rank", out)
        end
    end

    @testset "_test_johansen — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "johansen.csv")
            out = _capture() do
                _test_johansen(; data=csv, lags=2, trend="constant",
                                format="csv", output=outfile)
            end
            @test isfile(outfile)
        end
    end

    @testset "_test_johansen — none trend" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_johansen(; data=csv, lags=2, trend="none", format="table")
            end
            @test occursin("Johansen", out)
        end
    end

    @testset "_test_normality" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_normality(; data=csv, lags=2, format="table")
            end
            @test occursin("Normality Test", out)
            @test occursin("reject normality", out) || occursin("tests reject", out)
        end
    end

    @testset "_test_normality — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_normality(; data=csv, lags=nothing, format="table")
            end
            @test occursin("Normality Test", out)
        end
    end

    @testset "_test_identifiability — all tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_identifiability(; data=csv, lags=2, test="all",
                                        method="fastica", format="table")
            end
            @test occursin("Identifiability", out)
            @test occursin("Identification Strength", out) || occursin("significant", out)
        end
    end

    @testset "_test_identifiability — individual tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for test_type in ["strength", "gaussianity", "independence", "overidentification"]
                out = _capture() do
                    _test_identifiability(; data=csv, lags=2, test=test_type,
                                            method="fastica", format="table")
                end
                @test occursin("Identifiability", out)
            end
        end
    end

    @testset "_test_identifiability — all 5 ICA methods" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for ica_method in ["fastica", "jade", "sobi", "dcov", "hsic"]
                out = _capture() do
                    _test_identifiability(; data=csv, lags=2, test="gaussianity",
                                            method=ica_method, format="table")
                end
                @test occursin("Identifiability", out)
            end
        end
    end

    @testset "_test_identifiability — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_identifiability(; data=csv, lags=nothing, test="strength",
                                        format="table")
            end
            @test occursin("Identifiability", out)
        end
    end

    @testset "_test_heteroskedasticity — markov" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="markov",
                                            regimes=2, format="table")
            end
            @test occursin("Heteroskedasticity SVAR", out)
            @test occursin("method=markov", out)
            @test occursin("Structural Impact Matrix", out)
        end
    end

    @testset "_test_heteroskedasticity — garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="garch",
                                            format="table")
            end
            @test occursin("method=garch", out)
        end
    end

    @testset "_test_heteroskedasticity — smooth_transition requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="smooth_transition",
                                            config="", format="table")
            end
        end
    end

    @testset "_test_heteroskedasticity — smooth_transition with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_ng_smooth_config(dir; transition_var="var2")
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="smooth_transition",
                                            config=cfg, format="table")
            end
            @test occursin("smooth_transition", out)
        end
    end

    @testset "_test_heteroskedasticity — external requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="external",
                                            config="", format="table")
            end
        end
    end

    @testset "_test_heteroskedasticity — external with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_ng_external_config(dir; regime_var="var3")
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="external",
                                            config=cfg, regimes=2, format="table")
            end
            @test occursin("Structural Impact Matrix", out)
        end
    end

    @testset "_test_var_lagselect" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_lagselect(; data=csv, max_lags=4, criterion="aic", format="table")
            end
            @test occursin("Lag order selection", out) || occursin("Lag Order Selection", out)
            @test occursin("Optimal lag order", out) || occursin("optimal", out)
        end
    end

    @testset "_test_var_lagselect — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_lagselect(; data=csv, max_lags=4, criterion="bic", format="json")
            end
            @test occursin("optimal_lag", out)
        end
    end

    @testset "_test_var_stability" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_stability(; data=csv, lags=2, format="table")
            end
            @test occursin("Stationarity Check", out)
            @test occursin("stable", out) || occursin("Stable", out) || occursin("NOT stable", out)
            @test occursin("Max modulus", out)
        end
    end

    @testset "_test_var_stability — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_stability(; data=csv, lags=nothing)
            end
            @test occursin("Stationarity Check", out)
        end
    end

    @testset "_test_arch_lm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_arch_lm(; data=csv, column=1, lags=4, format="table")
            end
            @test occursin("ARCH-LM Test", out)
            @test occursin("LM statistic", out) || occursin("statistic", out)
            @test occursin("p-value", out)
        end
    end

    @testset "_test_ljung_box" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_ljung_box(; data=csv, column=1, lags=10, format="table")
            end
            @test occursin("Ljung-Box", out)
            @test occursin("Q statistic", out) || occursin("statistic", out)
            @test occursin("p-value", out)
        end
    end

end  # Test handlers

# ═══════════════════════════════════════════════════════════════
# IRF handlers (irf.jl)
# ═══════════════════════════════════════════════════════════════

@testset "IRF handlers" begin

    @testset "register_irf_commands!" begin
        node = register_irf_commands!()
        @test node isa NodeCommand
        @test node.name == "irf"
        @test length(node.subcmds) == 5
        for cmd in ["var", "bvar", "lp", "vecm", "pvar"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_irf_var — cholesky with bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                              ci="bootstrap", replications=100, format="table")
                end
            end
            @test occursin("Computing IRFs", out)
            @test occursin("cholesky", out)
            @test occursin("IRF to", out)
        end
    end

    @testset "_irf_var — cholesky with no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                              ci="none", format="table")
                end
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_irf_var — arias identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_arias_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="arias",
                              config=cfg, format="table")
                end
            end
            @test occursin("Arias", out)
        end
    end

    @testset "_irf_var — arias without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="arias",
                              config="", format="table")
                end
            end
        end
    end

    @testset "_irf_var — uhlig identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_uhlig_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="uhlig",
                              config=cfg, format="table")
                end
            end
            @test occursin("Uhlig", out)
            @test occursin("penalty", out)
        end
    end

    @testset "_irf_var — uhlig without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="uhlig",
                              config="", format="table")
                end
            end
        end
    end

    @testset "_irf_var — sign identification with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                              config=cfg, ci="none", format="table")
                end
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_irf_var — shock > n_vars uses generic name" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=3, horizons=10, id="cholesky",
                              ci="none", format="table")
                end
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_irf_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_bvar(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                               draws=100, sampler="direct", config="", format="table")
                end
            end
            @test occursin("Bayesian IRF", out)
            @test occursin("68% credible", out)
        end
    end

    @testset "_irf_bvar — shock 2" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_bvar(; data=csv, lags=2, shock=2, horizons=10, id="cholesky",
                               draws=100, sampler="direct", config="", format="table")
                end
            end
            @test occursin("Bayesian IRF", out)
        end
    end

    @testset "_irf_lp — single shock" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, horizons=10, lags=4, id="cholesky",
                             ci="none", vcov="newey_west", config="", format="table")
                end
            end
            @test occursin("LP IRF", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_irf_lp — multi-shock" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, shocks="1,2", horizons=10, lags=4,
                             id="cholesky", ci="none", vcov="newey_west", config="",
                             format="table")
                end
            end
            @test occursin("LP IRF", out)
            @test count("LP IRF to", out) >= 2
        end
    end

    @testset "_irf_lp — with bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, horizons=10, lags=4, id="cholesky",
                             ci="bootstrap", replications=50, vcov="newey_west",
                             config="", format="table")
                end
            end
            @test occursin("LP IRF", out)
        end
    end

    @testset "_irf_lp — with var_lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, horizons=10, lags=4, var_lags=6,
                             id="cholesky", ci="none", vcov="newey_west", config="",
                             format="table")
                end
            end
            @test occursin("LP IRF", out)
        end
    end

    @testset "_irf_lp — invalid shock index" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, shocks="5", horizons=10, lags=4,
                             id="cholesky", ci="none", vcov="newey_west", config="",
                             format="table")
                end
            end
        end
    end

end  # IRF handlers

# ═══════════════════════════════════════════════════════════════
# FEVD handlers (fevd.jl)
# ═══════════════════════════════════════════════════════════════

@testset "FEVD handlers" begin

    @testset "register_fevd_commands!" begin
        node = register_fevd_commands!()
        @test node isa NodeCommand
        @test node.name == "fevd"
        @test length(node.subcmds) == 5
        for cmd in ["var", "bvar", "lp", "vecm", "pvar"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_fevd_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="cholesky", format="table")
                end
            end
            @test occursin("FEVD", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_fevd_var — with output file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "fevd.csv")
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="cholesky",
                               format="csv", output=outfile)
                end
            end
            # Output gets split per variable
            @test any(isfile, [replace(outfile, "." => s) for s in ["_var1.", "_var2.", "_var3."]])
        end
    end

    @testset "_fevd_var — uhlig identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_uhlig_config(dir)
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="uhlig",
                               config=cfg, format="table")
                end
            end
            @test occursin("FEVD", out)
            @test occursin("uhlig", out)
        end
    end

    @testset "_fevd_var — arias identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_arias_config(dir)
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="arias",
                               config=cfg, format="table")
                end
            end
            @test occursin("FEVD", out)
            @test occursin("arias", out)
        end
    end

    @testset "_fevd_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_bvar(; data=csv, lags=2, horizons=10, id="cholesky",
                                draws=100, sampler="direct", config="", format="table")
                end
            end
            @test occursin("Bayesian FEVD", out)
        end
    end

    @testset "_fevd_lp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_lp(; data=csv, horizons=10, lags=4, id="cholesky",
                              vcov="newey_west", config="", format="table")
                end
            end
            @test occursin("LP FEVD", out)
        end
    end

end  # FEVD handlers

# ═══════════════════════════════════════════════════════════════
# HD handlers (hd.jl)
# ═══════════════════════════════════════════════════════════════

@testset "HD handlers" begin

    @testset "register_hd_commands!" begin
        node = register_hd_commands!()
        @test node isa NodeCommand
        @test node.name == "hd"
        @test length(node.subcmds) == 4
        for cmd in ["var", "bvar", "lp", "vecm"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_hd_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_var(; data=csv, lags=2, id="cholesky", format="table")
                end
            end
            @test occursin("Historical Decomposition", out)
            @test occursin("verified", out) || occursin("Decomposition", out)
        end
    end

    @testset "_hd_var — uhlig identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_uhlig_config(dir)
            out = cd(dir) do
                _capture() do
                    _hd_var(; data=csv, lags=2, id="uhlig", config=cfg, format="table")
                end
            end
            @test occursin("Historical Decomposition", out)
            @test occursin("uhlig", out)
        end
    end

    @testset "_hd_var — arias identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_arias_config(dir)
            out = cd(dir) do
                _capture() do
                    _hd_var(; data=csv, lags=2, id="arias", config=cfg, format="table")
                end
            end
            @test occursin("Historical Decomposition", out)
            @test occursin("arias", out)
        end
    end

    @testset "_hd_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_bvar(; data=csv, lags=2, id="cholesky", draws=100,
                              sampler="direct", config="", format="table")
                end
            end
            @test occursin("Bayesian Historical Decomposition", out) ||
                  occursin("Bayesian HD", out)
        end
    end

    @testset "_hd_lp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_lp(; data=csv, lags=4, id="cholesky", vcov="newey_west",
                            config="", format="table")
                end
            end
            @test occursin("LP Historical Decomposition", out)
            @test occursin("verified", out) || occursin("Decomposition", out)
        end
    end

    @testset "_hd_lp — with var_lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_lp(; data=csv, lags=4, var_lags=6, id="cholesky",
                            vcov="newey_west", config="", format="table")
                end
            end
            @test occursin("LP Historical Decomposition", out)
        end
    end

end  # HD handlers

# ═══════════════════════════════════════════════════════════════
# Forecast handlers (forecast.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Forecast handlers" begin

    @testset "register_forecast_commands!" begin
        node = register_forecast_commands!()
        @test node isa NodeCommand
        @test node.name == "forecast"
        @test length(node.subcmds) == 13
        for cmd in ["var", "bvar", "lp", "arima", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "vecm"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_forecast_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_var(; data=csv, lags=2, horizons=5, confidence=0.95, format="table")
                end
            end
            @test occursin("Forecast", out)
            @test occursin("95%", out)
        end
    end

    @testset "_forecast_var — custom confidence" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_var(; data=csv, lags=2, horizons=5, confidence=0.90, format="table")
                end
            end
            @test occursin("90%", out)
        end
    end

    @testset "_forecast_var — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_var(; data=csv, lags=nothing, horizons=5, format="table")
                end
            end
            @test occursin("Forecast", out)
        end
    end

    @testset "_forecast_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_bvar(; data=csv, lags=2, horizons=5, draws=100,
                                    sampler="direct", config="", format="table")
                end
            end
            @test occursin("Bayesian", out)
            @test occursin("68% credible", out)
        end
    end

    @testset "_forecast_bvar — with prior config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_prior_config(dir; optimize=true)
            out = cd(dir) do
                _capture() do
                    _forecast_bvar(; data=csv, lags=2, horizons=5, draws=100,
                                    sampler="hmc", config=cfg, format="table")
                end
            end
            @test occursin("Bayesian", out)
        end
    end

    @testset "_forecast_lp — analytical CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_lp(; data=csv, shock=1, horizons=5, shock_size=1.0,
                                  lags=4, vcov="newey_west", ci_method="analytical",
                                  conf_level=0.95, n_boot=100, format="table")
                end
            end
            @test occursin("LP Forecast", out) || occursin("LP forecast", out)
        end
    end

    @testset "_forecast_lp — no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_lp(; data=csv, shock=1, horizons=5, shock_size=1.0,
                                  lags=4, vcov="newey_west", ci_method="none",
                                  conf_level=0.95, format="table")
                end
            end
            @test occursin("LP Forecast", out) || occursin("LP forecast", out)
        end
    end

    @testset "_forecast_lp — custom shock size" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_lp(; data=csv, shock=1, horizons=5, shock_size=2.0,
                                  lags=4, vcov="newey_west", ci_method="analytical",
                                  format="table")
                end
            end
            @test occursin("shock_size=2.0", out)
        end
    end

    @testset "_forecast_arima — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_arima(; data=csv, column=1, p=nothing, d=0, q=0,
                                     horizons=5, confidence=0.95, method="css_mle", format="table")
                end
            end
            @test occursin("Auto ARIMA", out)
            @test occursin("Forecast", out)
        end
    end

    @testset "_forecast_arima — explicit" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_arima(; data=csv, column=1, p=2, d=0, q=0,
                                     horizons=5, confidence=0.90, method="ols", format="table")
                end
            end
            @test occursin("AR(2)", out)
            @test occursin("Forecast", out)
            @test occursin("90%", out)
        end
    end

    @testset "_forecast_arima — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "forecast.csv")
            out = cd(dir) do
                _capture() do
                    _forecast_arima(; data=csv, column=1, p=2, d=0, q=0,
                                     horizons=5, method="ols", format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test nrow(result_df) == 5
            @test "forecast" in names(result_df)
        end
    end

    @testset "_forecast_static — no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_static(; data=csv, nfactors=2,
                                      horizons=5, ci_method="none", format="table")
                end
            end
            @test occursin("Static Factor Forecast", out)
        end
    end

    @testset "_forecast_static — bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_static(; data=csv, nfactors=2,
                                      horizons=5, ci_method="bootstrap", format="table")
                end
            end
            @test occursin("Static Factor Forecast", out)
            @test occursin("standard errors", out) || occursin("_lower", out)
        end
    end

    @testset "_forecast_static — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_static(; data=csv, nfactors=nothing,
                                      horizons=5, format="table")
                end
            end
            @test occursin("Selecting number of factors", out)
        end
    end

    @testset "_forecast_dynamic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_dynamic(; data=csv, nfactors=2,
                                       horizons=5, factor_lags=1, method="twostep", format="table")
                end
            end
            @test occursin("Dynamic Factor Forecast", out)
        end
    end

    @testset "_forecast_gdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_gdfm(; data=csv, nfactors=2,
                                    horizons=5, dynamic_rank=2, format="table")
                end
            end
            @test occursin("GDFM Forecast", out)
        end
    end

    @testset "_forecast_gdfm — auto dynamic_rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_gdfm(; data=csv, nfactors=2,
                                    horizons=5, dynamic_rank=nothing, format="table")
                end
            end
            @test occursin("GDFM Forecast", out)
        end
    end

    @testset "_forecast_arch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_arch(; data=csv, column=1, q=1, horizons=5, format="table")
                end
            end
            @test occursin("ARCH", out)
            @test occursin("Volatility Forecast", out) || occursin("Forecast", out)
        end
    end

    @testset "_forecast_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_garch(; data=csv, column=1, p=1, q=1, horizons=5, format="table")
                end
            end
            @test occursin("GARCH", out)
            @test occursin("Volatility Forecast", out) || occursin("Forecast", out)
            @test occursin("Unconditional variance", out)
        end
    end

    @testset "_forecast_egarch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_egarch(; data=csv, column=1, p=1, q=1, horizons=5, format="table")
                end
            end
            @test occursin("EGARCH", out)
            @test occursin("Volatility Forecast", out) || occursin("Forecast", out)
        end
    end

    @testset "_forecast_gjr_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_gjr_garch(; data=csv, column=1, p=1, q=1, horizons=5, format="table")
                end
            end
            @test occursin("GJR-GARCH", out)
            @test occursin("Volatility Forecast", out) || occursin("Forecast", out)
        end
    end

    @testset "_forecast_sv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_sv(; data=csv, column=1, draws=100, horizons=5, format="table")
                end
            end
            @test occursin("SV", out) || occursin("Stochastic Volatility", out)
            @test occursin("Volatility Forecast", out) || occursin("Forecast", out)
        end
    end

end  # Forecast handlers

# ═══════════════════════════════════════════════════════════════
# VECM handlers (estimate, irf, fevd, hd, forecast vecm + test granger)
# ═══════════════════════════════════════════════════════════════

@testset "VECM handlers" begin

    # ── Structure tests ──────────────────────────────────────

    @testset "register_estimate_commands! includes vecm" begin
        node = register_estimate_commands!()
        @test length(node.subcmds) == 17
        @test haskey(node.subcmds, "vecm")
        @test node.subcmds["vecm"] isa LeafCommand
    end

    @testset "register_irf_commands! includes vecm" begin
        node = register_irf_commands!()
        @test length(node.subcmds) == 5
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_fevd_commands! includes vecm" begin
        node = register_fevd_commands!()
        @test length(node.subcmds) == 5
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_hd_commands! includes vecm" begin
        node = register_hd_commands!()
        @test length(node.subcmds) == 4
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_forecast_commands! includes vecm" begin
        node = register_forecast_commands!()
        @test length(node.subcmds) == 13
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_test_commands! includes granger" begin
        node = register_test_commands!()
        @test length(node.subcmds) == 16
        @test haskey(node.subcmds, "granger")
        @test node.subcmds["granger"] isa LeafCommand
    end

    # ── _load_and_estimate_vecm ──────────────────────────────

    @testset "_load_and_estimate_vecm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            vecm, Y, varnames, p = _load_and_estimate_vecm(csv, 2, "auto", "constant", "johansen", 0.05)
            @test vecm isa MacroEconometricModels.VECMModel
            @test cointegrating_rank(vecm) == 1
            @test size(Y, 2) == 3
            @test p == 2
        end
    end

    @testset "_load_and_estimate_vecm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            vecm, Y, varnames, p = _load_and_estimate_vecm(csv, 3, "2", "constant", "johansen", 0.05)
            @test cointegrating_rank(vecm) == 2
            @test p == 3
        end
    end

    # ── estimate vecm ────────────────────────────────────────

    @testset "_estimate_vecm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_vecm(; data=csv, lags=2, rank="auto", format="table")
                end
            end
            @test occursin("VECM", out)
            @test occursin("rank", out) || occursin("Cointegrat", out)
            @test occursin("beta", out) || occursin("Cointegrating", out)
            @test occursin("alpha", out) || occursin("Adjustment", out)
        end
    end

    @testset "_estimate_vecm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_vecm(; data=csv, lags=3, rank="2", format="table")
                end
            end
            @test occursin("VECM", out)
            @test occursin("rank: 2", out) || occursin("rank=2", out)
        end
    end

    @testset "_estimate_vecm — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "vecm.json")
            out = cd(dir) do
                _capture() do
                    _estimate_vecm(; data=csv, lags=2, rank="auto", format="json", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_estimate_vecm — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "vecm.csv")
            out = cd(dir) do
                _capture() do
                    _estimate_vecm(; data=csv, lags=2, rank="auto", format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_estimate_vecm — deterministic=none" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_vecm(; data=csv, lags=2, rank="1", deterministic="none", format="table")
                end
            end
            @test occursin("VECM", out)
        end
    end

    # ── irf vecm ─────────────────────────────────────────────

    @testset "_irf_vecm — cholesky with bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_vecm(; data=csv, lags=2, rank="auto", shock=1, horizons=10,
                               id="cholesky", ci="bootstrap", replications=100, format="table")
                end
            end
            @test occursin("VECM IRF", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_irf_vecm — no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_vecm(; data=csv, lags=2, rank="1", shock=1, horizons=10,
                               id="cholesky", ci="none", format="table")
                end
            end
            @test occursin("VECM IRF", out)
        end
    end

    @testset "_irf_vecm — sign identification with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_vecm(; data=csv, lags=2, rank="auto", shock=1, horizons=10,
                               id="sign", ci="none", config=cfg, format="table")
                end
            end
            @test occursin("VECM IRF", out) || occursin("sign", out)
        end
    end

    # ── fevd vecm ────────────────────────────────────────────

    @testset "_fevd_vecm — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_vecm(; data=csv, lags=2, rank="auto", horizons=10,
                                id="cholesky", format="table")
                end
            end
            @test occursin("VECM FEVD", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_fevd_vecm — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "fevd.csv")
            out = cd(dir) do
                _capture() do
                    _fevd_vecm(; data=csv, lags=2, rank="1", horizons=10,
                                id="cholesky", format="csv", output=outfile)
                end
            end
            @test any(isfile, [replace(outfile, "." => s) for s in ["_var1.", "_var2.", "_var3."]])
        end
    end

    # ── hd vecm ──────────────────────────────────────────────

    @testset "_hd_vecm — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_vecm(; data=csv, lags=2, rank="auto", id="cholesky", format="table")
                end
            end
            @test occursin("VECM Historical Decomposition", out)
            @test occursin("verified", out) || occursin("Decomposition", out)
        end
    end

    @testset "_hd_vecm — explicit rank with sign id" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            out = cd(dir) do
                _capture() do
                    _hd_vecm(; data=csv, lags=2, rank="1", id="sign", config=cfg, format="table")
                end
            end
            @test occursin("VECM Historical Decomposition", out) ||
                  occursin("Historical Decomposition", out)
        end
    end

    # ── forecast vecm ────────────────────────────────────────

    @testset "_forecast_vecm — no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_vecm(; data=csv, lags=2, rank="auto", horizons=8, format="table")
                end
            end
            @test occursin("VECM Forecast", out)
            @test occursin("rank=1", out) || occursin("rank", out)
        end
    end

    @testset "_forecast_vecm — bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_vecm(; data=csv, lags=2, rank="auto", horizons=8,
                                    ci_method="bootstrap", replications=100, confidence=0.90,
                                    format="table")
                end
            end
            @test occursin("VECM Forecast", out)
            @test occursin("90%", out)
        end
    end

    @testset "_forecast_vecm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_vecm(; data=csv, lags=3, rank="2", horizons=5, format="table")
                end
            end
            @test occursin("VECM Forecast", out)
            @test occursin("rank=2", out) || occursin("rank", out)
        end
    end

    @testset "_forecast_vecm — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "fc.json")
            out = cd(dir) do
                _capture() do
                    _forecast_vecm(; data=csv, lags=2, rank="auto", horizons=5,
                                    format="json", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    # ── test granger ─────────────────────────────────────────

    @testset "_test_granger — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2, rank="auto", format="table")
                end
            end
            @test occursin("Granger Causality", out)
            @test occursin("Short-run", out) || occursin("short", out)
            @test occursin("Long-run", out) || occursin("long", out)
            @test occursin("Strong", out) || occursin("joint", out)
        end
    end

    @testset "_test_granger — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=3, rank="1", format="table")
                end
            end
            @test occursin("Granger", out)
        end
    end

    @testset "_test_granger — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "granger.csv")
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2, rank="auto",
                                   format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_test_granger — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "granger.json")
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2, rank="auto",
                                   format="json", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_test_granger — reversed direction" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=2, effect=1, lags=2, rank="auto", format="table")
                end
            end
            @test occursin("Granger", out)
            @test occursin("var2", out) || occursin("Granger Causality", out)
        end
    end

end  # VECM handlers


# ═══════════════════════════════════════════════════════════════
# Predict handlers (predict.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Predict handlers" begin

    @testset "register_predict_commands!" begin
        node = register_predict_commands!()
        @test node isa NodeCommand
        @test node.name == "predict"
        @test length(node.subcmds) == 12
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_predict_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_var(; data=csv, lags=2, format="table")
                end
            end
            @test occursin("In-Sample Predictions", out)
            @test occursin("VAR(2)", out)
        end
    end

    @testset "_predict_var — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_var(; data=csv, lags=nothing, format="table")
                end
            end
            @test occursin("In-Sample Predictions", out)
        end
    end

    @testset "_predict_var — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "pred.json")
            out = cd(dir) do
                _capture() do
                    _predict_var(; data=csv, lags=2, format="json", output=outfile)
                end
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            @test length(json_data) > 0
        end
    end

    @testset "_predict_var — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "pred.csv")
            out = cd(dir) do
                _capture() do
                    _predict_var(; data=csv, lags=2, format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "t" in names(result_df)
            @test nrow(result_df) > 0
        end
    end

    @testset "_predict_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_bvar(; data=csv, lags=2, draws=100, sampler="nuts",
                                   config="", format="table")
                end
            end
            @test occursin("BVAR(2)", out)
            @test occursin("In-Sample Predictions", out)
        end
    end

    @testset "_predict_bvar — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_bvar(; data=csv, lags=2, draws=100, sampler="nuts",
                                   config="", format="json")
                end
            end
            @test !isempty(out)
        end
    end

    @testset "_predict_arima — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_arima(; data=csv, column=1, format="table")
                end
            end
            @test occursin("In-Sample Predictions", out)
        end
    end

    @testset "_predict_arima — explicit order" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_arima(; data=csv, column=1, p=2, d=1, q=1, format="table")
                end
            end
            @test occursin("ARIMA(2,1,1)", out)
            @test occursin("In-Sample Predictions", out)
        end
    end

    @testset "_predict_arima — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_arima(; data=csv, column=1, format="json")
                end
            end
            @test !isempty(out)
        end
    end

    @testset "_predict_vecm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_vecm(; data=csv, lags=2, rank="1", format="table")
                end
            end
            @test occursin("VECM", out)
            @test occursin("In-Sample Predictions", out)
        end
    end

    @testset "_predict_vecm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_vecm(; data=csv, lags=2, rank="auto", format="table")
                end
            end
            @test occursin("VECM", out)
        end
    end

    # ── Factor model predict tests ──

    @testset "_predict_static" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_static(; data=csv, format="table")
                end
            end
            @test occursin("Static", out) || occursin("factor", out) || occursin("Common Component", out)
        end
    end

    @testset "_predict_static — explicit nfactors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_static(; data=csv, nfactors=2, format="table")
                end
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_predict_dynamic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_dynamic(; data=csv, format="table")
                end
            end
            @test occursin("Dynamic", out) || occursin("Common Component", out)
        end
    end

    @testset "_predict_dynamic — explicit options" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_dynamic(; data=csv, nfactors=2, factor_lags=2, method="twostep", format="table")
                end
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_predict_gdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_gdfm(; data=csv, format="table")
                end
            end
            @test occursin("GDFM", out) || occursin("Common Component", out)
        end
    end

    @testset "_predict_gdfm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_gdfm(; data=csv, dynamic_rank=2, format="table")
                end
            end
            @test occursin("q=2", out)
        end
    end

    @testset "_predict_arch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_arch(; data=csv, column=1, q=1, format="table")
                end
            end
            @test occursin("ARCH", out)
            @test occursin("Conditional Variance", out) || occursin("variance", out)
        end
    end

    @testset "_predict_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GARCH", out)
        end
    end

    @testset "_predict_egarch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_egarch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("EGARCH", out)
        end
    end

    @testset "_predict_gjr_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_gjr_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GJR-GARCH", out)
        end
    end

    @testset "_predict_sv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_sv(; data=csv, column=1, draws=100, format="table")
                end
            end
            @test occursin("SV", out)
        end
    end

    @testset "_predict_arch — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_arch(; data=csv, column=1, q=1, format="json")
                end
            end
            @test !isempty(out)
        end
    end

end  # Predict handlers

# ═══════════════════════════════════════════════════════════════
# Residuals handlers (residuals.jl)
# ═══════════════════════════════════════════════════════════════

@testset "Residuals handlers" begin

    @testset "register_residuals_commands!" begin
        node = register_residuals_commands!()
        @test node isa NodeCommand
        @test node.name == "residuals"
        @test length(node.subcmds) == 12
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_residuals_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_var(; data=csv, lags=2, format="table")
                end
            end
            @test occursin("Residuals", out)
            @test occursin("VAR(2)", out)
        end
    end

    @testset "_residuals_var — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_var(; data=csv, lags=nothing, format="table")
                end
            end
            @test occursin("Residuals", out)
        end
    end

    @testset "_residuals_var — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "resid.json")
            out = cd(dir) do
                _capture() do
                    _residuals_var(; data=csv, lags=2, format="json", output=outfile)
                end
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            @test length(json_data) > 0
        end
    end

    @testset "_residuals_var — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "resid.csv")
            out = cd(dir) do
                _capture() do
                    _residuals_var(; data=csv, lags=2, format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "t" in names(result_df)
            @test nrow(result_df) > 0
        end
    end

    @testset "_residuals_bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_bvar(; data=csv, lags=2, draws=100, sampler="nuts",
                                     config="", format="table")
                end
            end
            @test occursin("BVAR(2)", out)
            @test occursin("Residuals", out)
        end
    end

    @testset "_residuals_bvar — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_bvar(; data=csv, lags=2, draws=100, sampler="nuts",
                                     config="", format="json")
                end
            end
            @test !isempty(out)
        end
    end

    @testset "_residuals_arima — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_arima(; data=csv, column=1, format="table")
                end
            end
            @test occursin("Residuals", out)
        end
    end

    @testset "_residuals_arima — explicit order" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_arima(; data=csv, column=1, p=2, d=1, q=1, format="table")
                end
            end
            @test occursin("ARIMA(2,1,1)", out)
            @test occursin("Residuals", out)
        end
    end

    @testset "_residuals_arima — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_arima(; data=csv, column=1, format="json")
                end
            end
            @test !isempty(out)
        end
    end

    @testset "_residuals_vecm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_vecm(; data=csv, lags=2, rank="1", format="table")
                end
            end
            @test occursin("VECM", out)
            @test occursin("Residuals", out)
        end
    end

    @testset "_residuals_vecm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_vecm(; data=csv, lags=2, rank="auto", format="table")
                end
            end
            @test occursin("VECM", out)
        end
    end

    # ── Factor model residuals tests ──

    @testset "_residuals_static" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_static(; data=csv, format="table")
                end
            end
            @test occursin("Static", out) || occursin("factor", out) || occursin("Idiosyncratic", out)
        end
    end

    @testset "_residuals_static — explicit nfactors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_static(; data=csv, nfactors=2, format="table")
                end
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_residuals_dynamic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_dynamic(; data=csv, format="table")
                end
            end
            @test occursin("Dynamic", out) || occursin("Idiosyncratic", out)
        end
    end

    @testset "_residuals_dynamic — explicit options" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_dynamic(; data=csv, nfactors=2, factor_lags=2, method="twostep", format="table")
                end
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_residuals_gdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_gdfm(; data=csv, format="table")
                end
            end
            @test occursin("GDFM", out) || occursin("Idiosyncratic", out)
        end
    end

    @testset "_residuals_gdfm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_gdfm(; data=csv, dynamic_rank=2, format="table")
                end
            end
            @test occursin("q=2", out)
        end
    end

    @testset "_residuals_arch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_arch(; data=csv, column=1, q=1, format="table")
                end
            end
            @test occursin("ARCH", out)
            @test occursin("Standardized Residuals", out) || occursin("residual", out)
        end
    end

    @testset "_residuals_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GARCH", out)
        end
    end

    @testset "_residuals_egarch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_egarch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("EGARCH", out)
        end
    end

    @testset "_residuals_gjr_garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_gjr_garch(; data=csv, column=1, p=1, q=1, format="table")
                end
            end
            @test occursin("GJR-GARCH", out)
        end
    end

    @testset "_residuals_sv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_sv(; data=csv, column=1, draws=100, format="table")
                end
            end
            @test occursin("SV", out)
        end
    end

    @testset "_residuals_arch — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _residuals_arch(; data=csv, column=1, q=1, format="json")
                end
            end
            @test !isempty(out)
        end
    end

end  # Residuals handlers

# ═══════════════════════════════════════════════════════════════
# Output format tests
# ═══════════════════════════════════════════════════════════════

@testset "Output format tests" begin

    @testset "csv output format for various commands" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "out.csv")
            for (fn, kwargs) in [
                (_test_var_stability, (; data=csv, lags=2, format="csv", output=outfile)),
                (_test_za, (; data=csv, column=1, format="csv", output=outfile)),
            ]
                out = _capture() do
                    fn(; kwargs...)
                end
                @test isfile(outfile)
                rm(outfile; force=true)
            end
        end
    end

    @testset "json output format for various commands" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for (fn, kwargs) in [
                (_test_adf, (; data=csv, column=1, format="json")),
                (_test_np, (; data=csv, column=1, format="json")),
            ]
                out = _capture() do
                    fn(; kwargs...)
                end
                @test !isempty(out)
            end
        end
    end

end  # Output format tests

# ═══════════════════════════════════════════════════════════════
# Edge cases and cross-cutting concerns
# ═══════════════════════════════════════════════════════════════

@testset "Edge Cases" begin

    @testset "nonexistent data file" begin
        mktempdir() do dir
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _estimate_var(; data="/nonexistent/path.csv", lags=2, format="table")
                end
            end
        end
    end

    @testset "nonexistent config file" begin
        @test_throws ErrorException _capture() do
            _build_prior("/nonexistent/config.toml", ones(100, 3), 2)
        end
    end

    @testset "2-variable system" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=80, n=2)
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=2, format="table")
                end
            end
            @test occursin("2 variables", out)
        end
    end

    @testset "single-variable tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _test_adf(; data=csv, column=1, format="table")
            end
            @test occursin("ADF Test", out)
        end
    end

    @testset "json output for estimate handlers" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_var(; data=csv, lags=2, format="json")
                end
            end
            @test !isempty(out)
        end
    end

    @testset "gmm output with file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            outfile = joinpath(dir, "gmm_params.csv")
            out = cd(dir) do
                _capture() do
                    _estimate_gmm(; data=csv, config=cfg, weighting="twostep",
                                   output=outfile, format="csv")
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "parameter" in names(result_df)
            @test "estimate" in names(result_df)
            @test "std_error" in names(result_df)
        end
    end

end  # Edge Cases

@testset "Filter handlers" begin

    @testset "register_filter_commands!" begin
        node = register_filter_commands!()
        @test node isa NodeCommand
        @test node.name == "filter"
        @test length(node.subcmds) == 5
        for cmd in ["hp", "hamilton", "bn", "bk", "bhp"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_filter_hp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_hp(; data=csv, lambda=1600.0, format="table")
                end
            end
            @test occursin("HP Filter", out)
            @test occursin("λ=1600.0", out)
            @test occursin("3 variable(s)", out)
            @test occursin("Cycle Variance Ratios", out)
        end
    end

    @testset "_filter_hp — columns selection" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_hp(; data=csv, lambda=1600.0, columns="1,3", format="table")
                end
            end
            @test occursin("HP Filter", out)
            @test occursin("2 variable(s)", out)
        end
    end

    @testset "_filter_hp — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "hp.json")
            out = cd(dir) do
                _capture() do
                    _filter_hp(; data=csv, lambda=1600.0, format="json", output=outfile)
                end
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            @test length(json_data) > 0
        end
    end

    @testset "_filter_hp — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "hp.csv")
            out = cd(dir) do
                _capture() do
                    _filter_hp(; data=csv, lambda=1600.0, format="csv", output=outfile)
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "t" in names(result_df)
            @test nrow(result_df) == 100
        end
    end

    @testset "_filter_hamilton" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_hamilton(; data=csv, horizon=8, lags=4, format="table")
                end
            end
            @test occursin("Hamilton Filter", out)
            @test occursin("h=8", out)
            @test occursin("p=4", out)
            @test occursin("Cycle Variance Ratios", out)
        end
    end

    @testset "_filter_hamilton — lost observations note" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_hamilton(; data=csv, horizon=8, lags=4, format="table")
                end
            end
            @test occursin("observations lost", out)
        end
    end

    @testset "_filter_bn" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bn(; data=csv, format="table")
                end
            end
            @test occursin("Beveridge-Nelson", out)
            @test occursin("p=auto", out)
            @test occursin("q=auto", out)
            @test occursin("Cycle Variance Ratios", out)
        end
    end

    @testset "_filter_bn — explicit orders" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bn(; data=csv, p=2, q=1, format="table")
                end
            end
            @test occursin("Beveridge-Nelson", out)
            @test occursin("p=2", out)
            @test occursin("q=1", out)
        end
    end

    @testset "_filter_bk" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bk(; data=csv, pl=6, pu=32, K=12, format="table")
                end
            end
            @test occursin("Baxter-King", out)
            @test occursin("pl=6", out)
            @test occursin("pu=32", out)
            @test occursin("K=12", out)
            @test occursin("Cycle Variance Ratios", out)
        end
    end

    @testset "_filter_bk — lost observations note" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bk(; data=csv, pl=6, pu=32, K=12, format="table")
                end
            end
            @test occursin("observations lost", out)
        end
    end

    @testset "_filter_bhp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bhp(; data=csv, lambda=1600.0, stopping="BIC", format="table")
                end
            end
            @test occursin("Boosted HP Filter", out)
            @test occursin("λ=1600.0", out)
            @test occursin("stopping=BIC", out)
            @test occursin("iteration(s)", out)
            @test occursin("Cycle Variance Ratios", out)
        end
    end

    @testset "_filter_bhp — ADF stopping" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bhp(; data=csv, stopping="ADF", sig_p=0.10, format="table")
                end
            end
            @test occursin("Boosted HP Filter", out)
            @test occursin("stopping=ADF", out)
        end
    end

end  # Filter handlers

# ═══════════════════════════════════════════════════════════════
# Panel VAR handlers
# ═══════════════════════════════════════════════════════════════

@testset "Panel VAR handlers" begin

    @testset "register_estimate_commands! includes pvar" begin
        node = register_estimate_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 17
    end

    @testset "register_irf_commands! includes pvar" begin
        node = register_irf_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 5
    end

    @testset "register_fevd_commands! includes pvar" begin
        node = register_fevd_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 5
    end

    @testset "register_test_commands! includes pvar, lr, lm" begin
        node = register_test_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa NodeCommand
        @test length(node.subcmds["pvar"].subcmds) == 4
        @test haskey(node.subcmds["pvar"].subcmds, "hansen_j")
        @test haskey(node.subcmds["pvar"].subcmds, "mmsc")
        @test haskey(node.subcmds["pvar"].subcmds, "lagselect")
        @test haskey(node.subcmds["pvar"].subcmds, "stability")
        @test haskey(node.subcmds, "lr")
        @test node.subcmds["lr"] isa LeafCommand
        @test haskey(node.subcmds, "lm")
        @test node.subcmds["lm"] isa LeafCommand
        @test length(node.subcmds) == 16
    end

    @testset "_parse_varlist" begin
        @test _parse_varlist("") == String[]
        @test _parse_varlist("var1,var2,var3") == ["var1", "var2", "var3"]
        @test _parse_varlist("x, y, z") == ["x", "y", "z"]
        @test _parse_varlist("single") == ["single"]
    end

    @testset "load_panel_data" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=3, T_per=10, n=2)
            panel = load_panel_data(csv, "group", "time")
            @test panel.n_groups == 3
            @test panel.n_vars == 2
            @test panel.T_obs == 30
            @test length(panel.varnames) == 2
        end
    end

    @testset "load_panel_data — missing id column" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException load_panel_data(csv, "nonexistent", "time")
        end
    end

    @testset "load_panel_data — missing time column" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException load_panel_data(csv, "group", "nonexistent")
        end
    end

    @testset "_estimate_pvar — default" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_pvar(; data=csv, id_col="group", time_col="time", lags=1)
                end
            end
            @test occursin("Panel VAR(1)", out)
            @test occursin("gmm", out)
        end
    end

    @testset "_estimate_pvar — feols method" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_pvar(; data=csv, id_col="group", time_col="time", method="feols")
                end
            end
            @test occursin("feols", out)
        end
    end

    @testset "_estimate_pvar — system GMM" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_pvar(; data=csv, id_col="group", time_col="time", system=true)
                end
            end
            @test occursin("System GMM", out)
        end
    end

    @testset "_estimate_pvar — json format" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            outfile = joinpath(dir, "pvar.json")
            out = cd(dir) do
                _capture() do
                    _estimate_pvar(; data=csv, id_col="group", time_col="time",
                                   format="json", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_estimate_pvar — missing id-col error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException cd(dir) do
                _estimate_pvar(; data=csv, time_col="time")
            end
        end
    end

    @testset "_estimate_pvar — missing time-col error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException cd(dir) do
                _estimate_pvar(; data=csv, id_col="group")
            end
        end
    end

    @testset "_estimate_pvar — invalid method error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException cd(dir) do
                _estimate_pvar(; data=csv, id_col="group", time_col="time", method="invalid")
            end
        end
    end

    @testset "_irf_pvar — oirf" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_pvar(; data=csv, id_col="group", time_col="time",
                              horizons=10, irf_type="oirf")
                end
            end
            @test occursin("Panel VAR OIRF", out)
        end
    end

    @testset "_irf_pvar — girf" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_pvar(; data=csv, id_col="group", time_col="time",
                              horizons=10, irf_type="girf")
                end
            end
            @test occursin("Panel VAR GIRF", out)
        end
    end

    @testset "_irf_pvar — missing data and tag error" begin
        mktempdir() do dir
            @test_throws ErrorException cd(dir) do
                _irf_pvar(; data="", id_col="group", time_col="time")
            end
        end
    end

    @testset "_fevd_pvar — default" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_pvar(; data=csv, id_col="group", time_col="time", horizons=10)
                end
            end
            @test occursin("Panel VAR FEVD", out)
        end
    end

    @testset "_test_pvar_hansen_j" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _test_pvar_hansen_j(; data=csv, id_col="group", time_col="time", lags=1)
                end
            end
            @test occursin("Hansen J", out)
            @test occursin("J statistic", out)
        end
    end

    @testset "_test_pvar_hansen_j — missing id error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws ErrorException cd(dir) do
                _test_pvar_hansen_j(; data=csv, time_col="time")
            end
        end
    end

    @testset "_test_pvar_mmsc" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _test_pvar_mmsc(; data=csv, id_col="group", time_col="time", max_lags=4)
                end
            end
            @test occursin("MMSC", out)
            @test occursin("Optimal lag order", out)
        end
    end

    @testset "_test_pvar_lagselect" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _test_pvar_lagselect(; data=csv, id_col="group", time_col="time", max_lags=3)
                end
            end
            @test occursin("Lag Selection", out)
            @test occursin("Optimal lag order", out)
        end
    end

    @testset "_test_pvar_stability" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = cd(dir) do
                _capture() do
                    _test_pvar_stability(; data=csv, id_col="group", time_col="time", lags=1)
                end
            end
            @test occursin("Stability Check", out)
            @test occursin("stable", out)
        end
    end

end  # Panel VAR handlers

# ═══════════════════════════════════════════════════════════════
# LR / LM test handlers
# ═══════════════════════════════════════════════════════════════

@testset "LR/LM test handlers" begin

    @testset "_test_lr" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_lr(; data1=csv, data2=csv, lags1=2, lags2=4)
                end
            end
            @test occursin("Likelihood Ratio Test", out)
            @test occursin("LR statistic", out)
            @test occursin("p-value", out)
        end
    end

    @testset "_test_lm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_lm(; data1=csv, data2=csv, lags1=2, lags2=4)
                end
            end
            @test occursin("Lagrange Multiplier Test", out)
            @test occursin("LM statistic", out)
        end
    end

end  # LR/LM test handlers

# ═══════════════════════════════════════════════════════════════
# Enhanced Granger causality handler
# ═══════════════════════════════════════════════════════════════

@testset "Enhanced Granger handlers" begin

    @testset "_test_granger — vecm (default)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2)
                end
            end
            @test occursin("VECM Granger Causality", out)
            @test occursin("Short-run", out)
        end
    end

    @testset "_test_granger — var model" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2, model="var")
                end
            end
            @test occursin("VAR Granger Causality", out)
            @test occursin("Test statistic", out)
        end
    end

    @testset "_test_granger — var all pairwise" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, lags=2, model="var", all=true)
                end
            end
            @test occursin("all pairwise", out)
            @test occursin("cause", out)
            @test occursin("effect", out)
        end
    end

    @testset "_test_granger — invalid model error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _test_granger(; data=csv, model="invalid")
            end
        end
    end

    @testset "_test_granger — vecm with explicit model option" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_granger(; data=csv, cause=1, effect=2, lags=2, model="vecm")
                end
            end
            @test occursin("VECM Granger Causality", out)
        end
    end

end  # Enhanced Granger handlers

# ═══════════════════════════════════════════════════════════════
# Data command handlers
# ═══════════════════════════════════════════════════════════════

@testset "Data handlers" begin

    @testset "register_data_commands!" begin
        node = register_data_commands!()
        @test node isa NodeCommand
        @test node.name == "data"
        @test length(node.subcmds) == 8
        for cmd in ["list", "load", "describe", "diagnose", "fix", "transform", "filter", "validate"]
            @test haskey(node.subcmds, cmd)
            @test node.subcmds[cmd] isa LeafCommand
        end
    end

    @testset "option counts" begin
        node = register_data_commands!()
        @test length(node.subcmds["list"].options) == 2
        @test length(node.subcmds["load"].options) == 4
        @test length(node.subcmds["describe"].options) == 2
        @test length(node.subcmds["diagnose"].options) == 2
        @test length(node.subcmds["fix"].options) == 3
        @test length(node.subcmds["transform"].options) == 3
        @test length(node.subcmds["filter"].options) == 8
        @test length(node.subcmds["validate"].options) == 3
    end

    @testset "_data_list — table" begin
        out = _capture() do
            _data_list(; format="table")
        end
        @test occursin("Available Datasets", out)
        @test occursin("fred_md", out)
        @test occursin("fred_qd", out)
        @test occursin("pwt", out)
        @test occursin("FRED-MD", out)
        @test occursin("Penn World Table", out)
    end

    @testset "_data_list — json" begin
        mktempdir() do dir
            outfile = joinpath(dir, "datasets.json")
            _capture() do
                _data_list(; format="json", output=outfile)
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            @test length(json_data) == 3
        end
    end

    @testset "_data_load — fred_md" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md")
                end
            end
            @test occursin("Loaded fred_md", out)
            @test occursin("804", out)
            @test occursin("126", out)
            @test occursin("monthly", out)
            @test isfile(joinpath(dir, "fred_md.csv"))
        end
    end

    @testset "_data_load — fred_qd" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_qd")
                end
            end
            @test occursin("Loaded fred_qd", out)
            @test occursin("268", out)
            @test occursin("245", out)
            @test occursin("quarterly", out)
        end
    end

    @testset "_data_load — pwt" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="pwt")
                end
            end
            @test occursin("Loaded pwt", out)
            @test occursin("Panel", out)
            @test occursin("38 groups", out)
        end
    end

    @testset "_data_load — with --transform" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md", transform=true)
                end
            end
            @test occursin("Applied FRED transformation codes", out)
            @test occursin("Loaded fred_md", out)
        end
    end

    @testset "_data_load — with --vars" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md", vars="INDPRO,CPIAUCSL,FEDFUNDS")
                end
            end
            @test occursin("Loaded fred_md", out)
            @test occursin("3", out)
            @test isfile(joinpath(dir, "fred_md.csv"))
            result_df = CSV.read(joinpath(dir, "fred_md.csv"), DataFrame)
            @test ncol(result_df) == 3
        end
    end

    @testset "_data_load — custom output" begin
        mktempdir() do dir
            outfile = joinpath(dir, "my_data.csv")
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md", output=outfile)
                end
            end
            @test isfile(outfile)
            @test occursin("Written to", out)
        end
    end

    @testset "_data_load — invalid name" begin
        mktempdir() do dir
            @test_throws ErrorException cd(dir) do
                _data_load(; name="nonexistent")
            end
        end
    end

    @testset "_data_load — pwt with --vars" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="pwt", vars="rgdpna,pop")
                end
            end
            @test isfile(joinpath(dir, "pwt.csv"))
            result_df = CSV.read(joinpath(dir, "pwt.csv"), DataFrame)
            @test "rgdpna" in names(result_df)
            @test "pop" in names(result_df)
            @test "group" in names(result_df)
            @test "time" in names(result_df)
            @test ncol(result_df) == 4  # group, time, rgdpna, pop
        end
    end

    @testset "_data_load — pwt with --country" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="pwt", country="USA")
                end
            end
            @test occursin("country=USA", out)
        end
    end

    @testset "_data_load — invalid var name" begin
        mktempdir() do dir
            @test_throws ErrorException cd(dir) do
                _data_load(; name="fred_md", vars="NONEXISTENT_VAR")
            end
        end
    end

    @testset "_data_describe — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_describe(; data=csv)
            end
            @test occursin("Data Summary", out)
            @test occursin("100 observations", out)
            @test occursin("3 variables", out)
            @test occursin("Descriptive Statistics", out)
        end
    end

    @testset "_data_describe — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "desc.csv")
            _capture() do
                _data_describe(; data=csv, format="csv", output=outfile)
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "variable" in names(result_df)
            @test "mean" in names(result_df)
            @test "std" in names(result_df)
            @test "skewness" in names(result_df)
            @test nrow(result_df) == 3
        end
    end

    @testset "_data_describe — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "desc.json")
            _capture() do
                _data_describe(; data=csv, format="json", output=outfile)
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            @test length(json_data) == 3
        end
    end

    @testset "_data_diagnose — clean data" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_diagnose(; data=csv)
            end
            @test occursin("Data Diagnostics", out)
            @test occursin("100 observations", out)
            @test occursin("Data is clean", out)
        end
    end

    @testset "_data_diagnose — json output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "diag.json")
            _capture() do
                _data_diagnose(; data=csv, format="json", output=outfile)
            end
            @test isfile(outfile)
        end
    end

    @testset "_data_fix — listwise (default)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _data_fix(; data=csv)
                end
            end
            @test occursin("Fixed data (listwise)", out)
            @test occursin("Written to", out)
            # Default output should be data_clean.csv
            @test occursin("data_clean.csv", out)
        end
    end

    @testset "_data_fix — interpolate" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _data_fix(; data=csv, method="interpolate")
                end
            end
            @test occursin("Fixed data (interpolate)", out)
        end
    end

    @testset "_data_fix — mean" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _data_fix(; data=csv, method="mean")
                end
            end
            @test occursin("Fixed data (mean)", out)
        end
    end

    @testset "_data_fix — custom output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "clean_data.csv")
            out = cd(dir) do
                _capture() do
                    _data_fix(; data=csv, output=outfile)
                end
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test nrow(result_df) == 100
        end
    end

    @testset "_data_fix — invalid method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_fix(; data=csv, method="invalid")
        end
    end

    @testset "_data_transform — explicit tcodes" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _data_transform(; data=csv, tcodes="5,5,1")
                end
            end
            @test occursin("Transformed 3 variable(s)", out)
            @test occursin("tcode=5", out)
            @test occursin("tcode=1", out)
            @test occursin("Δlog", out)
            @test occursin("level", out)
            @test occursin("Written to", out)
        end
    end

    @testset "_data_transform — custom output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "trans.csv")
            out = cd(dir) do
                _capture() do
                    _data_transform(; data=csv, tcodes="1,2,3", output=outfile)
                end
            end
            @test isfile(outfile)
        end
    end

    @testset "_data_transform — missing tcodes error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_transform(; data=csv, tcodes="")
        end
    end

    @testset "_data_transform — wrong number of tcodes" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_transform(; data=csv, tcodes="5,5")
        end
    end

    @testset "_data_filter — hp default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hp")
            end
            @test occursin("Data Filter (hp, component=cycle)", out)
            @test occursin("3 variable(s)", out)
        end
    end

    @testset "_data_filter — hamilton" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hamilton", horizon=8, lags=4)
            end
            @test occursin("Data Filter (hamilton", out)
        end
    end

    @testset "_data_filter — bhp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="bhp")
            end
            @test occursin("Data Filter (bhp", out)
        end
    end

    @testset "_data_filter — trend component" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hp", component="trend")
            end
            @test occursin("trend component", out)
        end
    end

    @testset "_data_filter — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "filtered.csv")
            _capture() do
                _data_filter(; data=csv, method="hp", format="csv", output=outfile)
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "t" in names(result_df)
            @test nrow(result_df) == 100
        end
    end

    @testset "_data_filter — column selection" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hp", columns="1,2")
            end
            @test occursin("2 variable(s)", out)
        end
    end

    @testset "_data_filter — invalid method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_filter(; data=csv, method="invalid")
        end
    end

    @testset "_data_filter — invalid component" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_filter(; data=csv, component="invalid")
        end
    end

    @testset "_data_validate — valid var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="var")
            end
            @test occursin("Data is valid for var estimation", out)
            @test occursin("3 variable(s)", out)
            @test occursin("100 observations", out)
        end
    end

    @testset "_data_validate — valid arima (univariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _data_validate(; data=csv, model="arima")
            end
            @test occursin("Data is valid for arima estimation", out)
        end
    end

    @testset "_data_validate — invalid arima (multivariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="arima")
            end
            @test occursin("validation failed", out)
            @test occursin("univariate", out)
        end
    end

    @testset "_data_validate — missing --model error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_validate(; data=csv, model="")
        end
    end

    @testset "_data_validate — invalid model type error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _data_validate(; data=csv, model="invalid")
        end
    end

    @testset "_data_validate — valid bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="bvar")
            end
            @test occursin("Data is valid for bvar estimation", out)
        end
    end

    @testset "_data_validate — valid garch (univariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _data_validate(; data=csv, model="garch")
            end
            @test occursin("Data is valid for garch estimation", out)
        end
    end

end  # Data handlers

# ──────────────────────────────────────────────────────────────────
# Plot Support Tests
# ──────────────────────────────────────────────────────────────────

@testset "Plot Support" begin

    @testset "_maybe_plot — no-op when both flags false" begin
        mktempdir() do dir
            cd(dir) do
                out = _capture() do
                    _maybe_plot(HPFilterResult(ones(10), zeros(10), 1600.0, 10);
                        plot=false, plot_save="")
                end
                @test out == ""
            end
        end
    end

    @testset "_maybe_plot — plot_save writes HTML file" begin
        mktempdir() do dir
            cd(dir) do
                html_path = joinpath(dir, "test_plot.html")
                out = _capture() do
                    _maybe_plot(HPFilterResult(ones(10), zeros(10), 1600.0, 10);
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
                @test occursin("Plot saved", out)
            end
        end
    end

    @testset "_maybe_plot — plot flag prints browser message" begin
        mktempdir() do dir
            cd(dir) do
                out = _capture() do
                    _maybe_plot(HPFilterResult(ones(10), zeros(10), 1600.0, 10);
                        plot=true, plot_save="")
                end
                @test occursin("Plot opened in browser", out)
            end
        end
    end

    @testset "_maybe_plot — both plot and plot_save" begin
        mktempdir() do dir
            cd(dir) do
                html_path = joinpath(dir, "both.html")
                out = _capture() do
                    _maybe_plot(HPFilterResult(ones(10), zeros(10), 1600.0, 10);
                        plot=true, plot_save=html_path)
                end
                @test isfile(html_path)
                @test occursin("Plot saved", out)
                @test occursin("Plot opened in browser", out)
            end
        end
    end

    # ── --plot-save integration on handlers ──────────────────────

    @testset "_irf_var — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir)
                html_path = joinpath(dir, "irf.html")
                out = _capture() do
                    _irf_var(; data=csv, horizons=20, id="cholesky",
                        ci="none", replications=100,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_fevd_var — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir)
                html_path = joinpath(dir, "fevd.html")
                out = _capture() do
                    _fevd_var(; data=csv, horizons=20, id="cholesky",
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_hd_var — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir)
                html_path = joinpath(dir, "hd.html")
                out = _capture() do
                    _hd_var(; data=csv, id="cholesky",
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_filter_hp — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir; n=1)
                html_path = joinpath(dir, "hp.html")
                out = _capture() do
                    _filter_hp(; data=csv, lambda=1600.0,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                # _per_var_output_path inserts variable name: hp.html → hp_var1.html
                actual_path = joinpath(dir, "hp_var1.html")
                @test isfile(actual_path)
                content = read(actual_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_estimate_arch — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir; n=1)
                html_path = joinpath(dir, "arch.html")
                out = _capture() do
                    _estimate_arch(; data=csv, q=1, column=1,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_estimate_static — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir; T=100, n=5)
                html_path = joinpath(dir, "static.html")
                out = _capture() do
                    _estimate_static(; data=csv, nfactors=0,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_forecast_arima — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir; n=1)
                html_path = joinpath(dir, "arima_fc.html")
                out = _capture() do
                    _forecast_arima(; data=csv, p=0, d=0, q=0,
                        horizons=12, column=1, confidence=0.95,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

    @testset "_forecast_vecm — --plot-save produces HTML" begin
        mktempdir() do dir
            cd(dir) do
                csv = _make_csv(dir; T=100, n=3)
                html_path = joinpath(dir, "vecm_fc.html")
                out = _capture() do
                    _forecast_vecm(; data=csv, lags=2, rank="auto",
                        horizons=12,
                        deterministic="constant",
                        ci_method="bootstrap", replications=100, confidence=0.95,
                        format="table", output="",
                        plot=false, plot_save=html_path)
                end
                @test isfile(html_path)
                content = read(html_path, String)
                @test occursin("mock plot", content)
            end
        end
    end

end  # Plot Support

end  # Command Handlers
