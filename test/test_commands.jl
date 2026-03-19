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
using Statistics: mean, median, var, quantile
using Random

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
include(joinpath(project_root, "src", "commands", "nowcast.jl"))
include(joinpath(project_root, "src", "commands", "dsge.jl"))
include(joinpath(project_root, "src", "commands", "did.jl"))
include(joinpath(project_root, "src", "commands", "spectral.jl"))

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
        @test length(node.subcmds) == 31
        for cmd in ["var", "bvar", "lp", "arima", "gmm", "smm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "fastica", "ml", "vecm", "pvar",
                     "favar", "sdfm", "reg", "iv", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit"]
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

    @testset "_estimate_smm — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _estimate_smm(; data=csv, format="table")
            end
            @test occursin("SMM", out)
            @test occursin("J-statistic", out) || occursin("Converged", out)
        end
    end

    @testset "_estimate_smm — with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            config_path = joinpath(dir, "smm.toml")
            write(config_path, """
            [smm]
            weighting = "optimal"
            sim_ratio = 10
            burn = 200
            """)
            out = _capture() do
                _estimate_smm(; data=csv, config=config_path, format="table")
            end
            @test occursin("SMM", out)
            @test occursin("optimal", out) || occursin("sim_ratio=10", out)
        end
    end

    @testset "_estimate_smm — custom weighting" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _estimate_smm(; data=csv, weighting="identity", sim_ratio=3, format="table")
            end
            @test occursin("SMM", out)
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

    @testset "_estimate_reg — OLS default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_reg(; data=csv, dep="var1", cov_type="hc1",
                               weights="", clusters="", format="table", output="")
            end
            @test occursin("OLS/WLS", out) || occursin("OLS Regression", out)
            @test occursin("R²", out) || occursin("R²", out)
            @test occursin("Coefficient", out)
        end
    end

    @testset "_estimate_reg — WLS with weights" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_reg(; data=csv, dep="var1", cov_type="ols",
                               weights="var4", clusters="", format="table", output="")
            end
            @test occursin("WLS Regression", out)
            @test occursin("R²", out) || occursin("R²", out)
        end
    end

    @testset "_estimate_reg — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            outfile = joinpath(dir, "reg_out.csv")
            out = _capture() do
                _estimate_reg(; data=csv, dep="var1", cov_type="hc1",
                               weights="", clusters="", format="csv", output=outfile)
            end
            @test isfile(outfile)
        end
    end

    @testset "_estimate_iv — 2SLS" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            out = _capture() do
                _estimate_iv(; data=csv, dep="y", endogenous="x1",
                              instruments="z1,z2", cov_type="hc1",
                              format="table", output="")
            end
            @test occursin("IV (2SLS)", out)
            @test occursin("Coefficient", out)
            @test occursin("First-stage F", out)
            @test occursin("Sargan", out)
        end
    end

    @testset "_estimate_iv — missing endogenous error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            @test_throws ErrorException _estimate_iv(; data=csv, dep="y",
                endogenous="", instruments="z1,z2", cov_type="hc1",
                format="table", output="")
        end
    end

    @testset "_estimate_iv — missing instruments error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            @test_throws ErrorException _estimate_iv(; data=csv, dep="y",
                endogenous="x1", instruments="", cov_type="hc1",
                format="table", output="")
        end
    end

    @testset "_estimate_logit — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_logit(; data=csv, dep="var1", cov_type="hc1",
                                 clusters="", maxiter=100, tol=1e-8,
                                 format="table", output="")
            end
            @test occursin("Logit Regression", out)
            @test occursin("Pseudo R²", out) || occursin("Pseudo R", out)
            @test occursin("Converged", out)
            @test occursin("Coefficient", out)
        end
    end

    @testset "_estimate_logit — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_logit(; data=csv, dep="var1", cov_type="ols",
                                 clusters="", maxiter=50, tol=1e-6,
                                 format="json", output="")
            end
            @test occursin("Logit", out)
        end
    end

    @testset "_estimate_probit — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_probit(; data=csv, dep="var1", cov_type="hc1",
                                  clusters="", maxiter=100, tol=1e-8,
                                  format="table", output="")
            end
            @test occursin("Probit Regression", out)
            @test occursin("Pseudo R²", out) || occursin("Pseudo R", out)
            @test occursin("Converged", out)
            @test occursin("Coefficient", out)
        end
    end

    @testset "_estimate_probit — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            outfile = joinpath(dir, "probit_out.csv")
            out = _capture() do
                _estimate_probit(; data=csv, dep="var1", cov_type="ols",
                                  clusters="", maxiter=100, tol=1e-8,
                                  format="csv", output=outfile)
            end
            @test isfile(outfile)
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
        @test length(node.subcmds) == 41
        for cmd in ["adf", "kpss", "pp", "za", "np", "johansen",
                     "normality", "identifiability", "heteroskedasticity",
                     "arch_lm", "ljung_box", "var", "granger", "pvar", "lr", "lm",
                     "andrews", "bai-perron", "panic", "cips", "moon-perron", "factor-break",
                     "fourier-adf", "fourier-kpss", "dfgls", "lm-unitroot",
                     "adf-2break", "gregory-hansen", "vif",
                     "hausman", "breusch-pagan", "f-fe", "pesaran-cd", "wooldridge-ar", "modified-wald",
                     "fisher", "bartlett-wn", "box-pierce", "durbin-watson",
                     "brant", "hausman-iia"]
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
        @test length(node.subcmds) == 7
        for cmd in ["var", "bvar", "lp", "vecm", "pvar", "favar", "sdfm"]
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

    @testset "_irf_var — cumulative flag" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                              ci="none", format="table", cumulative=true)
                end
            end
            @test occursin("Cumulative IRFs computed", out)
            @test occursin("IRF to", out)
        end
    end

    @testset "_irf_bvar — cumulative flag" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_bvar(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                               draws=100, sampler="direct", config="", format="table",
                               cumulative=true)
                end
            end
            @test occursin("Cumulative IRFs computed", out)
            @test occursin("Bayesian IRF", out)
        end
    end

    @testset "_irf_lp — cumulative flag" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_lp(; data=csv, shock=1, horizons=10, lags=4, id="cholesky",
                             ci="none", vcov="newey_west", config="", format="table",
                             cumulative=true)
                end
            end
            @test occursin("Cumulative IRFs computed", out)
            @test occursin("LP IRF", out)
        end
    end

    @testset "_irf_var — identified-set flag" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                              config=cfg, ci="none", format="table", identified_set=true)
                end
            end
            @test occursin("Sign-Identified Set", out)
            @test occursin("accepted", out)
            @test occursin("IRF Identified Set", out)
        end
    end

    @testset "_irf_var — identified-set without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                              config="", ci="none", format="table", identified_set=true)
                end
            end
        end
    end

    @testset "_irf_var — stationary-only flag with bootstrap" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                              ci="bootstrap", replications=100, format="table",
                              stationary_only=true)
                end
            end
            @test occursin("Computing IRFs", out)
            @test occursin("IRF to", out)
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
        @test length(node.subcmds) == 7
        for cmd in ["var", "bvar", "lp", "vecm", "pvar", "favar", "sdfm"]
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
        @test length(node.subcmds) == 5
        for cmd in ["var", "bvar", "lp", "vecm", "favar"]
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
        @test length(node.subcmds) == 14
        for cmd in ["var", "bvar", "lp", "arima", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "vecm", "favar"]
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

    @testset "_forecast_var — bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _forecast_var(; data=csv, lags=2, horizons=5, confidence=0.95,
                                   ci_method="bootstrap", format="table")
                end
            end
            @test occursin("bootstrap", out)
            @test occursin("Forecast", out)
            @test occursin("95%", out)
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
        @test length(node.subcmds) == 31
        @test haskey(node.subcmds, "vecm")
        @test node.subcmds["vecm"] isa LeafCommand
    end

    @testset "register_irf_commands! includes vecm" begin
        node = register_irf_commands!()
        @test length(node.subcmds) == 7
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_fevd_commands! includes vecm" begin
        node = register_fevd_commands!()
        @test length(node.subcmds) == 7
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_hd_commands! includes vecm" begin
        node = register_hd_commands!()
        @test length(node.subcmds) == 5
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_forecast_commands! includes vecm" begin
        node = register_forecast_commands!()
        @test length(node.subcmds) == 14
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_test_commands! includes granger" begin
        node = register_test_commands!()
        @test length(node.subcmds) == 41
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
        @test length(node.subcmds) == 23
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "favar",
                     "reg", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit"]
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

    @testset "_predict_reg — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_reg(; data=csv, dep="var1", cov_type="hc1",
                               weights="", clusters="", format="table", output="")
            end
            @test occursin("OLS Fitted Values", out) || occursin("Fitted Values", out)
            @test occursin("fitted_value", out)
        end
    end

    @testset "_predict_reg — WLS" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_reg(; data=csv, dep="var1", cov_type="ols",
                               weights="var4", clusters="", format="table", output="")
            end
            @test occursin("WLS Fitted Values", out) || occursin("Fitted Values", out)
        end
    end

    @testset "_predict_logit — default fitted" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_logit(; data=csv, dep="var1", cov_type="hc1",
                                 clusters="", threshold=0.5,
                                 marginal_effects=false, odds_ratio=false,
                                 classification_table=false,
                                 format="table", output="")
            end
            @test occursin("Logit Fitted Probabilities", out)
            @test occursin("fitted_prob", out)
        end
    end

    @testset "_predict_logit — marginal effects" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_logit(; data=csv, dep="var1", cov_type="hc1",
                                 clusters="", threshold=0.5,
                                 marginal_effects=true, odds_ratio=false,
                                 classification_table=false,
                                 format="table", output="")
            end
            @test occursin("Marginal Effects", out)
            @test occursin("Effect", out)
        end
    end

    @testset "_predict_logit — odds ratio" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_logit(; data=csv, dep="var1", cov_type="hc1",
                                 clusters="", threshold=0.5,
                                 marginal_effects=false, odds_ratio=true,
                                 classification_table=false,
                                 format="table", output="")
            end
            @test occursin("Odds Ratio", out)
        end
    end

    @testset "_predict_logit — classification table" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_logit(; data=csv, dep="var1", cov_type="hc1",
                                 clusters="", threshold=0.5,
                                 marginal_effects=false, odds_ratio=false,
                                 classification_table=true,
                                 format="table", output="")
            end
            @test occursin("Classification Table", out)
            @test occursin("accuracy", out)
        end
    end

    @testset "_predict_probit — default fitted" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_probit(; data=csv, dep="var1", cov_type="hc1",
                                  clusters="", threshold=0.5,
                                  marginal_effects=false,
                                  classification_table=false,
                                  format="table", output="")
            end
            @test occursin("Probit Fitted Probabilities", out)
            @test occursin("fitted_prob", out)
        end
    end

    @testset "_predict_probit — marginal effects" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_probit(; data=csv, dep="var1", cov_type="hc1",
                                  clusters="", threshold=0.5,
                                  marginal_effects=true,
                                  classification_table=false,
                                  format="table", output="")
            end
            @test occursin("Marginal Effects", out)
            @test occursin("Probit", out)
        end
    end

    @testset "_predict_probit — classification table" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_probit(; data=csv, dep="var1", cov_type="hc1",
                                  clusters="", threshold=0.5,
                                  marginal_effects=false,
                                  classification_table=true,
                                  format="table", output="")
            end
            @test occursin("Classification Table", out)
            @test occursin("accuracy", out)
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
        @test length(node.subcmds) == 23
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr_garch", "sv", "favar",
                     "reg", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit"]
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

    @testset "_residuals_reg — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_reg(; data=csv, dep="var1", cov_type="hc1",
                                 weights="", clusters="", format="table", output="")
            end
            @test occursin("OLS Residuals", out) || occursin("Residuals", out)
            @test occursin("residual", out)
        end
    end

    @testset "_residuals_reg — WLS" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_reg(; data=csv, dep="var1", cov_type="ols",
                                 weights="var4", clusters="", format="table", output="")
            end
            @test occursin("WLS Residuals", out) || occursin("Residuals", out)
        end
    end

    @testset "_residuals_logit — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_logit(; data=csv, dep="var1", cov_type="hc1",
                                   clusters="", format="table", output="")
            end
            @test occursin("Logit Residuals", out)
            @test occursin("residual", out)
        end
    end

    @testset "_residuals_logit — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_logit(; data=csv, dep="var1", cov_type="ols",
                                   clusters="", format="json", output="")
            end
            @test occursin("Logit", out)
            @test !isempty(out)
        end
    end

    @testset "_residuals_probit — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_probit(; data=csv, dep="var1", cov_type="hc1",
                                    clusters="", format="table", output="")
            end
            @test occursin("Probit Residuals", out)
            @test occursin("residual", out)
        end
    end

    @testset "_residuals_probit — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            outfile = joinpath(dir, "probit_resid.csv")
            out = _capture() do
                _residuals_probit(; data=csv, dep="var1", cov_type="ols",
                                    clusters="", format="csv", output=outfile)
            end
            @test isfile(outfile)
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

    @testset "_filter_bn — statespace method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _filter_bn(; data=csv, method="statespace", format="table")
                end
            end
            @test occursin("Beveridge-Nelson", out)
            @test occursin("method=statespace", out)
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
        @test length(node.subcmds) == 31
    end

    @testset "register_irf_commands! includes pvar" begin
        node = register_irf_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 7
    end

    @testset "register_fevd_commands! includes pvar" begin
        node = register_fevd_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 7
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
        @test length(node.subcmds) == 41
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
        @test length(node.subcmds) == 11
        for cmd in ["list", "load", "describe", "diagnose", "fix", "transform", "filter", "validate", "balance", "dropna", "keeprows"]
            @test haskey(node.subcmds, cmd)
            @test node.subcmds[cmd] isa LeafCommand
        end
    end

    @testset "option counts" begin
        node = register_data_commands!()
        @test length(node.subcmds["list"].options) == 2
        @test length(node.subcmds["load"].options) == 6
        @test length(node.subcmds["describe"].options) == 2
        @test length(node.subcmds["diagnose"].options) == 2
        @test length(node.subcmds["fix"].options) == 3
        @test length(node.subcmds["transform"].options) == 3
        @test length(node.subcmds["filter"].options) == 8
        @test length(node.subcmds["validate"].options) == 3
        @test length(node.subcmds["balance"].options) == 5
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
            @test length(json_data) == 5
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

    @testset "_data_list includes mpdta and ddcg" begin
        out = _capture() do
            _data_list(; format="table")
        end
        @test occursin("mpdta", out)
        @test occursin("ddcg", out)
        @test occursin("Callaway", out)
        @test occursin("Acemoglu", out)
    end

    @testset "_data_load — mpdta" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="mpdta")
                end
            end
            @test occursin("Loaded mpdta", out)
            @test occursin("Panel", out)
            @test occursin("500 groups", out)
            @test isfile(joinpath(dir, "mpdta.csv"))
            result_df = CSV.read(joinpath(dir, "mpdta.csv"), DataFrame)
            @test "lemp" in names(result_df)
            @test "group" in names(result_df)
            @test "time" in names(result_df)
        end
    end

    @testset "_data_load — ddcg" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="ddcg")
                end
            end
            @test occursin("Loaded ddcg", out)
            @test occursin("Panel", out)
            @test occursin("184 groups", out)
            @test isfile(joinpath(dir, "ddcg.csv"))
            result_df = CSV.read(joinpath(dir, "ddcg.csv"), DataFrame)
            @test "y" in names(result_df)
            @test "dem" in names(result_df)
        end
    end

    @testset "_data_load — mpdta with --vars" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="mpdta", vars="lemp,lpop")
                end
            end
            @test isfile(joinpath(dir, "mpdta.csv"))
            result_df = CSV.read(joinpath(dir, "mpdta.csv"), DataFrame)
            @test "lemp" in names(result_df)
            @test "lpop" in names(result_df)
            @test ncol(result_df) == 4  # group, time, lemp, lpop
        end
    end

    @testset "_data_load — ddcg with --vars" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="ddcg", vars="y,dem")
                end
            end
            @test isfile(joinpath(dir, "ddcg.csv"))
            result_df = CSV.read(joinpath(dir, "ddcg.csv"), DataFrame)
            @test "y" in names(result_df)
            @test "dem" in names(result_df)
            @test ncol(result_df) == 4  # group, time, y, dem
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

    # ── data balance ──────────────────────────────────────────

    @testset "_data_balance — basic" begin
        cd(mktempdir()) do
            write("data.csv", "a,b\n1.0,2.0\n3.0,NaN\n5.0,6.0\n7.0,8.0\n")
            out = _capture() do
                _data_balance(; data="data.csv")
            end
            @test occursin("Balanc", out)
        end
    end

    @testset "_data_balance — custom method and factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=4)
            out = _capture() do
                _data_balance(; data=csv, method="dfm", factors=2, lags=1)
            end
            @test occursin("Balanc", out)
            @test occursin("dfm", out)
        end
    end

    @testset "_data_balance — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            outfile = joinpath(dir, "balanced.csv")
            out = _capture() do
                _data_balance(; data=csv, format="csv", output=outfile)
            end
            @test isfile(outfile)
        end
    end

    # ── data load --dates ────────────────────────────────────

    @testset "_data_load — --dates option with --path" begin
        cd(mktempdir()) do
            write("data.csv", "date,a,b\n2020Q1,1.0,2.0\n2020Q2,3.0,4.0\n2020Q3,5.0,6.0\n")
            out = _capture() do
                _data_load(; name="", path="data.csv", dates="date")
            end
            @test occursin("Date", out) || occursin("date", out)
        end
    end

    @testset "_data_load — --dates missing column warning" begin
        cd(mktempdir()) do
            write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
            out = _capture() do
                _data_load(; name="", path="data.csv", dates="nonexistent")
            end
            @test occursin("Warning", out) || occursin("not found", out)
        end
    end

    @testset "_data_load — --path without dates" begin
        cd(mktempdir()) do
            write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n")
            out = _capture() do
                _data_load(; name="", path="data.csv")
            end
            @test occursin("Loaded", out)
        end
    end

    @testset "_data_load — named dataset with --dates" begin
        cd(mktempdir()) do
            out = _capture() do
                _data_load(; name="fred_md", dates="INDPRO")
            end
            @test occursin("Loaded", out)
            @test occursin("Date labels", out) || occursin("fred_md", out)
        end
    end

end  # Data handlers

# ──────────────────────────────────────────────────────────────────
# Nowcast Command Tests
# ──────────────────────────────────────────────────────────────────

@testset "Nowcast handlers" begin

    @testset "register_nowcast_commands!" begin
        node = register_nowcast_commands!()
        @test node isa NodeCommand
        @test node.name == "nowcast"
        @test length(node.subcmds) == 5
        for cmd in ["dfm", "bvar", "bridge", "news", "forecast"]
            @test haskey(node.subcmds, cmd)
            @test node.subcmds[cmd] isa LeafCommand
        end
    end

    @testset "option counts" begin
        node = register_nowcast_commands!()
        @test length(node.subcmds["dfm"].options) == 10
        @test length(node.subcmds["bvar"].options) == 6
        @test length(node.subcmds["bridge"].options) == 8
        @test length(node.subcmds["news"].options) == 12
        @test length(node.subcmds["forecast"].options) == 10
    end

    @testset "_nowcast_dfm — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_dfm(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
            @test occursin("Nowcast DFM", out)
            @test occursin("Nowcast:", out) || occursin("nowcast", out)
        end
    end

    @testset "_nowcast_dfm — custom factors and lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_dfm(; data=csv, monthly_vars=4, quarterly_vars=1,
                    factors=3, lags=2, idio="iid")
            end
            @test occursin("Factors: 3", out)
            @test occursin("VAR lags: 2", out)
        end
    end

    @testset "_nowcast_dfm — auto var split" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_dfm(; data=csv)
            end
            @test occursin("4 monthly", out)
            @test occursin("1 quarterly", out)
        end
    end

    @testset "_nowcast_dfm — invalid var split" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            @test_throws ErrorException _nowcast_dfm(; data=csv,
                monthly_vars=3, quarterly_vars=1)
        end
    end

    @testset "_nowcast_bvar — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
            @test occursin("Nowcast BVAR", out)
            @test occursin("Nowcast:", out) || occursin("nowcast", out)
        end
    end

    @testset "_nowcast_bvar — custom lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1, lags=3)
            end
            @test occursin("Lags: 3", out)
        end
    end

    @testset "_nowcast_bridge — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bridge(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
            @test occursin("Nowcast Bridge", out)
            @test occursin("Nowcast:", out) || occursin("nowcast", out)
        end
    end

    @testset "_nowcast_bridge — custom lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bridge(; data=csv, monthly_vars=4, quarterly_vars=1,
                    lag_m=2, lag_q=2, lag_y=2)
            end
            @test occursin("lagM=2", out)
            @test occursin("lagQ=2", out)
            @test occursin("lagY=2", out)
        end
    end

    @testset "_nowcast_news — basic" begin
        mktempdir() do dir
            csv_old = _make_csv(dir; T=100, n=5, colnames=["m1","m2","m3","m4","q1"])
            csv_new = joinpath(dir, "data_new.csv")
            # Create new vintage with slightly more data
            data = Dict{String,Vector{Float64}}()
            for name in ["m1","m2","m3","m4","q1"]
                data[name] = randn(105) .+ 1.0
            end
            CSV.write(csv_new, DataFrame(data))

            out = _capture() do
                _nowcast_news(; data_new=csv_new, data_old=csv_old,
                    monthly_vars=4, quarterly_vars=1, method="dfm")
            end
            @test occursin("News", out) || occursin("news", out)
            @test occursin("Old", out) || occursin("old", out)
        end
    end

    @testset "_nowcast_news — missing data errors" begin
        @test_throws ErrorException _nowcast_news(; data_new="", data_old="old.csv")
        @test_throws ErrorException _nowcast_news(; data_new="new.csv", data_old="")
    end

    @testset "_nowcast_news — bvar method" begin
        mktempdir() do dir
            csv_old = _make_csv(dir; T=100, n=5, colnames=["m1","m2","m3","m4","q1"])
            csv_new = joinpath(dir, "data_new.csv")
            data = Dict{String,Vector{Float64}}()
            for name in ["m1","m2","m3","m4","q1"]
                data[name] = randn(105) .+ 1.0
            end
            CSV.write(csv_new, DataFrame(data))

            out = _capture() do
                _nowcast_news(; data_new=csv_new, data_old=csv_old,
                    monthly_vars=4, quarterly_vars=1, method="bvar")
            end
            @test occursin("bvar", out)
        end
    end

    @testset "_nowcast_forecast — dfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="dfm", horizons=4)
            end
            @test occursin("Nowcast Forecast", out)
            @test occursin("dfm", out)
        end
    end

    @testset "_nowcast_forecast — bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="bvar", horizons=4)
            end
            @test occursin("bvar", out)
        end
    end

    @testset "_nowcast_forecast — bridge" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="bridge", horizons=4)
            end
            @test occursin("bridge", out)
        end
    end

    @testset "_nowcast_forecast — invalid method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            @test_throws ErrorException _nowcast_forecast(; data=csv,
                monthly_vars=4, quarterly_vars=1, method="invalid")
        end
    end

    @testset "_nowcast_forecast — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            outfile = joinpath(dir, "nc_fc.csv")
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="dfm", horizons=4, format="csv", output=outfile)
            end
            @test isfile(outfile)
        end
    end

end  # Nowcast handlers

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

# ─── DSGE Shared Helpers ─────────────────────────────────────────

@testset "DSGE shared helpers" begin
    @testset "_load_dsge_model — TOML file" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9, sigma = 0.01, beta = 0.99 }
            endogenous = ["C", "K", "Y"]
            exogenous = ["e_A"]

            [[model.equations]]
            expr = "C[t] + K[t] = Y[t]"
            [[model.equations]]
            expr = "Y[t] = K[t-1]"
            [[model.equations]]
            expr = "K[t] = rho * K[t-1] + sigma * e_A[t]"
            """)
            out = _capture() do
                spec = _load_dsge_model(toml_path)
                @test spec isa MacroEconometricModels.DSGESpec
                @test spec.n_endog == 3
                @test spec.n_exog == 1
            end
            @test occursin("Loaded DSGE model from TOML", out)
        end
    end

    @testset "_load_dsge_model — .jl file" begin
        mktempdir() do dir
            jl_path = joinpath(dir, "model.jl")
            write(jl_path, """
            model = MacroEconometricModels.DSGESpec(; n_endog=4, n_exog=2)
            """)
            out = _capture() do
                spec = _load_dsge_model(jl_path)
                @test spec isa MacroEconometricModels.DSGESpec
                @test spec.n_endog == 4
                @test spec.n_exog == 2
            end
            @test occursin("Loaded DSGE model from Julia file", out)
        end
    end

    @testset "_load_dsge_model — missing file" begin
        @test_throws ErrorException _load_dsge_model("/nonexistent/model.toml")
    end

    @testset "_load_dsge_model — unsupported extension" begin
        mktempdir() do dir
            bad_path = joinpath(dir, "model.csv")
            write(bad_path, "a,b\n1,2\n")
            @test_throws ErrorException _load_dsge_model(bad_path)
        end
    end

    @testset "_load_dsge_model — TOML missing endogenous" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            """)
            @test_throws ErrorException _load_dsge_model(toml_path)
        end
    end

    @testset "_solve_dsge — default method" begin
        spec = MacroEconometricModels.DSGESpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec)
            @test sol isa MacroEconometricModels.DSGESolution
        end
        @test occursin("Computing steady state", out)
        @test occursin("Linearizing", out)
        @test occursin("Solving", out)
        @test occursin("Determinacy", out)
    end

    @testset "_solve_dsge — perturbation" begin
        spec = MacroEconometricModels.DSGESpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec; method="perturbation", order=1)
            @test sol isa MacroEconometricModels.PerturbationSolution
        end
        @test occursin("perturbation", out)
    end

    @testset "_solve_dsge — projection" begin
        spec = MacroEconometricModels.DSGESpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec; method="projection", degree=5)
            @test sol isa MacroEconometricModels.ProjectionSolution
        end
    end

    @testset "_solve_dsge — with constraint_solver" begin
        spec = MacroEconometricModels.DSGESpec(; n_endog=2, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec; method="gensys", constraint_solver="optim")
            @test sol isa MacroEconometricModels.DSGESolution
        end
    end

    @testset "_load_dsge_constraints" begin
        mktempdir() do dir
            con_path = joinpath(dir, "constraints.toml")
            write(con_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0
            [[constraints.bounds]]
            variable = "c"
            lower = 0.0
            upper = 10.0
            """)
            cons = _load_dsge_constraints(con_path)
            @test length(cons) == 2
            @test cons[1] isa MacroEconometricModels.OccBinConstraint
        end
    end

    @testset "_load_dsge_constraints — nonlinear" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "constraints.toml")
            write(toml_path, """
            [[constraints.nonlinear]]
            expr = "K[t] + C[t] <= Y[t]"
            label = "resource constraint"
            """)
            spec = MacroEconometricModels.DSGESpec(; n_endog=3, n_exog=1)
            cons = _load_dsge_constraints(toml_path; spec=spec)
            @test length(cons) == 1
            @test cons[1] isa MacroEconometricModels.NonlinearConstraint
        end
    end

    @testset "_load_dsge_constraints — nonlinear without spec errors" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "constraints.toml")
            write(toml_path, """
            [[constraints.nonlinear]]
            expr = "K[t] <= Y[t]"
            """)
            @test_throws ErrorException _load_dsge_constraints(toml_path)
        end
    end

    @testset "_load_dsge_constraints — mixed bounds + nonlinear" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "constraints.toml")
            write(toml_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0

            [[constraints.nonlinear]]
            expr = "K[t] <= Y[t]"
            label = "cap"
            """)
            spec = MacroEconometricModels.DSGESpec(; n_endog=3, n_exog=1)
            cons = _load_dsge_constraints(toml_path; spec=spec)
            @test length(cons) == 2
            @test any(c -> c isa MacroEconometricModels.OccBinConstraint, cons)
            @test any(c -> c isa MacroEconometricModels.NonlinearConstraint, cons)
        end
    end

    @testset "_load_dsge_constraints — bounds only backward compat (no spec)" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "constraints.toml")
            write(toml_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0
            """)
            cons = _load_dsge_constraints(toml_path)
            @test length(cons) == 1
            @test cons[1] isa MacroEconometricModels.OccBinConstraint
        end
    end
end

@testset "DSGE commands" begin
    @testset "_dsge_solve — TOML model, default method" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_solve(; model=toml_path, format="table")
            end
            @test occursin("DSGE Solution", out) || occursin("Solving", out)
            @test occursin("Determinacy", out) || occursin("unique", out)
        end
    end

    @testset "_dsge_solve — perturbation method" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_solve(; model=toml_path, method="perturbation", order=1, format="table")
            end
            @test occursin("perturbation", lowercase(out)) || occursin("Perturbation", out)
        end
    end

    @testset "_dsge_solve — OccBin constraints" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            con_path = joinpath(dir, "constraints.toml")
            write(con_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0
            """)
            out = _capture() do
                _dsge_solve(; model=toml_path, constraints=con_path, format="table")
            end
            @test occursin("OccBin", out) || occursin("constraint", out)
        end
    end

    @testset "_dsge_solve — constraint-solver option" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.toml")
            write(model_path, """
            [model]
            endogenous = ["Y", "C"]
            exogenous = ["e_A"]
            parameters = { alpha = 0.36 }
            [[model.equations]]
            expr = "Y[t] = C[t]"
            [solver]
            method = "gensys"
            """)
            cons_path = joinpath(dir, "constraints.toml")
            write(cons_path, """
            [[constraints.nonlinear]]
            expr = "C[t] <= Y[t]"
            label = "resource"
            """)
            out = _capture() do
                _dsge_solve(; model=model_path, constraints=cons_path,
                              constraint_solver="optim")
            end
            @test contains(out, "constraint-solver=optim")
        end
    end

    @testset "_dsge_solve — invalid constraint-solver" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.toml")
            write(model_path, """
            [model]
            endogenous = ["Y", "C"]
            exogenous = ["e_A"]
            parameters = { alpha = 0.36 }
            [[model.equations]]
            expr = "Y[t] = C[t]"
            [solver]
            method = "gensys"
            """)
            @test_throws ErrorException _dsge_solve(;
                model=model_path, constraint_solver="invalid")
        end
    end

    @testset "_dsge_solve — projection method" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_solve(; model=toml_path, method="projection", degree=5, format="table")
            end
            @test occursin("Projection", out) || occursin("projection", out)
        end
    end

    @testset "_dsge_steady_state" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_steady_state(; model=toml_path, format="table")
            end
            @test occursin("Steady State", out)
        end
    end

    @testset "_dsge_steady_state — with constraints" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            con_path = joinpath(dir, "constraints.toml")
            write(con_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0
            """)
            out = _capture() do
                _dsge_steady_state(; model=toml_path, constraints=con_path, format="table")
            end
            @test occursin("Steady State", out)
        end
    end

    @testset "_dsge_steady_state — constraint-solver" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.toml")
            write(model_path, """
            [model]
            endogenous = ["Y", "C"]
            exogenous = ["e_A"]
            parameters = { alpha = 0.36 }
            [[model.equations]]
            expr = "Y[t] = C[t]"
            [solver]
            method = "gensys"
            """)
            cons_path = joinpath(dir, "constraints.toml")
            write(cons_path, """
            [[constraints.bounds]]
            variable = "Y"
            lower = 0.0
            """)
            out = _capture() do
                _dsge_steady_state(; model=model_path, constraints=cons_path,
                                     constraint_solver="nlopt")
            end
            @test contains(out, "Steady State")
        end
    end

    @testset "_dsge_simulate — default" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_simulate(; model=toml_path, periods=50, burn=10, format="table")
            end
            @test occursin("Simulat", out)
        end
    end

    @testset "_dsge_simulate — with seed" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_simulate(; model=toml_path, method="perturbation",
                                 periods=50, burn=10, seed=42, format="table")
            end
            @test occursin("Simulat", out)
        end
    end

    @testset "_dsge_simulate — csv output" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out_path = joinpath(dir, "sim.csv")
            out = _capture() do
                _dsge_simulate(; model=toml_path, periods=20, burn=5,
                                 output=out_path, format="csv")
            end
            @test isfile(out_path)
        end
    end

    @testset "_dsge_irf — standard" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_irf(; model=toml_path, horizon=20, format="table")
            end
            @test occursin("IRF", out) || occursin("Impulse", out)
        end
    end

    @testset "_dsge_irf — OccBin" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            con_path = joinpath(dir, "constraints.toml")
            write(con_path, """
            [[constraints.bounds]]
            variable = "i"
            lower = 0.0
            """)
            out = _capture() do
                _dsge_irf(; model=toml_path, horizon=20, constraints=con_path, format="table")
            end
            @test occursin("OccBin", out) || occursin("IRF", out)
        end
    end

    @testset "_dsge_fevd" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            out = _capture() do
                _dsge_fevd(; model=toml_path, horizon=20, format="table")
            end
            @test occursin("FEVD", out) || occursin("Variance", out)
        end
    end

    @testset "_dsge_estimate — irf_matching" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9, sigma = 0.01 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _dsge_estimate(; model=toml_path, data=csv, method="irf_matching",
                                params="rho,sigma", format="table")
            end
            @test occursin("Estimation", out) || occursin("Estimate", out)
            @test occursin("rho", out) || occursin("sigma", out)
        end
    end

    @testset "_dsge_estimate — missing data" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = rho * Y[t-1] + e[t]"
            """)
            @test_throws ErrorException _dsge_estimate(;
                model=toml_path, data="", method="irf_matching",
                params="rho", format="table")
        end
    end

    @testset "_dsge_estimate — missing params" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = e[t]"
            """)
            csv = _make_csv(dir; T=100, n=1)
            @test_throws ErrorException _dsge_estimate(;
                model=toml_path, data=csv, method="irf_matching",
                params="", format="table")
        end
    end

    @testset "_dsge_perfect_foresight" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C", "K"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + K[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            [[model.equations]]
            expr = "K[t] = e[t]"
            """)
            shock_csv = joinpath(dir, "shocks.csv")
            CSV.write(shock_csv, DataFrame(e = [1.0, 0.5, 0.25, 0.0, 0.0]))
            out = _capture() do
                _dsge_perfect_foresight(; model=toml_path, shocks=shock_csv,
                                         periods=50, format="table")
            end
            @test occursin("Perfect Foresight", out) || occursin("Transition", out)
        end
    end

    @testset "_dsge_perfect_foresight — missing shocks" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = e[t]"
            """)
            @test_throws ErrorException _dsge_perfect_foresight(;
                model=toml_path, shocks="", periods=50, format="table")
        end
    end

    @testset "_dsge_perfect_foresight — constraint-solver" begin
        mktempdir() do dir
            model_path = joinpath(dir, "model.toml")
            write(model_path, """
            [model]
            endogenous = ["Y", "C"]
            exogenous = ["e_A"]
            parameters = { alpha = 0.36 }
            [[model.equations]]
            expr = "Y[t] = C[t]"
            [solver]
            method = "gensys"
            """)
            shock_path = joinpath(dir, "shocks.csv")
            CSV.write(shock_path, DataFrame(e_A = [1.0, 0.5, 0.0]))
            out = _capture() do
                _dsge_perfect_foresight(; model=model_path, shocks=shock_path,
                                          constraint_solver="ipopt")
            end
            @test contains(out, "Perfect Foresight")
        end
    end

    @testset "register_dsge_commands! — structure" begin
        node = register_dsge_commands!()
        @test node isa NodeCommand
        @test node.name == "dsge"
        @test haskey(node.subcmds, "solve")
        @test haskey(node.subcmds, "irf")
        @test haskey(node.subcmds, "fevd")
        @test haskey(node.subcmds, "simulate")
        @test haskey(node.subcmds, "estimate")
        @test haskey(node.subcmds, "perfect-foresight")
        @test haskey(node.subcmds, "steady-state")
        @test haskey(node.subcmds, "bayes")
        @test haskey(node.subcmds, "hd")
        @test length(node.subcmds) == 9
    end
end

# ─── DID Shared Helpers ─────────────────────────────────────────

@testset "DID shared helpers" begin
    @testset "_load_panel_for_did — basic" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3,
                colnames=["outcome", "treat", "covar1"])
            out = _capture() do
                pd = _load_panel_for_did(csv, "group", "time")
                @test pd isa MacroEconometricModels.PanelData
                @test pd.n_groups == 5
                @test pd.n_vars == 3
            end
            @test occursin("Panel", out) || occursin("panel", out)
        end
    end

    @testset "_load_panel_for_did — custom id/time cols" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=3, T_per=10, n=2,
                colnames=["y", "d"])
            out = _capture() do
                pd = _load_panel_for_did(csv, "group", "time")
                @test pd.n_groups == 3
            end
        end
    end
end

# ─── DID Commands ────────────────────────────────────────────────

@testset "DID commands" begin

    function _make_did_csv(dir; G=5, T_per=20)
        rows = G * T_per
        data = Dict{String,Vector}()
        data["unit"] = repeat(1:G, inner=T_per)
        data["time"] = repeat(1:T_per, outer=G)
        data["outcome"] = randn(rows) .+ 1.0
        treat = zeros(Int, rows)
        for i in 1:rows
            g = data["unit"][i]
            t = data["time"][i]
            if g <= 2 && t >= 10
                treat[i] = 1
            elseif g == 3 && t >= 15
                treat[i] = 1
            end
        end
        data["treat"] = treat
        data["covar1"] = randn(rows)
        path = joinpath(dir, "did_panel.csv")
        CSV.write(path, DataFrame(data))
        return path
    end

    @testset "_did_estimate — twfe default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("DID Estimation", out)
            @test occursin("TWFE", out)
            @test occursin("ATT", out)
            @test occursin("Overall ATT", out)
        end
    end

    @testset "_did_estimate — callaway_santanna with group-time" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                    method="cs", id_col="unit", time_col="time", format="table")
            end
            @test occursin("DID Estimation", out)
            @test occursin("CS", out)
            @test occursin("Group-Time ATT", out)
        end
    end

    @testset "_did_estimate — methods cycle" begin
        for m in ["twfe", "sa", "bjs", "dcdh"]
            mktempdir() do dir
                csv = _make_did_csv(dir)
                out = _capture() do
                    _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                        method=m, id_col="unit", time_col="time", format="table")
                end
                @test occursin("DID Estimation", out)
            end
        end
    end

    @testset "_did_estimate — missing outcome" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            @test_throws ErrorException _did_estimate(;
                data=csv, outcome="", treatment="treat",
                id_col="unit", time_col="time", format="table")
        end
    end

    @testset "_did_estimate — csv output" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out_path = joinpath(dir, "result.csv")
            out = _capture() do
                _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", output=out_path, format="csv")
            end
            @test isfile(out_path)
        end
    end

    @testset "_did_event_study — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_event_study(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("Event Study LP", out)
            @test occursin("Coefficient", out)
            @test occursin("Lags", out)
        end
    end

    @testset "_did_event_study — custom leads/horizon" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_event_study(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", leads=5, horizon=10, lags=2,
                    format="table")
            end
            @test occursin("Event Study LP", out)
        end
    end

    @testset "_did_lp_did — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_lp_did(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("LP-DiD", out)
            @test occursin("Specification", out)
            @test occursin("Window", out)
            @test occursin("Pooled post-treatment", out)
        end
    end

    @testset "_did_lp_did — with pmd and reweight" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_lp_did(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", pmd="ccs",
                    reweight=true, pre_window=2, post_window=4,
                    ylags=1, dylags=1, format="table")
            end
            @test occursin("LP-DiD", out)
            @test occursin("pre=2", out)
            @test occursin("post=4", out)
        end
    end

    @testset "_did_lp_did — oneoff spec" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_lp_did(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", oneoff=true, format="table")
            end
            @test occursin("LP-DiD", out)
            @test occursin("oneoff", out)
        end
    end

    @testset "_did_estimate — base_period" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                    method="cs", base_period="universal",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("DID Estimation", out)
            @test occursin("CS", out)
        end
    end

    @testset "_did_test_bacon — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_bacon(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("Bacon Decomposition", out)
            @test occursin("Weight", out)
            @test occursin("Overall ATT", out)
        end
    end

    @testset "_did_test_pretrend — did method" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_pretrend(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", method="did",
                    did_method="twfe", format="table")
            end
            @test occursin("Pre-Trend Test", out)
            @test occursin("p-value", out) || occursin("pvalue", out)
        end
    end

    @testset "_did_test_pretrend — event-study method" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_pretrend(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", method="event-study",
                    format="table")
            end
            @test occursin("Pre-Trend Test", out)
        end
    end

    @testset "_did_test_negweight — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_negweight(; data=csv, treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
            @test occursin("Negative Weight", out)
            @test occursin("Weight Details", out) || occursin("weight", out)
        end
    end

    @testset "_did_test_honest — did method" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_honest(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", mbar=1.5,
                    method="did", did_method="twfe", format="table")
            end
            @test occursin("HonestDiD", out)
            @test occursin("Breakdown", out) || occursin("breakdown", out)
        end
    end

    @testset "_did_test_honest — event-study method" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_honest(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", mbar=2.0,
                    method="event-study", format="table")
            end
            @test occursin("HonestDiD", out)
        end
    end

    @testset "register_did_commands! — structure" begin
        node = register_did_commands!()
        @test node isa NodeCommand
        @test node.name == "did"
        @test haskey(node.subcmds, "estimate")
        @test haskey(node.subcmds, "event-study")
        @test haskey(node.subcmds, "lp-did")
        @test haskey(node.subcmds, "test")
        @test length(node.subcmds) == 4
        test_node = node.subcmds["test"]
        @test test_node isa NodeCommand
        @test haskey(test_node.subcmds, "bacon")
        @test haskey(test_node.subcmds, "pretrend")
        @test haskey(test_node.subcmds, "negweight")
        @test haskey(test_node.subcmds, "honest")
        @test length(test_node.subcmds) == 4
    end
end

# ═══════════════════════════════════════════════════════════════
# FAVAR / SDFM handler tests
# ═══════════════════════════════════════════════════════════════

@testset "FAVAR & SDFM handlers" begin

    @testset "_estimate_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                  method="two_step", draws=5000, format="table")
            end
            @test occursin("FAVAR", out)
        end
    end

    @testset "_estimate_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_sdfm(; data=csv, factors=2, id="cholesky",
                                 var_lags=1, horizon=20, format="table")
            end
            @test occursin("Structural DFM", out)
        end
    end

    @testset "_irf_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _irf_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                             horizons=10, id="cholesky", format="table")
            end
            @test occursin("FAVAR IRF", out)
        end
    end

    @testset "_irf_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _irf_sdfm(; data=csv, factors=2, horizons=10, format="table")
            end
            @test occursin("Structural DFM IRF", out)
        end
    end

    @testset "_fevd_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _fevd_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                              horizons=10, format="table")
            end
            @test occursin("FAVAR FEVD", out)
        end
    end

    @testset "_fevd_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _fevd_sdfm(; data=csv, factors=2, horizons=10, format="table")
            end
            @test occursin("SDFM FEVD", out)
        end
    end

    @testset "_hd_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _hd_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                            horizons=10, format="table")
            end
            @test occursin("FAVAR", out)
        end
    end

    @testset "_forecast_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _forecast_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                  horizons=5, format="table")
            end
            @test occursin("FAVAR Forecast", out)
        end
    end

    @testset "_predict_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _predict_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                 format="table")
            end
            @test occursin("FAVAR", out)
        end
    end

    @testset "_residuals_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _residuals_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                   format="table")
            end
            @test occursin("FAVAR", out)
        end
    end

end

# ═══════════════════════════════════════════════════════════════
# Structural break test handlers
# ═══════════════════════════════════════════════════════════════

@testset "Structural break test handlers" begin

    @testset "_test_andrews" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_andrews(; data=csv, response=1, test="supwald",
                                trimming=0.15, format="table")
            end
            @test occursin("Andrews", out)
        end
    end

    @testset "_test_bai_perron" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_bai_perron(; data=csv, response=1, max_breaks=5,
                                   trimming=0.15, criterion="bic", format="table")
            end
            @test occursin("Bai-Perron", out)
        end
    end

end

# ═══════════════════════════════════════════════════════════════
# Panel unit root test handlers
# ═══════════════════════════════════════════════════════════════

@testset "Panel unit root test handlers" begin

    @testset "_test_panic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_panic(; data=csv, factors="2", method="pooled",
                              id_col="", time_col="", format="table")
            end
            @test occursin("PANIC", out)
        end
    end

    @testset "_test_cips" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_cips(; data=csv, lags="2", deterministic="constant",
                             id_col="", time_col="", format="table")
            end
            @test occursin("CIPS", out)
        end
    end

    @testset "_test_moon_perron" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_moon_perron(; data=csv, factors="2",
                                    id_col="", time_col="", format="table")
            end
            @test occursin("Moon-Perron", out)
        end
    end

    @testset "_test_factor_break" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_factor_break(; data=csv, factors=2, method="breitung_eickmeier",
                                     id_col="", time_col="", format="table")
            end
            @test occursin("Factor Break", out)
        end
    end

end

# ═══════════════════════════════════════════════════════════════
# Bayesian DSGE handler tests
# ═══════════════════════════════════════════════════════════════

@testset "Bayesian DSGE handlers" begin

    # Shared helper to create temp DSGE model + priors + data files
    function _make_bayes_dsge_files(dir)
        model_path = joinpath(dir, "model.toml")
        write(model_path, """
        [model]
        parameters = { rho = 0.9, sigma = 0.01 }
        endogenous = ["Y", "C"]
        exogenous = ["e"]
        [[model.equations]]
        expr = "Y[t] = rho * Y[t-1] + sigma * e[t]"
        [[model.equations]]
        expr = "C[t] = Y[t]"
        [solver]
        method = "gensys"
        """)
        priors_path = joinpath(dir, "priors.toml")
        write(priors_path, """
        [priors]
        [priors.rho]
        dist = "beta"
        a = 0.5
        b = 0.2
        [priors.sigma]
        dist = "inv_gamma"
        a = 2.0
        b = 0.1
        """)
        csv = _make_csv(dir; T=50, n=2)
        return model_path, priors_path, csv
    end

    @testset "_dsge_bayes_estimate" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_estimate(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false, output="", format="table")
            end
            @test occursin("Bayesian DSGE Posterior", out)
            @test occursin("Log marginal likelihood", out)
            @test occursin("Acceptance rate", out)
        end
    end

    @testset "_dsge_bayes_estimate — missing data" begin
        mktempdir() do dir
            model_path, priors_path, _ = _make_bayes_dsge_files(dir)
            @test_throws ErrorException _dsge_bayes_estimate(;
                model=model_path, data="", params="rho,sigma",
                priors=priors_path, sampler="smc",
                n_smc=100, n_particles=50, n_draws=100, burnin=10,
                ess_target=0.5, observables="", solver="gensys", order=1,
                delayed_acceptance=false, output="", format="table")
        end
    end

    @testset "_dsge_bayes_estimate — missing params" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            @test_throws ErrorException _dsge_bayes_estimate(;
                model=model_path, data=csv, params="",
                priors=priors_path, sampler="smc",
                n_smc=100, n_particles=50, n_draws=100, burnin=10,
                ess_target=0.5, observables="", solver="gensys", order=1,
                delayed_acceptance=false, output="", format="table")
        end
    end

    @testset "_dsge_bayes_estimate — missing priors" begin
        mktempdir() do dir
            model_path, _, csv = _make_bayes_dsge_files(dir)
            @test_throws ErrorException _dsge_bayes_estimate(;
                model=model_path, data=csv, params="rho,sigma",
                priors="", sampler="smc",
                n_smc=100, n_particles=50, n_draws=100, burnin=10,
                ess_target=0.5, observables="", solver="gensys", order=1,
                delayed_acceptance=false, output="", format="table")
        end
    end

    @testset "_dsge_bayes_estimate — constraint-solver" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_estimate(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    constraint_solver="path")
            end
            @test contains(out, "Bayesian")
        end
    end

    @testset "_dsge_bayes_irf" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_irf(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    horizon=20, output="", format="table",
                    plot=false, plot_save="")
            end
            @test occursin("Bayesian DSGE IRF", out)
            @test occursin("shock", out)
        end
    end

    @testset "_dsge_bayes_fevd" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_fevd(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    horizon=20, output="", format="table",
                    plot=false, plot_save="")
            end
            @test occursin("Bayesian DSGE FEVD", out)
        end
    end

    @testset "_dsge_bayes_simulate" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_simulate(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    periods=50, output="", format="table",
                    plot=false, plot_save="")
            end
            @test occursin("Bayesian DSGE Simulation", out)
            @test occursin("period", out)
        end
    end

    @testset "_dsge_bayes_summary" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_summary(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    output="", format="table")
            end
            @test occursin("Posterior Summary", out)
            @test occursin("Prior vs Posterior", out)
            @test occursin("Log marginal likelihood", out)
        end
    end

    @testset "_dsge_bayes_compare" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            # Use same model as model2 for simplicity
            out = _capture() do
                _dsge_bayes_compare(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    model2=model_path, params2="rho,sigma", priors2=priors_path,
                    output="", format="table")
            end
            @test occursin("Bayesian Model Comparison", out)
            @test occursin("Bayes factor", out)
            @test occursin("Evidence favors", out)
        end
    end

    @testset "_dsge_bayes_compare — missing model2" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            @test_throws ErrorException _dsge_bayes_compare(;
                model=model_path, data=csv, params="rho,sigma",
                priors=priors_path, sampler="smc",
                n_smc=100, n_particles=50, n_draws=100, burnin=10,
                ess_target=0.5, observables="", solver="gensys", order=1,
                delayed_acceptance=false,
                model2="", params2="rho", priors2=priors_path,
                output="", format="table")
        end
    end

    @testset "_dsge_bayes_predictive" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            out = _capture() do
                _dsge_bayes_predictive(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    n_sim=10, periods=20, output="", format="table",
                    plot=false, plot_save="")
            end
            @test occursin("Posterior Predictive Summary", out)
        end
    end

end

# ═══════════════════════════════════════════════════════════════
# Advanced unit root test handlers
# ═══════════════════════════════════════════════════════════════

@testset "Advanced unit root test handlers" begin

    @testset "_test_fourier_adf" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_fourier_adf(; data=csv, column=1, regression="constant",
                                   fmax=3, lags="aic", max_lags=nothing,
                                   trim=0.15, format="table", output="")
            end
            @test occursin("Fourier ADF", out)
            @test occursin("stationary", out) || occursin("Reject", out)
        end
    end

    @testset "_test_fourier_adf — explicit lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_fourier_adf(; data=csv, column=1, regression="trend",
                                   fmax=2, lags="4", max_lags=nothing,
                                   trim=0.15, format="table", output="")
            end
            @test occursin("Fourier ADF", out)
        end
    end

    @testset "_test_fourier_kpss" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_fourier_kpss(; data=csv, column=1, regression="constant",
                                    fmax=3, bandwidth=nothing,
                                    format="table", output="")
            end
            @test occursin("Fourier KPSS", out)
            @test occursin("stationary", out) || occursin("Reject", out)
        end
    end

    @testset "_test_fourier_kpss — explicit bandwidth" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_fourier_kpss(; data=csv, column=1, regression="trend",
                                    fmax=2, bandwidth=5,
                                    format="table", output="")
            end
            @test occursin("Fourier KPSS", out)
        end
    end

    @testset "_test_dfgls" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_dfgls(; data=csv, column=1, regression="constant",
                              lags="aic", max_lags=nothing,
                              format="table", output="")
            end
            @test occursin("DF-GLS", out)
            @test occursin("stationary", out) || occursin("Reject", out)
        end
    end

    @testset "_test_dfgls — explicit lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_dfgls(; data=csv, column=1, regression="trend",
                              lags="3", max_lags=nothing,
                              format="table", output="")
            end
            @test occursin("DF-GLS", out)
        end
    end

    @testset "_test_lm_unitroot — no breaks" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_lm_unitroot(; data=csv, column=1, breaks=0,
                                    regression="level", lags="aic",
                                    max_lags=nothing, trim=0.15,
                                    format="table", output="")
            end
            @test occursin("LM Unit Root", out)
            @test occursin("stationary", out) || occursin("Reject", out)
        end
    end

    @testset "_test_lm_unitroot — with breaks" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_lm_unitroot(; data=csv, column=1, breaks=2,
                                    regression="trend", lags="aic",
                                    max_lags=nothing, trim=0.15,
                                    format="table", output="")
            end
            @test occursin("LM Unit Root", out)
            @test occursin("Break", out)
        end
    end

    @testset "_test_adf_2break" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_adf_2break(; data=csv, column=1, model="level",
                                   lags="aic", max_lags=nothing,
                                   trim=0.10, format="table", output="")
            end
            @test occursin("ADF 2-Break", out)
            @test occursin("Break 1", out) || occursin("observations", out)
        end
    end

    @testset "_test_adf_2break — trend model" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_adf_2break(; data=csv, column=1, model="trend",
                                   lags="3", max_lags=nothing,
                                   trim=0.10, format="table", output="")
            end
            @test occursin("ADF 2-Break", out)
        end
    end

    @testset "_test_gregory_hansen" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_gregory_hansen(; data=csv, model="C",
                                      lags="aic", max_lags=nothing,
                                      trim=0.15, format="table", output="")
            end
            @test occursin("Gregory-Hansen", out)
            @test occursin("ADF*", out) || occursin("cointegration", out)
        end
    end

    @testset "_test_gregory_hansen — C/T model" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_gregory_hansen(; data=csv, model="C/T",
                                      lags="2", max_lags=nothing,
                                      trim=0.15, format="table", output="")
            end
            @test occursin("Gregory-Hansen", out)
        end
    end

    @testset "_test_vif" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _test_vif(; data=csv, dep="var1", cov_type="hc1",
                            format="table", output="")
            end
            @test occursin("Variance Inflation", out)
            @test occursin("VIF", out)
            @test occursin("Tolerance", out)
        end
    end

    @testset "_test_vif — default dep" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _test_vif(; data=csv, dep="", cov_type="ols",
                            format="table", output="")
            end
            @test occursin("Variance Inflation", out)
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Spectral Commands
    # ══════════════════════════════════════════════════

    @testset "Spectral Commands" begin
        @testset "_spectral_acf" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_acf(; data=csv, column=1, max_lag=20, format="table", output="")
                end
                @test occursin("ACF", out)
            end
        end

        @testset "_spectral_periodogram" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_periodogram(; data=csv, column=1, format="table", output="")
                end
                @test occursin("Periodogram", out)
            end
        end

        @testset "_spectral_density" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_density(; data=csv, column=1, method="welch", format="table", output="")
                end
                @test occursin("Spectral Density", out)
            end
        end

        @testset "_spectral_cross" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_cross(; data=csv, var1=1, var2=2, format="table", output="")
                end
                @test occursin("Cross", out)
            end
        end

        @testset "_spectral_transfer" begin
            out = _capture() do
                _spectral_transfer(; filter="hp", lambda=1600.0, nobs=200, format="table", output="")
            end
            @test occursin("Transfer", out)
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — DSGE HD Commands
    # ══════════════════════════════════════════════════

    @testset "DSGE HD Commands" begin
        @testset "_dsge_hd" begin
            mktempdir() do dir
                toml_path = joinpath(dir, "model.toml")
                write(toml_path, """
                [model]
                parameters = { rho = 0.9, sigma = 0.01, beta = 0.99 }
                endogenous = ["C", "K", "Y"]
                exogenous = ["e_A"]

                [[model.equations]]
                expr = "C[t] + K[t] = Y[t]"
                [[model.equations]]
                expr = "Y[t] = K[t-1]"
                [[model.equations]]
                expr = "K[t] = rho * K[t-1] + sigma * e_A[t]"
                """)
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _dsge_hd(; model=toml_path, data=csv, observables="var1,var2,var3",
                              format="table", output="")
                end
                @test occursin("Historical Decomposition", out)
            end
        end

        @testset "_dsge_bayes_hd" begin
            mktempdir() do dir
                toml_path = joinpath(dir, "model.toml")
                write(toml_path, """
                [model]
                parameters = { rho = 0.9, sigma = 0.01, beta = 0.99 }
                endogenous = ["C", "K", "Y"]
                exogenous = ["e_A"]

                [[model.equations]]
                expr = "C[t] + K[t] = Y[t]"
                [[model.equations]]
                expr = "Y[t] = K[t-1]"
                [[model.equations]]
                expr = "K[t] = rho * K[t-1] + sigma * e_A[t]"
                """)
                csv = _make_csv(dir; T=100, n=3)
                params_path = joinpath(dir, "params.toml")
                write(params_path, """
                [parameters]
                rho = {init = 0.9, lower = 0.0, upper = 1.0}
                sigma = {init = 0.01, lower = 0.001, upper = 0.1}
                """)
                priors_path = joinpath(dir, "priors.toml")
                write(priors_path, """
                [priors]
                [priors.rho]
                dist = "beta"
                a = 0.5
                b = 0.2
                [priors.sigma]
                dist = "inv_gamma"
                a = 2.0
                b = 0.1
                """)
                out = _capture() do
                    _dsge_bayes_hd(; model=toml_path, data=csv, params=params_path,
                                    priors=priors_path, observables="var1,var2,var3",
                                    n_draws=100, sampler="smc",
                                    n_hd_draws=50, format="table", output="")
                end
                @test occursin("Historical Decomposition", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Data Enhancement Commands
    # ══════════════════════════════════════════════════

    @testset "Data Enhancement Commands" begin
        @testset "_data_dropna" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=50, n=3)
                out = _capture() do
                    _data_dropna(; data=csv, format="table", output="")
                end
                @test occursin("Drop NA", out) || occursin("Cleaned", out)
            end
        end

        @testset "_data_keeprows" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=50, n=3)
                out = _capture() do
                    _data_keeprows(; data=csv, rows="1:20", format="table", output="")
                end
                @test occursin("Keep Rows", out) || occursin("Filtered Data", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Panel Regression Commands
    # ══════════════════════════════════════════════════

    @testset "Panel Regression Commands" begin
        @testset "_estimate_preg" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _estimate_preg(; data=csv, dep="var1", indep="var2,var3",
                        method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Regression", out)
                @test occursin("Coefficient", out)
            end
        end

        @testset "_estimate_piv" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=4)
                out = _capture() do
                    _estimate_piv(; data=csv, dep="var1", exog="var2", endog="var3",
                        instruments="var4", method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel IV", out)
            end
        end

        @testset "_estimate_plogit" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _estimate_plogit(; data=csv, dep="var1", indep="var2,var3",
                        method="pooled", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Logit", out)
            end
        end

        @testset "_estimate_pprobit" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _estimate_pprobit(; data=csv, dep="var1", indep="var2,var3",
                        method="pooled", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Probit", out)
            end
        end

        @testset "_predict_preg" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _predict_preg(; data=csv, dep="var1", indep="var2,var3",
                        method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Regression Fitted", out)
            end
        end

        @testset "_predict_piv" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=4)
                out = _capture() do
                    _predict_piv(; data=csv, dep="var1", exog="var2", endog="var3",
                        instruments="var4", method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel IV Fitted", out)
            end
        end

        @testset "_predict_plogit" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _predict_plogit(; data=csv, dep="var1", indep="var2,var3",
                        method="pooled", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Logit Fitted", out)
            end
        end

        @testset "_predict_pprobit" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _predict_pprobit(; data=csv, dep="var1", indep="var2,var3",
                        method="pooled", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Probit Fitted", out)
            end
        end

        @testset "_residuals_preg" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _residuals_preg(; data=csv, dep="var1", indep="var2,var3",
                        method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Regression Residuals", out)
            end
        end

        @testset "_residuals_plogit" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _residuals_plogit(; data=csv, dep="var1", indep="var2,var3",
                        method="pooled", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Panel Logit Residuals", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Ordered/Multinomial Choice Commands
    # ══════════════════════════════════════════════════

    @testset "Ordered/Multinomial Choice Commands" begin
        @testset "_estimate_ologit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _estimate_ologit(; data=csv, dep="var1", cov_type="ols",
                                      clusters="", output="", format="table")
                end
                @test occursin("Ordered Logit", out)
            end
        end

        @testset "_estimate_oprobit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _estimate_oprobit(; data=csv, dep="var1", cov_type="ols",
                                       clusters="", output="", format="table")
                end
                @test occursin("Ordered Probit", out)
            end
        end

        @testset "_estimate_mlogit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _estimate_mlogit(; data=csv, dep="var1", cov_type="ols",
                                      output="", format="table")
                end
                @test occursin("Multinomial Logit", out)
            end
        end

        @testset "_predict_ologit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _predict_ologit(; data=csv, dep="var1", cov_type="hc1",
                                     clusters="", output="", format="table")
                end
                @test occursin("Ordered Logit Predicted", out)
            end
        end

        @testset "_predict_mlogit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _predict_mlogit(; data=csv, dep="var1", cov_type="ols",
                                     output="", format="table")
                end
                @test occursin("Multinomial Logit Predicted", out)
            end
        end

        @testset "_residuals_ologit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _residuals_ologit(; data=csv, dep="var1", cov_type="hc1",
                                       clusters="", output="", format="table")
                end
                @test occursin("Ordered Logit Residuals", out)
            end
        end

        @testset "_residuals_mlogit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _residuals_mlogit(; data=csv, dep="var1", cov_type="ols",
                                       output="", format="table")
                end
                @test occursin("Multinomial Logit Residuals", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Panel Specification Tests
    # ══════════════════════════════════════════════════

    @testset "Panel Specification Test Commands" begin
        @testset "_test_hausman" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_hausman(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Hausman", out)
            end
        end

        @testset "_test_breusch_pagan" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_breusch_pagan(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Breusch-Pagan", out)
            end
        end

        @testset "_test_f_fe" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_f_fe(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("F-Test", out) || occursin("Fixed Effect", out)
            end
        end

        @testset "_test_pesaran_cd" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_pesaran_cd(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Pesaran", out)
            end
        end

        @testset "_test_wooldridge_ar" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_wooldridge_ar(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Wooldridge", out)
            end
        end

        @testset "_test_modified_wald" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_modified_wald(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Modified Wald", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Spectral/Portmanteau Test Commands
    # ══════════════════════════════════════════════════

    @testset "Spectral Test Commands" begin
        @testset "_test_fisher" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_fisher(; data=csv, column=1, format="table", output="")
                end
                @test occursin("Fisher", out)
            end
        end

        @testset "_test_bartlett_wn" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_bartlett_wn(; data=csv, column=1, format="table", output="")
                end
                @test occursin("Bartlett", out)
            end
        end

        @testset "_test_box_pierce" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_box_pierce(; data=csv, column=1, lags=20, format="table", output="")
                end
                @test occursin("Box-Pierce", out)
            end
        end

        @testset "_test_durbin_watson" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_durbin_watson(; data=csv, column=1, format="table", output="")
                end
                @test occursin("Durbin-Watson", out)
            end
        end
    end

    # ══════════════════════════════════════════════════
    # v0.4.0 — Discrete Choice Test Commands
    # ══════════════════════════════════════════════════

    @testset "Discrete Choice Test Commands" begin
        @testset "_test_brant" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _test_brant(; data=csv, dep="var1", cov_type="hc1",
                                  format="table", output="")
                end
                @test occursin("Brant", out)
            end
        end

        @testset "_test_hausman_iia" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _test_hausman_iia(; data=csv, dep="var1", omit_category=1,
                                       format="table", output="")
                end
                @test occursin("Hausman", out) || occursin("IIA", out)
            end
        end
    end

end

end  # Command Handlers
