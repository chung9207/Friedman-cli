# Comprehensive handler tests for src/commands/*.jl
# Uses mock MacroEconometricModels from test/mocks.jl to test all command handlers
# without requiring the actual MacroEconometricModels package.

using Test
using CSV, DataFrames, JSON3, PrettyTables, TOML
using LinearAlgebra: eigvals, diag, I, svd, diagm
using Statistics: mean, median

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
include(joinpath(project_root, "src", "commands", "var.jl"))
include(joinpath(project_root, "src", "commands", "bvar.jl"))
include(joinpath(project_root, "src", "commands", "lp.jl"))
include(joinpath(project_root, "src", "commands", "factor.jl"))
include(joinpath(project_root, "src", "commands", "test_cmd.jl"))
include(joinpath(project_root, "src", "commands", "gmm.jl"))
include(joinpath(project_root, "src", "commands", "arima.jl"))
include(joinpath(project_root, "src", "commands", "nongaussian.jl"))

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
# shared.jl
# ═══════════════════════════════════════════════════════════════

@testset "shared.jl" begin

    @testset "ID_METHOD_MAP" begin
        @test ID_METHOD_MAP["cholesky"] == :cholesky
        @test ID_METHOD_MAP["sign"] == :sign
        @test ID_METHOD_MAP["narrative"] == :narrative
        @test ID_METHOD_MAP["longrun"] == :long_run
        @test length(ID_METHOD_MAP) == 4
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
                chain, Y, varnames, p, n = _load_and_estimate_bvar(csv, 2, cfg, 500, "nuts")
                @test chain isa MockChains
                @test size(Y) == (100, 3)
                @test n == 3
                @test p == 2
            end

            # With empty config (no prior)
            out = _capture() do
                chain, Y, varnames, p, n = _load_and_estimate_bvar(csv, 2, "", 500, "hmc")
                @test chain isa MockChains
            end
        end
    end

    @testset "_build_prior" begin
        mktempdir() do dir
            Y = ones(100, 3) .+ randn(100, 3) * 0.1

            # Empty config → nothing
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
            # Empty config → (nothing, nothing)
            cf, nc = _build_check_func("")
            @test isnothing(cf)
            @test isnothing(nc)

            # Sign restrictions
            sign_cfg = _make_sign_config(dir)
            cf, nc = _build_check_func(sign_cfg)
            @test cf isa Function
            @test isnothing(nc)
            # Sign check is a real function — verify it's callable
            mock_irf = ones(3, 3, 3) * 0.1
            @test cf(mock_irf) isa Bool

            # Narrative restrictions
            # get_identification only extracts sign_matrix when method=="sign",
            # so for method=="narrative", check_func is nothing, narrative_check is a Function
            narr_cfg = _make_narrative_config(dir)
            cf2, nc2 = _build_check_func(narr_cfg)
            # narrative config has method="narrative", sign_matrix only extracted for "sign"
            # but narrative block is extracted
            @test nc2 isa Function
            # Narrative check function is callable
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

        # Unknown method → falls back to :cholesky
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
        B[end, :] .= 0.01  # constant

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

end  # shared.jl

# ═══════════════════════════════════════════════════════════════
# var.jl
# ═══════════════════════════════════════════════════════════════

@testset "var.jl" begin

    @testset "register_var_commands!" begin
        node = register_var_commands!()
        @test node isa NodeCommand
        @test node.name == "var"
        @test length(node.subcmds) == 7
        for cmd in ["estimate", "lagselect", "stability", "irf", "fevd", "hd", "forecast"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_var_estimate — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_estimate(; data=csv, lags=nothing, format="table")
            end
            @test occursin("Estimating VAR(", out)
            @test occursin("Coefficients", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_var_estimate — explicit lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_estimate(; data=csv, lags=3, format="table")
            end
            @test occursin("VAR(3)", out)
        end
    end

    @testset "_var_estimate — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_estimate(; data=csv, lags=2, format="json")
            end
            @test occursin("AIC", out)
        end
    end

    @testset "_var_estimate — csv output to file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "coefs.csv")
            out = _capture() do
                _var_estimate(; data=csv, lags=2, output=outfile, format="csv")
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test nrow(result_df) > 0
        end
    end

    @testset "_var_lagselect" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_lagselect(; data=csv, max_lags=4, criterion="aic", format="table")
            end
            @test occursin("Lag order selection", out)
            @test occursin("Optimal lag order", out)
        end
    end

    @testset "_var_lagselect — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_lagselect(; data=csv, max_lags=4, criterion="bic", format="json")
            end
            @test occursin("optimal_lag", out)
        end
    end

    @testset "_var_stability" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_stability(; data=csv, lags=2, format="table")
            end
            @test occursin("Stationarity Check", out)
            @test occursin("stable", out) || occursin("Stable", out) || occursin("NOT stable", out)
            @test occursin("Max modulus", out)
        end
    end

    @testset "_var_stability — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_stability(; data=csv, lags=nothing)
            end
            @test occursin("Stationarity Check", out)
        end
    end

    @testset "_var_irf — cholesky with bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_irf(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                          ci="bootstrap", replications=100, format="table")
            end
            @test occursin("Computing IRFs", out)
            @test occursin("cholesky", out)
            @test occursin("IRF to", out)
        end
    end

    @testset "_var_irf — cholesky with no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_irf(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                          ci="none", format="table")
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_var_irf — arias identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_arias_config(dir)
            out = _capture() do
                _var_irf(; data=csv, lags=2, shock=1, horizons=10, id="arias",
                          config=cfg, format="table")
            end
            @test occursin("Arias", out)
        end
    end

    @testset "_var_irf — arias without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _var_irf(; data=csv, lags=2, shock=1, horizons=10, id="arias",
                          config="", format="table")
            end
        end
    end

    @testset "_var_irf — sign identification with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            out = _capture() do
                _var_irf(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                          config=cfg, ci="none", format="table")
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_var_irf — shock > n_vars uses generic name" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_irf(; data=csv, lags=2, shock=3, horizons=10, id="cholesky",
                          ci="none", format="table")
            end
            @test occursin("IRF to", out)
        end
    end

    @testset "_var_fevd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_fevd(; data=csv, lags=2, horizons=10, id="cholesky", format="table")
            end
            @test occursin("FEVD", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_var_fevd — with output file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "fevd.csv")
            out = _capture() do
                _var_fevd(; data=csv, lags=2, horizons=10, id="cholesky",
                           format="csv", output=outfile)
            end
            # Output gets split per variable, check at least one file exists
            @test any(isfile, [replace(outfile, "." => s) for s in ["_var1.", "_var2.", "_var3."]])
        end
    end

    @testset "_var_hd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_hd(; data=csv, lags=2, id="cholesky", format="table")
            end
            @test occursin("Historical Decomposition", out)
            @test occursin("verified", out) || occursin("Decomposition", out)
        end
    end

    @testset "_var_forecast" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_forecast(; data=csv, lags=2, horizons=5, confidence=0.95, format="table")
            end
            @test occursin("Forecast", out)
            @test occursin("95%", out)
        end
    end

    @testset "_var_forecast — custom confidence" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_forecast(; data=csv, lags=2, horizons=5, confidence=0.90, format="table")
            end
            @test occursin("90%", out)
        end
    end

    @testset "_var_forecast — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _var_forecast(; data=csv, lags=nothing, horizons=5, format="table")
            end
            @test occursin("Forecast", out)
        end
    end

end  # var.jl

# ═══════════════════════════════════════════════════════════════
# bvar.jl
# ═══════════════════════════════════════════════════════════════

@testset "bvar.jl" begin

    @testset "register_bvar_commands!" begin
        node = register_bvar_commands!()
        @test node isa NodeCommand
        @test node.name == "bvar"
        @test length(node.subcmds) == 6
        for cmd in ["estimate", "posterior", "irf", "fevd", "hd", "forecast"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_bvar_estimate" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_estimate(; data=csv, lags=2, prior="minnesota", draws=100,
                                sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian VAR(2)", out)
            @test occursin("Posterior Mean Coefficients", out)
        end
    end

    @testset "_bvar_estimate — with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_prior_config(dir; optimize=false)
            out = _capture() do
                _bvar_estimate(; data=csv, lags=2, prior="minnesota", draws=100,
                                sampler="nuts", config=cfg, format="table")
            end
            @test occursin("BVAR(2)", out)
        end
    end

    @testset "_bvar_posterior — mean" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_posterior(; data=csv, lags=2, draws=100, sampler="nuts",
                                method="mean", config="", format="table")
            end
            @test occursin("posterior mean", out) || occursin("Mean", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_bvar_posterior — median" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_posterior(; data=csv, lags=2, draws=100, sampler="nuts",
                                method="median", config="", format="table")
            end
            @test occursin("median", out) || occursin("Median", out)
        end
    end

    @testset "_bvar_irf" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_irf(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                           draws=100, sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian IRF", out)
            @test occursin("68% credible", out)
        end
    end

    @testset "_bvar_irf — shock 2" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_irf(; data=csv, lags=2, shock=2, horizons=10, id="cholesky",
                           draws=100, sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian IRF", out)
        end
    end

    @testset "_bvar_fevd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_fevd(; data=csv, lags=2, horizons=10, id="cholesky",
                            draws=100, sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian FEVD", out)
        end
    end

    @testset "_bvar_hd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_hd(; data=csv, lags=2, id="cholesky", draws=100,
                          sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian Historical Decomposition", out) ||
                  occursin("Bayesian HD", out)
        end
    end

    @testset "_bvar_forecast" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _bvar_forecast(; data=csv, lags=2, horizons=5, draws=100,
                                sampler="nuts", config="", format="table")
            end
            @test occursin("Bayesian forecast", out) || occursin("Bayesian VAR(2) Forecast", out)
            @test occursin("68% credible", out)
        end
    end

    @testset "_bvar_forecast — with prior config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_prior_config(dir; optimize=true)
            out = _capture() do
                _bvar_forecast(; data=csv, lags=2, horizons=5, draws=100,
                                sampler="hmc", config=cfg, format="table")
            end
            @test occursin("Bayesian", out)
        end
    end

end  # bvar.jl

# ═══════════════════════════════════════════════════════════════
# lp.jl
# ═══════════════════════════════════════════════════════════════

@testset "lp.jl" begin

    @testset "register_lp_commands!" begin
        node = register_lp_commands!()
        @test node isa NodeCommand
        @test node.name == "lp"
        @test length(node.subcmds) == 5
        for cmd in ["estimate", "irf", "fevd", "hd", "forecast"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_lp_estimate — standard" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="standard", shock=1, horizons=10,
                              control_lags=4, vcov="newey_west", format="table")
            end
            @test occursin("Local Projections", out)
            @test occursin("LP IRF", out)
        end
    end

    @testset "_lp_estimate — iv" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            iv_csv = _make_instruments_csv(dir; T=100, n_inst=2)
            out = _capture() do
                _lp_estimate(; data=csv, method="iv", shock=1, horizons=10,
                              control_lags=4, vcov="newey_west", instruments=iv_csv,
                              format="table")
            end
            @test occursin("LP-IV", out)
            @test occursin("F-statistic", out)
        end
    end

    @testset "_lp_estimate — iv missing instruments error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _lp_estimate(; data=csv, method="iv", shock=1, horizons=10,
                              control_lags=4, vcov="newey_west", instruments="",
                              format="table")
            end
        end
    end

    @testset "_lp_estimate — smooth with auto lambda" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="smooth", shock=1, horizons=10,
                              knots=3, lambda=0.0, format="table")
            end
            @test occursin("Smooth LP", out)
            @test occursin("Cross-validating", out)
        end
    end

    @testset "_lp_estimate — smooth with explicit lambda" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="smooth", shock=1, horizons=10,
                              knots=3, lambda=0.5, format="table")
            end
            @test occursin("Smooth LP", out)
            @test !occursin("Cross-validating", out)
        end
    end

    @testset "_lp_estimate — state" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="state", shock=1, horizons=10,
                              state_var=2, gamma=1.5, transition="logistic", format="table")
            end
            @test occursin("State-Dependent LP", out)
            @test occursin("Expansion", out) || occursin("expansion", out)
            @test occursin("Recession", out) || occursin("recession", out)
            @test occursin("Regime Difference Test", out)
        end
    end

    @testset "_lp_estimate — state missing state_var error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _lp_estimate(; data=csv, method="state", shock=1, horizons=10,
                              state_var=nothing, gamma=1.5, format="table")
            end
        end
    end

    @testset "_lp_estimate — propensity" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="propensity", treatment=1, horizons=10,
                              score_method="logit", format="table")
            end
            @test occursin("Propensity Score LP", out)
            @test occursin("Diagnostics", out)
            @test occursin("ATE", out)
        end
    end

    @testset "_lp_estimate — robust" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_estimate(; data=csv, method="robust", treatment=1, horizons=10,
                              score_method="logit", format="table")
            end
            @test occursin("Doubly Robust LP", out)
            @test occursin("Diagnostics", out)
        end
    end

    @testset "_lp_estimate — unknown method error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _lp_estimate(; data=csv, method="invalid", format="table")
            end
        end
    end

    @testset "_lp_irf — single shock" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_irf(; data=csv, shock=1, horizons=10, lags=4, id="cholesky",
                         ci="none", vcov="newey_west", config="", format="table")
            end
            @test occursin("LP IRF", out)
            @test occursin("cholesky", out)
        end
    end

    @testset "_lp_irf — multi-shock" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_irf(; data=csv, shock=1, shocks="1,2", horizons=10, lags=4,
                         id="cholesky", ci="none", vcov="newey_west", config="",
                         format="table")
            end
            @test occursin("LP IRF", out)
            # Should output tables for both shocks
            @test count("LP IRF to", out) >= 2
        end
    end

    @testset "_lp_irf — with bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_irf(; data=csv, shock=1, horizons=10, lags=4, id="cholesky",
                         ci="bootstrap", replications=50, vcov="newey_west",
                         config="", format="table")
            end
            @test occursin("LP IRF", out)
        end
    end

    @testset "_lp_irf — with var_lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_irf(; data=csv, shock=1, horizons=10, lags=4, var_lags=6,
                         id="cholesky", ci="none", vcov="newey_west", config="",
                         format="table")
            end
            @test occursin("LP IRF", out)
        end
    end

    @testset "_lp_irf — invalid shock index" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _lp_irf(; data=csv, shock=1, shocks="5", horizons=10, lags=4,
                         id="cholesky", ci="none", vcov="newey_west", config="",
                         format="table")
            end
        end
    end

    @testset "_lp_fevd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_fevd(; data=csv, horizons=10, lags=4, id="cholesky",
                          vcov="newey_west", config="", format="table")
            end
            @test occursin("LP FEVD", out)
        end
    end

    @testset "_lp_hd" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_hd(; data=csv, lags=4, id="cholesky", vcov="newey_west",
                        config="", format="table")
            end
            @test occursin("LP Historical Decomposition", out)
            @test occursin("verified", out) || occursin("Decomposition", out)
        end
    end

    @testset "_lp_hd — with var_lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_hd(; data=csv, lags=4, var_lags=6, id="cholesky",
                        vcov="newey_west", config="", format="table")
            end
            @test occursin("LP Historical Decomposition", out)
        end
    end

    @testset "_lp_forecast — analytical CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_forecast(; data=csv, shock=1, horizons=5, shock_size=1.0,
                              lags=4, vcov="newey_west", ci_method="analytical",
                              conf_level=0.95, n_boot=100, format="table")
            end
            @test occursin("LP Forecast", out) || occursin("LP forecast", out)
        end
    end

    @testset "_lp_forecast — no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_forecast(; data=csv, shock=1, horizons=5, shock_size=1.0,
                              lags=4, vcov="newey_west", ci_method="none",
                              conf_level=0.95, format="table")
            end
            @test occursin("LP Forecast", out) || occursin("LP forecast", out)
        end
    end

    @testset "_lp_forecast — custom shock size" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _lp_forecast(; data=csv, shock=1, horizons=5, shock_size=2.0,
                              lags=4, vcov="newey_west", ci_method="analytical",
                              format="table")
            end
            @test occursin("shock_size=2.0", out)
        end
    end

end  # lp.jl

# ═══════════════════════════════════════════════════════════════
# factor.jl
# ═══════════════════════════════════════════════════════════════

@testset "factor.jl" begin

    @testset "register_factor_commands!" begin
        node = register_factor_commands!()
        @test node isa NodeCommand
        @test node.name == "factor"
        @test length(node.subcmds) == 2
        @test haskey(node.subcmds, "estimate")
        @test haskey(node.subcmds, "forecast")
        est_node = node.subcmds["estimate"]
        @test est_node isa NodeCommand
        @test length(est_node.subcmds) == 3
        for sub in ["static", "dynamic", "gdfm"]
            @test haskey(est_node.subcmds, sub)
        end
    end

    @testset "_factor_static — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_static(; data=csv, nfactors=nothing, criterion="ic1", format="table")
            end
            @test occursin("static factor model", out)
            @test occursin("Scree Data", out)
            @test occursin("Factor Loadings", out)
        end
    end

    @testset "_factor_static — explicit factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_static(; data=csv, nfactors=3, format="table")
            end
            @test occursin("3 factors", out)
        end
    end

    @testset "_factor_dynamic — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_dynamic(; data=csv, nfactors=nothing, factor_lags=1,
                                 method="twostep", format="table")
            end
            @test occursin("dynamic factor model", out)
            @test occursin("stationary", out) || occursin("Stationary", out) || occursin("not stationary", out)
            @test occursin("Companion Matrix", out) || occursin("eigenvalues", out)
        end
    end

    @testset "_factor_dynamic — explicit factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_dynamic(; data=csv, nfactors=2, factor_lags=2,
                                 method="twostep", format="table")
            end
            @test occursin("2 factors", out)
        end
    end

    @testset "_factor_gdfm — auto rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_gdfm(; data=csv, nfactors=nothing, dynamic_rank=nothing, format="table")
            end
            @test occursin("GDFM", out)
            @test occursin("Common Variance Shares", out)
            @test occursin("Average common variance share", out)
        end
    end

    @testset "_factor_gdfm — explicit rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_gdfm(; data=csv, nfactors=3, dynamic_rank=2, format="table")
            end
            @test occursin("static rank=3", out)
            @test occursin("dynamic rank=2", out)
        end
    end

    @testset "_factor_forecast — static, no CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="static", nfactors=2,
                                  horizon=5, ci_method="none", format="table")
            end
            @test occursin("Static Factor Forecast", out)
        end
    end

    @testset "_factor_forecast — static, bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="static", nfactors=2,
                                  horizon=5, ci_method="bootstrap", format="table")
            end
            @test occursin("Static Factor Forecast", out)
            @test occursin("standard errors", out) || occursin("_lower", out)
        end
    end

    @testset "_factor_forecast — dynamic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="dynamic", nfactors=2,
                                  horizon=5, factor_lags=1, method="twostep", format="table")
            end
            @test occursin("Dynamic Factor Forecast", out)
        end
    end

    @testset "_factor_forecast — gdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="gdfm", nfactors=2,
                                  horizon=5, dynamic_rank=2, format="table")
            end
            @test occursin("GDFM Forecast", out)
        end
    end

    @testset "_factor_forecast — gdfm auto dynamic_rank" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="gdfm", nfactors=2,
                                  horizon=5, dynamic_rank=nothing, format="table")
            end
            @test occursin("GDFM Forecast", out)
        end
    end

    @testset "_factor_forecast — unknown model error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            @test_throws ErrorException _capture() do
                _factor_forecast(; data=csv, model="invalid", format="table")
            end
        end
    end

    @testset "_factor_forecast — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _factor_forecast(; data=csv, model="static", nfactors=nothing,
                                  horizon=5, format="table")
            end
            @test occursin("Selecting number of factors", out)
        end
    end

end  # factor.jl

# ═══════════════════════════════════════════════════════════════
# test_cmd.jl
# ═══════════════════════════════════════════════════════════════

@testset "test_cmd.jl" begin

    @testset "register_test_commands!" begin
        node = register_test_commands!()
        @test node isa NodeCommand
        @test node.name == "test"
        @test length(node.subcmds) == 6
        for cmd in ["adf", "kpss", "pp", "za", "np", "johansen"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_extract_series — valid column" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            y, vname = _extract_series(csv, 1)
            @test length(y) == 50
            @test vname isa String
        end
    end

    @testset "_extract_series — out of range" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=3)
            @test_throws ErrorException _extract_series(csv, 10)
        end
    end

    @testset "_test_adf — reject" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_adf(; data=csv, column=1, max_lags=nothing, trend="constant", format="table")
            end
            @test occursin("ADF Test", out)
            # Mock returns pvalue=0.01, so should reject
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
            # Mock KPSSResult has no pvalue field, so should hit the else branch
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
            # Mock pvalue=0.02 < 0.05, so should reject
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

end  # test_cmd.jl

# ═══════════════════════════════════════════════════════════════
# gmm.jl
# ═══════════════════════════════════════════════════════════════

@testset "gmm.jl" begin

    @testset "register_gmm_commands!" begin
        node = register_gmm_commands!()
        @test node isa NodeCommand
        @test node.name == "gmm"
        @test length(node.subcmds) == 1
        @test haskey(node.subcmds, "estimate")
    end

    @testset "_gmm_estimate — missing config error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _gmm_estimate(; data=csv, config="", weighting="twostep", format="table")
            end
        end
    end

    @testset "_gmm_estimate — with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            out = _capture() do
                _gmm_estimate(; data=csv, config=cfg, weighting="twostep", format="table")
            end
            @test occursin("GMM", out) || occursin("Estimating GMM", out)
            @test occursin("J-test", out) || occursin("Hansen", out)
        end
    end

    @testset "_gmm_estimate — j-test p > 0.05 (cannot reject)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            out = _capture() do
                _gmm_estimate(; data=csv, config=cfg, weighting="twostep", format="table")
            end
            # Mock J_pvalue = 0.65, so cannot reject
            @test occursin("Cannot reject", out) || occursin("cannot reject", out)
        end
    end

    @testset "_gmm_estimate — with output file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            outfile = joinpath(dir, "gmm_params.csv")
            out = _capture() do
                _gmm_estimate(; data=csv, config=cfg, weighting="twostep",
                               output=outfile, format="csv")
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test "parameter" in names(result_df)
            @test "estimate" in names(result_df)
            @test "std_error" in names(result_df)
        end
    end

    @testset "_gmm_estimate — different weightings" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["output", "inflation", "rate"])
            cfg = _make_gmm_config(dir; colnames=["output", "inflation", "rate"])
            for w in ["identity", "optimal", "twostep", "iterated"]
                out = _capture() do
                    _gmm_estimate(; data=csv, config=cfg, weighting=w, format="table")
                end
                @test occursin("GMM", out) || occursin("Estimating GMM", out)
            end
        end
    end

end  # gmm.jl

# ═══════════════════════════════════════════════════════════════
# arima.jl
# ═══════════════════════════════════════════════════════════════

@testset "arima.jl" begin

    @testset "register_arima_commands!" begin
        node = register_arima_commands!()
        @test node isa NodeCommand
        @test node.name == "arima"
        @test length(node.subcmds) == 2
        @test haskey(node.subcmds, "estimate")
        @test haskey(node.subcmds, "forecast")
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

    @testset "_arima_estimate — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_estimate(; data=csv, column=1, p=nothing, d=0, q=0,
                                 max_p=3, max_d=1, max_q=3, criterion="bic",
                                 method="css_mle", format="table")
            end
            @test occursin("Auto ARIMA", out)
            @test occursin("Selected model", out)
            @test occursin("Coefficients", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_arima_estimate — explicit AR" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_estimate(; data=csv, column=1, p=2, d=0, q=0,
                                 method="ols", format="table")
            end
            @test occursin("AR(2)", out)
        end
    end

    @testset "_arima_estimate — explicit ARIMA" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_estimate(; data=csv, column=1, p=1, d=1, q=1,
                                 method="css_mle", format="table")
            end
            @test occursin("ARIMA(1,1,1)", out)
        end
    end

    @testset "_arima_estimate — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_estimate(; data=csv, column=1, p=2, d=0, q=0,
                                 method="ols", format="json")
            end
            @test occursin("AIC", out)
        end
    end

    @testset "_arima_forecast — auto" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_forecast(; data=csv, column=1, p=nothing, d=0, q=0,
                                 horizons=5, confidence=0.95, method="css_mle", format="table")
            end
            @test occursin("Auto ARIMA", out)
            @test occursin("Forecast", out)
        end
    end

    @testset "_arima_forecast — explicit" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _arima_forecast(; data=csv, column=1, p=2, d=0, q=0,
                                 horizons=5, confidence=0.90, method="ols", format="table")
            end
            @test occursin("AR(2)", out)
            @test occursin("Forecast", out)
            @test occursin("90%", out)
        end
    end

    @testset "_arima_forecast — csv output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "forecast.csv")
            out = _capture() do
                _arima_forecast(; data=csv, column=1, p=2, d=0, q=0,
                                 horizons=5, method="ols", format="csv", output=outfile)
            end
            @test isfile(outfile)
            result_df = CSV.read(outfile, DataFrame)
            @test nrow(result_df) == 5
            @test "forecast" in names(result_df)
        end
    end

end  # arima.jl

# ═══════════════════════════════════════════════════════════════
# nongaussian.jl
# ═══════════════════════════════════════════════════════════════

@testset "nongaussian.jl" begin

    @testset "register_nongaussian_commands!" begin
        node = register_nongaussian_commands!()
        @test node isa NodeCommand
        @test node.name == "nongaussian"
        @test length(node.subcmds) == 5
        for cmd in ["fastica", "ml", "heteroskedasticity", "normality", "identifiability"]
            @test haskey(node.subcmds, cmd)
        end
    end

    @testset "_nongaussian_fastica — default (fastica)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_fastica(; data=csv, lags=2, method="fastica",
                                      contrast="logcosh", format="table")
            end
            @test occursin("Non-Gaussian SVAR", out)
            @test occursin("method=fastica", out)
            @test occursin("Structural Impact Matrix", out)
            @test occursin("Structural Shocks", out)
            @test occursin("Converged", out) || occursin("converged", out)
        end
    end

    @testset "_nongaussian_fastica — all 6 ICA methods" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for method in ["fastica", "infomax", "jade", "sobi", "dcov", "hsic"]
                out = _capture() do
                    _nongaussian_fastica(; data=csv, lags=2, method=method, format="table")
                end
                @test occursin("method=$method", out)
                @test occursin("Structural Impact Matrix", out)
            end
        end
    end

    @testset "_nongaussian_fastica — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_fastica(; data=csv, lags=nothing, method="fastica", format="table")
            end
            @test occursin("Non-Gaussian SVAR", out)
        end
    end

    @testset "_nongaussian_fastica — with output file" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "fastica.csv")
            out = _capture() do
                _nongaussian_fastica(; data=csv, lags=2, method="fastica",
                                      output=outfile, format="csv")
            end
            @test isfile(outfile)
        end
    end

    @testset "_nongaussian_ml — student_t" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="student_t", format="table")
            end
            @test occursin("Non-Gaussian ML SVAR", out)
            @test occursin("distribution=student_t", out)
            @test occursin("Structural Impact Matrix", out)
            @test occursin("Log-likelihood", out)
            @test occursin("AIC", out)
        end
    end

    @testset "_nongaussian_ml — mixture_normal" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="mixture_normal", format="table")
            end
            @test occursin("mixture_normal", out)
        end
    end

    @testset "_nongaussian_ml — pml" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="pml", format="table")
            end
            @test occursin("pml", out)
        end
    end

    @testset "_nongaussian_ml — skew_normal" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="skew_normal", format="table")
            end
            @test occursin("skew_normal", out)
        end
    end

    @testset "_nongaussian_ml — skew_t (default dispatch)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="skew_t", format="table")
            end
            @test occursin("skew_t", out)
        end
    end

    @testset "_nongaussian_ml — dist_params and se output" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_ml(; data=csv, lags=2, distribution="student_t", format="table")
            end
            # Mock has non-empty dist_params and se
            @test occursin("Distribution parameters", out)
            @test occursin("Parameter Estimates with Standard Errors", out) ||
                  occursin("std_error", out)
        end
    end

    @testset "_nongaussian_heteroskedasticity — markov" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="markov",
                                                  regimes=2, format="table")
            end
            @test occursin("Heteroskedasticity SVAR", out)
            @test occursin("method=markov", out)
            @test occursin("Structural Impact Matrix", out)
        end
    end

    @testset "_nongaussian_heteroskedasticity — garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="garch",
                                                  format="table")
            end
            @test occursin("method=garch", out)
        end
    end

    @testset "_nongaussian_heteroskedasticity — smooth_transition requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="smooth_transition",
                                                  config="", format="table")
            end
        end
    end

    @testset "_nongaussian_heteroskedasticity — smooth_transition with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_ng_smooth_config(dir; transition_var="var2")
            out = _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="smooth_transition",
                                                  config=cfg, format="table")
            end
            @test occursin("smooth_transition", out)
        end
    end

    @testset "_nongaussian_heteroskedasticity — external requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws ErrorException _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="external",
                                                  config="", format="table")
            end
        end
    end

    @testset "_nongaussian_heteroskedasticity — external with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_ng_external_config(dir; regime_var="var3")
            out = _capture() do
                _nongaussian_heteroskedasticity(; data=csv, lags=2, method="external",
                                                  config=cfg, regimes=2, format="table")
            end
            @test occursin("Structural Impact Matrix", out)
        end
    end

    @testset "_nongaussian_normality" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_normality(; data=csv, lags=2, format="table")
            end
            @test occursin("Normality Test Suite", out)
            @test occursin("Normality Tests", out)
            # Mock has 2 out of 3 tests rejecting normality
            @test occursin("reject normality", out) || occursin("tests reject", out)
        end
    end

    @testset "_nongaussian_normality — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_normality(; data=csv, lags=nothing, format="table")
            end
            @test occursin("Normality Test Suite", out)
        end
    end

    @testset "_nongaussian_identifiability — all tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_identifiability(; data=csv, lags=2, test="all",
                                              method="fastica", format="table")
            end
            @test occursin("Identifiability Tests", out)
            @test occursin("Identification Strength", out) || occursin("Identifiability", out)
            # Should have multiple tests
            @test occursin("significant", out) || occursin("tests", out)
        end
    end

    @testset "_nongaussian_identifiability — individual tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for test_type in ["strength", "gaussianity", "independence", "overidentification"]
                out = _capture() do
                    _nongaussian_identifiability(; data=csv, lags=2, test=test_type,
                                                  method="fastica", format="table")
                end
                @test occursin("Identifiability", out)
            end
        end
    end

    @testset "_nongaussian_identifiability — all 6 ICA methods" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for ica_method in ["fastica", "infomax", "jade", "sobi", "dcov", "hsic"]
                out = _capture() do
                    _nongaussian_identifiability(; data=csv, lags=2, test="gaussianity",
                                                  method=ica_method, format="table")
                end
                @test occursin("Identifiability", out)
            end
        end
    end

    @testset "_nongaussian_identifiability — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _nongaussian_identifiability(; data=csv, lags=nothing, test="strength",
                                              format="table")
            end
            @test occursin("Identifiability", out)
        end
    end

end  # nongaussian.jl

# ═══════════════════════════════════════════════════════════════
# Edge cases and cross-cutting concerns
# ═══════════════════════════════════════════════════════════════

@testset "Edge Cases" begin

    @testset "nonexistent data file" begin
        @test_throws ErrorException _capture() do
            _var_estimate(; data="/nonexistent/path.csv", lags=2, format="table")
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
            out = _capture() do
                _var_estimate(; data=csv, lags=2, format="table")
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

    @testset "json output format for various commands" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for (fn, kwargs) in [
                (_var_estimate, (; data=csv, lags=2, format="json")),
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

    @testset "csv output format for various commands" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            outfile = joinpath(dir, "out.csv")
            for (fn, kwargs) in [
                (_var_stability, (; data=csv, lags=2, format="csv", output=outfile)),
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

end  # Edge Cases

end  # Command Handlers
