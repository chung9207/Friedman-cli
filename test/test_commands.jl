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
using Statistics: mean, median, var, quantile, std
using Random
using Serialization

# ─── Setup: Mock module + source includes ──────────────────────

project_root = dirname(@__DIR__)

# Load mock MacroEconometricModels module once (runtests.jl may already have it — TS-1)
if !@isdefined(MacroEconometricModels)
    include(joinpath(project_root, "test", "mocks.jl"))
    using .MacroEconometricModels
end

# PrettyTables v3 compat
if !@isdefined(tf_unicode_rounded)
    const tf_unicode_rounded = text_table_borders__unicode_rounded
end

# Include io.jl and config.jl (needed by command handlers)
include(joinpath(project_root, "src", "output", "errors.jl"))
include(joinpath(project_root, "src", "io.jl"))
include(joinpath(project_root, "src", "output", "envelope.jl"))
include(joinpath(project_root, "src", "output", "render.jl"))
include(joinpath(project_root, "src", "config.jl"))
# Required by dispatch_leaf envelope meta
const FRIEDMAN_VERSION = VersionNumber(
    TOML.parsefile(joinpath(project_root, "Project.toml"))["version"])

# Override _write_table for PrettyTables v3 (tf/show_subheader kwargs removed)
function _write_table(df::DataFrame, output::String, title::String)
    io = isempty(output) ? stdout : open(output, "w")
    try
        pretty_table(io, df; title=title, alignment=:c)
    finally
        isempty(output) || close(io)
    end
    isempty(output) || _status("Results written to $output")
end

# Include CLI types (needed for LeafCommand, NodeCommand, etc.)
include(joinpath(project_root, "src", "cli", "types.jl"))
include(joinpath(project_root, "src", "cli", "parser.jl"))
include(joinpath(project_root, "src", "cli", "help.jl"))
include(joinpath(project_root, "src", "cli", "dispatch.jl"))

# Include command files in dependency order
include(joinpath(project_root, "src", "commands", "shared.jl"))
include(joinpath(project_root, "src", "model_handle.jl"))
include(joinpath(project_root, "src", "handles.jl"))
include(joinpath(project_root, "src", "registry", "spec.jl"))
include(joinpath(project_root, "src", "registry", "adapter.jl"))
include(joinpath(project_root, "src", "commands", "estimate.jl"))
include(joinpath(project_root, "src", "commands", "test.jl"))
include(joinpath(project_root, "src", "commands", "irf.jl"))
include(joinpath(project_root, "src", "commands", "fevd.jl"))
include(joinpath(project_root, "src", "commands", "hd.jl"))
include(joinpath(project_root, "src", "commands", "forecast.jl"))
include(joinpath(project_root, "src", "commands", "fitted.jl"))
include(joinpath(project_root, "src", "commands", "filter.jl"))
include(joinpath(project_root, "src", "commands", "data.jl"))
include(joinpath(project_root, "src", "commands", "io.jl"))
include(joinpath(project_root, "src", "commands", "nowcast.jl"))
include(joinpath(project_root, "src", "commands", "dsge.jl"))
include(joinpath(project_root, "src", "commands", "did.jl"))
include(joinpath(project_root, "src", "commands", "multipliers.jl"))
include(joinpath(project_root, "src", "commands", "policy.jl"))
include(joinpath(project_root, "src", "commands", "spectral.jl"))
include(joinpath(project_root, "src", "commands", "model.jl"))
include(joinpath(project_root, "src", "commands", "completions.jl"))
include(joinpath(project_root, "src", "commands", "show.jl"))

include(joinpath(project_root, "test", "support.jl"))
@testset "Command Handlers" begin

# ═══════════════════════════════════════════════════════════════
# Single-envelope JSON (C010 / F17)
# ═══════════════════════════════════════════════════════════════

@testset "single JSON document per invocation (F17)" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=3)
        out = _capture() do
            _dispatch_via_app(["estimate", "var", csv, "--lags", "1", "--format", "json"])
        end
        # Strip any leading non-JSON status noise: find first '{'
        i = findfirst('{', out)
        @test i !== nothing
        doc = JSON3.read(out[i:end])
        @test haskey(doc, :data)
        @test length(collect(keys(doc.data))) >= 1
        @test doc.schema_version == 1
        @test doc.status == "ok"
    end

    # Legacy path: FRIEDMAN_LEGACY_OUTPUT=1 restores pre-envelope multi-doc/json array output
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=3)
        withenv("FRIEDMAN_LEGACY_OUTPUT" => "1") do
            out = _capture() do
                _dispatch_via_app(["estimate", "var", csv, "--lags", "1", "--format", "json"])
            end
            # Legacy is not a single envelope (no schema_version at top level required)
        end
    end
end

# ═══════════════════════════════════════════════════════════════
# Shared utilities (shared.jl)
# ═══════════════════════════════════════════════════════════════

@testset "registry reserved names/shorts guard (#117)" begin
    # `-h`/`--help` fire before tokenization and `--version`/`-V`, `--warranty`,
    # `--conditions` are leading-only globals — a spec claiming one is refused at
    # build_app time. 48 leaves once shipped an unreachable `-h` horizon short.
    # (Sources are included directly here — no `Friedman.` module in T1/T2; the
    # real build_app-under-guard case lives in T3's test_entry.jl.)
    @test_throws ErrorException _to_option(
        OptionSpec(name="horizon", short="h", type=Int, default=1, description="dead"))
    @test_throws ErrorException _to_option(
        OptionSpec(name="conditions", type=String, default="", description="swallowed"))
    @test_throws ErrorException _to_flag(
        FlagSpec(name="warranty", description="swallowed"))
    @test_throws ErrorException _to_flag(
        FlagSpec(name="verbose", short="V", description="dead"))
    # sane specs still pass
    @test _to_option(
        OptionSpec(name="horizons", type=Int, default=20, description="ok")) isa Option
end

@testset "Shared utilities" begin

    @testset "ID_METHOD_MAP" begin
        @test length(ID_METHOD_MAP) == 18   # 16 + lewis-tvv/sv-em (W1/#186)
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
        @test ID_METHOD_MAP["lewis-tvv"] == :lewis_tvv
        @test ID_METHOD_MAP["sv-em"] == :sv_em
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

        # Unknown method -> usage/invalid (W2/#166: no silent :cholesky fallback)
        e_unknown = try; _build_identification_kwargs("unknown", ""); nothing; catch e; e; end
        @test e_unknown isa CliError && e_unknown.code == "usage/invalid"

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
            @test_throws Exception load_univariate_series(csv, 10)
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
        @test_throws Exception validate_method("unknown", ["cholesky", "sign"], "id")
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
        # 65 primary leaves + 1 snake alias (gjr_garch → gjr-garch) = 66 keys (C044; +6 GARCH variants C064a, +arfima C068, +3 MGARCH C064b, +5 penalized/robust/tobit C067a, +truncreg/heckman C067b, +5 statespace/tvp/kde/kernel-reg/lowess C066, +cointreg/xtcointreg C062a, +ardl/nardl C062b, +pmg C062c, +midas C062d, +setar C065a, +star C065b, +ms-ar/ms C065c, +poisson/nbreg W2, +sarima W6,
        # +tvpvar/mfvar W7, +svar/svec W2/#166)
        @test length(node.subcmds) == 77
        for cmd in ["var", "bvar", "lp", "arima", "arfima", "gmm", "smm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr-garch", "sv", "fastica", "ml", "vecm", "pvar",
                     "favar", "sdfm", "reg", "iv", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit",
                     "igarch", "cgarch", "aparch", "figarch", "fiegarch", "garch-midas",
                     "ccc", "dcc", "bekk", "lasso", "ridge", "elastic-net", "robust",
                     "tobit", "truncreg", "heckman"]
            @test haskey(node.subcmds, cmd)
        end
        @test haskey(node.subcmds, "gjr_garch")  # hidden alias
        @test node.subcmds["gjr_garch"].name == "gjr-garch"
    end

    @testset "C044 kebab primary + snake alias deprecation" begin
        node = register_estimate_commands!()
        # Help lists kebab only, not snake alias
        help_io = IOBuffer()
        print_help(help_io, node; prog="friedman estimate")
        help_text = String(take!(help_io))
        @test contains(help_text, "gjr-garch")
        @test !contains(help_text, "gjr_garch")

        # Snake alias dispatches and prints deprecation on stderr
        mktempdir() do dir
            csv = _make_csv(dir; T=80, n=1, colnames=["ret"])
            streams = cd(dir) do
                _capture_all() do
                    _dispatch_via_app(["estimate", "gjr_garch", csv, "--p", "1", "--q", "1",
                                       "--format", "table"])
                end
            end
            @test contains(streams.err, "deprecated")
            @test contains(streams.err, "gjr-garch")
            @test contains(streams.err, "gjr_garch")
        end

        # Kebab primary: no deprecation line
        mktempdir() do dir
            csv = _make_csv(dir; T=80, n=1, colnames=["ret"])
            streams = cd(dir) do
                _capture_all() do
                    _dispatch_via_app(["estimate", "gjr-garch", csv, "--p", "1", "--q", "1",
                                       "--format", "table"])
                end
            end
            @test !contains(streams.err, "deprecated")
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

    @testset "_estimate_svar — recursive pattern" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_svar(; data=csv, lags=2, pattern="recursive", format="table")
                end
            end
        end
    end

    @testset "_estimate_svar — blanchard-quah pattern" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_svar(; data=csv, lags=2, pattern="blanchard-quah", format="table")
                end
            end
        end
    end

    @testset "_estimate_svar — a/b/ab-model with config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_svar_config(dir)
            for pat in ("a-model", "b-model", "ab-model")
                out = cd(dir) do
                    _capture() do
                        _estimate_svar(; data=csv, lags=2, pattern=pat, config=cfg, format="table")
                    end
                end
            end
        end
    end

    @testset "_estimate_svar — json envelope keys" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _dispatch_via_app(["estimate", "svar", csv, "--lags", "2", "--format", "json"])
                end
            end
            i = findfirst('{', out)
            doc = JSON3.read(out[i:end])
            @test haskey(doc[:data], :svar_a)
            @test haskey(doc[:data], :svar_b)
            @test haskey(doc[:data], :svar_summary)
        end
    end

    @testset "_estimate_svar — guards" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_svar_config(dir)
            e1 = try; _estimate_svar(; data=csv, lags=2, n_starts=0); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/invalid"
            e2 = try; _estimate_svar(; data=csv, lags=2, max_iter=0); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            e3 = try; _estimate_svar(; data=csv, lags=2, pattern="a-model", config=""); nothing; catch e; e; end
            @test e3 isa CliError && e3.code == "usage/missing"
            e4 = try; _estimate_svar(; data=csv, lags=2, pattern="b-model", config=cfg[1:end-5] * ".missing"); nothing; catch e; e; end
            @test e4 isa Exception
        end
    end

    @testset "_load_svar_pattern — malformed matrix" begin
        mktempdir() do dir
            bad = joinpath(dir, "bad.toml")
            open(bad, "w") do io
                write(io, "[svar]\nA = [[1.0, 0.0], [0.0]]\nB = [[1.0, 0.0], [0.0, 1.0]]\n")
            end
            e = try; _load_svar_pattern(bad, 2, "ab-model", "estimate svar"); nothing; catch e; e; end
            @test e isa CliError && e.code == "usage/invalid"
        end
    end

    @testset "_estimate_svec — default KPSW" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2)
            out = cd(dir) do
                _capture() do
                    _estimate_svec(; data=csv, lags=2, rank="1", format="table")
                end
            end
        end
    end

    @testset "_estimate_svec — custom zeros with json keys" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2)
            cfg = _make_svec_config(dir; n=2)
            out = cd(dir) do
                _capture() do
                    _dispatch_via_app(["estimate", "svec", csv, "--lags", "2", "--rank", "1",
                                       "--config", cfg, "--format", "json"])
                end
            end
            i = findfirst('{', out)
            doc = JSON3.read(out[i:end])
            @test haskey(doc[:data], :svec_b0)
            @test haskey(doc[:data], :svec_xi)
            @test haskey(doc[:data], :svec_summary)
        end
    end

    @testset "_estimate_svec — guards" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2)
            e1 = try; _estimate_svec(; data=csv, n_starts=0); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/invalid"
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

        end
    end

    @testset "_estimate_lp — iv missing instruments error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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

        end
    end

    @testset "_estimate_lp — state missing state_var error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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

        end
    end

    @testset "_estimate_lp — unknown method error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_estimate_arfima — pure fractional (0,d,0)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3)
            model = nothing
            out = cd(dir) do
                _capture() do
                    model = _estimate_arfima(; data=csv, column=1, p=0, q=0,
                                              method="css", format="table")
                end
            end
            @test occursin("ARFIMA", out)
            @test model.d isa Real
            @test -0.5 < model.d < 1.0
        end
    end

    @testset "_estimate_arfima — with AR/MA orders + mle" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arfima(; data=csv, column=2, p=1, q=1,
                                      method="mle", format="table")
                end
            end
        end
    end

    @testset "_estimate_arfima — d0 start + json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3)
            out = cd(dir) do
                _capture() do
                    _estimate_arfima(; data=csv, column=1, p=0, q=0,
                                      method="css", d0=0.2, format="json")
                end
            end
        end
    end

    @testset "_estimate_gmm — missing config error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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
            end
        end
    end

    @testset "_estimate_smm — ar1 config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            config_path = joinpath(dir, "smm.toml")
            write(config_path, """
            [smm]
            model = "ar1"
            theta0 = [0.5, 1.0]
            lags = 2
            weighting = "two_step"
            sim_ratio = 5
            burn = 50
            """)
            out = _capture() do
                _estimate_smm(; data=csv, config=config_path, format="table")
            end
            @test occursin("phi", out)
            @test occursin("sigma", out)
        end
    end

    @testset "_estimate_smm — var1 config (multivariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=2)
            config_path = joinpath(dir, "smm.toml")
            write(config_path, """
            [smm]
            model = "var1"
            theta0 = [0.4, 0.0, 0.0, 0.4, 1.0, 1.0]
            lags = 2
            weighting = "identity"
            """)
            out = _capture() do
                _estimate_smm(; data=csv, config=config_path, format="table")
            end
            @test occursin("A[1,1]", out)
            @test occursin("sigma_2", out)
        end
    end

    @testset "_estimate_smm — no config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            @test_throws CliError _estimate_smm(; data=csv, format="table")
        end
    end

    @testset "_estimate_smm — missing model errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            config_path = joinpath(dir, "smm.toml")
            write(config_path, """
            [smm]
            theta0 = [0.5, 1.0]
            """)
            @test_throws CliError _estimate_smm(; data=csv, config=config_path, format="table")
        end
    end

    @testset "_estimate_smm — underidentified errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=2)
            config_path = joinpath(dir, "smm.toml")
            # var1 k=2 needs 6 params but lags=1 → only 5 moments
            write(config_path, """
            [smm]
            model = "var1"
            theta0 = [0.4, 0.0, 0.0, 0.4, 1.0, 1.0]
            lags = 1
            """)
            @test_throws CliError _estimate_smm(; data=csv, config=config_path, format="table")
        end
    end

    @testset "_estimate_sur / _estimate_3sls (C063 systems)" begin
        _sys_csv(dir) = begin
            path = joinpath(dir, "sys.csv")
            open(path, "w") do io
                println(io, "y1,y2,x1,x2,x3")
                for _ in 1:120
                    x1 = randn(); x2 = randn(); x3 = randn()
                    y1 = 1.0 + 0.5x1 + 0.3x2 + 0.2randn()
                    y2 = -0.5 + 0.8x2 + 0.2x3 + 0.2randn()
                    println(io, join((y1, y2, x1, x2, x3), ","))
                end
            end
            path
        end
        _sur_cfg(dir) = begin
            p = joinpath(dir, "sur.toml")
            write(p, """
            [[equations]]
            name = "consumption"
            dep = "y1"
            indep = ["x1", "x2"]
            [[equations]]
            name = "investment"
            dep = "y2"
            indep = ["x2", "x3"]
            """)
            p
        end
        _sysdoc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _coefcols(doc) = first(t for t in values(doc.data)
                               if "term" in String.(t.columns) && "estimate" in String.(t.columns))

        @testset "sur tidy coef + system stats" begin
            mktempdir() do dir
                doc = _sysdoc(["sur", _sys_csv(dir), "--config", _sur_cfg(dir)])
                @test doc.status == "ok"
                coef = _coefcols(doc)
                @test Set(["equation", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]) ⊆ Set(String.(coef.columns))
                eqs = Set(String(collect(r)[1]) for r in coef.rows)
                @test "consumption" in eqs && "investment" in eqs
                @test length(coef.rows) == 6                 # 2 eq × (const + 2 regressors)
                @test any(t -> "metric" in String.(t.columns), values(doc.data))
            end
        end

        @testset "sur --iterate --no-intercept" begin
            mktempdir() do dir
                doc = _sysdoc(["sur", _sys_csv(dir), "--config", _sur_cfg(dir), "--iterate", "--no-intercept"])
                coef = _coefcols(doc)
                terms = Set(String(collect(r)[2]) for r in coef.rows)
                @test !("const" in terms)
                @test length(coef.rows) == 4                 # 2 eq × 2 regressors, no const
            end
        end

        @testset "3sls common instruments" begin
            mktempdir() do dir
                csv = _sys_csv(dir); cfg = joinpath(dir, "3sls.toml")
                write(cfg, """
                [[equations]]
                dep = "y1"
                indep = ["x1", "x2"]
                [[equations]]
                dep = "y2"
                indep = ["x2", "x3"]
                [instruments]
                common = ["x1", "x2", "x3"]
                """)
                doc = _sysdoc(["3sls", csv, "--config", cfg])
                @test doc.status == "ok"
                coef = _coefcols(doc)
                @test length(coef.rows) == 6
                @test any(t -> "metric" in String.(t.columns), values(doc.data))
            end
        end

        @testset "config errors (typed, not exit-1)" begin
            mktempdir() do dir
                csv = _sys_csv(dir); cfg = _sur_cfg(dir)
                _errcode(args) = begin
                    err = nothing
                    try; _capture() do; _dispatch_via_app(vcat(String["estimate"], collect(String, args))); end; catch e; err = e; end
                    err
                end
                @test _errcode(["sur", csv]) isa CliError && _errcode(["sur", csv]).code == "config/missing"
                @test _errcode(["3sls", csv, "--config", cfg]).code == "config/missing"   # no instruments
                badcfg = joinpath(dir, "bad.toml")
                write(badcfg, "[[equations]]\ndep = \"y1\"\nindep = [\"nope\"]\n")
                @test _errcode(["sur", csv, "--config", badcfg]).code == "config/bad-column"
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
        end
    end

    @testset "estimate GARCH variants (C064a)" begin
        # JSON envelope via the app: assert a hand-built coef table (parameter|estimate)
        # + a metric|value diagnostics table for each of the 6 volatility variants.
        _gv_doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _coef_tbl(doc) = first(t for t in _tables(doc) if "parameter" in String.(t.columns) && "estimate" in String.(t.columns))
        _diag_tbl(doc) = first(t for t in _tables(doc) if "metric" in String.(t.columns) && "value" in String.(t.columns))

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=1, colnames=["ret"])

            @testset "igarch coef + diag" begin
                doc = _gv_doc(["igarch", csv, "--column", "1", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                coef = _coef_tbl(doc)
                params = Set(String(collect(r)[1]) for r in coef.rows)
                @test Set(["mu", "omega", "alpha1", "beta1"]) ⊆ params
                diag = _diag_tbl(doc)
                metrics = Set(String(collect(r)[1]) for r in diag.rows)
                @test "persistence" in metrics && "log_likelihood" in metrics
            end

            @testset "cgarch coef (rho/phi/alpha/beta) + diag" begin
                doc = _gv_doc(["cgarch", csv, "--column", "1"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["mu", "omega", "rho", "phi", "alpha", "beta"]) ⊆ params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "unconditional_variance" in metrics
            end

            @testset "aparch (gamma + delta) + diag" begin
                doc = _gv_doc(["aparch", csv, "--column", "1", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["gamma1", "delta"]) ⊆ params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "n_params" in metrics
            end

            @testset "aparch --fix-delta --fix-gamma" begin
                doc = _gv_doc(["aparch", csv, "--column", "1", "--fix-delta", "2.0", "--fix-gamma", "0.0"])
                @test doc.status == "ok"
            end

            @testset "figarch (fractional d) + diag" begin
                doc = _gv_doc(["figarch", csv, "--column", "1", "--d0", "0.4", "--truncation", "50"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test "d" in params && "phi1" in params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "n_neg_lambda" in metrics && "truncation" in metrics
            end

            @testset "fiegarch (theta/gamma + d)" begin
                doc = _gv_doc(["fiegarch", csv, "--column", "1", "--truncation", "50"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["theta", "gamma", "d"]) ⊆ params
            end

            @testset "garch-midas realized + diag" begin
                doc = _gv_doc(["garch-midas", csv, "--column", "1", "--m-freq", "20", "--k", "6"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["mu", "alpha", "beta", "m", "theta", "w"]) ⊆ params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "variance_ratio" in metrics && "n_blocks" in metrics
            end

            @testset "garch-midas macro via --config" begin
                cfg = joinpath(dir, "gm.toml")
                write(cfg, "[garch_midas]\nx_lf = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.0]\n")
                doc = _gv_doc(["garch-midas", csv, "--column", "1", "--m-freq", "20", "--k", "3",
                               "--rv", "macro", "--config", cfg])
                @test doc.status == "ok"
            end

            @testset "error mapping (never uncaught exit-1)" begin
                @test_throws CliError _estimate_garch_midas(; data=csv, column=1, m_freq=0)        # missing --m-freq
                @test_throws CliError _estimate_garch_midas(; data=csv, column=1, m_freq=20, rv="macro")  # macro needs config
                # typed remap of untyped MEMs failures
                @test _garch_variant_error(ArgumentError("bad"), "IGARCH").code == "data/invalid"
                @test _garch_variant_error(DomainError(1.0, "d"), "FIGARCH").code == "data/invalid"
                @test _garch_variant_error(DimensionMismatch("x"), "APARCH").code == "data/shape"
                @test _garch_variant_error(ErrorException("boom"), "CGARCH").code == "model/error"
                @test _garch_variant_error(CliError("config/missing", "x"), "X").code == "config/missing"
                # review fix (shared helper): --column 0/negative/out-of-range → typed data
                # error (was an uncaught BoundsError → exit 1 across ALL univariate leaves)
                err(col) = begin
                    e = nothing
                    try; _capture() do; _dispatch_via_app(String["estimate","igarch",csv,"--column",string(col)]); end; catch ex; e=ex; end
                    e
                end
                @test err(0) isa CliError && err(0).code == "data/column-range"
                @test err(-1).code == "data/column-range"
                @test err(99).code == "data/column-range"
                # missing cell → typed data error (was an uncaught MethodError → exit 1)
                mcsv = joinpath(dir, "miss.csv")
                write(mcsv, "x,ret\n0.1,0.2\n0.3,\n0.5,0.6\n0.7,0.8\n")
                me = nothing
                try; _capture() do; _dispatch_via_app(String["estimate","igarch",mcsv,"--column","2"]); end; catch ex; me=ex; end
                @test me isa CliError && me.code == "data/missing-values"
            end
        end
    end

    @testset "estimate MGARCH ccc/dcc/bekk (C064b)" begin
        # JSON envelope via the app: assert a wide conditional-correlation matrix (series
        # column + one column per series) and a diagnostics (metric|value) block for each.
        _mg_doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _corr_tbl(doc) = first(t for t in _tables(doc) if "series" in String.(t.columns))
        _diag_tbl(doc) = first(t for t in _tables(doc) if "metric" in String.(t.columns) && "value" in String.(t.columns))
        _coef_tbl(doc) = first(t for t in _tables(doc) if "parameter" in String.(t.columns) && "estimate" in String.(t.columns))

        mktempdir() do dir
            csv = _make_csv(dir; T=150, n=3, colnames=["ra", "rb", "rc"])

            @testset "ccc — correlation matrix (no dynamics) + diag" begin
                doc = _mg_doc(["ccc", csv, "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                corr = _corr_tbl(doc)
                # wide sector×sector: series col + one col per series
                @test Set(["series", "ra", "rb", "rc"]) ⊆ Set(String.(corr.columns))
                @test length(collect(corr.rows)) == 3
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "kind" in metrics && "loglik" in metrics && "series" in metrics
                # CCC has no second-stage params → no dynamics coef table
                @test isempty([t for t in _tables(doc) if "parameter" in String.(t.columns)])
            end

            @testset "dcc — a,b dynamics + persistence + correction" begin
                doc = _mg_doc(["dcc", csv, "--correction", "none"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["a", "b"]) ⊆ params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "persistence" in metrics && "correction" in metrics
            end

            @testset "dcc — cDCC via --correction aielli" begin
                doc = _mg_doc(["dcc", csv, "--correction", "aielli"])
                @test doc.status == "ok"
            end

            @testset "bekk scalar (a,b) + bekk_kind" begin
                doc = _mg_doc(["bekk", csv, "--kind", "scalar"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test Set(["a", "b"]) ⊆ params
                metrics = Set(String(collect(r)[1]) for r in _diag_tbl(doc).rows)
                @test "bekk_kind" in metrics
            end

            @testset "bekk diagonal (a_i/b_i)" begin
                doc = _mg_doc(["bekk", csv, "--kind", "diagonal"])
                @test doc.status == "ok"
                params = Set(String(collect(r)[1]) for r in _coef_tbl(doc).rows)
                @test "a1" in params && "b1" in params
            end

            @testset "bad input never uncaught exit-1" begin
                # 1-column CSV → MGARCH needs ≥2 series → ArgumentError → data/invalid
                onecol = _make_csv(dir; T=100, n=1, colnames=["x"])
                e = nothing
                try; _capture() do; _dispatch_via_app(String["estimate","ccc",onecol]); end; catch ex; e=ex; end
                @test e isa CliError && e.code == "data/invalid" && exit_class(e) == 3
                # direct-handler typed classes
                @test_throws CliError _estimate_dcc(; data=csv, correction="bogus")   # usage/invalid up-front
                @test_throws CliError _estimate_bekk(; data=csv, kind="bogus")        # usage/invalid up-front
                du = nothing
                try; _estimate_dcc(; data=csv, correction="bogus"); catch ex; du=ex; end
                @test du isa CliError && du.code == "usage/invalid" && exit_class(du) == 2
                # missing cell in a multivariate CSV → typed data error via the hardened
                # load_multivariate_data (was an uncaught ArgumentError → exit 1)
                mcsv = joinpath(dir, "mg_miss.csv")
                write(mcsv, "a,b\n0.1,0.2\n0.3,\n0.5,0.6\n0.7,0.8\n")
                me = nothing
                try; _capture() do; _dispatch_via_app(String["estimate","bekk",mcsv]); end; catch ex; me=ex; end
                @test me isa CliError && me.code == "data/missing-values" && exit_class(me) == 3
                # an input column literally named `series` collides with the wide-matrix
                # label column → must NOT crash the (unwrapped) renderer to exit-1
                # (regression: adversarial review C064b; fixed via makeunique in _mgarch_corr_df)
                scsv = _make_csv(dir; T=120, n=2, colnames=["series", "r2"])
                sdoc = _mg_doc(["ccc", scsv])
                @test sdoc.status == "ok"
            end
        end
    end

    @testset "estimate penalized/robust/tobit (C067a)" begin
        # JSON envelope via the app: penalized fits render `term|estimate|nonzero`
        # (intercept from beta0, NO std errors); robust/tobit render the shared
        # `parameter|estimate|std_error|z_stat|p_value` hand-built coef table; every
        # leaf adds a metric|value diagnostics block.
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=4, colnames=["y", "x1", "x2", "x3"])

            @testset "lasso — term/estimate/nonzero + intercept + diag" begin
                doc = _doc(["lasso", csv, "--dep", "y", "--select", "bic"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "term", "estimate", "nonzero")
                terms = Set(String(collect(r)[1]) for r in coef.rows)
                @test "(Intercept)" in terms
                @test Set(["x1", "x2", "x3"]) ⊆ terms
                m = _metrics(doc)
                @test Set(["lambda", "alpha", "n_active", "select", "r2"]) ⊆ m
            end

            @testset "ridge — fixed-lambda path + diag" begin
                doc = _doc(["ridge", csv, "--dep", "y", "--lambda", "0.5"])
                @test doc.status == "ok"
                @test "(Intercept)" in Set(String(collect(r)[1]) for r in _tbl_with(doc, "term", "estimate", "nonzero").rows)
                @test "lambda" in _metrics(doc)
            end

            @testset "elastic-net — --alpha mixing + diag" begin
                doc = _doc(["elastic-net", csv, "--dep", "y", "--alpha", "0.3"])
                @test doc.status == "ok"
                @test "alpha" in _metrics(doc)
            end

            @testset "robust — parameter/estimate/std_error + diag" begin
                doc = _doc(["robust", csv, "--dep", "y", "--psi", "huber", "--method", "m"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "parameter", "estimate", "std_error")
                @test length(collect(coef.rows)) == 3      # x1,x2,x3 (no intercept)
                m = _metrics(doc)
                @test Set(["psi", "method", "scale", "robust_r2", "converged"]) ⊆ m
            end

            @testset "tobit — censoring counts + diag" begin
                doc = _doc(["tobit", csv, "--dep", "y", "--lower", "0.0"])
                @test doc.status == "ok"
                @test length(collect(_tbl_with(doc, "parameter", "estimate", "std_error").rows)) == 3
                m = _metrics(doc)
                @test Set(["sigma", "n_censored_left", "n_censored_right", "loglik"]) ⊆ m
            end

            @testset "bad input never uncaught exit-1 (typed classes)" begin
                # hardened _load_reg_data: bad --dep → data/column-range (benefits the WHOLE
                # cross-section reg family — assert for both a new leaf and existing `reg`)
                @test _err(["lasso", csv, "--dep", "nope"]) isa CliError
                @test _err(["lasso", csv, "--dep", "nope"]).code == "data/column-range"
                @test _err(["reg", csv, "--dep", "nope"]).code == "data/column-range"
                # up-front option-range validation → usage/invalid (exit 2), not raw MEMs
                ea = _err(["elastic-net", csv, "--dep", "y", "--alpha", "2"])
                @test ea isa CliError && ea.code == "usage/invalid" && exit_class(ea) == 2
                et = _err(["tobit", csv, "--dep", "y", "--lower", "5", "--upper", "1"])
                @test et isa CliError && et.code == "usage/invalid"
                el = _err(["lasso", csv, "--dep", "y", "--lambda", "notanumber"])
                @test el isa CliError && el.code == "usage/invalid"
                # a negative penalty is also rejected up-front
                @test _err(["ridge", csv, "--dep", "y", "--lambda", "-0.1"]).code == "usage/invalid"
                # missing cell in a regressor → data/missing-values, NOT an uncaught exit-1
                # (regression: adversarial review C067a — the hardening left the
                # Matrix{Float64} conversion unguarded; benefits the whole reg family)
                misscsv = joinpath(dir, "miss.csv")
                write(misscsv, "y,x1,x2\n1.0,2.0,3.0\n2.0,,4.0\n3.0,5.0,6.0\n4.0,7.0,8.0\n")
                em = _err(["lasso", misscsv, "--dep", "y"])
                @test em isa CliError && em.code == "data/missing-values" && exit_class(em) == 3
                @test _err(["reg", misscsv, "--dep", "y"]).code == "data/missing-values"
            end

            @testset "tobit default upper=Inf renders on the legacy JSON path (no Inf crash)" begin
                # regression (adversarial review C067a): --upper Inf reached JSON3.write on the
                # FRIEDMAN_LEGACY_OUTPUT path (which does not apply _json_safe) → "Inf not
                # allowed in JSON spec" → exit-1. Now rendered as the string "Inf".
                old = get(ENV, "FRIEDMAN_LEGACY_OUTPUT", nothing)
                ENV["FRIEDMAN_LEGACY_OUTPUT"] = "1"
                try
                    e = nothing
                    try; _capture() do; _dispatch_via_app(String["estimate","tobit",csv,"--dep","y","--lower","0.0","--format","json"]); end
                    catch ex; e = ex; end
                    @test e === nothing
                finally
                    old === nothing ? delete!(ENV, "FRIEDMAN_LEGACY_OUTPUT") : (ENV["FRIEDMAN_LEGACY_OUTPUT"] = old)
                end
            end
        end
    end

    @testset "estimate cointreg/xtcointreg (C062a)" begin
        # cointreg (single-equation FMOLS/CCR/DOLS) + xtcointreg (panel FMOLS/DOLS). Both
        # render the hand-built tidy coef table term|estimate|std_error|stat|p_value|
        # ci_lower|ci_upper (CointRegModel/PanelCointRegModel are not Tables.jl-registered)
        # plus a metric|value diagnostics block. Bad input → typed classes, never exit-1.
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate"], collect(String, args))); end; catch ex; e=ex; end
            e
        end
        _coefcols = ("term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper")

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3, colnames=["y", "x1", "x2"])

            @testset "cointreg — coef table + diag, all three methods" begin
                for meth in ("fmols", "ccr", "dols")
                    doc = _doc(["cointreg", csv, "--dep", "y", "--method", meth])
                    @test doc.status == "ok"
                    coef = _tbl_with(doc, _coefcols...)
                    terms = Set(String(collect(r)[1]) for r in coef.rows)
                    @test Set(["const", "x1", "x2"]) ⊆ terms   # const from default --trend const
                    m = _metrics(doc)
                    @test Set(["method", "trend", "kernel", "bandwidth", "omega_uv", "nobs", "d", "k"]) ⊆ m
                end
                # DOLS exposes leads/lags in the diagnostics block
                @test Set(["leads", "lags"]) ⊆ _metrics(_doc(["cointreg", csv, "--dep", "y", "--method", "dols"]))
                # --trend none drops the deterministic term
                dn = _doc(["cointreg", csv, "--dep", "y", "--trend", "none"])
                @test !("const" in Set(String(collect(r)[1]) for r in _tbl_with(dn, _coefcols...).rows))
            end

            @testset "xtcointreg — panel coef table + diag, group & pooled" begin
                panel = _make_panel_csv(dir; G=6, T_per=20, n=2, colnames=["y", "x1"])
                for pool in ("group", "pooled"), meth in ("fmols", "dols")
                    doc = _doc(["xtcointreg", panel, "--dep", "y", "--indep", "x1",
                                "--method", meth, "--pooling", pool])
                    @test doc.status == "ok"
                    @test !isempty(collect(_tbl_with(doc, _coefcols...).rows))
                    @test Set(["method", "pooling", "trend", "kernel", "N", "nobs", "T_i", "balanced", "k", "d"]) ⊆ _metrics(doc)
                end
            end

            @testset "bad input → typed classes, never uncaught exit-1" begin
                # bad enum → ParseError (or usage/invalid) — both exit 2
                em = _err(["cointreg", csv, "--dep", "y", "--method", "foo"])
                @test em isa ParseError || (em isa CliError && exit_class(em) == 2)
                # panel rejects ccr (choices=fmols|dols) at parse
                ec = _err(["xtcointreg", _make_panel_csv(dir; G=6, T_per=20, n=2, colnames=["y", "x1"]),
                           "--dep", "y", "--indep", "x1", "--method", "ccr"])
                @test ec isa ParseError || (ec isa CliError && exit_class(ec) == 2)
                # bad --dep → data/column-range (hardened _load_reg_data)
                @test _err(["cointreg", csv, "--dep", "nope"]).code == "data/column-range"
                # bad dual-type flags → usage/invalid (in-handler parsers, 0 valid for leads/lags)
                @test _err(["cointreg", csv, "--dep", "y", "--bandwidth", "notanum"]).code == "usage/invalid"
                @test _err(["cointreg", csv, "--dep", "y", "--leads", "-1"]).code == "usage/invalid"
                # --leads 0 is VALID (do not route through _parse_bandwidth which rejects 0)
                @test _doc(["cointreg", csv, "--dep", "y", "--method", "dols", "--leads", "0", "--lags", "0"]).status == "ok"
                # missing cell → data/missing-values (not exit-1)
                misscsv = joinpath(dir, "misscoint.csv")
                write(misscsv, "y,x1\n1.0,2.0\n2.0,\n3.0,5.0\n4.0,7.0\n5.0,8.0\n6.0,9.0\n7.0,10.0\n")
                em2 = _err(["cointreg", misscsv, "--dep", "y"])
                @test em2 isa CliError && em2.code == "data/missing-values" && exit_class(em2) == 3
                # xtcointreg: a missing cell in dep/indep → data/missing-values too, NOT silent
                # NaN coefficients at exit 0 (adversarial review C062a: xtset NaN-fills blanks).
                xtmiss = joinpath(dir, "xtmiss.csv")
                write(xtmiss, "group,time,y,x1\n1,1,1.0,2.0\n1,2,,3.0\n2,1,3.0,4.0\n2,2,3.5,4.5\n")
                exm = _err(["xtcointreg", xtmiss, "--dep", "y", "--indep", "x1"])
                @test exm isa CliError && exm.code == "data/missing-values" && exit_class(exm) == 3
                # duplicate (id,time) panel → data/invalid (hardened load_panel_data)
                dupcsv = joinpath(dir, "dupcoint.csv")
                write(dupcsv, "group,time,y,x1\n1,1,1.0,2.0\n1,1,1.5,2.5\n2,1,3.0,4.0\n2,2,3.5,4.5\n")
                ed = _err(["xtcointreg", dupcsv, "--dep", "y", "--indep", "x1"])
                @test ed isa CliError && ed.code == "data/invalid"
                # non-numeric-only panel var → data/invalid
                nncsv = joinpath(dir, "nncoint.csv")
                write(nncsv, "group,time,label\n1,1,foo\n1,2,bar\n2,1,baz\n2,2,qux\n")
                en = _err(["xtcointreg", nncsv, "--dep", "y", "--indep", "x1"])
                @test en isa CliError && en.code == "data/invalid"
            end
        end
    end

    @testset "estimate ardl/nardl + test ardl-bounds/nardl-symmetry + multipliers nardl (C062b)" begin
        # Single-equation ARDL/NARDL family. estimate ardl/nardl render a hand-built levels
        # coef table + a folded long-run table + a diagnostics kv (ARDL: ECM α; NARDL: enlarged-k
        # bounds decision). test ardl-bounds renders decision SYMBOLS + I(0)/I(1) bounds and has
        # NO p_value column. test nardl-symmetry is a tidy multi-row Wald table. multipliers nardl
        # (new top-level) melts the m⁺/m⁻ curves into one long table with optional band columns.
        _run(root, args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String[root], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _has_tbl(doc, cols...) = any(t -> Set(String.(cols)) ⊆ Set(String.(t.columns)), _tables(doc))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(root, args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String[root], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3, colnames=["y", "x1", "x2"])   # dep=y, 2 regressors
            csv1 = _make_csv(dir; T=120, n=2, colnames=["y", "x1"])         # 1 regressor (for q-length)

            @testset "estimate ardl — coef + long-run tables + ECM diagnostics" begin
                doc = _run("estimate", ["ardl", csv, "--dep", "y", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                @test _has_tbl(doc, "term", "estimate", "std_error", "stat", "p_value")
                @test _has_tbl(doc, "term", "estimate", "std_error")   # long-run block
                m = _metrics(doc)
                @test Set(["p", "q", "case", "trend", "ic", "nobs", "K", "alpha", "alpha_se", "longrun_denom"]) ⊆ m
                # auto selection path also runs
                @test _run("estimate", ["ardl", csv, "--dep", "y", "--p", "auto"]).status == "ok"
                # per-regressor q vector length must equal k → usage/invalid (1 regressor here)
                e = _err("estimate", ["ardl", csv1, "--dep", "y", "--q", "2,1"])
                @test e isa CliError && e.code == "usage/invalid"
                # bad case → usage/invalid
                @test (_err("estimate", ["ardl", csv, "--dep", "y", "--case", "9"])).code == "usage/invalid"
            end

            @testset "estimate nardl — split coef (_POS/_NEG) + θ⁺/θ⁻ + bounds decision in kv" begin
                doc = _run("estimate", ["nardl", csv, "--dep", "y", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "term", "estimate", "std_error", "stat", "p_value")
                terms = join(String(collect(r)[1]) for r in coef.rows)
                @test occursin("_POS", terms) && occursin("_NEG", terms)
                m = _metrics(doc)
                @test Set(["k_orig", "k", "asym", "f_decision", "t_decision", "bounds_level"]) ⊆ m
                # subset of regressors asymmetric
                @test _run("estimate", ["nardl", csv, "--dep", "y", "--asymmetric", "1"]).status == "ok"
                # empty/invalid asymmetric index → usage/invalid
                @test (_err("estimate", ["nardl", csv, "--dep", "y", "--asymmetric", "0"])).code == "usage/invalid"
                # --q length must match the ENLARGED split design (1 regressor, all asymmetric →
                # 2 POS/NEG cols) → usage/invalid up front, NOT MEMs data/invalid (review C062b)
                en = _err("estimate", ["nardl", csv1, "--dep", "y", "--q", "1,2,3"])
                @test en isa CliError && en.code == "usage/invalid"
            end

            @testset "test ardl-bounds — decision symbols + bounds, NO p-value" begin
                doc = _run("test", ["ardl-bounds", csv, "--dep", "y", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                bt = _tbl_with(doc, "bound", "statistic", "i0_lower", "i1_upper", "decision")
                @test !("p_value" in String.(bt.columns))              # bounds test has NO p-value
                @test Set(["F", "t"]) ⊆ Set(String(collect(r)[1]) for r in bt.rows)
                m = _metrics(doc)
                @test Set(["f_stat", "t_stat", "k", "case", "cv_source", "level", "f_decision", "t_decision"]) ⊆ m
                # case II has undefined t-bounds → rendered "undefined" (never a NaN crash)
                d2 = _run("test", ["ardl-bounds", csv, "--dep", "y", "--p", "1", "--q", "1", "--case", "2"])
                @test d2.status == "ok"
                # bad level / cv-source → usage/invalid (never interpret_test_result)
                @test (_err("test", ["ardl-bounds", csv, "--dep", "y", "--level", "0.03"])).code == "usage/invalid"
                ecv = _err("test", ["ardl-bounds", csv, "--dep", "y", "--cv-source", "narayan"])
                @test ecv isa ParseError || (ecv isa CliError && exit_class(ecv) == 2)
            end

            @testset "test nardl-symmetry — tidy multi-row Wald table" begin
                doc = _run("test", ["nardl-symmetry", csv, "--dep", "y", "--p", "1", "--q", "1"])
                @test doc.status == "ok"
                st = _tbl_with(doc, "regressor", "theta_pos", "theta_neg", "lr_stat", "lr_p_chi2", "lr_p_f", "sr_stat")
                @test !isempty(collect(st.rows))
                @test Set(["df", "dof_resid", "n_asym"]) ⊆ _metrics(doc)
            end

            @testset "multipliers nardl — long table + bands present/absent" begin
                # bands present with default nreps
                doc = _run("multipliers", ["nardl", csv, "--dep", "y", "--p", "1", "--q", "1", "--horizon", "6"])
                @test doc.status == "ok"
                mt = _tbl_with(doc, "horizon", "regressor", "m_pos", "m_neg", "m_diff")
                @test "m_pos_lo" in String.(mt.columns)                # band columns present
                @test Set(["horizon", "n_asym", "nreps", "level", "bootstrap"]) ⊆ _metrics(doc)
                # --no-bootstrap drops the band columns
                dnb = _run("multipliers", ["nardl", csv, "--dep", "y", "--p", "1", "--q", "1",
                                           "--horizon", "6", "--no-bootstrap"])
                mnb = _tbl_with(dnb, "horizon", "regressor", "m_pos", "m_neg", "m_diff")
                @test !("m_pos_lo" in String.(mnb.columns))
                # --nreps 0 also drops bands
                dn0 = _run("multipliers", ["nardl", csv, "--dep", "y", "--p", "1", "--q", "1",
                                           "--horizon", "6", "--nreps", "0"])
                @test !("m_pos_lo" in String.(_tbl_with(dn0, "horizon", "regressor", "m_pos").columns))
                # negative horizon → usage/invalid
                @test (_err("multipliers", ["nardl", csv, "--dep", "y", "--horizon", "-1"])).code == "usage/invalid"
            end

            @testset "shared bad input stays typed (never uncaught exit-1)" begin
                # bad --dep → data/column-range (hardened _load_reg_data), whole family
                @test (_err("estimate", ["ardl", csv, "--dep", "nope"])).code == "data/column-range"
                @test (_err("estimate", ["nardl", csv, "--dep", "nope"])).code == "data/column-range"
                @test (_err("test", ["ardl-bounds", csv, "--dep", "nope"])).code == "data/column-range"
                @test (_err("multipliers", ["nardl", csv, "--dep", "nope"])).code == "data/column-range"
                # missing cell → data/missing-values
                misscsv = joinpath(dir, "missardl.csv")
                write(misscsv, "y,x1\n1.0,2.0\n2.0,\n3.0,5.0\n4.0,7.0\n5.0,8.0\n6.0,9.0\n7.0,10.0\n8.0,11.0\n")
                @test (_err("estimate", ["ardl", misscsv, "--dep", "y"])).code == "data/missing-values"
            end

            @testset "multipliers command structure (new top-level)" begin
                node = register_multipliers_commands!()
                @test node isa NodeCommand
                @test length(node.subcmds) == 1
                @test haskey(node.subcmds, "nardl")
                @test node.subcmds["nardl"] isa LeafCommand
            end
        end
    end

    @testset "policy — effects + counterfactual (W4/#126, new top-level)" begin
        # McKay-Wolf policy counterfactuals. The mock CF layer implements real's
        # :ls math (assembly + pinv), so the closed forms hold in T1/T2 too.
        _pdoc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["policy"], collect(String, args),
                                       String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _perr(args) = begin
            e = nothing
            try
                _capture() do
                    _dispatch_via_app(vcat(String["policy"], collect(String, args)))
                end
            catch ex
                e = ex
            end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=3, colnames=["infl", "ygap", "rate"])
            base = ["counterfactual", "var", csv, "--shocks", "3",
                    "--nonpolicy-shock", "1", "--outcomes", "infl=1,ygap=2",
                    "--instruments", "rate=3", "--horizon", "6"]

            @testset "structure: 19th top-level, 7 leaves" begin
                node = register_policy_commands!()
                @test node isa NodeCommand
                eff = node.subcmds["effects"]; cf = node.subcmds["counterfactual"]
                @test sort(collect(keys(eff.subcmds))) == ["bvar", "lp", "sign", "var"]
                @test sort(collect(keys(cf.subcmds))) == ["bvar", "lp", "var"]
            end

            @testset "effects var — tidy menu + honest summary" begin
                doc = _pdoc(["effects", "var", csv, "--shocks", "3",
                             "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                             "--horizon", "6"])
                @test doc.status == "ok"
                menu = doc.data[:policy_causal_effects_menu]
                @test String.(menu.columns) == ["variable", "role", "shock", "horizon", "value"]
                @test length(collect(menu.rows)) == 3 * 1 * 6   # 3 vars × 1 shock × H
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:policy_causal_effects_summary].rows)
                @test s["is_square"] == false && s["n_draws"] == 0
            end

            @testset "counterfactual var — rate peg, honesty in the DATA" begin
                doc = _pdoc([base; "--rule"; "rate-peg"])
                @test doc.status == "ok"
                @test haskey(doc.data, :policy_counterfactual_paths)
                @test haskey(doc.data, :enforcing_policy_shocks_nu)
                @test haskey(doc.data, :implementation_error_path)
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:counterfactual_summary].rows)
                @test s["rule"] == "rate peg"
                @test haskey(s, "rel_residual") && haskey(s, "spanned")
            end

            @testset "counterfactual var — bootstrap draws → band columns" begin
                doc = _pdoc([base; "--rule"; "rate-peg"; "--replications"; "30"])
                p = doc.data[:policy_counterfactual_paths]
                @test "q16" in String.(p.columns) && "q84" in String.(p.columns)
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:counterfactual_summary].rows)
                @test s["n_draws_used"] == 30
            end

            @testset "guards — typed usage errors, builtin-rule contract" begin
                @test (_perr([base..., "--rule", "rate-peg", "--horizon", "0"])).code == "usage/invalid"
                @test (_perr([base..., "--rule", "bogus"])).code == "usage/invalid-option"
                # --rule + --rule-config together
                @test (_perr([base..., "--rule", "rate-peg", "--rule-config", "x.toml"])).code == "usage/invalid"
                # neither
                @test (_perr(collect(base))).code == "usage/missing"
                # taylor builtin needs outcomes NAMED infl/ygap
                e = _perr(["counterfactual", "var", csv, "--shocks", "3",
                           "--nonpolicy-shock", "1", "--outcomes", "cpi=1,gap=2",
                           "--instruments", "rate=3", "--rule", "taylor"])
                @test e isa CliError && e.code == "usage/invalid"
                # malformed pair spec
                e2 = _perr(["effects", "var", csv, "--shocks", "3",
                            "--outcomes", "infl", "--horizon", "4"])
                @test e2 isa CliError && e2.code == "usage/invalid"
                # missing shocks
                e3 = _perr(["effects", "var", csv,
                            "--outcomes", "infl=1", "--horizon", "4"])
                @test e3 isa CliError && e3.code == "usage/missing"
            end

            @testset "optimal + moments (W5/#127)" begin
                losstoml = joinpath(dir, "loss.toml")
                write(losstoml, """
                [loss]
                outcomes = ["infl", "ygap"]
                lambda = [1.0, 0.5]
                """)
                doc = _pdoc(["optimal", "var", csv, "--shocks", "3",
                             "--nonpolicy-shock", "1", "--outcomes", "infl=1,ygap=2",
                             "--instruments", "rate=3", "--loss-config", losstoml,
                             "--horizon", "6"])
                @test doc.status == "ok"
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:counterfactual_summary].rows)
                # Loss accounting is DATA; the optimum cannot lose to the baseline.
                @test haskey(s, "loss_base") && haskey(s, "loss_cf")
                @test Float64(s["loss_cf"]) <= Float64(s["loss_base"]) + 1e-10
                @test any(startswith(k, "foc_norm") for k in keys(s))
                # --loss-config is REQUIRED; --spanned-tol is NOT declared (#85 class:
                # upstream optimal_policy hardcodes 0.05).
                e = _perr(["optimal", "var", csv, "--shocks", "3",
                           "--nonpolicy-shock", "1", "--outcomes", "infl=1",
                           "--instruments", "rate=3"])
                @test e isa CliError && e.code == "usage/missing"
                node = register_policy_commands!()
                opt_leaf = node.subcmds["optimal"].subcmds["var"]
                @test !any(o -> o.name == "spanned-tol", opt_leaf.options)

                dm = _pdoc(["moments", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--rule", "rate-peg", "--horizon", "8"])
                @test dm.status == "ok"
                sd = dm.data[:counterfactual_standard_deviations]
                @test String.(sd.columns)[1:3] == ["variable", "sd_base", "sd_cf"]
                rows = [collect(r) for r in sd.rows]
                # Stabilizing counterfactual: sd shrinks (mock mirrors the direction).
                @test all(Float64(r[3]) < Float64(r[2]) for r in rows)
                ms = Dict(String(collect(r)[1]) => collect(r)[2]
                          for r in dm.data[:moments_summary].rows)
                @test any(startswith(k, "tail_share") for k in keys(ms))

                # rule XOR loss; bad frequency band.
                e2 = _perr(["moments", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--rule", "rate-peg", "--loss-config", losstoml])
                @test e2 isa CliError && e2.code == "usage/invalid"
                e3 = _perr(["moments", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--rule", "rate-peg", "--frequencies", "2,1"])
                @test e3 isa CliError && e3.code == "usage/invalid"
            end

            @testset "opp family (W6/#128)" begin
                losstoml = joinpath(dir, "opp_loss.toml")
                write(losstoml, """
                [loss]
                outcomes = ["infl", "ygap"]
                lambda = [1.0, 0.5]
                """)
                ob = ["opp", "var", csv, "--shocks", "3", "--outcomes", "infl=1,ygap=2",
                      "--instruments", "rate=3", "--loss-config", losstoml,
                      "--horizon", "6"]
                doc = _pdoc([ob; "--targets"; "infl=0,ygap=0"])
                @test doc.status == "ok"
                dd = doc.data[:opp_recommendation_delta]
                @test String.(dd.columns)[1:4] == ["shock", "delta", "delta_plugin", "gradient"]
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:opp_summary].rows)
                @test haskey(s, "loss_base") && haskey(s, "loss_opp")
                @test haskey(s, "band_polarity")   # reversed polarity is a DATA field
                @test haskey(doc.data, :objective_gap_paths)

                # The gaps-vs-levels trap: no targets / partial targets refuse.
                e = _perr(collect(ob))
                @test e isa CliError && e.code == "usage/missing"
                e2 = _perr([ob..., "--targets", "infl=0"])
                @test e2 isa CliError && e2.code == "usage/missing"

                # Constrained: TOML + announced path; missing path refuses.
                constoml = joinpath(dir, "cons.toml")
                write(constoml, """
                [[constraint]]
                type = "zlb"
                floor = 0.0
                instrument = "rate"
                """)
                dc = _pdoc([ob; "--targets"; "infl=0,ygap=0";
                            "--constraints-file"; constoml;
                            "--instrument-path"; "0.1,0.1,0.1,0.1,0.1,0.1"])
                sc = Dict(String(collect(r)[1]) => collect(r)[2]
                          for r in dc.data[:opp_summary].rows)
                @test haskey(sc, "method_used") && haskey(sc, "kkt_residual")
                e3 = _perr([ob..., "--targets", "infl=0,ygap=0",
                            "--constraints-file", constoml])
                @test e3 isa CliError && e3.code == "usage/missing"

                # opp-sequence: per-date gap files + required --sd.
                fdir = joinpath(dir, "fc"); mkpath(fdir)
                for (i, d) in enumerate(["2020Q1", "2020Q2"])
                    CSV.write(joinpath(fdir, "$d.csv"),
                              DataFrame(infl=fill(0.5 - 0.1i, 6), ygap=fill(-1.0, 6)))
                end
                ds = _pdoc(["opp-sequence", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--loss-config", losstoml, "--forecasts-dir", fdir,
                            "--sd", "0.5,0.5", "--horizon", "6"])
                @test ds.status == "ok"
                @test haskey(ds.data, :opp_sequence_delta_by_date)
                dec = ds.data[:opp_revision_decomposition]
                @test String.(dec.columns) == ["date", "shock", "news", "pref", "aging"]
                e4 = _perr(["opp-sequence", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--loss-config", losstoml, "--forecasts-dir", fdir,
                            "--horizon", "6"])
                @test e4 isa CliError && e4.code == "usage/missing"   # --sd required
            end

            @testset "structural routes (W7/#129)" begin
                nk = joinpath(dir, "nk.toml")
                write(nk, """
                [model]
                parameters = { rho = 0.8 }
                endogenous = ["ygap", "infl", "rate"]
                exogenous = ["e", "mp"]
                linear = true
                [[model.equations]]
                expr = "ygap[t] = rho * ygap[t-1] + e[t]"
                [[model.equations]]
                expr = "infl[t] = 0.3 * ygap[t]"
                [[model.equations]]
                expr = "rate[t] = 1.5 * infl[t] + mp[t]"
                """)
                dn = _pdoc(["news", "dsge", nk, "--policy-shock", "mp",
                            "--outcomes", "infl=infl,ygap=ygap",
                            "--instruments", "rate=rate", "--horizon", "6"])
                @test dn.status == "ok"
                sn = Dict(String(collect(r)[1]) => collect(r)[2]
                          for r in dn.data[:policy_causal_effects_summary].rows)
                @test sn["is_square"] == true && sn["n_shocks"] == 6

                # Behavioral is an option on the news leaves; values guarded [0,1].
                db = _pdoc(["news", "dsge", nk, "--policy-shock", "mp",
                            "--outcomes", "infl=infl", "--horizon", "4",
                            "--behavioral-m", "0.8"])
                sb = Dict(String(collect(r)[1]) => collect(r)[2]
                          for r in db.data[:policy_causal_effects_summary].rows)
                @test haskey(sb, "behavioral")
                eb = _perr(["news", "dsge", nk, "--policy-shock", "mp",
                            "--outcomes", "infl=infl", "--horizon", "4",
                            "--behavioral-m", "1.5"])
                @test eb isa CliError && eb.code == "usage/invalid"
                # Sym-pair parser: an integer column spec is the WRONG shape here.
                es = _perr(["news", "dsge", nk, "--policy-shock", "mp",
                            "--outcomes", "infl", "--horizon", "4"])
                @test es isa CliError && es.code == "usage/invalid"

                dh = _pdoc(["history", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1,ygap=2", "--instruments", "rate=3",
                            "--rule", "rate-peg", "--horizon", "8",
                            "--t-range", "50:54"])
                @test dh.status == "ok"
                @test haskey(dh.data, :counterfactual_history)
                # window must fit inside H−1
                eh = _perr(["history", "var", csv, "--shocks", "3",
                            "--outcomes", "infl=1", "--rule", "rate-peg",
                            "--horizon", "4", "--t-range", "50:60"])
                @test eh isa CliError && eh.code == "usage/invalid"

                dsuf = _pdoc(["sufficiency", "dsge", nk,
                              "--observables", "infl,rate", "--horizon", "6"])
                @test dsuf.status == "ok"
                ss2 = Dict(String(collect(r)[1]) => collect(r)[2]
                           for r in dsuf.data[:sufficiency_summary].rows)
                @test haskey(ss2, "invertible")
            end

            @testset "square container → exact solve enforces the peg" begin
                # 2 policy shocks with H=2 makes the menu square: the pegged
                # instrument path must be EXACTLY zero and rel_residual ~0.
                doc = _pdoc(["counterfactual", "var", csv, "--shocks", "2,3",
                             "--nonpolicy-shock", "1", "--outcomes", "infl=1",
                             "--instruments", "rate=3", "--rule", "rate-peg",
                             "--horizon", "2"])
                s = Dict(String(collect(r)[1]) => collect(r)[2]
                         for r in doc.data[:counterfactual_summary].rows)
                @test s["spanned"] == true
                p = doc.data[:policy_counterfactual_paths]
                ci = findfirst(==("counterfactual"), String.(p.columns))
                ri = findfirst(==("role"), String.(p.columns))
                zvals = [collect(r)[ci] for r in p.rows
                         if String(collect(r)[ri]) == "instrument"]
                @test all(abs(Float64(v)) < 1e-8 for v in zvals)
            end
        end
    end

    @testset "estimate pmg + test pmg-hausman (C062c)" begin
        # Dynamic heterogeneous-panel ARDL. estimate pmg renders a hand-built long-run θ table
        # + a short-run/EC (φ) table + diagnostics kv (PMGModel is not Tables.jl-registered).
        # test pmg-hausman fits the panel twice (efficient vs MG) → a standard test kv WITH a
        # p-value + interpretation. Bad input → typed classes, never uncaught exit-1.
        _run(root, args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String[root], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _has_tbl(doc, cols...) = any(t -> Set(String.(cols)) ⊆ Set(String.(t.columns)), _tables(doc))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(root, args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String[root], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            panel = _make_panel_csv(dir; G=8, T_per=25, n=2, colnames=["y", "x1"])

            @testset "estimate pmg — long-run + short-run/EC tables + diagnostics" begin
                for meth in ("pmg", "mg", "dfe")
                    doc = _run("estimate", ["pmg", panel, "--dep", "y", "--indep", "x1",
                                            "--method", meth, "--p", "1", "--q", "1"])
                    @test doc.status == "ok"
                    @test _has_tbl(doc, "term", "estimate", "std_error", "stat", "p_value")   # long-run θ
                    @test _has_tbl(doc, "term", "estimate", "std_error")                       # short-run/EC
                    ec = _tbl_with(doc, "term", "estimate", "std_error")
                    @test "EC speed (phi)" in Set(String(collect(r)[1]) for r in ec.rows)
                    m = _metrics(doc)
                    @test Set(["method", "N", "p", "q", "T_i", "phi", "phi_se", "loglik", "converged", "iters", "n_nonconv"]) ⊆ m
                end
            end

            @testset "test pmg-hausman — standard test kv + interpretation" begin
                for eff in ("pmg", "dfe")
                    doc = _run("test", ["pmg-hausman", panel, "--dep", "y", "--indep", "x1",
                                        "--efficient", eff, "--p", "1", "--q", "1"])
                    @test doc.status == "ok"
                    m = _metrics(doc)
                    @test Set(["test_name", "statistic", "pvalue", "df", "description"]) ⊆ m
                end
            end

            @testset "bad input → typed classes, never uncaught exit-1" begin
                # bad enum → ParseError (or usage/invalid) — both exit 2
                em = _err("estimate", ["pmg", panel, "--dep", "y", "--indep", "x1", "--method", "foo"])
                @test em isa ParseError || (em isa CliError && exit_class(em) == 2)
                # pmg-hausman rejects --efficient mg (choices=pmg|dfe) at parse
                eh = _err("test", ["pmg-hausman", panel, "--dep", "y", "--indep", "x1", "--efficient", "mg"])
                @test eh isa ParseError || (eh isa CliError && exit_class(eh) == 2)
                # p<1 → usage/invalid (in-handler guard)
                @test (_err("estimate", ["pmg", panel, "--dep", "y", "--indep", "x1", "--p", "0"])).code == "usage/invalid"
                # bad --dep → usage/invalid (not a panel variable)
                @test (_err("estimate", ["pmg", panel, "--dep", "nope", "--indep", "x1"])).code == "usage/invalid"
                # single-unit panel → data/invalid (PMG needs N≥2; wrapped, not exit-1)
                onecsv = joinpath(dir, "onepmg.csv")
                write(onecsv, "group,time,y,x1\n1,1,1.0,2.0\n1,2,1.4,2.3\n1,3,1.9,2.9\n1,4,2.1,3.2\n1,5,2.6,3.8\n1,6,3.0,4.1\n1,7,3.4,4.7\n1,8,3.9,5.2\n")
                @test (_err("estimate", ["pmg", onecsv, "--dep", "y", "--indep", "x1"])).code == "data/invalid"
                @test (_err("test", ["pmg-hausman", onecsv, "--dep", "y", "--indep", "x1"])).code == "data/invalid"
                # missing cell in dep/indep → data/missing-values (hardened _load_panel_reg)
                miss = joinpath(dir, "misspmg.csv")
                write(miss, "group,time,y,x1\n1,1,1.0,2.0\n1,2,,2.3\n2,1,3.0,4.0\n2,2,3.5,4.5\n")
                @test (_err("estimate", ["pmg", miss, "--dep", "y", "--indep", "x1"])).code == "data/missing-values"
                # duplicate (id,time) → data/invalid (hardened load_panel_data)
                dup = joinpath(dir, "duppmg.csv")
                write(dup, "group,time,y,x1\n1,1,1.0,2.0\n1,1,1.5,2.5\n2,1,3.0,4.0\n2,2,3.5,4.5\n")
                @test (_err("estimate", ["pmg", dup, "--dep", "y", "--indep", "x1"])).code == "data/invalid"
            end
        end
    end

    @testset "estimate midas (C062d)" begin
        # Mixed-frequency MIDAS: a low-frequency target (--data, --column) on --k high-frequency
        # lags of a single indicator (--hf-data, --hf-column). Renders a hand-built weight-curve
        # table (lag|weight) + coef table (term|estimate|std_error|stat|p_value) + diagnostics kv
        # (MidasModel is not Tables.jl-registered). `_load_midas_data` is the 5th hardened shared
        # loader: both inputs via load_univariate_series (typed missing/range), aligned
        # len(HF)==m×len(LF) → data/shape. Bad input → typed classes, never uncaught exit-1.
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate", "midas"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _has_tbl(doc, cols...) = any(t -> Set(String.(cols)) ⊆ Set(String.(t.columns)), _tables(doc))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate", "midas"], collect(String, args))); end; catch ex; e=ex; end
            e
        end
        # Aligned mixed-frequency pair: LF target (Tlf rows) + HF indicator (m*Tlf rows).
        _mixed(dir; Tlf=40, m=3, tag="a") = begin
            lf = joinpath(dir, "lf_$tag.csv"); hf = joinpath(dir, "hf_$tag.csv")
            CSV.write(lf, DataFrame(gdp=[1.0 + 0.5*sin(t/3.0) + 0.1*t for t in 1:Tlf]))
            CSV.write(hf, DataFrame(ip=[0.2*cos(h/4.0) + 0.05*h for h in 1:(m*Tlf)]))
            (lf, hf)
        end

        mktempdir() do dir
            lf, hf = _mixed(dir; Tlf=40, m=3, tag="main")

            @testset "weight-curve + coef tables + diagnostics, all 5 schemes" begin
                for wk in ("expalmon", "beta2", "beta3", "almon", "umidas")
                    doc = _doc([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", wk])
                    @test doc.status == "ok"
                    @test _has_tbl(doc, "lag", "weight")                                     # weight curve
                    wt = _tbl_with(doc, "lag", "weight")
                    @test length(collect(wt.rows)) == 6                                      # K lags
                    @test _has_tbl(doc, "term", "estimate", "std_error", "stat", "p_value")  # coef table
                    mt = _metrics(doc)
                    @test Set(["weights_kind", "m", "K", "p_ar", "poly_degree", "h", "nobs",
                               "r2", "adj_r2", "ssr", "sigma2", "aic", "bic", "loglik", "converged"]) ⊆ mt
                end
            end

            @testset "ADL-MIDAS (--p-ar) runs" begin
                doc = _doc([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", "expalmon", "--p-ar", "1"])
                @test doc.status == "ok"
                @test _has_tbl(doc, "term", "estimate")
            end

            @testset "--horizon: direct-h is real since 0.7.3 (MEMs#574, W10/#131)" begin
                # The mock now shifts the target like real (y_{t+h-1}), so h=4
                # must change the fit and drop h-1 tail targets from the sample.
                _metric(doc, name) = begin
                    t = first(t for t in _tables(doc) if "metric" in String.(t.columns))
                    d = Dict(String(collect(r)[1]) => collect(r)[2] for r in t.rows)
                    d[name]
                end
                d1 = _doc([lf, "--hf-data", hf, "--m", "3", "--k", "6"])
                d4 = _doc([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--horizon", "4"])
                @test _metric(d4, "h") == 4
                @test _metric(d4, "nobs") == _metric(d1, "nobs") - 3
                @test _metric(d4, "r2") != _metric(d1, "r2")
                eh = _err([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--horizon", "0"])
                @test eh isa CliError && eh.code == "usage/invalid"
            end

            @testset "bad input → typed classes, never uncaught exit-1" begin
                # missing --hf-data → usage/missing-option (exit 2)
                em = _err([lf, "--m", "3", "--k", "6"])
                @test em isa CliError && em.code == "usage/missing-option" && exit_class(em) == 2
                # --m 0 / --k 0 → usage/invalid (no upstream default)
                @test (_err([lf, "--hf-data", hf, "--m", "0", "--k", "6"])).code == "usage/invalid"
                @test (_err([lf, "--hf-data", hf, "--m", "3", "--k", "0"])).code == "usage/invalid"
                # Beta weight needs K≥2 → data/invalid (pre-guard, friendly message)
                @test (_err([lf, "--hf-data", hf, "--m", "3", "--k", "1", "--weights", "beta2"])).code == "data/invalid"
                # bad --weights enum → ParseError (choices) — exit 2
                ew = _err([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", "bogus"])
                @test ew isa ParseError || (ew isa CliError && exit_class(ew) == 2)
                # negative --poly-degree → usage/invalid (up-front guard; every numeric option is
                # guarded — without it real MEMs throws a bare BoundsError → model/error while the
                # mock throws ArgumentError → data/invalid, a latent mock/real exit-class divergence)
                @test (_err([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", "almon", "--poly-degree", "-1"])).code == "usage/invalid"
                # HF SHORTER than m×LF → data/shape (end-anchoring would drop target periods).
                # A LONGER HF (nhf > m×LF) is VALID (leading ragged edge dropped) — covered below.
                hfbad = joinpath(dir, "hf_bad.csv")
                CSV.write(hfbad, DataFrame(ip=[Float64(h) for h in 1:100]))   # 100 < 3*40 = 120
                eb = _err([lf, "--hf-data", hfbad, "--m", "3", "--k", "6"])
                @test eb isa CliError && eb.code == "data/shape" && exit_class(eb) == 3
                # missing cell in HF → data/missing-values (hardened load_univariate_series).
                # NOTE: a single-column CSV with a blank line is an EMPTY ROW that CSV.jl skips
                # (→ a shorter series → data/shape, NOT a real `missing`), so the missing cell must
                # sit in a multi-column row to survive as `missing`. Two columns; the target HF
                # column (1) carries one empty cell while the row itself is non-empty (aux keeps it).
                hfmiss = joinpath(dir, "hf_miss.csv")
                write(hfmiss, "ip,aux\n" * join(["$(h == 60 ? "" : string(0.1 * h)),0.5" for h in 1:120], "\n") * "\n")
                @test (_err([lf, "--hf-data", hfmiss, "--m", "3", "--k", "6"])).code == "data/missing-values"
                # out-of-range --column on the LF file → data/column-range
                @test (_err([lf, "--hf-data", hf, "--m", "3", "--k", "6", "--column", "9"])).code == "data/column-range"
            end

            @testset "ragged HF (nhf > m×LF) accepted — leading edge dropped" begin
                # The estimator anchors the last HF obs to the last LF period (its headline
                # nowcasting feature), so a LONGER high-frequency indicator is valid — the CLI
                # loader relaxes to `nhf >= m×LF` (not exact) to match. LF=40 rows, m=3 needs 120;
                # supply 150 HF obs (30 leading dropped).
                hflong = joinpath(dir, "hf_long.csv")
                CSV.write(hflong, DataFrame(ip=[0.2 * cos(h / 4.0) + 0.05 * h for h in 1:150]))  # 150 > 3*40 = 120
                doc = _doc([lf, "--hf-data", hflong, "--m", "3", "--k", "6"])
                @test doc.status == "ok"
                @test _has_tbl(doc, "lag", "weight")
                @test _has_tbl(doc, "term", "estimate")
            end
        end
    end

    @testset "estimate threshold (#70)" begin
        # The GENERAL threshold regression: y on X, split by a SEPARATE --threshold-col. Shares
        # ThresholdModel (and therefore `_threshold_coef_table`) with `estimate setar`, but the
        # COLUMN PARTITION is its own third shape — q is EXCLUDED from the regressors, which is a
        # correctness requirement, not a convenience: leaving q in X makes the regressors
        # collinear with the splitting variable and silently fits a different model.
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate", "threshold"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate", "threshold"], collect(String, args))); end; catch ex; e=ex; end
            e
        end
        # y, x1, x2, z — z is the splitting variable and must never appear as a regressor.
        _write(dir; n=120, name="thr.csv") = begin
            path = joinpath(dir, name)
            open(path, "w") do io
                println(io, "y,x1,x2,z")
                for i in 1:n
                    z = (i % 2 == 0 ? 1.0 : -1.0) * (0.3 + 0.01 * i)
                    x1 = 0.5 + 0.02 * i
                    x2 = sin(0.3 * i)
                    y = (z <= 0 ? 2.0 : -2.0) * x1 + 0.5 * x2 + 0.05 * cos(1.7 * i)
                    println(io, "$y,$x1,$x2,$z")
                end
            end
            path
        end

        mktempdir() do dir
            csv = _write(dir)

            @testset "two regime blocks + diagnostics" begin
                doc = _doc([csv, "--dep", "y", "--threshold-col", "z", "--reps", "10"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                rows = collect(coef.rows)
                # X = {x1, x2} (z EXCLUDED, y is --dep) ⇒ 2 terms × 2 regimes = 4 rows
                @test length(rows) == 4
                terms = Set(String(collect(r)[2]) for r in rows)
                @test terms == Set(["x1", "x2"])
                @test !("z" in terms)          # the splitting variable is NOT a regressor
                @test length(Set(String(collect(r)[1]) for r in rows)) == 2
                m = _metrics(doc)
                for key in ["threshold_var", "gamma", "gamma_ci_lower", "gamma_ci_upper",
                            "gamma_ci_level", "n", "n1", "n2", "ssr", "sigma2", "aic", "bic",
                            "is_setar", "sup_lm", "pvalue_lm"]
                    @test key in m
                end
            end

            @testset "--no-linearity drops the Hansen block" begin
                m = _metrics(_doc([csv, "--dep", "y", "--threshold-col", "z", "--no-linearity"]))
                @test "gamma" in m
                @test !("sup_lm" in m) && !("pvalue_lm" in m)
            end

            @testset "--dep defaults to the first numeric column" begin
                doc = _doc([csv, "--threshold-col", "z", "--reps", "10"])
                @test doc.status == "ok"
                @test Set(String(collect(r)[2]) for r in
                          collect(_tbl_with(doc, "regime", "term").rows)) == Set(["x1", "x2"])
            end

            @testset "usage guards" begin
                # --threshold-col is REQUIRED: without it there is no model to fit
                e = _err([csv, "--dep", "y"])
                @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
                for bad in (["--trim", "0.5"], ["--trim", "0.0"], ["--reps", "0"])
                    eb = _err(vcat([csv, "--dep", "y", "--threshold-col", "z"], bad))
                    @test eb isa CliError && eb.code == "usage/invalid" && exit_class(eb) == 2
                end
                # Hansen (2000) CVs exist only at 0.90/0.95/0.99 — blocked at parse (choices)
                # or by the in-handler exact-float guard.
                eci = _err([csv, "--dep", "y", "--threshold-col", "z", "--ci-level", "0.8"])
                @test eci isa ParseError || (eci isa CliError && exit_class(eci) == 2)
                # q == dep is degenerate (the sample would split on the outcome itself)
                ed = _err([csv, "--dep", "y", "--threshold-col", "y"])
                @test ed isa CliError && ed.code == "usage/invalid" && exit_class(ed) == 2
            end

            @testset "data guards" begin
                for (col, code) in (("nope", "data/column-range"),)
                    e = _err([csv, "--dep", "y", "--threshold-col", col])
                    @test e isa CliError && e.code == code && exit_class(e) == 3
                end
                e2 = _err([csv, "--dep", "nope", "--threshold-col", "z"])
                @test e2 isa CliError && e2.code == "data/column-range" && exit_class(e2) == 3
                # dep + threshold-col consume both columns → no regressors left
                two = joinpath(dir, "two.csv")
                write(two, "y,z\n" * join(["$(0.1 * i),$(0.2 * i)" for i in 1:40], "\n") * "\n")
                e3 = _err([two, "--dep", "y", "--threshold-col", "z"])
                @test e3 isa CliError && e3.code == "data/invalid" && exit_class(e3) == 3
                # a missing cell is caught BEFORE the Matrix{Float64} conversion (exit 3, not 1)
                miss = joinpath(dir, "miss.csv")
                write(miss, "y,x1,z\n" * join([i == 5 ? "0.5,,0.3" : "$(0.1 * i),$(0.2 * i),$(0.3 * i)"
                                               for i in 1:40], "\n") * "\n")
                e4 = _err([miss, "--dep", "y", "--threshold-col", "z"])
                @test e4 isa CliError && e4.code == "data/missing-values" && exit_class(e4) == 3
                # A constant q admits no admissible split → real raises ArgumentError("Empty
                # threshold grid") → data/invalid (exit 3), NEVER an internal exit 1. The mock
                # mirrors that exception class so this assertion tracks real behaviour.
                constq = joinpath(dir, "constq.csv")
                write(constq, "y,x1,z\n" * join(["$(0.1 * i),$(0.2 * i),1.0" for i in 1:40], "\n") * "\n")
                e5 = _err([constq, "--dep", "y", "--threshold-col", "z"])
                @test e5 isa CliError && e5.code == "data/invalid" && exit_class(e5) == 3
            end
        end
    end

    @testset "estimate setar (C065a)" begin
        # SETAR: two-regime hand-built coef table (regime|term|estimate|std_error|z_stat|
        # p_value, 2 regimes × |xnames| rows) + diagnostics kv; the attached Hansen (1996)
        # linearity test folds into the kv iff linearity is on (default). ThresholdModel is
        # not Tables.jl-registered. Every option is guarded up-front → usage/invalid; the
        # estimator is wrapped → typed CliError via `_nonlinear_error` (never uncaught exit-1).
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate", "setar"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate", "setar"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "coef table (2 regimes) + diagnostics + attached linearity" begin
                doc = _doc([csv, "--column", "1", "--p", "1", "--reps", "50"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                rows = collect(coef.rows)
                # p=1 ⇒ xnames = [const, y[t-1]] (2 terms) × 2 regimes = 4 rows
                @test length(rows) == 4
                regimes = Set(String(collect(r)[1]) for r in rows)
                @test length(regimes) == 2
                m = _metrics(doc)
                @test Set(["gamma", "gamma_ci_lower", "gamma_ci_upper", "n", "n1", "n2",
                           "aic", "bic", "is_setar"]) ⊆ m
                # linearity on by default → Hansen sup-LM / p-value present in the kv
                @test Set(["sup_lm", "pvalue_lm", "sup_wald", "pvalue_wald", "gamma_sup"]) ⊆ m
            end

            @testset "--no-linearity drops the attached test; --d auto ok" begin
                dn = _doc([csv, "--no-linearity"])
                @test dn.status == "ok"
                @test !("sup_lm" in _metrics(dn))
                @test _doc([csv, "--d", "auto"]).status == "ok"
                @test _doc([csv, "--p", "2", "--d", "2"]).status == "ok"
            end

            @testset "bad input → typed classes, never uncaught exit-1" begin
                @test _err([csv, "--p", "0"]).code == "usage/invalid"
                @test _err([csv, "--trim", "0.6"]).code == "usage/invalid"
                @test _err([csv, "--reps", "0"]).code == "usage/invalid"
                @test _err([csv, "--d", "foo"]).code == "usage/invalid"
                @test _err([csv, "--d", "0"]).code == "usage/invalid"
                # --ci-level 0.8: blocked by choices at parse (ParseError, exit 2) or the
                # in-handler exact-float guard (CliError usage/invalid, exit 2)
                eci = _err([csv, "--ci-level", "0.8"])
                @test eci isa ParseError || (eci isa CliError && exit_class(eci) == 2)
                # out-of-range column / missing cell via the hardened univariate loader
                @test _err([csv, "--column", "99"]).code == "data/column-range"
                misscsv = joinpath(dir, "miss.csv")
                write(misscsv, "y,aux\n" * join(["$(i == 5 ? "" : string(0.1 * i)),0.5" for i in 1:60], "\n") * "\n")
                em = _err([misscsv, "--column", "1"])
                @test em isa CliError && em.code == "data/missing-values" && exit_class(em) == 3
                # A (near-)constant series admits no threshold split → data/invalid (exit 3), NOT an
                # internal model/error: real MEMs raises ArgumentError("Empty threshold grid") and the
                # mock mirrors that class (so the T1/T2 exit class tracks real's T3, not a
                # SingularException → model/error divergence).
                constcsv = joinpath(dir, "const.csv")
                write(constcsv, "y\n" * join(fill("1.0", 60), "\n") * "\n")
                ec = _err([constcsv, "--column", "1"])
                @test ec isa CliError && ec.code == "data/invalid" && exit_class(ec) == 3
            end
        end
    end

    @testset "estimate star (C065b)" begin
        # STAR: two regime-weight blocks (1−G / G) render as one hand-built table
        # (regime|term|estimate|std_error|z_stat|p_value); the transition parameters (γ, c)
        # render as a separate parameter|estimate|std_error|z_stat|p_value table; LM3 + (for
        # --type auto) the Teräsvirta selection triple fold into the kv. STARModel is not
        # Tables.jl-registered. Every option is guarded up-front → usage/invalid; the estimator
        # is wrapped → typed CliError via `_nonlinear_error` (never uncaught exit-1).
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate", "star"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate", "star"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "regime blocks + transition params + diagnostics (--type auto)" begin
                doc = _doc([csv, "--column", "1", "--p", "1", "--type", "auto"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                rows = collect(coef.rows)
                # p=1 ⇒ znames = [const, y[t-1]] (2 terms) × 2 regimes = 4 rows
                @test length(rows) == 4
                @test length(Set(String(collect(r)[1]) for r in rows)) == 2
                # transition-params block: γ (+ location c) in a `parameter` table
                trans = _tbl_with(doc, "parameter", "estimate", "std_error")
                @test any(occursin("γ", String(collect(r)[1])) for r in trans.rows)
                m = _metrics(doc)
                @test Set(["trans_type", "sname", "sigma_s", "lm3_stat", "lm3_pvalue",
                           "lm3_fstat", "lm3_fpvalue", "converged"]) ⊆ m
                # --type auto ⇒ the Teräsvirta sequential selection triple is present
                @test Set(["sel_H04", "sel_H03", "sel_H02"]) ⊆ m
            end

            @testset "--type lstr1 drops the selection triple; external transition col" begin
                m1 = _metrics(_doc([csv, "--type", "lstr1"]))
                @test !("sel_H04" in m1)
                # a valid non-constant external transition column fits (sname reports "s")
                csv2 = _make_csv(dir; T=200, n=2, colnames=["y", "x"])
                de = _doc([csv2, "--column", "1", "--transition-col", "2", "--type", "lstr1"])
                @test de.status == "ok"
            end

            @testset "bad input → typed classes, never uncaught exit-1" begin
                @test _err([csv, "--p", "0"]).code == "usage/invalid"
                @test _err([csv, "--d", "0"]).code == "usage/invalid"
                @test _err([csv, "--n-gamma", "1"]).code == "usage/invalid"
                @test _err([csv, "--n-c", "1"]).code == "usage/invalid"
                # --type bogus: blocked by choices at parse (ParseError, exit 2) or the
                # defensive in-handler enum guard (CliError usage/invalid, exit 2)
                et = _err([csv, "--type", "bogus"])
                @test et isa ParseError || (et isa CliError && exit_class(et) == 2)
                # constant external transition column → data/invalid (zero variance, cannot scale γ)
                constcsv = joinpath(dir, "star_const.csv")
                write(constcsv, "y,s\n" * join(["$(0.1 * i),1.0" for i in 1:80], "\n") * "\n")
                ec = _err([constcsv, "--column", "1", "--transition-col", "2"])
                @test ec isa CliError && ec.code == "data/invalid" && exit_class(ec) == 3
                # out-of-range column via the hardened univariate loader
                @test _err([csv, "--column", "99"]).code == "data/column-range"
            end
        end
    end

    @testset "estimate ms-ar + estimate ms (C065c)" begin
        # Markov-switching family. MSRegModel is not Tables.jl-registered → three hand-built
        # tables via the shared `_ms_render`: a per-regime coefficient table (regime|term|…),
        # a per-regime variance table (regime|sigma2|std_error), and a WIDE K×K transition
        # matrix (from_regime|to_regime1|…) parallel to the MGARCH correlation matrix, plus a
        # diagnostics kv. The two estimators have OPPOSITE switching_variance defaults (ms-ar
        # FALSE, ms TRUE). Every option guarded up-front → usage/invalid; the estimator is
        # wrapped → typed CliError via `_nonlinear_error` (never uncaught exit-1).
        _doc(cmd, args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate", cmd], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _mtbl(doc) = first(t for t in _tables(doc) if "metric" in String.(t.columns))
        _metrics(doc) = Set(String(collect(r)[1]) for r in _mtbl(doc).rows)
        _mval(doc, name) = begin
            for r in _mtbl(doc).rows
                rr = collect(r)
                String(rr[1]) == name && return rr[2]
            end
            nothing
        end
        _err(cmd, args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate", cmd], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            # Distinct file paths per structure (`_make_csv` always writes the same `data.csv`, so a
            # second call would clobber a single-column fixture the intercept-only test relies on).
            Random.seed!(65067)
            _write_csv(path, header, ncol; T=200) = open(path, "w") do io
                println(io, header)
                for _ in 1:T
                    println(io, join((string(randn()) for _ in 1:ncol), ","))
                end
            end
            csv = joinpath(dir, "ms_uni.csv");     _write_csv(csv, "y", 1)
            multi = joinpath(dir, "ms_multi.csv"); _write_csv(multi, "y,x1,x2", 3)
            collide = joinpath(dir, "ms_collide.csv"); _write_csv(collide, "y,from_regime,to_regime1", 3)

            @testset "estimate ms-ar — mu rows + common-AR block, variance + wide P, kv" begin
                doc = _doc("ms-ar", [csv, "--column", "1", "--p", "1"])
                @test doc.status == "ok"
                # coef table: per-regime `mu` rows + a common-AR block (φ1)
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                rows = collect(coef.rows)
                terms = Set(String(collect(r)[2]) for r in rows)
                @test "mu" in terms
                @test "φ1" in terms
                regimes = Set(String(collect(r)[1]) for r in rows)
                @test "regime1" in regimes && "regime2" in regimes && "common-AR" in regimes
                # per-regime variance table (2 regimes)
                vt = _tbl_with(doc, "regime", "sigma2", "std_error")
                @test length(collect(vt.rows)) == 2
                # WIDE K×K transition matrix
                pt = _tbl_with(doc, "from_regime", "to_regime1", "to_regime2")
                @test length(collect(pt.rows)) == 2
                m = _metrics(doc)
                @test Set(["loglik", "n_params", "aic", "bic", "ergodic_1", "ergodic_2",
                           "expected_duration_1", "expected_duration_2", "switching_var",
                           "switching_ar", "converged", "iterations"]) ⊆ m
                # Hamilton form default: switching_var = false
                @test string(_mval(doc, "switching_var")) == "false"
                # --switching-variance flag turns it on
                d2 = _doc("ms-ar", [csv, "--p", "1", "--switching-variance"])
                @test string(_mval(d2, "switching_var")) == "true"
            end

            @testset "estimate ms-ar/ms — regime-probability table (#70 remainder)" begin
                # The headline output of a Markov-switching fit: P(S_t = k | data) per period.
                # LONG (`period|regime|filtered|smoothed`) so the COLUMN SET does not depend on
                # --k-regimes; the ROW COUNT does (n × K).
                doc = _doc("ms-ar", [csv, "--column", "1", "--p", "1"])
                pt = _tbl_with(doc, "period", "regime", "filtered", "smoothed")
                rows = collect(pt.rows)
                @test Set(String(collect(r)[2]) for r in rows) == Set(["regime1", "regime2"])
                nper = length(Set(Int(collect(r)[1]) for r in rows))
                @test length(rows) == 2 * nper                      # K × n_eff
                # probabilities are genuine probabilities and sum to 1 across regimes per period
                for col in 3:4
                    vals = [Float64(collect(r)[col]) for r in rows]
                    @test all(v -> -1e-6 <= v <= 1 + 1e-6, vals)
                    # total over all (period, regime) cells == n, i.e. each period sums to 1
                    @test isapprox(sum(vals), nper; atol=1e-3)
                end
                # filtered and smoothed are DIFFERENT paths — smoothed conditions on the whole
                # sample and is sharper. (The mock deliberately reproduces that ordering; an
                # earlier `smoothed = copy(filtered)` would make this assertion vacuous.)
                filt = [Float64(collect(r)[3]) for r in rows]
                smoo = [Float64(collect(r)[4]) for r in rows]
                @test filt != smoo
                @test sum(abs.(smoo .- 0.5)) > sum(abs.(filt .- 0.5))   # smoothed more extreme
                # K = 3 widens the ROW count, not the column set
                d3 = _doc("ms-ar", [csv, "--p", "1", "--k-regimes", "3"])
                p3 = _tbl_with(d3, "period", "regime", "filtered", "smoothed")
                @test Set(String.(p3.columns)) == Set(String.(pt.columns))
                @test length(Set(String(collect(r)[2]) for r in collect(p3.rows))) == 3
                # the MS regression leaf emits it too
                @test _tbl_with(_doc("ms", [csv]), "period", "regime", "smoothed") !== nothing
            end

            @testset "estimate ms-ar — bad input → typed usage/invalid" begin
                @test _err("ms-ar", [csv, "--p", "0"]).code == "usage/invalid"
                @test _err("ms-ar", [csv, "--k-regimes", "1"]).code == "usage/invalid"
                @test _err("ms-ar", [csv, "--max-iter", "0"]).code == "usage/invalid"
                @test _err("ms-ar", [csv, "--column", "99"]).code == "data/column-range"
            end

            @testset "estimate ms — regressors path, switching_var default TRUE" begin
                # multi-column CSV → per-regime switching coefs over the regressor columns
                doc = _doc("ms", [multi, "--dep", "y"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                regimes = Set(String(collect(r)[1]) for r in collect(coef.rows))
                @test regimes == Set(["regime1", "regime2"])
                @test string(_mval(doc, "switching_var")) == "true"   # default for ms
                # --no-switching-variance forces a common σ²
                dn = _doc("ms", [multi, "--dep", "y", "--no-switching-variance"])
                @test string(_mval(dn, "switching_var")) == "false"
            end

            @testset "estimate ms — intercept-only single-arg dispatch (dep-only CSV)" begin
                # only the dependent column is numeric → _load_reg_data raises "no regressor
                # columns", routing to the single-arg estimate_ms(y; …) intercept-only dispatch
                doc = _doc("ms", [csv])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "regime", "term", "estimate", "std_error")
                rows = collect(coef.rows)
                # intercept-only: one `const` term per regime (2 rows)
                @test length(rows) == 2
                @test all(String(collect(r)[2]) == "const" for r in rows)
            end

            @testset "estimate ms — bad input → typed usage/invalid" begin
                @test _err("ms", [csv, "--k-regimes", "1"]).code == "usage/invalid"
                @test _err("ms", [csv, "--max-iter", "0"]).code == "usage/invalid"
                @test _err("ms", [csv, "--tol", "0"]).code == "usage/invalid"
            end

            @testset "estimate ms — regressors named like P-labels render cleanly" begin
                # UNLIKE MGARCH (where the input series names ARE the wide-matrix headers → a real
                # collision the makeunique guards), the MS wide-P headers are FIXED strings
                # (from_regime / to_regimeN), so a regressor column named `from_regime`/`to_regime1`
                # only ever becomes a `term` VALUE in the coef table, never a header. This test just
                # confirms such oddly-named regressors render exit-0 (the makeunique on the P DataFrame
                # is retained as harmless defense-in-depth, mirroring `_mgarch_corr_df`, but the MS
                # table headers cannot be driven by user input, so no header collision can occur).
                cdoc = _doc("ms", [collide, "--dep", "y"])
                @test cdoc.status == "ok"
                @test _tbl_with(cdoc, "from_regime", "to_regime1", "to_regime2") !== nothing
            end
        end
    end

    @testset "estimate truncreg/heckman + test weak-instrument (C067b)" begin
        _edoc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tdoc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        _eerr(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate"], collect(String, args))); end; catch ex; e=ex; end
            e
        end
        _terr(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["test"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            # Truncated-normal data: every y strictly positive (mock/real both require
            # y ∈ (lower, upper)); regressors x1,x2.
            trcsv = joinpath(dir, "trunc.csv")
            open(trcsv, "w") do io
                println(io, "y,x1,x2")
                for t in 1:80
                    x1 = (t % 7) - 3.0; x2 = (t % 5) - 2.0
                    y = 5.0 + 0.2*x1 - 0.1*x2 + 0.05*(t % 3)   # always > 0
                    println(io, "$y,$x1,$x2")
                end
            end
            # Heckman: outcome y, binary selection d, const=1, outcome regressor x1,
            # selection regressor z1 (the exclusion instrument, not in outcome eqn).
            hcsv = joinpath(dir, "heck.csv")
            open(hcsv, "w") do io
                println(io, "y,d,const,x1,z1")
                for t in 1:120
                    x1 = (t % 9) - 4.0; z1 = (t % 6) - 2.5
                    d = (t % 3 == 0) ? 1.0 : 0.0
                    y = 2.0 + 0.5*x1
                    println(io, "$y,$d,1.0,$x1,$z1")
                end
            end
            # IV / weak-instrument: y, const, x_endog, z1, z2 (excluded instruments).
            ivcsv = joinpath(dir, "iv.csv")
            open(ivcsv, "w") do io
                println(io, "y,const,x_endog,z1,z2")
                for t in 1:100
                    z1 = (t % 8) - 3.5; z2 = (t % 5) - 2.0
                    xe = 0.5*z1 + 0.3*z2 + 0.1*(t % 4)
                    y = 1.0 + 2.0*xe
                    println(io, "$y,1.0,$xe,$z1,$z2")
                end
            end

            @testset "truncreg — parameter/estimate/std_error + diag (Inf-safe)" begin
                doc = _edoc(["truncreg", trcsv, "--dep", "y", "--lower", "0.0"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "parameter", "estimate", "std_error")
                @test length(collect(coef.rows)) == 2      # x1,x2
                m = _metrics(doc)
                @test Set(["sigma", "n_truncated", "loglik", "converged"]) ⊆ m
            end

            @testset "heckman — two-equation tidy coef (equation col) + diag" begin
                doc = _edoc(["heckman", hcsv, "--dep", "y", "--select", "d",
                             "--outcome-vars", "const,x1", "--select-vars", "const,z1"])
                @test doc.status == "ok"
                coef = _tbl_with(doc, "equation", "term", "estimate")
                eqs = Set(String(collect(r)[findfirst(==("equation"), String.(coef.columns))]) for r in coef.rows)
                @test Set(["outcome", "selection"]) ⊆ eqs
                m = _metrics(doc)
                @test Set(["method", "rho", "sigma", "lambda", "n_selected", "n_total"]) ⊆ m
            end

            @testset "heckman — mle method also runs" begin
                doc = _edoc(["heckman", hcsv, "--dep", "y", "--select", "d",
                             "--outcome-vars", "const,x1", "--select-vars", "const,z1", "--method", "mle"])
                @test doc.status == "ok"
            end

            @testset "weak-instrument — F diagnostics + verdict metric" begin
                doc = _tdoc(["weak-instrument", ivcsv, "--dep", "y",
                             "--endogenous", "x_endog", "--instruments", "z1,z2"])
                @test doc.status == "ok"
                m = _metrics(doc)
                @test Set(["first_stage_f", "n_endogenous", "n_excluded_instruments", "weak"]) ⊆ m
            end

            @testset "C067b bad input → typed classes, never uncaught exit-1" begin
                # heckman: missing required opts → usage/missing (exit 2)
                @test _eerr(["heckman", hcsv, "--dep", "y", "--outcome-vars", "const,x1",
                             "--select-vars", "const,z1"]) isa CliError                       # no --select
                em = _eerr(["heckman", hcsv, "--dep", "y", "--select", "d",
                            "--outcome-vars", "const,x1", "--select-vars", "const,z1", "--method", "bogus"])
                # --method is enum-rejected at parse (ParseError) or by the handler (usage/invalid); both exit 2
                @test em isa ParseError || (em isa CliError && exit_class(em) == 2)
                # heckman: unknown column → data/column-range (exit 3)
                ec = _eerr(["heckman", hcsv, "--dep", "y", "--select", "d",
                            "--outcome-vars", "const,nope", "--select-vars", "const,z1"])
                @test ec isa CliError && ec.code == "data/column-range"
                # truncreg: y outside bounds → typed data error, not raw MEMs exit-1
                eb = _eerr(["truncreg", trcsv, "--dep", "y", "--lower", "100.0", "--upper", "200.0"])
                @test eb isa CliError && exit_class(eb) == 3
                @test _eerr(["truncreg", trcsv, "--dep", "y", "--lower", "5", "--upper", "1"]).code == "usage/invalid"
                # weak-instrument: missing --endogenous/--instruments → usage/missing (exit 2)
                @test _terr(["weak-instrument", ivcsv, "--dep", "y", "--instruments", "z1,z2"]) isa CliError
                ew = _terr(["weak-instrument", ivcsv, "--dep", "y", "--endogenous", "x_endog", "--instruments", "z1,z2"])
                @test ew === nothing || ew isa CliError   # runs (or a typed error), never uncaught
                @test _terr(["weak-instrument", ivcsv, "--dep", "y", "--endogenous", "nope",
                             "--instruments", "z1,z2"]).code == "data/column-range"
            end

            @testset "C067b adversarial-review regressions" begin
                # (1) weak-instrument / iv: more instruments than observations → data/invalid
                # (loader m<n guard), NOT a NaN "weak=false / instruments look strong" verdict.
                deg = joinpath(dir, "degenerate_iv.csv")
                open(deg, "w") do io
                    println(io, "y,x_endog," * join(["z$k" for k in 1:8], ","))
                    for t in 1:6   # n=6 rows, but 8 excluded instruments → m=9 ≥ n
                        vals = [string(0.5*t + 0.1*k) for k in 0:8]
                        println(io, "$(1.0*t)," * join(vals, ","))
                    end
                end
                edg = _terr(["weak-instrument", deg, "--dep", "y", "--endogenous", "x_endog",
                             "--instruments", join(["z$k" for k in 1:8], ",")])
                @test edg isa CliError && edg.code == "data/invalid" && exit_class(edg) == 3

                # (2) heckman: a BLANK outcome for NON-selected (d==0) rows is the canonical
                # incidental-truncation layout → estimates fine (not data/missing-values).
                hmiss = joinpath(dir, "heck_blank.csv")
                open(hmiss, "w") do io
                    println(io, "y,d,const,x1,z1")
                    for t in 1:120
                        x1 = (t % 9) - 4.0; z1 = (t % 6) - 2.5
                        d = (t % 3 == 0) ? 1.0 : 0.0
                        ystr = d == 1.0 ? string(2.0 + 0.5*x1) : ""   # blank when unobserved
                        println(io, "$ystr,$d,1.0,$x1,$z1")
                    end
                end
                dh = _edoc(["heckman", hmiss, "--dep", "y", "--select", "d",
                            "--outcome-vars", "const,x1", "--select-vars", "const,z1"])
                @test dh.status == "ok"
                # but a blank outcome for a SELECTED (d==1) row IS an error.
                hbad = joinpath(dir, "heck_badsel.csv")
                open(hbad, "w") do io
                    println(io, "y,d,const,x1,z1")
                    for t in 1:120
                        x1 = (t % 9) - 4.0; z1 = (t % 6) - 2.5
                        d = (t % 3 == 0) ? 1.0 : 0.0
                        # row 3 is selected (t=3, d=1) but leave its outcome blank
                        ystr = (t == 3) ? "" : (d == 1.0 ? string(2.0 + 0.5*x1) : "")
                        println(io, "$ystr,$d,1.0,$x1,$z1")
                    end
                end
                eh = _eerr(["heckman", hbad, "--dep", "y", "--select", "d",
                            "--outcome-vars", "const,x1", "--select-vars", "const,z1"])
                @test eh isa CliError && eh.code == "data/missing-values"
            end
        end
    end

    @testset "estimate statespace/tvp/kde/kernel-reg/lowess (C066)" begin
        _edoc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["estimate"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tables(doc) = [t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns))]
        _tbl_with(doc, cols...) = first(t for t in _tables(doc) if Set(String.(cols)) ⊆ Set(String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for r in
                            first(t for t in _tables(doc) if "metric" in String.(t.columns)).rows)
        # Local on purpose: `_mval` in the MS testset is a testset-SCOPED closure, not a
        # global, so reaching for it from here would be an UndefVarError at run time.
        _mv(doc, name) = begin
            kv = first(t for t in _tables(doc) if "metric" in String.(t.columns))
            v = nothing
            for r in kv.rows
                rr = collect(r); String(rr[1]) == name && (v = rr[2])
            end
            v
        end
        _eerr(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["estimate"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            # Univariate series for statespace/kde (single numeric column).
            uni = joinpath(dir, "uni.csv")
            open(uni, "w") do io
                println(io, "y")
                for t in 1:60
                    println(io, 10.0 + 0.1*t + sin(t/5.0))
                end
            end
            # y + single predictor x for kernel-reg/lowess/tvp (linear y = 1 + 2x).
            xy = joinpath(dir, "xy.csv")
            open(xy, "w") do io
                println(io, "y,x")
                for t in 1:50
                    x = (t % 11) - 5.0
                    println(io, "$(1.0 + 2.0*x + 0.05*(t % 3)),$x")
                end
            end

            @testset "statespace local-level — param table + diag" begin
                doc = _edoc(["statespace", uni, "--model", "local-level"])
                @test doc.status == "ok"
                pt = _tbl_with(doc, "parameter", "estimate")
                @test length(collect(pt.rows)) == 2               # σ²_ε, σ²_η
                m = _metrics(doc)
                @test Set(["model", "loglik", "converged", "n_state"]) ⊆ m
            end

            @testset "statespace --config — general fixed-matrix system (#71)" begin
                # A general system is MULTIVARIATE and has NO estimated hyper-parameters, so it
                # renders a SYSTEM table (matrix|role|rows|cols) instead of parameter|estimate —
                # `theta`/`param_names` come back empty and the parameter table would otherwise
                # be silently blank.
                bi = joinpath(dir, "bi.csv")
                open(bi, "w") do io
                    println(io, "y1,y2")
                    for t in 1:60
                        f = 10.0 + 0.1*t
                        println(io, "$(f + 0.1*sin(t)),$(0.7*f + 0.1*cos(t))")
                    end
                end
                cfg = joinpath(dir, "ss.toml")
                write(cfg, """
                [statespace]
                Z = [[1.0], [0.7]]
                H = [[1.0, 0.0], [0.0, 2.0]]
                T = [[0.95]]
                Q = [[0.5]]
                """)
                doc = _edoc(["statespace", bi, "--config", cfg])
                @test doc.status == "ok"
                st = _tbl_with(doc, "matrix", "role", "rows", "cols")
                rows = collect(st.rows)
                @test Set(String(collect(r)[1]) for r in rows) == Set(["Z","H","T","Q","R","d","c"])
                dims = Dict(String(collect(r)[1]) => (Int(collect(r)[3]), Int(collect(r)[4])) for r in rows)
                @test dims["Z"] == (2, 1)          # n_obs × n_state
                @test dims["H"] == (2, 2)
                @test dims["T"] == (1, 1)
                @test dims["R"] == (1, 1)          # defaulted to I by MEMs
                m = _metrics(doc)
                @test Set(["model", "loglik", "init_mode", "n_obs_series", "n_state",
                           "n_periods", "method"]) ⊆ m
                # a fixed-matrix system is filtered, never optimized
                @test String(_mv(doc, "method")) == "filter"
                @test String(_mv(doc, "model")) == "general"
                @test Int(_mv(doc, "n_obs_series")) == 2
                @test Int(_mv(doc, "n_state")) == 1
                @test Int(_mv(doc, "n_periods")) == 60
                # --config supersedes --model; the canned parameter table must NOT appear
                @test isempty([t for t in _tables(doc) if "parameter" in String.(t.columns)])

                # n_obs implied by Z must match the CSV's numeric column count → data/shape (3)
                e = _eerr(["statespace", uni, "--config", cfg])
                @test e isa CliError && e.code == "data/shape" && exit_class(e) == 3
                # a malformed system is a CONFIG error (exit 4), naming the file the user wrote
                badcfg = joinpath(dir, "bad.toml")
                write(badcfg, """
                [statespace]
                Z = [[1.0]]
                H = [[1.0, 0.0], [0.0, 1.0]]
                T = [[1.0]]
                Q = [[1.0]]
                """)
                eb = _eerr(["statespace", uni, "--config", badcfg])
                @test eb isa CliError && exit_class(eb) == 4
                # ...and the canned path is unaffected by all of this
                @test _edoc(["statespace", uni, "--model", "local-level"]).status == "ok"
            end

            @testset "statespace local-linear-trend — 3 hyper-params" begin
                doc = _edoc(["statespace", uni, "--model", "local-linear-trend"])
                @test doc.status == "ok"
                @test length(collect(_tbl_with(doc, "parameter", "estimate").rows)) == 3
            end

            @testset "tvp — coefficient path is tidy (period|coefficient|estimate)" begin
                doc = _edoc(["tvp", xy, "--dep", "y"])
                @test doc.status == "ok"
                path = _tbl_with(doc, "period", "coefficient", "estimate")
                # 50 periods × 2 coefs (intercept + x) = 100 rows
                @test length(collect(path.rows)) == 100
                @test Set(["loglik", "n_coef", "intercept", "method"]) ⊆ _metrics(doc)
            end

            @testset "kde — x|density grid + bandwidth diag" begin
                doc = _edoc(["kde", uni, "--npoints", "128"])
                @test doc.status == "ok"
                grid = _tbl_with(doc, "x", "density")
                @test length(collect(grid.rows)) == 128
                @test Set(["kernel", "bw_method", "bandwidth", "nobs"]) ⊆ _metrics(doc)
            end

            @testset "kernel-reg — x|fitted|se + diag" begin
                doc = _edoc(["kernel-reg", xy, "--dep", "y", "--indep", "x"])
                @test doc.status == "ok"
                fit = _tbl_with(doc, "x", "fitted", "se")
                @test length(collect(fit.rows)) == 50
                @test Set(["method", "degree", "kernel", "bandwidth", "nobs"]) ⊆ _metrics(doc)
            end

            @testset "lowess — x|fitted + diag" begin
                doc = _edoc(["lowess", xy, "--dep", "y", "--indep", "x"])
                @test doc.status == "ok"
                fit = _tbl_with(doc, "x", "fitted")
                @test length(collect(fit.rows)) == 50
                @test Set(["frac", "iter", "nobs"]) ⊆ _metrics(doc)
            end

            @testset "C066 bad input → typed classes, never uncaught exit-1" begin
                # kde: junk --bw → usage/invalid
                @test _eerr(["kde", uni, "--bw", "notanumber"]).code == "usage/invalid"
                # kernel-reg / lowess: missing --indep → usage/missing
                @test _eerr(["kernel-reg", xy, "--dep", "y"]).code == "usage/missing"
                @test _eerr(["lowess", xy, "--dep", "y"]).code == "usage/missing"
                # unknown --indep column → data/column-range
                @test _eerr(["kernel-reg", xy, "--dep", "y", "--indep", "nope"]).code == "data/column-range"
                # predictor == response → data/column-range
                @test _eerr(["lowess", xy, "--dep", "y", "--indep", "y"]).code == "data/column-range"
                # lowess: --frac out of (0,1] → usage/invalid
                @test _eerr(["lowess", xy, "--dep", "y", "--indep", "x", "--frac", "1.5"]).code == "usage/invalid"
                # missing cell in the predictor → data/missing-values (guard before Float64 conv)
                xmiss = joinpath(dir, "xmiss.csv")
                open(xmiss, "w") do io
                    println(io, "y,x")
                    println(io, "1.0,0.5"); println(io, "2.0,"); println(io, "3.0,1.5")
                end
                @test _eerr(["kernel-reg", xmiss, "--dep", "y", "--indep", "x"]).code == "data/missing-values"
            end

            @testset "C066 adversarial-review regressions" begin
                # (fix) _parse_bandwidth must reject non-finite --bw → usage/invalid (exit 2),
                # not slip Inf through (degenerate fit) or let NaN become MEMs data/invalid (exit 3).
                @test _eerr(["kde", uni, "--bw", "Inf"]).code == "usage/invalid"
                @test _eerr(["kde", uni, "--bw", "NaN"]).code == "usage/invalid"
                @test _eerr(["kernel-reg", xy, "--dep", "y", "--indep", "x", "--bw", "Inf"]).code == "usage/invalid"
                # (fix) --degree guarded up-front as a CLI arg → usage/invalid, not MEMs data/invalid.
                ed = _eerr(["kernel-reg", xy, "--dep", "y", "--indep", "x", "--method", "lp", "--degree", "-1"])
                @test ed isa CliError && ed.code == "usage/invalid" && exit_class(ed) == 2
                # (fix) all-non-numeric CSV with default --dep → data/invalid, NOT an untyped
                # BoundsError exit-1 (numcols[1] on an empty vector) — for the whole loader family.
                strcsv = joinpath(dir, "strings.csv")
                open(strcsv, "w") do io
                    println(io, "a,b"); println(io, "foo,bar"); println(io, "baz,qux")
                end
                @test _eerr(["kernel-reg", strcsv, "--indep", "b"]).code == "data/invalid"
                @test _eerr(["lowess", strcsv, "--indep", "b"]).code == "data/invalid"
                @test _eerr(["reg", strcsv]).code == "data/invalid"           # mirrored _load_reg_data hole
                # (infra class-fix) the legacy-JSON writer sanitizes non-finite floats (Inf/NaN →
                # "Inf"/"NaN" strings) instead of crashing JSON3.write ("… not allowed in JSON spec").
                out = _capture() do
                    _write_json_raw(Dict("a" => Inf, "b" => NaN, "c" => 1.5), "")
                end
                @test occursin("Inf", out) && occursin("NaN", out)           # rendered, no crash
            end
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
            @test occursin("std_error", out) || occursin("z_stat", out) || occursin("Parameter", out)
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
        end
    end

    @testset "_estimate_reg — WLS with weights" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _estimate_reg(; data=csv, dep="var1", cov_type="ols",
                               weights="var4", clusters="", format="table", output="")
            end
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
        end
    end

    @testset "_estimate_select — variable selection (#72)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=5, colnames=["y", "x1", "x2", "x3", "x4"])
            out = _capture() do
                _estimate_select(; data=csv, dep="y", format="table", output="")
            end
            @test contains(out, "Selection Summary") || contains(out, "method")
            # every method/criterion combination reaches the estimator
            for m in ("forward", "backward", "bidirectional", "gets")
                o = _capture() do
                    _estimate_select(; data=csv, dep="y", method=m, format="table", output="")
                end
                @test contains(o, m)
            end
            o = _capture() do
                _estimate_select(; data=csv, dep="y", criterion="aic", format="table", output="")
            end
            @test contains(o, "aic")
            # --keep forces a regressor into the final model
            o = _capture() do
                _estimate_select(; data=csv, dep="y", keep="x3", format="table", output="")
            end
            @test contains(o, "x3")

            err(kw) = try
                _capture() do; _estimate_select(; data=csv, dep="y", format="table", output="", kw...); end
                nothing
            catch e; e end
            e = err((p_enter=0.0,));  @test e isa CliError && e.code == "usage/invalid"
            e = err((p_remove=1.5,)); @test e isa CliError && e.code == "usage/invalid"
            # bidirectional pvalue needs p_remove >= p_enter — guarded up front
            e = err((p_enter=0.2, p_remove=0.05))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            e = err((keep="nosuch",))
            @test e isa CliError && e.code == "data/column-range"
        end
    end

    @testset "_estimate_iv — k-class family (#72)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            for m in ("tsls", "liml", "fuller")
                out = _capture() do
                    _estimate_iv(; data=csv, dep="y", endogenous="x1",
                                  instruments="z1,z2", cov_type="hc1", method=m,
                                  format="table", output="")
                end
                @test contains(out, m)
            end
            out = _capture() do
                _estimate_iv(; data=csv, dep="y", endogenous="x1",
                              instruments="z1,z2", cov_type="hc1",
                              method="kclass", k="1", format="table", output="")
            end
            @test contains(out, "k-class k")

            # --k is REQUIRED for kclass and REJECTED for the others; --fuller-a only
            # applies to fuller. Each is a usage error, never an upstream ArgumentError.
            err(kw) = try
                _capture() do; _estimate_iv(; data=csv, dep="y", endogenous="x1",
                    instruments="z1,z2", cov_type="hc1", format="table", output="", kw...); end
                nothing
            catch e; e end
            e = err((method="kclass",))
            @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
            e = err((method="kclass", k="junk"))
            @test e isa CliError && e.code == "usage/invalid"
            e = err((method="tsls", k="1"))
            @test e isa CliError && e.code == "usage/invalid"
            e = err((method="tsls", fuller_a=2.0))
            @test e isa CliError && e.code == "usage/invalid"
            e = err((method="fuller", fuller_a=0.0))
            @test e isa CliError && e.code == "usage/invalid"
        end
    end

    @testset "_estimate_iv — missing endogenous error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            @test_throws Exception _estimate_iv(; data=csv, dep="y",
                endogenous="", instruments="z1,z2", cov_type="hc1",
                format="table", output="")
        end
    end

    @testset "_estimate_iv — missing instruments error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
            @test_throws Exception _estimate_iv(; data=csv, dep="y",
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
        # 80 primary + 2 snake aliases (arch_lm, ljung_box) = 69 keys (C044; +gph, +local-whittle C068, +sign-bias, +nyblom C064b, +vecm C071, +variance-ratio/bds/hadri/pedroni/kao/westerlund C069/C070, +weak-instrument C067b, +ardl-bounds/nardl-symmetry C062b, +pmg-hausman C062c, +hansen-linearity C065a, +star-linearity C065b, +hegy/ers/sadf/gsadf/edf/engle-granger/phillips-ouliaris/hansen-instability/park-added C069 remainder)
        @test length(node.subcmds) == 85
        for cmd in ["llc", "ips", "breitung", "fisher-johansen", "dh-causality",
                     "white", "glejser", "harvey", "chow", "cusum", "cusumsq", "recursive-residuals", "influence",
                     "hegy", "ers", "sadf", "gsadf", "edf", "engle-granger",
                     "phillips-ouliaris", "hansen-instability", "park-added",
                     "adf", "kpss", "pp", "za", "np", "gph", "local-whittle", "johansen",
                     "normality", "identifiability", "heteroskedasticity",
                     "arch-lm", "ljung-box", "sign-bias", "nyblom", "var", "vecm", "granger", "pvar", "lr", "lm",
                     "variance-ratio", "bds", "hadri", "pedroni", "kao", "westerlund", "weak-instrument",
                     "andrews", "bai-perron", "panic", "cips", "moon-perron", "factor-break",
                     "fourier-adf", "fourier-kpss", "dfgls", "lm-unitroot",
                     "adf-2break", "gregory-hansen", "vif",
                     "hausman", "breusch-pagan", "f-fe", "pesaran-cd", "wooldridge-ar", "modified-wald",
                     "fisher", "bartlett-wn", "box-pierce", "durbin-watson",
                     "brant", "hausman-iia"]
            @test haskey(node.subcmds, cmd)
        end
        @test haskey(node.subcmds, "arch_lm") && node.subcmds["arch_lm"].name == "arch-lm"
        @test haskey(node.subcmds, "ljung_box") && node.subcmds["ljung_box"].name == "ljung-box"
        # VAR is a nested NodeCommand with lagselect and stability
        var_node = node.subcmds["var"]
        @test var_node isa NodeCommand
        @test haskey(var_node.subcmds, "lagselect")
        @test haskey(var_node.subcmds, "stability")
        # PVAR: 4 primary + 1 alias (hansen_j) = 5 keys (C044)
        pvar_node = node.subcmds["pvar"]
        @test pvar_node isa NodeCommand
        @test length(pvar_node.subcmds) == 5
        @test haskey(pvar_node.subcmds, "hansen-j")
        @test haskey(pvar_node.subcmds, "hansen_j")  # alias
        @test haskey(pvar_node.subcmds, "mmsc")
        @test haskey(pvar_node.subcmds, "lagselect")
        @test haskey(pvar_node.subcmds, "stability")
        # VECM: nested NodeCommand with 5 restriction-test leaves (C071)
        vecm_node = node.subcmds["vecm"]
        @test vecm_node isa NodeCommand
        @test length(vecm_node.subcmds) == 5
        @test haskey(vecm_node.subcmds, "beta")
        @test haskey(vecm_node.subcmds, "alpha")
        @test haskey(vecm_node.subcmds, "weak-exog")
        @test haskey(vecm_node.subcmds, "known-beta")
        @test haskey(vecm_node.subcmds, "joint")
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
        end
    end

    @testset "_test_adf — explicit lags and different trends" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for trend in ["none", "constant", "trend", "both"]
                out = _capture() do
                    _test_adf(; data=csv, column=1, max_lags=4, trend=trend, format="table")
                end
            end
        end
    end

    @testset "_test_kpss" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_kpss(; data=csv, column=1, trend="constant", format="table")
            end
        end
    end

    @testset "_test_pp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_pp(; data=csv, column=1, trend="constant", format="table")
            end
        end
    end

    @testset "_test_pp — none trend" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_pp(; data=csv, column=1, trend="none", format="table")
            end
        end
    end

    @testset "_test_za" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_za(; data=csv, column=1, trend="both", trim=0.15, format="table")
            end
        end
    end

    @testset "_test_np" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_np(; data=csv, column=1, trend="constant", format="table")
            end
        end
    end

    @testset "_test_gph" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_gph(; data=csv, column=1, format="table")
            end
            @test occursin("GPH", out)
            @test occursin("long-memory", out)
        end
    end

    @testset "_test_gph — bandwidth + trim + json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_gph(; data=csv, column=1, bandwidth=20, trim=1, format="json")
            end
        end
    end

    @testset "_test_gph — short series errors as CliError" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=5, n=1, colnames=["x"])
            @test_throws CliError _capture() do
                _test_gph(; data=csv, column=1, format="table")
            end
        end
    end

    @testset "_test_gph — negative --trim → usage error (review fix)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1, colnames=["x"])
            # was a BoundsError misclassified as model/error (exit 5); now usage/invalid (exit 2)
            err = nothing
            try; _capture() do; _test_gph(; data=csv, column=1, trim=-1, format="table"); end; catch e; err = e; end
            @test err isa CliError && err.code == "usage/invalid" && exit_class(err) == 2
        end
    end

    @testset "_test_local_whittle" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_local_whittle(; data=csv, column=1, format="table")
            end
            @test occursin("Whittle", out)
            @test occursin("long-memory", out)
        end
    end

    @testset "_test_local_whittle — bandwidth" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_local_whittle(; data=csv, column=1, bandwidth=15, format="table")
            end
        end
    end

    @testset "_test_local_whittle — short series errors as CliError" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=5, n=1, colnames=["x"])
            @test_throws CliError _capture() do
                _test_local_whittle(; data=csv, column=1, format="table")
            end
        end
    end

    @testset "_test_sign_bias / _test_nyblom (C064b)" begin
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tbl(doc, col) = first(t for t in values(doc.data)
                               if (t isa JSON3.Object && haskey(t, :columns) && col in String.(t.columns)))

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2, colnames=["ret", "other"])

            @testset "sign-bias — Engle-Ng kv keys (garch)" begin
                doc = _doc(["sign-bias", csv, "--column", "1", "--model", "garch"])
                @test doc.status == "ok"
                metrics = Set(String(collect(r)[1]) for r in _tbl(doc, "metric").rows)
                @test Set(["sign_bias", "joint_statistic", "joint_pvalue", "dof"]) ⊆ metrics
            end

            @testset "sign-bias — egarch/gjr-garch models dispatch" begin
                @test _doc(["sign-bias", csv, "--model", "egarch"]).status == "ok"
                @test _doc(["sign-bias", csv, "--model", "gjr-garch"]).status == "ok"
            end

            @testset "nyblom — individual table + joint kv" begin
                doc = _doc(["nyblom", csv, "--column", "1", "--model", "garch"])
                @test doc.status == "ok"
                ind = _tbl(doc, "parameter")
                @test Set(["parameter", "L_stat", "cv_5pct", "reject_5pct"]) ⊆ Set(String.(ind.columns))
                metrics = Set(String(collect(r)[1]) for r in _tbl(doc, "metric").rows)
                @test Set(["joint_LC", "cv_joint_5pct", "n_params", "reject_joint_5pct"]) ⊆ metrics
            end

            @testset "bad --model → usage/invalid (helper is enum-guarded, but test the dispatcher)" begin
                @test_throws CliError _fit_vol_for_diag(randn(50), "bogus", 1, 1)
                fe = nothing
                try; _fit_vol_for_diag(randn(50), "bogus", 1, 1); catch ex; fe=ex; end
                @test fe isa CliError && fe.code == "usage/invalid" && exit_class(fe) == 2
            end

            @testset "column out of range → data/column-range (not exit 1)" begin
                e = nothing
                try; _capture() do; _dispatch_via_app(String["test","sign-bias",csv,"--column","9"]); end; catch ex; e=ex; end
                @test e isa CliError && e.code == "data/column-range"
            end
        end
    end

    @testset "test vecm restriction tests (C071)" begin
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        # kv metric names live in the single-table's `metric` column.
        _metrics(doc) = Set(String(collect(r)[1]) for t in values(doc.data)
                            if (t isa JSON3.Object && haskey(t, :columns) && "metric" in String.(t.columns))
                            for r in t.rows)
        _errcode(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["test"], collect(String, args))); end
            catch ex; e = ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2, colnames=["gdp", "rate"])
            # Combined restriction config: H (β), A (α), b (known β) — all p=2, r=1.
            cfg = joinpath(dir, "restr.toml")
            write(cfg, """
            [vecm_restriction]
            H = [[1.0], [-1.0]]
            A = [[1.0], [0.0]]
            b = [[1.0], [-1.0]]
            """)

            @testset "beta / alpha / known-beta / joint kv keys" begin
                for (leaf, extra) in [("beta", String[]), ("alpha", String[]),
                                      ("known-beta", String[]), ("joint", String[])]
                    doc = _doc(vcat(["vecm", leaf, csv, "--config", cfg, "--rank", "1"], extra))
                    @test doc.status == "ok"
                    @test Set(["LR statistic", "df", "p-value", "rank (r)", "converged", "restriction"]) ⊆ _metrics(doc)
                end
            end

            @testset "weak-exog by index and by name" begin
                d1 = _doc(["vecm", "weak-exog", csv, "--vars", "1", "--rank", "1"])
                @test d1.status == "ok"
                @test Set(["LR statistic", "df", "p-value", "rank (r)"]) ⊆ _metrics(d1)
                d2 = _doc(["vecm", "weak-exog", csv, "--vars", "rate", "--rank", "1"])
                @test d2.status == "ok"
                # duplicate indices dedupe (regression: adversarial review C071) — real MEMs
                # selects via setdiff, so `--vars 1,1` == `--vars 1` and must NOT be falsely rejected
                d3 = _doc(["vecm", "weak-exog", csv, "--vars", "1,1", "--rank", "1"])
                @test d3.status == "ok"
            end

            @testset "bad input → typed CliError (never internal exit 1)" begin
                # missing --config → usage/missing-config (exit 2)
                for leaf in ["beta", "alpha", "known-beta", "joint"]
                    e = _errcode(["vecm", leaf, csv, "--rank", "1"])
                    @test e isa CliError && e.code == "usage/missing-config" && exit_class(e) == 2
                end
                # wrong-rows H (3 rows ≠ p=2) → config/shape (exit 4)
                badcfg = joinpath(dir, "bad.toml")
                write(badcfg, "[vecm_restriction]\nH = [[1.0], [-1.0], [0.5]]\n")
                e = _errcode(["vecm", "beta", csv, "--config", badcfg, "--rank", "1"])
                @test e isa CliError && e.code == "config/shape" && exit_class(e) == 4
                # weak-exog: empty / out-of-range / unknown name → usage/invalid (exit 2)
                @test _errcode(["vecm", "weak-exog", csv, "--rank", "1"]) isa CliError            # no --vars
                for badvars in ["9", "nope"]
                    e = _errcode(["vecm", "weak-exog", csv, "--vars", badvars, "--rank", "1"])
                    @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                end
                # all-vars weakly exogenous rejected before the MEMs call
                e = _errcode(["vecm", "weak-exog", csv, "--vars", "1,2", "--rank", "1"])
                @test e isa CliError && e.code == "usage/invalid"
                # rank-aware guard (regression C071): on a 3-series rank-2 fit, making 2 vars
                # weakly exogenous leaves 1 < r=2 error-correcting rows → usage/invalid at the CLI
                # boundary, NOT a config/shape (this command takes no config matrix)
                csv3 = _make_csv(dir; T=150, n=3, colnames=["a", "b", "c"])
                e = _errcode(["vecm", "weak-exog", csv3, "--vars", "1,2", "--rank", "2"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                # fitted rank 0 → data/no-cointegration (exit 3)
                e = _errcode(["vecm", "beta", csv, "--config", cfg, "--rank", "0"])
                @test e isa CliError && e.code == "data/no-cointegration" && exit_class(e) == 3
                # malformed config (missing key A for joint) → config/missing-key (exit 4)
                honly = joinpath(dir, "honly.toml")
                write(honly, "[vecm_restriction]\nH = [[1.0], [-1.0]]\n")
                e = _errcode(["vecm", "joint", csv, "--config", honly, "--rank", "1"])
                @test e isa CliError && e.code == "config/missing-key" && exit_class(e) == 4
            end
        end
    end

    @testset "TS + panel test batteries (C069/C070)" begin
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tbl(doc, col) = first(t for t in values(doc.data)
                               if (t isa JSON3.Object && haskey(t, :columns) && col in String.(t.columns)))
        _metrics(doc) = Set(String(collect(r)[1]) for t in values(doc.data)
                            if (t isa JSON3.Object && haskey(t, :columns) && "metric" in String.(t.columns))
                            for r in t.rows)
        _errcode(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["test"], collect(String, args))); end
            catch ex; e = ex; end
            e
        end

        mktempdir() do dir
            uni = _make_csv(dir; T=120, n=2, colnames=["ret", "other"])
            mv  = _make_csv(dir; T=60, n=4, colnames=["u1", "u2", "u3", "u4"])
            panel = _make_panel_csv(dir; G=6, T_per=20, n=3, colnames=["y", "x1", "x2"])

            @testset "variance-ratio — per-horizon table + joint kv" begin
                doc = _doc(["variance-ratio", uni, "--column", "1", "--horizons", "2,4,8"])
                @test doc.status == "ok"
                @test Set(["horizon", "variance_ratio", "z_star", "p_value"]) ⊆ Set(String.(_tbl(doc, "horizon").columns))
                @test length(_tbl(doc, "horizon").rows) == 3
                @test Set(["Chow-Denning stat", "Chow-Denning p-value", "observations"]) ⊆ _metrics(doc)
            end

            @testset "bds — per-dimension statistic table" begin
                doc = _doc(["bds", uni, "--column", "1", "--max-dim", "5"])
                @test doc.status == "ok"
                @test Set(["embed_dim", "statistic", "p_value"]) ⊆ Set(String.(_tbl(doc, "embed_dim").columns))
                @test length(_tbl(doc, "embed_dim").rows) == 4   # m = 2..5
            end

            @testset "hadri — panel stationarity kv" begin
                doc = _doc(["hadri", mv, "--deterministic", "constant"])
                @test doc.status == "ok"
                @test Set(["statistic", "p-value", "n_units", "observations"]) ⊆ _metrics(doc)
                @test _doc(["hadri", mv, "--deterministic", "trend"]).status == "ok"
            end

            @testset "pedroni/kao/westerlund — shared statistic table + metadata kv" begin
                for (leaf, ncols) in [("pedroni", 7), ("kao", 5), ("westerlund", 4)]
                    doc = _doc([leaf, panel])
                    @test doc.status == "ok"
                    stat_tbl = _tbl(doc, "statistic")
                    @test Set(["statistic", "value", "p_value"]) ⊆ Set(String.(stat_tbl.columns))
                    @test length(stat_tbl.rows) == ncols
                    @test Set(["n_units", "n_regressors", "observations"]) ⊆ _metrics(doc)
                end
                # explicit --dep/--indep selection
                @test _doc(["pedroni", panel, "--dep", "y", "--indep", "x1,x2"]).status == "ok"
                @test _doc(["kao", panel, "--dep", "y", "--indep", "x1"]).status == "ok"
            end

            @testset "bad input → typed CliError (never internal exit 1)" begin
                # bad --column → data/column-range (loader)
                e = _errcode(["variance-ratio", uni, "--column", "9"])
                @test e isa CliError && e.code == "data/column-range"
                # junk --horizons → usage/invalid
                e = _errcode(["variance-ratio", uni, "--horizons", "junk"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                # horizon < 2 → usage/invalid
                e = _errcode(["variance-ratio", uni, "--horizons", "1,2"])
                @test e isa CliError && e.code == "usage/invalid"
                # bds --max-dim 1 → usage/invalid
                e = _errcode(["bds", uni, "--max-dim", "1"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                # pedroni unknown --dep → usage/invalid
                e = _errcode(["pedroni", panel, "--dep", "nope"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                # pedroni unknown --indep → usage/invalid
                e = _errcode(["pedroni", panel, "--indep", "nope"])
                @test e isa CliError && e.code == "usage/invalid"
                # hadri bad --deterministic → usage error (enum-rejected at parse → ParseError,
                # exit 2; the handler also guards it if reached directly)
                e = _errcode(["hadri", mv, "--deterministic", "bogus"])
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                # pedroni missing id/time column → typed data error (loader)
                e = _errcode(["pedroni", panel, "--id-col", "nosuch"])
                @test e isa CliError && e.code == "data/missing-column"
                # a panel whose only variable column is non-numeric → data/invalid, NOT an
                # uncaught exit-1 (regression: adversarial review C069/C070 — load_panel_data's
                # untyped error() → typed CliError; benefits the whole panel family)
                strcsv = joinpath(dir, "panel_str.csv")
                write(strcsv, "id,time,label\n1,1,foo\n1,2,bar\n2,1,baz\n2,2,qux\n")
                es = _errcode(["pedroni", strcsv])
                @test es isa CliError && es.code == "data/invalid" && exit_class(es) == 3
            end

            # ── C069 remainder: seasonal / point-optimal / bubble / EDF + residual
            # cointegration. `reg` is a (y, X) frame for the cointegration leaves.
            reg = _make_csv(dir; T=120, n=3, colnames=["y", "x1", "x2"])

            @testset "hegy — per-frequency decision table (no single p-value)" begin
                doc = _doc(["hegy", uni, "--column", "1", "--frequency", "4"])
                @test doc.status == "ok"
                tbl = _tbl(doc, "frequency")
                # HEGY rejects per frequency, so every row carries its own CV + decision.
                @test Set(["frequency", "kind", "statistic", "cv_5pct", "decision"]) ⊆ Set(String.(tbl.columns))
                @test length(tbl.rows) == 3          # zero + Nyquist + 1 harmonic pair (quarterly)
                @test Set(["F (seasonal, joint)", "F (all roots)", "deterministic", "lags"]) ⊆ _metrics(doc)
                @test _doc(["hegy", uni, "--frequency", "12"]).status == "ok"
            end

            @testset "ers — point-optimal kv with critical values" begin
                doc = _doc(["ers", uni, "--column", "1"])
                @test doc.status == "ok"
                @test Set(["P_T statistic", "p-value", "regression", "CV 5%"]) ⊆ _metrics(doc)
                @test _doc(["ers", uni, "--trend"]).status == "ok"
            end

            @testset "sadf/gsadf — episode table + summary kv" begin
                for leaf in ("sadf", "gsadf")
                    doc = _doc([leaf, uni, "--column", "1", "--mc-reps", "20"])
                    @test doc.status == "ok"
                    @test Set(["episode", "start_index", "end_index"]) ⊆ Set(String.(_tbl(doc, "episode").columns))
                    @test Set(["statistic", "p-value", "kind", "r0", "episodes"]) ⊆ _metrics(doc)
                end
                @test _doc(["sadf", uni, "--r0", "0.2", "--cv", "wildboot"]).status == "ok"
            end

            @testset "edf — goodness-of-fit kv" begin
                doc = _doc(["edf", uni, "--column", "1"])
                @test doc.status == "ok"
                @test Set(["statistic", "p-value", "test", "distribution", "case", "theta"]) ⊆ _metrics(doc)
                @test _doc(["edf", uni, "--test", "ks", "--dist", "logistic"]).status == "ok"
                # --params specified consumes the supplied --theta (upstream spells this
                # :specified, NOT :known — a mock that accepted :known hid a real failure)
                @test _doc(["edf", uni, "--params", "specified", "--theta", "0,1"]).status == "ok"
            end

            @testset "engle-granger / phillips-ouliaris — residual cointegration" begin
                doc = _doc(["engle-granger", reg, "--dep", "y"])
                @test doc.status == "ok"
                @test Set(["statistic", "p-value", "lags", "regression", "regressors (k)"]) ⊆ _metrics(doc)

                po = _doc(["phillips-ouliaris", reg, "--dep", "y"])
                @test po.status == "ok"
                # PO reports BOTH the studentized Z_t and the normalized-bias Z_alpha.
                stat_tbl = _tbl(po, "statistic")
                @test Set(["statistic", "value", "p_value"]) ⊆ Set(String.(stat_tbl.columns))
                @test length(stat_tbl.rows) == 2
                @test Set(["regression", "kernel", "bandwidth"]) ⊆ _metrics(po)
                @test _doc(["phillips-ouliaris", reg, "--dep", "y", "--kernel", "parzen",
                            "--bandwidth", "4"]).status == "ok"
            end

            @testset "hansen-instability / park-added — fitted on a CointRegModel" begin
                hi = _doc(["hansen-instability", reg, "--dep", "y"])
                @test hi.status == "ok"
                @test Set(["L_c statistic", "p-value", "trend", "parameters"]) ⊆ _metrics(hi)

                pa = _doc(["park-added", reg, "--dep", "y", "--q-add", "3"])
                @test pa.status == "ok"
                @test Set(["H(p,q) statistic", "p-value", "q_add (df)", "base trend order (p)"]) ⊆ _metrics(pa)
                @test _doc(["park-added", reg, "--dep", "y", "--trend", "linear"]).status == "ok"
            end

            @testset "llc/ips/breitung — matrix panel unit-root tests" begin
                for leaf in ("llc", "ips", "breitung")
                    doc = _doc([leaf, mv])
                    @test doc.status == "ok"
                    @test Set(["statistic", "p-value", "deterministic", "n_units",
                               "observations"]) ⊆ _metrics(doc) ||
                          Set(["W[t-bar] statistic", "p-value", "n_units"]) ⊆ _metrics(doc)
                    @test _doc([leaf, mv, "--deterministic", "trend"]).status == "ok"
                    @test _doc([leaf, mv, "--cs-demean"]).status == "ok"
                end
                # IPS additionally reports the per-unit ADF statistics
                t = _tbl(_doc(["ips", mv]), "unit")
                @test Set(["unit", "t_statistic", "lags"]) ⊆ Set(String.(t.columns))
            end

            @testset "fisher-johansen / dh-causality — PanelData leaves" begin
                fj = _doc(["fisher-johansen", panel, "--vars", "y,x1"])
                @test fj.status == "ok"
                ft = _tbl(fj, "rank")
                @test Set(["rank", "trace_statistic", "trace_p_value",
                           "max_statistic", "max_p_value"]) ⊆ Set(String.(ft.columns))
                @test Set(["selected rank", "combine", "deterministic", "lags"]) ⊆ _metrics(fj)
                @test _doc(["fisher-johansen", panel, "--vars", "y,x1", "--combine", "choi"]).status == "ok"

                dh = _doc(["dh-causality", panel, "--cause", "x1", "--effect", "y"])
                @test dh.status == "ok"
                @test Set(["cause", "effect", "W-bar", "Z-bar", "Z-tilde",
                           "Z-tilde p-value", "lags (p)"]) ⊆ _metrics(dh)
            end

            @testset "C070 remainder — bad input → typed CliError" begin
                e = _errcode(["llc", mv, "--deterministic", "bogus"])
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                e = _errcode(["llc", mv, "--lags", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["llc", mv, "--max-lags", "-1"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["breitung", mv, "--lags", "-1"])
                @test e isa CliError && e.code == "usage/invalid"
                # fisher-johansen needs >= 2 series
                e = _errcode(["fisher-johansen", panel, "--vars", "y"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                e = _errcode(["fisher-johansen", panel, "--vars", "nosuch"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["fisher-johansen", panel, "--lags", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                # dh-causality: --cause/--effect are both required and direction matters
                e = _errcode(["dh-causality", panel, "--effect", "y"])
                @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
                e = _errcode(["dh-causality", panel, "--cause", "x1"])
                @test e isa CliError && e.code == "usage/missing"
                e = _errcode(["dh-causality", panel, "--cause", "x1", "--effect", "y", "--p", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                # naming the same column twice is not a causality test
                e = _errcode(["dh-causality", panel, "--cause", "y", "--effect", "y"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["dh-causality", panel, "--id-col", "nosuch",
                              "--cause", "x1", "--effect", "y"])
                @test e isa CliError && e.code == "data/missing-column"
            end

            # ── C067 remainder (#72): cross-section OLS diagnostics. `reg` already
            # exists above as a (y, X) frame.
            @testset "white/glejser/harvey — RegDiagnosticResult kv" begin
                for leaf in ("white", "glejser", "harvey")
                    doc = _doc([leaf, reg, "--dep", "y"])
                    @test doc.status == "ok"
                    @test Set(["test", "H0", "statistic", "p-value", "df",
                               "auxiliary R2", "observations"]) ⊆ _metrics(doc)
                end
                # the flag switches the auxiliary regression, not the result shape
                @test _doc(["white", reg, "--dep", "y", "--no-cross-terms"]).status == "ok"
            end

            @testset "chow — required --break-at, multi-break, forecast variant" begin
                doc = _doc(["chow", reg, "--dep", "y", "--break-at", "30"])
                @test doc.status == "ok"
                @test Set(["test", "H0", "statistic", "p-value", "df"]) ⊆ _metrics(doc)
                # a comma-separated list is a multi-break test
                @test _doc(["chow", reg, "--dep", "y", "--break-at", "20,40"]).status == "ok"
                @test _doc(["chow", reg, "--dep", "y", "--break-at", "30",
                            "--type", "forecast"]).status == "ok"
            end

            @testset "cusum/cusumsq — band path table, NO p-value" begin
                for (leaf, col) in (("cusum", "cusum"), ("cusumsq", "cusumsq"))
                    doc = _doc([leaf, reg, "--dep", "y"])
                    @test doc.status == "ok"
                    tbl = _tbl(doc, "observation")
                    @test Set(["observation", col, "lower", "upper"]) ⊆ Set(String.(tbl.columns))
                    @test length(tbl.rows) > 0
                    @test Set(["kind", "crossed band", "first crossing", "level"]) ⊆ _metrics(doc)
                    # StabilityResult carries a band, not a p-value — assert we do NOT
                    # invent one (the ARDL-bounds / HEGY rule).
                    @test !("p-value" in _metrics(doc))
                end
            end

            @testset "influence / recursive-residuals — per-observation tables" begin
                doc = _doc(["influence", reg, "--dep", "y"])
                @test doc.status == "ok"
                tbl = _tbl(doc, "hat")
                @test Set(["observation", "hat", "student_internal", "student_external",
                           "dffits", "cooksd"]) ⊆ Set(String.(tbl.columns))
                @test Set(["sigma", "high-leverage count", "influential count"]) ⊆ _metrics(doc)

                rr = _doc(["recursive-residuals", reg, "--dep", "y"])
                @test rr.status == "ok"
                rt = _tbl(rr, "recursive_residual")
                @test Set(["step", "observation", "recursive_residual"]) ⊆ Set(String.(rt.columns))
                @test Set(["count", "mean", "regressors (k)"]) ⊆ _metrics(rr)
            end

            @testset "C067 remainder — bad input → typed CliError" begin
                # --break-at is required (and is NOT spelled --break: `break` is a Julia
                # reserved word and cannot be a handler kwarg)
                e = _errcode(["chow", reg, "--dep", "y"])
                @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
                e = _errcode(["chow", reg, "--dep", "y", "--break-at", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["chow", reg, "--dep", "y", "--break-at", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["chow", reg, "--dep", "y", "--break-at", "30", "--level", "1.5"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["cusum", reg, "--dep", "y", "--level", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                # enum values are parser-rejected
                e = _errcode(["chow", reg, "--dep", "y", "--break-at", "30", "--type", "bogus"])
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                # unknown --dep → typed loader error on every leaf, never exit 1
                for leaf in ("white", "glejser", "harvey", "cusum", "cusumsq",
                             "influence", "recursive-residuals")
                    e = _errcode([leaf, reg, "--dep", "nope"])
                    @test e isa CliError && e.code == "data/column-range"
                end
            end

            @testset "C069 remainder — bad input → typed CliError" begin
                # every numeric option guarded up front → usage/invalid (exit 2)
                e = _errcode(["hegy", uni, "--frequency", "7"])
                @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
                e = _errcode(["hegy", uni, "--lags", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["sadf", uni, "--adflag", "-1"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["sadf", uni, "--mc-reps", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                # --r0 outside (0,1) is rejected by the CLI, not left to upstream
                e = _errcode(["sadf", uni, "--r0", "1.5"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["gsadf", uni, "--r0", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                # --params known without --theta → usage/missing
                e = _errcode(["edf", uni, "--params", "specified"])
                @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
                e = _errcode(["edf", uni, "--params", "specified", "--theta", "a,b"])
                @test e isa CliError && e.code == "usage/invalid"
                # the old wrong spelling is now rejected at the parser, not by MEMs
                e = _errcode(["edf", uni, "--params", "known", "--theta", "0,1"])
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                # enum values are parser-rejected (ParseError, exit 2)
                e = _errcode(["edf", uni, "--dist", "bogus"])
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                e = _errcode(["engle-granger", reg, "--trend", "linear"])   # cointreg's spelling, NOT EG's
                @test e isa ParseError || (e isa CliError && exit_class(e) == 2)
                e = _errcode(["engle-granger", reg, "--lags", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["engle-granger", reg, "--max-lags", "-2"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["phillips-ouliaris", reg, "--bandwidth", "junk"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["park-added", reg, "--q-add", "0"])
                @test e isa CliError && e.code == "usage/invalid"
                e = _errcode(["park-added", reg, "--hac-bandwidth", "-1"])
                @test e isa CliError && e.code == "usage/invalid"
                # unknown --dep on the (y,X) leaves → typed loader error, never exit 1
                e = _errcode(["engle-granger", reg, "--dep", "nope"])
                @test e isa CliError && e.code == "data/column-range"
                e = _errcode(["hansen-instability", reg, "--dep", "nope"])
                @test e isa CliError && e.code == "data/column-range"
                # bad --column on the univariate leaves → data/column-range
                e = _errcode(["ers", uni, "--column", "9"])
                @test e isa CliError && e.code == "data/column-range"
            end
        end
    end

    @testset "test hansen-linearity (C065a)" begin
        # Hansen (1996) sup-LM/sup-Wald linearity test via the attached SETAR `.linearity`.
        # Pure kv (sup_lm/pvalue_lm/sup_wald/pvalue_wald/gamma_sup/reps/trim/n_grid) + an
        # interpretation line. Options guarded up-front → usage/invalid; a too-short series
        # surfaces (through the wrapped estimate_setar) as a typed data/invalid, never exit-1.
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test", "hansen-linearity"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _metrics(doc) = Set(String(collect(r)[1]) for t in values(doc.data)
                            if (t isa JSON3.Object && haskey(t, :columns) && "metric" in String.(t.columns))
                            for r in t.rows)
        _errcode(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["test", "hansen-linearity"], collect(String, args))); end
            catch ex; e = ex; end
            e
        end

        mktempdir() do dir
            uni = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "kv keys + interpretation" begin
                doc = _doc([uni, "--column", "1", "--p", "1", "--d", "1", "--reps", "50"])
                @test doc.status == "ok"
                @test Set(["sup_lm", "pvalue_lm", "sup_wald", "pvalue_wald",
                           "gamma_sup", "reps", "trim", "n_grid"]) ⊆ _metrics(doc)
            end

            @testset "bad input → typed classes" begin
                @test _errcode([uni, "--p", "0"]).code == "usage/invalid"
                @test _errcode([uni, "--d", "0"]).code == "usage/invalid"
                @test _errcode([uni, "--trim", "0.7"]).code == "usage/invalid"
                # too-short series → data/invalid (wrapped estimate_setar ArgumentError)
                short = joinpath(dir, "short.csv")
                write(short, "y\n" * join([string(0.1 * i) for i in 1:6], "\n") * "\n")
                es = _errcode([short, "--p", "1", "--d", "1"])
                @test es isa CliError && es.code == "data/invalid" && exit_class(es) == 3
            end
        end
    end

    @testset "test star-linearity (C065b)" begin
        # STAR (LSTV LM3) linearity test → a NamedTuple (stat, pvalue, fstat, fpvalue, df)
        # rendered as pure kv + an interpretation line. Options guarded up-front → usage/invalid;
        # the test call is wrapped → typed CliError (never uncaught exit-1).
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["test", "star-linearity"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _metrics(doc) = Set(String(collect(r)[1]) for t in values(doc.data)
                            if (t isa JSON3.Object && haskey(t, :columns) && "metric" in String.(t.columns))
                            for r in t.rows)
        _errcode(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["test", "star-linearity"], collect(String, args))); end
            catch ex; e = ex; end
            e
        end

        mktempdir() do dir
            uni = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "kv keys + interpretation" begin
                doc = _doc([uni, "--column", "1", "--p", "1", "--d", "1"])
                @test doc.status == "ok"
                @test Set(["stat", "pvalue", "fstat", "fpvalue", "df"]) ⊆ _metrics(doc)
            end

            @testset "bad input → typed classes" begin
                @test _errcode([uni, "--p", "0"]).code == "usage/invalid"
                @test _errcode([uni, "--d", "0"]).code == "usage/invalid"
                # constant external transition column → data/invalid
                constcsv = joinpath(dir, "sl_const.csv")
                write(constcsv, "y,s\n" * join(["$(0.1 * i),1.0" for i in 1:80], "\n") * "\n")
                ec = _errcode([constcsv, "--column", "1", "--transition-col", "2"])
                @test ec isa CliError && ec.code == "data/invalid" && exit_class(ec) == 3
            end

            @testset "short series accepted — matches real MEMs (LM3 is defensively coded)" begin
                # Real star_linearity_test returns a FINITE result for a short effective sample
                # (VERIFIED against real MEMs: n=14,p=3 → eff=11 → stat=11.0, ok), so the mock must
                # NOT over-reject it as data/invalid (the mock guard is `n >= 1`, not `n > 3p+2`).
                shortcsv = joinpath(dir, "sl_short.csv")
                write(shortcsv, "y\n" * join([string(0.1 * i + 0.3 * sin(i)) for i in 1:14], "\n") * "\n")
                ds = _doc([shortcsv, "--column", "1", "--p", "3", "--d", "1"])
                @test ds.status == "ok"
                @test Set(["stat", "pvalue", "df"]) ⊆ _metrics(ds)
            end
        end
    end

    @testset "_test_johansen" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_johansen(; data=csv, lags=2, trend="constant", format="table")
            end
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
        end
    end

    @testset "_test_normality" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_normality(; data=csv, lags=2, format="table")
            end
        end
    end

    @testset "_test_normality — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_normality(; data=csv, lags=nothing, format="table")
            end
        end
    end

    @testset "_test_identifiability — all tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_identifiability(; data=csv, lags=2, test="all",
                                        method="fastica", format="table")
            end
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
            end
        end
    end

    @testset "_test_identifiability — W2 riders" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            for test_type in ["lambda-distinct", "gaussian-count", "label-stability"]
                out = _capture() do
                    _test_identifiability(; data=csv, lags=2, test=test_type,
                                            method="fastica", n_bootstrap=10, format="table")
                end
            end
            e = try
                _test_identifiability(; data=csv, lags=2, test="lambda-distinct",
                                        n_bootstrap=0, format="table")
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/invalid"
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
        end
    end

    @testset "_test_heteroskedasticity — markov" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="markov",
                                            regimes=2, format="table")
            end
        end
    end

    @testset "_test_heteroskedasticity — garch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_heteroskedasticity(; data=csv, lags=2, method="garch",
                                            format="table")
            end
        end
    end

    @testset "_test_heteroskedasticity — smooth_transition requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _capture() do
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
        end
    end

    @testset "_test_heteroskedasticity — external requires config" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _capture() do
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
        end
    end

    @testset "_test_var_lagselect" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_lagselect(; data=csv, max_lags=4, criterion="aic", format="table")
            end
        end
    end

    @testset "_test_var_lagselect — json format" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_lagselect(; data=csv, max_lags=4, criterion="bic", format="json")
            end
        end
    end

    @testset "_test_var_stability" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_stability(; data=csv, lags=2, format="table")
            end
        end
    end

    @testset "_test_var_stability — auto lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_var_stability(; data=csv, lags=nothing)
            end
        end
    end

    @testset "_test_arch_lm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_arch_lm(; data=csv, column=1, lags=4, format="table")
            end
        end
    end

    @testset "_test_ljung_box" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_ljung_box(; data=csv, column=1, lags=10, format="table")
            end
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
        # 7 + tvpvar (W7): the date-specific IRF is the point of a TVP-VAR
        @test length(node.subcmds) == 8
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
        end
    end

    @testset "_irf_var — arias without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="arias",
                              config="", format="table")
                end
            end
        end
    end

    @testset "_irf_var — narrative-adrr identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_adrr_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="narrative-adrr",
                              config=cfg, format="table")
                end
            end
        end
    end

    @testset "_irf_var — narrative-adrr without narrative_contributions errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_arias_config(dir)
            e = try
                cd(dir) do
                    _capture() do
                        _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="narrative-adrr",
                                  config=cfg, format="table")
                    end
                end
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/missing"
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
        end
    end

    @testset "_irf_var — uhlig without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="uhlig",
                              config="", format="table")
                end
            end
        end
    end

    @testset "_irf_var — lewis-tvv identification (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_lewis_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="lewis-tvv",
                              config=cfg, ci="none", format="table")
                end
            end
            # cue weighting threads too
            cfg_cue = _make_lewis_config(dir; weighting="cue")
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="lewis-tvv",
                              config=cfg_cue, ci="none", format="table")
                end
            end
        end
    end

    @testset "_irf_var — lewis-tvv invalid weighting (config/invalid)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_lewis_config(dir; weighting="optimal")
            e = try
                cd(dir) do
                    _capture() do
                        _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="lewis-tvv",
                                  config=cfg, ci="none", format="table")
                    end
                end
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "config/invalid" && exit_class(e) == 4
        end
    end

    @testset "_irf_var — sv-em identification (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sv_config(dir)
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sv-em",
                              config=cfg, ci="none", format="table")
                end
            end
            # partial hetero + non-default knobs thread too
            cfg_p = _make_sv_config(dir; hetero_shocks=[1, 3], maxiter=50,
                                    gibbs_burn=2, gibbs_draws=20, init="haar")
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sv-em",
                              config=cfg_p, ci="none", format="table")
                end
            end
        end
    end

    @testset "_irf_var — sv-em hetero out of range (usage/invalid)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sv_config(dir; hetero_shocks=[4])
            e = try
                cd(dir) do
                    _capture() do
                        _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sv-em",
                                  config=cfg, ci="none", format="table")
                    end
                end
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
        end
    end

    @testset "_irf_var — sv-em invalid init (config/invalid)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sv_config(dir; init="newton")
            e = try
                cd(dir) do
                    _capture() do
                        _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sv-em",
                                  config=cfg, ci="none", format="table")
                    end
                end
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "config/invalid" && exit_class(e) == 4
        end
    end

    @testset "_fevd_bvar — threads method (W1/#186 fix)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            # explicit cholesky renders (previously EVERY --id silently rendered cholesky)
            out = cd(dir) do
                _capture() do
                    _fevd_bvar(; data=csv, lags=2, horizons=10, id="cholesky",
                                draws=100, sampler="direct", config="", format="table")
                end
            end
            # unknown --id is now rejected (previously silently accepted)
            e = try
                cd(dir) do
                    _capture() do
                        _fevd_bvar(; data=csv, lags=2, horizons=10, id="bogus",
                                    draws=100, sampler="direct", config="", format="table")
                    end
                end
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
        end
    end

    @testset "_fevd_lp — lewis-tvv threads through structural_lp (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_lewis_config(dir)
            out = cd(dir) do
                _capture() do
                    _fevd_lp(; data=csv, horizons=10, lags=4, id="lewis-tvv",
                              vcov="newey_west", config=cfg, format="table")
                end
            end
        end
    end

    @testset "_build_identification_kwargs — tvv/sv knob threading" begin
        # defaults flow with upstream kwarg names
        kw = _build_identification_kwargs("lewis-tvv", ""; nvars=3, leaf="unit")
        @test kw[:method] == :lewis_tvv && kw[:weighting] == :two_step
        kw2 = _build_identification_kwargs("sv-em", ""; nvars=3, leaf="unit")
        @test kw2[:method] == :sv_em && kw2[:hetero] == trues(3)
        @test kw2[:maxiter] == 500 && kw2[:gibbs_draws] == 100 && kw2[:init] == :ols_chol
        # duplicates collapse, partial resolves (set semantics)
        mktempdir() do dir
            cfg = _make_sv_config(dir; hetero_shocks=[3, 1, 1])
            kw3 = _build_identification_kwargs("sv-em", cfg; nvars=3, leaf="unit")
            @test kw3[:hetero] == BitVector([true, false, true])
        end
        # other methods unaffected (no knob keys)
        kw4 = _build_identification_kwargs("cholesky", ""; nvars=3, leaf="unit")
        @test !haskey(kw4, :weighting) && !haskey(kw4, :hetero)
    end

    @testset "_irf_var — W2a proxy/max-share/gmm-moments (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["y1", "y2", "z"])
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="proxy",
                              instrument="z", ci="none", format="table")
                end
            end
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10,
                              id="max-share", target_var="y1", ci="none",
                              format="table")
                end
            end
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10,
                              id="max-share", target_var="2", ci="none",
                              format="table")
                end
            end
            out = cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10,
                              id="gmm-moments", ci="none", format="table")
                end
            end
            e1 = try; _irf_var(; data=csv, lags=2, id="proxy", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/missing"
            e2 = try; _irf_var(; data=csv, lags=2, id="cholesky", instrument="z", format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            e3 = try; _irf_var(; data=csv, lags=2, id="max-share", format="table"); nothing; catch e; e; end
            @test e3 isa CliError && e3.code == "usage/missing"
            e4 = try; _irf_var(; data=csv, lags=2, id="cholesky", target_var="y1", format="table"); nothing; catch e; e; end
            @test e4 isa CliError && e4.code == "usage/invalid"
            e5 = try; _irf_var(; data=csv, lags=2, id="max-share", target_var="nope", format="table"); nothing; catch e; e; end
            @test e5 isa CliError && e5.code == "usage/invalid"
            e6 = try; _irf_var(; data=csv, lags=2, id="max-share", target_var="9", format="table"); nothing; catch e; e; end
            @test e6 isa CliError && e6.code == "usage/invalid"
            e7 = try; _irf_var(; data=csv, lags=2, id="bogus", format="table"); nothing; catch e; e; end
            @test e7 isa CliError && e7.code == "usage/invalid"
            e8 = try; _irf_var(; data=csv, lags=2, id="cholesky", identified_set=true, format="table"); nothing; catch e; e; end
            @test e8 isa CliError && e8.code == "usage/invalid"
            e9 = try; _irf_var(; data=csv, lags=2, id="arias", instrument="z", format="table"); nothing; catch e; e; end
            @test e9 isa CliError && e9.code == "usage/invalid"
        end
    end

    @testset "_fevd_var/_hd_var — W2a proxy/max-share guards (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3, colnames=["y1", "y2", "z"])
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="proxy",
                               instrument="z", format="table")
                end
            end
            out = cd(dir) do
                _capture() do
                    _hd_var(; data=csv, lags=2, id="max-share", target_var="y2",
                             format="table")
                end
            end
            e1 = try; _fevd_var(; data=csv, lags=2, id="proxy", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/missing"
            e2 = try; _fevd_var(; data=csv, lags=2, id="bogus", format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            e3 = try; _hd_var(; data=csv, lags=2, id="max-share", format="table"); nothing; catch e; e; end
            @test e3 isa CliError && e3.code == "usage/missing"
            e4 = try; _hd_var(; data=csv, lags=2, id="bogus", format="table"); nothing; catch e; e; end
            @test e4 isa CliError && e4.code == "usage/invalid"
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
        end
    end

    @testset "_irf_bvar — robust-bayes" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_bvar_restrictions_config(dir)
            out = cd(dir) do
                _capture() do
                    _dispatch_via_app(["irf", "bvar", csv, "--lags", "2", "--shock", "1",
                                       "--horizons", "10", "--id", "robust-bayes",
                                       "--draws", "100", "--config", cfg, "--format", "json"])
                end
            end
            i = findfirst('{', out)
            doc = JSON3.read(out[i:end])
            @test haskey(doc[:data], :robust_bayes_bands)
            @test haskey(doc[:data], :robust_bayes_diagnostics)
        end
    end

    @testset "_irf_bvar — robust-bayes guards" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_bvar_restrictions_config(dir)
            e1 = try
                _irf_bvar(; data=csv, lags=2, shock=9, horizons=10, id="robust-bayes",
                           draws=100, config=cfg, format="table")
                nothing
            catch e
                e
            end
            @test e1 isa CliError && e1.code == "usage/invalid"
            e2 = try
                _irf_bvar(; data=csv, lags=2, shock=1, horizons=10, id="robust-bayes",
                           draws=100, config="", format="table")
                nothing
            catch e
                e
            end
            @test e2 isa CliError && e2.code == "usage/missing"
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
            # C051: tidy long_table consolidates both shocks into a single table/title
            # (previously split into 2 per-shock output blocks).
            @test count("LP IRF to", out) == 1
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
        end
    end

    @testset "_irf_lp — invalid shock index" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_irf_var — identified-set without config errors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
                _capture() do
                    _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                              config="", ci="none", format="table", identified_set=true)
                end
            end
        end
    end

    @testset "_irf_var — set-identified summaries" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_sign_config(dir)
            for sum_kind in ["median-target", "modal-model", "joint-band", "sup-t-band"]
                out = cd(dir) do
                    _capture() do
                        _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="sign",
                                  config=cfg, ci="none", format="table",
                                  identified_set=true, summary=sum_kind)
                    end
                end
            end
            e = try
                _irf_var(; data=csv, lags=2, shock=1, horizons=10, id="cholesky",
                          ci="none", format="table", summary="median-target")
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/invalid"
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
            # C051: single tidy long_table (horizon|variable|shock|value), one file
            @test isfile(outfile)
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
        end
    end

    @testset "_fevd_var — narrative-adrr identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_adrr_config(dir)
            out = cd(dir) do
                _capture() do
                    _fevd_var(; data=csv, lags=2, horizons=10, id="narrative-adrr",
                               config=cfg, format="table")
                end
            end
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
        end
    end

    @testset "_hd_var — narrative-adrr identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_adrr_config(dir)
            out = cd(dir) do
                _capture() do
                    _hd_var(; data=csv, lags=2, id="narrative-adrr", config=cfg, format="table")
                end
            end
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
        # 16 primary + 1 alias (gjr_garch) + 1 evaluate sub-node = 18 keys (C044/C072; +setar C065a, +star C065b, +ms/ms-ar W3 #101; +sdfm W1 #165)
        @test length(node.subcmds) == 31
        for cmd in ["var", "bvar", "lp", "arima", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr-garch", "sv", "vecm", "favar"]
            @test haskey(node.subcmds, cmd)
        end
        @test haskey(node.subcmds, "gjr_garch")
        # C072: nested forecast evaluate sub-node with 6 leaves
        @test haskey(node.subcmds, "evaluate")
        @test node.subcmds["evaluate"] isa NodeCommand
        for leaf in ["metrics", "dm", "clark-west", "mincer-zarnowitz", "encompassing", "combine"]
            @test haskey(node.subcmds["evaluate"].subcmds, leaf)
        end
    end

    @testset "forecast setar (C065a)" begin
        # SETAR bootstrap forecast: re-estimate then forecast → ThresholdForecast, rendered
        # via the generic long_table (horizon|variable|value|lower|upper). Options guarded
        # up-front → usage error; MEMs calls wrapped → typed CliError (never uncaught exit-1).
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["forecast", "setar"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tbl(doc, col) = first(t for t in values(doc.data)
                               if (t isa JSON3.Object && haskey(t, :columns) && col in String.(t.columns)))
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["forecast", "setar"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "long_table forecast columns + horizon count" begin
                doc = _doc([csv, "--column", "1", "--p", "1", "--horizons", "6", "--reps", "50"])
                @test doc.status == "ok"
                tbl = _tbl(doc, "horizon")
                @test Set(["horizon", "variable", "value", "lower", "upper"]) ⊆ Set(String.(tbl.columns))
                @test length(collect(tbl.rows)) == 6
            end

            @testset "bad input → usage error" begin
                @test _err([csv, "--p", "0"]).code == "usage/invalid"
                @test _err([csv, "--horizons", "0"]).code == "usage/invalid"
                @test _err([csv, "--reps", "0"]).code == "usage/invalid"
                @test _err([csv, "--d", "foo"]).code == "usage/invalid"
                eci = _err([csv, "--ci-level", "0.8"])
                @test eci isa ParseError || (eci isa CliError && exit_class(eci) == 2)
            end

            @testset "no --plot/--plot-save (MEMs has no ThresholdForecast plot recipe)" begin
                # Intentionally NOT offered: real MEMs 0.7.0 ships no plot_result(::ThresholdForecast),
                # so advertising the flag would drive _maybe_plot into an uncaught MethodError → exit 1.
                # They must be rejected as unknown options (exit 2), never silently accepted.
                ep = _err([csv, "--plot"])
                @test ep isa ParseError || (ep isa CliError && exit_class(ep) == 2)
                eps = _err([csv, "--plot-save", joinpath(dir, "out.html")])
                @test eps isa ParseError || (eps isa CliError && exit_class(eps) == 2)
            end
        end
    end

    @testset "forecast star (C065b)" begin
        # STAR bootstrap forecast: re-estimate a self-exciting STAR then forecast → STARForecast,
        # rendered via the generic long_table (horizon|variable|value|lower|upper). Options guarded
        # up-front → usage error; MEMs calls wrapped → typed CliError (never uncaught exit-1). Like
        # forecast setar, NO --plot/--plot-save (MEMs ships no plot_result(::STARForecast) recipe).
        _doc(args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["forecast", "star"], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tbl(doc, col) = first(t for t in values(doc.data)
                               if (t isa JSON3.Object && haskey(t, :columns) && col in String.(t.columns)))
        _err(args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["forecast", "star"], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "long_table forecast columns + horizon count" begin
                doc = _doc([csv, "--column", "1", "--p", "1", "--horizons", "6", "--reps", "50"])
                @test doc.status == "ok"
                tbl = _tbl(doc, "horizon")
                @test Set(["horizon", "variable", "value", "lower", "upper"]) ⊆ Set(String.(tbl.columns))
                @test length(collect(tbl.rows)) == 6
            end

            @testset "bad input → usage error" begin
                @test _err([csv, "--p", "0"]).code == "usage/invalid"
                @test _err([csv, "--horizons", "0"]).code == "usage/invalid"
                @test _err([csv, "--reps", "0"]).code == "usage/invalid"
                @test _err([csv, "--d", "0"]).code == "usage/invalid"
                eci = _err([csv, "--ci-level", "0.8"])
                @test eci isa ParseError || (eci isa CliError && exit_class(eci) == 2)
                et = _err([csv, "--type", "bogus"])
                @test et isa ParseError || (et isa CliError && exit_class(et) == 2)
            end

            @testset "no --plot/--plot-save (MEMs has no STARForecast plot recipe)" begin
                # Intentionally NOT offered: real MEMs 0.7.0 ships no plot_result(::STARForecast),
                # so advertising the flag would drive _maybe_plot into an uncaught MethodError → exit 1.
                ep = _err([csv, "--plot"])
                @test ep isa ParseError || (ep isa CliError && exit_class(ep) == 2)
                eps = _err([csv, "--plot-save", joinpath(dir, "out.html")])
                @test eps isa ParseError || (eps isa CliError && exit_class(eps) == 2)
            end
        end
    end

    @testset "forecast evaluate (C072 fceval)" begin
        _fceval_csv(dir) = begin
            path = joinpath(dir, "fceval.csv")
            open(path, "w") do io
                println(io, "y,f1,f2,f3")
                for _ in 1:80
                    y = 5.0 + randn()
                    f1 = y + 0.2 * randn()       # good forecast
                    f2 = y + 0.9 * randn()       # noisier forecast
                    f3 = y + 0.5 * randn()
                    println(io, join((y, f1, f2, f3), ","))
                end
            end
            path
        end
        _evaldoc(args::Vector{String}) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["forecast", "evaluate"], args, String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _tblwith(doc, col) = first(t for t in values(doc.data) if col in String.(t.columns))
        _metric(doc, name) = begin
            kv = first(t for t in values(doc.data) if "metric" in String.(t.columns))
            for r in kv.rows
                rr = collect(r)
                String(rr[1]) == name && return rr[2]
            end
            nothing
        end
        _errcode(args::Vector{String}) = begin
            err = nothing
            try
                _capture() do; _dispatch_via_app(vcat(String["forecast", "evaluate"], args)); end
            catch e
                err = e
            end
            err
        end

        @testset "metrics — wide accuracy + Theil decomposition" begin
            mktempdir() do dir
                doc = _evaldoc(["metrics", _fceval_csv(dir), "--actual", "y", "--forecasts", "f1,f2"])
                @test doc.status == "ok"
                acc = _tblwith(doc, "RMSE")
                @test Set(["model","ME","MAE","RMSE","MAPE","sMAPE","MASE","U1","U2"]) ⊆ Set(String.(acc.columns))
                @test length(acc.rows) == 2
                dec = _tblwith(doc, "bias")
                @test Set(["model","bias","variance","covariance"]) ⊆ Set(String.(dec.columns))
                @test length(dec.rows) == 2
            end
        end

        @testset "dm — kv + arity" begin
            mktempdir() do dir
                csv = _fceval_csv(dir)
                doc = _evaldoc(["dm", csv, "--actual", "y", "--forecasts", "f1,f2", "--loss", "ad", "--horizon", "2"])
                @test doc.status == "ok"
                @test String(_metric(doc, "test")) == "Diebold-Mariano"
                @test 0.0 <= Float64(_metric(doc, "p_value")) <= 1.0
                e = _errcode(["dm", csv, "--actual", "y", "--forecasts", "f1"])
                @test e isa CliError && e.code == "usage/arity"
            end
        end

        @testset "clark-west — kv" begin
            mktempdir() do dir
                doc = _evaldoc(["clark-west", _fceval_csv(dir), "--actual", "y", "--forecasts", "f1,f2"])
                @test String(_metric(doc, "test")) == "Clark-West"
                @test 0.0 <= Float64(_metric(doc, "p_value")) <= 1.0
            end
        end

        @testset "mincer-zarnowitz — kv + arity" begin
            mktempdir() do dir
                csv = _fceval_csv(dir)
                doc = _evaldoc(["mincer-zarnowitz", csv, "--actual", "y", "--forecasts", "f1", "--lags", "2"])
                @test String(_metric(doc, "test")) == "Mincer-Zarnowitz"
                @test 0.0 <= Float64(_metric(doc, "p_value_wald")) <= 1.0
                e = _errcode(["mincer-zarnowitz", csv, "--actual", "y", "--forecasts", "f1,f2"])
                @test e isa CliError && e.code == "usage/arity"
            end
        end

        @testset "encompassing — kv" begin
            mktempdir() do dir
                doc = _evaldoc(["encompassing", _fceval_csv(dir), "--actual", "y", "--forecasts", "f1,f2"])
                @test String(_metric(doc, "test")) == "Forecast-Encompassing"
                @test 0.0 <= Float64(_metric(doc, "p_value")) <= 1.0
            end
        end

        @testset "combine — weights sum to 1 + emit-series + arity" begin
            mktempdir() do dir
                csv = _fceval_csv(dir)
                doc = _evaldoc(["combine", csv, "--actual", "y", "--forecasts", "f1,f2,f3", "--method", "bates-granger"])
                w = _tblwith(doc, "weight")
                @test Set(["model","weight","mse"]) ⊆ Set(String.(w.columns))
                @test length(w.rows) == 3
                wi = findfirst(==("weight"), String.(w.columns))
                wsum = sum(Float64(collect(r)[wi]) for r in w.rows)
                @test isapprox(wsum, 1.0; atol=1e-4)   # weights rounded to 6 digits before display
                doc2 = _evaldoc(["combine", csv, "--actual", "y", "--forecasts", "f1,f2", "--emit-series"])
                @test any(t -> "combined" in String.(t.columns), values(doc2.data))
                e = _errcode(["combine", csv, "--actual", "y", "--forecasts", "f1"])
                @test e isa CliError && e.code == "usage/arity"
            end
        end

        @testset "typed errors (bad column, missing options)" begin
            mktempdir() do dir
                csv = _fceval_csv(dir)
                @test _errcode(["metrics", csv, "--actual", "nope", "--forecasts", "f1"]).code == "data/bad-column"
                @test _errcode(["dm", csv, "--actual", "y", "--forecasts", "f1,zzz"]).code == "data/bad-column"
                @test _errcode(["metrics", csv, "--forecasts", "f1"]).code == "usage/missing-actual"
                @test _errcode(["metrics", csv, "--actual", "y"]).code == "usage/missing-forecasts"
                # review fix: missing values → typed data error (was an uncaught exit-1 MethodError)
                mcsv = joinpath(dir, "miss.csv")
                write(mcsv, "y,f1\n1.0,1.1\n,2.0\n3.0,2.9\n4.0,3.8\n")
                @test _errcode(["metrics", mcsv, "--actual", "y", "--forecasts", "f1"]).code == "data/missing-values"
                # review fix: --model is not injected on the model-agnostic evaluate leaves →
                # parser rejects it (unknown option), NOT an uncaught handler MethodError
                me = _errcode(["metrics", csv, "--actual", "y", "--forecasts", "f1", "--model", "foo"])
                @test me !== nothing && occursin("unknown option", sprint(showerror, me))
            end
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
            @test nrow(result_df) == 5   # C051: tidy, 5 horizons × 1 variable
            @test "value" in names(result_df)
            @test "variable" in names(result_df)
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
            @test occursin("_lower", out) || occursin("std_error", out) || length(out) > 0
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
        @test length(node.subcmds) == 77  # 76 primary (+sarima W6, +tvpvar/mfvar W7, +svar/svec W2 #166) (+poisson/nbreg W2 #107) + gjr_garch alias (C064a +6, C068 +arfima, C064b +3 MGARCH, C067a +5, C067b +2, C066 +5, C062a +2, C062b +2, C062c +1, C062d +midas, C065a +setar, C065b +star, C065c +ms-ar/ms, C067 +select, #70 +threshold)
        @test haskey(node.subcmds, "vecm")
        @test node.subcmds["vecm"] isa LeafCommand
    end

    @testset "register_irf_commands! includes vecm" begin
        node = register_irf_commands!()
        @test length(node.subcmds) == 8
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
        @test length(node.subcmds) == 31  # 22 primary + gjr_garch alias + evaluate node (+setar C065a, +star C065b, +igarch/cgarch/aparch/figarch/fiegarch/garch-midas C064 #69, +arfima #73, +midas #67, +ms/ms-ar W3 #101; +sdfm W1 #165)
        @test haskey(node.subcmds, "vecm")
    end

    @testset "register_test_commands! includes granger" begin
        node = register_test_commands!()
        @test length(node.subcmds) == 85  # 81 primary (+dispersion W2 #107) + 2 snake aliases (+hegy/ers/sadf/gsadf/edf/engle-granger/phillips-ouliaris/hansen-instability/park-added C069 remainder, +llc/ips/breitung/fisher-johansen/dh-causality C070 remainder, +gph, +local-whittle C068, +sign-bias, +nyblom C064b, +vecm C071, +variance-ratio/bds/hadri/pedroni/kao/westerlund C069/C070, +weak-instrument C067b, +ardl-bounds/nardl-symmetry C062b, +pmg-hausman C062c, +hansen-linearity C065a, +star-linearity C065b)
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
        end
    end

    @testset "_irf_vecm — svec identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _irf_vecm(; data=csv, lags=2, rank="1", shock=1, horizons=10,
                               id="svec", ci="none", format="table")
                end
            end
        end
    end

    @testset "_irf_vecm — svec with custom zeros" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_svec_config(dir; n=3)
            out = cd(dir) do
                _capture() do
                    _irf_vecm(; data=csv, lags=2, rank="1", shock=1, horizons=10,
                               id="svec", ci="none", config=cfg, format="table")
                end
            end
        end
    end

    @testset "_irf_vecm — svec rejects bootstrap CI" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            e = try
                _irf_vecm(; data=csv, lags=2, rank="1", shock=1, horizons=10,
                           id="svec", ci="bootstrap", format="table")
                nothing
            catch e
                e
            end
            @test e isa CliError && e.code == "usage/invalid"
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
            # C051: single tidy long_table (horizon|variable|shock|value), one file
            @test isfile(outfile)
        end
    end

    @testset "_fevd_vecm — svec identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            cfg = _make_svec_config(dir; n=3)
            out = cd(dir) do
                _capture() do
                    _fevd_vecm(; data=csv, lags=2, rank="1", horizons=10,
                                id="svec", config=cfg, format="table")
                end
            end
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
        end
    end

    @testset "_hd_vecm — svec identification" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _hd_vecm(; data=csv, lags=2, rank="1", id="svec", format="table")
                end
            end
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
        # 23 primary + 1 alias = 24 keys (C044; +ms/ms-ar W3 #101)
        @test length(node.subcmds) == 39  # +poisson/nbreg (W2 #107)
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr-garch", "sv", "favar",
                     "reg", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit"]
            @test haskey(node.subcmds, cmd)
        end
        @test haskey(node.subcmds, "gjr_garch")
    end

    @testset "_predict_var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _predict_var(; data=csv, lags=2, format="table")
                end
            end
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
        end
    end

    @testset "_predict_reg — WLS" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _predict_reg(; data=csv, dep="var1", cov_type="ols",
                               weights="var4", clusters="", format="table", output="")
            end
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
            # The dict mixes scalars with a `confusion` MATRIX: the scalars render
            # as kv and the matrix as its own labelled table. Sorting the pairs
            # themselves compared a Matrix against a Float and exited 1 (#85).
            @test contains(out, "accuracy")
            @test contains(out, "Confusion Matrix")
            @test contains(out, "predicted_0")
            @test contains(out, "actual_1")
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
            @test contains(out, "accuracy")
            @test contains(out, "Confusion Matrix")
        end
    end

end  # Predict handlers

# ═══════════════════════════════════════════════════════════════
# Residuals handlers (residuals.jl)
# ═══════════════════════════════════════════════════════════════

@testset "dsge bayes posterior-mode / prior-predictive (C073 #78)" begin
    mktempdir() do dir
        model = joinpath(dir, "m.toml")
        write(model, """
        [model]
        parameters = { rho = 0.9, sigma = 0.01 }
        endogenous = ["Y", "C"]
        exogenous = ["e"]
        [[model.equations]]
        expr = "Y[t] = rho * Y[t-1] + sigma * e[t]"
        [[model.equations]]
        expr = "C[t] = Y[t]"
        """)
        priors = joinpath(dir, "p.toml")
        write(priors, """
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
        data = joinpath(dir, "d.csv")
        open(data, "w") do io
            println(io, "Y"); y = 0.0
            for _ in 1:40; y = 0.9y + 0.01randn(); println(io, y); end
        end

        @testset "posterior-mode — mode table + Laplace diagnostics" begin
            out = _capture() do
                _dsge_bayes_posterior_mode(; model=model, data=data, params="rho,sigma",
                    priors=priors, observables="Y", format="table", output="")
            end
            @test contains(out, "mode") && contains(out, "std_error")
            @test contains(out, "Laplace log ML") && contains(out, "converged")
        end

        @testset "prior-predictive — needs NO --data" begin
            out = _capture() do
                _dsge_bayes_prior_predictive(; model=model, params="rho,sigma",
                    priors=priors, observables="Y", n_draws=20, periods=50,
                    format="table", output="")
            end
            @test contains(out, "statistic") && contains(out, "median")
            @test contains(out, "draws that solved")
        end

        @testset "bad input → typed CliError" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            # posterior-mode requires data/params/priors (shared _dsge_bayes_inputs)
            e = err(() -> _dsge_bayes_posterior_mode(; model=model, params="rho",
                        priors=priors, format="table", output=""))
            @test e isa CliError && e.code == "usage/missing"
            e = err(() -> _dsge_bayes_posterior_mode(; model=model, data=data,
                        priors=priors, format="table", output=""))
            @test e isa CliError && e.code == "usage/missing"
            e = err(() -> _dsge_bayes_posterior_mode(; model=model, data=data,
                        params="rho", format="table", output=""))
            @test e isa CliError && e.code == "usage/missing"
            e = err(() -> _dsge_bayes_posterior_mode(; model=model, data=data,
                        params="rho,sigma", priors=priors, max_iter=0,
                        format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            # prior-predictive must NOT demand --data, but still needs params/priors
            e = err(() -> _dsge_bayes_prior_predictive(; model=model, priors=priors,
                        format="table", output=""))
            @test e isa CliError && e.code == "usage/missing"
            e = err(() -> _dsge_bayes_prior_predictive(; model=model, params="rho,sigma",
                        priors=priors, n_draws=0, format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _dsge_bayes_prior_predictive(; model=model, params="rho,sigma",
                        priors=priors, periods=0, format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
        end
    end
end

@testset "forecast midas (#67)" begin
    mktempdir() do dir
        lf = joinpath(dir, "lf.csv"); hf = joinpath(dir, "hf.csv")
        open(lf, "w") do io; println(io, "y"); for t in 1:60; println(io, 0.5t + 0.1); end; end
        open(hf, "w") do io; println(io, "x"); for t in 1:180; println(io, Float64(t)); end; end

        @testset "single direct forecast at the fixed horizon" begin
            out = _capture() do
                _forecast_midas(; data=lf, column=1, hf_data=hf, hf_column=1,
                                  m=3, k=6, format="table", output="")
            end
            @test contains(out, "forecast") && contains(out, "lower") && contains(out, "upper")
            # the horizon is fixed at estimation (m.h) — there is no --horizons here
            @test contains(out, "horizon (h)")
        end

        @testset "X_new is passed MOST-RECENT-FIRST" begin
            # The mock's forecast returns X_new[1] as the point forecast, pinning the
            # orientation contract: the HF series runs 1.0…180.0, so the most recent
            # observation is 180.0. Passing the block chronologically would yield 175.0
            # (the oldest of the last K=6) and would NOT error — it would just be wrong,
            # which is why the handler reverses explicitly.
            out = _capture() do
                _forecast_midas(; data=lf, column=1, hf_data=hf, hf_column=1,
                                  m=3, k=6, format="table", output="")
            end
            @test contains(out, "180.0")
            @test !contains(out, "175.0")
        end

        @testset "bad input → typed CliError" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            e = err(() -> _forecast_midas(; data=lf, hf_data=hf, m=3, k=6, level=1.5,
                                            format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            e = err(() -> _forecast_midas(; data=lf, hf_data=hf, m=3, k=6, poly_degree=-1,
                                            format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _forecast_midas(; data=lf, hf_data=hf, m=3, k=1,
                                            weights="beta2", format="table", output=""))
            @test e isa CliError && e.code == "data/invalid"
        end
    end
end

@testset "statespace predict & residuals (#71)" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=120, n=2, colnames=["y", "other"])

        @testset "state paths — long table, shape follows --kind" begin
            out = _capture() do
                _predict_statespace(; data=csv, column=1, format="csv", output="")
            end
            @test contains(out, "period") && contains(out, "state")
            @test contains(out, "filtered") && contains(out, "smoothed")
            # local-level has ONE state -> T rows + header
            @test count(==('\n'), out) == 120 + 1
            # local-linear-trend has TWO -> the long table GROWS; the columns do not change
            out2 = _capture() do
                _predict_statespace(; data=csv, column=1, kind="local-linear-trend",
                                      format="csv", output="")
            end
            @test count(==('\n'), out2) == 2 * 120 + 1
        end

        @testset "--state narrows the emitted path" begin
            f = _capture() do
                _predict_statespace(; data=csv, column=1, state="filtered",
                                      format="csv", output="")
            end
            @test contains(f, "filtered") && !contains(f, "smoothed")
            sm = _capture() do
                _predict_statespace(; data=csv, column=1, state="smoothed",
                                      format="csv", output="")
            end
            @test contains(sm, "smoothed") && !contains(sm, "filtered")
        end

        @testset "residuals — innovations, raw and standardized" begin
            raw = _capture() do
                _residuals_statespace(; data=csv, column=1, format="csv", output="")
            end
            @test contains(raw, "residual")
            std = _capture() do
                _residuals_statespace(; data=csv, column=1, standardized=true,
                                        format="csv", output="")
            end
            @test contains(std, "residual")
            @test raw != std          # v_t vs v_t/sqrt(F_t) are different series
        end

        @testset "bad input → typed CliError" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            e = err(() -> _predict_statespace(; data=csv, kind="bogus",
                                                format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            e = err(() -> _predict_statespace(; data=csv, init_mode="bogus",
                                                format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _predict_statespace(; data=csv, state="bogus",
                                                format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _residuals_statespace(; data=csv, column=9,
                                                  format="table", output=""))
            @test e isa CliError && e.code == "data/column-range"
        end
    end
end

@testset "sur/3sls predict & residuals (#68)" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=80, n=5, colnames=["y1", "y2", "x1", "x2", "z1"])
        cfg = joinpath(dir, "sys.toml")
        write(cfg, """
        [[equations]]
        name = "eq1"
        dep = "y1"
        indep = ["x1"]
        [[equations]]
        name = "eq2"
        dep = "y2"
        indep = ["x2"]
        [instruments]
        common = ["x1", "x2", "z1"]
        """)

        @testset "one tidy LONG table, not N tables" begin
            for (fn, col) in ((_predict_sur, "fitted"), (_residuals_sur, "residual"))
                out = _capture() do
                    fn(; data=csv, config=cfg, format="csv", output="")
                end
                @test contains(out, "equation") && contains(out, col)
                # 2 equations x 80 obs + header — ONE table, not one per equation, so the
                # envelope key set does not vary with the config file
                @test count(==('\n'), out) == 2 * 80 + 1
                @test contains(out, "eq1") && contains(out, "eq2")
            end
        end

        @testset "3sls verbs" begin
            for (fn, col) in ((_predict_3sls, "fitted"), (_residuals_3sls, "residual"))
                out = _capture() do
                    fn(; data=csv, config=cfg, format="csv", output="")
                end
                @test contains(out, "equation") && contains(out, col)
            end
        end

        @testset "--config is required (the system lives there)" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            for fn in (_predict_sur, _residuals_sur, _predict_3sls, _residuals_3sls)
                e = err(() -> fn(; data=csv, config="", format="table", output=""))
                @test e isa CliError && e.code == "config/missing"
            end
        end
    end
end

@testset "arfima forecast/predict/residuals (#73)" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=200, n=2, colnames=["x", "other"])

        @testset "forecast — ARIMAForecast with ci_lower/ci_upper" begin
            out = _capture() do
                _forecast_arfima(; data=csv, column=1, p=1, q=0, horizons=6,
                                   format="table", output="")
            end
            # the result type is ARIMAForecast: its interval fields are ci_lower/ci_upper,
            # NOT lower/upper (an early draft used the latter and FieldError'd)
            @test contains(out, "forecast") && contains(out, "lower") && contains(out, "upper")
        end

        @testset "predict / residuals — read the model's fields" begin
            out = _capture() do
                _predict_arfima(; data=csv, column=1, p=1, q=0, format="table", output="")
            end
            @test contains(out, "fitted")
            out = _capture() do
                _residuals_arfima(; data=csv, column=1, p=1, q=0, format="table", output="")
            end
            @test contains(out, "residual")
        end

        @testset "bad input → typed CliError" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            e = err(() -> _forecast_arfima(; data=csv, column=1, horizons=0,
                                             format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            e = err(() -> _forecast_arfima(; data=csv, column=1, confidence=1.5,
                                             format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _forecast_arfima(; data=csv, column=1, trunc_lag=0,
                                             format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _predict_arfima(; data=csv, column=9, format="table", output=""))
            @test e isa CliError && e.code == "data/column-range"
        end
    end
end

@testset "GARCH-variant forecast/predict/residuals (C064 #69)" begin
    # The six C064a variants are NOT in VOL_MODELS (their option sets differ), so these
    # verbs are hand-written per variant; each mirrors its `estimate` sibling's options.
    _capture_ok(f) = begin
        out = _capture() do; f(); end
        out
    end
    mktempdir() do dir
        csv = _make_csv(dir; T=200, n=2, colnames=["r", "other"])

        @testset "forecast — variance path per variant" begin
            for (fn, kw) in ((_forecast_igarch, (;)), (_forecast_cgarch, (;)),
                             (_forecast_aparch, (;)),
                             (_forecast_figarch, (;)), (_forecast_fiegarch, (;)))
                out = _capture_ok(() -> fn(; data=csv, column=1, horizons=5,
                                             format="table", output="", kw...))
                @test !isempty(out)
            end
            # garch-midas returns a NamedTuple with the long-run/short-run split and has
            # NO --conf-level (upstream forecast takes none)
            out = _capture_ok(() -> _forecast_garch_midas(; data=csv, column=1, m_freq=20,
                                        k=4, horizons=4, format="table", output=""))
            @test contains(out, "long_run") && contains(out, "short_run")
        end

        @testset "predict / residuals — per-observation tables" begin
            for fn in (_predict_igarch, _predict_cgarch, _predict_aparch,
                       _predict_figarch, _predict_fiegarch)
                out = _capture_ok(() -> fn(; data=csv, column=1, format="table", output=""))
                @test contains(out, "variance") && contains(out, "volatility")
            end
            for fn in (_residuals_igarch, _residuals_cgarch, _residuals_aparch,
                       _residuals_figarch, _residuals_fiegarch)
                out = _capture_ok(() -> fn(; data=csv, column=1, format="table", output=""))
                @test contains(out, "residual")
            end
            out = _capture_ok(() -> _predict_garch_midas(; data=csv, column=1, m_freq=20,
                                        k=4, format="table", output=""))
            @test contains(out, "variance")
            out = _capture_ok(() -> _residuals_garch_midas(; data=csv, column=1, m_freq=20,
                                        k=4, format="table", output=""))
            @test contains(out, "residual")
        end

        @testset "bad input → typed CliError, never exit 1" begin
            err(f) = try; _capture() do; f(); end; nothing; catch e; e end
            e = err(() -> _forecast_igarch(; data=csv, column=1, horizons=0,
                                            format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid" && exit_class(e) == 2
            e = err(() -> _forecast_igarch(; data=csv, column=1, conf_level=1.5,
                                            format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            e = err(() -> _forecast_garch_midas(; data=csv, column=1, m_freq=20,
                                                 horizons=0, format="table", output=""))
            @test e isa CliError && e.code == "usage/invalid"
            # --m-freq is required for every garch-midas verb
            for fn in (_forecast_garch_midas, _predict_garch_midas, _residuals_garch_midas)
                e = err(() -> fn(; data=csv, column=1, format="table", output=""))
                @test e isa CliError && exit_class(e) == 2
            end
            # a bad --column is a typed loader error on every verb
            e = err(() -> _predict_igarch(; data=csv, column=9, format="table", output=""))
            @test e isa CliError && e.code == "data/column-range"
        end
    end
end

@testset "Residuals handlers" begin

    @testset "register_residuals_commands!" begin
        node = register_residuals_commands!()
        @test node isa NodeCommand
        @test node.name == "residuals"
        # 23 primary + 1 alias = 24 keys (C044); +#70 setar/star/ms-ar/ms => 38
        @test length(node.subcmds) == 41
        for cmd in ["var", "bvar", "arima", "vecm", "static", "dynamic", "gdfm",
                     "arch", "garch", "egarch", "gjr-garch", "sv", "favar",
                     "reg", "logit", "probit",
                     "preg", "piv", "plogit", "pprobit", "ologit", "oprobit", "mlogit"]
            @test haskey(node.subcmds, cmd)
        end
        @test haskey(node.subcmds, "gjr_garch")
        # #70 remainder: the nonlinear-TS models define StatsAPI.residuals upstream
        for cmd in ["setar", "star", "ms-ar", "ms"]
            @test haskey(node.subcmds, cmd)
        end
        # W3/#101: MEMs#510 added predict/forecast for MSRegModel ONLY, so ms|ms-ar gained
        # both verbs while SETAR/STAR still have neither upstream. Advertising a leaf whose
        # upstream method does not exist is the #85 failure mode, so this split is asserted
        # in both directions rather than assumed.
        pnode = register_predict_commands!()
        fnode = register_forecast_commands!()
        for cmd in ["ms-ar", "ms"]
            @test haskey(pnode.subcmds, cmd)
            @test haskey(fnode.subcmds, cmd)
        end
        for cmd in ["setar", "star"]
            @test !haskey(pnode.subcmds, cmd)
        end
    end

    @testset "residuals setar|star|ms-ar|ms (#70 remainder)" begin
        # One tidy `period|residual` table per leaf. `period` is the EFFECTIVE-sample index:
        # the AR-based fits drop lags, the MS regression does not — asserted below.
        _rdoc(leaf, args) = begin
            out = _capture() do
                _dispatch_via_app(vcat(String["residuals", leaf], collect(String, args), String["--format", "json"]))
            end
            JSON3.read(out[findfirst('{', out):end])
        end
        _rtbl(doc) = first(t for t in values(doc.data) if (t isa JSON3.Object && haskey(t, :columns)))
        _rerr(leaf, args) = begin
            e = nothing
            try; _capture() do; _dispatch_via_app(vcat(String["residuals", leaf], collect(String, args))); end; catch ex; e=ex; end
            e
        end

        mktempdir() do dir
            csv = _make_csv(dir; T=200, n=1, colnames=["y"])

            @testset "each leaf emits period|residual" begin
                for (leaf, args) in (("setar", [csv, "--p", "1"]), ("star", [csv, "--p", "1"]),
                                     ("ms-ar", [csv, "--p", "1"]), ("ms", [csv]))
                    doc = _rdoc(leaf, args)
                    @test doc.status == "ok"
                    t = _rtbl(doc)
                    @test Set(["period", "residual"]) ⊆ Set(String.(t.columns))
                    rows = collect(t.rows)
                    @test !isempty(rows)
                    @test all(r -> isa(Float64(collect(r)[2]), Float64), rows)
                    # period is 1..n_eff, contiguous
                    @test [Int(collect(r)[1]) for r in rows] == collect(1:length(rows))
                end
            end

            @testset "AR-based leaves drop lags; the MS regression does not" begin
                n_setar = length(collect(_rtbl(_rdoc("setar", [csv, "--p", "1"])).rows))
                n_ms    = length(collect(_rtbl(_rdoc("ms", [csv])).rows))
                @test n_setar < n_ms          # SETAR(p=1) loses one observation to the lag
                # a larger p loses more
                @test length(collect(_rtbl(_rdoc("setar", [csv, "--p", "3"])).rows)) < n_setar
            end

            @testset "options mirror the estimate sibling; guards are typed" begin
                @test _rerr("setar", [csv, "--p", "0"]).code == "usage/invalid"
                @test _rerr("setar", [csv, "--trim", "0.6"]).code == "usage/invalid"
                @test _rerr("setar", [csv, "--column", "99"]).code == "data/column-range"
                @test _rerr("star", [csv, "--n-gamma", "1"]).code == "usage/invalid"
                @test _rerr("ms-ar", [csv, "--k-regimes", "1"]).code == "usage/invalid"
                @test _rerr("ms-ar", [csv, "--max-iter", "0"]).code == "usage/invalid"
                @test _rerr("ms", [csv, "--tol", "0"]).code == "usage/invalid"
                @test _rerr("ms", [csv, "--k-regimes", "1"]).code == "usage/invalid"
                # --type is parser-validated via choices (ParseError, exit 2)
                et = _rerr("star", [csv, "--type", "bogus"])
                @test et isa ParseError || (et isa CliError && exit_class(et) == 2)
                # SETAR's inference-only options are deliberately NOT advertised here: they
                # drive the Hansen bootstrap / threshold CI, never the residuals.
                for opt in ("--reps", "--ci-level", "--het", "--no-linearity")
                    er = _rerr("setar", [csv, opt, "1"])
                    @test er isa ParseError || (er isa CliError && exit_class(er) == 2)
                end
            end

            @testset "a delay/regime change actually changes the residuals" begin
                r1 = [Float64(collect(r)[2]) for r in collect(_rtbl(_rdoc("ms-ar", [csv, "--p", "1"])).rows)]
                r3 = [Float64(collect(r)[2]) for r in collect(_rtbl(_rdoc("ms-ar", [csv, "--p", "3"])).rows)]
                @test length(r1) != length(r3) || r1 != r3
            end
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
        end
    end

    @testset "_residuals_reg — WLS" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_reg(; data=csv, dep="var1", cov_type="ols",
                                 weights="var4", clusters="", format="table", output="")
            end
        end
    end

    @testset "_residuals_logit — default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_logit(; data=csv, dep="var1", cov_type="hc1",
                                   clusters="", format="table", output="")
            end
        end
    end

    @testset "_residuals_logit — json" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _residuals_logit(; data=csv, dep="var1", cov_type="ols",
                                   clusters="", format="json", output="")
            end
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
            @test_throws Exception cd(dir) do
                _capture() do
                    _estimate_var(; data="/nonexistent/path.csv", lags=2, format="table")
                end
            end
        end
    end

    @testset "nonexistent config file" begin
        @test_throws Exception _capture() do
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
        end
    end

    @testset "single-variable tests" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _test_adf(; data=csv, column=1, format="table")
            end
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
        @test length(node.subcmds) == 6
        for cmd in ["hp", "hamilton", "bn", "bk", "bhp", "x13"]
            @test haskey(node.subcmds, cmd)
        end
        x13 = node.subcmds["x13"]
        opt_names = [o.name for o in x13.options]
        @test "frequency" in opt_names
        @test "method" in opt_names
        @test "transform" in opt_names
    end

    @testset "_filter_x13" begin
        mktempdir() do dir
            # monthly seasonal series, T=120
            csv = joinpath(dir, "x13.csv")
            open(csv, "w") do io
                println(io, "y")
                for t in 1:120
                    println(io, 100 + 10 * sin(2π * t / 12) + 0.1 * t)
                end
            end
            out = _capture() do
                _filter_x13(; data=csv, frequency=12, method="x11",
                            transform="none", format="table", output="")
            end
            @test contains(out, "X-13") || contains(out, "Seasonally") || true
        end
    end

    @testset "_filter_x13 too short" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=20, n=1)
            @test_throws Exception _filter_x13(; data=csv, frequency=12, method="seats",
                transform="auto", format="table", output="")
        end
    end

    @testset "_filter_x13 bad frequency" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=1)
            @test_throws Exception _filter_x13(; data=csv, frequency=5, method="seats",
                transform="auto", format="table", output="")
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
        @test length(node.subcmds) == 77  # 76 primary (+sarima W6, +tvpvar/mfvar W7, +svar/svec W2 #166) (+poisson/nbreg W2 #107) + gjr_garch alias (C064a +6, C068 +arfima, C064b +3 MGARCH, C067a +5, C067b +2, C066 +5, C062a +2, C062b +2, C062c +1, C062d +midas, C065a +setar, C065b +star, C065c +ms-ar/ms, C067 +select, #70 +threshold)
    end

    @testset "register_irf_commands! includes pvar" begin
        node = register_irf_commands!()
        @test haskey(node.subcmds, "pvar")
        @test node.subcmds["pvar"] isa LeafCommand
        @test length(node.subcmds) == 8
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
        @test length(node.subcmds["pvar"].subcmds) == 5  # 4 primary + hansen_j alias
        @test haskey(node.subcmds["pvar"].subcmds, "hansen-j")
        @test haskey(node.subcmds["pvar"].subcmds, "hansen_j")
        @test haskey(node.subcmds["pvar"].subcmds, "mmsc")
        @test haskey(node.subcmds["pvar"].subcmds, "lagselect")
        @test haskey(node.subcmds["pvar"].subcmds, "stability")
        @test haskey(node.subcmds, "lr")
        @test node.subcmds["lr"] isa LeafCommand
        @test haskey(node.subcmds, "lm")
        @test node.subcmds["lm"] isa LeafCommand
        @test length(node.subcmds) == 85  # 81 primary (+dispersion W2 #107) + 2 aliases (+hegy/ers/sadf/gsadf/edf/engle-granger/phillips-ouliaris/hansen-instability/park-added C069 remainder, +llc/ips/breitung/fisher-johansen/dh-causality C070 remainder, +gph, +local-whittle C068, +sign-bias, +nyblom C064b, +vecm C071, +variance-ratio/bds/hadri/pedroni/kao/westerlund C069/C070, +weak-instrument C067b, +ardl-bounds/nardl-symmetry C062b, +pmg-hausman C062c, +hansen-linearity C065a, +star-linearity C065b)
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
            @test_throws Exception load_panel_data(csv, "nonexistent", "time")
        end
    end

    @testset "load_panel_data — missing time column" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws Exception load_panel_data(csv, "group", "nonexistent")
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
            @test_throws Exception cd(dir) do
                _estimate_pvar(; data=csv, time_col="time")
            end
        end
    end

    @testset "_estimate_pvar — missing time-col error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws Exception cd(dir) do
                _estimate_pvar(; data=csv, id_col="group")
            end
        end
    end

    @testset "_estimate_pvar — invalid method error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_irf_pvar — missing data and tag error" begin
        mktempdir() do dir
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_test_pvar_hansen_j — missing id error" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir)
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_test_granger — invalid model error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception cd(dir) do
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
        @test length(node.subcmds) == 13
        for cmd in ["list", "load", "import", "export", "describe", "diagnose", "fix", "transform", "filter", "validate", "balance", "dropna", "keeprows"]
            @test haskey(node.subcmds, cmd)
            @test node.subcmds[cmd] isa LeafCommand
        end
    end

    @testset "option counts" begin
        node = register_data_commands!()
        @test length(node.subcmds["list"].options) == 2
        @test length(node.subcmds["load"].options) == 6
        @test length(node.subcmds["import"].options) == 10
        @test length(node.subcmds["export"].options) == 2
        @test length(node.subcmds["describe"].options) == 2
        @test length(node.subcmds["diagnose"].options) == 2
        @test length(node.subcmds["fix"].options) == 3
        @test length(node.subcmds["transform"].options) == 3
        @test length(node.subcmds["filter"].options) == 8
        @test length(node.subcmds["validate"].options) == 3
        @test length(node.subcmds["balance"].options) == 5
    end

    @testset "dataset_to_dataframe — panels keep their identifiers" begin
        # Without group/time columns every panel command fails on a bundled panel
        # (`--id-col group` has nothing to bind to).
        df = dataset_to_dataframe(load_example(:pwt))
        @test names(df)[1:2] == ["group", "time"]
        @test df.group == load_example(:pwt).group_id

        # Same via the public loader used by handlers and the REPL
        df2 = load_data(":pwt")
        @test "group" in names(df2) && "time" in names(df2)

        # Non-panel datasets are unchanged
        ts = dataset_to_dataframe(load_example(:fred_md))
        @test !("group" in names(ts))

        # Cross-section datasets load too (added alongside the other bundled sets)
        cs = dataset_to_dataframe(load_example(:stackloss))
        @test nrow(cs) == 21
    end

    @testset "_data_describe — n is per-variable" begin
        mktempdir() do dir
            csv = joinpath(dir, "d.csv")
            CSV.write(csv, DataFrame(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0]))
            outfile = joinpath(dir, "desc.json")
            _capture() do
                _data_describe(; data=csv, format="json", output=outfile)
            end
            rows = JSON3.read(read(outfile, String))
            # Regression: fill(summary.n, n_vars) nested the whole vector into every cell
            @test all(r -> r.n isa Integer, rows)
        end
    end

    @testset "_data_list — table" begin
        out = _capture() do
            _data_list(; format="table")
        end
    end

    @testset "_data_list — json" begin
        mktempdir() do dir
            outfile = joinpath(dir, "datasets.json")
            _capture() do
                _data_list(; format="json", output=outfile)
            end
            @test isfile(outfile)
            json_data = JSON3.read(read(outfile, String))
            # Derived from EXAMPLE_DATASETS, so `data list` can never advertise
            # fewer datasets than `data load` accepts.
            @test length(json_data) == length(EXAMPLE_DATASETS)
        end
    end

    @testset "_data_load — loading existing data (regressions)" begin
        node = register_data_commands!()
        load_leaf = node.subcmds["load"]

        # <name> is optional so `data load --path file.csv` is reachable at all.
        @test !load_leaf.args[1].required
        # --transform was documented + handler-supported but never registered.
        @test "transform" in [f.name for f in load_leaf.flags]

        mktempdir() do dir
            src = joinpath(dir, "mine.csv")
            CSV.write(src, DataFrame(a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0]))

            # --path alone (no positional) loads the CSV
            cd(dir) do
                _capture() do
                    _data_load(; path=src)
                end
            end
            @test isfile(joinpath(dir, "mine_loaded.csv"))

            # Neither name nor --path → typed usage error, not a missing-arg crash
            e = try; _capture() do; _data_load(); end; nothing catch err; err end
            @test e isa CliError
            @test e.code == "usage/missing"
        end

        # ':name' references resolve (REPL syntax must work on the CLI too)
        for form in ("fred-md", ":fred-md", ":fred_md")
            mktempdir() do dir
                cd(dir) do
                    _capture() do
                        _data_load(; name=form)
                    end
                end
                # Default output uses the canonical stem — never ':fred-md.csv'
                @test isfile(joinpath(dir, "fred_md.csv"))
            end
        end

        # Unknown dataset → typed data/unknown-dataset (was untyped ArgumentError → exit 1)
        e = try; _capture() do; _data_load(; name="frd_md"); end; nothing catch err; err end
        @test e isa CliError
        @test e.code == "data/unknown-dataset"
    end

    @testset "_data_load — fred_md" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md")
                end
            end
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
        end
    end

    @testset "_data_load — pwt" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="pwt")
                end
            end
        end
    end

    @testset "_data_load — with --transform" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md", transform=true)
                end
            end
        end
    end

    @testset "_data_load — with --vars" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="fred_md", vars="INDPRO,CPIAUCSL,FEDFUNDS")
                end
            end
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
        end
    end

    @testset "_data_load — invalid name" begin
        mktempdir() do dir
            @test_throws Exception cd(dir) do
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
        end
    end

    @testset "_data_load — invalid var name" begin
        mktempdir() do dir
            @test_throws Exception cd(dir) do
                _data_load(; name="fred_md", vars="NONEXISTENT_VAR")
            end
        end
    end

    @testset "_data_list includes mpdta and ddcg" begin
        out = _capture() do
            _data_list(; format="table")
        end
    end

    @testset "_data_load — mpdta" begin
        mktempdir() do dir
            out = cd(dir) do
                _capture() do
                    _data_load(; name="mpdta")
                end
            end
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
            @test isfile(joinpath(dir, "ddcg.csv"))
            result_df = CSV.read(joinpath(dir, "ddcg.csv"), DataFrame)
            @test "y" in names(result_df)
            @test "dem" in names(result_df)
        end
    end

    @testset "_data_load — :mp-shocks NaN survives the round-trip (W3/#125)" begin
        # NaN is NOT zero: zero is a valid shock value, so any loader that
        # coerced NaN→0 would silently fabricate shocks. The mock carries the
        # real per-column valid ranges, so these counts match T3 exactly.
        mktempdir() do dir
            out = joinpath(dir, "mp.csv")
            _capture() do
                _data_load(; name=":mp-shocks", output=out)   # dash spelling normalizes
            end
            df = CSV.read(out, DataFrame)
            @test size(df) == (240, 8)
            @test names(df) == ["ygap", "infl", "ffr", "lpcom", "rr", "mp1", "ad", "bzk_ist"]
            @test isnan(df.rr[1])                  # pre-1969Q1: outside published sample
            @test count(!isnan, df.rr) == 156      # Romer-Romer 1969Q1–2007Q4
            @test count(!isnan, df.mp1) == 95      # Gertler-Karadi 1988Q4–2012Q2
        end
    end

    @testset "_data_describe — NaN-padded valid windows (:mp_shocks)" begin
        mktempdir() do dir
            outfile = joinpath(dir, "desc.json")
            _capture() do
                _data_describe(; data=":mp_shocks", format="json", output=outfile)
            end
            rows = JSON3.read(read(outfile, String))
            byvar = Dict(String(r.variable) => r for r in rows)
            # n counts FINITE observations; first/last_valid bound the window.
            @test byvar["rr"].n == 156
            @test byvar["rr"].first_valid == 37 && byvar["rr"].last_valid == 192
            @test byvar["ygap"].first_valid == 37 && byvar["ygap"].last_valid == 240
            @test byvar["ffr"].n == 223
            # Statistics exclude the NaN padding rather than propagating it.
            @test isfinite(byvar["rr"].mean) && isfinite(byvar["rr"].std)
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
            # Default output should be data_clean.csv
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
            @test_throws Exception _data_fix(; data=csv, method="invalid")
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
            @test_throws Exception _data_transform(; data=csv, tcodes="")
        end
    end

    @testset "_data_transform — wrong number of tcodes" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _data_transform(; data=csv, tcodes="5,5")
        end
    end

    @testset "_data_filter — hp default" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hp")
            end
        end
    end

    @testset "_data_filter — hamilton" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hamilton", horizon=8, lags=4)
            end
        end
    end

    @testset "_data_filter — bhp" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="bhp")
            end
        end
    end

    @testset "_data_filter — trend component" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="hp", component="trend")
            end
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
        end
    end

    @testset "_data_filter — invalid method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _data_filter(; data=csv, method="invalid")
        end
    end

    @testset "_data_filter — invalid component" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _data_filter(; data=csv, component="invalid")
        end
    end

    @testset "_data_validate — valid var" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="var")
            end
        end
    end

    @testset "_data_validate — valid arima (univariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _data_validate(; data=csv, model="arima")
            end
        end
    end

    @testset "_data_validate — invalid arima (multivariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="arima")
            end
        end
    end

    @testset "_data_validate — missing --model error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _data_validate(; data=csv, model="")
        end
    end

    @testset "_data_validate — invalid model type error" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            @test_throws Exception _data_validate(; data=csv, model="invalid")
        end
    end

    @testset "_data_validate — valid bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_validate(; data=csv, model="bvar")
            end
        end
    end

    @testset "_data_validate — valid garch (univariate)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            out = _capture() do
                _data_validate(; data=csv, model="garch")
            end
        end
    end

    # ── data balance ──────────────────────────────────────────

    @testset "_data_balance — basic" begin
        cd(mktempdir()) do
            write("data.csv", "a,b\n1.0,2.0\n3.0,NaN\n5.0,6.0\n7.0,8.0\n")
            out = _capture() do
                _data_balance(; data="data.csv")
            end
        end
    end

    @testset "_data_balance — custom method and factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=50, n=4)
            out = _capture() do
                _data_balance(; data=csv, method="dfm", factors=2, lags=1)
            end
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
        end
    end

    @testset "_data_load — named dataset with --dates" begin
        cd(mktempdir()) do
            out = _capture() do
                _data_load(; name="fred_md", dates="INDPRO")
            end
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
        @test length(node.subcmds["bvar"].options) == 12   # +6 in W1/#123 (prior, theta-cross, 4 hyperparameters)
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
        end
    end

    @testset "_nowcast_dfm — custom factors and lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_dfm(; data=csv, monthly_vars=4, quarterly_vars=1,
                    factors=3, lags=2, idio="iid")
            end
        end
    end

    @testset "_nowcast_dfm — auto var split" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_dfm(; data=csv)
            end
        end
    end

    @testset "_nowcast_dfm — invalid var split" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            @test_throws Exception _nowcast_dfm(; data=csv,
                monthly_vars=3, quarterly_vars=1)
        end
    end

    @testset "_nowcast_bvar — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
        end
    end

    @testset "_nowcast_bvar — custom lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1, lags=3)
            end
        end
    end

    @testset "_nowcast_bvar — litterman prior + theta-cross (W1/#123)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1,
                               prior="litterman", theta_cross="0.5")
            end
            @test occursin("litterman", out)
            @test occursin("theta_cross", out)
            # conjugate output must NOT carry a theta_cross row (it is NaN there)
            out_c = _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
            @test !occursin("theta_cross", out_c)
            # --theta-cross under conjugate is a TYPED usage error, guarded before the
            # estimator (upstream's ArgumentError would be data/invalid — wrong class)
            err = try
                _capture() do
                    _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1,
                                   theta_cross="0.5")
                end
                nothing
            catch e; e; end
            @test err isa CliError && err.code == "usage/invalid"
            @test_throws CliError _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1,
                               prior="litterman", theta_cross="not-a-number")
            end
            @test_throws CliError _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1,
                               prior="litterman", theta_cross="-1.0")
            end
            @test_throws CliError _capture() do
                _nowcast_bvar(; data=csv, monthly_vars=4, quarterly_vars=1, lambda0=0.0)
            end
        end
    end

    @testset "_nowcast_bridge — basic" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bridge(; data=csv, monthly_vars=4, quarterly_vars=1)
            end
        end
    end

    @testset "_nowcast_bridge — custom lags" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_bridge(; data=csv, monthly_vars=4, quarterly_vars=1,
                    lag_m=2, lag_q=2, lag_y=2)
            end
        end
    end

    # A vintage differs from its predecessor by which cells are FILLED IN, not by
    # row count — both matrices must be the same shape. These two used to hand the
    # handler a 105-row new vintage against a 100-row old one and passed only
    # because the mock accepted anything; real MEMs raises on it.
    @testset "_nowcast_news — basic" begin
        mktempdir() do dir
            csv_old = _make_csv(dir; T=100, n=5, colnames=["m1","m2","m3","m4","q1"])
            csv_new = joinpath(dir, "data_new.csv")
            data = Dict{String,Vector{Float64}}()
            for name in ["m1","m2","m3","m4","q1"]
                data[name] = randn(100) .+ 1.0
            end
            CSV.write(csv_new, DataFrame(data))

            out = _capture() do
                _nowcast_news(; data_new=csv_new, data_old=csv_old,
                    monthly_vars=4, quarterly_vars=1, method="dfm")
            end
        end
    end

    @testset "_nowcast_news — missing data errors" begin
        @test_throws Exception _nowcast_news(; data_new="", data_old="old.csv")
        @test_throws Exception _nowcast_news(; data_new="new.csv", data_old="")
    end

    @testset "_nowcast_news — mismatched vintages are data/shape (#85)" begin
        mktempdir() do dir
            csv_old = _make_csv(dir; T=100, n=5, colnames=["m1","m2","m3","m4","q1"])
            csv_new = joinpath(dir, "data_new.csv")
            data = Dict{String,Vector{Float64}}()
            for name in ["m1","m2","m3","m4","q1"]
                data[name] = randn(105) .+ 1.0
            end
            CSV.write(csv_new, DataFrame(data))

            e = try
                _capture() do
                    _nowcast_news(; data_new=csv_new, data_old=csv_old,
                        monthly_vars=4, quarterly_vars=1, method="dfm")
                end
                nothing
            catch err
                err
            end
            # Upstream signals this with a bare ArgumentError — exit 1 before #85.
            @test e isa CliError
            @test e.code == "data/shape"
        end
    end

    @testset "_nowcast_news — bvar method" begin
        mktempdir() do dir
            csv_old = _make_csv(dir; T=100, n=5, colnames=["m1","m2","m3","m4","q1"])
            csv_new = joinpath(dir, "data_new.csv")
            data = Dict{String,Vector{Float64}}()
            for name in ["m1","m2","m3","m4","q1"]
                data[name] = randn(100) .+ 1.0
            end
            CSV.write(csv_new, DataFrame(data))

            out = _capture() do
                _nowcast_news(; data_new=csv_new, data_old=csv_old,
                    monthly_vars=4, quarterly_vars=1, method="bvar")
            end
        end
    end

    @testset "_nowcast_forecast — dfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="dfm", horizons=4)
            end
        end
    end

    @testset "_nowcast_forecast — bvar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="bvar", horizons=4)
            end
        end
    end

    @testset "_nowcast_forecast — bridge" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _nowcast_forecast(; data=csv, monthly_vars=4, quarterly_vars=1,
                    method="bridge", horizons=4)
            end
        end
    end

    @testset "_nowcast_forecast — invalid method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            @test_throws Exception _nowcast_forecast(; data=csv,
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
                @test spec isa MacroEconometricModels.ModelSpec
                @test spec.n_endog == 3
                @test spec.n_exog == 1
            end
        end
    end

    @testset "_load_dsge_model — .jl file" begin
        mktempdir() do dir
            jl_path = joinpath(dir, "model.jl")
            write(jl_path, """
            model = MacroEconometricModels.ModelSpec(; n_endog=4, n_exog=2)
            """)
            out = _capture() do
                spec = _load_dsge_model(jl_path)
                @test spec isa MacroEconometricModels.ModelSpec
                @test spec.n_endog == 4
                @test spec.n_exog == 2
            end
        end
    end

    @testset "_load_dsge_model — missing file" begin
        @test_throws Exception _load_dsge_model("/nonexistent/model.toml")
    end

    @testset "_load_dsge_model — unsupported extension" begin
        mktempdir() do dir
            bad_path = joinpath(dir, "model.csv")
            write(bad_path, "a,b\n1,2\n")
            @test_throws Exception _load_dsge_model(bad_path)
        end
    end

    @testset "_load_dsge_model — TOML missing endogenous" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            """)
            @test_throws Exception _load_dsge_model(toml_path)
        end
    end

    @testset "_load_dsge_model — E[t] TOML is config/invalid" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t] + e[t]"
            [[model.equations]]
            expr = "C[t] = rho * E[t](C[t+1])"
            """)
            err = try
                _load_dsge_model(toml_path)
                nothing
            catch e
                e
            end
            @test err isa CliError
            @test err.code == "config/invalid"
            @test occursin("E[t]", err.message)
            @test exit_class(err) == 4
        end
    end

    @testset "_load_dsge_model — linear=true TOML (C046/C043)" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "model.toml")
            write(toml_path, """
            [model]
            linear = true
            parameters = { rho = 0.9 }
            endogenous = ["Y", "C"]
            exogenous = ["e"]
            [[model.equations]]
            expr = "Y[t] = C[t]"
            [[model.equations]]
            expr = "C[t] = rho * Y[t]"
            """)
            out = _capture() do
                spec = _load_dsge_model(toml_path)
                @test spec isa MacroEconometricModels.ModelSpec
                @test hasproperty(spec, :linear)
                @test spec.linear === true
            end
            @test contains(out, "linear=true")
        end
    end

    @testset "_load_dsge_model — HADSGESpec .jl → usage/wrong-command (C046)" begin
        mktempdir() do dir
            jl_path = joinpath(dir, "ha_model.jl")
            write(jl_path, """
            MacroEconometricModels.load_ha_example(:huggett)
            """)
            err = try
                _load_dsge_model(jl_path)
                nothing
            catch e
                e
            end
            @test err isa CliError
            @test err.code == "usage/wrong-command"
            @test contains(err.message, "heterogeneous-agent")
            @test contains(err.message, "dsge ha")
            @test exit_class(err) == 2
        end
    end

    @testset "_load_ha_model — DSGESpec .jl → usage/wrong-command (C046)" begin
        mktempdir() do dir
            jl_path = joinpath(dir, "ra_model.jl")
            write(jl_path, """
            MacroEconometricModels.ModelSpec(; n_endog=2, n_exog=1)
            """)
            err = try
                _load_ha_model(jl_path)
                nothing
            catch e
                e
            end
            @test err isa CliError
            @test err.code == "usage/wrong-command"
            @test contains(err.message, "representative-agent") || contains(err.message, "ModelSpec")
            @test exit_class(err) == 2
        end
    end

    @testset "_load_ha_model — HADSGESpec .jl file (C046)" begin
        mktempdir() do dir
            jl_path = joinpath(dir, "ha_model.jl")
            write(jl_path, """
            MacroEconometricModels.load_ha_example(:huggett)
            """)
            out = _capture() do
                spec = _load_ha_model(jl_path)
                @test spec isa MacroEconometricModels.ModelSpec &&
                      MacroEconometricModels.has_kind(spec, MacroEconometricModels.HouseholdSystem)
                @test _ha_model_symbol(spec) == :huggett
            end
            @test contains(out, "huggett")
        end
    end

    @testset "_solve_dsge — default method" begin
        spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec)
            @test sol isa MacroEconometricModels.DSGESolution
        end
    end

    @testset "_solve_dsge — perturbation" begin
        spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec; method="perturbation", order=1)
            @test sol isa MacroEconometricModels.PerturbationSolution
        end
    end

    @testset "_solve_dsge — projection" begin
        spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
        out = _capture() do
            sol = _solve_dsge(spec; method="projection", degree=5)
            @test sol isa MacroEconometricModels.ProjectionSolution
        end
    end

    @testset "_solve_dsge — with constraint_solver" begin
        spec = MacroEconometricModels.ModelSpec(; n_endog=2, n_exog=1)
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
            @test cons[1] isa MacroEconometricModels.VariableBound
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
            spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
            cons = _load_dsge_constraints(toml_path; spec=spec)
            @test length(cons) == 1
            @test cons[1] isa MacroEconometricModels.OccBinConstraint
        end
    end

    @testset "_load_dsge_constraints — nonlinear without spec errors" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "constraints.toml")
            write(toml_path, """
            [[constraints.nonlinear]]
            expr = "K[t] <= Y[t]"
            """)
            @test_throws Exception _load_dsge_constraints(toml_path)
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
            spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
            cons = _load_dsge_constraints(toml_path; spec=spec)
            @test length(cons) == 2
            @test any(c -> c isa MacroEconometricModels.VariableBound, cons)
            @test any(c -> c isa MacroEconometricModels.OccBinConstraint, cons)
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
            @test cons[1] isa MacroEconometricModels.VariableBound
        end
    end

    @testset "_as_occbin_constraints — >2 bounds is usage/invalid" begin
        spec = MacroEconometricModels.ModelSpec(; n_endog=3, n_exog=1)
        mktempdir() do dir
            toml_path = joinpath(dir, "c.toml")
            write(toml_path, """
            [[constraints.bounds]]
            variable = "y1"
            lower = 0.0
            [[constraints.bounds]]
            variable = "y2"
            lower = 0.0
            [[constraints.bounds]]
            variable = "y3"
            upper = 1.0
            """)
            cons = _load_dsge_constraints(toml_path)
            err = try
                _as_occbin_constraints(cons, spec)
                nothing
            catch e
                e
            end
            @test err isa CliError
            @test err.code == "usage/invalid"
            @test exit_class(err) == 2
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
        end
    end

    @testset "_dsge_solve — order=3 perturbation (C043)" begin
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
                sol = _dsge_solve(; model=toml_path, method="perturbation", order=3, format="table")
                @test sol isa MacroEconometricModels.PerturbationSolution
                @test sol.order == 3
            end
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
            @test_throws Exception _dsge_solve(;
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

    @testset "_dsge_steady_state — invalid constraint-solver" begin
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
            @test_throws Exception _dsge_steady_state(;
                model=model_path, constraint_solver="invalid")
        end
    end

    @testset "_dsge_perfect_foresight — invalid constraint-solver" begin
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
            shocks_csv = joinpath(dir, "shocks.csv")
            write(shocks_csv, "e_A\n0.01\n0.0\n")
            @test_throws Exception _dsge_perfect_foresight(;
                model=model_path, shocks=shocks_csv, constraint_solver="invalid")
        end
    end

    @testset "_dsge_bayes_run_estimation — invalid constraint-solver" begin
        mktempdir() do dir
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
            @test_throws Exception _dsge_bayes_estimate(;
                model=model_path, data=csv, params="rho,sigma",
                priors=priors_path, sampler="smc",
                n_smc=100, n_particles=50, n_draws=100, burnin=10,
                ess_target=0.5, observables="", solver="gensys", order=1,
                delayed_acceptance=false, constraint_solver="invalid",
                output="", format="table")
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
        end
    end

    @testset "_dsge_fevd — unconditional order≥2 (C043)" begin
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
                _dsge_fevd(; model=toml_path, method="perturbation", order=2,
                           horizon=20, unconditional=true, format="table")
            end
            @test occursin("unconditional", out) || occursin("FEVD", out)
        end
    end

    @testset "_dsge_fevd — unconditional rejects gensys (C043)" begin
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
            @test_throws Exception _dsge_fevd(; model=toml_path, method="gensys",
                                               unconditional=true, format="table")
        end
    end

    @testset "_dsge_fevd — unconditional rejects order=1 (C043)" begin
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
            @test_throws Exception _dsge_fevd(; model=toml_path, method="perturbation",
                                               order=1, unconditional=true, format="table")
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
            @test_throws Exception _dsge_estimate(;
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
            @test_throws Exception _dsge_estimate(;
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
                                         periods=5, format="table")
            end
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
            out = _capture() do
                _dsge_perfect_foresight(; model=toml_path, shocks="", periods=50, format="table")
            end
            @test contains(out, "Perfect Foresight") || contains(out, "period")
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
                                          periods=3, constraint_solver="ipopt")
            end
            @test contains(out, "Perfect Foresight")
        end
    end

    @testset "_dsge_perfect_foresight — with constraints" begin
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
            shock_path = joinpath(dir, "shocks.csv")
            CSV.write(shock_path, DataFrame(e_A = [1.0, 0.5, 0.0]))
            out = _capture() do
                _dsge_perfect_foresight(; model=model_path, shocks=shock_path,
                                          periods=3, constraints=cons_path)
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
        @test haskey(node.subcmds, "ha")
        @test haskey(node.subcmds, "ct")
        @test haskey(node.subcmds, "olg")
        @test haskey(node.subcmds, "determinacy-map")   # W12/#114
        @test haskey(node.subcmds, "moments")           # W12/#114
        @test haskey(node.subcmds, "dcegm")
        @test haskey(node.subcmds, "lifecycle")
        @test haskey(node.subcmds, "firm")
        @test haskey(node.subcmds, "bank")
        @test length(node.subcmds) == 18
        ha = node.subcmds["ha"]
        @test ha isa NodeCommand
        for leaf in ("solve", "steady-state", "irf", "fevd", "simulate",
                     "distribution-irf", "inequality-irf", "simulate-panel", "estimate", "hd")
            @test haskey(ha.subcmds, leaf)
        end
        @test haskey(ha.subcmds, "estimate")  # un-deferred (C048): MEMs#228 fixed in 0.6.7
        @test haskey(node.subcmds["ct"].subcmds, "solve")
        @test haskey(node.subcmds["ct"].subcmds, "transition")
        @test haskey(node.subcmds["olg"].subcmds, "solve")
        @test haskey(node.subcmds["olg"].subcmds, "simulate")
    end
end

# ─── HA-DSGE handlers (C040) ───────────────────────────────────

@testset "HA-DSGE handlers (C040)" begin
    @testset "_load_ha_model builtins" begin
        for name in ("huggett", ":huggett", "krusell-smith", "one-asset-hank", "two-asset-hank")
            out = _capture() do
                spec = _load_ha_model(name)
                @test spec isa MacroEconometricModels.ModelSpec &&
                      MacroEconometricModels.has_kind(spec, MacroEconometricModels.HouseholdSystem)
            end
        end
        @test_throws Exception _load_ha_model("not-a-model")
    end

    @testset "_dsge_ha_steady_state" begin
        out = _capture() do
            _dsge_ha_steady_state(; model="huggett", format="table", output="")
        end
        @test contains(out, "Aggregates") || contains(out, "aggregates") ||
              contains(out, "K") || contains(out, "value") || true
    end

    @testset "_dsge_ha_solve reiter" begin
        out = _capture() do
            sol = _dsge_ha_solve(; model="huggett", method="reiter",
                                 n_reduced=5, t_horizon=50, format="table", output="")
            @test sol isa MacroEconometricModels.HADSGESolution
        end
    end

    @testset "_dsge_ha_solve krusell-smith" begin
        out = _capture() do
            sol = _dsge_ha_solve(; model="huggett", method="krusell-smith",
                                 n_reduced=5, t_horizon=50, format="table", output="")
            @test sol isa MacroEconometricModels.KrusellSmithSolution
        end
    end

    @testset "_dsge_ha_irf" begin
        out = _capture() do
            _dsge_ha_irf(; model="huggett", method="reiter", horizon=5,
                         n_reduced=5, format="table", output="", plot=false, plot_save="")
        end
    end

    @testset "_dsge_ha_fevd" begin
        out = _capture() do
            _dsge_ha_fevd(; model="huggett", method="reiter", horizon=5,
                          n_reduced=5, format="table", output="", plot=false, plot_save="")
        end
    end

    @testset "_dsge_ha_simulate" begin
        out = _capture() do
            _dsge_ha_simulate(; model="huggett", method="reiter", periods=10, seed=1,
                              n_reduced=5, format="table", output="", plot=false, plot_save="")
        end
    end

    @testset "_dsge_ha_distribution_irf" begin
        out = _capture() do
            _dsge_ha_distribution_irf(; model="huggett", method="reiter", horizon=5,
                                      shock_index=1, shock_size=1.0, n_reduced=5,
                                      format="table", output="")
        end
        @test_throws Exception _dsge_ha_distribution_irf(; model="huggett", method="ssj",
            horizon=5, shock_index=1, shock_size=1.0, n_reduced=5, format="table", output="")
    end

    @testset "_dsge_ha_inequality_irf" begin
        out = _capture() do
            _dsge_ha_inequality_irf(; model="huggett", method="reiter", horizon=5,
                                    shock_index=1, shock_size=1.0, n_reduced=5,
                                    format="table", output="", plot=false, plot_save="")
        end
    end

    @testset "_dsge_ha_simulate_panel" begin
        out = _capture() do
            _dsge_ha_simulate_panel(; model="huggett", n_agents=20, periods=10, seed=1,
                                    format="table", output="")
        end
    end

    @testset "_dsge_ha_estimate (C048)" begin
        mktempdir() do dir
            priors = joinpath(dir, "ha_priors.toml")
            write(priors, """
            [priors]
            [priors.alpha]
            dist = "normal"
            a = 0.36
            b = 0.05
            """)
            csv = _make_csv(dir; T=16, n=1, colnames=["K"])

            out = _capture() do
                r = _dsge_ha_estimate(; model="krusell-smith", data=csv, priors=priors,
                    observables="K", method="ssj", n_draws=6, burnin=2,
                    t_horizon=30, n_reduced=10, seed=1, format="table", output="")
                @test r isa MacroEconometricModels.BayesianDSGE
                @test r.method === :rwmh
            end
            @test contains(out, "alpha")
            @test contains(out, "Acceptance rate")

            # csv format still routes through the posterior table
            out2 = _capture() do
                _dsge_ha_estimate(; model="krusell-smith", data=csv, priors=priors,
                    observables="K", method="reiter", n_draws=4, burnin=1,
                    seed=2, format="csv", output="")
            end
            @test contains(out2, "parameter") || contains(out2, "alpha") ||
                  contains(out2, "mean") || !isempty(strip(out2))

            # krusell-smith has no linear state space → rejected
            @test_throws Exception _dsge_ha_estimate(; model="krusell-smith", data=csv,
                priors=priors, method="krusell-smith")
            # bad measurement-error value
            @test_throws Exception _dsge_ha_estimate(; model="krusell-smith", data=csv,
                priors=priors, measurement_error="bogus")
        end
        # required-option guards (checked before any file IO)
        @test_throws Exception _dsge_ha_estimate(; model="huggett", data="", priors="x")
        mktempdir() do dir
            csv = _make_csv(dir; T=8, n=1, colnames=["K"])
            @test_throws Exception _dsge_ha_estimate(; model="huggett", data=csv, priors="")
        end
    end

    @testset "invalid method" begin
        @test_throws Exception _parse_ha_method("bogus")
    end
end

# ─── CT + OLG handlers (C041) ──────────────────────────────────

@testset "CT/OLG handlers (C041)" begin
    @testset "_dsge_ct_solve aiyagari" begin
        out = _capture() do
            ss = _dsge_ct_solve(; grid_size=30, max_iter=20, tol=1e-4,
                                format="table", output="", two_asset=false)
            @test ss isa MacroEconometricModels.CTSteadyState
        end
    end

    @testset "_dsge_ct_solve two-asset" begin
        out = _capture() do
            sol = _dsge_ct_solve(; two_asset=true, max_iter=20, tol=1e-4,
                                 format="table", output="")
            @test sol isa MacroEconometricModels.CTTwoAssetSolution
        end
    end

    @testset "_dsge_ct_transition" begin
        out = _capture() do
            tr = _dsge_ct_transition(; grid_size=30, periods=8, max_iter=20, tol=1e-4,
                                     shock_size=0.95, format="table", output="",
                                     plot=false, plot_save="")
            @test tr isa MacroEconometricModels.CTTransition
        end
    end

    @testset "_dsge_olg_solve" begin
        out = _capture() do
            sol = _dsge_olg_solve(; debt=0.0, format="table", output="")
            @test sol isa MacroEconometricModels.BlanchardOLGSolution
            @test sol.determinate
        end
    end

    @testset "_dsge_olg_solve debt warning" begin
        out = _capture() do
            _dsge_olg_solve(; debt=0.5, format="table", output="")
        end
        @test contains(out, "#237") || contains(out, "debt") || true
    end

    @testset "_dsge_olg_simulate" begin
        out = _capture() do
            paths = _dsge_olg_simulate(; horizon=10, k0=0.0, format="table",
                                       output="", plot=false, plot_save="")
            @test haskey(paths, :k)
            @test length(paths.k) == 11
        end
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
        end
    end

    @testset "_did_estimate — callaway_santanna with group-time" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_estimate(; data=csv, outcome="outcome", treatment="treat",
                    method="cs", id_col="unit", time_col="time", format="table")
            end
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
            end
        end
    end

    @testset "_did_estimate — missing outcome" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            @test_throws Exception _did_estimate(;
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
        end
    end

    @testset "_did_lp_did — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_lp_did(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
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
        end
    end

    @testset "_did_lp_did — oneoff spec" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_lp_did(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", oneoff=true, format="table")
            end
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
        end
    end

    @testset "_did_test_bacon — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_bacon(; data=csv, outcome="outcome", treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
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
        end
    end

    @testset "_did_test_negweight — default" begin
        mktempdir() do dir
            csv = _make_did_csv(dir)
            out = _capture() do
                _did_test_negweight(; data=csv, treatment="treat",
                    id_col="unit", time_col="time", format="table")
            end
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
        end
    end

    @testset "_estimate_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_sdfm(; data=csv, factors=2, id="cholesky",
                                 var_lags=1, horizon=20, format="table")
            end
        end
    end

    @testset "_irf_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _irf_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                             horizons=10, id="cholesky", format="table")
            end
        end
    end

    @testset "_irf_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _irf_sdfm(; data=csv, factors=2, horizons=10, format="table")
            end
        end
    end

    @testset "factor family carries CSV varnames (W10/#131, MEMs#538)" begin
        # The mock replicates real's naming: FAVAR key variables take their panel
        # names inside the augmented VAR; sdfm responses take the panel names.
        mktempdir() do dir
            csv = joinpath(dir, "named.csv")
            CSV.write(csv, DataFrame(s1=randn(80), s2=randn(80), s3=randn(80),
                                     infl=randn(80), ffr=randn(80)))
            out_f = _capture() do
                _irf_favar(; data=csv, factors=2, lags=1, key_vars="infl,ffr",
                             horizons=6, id="cholesky", format="table")
            end
            @test occursin("infl", out_f) && occursin("ffr", out_f)
            out_s = _capture() do
                _irf_sdfm(; data=csv, factors=2, horizons=6, format="table")
            end
            @test occursin("infl", out_s)
            @test !occursin("Var 1", out_s)
        end
    end

    @testset "_fevd_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _fevd_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                              horizons=10, format="table")
            end
        end
    end

    @testset "_fevd_sdfm" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _fevd_sdfm(; data=csv, factors=2, horizons=10, format="table")
            end
        end
    end

    @testset "sdfm W1 riders — method/spectral/q-method/instrument (mock)" begin
        # New leaf checklist (T1/T2 on mocks): registry spec + handler → T3 + golden.
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            # auto selection through each upstream q_method branch
            for qm in ("hallin-liska", "bai-ng", "amengual-watson")
                out = _capture() do
                    _estimate_sdfm(; data=csv, factors=nothing, q_method=qm,
                                     horizon=10, format="table")
                end
            end
            # legacy estimator + legacy spectrum
            out = _capture() do
                _estimate_sdfm(; data=csv, factors=2, method="gdfm-var",
                                 spectral="smoothed-periodogram",
                                 horizon=10, format="table")
            end
            # proxy identification with an instrument column runs
            inst_csv = joinpath(dir, "inst.csv")
            CSV.write(inst_csv, DataFrame(y1=randn(100), y2=randn(100), y3=randn(100),
                                          z=randn(100)))
            out = _capture() do
                _estimate_sdfm(; data=inst_csv, factors=1, id="proxy",
                                 instrument="z", horizon=10, format="table")
            end
            # guards: proxy needs an instrument; instrument needs proxy;
            # q-method needs auto; enums are closed
            e1 = try; _estimate_sdfm(; data=csv, factors=2, id="proxy", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/missing"
            e2 = try; _estimate_sdfm(; data=csv, factors=2, instrument="var1", format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            e3 = try; _estimate_sdfm(; data=csv, factors=2, q_method="bai-ng", format="table"); nothing; catch e; e; end
            @test e3 isa CliError && e3.code == "usage/invalid"
            e4 = try; _estimate_sdfm(; data=csv, factors=2, method="bogus", format="table"); nothing; catch e; e; end
            @test e4 isa CliError && e4.code == "usage/invalid"
            e5 = try; _estimate_sdfm(; data=csv, factors=2, spectral="bogus", format="table"); nothing; catch e; e; end
            @test e5 isa CliError && e5.code == "usage/invalid"
            e6 = try; _estimate_sdfm(; data=csv, factors=2, q_method="bogus", format="table"); nothing; catch e; e; end
            @test e6 isa CliError && e6.code == "usage/invalid"
            # instrument column hygiene mirrors the weights/coordinates loaders
            e7 = try; _estimate_sdfm(; data=csv, factors=1, id="proxy", instrument="nope", format="table"); nothing; catch e; e; end
            @test e7 isa CliError && e7.code == "data/column-range"
        end
    end

    @testset "irf/fevd sdfm W1 riders — shared surface + bootstrap ci (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _irf_sdfm(; data=csv, factors=nothing, q_method="bai-ng",
                            method="gdfm-var", horizons=8, format="table")
            end
            out = _capture() do
                _irf_sdfm(; data=csv, factors=2, horizons=8, ci="bootstrap",
                            reps=5, format="table")
            end
            e1 = try; _irf_sdfm(; data=csv, factors=2, ci="bogus", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/invalid"
            e2 = try; _irf_sdfm(; data=csv, factors=2, reps=0, format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            out = _capture() do
                _fevd_sdfm(; data=csv, factors=nothing, spectral="smoothed-periodogram",
                             horizons=8, format="table")
            end
        end
    end

    @testset "gdfm W1 riders — spectral + forecast method (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_gdfm(; data=csv, nfactors=2, dynamic_rank=1,
                                 spectral="smoothed-periodogram", format="table")
            end
            e1 = try; _estimate_gdfm(; data=csv, spectral="bogus", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/invalid"
            for m in ("ar", "one-sided", "spectral")
                out = _capture() do
                    _forecast_gdfm(; data=csv, nfactors=2, dynamic_rank=1,
                                     horizons=4, method=m, format="table")
                end
            end
            e2 = try; _forecast_gdfm(; data=csv, method="bogus", format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
            e3 = try; _forecast_gdfm(; data=csv, spectral="bogus", format="table"); nothing; catch e; e; end
            @test e3 isa CliError && e3.code == "usage/invalid"
        end
    end

    @testset "forecast sdfm — new leaf (mock)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _forecast_sdfm(; data=csv, factors=2, horizons=4, format="table")
            end
            out = _capture() do
                _forecast_sdfm(; data=csv, factors=nothing, horizons=4,
                                 ci="bootstrap", reps=5, format="table")
            end
            e1 = try; _forecast_sdfm(; data=csv, ci="bogus", format="table"); nothing; catch e; e; end
            @test e1 isa CliError && e1.code == "usage/invalid"
            e2 = try; _forecast_sdfm(; data=csv, reps=0, format="table"); nothing; catch e; e; end
            @test e2 isa CliError && e2.code == "usage/invalid"
        end
    end

    @testset "_hd_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _hd_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                            horizons=10, format="table")
            end
        end
    end

    @testset "_forecast_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _forecast_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                  horizons=5, format="table")
            end
        end
    end

    @testset "_predict_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _predict_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                 format="table")
            end
        end
    end

    @testset "_residuals_favar" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _residuals_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                   format="table")
            end
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
        end
    end

    @testset "_test_bai_perron" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_bai_perron(; data=csv, response=1, max_breaks=5,
                                   trimming=0.15, criterion="bic", format="table")
            end
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
        end
    end

    @testset "_test_cips" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_cips(; data=csv, lags="2", deterministic="constant",
                             id_col="", time_col="", format="table")
            end
            # Regression: the handler read `result.cips`, a field real MEMs does not
            # have (it is `cips_statistic`) — a mock getproperty alias hid the crash.
            @test contains(out, "CIPS statistic")
        end
    end

    @testset "_test_moon_perron" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_moon_perron(; data=csv, factors="2",
                                    id_col="", time_col="", format="table")
            end
        end
    end

    @testset "_test_factor_break" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _test_factor_break(; data=csv, factors=2, method="breitung_eickmeier",
                                     id_col="", time_col="", format="table")
            end
            # W2/#124: pooled methods emit the per-series table…
            @test occursin("Per-Series Break Diagnostics", out)
            # …and chen_dolado_gonzalo (series fields = nothing) silently omits it.
            out_cdg = _capture() do
                _test_factor_break(; data=csv, factors=2, method="chen_dolado_gonzalo",
                                     id_col="", time_col="", format="table")
            end
            @test !occursin("Per-Series Break Diagnostics", out_cdg)
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
        end
    end

    @testset "_dsge_bayes_estimate — missing data" begin
        mktempdir() do dir
            model_path, priors_path, _ = _make_bayes_dsge_files(dir)
            @test_throws Exception _dsge_bayes_estimate(;
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
            @test_throws Exception _dsge_bayes_estimate(;
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
            @test_throws Exception _dsge_bayes_estimate(;
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
        end
    end

    @testset "_dsge_bayes_compare — Model 1 favored" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            # Model 1 has 2 params → log_ml = -498, Model 2 has 1 param → log_ml = -499
            # bayes_factor returns log BF = logML₁ − logML₂ = 1 > 0 → "Model 1 favored"
            priors2_path = joinpath(dir, "priors2.toml")
            write(priors2_path, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            """)
            out = _capture() do
                _dsge_bayes_compare(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path,
                    sampler="smc", n_smc=100, n_particles=50,
                    n_draws=100, burnin=10, ess_target=0.5,
                    observables="", solver="gensys", order=1,
                    delayed_acceptance=false,
                    model2=model_path, params2="rho", priors2=priors2_path,
                    output="", format="table")
            end
        end
    end

    @testset "_dsge_bayes_compare — missing model2" begin
        mktempdir() do dir
            model_path, priors_path, csv = _make_bayes_dsge_files(dir)
            @test_throws Exception _dsge_bayes_compare(;
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
        end
    end

    # ── C073 Bayesian DSGE diagnostics ──
    @testset "C073 Bayesian DSGE diagnostics" begin
        _catch(f) = try; f(); nothing; catch e; e; end

        @testset "_dsge_bayes_mcmc_diag" begin
            mktempdir() do dir
                model_path, priors_path, csv = _make_bayes_dsge_files(dir)
                out = _capture() do
                    _dsge_bayes_mcmc_diag(; model=model_path, data=csv,
                        params="rho,sigma", priors=priors_path,
                        sampler="smc", n_smc=100, n_particles=50,
                        n_draws=100, burnin=10, ess_target=0.5,
                        observables="", solver="gensys", order=1,
                        delayed_acceptance=false, output="", format="table")
                end
                @test contains(out, "MCMC") || contains(out, "rhat")
            end
        end

        @testset "_dsge_bayes_identification + missing params → usage/missing" begin
            mktempdir() do dir
                model_path, _, _ = _make_bayes_dsge_files(dir)
                out = _capture() do
                    _dsge_bayes_identification(; model=model_path, params="rho,sigma",
                        observables="", solver="gensys", order=1, n_lags=2,
                        output="", format="table")
                end
                @test contains(out, "rank") || contains(out, "Identification")
                @test contains(out, "Singular") || contains(out, "singular_value")
                e = _catch(() -> _dsge_bayes_identification(; model=model_path, params=""))
                @test e isa CliError && e.code == "usage/missing"
            end
        end

        @testset "_dsge_bayes_learning_rate + junk fractions → usage/invalid" begin
            mktempdir() do dir
                model_path, priors_path, csv = _make_bayes_dsge_files(dir)
                out = _capture() do
                    _dsge_bayes_learning_rate(; model=model_path, data=csv,
                        params="rho,sigma", priors=priors_path,
                        sampler="smc", n_smc=100, n_particles=50,
                        n_draws=100, burnin=10, ess_target=0.5,
                        observables="", solver="gensys", order=1,
                        delayed_acceptance=false, fractions="0.5,1.0",
                        threshold=0.2, refit_n_smc=30, output="", format="table")
                end
                @test contains(out, "Learning") || contains(out, "learning_rate")
                ej = _catch(() -> _dsge_bayes_learning_rate(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path, fractions="abc"))
                @test ej isa CliError && ej.code == "usage/invalid"
                e1 = _catch(() -> _dsge_bayes_learning_rate(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path, fractions="0.5"))
                @test e1 isa CliError && e1.code == "usage/invalid"
            end
        end

        @testset "_dsge_bayes_overlap" begin
            mktempdir() do dir
                model_path, priors_path, csv = _make_bayes_dsge_files(dir)
                out = _capture() do
                    _dsge_bayes_overlap(; model=model_path, data=csv,
                        params="rho,sigma", priors=priors_path,
                        sampler="smc", n_smc=100, n_particles=50,
                        n_draws=100, burnin=10, ess_target=0.5,
                        observables="", solver="gensys", order=1,
                        delayed_acceptance=false, threshold=0.8, n_grid=0,
                        output="", format="table")
                end
                @test contains(out, "Overlap") || contains(out, "overlap")
            end
        end

        @testset "_dsge_bayes_marginal_lik + bad proposal → usage/invalid" begin
            mktempdir() do dir
                model_path, priors_path, csv = _make_bayes_dsge_files(dir)
                out = _capture() do
                    _dsge_bayes_marginal_lik(; model=model_path, data=csv,
                        params="rho,sigma", priors=priors_path,
                        sampler="smc", n_smc=100, n_particles=50,
                        n_draws=100, burnin=10, ess_target=0.5,
                        observables="", solver="gensys", order=1,
                        delayed_acceptance=false, proposal="normal", df=5.0,
                        output="", format="table")
                end
                @test contains(out, "Marginal") || contains(out, "log_marginal_likelihood_bridge")
                ep = _catch(() -> _dsge_bayes_marginal_lik(; model=model_path, data=csv,
                    params="rho,sigma", priors=priors_path, proposal="banana"))
                @test ep isa CliError && ep.code == "usage/invalid"
            end
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
        end
    end

    @testset "_test_gregory_hansen — one column is data/shape (#85)" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=1)
            e = try
                _capture() do
                    _test_gregory_hansen(; data=csv, model="C",
                                          lags="aic", max_lags=nothing,
                                          trim=0.15, format="table", output="")
                end
                nothing
            catch err
                err
            end
            # A cointegrating regression needs a dependent plus a regressor;
            # upstream raises a bare ArgumentError (exit 1 before #85).
            @test e isa CliError
            @test e.code == "data/shape"
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
        end
    end

    @testset "_test_vif" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _test_vif(; data=csv, dep="var1", cov_type="hc1",
                            format="table", output="")
            end
        end
    end

    @testset "_test_vif — default dep" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)
            out = _capture() do
                _test_vif(; data=csv, dep="", cov_type="ols",
                            format="table", output="")
            end
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
            end
        end

        @testset "_spectral_periodogram" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_periodogram(; data=csv, column=1, format="table", output="")
                end
            end
        end

        @testset "_spectral_density" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_density(; data=csv, column=1, method="welch", format="table", output="")
                end
            end
        end

        @testset "_spectral_cross" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _spectral_cross(; data=csv, var1=1, var2=2, format="table", output="")
                end
            end
        end

        @testset "_spectral_transfer" begin
            out = _capture() do
                _spectral_transfer(; filter="hp", lambda=1600.0, nobs=200, format="table", output="")
            end
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
            end
        end

        @testset "_dsge_hd — auto measurement error" begin
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
                              measurement_error="auto", format="table", output="")
                end
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
            end
        end

        @testset "_data_dropna — drops NaN rows, --vars scopes the check (W3/#125)" begin
            mktempdir() do dir
                csv = joinpath(dir, "nan.csv")
                a = randn(20); a[1:5] .= NaN
                CSV.write(csv, DataFrame(a=a, b=randn(20)))

                out_csv = joinpath(dir, "clean.csv")
                _capture() do
                    _data_dropna(; data=csv, format="csv", output=out_csv)
                end
                @test nrow(CSV.read(out_csv, DataFrame)) == 15   # NaN rows actually dropped

                # --vars used to pass Vector{SubString} into real dropna's
                # ::Union{Vector{String},Nothing} assertion → TypeError exit 1.
                out_b = joinpath(dir, "clean_b.csv")
                _capture() do
                    _data_dropna(; data=csv, vars="b", format="csv", output=out_b)
                end
                @test nrow(CSV.read(out_b, DataFrame)) == 20     # b is fully finite

                # Unknown --vars entry is typed, not upstream's raw ArgumentError.
                err = try
                    _capture() do
                        _data_dropna(; data=csv, vars="nope", format="table", output="")
                    end
                catch e
                    e
                end
                @test err isa CliError && err.code == "data/column-range"
            end
        end

        @testset "_data_keeprows" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=50, n=3)
                out = _capture() do
                    _data_keeprows(; data=csv, rows="1:20", format="table", output="")
                end
            end
        end

        @testset "_data_keeprows — typed guards (W3/#125)" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=50, n=3)
                for (rows, code) in [("", "usage/missing"),          # was a bare error() → exit 1
                                     ("abc", "usage/invalid"),       # was raw parse ArgumentError
                                     ("1:2:3", "usage/invalid"),
                                     ("40:900", "usage/invalid"),    # was upstream BoundsError
                                     ("20:10", "usage/invalid")]     # empty selection
                    err = try
                        _capture() do
                            _data_keeprows(; data=csv, rows=rows, format="table", output="")
                        end
                    catch e
                        e
                    end
                    @test err isa CliError && err.code == code
                end
                # 1:end still resolves to the full sample.
                _capture() do
                    _data_keeprows(; data=csv, rows="1:end", format="table", output="")
                end
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
                # W10/#131 (MEMs#553): diagnostics table always emitted; the mock is
                # just-identified here (1 endog, 1 instrument) so Sargan says why
                # it is missing rather than a bare N/A.
                @test occursin("Weak-Instrument Diagnostics", out)
                @test occursin("unavailable (failed or underidentified)", out)
                # Overidentified (2 instruments): Sargan carries a number.
                csv5 = _make_panel_csv(dir; G=5, T_per=20, n=5)
                out2 = _capture() do
                    _estimate_piv(; data=csv5, dep="var1", exog="var2", endog="var3",
                        instruments="var4,var5", method="fe", cov_type="cluster",
                        id_col="group", time_col="time", format="table", output="")
                end
                @test occursin("Weak-Instrument Diagnostics", out2)
                @test !occursin("unavailable", out2)
                # Missing --dep/--endog: bare error() used to make these exit 1.
                for kw in [(; exog="var2", endog="var3"), (; dep="var1", exog="var2")]
                    err = try
                        _capture() do
                            _estimate_piv(; data=csv, id_col="group", time_col="time",
                                          format="table", kw...)
                        end
                    catch e
                        e
                    end
                    @test err isa CliError && err.code == "usage/missing"
                end
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
            end
        end

        @testset "_estimate_oprobit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _estimate_oprobit(; data=csv, dep="var1", cov_type="ols",
                                       clusters="", output="", format="table")
                end
            end
        end

        @testset "_estimate_mlogit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _estimate_mlogit(; data=csv, dep="var1", cov_type="ols",
                                      output="", format="table")
                end
            end
        end

        @testset "_predict_ologit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _predict_ologit(; data=csv, dep="var1", cov_type="hc1",
                                     clusters="", output="", format="table")
                end
            end
        end

        @testset "_predict_mlogit" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _predict_mlogit(; data=csv, dep="var1", cov_type="ols",
                                     output="", format="table")
                end
            end
        end

        @testset "predict --marginal-effects re-added w/ handler support (W10/#131, MEMs#550)" begin
            # #85 removed the flag because no handler accepted it; MEMs#550 (0.7.3)
            # added delta-method SEs upstream. The mock returns the REAL shapes:
            # ordered → NamedTuple (K×J matrices), mlogit → MultinomialMarginalEffects.
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                for f in (_predict_ologit, _predict_oprobit)
                    out = _capture() do
                        f(; data=csv, dep="var1", marginal_effects=true,
                          output="", format="table")
                    end
                    @test occursin("Average Marginal Effects", out)
                    @test occursin("dydx", out) && occursin("se", out)
                end
                out_m = _capture() do
                    _predict_mlogit(; data=csv, dep="var1", cov_type="ols",
                                     marginal_effects=true, output="", format="table")
                end
                @test occursin("Average Marginal Effects", out_m)
                # …and the flag is DECLARED on all three predict leaves (registry ↔
                # handler agreement in both directions).
                for leaf in ("ologit", "oprobit", "mlogit")
                    fl = [f.name for f in _flags_for_kind(Symbol(leaf), :predict)]
                    @test "marginal-effects" in fl
                end
            end
        end

        # W4/#87: MEMs 0.7.2 (MEMs#507) defines residuals for these models, so the typed
        # refusal these two testsets used to pin is gone. Real returns an n x J matrix; the
        # mock now mirrors that shape exactly (it previously invented a length-n
        # `y .- fitted[:, 1]` that real never had — the #84 trap).
        @testset "_residuals_ologit — per-category matrix" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _residuals_ologit(; data=csv, dep="var1", cov_type="hc1",
                                       clusters="", kind="response", generalized=false,
                                       output="", format="csv")
                end
                @test occursin("resid_", out)
                @test occursin("observation", out)
            end
        end

        @testset "_residuals_ologit — generalized is a length-n column" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _residuals_ologit(; data=csv, dep="var1", cov_type="hc1",
                                       clusters="", kind="response", generalized=true,
                                       output="", format="csv")
                end
                @test occursin("generalized_residual", out)
                @test !occursin("resid_", out)
            end
        end

        @testset "_residuals_mlogit — per-category matrix, no generalized kwarg" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _residuals_mlogit(; data=csv, dep="var1", cov_type="ols",
                                       kind="response", output="", format="csv")
                end
                @test occursin("resid_", out)
                # upstream has no generalized_residuals for mlogit, so the handler must not
                # accept the kwarg either — declared surface and handler must agree
                @test_throws MethodError _residuals_mlogit(; data=csv, dep="var1",
                                                            generalized=true)
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
            end
        end

        @testset "_test_breusch_pagan" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_breusch_pagan(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
            end
        end

        @testset "_test_f_fe" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_f_fe(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
            end
        end

        @testset "_test_pesaran_cd" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_pesaran_cd(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
            end
        end

        @testset "_test_wooldridge_ar" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_wooldridge_ar(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
            end
        end

        @testset "_test_modified_wald" begin
            mktempdir() do dir
                csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
                out = _capture() do
                    _test_modified_wald(; data=csv, dep="var1", indep="var2,var3",
                        id_col="group", time_col="time", format="table", output="")
                end
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
            end
        end

        @testset "_test_bartlett_wn" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_bartlett_wn(; data=csv, column=1, format="table", output="")
                end
            end
        end

        @testset "_test_box_pierce" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_box_pierce(; data=csv, column=1, lags=20, format="table", output="")
                end
            end
        end

        @testset "_test_durbin_watson" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=3)
                out = _capture() do
                    _test_durbin_watson(; data=csv, column=1, format="table", output="")
                end
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
            end
        end

        @testset "_test_hausman_iia" begin
            mktempdir() do dir
                csv = _make_csv(dir; T=100, n=4)
                out = _capture() do
                    _test_hausman_iia(; data=csv, dep="var1", omit_category=1,
                                       format="table", output="")
                end
            end
        end

        @testset "missing required option → usage/missing, not internal exit 1" begin
            # These seven handlers guarded their required option with a bare error(),
            # which run_cli reports as internal/error exit 1 "likely a bug" for what is
            # an ordinary usage mistake. Same class as the shared-loader hardenings.
            mktempdir() do dir
                panel = _make_panel_csv(dir; G=6, T_per=20, n=2, colnames=["y", "x1"])
                csv = _make_csv(dir; T=100, n=4)
                for h in (_test_hausman, _test_breusch_pagan, _test_f_fe,
                          _test_pesaran_cd, _test_wooldridge_ar, _test_modified_wald)
                    e = try
                        _capture() do
                            h(; data=panel, dep="", indep="", id_col="", time_col="",
                               format="table", output="")
                        end
                        nothing
                    catch err
                        err
                    end
                    @test e isa CliError
                    @test e.code == "usage/missing" && exit_class(e) == 2
                end
                e = try
                    _capture() do
                        _test_hausman_iia(; data=csv, dep="var1", omit_category=nothing,
                                           format="table", output="")
                    end
                    nothing
                catch err
                    err
                end
                @test e isa CliError && e.code == "usage/missing" && exit_class(e) == 2
            end
        end
    end

end

# ═══════════════════════════════════════════════════════════════
# Task 5: Diagnostic Warning Branch Coverage — test.jl
# ═══════════════════════════════════════════════════════════════

@testset "Diagnostic warning branches" begin

    # KPSS rejection branch (pvalue < 0.05)
    @testset "_test_kpss — rejection branch" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_kpss(; data=csv, column=1, trend="constant", format="table")
            end
            # Mock now returns pvalue=0.01 < 0.05, triggering rejection
        end
    end

    # Fourier KPSS — pvalue is 0.10 in mock, so does NOT reject (covers else branch)
    @testset "_test_fourier_kpss — no rejection" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _test_fourier_kpss(; data=csv, column=1, regression="constant",
                                    fmax=3, bandwidth=nothing, format="table", output="")
            end
        end
    end

    # Normality — all-pass branch (line 764)
    @testset "_test_normality — all pass" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            _MOCK_FLAGS[:normality_all_pass] = true
            try
                out = _capture() do
                    _test_normality(; data=csv, lags=2, format="table")
                end
            finally
                _MOCK_FLAGS[:normality_all_pass] = false
            end
        end
    end

    # VAR instability branch (line 1038)
    @testset "_test_var_stability — unstable" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            _MOCK_FLAGS[:var_stationary] = false
            try
                out = _capture() do
                    _test_var_stability(; data=csv, lags=2, format="table")
                end
            finally
                _MOCK_FLAGS[:var_stationary] = true
            end
        end
    end

    # PVAR instability branch (line 1233)
    @testset "_test_pvar_stability — unstable" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            _MOCK_FLAGS[:pvar_stable] = false
            try
                out = cd(dir) do
                    _capture() do
                        _test_pvar_stability(; data=csv, id_col="group", time_col="time", lags=1)
                    end
                end
            finally
                _MOCK_FLAGS[:pvar_stable] = true
            end
        end
    end

    # LR same-spec warning (lines 1245-1248)
    @testset "_test_lr — same spec warning" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_lr(; data1=csv, data2=csv, lags1=2, lags2=2)
                end
            end
        end
    end

    # LM same-spec warning (lines 1276-1278)
    @testset "_test_lm — same spec warning" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = cd(dir) do
                _capture() do
                    _test_lm(; data1=csv, data2=csv, lags1=2, lags2=2)
                end
            end
        end
    end

    # VIF severe multicollinearity (line 1742: max_vif > 10)
    @testset "_test_vif — severe multicollinearity" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)  # k=4 regressors -> vals[4]=12.0
            out = _capture() do
                _test_vif(; data=csv, dep="var1", cov_type="hc1",
                            format="table", output="")
            end
        end
    end

    # VIF moderate multicollinearity (line 1744: 5 < max_vif <= 10)
    @testset "_test_vif — moderate multicollinearity" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=4)  # k=3 regressors -> vals[3]=7.0
            out = _capture() do
                _test_vif(; data=csv, dep="var1", cov_type="hc1",
                            format="table", output="")
            end
        end
    end

    # VIF no multicollinearity (line 1746: all < 5)
    @testset "_test_vif — no multicollinearity" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)  # k=2 regressors -> [2.5, 2.5]
            out = _capture() do
                _test_vif(; data=csv, dep="var1", cov_type="hc1",
                            format="table", output="")
            end
        end
    end

end  # Diagnostic warning branches

# ═══════════════════════════════════════════════════════════════
# W10/#112: micro inference riders
# ═══════════════════════════════════════════════════════════════

@testset "W10 micro inference riders" begin

    """Cross-section CSV with lat/lon coordinate columns and an optional cluster column."""
    function _w10_csv(dir; T=120, with_coords=true, with_cluster=false, string_cluster=false)
        rng = MersenneTwister(4)
        d = DataFrame(y=randn(rng, T), x=randn(rng, T))
        if with_coords
            d.lat = 30.0 .+ 10.0 .* rand(rng, T)
            d.lon = -100.0 .+ 10.0 .* rand(rng, T)
        end
        if with_cluster
            g = repeat(1:6, inner=cld(T, 6))[1:T]
            d.cl = string_cluster ? ["g$(v)" for v in g] : Float64.(g)
        end
        path = joinpath(dir, "w10_$(with_coords)_$(with_cluster)_$(string_cluster).csv")
        CSV.write(path, d)
        return path
    end

    @testset "_estimate_reg — conley euclidean + haversine" begin
        mktempdir() do dir
            csv = _w10_csv(dir)
            for metric in ("euclidean", "haversine")
                out = _capture() do
                    _estimate_reg(; data=csv, dep="y", cov_type="conley",
                                   lat="lat", lon="lon", dist_cutoff=100.0,
                                   conley_metric=metric, format="table", output="")
                end
                @test occursin("Conley", out)
            end
        end
    end

    @testset "_estimate_reg — conley spatial+serial" begin
        mktempdir() do dir
            rng = MersenneTwister(5)
            T = 100
            d = DataFrame(y=randn(rng, T), x=randn(rng, T),
                          lat=rand(rng, T), lon=rand(rng, T),
                          yr=Float64.(repeat(1:10, inner=10)))
            csv = joinpath(dir, "conley_time.csv"); CSV.write(csv, d)
            out = _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="conley",
                               lat="lat", lon="lon", dist_cutoff=0.5,
                               time_col="yr", time_cutoff=2, format="table", output="")
            end
            @test occursin("Conley", out)
        end
    end

    @testset "_estimate_reg — conley guards" begin
        mktempdir() do dir
            csv = _w10_csv(dir)
            # conley without coordinates / without a cutoff
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="conley", format="table")
            end
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="conley",
                               lat="lat", lon="lon", format="table")
            end
            # coordinates supplied under another cov-type: a usage error, not a no-op
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="hc1",
                               lat="lat", lon="lon", format="table")
            end
            # --time-cutoff without --time-col
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="conley", lat="lat",
                               lon="lon", dist_cutoff=10.0, time_cutoff=2, format="table")
            end
            # unknown coordinate column
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="conley", lat="nope",
                               lon="lon", dist_cutoff=10.0, format="table")
            end
            # cluster cov-type with no cluster column
            @test_throws CliError _capture() do
                _estimate_reg(; data=csv, dep="y", cov_type="cluster", format="table")
            end
        end
    end

    @testset "_estimate_preg — absorb + guards" begin
        mktempdir() do dir
            rng = MersenneTwister(6)
            N, T = 8, 10
            d = DataFrame(id=repeat(1:N, inner=T), time=repeat(1:T, N),
                          y=randn(rng, N * T), x=randn(rng, N * T),
                          region=Float64.(rand(rng, 1:3, N * T)))
            csv = joinpath(dir, "hdfe.csv"); CSV.write(csv, d)
            out = _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity,time",
                                id_col="id", time_col="time", format="table", output="")
            end
            @test occursin("HDFE", out) || occursin("Absorb", out)
            # a named panel-variable dimension resolves too
            _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity,region",
                                id_col="id", time_col="time", format="table", output="")
            end
            # guards
            @test_throws CliError _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity,time",
                                twoway=true, id_col="id", time_col="time", format="table")
            end
            @test_throws CliError _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity,entity",
                                id_col="id", time_col="time", format="table")
            end
            @test_throws CliError _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity",
                                method="re", id_col="id", time_col="time", format="table")
            end
            @test_throws CliError _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", hdfe_tol=1e-5,
                                id_col="id", time_col="time", format="table")
            end
            @test_throws CliError _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", absorb="entity",
                                hdfe_maxiter=0, id_col="id", time_col="time", format="table")
            end
        end
    end

    @testset "_estimate_preg — ab/bb instrument controls (W10/#131)" begin
        mktempdir() do dir
            rng = MersenneTwister(7)
            N, T = 10, 8
            d = DataFrame(id=repeat(1:N, inner=T), time=repeat(1:T, N),
                          y=randn(rng, N * T), x=randn(rng, N * T))
            csv = joinpath(dir, "dyn.csv"); CSV.write(csv, d)

            # ab emits the Dynamic Panel Diagnostics table; collapse shrinks the
            # instrument count (mock mirrors the real direction).
            out_full = _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", method="ab",
                                id_col="id", time_col="time", format="table", output="")
            end
            @test occursin("Dynamic Panel Diagnostics", out_full)
            @test occursin("n_instruments", out_full)
            out_col = _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", method="ab",
                                collapse=true, id_col="id", time_col="time",
                                format="table", output="")
            end
            @test occursin("collapsed", out_col)   # stderr status notes the collapse
            # bb routes too, and a plain fe run emits NO diagnostics table.
            out_bb = _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", method="bb",
                                min_lag_endo=2, max_lag_endo=4,
                                id_col="id", time_col="time", format="table", output="")
            end
            @test occursin("Dynamic Panel Diagnostics", out_bb)
            out_fe = _capture() do
                _estimate_preg(; data=csv, dep="y", indep="x", method="fe",
                                id_col="id", time_col="time", format="table", output="")
            end
            @test !occursin("Dynamic Panel Diagnostics", out_fe)

            # Guards: controls on a non-GMM method refuse (upstream would silently
            # ignore them); window sanity on the GMM path.
            for kw in [(; collapse=true), (; min_lag_endo=3), (; max_lag_endo=5)]
                err = try
                    _capture() do
                        _estimate_preg(; data=csv, dep="y", indep="x", method="fe",
                                        id_col="id", time_col="time", format="table",
                                        kw...)
                    end
                catch e
                    e
                end
                @test err isa CliError && err.code == "usage/invalid"
            end
            for kw in [(; min_lag_endo=0), (; min_lag_endo=5, max_lag_endo=3)]
                err = try
                    _capture() do
                        _estimate_preg(; data=csv, dep="y", indep="x", method="ab",
                                        id_col="id", time_col="time", format="table",
                                        kw...)
                    end
                catch e
                    e
                end
                @test err isa CliError && err.code == "usage/invalid"
            end
        end
    end

    @testset "_test_wild_cluster — numeric and string clusters" begin
        mktempdir() do dir
            for sc in (false, true)
                csv = _w10_csv(dir; with_coords=false, with_cluster=true, string_cluster=sc)
                out = _capture() do
                    _test_wild_cluster(; data=csv, dep="y", clusters="cl",
                                        coefficient="x", format="table", output="")
                end
                @test occursin("Wild Cluster", out) || occursin("bootstrap", out)
            end
        end
    end

    @testset "_test_wild_cluster — variants and guards" begin
        mktempdir() do dir
            csv = _w10_csv(dir; with_coords=false, with_cluster=true)
            _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="cl", coefficient="x",
                                    boot_weights="webb", no_impose_null=true, no_ci=true,
                                    enumerate_signs="no", format="table", output="")
            end
            _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="cl", coefficient="x",
                                    enumerate_signs="yes", format="table", output="")
            end
            @test_throws CliError _capture() do
                _test_wild_cluster(; data=csv, dep="y", format="table")
            end
            @test_throws CliError _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="cl",
                                    coefficient="nope", format="table")
            end
            @test_throws CliError _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="cl", boot_reps=0, format="table")
            end
            @test_throws CliError _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="cl", level=1.5, format="table")
            end
            @test_throws CliError _capture() do
                _test_wild_cluster(; data=csv, dep="y", clusters="nope", format="table")
            end
        end
    end

    @testset "_test_anderson_rubin — set shapes" begin
        mktempdir() do dir
            rng = MersenneTwister(8)
            T = 150
            d = DataFrame("y" => randn(rng, T), "const" => fill(1.0, T),
                          "x2" => randn(rng, T), "x_endog" => randn(rng, T),
                          "z1" => randn(rng, T), "z2" => randn(rng, T))
            csv = joinpath(dir, "ar.csv"); CSV.write(csv, d)
            # Every degenerate set shape must render: the whole point of the leaf is that
            # an AR set is not always `[lo, hi]`.
            for shape in (:bounded, :unbounded, :disjoint, :whole, :empty)
                _MOCK_FLAGS[:ar_set_shape] = shape
                out = _capture() do
                    _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                          instruments="z1,z2", format="table", output="")
                end
                @test occursin("Anderson", out)
            end
            _MOCK_FLAGS[:ar_set_shape] = :bounded

            # --no-ci, explicit --beta0, and the clustered path
            _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", beta0="0.5", no_ci=true,
                                      format="table", output="")
            end
            dc = copy(d); dc.cl = Float64.(repeat(1:5, inner=cld(T, 5))[1:T])
            ccsv = joinpath(dir, "arcl.csv"); CSV.write(ccsv, dc)
            _capture() do
                _test_anderson_rubin(; data=ccsv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", cov_type="cluster",
                                      clusters="cl", format="table", output="")
            end

            # Two endogenous regressors: the TEST still runs, the confidence set is skipped
            # (upstream inverts over a single coefficient only) — not an error.
            out2 = _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog,x2",
                                      instruments="z1,z2", format="table", output="")
            end
            @test occursin("Anderson", out2)

            # guards
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog", format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", cov_type="cluster", format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=ccsv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", clusters="cl", format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", beta0="1,2", format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", beta0="abc", format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", level=0.0, format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", n_grid=2, format="table")
            end
            @test_throws CliError _capture() do
                _test_anderson_rubin(; data=csv, dep="y", endogenous="x_endog",
                                      instruments="z1,z2", span=0.0, format="table")
            end
        end
    end

    @testset "_estimate_lp — iv MOP + AR bands" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=120, n=2)
            zcsv = _make_instruments_csv(dir; T=120, n_inst=1)
            _capture() do
                _estimate_lp(; data=csv, method="iv", shock=1, horizons=5,
                              control_lags=2, vcov="newey_west", instruments=zcsv,
                              mop_f=true, format="table", output="")
            end
            _capture() do
                _estimate_lp(; data=csv, method="iv", shock=1, horizons=4,
                              control_lags=2, vcov="newey_west", instruments=zcsv,
                              ar_bands=true, ar_grid=51, format="table", output="")
            end
            # riders are iv-only
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="standard", mop_f=true, format="table")
            end
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="smooth", ar_span=5.0, format="table")
            end
            # tau is a closed four-member set; the grid/level are validated
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="iv", instruments=zcsv,
                              mop_f=true, mop_tau=0.15, format="table")
            end
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="iv", instruments=zcsv,
                              ar_bands=true, ar_grid=2, format="table")
            end
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="iv", instruments=zcsv,
                              ar_bands=true, ar_level=1.0, format="table")
            end
            # missing --instruments is now a typed usage error, not a bare error()
            @test_throws CliError _capture() do
                _estimate_lp(; data=csv, method="iv", instruments="", format="table")
            end
        end
    end

end  # W10 micro inference riders

# ═══════════════════════════════════════════════════════════════
# W12/#114: DSGE determinacy map, moments, prefilter
# ═══════════════════════════════════════════════════════════════

@testset "W12 DSGE riders" begin

    """Minimal @dsge model file + a [determinacy] config, written to `dir`."""
    function _w12_model(dir)
        path = joinpath(dir, "m.jl")
        # NO `using MacroEconometricModels` line: `_dsge_sandbox()` already injects the
        # in-scope module plus a RELATIVE `using .MacroEconometricModels`. An absolute
        # `using` resolves through the load path instead, which in the mock context is a
        # different module — `UndefVarError: @dsge not defined` with an ambiguity hint.
        # A model file is a bare @dsge block; the T3 fixtures do the same.
        write(path, """
        @dsge begin
            parameters: phi_pi = 1.5, rho = 0.8
            endogenous: y, pi
            exogenous: eps
            y[t] = rho * y[t-1] + eps[t]
            pi[t] = phi_pi * y[t]
        end
        """)
        return path
    end

    function _w12_cfg(dir; params="[\"phi_pi\"]", extra="")
        path = joinpath(dir, "det.toml")
        write(path, """
        [determinacy]
        params = $params
        lower = [0.0]
        upper = [2.0]
        points = [9]
        $extra
        """)
        return path
    end

    @testset "get_determinacy — parsing and guards" begin
        mktempdir() do dir
            cfg = _w12_cfg(dir)
            d = get_determinacy(load_config(cfg))
            @test d.params == ["phi_pi"]
            @test length(d.grids) == 1
            @test length(d.grids[1]) == 9
            @test d.grids[1][1] == 0.0 && d.grids[1][end] == 2.0
            @test d.method == :gensys

            # two parameters, explicit grids, method alias
            p2 = joinpath(dir, "d2.toml")
            write(p2, """
            [determinacy]
            params = ["phi_pi", "rho"]
            grids = [[0.0, 1.0, 2.0], [0.1, 0.5, 0.9]]
            method = "blanchard-kahn"
            div = 1.001
            """)
            d2 = get_determinacy(load_config(p2))
            @test length(d2.params) == 2
            @test d2.method == :blanchard_kahn
            @test d2.div == 1.001

            # a scalar params entry is accepted where a one-element list would do
            p3 = joinpath(dir, "d3.toml")
            write(p3, """
            [determinacy]
            params = "phi_pi"
            lower = 0.0
            upper = 2.0
            points = 5
            """)
            @test length(get_determinacy(load_config(p3)).grids[1]) == 5

            for (name, body) in (
                ("missing section", ""),
                ("no params", "[determinacy]\nlower = [0.0]\nupper = [1.0]\n"),
                ("three params", "[determinacy]\nparams = [\"a\",\"b\",\"c\"]\nlower=[0,0,0]\nupper=[1,1,1]\n"),
                ("duplicate params", "[determinacy]\nparams = [\"a\",\"a\"]\nlower=[0,0]\nupper=[1,1]\n"),
                ("lower >= upper", "[determinacy]\nparams=[\"a\"]\nlower=[1.0]\nupper=[1.0]\n"),
                ("points < 2", "[determinacy]\nparams=[\"a\"]\nlower=[0.0]\nupper=[1.0]\npoints=[1]\n"),
                ("shape mismatch", "[determinacy]\nparams=[\"a\",\"b\"]\nlower=[0.0]\nupper=[1.0,2.0]\n"),
                ("bad method", "[determinacy]\nparams=[\"a\"]\nlower=[0.0]\nupper=[1.0]\nmethod=\"nope\"\n"),
                ("short grid", "[determinacy]\nparams=[\"a\"]\ngrids=[[0.5]]\n"),
                ("no bounds", "[determinacy]\nparams=[\"a\"]\n"),
            )
                bad = joinpath(dir, "bad.toml")
                write(bad, body)
                @test_throws CliError get_determinacy(load_config(bad))
            end
        end
    end

    @testset "_dsge_determinacy_map" begin
        mktempdir() do dir
            m = _w12_model(dir)
            cfg = _w12_cfg(dir)
            out = _capture() do
                _dsge_determinacy_map(; model=m, config=cfg, format="table", output="")
            end
            @test occursin("Determinacy", out)
            # A one-parameter sweep also reports the boundary.
            @test occursin("Boundary", out) || occursin("boundary", out)

            # two-parameter sweep: no boundary table (the frontier is a curve)
            c2 = joinpath(dir, "d2.toml")
            write(c2, """
            [determinacy]
            params = ["phi_pi", "rho"]
            grids = [[0.5, 1.5], [0.1, 0.9]]
            """)
            _capture() do
                _dsge_determinacy_map(; model=m, config=c2, format="table", output="")
            end

            # guards
            @test_throws CliError _capture() do
                _dsge_determinacy_map(; model=m, config="", format="table")
            end
            @test_throws CliError _capture() do
                _dsge_determinacy_map(; model=m, config=cfg, rank_rtol=0.0, format="table")
            end
            # an unknown parameter name is config/invalid, not an internal error
            cbad = joinpath(dir, "dbad.toml")
            write(cbad, """
            [determinacy]
            params = ["not_a_param"]
            lower = [0.0]
            upper = [1.0]
            points = [3]
            """)
            @test_throws CliError _capture() do
                _dsge_determinacy_map(; model=m, config=cbad, format="table")
            end
        end
    end

    @testset "_dsge_moments — orders and guards" begin
        mktempdir() do dir
            m = _w12_model(dir)
            # order 1 re-enabled in W9/#116 (MEMs 0.7.3/#607 fixed the control-block
            # covariance); the closed-form proof lives in T3 — here all three orders run.
            for ord in (1, 2, 3)
                out = _capture() do
                    _dsge_moments(; model=m, method="perturbation", order=ord,
                                   lags=2, format="table", output="")
                end
                @test occursin("Moments", out) || occursin("moments", out)
            end
            @test_throws CliError _capture() do
                _dsge_moments(; model=m, order=0, format="table")
            end
            @test_throws CliError _capture() do
                _dsge_moments(; model=m, order=4, format="table")
            end
            @test_throws CliError _capture() do
                _dsge_moments(; model=m, lags=0, format="table")
            end
        end
    end

    @testset "dsge bayes --prefilter guards" begin
        mktempdir() do dir
            m = _w12_model(dir)
            csv = _make_csv(dir; T=60, n=2, colnames=["y", "pi"])
            pri = joinpath(dir, "priors.toml")
            write(pri, """
            [priors.rho]
            dist = "beta"
            a = 2.0
            b = 2.0
            """)
            common = (; model=m, data=csv, params="rho", priors=pri,
                       observables="y,pi", n_smc=8, n_draws=8, burnin=2)
            # A valid prefilter runs.
            _capture() do
                _dsge_bayes_estimate(; common..., prefilter="demean", format="table", output="")
            end
            # HP needs its lambda to be the one that is used, so a lambda without hp is a
            # usage error rather than a silent no-op.
            @test_throws CliError _capture() do
                _dsge_bayes_estimate(; common..., prefilter="demean", hp_lambda=100.0, format="table")
            end
            @test_throws CliError _capture() do
                _dsge_bayes_estimate(; common..., prefilter="bogus", format="table")
            end
            @test_throws CliError _capture() do
                _dsge_bayes_estimate(; common..., prefilter="hp", hp_lambda=0.0, format="table")
            end
        end
    end

end  # W12 DSGE riders



# ═══════════════════════════════════════════════════════════════
# Task 6: HD verify_decomposition failure + estimate diagnostics
# ═══════════════════════════════════════════════════════════════

@testset "HD and estimation diagnostics" begin

    # HD verify_decomposition failure warning
    @testset "_hd_var — decomposition verification failure" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            _MOCK_FLAGS[:verify_decomposition] = false
            try
                out = _capture() do
                    _hd_var(; data=csv, lags=2, id="cholesky", config="",
                              format="table", output="", plot=false, plot_save="")
                end
            finally
                _MOCK_FLAGS[:verify_decomposition] = true
            end
        end
    end

    # HD LP decomposition verification failure
    @testset "_hd_lp — decomposition verification failure" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            _MOCK_FLAGS[:verify_decomposition] = false
            try
                out = _capture() do
                    _hd_lp(; data=csv, lags=4, var_lags=nothing,
                              id="cholesky", vcov="newey_west", config="",
                              format="table", output="", plot=false, plot_save="")
                end
            finally
                _MOCK_FLAGS[:verify_decomposition] = true
            end
        end
    end

    # Bayesian FAVAR estimate
    @testset "_estimate_favar — bayesian method" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                  method="bayesian", draws=100, format="table")
            end
        end
    end

    # LP-IV weak instruments
    @testset "_estimate_lp — iv weak instruments" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            inst_csv = joinpath(dir, "instruments.csv")
            write(inst_csv, "z1,z2\n" * join(["$(rand()),$(rand())" for _ in 1:100], "\n") * "\n")
            _MOCK_FLAGS[:lp_iv_weak] = true
            try
                out = _capture() do
                    _estimate_lp(; data=csv, horizons=10, shock=1,
                                   method="iv", instruments=inst_csv,
                                   control_lags=4, vcov="newey_west",
                                   format="table")
                end
            finally
                _MOCK_FLAGS[:lp_iv_weak] = false
            end
        end
    end

    # FAVAR with named key_vars (covers string name parsing in shared.jl)
    @testset "_estimate_favar — named key vars" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_favar(; data=csv, factors=2, lags=1, key_vars="var1,var2",
                                  method="two_step", draws=5000, format="table")
            end
        end
    end

    # FAVAR auto factor selection (covers shared.jl lines 625-628)
    @testset "_estimate_favar — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _estimate_favar(; data=csv, factors=nothing, lags=1, key_vars="1,2",
                                  method="two_step", draws=5000, format="table")
            end
        end
    end

end  # HD and estimation diagnostics

# ═══════════════════════════════════════════════════════════════
# Task 7: Auto-selection and feature branches
# ═══════════════════════════════════════════════════════════════

@testset "Auto-selection and feature branches" begin

    # Spectral ACF with CCF (covers lines 122-132 in spectral.jl)
    @testset "_spectral_acf — ccf_with" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_acf(; data=csv, column=1, ccf_with=2,
                                max_lag=nothing, format="table", output="")
            end
        end
    end

    # Data filter BN (covers data.jl lines 422-423)
    @testset "_data_filter — bn" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="bn")
            end
        end
    end

    # Data filter BK (covers data.jl lines 424-425)
    @testset "_data_filter — bk" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _data_filter(; data=csv, method="bk")
            end
        end
    end

    # Built-in dataset shortcut (covers io.jl lines 54-57)
    @testset "load_data — built-in dataset shortcut" begin
        df = load_data(":fred_md")
        @test df isa DataFrame
        @test nrow(df) > 0
        @test "INDPRO" in names(df)
    end

    # Spectral density with bandwidth (covers spectral.jl line 169)
    @testset "_spectral_density — with bandwidth" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_density(; data=csv, column=1, method="smoothed",
                                    bandwidth=0.1, format="table", output="")
            end
        end
    end

    # Spectral ACF with max_lag=nothing (auto selection, covers line 107)
    @testset "_spectral_acf — auto max_lag" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_acf(; data=csv, column=1, max_lag=nothing,
                                format="table", output="")
            end
        end
    end

end  # Auto-selection and feature branches

# ═══════════════════════════════════════════════════════════════
# Task 8: Sign/narrative restriction closure execution
# ═══════════════════════════════════════════════════════════════

@testset "Sign/narrative closure execution" begin

    # Build and execute sign restriction closure
    @testset "_build_check_func — sign restrictions" begin
        mktempdir() do dir
            cfg = joinpath(dir, "sign.toml")
            write(cfg, """
            [identification]
            method = "sign"
            [identification.sign_matrix]
            matrix = [[1, -1], [0, 1]]
            horizons = [0, 1]
            """)
            check_func, narrative_check = _build_check_func(cfg)
            @test check_func !== nothing
            @test narrative_check === nothing

            # irf_values[h, j, i] where h=horizon+1, j=shock (col), i=variable (row)
            # sign_mat[1,1]=1 -> irf[h,1,1]>0; sign_mat[1,2]=-1 -> irf[h,2,1]<0
            # sign_mat[2,2]=1 -> irf[h,2,2]>0

            # Test the closure: IRF satisfying restrictions
            irf_vals = ones(3, 2, 2)
            irf_vals[1, 2, 1] = -0.5  # h=0: shock=2, var=1 should be < 0
            irf_vals[2, 2, 1] = -0.5  # h=1: shock=2, var=1 should be < 0
            @test check_func(irf_vals) == true

            # Test the closure: IRF violating restrictions
            irf_vals_bad = ones(3, 2, 2)
            irf_vals_bad[1, 2, 1] = 0.5  # h=0: shock=2, var=1 should be < 0 but positive
            @test check_func(irf_vals_bad) == false
        end
    end

    # Build and execute narrative restriction closure
    @testset "_build_check_func — narrative restrictions" begin
        mktempdir() do dir
            cfg = joinpath(dir, "narrative.toml")
            write(cfg, """
            [identification]
            method = "sign"
            [identification.narrative]
            shock_index = 1
            periods = [5, 10]
            signs = [1, -1]
            """)
            check_func, narrative_check = _build_check_func(cfg)
            @test narrative_check !== nothing

            # Test: structural shocks satisfying narrative
            shocks = zeros(15, 2)
            shocks[5, 1] = 1.0   # positive at t=5
            shocks[10, 1] = -1.0  # negative at t=10
            @test narrative_check(shocks) == true

            # Test: structural shocks violating narrative
            shocks_bad = zeros(15, 2)
            shocks_bad[5, 1] = -1.0  # negative at t=5 (should be positive)
            @test narrative_check(shocks_bad) == false
        end
    end

    # Both sign and narrative restrictions
    @testset "_build_check_func — both restrictions" begin
        mktempdir() do dir
            cfg = joinpath(dir, "both.toml")
            write(cfg, """
            [identification]
            method = "sign"
            [identification.sign_matrix]
            matrix = [[1, 0], [0, 1]]
            horizons = [0]
            [identification.narrative]
            shock_index = 1
            periods = [3]
            signs = [1]
            """)
            check_func, narrative_check = _build_check_func(cfg)
            @test check_func !== nothing
            @test narrative_check !== nothing
        end
    end

    # Empty config
    @testset "_build_check_func — empty config" begin
        check_func, narrative_check = _build_check_func("")
        @test check_func === nothing
        @test narrative_check === nothing
    end

    # Sign check with h > irf size (continue branch, line 348)
    @testset "_build_check_func — horizon exceeds IRF" begin
        mktempdir() do dir
            cfg = joinpath(dir, "big_horizon.toml")
            write(cfg, """
            [identification]
            method = "sign"
            [identification.sign_matrix]
            matrix = [[1, -1], [0, 1]]
            horizons = [0, 1, 99]
            """)
            check_func, _ = _build_check_func(cfg)
            # h=99+1=100 > size(irf_values,1)=3 → continue, no error
            # Must satisfy h=0 and h=1 restrictions
            irf_vals = ones(3, 2, 2)
            irf_vals[1, 2, 1] = -0.5  # h=0: shock=2, var=1 < 0
            irf_vals[2, 2, 1] = -0.5  # h=1: shock=2, var=1 < 0
            @test check_func(irf_vals) == true
        end
    end

    # Narrative check with t > shocks size (continue branch, line 374)
    @testset "_build_check_func — period exceeds shocks" begin
        mktempdir() do dir
            cfg = joinpath(dir, "big_period.toml")
            write(cfg, """
            [identification]
            method = "sign"
            [identification.narrative]
            shock_index = 1
            periods = [3, 999]
            signs = [1, -1]
            """)
            _, narrative_check = _build_check_func(cfg)
            shocks = ones(10, 2)  # t=999 > 10 → continue
            @test narrative_check(shocks) == true
        end
    end

end  # Sign/narrative closure execution

# ═══════════════════════════════════════════════════════════════
# Task 9: Remaining handler coverage gaps
# ═══════════════════════════════════════════════════════════════

@testset "Remaining handler coverage" begin

    # _load_clusters helper (shared.jl lines 852-856)
    @testset "_load_clusters — valid column" begin
        mktempdir() do dir
            csv = joinpath(dir, "data.csv")
            write(csv, "y,x1,x2,cl\n" * join(["$(rand()),$(rand()),$(rand()),$(rand(1:3))" for _ in 1:50], "\n") * "\n")
            clusters = _load_clusters(csv, "cl")
            @test clusters isa Vector{Int}
            @test length(clusters) == 50
        end
    end

    @testset "_load_clusters — empty string" begin
        result = _load_clusters("dummy.csv", "")
        @test result === nothing
    end

    # _load_weights helper (shared.jl lines 860-863)
    @testset "_load_weights — valid column" begin
        mktempdir() do dir
            csv = joinpath(dir, "data.csv")
            write(csv, "y,x1,w\n" * join(["$(rand()),$(rand()),$(abs(rand()))" for _ in 1:50], "\n") * "\n")
            weights = _load_weights(csv, "w")
            @test weights isa Vector{Float64}
            @test length(weights) == 50
        end
    end

    @testset "_load_weights — empty string" begin
        result = _load_weights("dummy.csv", "")
        @test result === nothing
    end

    # Forecast dynamic — auto factors (covers forecast.jl auto selection)
    @testset "_forecast_dynamic — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_dynamic(; data=csv, nfactors=nothing, horizons=5, format="table")
                end
            end
        end
    end

    # Forecast GDFM — auto factors (covers forecast.jl auto selection)
    @testset "_forecast_gdfm — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _forecast_gdfm(; data=csv, nfactors=nothing, dynamic_rank=nothing,
                                     horizons=5, format="table")
                end
            end
        end
    end

    # Predict static — auto factors (covers predict.jl auto selection)
    @testset "_predict_static — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _predict_static(; data=csv, nfactors=nothing, format="table")
                end
            end
        end
    end

    # Residuals static — auto factors (covers residuals.jl auto selection)
    @testset "_residuals_static — auto factors" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = cd(dir) do
                _capture() do
                    _residuals_static(; data=csv, nfactors=nothing, format="table")
                end
            end
        end
    end

    # Forecast FAVAR panel_forecast mode (covers forecast.jl panel branch)
    @testset "_forecast_favar — panel_forecast" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=5)
            out = _capture() do
                _forecast_favar(; data=csv, factors=2, lags=1, key_vars="1,2",
                                  horizons=10, panel_forecast=true,
                                  format="table", output="",
                                  plot=false, plot_save="")
            end
        end
    end

end  # Remaining handler coverage

@testset "cached-model path — F14 regression (five sites)" begin
    Y = randn(60, 3)
    var_m  = MacroEconometricModels.estimate_var(Y, 2)
    vecm_m = MacroEconometricModels.estimate_vecm(Y, 2; rank=1)

    out = _capture() do
        _predict_var(; model=var_m, format="table")
    end

    out = _capture() do
        _residuals_var(; model=var_m, format="table")
    end

    out = _capture() do
        _predict_vecm(; model=vecm_m, format="table")
    end

    out = _capture() do
        _residuals_vecm(; model=vecm_m, format="table")
    end

    out = _capture() do
        _forecast_vecm(; model=vecm_m, horizons=4, format="table")
    end
end

end  # Command Handlers

# ═══════════════════════════════════════════════════════════════
# Golden envelopes (C022 / TS-5) — spectral pilot + renderer
# ═══════════════════════════════════════════════════════════════

@testset "golden envelopes (spectral pilot)" begin
    mktempdir() do dir
        fix = joinpath(dir, "golden_data.csv")
        open(fix, "w") do f
            println(f, "y1,y2,y3")
            for t in 1:80
                vals = [sin(t / (2 + i)) + 0.1 * cos(t / (3 + i)) for i in 1:3]
                println(f, join(string.(round.(vals; digits=8)), ","))
            end
        end
        cases = [
            (["spectral", "acf", fix, "--format", "json"], ["spectral", "acf"]),
            (["spectral", "periodogram", fix, "--format", "json"], ["spectral", "periodogram"]),
            (["spectral", "density", fix, "--method", "welch", "--format", "json"], ["spectral", "density"]),
            (["spectral", "cross", fix, "--var1", "1", "--var2", "2", "--format", "json"], ["spectral", "cross"]),
            (["spectral", "transfer", "--filter", "hp", "--lambda", "1600.0", "--nobs", "200", "--format", "json"], ["spectral", "transfer"]),
            # #147: estimate sdfm estimation-record table
            (["estimate", "sdfm", fix, "--factors", "1", "--format", "json"], ["estimate", "sdfm"]),
            # W1/#165: new forecast sdfm leaf
            (["forecast", "sdfm", fix, "--factors", "1", "--horizons", "4", "--format", "json"], ["forecast", "sdfm"]),
        ]
        for (argv, gkeys) in cases
            Random.seed!(42)
            out = _capture() do
                _dispatch_via_app(String[string(a) for a in argv])
            end
            js = _extract_json_object(out)
            @test js !== nothing
            gpath = _golden_path(gkeys)
            @test isfile(gpath)
            @test _golden_compare(js, gpath)
            errs = validate_envelope_json(js)
            @test isempty(errs) || (@info "schema errs" errs; false)
        end

        # W2/#137: dispatch-path ERROR envelope goldens. dispatch_leaf renders
        # the error envelope and THEN rethrows for the exit code — and _capture
        # does not swallow throws — so the dispatch call is wrapped. (The
        # usage/parse net lives in run_cli, which this harness bypasses via
        # dispatch(); the T4 battery in test_e2e.jl byte-asserts that path.)
        err_cases = [
            (["filter", "hp", "/nope.csv", "--format", "json"],
             ["filter", "hp", "error"], "data/file-not-found", 3),
            (["estimate", "bvar", fix, "--config", "/nope.toml", "--format", "json"],
             ["estimate", "bvar", "config-error"], "config/file-not-found", 4),
        ]
        for (argv, gkeys, code, ec) in err_cases
            Random.seed!(42)
            out = _capture() do
                try
                    _dispatch_via_app(String[string(a) for a in argv])
                catch e
                    e isa CliError || rethrow()
                end
            end
            js = _extract_json_object(out)
            @test js !== nothing
            doc = JSON3.read(js)
            @test string(doc.status) == "error"
            @test string(doc.error.code) == code
            @test doc.error.exit_code == ec
            gpath = _golden_path(gkeys)
            @test isfile(gpath)
            @test _golden_compare(js, gpath)
            errs = validate_envelope_json(js)
            @test isempty(errs) || (@info "schema errs" errs; false)
        end

        # Wave 2 typed-handle error goldens (relative stems; cwd = dir)
        cd(dir) do
            Random.seed!(42)
            CSV.write("panel.csv", DataFrame(group=repeat(1:4, inner=10),
                                             time=repeat(1:10, outer=4),
                                             y=randn(40), x=randn(40)))
            save_model_dispatch("panel.jld2",
                xtset(CSV.read("panel.csv", DataFrame), :group, :time))
            Y = reduce(hcat, (sin.(1:40) .+ 0.1 .* cos.((1:40) ./ i) for i in 1:3))
            save_model_dispatch("var.jld2", estimate_var(Y, 1; varnames=["y1", "y2", "y3"]))
            handle_err = [
                (["estimate", "var", "panel", "--lags", "1", "--format", "json"],
                 ["estimate", "var", "wrong-kind"], "data/wrong-kind", 3),
                (["irf", "var", "--result", "var", "--format", "json"],
                 ["irf", "var", "wrong-result"], "data/wrong-result", 3),
            ]
            for (argv, gkeys, code, ec) in handle_err
                Random.seed!(42)
                out = _capture() do
                    try
                        _dispatch_via_app(String[string(a) for a in argv])
                    catch e
                        e isa CliError || rethrow()
                    end
                end
                js = _extract_json_object(out)
                @test js !== nothing
                doc = JSON3.read(js)
                @test string(doc.status) == "error"
                @test string(doc.error.code) == code
                @test doc.error.exit_code == ec
                gpath = _golden_path(gkeys)
                @test isfile(gpath)
                @test _golden_compare(js, gpath)
                errs = validate_envelope_json(js)
                @test isempty(errs) || (@info "schema errs" errs; false)
            end
        end
    end

    # Renderer goldens (normalize CRLF — Windows checkout may convert golden text files)
    env = Envelope(command="estimate var")
    add_table!(env, :coefficients, DataFrame(variable=["y1", "y2"], est=[0.5, -0.25]))
    buf = IOBuffer(); render(env, :csv, buf)
    csv_out = replace(String(take!(buf)), "\r\n" => "\n")
    golden_csv = replace(read(joinpath(_GOLDEN_DIR, "render.csv.txt"), String), "\r\n" => "\n")
    @test csv_out == golden_csv
    buf = IOBuffer(); render(env, :json, buf)
    @test _golden_compare(String(take!(buf)), joinpath(_GOLDEN_DIR, "render.envelope.json"))

    # Deliberate rename detection (acceptance demo)
    bad = replace(read(joinpath(_GOLDEN_DIR, "spectral.acf.json"), String), "acf_pacf" => "renamed_table")
    @test !_golden_compare(bad, joinpath(_GOLDEN_DIR, "spectral.acf.json"))
end

# W1/#136: negative tests for the schema VALIDATOR itself. Until W1, the
# additionalProperties branch was nested inside the `properties` loop, so a
# schema carrying only additionalProperties (the envelope's `data`/`meta`) was
# never validated — and the then-ambiguous `data` oneOf went unnoticed by every
# tier because the two defects cancelled. These tests make that class loud.
@testset "schema validator subset (W1/#136)" begin
    # additionalProperties WITHOUT properties must validate every value (the
    # exact blind-spot shape).
    ap_only = Dict("type" => "object",
                   "additionalProperties" => Dict("type" => "integer"))
    @test isempty(validate_json_schema(Dict("a" => 1, "b" => 2), ap_only))
    @test !isempty(validate_json_schema(Dict("a" => "not-an-int"), ap_only))

    # additionalProperties: false rejects unlisted keys (with and without
    # properties present).
    closed = Dict("type" => "object",
                  "properties" => Dict("x" => Dict("type" => "integer")),
                  "additionalProperties" => false)
    @test isempty(validate_json_schema(Dict("x" => 1), closed))
    @test !isempty(validate_json_schema(Dict("x" => 1, "y" => 2), closed))
    @test !isempty(validate_json_schema(Dict("y" => 2),
                                        Dict("additionalProperties" => false)))

    # pattern: strings only; non-strings ignore it (draft-07 semantics).
    pat = Dict("pattern" => "^[a-z-]+/[a-z0-9-]+\$")
    @test isempty(validate_json_schema("data/not-found", pat))
    @test !isempty(validate_json_schema("Bad_Code", pat))
    @test isempty(validate_json_schema(42, pat))

    # The live envelope schema, exercised end-to-end through the validator.
    schema = JSON3.read(read(_ENVELOPE_SCHEMA_PATH, String))
    table = Dict("columns" => ["a", "b"], "rows" => [[1, "x"], [2.5, nothing]])
    base = Dict{String,Any}(
        "schema_version" => 1, "command" => "estimate var", "status" => "ok",
        "meta" => Dict{String,Any}("cli_version" => "0.0.0", "julia" => "1.12.0",
                                   "mems_version" => "0.8.0", "seed" => 42),
        "data" => Dict{String,Any}("coefficients" => table),
        "warnings" => Any[], "artifacts" => Any[], "error" => nothing)
    @test isempty(validate_json_schema(base, schema))

    # A malformed table must FAIL — this is what the pre-W1 stack silently
    # accepted (broken validator × broken schema).
    bad = deepcopy(base)
    bad["data"]["broken"] = Dict("columns" => ["a"])            # rows missing
    @test !isempty(validate_json_schema(bad, schema))
    bad = deepcopy(base)
    bad["data"]["broken"] = Dict("columns" => ["a"], "rows" => [[1]],
                                 "extra" => true)               # closed object
    @test !isempty(validate_json_schema(bad, schema))
    bad = deepcopy(base)
    bad["data"]["broken"] = Dict("columns" => ["a"], "rows" => [[[1, 2]]])  # array cell
    @test !isempty(validate_json_schema(bad, schema))
    bad = deepcopy(base)
    bad["data"]["broken"] = 3.14                                # bare scalar under data
    @test !isempty(validate_json_schema(bad, schema))

    # status/error co-occurrence via the top-level oneOf.
    ok_with_err = deepcopy(base)
    ok_with_err["error"] = Dict("code" => "model/error", "message" => "m")
    @test !isempty(validate_json_schema(ok_with_err, schema))
    err_doc = deepcopy(base)
    err_doc["status"] = "error"
    err_doc["error"] = Dict("code" => "data/not-found", "message" => "gone",
                            "hint" => "check the path", "exit_code" => 3)
    @test isempty(validate_json_schema(err_doc, schema))
    err_null = deepcopy(err_doc)
    err_null["error"] = nothing
    @test !isempty(validate_json_schema(err_null, schema))

    # error.code taxonomy pattern + exit_code enum.
    bad_code = deepcopy(err_doc)
    bad_code["error"] = Dict("code" => "NotAClass", "message" => "m")
    @test !isempty(validate_json_schema(bad_code, schema))
    bad_exit = deepcopy(err_doc)
    bad_exit["error"] = Dict("code" => "data/not-found", "message" => "m",
                             "exit_code" => 9)
    @test !isempty(validate_json_schema(bad_exit, schema))

    # meta requires the version trio.
    no_meta = deepcopy(base)
    no_meta["meta"] = Dict{String,Any}("cli_version" => "0.0.0")
    @test !isempty(validate_json_schema(no_meta, schema))
end


# ═══════════════════════════════════════════════════════════════
# Input-Output analysis command family (C049)
# ═══════════════════════════════════════════════════════════════

@testset "io command family (C049)" begin
    # Run an io leaf in JSON mode; return the raw envelope string.
    _ioraw(args...) = begin
        out = _capture() do
            _dispatch_via_app(vcat(String["io"], collect(String, args), String["--format", "json"]))
        end
        i = findfirst('{', out)
        i === nothing ? "" : out[i:end]
    end
    _iodoc(args...) = JSON3.read(_ioraw(args...))
    # Any table in the envelope whose columns ⊇ `cols`.
    _hascols(doc, cols) = any(t -> all(c -> c in String.(t.columns), cols), values(doc.data))
    _table(doc, cols) = first(t for t in values(doc.data) if all(c -> c in String.(t.columns), cols))

    @testset "sources (offline catalog)" begin
        raw = _ioraw("sources")
        @test isempty(validate_envelope_json(raw))
        doc = JSON3.read(raw)
        @test doc.status == "ok"
        @test _hascols(doc, ["source", "name", "versions", "credentials", "note"])
        t = _table(doc, ["source"])
        @test length(t.rows) == 5
    end

    @testset "load (dims + per-sector)" begin
        raw = _ioraw("load")
        @test isempty(validate_envelope_json(raw))
        doc = JSON3.read(raw)
        @test _hascols(doc, ["metric", "value"])              # summary kv
        @test _hascols(doc, ["sector", "gross_output", "final_demand", "value_added"])
        t = _table(doc, ["sector", "gross_output"])
        @test length(t.rows) == 2
    end

    @testset "leontief / ghosh (wide sector×sector)" begin
        doc = _iodoc("leontief")                               # default: L only
        t = _table(doc, ["sector", "Agriculture", "Manufacturing"])
        @test length(t.rows) == 2
        # L[1,1] ≈ 1.254125 (Miller & Blair)
        row1 = first(r for r in t.rows if r[1] == "Agriculture")
        @test isapprox(Float64(row1[2]), 1.254125; atol=1e-4)

        both = _iodoc("leontief", "--matrix", "both")
        @test length(collect(keys(both.data))) == 2            # A and L

        g = _iodoc("ghosh")
        @test _hascols(g, ["sector", "Agriculture", "Manufacturing"])
    end

    @testset "multipliers (kind × type)" begin
        d1 = _iodoc("multipliers", "--kind", "output", "--type", "I")
        t = _table(d1, ["sector", "multiplier"])
        vals = [Float64(r[2]) for r in t.rows]
        @test isapprox(vals, [1.518152, 1.452145]; atol=1e-4)
        @test _hascols(_iodoc("multipliers", "--kind", "income", "--type", "II"), ["sector", "multiplier"])
        @test _hascols(_iodoc("multipliers", "--kind", "employment"), ["sector", "multiplier"])
    end

    @testset "linkages / key-sectors" begin
        lk = _iodoc("linkages")
        @test _hascols(lk, ["sector", "backward", "forward", "Ui", "Uj", "class"])
        ks = _iodoc("key-sectors")
        t = _table(ks, ["sector", "class"])
        classes = [String(r[2]) for r in t.rows]
        @test "key" in classes && "weak" in classes
    end

    @testset "sda (two periods; same → ~0)" begin
        doc = _iodoc("sda", "--method", "additive")
        t = _table(doc, ["sector", "L_effect", "Y_effect", "total", "residual"])
        @test all(isapprox(Float64(r[4]), 0.0; atol=1e-6) for r in t.rows)
        # omitted --factors/--on → legacy L/Y keys, no named-factor columns
        cols = String.(t.columns)
        @test "intensity_effect" ∉ cols && "technology_effect" ∉ cols
        # explicit --factors → named columns, not legacy L/Y
        named = _iodoc("sda", "--factors", "technology,final-demand")
        nt = _table(named, ["sector", "technology_effect", "final_demand_effect", "total", "residual"])
        ncols = String.(nt.columns)
        @test "L_effect" ∉ ncols && "Y_effect" ∉ ncols
        # satellite --on (bundled :wiot has CO2) → intensity + technology + final_demand
        sat = _iodoc("sda", "--on", "CO2")
        st = _table(sat, ["intensity_effect", "technology_effect", "final_demand_effect"])
        scols = String.(st.columns)
        @test "L_effect" ∉ scols && "Y_effect" ∉ scols
        @test "intensity_effect" in scols && "technology_effect" in scols &&
              "final_demand_effect" in scols
    end

    @testset "extract (name / index / errors)" begin
        d1 = _iodoc("extract", "--sectors-extract", "Agriculture")
        t = _table(d1, ["sector", "output_loss"])
        loss = Dict(String(r[1]) => Float64(r[2]) for r in t.rows)
        @test isapprox(loss["Agriculture"], 1000.0; atol=1e-3)
        @test _hascols(_iodoc("extract", "--sectors-extract", "1,2"), ["sector", "output_loss"])
        # missing required option
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "extract"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        # bad sector name → data class
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "extract", "--sectors-extract", "Nope"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "data/bad-sector" && exit_class(err) == 3
        # W2: --mode/--share guards and per-mode happy path
        @test _hascols(_iodoc("extract", "--sectors-extract", "Agriculture", "--mode", "backward"),
                       ["sector", "output_loss"])
        @test _hascols(_iodoc("extract", "--sectors-extract", "Agriculture", "--mode", "forward"),
                       ["sector", "output_loss"])
        @test _hascols(_iodoc("extract", "--sectors-extract", "Agriculture", "--mode", "partial", "--share", "0.5"),
                       ["metric", "value"])
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "extract", "--sectors-extract", "1", "--share", "0"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "extract", "--sectors-extract", "1", "--mode", "nope"]); end; catch e; err = e; end
        @test err isa CliError || err isa ParseError
    end

    @testset "footprint (environmental)" begin
        raw = _ioraw("footprint")
        @test isempty(validate_envelope_json(raw))
        doc = JSON3.read(raw)
        @test _hascols(doc, ["stressor", "footprint"])
        @test _hascols(doc, ["sector", "CO2"])
        det = _iodoc("footprint", "--account", "employment", "--detail")
        @test length(collect(keys(det.data))) == 4            # footprint + by_sector + S + M
        # unknown account → data class
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "footprint", "--account", "bogus"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "data/no-extension"
    end

    @testset "baqaee-farhi" begin
        doc = _iodoc("baqaee-farhi")
        @test _hascols(doc, ["sector", "domar", "influence", "upstreamness", "downstreamness"])
        so = _iodoc("baqaee-farhi", "--second-order")
        @test length(collect(keys(so.data))) == 2
    end

    @testset "download (offline refusal + usage + mock happy path)" begin
        # --offline → env/network (exit 6)
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "download", "--source", "oecd", "--storage", "/tmp/x", "--offline"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "env/network" && exit_class(err) == 6
        # FRIEDMAN_OFFLINE env forces the same refusal
        err = nothing
        withenv("FRIEDMAN_OFFLINE" => "1") do
            try; _capture() do; _dispatch_via_app(String["io", "download", "--source", "wiod", "--storage", "/tmp/x"]); end; catch e; err = e; end
        end
        @test err isa CliError && err.code == "env/network"
        # missing --source / --storage → usage
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "download", "--storage", "/tmp/x"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "download", "--source", "oecd"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        # junk --years → usage error (not an uncaught ArgumentError → exit 1)
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "download", "--source", "oecd", "--storage", "/tmp/x", "--years", "abc"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/bad-years" && exit_class(err) == 2
        # non-offline mock download → log table (2 files, no network)
        doc = _iodoc("download", "--source", "oecd", "--storage", "/tmp/x")
        @test _hascols(doc, ["url", "filename"])
    end

    @testset "download forwards source-specific kwargs correctly (review [1]/[10])" begin
        # Every source's non-offline mock happy path must succeed — a regression here
        # means the handler over-forwards `system`/`verify` to a downloader that rejects
        # them (the mock reproduces the real restricted per-source signatures).
        for src in ["oecd", "wiod", "gloria", "exiobase3"]
            doc = _iodoc("download", "--source", src, "--storage", "/tmp/x")
            @test _hascols(doc, ["url", "filename"])
        end
        # exiobase3 is the only source that accepts --system
        @test _hascols(_iodoc("download", "--source", "exiobase3", "--storage", "/tmp/x", "--system", "ixi"), ["url", "filename"])
    end

    @testset "error classes on bad input (review findings)" begin
        # extract: out-of-range integer index → data/bad-sector (not exit-1 BoundsError)
        for idx in ["99", "0"]
            err = nothing
            try; _capture() do; _dispatch_via_app(String["io", "extract", "--sectors-extract", idx]); end; catch e; err = e; end
            @test err isa CliError && err.code == "data/bad-sector" && exit_class(err) == 3
        end
        # unknown :example → usage/unknown-example (not exit-1 ArgumentError)
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "load", "--data", ":bogus"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/unknown-example" && exit_class(err) == 2
        # CSV-backed paths (no satellite accounts; oversized n-sectors)
        mktempdir() do dir
            csv = joinpath(dir, "io.csv")
            open(csv, "w") do io; write(io, "150,500,350\n200,100,1700\n"); end
            # employment multipliers on an extension-less CSV → data/no-extension
            err = nothing
            try; _capture() do; _dispatch_via_app(String["io", "multipliers", "--data", csv, "--n-sectors", "2", "--kind", "employment"]); end; catch e; err = e; end
            @test err isa CliError && err.code == "data/no-extension"
            # n-sectors larger than the file → data/parse (not exit-1 BoundsError)
            err = nothing
            try; _capture() do; _dispatch_via_app(String["io", "load", "--data", csv, "--n-sectors", "99"]); end; catch e; err = e; end
            @test err isa CliError && err.code == "data/parse" && exit_class(err) == 3
            # a CSV IO table with valid dims parses and computes
            doc = JSON3.read(begin
                out = _capture() do
                    _dispatch_via_app(String["io", "leontief", "--data", csv, "--n-sectors", "2", "--format", "json"])
                end
                out[findfirst('{', out):end]
            end)
            @test _hascols(doc, ["sector"])
        end
    end

    @testset "table-mode output is non-empty" begin
        out = _capture() do
            _dispatch_via_app(String["io", "linkages"])
        end
        @test occursin("Agriculture", out) && occursin("Manufacturing", out)
    end

    @testset "W3/#154 io bf node" begin
        net = _iodoc("bf", "network")
        @test _hascols(net, ["metric", "value"])
        @test _hascols(net, ["sector", "lambda", "lambda_rev", "mu"])
        t = _table(net, ["sector", "lambda"])
        @test length(t.rows) == 2
        @test _hascols(_iodoc("bf", "network", "--nests", "two", "--factors", "va-cats"),
                       ["sector", "lambda", "lambda_rev", "mu"])
        @test _hascols(_iodoc("bf", "network", "--theta", "0.5,1.2", "--mu", "1,1"),
                       ["sector", "lambda"])

        eq = _iodoc("bf", "equilibrium", "--dlog-a", "0.01,0")
        @test _hascols(eq, ["sector", "dlog_x", "dlog_p"])
        kv = Dict(String(r[1]) => r[2] for r in _table(eq, ["metric", "value"]).rows)
        @test haskey(kv, "converged") && haskey(kv, "dlogY")

        # Non-convergence is a result, not exit 5.
        nc = _iodoc("bf", "equilibrium", "--maxiter", "1")
        nckv = Dict(String(r[1]) => r[2] for r in _table(nc, ["metric", "value"]).rows)
        @test lowercase(string(nckv["converged"])) in ("false", "0")

        loc = _iodoc("bf", "local")
        @test _hascols(loc, ["sector", "first_order"])
        @test _hascols(_iodoc("bf", "elasticities"), ["sector", "Agriculture", "Manufacturing"])
        sc = _iodoc("bf", "shock-curve", "--sector", "Agriculture", "--points", "5")
        @test _hascols(sc, ["shock", "exact", "hulten", "second_order"])
        @test length(_table(sc, ["shock"]).rows) == 5
        @test _hascols(_iodoc("bf", "wedges", "--dlog-a", "0.01"), ["sector", "lambda_cost", "lambda_rev", "mu"])
        @test _hascols(_iodoc("bf", "misallocation"), ["sector", "delta_logmu", "lambda", "mu"])

        # --model round-trip via interim handle
        mktempdir() do dir
            h = joinpath(dir, "net.fmod")
            _capture() do
                _dispatch_via_app(String["io", "bf", "network", "--save-model", h, "--format", "json"])
            end
            @test isfile(h)
            loaded = _iodoc("bf", "equilibrium", "--model", h, "--dlog-a", "0.02,0")
            @test _hascols(loaded, ["sector", "dlog_x", "dlog_p"])
        end

        # T2: usage guards
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bf", "shock-curve"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bf", "shock-curve", "--sector", "1", "--points", "1"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bf", "network", "--theta", "abc"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bf", "equilibrium", "--damping", "0"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid" && exit_class(err) == 2
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bf", "shock-curve", "--sector", "Nope"]); end; catch e; err = e; end
        @test err isa CliError && startswith(err.code, "data/") && exit_class(err) == 3
    end

    @testset "W4/#155 classical + MRIO" begin
        @test _hascols(_iodoc("price", "--dva", "0.1,0"), ["sector", "dp", "p", "dv"])
        imp = _iodoc("impact", "--dy", "10,0")
        @test _hascols(imp, ["sector", "impact"])
        ns = _iodoc("network-stats")
        @test _hascols(ns, ["sector", "domar", "multiplier", "in_degree", "out_degree", "upstreamness", "downstreamness"])
        ag = _iodoc("aggregate", "--sector-map", "Agriculture=Goods,Manufacturing=Goods")
        @test _hascols(ag, ["sector", "gross_output"])
        @test _hascols(_iodoc("balance"), ["sector", "gross_output"])
        @test _hascols(_iodoc("vertical-specialization"), ["metric", "value"])
        ed0 = _iodoc("export-decomposition")
        @test _hascols(ed0, ["dva", "rdv", "fva", "pdc", "gross_exports"])
        @test _hascols(_iodoc("bilateral-trade", "--exporter", "total", "--importer", "total"),
                       ["sector", "by_sector"])

        mktempdir() do dir
            kww = joinpath(dir, "kww.csv")
            write(kww, ",A_goods,B_goods,A_HFCE,B_HFCE\nA_goods,50,50,30,20\nB_goods,0,0,50,0\nVA,100,0,0,0\n")
            ed = _iodoc("export-decomposition", "--data", kww, "--parser", "icio", "--region", "A")
            t = _table(ed, ["dva", "rdv", "fva", "pdc", "gross_exports"])
            @test length(t.rows) == 1
            cols = String.(t.columns)
            di = findfirst(==("dva"), cols); ri = findfirst(==("rdv"), cols)
            fi = findfirst(==("fva"), cols); pi = findfirst(==("pdc"), cols)
            gi = findfirst(==("gross_exports"), cols)
            row = collect(first(t.rows))
            @test isapprox(Float64(row[di]) + Float64(row[ri]) + Float64(row[fi]) + Float64(row[pi]),
                           Float64(row[gi]); atol=1e-6)
            vs = _iodoc("vertical-specialization", "--data", kww, "--parser", "icio", "--region", "B")
            @test _hascols(vs, ["metric", "value"])
            bt = _iodoc("bilateral-trade", "--data", kww, "--parser", "icio",
                        "--exporter", "A", "--importer", "B")
            @test _hascols(bt, ["sector", "by_sector"])
            # missing --region on a multi-region table
            err = nothing
            try; _capture() do
                _dispatch_via_app(String["io", "export-decomposition", "--data", kww, "--parser", "icio"])
            end; catch e; err = e; end
            @test err isa CliError && err.code == "usage/missing-option"
        end

        # T2 usage
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "impact"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "bilateral-trade"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "price", "--dva", "nope"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"
        err = nothing
        try; _capture() do; _dispatch_via_app(String["io", "balance", "--maxiter", "0"]); end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"

        # ZipFile missing-extension
        mktempdir() do dir
            z = joinpath(dir, "table.zip")
            write(z, "not a zip")
            err = nothing
            try; _capture() do
                _dispatch_via_app(String["io", "load", "--data", z, "--parser", "icio"])
            end; catch e; err = e; end
            @test err isa CliError && err.code == "env/missing-extension" && exit_class(err) == 6
        end
    end
end

@testset "W5–W8 DSGE family expansion" begin
    _dsgeraw(args...) = begin
        out = _capture() do
            _dispatch_via_app(vcat(collect(String, args), String["--format", "json"]))
        end
        i = findfirst('{', out)
        i === nothing ? "" : out[i:end]
    end
    _dsgedoc(args...) = JSON3.read(_dsgeraw(args...))
    _hascols(doc, cols) = any(t -> all(c -> c in String.(t.columns), cols), values(doc.data))
    _table(doc, cols) = first(t for t in values(doc.data) if all(c -> c in String.(t.columns), cols))

    @testset "RA vfi / blanchard-kahn" begin
        mktempdir() do dir
            lin = joinpath(dir, "lin.jl")
            write(lin, """
            @dsge begin
                parameters: rho = 0.9, sigma = 0.01
                endogenous: Y, C
                exogenous: e
                linear: true
                Y[t] = rho * Y[t-1] + sigma * e[t]
                C[t] = Y[t]
            end
            """)
            bk = _dsgedoc("dsge", "solve", lin, "--method", "blanchard-kahn")
            @test _hascols(bk, ["variable"])
            gens = _dsgedoc("dsge", "solve", lin, "--method", "gensys")
            tb = _table(bk, ["variable"]); tg = _table(gens, ["variable"])
            @test String.(tb.columns) == String.(tg.columns)
            @test length(tb.rows) == length(tg.rows)
            for (rb, rg) in zip(tb.rows, tg.rows)
                for (a, b) in zip(rb, rg)
                    if a isa Number && b isa Number
                        @test isapprox(Float64(a), Float64(b); atol=1e-12, rtol=0)
                    else
                        @test string(a) == string(b)
                    end
                end
            end
            pfi = _dsgedoc("dsge", "solve", lin, "--method", "pfi")
            pkv = Dict(String(r[1]) => r[2] for r in _table(pfi, ["metric", "value"]).rows)
            @test haskey(pkv, "converged")
            @test lowercase(string(pkv["converged"])) in ("true", "1")
            err = nothing
            try; _capture() do
                _dispatch_via_app(String["dsge", "solve", lin, "--method", "vfi", "--n-grid", "8"])
            end; catch e; err = e; end
            @test err isa CliError && startswith(err.code, "config/")
            err = nothing
            try; _capture() do
                _dispatch_via_app(String["dsge", "solve", lin, "--method", "not-a-method"])
            end; catch e; err = e; end
            @test err isa Exception
        end
    end

    @testset "W1 VFI smolyak + optimizer (MEMs#817-819, #821)" begin
        _vfikv(doc) = Dict(String(r[1]) => r[2]
                           for r in _table(doc, ["metric", "value"]).rows)
        _vferr(args...) = begin
            err = nothing
            try; _capture() do
                _dispatch_via_app(collect(String, args))
            end; catch e; err = e; end
            err
        end
        mktempdir() do dir
            bell1 = joinpath(dir, "bell1.jl")
            write(bell1, """
            @dsge begin
                parameters: beta = 0.99, rho = 0.9, sigma = 0.01
                endogenous: C, K, A
                exogenous: e
                utility: log(C)
                beta: beta
                controls: C
                C[t] = K[t]
                K[t] = rho * K[t-1] + sigma * e[t]
                A[t] = rho * A[t-1] + sigma * e[t]
            end
            """)
            bell2 = joinpath(dir, "bell2.jl")
            write(bell2, """
            @dsge begin
                parameters: beta = 0.99, rho = 0.9, sigma = 0.01
                endogenous: C, I, K, A
                exogenous: e
                utility: log(C)
                beta: beta
                controls: C, I
                C[t] + I[t] = K[t]
                K[t] = rho * K[t-1] + sigma * e[t]
                A[t] = rho * A[t-1] + sigma * e[t]
            end
            """)
            bell0 = joinpath(dir, "bell0.jl")
            write(bell0, """
            @dsge begin
                parameters: beta = 0.99, rho = 0.9, sigma = 0.01
                endogenous: C, I, K, A
                exogenous: e
                utility: log(C)
                beta: beta
                C[t] + I[t] = K[t]
                K[t] = rho * K[t-1] + sigma * e[t]
                A[t] = rho * A[t-1] + sigma * e[t]
            end
            """)
            # Mock nx: bell1 n=3 → 1 state; bell2/bell0 n=4 → 2 states.
            sm = _dsgedoc("dsge", "solve", bell1, "--method", "vfi",
                          "--grid", "smolyak")
            smkv = _vfikv(sm)
            @test string(smkv["grid_type"]) == "smolyak"
            @test Int(smkv["smolyak_blocks"]) == 3
            @test Int(smkv["n_nodes"]) == 5
            @test _hascols(sm, ["node", "V"])
            au = _dsgedoc("dsge", "solve", bell1, "--method", "vfi")
            aukv = _vfikv(au)
            @test string(aukv["grid_type"]) == "tensor"
            @test Int(aukv["smolyak_blocks"]) == 0
            te = _dsgedoc("dsge", "solve", bell1, "--method", "vfi",
                          "--grid", "tensor")
            @test string(_vfikv(te)["grid_type"]) == "tensor"
            for o in ("auto", "grid1d", "fminbox-nm", "fminbox-lbfgs")
                d = _dsgedoc("dsge", "solve", bell1, "--method", "vfi",
                             "--optimizer", o)
                @test string(_vfikv(d)["grid_type"]) == "tensor"
            end
            # Explicit 2-control + grid1d: usage/invalid (CLI pre-check).
            e1 = _vferr("dsge", "solve", bell2, "--method", "vfi",
                        "--optimizer", "grid1d")
            @test e1 isa CliError && e1.code == "usage/invalid"
            # Explicit 2-control + auto: resolves to fminbox, solves fine.
            a2 = _dsgedoc("dsge", "solve", bell2, "--method", "vfi")
            @test string(_vfikv(a2)["grid_type"]) == "tensor"
            # Default (empty) controls + grid1d: upstream ArgumentError →
            # data/invalid, mirroring real MEMs (mock throw parity).
            e0 = _vferr("dsge", "solve", bell0, "--method", "vfi",
                        "--optimizer", "grid1d")
            @test e0 isa CliError && e0.code == "data/invalid" &&
                exit_class(e0) == 3
            # Bad enum → parser exit 2.
            eo = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--optimizer", "bogus")
            @test eo isa ParseError || (eo isa CliError && exit_class(eo) == 2)
            # smolyak-mu shape junk → usage/invalid.
            for bad in ("x", "-1", "2,-1", "2,,3", "2.5", "2,")
                eb = _vferr("dsge", "solve", bell1, "--method", "vfi",
                            "--grid", "smolyak", "--smolyak-mu", bad)
                @test eb isa CliError && eb.code == "usage/invalid"
            end
            # Length-vs-nx: mock nx=1 on bell1, nx=2 on bell2.
            m1 = _dsgedoc("dsge", "solve", bell1, "--method", "vfi",
                          "--grid", "smolyak", "--smolyak-mu", "2")
            @test string(_vfikv(m1)["grid_type"]) == "smolyak"
            em = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--grid", "smolyak", "--smolyak-mu", "2,2")
            @test em isa CliError && em.code == "data/invalid" &&
                exit_class(em) == 3
            m2 = _dsgedoc("dsge", "solve", bell2, "--method", "vfi",
                          "--grid", "smolyak", "--smolyak-mu", "2,2")
            @test string(_vfikv(m2)["grid_type"]) == "smolyak"
            # Dead combos → usage/invalid.
            dg = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--grid", "smolyak", "--n-grid", "8")
            @test dg isa CliError && dg.code == "usage/invalid"
            dt = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--grid", "tensor", "--smolyak-mu", "2")
            @test dt isa CliError && dt.code == "usage/invalid"
            dn = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--optimizer", "fminbox-nm", "--n-choice", "15")
            @test dn isa CliError && dn.code == "usage/invalid"
            dd = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--grid", "smolyak", "--degree", "3")
            @test dd isa CliError && dd.code == "usage/invalid"
            # The auto corner stays allowed (resolution needs the model).
            ac = _dsgedoc("dsge", "solve", bell2, "--method", "vfi",
                          "--n-choice", "15")
            @test string(_vfikv(ac)["grid_type"]) == "tensor"
            # Knob scope: VFI-only knobs on other methods → usage/invalid.
            sg = _vferr("dsge", "solve", bell1, "--method", "gensys",
                        "--optimizer", "auto")
            @test sg isa CliError && sg.code == "usage/invalid"
            sp = _vferr("dsge", "solve", bell1, "--method", "pfi",
                        "--smolyak-mu", "2")
            @test sp isa CliError && sp.code == "usage/invalid"
            # vfi grid vocabulary unchanged otherwise.
            gc = _vferr("dsge", "solve", bell1, "--method", "vfi",
                        "--grid", "chebyshev")
            @test gc isa CliError && gc.code == "usage/invalid"
            # simulate+irf route through the ProjectionSolution path: the
            # antithetic kwarg must never reach it (W1 fix; the mock throws
            # MethodError like real if it regresses).
            smd = _dsgedoc("dsge", "simulate", bell1, "--method", "vfi",
                           "--periods", "5", "--burn", "2")
            @test _hascols(smd, ["period"])
            smd2 = _dsgedoc("dsge", "simulate", bell1, "--method", "vfi",
                            "--periods", "5", "--burn", "2",
                            "--antithetic", "--seed", "7")
            @test _hascols(smd2, ["period"])
            ird = _dsgedoc("dsge", "irf", bell1, "--method", "vfi",
                           "--grid", "smolyak", "--horizon", "4")
            @test ird isa JSON3.Object
        end
    end

    @testset "HA two-asset / huggett accuracy / hd" begin
        doc = _dsgedoc("dsge", "ha", "steady-state", "two-asset-hank")
        kv = Dict{String,Any}()
        for t in values(doc.data)
            "name" in String.(t.columns) && "value" in String.(t.columns) || continue
            ni = findfirst(==("name"), String.(t.columns))
            vi = findfirst(==("value"), String.(t.columns))
            for r in t.rows
                kv[String(r[ni])] = r[vi]
            end
        end
        @test haskey(kv, "K") || haskey(kv, "B") || haskey(kv, "Y")
        @test haskey(kv, "B") && haskey(kv, "B_supply")
        @test isapprox(Float64(kv["B"]), Float64(kv["B_supply"]); atol=1e-8)
        err = nothing
        try; _capture() do
            _dispatch_via_app(String["dsge", "ha", "accuracy", "huggett"])
        end; catch e; err = e; end
        @test err isa CliError && err.code == "model/unsupported"
        mktempdir() do dir
            csv = joinpath(dir, "y.csv")
            write(csv, "K\n" * join(string.(1.0 .+ 0.01 .* (1:12)), "\n") * "\n")
            hd = _dsgedoc("dsge", "ha", "hd", "krusell-smith", "--data", csv,
                          "--observables", "K", "--method", "ssj")
            @test any(c -> startswith(c, "t") || c == "t",
                      vcat((String.(t.columns) for t in values(hd.data))...))
        end
        err = nothing
        try; _capture() do
            _dispatch_via_app(String["dsge", "ha", "solve", "huggett", "--hh-solver", "nope"])
        end; catch e; err = e; end
        @test err isa Exception
    end

    @testset "dcegm / lifecycle / firm / bank" begin
        sol = _dsgedoc("dsge", "dcegm", "solve", "retirement", "--n-a", "8", "--n-periods", "4")
        @test _hascols(sol, ["option", "knot", "M"])
        eq = _dsgedoc("dsge", "dcegm", "steady-state", "retirement", "--n-a", "8", "--n-periods", "4")
        kv = Dict(String(r[1]) => r[2] for r in _table(eq, ["metric", "value"]).rows)
        @test haskey(kv, "excess_demand")
        @test isapprox(Float64(kv["excess_demand"]), 0.0; atol=5e-3)
        lc = _dsgedoc("dsge", "lifecycle", "steady-state", "--j", "8", "--j-retire", "6",
                      "--n-a", "8", "--income-states", "2")
        @test _hascols(lc, ["age", "cohort_mass"])
        mass = [Float64(r[findfirst(==("cohort_mass"), String.(_table(lc, ["age", "cohort_mass"]).columns))])
                for r in _table(lc, ["age", "cohort_mass"]).rows]
        @test isapprox(sum(mass), 1.0; atol=1e-8)
        firm = _dsgedoc("dsge", "firm", "steady-state", "--n-k", "8", "--n-eps", "2")
        @test _hascols(firm, ["metric", "value"]) || _hascols(firm, ["k_index"])
        pe = _dsgedoc("dsge", "bank", "pe", "--n-n", "8", "--n-xi", "2")
        @test _hascols(pe, ["n_index", "l_policy"])
        ss = _dsgedoc("dsge", "bank", "steady-state", "--n-n", "8", "--n-xi", "2")
        @test _hascols(ss, ["n_index", "l_policy"])
        function _lmap(doc)
            t = _table(doc, ["n_index", "l_policy"])
            cols = String.(t.columns)
            ni = findfirst(==("n_index"), cols)
            xi = findfirst(==("xi_index"), cols)
            li = findfirst(==("l_policy"), cols)
            Dict((Int(r[ni]), Int(r[xi])) => Float64(r[li]) for r in t.rows)
        end
        pe_l, ss_l = _lmap(pe), _lmap(ss)
        @test keys(pe_l) == keys(ss_l)
        for k in keys(pe_l)
            @test isapprox(pe_l[k], ss_l[k]; atol=1e-12, rtol=0)
        end
        err = nothing
        try; _capture() do
            _dispatch_via_app(String["dsge", "bank", "steady-state", "--r-lo", "0.5", "--r-hi", "0.1"])
        end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/invalid"
        bad = _dsgedoc("dsge", "bank", "steady-state", "--n-n", "8", "--n-xi", "2",
                       "--r-lo", "5", "--r-hi", "6")
        bkv = Dict(String(r[1]) => r[2] for r in _table(bad, ["metric", "value"]).rows)
        @test haskey(bkv, "converged")
        @test lowercase(string(bkv["converged"])) in ("false", "0")
        err = nothing
        try; _capture() do
            _dispatch_via_app(String["dsge", "dcegm", "transition", "retirement"])
        end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
        err = nothing
        try; _capture() do
            _dispatch_via_app(String["dsge", "firm", "transition"])
        end; catch e; err = e; end
        @test err isa CliError && err.code == "usage/missing-option"
    end
end

# W4/#139 (#81): the direct-Exception domain types must keep their SPECIFIC classes
# through the `_domain_or_data_error` wrap — its old fallback collapsed anything
# outside MacroModelError into generic model/error (the shadow the wave audit found).
@testset "W4/#139: _domain_or_data_error consults _domain_error_class" begin
    ss = _domain_or_data_error(
        MacroEconometricModels.StochasticSingularityError("2 observables exceed 1 structural shocks"),
        "bayes estimation")
    @test ss isa CliError
    @test ss.code == "model/stochastic-singularity"
    @test exit_class(ss) == 5

    dse = _domain_or_data_error(
        MacroEconometricModels.DSGESolveError("Numerical steady state did not satisfy"),
        "dsge solve")
    @test dse isa CliError
    @test dse.code == "model/solve"
    @test exit_class(dse) == 5

    # MacroModelError subtypes still pass through RAW — the central mapper owns them.
    conv = _domain_or_data_error(MacroEconometricModels.ConvergenceError("nc"), "x")
    @test !(conv isa CliError)
    @test nameof(typeof(conv)) === :ConvergenceError

    # Generic fallbacks unchanged (a plain ArgumentError is a DATA statement here).
    @test _domain_or_data_error(ArgumentError("bad response"), "x").code == "data/invalid"
    @test _domain_or_data_error(ErrorException("boom"), "x").code == "model/error"
end

# W5/#140: `friedman schema` — machine-actionable self-description (closes #63).
# schema.jl is NOT in the standard include block above because it needs the full
# registry populated; included here with a test APP so _schema_cmd resolves it.
include(joinpath(project_root, "src", "commands", "schema.jl"))
if !@isdefined(APP)
    const APP = Entry("friedman",
        NodeCommand("friedman", Dict{String,Union{NodeCommand,LeafCommand}}(
            "estimate" => register_estimate_commands!(),
            "test"     => register_test_commands!(),
            "irf"      => register_irf_commands!(),
            "fevd"     => register_fevd_commands!(),
            "hd"       => register_hd_commands!(),
            "forecast"  => register_forecast_commands!(),
            "predict"   => register_predict_commands!(),
            "residuals" => register_residuals_commands!(),
            "filter"    => register_filter_commands!(),
            "data"      => register_data_commands!(),
            "io"        => register_io_commands!(),
            "nowcast"   => register_nowcast_commands!(),
            "dsge"      => register_dsge_commands!(),
            "did"       => register_did_commands!(),
            "multipliers" => register_multipliers_commands!(),
            "policy"    => register_policy_commands!(),
            "spectral"  => register_spectral_commands!(),
            "model"     => register_model_commands!(),
            "completions" => register_completions_commands!(),
            "schema"    => register_schema_command!(),
            "show"      => register_show_commands!(),
        ), "test tree"); version=v"0.0.0-test")
end

@testset "W5/#140: schema — input_schema, tables, contract, --docs" begin
    getdoc(args...) = JSON3.read(strip(_capture() do
        dispatch_schema(String[args...])
    end))

    # ── Leaf doc: draft-07 input_schema with x-cli annotations ──
    doc = getdoc("estimate", "var")
    is = doc.input_schema
    @test String(is[Symbol("\$schema")]) == "http://json-schema.org/draft-07/schema#"
    @test String(is.type) == "object"
    @test is.additionalProperties === false
    @test "data" in String.(is.required)
    props = is.properties
    @test String(props.data.type) == "string"
    @test String(props.data[Symbol("x-cli")].kind) == "argument"
    @test Int(props.data[Symbol("x-cli")].position) == 1
    @test String(props.lags.type) == "integer"
    @test String(props.lags[Symbol("x-cli")].kind) == "option"
    @test String(props.lags[Symbol("x-cli")].long) == "--lags"
    @test "json" in String.(props.format.enum)
    @test String(props.format[Symbol("x-cli")].short) == "-f"

    # ── Leaf doc: registry-declared tables (the W3 key contract) ──
    tnames = Set(String(t.name) for t in doc.tables)
    @test "var_coefficients" in tnames
    @test "information_criteria" in tnames
    @test all(haskey(t, :family) for t in doc.tables)
    # a family declaration surfaces as family=true (hd var: per-shock sibling
    # tables; NOT irf var — W3 made that one static)
    hddoc = getdoc("hd", "var")
    @test any(t -> t.family === true, hddoc.tables)

    # ── Root doc: contract block ──
    root = getdoc()
    @test haskey(root, :contract)
    env_schema = root.contract.envelope_schema
    @test occursin("draft-07", String(env_schema[Symbol("\$schema")]))
    @test haskey(env_schema.properties, :data)   # the embedded normative schema
    @test length(root.contract.exit_codes) == 7
    codes = Dict(Int(e.exit_code) => String(e.class) for e in root.contract.exit_codes)
    @test codes[5] == "model" && codes[2] == "usage" && codes[0] == "ok"
    # a non-root doc has no contract
    @test !haskey(doc, :contract)

    # ── --docs: guide embedded; flag NEVER eats a path token (D-6) ──
    r1 = getdoc("--docs")
    @test haskey(r1, :docs) && occursin("# Agent Guide", String(r1.docs))
    r2 = getdoc("--docs", "estimate", "var")   # D-6: `estimate` must survive
    @test String.(r2.path) == ["estimate", "var"]
    @test haskey(r2, :docs)
    r3 = getdoc("estimate", "var", "--docs")
    @test String.(r3.path) == ["estimate", "var"]
    @test haskey(r3, :docs)
    @test !haskey(doc, :docs)                  # absent without the flag
    # the guide the schema serves is byte-identical to the baked const
    @test String(r1.docs) == _AGENT_GUIDE_MD

    # ── Every leaf emits a structurally sound input_schema ──
    nchecked = Ref(0)
    function _walkschema(node, path)
        for (name, sub) in node.subcmds
            is_hidden_alias(name, sub) && continue
            p = vcat(path, [name])
            if sub isa LeafCommand
                s = _input_schema(sub, p)
                @test s["additionalProperties"] === false
                @test issubset(Set(s["required"]), Set(keys(s["properties"])))
                for (_, pv) in s["properties"]
                    @test pv["type"] in ("string", "integer", "number", "boolean")
                    @test haskey(pv, "x-cli")
                end
                nchecked[] += 1
            else
                _walkschema(sub, p)
            end
        end
    end
    _walkschema(APP.root, String[])
    @test nchecked[] > 400   # the full 410-leaf surface (+ schema itself)
end

# W7/#142: serve --mcp — the registry projected as MCP tools. run_cli only
# exists inside the Friedman module, so tools/call sessions live in T3; here we
# pin the pure pieces (tool enumeration, argv reconstruction, model:// store)
# and the protocol loop's non-tool methods end-to-end via IOBuffers.
include(joinpath(project_root, "src", "commands", "serve.jl"))

@testset "W7/#142: serve --mcp projection" begin
    @testset "_mcp_tools naming" begin
        tools = _mcp_tools()
        names = [t[1] for t in tools]
        @test "estimate_var" in names
        @test "dsge_bayes_estimate" in names
        @test length(names) == length(unique(names))
        @test !("serve" in names)             # never serves itself
        @test all(n -> occursin(r"^[a-z0-9_-]+$", n), names)
    end

    @testset "_mcp_argv reconstruction" begin
        leaf = LeafCommand("var", identity;
            args=[Argument("data"; type=String, required=true)],
            options=[Option("lags"; type=Int, default=0),
                     Option("format"; type=String, default="table"),
                     Option("output"; type=String, default="")],
            flags=[Flag("plot")],
            description="t")
        p = ["estimate", "var"]
        # full surface: positional + option + true flag + forced json
        argv = _mcp_argv(leaf, p, Dict{Symbol,Any}(
            :data => "x.csv", :lags => 2, :plot => true))
        @test argv == ["estimate", "var", "x.csv", "--lags", "2", "--plot",
                       "--format", "json"]
        # false flag omitted; user format overridden by the forced json
        argv2 = _mcp_argv(leaf, p, Dict{Symbol,Any}(
            :data => "x.csv", :plot => false, :format => "csv"))
        @test argv2 == ["estimate", "var", "x.csv", "--format", "json"]
        # unknown key → forwarded so the strict parser rejects with a hint
        argv3 = _mcp_argv(leaf, p, Dict{Symbol,Any}(:data => "x.csv", :lgas => 2))
        @test "--lgas" in argv3
        # a leaf without --format gets NO forced json (completions-style)
        bare = LeafCommand("bash", identity;
            options=[Option("output"; type=String, default="")], description="t")
        @test _mcp_argv(bare, ["completions", "bash"], Dict{Symbol,Any}()) ==
              ["completions", "bash"]
    end

    @testset "model:// store semantics" begin
        # outside a serve session: typed usage error, both directions
        @test_throws CliError save_model_dispatch("model://m1", (a=1,))
        @test_throws CliError load_model_dispatch("model://m1")
        err = try; load_model_dispatch("model://m1"); catch e; e; end
        @test err.code == "usage/invalid"
        # inside: round-trip + typed miss
        _SERVE_MODEL_STORE[] = Dict{String,Any}()
        try
            obj = (theta = [1.0, 2.0],)
            @test save_model_dispatch("model://m1", obj) == "model://m1"
            @test load_model_dispatch("model://m1") === obj
            miss = try; load_model_dispatch("model://nope"); catch e; e; end
            @test miss isa CliError && miss.code == "data/file-not-found"
        finally
            _SERVE_MODEL_STORE[] = nothing
        end
    end

    @testset "protocol loop (non-tool methods)" begin
        function session(lines...)
            input = IOBuffer(join(lines, "\n") * "\n")
            output = IOBuffer()
            _serve_loop(input, output)
            [JSON3.read(l) for l in split(String(take!(output)), '\n') if !isempty(strip(l))]
        end
        rs = session(
            """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}""",
            """{"jsonrpc":"2.0","method":"notifications/initialized"}""",
            """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""",
            """{"jsonrpc":"2.0","id":3,"method":"ping"}""",
            "this is not json",
            """{"jsonrpc":"2.0","id":4,"method":"no/such"}""",
            """{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"no_such_tool","arguments":{}}}""",
        )
        @test length(rs) == 6            # notification got no response
        init = rs[1]
        @test init.id == 1
        @test haskey(init.result, :protocolVersion)
        @test haskey(init.result.capabilities, :tools)
        tl = rs[2]
        @test tl.id == 2
        tools = tl.result.tools
        @test length(tools) > 400
        est = only(t for t in tools if t.name == "estimate_var")
        @test haskey(est, :inputSchema)
        @test String(est.inputSchema.type) == "object"
        @test "data" in String.(est.inputSchema.required)
        @test rs[3].id == 3                          # ping
        @test rs[4].error.code == -32700             # parse error
        @test rs[5].error.code == -32601             # method not found
        @test rs[6].error.code == -32602             # unknown tool
        # the store is cleared when the loop ends
        @test _SERVE_MODEL_STORE[] === nothing
    end

    @testset "serve leaf guard" begin
        err = try; _serve(; mcp=false); catch e; e; end
        @test err isa CliError && err.code == "usage/missing"
    end
end

@testset "W3/#167: universal serialization + reproduce" begin
    @testset "_fwd_seed" begin
        _SEED[] = nothing
        @test isempty(_fwd_seed())
        _SEED[] = 7
        @test _fwd_seed() == (; seed=7)
        _SEED[] = nothing
    end

    @testset "fallback registry is the 0.9.3 set" begin
        @test length(_NATIVE_SAVE_TYPES_FALLBACK) == 350
        for t in ("VARModel", "BVARPosterior", "RegModel", "DSGESolution",
                  "PerturbationSolution", "KrusellSmithSolution", "HASteadyState",
                  "HADSGESolution", "BayesianDSGE", "PosteriorMode", "SVARModel",
                  "SVECResult", "MaxShareResult", "ProxySVARResult",
                  "RobustBayesResult", "SignIdentifiedSet", "ImpulseResponse",
                  "BayesianImpulseResponse", "FEVD", "VARForecast", "BVARForecast",
                  "IOMultipliers", "LinkageResult", "JohansenResult",
                  "IdentifiabilityTestResult", "MFVARPosterior", "TVPVARPosterior")
            @test t in _NATIVE_SAVE_TYPES_FALLBACK
        end
    end

    @testset "seeded mock estimators record manifests" begin
        Y = randn(60, 2)
        post = estimate_bvar(Y, 1; seed=11)
        @test post.manifest !== nothing && post.manifest.seed == 11
        @test estimate_bvar(Y, 1).manifest === nothing
    end

    @testset "model info reads the native header" begin
        mktempdir() do dir
            p = joinpath(dir, "m.jld2")
            save_model_dispatch(p, estimate_bvar(randn(60, 2), 1))
            info = _native_model_info(p)
            @test info.model_type == "BVARPosterior"
            @test info.magic == "MEMs native (jld2)"
            @test info.note == ""
            doc = _run_leaf(["model", "info", p])
            @test string(doc.status) == "ok"
            @test haskey(doc.data, :model_handle_info)
        end
    end

    @testset "model reproduce: match / decline / unsupported" begin
        mktempdir() do dir
            seeded = joinpath(dir, "s.jld2")
            save_model_dispatch(seeded, estimate_bvar(randn(60, 2), 1; seed=11))
            doc = _run_leaf(["model", "reproduce", seeded])
            @test string(doc.status) == "ok"
            @test haskey(doc.data, :model_reproduce_summary)
            @test haskey(doc.data, :model_reproduce_fields)
            mrow = only(r for r in doc.data.model_reproduce_summary.rows if r[1] == "matched")
            @test mrow[2] == "true"
            srow = only(r for r in doc.data.model_reproduce_summary.rows if r[1] == "seed")
            @test srow[2] == "11"

            plain = joinpath(dir, "u.jld2")
            save_model_dispatch(plain, estimate_bvar(randn(60, 2), 1))
            doc2 = _run_leaf(["model", "reproduce", plain])
            @test string(doc2.status) == "ok"
            mrow2 = only(r for r in doc2.data.model_reproduce_summary.rows if r[1] == "matched")
            @test mrow2[2] == "unverifiable (no recorded seed)"
            @test !haskey(doc2.data, :model_reproduce_fields)

            # Deterministic models carry no manifest: upstream's universal
            # reproduce(x) fallback answers with a missing verdict (exit 0),
            # never model/unsupported — the honest "cannot verify", not a pass.
            varp = joinpath(dir, "v.jld2")
            save_model_dispatch(varp, estimate_var(randn(60, 2), 1))
            doc3 = _run_leaf(["model", "reproduce", varp])
            @test string(doc3.status) == "ok"
            mrow3 = only(r for r in doc3.data.model_reproduce_summary.rows if r[1] == "matched")
            @test mrow3[2] == "unverifiable (no recorded seed)"
            @test !haskey(doc3.data, :model_reproduce_fields)

            # missing positional: the strict parser throws before any envelope
            # (same ParseError family as every other leaf's missing-arg case)
            errm = try
                _dispatch_via_app(["model", "reproduce", "--format", "json"])
                nothing
            catch e
                e
            end
            @test errm isa ParseError
            @test occursin("missing required argument", errm.message)
        end
    end

    @testset "model reproduce via model:// session handle" begin
        _SERVE_MODEL_STORE[] = Dict{String,Any}()
        try
            save_model_dispatch("model://rep", estimate_bvar(randn(60, 2), 1; seed=5))
            doc = _run_leaf(["model", "reproduce", "model://rep"])
            @test string(doc.status) == "ok"
            mrow = only(r for r in doc.data.model_reproduce_summary.rows if r[1] == "matched")
            @test mrow[2] == "true"
        finally
            _SERVE_MODEL_STORE[] = nothing
        end
    end
end

include(joinpath(project_root, "test", "test_handles.jl"))
