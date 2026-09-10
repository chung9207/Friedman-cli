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

# DSGE commands: solve, irf, fevd, simulate, estimate, perfect-foresight, steady-state, bayes (NodeCommand)

const RA_METHOD_OPTION = OptionSpec(
    name="method", type=String, default="gensys",
    description="Solution method: gensys|klein|perturbation|projection|pfi|vfi|blanchard-kahn",
    choices=_RA_METHOD_CHOICES)
const RA_SOLVER_KNOB_OPTIONS = [
    OptionSpec(name="next-state", type=String, default="",
               description="VFI: auto|linear|residual; PFI: linear|policy|nonlinear"),
    OptionSpec(name="howard-steps", type=Int, default=-1,
               description="Howard policy-evaluation steps (vfi default 20, pfi 0; -1 = method default)"),
    OptionSpec(name="n-grid", type=Int, default=0, description="VFI tensor-grid nodes per state (tensor path only; ≥3; 0 = default 12)"),
    OptionSpec(name="n-choice", type=Int, default=0, description="VFI line-search points (grid1d only; ≥3; 0 = default 41)"),
    OptionSpec(name="optimizer", type=String, default="", choices=_VFI_OPTIMIZER_CHOICES,
               description="VFI Bellman maximizer: auto (grid1d for 1 control, fminbox-nm for vectors)|grid1d|fminbox-nm|fminbox-lbfgs"),
    OptionSpec(name="smolyak-mu", type=String, default="",
               description="VFI Smolyak level: scalar μ ≥ 0 or comma-separated per-dimension levels (Smolyak path only; unset = default 2)"),
    OptionSpec(name="n-quad", type=Int, default=0, description="VFI/PFI quadrature nodes per shock (0 = default 5)"),
    OptionSpec(name="scale", type=Float64, default=0.0, description="VFI/PFI state-bound scale (0 = default 3.0)"),
    OptionSpec(name="tol", type=Float64, default=0.0, description="VFI/PFI convergence tolerance (0 = default 1e-8)"),
    OptionSpec(name="max-iter", type=Int, default=0, description="VFI/PFI max iterations (0 = default 500)"),
    OptionSpec(name="damping", type=Float64, default=0.0, description="VFI/PFI mixing factor (0 = default 1.0)"),
    OptionSpec(name="anderson-m", type=Int, default=0, description="PFI Anderson acceleration memory (PFI only)"),
]
const HA_HH_OPTIONS = [
    OptionSpec(name="hh-solver", type=String, default="egm",
               choices=["egm", "vfi"],
               description="Household solver: egm|vfi (SS + Reiter; SSJ is EGM; not with krusell-smith)"),
    OptionSpec(name="distribution", type=String, default="young",
               choices=["young", "winberry"],
               description="Distribution method: young|winberry"),
]

function dsge_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["dsge", "solve"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                RA_METHOD_OPTION,
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="degree", type=Int, default=5, description="Polynomial degree (projection/pfi/vfi)"),
                OptionSpec(name="grid", type=String, default="auto", description="Grid type: auto|chebyshev|smolyak (vfi: auto|tensor|smolyak; auto routes nx≥4 to Smolyak)"),
                RA_SOLVER_KNOB_OPTIONS...,
                OptionSpec(name="evaluate-at", type=String, default="",
                           description="State vector x1,x2,… at which to evaluate the VFI value function"),
                OptionSpec(name="constraints", type=String, default="", description="Path to OccBin constraints TOML"),
                OptionSpec(name="constraint-solver", type=String, default="", description="Constraint solver: nonlinearsolve|optim|nlopt|ipopt|path"),
                OptionSpec(name="periods", type=Int, default=40, description="Number of periods for OccBin simulation"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[
                TableSpec(name=:dsge_solution, description="Gensys/Klein state-transition policy matrix G1, one column per variable"),
                TableSpec(name=:perturbation_policy_gx, description="Perturbation control policy gx: control responses to states and shocks"),
                TableSpec(name=:projection_solution, description="Projection/PFI/VFI basis coefficients, one row per control"),
                TableSpec(name=:projection_diagnostics, description="Projection/PFI/VFI convergence, iterations, residual norm, grid and degree"),
                TableSpec(name=:vfi_value_function, description="Bellman value on collocation nodes (--method vfi)"),
                TableSpec(name=:vfi_value_coefficients, description="Chebyshev coefficients of the Bellman value (--method vfi)"),
                TableSpec(name=:vfi_value_at, description="evaluate_value at --evaluate-at (--method vfi)"),
                TableSpec(name=:determinacy_verdict, description="Sims existence/uniqueness pair and the collapsed determinacy verdict"),
                TableSpec(name=:dsge_occbin_solution, description="OccBin piecewise path per variable (--constraints without --constraint-solver)"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_solve),
        ),
        CommandSpec(
            path=["dsge", "irf"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                RA_METHOD_OPTION,
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="degree", type=Int, default=5, description="Polynomial degree (projection/pfi/vfi)"),
                OptionSpec(name="grid", type=String, default="auto", description="Grid type: auto|chebyshev|smolyak (vfi: auto|tensor|smolyak; auto routes nx≥4 to Smolyak)"),
                RA_SOLVER_KNOB_OPTIONS...,
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="shock-size", type=Float64, default=1.0, description="Shock size (std devs)"),
                OptionSpec(name="n-sim", type=Int, default=0, description="Simulation-based IRF draws (0=analytical)"),
                OptionSpec(name="constraints", type=String, default="", description="Path to OccBin constraints TOML"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[
                TableSpec(name=:dsge_irf, description="Responses of every variable to one shock, one table per shock", family=true),
                TableSpec(name=:occbin_irf, description="Linear vs piecewise OccBin response of one variable, one table per variable (--constraints)", family=true),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_irf),
        ),
        CommandSpec(
            path=["dsge", "fevd"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                RA_METHOD_OPTION,
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="degree", type=Int, default=5, description="Polynomial degree (projection/pfi/vfi)"),
                OptionSpec(name="grid", type=String, default="auto", description="Grid type: auto|chebyshev|smolyak (vfi: auto|tensor|smolyak; auto routes nx≥4 to Smolyak)"),
                RA_SOLVER_KNOB_OPTIONS...,
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="unconditional", description="Unconditional (asymptotic) FEVD for order≥2 perturbation (Andreasen et al. 2018)"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:dsge_fevd, description="Variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_fevd),
        ),
        CommandSpec(
            path=["dsge", "simulate"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                RA_METHOD_OPTION,
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="degree", type=Int, default=5, description="Polynomial degree (projection/pfi/vfi)"),
                OptionSpec(name="grid", type=String, default="auto", description="Grid type: auto|chebyshev|smolyak (vfi: auto|tensor|smolyak; auto routes nx≥4 to Smolyak)"),
                RA_SOLVER_KNOB_OPTIONS...,
                OptionSpec(name="periods", type=Int, default=200, description="Simulation periods (after burn-in)"),
                OptionSpec(name="burn", type=Int, default=100, description="Burn-in periods to discard"),
                OptionSpec(name="seed", type=Int, default=0, description="Random seed (0=no seed)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="antithetic", description="Use antithetic sampling for variance reduction"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:dsge_simulation, description="Simulated path of every endogenous variable, burn-in dropped")],
            category="dsge",
            handler=wrap_legacy(_dsge_simulate),
        ),
        # W12/#114: determinacy-region mapping (MEMs#367). Config-driven because the sweep
        # is two parameter names + two grids + a resolution — well past the "few flags"
        # threshold. `plot_result(::DeterminacyMap)` EXISTS upstream
        # (plotting/dsge_extra.jl:258), so --plot is honoured, not advertised on faith.
        CommandSpec(
            path=["dsge", "determinacy-map"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="config", type=String, default="",
                           description="TOML with a [determinacy] section (params, lower/upper/points or grids); REQUIRED"),
                OptionSpec(name="rank-rtol", type=Float64, default=1e-8,
                           description="Relative tolerance of the Sims rank tests"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="threaded", description="Evaluate grid points on all available threads (results are identical to the serial sweep)"),
                FlagSpec(name="verbose-solver", description="Do NOT suppress per-grid-point solver warnings"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[
                TableSpec(name=:dsge_determinacy_map, description="One row per grid cell: swept parameter values, verdict code/label and the Sims existence/uniqueness pair"),
                TableSpec(name=:determinacy_region_summary, description="Grid-point counts per determinacy region plus the solver settings"),
                TableSpec(name=:determinacy_boundary, description="Parameter values where the verdict changes (one-parameter sweeps only)"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_determinacy_map),
        ),
        # W12/#114: closed-form (simulation-free) theoretical moments at perturbation order
        # 1/2/3 (MEMs#368). NOTE there is deliberately no `--pruned` switch: upstream's
        # `simulate(::PerturbationSolution)` is ALWAYS pruned (Kim et al. 2008) and exposes
        # no unpruned path, so a flag would advertise a choice that does not exist.
        CommandSpec(
            path=["dsge", "moments"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="method", type=String, default="perturbation",
                           description="Solution method (moments need a PerturbationSolution)"),
                OptionSpec(name="order", type=Int, default=2,
                           description="Perturbation order: 1, 2 or 3 (default 2; for linear models order 2 equals order 1 exactly)"),
                OptionSpec(name="lags", type=Int, default=1, description="Autocovariance lags to report (>= 1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:dsge_theoretical_moments, description="Per-variable steady state, unconditional mean, risk correction and standard deviation"),
                TableSpec(name=:variance_covariance, description="Pairwise covariance and correlation for every variable pair"),
                TableSpec(name=:autocovariances, description="Own autocovariance and autocorrelation at each reported lag"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_moments),
        ),
        CommandSpec(
            path=["dsge", "estimate"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="data", short="d", type=String, default="", description="Path to CSV data file"),
                OptionSpec(name="method", type=String, default="irf_matching", description="Estimation method: irf_matching|likelihood|bayesian|smm"),
                OptionSpec(name="params", type=String, default="", description="Comma-separated parameter names to estimate"),
                OptionSpec(name="solve-method", type=String, default="gensys", description="DSGE solution method"),
                OptionSpec(name="solve-order", type=Int, default=1, description="Perturbation order for solution"),
                OptionSpec(name="weighting", type=String, default="optimal", description="Weighting matrix: identity|optimal|diagonal"),
                OptionSpec(name="irf-horizon", type=Int, default=20, description="IRF horizon for matching"),
                OptionSpec(name="var-lags", type=Int, default=4, description="VAR lags for empirical IRF"),
                OptionSpec(name="sim-ratio", type=Int, default=5, description="Simulation-to-data ratio (SMM)"),
                OptionSpec(name="bounds", type=String, default="", description="Path to parameter bounds TOML"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:dsge_estimation, description="Estimated structural parameters with standard errors, t-stats and p-values")],
            category="dsge",
            handler=wrap_legacy(_dsge_estimate),
        ),
        CommandSpec(
            path=["dsge", "perfect-foresight"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="shocks", type=String, default="", description="Path to shock sequence CSV"),
                OptionSpec(name="constraints", type=String, default="", description="Path to constraints TOML"),
                OptionSpec(name="constraint-solver", type=String, default="", description="Constraint solver: nonlinearsolve|optim|nlopt|ipopt|path"),
                OptionSpec(name="periods", type=Int, default=100, description="Simulation periods"),
                OptionSpec(name="sparsity", type=String, default="auto",
                           choices=["auto", "dense"],
                           description="Jacobian: auto (sparse BT) or dense (nlopt/path/ipopt ignore this)"),
                OptionSpec(name="max-iter", type=Int, default=100, description="Newton iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="Convergence tolerance (> 0)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:perfect_foresight_path, description="Deterministic transition path of every endogenous variable")],
            category="dsge",
            handler=wrap_legacy(_dsge_perfect_foresight),
        ),
        CommandSpec(
            path=["dsge", "steady-state"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="constraints", type=String, default="", description="Path to OccBin constraints TOML"),
                OptionSpec(name="constraint-solver", type=String, default="", description="Constraint solver: nonlinearsolve|optim|nlopt|ipopt|path"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:dsge_steady_state, description="Steady-state level of every endogenous variable")],
            category="dsge",
            handler=wrap_legacy(_dsge_steady_state),
        ),
        CommandSpec(
            path=["dsge", "hd"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                RA_METHOD_OPTION,
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="degree", type=Int, default=5, description="Polynomial degree (projection/pfi/vfi)"),
                OptionSpec(name="grid", type=String, default="auto", description="Grid type: auto|chebyshev|smolyak (vfi: auto|tensor|smolyak; auto routes nx≥4 to Smolyak)"),
                RA_SOLVER_KNOB_OPTIONS...,
                OptionSpec(name="data", short="d", type=String, default="", description="Path to CSV data file"),
                OptionSpec(name="observables", type=String, default="", description="Observable variable names (comma-separated)"),
                OptionSpec(name="states", type=String, default="observables", description="observables|all"),
                OptionSpec(name="measurement-error", type=String, default="", description="Measurement error std devs (comma-separated) or auto"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:dsge_historical_decomposition, description="Per-variable contribution path of one shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_hd),
        ),
        CommandSpec(
            path=["dsge", "bayes", "estimate"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH (Christen & Fox 2005)")
            ],
            tables=[TableSpec(name=:bayesian_dsge_posterior, description="Posterior mean, std, median and 5/95% quantiles per estimated parameter")],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_estimate),
        ),
        CommandSpec(
            path=["dsge", "bayes", "irf"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bayesian_dsge_irf, description="Posterior-mean responses of every variable to one shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_irf),
        ),
        CommandSpec(
            path=["dsge", "bayes", "fevd"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bayesian_dsge_fevd, description="Posterior-mean variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_fevd),
        ),
        CommandSpec(
            path=["dsge", "bayes", "simulate"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="periods", type=Int, default=200, description="Simulation periods"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bayesian_dsge_simulation, description="Posterior-mean simulated path of every variable")],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_simulate),
        ),
        CommandSpec(
            path=["dsge", "bayes", "summary"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[
                TableSpec(name=:bayesian_dsge_posterior_summary, description="Posterior mean, median, std and 5/95% quantiles per parameter"),
                TableSpec(name=:prior_vs_posterior_comparison, description="Prior mean/std against posterior mean/std and quantiles per parameter"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_summary),
        ),
        CommandSpec(
            path=["dsge", "bayes", "compare"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="model2", type=String, default="", description="Path to second DSGE model file"),
                OptionSpec(name="params2", type=String, default="", description="Parameters for second model (comma-separated)"),
                OptionSpec(name="priors2", type=String, default="", description="Priors TOML for second model")
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[TableSpec(name=:bayesian_model_comparison, description="Log marginal likelihood and acceptance rate for each of the two models")],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_compare),
        ),
        CommandSpec(
            path=["dsge", "bayes", "predictive"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="n-sim", type=Int, default=500, description="Number of predictive simulations"),
                OptionSpec(name="periods", type=Int, default=100, description="Periods per simulation"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:posterior_predictive_summary, description="Mean, std, min and max of each variable across the predictive simulations")],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_predictive),
        ),
        CommandSpec(
            path=["dsge", "bayes", "hd"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="n-hd-draws", type=Int, default=200, description="Number of posterior draws for HD"),
                OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84", description="Quantile levels"),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="mode-only", description="Use posterior mode only (no full posterior)"),
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bayesian_dsge_historical_decomposition, description="Posterior-mean per-variable contribution path of one shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_hd),
        ),
        # ── Bayesian DSGE diagnostics (C073 / MEMs 0.7.0) ──
        CommandSpec(
            path=["dsge", "bayes", "mcmc-diag"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[
                TableSpec(name=:mcmc_convergence_diagnostics, description="Per-parameter R-hat, bulk/tail ESS and Geweke z and p-value"),
                TableSpec(name=:mcmc_diagnostics_summary, description="Draw count and sampler behind the diagnostics"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_mcmc_diag),
        ),
        CommandSpec(
            path=["dsge", "bayes", "identification"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                OptionSpec(name="params", type=String, default="", description="Comma-separated estimated parameter names (required)"),
                OptionSpec(name="observables", type=String, default="", description="Observable variable names (comma-separated; default: all endogenous)"),
                OptionSpec(name="solver", type=String, default="gensys", description="gensys|klein|perturbation", choices=["gensys","klein","perturbation"]),
                OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
                OptionSpec(name="n-lags", type=Int, default=2, description="Autocovariance lags in the Iskrev moment vector"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:identification_diagnostics, description="Iskrev rank test: rank, parameter/moment counts, tolerance and the identified verdict"),
                TableSpec(name=:singular_values, description="Singular values of the moment Jacobian, in order"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_identification),
        ),
        CommandSpec(
            path=["dsge", "bayes", "learning-rate"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="fractions", type=String, default="0.5,1.0", description="Nested subsample fractions (comma-separated, in (0,1])"),
                OptionSpec(name="threshold", type=Float64, default=0.2, description="Flag threshold on the learning rate α"),
                OptionSpec(name="refit-n-smc", type=Int, default=100, description="SMC particles per subsample refit"),
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[
                TableSpec(name=:learning_rate_check, description="Per-parameter Koop-Pesaran-Smith learning rate and its flag"),
                TableSpec(name=:learning_rate_summary, description="Flag threshold and the nested subsample sizes"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_learning_rate),
        ),
        CommandSpec(
            path=["dsge", "bayes", "overlap"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="threshold", type=Float64, default=0.8, description="Flag threshold on the prior/posterior overlap"),
                OptionSpec(name="n-grid", type=Int, default=0, description="Histogram bins (0 = auto ≈ √N)"),
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[
                TableSpec(name=:prior_posterior_overlap, description="Per-parameter prior/posterior overlap and weak-identification flag"),
                TableSpec(name=:overlap_summary, description="Flag threshold applied to the overlap"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_overlap),
        ),
        CommandSpec(
            path=["dsge", "bayes", "marginal-lik"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                BAYES_OPTIONS...,
                OptionSpec(name="proposal", type=String, default="normal", description="Bridge proposal family: normal|t", choices=["normal","t"]),
                OptionSpec(name="df", type=Float64, default=5.0, description="Degrees of freedom for the t proposal"),
            ],
            flags=[
                FlagSpec(name="delayed-acceptance", description="Use delayed acceptance for MH")
            ],
            tables=[TableSpec(name=:marginal_likelihood_bridge_sampling, description="Bridge-sampling and SMC log marginal likelihoods with the proposal settings")],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_marginal_lik),
        ),
        # C073 remainder (#78). NOTE: BAYES_OPTIONS is NOT splatted here — it carries
        # sampler/n-smc/n-particles/burnin/ess-target, which neither handler accepts, and
        # declaring an option a handler cannot take MethodErrors on every use (#85).
        # select_options picks exactly the ones each signature has.
        CommandSpec(
            path=["dsge", "bayes", "posterior-mode"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                select_options(BAYES_OPTIONS, "data", "params", "priors", "observables",
                               "solver", "order", "constraint-solver", "output", "format")...,
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations (≥ 1)"),
                OptionSpec(name="f-reltol", type=Float64, default=1e-8, description="Relative function tolerance (> 0)"),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:posterior_mode, description="Posterior mode and Laplace standard error per parameter"),
                TableSpec(name=:posterior_mode_diagnostics, description="Log posterior/likelihood, Laplace log ML, convergence flag and iteration count"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_posterior_mode),
        ),
        CommandSpec(
            path=["dsge", "bayes", "prior-predictive"],
            summary="Path to DSGE model file (.toml or .jl)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing, description="")],
            options=[
                # no --data: prior_predictive draws from the PRIOR and needs none
                select_options(BAYES_OPTIONS, "params", "priors", "observables", "solver",
                               "order", "constraint-solver", "n-draws", "output", "format")...,
                OptionSpec(name="periods", type=Int, default=200, description="Periods to simulate per draw (≥ 1)"),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:prior_predictive_distribution, description="Mean, std, median and 5/95% quantiles of each summary statistic across prior draws"),
                TableSpec(name=:prior_predictive_summary, description="Draws requested, draws that solved and periods simulated per draw"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bayes_prior_predictive),
        ),
        # ── HA-DSGE node (C040 / MEMs 0.6.7) ──
        # estimate un-deferred (C048): MEMs#228 fixed in 0.6.7 — observation matrix Z is
        # now built from the reduction C rows, so HA Bayesian estimation is meaningful.
        CommandSpec(
            path=["dsge", "ha", "accuracy"],
            summary="Den Haan (2010) accuracy of the aggregate law of motion",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Capital builtin (krusell-smith|one-asset-hank) or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="krusell-smith",
                           description="Solution to score: krusell-smith|ssj|reiter",
                           choices=["krusell-smith", "ssj", "reiter"]),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced distribution states"),
                OptionSpec(name="t-sim", type=Int, default=10000, description="Simulation length (must exceed --t-burn by >= 10)"),
                OptionSpec(name="t-burn", type=Int, default=1000, description="Burn-in discarded before scoring"),
                OptionSpec(name="t-fit", type=Int, default=4000,
                           description="Fitting length for the implied law (> 100; ssj|reiter only)"),
                OptionSpec(name="rho-z", type=Float64, default=0.95, description="Aggregate shock persistence, |rho| < 1"),
                OptionSpec(name="sigma-z", type=Float64, default=0.007, description="Aggregate shock s.d. (> 0)"),
                OptionSpec(name="seed", type=Int, default=98765, description="Simulation seed"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                PLOT_OPTIONS...,
            ],
            flags=copy(PLOT_FLAGS),
            tables=[
                TableSpec(name=:den_haan_accuracy, description="Max/mean percentage deviation plus the reference and PLM standard deviations"),
                TableSpec(name=:reference_vs_plm_only_aggregate_path, description="Simulated reference and PLM-only aggregate paths side by side"),
                TableSpec(name=:den_haan_simulation_settings, description="Solution method, scored aggregate, simulation lengths and seed"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_accuracy),
        ),
        CommandSpec(
            path=["dsge", "ha", "solve"],
            summary="Solve HA-DSGE (SSJ / Reiter / Krusell-Smith)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin (huggett|krusell-smith|one-asset-hank|two-asset-hank|endogenous-labor) or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="ssj",
                           description="HA solution method: ssj|reiter|krusell-smith",
                           choices=["ssj", "reiter", "krusell-smith"]),
                OptionSpec(name="n-reduced", type=Int, default=30,
                           description="Reduced distribution states (SSJ/Reiter)"),
                OptionSpec(name="t-horizon", type=Int, default=300,
                           description="Sequence-space horizon (SSJ)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:ha_dsge_solve_diagnostics, description="Solution method with its size and fit diagnostics"),
                TableSpec(name=:krusell_smith_plm_coefficients, description="Fitted perceived-law-of-motion coefficients (--method krusell-smith)"),
                TableSpec(name=:ha_steady_state_aggregates, description="Steady-state aggregate quantities"),
                TableSpec(name=:ha_steady_state_prices, description="Steady-state prices"),
                TableSpec(name=:ha_steady_state_diagnostics, description="Steady-state convergence, iterations, Euler error and excess demand"),
                TableSpec(name=:ha_euler_accuracy_log10_by_convention, description="log10 Euler errors under both the midpoints and nodes conventions"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_solve),
        ),
        CommandSpec(
            path=["dsge", "ha", "steady-state"],
            summary="Compute HA-DSGE stationary equilibrium",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="euler-points", type=String, default="midpoints",
                           description="Euler-error evaluation points: midpoints|nodes",
                           choices=["midpoints", "nodes"]),
                OptionSpec(name="max-iter", type=Int, default=0,
                           description="GE iterations (0 = upstream default: 200 one-asset, 60 two-asset closer)"),
                OptionSpec(name="tol", type=Float64, default=0.0,
                           description="Market-clearing tolerance (0 = upstream default)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:ha_steady_state_aggregates, description="Steady-state aggregate quantities"),
                TableSpec(name=:ha_steady_state_prices, description="Steady-state prices"),
                TableSpec(name=:ha_steady_state_diagnostics, description="Convergence, iterations, Euler error and excess demand"),
                TableSpec(name=:ha_euler_accuracy_log10_by_convention, description="log10 Euler errors under both the midpoints and nodes conventions"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_steady_state),
        ),
        CommandSpec(
            path=["dsge", "ha", "irf"],
            summary="Aggregate IRFs from linearized HA-DSGE solution",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="reiter",
                           description="HA solution method: ssj|reiter (krusell-smith has no linear IRF)",
                           choices=["ssj", "reiter"]),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:ha_dsge_irf, description="Aggregate responses of every variable to one shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_irf),
        ),
        CommandSpec(
            path=["dsge", "ha", "fevd"],
            summary="Aggregate FEVD from linearized HA-DSGE solution",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="reiter",
                           description="HA solution method: ssj|reiter",
                           choices=["ssj", "reiter"]),
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:ha_dsge_fevd, description="Aggregate variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_fevd),
        ),
        CommandSpec(
            path=["dsge", "ha", "simulate"],
            summary="Simulate aggregate paths from linearized HA-DSGE",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="reiter",
                           description="HA solution method: ssj|reiter",
                           choices=["ssj", "reiter"]),
                OptionSpec(name="periods", type=Int, default=200, description="Simulation periods"),
                OptionSpec(name="seed", type=Int, default=0, description="Random seed (0=no seed)"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:ha_dsge_simulation, description="Simulated path of every aggregate deviation")],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_simulate),
        ),
        CommandSpec(
            path=["dsge", "ha", "distribution-irf"],
            summary="Wealth distribution IRF after an aggregate shock (Reiter only)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="reiter",
                           description="Must be reiter (SSJ has no distribution basis)",
                           choices=["reiter"]),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="shock-index", type=Int, default=1, description="Aggregate shock index (1-based)"),
                OptionSpec(name="shock-size", type=Float64, default=1.0, description="Shock size (std devs)"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ha_distribution_irf, description="Per-horizon L1 and max wealth-distribution mass deviation with the grid sizes")],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_distribution_irf),
        ),
        CommandSpec(
            path=["dsge", "ha", "inequality-irf"],
            summary="Gini and wealth-percentile IRFs after an aggregate shock",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="reiter",
                           description="Must be reiter for dynamic inequality IRF",
                           choices=["reiter"]),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="shock-index", type=Int, default=1, description="Aggregate shock index (1-based)"),
                OptionSpec(name="shock-size", type=Float64, default=1.0, description="Shock size (std devs)"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:ha_inequality_irf, description="Per-horizon Gini and wealth-percentile (p10-p90) responses")],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_inequality_irf),
        ),
        CommandSpec(
            path=["dsge", "ha", "simulate-panel"],
            summary="Simulate individual asset holdings from steady-state policies",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="n-agents", type=Int, default=1000, description="Number of agents"),
                OptionSpec(name="periods", type=Int, default=100, description="Time periods"),
                OptionSpec(name="seed", type=Int, default=0, description="Random seed (0=no seed)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ha_panel_simulation_summary, description="Cross-sectional mean and sd of assets per period with the agent count")],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_simulate_panel),
        ),
        CommandSpec(
            path=["dsge", "ha", "estimate"],
            summary="Bayesian estimation of HA-DSGE parameters (MH/SMC; MEMs#228 fixed in 0.6.7)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="data", type=String, default="", description="Path to observed aggregates CSV (required)"),
                OptionSpec(name="priors", type=String, default="", description="Path to priors TOML with [priors] section (required)"),
                OptionSpec(name="observables", type=String, default="",
                           description="Comma-separated observed aggregates (e.g. K,Y); default: first aggregates"),
                OptionSpec(name="method", type=String, default="ssj",
                           description="HA solution method re-solved each draw: ssj|reiter",
                           choices=["ssj", "reiter"]),
                OptionSpec(name="sampler", type=String, default="mh",
                           description="Posterior sampler: mh (RWMH) or smc",
                           choices=["mh", "smc"]),
                OptionSpec(name="n-draws", type=Int, default=2000, description="Total RWMH draws (including burn-in)"),
                OptionSpec(name="burnin", type=Int, default=500, description="Burn-in draws to discard"),
                OptionSpec(name="n-smc", type=Int, default=500, description="SMC particles (HA default 500)"),
                OptionSpec(name="n-mh-steps", type=Int, default=1, description="MH mutation steps per SMC stage"),
                OptionSpec(name="ess-target", type=Float64, default=0.5, description="ESS target for SMC resampling"),
                OptionSpec(name="t-horizon", type=Int, default=300,
                           description="Sequence-space truncation length (SSJ); default 300 (ABRS 2021)"),
                OptionSpec(name="n-reduced", type=Int, default=15, description="Reduced distribution states"),
                OptionSpec(name="proposal-scale", type=Float64, default=0.01, description="Initial RWMH proposal scale"),
                OptionSpec(name="adapt-interval", type=Int, default=100, description="Adapt proposal covariance every N draws"),
                OptionSpec(name="measurement-error", type=String, default="none",
                           description="Measurement error: none|auto (auto adds 10% per-obs variance)",
                           choices=["none", "auto"]),
                OptionSpec(name="seed", type=Int, default=0, description="Random seed (0=no seed)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:ha_dsge_bayesian_posterior, description="Posterior mean, std, median and 5/95% quantiles per parameter"),
                TableSpec(name=:ha_dsge_bayesian_settings, description="Sampler, solution method, observables and measurement-error provenance"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_estimate),
        ),
        CommandSpec(
            path=["dsge", "ha", "hd"],
            summary="Historical decomposition of HA-DSGE aggregates",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin name or .jl HA ModelSpec")],
            options=[
                HA_HH_OPTIONS...,
                OptionSpec(name="method", type=String, default="ssj",
                           description="HA solution method: ssj|reiter",
                           choices=["ssj", "reiter"]),
                OptionSpec(name="data", short="d", type=String, default="", description="Path to CSV data file (levels)"),
                OptionSpec(name="observables", type=String, default="", description="Observable aggregates (comma-separated; keys of ss.aggregates/ss.prices)"),
                OptionSpec(name="measurement-error", type=String, default="", description="Measurement error std devs (comma-separated) or auto"),
                OptionSpec(name="n-reduced", type=Int, default=30, description="Reduced states"),
                OptionSpec(name="t-horizon", type=Int, default=300, description="Sequence-space horizon (SSJ)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:ha_historical_decomposition, description="Per-variable contribution path of one shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_ha_hd),
        ),
        # ── Continuous-time HA (C041) ──
        CommandSpec(
            path=["dsge", "ct", "solve"],
            summary="Continuous-time Aiyagari (or two-asset KMV) stationary equilibrium",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="rho", type=Float64, default=0.05, description="Discount rate"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion"),
                OptionSpec(name="delta", type=Float64, default=0.05, description="Depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP level"),
                OptionSpec(name="a-min", type=Float64, default=0.0, description="Asset grid lower bound"),
                OptionSpec(name="a-max", type=Float64, default=30.0, description="Asset grid upper bound"),
                OptionSpec(name="grid-size", type=Int, default=100, description="Asset grid points (I)"),
                OptionSpec(name="max-iter", type=Int, default=100, description="Outer equilibrium iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-6, description="Convergence tolerance"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=[
                FlagSpec(name="two-asset", description="Solve Kaplan-Moll-Violante two-asset model instead"),
                FlagSpec(name="ge", description="Two-asset general equilibrium (ct_two_asset_ge; requires --two-asset)"),
            ],
            tables=[
                TableSpec(name=:ct_aiyagari_prices, description="Equilibrium interest rate and wage"),
                TableSpec(name=:ct_aiyagari_aggregates, description="Equilibrium capital, labour and the convergence flag"),
                TableSpec(name=:ct_two_asset_solution, description="Liquid/illiquid holdings, distribution mass and HJB convergence (--two-asset)"),
                TableSpec(name=:ct_two_asset_ge, description="Two-asset GE prices, aggregates and market residuals (--two-asset --ge)"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ct_solve),
        ),
        CommandSpec(
            path=["dsge", "ct", "transition"],
            summary="MIT-shock perfect-foresight transition (ct_mit_shock)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="rho", type=Float64, default=0.05, description="Discount rate"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion"),
                OptionSpec(name="delta", type=Float64, default=0.05, description="Depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Steady-state TFP"),
                OptionSpec(name="shock-size", type=Float64, default=0.95, description="Impact TFP multiplier (Z_0 = shock-size * z)"),
                OptionSpec(name="periods", type=Int, default=40, description="Transition length (time points)"),
                OptionSpec(name="dt", type=Float64, default=0.25, description="Time step"),
                OptionSpec(name="a-max", type=Float64, default=30.0, description="Asset grid upper bound"),
                OptionSpec(name="grid-size", type=Int, default=100, description="Asset grid points"),
                OptionSpec(name="max-iter", type=Int, default=100, description="Shooting iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-6, description="Convergence tolerance"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="z-path", type=String, default="", description="CSV of TFP path (two-asset MIT; length ≥ 2, all positive)"),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser (one-asset CTTransition only)"),
                FlagSpec(name="two-asset", description="Two-asset MIT (ct_two_asset_mit; no plot)"),
            ],
            tables=[
                TableSpec(name=:ct_mit_shock_transition, description="MIT-shock transition path of t, Z, K, r, w and C"),
                TableSpec(name=:ct_two_asset_transition, description="Two-asset MIT path of t, Z, K, r_a, r_b, w, B, C (--two-asset)"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_ct_transition),
        ),
        CommandSpec(
            path=["dsge", "ct", "irf"],
            summary="MIT impulse response of CT Aiyagari or two-asset GE",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="rho", type=Float64, default=0.05, description="Discount rate"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion"),
                OptionSpec(name="delta", type=Float64, default=0.05, description="Depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP level"),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="dt", type=Float64, default=0.25, description="Time step"),
                OptionSpec(name="grid-size", type=Int, default=100, description="Asset grid points"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="two-asset", description="Two-asset GE IRF (K, r_a, r_b, w, B, Z)"),
                FlagSpec(name="plot", description="Open interactive plot in browser"),
            ],
            tables=[TableSpec(name=:ct_irf, description="MIT responses of every variable to the TFP shock, one table per shock", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_ct_irf),
        ),
        CommandSpec(
            path=["dsge", "ct", "fevd"],
            summary="FEVD from the CT MIT impulse (single TFP shock)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="rho", type=Float64, default=0.05, description="Discount rate"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion"),
                OptionSpec(name="delta", type=Float64, default=0.05, description="Depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP level"),
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="dt", type=Float64, default=0.25, description="Time step"),
                OptionSpec(name="grid-size", type=Int, default=100, description="Asset grid points"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="two-asset", description="Two-asset GE FEVD"),
                FlagSpec(name="plot", description="Open interactive plot in browser"),
            ],
            tables=[TableSpec(name=:ct_fevd, description="Variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_ct_fevd),
        ),
        # ── Blanchard OLG (C041) ──
        CommandSpec(
            path=["dsge", "olg", "solve"],
            summary="Blanchard perpetual-youth OLG: steady state + saddle path",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="beta", type=Float64, default=0.96, description="Discount factor"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Depreciation"),
                OptionSpec(name="gamma", type=Float64, default=0.98, description="Survival probability"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="debt", type=Float64, default=0.0, description="Government debt b (see MEMs#237 if b≠0)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:blanchard_olg_steady_state, description="Steady-state k, C, r, w, human wealth, MPC, debt and convergence"),
                TableSpec(name=:blanchard_olg_dynamics, description="Stable eigenvalue, policy slope, determinacy verdict and both eigenvalue moduli"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_olg_solve),
        ),
        CommandSpec(
            path=["dsge", "olg", "simulate"],
            summary="Blanchard OLG transitional dynamics from k0 along the saddle path",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="beta", type=Float64, default=0.96, description="Discount factor"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Depreciation"),
                OptionSpec(name="gamma", type=Float64, default=0.98, description="Survival probability"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="debt", type=Float64, default=0.0, description="Government debt b"),
                OptionSpec(name="k0", type=Float64, default=0.0,
                           description="Initial capital (0 = 80% of steady-state k)"),
                OptionSpec(name="horizon", type=Int, default=50, description="Transition periods H"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:blanchard_olg_transition, description="Transition paths of k, C, r and w from the initial capital stock")],
            category="dsge",
            handler=wrap_legacy(_dsge_olg_simulate),
        ),
        CommandSpec(
            path=["dsge", "olg", "irf"],
            summary="Blanchard OLG IRF via to_spec (TFP; NK adds monetary)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="beta", type=Float64, default=0.96, description="Discount factor"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Depreciation"),
                OptionSpec(name="gamma", type=Float64, default=0.98, description="Survival probability"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="debt", type=Float64, default=0.0, description="Government debt b"),
                OptionSpec(name="rho-z", type=Float64, default=0.0, description="TFP persistence"),
                OptionSpec(name="sigma-z", type=Float64, default=0.0, description="TFP shock scale"),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon"),
                OptionSpec(name="shock-size", type=Float64, default=1.0, description="Shock size (std devs)"),
                OptionSpec(name="kappa", type=Float64, default=0.1, description="NK Phillips slope (--nk)"),
                OptionSpec(name="phi-pi", type=Float64, default=1.5, description="NK Taylor φπ (--nk)"),
                OptionSpec(name="phi-y", type=Float64, default=0.125, description="NK Taylor φy (--nk)"),
                OptionSpec(name="rho-i", type=Float64, default=0.0, description="NK interest smoothing (--nk)"),
                OptionSpec(name="sigma-i", type=Float64, default=0.0, description="NK monetary shock scale (--nk)"),
                OptionSpec(name="omega", type=Float64, default=0.0, description="NK indexation ω in [0,1] (--nk)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="nk", description="NK perpetual-youth variant (adds π, i, rr and eps_i)"),
                FlagSpec(name="plot", description="Open interactive plot in browser"),
            ],
            tables=[TableSpec(name=:blanchard_olg_irf, description="Responses of every variable to one shock (eps_Z; NK also eps_i)", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_olg_irf),
        ),
        CommandSpec(
            path=["dsge", "olg", "fevd"],
            summary="Blanchard OLG FEVD via to_spec (TFP; NK adds monetary)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="beta", type=Float64, default=0.96, description="Discount factor"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Depreciation"),
                OptionSpec(name="gamma", type=Float64, default=0.98, description="Survival probability"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="debt", type=Float64, default=0.0, description="Government debt b"),
                OptionSpec(name="rho-z", type=Float64, default=0.0, description="TFP persistence"),
                OptionSpec(name="sigma-z", type=Float64, default=0.0, description="TFP shock scale"),
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon"),
                OptionSpec(name="kappa", type=Float64, default=0.1, description="NK Phillips slope (--nk)"),
                OptionSpec(name="phi-pi", type=Float64, default=1.5, description="NK Taylor φπ (--nk)"),
                OptionSpec(name="phi-y", type=Float64, default=0.125, description="NK Taylor φy (--nk)"),
                OptionSpec(name="rho-i", type=Float64, default=0.0, description="NK interest smoothing (--nk)"),
                OptionSpec(name="sigma-i", type=Float64, default=0.0, description="NK monetary shock scale (--nk)"),
                OptionSpec(name="omega", type=Float64, default=0.0, description="NK indexation ω in [0,1] (--nk)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="nk", description="NK perpetual-youth variant (adds π, i, rr and eps_i)"),
                FlagSpec(name="plot", description="Open interactive plot in browser"),
            ],
            tables=[TableSpec(name=:blanchard_olg_fevd, description="Variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_olg_fevd),
        ),
        # ── DCEGM (MEMs 0.9.0) ──
        CommandSpec(
            path=["dsge", "dcegm", "solve"],
            summary="Discrete-continuous EGM household (builtin retirement or .jl DCEGMProblem)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon (0 = infinite)"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor in (0,1)"),
                OptionSpec(name="r", type=Float64, default=1.0, description="Gross return R"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="disutility", type=Float64, default=1.0, description="Work disutility"),
                OptionSpec(name="sigma", type=Float64, default=0.0, description="Income-shock s.d."),
                OptionSpec(name="n-shocks", type=Int, default=1, description="Income quadrature nodes (≥ 1)"),
                OptionSpec(name="taste-shock-scale", type=Float64, default=0.0, description="Taste-shock scale (≥ 0)"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="n-a", type=Int, default=200, description="Asset grid points"),
                OptionSpec(name="pension", type=Float64, default=0.0, description="Retirement income"),
                OptionSpec(name="credit-limit", type=Float64, default=0.0, description="Borrowing limit"),
                OptionSpec(name="curvature", type=Float64, default=2.0, description="Grid curvature (≥ 1)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Infinite-horizon iteration cap"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="Policy tolerance"),
                OptionSpec(name="period", type=Int, default=1, description="Stored period for the policy table"),
                OptionSpec(name="income", type=Int, default=1, description="Income-state index for the policy table"),
                OptionSpec(name="view", type=String, default="policy", choices=["policy", "threshold"],
                           description="plot_result view: policy|threshold"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser (DCEGMSolution)")],
            tables=[
                TableSpec(name=:dcegm_solve_diagnostics, description="Convergence, iterations, kinks and sup-norm policy change"),
                TableSpec(name=:dcegm_policy, description="Long policy: one row per option knot at --period/--income"),
                TableSpec(name=:dcegm_kinks, description="Switching-threshold counts per period × option × income"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_solve),
        ),
        CommandSpec(
            path=["dsge", "dcegm", "steady-state"],
            summary="DCEGM capital-market equilibrium (dcegm_steady_state)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon (0 = infinite)"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor in (0,1)"),
                OptionSpec(name="r", type=Float64, default=1.0, description="Gross return R (PE; GE solves for r)"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="disutility", type=Float64, default=1.0, description="Work disutility"),
                OptionSpec(name="sigma", type=Float64, default=0.0, description="Income-shock s.d."),
                OptionSpec(name="n-shocks", type=Int, default=1, description="Income quadrature nodes"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="n-a", type=Int, default=200, description="Asset grid points"),
                OptionSpec(name="pension", type=Float64, default=0.0, description="Retirement income"),
                OptionSpec(name="credit-limit", type=Float64, default=0.0, description="Borrowing limit"),
                OptionSpec(name="curvature", type=Float64, default=2.0, description="Grid curvature"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Firm capital share"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Firm depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Firm TFP"),
                OptionSpec(name="l", type=Float64, default=1.0, description="Firm labour endowment"),
                OptionSpec(name="r-lo", type=Float64, default=0.001, description="Net-return bracket lower end"),
                OptionSpec(name="r-hi", type=Float64, default=0.20, description="Net-return bracket upper end"),
                OptionSpec(name="labor", type=String, default="exogenous", choices=["exogenous", "measured"],
                           description="Labour: exogenous|measured"),
                OptionSpec(name="work-option", type=String, default="work", description="Discrete option treated as work"),
                OptionSpec(name="n-sim", type=Int, default=40, description="Simulation periods (infinite horizon)"),
                OptionSpec(name="tol", type=Float64, default=1e-4, description="|A − K^d| tolerance"),
                OptionSpec(name="max-iter", type=Int, default=40, description="Bisection cap"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=[FlagSpec(name="reprice-wage", description="Feed the firm wage back into work-option income")],
            tables=[TableSpec(name=:dcegm_equilibrium, description="Clearing r, w, K, L, Y, excess demand and convergence")],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_steady_state),
        ),
        CommandSpec(
            path=["dsge", "dcegm", "irf"],
            summary="MIT IRF of a DCEGM equilibrium (needs GE, not DCEGMSolution)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="n-a", type=Int, default=80, description="Asset grid points"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Firm capital share"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Firm depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Firm TFP"),
                OptionSpec(name="horizon", type=Int, default=40, description="IRF horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:dcegm_irf, description="MIT responses of K, r, w, Y, Z to TFP", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_irf),
        ),
        CommandSpec(
            path=["dsge", "dcegm", "fevd"],
            summary="FEVD of a DCEGM equilibrium (single TFP shock)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="n-a", type=Int, default=80, description="Asset grid points"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Firm capital share"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Firm depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Firm TFP"),
                OptionSpec(name="horizon", type=Int, default=40, description="FEVD horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:dcegm_fevd, description="Variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_fevd),
        ),
        CommandSpec(
            path=["dsge", "dcegm", "simulate"],
            summary="MIT simulation of a DCEGM equilibrium (levels)",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="n-a", type=Int, default=80, description="Asset grid points"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Firm capital share"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Firm depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Firm TFP"),
                OptionSpec(name="periods", type=Int, default=40, description="Simulation length (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.0, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:dcegm_simulation, description="Simulated path of K, r, w, Y, Z")],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_simulate),
        ),
        CommandSpec(
            path=["dsge", "dcegm", "transition"],
            summary="MIT TFP path of a DCEGM equilibrium",
            args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                          description="Builtin `retirement` or .jl DCEGMProblem / DCEGMSystem spec")],
            options=[
                OptionSpec(name="z-path", type=String, default="", description="CSV of TFP path (length ≥ 2, all positive; required)"),
                OptionSpec(name="n-periods", type=Int, default=20, description="Finite horizon"),
                OptionSpec(name="beta", type=Float64, default=0.98, description="Discount factor"),
                OptionSpec(name="wage", type=Float64, default=20.0, description="Work-option wage"),
                OptionSpec(name="n-a", type=Int, default=80, description="Asset grid points"),
                OptionSpec(name="a-max", type=Float64, default=50.0, description="Asset grid upper bound"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Firm capital share"),
                OptionSpec(name="delta", type=Float64, default=0.08, description="Firm depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Firm TFP"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:dcegm_transition_path, description="MIT path of t, Z, K, r, w, A, Y"),
                TableSpec(name=:dcegm_transition_diagnostics, description="Method and convergence"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_dcegm_transition),
        ),
        # ── Life-cycle OLG (MEMs 0.9.0) ──
        CommandSpec(
            path=["dsge", "lifecycle", "steady-state"],
            summary="Life-cycle OLG stationary equilibrium",
            args=ArgSpec[],
            options=[
                OptionSpec(name="j", type=Int, default=60, description="Maximum age J (≥ 2)"),
                OptionSpec(name="j-retire", type=Int, default=45, description="Retirement age (≥ 2)"),
                OptionSpec(name="survival", type=Float64, default=0.99, description="Scalar survival probability (overridden by --config vector)"),
                OptionSpec(name="a-max", type=Float64, default=60.0, description="Asset grid upper bound"),
                OptionSpec(name="n-a", type=Int, default=200, description="Asset grid points (≥ 3)"),
                OptionSpec(name="beta", type=Float64, default=0.97, description="Discount factor in (0,1)"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion (> 0)"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share in (0,1)"),
                OptionSpec(name="delta", type=Float64, default=0.06, description="Depreciation in [0,1]"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="n-pop", type=Float64, default=0.0, description="Population growth (> −1)"),
                OptionSpec(name="replacement", type=Float64, default=0.4, description="Pension replacement (≥ 0)"),
                OptionSpec(name="credit-limit", type=Float64, default=0.0, description="Borrowing limit"),
                OptionSpec(name="income-rho", type=Float64, default=0.95, description="Idiosyncratic income persistence"),
                OptionSpec(name="income-sigma", type=Float64, default=0.2, description="Idiosyncratic income s.d."),
                OptionSpec(name="income-states", type=Int, default=5, description="Income states"),
                OptionSpec(name="config", type=String, default="", description="TOML with [lifecycle] survival/earnings vectors"),
                OptionSpec(name="r-lo", type=Float64, default=-0.02, description="Net-return bracket lower end"),
                OptionSpec(name="r-hi", type=Float64, default=0.10, description="Net-return bracket upper end"),
                OptionSpec(name="tol", type=Float64, default=1e-6, description="Excess-demand tolerance"),
                OptionSpec(name="max-iter", type=Int, default=60, description="Bisection cap"),
                OptionSpec(name="bequest-iter", type=Int, default=50, description="Inner bequest iterations"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[
                FlagSpec(name="no-annuities", description="Turn off perfect annuities"),
                FlagSpec(name="plot", description="Open interactive plot in browser (LifeCycleSteadyState)"),
            ],
            tables=[
                TableSpec(name=:lifecycle_steady_state, description="r, w, K, L, Y, tau, pension, transfer, excess demand and convergence"),
                TableSpec(name=:lifecycle_age_profiles, description="Cohort mass and asset/consumption/income profiles by age"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_lifecycle_steady_state),
        ),
        CommandSpec(
            path=["dsge", "lifecycle", "transition"],
            summary="Life-cycle OLG perfect-foresight transition (--k0 XOR --z-path)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="j", type=Int, default=60, description="Maximum age J"),
                OptionSpec(name="j-retire", type=Int, default=45, description="Retirement age"),
                OptionSpec(name="survival", type=Float64, default=0.99, description="Scalar survival probability"),
                OptionSpec(name="a-max", type=Float64, default=60.0, description="Asset grid upper bound"),
                OptionSpec(name="n-a", type=Int, default=80, description="Asset grid points"),
                OptionSpec(name="beta", type=Float64, default=0.97, description="Discount factor"),
                OptionSpec(name="sigma", type=Float64, default=2.0, description="CRRA risk aversion"),
                OptionSpec(name="alpha", type=Float64, default=0.36, description="Capital share"),
                OptionSpec(name="delta", type=Float64, default=0.06, description="Depreciation"),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP"),
                OptionSpec(name="income-rho", type=Float64, default=0.95, description="Idiosyncratic income persistence"),
                OptionSpec(name="income-sigma", type=Float64, default=0.2, description="Idiosyncratic income s.d."),
                OptionSpec(name="income-states", type=Int, default=3, description="Income states"),
                OptionSpec(name="config", type=String, default="", description="TOML with [lifecycle] survival/earnings vectors"),
                OptionSpec(name="k0", type=Float64, default=NaN, description="Initial capital (XOR --z-path)"),
                OptionSpec(name="z-path", type=String, default="", description="CSV of TFP path (length ≥ 3; XOR --k0)"),
                OptionSpec(name="horizon", type=Int, default=80, description="Transition horizon H (ignored when --z-path is set)"),
                OptionSpec(name="tol", type=Float64, default=1e-5, description="Shooting tolerance"),
                OptionSpec(name="max-iter", type=Int, default=80, description="Shooting iterations"),
                OptionSpec(name="relax", type=Float64, default=0.5, description="Damping on the capital update"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:lifecycle_transition_path, description="Path of t, K, r, w, Y, C, Z, pension, transfer"),
                TableSpec(name=:lifecycle_transition_diagnostics, description="tau, convergence and iterations"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_lifecycle_transition),
        ),
        CommandSpec(
            path=["dsge", "lifecycle", "irf"],
            summary="MIT IRF of a life-cycle OLG steady state",
            args=ArgSpec[],
            options=[
                OptionSpec(name="j", type=Int, default=40, description="Maximum age J"),
                OptionSpec(name="j-retire", type=Int, default=30, description="Retirement age"),
                OptionSpec(name="n-a", type=Int, default=40, description="Asset grid points"),
                OptionSpec(name="income-states", type=Int, default=3, description="Income states"),
                OptionSpec(name="horizon", type=Int, default=20, description="IRF horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:lifecycle_irf, description="MIT responses of K, r, w, Y, Z to TFP", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_lifecycle_irf),
        ),
        CommandSpec(
            path=["dsge", "lifecycle", "fevd"],
            summary="FEVD of a life-cycle OLG steady state",
            args=ArgSpec[],
            options=[
                OptionSpec(name="j", type=Int, default=40, description="Maximum age J"),
                OptionSpec(name="j-retire", type=Int, default=30, description="Retirement age"),
                OptionSpec(name="n-a", type=Int, default=40, description="Asset grid points"),
                OptionSpec(name="income-states", type=Int, default=3, description="Income states"),
                OptionSpec(name="horizon", type=Int, default=20, description="FEVD horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:lifecycle_fevd, description="Variance shares by shock across horizons, one table per variable", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_lifecycle_fevd),
        ),
        CommandSpec(
            path=["dsge", "lifecycle", "simulate"],
            summary="MIT simulation of a life-cycle OLG steady state (levels)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="j", type=Int, default=40, description="Maximum age J"),
                OptionSpec(name="j-retire", type=Int, default=30, description="Retirement age"),
                OptionSpec(name="n-a", type=Int, default=40, description="Asset grid points"),
                OptionSpec(name="income-states", type=Int, default=3, description="Income states"),
                OptionSpec(name="periods", type=Int, default=20, description="Simulation length (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.0, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.0, description="AR(1) decay of the TFP impulse"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:lifecycle_simulation, description="Simulated path of K, r, w, Y, Z")],
            category="dsge",
            handler=wrap_legacy(_dsge_lifecycle_simulate),
        ),
        # ── Khan–Thomas firms (MEMs 0.9.0) ──
        CommandSpec(
            path=["dsge", "firm", "steady-state"],
            summary="Khan–Thomas plant-level stationary equilibrium",
            args=ArgSpec[],
            options=[
                OptionSpec(name="n-k", type=Int, default=16, description="Capital-grid nodes (≥ 3)"),
                OptionSpec(name="n-eps", type=Int, default=3, description="Idiosyncratic productivity states"),
                OptionSpec(name="alpha", type=Float64, default=0.256, description="Capital exponent (> 0; α+ν<1)"),
                OptionSpec(name="nu", type=Float64, default=0.640, description="Labour exponent (> 0)"),
                OptionSpec(name="delta", type=Float64, default=0.069, description="Depreciation in (0,1)"),
                OptionSpec(name="beta", type=Float64, default=0.977, description="Discount factor in (0,1)"),
                OptionSpec(name="gamma", type=Float64, default=1.016, description="Utility curvature (≥ 1)"),
                OptionSpec(name="xi-bar", type=Float64, default=0.0083, description="Fixed-cost upper bound (≥ 0)"),
                OptionSpec(name="b", type=Float64, default=0.011, description="Exemption band (≥ 0)"),
                OptionSpec(name="phi", type=Float64, default=2.4, description="Frisch inverse (> 0)"),
                OptionSpec(name="rho-z", type=Float64, default=0.859, description="Aggregate TFP persistence (|ρ|<1)"),
                OptionSpec(name="sigma-z", type=Float64, default=0.014, description="Aggregate TFP s.d. (≥ 0)"),
                OptionSpec(name="rho-e", type=Float64, default=0.859, description="Idiosyncratic persistence"),
                OptionSpec(name="sigma-e", type=Float64, default=0.022, description="Idiosyncratic s.d."),
                OptionSpec(name="z", type=Float64, default=1.0, description="TFP level (> 0)"),
                OptionSpec(name="tol", type=Float64, default=1e-5, description="Wage fixed-point tolerance"),
                OptionSpec(name="max-iter", type=Int, default=16, description="Outer wage iterations"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:khan_thomas_steady_state, description="w, p, K, N, Y, I, C, inaction, convergence"),
                TableSpec(name=:khan_thomas_policy, description="k-grid × ε: unconstrained k*, constrained k' and adjustment probability"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_firm_steady_state),
        ),
        CommandSpec(
            path=["dsge", "firm", "transition"],
            summary="Khan–Thomas MIT TFP path (--prices ss|ge)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="z-path", type=String, default="", description="CSV of TFP path (length ≥ 2, all positive; required)"),
                OptionSpec(name="prices", type=String, default="ss", choices=["ss", "ge"],
                           description="Hold (w,p) at SS or iterate GE wages"),
                OptionSpec(name="n-k", type=Int, default=16, description="Capital-grid nodes"),
                OptionSpec(name="n-eps", type=Int, default=3, description="Idiosyncratic productivity states"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Steady-state TFP"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:khan_thomas_transition, description="Path of t, Z, Y, I, K, N, C, w"),
                TableSpec(name=:khan_thomas_transition_diagnostics, description="Method and convergence"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_firm_transition),
        ),
        CommandSpec(
            path=["dsge", "firm", "irf"],
            summary="Khan–Thomas MIT IRF of Y, I, K, N, C, Z",
            args=ArgSpec[],
            options=[
                OptionSpec(name="horizon", type=Int, default=20, description="IRF horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=NaN, description="AR(1) decay (NaN = firm.rho_z)"),
                OptionSpec(name="prices", type=String, default="ss", choices=["ss", "ge"],
                           description="Hold (w,p) at SS or iterate GE wages"),
                OptionSpec(name="n-k", type=Int, default=16, description="Capital-grid nodes"),
                OptionSpec(name="n-eps", type=Int, default=3, description="Idiosyncratic productivity states"),
                OptionSpec(name="z", type=Float64, default=1.0, description="Steady-state TFP"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser (ImpulseResponse)")],
            tables=[TableSpec(name=:khan_thomas_irf, description="MIT responses of Y, I, K, N, C, Z to TFP", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_firm_irf),
        ),
        # ── Bewley banks (MEMs 0.9.0) ──
        CommandSpec(
            path=["dsge", "bank", "pe"],
            summary="Bewley-bank partial equilibrium at given (R, rᵏ)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="n-n", type=Int, default=25, description="Net-worth grid points"),
                OptionSpec(name="n-xi", type=Int, default=3, description="Idiosyncratic ξ states"),
                OptionSpec(name="n-min", type=Float64, default=0.05, description="Net-worth grid lower bound (> 0)"),
                OptionSpec(name="n-max", type=Float64, default=8.0, description="Net-worth grid upper bound"),
                OptionSpec(name="beta", type=Float64, default=0.99, description="Discount factor in (0,1)"),
                OptionSpec(name="sigma", type=Float64, default=0.95, description="Survival probability in (0,1]"),
                OptionSpec(name="lambda", type=Float64, default=0.20, description="Diversion parameter (> 0)"),
                OptionSpec(name="zeta1", type=Float64, default=0.02, description="Operating-cost scale (≥ 0)"),
                OptionSpec(name="zeta2", type=Float64, default=2.0, description="Operating-cost curvature (> 1)"),
                OptionSpec(name="r", type=Float64, default=1.01, description="Gross deposit rate R (> 0)"),
                OptionSpec(name="rk", type=Float64, default=0.05, description="Net claim return rᵏ"),
                OptionSpec(name="z", type=Float64, default=0.25, description="TFP"),
                OptionSpec(name="alpha", type=Float64, default=0.33, description="Capital share in (0,1)"),
                OptionSpec(name="max-iter", type=Int, default=250, description="PE VFI iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-6, description="PE VFI tolerance"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:bewley_banks_pe, description="PE prices, convergence and iterations"),
                TableSpec(name=:bewley_banks_pe_policy, description="Lending and deposit policy over the net-worth grid"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bank_pe),
        ),
        CommandSpec(
            path=["dsge", "bank", "steady-state"],
            summary="Bewley-bank credit-market stationary equilibrium",
            args=ArgSpec[],
            options=[
                OptionSpec(name="n-n", type=Int, default=25, description="Net-worth grid points"),
                OptionSpec(name="n-xi", type=Int, default=3, description="Idiosyncratic ξ states"),
                OptionSpec(name="n-min", type=Float64, default=0.05, description="Net-worth grid lower bound"),
                OptionSpec(name="n-max", type=Float64, default=8.0, description="Net-worth grid upper bound"),
                OptionSpec(name="beta", type=Float64, default=0.99, description="Discount factor"),
                OptionSpec(name="sigma", type=Float64, default=0.95, description="Survival probability"),
                OptionSpec(name="lambda", type=Float64, default=0.20, description="Diversion parameter"),
                OptionSpec(name="r", type=Float64, default=1.01, description="Gross deposit rate R (held fixed)"),
                OptionSpec(name="z", type=Float64, default=0.25, description="TFP"),
                OptionSpec(name="alpha", type=Float64, default=0.33, description="Capital share"),
                OptionSpec(name="r-lo", type=Float64, default=NaN, description="rᵏ bracket lower end (NaN = default)"),
                OptionSpec(name="r-hi", type=Float64, default=NaN, description="rᵏ bracket upper end (NaN = default)"),
                OptionSpec(name="tol", type=Float64, default=1e-4, description="Credit-market tolerance"),
                OptionSpec(name="max-iter", type=Int, default=24, description="Bisection cap"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:bewley_banks_steady_state, description="R, rk, L, N, B, leverage, Y, excess demand and convergence"),
                TableSpec(name=:bewley_banks_steady_state_policy, description="Lending and deposit policy over the net-worth grid at SS prices"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bank_steady_state),
        ),
        CommandSpec(
            path=["dsge", "bank", "transition"],
            summary="Bewley-bank MIT TFP path",
            args=ArgSpec[],
            options=[
                OptionSpec(name="z-path", type=String, default="", description="CSV of TFP path (length ≥ 2, all positive; required)"),
                OptionSpec(name="n-n", type=Int, default=25, description="Net-worth grid points"),
                OptionSpec(name="n-xi", type=Int, default=3, description="Idiosyncratic ξ states"),
                OptionSpec(name="z", type=Float64, default=0.25, description="Steady-state TFP"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:bewley_banks_transition, description="Path of t, Z, L, Y, K, rk"),
                TableSpec(name=:bewley_banks_transition_diagnostics, description="Method and convergence"),
            ],
            category="dsge",
            handler=wrap_legacy(_dsge_bank_transition),
        ),
        CommandSpec(
            path=["dsge", "bank", "irf"],
            summary="Bewley-bank MIT IRF (variable names from the ImpulseResponse)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="horizon", type=Int, default=20, description="IRF horizon (≥ 2)"),
                OptionSpec(name="shock-size", type=Float64, default=0.01, description="TFP impulse size"),
                OptionSpec(name="persist", type=Float64, default=0.5, description="AR(1) decay (default 0.5)"),
                OptionSpec(name="n-n", type=Int, default=25, description="Net-worth grid points"),
                OptionSpec(name="n-xi", type=Int, default=3, description="Idiosyncratic ξ states"),
                OptionSpec(name="z", type=Float64, default=0.25, description="Steady-state TFP"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser (ImpulseResponse)")],
            tables=[TableSpec(name=:bewley_banks_irf, description="MIT responses of the bank IRF variables to TFP", family=true)],
            category="dsge",
            handler=wrap_legacy(_dsge_bank_irf),
        ),
    ]
end

function register_dsge_commands!()
    # --save-model on the solution-returning leaves (W3/#167 adds the HA pair +
    # bayes estimate: HASteadyState / HADSGESolution / KrusellSmithSolution /
    # BayesianDSGE are all natively registered at MEMs 0.9.3, so they persist
    # via .jld2 — the retired .fmod carve-out).
    specs = map(dsge_specs()) do s
        s.path in (["dsge", "solve"], ["dsge", "ha", "solve"],
                   ["dsge", "ha", "steady-state"],
                   ["dsge", "bayes", "estimate"]) ? with_save_model([s])[1] : s
    end
    out = CommandSpec[]
    for s in specs
        if _has_data_slot(s)
            push!(out, _copy_spec(s; data_kinds=[:timeseries, :csv]))
        else
            push!(out, s)
        end
    end
    out = with_default_csv_kinds(out)
    register!(out)
    return build_node("dsge", out; description="DSGE models: RA, Bayesian, HA, CT, OLG, DCEGM, lifecycle, firm, bank")
end


# ── Implemented Handlers ─────────────────────────────────────

function _dsge_solve(; model::String, method::String="gensys", order::Int=1,
                      degree::Int=5, grid::String="auto",
                      next_state::String="", howard_steps::Int=-1,
                      n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                      scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                      damping::Float64=0.0, anderson_m::Int=0,
                      optimizer::String="", smolyak_mu::String="",
                      evaluate_at::String="",
                      constraints::String="", constraint_solver::String="",
                      periods::Int=40,
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="")
    if !isempty(constraint_solver) && !(constraint_solver in ("nonlinearsolve", "optim", "nlopt", "ipopt", "path"))
        error("invalid --constraint-solver value '$constraint_solver'; must be one of: nonlinearsolve, optim, nlopt, ipopt, path")
    end

    spec = _load_dsge_model(model)

    if !isempty(constraints)
        cons = _load_dsge_constraints(constraints; spec=spec)
        if isempty(constraint_solver)
            # Default: OccBin path (backward compatible)
            _status("\nSolving with OccBin constraints...")
            sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                              next_state=next_state, howard_steps=howard_steps,
                              n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                              scale=scale, tol=tol, max_iter=max_iter,
                              damping=damping, anderson_m=anderson_m,
                              optimizer=optimizer, smolyak_mu=smolyak_mu)
            ob_sol = _occbin_solve_call(spec, cons; periods=periods)

            _maybe_plot(ob_sol; plot=plot, plot_save=plot_save)

            path_df = DataFrame()
            path_df.period = 1:periods
            for (vi, vname) in enumerate(spec.varnames)
                if vi <= size(ob_sol.piecewise_path, 2)
                    path_df[!, vname] = ob_sol.piecewise_path[:, vi]
                end
            end
            output_result(path_df; format=Symbol(format), output=output,
                          title="DSGE OccBin Solution ($(length(cons)) constraint(s), T=$periods)",
                          key="dsge_occbin_solution")
            return
        else
            # New solver hierarchy path
            _status("\nSolving with constraint-solver=$constraint_solver...")
            sol = _solve_dsge(spec; method=method, order=order, degree=degree,
                              grid=grid, constraint_solver=constraint_solver,
                              next_state=next_state, howard_steps=howard_steps,
                              n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                              scale=scale, tol=tol, max_iter=max_iter,
                              damping=damping, anderson_m=anderson_m,
                              optimizer=optimizer, smolyak_mu=smolyak_mu)
        end
    else
        sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                          constraint_solver=constraint_solver,
                          next_state=next_state, howard_steps=howard_steps,
                          n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                          scale=scale, tol=tol, max_iter=max_iter,
                          damping=damping, anderson_m=anderson_m,
                          optimizer=optimizer, smolyak_mu=smolyak_mu)
    end

    # W12/#114 item 4: the Sims [existence, uniqueness] verdict pair. Both DSGESolution and
    # PerturbationSolution carry `eu::Vector{Int}`. Collapsing it to a single "determinate"
    # boolean loses the distinction that matters: e=0 means NO stable solution exists, while
    # e=1,u=0 means solutions exist but are not unique (sunspots) — different diagnoses with
    # different fixes. Emitted for every solution type that carries the pair.
    if hasproperty(sol, :eu) && sol.eu isa AbstractVector && length(sol.eu) >= 2
        e_, u_ = Int(sol.eu[1]), Int(sol.eu[2])
        verdict = e_ == 0 ? "no stable solution (existence fails)" :
                  u_ == 1 ? "determinate (unique stable solution)" :
                            "indeterminate (solutions exist but are not unique)"
        output_kv(Pair{String,Any}[
            "existence" => e_,
            "uniqueness" => u_,
            "verdict" => verdict,
            "solver" => string(hasproperty(sol, :method) ? sol.method : method),
        ]; format=format, output=_per_var_output_path(output, "determinacy"),
           title="Determinacy Verdict")
        e_ == 1 && u_ == 1 ||
            _status_styled("  Determinacy: $verdict\n"; color=:yellow)
    end

    # Standard solve output
    if sol isa MacroEconometricModels.DSGESolution
        n = spec.n_endog
        policy_df = DataFrame()
        policy_df.variable = spec.varnames
        for (vi, vname) in enumerate(spec.varnames)
            if vi <= size(sol.G1, 2)
                policy_df[!, "G1_$vname"] = sol.G1[:, vi]
            end
        end
        output_result(policy_df; format=Symbol(format), output=output,
                      title="DSGE Solution (method=$method)", key="dsge_solution")
    elseif sol isa MacroEconometricModels.PerturbationSolution
        n_s = length(sol.state_indices)
        n_c = length(sol.control_indices)
        _status("\n  State variables ($n_s): $(join([spec.varnames[i] for i in sol.state_indices], ", "))")
        _status("  Control variables ($n_c): $(join([spec.varnames[i] for i in sol.control_indices], ", "))")

        # gx is ny×nv with v = [states; shocks] (Stage-14 #368 layout, MEMs ≥0.7.2) —
        # labelling only the state columns was a DimensionMismatch on every model with
        # a shock, i.e. every model. Broken since the 0.7.2 bump; no per-branch T3.
        state_names = [spec.varnames[i] for i in sol.state_indices]
        shock_names = String.(spec.exog)
        nv = size(sol.gx, 2)
        col_names = nv == length(state_names) + length(shock_names) ?
            vcat(state_names, shock_names) :
            (nv == length(state_names) ? state_names : ["v$i" for i in 1:nv])
        gx_df = DataFrame(sol.gx, col_names)
        insertcols!(gx_df, 1, :control => [spec.varnames[i] for i in sol.control_indices])
        output_result(gx_df; format=Symbol(format), output=output,
                      title="Perturbation Policy (gx, order=$order)",
                      key="perturbation_policy_gx")
    elseif sol isa MacroEconometricModels.ProjectionSolution
        _status("\n  Grid type: $(sol.grid_type), Degree: $(sol.degree)")
        _status("  Converged: $(sol.converged), Iterations: $(sol.iterations)")
        _status_styled("  Residual norm: $(round(sol.residual_norm; sigdigits=4))\n";
                    color = sol.residual_norm < 1e-6 ? :green : :yellow)
        # `smolyak_levels` is a struct field on real MEMs ≥ 0.9.0 (0×0 on
        # tensor grids) and mirrored on the mock — read directly, no guard.
        output_kv(Pair{String,Any}[
            "converged" => sol.converged,
            "iterations" => sol.iterations,
            "residual_norm" => sol.residual_norm,
            "grid_type" => string(sol.grid_type),
            "degree" => sol.degree,
            "n_nodes" => size(sol.collocation_nodes, 1),
            "smolyak_blocks" => size(sol.smolyak_levels, 1),
            "method" => string(hasproperty(sol, :method) ? sol.method : method),
        ]; format=format, output=_per_var_output_path(output, "diagnostics"),
           title="Projection Diagnostics", key="projection_diagnostics")

        coef_df = DataFrame(sol.coefficients,
                           ["basis_$i" for i in 1:size(sol.coefficients, 2)])
        nrows = size(coef_df, 1)
        ctrls = length(sol.control_indices) == nrows ?
            [spec.varnames[i] for i in sol.control_indices] :
            [i <= length(spec.varnames) ? spec.varnames[i] : "row$i" for i in 1:nrows]
        insertcols!(coef_df, 1, :control => ctrls)
        title = (hasproperty(sol, :method) && sol.method === :vfi) ?
            "VFI Solution (degree=$(sol.degree), grid=$(sol.grid_type))" :
            "Projection Solution (degree=$(sol.degree), grid=$(sol.grid_type))"
        output_result(coef_df; format=Symbol(format), output=output,
                      title=title, key="projection_solution")
        if hasproperty(sol, :method) && sol.method === :vfi &&
           hasproperty(sol, :value_fn) && !isempty(sol.value_fn)
            nodes = hasproperty(sol, :collocation_nodes) ? sol.collocation_nodes :
                    zeros(eltype(sol.value_fn), size(sol.value_fn, 1), 0)
            n_nodes = size(sol.value_fn, 1)
            vfi_df = DataFrame(node = 1:n_nodes)
            nx = size(nodes, 2)
            for d in 1:nx
                sidx = d <= length(sol.state_indices) ? sol.state_indices[d] : d
                cname = sidx <= length(spec.varnames) ? spec.varnames[sidx] : "x$d"
                vfi_df[!, cname] = nodes[:, d]
            end
            vfi_df[!, :V] = vec(sol.value_fn)
            output_result(vfi_df; format=Symbol(format),
                          output=_per_var_output_path(output, "value"),
                          title="VFI Value Function", key="vfi_value_function")
            if hasproperty(sol, :value_coefficients) && !isempty(sol.value_coefficients)
                output_result(DataFrame(index=1:length(sol.value_coefficients),
                                        coefficient=Float64.(sol.value_coefficients));
                              format=Symbol(format),
                              output=_per_var_output_path(output, "value_coefficients"),
                              title="VFI Value Coefficients", key="vfi_value_coefficients")
            end
            if !isempty(strip(evaluate_at))
                xs = Float64[parse(Float64, strip(s)) for s in split(evaluate_at, ",") if !isempty(strip(s))]
                n_state = length(sol.state_indices)
                length(xs) == n_state || throw(CliError("usage/invalid",
                    "--evaluate-at needs $n_state state value(s) (got $(length(xs)))"))
                v = evaluate_value(sol, xs)
                output_kv(Pair{String,Any}["V" => Float64(v),
                                          "n_states" => n_state];
                          format=format, output=_per_var_output_path(output, "value_at"),
                          title="VFI Value at evaluate-at", key="vfi_value_at")
            end
        end
    end
    _status()
    return sol  # for --save-model (C029)
end

# ── W12/#114: determinacy region mapping ──────────────────────────────────
# `determinacy_region` re-specs the model and calls `solve` at every grid point, i.e. it
# EVALUATES the runtime-loaded spec's @dsge residual closures — so it goes through the
# `_dsge_call` world-age barrier like every other spec-evaluating path (the RA loader
# lesson). Failed grid points are recorded by upstream as code -2 and the sweep continues;
# they are surfaced here rather than swallowed, because a solve failure is missing
# information, not a fourth determinacy region.
function _dsge_determinacy_map(; model::String, config::String="", rank_rtol::Float64=1e-8,
                                threaded::Bool=false, verbose_solver::Bool=false,
                                output::String="", format::String="table",
                                plot::Bool=false, plot_save::String="")
    isempty(config) && throw(CliError("usage/missing",
        "dsge determinacy-map requires --config <toml> with a [determinacy] section";
        hint="params = [\"phi_pi\"], lower = [0.0], upper = [3.0], points = [61]"))
    rank_rtol > 0 || throw(CliError("usage/invalid",
        "dsge determinacy-map: --rank-rtol must be > 0 (got $rank_rtol)"))

    cfg = get_determinacy(load_config(config))
    spec = _load_dsge_model(model)

    # Reject unknown parameter names HERE: upstream raises a bare ArgumentError, and a typo
    # in a config file is user input (exit 4), not a model failure.
    known = Set(String.(spec.params))
    for p in cfg.params
        p in known || throw(CliError("config/invalid",
            "[determinacy] parameter '$p' is not a parameter of this model " *
            "(have $(join(sort(collect(known)), ", ")))"))
    end

    npts = prod(length.(cfg.grids))
    _status("Determinacy map: sweeping $(join(cfg.params, " × ")) over $npts grid point(s)")
    for (i, p) in enumerate(cfg.params)
        g = cfg.grids[i]
        _status("  $p: $(length(g)) points in [$(first(g)), $(last(g))]")
    end
    _status("  Solver: $(cfg.method), div=$(cfg.div), rank_rtol=$rank_rtol" *
            (threaded ? ", threaded" : ""))
    _status()

    dmap = try
        _dsge_call(determinacy_region, spec;
            params=Symbol[Symbol(p) for p in cfg.params],
            grids=cfg.grids, div=cfg.div, rank_rtol=rank_rtol,
            method=cfg.method, threaded=threaded, quiet=!verbose_solver)
    catch e
        e isa CliError && rethrow()
        e isa ArgumentError && throw(CliError("config/invalid",
            "determinacy map: $(sprint(showerror, e))"))
        throw(_domain_or_data_error(e, "determinacy region sweep"))
    end

    _maybe_plot(dmap; plot=plot, plot_save=plot_save)

    # Tidy long table: one row per grid cell. The raw Sims [existence, uniqueness] pair
    # rides along with the collapsed verdict — the pair is what distinguishes "no solution"
    # (e=0) from "indeterminate" (e=1, u=0), and an agent should not have to re-derive it.
    p1 = cfg.params[1]
    p2 = length(cfg.params) == 2 ? cfg.params[2] : ""
    n1, n2 = size(dmap.verdict)
    rows = NamedTuple[]
    for j in 1:n2, i in 1:n1
        code = dmap.verdict[i, j]
        push!(rows, (; Symbol(p1) => dmap.axes[1][i],
                     (isempty(p2) ? (;) : (; Symbol(p2) => dmap.axes[2][j]))...,
                     verdict = code,
                     label = determinacy_label(code),
                     existence = dmap.eu[i, j, 1],
                     uniqueness = dmap.eu[i, j, 2]))
    end
    output_result(DataFrame(rows); format=Symbol(format), output=output,
                  title="DSGE Determinacy Map ($(join(cfg.params, " × ")))",
                  key="dsge_determinacy_map")

    # Region counts. A `failed` count > 0 is reported explicitly rather than folded into
    # "indeterminate": it means the sweep has holes, not that the model is indeterminate there.
    counts = Dict{Int,Int}()
    for c in dmap.verdict
        counts[c] = get(counts, c, 0) + 1
    end
    total = length(dmap.verdict)
    pairs = Pair{String,Any}[
        "n_grid_points" => total,
        "n_determinate" => get(counts, DETERMINACY_CODES.determinate, 0),
        "n_indeterminate" => get(counts, DETERMINACY_CODES.indeterminate, 0),
        "n_no_solution" => get(counts, DETERMINACY_CODES.no_solution, 0),
        "n_failed" => get(counts, DETERMINACY_CODES.failed, 0),
        "method" => string(dmap.method),
        "div" => dmap.div,
    ]
    _status()
    output_kv(pairs; format=format, output=_per_var_output_path(output, "summary"),
              title="Determinacy Region Summary")

    n_failed = get(counts, DETERMINACY_CODES.failed, 0)
    n_failed > 0 && _status_styled(
        "  $n_failed of $total grid point(s) could not be solved and are recorded as " *
        "'failed' — holes in the sweep, NOT an indeterminacy region\n"; color=:yellow)

    # Boundary: one-parameter sweeps only (for two, the boundary is a curve — read the map).
    if length(cfg.params) == 1
        bnd = try
            determinacy_boundary(dmap)
        catch e
            throw(_domain_or_data_error(e, "determinacy boundary"))
        end
        _status()
        output_result(DataFrame(; boundary = collect(Float64.(bnd)));
            format=Symbol(format), output=_per_var_output_path(output, "boundary"),
            title="Determinacy Boundary ($p1)", key="determinacy_boundary")
        isempty(bnd) ?
            _status("  Verdict is constant across the grid — no boundary crossing.") :
            _status("  Boundary at $p1 ≈ $(join(round.(Float64.(bnd); digits=6), ", ")) " *
                    "(resolution = the grid spacing; refine the grid to sharpen it)")
    end
    return dmap
end

# ── W12/#114: closed-form theoretical moments at order 1/2/3 ───────────────
# Uses the exported `analytical_moments` with format=:gmm, which is a SUPERSET of the
# :covariance packing: means, then upper-triangle PRODUCT moments E[yᵢyⱼ], then diagonal
# E[yᵢ,ₜ·yᵢ,ₜ₋ₖ]. Central moments are recovered from it exactly, so one call yields means,
# variances/covariances and autocovariances rather than two calls with different layouts.
# The mean matters at order ≥ 2: the risk-adjusted mean differs from the steady state, and
# that difference is the point of solving at higher order.
function _dsge_moments(; model::String, method::String="perturbation", order::Int=2,
                        lags::Int=1, output::String="", format::String="table")
    (1 <= order <= 3) || throw(CliError("usage/invalid",
        "dsge moments: --order must be 1, 2 or 3 (got $order)"))
    # Order 1 was REFUSED from W12/#114 through v0.9.1: upstream's order-1 state↔control
    # covariance dropped the contemporaneous shock term (corr(state, control) reported as
    # rho where the truth is 1, control autocorrelations as rho^(k+2) instead of rho^k).
    # Fixed in MEMs 0.7.3 (#607) — `Sigma_xy = hx*Var(z)*gx' + eta_x*eta_y'` and the
    # `G1_equiv` control map now match the closed form — and re-enabled here (W9/#116 of
    # the v0.9.2 program). The order-1 closed-form AR(1) T3 case is the proof; --order 2
    # stays the default (for a linear model it reproduces first-order moments exactly).
    lags >= 1 || throw(CliError("usage/invalid",
        "dsge moments: --lags must be ≥ 1 (got $lags)"))

    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec; method=method, order=order)
    sol isa MacroEconometricModels.PerturbationSolution || throw(CliError("usage/invalid",
        "dsge moments needs a perturbation solution; --method $method produced a " *
        "$(nameof(typeof(sol)))";
        hint="use --method perturbation (the default), optionally with --order 2 or 3"))

    # Augmented specs report moments over the ORIGINAL variables only, so the labels must be
    # filtered the same way upstream filters the matrices — otherwise every moment is
    # attributed to the wrong variable. Replicated from public fields rather than calling
    # the private `_original_var_indices`.
    idx = spec.augmented ? Int[findfirst(==(v), spec.endog) for v in spec.original_endog] :
                           collect(1:spec.n_endog)
    any(isnothing, idx) && throw(CliError("model/error",
        "could not map the augmented model's original variables back to the solved set"))
    names = String[spec.varnames[i] for i in idx]
    k = length(names)

    _status("DSGE theoretical moments: order=$order, $k variable(s), $lags autocovariance lag(s)")
    order == 1 || _status("  Closed-form pruned-state-space moments (simulation-free)")
    _status()

    mv = try
        analytical_moments(sol; lags=lags, format=:gmm)
    catch e
        throw(_domain_or_data_error(e, "theoretical moments"))
    end
    expected = k + div(k * (k + 1), 2) + k * lags
    length(mv) == expected || throw(CliError("model/error",
        "moment vector has $(length(mv)) entries but the $k-variable × $lags-lag layout " *
        "implies $expected — the upstream packing changed"))

    E = Float64[mv[i] for i in 1:k]
    pos = k
    Var = zeros(Float64, k, k)
    for i in 1:k, j in i:k
        pos += 1
        # :gmm packs the PRODUCT moment E[yᵢyⱼ]; the central second moment is that minus E·E.
        v = mv[pos] - E[i] * E[j]
        Var[i, j] = v
        Var[j, i] = v
    end
    ss = Float64.(sol.steady_state)[idx]

    mean_df = DataFrame(
        variable = names,
        steady_state = round.(ss; digits=8),
        mean = round.(E; digits=8),
        # At order 1 this is identically zero; at order ≥ 2 it is the risk correction.
        mean_minus_ss = round.(E .- ss; digits=8),
        std_dev = round.(sqrt.(max.(diag(Var), 0.0)); digits=8),
    )
    output_result(mean_df; format=Symbol(format), output=output,
                  title="DSGE Theoretical Moments (order=$order)",
                  key="dsge_theoretical_moments")

    cov_rows = NamedTuple[]
    for i in 1:k, j in i:k
        sd = sqrt(max(Var[i, i], 0.0)) * sqrt(max(Var[j, j], 0.0))
        push!(cov_rows, (variable1 = names[i], variable2 = names[j],
                         covariance = round(Var[i, j]; digits=8),
                         correlation = sd > 0 ? round(Var[i, j] / sd; digits=8) : NaN))
    end
    output_result(DataFrame(cov_rows); format=Symbol(format),
                  output=_per_var_output_path(output, "covariance"),
                  title="Variance-Covariance (order=$order)", key="variance_covariance")

    ac_rows = NamedTuple[]
    for lag in 1:lags, i in 1:k
        pos += 1
        # Diagonal autocovariance, again de-centred out of the :gmm product moment.
        ac = mv[pos] - E[i]^2
        push!(ac_rows, (variable = names[i], lag = lag,
                        autocovariance = round(ac; digits=8),
                        autocorrelation = Var[i, i] > 0 ? round(ac / Var[i, i]; digits=8) : NaN))
    end
    output_result(DataFrame(ac_rows); format=Symbol(format),
                  output=_per_var_output_path(output, "autocovariance"),
                  title="Autocovariances (order=$order)", key="autocovariances")
    return sol
end

function _dsge_steady_state(; model::String, constraints::String="",
                             constraint_solver::String="",
                             output::String="", format::String="table")
    if !isempty(constraint_solver) && !(constraint_solver in ("nonlinearsolve", "optim", "nlopt", "ipopt", "path"))
        error("invalid --constraint-solver value '$constraint_solver'; must be one of: nonlinearsolve, optim, nlopt, ipopt, path")
    end

    spec = _load_dsge_model(model)

    solver_kw = isempty(constraint_solver) ? (;) : (; solver=Symbol(constraint_solver))
    if !isempty(constraints)
        cons = _load_dsge_constraints(constraints; spec=spec)
        spec = _dsge_call(compute_steady_state, spec; constraints=cons, solver_kw...)
    else
        spec = _dsge_call(compute_steady_state, spec; solver_kw...)
    end

    ss_df = DataFrame(
        variable = spec.varnames,
        steady_state = spec.steady_state
    )
    output_result(ss_df; format=Symbol(format), output=output,
                  title="DSGE Steady State")
end

function _dsge_simulate(; model::String, method::String="gensys", order::Int=1,
                         degree::Int=5, grid::String="auto",
                         next_state::String="", howard_steps::Int=-1,
                         n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                         scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                         damping::Float64=0.0, anderson_m::Int=0,
                         optimizer::String="", smolyak_mu::String="",
                         periods::Int=200, burn::Int=100,
                         antithetic::Bool=false, seed::Int=0,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                      next_state=next_state, howard_steps=howard_steps,
                      n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                      scale=scale, tol=tol, max_iter=max_iter,
                      damping=damping, anderson_m=anderson_m,
                      optimizer=optimizer, smolyak_mu=smolyak_mu)

    _status("Simulating $(periods + burn) periods (burn-in=$burn)...")

    if sol isa MacroEconometricModels.ProjectionSolution
        # Nonlinear-policy simulations take seed/rng/shock_draws only
        # (real simulate/simulation.jl:206) — no antithetic variates here.
        antithetic && _status("note: --antithetic ignored for " *
            "projection/pfi/vfi simulations (no antithetic draws on " *
            "nonlinear policies)")
        sim = seed > 0 ? simulate(sol, periods + burn; seed=seed) :
            simulate(sol, periods + burn)
    elseif seed > 0
        sim = simulate(sol, periods + burn; antithetic=antithetic, rng=Random.MersenneTwister(seed))
    else
        sim = simulate(sol, periods + burn; antithetic=antithetic)
    end

    # Drop burn-in
    sim_data = sim[burn+1:end, :]

    sim_df = DataFrame(sim_data, spec.varnames)
    insertcols!(sim_df, 1, :period => 1:periods)

    _maybe_plot(sim_df; plot=plot, plot_save=plot_save)

    output_result(sim_df; format=Symbol(format), output=output,
                  title="DSGE Simulation (method=$method, T=$periods)",
                  key="dsge_simulation")
end

# ── IRF / FEVD / Estimate / Perfect Foresight ──────────────────

function _dsge_irf(; model::String, method::String="gensys", order::Int=1,
                    degree::Int=5, grid::String="auto",
                    next_state::String="", howard_steps::Int=-1,
                    n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                    scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                    damping::Float64=0.0, anderson_m::Int=0,
                    optimizer::String="", smolyak_mu::String="",
                    horizon::Int=40, shock_size::Float64=1.0, n_sim::Int=500,
                    constraints::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="")
    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                      next_state=next_state, howard_steps=howard_steps,
                      n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                      scale=scale, tol=tol, max_iter=max_iter,
                      damping=damping, anderson_m=anderson_m,
                      optimizer=optimizer, smolyak_mu=smolyak_mu)

    if !isempty(constraints)
        _status("\nComputing OccBin IRF...")
        cons = _load_dsge_constraints(constraints; spec=spec)
        ob_irf = _occbin_irf_call(spec, cons; shock_idx=1, horizon=horizon, magnitude=shock_size)

        _maybe_plot(ob_irf; plot=plot, plot_save=plot_save)

        n_h = size(ob_irf.piecewise, 1)
        for (vi, vname) in enumerate(spec.varnames)
            vi > size(ob_irf.piecewise, 2) && break
            irf_df = DataFrame(
                horizon = 0:(n_h - 1),
                linear = ob_irf.linear[:, vi, 1],
                piecewise = ob_irf.piecewise[:, vi, 1],
            )
            output_result(irf_df; format=Symbol(format),
                          output=_per_var_output_path(output, vname),
                          title="OccBin IRF: $vname ← $(ob_irf.shock_name)",
                          key=_table_key("occbin_irf", vname))
        end
        return
    end

    _status("\nComputing IRF: horizon=$horizon, shock_size=$shock_size")
    irf_result = irf(sol, horizon; shock_size=shock_size, n_sim=n_sim)

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    irf_vals = irf_result.values
    n_h = size(irf_vals, 1)
    ne = nshocks(sol)
    for si in 1:ne
        shock_name = si <= spec.n_exog ? String(spec.exog[si]) : "shock_$si"
        irf_df = DataFrame()
        irf_df.horizon = 0:(n_h - 1)
        for (vi, vname) in enumerate(spec.varnames)
            vi > size(irf_vals, 2) && break
            si > size(irf_vals, 3) && break
            irf_df[!, vname] = irf_vals[:, vi, si]
        end
        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock_name),
                      title="DSGE IRF: shock=$shock_name (method=$method, h=$horizon)",
                      key=_table_key("dsge_irf", shock_name))
    end
end

function _dsge_fevd(; model::String, method::String="gensys", order::Int=1,
                     degree::Int=5, grid::String="auto",
                     next_state::String="", howard_steps::Int=-1,
                     n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                     scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                     damping::Float64=0.0, anderson_m::Int=0,
                     optimizer::String="", smolyak_mu::String="",
                     horizon::Int=40, unconditional::Bool=false,
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="")
    meth = _parse_ra_method(method)
    meth in (:projection, :pfi, :vfi) && throw(CliError("usage/invalid",
        "dsge fevd has no ProjectionSolution method; use gensys|klein|blanchard-kahn|perturbation"))
    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                      next_state=next_state, howard_steps=howard_steps,
                      n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                      scale=scale, tol=tol, max_iter=max_iter,
                      damping=damping, anderson_m=anderson_m,
                      optimizer=optimizer, smolyak_mu=smolyak_mu)

    if unconditional
        # MEMs: unconditional FEVD is only defined for PerturbationSolution with order ≥ 2
        if !(sol isa MacroEconometricModels.PerturbationSolution)
            throw(CliError("usage/invalid-option",
                "--unconditional requires --method=perturbation (got solution type $(typeof(sol).name.name))"))
        end
        if sol.order < 2
            throw(CliError("usage/invalid-option",
                "--unconditional requires --order ≥ 2 (got order=$(sol.order))"))
        end
        _status("\nComputing unconditional FEVD (order=$(sol.order) perturbation)")
        fevd_result = fevd(sol, horizon; unconditional=true)
    else
        _status("\nComputing FEVD: horizon=$horizon")
        fevd_result = fevd(sol, horizon)
    end

    _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)

    n_v = size(fevd_result.proportions, 1)
    ne = size(fevd_result.proportions, 2)
    n_h = size(fevd_result.proportions, 3)

    mode_label = unconditional ? "unconditional" : "method=$method, h=$horizon"
    for vi in 1:min(n_v, length(spec.varnames))
        vname = spec.varnames[vi]
        fevd_df = DataFrame()
        fevd_df.horizon = 1:n_h
        for si in 1:ne
            shock_name = si <= spec.n_exog ? String(spec.exog[si]) : "shock_$si"
            fevd_df[!, shock_name] = fevd_result.proportions[vi, si, :]
        end
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="DSGE FEVD: $vname ($mode_label)",
                      key=_table_key("dsge_fevd", vname))
    end
end

function _dsge_estimate(; model::String, data::String="", method::String="irf_matching",
                         params::String="", solve_method::String="gensys", solve_order::Int=1,
                         weighting::String="optimal",
                         irf_horizon::Int=20, var_lags::Int=4,
                         sim_ratio::Int=5, bounds::String="",
                         output::String="", format::String="table")
    isempty(data) && error("--data/-d is required for DSGE estimation")
    isempty(params) && error("--params is required (comma-separated parameter names)")

    spec = _load_dsge_model(model)
    Y, varnames = load_multivariate_data(data)
    param_names = [strip(s) for s in split(params, ",") if !isempty(strip(s))]

    isempty(param_names) && error("--params is required (comma-separated parameter names)")

    _status("Estimating DSGE model: method=$method, params=$(join(param_names, ", "))")
    _status("  Data: $(size(Y, 1)) obs × $(size(Y, 2)) vars")
    _status("  Solver: $solve_method, order=$solve_order")
    _status()

    est = try
        _dsge_call(estimate_dsge, spec, Y, param_names;
                            method=Symbol(method), solve_method=Symbol(solve_method),
                            solve_order=solve_order, weighting=Symbol(weighting),
                            irf_horizon=irf_horizon, var_lags=var_lags,
                            sim_ratio=sim_ratio)
    catch e
        throw(_domain_or_data_error(e, "dsge estimate"))
    end

    se = sqrt.(abs.(diag(est.vcov)))
    t_stats = est.theta ./ se
    p_vals = [2.0 * (1.0 - _normal_cdf(abs(t))) for t in t_stats]

    est_df = DataFrame(
        parameter = est.param_names,
        estimate = round.(est.theta; digits=6),
        std_error = round.(se; digits=6),
        t_stat = round.(t_stats; digits=4),
        p_value = round.(p_vals; digits=4),
    )
    output_result(est_df; format=Symbol(format), output=output,
                  title="DSGE Estimation ($method)", key="dsge_estimation")

    _status()
    _status_styled("  J-statistic: $(round(est.J_stat; digits=4))\n"; color=:cyan)
    _status_styled("  J p-value:   $(round(est.J_pvalue; digits=4))\n"; color=:cyan)
    _status_styled("  Converged:   $(est.converged)\n";
                color = est.converged ? :green : :red)
end

function _dsge_perfect_foresight(; model::String, shocks::String="",
                                  constraints::String="", constraint_solver::String="",
                                  periods::Int=100,
                                  sparsity::String="auto", max_iter::Int=100, tol::Float64=1e-8,
                                  output::String="", format::String="table",
                                  plot::Bool=false, plot_save::String="")
    if !isempty(constraint_solver) && !(constraint_solver in ("nonlinearsolve", "optim", "nlopt", "ipopt", "path"))
        error("invalid --constraint-solver value '$constraint_solver'; must be one of: nonlinearsolve, optim, nlopt, ipopt, path")
    end
    periods >= 1 || throw(CliError("usage/invalid",
        "perfect-foresight: --periods must be ≥ 1 (got $periods)"))
    sp = lowercase(strip(sparsity))
    sp in ("auto", "dense") || throw(CliError("usage/invalid",
        "perfect-foresight: --sparsity must be auto|dense (got '$sparsity')"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "perfect-foresight: --max-iter must be ≥ 1 (got $max_iter)"))
    tol > 0 || throw(CliError("usage/invalid",
        "perfect-foresight: --tol must be > 0 (got $tol)"))

    spec = _load_dsge_model(model)

    shock_mat = if isempty(shocks)
        nothing
    else
        shock_df = load_data(shocks)
        mat = df_to_matrix(shock_df)
        size(mat, 1) == periods || throw(CliError("data/shape",
            "perfect-foresight shock_path has $(size(mat, 1)) row(s) but --periods=$periods; " *
            "shock_path must have T_periods rows (one per transition period)"))
        size(mat, 2) == spec.n_exog || throw(CliError("data/shape",
            "perfect-foresight shock_path has $(size(mat, 2)) column(s) but the model has " *
            "$(spec.n_exog) exogenous shock(s)"))
        mat
    end

    _status("Computing perfect foresight transition path...")
    _status("  Shock periods: $(shock_mat === nothing ? periods : size(shock_mat, 1)), transition periods: $periods")
    _status()

    solver_kw = isempty(constraint_solver) ? (;) : (; solver=Symbol(constraint_solver))
    cons_kw = if !isempty(constraints)
        cons = _load_dsge_constraints(constraints; spec=spec)
        (; constraints=cons)
    else
        (;)
    end
    # Spec must have a steady state first (upstream ArgumentError otherwise).
    spec = _dsge_call(compute_steady_state, spec; solver_kw...)
    pf = _dsge_call(perfect_foresight, spec; shock_path=shock_mat, T_periods=periods,
                    sparsity=Symbol(sp), max_iter=max_iter, tol=tol,
                    solver_kw..., cons_kw...)

    _maybe_plot(pf; plot=plot, plot_save=plot_save)

    path_df = DataFrame()
    n_periods = size(pf.path, 1)
    path_df.period = 1:n_periods
    for (vi, vname) in enumerate(spec.varnames)
        if vi <= size(pf.path, 2)
            path_df[!, vname] = pf.path[:, vi]
        end
    end

    output_result(path_df; format=Symbol(format), output=output,
                  title="Perfect Foresight Path (T=$n_periods, converged=$(pf.converged))",
                  key="perfect_foresight_path")
end

# ── Bayesian DSGE Handlers ─────────────────────────────────────

"""
    _dsge_prior_distribution(name, spec) → Distribution

Map one `[priors.<name>]` TOML entry (`{dist, a, b}` from `get_dsge_priors`) to a
`Distributions.jl` object. The two numbers `a`, `b` are the distribution's
positional constructor arguments (MEMs/Dynare convention): `beta` → `Beta(a,b)`,
`normal` → `Normal(mean, sd)`, `inv_gamma` → `InverseGamma(a,b)`,
`gamma` → `Gamma(a,b)`, `uniform` → `Uniform(lo, hi)`.
"""
function _dsge_prior_distribution(name::AbstractString, spec)
    D = MacroEconometricModels.Distributions
    dist = lowercase(strip(string(spec["dist"])))
    a = Float64(spec["a"]); b = Float64(spec["b"])
    dist == "beta"                                    ? D.Beta(a, b) :
    dist in ("normal", "gaussian")                    ? D.Normal(a, b) :
    dist in ("inv_gamma", "inverse_gamma", "invgamma") ? D.InverseGamma(a, b) :
    dist == "gamma"                                   ? D.Gamma(a, b) :
    dist == "uniform"                                 ? D.Uniform(a, b) :
    throw(CliError("config/bad-prior",
        "unknown prior dist '$(spec["dist"])' for parameter '$name'; " *
        "use one of beta|normal|inv_gamma|gamma|uniform",
        hint="see the [priors] TOML reference"))
end

"""
    _dsge_priors_distributions(priors_config) → Dict{Symbol,<:Distribution}

Bridge the `[priors]` TOML to the `Dict{Symbol,<:Distribution}` that MEMs
`estimate_dsge_bayes` requires (both RA and HA `ModelSpec`
methods). `get_dsge_priors` yields `{name => {dist,a,b}}`; each entry becomes a
concrete distribution via [`_dsge_prior_distribution`].
"""
function _dsge_priors_distributions(priors_config::Dict)
    raw = get_dsge_priors(priors_config)  # throws config/missing-key if no [priors]
    return Dict(Symbol(name) => _dsge_prior_distribution(name, spec)
                for (name, spec) in raw)
end

"""Gather the shared inputs for every Bayesian DSGE leaf: the spec, the data matrix,
`theta0` as a name→value Dict, the bridged priors, observables and solver kwargs. Split
out of `_dsge_bayes_run_estimation` so `dsge bayes posterior-mode`/`prior-predictive`
(#78) get IDENTICAL input handling rather than a second, drifting copy.

`require_data=false` serves `prior-predictive`, which needs no data at all — it draws
from the prior, so demanding `--data` would be a false requirement."""
function _dsge_bayes_inputs(; model::String, data::String, params::String,
        priors::String, observables::String, solver::String, order::Int,
        constraint_solver::String="", require_data::Bool=true)
    require_data && isempty(data) &&
        throw(CliError("usage/missing", "--data is required (path to CSV data file)"))
    isempty(params) && throw(CliError("usage/missing", "--params is required (comma-separated parameter names)"))
    isempty(priors) && throw(CliError("usage/missing", "--priors is required (path to priors TOML)"))

    if !isempty(constraint_solver) && !(constraint_solver in ("nonlinearsolve", "optim", "nlopt", "ipopt", "path"))
        throw(CliError("usage/invalid",
            "invalid --constraint-solver value '$constraint_solver'; must be one of: nonlinearsolve, optim, nlopt, ipopt, path"))
    end

    spec = _load_dsge_model(model)

    Y = isempty(data) ? zeros(0, 0) : df_to_matrix(load_data(data))

    param_names = [strip(p) for p in split(params, ",")]
    # theta0 as a name→value Dict (MEMs #136 / C054): the by-name path resolves
    # start values against the (internally sorted) prior keys, so it is immune to
    # silent alphabetical reordering AND validates that --params matches the
    # priors' parameter set. A bare positional vector would only be length-checked.
    theta0 = Dict(Symbol(p) => 0.5 for p in param_names)

    priors_config = load_config(priors)
    # Bridge {dist,a,b} TOML → Dict{Symbol,<:Distribution} (MEMs requires distribution
    # objects, not the raw config dict — C048; previously passed the wrong type).
    priors_dict = _dsge_priors_distributions(priors_config)

    obs_syms = isempty(observables) ? Symbol[] : Symbol.(strip.(split(observables, ",")))

    solver_kwargs = order > 1 ? (order=order,) : NamedTuple()

    return (spec=spec, Y=Y, theta0=theta0, priors_dict=priors_dict,
            obs_syms=obs_syms, solver_kwargs=solver_kwargs, param_names=param_names)
end

"""Shared helper: run Bayesian DSGE estimation and return the result."""
function _dsge_bayes_run_estimation(; model::String, data::String, params::String,
        priors::String, sampler::String, n_smc::Int, n_particles::Int,
        n_draws::Int, burnin::Int, ess_target::Float64, observables::String,
        solver::String, order::Int, delayed_acceptance::Bool,
        constraint_solver::String="",
        prefilter::String="none", hp_lambda::Float64=1600.0,
        measurement_error::String="none")
    inp = _dsge_bayes_inputs(; model, data, params, priors, observables, solver, order,
                             constraint_solver)
    spec, Y, theta0 = inp.spec, inp.Y, inp.theta0
    priors_dict, obs_syms, solver_kwargs = inp.priors_dict, inp.obs_syms, inp.solver_kwargs
    param_names = inp.param_names

    _status("Bayesian DSGE Estimation:")
    _status("  Sampler: $sampler")
    _status("  Parameters: $(join(param_names, ", "))")
    _status("  Data: $(size(Y, 1)) obs × $(size(Y, 2)) vars")
    _status("  Solver: $solver" * (order > 1 ? ", order=$order" : ""))
    _status()

    # W12/#114 (MEMs#339): observables with trends. `estimate_dsge_bayes` is the ONLY
    # upstream estimation entry point that takes `prefilter` — the frequentist
    # `estimate_dsge` has no such kwarg, and `posterior_mode` takes `trends` but not
    # `prefilter`, so the option is declared on the Bayesian leaves only (#85: never declare
    # an option the handler cannot feed through). It rides on the SHARED runner because
    # every `dsge bayes` leaf re-estimates from scratch — a prefilter that only worked on
    # `bayes estimate` could not be carried into `bayes irf`/`fevd`/`hd`.
    prefilter in ("none", "demean", "first-difference", "linear-detrend", "hp") ||
        throw(CliError("usage/invalid",
            "dsge bayes: --prefilter must be none|demean|first-difference|linear-detrend|hp, got '$prefilter'"))
    (prefilter == "hp" || hp_lambda == 1600.0) || throw(CliError("usage/invalid",
        "dsge bayes: --hp-lambda applies only to --prefilter hp (got --prefilter $prefilter)"))
    hp_lambda > 0 || throw(CliError("usage/invalid",
        "dsge bayes: --hp-lambda must be > 0 (got $hp_lambda)"))
    pf = Symbol(replace(prefilter, '-' => '_'))
    pf === :none || _status("  Prefilter: $prefilter" *
        (pf === :hp ? " (lambda=$hp_lambda)" : ""))

    # #148: measurement error (none|auto|comma-separated SDs, one per observable).
    # Upstream default is NONE — with n_obs > n_shocks the likelihood is then
    # stochastically singular (model/stochastic-singularity), and this option is
    # the in-CLI remedy the error hint points at.
    me = if measurement_error in ("", "none")
        nothing
    elseif measurement_error == "auto"
        :auto
    else
        try
            [parse(Float64, strip(v)) for v in split(measurement_error, ",")]
        catch
            throw(CliError("usage/invalid-option",
                "dsge bayes: --measurement-error must be none|auto|comma-separated numbers, " *
                "got '$measurement_error'"))
        end
    end
    # Guard the vector form up front (usage/invalid) — upstream raises an untyped
    # ArgumentError on a length mismatch, which would surface as internal/error.
    me isa Vector && length(me) != size(Y, 2) && throw(CliError("usage/invalid",
        "dsge bayes: --measurement-error has $(length(me)) value(s) but there are " *
        "$(size(Y, 2)) observables — give one SD per observable, or use auto"))
    me === nothing || _status("  Measurement error: $measurement_error")

    # --constraint-solver threads into solve via solver_kwargs.solver (the constrained
    # steady-state solver), the same kwarg the frequentist path uses. There is no
    # `solver_obj=` upstream — passing it is a MethodError/ArgumentError on every call.
    if !isempty(constraint_solver)
        solver_kwargs = merge(solver_kwargs, (solver=Symbol(constraint_solver),))
    end
    # World-age barrier: estimate_dsge_bayes re-solves the spec (evaluating its @dsge
    # residual fns) on every posterior draw — must run at the latest world age.
    result = _dsge_call(estimate_dsge_bayes, spec, Y, theta0;
        priors=priors_dict, method=Symbol(sampler),
        observables=obs_syms,
        n_smc=n_smc, n_particles=n_particles,
        n_draws=n_draws, burnin=burnin, ess_target=ess_target,
        solver=Symbol(solver), solver_kwargs=solver_kwargs,
        delayed_acceptance=delayed_acceptance,
        measurement_error=me,
        prefilter=pf, hp_lambda=hp_lambda,
        _fwd_seed()...)

    return result
end

function _dsge_bayes_estimate(; model::String, data::String="", params::String="",
                               priors::String="", sampler::String="smc",
                               n_smc::Int=5000, n_particles::Int=500,
                               n_draws::Int=10000, burnin::Int=5000,
                               ess_target::Float64=0.5, observables::String="",
                               solver::String="gensys", order::Int=1,
                               constraint_solver::String="",
                               delayed_acceptance::Bool=false,
                               prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                               output::String="", format::String="table")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    # Posterior summary table
    draws = result.theta_draws
    np = size(draws, 2)
    est_df = DataFrame(
        parameter = result.param_names,
        mean = [round(mean(draws[:, i]); digits=6) for i in 1:np],
        std = [round(sqrt(var(draws[:, i])); digits=6) for i in 1:np],
        q05 = [round(quantile(draws[:, i], 0.05); digits=6) for i in 1:np],
        median = [round(median(draws[:, i]); digits=6) for i in 1:np],
        q95 = [round(quantile(draws[:, i], 0.95); digits=6) for i in 1:np],
    )
    output_result(est_df; format=Symbol(format), output=output,
                  title="Bayesian DSGE Posterior ($sampler)", key="bayesian_dsge_posterior")

    _status()
    _status_styled("  Log marginal likelihood: $(round(result.log_marginal_likelihood; digits=4))\n"; color=:cyan)
    _status_styled("  Acceptance rate: $(round(result.acceptance_rate; digits=4))\n"; color=:cyan)
    _status_styled("  Method: $(result.method)\n"; color=:cyan)
end

function _dsge_bayes_irf(; model::String, data::String="", params::String="",
                          priors::String="", sampler::String="smc",
                          n_smc::Int=5000, n_particles::Int=500,
                          n_draws::Int=10000, burnin::Int=5000,
                          ess_target::Float64=0.5, observables::String="",
                          solver::String="gensys", order::Int=1,
                          constraint_solver::String="",
                          delayed_acceptance::Bool=false,
                          prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                          horizon::Int=40,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    solver_kwargs = order > 1 ? (order=order,) : NamedTuple()

    _status("Computing Bayesian DSGE IRF: horizon=$horizon")
    # re-solves per draw → world-age barrier (see _dsge_call)
    irf_result = _dsge_call(irf, result, horizon; n_draws=n_draws,
        solver=Symbol(solver), solver_kwargs=solver_kwargs)

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    n_h = size(irf_result.point_estimate, 1)
    ns = size(irf_result.point_estimate, 3)
    varnames = irf_result.variables
    for si in 1:ns
        shock_name = si <= length(irf_result.shocks) ? irf_result.shocks[si] : "shock_$si"
        irf_df = DataFrame()
        irf_df.horizon = 0:(n_h - 1)
        for (vi, vname) in enumerate(varnames)
            vi > size(irf_result.point_estimate, 2) && break
            irf_df[!, vname] = irf_result.point_estimate[:, vi, si]
        end
        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock_name),
                      title="Bayesian DSGE IRF: shock=$shock_name ($sampler, h=$horizon)",
                      key=_table_key("bayesian_dsge_irf", shock_name))
    end
end

function _dsge_bayes_fevd(; model::String, data::String="", params::String="",
                           priors::String="", sampler::String="smc",
                           n_smc::Int=5000, n_particles::Int=500,
                           n_draws::Int=10000, burnin::Int=5000,
                           ess_target::Float64=0.5, observables::String="",
                           solver::String="gensys", order::Int=1,
                           constraint_solver::String="",
                           delayed_acceptance::Bool=false,
                           horizon::Int=40,
                           prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                           output::String="", format::String="table",
                           plot::Bool=false, plot_save::String="")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    solver_kwargs = order > 1 ? (order=order,) : NamedTuple()

    _status("Computing Bayesian DSGE FEVD: horizon=$horizon")
    # re-solves per draw → world-age barrier (see _dsge_call)
    fevd_result = _dsge_call(fevd, result, horizon; n_draws=n_draws,
        solver=Symbol(solver), solver_kwargs=solver_kwargs)

    _maybe_plot(fevd_result; plot=plot, plot_save=plot_save)

    n_v = size(fevd_result.point_estimate, 2)
    ns = size(fevd_result.point_estimate, 3)
    n_h = size(fevd_result.point_estimate, 1)
    varnames = fevd_result.variables
    for vi in 1:min(n_v, length(varnames))
        vname = varnames[vi]
        fevd_df = DataFrame()
        fevd_df.horizon = 1:n_h
        for si in 1:ns
            shock_name = si <= length(fevd_result.shocks) ? fevd_result.shocks[si] : "shock_$si"
            fevd_df[!, shock_name] = fevd_result.point_estimate[:, vi, si]
        end
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="Bayesian DSGE FEVD: $vname ($sampler, h=$horizon)",
                      key=_table_key("bayesian_dsge_fevd", vname))
    end
end

function _dsge_bayes_simulate(; model::String, data::String="", params::String="",
                               priors::String="", sampler::String="smc",
                               n_smc::Int=5000, n_particles::Int=500,
                               n_draws::Int=10000, burnin::Int=5000,
                               ess_target::Float64=0.5, observables::String="",
                               solver::String="gensys", order::Int=1,
                               constraint_solver::String="",
                               delayed_acceptance::Bool=false,
                               prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                               periods::Int=200,
                               output::String="", format::String="table",
                               plot::Bool=false, plot_save::String="")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    solver_kwargs = order > 1 ? (order=order,) : NamedTuple()

    _status("Simulating from Bayesian DSGE posterior: T=$periods")
    # re-solves per draw → world-age barrier (see _dsge_call)
    sim = _dsge_call(simulate, result, periods; n_draws=n_draws,
        solver=Symbol(solver), solver_kwargs=solver_kwargs)

    _maybe_plot(sim; plot=plot, plot_save=plot_save)

    varnames = sim.variables
    sim_df = DataFrame()
    sim_df.period = 1:periods
    for (vi, vname) in enumerate(varnames)
        vi > size(sim.point_estimate, 2) && break
        sim_df[!, vname] = sim.point_estimate[:, vi]
    end
    output_result(sim_df; format=Symbol(format), output=output,
                  title="Bayesian DSGE Simulation ($sampler, T=$periods)",
                  key="bayesian_dsge_simulation")
end

function _dsge_bayes_summary(; model::String, data::String="", params::String="",
                              priors::String="", sampler::String="smc",
                              n_smc::Int=5000, n_particles::Int=500,
                              n_draws::Int=10000, burnin::Int=5000,
                              ess_target::Float64=0.5, observables::String="",
                              solver::String="gensys", order::Int=1,
                              constraint_solver::String="",
                              delayed_acceptance::Bool=false,
                              prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                              output::String="", format::String="table")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    summary = posterior_summary(result)
    pp_table = prior_posterior_table(result)

    # Posterior summary table
    pnames = result.param_names
    sum_df = DataFrame(
        parameter = pnames,
        mean = [round(summary[p][:mean]; digits=6) for p in pnames],
        median = [round(summary[p][:median]; digits=6) for p in pnames],
        std = [round(summary[p][:std]; digits=6) for p in pnames],
        q05 = [round(summary[p][:q05]; digits=6) for p in pnames],
        q95 = [round(summary[p][:q95]; digits=6) for p in pnames],
    )
    output_result(sum_df; format=Symbol(format), output=output,
                  title="Bayesian DSGE Posterior Summary ($sampler)",
                  key="bayesian_dsge_posterior_summary")

    # Prior-posterior comparison
    pp_df = DataFrame(
        parameter = [r.param for r in pp_table],
        prior_mean = [round(r.prior_mean; digits=6) for r in pp_table],
        prior_std = [round(r.prior_std; digits=6) for r in pp_table],
        post_mean = [round(r.post_mean; digits=6) for r in pp_table],
        post_std = [round(r.post_std; digits=6) for r in pp_table],
        post_q05 = [round(r.post_q05; digits=6) for r in pp_table],
        post_q95 = [round(r.post_q95; digits=6) for r in pp_table],
    )
    output_result(pp_df; format=Symbol(format),
                  output=_per_var_output_path(output, "prior_posterior"),
                  title="Prior vs Posterior Comparison")

    _status()
    _status_styled("  Log marginal likelihood: $(round(result.log_marginal_likelihood; digits=4))\n"; color=:cyan)
    _status_styled("  Acceptance rate: $(round(result.acceptance_rate; digits=4))\n"; color=:cyan)
end

function _dsge_bayes_compare(; model::String, data::String="", params::String="",
                              priors::String="", sampler::String="smc",
                              n_smc::Int=5000, n_particles::Int=500,
                              n_draws::Int=10000, burnin::Int=5000,
                              ess_target::Float64=0.5, observables::String="",
                              solver::String="gensys", order::Int=1,
                              constraint_solver::String="",
                              delayed_acceptance::Bool=false,
                              model2::String="", params2::String="", priors2::String="",
                              prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                              output::String="", format::String="table")
    isempty(model2) && error("--model2 is required for model comparison")
    isempty(params2) && error("--params2 is required for model comparison")
    isempty(priors2) && error("--priors2 is required for model comparison")

    _status("Estimating Model 1...")
    r1 = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Estimating Model 2...")
    r2 = _dsge_bayes_run_estimation(; model=model2, data, params=params2,
        priors=priors2, sampler, n_smc, n_particles, n_draws, burnin,
        ess_target, observables, solver, order, delayed_acceptance, constraint_solver)

    # MEMs `bayes_factor` returns the LOG Bayes factor: log BF₁₂ = logML₁ − logML₂
    # (positive favors Model 1). Do NOT take log() of it again. `bf = exp(log_bf)` may
    # overflow to Inf under strong evidence — that is fine to display.
    log_bf = bayes_factor(r1, r2)
    bf = exp(log_bf)

    comp_df = DataFrame(
        model = ["Model 1", "Model 2"],
        log_marginal_likelihood = [round(r1.log_marginal_likelihood; digits=4),
                                   round(r2.log_marginal_likelihood; digits=4)],
        acceptance_rate = [round(r1.acceptance_rate; digits=4),
                          round(r2.acceptance_rate; digits=4)],
    )
    output_result(comp_df; format=Symbol(format), output=output,
                  title="Bayesian Model Comparison")

    _status()
    _status_styled("  Bayes factor (M1 vs M2): $(round(bf; digits=4))\n"; color=:cyan)
    _status_styled("  Log Bayes factor: $(round(log_bf; digits=4))\n"; color=:cyan)
    # Kass & Raftery (1995): 2·log BF > 6 is strong evidence for Model 1.
    if log_bf > 0
        _status_styled("  Evidence favors Model 1\n"; color=:green)
    else
        _status_styled("  Evidence favors Model 2\n"; color=:yellow)
    end
end

function _dsge_bayes_predictive(; model::String, data::String="", params::String="",
                                 priors::String="", sampler::String="smc",
                                 n_smc::Int=5000, n_particles::Int=500,
                                 n_draws::Int=10000, burnin::Int=5000,
                                 ess_target::Float64=0.5, observables::String="",
                                 solver::String="gensys", order::Int=1,
                                 constraint_solver::String="",
                                 delayed_acceptance::Bool=false,
                                 n_sim::Int=500, periods::Int=100,
                                 prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                                 output::String="", format::String="table",
                                 plot::Bool=false, plot_save::String="")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Generating posterior predictive simulations: n=$n_sim, T=$periods")
    # re-solves per draw → world-age barrier (see _dsge_call)
    pp = _dsge_call(posterior_predictive, result, n_sim; T_periods=periods, _fwd_seed()...)

    _maybe_plot(pp; plot=plot, plot_save=plot_save)

    # Summary statistics across simulations
    nv = size(pp, 3)
    varnames = result.param_names
    pp_df = DataFrame(
        variable = varnames[1:min(nv, length(varnames))],
        mean = [round(mean(pp[:, :, vi]); digits=6) for vi in 1:min(nv, length(varnames))],
        std = [round(sqrt(var(vec(pp[:, :, vi]))); digits=6) for vi in 1:min(nv, length(varnames))],
        min = [round(minimum(pp[:, :, vi]); digits=6) for vi in 1:min(nv, length(varnames))],
        max = [round(maximum(pp[:, :, vi]); digits=6) for vi in 1:min(nv, length(varnames))],
    )
    output_result(pp_df; format=Symbol(format), output=output,
                  title="Posterior Predictive Summary ($sampler, n=$n_sim, T=$periods)",
                  key="posterior_predictive_summary")
end

function _dsge_hd(; model::String, method::String="gensys", order::Int=1,
                   degree::Int=5, grid::String="auto",
                   next_state::String="", howard_steps::Int=-1,
                   n_grid::Int=0, n_choice::Int=0, n_quad::Int=0,
                   scale::Float64=0.0, tol::Float64=0.0, max_iter::Int=0,
                   damping::Float64=0.0, anderson_m::Int=0,
                   optimizer::String="", smolyak_mu::String="",
                   data::String="", observables::String="",
                   states::String="observables",
                   measurement_error::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="")
    isempty(data) && error("--data is required for DSGE historical decomposition")
    isempty(observables) && error("--observables is required (comma-separated variable names)")
    meth = _parse_ra_method(method)
    meth in (:projection, :pfi, :vfi) && throw(CliError("usage/invalid",
        "dsge hd has no ProjectionSolution method; use gensys|klein|blanchard-kahn|perturbation"))

    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec; method=method, order=order, degree=degree, grid=grid,
                      next_state=next_state, howard_steps=howard_steps,
                      n_grid=n_grid, n_choice=n_choice, n_quad=n_quad,
                      scale=scale, tol=tol, max_iter=max_iter,
                      damping=damping, anderson_m=anderson_m,
                      optimizer=optimizer, smolyak_mu=smolyak_mu)

    df = load_data(data)
    Y = df_to_matrix(df)
    obs_syms = Symbol[Symbol(strip(s)) for s in split(observables, ",")]

    _status("DSGE Historical Decomposition")
    _status("  Model: $model")
    _status("  Observations: $(size(Y, 1)), Observable variables: $(length(obs_syms))")
    _status("  States: $states")
    _status()

    me = if isempty(measurement_error)
        nothing
    elseif measurement_error == "auto"
        :auto
    else
        [parse(Float64, strip(s)) for s in split(measurement_error, ",")]
    end

    hd = historical_decomposition(sol, Y, obs_syms;
        states=Symbol(states), measurement_error=me)

    ok = verify_decomposition(hd)
    ok && _status_styled("  Decomposition verified\n"; color=:green)

    for (si, sname) in enumerate(hd.shock_names)
        contrib = hd.contributions[:, :, si]
        contrib_df = DataFrame(contrib, hd.variables)
        insertcols!(contrib_df, 1, :t => 1:hd.T_eff)
        output_result(contrib_df; format=Symbol(format),
            output=_per_var_output_path(output, string(sname)),
            title="Shock: $sname contributions",
            key=_table_key("dsge_historical_decomposition", sname))
    end

    _maybe_plot(hd; plot=plot, plot_save=plot_save)
    return hd
end

function _dsge_bayes_hd(; model::String, data::String="", params::String="",
                         priors::String="", observables::String="",
                         sampler::String="smc", n_smc::Int=5000,
                         n_particles::Int=500,
                         n_draws::Int=10000, burnin::Int=5000,
                         ess_target::Float64=0.5,
                         solver::String="gensys", order::Int=1,
                         constraint_solver::String="",
                         n_hd_draws::Int=200, quantiles::String="0.16,0.5,0.84",
                         mode_only::Bool=false,
                         delayed_acceptance::Bool=false,
                         horizon::Int=40,
                         prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    isempty(observables) && error("--observables is required (comma-separated variable names)")

    bd = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    df = load_data(data)
    Y = df_to_matrix(df)
    obs_syms = Symbol[Symbol(strip(s)) for s in split(observables, ",")]
    q_levels = [parse(Float64, strip(s)) for s in split(quantiles, ",")]

    _status("Historical Decomposition from Bayesian DSGE posterior")
    _status()

    # re-solves per draw → world-age barrier (see _dsge_call)
    hd = _dsge_call(historical_decomposition, bd, Y, obs_syms;
        mode_only=mode_only, n_draws=n_hd_draws, quantiles=q_levels)

    for (si, sname) in enumerate(hd.shock_names)
        pe = hd.point_estimate[:, :, si]
        pe_df = DataFrame(pe, hd.variables)
        insertcols!(pe_df, 1, :t => 1:hd.T_eff)
        output_result(pe_df; format=Symbol(format),
            output=_per_var_output_path(output, string(sname)),
            title="Shock: $sname (posterior mean)",
            key=_table_key("bayesian_dsge_historical_decomposition", sname))
    end

    _maybe_plot(hd; plot=plot, plot_save=plot_save)
    return hd
end

# ── Bayesian DSGE diagnostics (C073) ────────────────────────
# Convergence + identification diagnostics on a fitted Bayesian DSGE posterior.
# All share _dsge_bayes_run_estimation to fit, then route the (world-age-sensitive)
# MEMs diagnostic through _dsge_call. Hand-built tables — none are Tables.jl types.

function _dsge_bayes_mcmc_diag(; model::String, data::String="", params::String="",
                                priors::String="", sampler::String="smc",
                                n_smc::Int=5000, n_particles::Int=500,
                                n_draws::Int=10000, burnin::Int=5000,
                                ess_target::Float64=0.5, observables::String="",
                                solver::String="gensys", order::Int=1,
                                constraint_solver::String="",
                                delayed_acceptance::Bool=false,
                                prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                                output::String="", format::String="table")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Computing MCMC convergence diagnostics (R-hat / ESS / Geweke)")
    # Pure post-processing of theta_draws, but wrapped for world-age consistency.
    d = _dsge_call(mcmc_diagnostics, result)

    diag_df = DataFrame(
        parameter = string.(d.param_names),
        rhat = round.(d.rhat; digits=4),
        ess_bulk = round.(d.ess_bulk; digits=1),
        ess_tail = round.(d.ess_tail; digits=1),
        geweke_z = round.(d.geweke_z; digits=4),
        geweke_p = round.(d.geweke_p; digits=4),   # may be NaN — _json_safe sanitizes both paths
    )
    output_result(diag_df; format=Symbol(format), output=output,
                  title="MCMC Convergence Diagnostics")

    output_kv(Pair{String,Any}[
        "n_draws" => d.n_draws,
        "method"  => string(d.method),
    ]; format=format, output=_per_var_output_path(output, "summary"),
       title="MCMC Diagnostics Summary")
    return nothing
end

"""Map an untyped `identification_diagnostics` failure to a typed CliError — these are NOT
`MacroModelError`s, so without this they reach run_cli's exit-1 tail. Bad `--params` name →
`KeyError`; bad `--observables` / a model that fails to solve or has a non-finite moment
Jacobian at θ → `ArgumentError`."""
function _identification_error(e)
    e isa CliError && return e
    if e isa KeyError
        return CliError("usage/invalid",
            "unknown --params name $(repr(e.key)): not a calibrated parameter of the model")
    elseif e isa ArgumentError
        msg = sprint(showerror, e)
        if occursin("observable", msg)
            return CliError("usage/invalid", "identification: $msg")
        elseif occursin("solve", msg) || occursin("non-finite", msg) || occursin("Jacobian", msg)
            return CliError("model/error", "identification: $msg";
                hint="the model does not solve, or has a singular/non-finite moment Jacobian, at the calibrated θ")
        end
        return CliError("data/invalid", "identification: $msg")
    elseif e isa DomainError
        return CliError("data/invalid", "identification: $(sprint(showerror, e))")
    end
    return CliError("model/error", "identification diagnostics failed: $(sprint(showerror, e))")
end

function _dsge_bayes_identification(; model::String, params::String="",
                                     observables::String="", solver::String="gensys",
                                     order::Int=1, n_lags::Int=2,
                                     output::String="", format::String="table")
    isempty(params) && throw(CliError("usage/missing",
        "--params is required (comma-separated estimated parameter names)"))
    spec = _load_dsge_model(model)
    param_syms = Symbol.(filter(!isempty, strip.(split(params, ","))))
    isempty(param_syms) && throw(CliError("usage/missing",
        "--params is required (comma-separated estimated parameter names)"))
    obs_syms = isempty(observables) ? Symbol[] : Symbol.(strip.(split(observables, ",")))
    solver_kwargs = order > 1 ? (order=order,) : NamedTuple()

    _status("Iskrev (2010) local-identification rank test: params=" * join(string.(param_syms), ", "))
    # Evaluates the runtime-loaded spec's residual fns (steady state + solve) →
    # world-age barrier (see _dsge_call). This is the ONLY bayes leaf that feeds raw user
    # --params/--observables straight to a MEMs call (the others go through
    # _dsge_bayes_run_estimation first), and identification_diagnostics throws UNTYPED
    # KeyError (bad --params name → spec.param_values[p]) / ArgumentError (bad --observables,
    # or a model that fails to solve / has a non-finite moment Jacobian at θ) — none are
    # MacroModelError, so they'd fall through run_cli to exit-1. Map them (adversarial review).
    idr = try
        _dsge_call(identification_diagnostics, spec, param_syms;
            observables=obs_syms, n_lags=n_lags, solver=Symbol(solver),
            solver_kwargs=solver_kwargs)
    catch e
        throw(_identification_error(e))
    end

    output_kv(Pair{String,Any}[
        "rank"       => idr.rank,
        "n_params"   => idr.n_params,
        "n_moments"  => idr.n_moments,
        "n_lags"     => idr.n_lags,
        "identified" => idr.identified,
        "tol"        => round(idr.tol; sigdigits=6),
    ]; format=format, output=output, title="Identification Diagnostics")

    sv_df = DataFrame(
        index = collect(1:length(idr.singular_values)),
        singular_value = round.(idr.singular_values; sigdigits=6),
    )
    output_result(sv_df; format=Symbol(format),
                  output=_per_var_output_path(output, "singular_values"),
                  title="Singular Values")
    return nothing
end

function _dsge_bayes_learning_rate(; model::String, data::String="", params::String="",
                                    priors::String="", sampler::String="smc",
                                    n_smc::Int=5000, n_particles::Int=500,
                                    n_draws::Int=10000, burnin::Int=5000,
                                    ess_target::Float64=0.5, observables::String="",
                                    solver::String="gensys", order::Int=1,
                                    constraint_solver::String="",
                                    delayed_acceptance::Bool=false,
                                    fractions::String="0.5,1.0", threshold::Float64=0.2,
                                    refit_n_smc::Int=100,
                                    prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                                    output::String="", format::String="table")
    frac_vec = try
        Float64[parse(Float64, strip(s)) for s in split(fractions, ",") if !isempty(strip(s))]
    catch
        throw(CliError("usage/invalid",
            "--fractions must be comma-separated numbers in (0,1], got '$fractions'"))
    end
    (length(frac_vec) >= 2 && all(f -> 0 < f <= 1, frac_vec)) || throw(CliError("usage/invalid",
        "--fractions must be at least two values in (0,1], got '$fractions'"))

    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Koop-Pesaran-Smith learning-rate check (refit n_smc=$refit_n_smc)")
    # Re-runs SMC on nested subsamples (evaluates the spec) → world-age barrier.
    lr = _dsge_call(learning_rate_check, result; fractions=frac_vec,
        n_smc=refit_n_smc, threshold=threshold)

    lr_df = DataFrame(
        parameter = string.(lr.param_names),
        learning_rate = round.(lr.learning_rate; digits=4),
        flagged = lr.flagged,
    )
    output_result(lr_df; format=Symbol(format), output=output,
                  title="Learning-Rate Check")

    output_kv(Pair{String,Any}[
        "threshold"    => threshold,
        "sample_sizes" => string(lr.sample_sizes),
    ]; format=format, output=_per_var_output_path(output, "summary"),
       title="Learning-Rate Summary")
    return nothing
end

function _dsge_bayes_overlap(; model::String, data::String="", params::String="",
                              priors::String="", sampler::String="smc",
                              n_smc::Int=5000, n_particles::Int=500,
                              n_draws::Int=10000, burnin::Int=5000,
                              ess_target::Float64=0.5, observables::String="",
                              solver::String="gensys", order::Int=1,
                              constraint_solver::String="",
                              delayed_acceptance::Bool=false,
                              threshold::Float64=0.8, n_grid::Int=0,
                              prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                              output::String="", format::String="table")
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Prior/posterior overlap (weak-identification signal)")
    ov = _dsge_call(prior_posterior_overlap, result; n_grid=n_grid, threshold=threshold)

    ov_df = DataFrame(
        parameter = string.(ov.param_names),
        overlap = round.(ov.overlap; digits=4),
        flagged = ov.flagged,
    )
    output_result(ov_df; format=Symbol(format), output=output,
                  title="Prior-Posterior Overlap")

    output_kv(Pair{String,Any}[
        "threshold" => threshold,
    ]; format=format, output=_per_var_output_path(output, "summary"),
       title="Overlap Summary")
    return nothing
end

# C073 remainder (#78): posterior mode + prior predictive.
#
# Both go through the shared `_dsge_bayes_inputs`, so they see the same theta0-as-Dict,
# priors bridging and observable handling as the sampler leaves. Both evaluate the
# runtime-loaded @dsge spec's residual fns, so both route through `_dsge_call` (the
# world-age barrier).
#
# NOTE `posterior_mode` reports the LAPLACE log marginal likelihood as a field
# (`laplace_log_ml`). There is deliberately no `--ml bridge` here: bridge sampling needs
# posterior DRAWS, which a mode-finder does not produce — `dsge bayes marginal-lik` is
# the leaf for that.
function _dsge_bayes_posterior_mode(; model::String, data::String="", params::String="",
        priors::String="", observables::String="", solver::String="gensys", order::Int=1,
        max_iter::Int=500, f_reltol::Float64=1e-8, constraint_solver::String="",
        output::String="", format::String="table")
    max_iter >= 1 || throw(CliError("usage/invalid",
        "dsge bayes posterior-mode: --max-iter must be ≥ 1 (got $max_iter)"))
    f_reltol > 0 || throw(CliError("usage/invalid",
        "dsge bayes posterior-mode: --f-reltol must be > 0 (got $f_reltol)"))
    inp = _dsge_bayes_inputs(; model, data, params, priors, observables, solver, order,
                             constraint_solver)
    _status("Bayesian DSGE Posterior Mode:")
    _status("  Parameters: $(join(inp.param_names, ", "))")
    _status("  Data: $(size(inp.Y, 1)) obs × $(size(inp.Y, 2)) vars")
    _status()
    res = try
        _dsge_call(posterior_mode, inp.spec, inp.Y, inp.theta0;
            priors=inp.priors_dict, observables=inp.obs_syms,
            solver=Symbol(solver), solver_kwargs=inp.solver_kwargs,
            f_reltol=f_reltol, max_iter=max_iter)
    catch e
        throw(_identification_error(e))
    end
    # Mode + the Laplace standard errors implied by the inverse Hessian.
    md = Float64.(collect(res.mode))
    se = [sqrt(abs(Float64(res.inv_hessian[i, i]))) for i in 1:length(md)]
    output_result(DataFrame(
            parameter = String.(res.param_names),
            mode = round.(md; digits=6),
            std_error = round.(se; digits=6));
        format=Symbol(format), output=output, title="Posterior Mode")
    output_kv(Pair{String,Any}[
        "log posterior" => round(Float64(res.log_posterior); digits=6),
        "log likelihood" => round(Float64(res.log_likelihood); digits=6),
        "Laplace log ML" => round(Float64(res.laplace_log_ml); digits=6),
        "converged" => res.converged,
        "iterations" => res.n_iterations];
        format=format, output=_per_var_output_path(output, "diagnostics"),
        title="Posterior Mode Diagnostics")
    res.converged || _status_styled("-> optimizer did NOT converge — treat the mode and its Laplace ML with caution\n"; color=:yellow)
    return res
end

# prior_predictive draws from the PRIOR, so it needs NO data — requiring --data would be
# a false requirement (hence require_data=false).
function _dsge_bayes_prior_predictive(; model::String, params::String="",
        priors::String="", observables::String="", solver::String="gensys", order::Int=1,
        n_draws::Int=500, periods::Int=200, constraint_solver::String="",
        output::String="", format::String="table")
    n_draws >= 1 || throw(CliError("usage/invalid",
        "dsge bayes prior-predictive: --n-draws must be ≥ 1 (got $n_draws)"))
    periods >= 1 || throw(CliError("usage/invalid",
        "dsge bayes prior-predictive: --periods must be ≥ 1 (got $periods)"))
    inp = _dsge_bayes_inputs(; model, data="", params, priors, observables, solver, order,
                             constraint_solver, require_data=false)
    _status("Bayesian DSGE Prior Predictive: draws=$n_draws, periods=$periods")
    _status()
    res = try
        _dsge_call(prior_predictive, inp.spec, inp.priors_dict;
            n_draws=n_draws, T_periods=periods, observables=inp.obs_syms,
            solver=Symbol(solver), solver_kwargs=inp.solver_kwargs,
            _fwd_seed()...)
    catch e
        throw(_identification_error(e))
    end
    # `stats` is (n_effective × n_stats): summarise each statistic across the draws that
    # actually solved, which is the point of a prior predictive check.
    S = Float64.(res.stats)
    nstat = size(S, 2)
    output_result(DataFrame(
            statistic = String.(res.stat_names),
            mean = [round(mean(view(S, :, j)); digits=6) for j in 1:nstat],
            std = [round(sqrt(var(view(S, :, j))); digits=6) for j in 1:nstat],
            q05 = [round(quantile(view(S, :, j), 0.05); digits=6) for j in 1:nstat],
            median = [round(median(view(S, :, j)); digits=6) for j in 1:nstat],
            q95 = [round(quantile(view(S, :, j), 0.95); digits=6) for j in 1:nstat]);
        format=Symbol(format), output=output, title="Prior Predictive Distribution")
    output_kv(Pair{String,Any}[
        "draws requested" => res.n_draws,
        "draws that solved" => res.n_effective,
        "periods simulated" => res.T_periods];
        format=format, output=_per_var_output_path(output, "summary"),
        title="Prior Predictive Summary")
    res.n_effective < res.n_draws && _status_styled(
        "-> $(res.n_draws - res.n_effective) prior draw(s) failed to solve and were dropped\n"; color=:yellow)
    return res
end

function _dsge_bayes_marginal_lik(; model::String, data::String="", params::String="",
                                   priors::String="", sampler::String="smc",
                                   n_smc::Int=5000, n_particles::Int=500,
                                   n_draws::Int=10000, burnin::Int=5000,
                                   ess_target::Float64=0.5, observables::String="",
                                   solver::String="gensys", order::Int=1,
                                   constraint_solver::String="",
                                   delayed_acceptance::Bool=false,
                                   proposal::String="normal", df::Float64=5.0,
                                   prefilter::String="none", hp_lambda::Float64=1600.0,
                               measurement_error::String="none",
                                   output::String="", format::String="table")
    proposal in ("normal", "t") || throw(CliError("usage/invalid",
        "--proposal must be normal|t, got '$proposal'"))
    result = _dsge_bayes_run_estimation(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance, constraint_solver, prefilter, hp_lambda, measurement_error)

    _status("Bridge-sampling marginal likelihood (proposal=$proposal)")
    # Re-evaluates the likelihood over the spec → world-age barrier. Returns a scalar
    # log-ML (NaN on a too-short/diffuse chain — handled gracefully).
    logml = _dsge_call(bridge_sampling_ml, result; proposal=Symbol(proposal), df=df)
    logml_smc = result.log_marginal_likelihood

    output_kv(Pair{String,Any}[
        "log_marginal_likelihood_bridge" => (isfinite(logml) ? round(logml; digits=4) : string(logml)),
        "log_marginal_likelihood_smc"    => (isfinite(logml_smc) ? round(logml_smc; digits=4) : string(logml_smc)),
        "proposal"                       => proposal,
        "df"                             => df,
    ]; format=format, output=output, title="Marginal Likelihood (Bridge Sampling)")
    return nothing
end

# ── HA-DSGE handlers (C040) ─────────────────────────────────

function _ha_ss_tables(ss; format::String, output::String, title_prefix::String="HA")
    agg = ss.aggregates
    prices = ss.prices
    if agg isa AbstractDict
        agg_df = DataFrame(
            name = String[string(k) for k in keys(agg)],
            value = [Float64(v) for v in values(agg)],
        )
    else
        agg_df = DataFrame(name=String[], value=Float64[])
    end
    if prices isa AbstractDict
        price_df = DataFrame(
            name = String[string(k) for k in keys(prices)],
            value = [Float64(v) for v in values(prices)],
        )
    else
        price_df = DataFrame(name=String[], value=Float64[])
    end
    diag_df = DataFrame(
        metric = ["converged", "iterations", "euler_error", "excess_demand"],
        value = [
            Float64(ss.converged ? 1.0 : 0.0),
            Float64(ss.iterations),
            Float64(ss.euler_error),
            Float64(ss.excess_demand),
        ],
    )
    # W3/#138: the envelope keys are fixed here rather than derived from the (prefixed)
    # titles — every caller passes title_prefix="HA", but a key must not be hostage to a
    # display string.
    output_result(agg_df; format=Symbol(format), output=output,
                  title="$title_prefix Steady-State Aggregates",
                  key="ha_steady_state_aggregates")
    # Distinct paths for the 2nd+ table, or `--output f.csv` silently keeps only the last.
    output_result(price_df; format=Symbol(format), output=_per_var_output_path(output, "prices"),
                  title="$title_prefix Steady-State Prices",
                  key="ha_steady_state_prices")
    output_result(diag_df; format=Symbol(format), output=_per_var_output_path(output, "diagnostics"),
                  title="$title_prefix Steady-State Diagnostics",
                  key="ha_steady_state_diagnostics")

    # Euler accuracy detail (MEMs#508). `ss.euler` carries BOTH conventions —
    # `(midpoints=…, nodes=…)`, each `(points, max, mean, n_evaluated, n_constrained,
    # n_offgrid)`. `euler_error` above is whichever one `--euler-points` selected, and the
    # two differ by 2.5–3.8 log10 units, so reporting only the selected number invites a
    # comparison against published figures measured the other way. `nothing` for steady
    # states built by paths that do not measure accuracy — then this table is absent.
    eu = hasproperty(ss, :euler) ? ss.euler : nothing
    if eu !== nothing
        rows = NamedTuple[]
        for conv in (:midpoints, :nodes)
            haskey(eu, conv) || continue
            s = eu[conv]
            push!(rows, (convention=String(conv),
                         max=Float64(s.max), mean=Float64(s.mean),
                         n_evaluated=Int(s.n_evaluated),
                         n_constrained=Int(s.n_constrained),
                         n_offgrid=Int(s.n_offgrid)))
        end
        if !isempty(rows)
            n_dims = try
                g = ss.grid
                g !== nothing && hasproperty(g, :n_dims) ? Int(g.n_dims) : 1
            catch
                1
            end
            euler_label = n_dims == 2 ? "liquid Euler" : "Euler"
            output_result(DataFrame(rows);
                          format=Symbol(format),
                          output=_per_var_output_path(output, "euler"),
                          title="$title_prefix $euler_label Accuracy (log10, by convention)",
                          key="ha_euler_accuracy_log10_by_convention")
        end
    end
    return (agg_df, price_df, diag_df)
end

function _dsge_ha_steady_state(; model::String, euler_points::String="midpoints",
                                hh_solver::String="egm", distribution::String="young",
                                max_iter::Int=0, tol::Float64=0.0,
                                output::String="", format::String="table")
    ep = lowercase(strip(euler_points))
    ep in ("midpoints", "nodes") || throw(CliError("usage/invalid-option",
        "invalid --euler-points '$euler_points'; must be midpoints or nodes"))
    max_iter >= 0 || throw(CliError("usage/invalid", "--max-iter must be ≥ 0"))
    tol >= 0 || throw(CliError("usage/invalid", "--tol must be ≥ 0"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    _status("Computing HA steady state for model=$(_ha_model_symbol(spec))...")
    extra = (;)
    max_iter > 0 && (extra = (; extra..., max_iter=max_iter))
    tol > 0 && (extra = (; extra..., tol=tol))
    ss = _dsge_call(compute_steady_state, spec; euler_points=Symbol(ep), hh_solver=hh, extra...)
    _ha_ss_tables(ss; format=format, output=output)
    return ss
end

# ── W13/#115: Den Haan (2010) accuracy for a Krusell-Smith solution ───────────
#
# The audit's strongest row and the one it required resolved concretely.
#
# `den_haan_test` has TWO methods: `KrusellSmithSolution` (the fitted PLM) and
# `HADSGESolution` for the linearized `:ssj`/`:reiter` solutions, which recover the implied
# law by regression over `T_fit` periods. Both are exposed via --method; the linearized ones
# additionally take --t-fit. The two are NOT comparable — upstream measures 0.07% for the
# fitted KS PLM against 12.2% (ssj) and 5.5% (reiter) on the same model — so the renderer
# labels which solution produced the number.
#
# Upstream's own guards are `@assert`s (T_sim/T_burn, T_fit, the z-augmented PLM) and bare
# `error()`s (:huggett, wrong method) — all UNTYPED, i.e. exit 1 — so each is either
# pre-guarded here or mapped. `DenHaanAccuracy` has a real plot recipe
# (plotting/ha_dynamics.jl), so the plot flags are genuinely backed.
function _dsge_ha_accuracy(; model::String, method::String="krusell-smith",
                            n_reduced::Int=30,
                            t_sim::Int=10000, t_burn::Int=1000, t_fit::Int=4000,
                            rho_z::Float64=0.95, sigma_z::Float64=0.007,
                            seed::Int=98765,
                            hh_solver::String="egm", distribution::String="young",
                            plot::Bool=false, plot_save::String="",
                            output::String="", format::String="table")
    meth = _parse_ha_method(method)
    hh = _parse_hh_solver(hh_solver)
    t_sim > t_burn + 10 || throw(CliError("usage/invalid",
        "dsge ha accuracy: --t-sim must exceed --t-burn by at least 10 " *
        "(got t_sim=$t_sim, t_burn=$t_burn)"))
    t_burn >= 0 || throw(CliError("usage/invalid",
        "dsge ha accuracy: --t-burn must be ≥ 0 (got $t_burn)"))
    sigma_z > 0 || throw(CliError("usage/invalid",
        "dsge ha accuracy: --sigma-z must be > 0 (got $sigma_z)"))
    abs(rho_z) < 1 || throw(CliError("usage/invalid",
        "dsge ha accuracy: --rho-z must satisfy |rho| < 1 (got $rho_z)"))
    if meth !== :krusell_smith
        t_fit > 100 || throw(CliError("usage/invalid",
            "dsge ha accuracy: --t-fit must be > 100 to fit the implied law of motion " *
            "(got $t_fit)"))
    end

    spec = _load_ha_model(model; distribution=distribution)
    # Refuse BEFORE the (expensive) solve. Upstream only errors once den_haan_test is
    # reached, so without this a user asking for an undefined combination waits through a
    # full solve just to be told no.
    _ha_model_symbol(spec) === :huggett && throw(CliError(
        "model/unsupported",
        "Den Haan accuracy is undefined for :huggett — it scores the aggregate CAPITAL " *
        "law of motion, and the Huggett clearing rate is driven by the wealth distribution " *
        "rather than the aggregate shock alone";
        hint="use krusell-smith or one-asset-hank (or an :aiyagari-family .jl spec)"))
    sol = _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh)

    dh_kwargs = meth === :krusell_smith ?
        (; T_sim=t_sim, T_burn=t_burn, rho_z=rho_z, sigma_z=sigma_z, seed=seed) :
        (; T_sim=t_sim, T_burn=t_burn, T_fit=t_fit, rho_z=rho_z, sigma_z=sigma_z, seed=seed)
    acc = try
        MacroEconometricModels.den_haan_test(sol; dh_kwargs...)
    catch e
        e isa CliError && rethrow()
        throw(CliError("model/unsupported",
            "Den Haan accuracy is not available for this model/solution combination " *
            "(model='$model', method=$meth)";
            hint=sprint(showerror, e)))
    end

    _status("Den Haan accuracy: aggregate=$(acc.aggregate), T_sim=$(acc.T_sim), " *
            "T_burn=$(acc.T_burn)"); _status()
    output_result(DataFrame(
        metric=["dh_max", "dh_mean", "sigma_ref", "sigma_plm"],
        value=round.(Float64[acc.dh_max, acc.dh_mean, acc.sigma_ref, acc.sigma_plm]; digits=6),
    ); format=Symbol(format), output=output,
       title="Den Haan Accuracy (% deviation, $(acc.aggregate), $(meth))",
       key="den_haan_accuracy")
    # The two simulated aggregate paths, tidy and long, under a distinct output path.
    n = min(length(acc.ref_path), length(acc.plm_path))
    output_result(DataFrame(t=1:n,
                            reference=round.(Float64.(acc.ref_path[1:n]); digits=6),
                            plm_only=round.(Float64.(acc.plm_path[1:n]); digits=6));
                  format=Symbol(format), output=_per_var_output_path(output, "paths"),
                  title="Reference vs PLM-only Aggregate Path")
    settings = Pair{String,Any}[
        "method"    => String(meth),
        "aggregate" => String(acc.aggregate),
        "source"    => String(acc.source),
        "T_sim"     => acc.T_sim,
        "T_burn"    => acc.T_burn,
        "seed"      => seed,
    ]
    meth === :krusell_smith || push!(settings, "T_fit" => t_fit)
    output_kv(settings; format=format,
              output=_per_var_output_path(output, "settings"),
              title="Den Haan Simulation Settings")
    _maybe_plot(acc; plot=plot, plot_save=plot_save)
    return acc
end

function _dsge_ha_solve(; model::String, method::String="ssj",
                         n_reduced::Int=30, t_horizon::Int=300,
                         hh_solver::String="egm", distribution::String="young",
                         output::String="", format::String="table")
    meth = _parse_ha_method(method)
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _solve_ha(spec; method=meth, n_reduced=n_reduced, T_horizon=t_horizon,
                    hh_solver=hh)

    if sol isa MacroEconometricModels.KrusellSmithSolution
        r2 = sol.r_squared
        r2_keys = sort!(collect(keys(r2)))
        r2_note = join(["$k=$(round(Float64(r2[k]); digits=4))" for k in r2_keys], ", ")
        _status_styled("  Krusell–Smith PLM R² $r2_note, " *
                       "converged=$(sol.converged), iterations=$(sol.iterations)\n";
                       color = sol.converged ? :green : :yellow)
        plm = sol.plm_coefficients
        plm_rows = NamedTuple{(:aggregate, :coefficient, :value), Tuple{String,String,Float64}}[]
        for k in sort!(collect(keys(plm)))
            coefs = plm[k]
            for (i, v) in enumerate(coefs)
                push!(plm_rows, (aggregate=String(k), coefficient="b$i", value=Float64(v)))
            end
        end
        plm_df = DataFrame(plm_rows)
        diag_df = DataFrame(
            aggregate = String.(r2_keys),
            r_squared = [round(Float64(r2[k]); digits=4) for k in r2_keys],
            converged = fill(sol.converged, length(r2_keys)),
            iterations = fill(sol.iterations, length(r2_keys)),
        )
        output_result(diag_df; format=Symbol(format),
                      output=_per_var_output_path(output, "solve_diagnostics"),
                      title="HA-DSGE Solve Diagnostics (krusell-smith)",
                      key="ha_dsge_solve_diagnostics")
        output_result(plm_df; format=Symbol(format),
                      output=_per_var_output_path(output, "plm"),
                      title="Krusell–Smith PLM Coefficients")
        _ha_ss_tables(sol.steady_state; format=format, output=output, title_prefix="HA")
        return sol
    end

    # HADSGESolution
    _status("  Method: $(sol.method), reduced states: $(sol.n_reduced)/$(sol.n_full_states)")
    if hasproperty(sol, :explained_variance)
        _status("  Explained variance: $(round(Float64(sol.explained_variance); digits=4))")
    end
    diag_df = DataFrame(
        metric = ["method", "n_full_states", "n_reduced", "explained_variance",
                  "obs_rows", "obs_cols"],
        value = [
            string(sol.method),
            string(sol.n_full_states),
            string(sol.n_reduced),
            string(sol.explained_variance),
            string(size(sol.C_obs, 1)),
            string(size(sol.C_obs, 2)),
        ],
    )
    output_result(diag_df; format=Symbol(format),
                  output=_per_var_output_path(output, "solve_diagnostics"),
                  title="HA-DSGE Solve Diagnostics (method=$(sol.method))",
                  key="ha_dsge_solve_diagnostics")
    _ha_ss_tables(sol.steady_state; format=format, output=output, title_prefix="HA")
    return sol
end

function _dsge_ha_require_linear(sol, meth::Symbol)
    sol isa MacroEconometricModels.HADSGESolution || throw(CliError("model/unsupported",
        "method=$meth does not produce a linearized HADSGESolution " *
        "(got $(typeof(sol))); use --method=ssj or --method=reiter"))
    return sol
end

function _dsge_ha_irf(; model::String, method::String="reiter",
                       horizon::Int=40, n_reduced::Int=30,
                       hh_solver::String="egm", distribution::String="young",
                       output::String="", format::String="table",
                       plot::Bool=false, plot_save::String="")
    meth = _parse_ha_method(method)
    meth === :krusell_smith && throw(CliError("usage/invalid-option",
        "HA IRF requires ssj or reiter (krusell-smith returns a PLM, not linear IRFs)"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh), meth)

    _status("Computing HA IRF: horizon=$horizon, method=$meth")
    ir = MacroEconometricModels.irf(sol, horizon)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)

    vals = ir.values
    n_h, n_v, n_s = size(vals)
    for si in 1:n_s
        shock = si <= length(ir.shocks) ? String(ir.shocks[si]) : "shock_$si"
        irf_df = DataFrame(horizon = 0:(n_h - 1))
        for vi in 1:n_v
            vname = vi <= length(ir.variables) ? String(ir.variables[vi]) : "y$vi"
            irf_df[!, vname] = vals[:, vi, si]
        end
        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock),
                      title="HA-DSGE IRF: shock=$shock (method=$meth, h=$horizon)",
                      key=_table_key("ha_dsge_irf", shock))
    end
    return ir
end

function _dsge_ha_fevd(; model::String, method::String="reiter",
                        horizon::Int=40, n_reduced::Int=30,
                        hh_solver::String="egm", distribution::String="young",
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    meth = _parse_ha_method(method)
    meth === :krusell_smith && throw(CliError("usage/invalid-option",
        "HA FEVD requires ssj or reiter"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh), meth)

    _status("Computing HA FEVD: horizon=$horizon, method=$meth")
    fv = MacroEconometricModels.fevd(sol, horizon)
    _maybe_plot(fv; plot=plot, plot_save=plot_save)

    props = fv.proportions
    n_v, n_s, n_h = size(props)
    varnames = hasproperty(fv, :variables) ? fv.variables : ["y$i" for i in 1:n_v]
    shocks = hasproperty(fv, :shocks) ? fv.shocks : ["shock_$i" for i in 1:n_s]
    for vi in 1:n_v
        vname = vi <= length(varnames) ? String(varnames[vi]) : "y$vi"
        fevd_df = DataFrame(horizon = 1:n_h)
        for si in 1:n_s
            sname = si <= length(shocks) ? String(shocks[si]) : "shock_$si"
            fevd_df[!, sname] = props[vi, si, :]
        end
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="HA-DSGE FEVD: $vname (method=$meth, h=$horizon)",
                      key=_table_key("ha_dsge_fevd", vname))
    end
    return fv
end

function _dsge_ha_simulate(; model::String, method::String="reiter",
                            periods::Int=200, seed::Int=0, n_reduced::Int=30,
                            hh_solver::String="egm", distribution::String="young",
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    meth = _parse_ha_method(method)
    meth === :krusell_smith && throw(CliError("usage/invalid-option",
        "HA simulate requires ssj or reiter"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh), meth)

    _status("Simulating HA aggregates: T=$periods, method=$meth")
    if seed > 0
        path = MacroEconometricModels.simulate(sol, periods; rng=Random.MersenneTwister(seed))
    else
        path = MacroEconometricModels.simulate(sol, periods)
    end

    n_out = size(path, 2)
    colnames = ["y$i" for i in 1:n_out]
    try
        ir0 = MacroEconometricModels.irf(sol, 1)
        if length(ir0.variables) == n_out
            colnames = String[string(v) for v in ir0.variables]
        end
    catch
    end
    sim_df = DataFrame(path, colnames)
    insertcols!(sim_df, 1, :period => 1:periods)
    _maybe_plot(sim_df; plot=plot, plot_save=plot_save)
    output_result(sim_df; format=Symbol(format), output=output,
                  title="HA-DSGE Simulation (method=$meth, T=$periods)",
                  key="ha_dsge_simulation")
    return path
end

function _dsge_ha_distribution_irf(; model::String, method::String="reiter",
                                    horizon::Int=40, shock_index::Int=1,
                                    shock_size::Float64=1.0, n_reduced::Int=30,
                                    hh_solver::String="egm", distribution::String="young",
                                    output::String="", format::String="table")
    meth = _parse_ha_method(method)
    meth === :reiter || throw(CliError("usage/invalid-option",
        "distribution-irf requires --method=reiter (SSJ has no distribution basis)"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh), meth)

    _status("Computing distribution IRF: h=$horizon, shock=$shock_index, size=$shock_size")
    d = MacroEconometricModels.distribution_irf(sol, horizon;
            shock_index=shock_index, shock_size=shock_size)
    # Summarize 3D (n_a × n_e × h) as horizon moments
    n_a, n_e, n_h = size(d)
    mean_dev = [sum(abs, d[:, :, h]) for h in 1:n_h]
    max_dev = [maximum(abs, d[:, :, h]) for h in 1:n_h]
    df = DataFrame(
        horizon = 0:(n_h - 1),
        l1_mass_deviation = mean_dev,
        max_abs_deviation = max_dev,
        n_asset_bins = fill(n_a, n_h),
        n_income = fill(n_e, n_h),
    )
    output_result(df; format=Symbol(format), output=output,
                  title="HA Distribution IRF (shock=$shock_index, h=$horizon)",
                  key="ha_distribution_irf")
    return d
end

function _dsge_ha_inequality_irf(; model::String, method::String="reiter",
                                  horizon::Int=40, shock_index::Int=1,
                                  shock_size::Float64=1.0, n_reduced::Int=30,
                                  hh_solver::String="egm", distribution::String="young",
                                  output::String="", format::String="table",
                                  plot::Bool=false, plot_save::String="")
    meth = _parse_ha_method(method)
    meth === :reiter || throw(CliError("usage/invalid-option",
        "inequality-irf requires --method=reiter"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, hh_solver=hh), meth)

    _status("Computing inequality IRF: h=$horizon, shock=$shock_index")
    d = MacroEconometricModels.inequality_irf(sol, horizon;
            shock_index=shock_index, shock_size=shock_size)
    df = DataFrame(
        horizon = 0:(horizon - 1),
        gini = d[:gini],
        p10 = d[:p10],
        p25 = d[:p25],
        p50 = d[:p50],
        p75 = d[:p75],
        p90 = d[:p90],
    )
    _maybe_plot(df; plot=plot, plot_save=plot_save)
    output_result(df; format=Symbol(format), output=output,
                  title="HA Inequality IRF (shock=$shock_index, h=$horizon)",
                  key="ha_inequality_irf")
    return d
end

function _dsge_ha_simulate_panel(; model::String,
                                  n_agents::Int=1000, periods::Int=100, seed::Int=0,
                                  hh_solver::String="egm", distribution::String="young",
                                  output::String="", format::String="table")
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    _status("Computing HA steady state for panel simulation...")
    ss = _dsge_call(compute_steady_state, spec; hh_solver=hh)
    _status("Simulating panel: N=$n_agents, T=$periods")
    if seed > 0
        panel = MacroEconometricModels.simulate_panel(ss;
            N_agents=n_agents, T_periods=periods, rng=Random.MersenneTwister(seed))
    else
        panel = MacroEconometricModels.simulate_panel(ss;
            N_agents=n_agents, T_periods=periods)
    end
    # Summary path: cross-sectional mean assets over time (full N×T is huge)
    mean_a = [mean(panel[:, t]) for t in 1:size(panel, 2)]
    # Sample std via sqrt(var); avoids depending on Statistics.std in bare includes
    sd_a = [sqrt(var(panel[:, t])) for t in 1:size(panel, 2)]
    df = DataFrame(
        period = 1:size(panel, 2),
        mean_assets = mean_a,
        sd_assets = sd_a,
        n_agents = fill(size(panel, 1), size(panel, 2)),
    )
    output_result(df; format=Symbol(format), output=output,
                  title="HA Panel Simulation Summary (N=$n_agents, T=$periods)",
                  key="ha_panel_simulation_summary")
    return panel
end

# HA Bayesian estimation (C048; un-deferred after MEMs#228). RWMH re-solves the full
# HA model at every draw (Auclert-Bardóczy-Rognlie-Straub 2021 "offline" approach), so
# runs are intentionally small by default relative to RA SMC.
function _dsge_ha_estimate(; model::String, data::String="", priors::String="",
                            observables::String="", method::String="ssj",
                            sampler::String="mh",
                            n_draws::Int=2000, burnin::Int=500,
                            n_smc::Int=500, n_mh_steps::Int=1, ess_target::Float64=0.5,
                            t_horizon::Int=300, n_reduced::Int=15,
                            proposal_scale::Float64=0.01, adapt_interval::Int=100,
                            measurement_error::String="none", seed::Int=0,
                            hh_solver::String="egm", distribution::String="young",
                            output::String="", format::String="table")
    isempty(data) && throw(CliError("usage/missing-option",
        "--data is required (path to observed aggregates CSV)"))
    isempty(priors) && throw(CliError("usage/missing-option",
        "--priors is required (path to priors TOML with a [priors] section)"))
    meth = _parse_ha_method(method)
    meth === :krusell_smith && throw(CliError("usage/invalid-option",
        "HA Bayesian estimation requires --method=ssj or reiter " *
        "(krusell-smith yields a PLM, not a linear state space for the Kalman filter)"))
    samp = lowercase(strip(sampler))
    samp in ("mh", "smc") || throw(CliError("usage/invalid-option",
        "invalid --sampler '$sampler'; must be mh|smc"))
    n_smc >= 1 || throw(CliError("usage/invalid", "--n-smc must be ≥ 1 (got $n_smc)"))
    n_mh_steps >= 1 || throw(CliError("usage/invalid", "--n-mh-steps must be ≥ 1 (got $n_mh_steps)"))
    0 < ess_target <= 1 || throw(CliError("usage/invalid",
        "--ess-target must be in (0, 1] (got $ess_target)"))
    me = measurement_error == "auto" ? :auto :
         (measurement_error in ("none", "") ? nothing :
          throw(CliError("usage/invalid-option", "--measurement-error must be none|auto")))

    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    df = load_data(data)
    Y = df_to_matrix(df)

    priors_config = load_config(priors)
    priors_dist = _dsge_priors_distributions(priors_config)
    param_names = sort!(collect(keys(priors_dist)))          # match DSGEPrior sorted order
    theta0 = Float64[mean(priors_dist[pn]) for pn in param_names]
    obs_syms = isempty(observables) ? Symbol[] :
               Symbol.(strip.(split(observables, ",")))

    sampler_label = samp == "smc" ? "SMC" : "RWMH"
    _status("HA-DSGE Bayesian Estimation ($sampler_label):")
    _status("  Model: $(_ha_model_symbol(spec)), method: $meth, sampler: $samp")
    _status("  Parameters: $(join(String.(param_names), ", "))")
    _status("  Observables: " * (isempty(obs_syms) ? "(default aggregates)" :
                                 join(String.(obs_syms), ", ")))
    _status("  Data: $(size(Y, 1)) obs × $(size(Y, 2)) vars; draws=$n_draws, burnin=$burnin")
    _status()

    rng = seed > 0 ? Random.MersenneTwister(seed) : Random.default_rng()
    # MEMs#769: seed= owns the RNG and records the manifest (seed wins over rng;
    # upstream builds Xoshiro(seed), so the MersenneTwister above is shadowed
    # whenever a seed is in play). Leaf --seed wins, else global.
    eff_seed = seed > 0 ? seed : _SEED[]
    result = _dsge_call(estimate_dsge_bayes, spec, Y, theta0;
        priors=priors_dist, observables=obs_syms,
        method=Symbol(samp), n_draws=n_draws, burnin=burnin, n_smc=n_smc,
        n_mh_steps=n_mh_steps, ess_target=ess_target, measurement_error=me,
        ha_method=meth, ha_kwargs=(T_horizon=t_horizon, n_reduced=n_reduced, hh_solver=hh),
        proposal_scale=proposal_scale, adapt_interval=adapt_interval, rng=rng,
        seed=eff_seed)

    draws = result.theta_draws
    np = size(draws, 2)
    est_df = DataFrame(
        parameter = String.(result.param_names),
        mean = [round(mean(draws[:, i]); digits=6) for i in 1:np],
        std = [round(sqrt(var(draws[:, i])); digits=6) for i in 1:np],
        q05 = [round(quantile(draws[:, i], 0.05); digits=6) for i in 1:np],
        median = [round(median(draws[:, i]); digits=6) for i in 1:np],
        q95 = [round(quantile(draws[:, i], 0.95); digits=6) for i in 1:np],
    )
    output_result(est_df; format=Symbol(format), output=output,
                  title="HA-DSGE Bayesian Posterior ($sampler_label, method=$meth)",
                  key="ha_dsge_bayesian_posterior")
    settings = Pair{String,Any}[
        "sampler" => samp,
        "ha_method" => String(meth),
        "hh_solver" => hh_solver,
        "observables" => (isempty(obs_syms) ? "(default)" : join(String.(obs_syms), ",")),
        "measurement_error" => (me === nothing ? "none" : string(me)),
        "n_draws" => n_draws,
        "burnin" => burnin,
        "n_smc" => n_smc,
    ]
    if hasproperty(result, :solver)
        push!(settings, "solver" => string(result.solver))
    end
    if hasproperty(result, :data) && result.data isa AbstractMatrix
        push!(settings, "data_T" => size(result.data, 1))
        push!(settings, "data_n" => size(result.data, 2))
    end
    output_kv(settings; format=format, output=_per_var_output_path(output, "settings"),
              title="HA-DSGE Bayesian Settings", key="ha_dsge_bayesian_settings")

    _status()
    _status_styled("  Log marginal likelihood: $(round(result.log_marginal_likelihood; digits=4))\n"; color=:cyan)
    _status_styled("  Acceptance rate: $(round(result.acceptance_rate; digits=4))\n"; color=:cyan)
    _status_styled("  Effective draws: $(size(draws, 1)) (after burnin=$burnin)\n"; color=:cyan)
    return result
end

function _dsge_ha_hd(; model::String, method::String="ssj",
                      data::String="", observables::String="",
                      measurement_error::String="",
                      n_reduced::Int=30, t_horizon::Int=300,
                      hh_solver::String="egm", distribution::String="young",
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="")
    isempty(data) && throw(CliError("usage/missing-option",
        "--data is required for HA historical decomposition"))
    isempty(observables) && throw(CliError("usage/missing-option",
        "--observables is required (comma-separated aggregate names)"))
    meth = _parse_ha_method(method)
    meth === :krusell_smith && throw(CliError("usage/invalid-option",
        "HA HD requires ssj or reiter (no states=; the RA `dsge hd` flag is not copied)"))
    spec = _load_ha_model(model; distribution=distribution)
    hh = _parse_hh_solver(hh_solver)
    sol = _dsge_ha_require_linear(
        _solve_ha(spec; method=meth, n_reduced=n_reduced, T_horizon=t_horizon, hh_solver=hh), meth)

    df = load_data(data)
    Y = df_to_matrix(df)
    obs_syms = Symbol[Symbol(strip(s)) for s in split(observables, ",") if !isempty(strip(s))]
    me = if isempty(measurement_error)
        nothing
    elseif measurement_error == "auto"
        :auto
    else
        [parse(Float64, strip(s)) for s in split(measurement_error, ",")]
    end
    hd = try
        historical_decomposition(sol, Y, obs_syms; measurement_error=me)
    catch e
        throw(_domain_or_data_error(e, "HA historical decomposition"))
    end
    for (si, sname) in enumerate(hd.shock_names)
        contrib = hd.contributions[:, :, si]
        contrib_df = DataFrame(contrib, hd.variables)
        insertcols!(contrib_df, 1, :t => 1:hd.T_eff)
        output_result(contrib_df; format=Symbol(format),
            output=_per_var_output_path(output, string(sname)),
            title="HA shock: $sname contributions",
            key=_table_key("ha_historical_decomposition", sname))
    end
    _maybe_plot(hd; plot=plot, plot_save=plot_save)
    return hd
end

# ── Continuous-time HA + Blanchard OLG (C041) ───────────────

function _dsge_ct_build_aiyagari(; alpha, rho, sigma, delta, z, a_min, a_max, grid_size)
    return MacroEconometricModels.CTAiyagari(;
        alpha=alpha, rho=rho, sigma=sigma, delta=delta, Z=z,
        a_min=a_min, a_max=a_max, I=grid_size)
end

function _dsge_ct_solve(; alpha::Float64=0.36, rho::Float64=0.05, sigma::Float64=2.0,
                         delta::Float64=0.05, z::Float64=1.0,
                         a_min::Float64=0.0, a_max::Float64=30.0, grid_size::Int=100,
                         max_iter::Int=100, tol::Float64=1e-6,
                         two_asset::Bool=false, ge::Bool=false,
                         output::String="", format::String="table")
    ge && !two_asset && throw(CliError("usage/invalid",
        "--ge requires --two-asset (ct_two_asset_ge)"))
    if two_asset && ge
        _status("Solving continuous-time two-asset GE (ct_two_asset_ge)...")
        m = MacroEconometricModels.CTTwoAsset(; sigma=sigma, rho=rho, alpha=alpha,
                                               delta=delta, Z=z)
        ge0 = MacroEconometricModels.ct_two_asset_ge(m; max_iter=max_iter, tol=tol)
        output_kv(Pair{String,Any}[
            "r_a" => ge0.r_a, "r_b" => ge0.r_b, "w" => ge0.w, "tau" => ge0.tau,
            "K" => ge0.K, "B" => ge0.B, "L" => ge0.L, "Y" => ge0.Y,
            "B_supply" => hasproperty(m, :B_supply) ? m.B_supply : 1.0,
            "resid_illiquid" => ge0.resid_illiquid, "resid_liquid" => ge0.resid_liquid,
            "markets_cleared" => ge0.markets_cleared,
            "converged" => ge0.converged, "iterations" => ge0.iterations,
        ]; format=format, output=output, title="CT Two-Asset GE", key="ct_two_asset_ge")
        return ge0
    end
    if two_asset
        _status("Solving continuous-time two-asset (KMV) model...")
        m = MacroEconometricModels.CTTwoAsset(; sigma=sigma, rho=rho)
        sol = MacroEconometricModels.ct_two_asset_solve(m; max_iter=max_iter, tol=tol)
        # Summarize two-asset solution
        gsum = sum(sol.g)
        diag_df = DataFrame(
            metric = ["hjb_converged", "B_liquid", "A_illiquid", "mass"],
            value = [
                Float64(sol.hjb_converged ? 1.0 : 0.0),
                Float64(sol.B),
                Float64(sol.A),
                Float64(gsum),
            ],
        )
        output_result(diag_df; format=Symbol(format), output=output,
                      title="CT Two-Asset Solution")
        return sol
    end

    _status("Solving continuous-time Aiyagari steady state (I=$grid_size)...")
    m = _dsge_ct_build_aiyagari(; alpha, rho, sigma, delta, z, a_min, a_max, grid_size)
    ss = MacroEconometricModels.ct_steady_state(m; max_iter=max_iter, tol=tol)
    _status_styled("  Converged: $(ss.converged)  r=$(round(ss.r; digits=5))  K=$(round(ss.K; digits=4))\n";
                   color = ss.converged ? :green : :yellow)
    price_df = DataFrame(name=["r", "w"], value=[ss.r, ss.w])
    agg_df = DataFrame(name=["K", "L", "converged"],
                       value=[ss.K, ss.L, Float64(ss.converged ? 1.0 : 0.0)])
    output_result(price_df; format=Symbol(format),
                  output=_per_var_output_path(output, "prices"), title="CT Aiyagari Prices")
    output_result(agg_df; format=Symbol(format), output=output, title="CT Aiyagari Aggregates")
    return ss
end

function _dsge_ct_transition(; alpha::Float64=0.36, rho::Float64=0.05, sigma::Float64=2.0,
                              delta::Float64=0.05, z::Float64=1.0,
                              shock_size::Float64=0.95, periods::Int=40, dt::Float64=0.25,
                              a_max::Float64=30.0, grid_size::Int=100,
                              max_iter::Int=100, tol::Float64=1e-6,
                              z_path::String="", two_asset::Bool=false,
                              output::String="", format::String="table",
                              plot::Bool=false, plot_save::String="")
    if two_asset
        isempty(z_path) && throw(CliError("usage/missing-option",
            "dsge ct transition --two-asset requires --z-path <csv>"))
        Z = _load_positive_path(z_path; min_length=2, name="Z_path")
        _status("CT two-asset MIT: T=$(length(Z))")
        m = MacroEconometricModels.CTTwoAsset(; sigma=sigma, rho=rho, alpha=alpha,
                                               delta=delta, Z=z)
        ge0 = MacroEconometricModels.ct_two_asset_ge(m; max_iter=max_iter, tol=tol)
        tr = MacroEconometricModels.ct_two_asset_mit(m, ge0, Z; dt=dt, max_iter=max_iter, tol=tol)
        df = DataFrame(t=tr.t, Z=tr.Z, K=tr.K, r_a=tr.r_a, r_b=tr.r_b, w=tr.w, B=tr.B, C=tr.C)
        output_result(df; format=Symbol(format), output=output,
                      title="CT Two-Asset MIT Transition (T=$(length(Z)))",
                      key="ct_two_asset_transition")
        (plot || !isempty(plot_save)) && throw(CliError("model/unsupported",
            "no plot_result recipe for CTTwoAssetTransition; drop --plot/--plot-save"))
        return tr
    end
    periods >= 2 || throw(CliError("usage/invalid-option",
        "--periods must be >= 2 (need impact + terminal)"))
    _status("CT MIT-shock transition: periods=$periods, impact Z=$(shock_size)*$z")
    m = _dsge_ct_build_aiyagari(; alpha, rho, sigma, delta, z, a_min=0.0, a_max, grid_size)
    ss = MacroEconometricModels.ct_steady_state(m; max_iter=max_iter, tol=tol)
    # Z_path: impact then reversion to m.Z
    Z_path = fill(Float64(m.Z), periods)
    Z_path[1] = shock_size * Float64(m.Z)
    tr = MacroEconometricModels.ct_mit_shock(m, ss, Z_path; dt=dt, max_iter=max_iter, tol=tol)
    df = DataFrame(
        t = tr.t,
        Z = tr.Z,
        K = tr.K,
        r = tr.r,
        w = tr.w,
        C = tr.C,
    )
    _status_styled("  Transition converged: $(tr.converged) (iters=$(tr.iterations))\n";
                   color = tr.converged ? :green : :yellow)
    _maybe_plot(df; plot=plot, plot_save=plot_save)
    output_result(df; format=Symbol(format), output=output,
                  title="CT MIT-Shock Transition (impact=$(shock_size), T=$periods)",
                  key="ct_mit_shock_transition")
    return tr
end

function _dsge_olg_build(; alpha, beta, delta, gamma, z, debt)
    return MacroEconometricModels.BlanchardOLG(;
        alpha=alpha, beta=beta, delta=delta, gamma=gamma, Z=z, b=debt)
end

function _dsge_olg_maybe_warn_debt(debt::Float64)
    if abs(debt) > 0
        # MEMs#237 was fixed in later tips for C = r*k + w; still surface a soft note
        # when debt is non-zero so agents know to re-check upstream notes.
        _status_styled(
            "  Note: debt b=$debt — verify MEMs Blanchard debt accounting (upstream #237).\n";
            color=:yellow)
    end
end

function _dsge_olg_solve(; alpha::Float64=0.36, beta::Float64=0.96, delta::Float64=0.08,
                          gamma::Float64=0.98, z::Float64=1.0, debt::Float64=0.0,
                          output::String="", format::String="table")
    _dsge_olg_maybe_warn_debt(debt)
    m = _dsge_olg_build(; alpha, beta, delta, gamma, z, debt)
    _status("Solving Blanchard OLG (γ=$gamma, b=$debt)...")
    sol = MacroEconometricModels.blanchard_solve(m)
    ss = sol.ss
    ss_df = DataFrame(
        variable = ["k", "C", "r", "w", "H", "mpc", "b", "converged"],
        value = [ss.k, ss.C, ss.r, ss.w, ss.H, ss.mpc, ss.b, Float64(ss.converged ? 1.0 : 0.0)],
    )
    dyn_df = DataFrame(
        metric = ["stable_eig", "policy_slope", "determinate",
                  "eig1_mod", "eig2_mod"],
        value = [
            Float64(sol.stable_eig),
            Float64(sol.policy_slope),
            Float64(sol.determinate ? 1.0 : 0.0),
            Float64(abs(sol.eigenvalues[1])),
            Float64(abs(sol.eigenvalues[2])),
        ],
    )
    _status_styled("  Determinate: $(sol.determinate)  stable_eig=$(round(sol.stable_eig; digits=4))\n";
                   color = sol.determinate ? :green : :red)
    output_result(ss_df; format=Symbol(format), output=output, title="Blanchard OLG Steady State")
    output_result(dyn_df; format=Symbol(format), output=output, title="Blanchard OLG Dynamics")
    return sol
end

function _dsge_olg_simulate(; alpha::Float64=0.36, beta::Float64=0.96, delta::Float64=0.08,
                             gamma::Float64=0.98, z::Float64=1.0, debt::Float64=0.0,
                             k0::Float64=0.0, horizon::Int=50,
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    _dsge_olg_maybe_warn_debt(debt)
    m = _dsge_olg_build(; alpha, beta, delta, gamma, z, debt)
    sol = MacroEconometricModels.blanchard_solve(m)
    k_init = k0 == 0.0 ? 0.8 * sol.ss.k : k0
    _status("OLG transition: k0=$(round(k_init; digits=4)), H=$horizon, k*=$(round(sol.ss.k; digits=4))")
    paths = MacroEconometricModels.blanchard_transition(m, sol, k_init; H=horizon)
    df = DataFrame(
        period = 0:horizon,
        k = paths.k,
        C = paths.C,
        r = paths.r,
        w = paths.w,
    )
    _maybe_plot(df; plot=plot, plot_save=plot_save)
    output_result(df; format=Symbol(format), output=output,
                  title="Blanchard OLG Transition (H=$horizon)",
                  key="blanchard_olg_transition")
    return paths
end

function _output_family_irf(ir; format::String, output::String, title::String, key_prefix::String)
    vals = ir.values
    n_h, n_v, n_s = size(vals)
    vars = hasproperty(ir, :variables) ? ir.variables : String[]
    shocks = hasproperty(ir, :shocks) ? ir.shocks : String[]
    for si in 1:n_s
        shock = si <= length(shocks) ? String(shocks[si]) : "shock_$si"
        irf_df = DataFrame(horizon = 0:(n_h - 1))
        for vi in 1:n_v
            vname = vi <= length(vars) ? String(vars[vi]) : "y$vi"
            irf_df[!, vname] = vals[:, vi, si]
        end
        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock),
                      title="$title: shock=$shock",
                      key=_table_key(key_prefix, shock))
    end
    return ir
end

function _output_family_fevd(fv; format::String, output::String, title::String, key_prefix::String)
    props = fv.proportions
    n_v, n_s, n_h = size(props)
    vars = hasproperty(fv, :variables) ? fv.variables : ["y$i" for i in 1:n_v]
    shocks = hasproperty(fv, :shocks) ? fv.shocks : ["shock_$i" for i in 1:n_s]
    for vi in 1:n_v
        vname = vi <= length(vars) ? String(vars[vi]) : "y$vi"
        fevd_df = DataFrame(horizon = 1:n_h)
        for si in 1:n_s
            sname = si <= length(shocks) ? String(shocks[si]) : "shock_$si"
            fevd_df[!, sname] = props[vi, si, :]
        end
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="$title: $vname",
                      key=_table_key(key_prefix, vname))
    end
    return fv
end

function _dsge_olg_to_spec(; alpha, beta, delta, gamma, z, debt, rho_z, sigma_z,
                            nk::Bool, kappa, phi_pi, phi_y, rho_i, sigma_i, omega)
    m = _dsge_olg_build(; alpha, beta, delta, gamma, z, debt)
    spec = MacroEconometricModels.to_spec(m; rho_z=rho_z, sigma_z=sigma_z)
    nk || return spec
    (0 <= omega <= 1) || throw(CliError("usage/invalid", "--omega must be in [0, 1]"))
    (0 <= rho_i < 1) || throw(CliError("usage/invalid", "--rho-i must be in [0, 1)"))
    return MacroEconometricModels.blanchard_nk_spec(spec; kappa=kappa, phi_pi=phi_pi,
                                                     phi_y=phi_y, rho_i=rho_i,
                                                     sigma_i=sigma_i, omega=omega)
end

function _dsge_olg_irf(; alpha::Float64=0.36, beta::Float64=0.96, delta::Float64=0.08,
                        gamma::Float64=0.98, z::Float64=1.0, debt::Float64=0.0,
                        rho_z::Float64=0.0, sigma_z::Float64=0.0,
                        horizon::Int=40, shock_size::Float64=1.0,
                        kappa::Float64=0.1, phi_pi::Float64=1.5, phi_y::Float64=0.125,
                        rho_i::Float64=0.0, sigma_i::Float64=0.0, omega::Float64=0.0,
                        nk::Bool=false,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid", "--horizon must be ≥ 1"))
    spec = _dsge_olg_to_spec(; alpha, beta, delta, gamma, z, debt, rho_z, sigma_z,
                              nk, kappa, phi_pi, phi_y, rho_i, sigma_i, omega)
    sol = _solve_dsge(spec; method="gensys")
    ir = irf(sol, horizon; shock_size=shock_size)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="Blanchard OLG IRF", key_prefix="blanchard_olg_irf")
end

function _dsge_olg_fevd(; alpha::Float64=0.36, beta::Float64=0.96, delta::Float64=0.08,
                         gamma::Float64=0.98, z::Float64=1.0, debt::Float64=0.0,
                         rho_z::Float64=0.0, sigma_z::Float64=0.0, horizon::Int=40,
                         kappa::Float64=0.1, phi_pi::Float64=1.5, phi_y::Float64=0.125,
                         rho_i::Float64=0.0, sigma_i::Float64=0.0, omega::Float64=0.0,
                         nk::Bool=false,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid", "--horizon must be ≥ 1"))
    spec = _dsge_olg_to_spec(; alpha, beta, delta, gamma, z, debt, rho_z, sigma_z,
                              nk, kappa, phi_pi, phi_y, rho_i, sigma_i, omega)
    sol = _solve_dsge(spec; method="gensys")
    fv = fevd(sol, horizon)
    _maybe_plot(fv; plot=plot, plot_save=plot_save)
    return _output_family_fevd(fv; format=format, output=output,
                               title="Blanchard OLG FEVD", key_prefix="blanchard_olg_fevd")
end

function _dsge_ct_irf(; alpha::Float64=0.36, rho::Float64=0.05, sigma::Float64=2.0,
                       delta::Float64=0.05, z::Float64=1.0,
                       horizon::Int=40, shock_size::Float64=0.01, persist::Float64=0.0,
                       dt::Float64=0.25, grid_size::Int=100,
                       two_asset::Bool=false,
                       output::String="", format::String="table",
                       plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2 (MIT path length)"))
    ir = if two_asset
        m = MacroEconometricModels.CTTwoAsset(; sigma=sigma, rho=rho, alpha=alpha,
                                               delta=delta, Z=z)
        irf(m, horizon; shock_size=shock_size, persist=persist, dt=dt)
    else
        m = _dsge_ct_build_aiyagari(; alpha, rho, sigma, delta, z, a_min=0.0,
                                    a_max=30.0, grid_size)
        irf(m, horizon; shock_size=shock_size, persist=persist, dt=dt)
    end
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="CT IRF", key_prefix="ct_irf")
end

function _dsge_ct_fevd(; alpha::Float64=0.36, rho::Float64=0.05, sigma::Float64=2.0,
                        delta::Float64=0.05, z::Float64=1.0,
                        horizon::Int=40, shock_size::Float64=0.01, persist::Float64=0.0,
                        dt::Float64=0.25, grid_size::Int=100,
                        two_asset::Bool=false,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2 (MIT path length)"))
    fv = if two_asset
        m = MacroEconometricModels.CTTwoAsset(; sigma=sigma, rho=rho, alpha=alpha,
                                               delta=delta, Z=z)
        fevd(m, horizon; shock_size=shock_size, persist=persist, dt=dt)
    else
        m = _dsge_ct_build_aiyagari(; alpha, rho, sigma, delta, z, a_min=0.0,
                                    a_max=30.0, grid_size)
        fevd(m, horizon; shock_size=shock_size, persist=persist, dt=dt)
    end
    _maybe_plot(fv; plot=plot, plot_save=plot_save)
    return _output_family_fevd(fv; format=format, output=output,
                               title="CT FEVD", key_prefix="ct_fevd")
end

# ── DCEGM ───────────────────────────────────────────────────

function _dcegm_guard(; n_periods, beta, n_shocks, n_a, curvature, taste_shock_scale)
    n_periods >= 0 || throw(CliError("usage/invalid", "--n-periods must be ≥ 0"))
    0 < beta < 1 || throw(CliError("usage/invalid", "--beta must lie in (0, 1)"))
    n_shocks >= 1 || throw(CliError("usage/invalid", "--n-shocks must be ≥ 1"))
    n_a >= 2 || throw(CliError("usage/invalid", "--n-a must be ≥ 2"))
    curvature >= 1 || throw(CliError("usage/invalid", "--curvature must be ≥ 1"))
    taste_shock_scale >= 0 || throw(CliError("usage/invalid", "--taste-shock-scale must be ≥ 0"))
end

function _load_dcegm_source(model::String; n_periods=20, beta=0.98, r=1.0, wage=20.0,
                             disutility=1.0, sigma=0.0, n_shocks=1,
                             taste_shock_scale=0.0, a_max=50.0, n_a=200,
                             pension=0.0, credit_limit=0.0, curvature=2.0)
    tok = lowercase(strip(model))
    startswith(tok, ":") && (tok = tok[2:end])
    if tok in ("retirement", "dcegm-retirement", "dcegm_retirement")
        _dcegm_guard(; n_periods, beta, n_shocks, n_a, curvature, taste_shock_scale)
        return MacroEconometricModels.dcegm_retirement_model(;
            n_periods=n_periods, beta=beta, R=r, wage=wage, disutility=disutility,
            sigma=sigma, n_shocks=n_shocks, taste_shock_scale=taste_shock_scale,
            a_max=a_max, n_a=n_a, pension=pension, credit_limit=credit_limit,
            curvature=curvature)
    end
    _validate_input_path(model)
    isfile(model) || throw(CliError("data/file-not-found", "DCEGM model file not found: $model"))
    ext = lowercase(splitext(model)[2])
    ext == ".jl" || throw(CliError("usage/invalid-option",
        "DCEGM model must be builtin `retirement` or a .jl file (got '$ext')"))
    mod = _dsge_sandbox()
    result = try
        Base.include(mod, model)
    catch e
        e isa CliError && rethrow()
        _dsge_eval_invalid(e, "could not evaluate the DCEGM model file '$model'";
            hint="the file should evaluate to a DCEGMProblem or a ModelSpec with DCEGMSystem")
    end
    if result isa MacroEconometricModels.DCEGMProblem
        return result
    end
    _is_model_spec(result) || throw(CliError("config/invalid",
        "DCEGM .jl file must evaluate to a DCEGMProblem or ModelSpec (got $(typeof(result)))"))
    MacroEconometricModels.has_kind(result, MacroEconometricModels.DCEGMSystem) ||
        throw(_wrong_command_for_kinds(result, "dsge dcegm"))
    return result
end

function _dcegm_as_problem(src)
    src isa MacroEconometricModels.DCEGMProblem && return src
    return only(MacroEconometricModels.agents_of(src, MacroEconometricModels.DCEGMSystem)).problem
end

function _dsge_dcegm_solve(; model::String,
                            n_periods::Int=20, beta::Float64=0.98, r::Float64=1.0,
                            wage::Float64=20.0, disutility::Float64=1.0, sigma::Float64=0.0,
                            n_shocks::Int=1, taste_shock_scale::Float64=0.0,
                            a_max::Float64=50.0, n_a::Int=200, pension::Float64=0.0,
                            credit_limit::Float64=0.0, curvature::Float64=2.0,
                            max_iter::Int=500, tol::Float64=1e-8,
                            period::Int=1, income::Int=1, view::String="policy",
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    vw = lowercase(strip(view))
    vw in ("policy", "threshold") || throw(CliError("usage/invalid",
        "--view must be policy|threshold"))
    src = _load_dcegm_source(model; n_periods, beta, r, wage, disutility, sigma, n_shocks,
                              taste_shock_scale, a_max, n_a, pension, credit_limit, curvature)
    prob = _dcegm_as_problem(src)
    sol = _dsge_call(MacroEconometricModels.dcegm_solve, prob; max_iter=max_iter, tol=tol)
    n_t, n_d, n_e = size(sol.M)
    output_kv(Pair{String,Any}[
        "converged" => sol.converged,
        "iterations" => sol.iterations,
        "sup_diff" => sol.sup_diff,
        "n_periods" => sol.n_periods,
        "n_options" => n_d,
        "n_income" => n_e,
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
       title="DCEGM Solve Diagnostics", key="dcegm_solve_diagnostics")
    t = clamp(period, 1, n_t)
    j = clamp(income, 1, n_e)
    opts = hasproperty(sol, :prob) && hasproperty(sol.prob, :options) ? sol.prob.options :
           Symbol[:d1]
    rows = NamedTuple[]
    for d in 1:n_d
        Md = sol.M[t, d, j]; cd = sol.c[t, d, j]; vd = sol.v[t, d, j]
        oname = d <= length(opts) ? String(opts[d]) : "d$d"
        for i in 1:length(Md)
            push!(rows, (option=oname, knot=i, M=Float64(Md[i]),
                         c=Float64(cd[i]), v=Float64(vd[i])))
        end
    end
    output_result(DataFrame(rows); format=Symbol(format), output=output,
                  title="DCEGM Policy (period=$t, income=$j)", key="dcegm_policy")
    kink_rows = NamedTuple[]
    for tt in 1:n_t, d in 1:n_d, jj in 1:n_e
        oname = d <= length(opts) ? String(opts[d]) : "d$d"
        push!(kink_rows, (period=tt, option=oname, income=jj, n_kinks=Int(sol.n_kinks[tt, d, jj])))
    end
    output_result(DataFrame(kink_rows); format=Symbol(format),
                  output=_per_var_output_path(output, "kinks"),
                  title="DCEGM Kinks", key="dcegm_kinks")
    _maybe_plot(sol; plot=plot, plot_save=plot_save, view=Symbol(vw), period=t, income=j)
    return sol
end

function _dcegm_equilibrium(src; alpha, delta, z, l, r_lo, r_hi, labor, reprice_wage,
                             work_option, n_sim, tol, max_iter)
    0 < alpha < 1 || throw(CliError("usage/invalid", "--alpha must lie in (0, 1)"))
    delta >= 0 || throw(CliError("usage/invalid", "--delta must be ≥ 0"))
    z > 0 || throw(CliError("usage/invalid", "--z must be > 0"))
    l > 0 || throw(CliError("usage/invalid", "--l must be > 0"))
    r_lo < r_hi || throw(CliError("usage/invalid", "--r-lo must be < --r-hi"))
    lab = Symbol(lowercase(strip(labor)))
    lab in (:exogenous, :measured) || throw(CliError("usage/invalid",
        "--labor must be exogenous|measured"))
    firm = MacroEconometricModels.DCEGMFirm(; alpha=alpha, delta=delta, Z=z, L=l)
    return _dsge_call(MacroEconometricModels.dcegm_steady_state, src, firm;
                      r_bounds=(r_lo, r_hi), labor=lab, reprice_wage=reprice_wage,
                      work_option=Symbol(work_option), n_sim=n_sim, tol=tol, max_iter=max_iter)
end

function _dsge_dcegm_steady_state(; model::String,
                                   n_periods::Int=20, beta::Float64=0.98, r::Float64=1.0,
                                   wage::Float64=20.0, disutility::Float64=1.0, sigma::Float64=0.0,
                                   n_shocks::Int=1, a_max::Float64=50.0, n_a::Int=200,
                                   pension::Float64=0.0, credit_limit::Float64=0.0,
                                   curvature::Float64=2.0,
                                   alpha::Float64=0.36, delta::Float64=0.08, z::Float64=1.0,
                                   l::Float64=1.0, r_lo::Float64=0.001, r_hi::Float64=0.20,
                                   labor::String="exogenous", work_option::String="work",
                                   n_sim::Int=40, tol::Float64=1e-4, max_iter::Int=40,
                                   reprice_wage::Bool=false,
                                   output::String="", format::String="table")
    src = _load_dcegm_source(model; n_periods, beta, r, wage, disutility, sigma, n_shocks,
                              a_max=a_max, n_a=n_a, pension=pension,
                              credit_limit=credit_limit, curvature=curvature)
    eq = _dcegm_equilibrium(src; alpha, delta, z, l, r_lo, r_hi, labor, reprice_wage,
                             work_option, n_sim, tol, max_iter)
    output_kv(Pair{String,Any}[
        "r" => eq.r, "w" => eq.w, "K" => eq.K, "L" => eq.L, "Y" => eq.Y,
        "K_demand" => eq.K_demand, "excess_demand" => eq.excess_demand,
        "converged" => eq.converged, "iterations" => eq.iterations,
    ]; format=format, output=output, title="DCEGM Equilibrium", key="dcegm_equilibrium")
    return eq
end

function _dsge_dcegm_irf(; model::String, n_periods::Int=20, beta::Float64=0.98,
                          wage::Float64=20.0, n_a::Int=80, a_max::Float64=50.0,
                          alpha::Float64=0.36, delta::Float64=0.08, z::Float64=1.0,
                          horizon::Int=40, shock_size::Float64=0.01, persist::Float64=0.0,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    src = _load_dcegm_source(model; n_periods, beta, wage=wage, n_a=n_a, a_max=a_max)
    eq = _dcegm_equilibrium(src; alpha, delta, z, l=1.0, r_lo=0.001, r_hi=0.20,
                             labor="exogenous", reprice_wage=false, work_option="work",
                             n_sim=40, tol=1e-4, max_iter=40)
    ir = irf(eq, horizon; shock_size=shock_size, persist=persist)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="DCEGM IRF", key_prefix="dcegm_irf")
end

function _dsge_dcegm_fevd(; model::String, n_periods::Int=20, beta::Float64=0.98,
                           wage::Float64=20.0, n_a::Int=80, a_max::Float64=50.0,
                           alpha::Float64=0.36, delta::Float64=0.08, z::Float64=1.0,
                           horizon::Int=40, shock_size::Float64=0.01, persist::Float64=0.0,
                           output::String="", format::String="table",
                           plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    src = _load_dcegm_source(model; n_periods, beta, wage=wage, n_a=n_a, a_max=a_max)
    eq = _dcegm_equilibrium(src; alpha, delta, z, l=1.0, r_lo=0.001, r_hi=0.20,
                             labor="exogenous", reprice_wage=false, work_option="work",
                             n_sim=40, tol=1e-4, max_iter=40)
    fv = fevd(eq, horizon; shock_size=shock_size, persist=persist)
    _maybe_plot(fv; plot=plot, plot_save=plot_save)
    return _output_family_fevd(fv; format=format, output=output,
                               title="DCEGM FEVD", key_prefix="dcegm_fevd")
end

function _dsge_dcegm_simulate(; model::String, n_periods::Int=20, beta::Float64=0.98,
                               wage::Float64=20.0, n_a::Int=80, a_max::Float64=50.0,
                               alpha::Float64=0.36, delta::Float64=0.08, z::Float64=1.0,
                               periods::Int=40, shock_size::Float64=0.0, persist::Float64=0.0,
                               output::String="", format::String="table")
    periods >= 2 || throw(CliError("usage/invalid", "--periods must be ≥ 2"))
    src = _load_dcegm_source(model; n_periods, beta, wage=wage, n_a=n_a, a_max=a_max)
    eq = _dcegm_equilibrium(src; alpha, delta, z, l=1.0, r_lo=0.001, r_hi=0.20,
                             labor="exogenous", reprice_wage=false, work_option="work",
                             n_sim=40, tol=1e-4, max_iter=40)
    path = simulate(eq, periods; shock_size=shock_size, persist=persist)
    df = DataFrame(path, ["K", "r", "w", "Y", "Z"])
    insertcols!(df, 1, :period => 1:size(path, 1))
    output_result(df; format=Symbol(format), output=output,
                  title="DCEGM Simulation (T=$periods)", key="dcegm_simulation")
    return path
end

function _dsge_dcegm_transition(; model::String, z_path::String="",
                                 n_periods::Int=20, beta::Float64=0.98,
                                 wage::Float64=20.0, n_a::Int=80, a_max::Float64=50.0,
                                 alpha::Float64=0.36, delta::Float64=0.08, z::Float64=1.0,
                                 output::String="", format::String="table")
    isempty(z_path) && throw(CliError("usage/missing-option",
        "dsge dcegm transition requires --z-path <csv>"))
    Z = _load_positive_path(z_path; min_length=2, name="Z_path")
    src = _load_dcegm_source(model; n_periods, beta, wage=wage, n_a=n_a, a_max=a_max)
    eq = _dcegm_equilibrium(src; alpha, delta, z, l=1.0, r_lo=0.001, r_hi=0.20,
                             labor="exogenous", reprice_wage=false, work_option="work",
                             n_sim=40, tol=1e-4, max_iter=40)
    tr = MacroEconometricModels.dcegm_mit(eq, Z)
    df = DataFrame(t=1:length(tr.Z), Z=tr.Z, K=tr.K, r=tr.r, w=tr.w, A=tr.A, Y=tr.Y)
    output_result(df; format=Symbol(format), output=output,
                  title="DCEGM Transition", key="dcegm_transition_path")
    output_kv(Pair{String,Any}["method" => string(tr.method), "converged" => tr.converged];
              format=format, output=_per_var_output_path(output, "diagnostics"),
              title="DCEGM Transition Diagnostics", key="dcegm_transition_diagnostics")
    return tr
end

# ── Life-cycle OLG ──────────────────────────────────────────

function _lifecycle_from_flags(; j, j_retire, survival, a_max, n_a, beta, sigma,
                                 alpha, delta, z, n_pop=0.0, replacement=0.4,
                                 credit_limit=0.0, income_rho, income_sigma, income_states,
                                 config, no_annuities=false)
    j >= 2 || throw(CliError("usage/invalid", "--j must be ≥ 2"))
    j_retire >= 2 || throw(CliError("usage/invalid", "--j-retire must be ≥ 2"))
    0 < beta < 1 || throw(CliError("usage/invalid", "--beta must lie in (0, 1)"))
    sigma > 0 || throw(CliError("usage/invalid", "--sigma must be > 0"))
    0 < alpha < 1 || throw(CliError("usage/invalid", "--alpha must lie in (0, 1)"))
    0 <= delta <= 1 || throw(CliError("usage/invalid", "--delta must lie in [0, 1]"))
    n_a >= 3 || throw(CliError("usage/invalid", "--n-a must be ≥ 3"))
    surv = survival
    earn = nothing
    if !isempty(config)
        cfg = load_config(config)
        sec = get(cfg, "lifecycle", cfg)
        if haskey(sec, "survival")
            surv = Float64[Float64(x) for x in sec["survival"]]
        end
        if haskey(sec, "earnings")
            earn = Float64[Float64(x) for x in sec["earnings"]]
        end
    end
    inc = MacroEconometricModels.lifecycle_income(income_rho, income_sigma, income_states)
    return MacroEconometricModels.LifeCycleOLG(;
        J=j, J_retire=j_retire, survival=surv, earnings=earn, income=inc,
        a_max=a_max, n_a=n_a, beta=beta, sigma=sigma, alpha=alpha, delta=delta,
        Z=z, n_pop=n_pop, replacement=replacement, credit_limit=credit_limit,
        annuities=!no_annuities)
end

function _dsge_lifecycle_steady_state(; j::Int=60, j_retire::Int=45, survival::Float64=0.99,
                                       a_max::Float64=60.0, n_a::Int=200, beta::Float64=0.97,
                                       sigma::Float64=2.0, alpha::Float64=0.36, delta::Float64=0.06,
                                       z::Float64=1.0, n_pop::Float64=0.0, replacement::Float64=0.4,
                                       credit_limit::Float64=0.0,
                                       income_rho::Float64=0.95, income_sigma::Float64=0.2,
                                       income_states::Int=5, config::String="",
                                       r_lo::Float64=-0.02, r_hi::Float64=0.10,
                                       tol::Float64=1e-6, max_iter::Int=60, bequest_iter::Int=50,
                                       no_annuities::Bool=false,
                                       output::String="", format::String="table",
                                       plot::Bool=false, plot_save::String="")
    r_lo < r_hi || throw(CliError("usage/invalid", "--r-lo must be < --r-hi"))
    m = _lifecycle_from_flags(; j, j_retire, survival, a_max, n_a, beta, sigma, alpha, delta,
                                z, n_pop, replacement, credit_limit, income_rho, income_sigma,
                                income_states, config, no_annuities)
    ss = MacroEconometricModels.lifecycle_steady_state(m; r_bounds=(r_lo, r_hi),
                                                        tol=tol, max_iter=max_iter,
                                                        bequest_iter=bequest_iter)
    output_kv(Pair{String,Any}[
        "r" => ss.r, "w" => ss.w, "K" => ss.K, "L" => ss.L, "Y" => ss.Y,
        "tau" => ss.tau, "pension" => ss.pension, "transfer" => ss.transfer,
        "excess_demand" => ss.excess_demand, "converged" => ss.converged,
        "iterations" => ss.iterations,
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
       title="Life-Cycle Steady State", key="lifecycle_steady_state")
    ages = 1:length(ss.cohort_mass)
    output_result(DataFrame(age=collect(ages), cohort_mass=ss.cohort_mass,
                            asset_profile=ss.asset_profile,
                            consumption_profile=ss.consumption_profile,
                            income_profile=ss.income_profile);
                  format=Symbol(format), output=output,
                  title="Life-Cycle Age Profiles", key="lifecycle_age_profiles")
    _maybe_plot(ss; plot=plot, plot_save=plot_save)
    return ss
end

function _dsge_lifecycle_transition(; j::Int=60, j_retire::Int=45, survival::Float64=0.99,
                                     a_max::Float64=60.0, n_a::Int=80, beta::Float64=0.97,
                                     sigma::Float64=2.0, alpha::Float64=0.36, delta::Float64=0.06,
                                     z::Float64=1.0,
                                     income_rho::Float64=0.95, income_sigma::Float64=0.2,
                                     income_states::Int=3, config::String="",
                                     k0::Float64=NaN, z_path::String="", horizon::Int=80,
                                     tol::Float64=1e-5, max_iter::Int=80, relax::Float64=0.5,
                                     output::String="", format::String="table")
    has_k0 = isfinite(k0)
    has_z = !isempty(z_path)
    xor(has_k0, has_z) || throw(CliError("usage/invalid",
        "dsge lifecycle transition requires exactly one of --k0 or --z-path"))
    m = _lifecycle_from_flags(; j, j_retire, survival, a_max, n_a, beta, sigma, alpha, delta,
                                z, income_rho, income_sigma, income_states, config)
    tr = if has_z
        Z = _load_positive_path(z_path; min_length=3, name="Z_path")
        MacroEconometricModels.lifecycle_transition(m, Z; tol=tol, max_iter=max_iter, relax=relax)
    else
        MacroEconometricModels.lifecycle_transition(m, k0; H=horizon, tol=tol,
                                                     max_iter=max_iter, relax=relax)
    end
    n = length(tr.K)
    output_result(DataFrame(t=0:(n-1), K=tr.K, r=tr.r, w=tr.w, Y=tr.Y, C=tr.C,
                            Z=tr.Z, pension=tr.pension, transfer=tr.transfer);
                  format=Symbol(format), output=output,
                  title="Life-Cycle Transition", key="lifecycle_transition_path")
    output_kv(Pair{String,Any}["tau" => tr.tau, "converged" => tr.converged,
                               "iterations" => tr.iterations];
              format=format, output=_per_var_output_path(output, "diagnostics"),
              title="Life-Cycle Transition Diagnostics",
              key="lifecycle_transition_diagnostics")
    return tr
end

function _lifecycle_ss_small(; j, j_retire, n_a, income_states)
    _lifecycle_from_flags(; j=j, j_retire=j_retire, survival=0.99, a_max=40.0, n_a=n_a,
                            beta=0.97, sigma=2.0, alpha=0.36, delta=0.06, z=1.0,
                            income_rho=0.95, income_sigma=0.2, income_states=income_states,
                            config="")
end

function _dsge_lifecycle_irf(; j::Int=40, j_retire::Int=30, n_a::Int=40, income_states::Int=3,
                              horizon::Int=20, shock_size::Float64=0.01, persist::Float64=0.0,
                              output::String="", format::String="table",
                              plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    m = _lifecycle_ss_small(; j, j_retire, n_a, income_states)
    ss = MacroEconometricModels.lifecycle_steady_state(m)
    ir = irf(ss, horizon; shock_size=shock_size, persist=persist)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="Life-Cycle IRF", key_prefix="lifecycle_irf")
end

function _dsge_lifecycle_fevd(; j::Int=40, j_retire::Int=30, n_a::Int=40, income_states::Int=3,
                               horizon::Int=20, shock_size::Float64=0.01, persist::Float64=0.0,
                               output::String="", format::String="table",
                               plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    m = _lifecycle_ss_small(; j, j_retire, n_a, income_states)
    ss = MacroEconometricModels.lifecycle_steady_state(m)
    fv = fevd(ss, horizon; shock_size=shock_size, persist=persist)
    _maybe_plot(fv; plot=plot, plot_save=plot_save)
    return _output_family_fevd(fv; format=format, output=output,
                               title="Life-Cycle FEVD", key_prefix="lifecycle_fevd")
end

function _dsge_lifecycle_simulate(; j::Int=40, j_retire::Int=30, n_a::Int=40,
                                   income_states::Int=3, periods::Int=20,
                                   shock_size::Float64=0.0, persist::Float64=0.0,
                                   output::String="", format::String="table")
    periods >= 2 || throw(CliError("usage/invalid", "--periods must be ≥ 2"))
    m = _lifecycle_ss_small(; j, j_retire, n_a, income_states)
    ss = MacroEconometricModels.lifecycle_steady_state(m)
    path = simulate(ss, periods; shock_size=shock_size, persist=persist)
    df = DataFrame(path, ["K", "r", "w", "Y", "Z"])
    insertcols!(df, 1, :period => 1:size(path, 1))
    output_result(df; format=Symbol(format), output=output,
                  title="Life-Cycle Simulation (T=$periods)", key="lifecycle_simulation")
    return path
end

# ── Khan–Thomas / Bewley banks ─────────────────────────────

function _khan_thomas_guard(; n_k, alpha, nu, delta, beta, gamma, xi_bar, b, phi,
                              rho_z, sigma_z, z)
    n_k >= 3 || throw(CliError("usage/invalid", "--n-k must be ≥ 3"))
    alpha > 0 || throw(CliError("usage/invalid", "--alpha must be > 0"))
    nu > 0 || throw(CliError("usage/invalid", "--nu must be > 0"))
    alpha + nu < 1 || throw(CliError("usage/invalid",
        "--alpha + --nu must be < 1 (decreasing returns)"))
    0 < delta < 1 || throw(CliError("usage/invalid", "--delta must lie in (0, 1)"))
    0 < beta < 1 || throw(CliError("usage/invalid", "--beta must lie in (0, 1)"))
    gamma >= 1 || throw(CliError("usage/invalid", "--gamma must be ≥ 1"))
    xi_bar >= 0 || throw(CliError("usage/invalid", "--xi-bar must be ≥ 0"))
    b >= 0 || throw(CliError("usage/invalid", "--b must be ≥ 0"))
    phi > 0 || throw(CliError("usage/invalid", "--phi must be > 0"))
    abs(rho_z) < 1 || throw(CliError("usage/invalid", "|--rho-z| must be < 1"))
    sigma_z >= 0 || throw(CliError("usage/invalid", "--sigma-z must be ≥ 0"))
    z > 0 || throw(CliError("usage/invalid", "--z must be > 0"))
end

function _khan_thomas_fs(; n_k=16, n_eps=3, alpha=0.256, nu=0.640, delta=0.069,
                           beta=0.977, gamma=1.016, xi_bar=0.0083, b=0.011, phi=2.4,
                           rho_z=0.859, sigma_z=0.014, rho_e=0.859, sigma_e=0.022, z=1.0)
    _khan_thomas_guard(; n_k, alpha, nu, delta, beta, gamma, xi_bar, b, phi, rho_z, sigma_z, z)
    return MacroEconometricModels.khan_thomas_example(;
        n_k=n_k, n_eps=n_eps, alpha=alpha, nu=nu, delta=delta, beta=beta, gamma=gamma,
        xi_bar=xi_bar, b=b, phi=phi, rho_z=rho_z, sigma_z=sigma_z, rho_e=rho_e,
        sigma_e=sigma_e, Z=z)
end

function _dsge_firm_steady_state(; n_k::Int=16, n_eps::Int=3,
                                  alpha::Float64=0.256, nu::Float64=0.640,
                                  delta::Float64=0.069, beta::Float64=0.977, gamma::Float64=1.016,
                                  xi_bar::Float64=0.0083, b::Float64=0.011, phi::Float64=2.4,
                                  rho_z::Float64=0.859, sigma_z::Float64=0.014,
                                  rho_e::Float64=0.859, sigma_e::Float64=0.022, z::Float64=1.0,
                                  tol::Float64=1e-5, max_iter::Int=16,
                                  output::String="", format::String="table")
    fs = _khan_thomas_fs(; n_k, n_eps, alpha, nu, delta, beta, gamma, xi_bar, b, phi,
                          rho_z, sigma_z, rho_e, sigma_e, z)
    ss = MacroEconometricModels.khan_thomas_steady_state(fs; tol=tol, max_iter=max_iter)
    output_kv(Pair{String,Any}[
        "w" => ss.w, "p" => ss.p, "K" => ss.K, "N" => ss.N, "Y" => ss.Y,
        "I" => ss.I, "C" => ss.C, "inaction" => ss.inaction,
        "converged" => ss.converged, "iterations" => ss.iterations,
        "method" => string(ss.method),
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
       title="Khan–Thomas Steady State", key="khan_thomas_steady_state")
    k_grid = fs.k_grid
    n_k_g = length(k_grid)
    n_e = size(ss.adj_prob, ndims(ss.adj_prob) == 2 ? 2 : 1)
    # Read orientation at runtime: k_constrained/adj_prob are n_k × n_ε; k_star is length n_ε.
    k_con = ss.k_constrained
    adj = ss.adj_prob
    if size(k_con, 1) != n_k_g && size(k_con, 2) == n_k_g
        k_con = permutedims(k_con)
        adj = permutedims(adj)
    end
    n_e = size(k_con, 2)
    kstar = ss.k_star
    rows = NamedTuple[]
    for i in 1:n_k_g, j in 1:n_e
        ks = j <= length(kstar) ? kstar[j] : kstar[1]
        push!(rows, (k_index=i, k=Float64(k_grid[i]), eps_index=j,
                     k_star=Float64(ks), k_constrained=Float64(k_con[i, j]),
                     adj_prob=Float64(adj[i, j])))
    end
    output_result(DataFrame(rows); format=Symbol(format), output=output,
                  title="Khan–Thomas Policy", key="khan_thomas_policy")
    return ss
end

function _dsge_firm_transition(; z_path::String="", prices::String="ss",
                                n_k::Int=16, n_eps::Int=3, z::Float64=1.0,
                                output::String="", format::String="table")
    isempty(z_path) && throw(CliError("usage/missing-option",
        "dsge firm transition requires --z-path <csv>"))
    pr = lowercase(strip(prices))
    pr in ("ss", "ge") || throw(CliError("usage/invalid", "--prices must be ss|ge"))
    Z = _load_positive_path(z_path; min_length=2, name="Z_path")
    fs = _khan_thomas_fs(; n_k, n_eps, z=z)
    ss = MacroEconometricModels.khan_thomas_steady_state(fs)
    tr = MacroEconometricModels.khan_thomas_mit(ss, Z; prices=Symbol(pr))
    n = length(tr.Z)
    output_result(DataFrame(t=1:n, Z=tr.Z, Y=tr.Y, I=tr.I, K=tr.K, N=tr.N, C=tr.C, w=tr.w);
                  format=Symbol(format), output=output,
                  title="Khan–Thomas Transition (prices=$pr)", key="khan_thomas_transition")
    output_kv(Pair{String,Any}["method" => string(tr.method), "converged" => tr.converged];
              format=format, output=_per_var_output_path(output, "diagnostics"),
              title="Khan–Thomas Transition Diagnostics",
              key="khan_thomas_transition_diagnostics")
    return tr
end

function _dsge_firm_irf(; horizon::Int=20, shock_size::Float64=0.01, persist::Float64=NaN,
                         prices::String="ss", n_k::Int=16, n_eps::Int=3, z::Float64=1.0,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    pr = lowercase(strip(prices))
    pr in ("ss", "ge") || throw(CliError("usage/invalid", "--prices must be ss|ge"))
    fs = _khan_thomas_fs(; n_k, n_eps, z=z)
    ss = MacroEconometricModels.khan_thomas_steady_state(fs)
    kw = isfinite(persist) ? (; shock_size=shock_size, persist=persist, prices=Symbol(pr)) :
                             (; shock_size=shock_size, prices=Symbol(pr))
    ir = irf(ss, horizon; kw...)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="Khan–Thomas IRF", key_prefix="khan_thomas_irf")
end

function _intermediary_guard(; n_min, n_max, n_n, beta, sigma, lambda, zeta1, zeta2, R, Z, alpha)
    n_min > 0 || throw(CliError("usage/invalid", "--n-min must be > 0"))
    n_max > n_min || throw(CliError("usage/invalid", "--n-max must exceed --n-min"))
    n_n >= 3 || throw(CliError("usage/invalid", "--n-n must be ≥ 3"))
    0 < beta < 1 || throw(CliError("usage/invalid", "--beta must lie in (0, 1)"))
    0 < sigma <= 1 || throw(CliError("usage/invalid", "--sigma (survival) must lie in (0, 1]"))
    lambda > 0 || throw(CliError("usage/invalid", "--lambda must be > 0"))
    zeta1 >= 0 || throw(CliError("usage/invalid", "--zeta1 must be ≥ 0"))
    zeta2 > 1 || throw(CliError("usage/invalid", "--zeta2 must exceed 1"))
    R > 0 || throw(CliError("usage/invalid", "--r (gross R) must be > 0"))
    Z > 0 || throw(CliError("usage/invalid", "--z must be > 0"))
    0 < alpha < 1 || throw(CliError("usage/invalid", "--alpha must lie in (0, 1)"))
end

function _intermediary_sys(; n_n=25, n_xi=3, n_min=0.05, n_max=8.0, beta=0.99, sigma=0.95,
                             lambda=0.20, zeta1=0.02, zeta2=2.0, R=1.01, rk=0.05,
                             z=0.25, alpha=0.33)
    _intermediary_guard(; n_min, n_max, n_n, beta, sigma, lambda, zeta1, zeta2, R, Z=z, alpha)
    return MacroEconometricModels.IntermediarySystem(;
        n_n=n_n, n_xi=n_xi, n_min=n_min, n_max=n_max, beta=beta, sigma=sigma,
        lambda=lambda, zeta1=zeta1, zeta2=zeta2, R=R, rk=rk, Z=z, alpha=alpha)
end

function _bewley_policy_df(sys, pol)
    n_grid = sys.grid.grids[1]
    n_n_g = length(n_grid)
    n_e = size(pol.l_policy, 2)
    rows = NamedTuple[]
    for i in 1:n_n_g, j in 1:n_e
        push!(rows, (n_index=i, n=Float64(n_grid[i]), xi_index=j,
                     l_policy=Float64(pol.l_policy[i, j]),
                     b_policy=Float64(pol.b_policy[i, j])))
    end
    return DataFrame(rows)
end

function _dsge_bank_pe(; n_n::Int=25, n_xi::Int=3, n_min::Float64=0.05, n_max::Float64=8.0,
                        beta::Float64=0.99, sigma::Float64=0.95, lambda::Float64=0.20,
                        zeta1::Float64=0.02, zeta2::Float64=2.0,
                        r::Float64=1.01, rk::Float64=0.05, z::Float64=0.25, alpha::Float64=0.33,
                        max_iter::Int=250, tol::Float64=1e-6,
                        output::String="", format::String="table")
    sys = _intermediary_sys(; n_n, n_xi, n_min, n_max, beta, sigma, lambda, zeta1, zeta2,
                             R=r, rk=rk, z=z, alpha=alpha)
    pe = MacroEconometricModels.intermediary_pe(sys; R=r, rk=rk, max_iter=max_iter, tol=tol)
    output_kv(Pair{String,Any}[
        "R" => pe.prices[:R], "rk" => pe.prices[:rk],
        "converged" => pe.converged, "iterations" => pe.iterations,
    ]; format=format, output=_per_var_output_path(output, "diagnostics"),
       title="Bewley Banks PE", key="bewley_banks_pe")
    output_result(_bewley_policy_df(sys, pe); format=Symbol(format), output=output,
                  title="Bewley Banks PE Policy", key="bewley_banks_pe_policy")
    return pe
end

function _dsge_bank_steady_state(; n_n::Int=25, n_xi::Int=3, n_min::Float64=0.05,
                                  n_max::Float64=8.0, beta::Float64=0.99, sigma::Float64=0.95,
                                  lambda::Float64=0.20, r::Float64=1.01, z::Float64=0.25,
                                  alpha::Float64=0.33, r_lo::Float64=NaN, r_hi::Float64=NaN,
                                  tol::Float64=1e-4, max_iter::Int=24,
                                  output::String="", format::String="table")
    sys = _intermediary_sys(; n_n, n_xi, n_min, n_max, beta, sigma, lambda, R=r, z=z, alpha=alpha)
    rb = (isfinite(r_lo) && isfinite(r_hi)) ? (r_lo, r_hi) : nothing
    rb !== nothing && r_lo >= r_hi && throw(CliError("usage/invalid",
        "--r-lo must be < --r-hi"))
    ss = MacroEconometricModels.intermediary_steady_state(sys; r_bounds=rb, tol=tol,
                                                           max_iter=max_iter)
    ag = ss.aggregates
    pr = ss.prices
    output_kv(Pair{String,Any}[
        "R" => pr[:R], "rk" => pr[:rk],
        "L" => ag[:L], "N" => ag[:N], "B" => ag[:B],
        "leverage" => ag[:leverage], "Y" => ag[:Y],
        "excess_demand" => ss.excess_demand,
        "converged" => ss.converged, "iterations" => ss.iterations,
    ]; format=format, output=output, title="Bewley Banks Steady State",
       key="bewley_banks_steady_state")
    output_result(_bewley_policy_df(sys, ss); format=Symbol(format),
                  output=_per_var_output_path(output, "policy"),
                  title="Bewley Banks SS Policy", key="bewley_banks_steady_state_policy")
    return ss
end

function _dsge_bank_transition(; z_path::String="", n_n::Int=25, n_xi::Int=3,
                                z::Float64=0.25,
                                output::String="", format::String="table")
    isempty(z_path) && throw(CliError("usage/missing-option",
        "dsge bank transition requires --z-path <csv>"))
    Z = _load_positive_path(z_path; min_length=2, name="Z_path")
    sys = _intermediary_sys(; n_n, n_xi, z=z)
    ss = MacroEconometricModels.intermediary_steady_state(sys)
    tr = MacroEconometricModels.intermediary_mit(ss, Z)
    n = length(tr.Z)
    output_result(DataFrame(t=1:n, Z=tr.Z, L=tr.L, Y=tr.Y, K=tr.K, rk=tr.rk);
                  format=Symbol(format), output=output,
                  title="Bewley Banks Transition", key="bewley_banks_transition")
    output_kv(Pair{String,Any}["method" => string(tr.method), "converged" => tr.converged];
              format=format, output=_per_var_output_path(output, "diagnostics"),
              title="Bewley Banks Transition Diagnostics",
              key="bewley_banks_transition_diagnostics")
    return tr
end

function _dsge_bank_irf(; horizon::Int=20, shock_size::Float64=0.01, persist::Float64=0.5,
                         n_n::Int=25, n_xi::Int=3, z::Float64=0.25,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    horizon >= 2 || throw(CliError("usage/invalid", "--horizon must be ≥ 2"))
    sys = _intermediary_sys(; n_n, n_xi, z=z)
    ss = MacroEconometricModels.intermediary_steady_state(sys)
    ir = irf(ss, horizon; shock_size=shock_size, persist=persist)
    _maybe_plot(ir; plot=plot, plot_save=plot_save)
    return _output_family_irf(ir; format=format, output=output,
                              title="Bewley Banks IRF", key_prefix="bewley_banks_irf")
end

