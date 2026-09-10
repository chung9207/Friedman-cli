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

# Estimate commands: var, bvar, lp, arima, gmm, smm, static, dynamic, gdfm, arch, garch, egarch, gjr_garch, sv, fastica, ml, favar, sdfm, reg, iv, logit, probit, preg, piv, plogit, pprobit, ologit, oprobit, mlogit

# C044: kebab primary CLI name; snake kept as hidden alias where renamed
const _VOL_CLI_NAMES = Dict("gjr_garch" => ("gjr-garch", ["gjr_garch"]))

"""Generate CommandSpecs for the volatility family — `:estimate` and `:forecast`
ONLY. The predict/residuals leaves are registered by fitted.jl (#145): wiring
those verbs here would double-register leaves fitted.jl owns, and registration
dedup is last-wins, so the W3 table declarations would silently flip."""
function _vol_specs(verb::Symbol)::Vector{CommandSpec}
    handlers = if verb === :estimate
        _VOL_ESTIMATE_HANDLERS
    elseif verb === :forecast
        _VOL_FORECAST_HANDLERS
    else
        error("_vol_specs handles only :estimate|:forecast (fitted.jl owns predict/residuals); got :$verb")
    end
    with_horizons = verb === :forecast
    order_opts = Dict{Symbol,Vector{OptionSpec}}(
        :q_only => [OptionSpec(name="q", type=Int, default=1, description="ARCH order")],
        :pq => [
            OptionSpec(name="p", type=Int, default=1, description="GARCH order"),
            OptionSpec(name="q", type=Int, default=1, description="ARCH order"),
        ],
        :sv => [OptionSpec(name="draws", short="n", type=Int, default=5000, description="MCMC draws")],
    )
    # historical option order: output before format, then plot-save
    out_opts = [
        OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
        OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
        OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file"),
    ]
    flags = [FlagSpec(name="plot", description="Open interactive plot in browser")]
    specs = CommandSpec[]
    for vol in VOL_MODELS
        opts = OptionSpec[
            OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
            order_opts[vol.order]...,
        ]
        if with_horizons
            push!(opts, OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"))
        end
        # W11/#113: only the three estimators that actually take a conditional distribution
        # upstream get --dist. arch/sv have no `dist` kwarg at all, so declaring it there
        # would be an option no handler path can honour (the #85 rule).
        if vol.supports_dist && (verb === :estimate || verb === :forecast)
            push!(opts, OptionSpec(name="dist", type=String, default="normal",
                                   choices=["normal", "student", "ged"],
                                   description="Conditional distribution of the innovations"))
        end
        append!(opts, out_opts)
        # predict/residuals historically omit p/q/draws from schema (defaults only)
        if verb === :predict || verb === :residuals
            opts = OptionSpec[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index"),
                OUTPUT_OPTIONS...,
            ]
        end
        cli_name, aliases = get(_VOL_CLI_NAMES, vol.name, (vol.name, String[]))
        label = vol.label(1, 1)
        # W3/#138: the four verbs share the shared.jl emitters, so their keys are
        # `<vol.name>_<what>` — `<label>` (which carries p/q) stays in the title only.
        tables = if verb === :estimate
            t = TableSpec[TableSpec(name=Symbol("$(vol.name)_coefficients"),
                    description="$label parameter estimates with standard errors, z-statistics and p-values")]
            # Only emitted when --dist selects a non-Gaussian innovation law; the shape
            # parameter is estimated jointly but lives outside coef(model).
            vol.supports_dist && push!(t, TableSpec(name=:conditional_distribution,
                    description="Estimated shape parameter of the non-Gaussian innovation distribution (--dist student|ged only)"))
            t
        elseif verb === :forecast
            [TableSpec(name=Symbol("$(vol.name)_volatility_forecast"),
                       description="Per-horizon forecast conditional variance and volatility")]
        elseif verb === :predict
            [TableSpec(name=Symbol("$(vol.name)_conditional_variance"),
                       description="In-sample conditional variance and volatility, one row per period")]
        else
            [TableSpec(name=Symbol("$(vol.name)_standardized_residuals"),
                       description="Standardized residuals, one row per period")]
        end
        push!(specs, CommandSpec(
            path=[string(verb), cli_name],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=opts,
            flags=flags,
            tables=tables,
            category=string(verb),
            aliases=aliases,
            handler=wrap_legacy(handlers[vol.name]),
        ))
    end
    return specs
end

function estimate_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["estimate", "var"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
                OptionSpec(name="trend", type=String, default="constant", description="none|constant|trend|both"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:var_coefficients,
                              description="Tidy VAR coefficient table: one row per equation x term with SEs, t-stats, p-values and CIs"),
                    TableSpec(name=:information_criteria,
                              description="AIC / BIC / HQC and the log-likelihood of the fitted VAR")],
            category="estimate",
            handler=wrap_legacy(_estimate_var),
        ),
        CommandSpec(
            path=["estimate", "bvar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
                OptionSpec(name="prior", type=String, default="minnesota", description="Prior type: minnesota"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws"),
                OptionSpec(name="sampler", type=String, default="direct", description="direct|gibbs"),
                OptionSpec(name="method", type=String, default="mean", description="mean|median (posterior extraction)"),
                OptionSpec(name="hyperopt", type=String, default="glp",
                           description="Minnesota hyperparameter selection: glp|grid",
                           choices=["glp", "grid"]),
                OptionSpec(name="config", type=String, default="", description="TOML config for prior hyperparameters"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:bvar_coefficients,
                              description="Posterior mean (or median) BVAR coefficients, one row per equation x regressor"),
                    TableSpec(name=:minnesota_hyperparameters,
                              description="Selected Minnesota hyperparameters and, for --hyperopt glp, the optimizer diagnostics"),
                    TableSpec(name=:information_criteria,
                              description="AIC / BIC / HQC and the log-likelihood of the posterior point model")],
            category="estimate",
            handler=wrap_legacy(_estimate_bvar),
        ),
        CommandSpec(
            path=["estimate", "qreg"],
            summary="Quantile regression (Koenker-Bassett)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable (default: first numeric column)"),
                OptionSpec(name="tau", type=String, default="0.5", description="Quantile(s) in (0,1); one value or a comma-list"),
                OptionSpec(name="se", type=String, default="iid", description="Standard errors: iid|robust|boot", choices=["iid","robust","boot"]),
                OptionSpec(name="n-boot", type=Int, default=500, description="Bootstrap replications for --se boot"),
                OptionSpec(name="alpha", type=Float64, default=0.05, description="Significance level for the CI, in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:quantile_regression_coefficients,
                              description="One row per quantile x term with standard errors, p-values and confidence bounds"),
                    TableSpec(name=:quantile_fit_diagnostics,
                              description="Per-quantile objective value, pseudo R-squared and convergence flag")],
            category="estimate",
            handler=wrap_legacy(_estimate_qreg),
        ),
        CommandSpec(
            path=["estimate", "rdd"],
            summary="Regression discontinuity (Calonico-Cattaneo-Titiunik)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable (default: first numeric column)"),
                OptionSpec(name="running", type=String, default="", description="Running/forcing variable (default: next numeric column)"),
                OptionSpec(name="fuzzy", type=String, default="", description="Treatment column for a FUZZY design (default: sharp)"),
                OptionSpec(name="cutoff", type=Float64, default=0.0, description="Threshold on the running variable"),
                OptionSpec(name="bandwidth", type=Float64, default=0.0, description="Main bandwidth h (0 = CCT auto-selection)"),
                OptionSpec(name="bias-bandwidth", type=Float64, default=0.0, description="Bias bandwidth b (0 = CCT auto-selection)"),
                OptionSpec(name="kernel", type=String, default="triangular", description="triangular|epanechnikov|uniform", choices=["triangular","epanechnikov","uniform"]),
                OptionSpec(name="order", type=Int, default=1, description="Local polynomial order (>= 1)"),
                OptionSpec(name="level", type=Float64, default=0.95, description="Confidence level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:rdd_treatment_effect,
                              description="Conventional, bias-corrected and robust treatment effects with standard errors and CIs"),
                    TableSpec(name=:rdd_settings_diagnostics,
                              description="Design, cutoff, kernel, bandwidths, effective counts and the robust z and p-value")],
            category="estimate",
            handler=wrap_legacy(_estimate_rdd),
        ),
        CommandSpec(
            path=["estimate", "tvpvar"],
            summary="TVP-VAR with stochastic volatility (Primiceri 2005)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="Retained Gibbs draws"),
                OptionSpec(name="burnin", type=Int, default=1000, description="Burn-in sweeps discarded"),
                OptionSpec(name="thin", type=Int, default=1, description="Keep every k-th draw"),
                OptionSpec(name="n-train", type=Int, default=0, description="Training sample used to calibrate priors"),
                OptionSpec(name="k-q", type=Float64, default=0.01, description="Coefficient random-walk prior scale (> 0)"),
                OptionSpec(name="k-s", type=Float64, default=0.1, description="Covariance random-walk prior scale (> 0)"),
                OptionSpec(name="k-w", type=Float64, default=0.01, description="Log-volatility random-walk prior scale (> 0)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-tvp", description="Hold coefficients constant (drop the time variation)"),
                   FlagSpec(name="no-sv", description="Hold volatilities constant (drop stochastic volatility)")],
            tables=[TableSpec(name=:tvp_var_stochastic_volatility_path_posterior_sd_68_band,
                              description="Posterior mean and 16/50/84 quantiles of the stochastic-volatility path, one row per period x variable"),
                    TableSpec(name=:tvp_var_specification,
                              description="Lags, variable count, effective sample, training sample, draws and the tvp/sv switches")],
            category="estimate",
            handler=wrap_legacy(_estimate_tvpvar),
        ),
        CommandSpec(
            path=["estimate", "mfvar"],
            summary="Mixed-frequency VAR (Schorfheide-Song 2015)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="CSV at the HIGH frequency; low-frequency series blank between observations")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order"),
                OptionSpec(name="low-freq", type=String, default="", description="1-based indices of low-frequency columns (default: those with gaps)"),
                OptionSpec(name="freq-ratio", type=Int, default=3, description="High- per low-frequency periods (3 = monthly/quarterly)"),
                OptionSpec(name="aggregation", type=String, default="growth", description="stock|flow|average|growth (one, or one per low-freq series)"),
                OptionSpec(name="draws", short="n", type=Int, default=1000, description="Retained Gibbs draws"),
                OptionSpec(name="burnin", type=Int, default=500, description="Burn-in sweeps discarded"),
                OptionSpec(name="prior", type=String, default="minnesota", description="minnesota|diffuse", choices=["minnesota","diffuse"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:mf_var_latent_high_frequency_path_68_credible_band,
                              description="Posterior mean and 16/50/84 quantiles of the latent high-frequency path"),
                    TableSpec(name=:mf_var_specification,
                              description="Lags, variable count, high-frequency length, low-frequency columns, ratio and aggregation")],
            category="estimate",
            handler=wrap_legacy(_estimate_mfvar),
        ),
        CommandSpec(
            path=["estimate", "lp"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="method", type=String, default="standard", description="standard|iv|smooth|state|propensity|robust"),
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="horizons", type=Int, default=20, description="IRF horizon"),
                OptionSpec(name="control-lags", type=Int, default=4, description="Number of control lags"),
                OptionSpec(name="vcov", type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
                OptionSpec(name="instruments", type=String, default="", description="Path to instruments CSV (iv only)"),
                OptionSpec(name="knots", type=Int, default=3, description="Number of B-spline knots (smooth only)"),
                OptionSpec(name="lambda", type=Float64, default=0.0, description="Smoothing penalty, 0=auto CV (smooth only)"),
                OptionSpec(name="state-var", type=Int, default=nothing, description="State variable index (state only)"),
                OptionSpec(name="gamma", type=Float64, default=1.5, description="Transition steepness (state only)"),
                OptionSpec(name="transition", type=String, default="logistic", description="logistic|exponential|indicator (state only)"),
                OptionSpec(name="treatment", type=Int, default=1, description="Treatment variable index (propensity/robust only)"),
                OptionSpec(name="score-method", type=String, default="logit", description="logit|probit (propensity/robust only)"),
                # W10/#112 weak-instrument-robust LP-IV (iv only). Both are OFF by default so
                # the existing `estimate lp --method iv` envelope is unchanged; the AR band in
                # particular inverts a test over a grid at every horizon × response and is far
                # from free.
                OptionSpec(name="mop-tau", type=Float64, default=0.10,
                           description="MOP worst-case relative-bias target: 0.05|0.10|0.20|0.30 (iv, with --mop-f)"),
                OptionSpec(name="mop-bandwidth", type=Int, default=0,
                           description="HAC lag length for the MOP effective F; 0 = auto (iv, with --mop-f)"),
                OptionSpec(name="ar-level", type=Float64, default=0.95,
                           description="Coverage for the AR bands (iv, with --ar-bands)"),
                OptionSpec(name="ar-grid", type=Int, default=401,
                           description="Grid points per horizon×response when inverting the AR test (iv, with --ar-bands)"),
                OptionSpec(name="ar-span", type=Float64, default=20.0,
                           description="AR search half-width in 2SLS standard errors (iv, with --ar-bands)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[
                FlagSpec(name="mop-f", description="Report the Montiel Olea-Pflueger effective first-stage F (LP-IV)"),
                FlagSpec(name="ar-bands", description="Report weak-instrument-robust Anderson-Rubin IRF bands (LP-IV)")
            ],
            tables=[TableSpec(name=:lp_coefficients,
                              description="Impulse responses by horizon with standard errors and t-statistics, one column per response variable"),
                    TableSpec(name=:lp_coefficients_expansion,
                              description="Expansion-regime responses (--method state only)"),
                    TableSpec(name=:lp_coefficients_recession,
                              description="Recession-regime responses (--method state only)"),
                    TableSpec(name=:lp_estimation_summary,
                              description="Effective sample size, covariance estimator and, for LP-IV, the first-stage F"),
                    TableSpec(name=:montiel_olea_pflueger_effective_f,
                              description="Montiel Olea-Pflueger effective first-stage F against its critical value (--mop-f)"),
                    TableSpec(name=:lp_iv_anderson_rubin_bands,
                              description="Weak-instrument-robust Anderson-Rubin bands per horizon x response (--ar-bands)")],
            category="estimate",
            handler=wrap_legacy(_estimate_lp),
        ),
        CommandSpec(
            path=["estimate", "arima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=nothing, description="AR order (default: auto selection)"),
                OptionSpec(name="d", type=Int, default=0, description="Differencing order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="max-p", type=Int, default=5, description="Max AR order for auto selection"),
                OptionSpec(name="max-d", type=Int, default=2, description="Max differencing order for auto selection"),
                OptionSpec(name="max-q", type=Int, default=5, description="Max MA order for auto selection"),
                OptionSpec(name="criterion", type=String, default="bic", description="aic|bic (for auto selection)"),
                OptionSpec(name="method", short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:arima_coefficients,
                              description="AR / MA / constant estimates with standard errors, z-statistics and p-values"),
                    TableSpec(name=:information_criteria,
                              description="AIC, BIC and the log-likelihood of the fitted ARIMA")],
            category="estimate",
            handler=wrap_legacy(_estimate_arima),
        ),
        CommandSpec(
            path=["estimate", "arfima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=0, description="AR order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="method", short="m", type=String, default="css", description="css|mle (fractional-integration estimator)", choices=["css","mle"]),
                OptionSpec(name="d0", type=Float64, default=nothing, description="Starting value for d (default: GPH pre-estimate)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:arfima_coefficients,
                              description="Constant, fractional-integration d, AR and MA estimates with standard errors"),
                    TableSpec(name=:arfima_diagnostics,
                              description="Estimated d with its standard error, log-likelihood, AIC/BIC and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_arfima),
        ),
        CommandSpec(
            path=["estimate", "gmm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML config for moment conditions and instruments"),
                OptionSpec(name="weighting", short="w", type=String, default="twostep", description="identity|optimal|twostep|iterated"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:gmm_estimates,
                              description="GMM parameter estimates with standard errors (written only when --output is given)")],
            category="estimate",
            handler=wrap_legacy(_estimate_gmm),
        ),
        CommandSpec(
            path=["estimate", "static"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
                OptionSpec(name="criterion", type=String, default="ic1", description="ic1|ic2|ic3 for auto selection"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:scree_data_eigenvalues_variance_shares,
                              description="Eigenvalue, explained-variance and cumulative-variance share per component"),
                    TableSpec(name=:factor_loadings,
                              description="Estimated factor loadings, one row per observed variable")],
            category="estimate",
            handler=wrap_legacy(_estimate_static),
        ),
        CommandSpec(
            path=["estimate", "dynamic"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
                OptionSpec(name="factor-lags", short="p", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="method", type=String, default="twostep", description="twostep|em"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:dynamic_factor_loadings,
                              description="Estimated dynamic-factor loadings, one row per observed variable")],
            category="estimate",
            handler=wrap_legacy(_estimate_dynamic),
        ),
        CommandSpec(
            path=["estimate", "gdfm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
                OptionSpec(name="dynamic-rank", short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
                OptionSpec(name="spectral", type=String, default="lag-window", description="Spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:gdfm_common_variance_shares,
                              description="Share of each variable variance explained by the common component")],
            category="estimate",
            handler=wrap_legacy(_estimate_gdfm),
        ),
        # Volatility 20-plex (estimate side): generated from VOL_MODELS
        _vol_specs(:estimate)...,
        # C064a: univariate GARCH variants (grouped with the volatility family).
        # Estimate-only leaves — result types are NOT in MEMs _COEF_TABLE_TYPES, so
        # the coefficient table is hand-built (the documented C051 exception).
        CommandSpec(
            path=["estimate", "igarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:igarch_coefficients,
                              description="IGARCH parameter estimates with standard errors, z-statistics and p-values"),
                    TableSpec(name=:igarch_diagnostics,
                              description="Log-likelihood, AIC/BIC, persistence, convergence and iteration count")],
            category="estimate",
            handler=wrap_legacy(_estimate_igarch),
        ),
        CommandSpec(
            path=["estimate", "cgarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:cgarch_coefficients,
                              description="Component-GARCH permanent/transitory parameter estimates with standard errors"),
                    TableSpec(name=:cgarch_diagnostics,
                              description="Log-likelihood, AIC/BIC, transitory persistence and unconditional variance")],
            category="estimate",
            handler=wrap_legacy(_estimate_cgarch),
        ),
        CommandSpec(
            path=["estimate", "aparch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="fix-delta", type=Float64, default=nothing, description="Pin power δ (>0); default estimates it"),
                OptionSpec(name="fix-gamma", type=Float64, default=nothing, description="Pin leverage γ ∈ (-1,1); default estimates it"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:aparch_coefficients,
                              description="APARCH parameter estimates including the power delta and asymmetry gamma"),
                    TableSpec(name=:aparch_diagnostics,
                              description="Log-likelihood, AIC/BIC, persistence, estimated delta and parameter count")],
            category="estimate",
            handler=wrap_legacy(_estimate_aparch),
        ),
        CommandSpec(
            path=["estimate", "figarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH β(L) order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH φ(L) order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional-integration order d ∈ (0,1)"),
                OptionSpec(name="truncation", type=Int, default=1000, description="ARCH(∞) truncation lag"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution (Gaussian QMLE)", choices=["normal"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:figarch_coefficients,
                              description="FIGARCH parameter estimates including the fractional-integration d"),
                    TableSpec(name=:figarch_diagnostics,
                              description="Log-likelihood, AIC/BIC, persistence, d, truncation lag and negative-lambda count")],
            category="estimate",
            handler=wrap_legacy(_estimate_figarch),
        ),
        CommandSpec(
            path=["estimate", "fiegarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH β(L) order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH φ(L) order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional-integration order d ∈ (0,1)"),
                OptionSpec(name="truncation", type=Int, default=1000, description="MA(∞) truncation lag"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution (Gaussian QMLE)", choices=["normal"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:fiegarch_coefficients,
                              description="FIEGARCH parameter estimates including the fractional-integration d"),
                    TableSpec(name=:fiegarch_diagnostics,
                              description="Log-likelihood, AIC/BIC, persistence, d and the truncation lag")],
            category="estimate",
            handler=wrap_legacy(_estimate_fiegarch),
        ),
        CommandSpec(
            path=["estimate", "garch-midas"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="High-frequency return series column (1-based)"),
                OptionSpec(name="m-freq", type=Int, default=0, description="High-frequency observations per low-frequency block (required)"),
                OptionSpec(name="k", type=Int, default=12, description="Number of MIDAS lags (K ≥ 2)"),
                OptionSpec(name="rv", type=String, default="realized", description="Long-run driver: realized (from returns) | macro (exogenous)", choices=["realized","macro"]),
                OptionSpec(name="span", type=String, default="fixed", description="τ span: fixed (per block) | rolling (rolling RV)", choices=["fixed","rolling"]),
                OptionSpec(name="config", type=String, default="", description="TOML with [garch_midas] x_lf = [...] (required for --rv macro)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:garch_midas_coefficients,
                              description="GARCH-MIDAS short-run and long-run (MIDAS weight) parameter estimates"),
                    TableSpec(name=:garch_midas_diagnostics,
                              description="Log-likelihood, AIC/BIC, variance ratio, K, m_freq and the block count")],
            category="estimate",
            handler=wrap_legacy(_estimate_garch_midas),
        ),
        # C064b: multivariate GARCH (CCC / DCC-cDCC / scalar-diagonal BEKK).
        # Multivariate — no --column; input is the full numeric matrix (T×n).
        CommandSpec(
            path=["estimate", "ccc"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p for the univariate margins"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q for the univariate margins"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ccc_garch_conditional_correlation,
                              description="Constant conditional-correlation matrix, series x series"),
                    TableSpec(name=:ccc_garch_diagnostics,
                              description="Log-likelihood, AIC/BIC, series and observation counts, convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_ccc),
        ),
        CommandSpec(
            path=["estimate", "dcc"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p for the univariate margins"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q for the univariate margins"),
                OptionSpec(name="correction", type=String, default="none", description="DCC targeting correction: none | aielli (cDCC)", choices=["none","aielli"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:dcc_garch_conditional_correlation,
                              description="Conditional-correlation matrix at the last observation, series x series"),
                    TableSpec(name=:dcc_garch_dynamics_parameters,
                              description="Second-stage DCC dynamics parameters with standard errors"),
                    TableSpec(name=:dcc_garch_diagnostics,
                              description="Log-likelihood, AIC/BIC, correction scheme and correlation persistence")],
            category="estimate",
            handler=wrap_legacy(_estimate_dcc),
        ),
        CommandSpec(
            path=["estimate", "bekk"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="kind", type=String, default="scalar", description="BEKK(1,1) parameterization: scalar | diagonal", choices=["scalar","diagonal"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:bekk_conditional_correlation,
                              description="Conditional-correlation matrix at the last observation, series x series"),
                    TableSpec(name=:bekk_dynamics_parameters,
                              description="BEKK dynamics parameters with standard errors"),
                    TableSpec(name=:bekk_diagnostics,
                              description="Log-likelihood, AIC/BIC, series and observation counts, and the BEKK parameterisation")],
            category="estimate",
            handler=wrap_legacy(_estimate_bekk),
        ),
        # C067a: penalized (lasso/ridge/elastic-net) & limited-dependent (robust/tobit)
        # cross-section regression. All share the `_load_reg_data` (y, X) loader — X = all
        # numeric columns except --dep, no constant prepended (same as `estimate reg`).
        CommandSpec(
            path=["estimate", "lasso"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="lambda", type=String, default="auto", description="L1 penalty: auto (CV path) or a non-negative number"),
                OptionSpec(name="select", type=String, default="cv", description="Lambda selection rule: cv|aic|bic|ebic", choices=["cv","aic","bic","ebic"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:lasso_coefficients,
                              description="Intercept and slope estimates with an active-set (non-zero) indicator"),
                    TableSpec(name=:lasso_diagnostics,
                              description="Selected lambda, active-set size, R-squared, AIC/BIC/EBIC and the selection rule")],
            category="estimate",
            handler=wrap_legacy(_estimate_lasso),
        ),
        CommandSpec(
            path=["estimate", "ridge"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="lambda", type=String, default="auto", description="L2 penalty: auto (CV path) or a non-negative number"),
                OptionSpec(name="select", type=String, default="cv", description="Lambda selection rule: cv|aic|bic|ebic", choices=["cv","aic","bic","ebic"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ridge_coefficients,
                              description="Intercept and slope estimates with an active-set (non-zero) indicator"),
                    TableSpec(name=:ridge_diagnostics,
                              description="Selected lambda, active-set size, R-squared, AIC/BIC/EBIC and the selection rule")],
            category="estimate",
            handler=wrap_legacy(_estimate_ridge),
        ),
        CommandSpec(
            path=["estimate", "elastic-net"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="alpha", type=Float64, default=0.5, description="L1/L2 mixing in [0,1] (1=lasso, 0=ridge)"),
                OptionSpec(name="lambda", type=String, default="auto", description="Penalty: auto (CV path) or a non-negative number"),
                OptionSpec(name="select", type=String, default="cv", description="Lambda selection rule: cv|aic|bic|ebic", choices=["cv","aic","bic","ebic"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:elastic_net_coefficients,
                              description="Intercept and slope estimates with an active-set (non-zero) indicator"),
                    TableSpec(name=:elastic_net_diagnostics,
                              description="Mixing alpha, selected lambda, active-set size, R-squared and information criteria")],
            category="estimate",
            handler=wrap_legacy(_estimate_elastic_net),
        ),
        CommandSpec(
            path=["estimate", "robust"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="psi", type=String, default="huber", description="ψ (weight) function: huber|bisquare", choices=["huber","bisquare"]),
                OptionSpec(name="method", type=String, default="m", description="Estimator: m|mm (MM = high-breakdown)", choices=["m","mm"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:robust_regression_coefficients,
                              description="M-estimator coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:robust_regression_diagnostics,
                              description="psi function, estimator, robust scale, robust R-squared, tuning and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_robust),
        ),
        CommandSpec(
            path=["estimate", "tobit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="lower", type=Float64, default=0.0, description="Lower censoring bound"),
                OptionSpec(name="upper", type=Float64, default=Inf, description="Upper censoring bound (default: none)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:tobit_coefficients,
                              description="Censored-regression coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:tobit_diagnostics,
                              description="sigma, log-likelihood, AIC/BIC, censoring bounds and left/right censored counts")],
            category="estimate",
            handler=wrap_legacy(_estimate_tobit),
        ),
        # C067b: truncated-normal regression (Hausman-Wise). Mirrors `estimate tobit` but
        # the sample is truncated (no censored mass): every y must lie strictly inside
        # (lower, upper) or MEMs throws → mapped to data/invalid.
        CommandSpec(
            path=["estimate", "truncreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="lower", type=Float64, default=0.0, description="Lower truncation bound"),
                OptionSpec(name="upper", type=Float64, default=Inf, description="Upper truncation bound (default: none)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:truncated_regression_coefficients,
                              description="Truncated-normal coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:truncated_regression_diagnostics,
                              description="sigma and its SE, log-likelihood, AIC/BIC, bounds and the truncated count")],
            category="estimate",
            handler=wrap_legacy(_estimate_truncreg),
        ),
        # C067b: Heckman sample-selection model (two-step or FIML). Two equations —
        # outcome `--dep ~ --outcome-vars` observed only when the binary `--select`
        # indicator is 1, selection `--select ~ --select-vars` (probit). Include a `const`
        # column in each var list if you want an intercept (same no-auto-intercept
        # convention as `estimate reg`). Exclusion restriction: --select-vars should hold a
        # variable not in --outcome-vars (else MEMs warns and identification is nonlinear).
        CommandSpec(
            path=["estimate", "heckman"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Outcome variable column name (default: first numeric column)"),
                OptionSpec(name="select", type=String, default="", description="Binary selection-indicator column (0/1), required"),
                OptionSpec(name="outcome-vars", type=String, default="", description="Outcome-equation regressor columns, comma-separated (required)"),
                OptionSpec(name="select-vars", type=String, default="", description="Selection-equation regressor columns, comma-separated (required)"),
                OptionSpec(name="method", type=String, default="twostep", description="twostep (Heckit) | mle (FIML)", choices=["twostep","mle"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:heckman_coefficients_outcome_selection,
                              description="Outcome- and selection-equation coefficients in one tidy table"),
                    TableSpec(name=:heckman_diagnostics,
                              description="rho, sigma and lambda with SEs, log-likelihood, AIC/BIC and selected/total counts")],
            category="estimate",
            handler=wrap_legacy(_estimate_heckman),
        ),
        # ── C066: state-space + nonparametric estimation (M5c) ──────────────
        # StateSpaceModel / KernelDensity / KernelRegression / LowessFit are NOT registered
        # in MEMs `_COEF_TABLE_TYPES`, so every table is hand-built (documented C051 exception,
        # like io/mgarch/sur). All MEMs calls are wrapped → typed CliError via
        # `_garch_variant_error` (never an uncaught exit-1 on bad input).
        CommandSpec(
            path=["estimate", "statespace"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="1-based numeric column to model"),
                OptionSpec(name="model", type=String, default="local-level", description="local-level | local-linear-trend (ignored with --config)", choices=["local-level","local-linear-trend"]),
                OptionSpec(name="init-mode", type=String, default="kappa", description="Kalman initialization: kappa | diffuse (--config uses [statespace] init_mode)", choices=["kappa","diffuse"]),
                OptionSpec(name="kappa", type=Float64, default=1e6, description="Large-variance diffuse-init constant (init-mode=kappa)"),
                # A [statespace] section switches to the GENERAL (multivariate, fixed-matrix)
                # system and supersedes --model/--init-mode/--column.
                OptionSpec(name="config", type=String, default="", description="TOML with [statespace] Z/H/T/Q (+ d, c, R, a1, P1, init_mode) for a general system"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:state_space_hyper_parameters,
                              description="Estimated hyper-parameters of a canned local-level / local-linear-trend system"),
                    TableSpec(name=:state_space_system_general,
                              description="Dimensions of each system matrix for a --config general system"),
                    TableSpec(name=:state_space_diagnostics,
                              description="Model kind, log-likelihood, state and period counts, and the estimation method")],
            category="estimate",
            handler=wrap_legacy(_estimate_statespace),
        ),
        CommandSpec(
            path=["estimate", "tvp"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="init-mode", type=String, default="kappa", description="Kalman initialization: kappa | diffuse", choices=["kappa","diffuse"]),
                OptionSpec(name="kappa", type=Float64, default=1e6, description="Large-variance diffuse-init constant (init-mode=kappa)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-intercept", description="Do NOT prepend a time-varying intercept coefficient")],
            tables=[TableSpec(name=:tvp_hyper_parameters,
                              description="Estimated state-space hyper-parameters of the TVP regression"),
                    TableSpec(name=:tvp_coefficient_paths,
                              description="Smoothed time-varying coefficient path, one row per period x coefficient"),
                    TableSpec(name=:tvp_diagnostics,
                              description="Log-likelihood, convergence, coefficient count and whether an intercept was included")],
            category="estimate",
            handler=wrap_legacy(_estimate_tvp),
        ),
        CommandSpec(
            path=["estimate", "kde"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="1-based numeric column"),
                OptionSpec(name="kernel", type=String, default="gaussian", description="gaussian | epanechnikov | triangular | uniform", choices=["gaussian","epanechnikov","triangular","uniform"]),
                OptionSpec(name="bw", type=String, default="silverman", description="Bandwidth: silverman | sj | a positive number"),
                OptionSpec(name="npoints", type=Int, default=512, description="Number of grid points"),
                OptionSpec(name="cut", type=Float64, default=3.0, description="Grid extends cut·h beyond the data range each side"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:kernel_density_estimate,
                              description="Evaluation grid and the estimated density at each grid point"),
                    TableSpec(name=:kernel_density_diagnostics,
                              description="Kernel, bandwidth rule, selected bandwidth and the observation count")],
            category="estimate",
            handler=wrap_legacy(_estimate_kde),
        ),
        CommandSpec(
            path=["estimate", "kernel-reg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Response variable column name (default: first numeric column)"),
                OptionSpec(name="indep", type=String, default="", description="Single predictor column name (required)"),
                OptionSpec(name="method", type=String, default="ll", description="nw (Nadaraya-Watson) | ll (local linear) | lp (local polynomial)", choices=["nw","ll","lp"]),
                OptionSpec(name="degree", type=Int, default=1, description="Local-polynomial degree (method=lp)"),
                OptionSpec(name="bw", type=String, default="cv", description="Bandwidth: cv | rot | a positive number"),
                OptionSpec(name="kernel", type=String, default="gaussian", description="gaussian | epanechnikov | triangular | uniform", choices=["gaussian","epanechnikov","triangular","uniform"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:kernel_regression_fit,
                              description="Evaluation grid with the fitted conditional mean and its standard error"),
                    TableSpec(name=:kernel_regression_diagnostics,
                              description="Method, polynomial degree, kernel, bandwidth rule and selected bandwidth")],
            category="estimate",
            handler=wrap_legacy(_estimate_kernel_reg),
        ),
        CommandSpec(
            path=["estimate", "lowess"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Response variable column name (default: first numeric column)"),
                OptionSpec(name="indep", type=String, default="", description="Single predictor column name (required)"),
                OptionSpec(name="frac", type=Float64, default=0.6667, description="Smoother span f ∈ (0,1] (fraction of points per window)"),
                OptionSpec(name="iter", type=Int, default=3, description="Number of bisquare robustifying passes"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:lowess_fit, description="Sorted x with the locally-weighted smoothed fit"),
                    TableSpec(name=:lowess_diagnostics,
                              description="Smoothing span, robustness iterations and the observation count")],
            category="estimate",
            handler=wrap_legacy(_estimate_lowess),
        ),
        # ── C062a: cointegrating regression (FMOLS / CCR / DOLS) ──────────────
        # `estimate cointreg` (single-equation, `_load_reg_data`) + `estimate xtcointreg`
        # (panel, `_load_panel_reg`). Hand-built coef table (C051 exception). NOTE the trend
        # vocabulary is cointreg-specific: none|const|linear (do NOT share the OptionSpec —
        # PMG uses :constant, ARDL uses :const/:trend). Dual-type --bandwidth/--leads/--lags
        # are declared String and parsed in-handler (`_parse_cointreg_*`).
        CommandSpec(
            path=["estimate", "cointreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent (levels) column name (default: first numeric column)"),
                OptionSpec(name="method", type=String, default="fmols", description="fmols|ccr|dols", choices=["fmols","ccr","dols"]),
                OptionSpec(name="trend", type=String, default="const", description="Deterministics: none|const|linear", choices=["none","const","linear"]),
                OptionSpec(name="kernel", type=String, default="bartlett", description="HAC kernel: bartlett|parzen|qs|tukey-hanning", choices=["bartlett","parzen","qs","tukey-hanning"]),
                OptionSpec(name="bandwidth", type=String, default="andrews", description="andrews|nw94 or a fixed truncation lag (>=0)"),
                OptionSpec(name="leads", type=String, default="auto", description="DOLS leads: auto or a non-negative integer"),
                OptionSpec(name="lags", type=String, default="auto", description="DOLS lags: auto or a non-negative integer"),
                OptionSpec(name="ic", type=String, default="aic", description="DOLS lead/lag selection: aic|bic", choices=["aic","bic"]),
                OptionSpec(name="dols-se", type=String, default="lrv", description="DOLS standard errors: lrv|robust", choices=["lrv","robust"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:cointegrating_regression_coefficients,
                              description="FMOLS / CCR / DOLS long-run coefficients with long-run-variance standard errors"),
                    TableSpec(name=:cointegrating_regression_diagnostics,
                              description="Method, trend, kernel, bandwidth, omega_uv and sample dimensions")],
            category="estimate",
            handler=wrap_legacy(_estimate_cointreg),
        ),
        CommandSpec(
            path=["estimate", "xtcointreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="id-col", type=String, default="", description="Panel group id column (default: first column)"),
                OptionSpec(name="time-col", type=String, default="", description="Panel time column (default: second column)"),
                OptionSpec(name="dep", type=String, default="", description="Dependent panel variable (default: first variable)"),
                OptionSpec(name="indep", type=String, default="", description="Regressors, comma-separated (default: all other variables)"),
                OptionSpec(name="method", type=String, default="fmols", description="fmols|dols (no ccr for panels)", choices=["fmols","dols"]),
                OptionSpec(name="pooling", type=String, default="group", description="group (between) | pooled (within)", choices=["group","pooled"]),
                OptionSpec(name="trend", type=String, default="const", description="Per-unit deterministics: none|const|linear", choices=["none","const","linear"]),
                OptionSpec(name="kernel", type=String, default="bartlett", description="HAC kernel: bartlett|parzen|qs|tukey-hanning", choices=["bartlett","parzen","qs","tukey-hanning"]),
                OptionSpec(name="bandwidth", type=String, default="andrews", description="andrews|nw94 or a fixed truncation lag (>=0)"),
                OptionSpec(name="leads", type=String, default="auto", description="DOLS leads: auto or a non-negative integer"),
                OptionSpec(name="lags", type=String, default="auto", description="DOLS lags: auto or a non-negative integer"),
                OptionSpec(name="ic", type=String, default="aic", description="DOLS lead/lag selection: aic|bic", choices=["aic","bic"]),
                OptionSpec(name="dols-se", type=String, default="lrv", description="DOLS standard errors: lrv|robust", choices=["lrv","robust"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:panel_cointegrating_regression_coefficients,
                              description="Panel FMOLS/DOLS long-run coefficients (group-mean or pooled per --pooling)"),
                    TableSpec(name=:panel_cointegrating_regression_diagnostics,
                              description="Method, pooling, trend, kernel, unit count and per-unit sample lengths")],
            category="estimate",
            handler=wrap_legacy(_estimate_xtcointreg),
        ),
        # ── C062b: single-equation ARDL / NARDL (autoregressive distributed lag) ──
        # `estimate ardl` (linear ARDL, folds long-run + ECM speed of adjustment) +
        # `estimate nardl` (nonlinear/asymmetric ARDL, folds θ⁺/θ⁻ long-run + the cached
        # enlarged-k bounds decision). Both load y+X via `_load_reg_data` (no intercept
        # prepended — ARDL adds its own deterministics per PSS `--case`). ARDLModel/NARDLModel
        # are StatsAPI models but NOT in MEMs `_COEF_TABLE_TYPES` → hand-built tidy tables
        # (C051 exception). NOTE the trend vocab is ARDL-specific: none|const|trend (NOT the
        # cointreg none|const|linear, NOT PMG's :constant). `--p`/`--q` accept auto|int|list.
        CommandSpec(
            path=["estimate", "ardl"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent column name (default: first numeric column)"),
                OptionSpec(name="p", type=String, default="auto", description="AR order: auto or an integer ≥ 1"),
                OptionSpec(name="q", type=String, default="auto", description="DL order: auto, an integer (all regressors), or a comma-separated per-regressor list"),
                OptionSpec(name="max-p", type=Int, default=4, description="Max AR order for auto IC selection"),
                OptionSpec(name="max-q", type=Int, default=4, description="Max DL order for auto IC selection"),
                OptionSpec(name="ic", type=String, default="aic", description="Selection criterion: aic|bic", choices=["aic","bic"]),
                OptionSpec(name="case", type=Int, default=3, description="Pesaran-Shin-Smith deterministic case (1..5)"),
                OptionSpec(name="trend", type=String, default="none", description="Informational trend label: none|const|trend", choices=["none","const","trend"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ardl_coefficients,
                              description="Levels-form ARDL coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:ardl_long_run_coefficients,
                              description="Long-run level multipliers with delta-method standard errors"),
                    TableSpec(name=:ardl_diagnostics,
                              description="Selected p/q, case, trend, information criterion, fit statistics and the ECM speed alpha")],
            category="estimate",
            handler=wrap_legacy(_estimate_ardl),
        ),
        CommandSpec(
            path=["estimate", "nardl"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent column name (default: first numeric column)"),
                OptionSpec(name="asymmetric", type=String, default="all", description="'all' or comma-separated 1-based regressor indices to split into +/- partial sums"),
                OptionSpec(name="p", type=String, default="auto", description="AR order: auto or an integer ≥ 1"),
                OptionSpec(name="q", type=String, default="auto", description="DL order: auto, an integer (all split regressors), or a comma-separated list"),
                OptionSpec(name="max-p", type=Int, default=4, description="Max AR order for auto IC selection"),
                OptionSpec(name="max-q", type=Int, default=4, description="Max DL order for auto IC selection"),
                OptionSpec(name="ic", type=String, default="aic", description="Selection criterion: aic|bic", choices=["aic","bic"]),
                OptionSpec(name="case", type=Int, default=3, description="Pesaran-Shin-Smith deterministic case (1..5)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:nardl_coefficients,
                              description="Levels-form NARDL coefficients on the split (positive/negative) regressor set"),
                    TableSpec(name=:nardl_asymmetric_long_run_coefficients,
                              description="Asymmetric long-run multipliers theta+ and theta- with standard errors"),
                    TableSpec(name=:nardl_diagnostics,
                              description="Split dimensions, selected orders, fit statistics and the enlarged-k PSS bounds decision")],
            category="estimate",
            handler=wrap_legacy(_estimate_nardl),
        ),
        # ── C062c: dynamic heterogeneous-panel ARDL (PMG / MG / DFE) ──────────
        # `estimate pmg` fits Pooled Mean Group (common long-run θ), Mean Group, or Dynamic
        # Fixed Effects on a long-format panel via the shared `_load_panel_reg` loader; folds a
        # long-run θ table + a short-run/EC (φ) table + diagnostics. PMGModel is a StatsAPI model
        # but NOT in MEMs `_COEF_TABLE_TYPES` → hand-built tables (C051 exception). NOTE the trend
        # vocab is PMG-specific: none|constant|trend (`:constant` spelled out — NOT ARDL's :const).
        CommandSpec(
            path=["estimate", "pmg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="id-col", type=String, default="", description="Panel group id column (default: first column)"),
                OptionSpec(name="time-col", type=String, default="", description="Panel time column (default: second column)"),
                OptionSpec(name="dep", type=String, default="", description="Dependent panel variable (default: first variable)"),
                OptionSpec(name="indep", type=String, default="", description="Long-run regressors, comma-separated (default: all other variables)"),
                OptionSpec(name="method", type=String, default="pmg", description="pmg (pooled mean group) | mg (mean group) | dfe (dynamic fixed effects)", choices=["pmg","mg","dfe"]),
                OptionSpec(name="trend", type=String, default="constant", description="Per-unit EC deterministics: none|constant|trend", choices=["none","constant","trend"]),
                OptionSpec(name="p", type=Int, default=1, description="Autoregressive order (≥ 1)"),
                OptionSpec(name="q", type=Int, default=1, description="Distributed-lag order for all regressors (≥ 0)"),
                OptionSpec(name="maxiter", type=Int, default=100, description="PMG outer-loop max iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="PMG outer-loop convergence tolerance"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:panel_ardl_long_run_coefficients,
                              description="Pooled or averaged long-run coefficients theta with standard errors"),
                    TableSpec(name=:panel_ardl_short_run_ec_coefficients,
                              description="Error-correction speed phi and the short-run coefficient block"),
                    TableSpec(name=:panel_ardl_diagnostics,
                              description="Estimator, unit count, orders, phi with its SE, log-likelihood and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_pmg),
        ),
        # ── C062d: MIDAS mixed-frequency regression ──────────────────────────
        # `estimate midas` fits a (restricted) MIDAS / ADL-MIDAS / U-MIDAS regression of a
        # low-frequency target (--data, --column) on --k high-frequency lags of a single indicator
        # (--hf-data, --hf-column) aggregated through a parsimonious weight function. The only new
        # plumbing is the mixed-frequency `_load_midas_data` loader (5th shared-loader hardening:
        # BOTH inputs go through the hardened `load_univariate_series`, and the aligned
        # len(HF)>=m×len(LF) rule is a typed `data/shape`, leading ragged edge dropped). `MidasModel` is a StatsAPI model but NOT
        # in MEMs `_COEF_TABLE_TYPES` → hand-built weight-curve + coef tables (C051 exception).
        # `--horizon` (the direct h-step target y_{t+h-1}) was INERT upstream at ≤0.7.2
        # and is real since 0.7.3 (MEMs#574, adopted W10/#131). `forecast midas` exists.
        CommandSpec(
            path=["estimate", "midas"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to the low-frequency target CSV")],
            options=[
                OptionSpec(name="column", type=Int, default=1, description="Low-frequency target column in --data (1-based)"),
                OptionSpec(name="hf-data", type=String, default="", description="High-frequency indicator CSV (REQUIRED)"),
                OptionSpec(name="hf-column", type=Int, default=1, description="High-frequency indicator column in --hf-data (1-based)"),
                OptionSpec(name="m", type=Int, default=0, description="Frequency ratio HF/LF, e.g. 3 = monthly→quarterly (REQUIRED, ≥ 1)"),
                OptionSpec(name="k", type=Int, default=0, description="Number of high-frequency lags (REQUIRED, ≥ 1)"),
                OptionSpec(name="weights", type=String, default="expalmon", description="Weight scheme: expalmon|beta2|beta3|almon|umidas", choices=["expalmon","beta2","beta3","almon","umidas"]),
                OptionSpec(name="p-ar", type=Int, default=0, description="Autoregressive lags of the target (ADL-MIDAS, ≥ 0)"),
                OptionSpec(name="poly-degree", type=Int, default=2, description="Polynomial degree for --weights almon"),
                OptionSpec(name="horizon", type=Int, default=1, description="Direct forecast horizon h stored in the model (1 = nowcast)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="LBFGS iteration cap per NLS start"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:midas_weight_curve,
                              description="Estimated MIDAS lag weights, most recent lag first"),
                    TableSpec(name=:midas_coefficients,
                              description="MIDAS regression coefficients with standard errors"),
                    TableSpec(name=:midas_diagnostics,
                              description="Weight scheme, m/K/p_ar, horizon, fit statistics and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_midas),
        ),
        # ── C065a: SETAR (self-exciting threshold autoregression) ──
        # Two-regime SETAR via `estimate_setar` (Hansen 2000 threshold CI + attached
        # Hansen 1996 linearity test). Hand-built two-regime coef table + kv diagnostics
        # (ThresholdModel is NOT in MEMs `_COEF_TABLE_TYPES`). `--d` is a `Int|:auto`
        # sentinel; `--ci-level` MUST be exactly 0.90/0.95/0.99 (Hansen 2000 tabulation).
        CommandSpec(
            path=["estimate", "threshold"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="threshold-col", type=String, default="", description="Required: the variable that splits the sample (excluded from the regressors)"),
                OptionSpec(name="trim", type=Float64, default=0.15, description="Trimming fraction for the threshold grid (0 < trim < 0.5)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Bootstrap replications for the linearity test (≥ 1)"),
                # Hansen (2000) CVs are tabulated only at 0.90/0.95/0.99. `choices` is
                # Vector{String} even for a Float64 option — same as `estimate setar`.
                OptionSpec(name="ci-level", type=Float64, default=0.95, description="Threshold CI level: 0.90|0.95|0.99", choices=["0.90", "0.95", "0.99"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="het", description="Heteroskedasticity-robust bootstrap for the linearity test"),
                   FlagSpec(name="no-linearity", description="Skip the Hansen (1996) linearity test"),
                   FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:threshold_regression_coefficients,
                              description="Per-regime coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:threshold_regression_diagnostics,
                              description="Threshold gamma with its Hansen CI, per-regime counts, fit statistics and the linearity test")],
            category="estimate",
            handler=wrap_legacy(_estimate_threshold),
        ),
        CommandSpec(
            path=["estimate", "setar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=String, default="1", description="Delay lag: an integer ≥ 1, or 'auto' (=1:p grid)"),
                OptionSpec(name="trim", type=Float64, default=0.15, description="Trimming fraction for the threshold grid (0 < trim < 0.5)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Bootstrap reps for the Hansen test / threshold CI (≥ 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.95, description="Threshold CI level: 0.90|0.95|0.99", choices=["0.90", "0.95", "0.99"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                PLOT_OPTIONS...
            ],
            flags=[
                FlagSpec(name="het", description="Heteroskedastic (White) bootstrap for the linearity test / CI"),
                FlagSpec(name="no-linearity", description="Skip the attached Hansen (1996) linearity test")
            , PLOT_FLAGS...],
            tables=[TableSpec(name=:setar_coefficients,
                              description="Per-regime SETAR coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:setar_diagnostics,
                              description="Threshold gamma with its Hansen CI, regime counts, p/d, fit statistics and the linearity test")],
            category="estimate",
            handler=wrap_legacy(_estimate_setar),
        ),
        # ── C065b: STAR (smooth-transition autoregression) ──
        # Two-regime STAR via `estimate_star` (Teräsvirta 1994 NLS). Hand-built two-regime
        # weight tables (1−G / G) + a transition-parameters block (γ, c) + kv diagnostics
        # (STARModel is NOT in MEMs `_COEF_TABLE_TYPES`). `--type` selects the transition
        # shape (lstr1|lstr2|estr) or `auto` (Teräsvirta sequential selection); an external
        # transition variable can be supplied via `--transition-col` (else self-exciting y[t-d]).
        CommandSpec(
            path=["estimate", "star"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=Int, default=1, description="Delay lag for the self-exciting transition var (≥ 1)"),
                OptionSpec(name="type", type=String, default="auto", description="Transition shape: lstr1|lstr2|estr|auto", choices=["lstr1", "lstr2", "estr", "auto"]),
                OptionSpec(name="n-gamma", type=Int, default=15, description="Grid points for the γ start values (≥ 2)"),
                OptionSpec(name="n-c", type=Int, default=15, description="Grid points for the c start values (≥ 2)"),
                OptionSpec(name="transition-col", type=Int, default=0, description="Column index of an external transition var s (0 = self-exciting y[t-d])"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                PLOT_OPTIONS...
            ],
            flags=copy(PLOT_FLAGS),
            tables=[TableSpec(name=:star_regime_coefficients,
                              description="Coefficients of the two regime blocks with standard errors and p-values"),
                    TableSpec(name=:star_transition_parameters,
                              description="Smoothness gamma and location c of the transition function, with standard errors"),
                    TableSpec(name=:star_diagnostics,
                              description="Transition type, fit statistics, the LM3 linearity test and any sequential-selection p-values")],
            category="estimate",
            handler=wrap_legacy(_estimate_star),
        ),
        # ── C065c: MS-AR (Markov-switching autoregression, Hamilton 1989 mean-switching) ──
        # `estimate_ms_ar` → MSRegModel(model_type=:ms_ar). Only the level μ switches across
        # regimes; the AR coefficients φ are COMMON. Hand-built per-regime coef table (μ rows +
        # a common-AR block) + a per-regime variance table + a WIDE K×K transition matrix +
        # diagnostics kv (MSRegModel is NOT in MEMs `_COEF_TABLE_TYPES`). NOTE the Hamilton form's
        # `switching_variance` default is FALSE (the `--switching-variance` flag turns it ON) —
        # the OPPOSITE polarity of `estimate ms`; do not unify.
        CommandSpec(
            path=["estimate", "ms-ar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=1000, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                PLOT_OPTIONS...
            ],
            flags=[FlagSpec(name="switching-variance", description="Let σ² switch across regimes (default: off, Hamilton form)"), PLOT_FLAGS...],
            tables=[TableSpec(name=:ms_ar_regime_coefficients,
                              description="Per-regime switching means plus the common AR block, with standard errors"),
                    TableSpec(name=:ms_ar_regime_variances,
                              description="Per-regime innovation variance with its standard error"),
                    TableSpec(name=:ms_ar_transition_matrix,
                              description="Markov transition matrix P, one row per originating regime"),
                    TableSpec(name=:ms_ar_regime_probabilities,
                              description="Filtered and smoothed regime probabilities, one row per period x regime"),
                    TableSpec(name=:ms_ar_diagnostics,
                              description="Log-likelihood, AIC/BIC, ergodic probabilities, expected durations and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_ms_ar),
        ),
        # ── C065c: MS regression (K-state Markov-switching regression) ──
        # `estimate_ms` → MSRegModel(model_type=:regression) — every coefficient switches with the
        # latent regime. Loaded via `_load_reg_data` (y + numeric regressors, NO auto-intercept —
        # include a `const` column, like `estimate reg`); when the dependent variable is the ONLY
        # numeric column, routes to the single-arg intercept-only dispatch `estimate_ms(y; …)`.
        # Same hand-built rendering as ms-ar via the shared `_ms_render`. NOTE the regression form's
        # `switching_variance` default is TRUE (the `--no-switching-variance` flag turns it OFF) —
        # the OPPOSITE polarity of `estimate ms-ar`.
        CommandSpec(
            path=["estimate", "ms"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="EM convergence tolerance (> 0)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                PLOT_OPTIONS...
            ],
            flags=[FlagSpec(name="no-switching-variance", description="Force common σ² across regimes (default: σ² switches)"), PLOT_FLAGS...],
            tables=[TableSpec(name=:ms_regression_regime_coefficients,
                              description="Per-regime switching regression coefficients with standard errors"),
                    TableSpec(name=:ms_regression_regime_variances,
                              description="Per-regime innovation variance with its standard error"),
                    TableSpec(name=:ms_regression_transition_matrix,
                              description="Markov transition matrix P, one row per originating regime"),
                    TableSpec(name=:ms_regression_regime_probabilities,
                              description="Filtered and smoothed regime probabilities, one row per period x regime"),
                    TableSpec(name=:ms_regression_diagnostics,
                              description="Log-likelihood, AIC/BIC, ergodic probabilities, expected durations and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_ms),
        ),
        CommandSpec(
            path=["estimate", "fastica"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
                OptionSpec(name="method", type=String, default="fastica", description="fastica|jade|sobi|dcov|hsic"),
                OptionSpec(name="contrast", type=String, default="logcosh", description="logcosh|exp|kurtosis (for FastICA)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:structural_impact_matrix_b0,
                              description="Identified structural impact matrix B0, one row per equation"),
                    TableSpec(name=:structural_shocks,
                              description="First observations of the recovered structural shocks")],
            category="estimate",
            handler=wrap_legacy(_estimate_fastica),
        ),
        CommandSpec(
            path=["estimate", "ml"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
                OptionSpec(name="distribution", short="d", type=String, default="student_t", description="student_t|skew_t|ghd|mixture_normal|pml|skew_normal"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:structural_impact_matrix_b0,
                              description="Identified structural impact matrix B0, one row per equation"),
                    TableSpec(name=:model_fit,
                              description="Non-Gaussian and Gaussian log-likelihoods, AIC/BIC and the assumed distribution"),
                    TableSpec(name=:parameter_estimates_with_standard_errors,
                              description="B0 elements with standard errors, when the estimator returns them")],
            category="estimate",
            handler=wrap_legacy(_estimate_ml),
        ),
        CommandSpec(
            path=["estimate", "pvar"],
            summary="Path to CSV panel data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV panel data file")],
            options=[
                OptionSpec(name="id-col", type=String, default="", description="Panel group identifier column (required)"),
                OptionSpec(name="time-col", type=String, default="", description="Time period column (required)"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Lag order"),
                OptionSpec(name="dependent", type=String, default="", description="Dependent variables (comma-separated)"),
                OptionSpec(name="predet", type=String, default="", description="Predetermined variables (comma-separated)"),
                OptionSpec(name="exog", type=String, default="", description="Exogenous variables (comma-separated)"),
                OptionSpec(name="transformation", type=String, default="fd", description="fd|fod (first-difference or forward orthogonal)"),
                OptionSpec(name="steps", type=String, default="twostep", description="onestep|twostep"),
                OptionSpec(name="method", type=String, default="gmm", description="gmm|feols"),
                OptionSpec(name="min-lag-endo", type=Int, default=2, description="Minimum lag for endogenous instruments"),
                OptionSpec(name="max-lag-endo", type=Int, default=99, description="Maximum lag for endogenous instruments"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[
                FlagSpec(name="system", description="Use system GMM (adds level equations)"),
                FlagSpec(name="collapse", description="Collapse instruments to limit count")
            ],
            tables=[TableSpec(name=:panel_var_coefficients,
                              description="Panel VAR coefficients, one row per equation x lagged regressor"),
                    TableSpec(name=:panel_summary,
                              description="Group and observation counts, instrument count, estimator and transformation")],
            category="estimate",
            handler=wrap_legacy(_estimate_pvar),
        ),
        CommandSpec(
            path=["estimate", "vecm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels, VECM uses p-1)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="method", type=String, default="johansen", description="johansen|engle_granger"),
                OptionSpec(name="significance", type=Float64, default=0.05, description="Significance level for rank selection"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:cointegrating_vectors_beta,
                              description="Cointegrating vectors beta, one row per variable and one column per relation"),
                    TableSpec(name=:adjustment_coefficients_alpha,
                              description="Adjustment coefficients alpha, one row per equation"),
                    TableSpec(name=:information_criteria,
                              description="AIC / BIC / HQC and the log-likelihood of the fitted VECM")],
            category="estimate",
            handler=wrap_legacy(_estimate_vecm),
        ),
        CommandSpec(
            path=["estimate", "svar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto via AIC)"),
                OptionSpec(name="pattern", type=String, default="recursive", description="AB-model pattern: recursive|blanchard-quah|a-model|b-model|ab-model", choices=["recursive", "blanchard-quah", "a-model", "b-model", "ab-model"]),
                OptionSpec(name="config", type=String, default="", description="TOML config with [svar] A/B matrices (a/b/ab-model)"),
                OptionSpec(name="n-starts", type=Int, default=5, description="Optimizer starting values (overidentified patterns)"),
                OptionSpec(name="max-iter", type=Int, default=400, description="Max optimizer iterations per start"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:svar_a,
                              description="SVAR contemporaneous A matrix, one row per equation"),
                    TableSpec(name=:svar_b,
                              description="SVAR structural B matrix, one row per equation"),
                    TableSpec(name=:svar_summary,
                              description="Log-likelihood, LR overidentification test and identification status")],
            category="estimate",
            handler=wrap_legacy(_estimate_svar),
        ),
        CommandSpec(
            path=["estimate", "svec"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels, VECM uses p-1)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="method", type=String, default="johansen", description="johansen|engle_granger"),
                OptionSpec(name="significance", type=Float64, default=0.05, description="Significance level for rank selection"),
                OptionSpec(name="config", type=String, default="", description="TOML config with optional [svec] long/short-run zero matrices"),
                OptionSpec(name="n-starts", type=Int, default=5, description="Optimizer starting values (restricted patterns)"),
                OptionSpec(name="max-iter", type=Int, default=400, description="Max optimizer iterations per start"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:svec_b0,
                              description="SVEC contemporaneous impact matrix B0, one row per equation"),
                    TableSpec(name=:svec_xi,
                              description="SVEC long-run impact matrix Xi, one row per equation"),
                    TableSpec(name=:svec_summary,
                              description="Permanent-shock count and identification status")],
            category="estimate",
            handler=wrap_legacy(_estimate_svec),
        ),
        CommandSpec(
            path=["estimate", "smm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML config for SMM specification"),
                OptionSpec(name="weighting", type=String, default="two_step", description="identity|optimal|two_step|iterated"),
                OptionSpec(name="sim-ratio", type=Int, default=5, description="Simulation-to-sample ratio"),
                OptionSpec(name="burn", type=Int, default=100, description="Burn-in periods"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:smm_estimates,
                              description="SMM parameter estimates with standard errors, t-statistics and p-values")],
            category="estimate",
            handler=wrap_legacy(_estimate_smm),
        ),
        CommandSpec(
            path=["estimate", "favar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lag order"),
                OptionSpec(name="key-vars", type=String, default="", description="Key variable names or indices (comma-separated)"),
                OptionSpec(name="method", type=String, default="two_step", description="two_step|bayesian"),
                OptionSpec(name="draws", short="n", type=Int, default=5000, description="MCMC draws (bayesian only)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:favar_coefficients,
                              description="FAVAR coefficients over factors and key variables, one row per equation x regressor"),
                    TableSpec(name=:bayesian_favar,
                              description="Factor, key-variable, lag and draw counts of a --method bayesian fit")],
            category="estimate",
            handler=wrap_legacy(_estimate_favar),
        ),
        CommandSpec(
            path=["estimate", "sdfm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="q", type=Int, default=nothing, description="Number of dynamic factors (default: auto via --q-method)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|proxy (--id proxy requires --instrument)"),
                OptionSpec(name="q-method", type=String, default="hallin-liska", description="Auto factor selection: hallin-liska|bai-ng|amengual-watson", choices=["hallin-liska","bai-ng","amengual-watson"]),
                OptionSpec(name="method", type=String, default="fglr", description="Estimator: fglr|gdfm-var (gdfm-var is the legacy path)", choices=["fglr","gdfm-var"]),
                OptionSpec(name="spectral", type=String, default="lag-window", description="GDFM spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="var-lags", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="horizon", type=Int, default=40, description="Structural IRF horizon"),
                OptionSpec(name="config", type=String, default="", description="TOML config for sign restrictions"),
                OptionSpec(name="bandwidth", type=Int, default=0, description="Spectral bandwidth (0=auto)"),
                OptionSpec(name="kernel", type=String, default="bartlett", description="bartlett|parzen|quadratic_spectral"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            # #147 (closed the W3/#138 gap): the leaf now emits an estimation record —
            # dimensions, identification, per-shock names, variance shares. The
            # structural arrays still live on `irf sdfm`/`fevd sdfm`.
            tables=[TableSpec(name=:sdfm_estimation_summary,
                              description="Structural DFM estimation record: panel dimensions, factor counts, identification, factor-VAR lags, shock names and the average common variance share")],
            category="estimate",
            handler=wrap_legacy(_estimate_sdfm),
        ),
        CommandSpec(
            path=["estimate", "reg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                # W10/#112: cov-type is WIDENED with `conley` for THIS leaf only. REG_OPTIONS
                # is shared with logit/probit/ologit/oprobit/mlogit and the predict/residuals
                # family, whose estimators take no :conley and whose handlers accept no
                # coords/cutoff kwargs — declaring an option a handler cannot feed through is
                # an exit-1 on every invocation (#85). Same pattern as `estimate preg`'s pcse.
                map(o -> o.name == "cov-type" ?
                        OptionSpec(name="cov-type", type=String, default="hc1",
                                   choices=["ols", "hc0", "hc1", "hc2", "hc3", "cluster", "conley"],
                                   description="ols|hc0|hc1|hc2|hc3|cluster|conley (Conley spatial HAC; needs --lat/--lon/--dist-cutoff)") : o,
                    REG_OPTIONS)...,
                OptionSpec(name="weights", type=String, default="", description="Weight column name (WLS)"),
                OptionSpec(name="lat", type=String, default="",
                           description="Latitude (or first coordinate) column — required for --cov-type conley"),
                OptionSpec(name="lon", type=String, default="",
                           description="Longitude (or second coordinate) column — required for --cov-type conley"),
                OptionSpec(name="dist-cutoff", type=Float64, default=0.0,
                           description="Conley spatial cutoff: km for --conley-metric haversine, coordinate units for euclidean"),
                OptionSpec(name="conley-kernel", type=String, default="bartlett",
                           choices=["bartlett", "uniform"],
                           description="Conley spatial kernel: bartlett (tapered) or uniform"),
                OptionSpec(name="conley-metric", type=String, default="euclidean",
                           choices=["euclidean", "haversine"],
                           description="Distance metric: euclidean (projected coords) or haversine (degrees → km)"),
                OptionSpec(name="time-col", type=String, default="",
                           description="Time column for Conley spatial+serial correlation (needs --time-cutoff)"),
                OptionSpec(name="time-cutoff", type=Int, default=0,
                           description="Conley serial-correlation lag cutoff (0 = spatial only)")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:reg_coefficients,
                              description="OLS/WLS coefficients with standard errors, t-statistics, p-values and CIs"),
                    TableSpec(name=:fit_statistics,
                              description="R-squared, adjusted R-squared, F-statistic with p-value, log-likelihood and AIC/BIC"),
                    TableSpec(name=:conley_spatial_hac_settings,
                              description="Coordinate columns, metric, kernel and cutoffs behind a --cov-type conley fit")],
            category="estimate",
            handler=wrap_legacy(_estimate_reg),
        ),
        CommandSpec(
            path=["estimate", "select"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="method", type=String, default="bidirectional", description="Search strategy", choices=["forward","backward","bidirectional","best-subset","gets"]),
                OptionSpec(name="criterion", type=String, default="pvalue", description="Selection criterion", choices=["pvalue","aic","bic"]),
                OptionSpec(name="p-enter", type=Float64, default=0.05, description="p-value to enter a regressor (0,1)"),
                OptionSpec(name="p-remove", type=Float64, default=0.10, description="p-value to remove a regressor (0,1); must be ≥ --p-enter for bidirectional pvalue"),
                OptionSpec(name="keep", type=String, default="", description="Comma-separated regressor names always retained"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:selected_model_coefficients,
                              description="Coefficients of the refitted model on the selected regressors"),
                    TableSpec(name=:selection_path,
                              description="The search trail: step, add/drop action, variable and the deciding statistic"),
                    TableSpec(name=:selection_summary,
                              description="Method, criterion, selected and forced variables, and any encompassing test")],
            category="estimate",
            handler=wrap_legacy(_estimate_select),
        ),
        CommandSpec(
            path=["estimate", "iv"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name (default: first numeric column)"),
                OptionSpec(name="endogenous", type=String, default="", description="Endogenous regressor column names, comma-separated (required)"),
                OptionSpec(name="instruments", type=String, default="", description="EXCLUDED instrument column names, comma-separated (required; other numeric cols are exogenous regressors — include a `const` for an intercept)"),
                OptionSpec(name="cov-type", type=String, default="hc1", description="ols|hc0|hc1|hc2|hc3"),
                OptionSpec(name="method", type=String, default="tsls", description="k-class estimator", choices=["tsls","liml","fuller","kclass"]),
                OptionSpec(name="k", type=String, default="", description="k-class scalar (required with --method kclass; k=0 is OLS, k=1 is 2SLS)"),
                OptionSpec(name="fuller-a", type=Float64, default=1.0, description="Fuller adjustment a > 0 (--method fuller only; a=1 is approximately unbiased)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:iv_coefficients,
                              description="Instrumental-variables coefficients with standard errors, t-statistics and p-values"),
                    TableSpec(name=:iv_diagnostics,
                              description="R-squared, the k-class constant, first-stage F and the Sargan overidentification test")],
            category="estimate",
            handler=wrap_legacy(_estimate_iv),
        ),
        CommandSpec(
            path=["estimate", "sur"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML config: [[equations]] blocks (dep + indep) (required)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[
                FlagSpec(name="iterate", description="Iterate FGLS to the Gaussian MLE"),
                FlagSpec(name="no-intercept", description="Do not add a per-equation constant"),
            ],
            tables=[TableSpec(name=:sur_coefficients,
                              description="System coefficients, one row per equation x term, with standard errors and p-values"),
                    TableSpec(name=:sur_system_statistics,
                              description="Estimator, equation and observation counts, det(Sigma), McElroy R-squared and log-likelihood")],
            category="estimate",
            handler=wrap_legacy(_estimate_sur),
        ),
        CommandSpec(
            path=["estimate", "3sls"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="config", type=String, default="", description="TOML config: [[equations]] + instruments (required)"),
                OptionSpec(name="instruments", type=String, default="common", choices=["common","perequation"], description="common|perequation instrument sets"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[
                FlagSpec(name="no-intercept", description="Do not add a per-equation constant"),
            ],
            tables=[TableSpec(name=Symbol("3sls_coefficients"),
                              description="System coefficients, one row per equation x term, with standard errors and p-values"),
                    TableSpec(name=Symbol("3sls_system_statistics"),
                              description="Equation and observation counts, det(Sigma), McElroy R-squared and instruments per equation")],
            category="estimate",
            handler=wrap_legacy(_estimate_3sls),
        ),
        CommandSpec(
            # W6/#108: multiplicative seasonal ARIMA. SARIMAModel <: AbstractARIMAModel,
            # which has a real plot_result recipe (an ABSTRACT dispatch — a by-name grep for
            # plot_result(::SARIMAModel) finds nothing and would have wrongly ruled plots out).
            path=["estimate", "sarima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[SARIMA_OPTIONS...,
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                PLOT_OPTIONS...],
            flags=[SARIMA_FLAGS..., PLOT_FLAGS...],
            tables=[TableSpec(name=:sarima_coefficients,
                              description="Non-seasonal and seasonal AR/MA estimates plus the innovation variance"),
                    TableSpec(name=:information_criteria,
                              description="AIC, BIC, log-likelihood and effective sample size")],
            category="estimate",
            handler=wrap_legacy(_estimate_sarima),
        ),
        CommandSpec(
            path=["estimate", "poisson"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                # Upstream's own choice set and default: estimate_poisson defaults to the
                # Gourieroux-Monfort-Trognon pseudo-ML sandwich, which stays consistent under
                # any conditional-mean-correct misspecification. NOT the shared REG_OPTIONS
                # cov-type, which has no `robust`/`mle` and defaults to hc1.
                OptionSpec(name="cov-type", type=String, default="robust",
                           choices=["robust", "mle", "hc0", "hc1", "hc2", "hc3", "cluster"],
                           description="robust (QMLE sandwich, default), mle, hc0-hc3, cluster"),
                OptionSpec(name="clusters", type=String, default="", description="Cluster variable column name"),
                OptionSpec(name="maxiter", type=Int, default=100, description="Maximum IRLS iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                COUNT_IRR_OPTIONS...],
            flags=[COUNT_IRR_FLAG],
            tables=[TableSpec(name=:poisson_regression_coefficients,
                              description="Poisson coefficients with standard errors, z-statistics, p-values and CIs"),
                    TableSpec(name=:incidence_rate_ratios,
                              description="exp(beta) incidence-rate ratios with delta-method SEs and CIs (--irr)"),
                    TableSpec(name=:fit_statistics,
                              description="Log-likelihood, deviance, pseudo R-squared and AIC/BIC")],
            category="estimate",
            handler=wrap_legacy(_estimate_poisson),
        ),
        CommandSpec(
            # NO --cov-type/--clusters: estimate_nbreg accepts neither (its vcov is the
            # joint (beta, log alpha) information matrix).
            path=["estimate", "nbreg"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[COUNT_COMMON_OPTIONS...,
                OptionSpec(name="maxiter", type=Int, default=1000, description="Maximum iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-10, description="Convergence tolerance (> 0)"),
                COUNT_IRR_OPTIONS...],
            flags=[COUNT_IRR_FLAG],
            tables=[TableSpec(name=:negative_binomial_regression_coefficients,
                              description="NB2 coefficients with standard errors, z-statistics, p-values and CIs"),
                    TableSpec(name=:overdispersion_parameter,
                              description="The NB2 dispersion parameter alpha with its delta-method standard error"),
                    TableSpec(name=:incidence_rate_ratios,
                              description="exp(beta) incidence-rate ratios with delta-method SEs and CIs (--irr)"),
                    TableSpec(name=:fit_statistics,
                              description="Log-likelihood, deviance, pseudo R-squared and AIC/BIC")],
            category="estimate",
            handler=wrap_legacy(_estimate_nbreg),
        ),
        CommandSpec(
            path=["estimate", "logit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                REG_OPTIONS...,
                OptionSpec(name="maxiter", type=Int, default=100, description="Maximum IRLS iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="Convergence tolerance")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:logit_regression_coefficients,
                              description="Logit coefficients with standard errors, z-statistics, p-values and CIs"),
                    TableSpec(name=:fit_statistics,
                              description="Pseudo R-squared, log-likelihood (fitted and null), AIC/BIC and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_logit),
        ),
        CommandSpec(
            path=["estimate", "probit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                REG_OPTIONS...,
                OptionSpec(name="maxiter", type=Int, default=100, description="Maximum IRLS iterations"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="Convergence tolerance")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:probit_regression_coefficients,
                              description="Probit coefficients with standard errors, z-statistics, p-values and CIs"),
                    TableSpec(name=:fit_statistics,
                              description="Pseudo R-squared, log-likelihood (fitted and null), AIC/BIC and convergence")],
            category="estimate",
            handler=wrap_legacy(_estimate_probit),
        ),
        CommandSpec(
            path=["estimate", "preg"],
            summary="Path to CSV panel data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV panel data file")],
            options=[
                # cov-type is WIDENED to include pcse for THIS leaf only, and the two new
                # options are appended HERE rather than to PREG_OPTIONS: that const is
                # shared with piv/plogit/pprobit, predict/residuals and several `test`
                # leaves whose estimators take no :pcse and whose handlers accept no
                # ar1/pcse_unbalanced kwarg (declaring an option a handler cannot take is
                # an exit-1 MethodError on every invocation — see #85).
                map(o -> o.name == "cov-type" ?
                        OptionSpec(name="cov-type", type=String, default="cluster",
                                   choices=["ols","cluster","twoway","driscoll-kraay","pcse"],
                                   description="ols|cluster|twoway|driscoll-kraay|pcse (Beck-Katz panel-corrected SEs)") : o,
                    PREG_OPTIONS)...;
                OptionSpec(name="ar1", type=String, default="none", description="Prais-Winsten AR(1) correction", choices=["none","common","panel-specific"]);
                OptionSpec(name="pcse-unbalanced", type=String, default="casewise", description="Unbalanced-panel handling for --cov-type pcse", choices=["casewise","pairwise"]);
                # W10/#112 reghdfe-style absorption. `entity`/`time`/`cohort` (and the
                # aliases id/unit/group/period) resolve to the panel indices; anything else
                # must name a panel variable column.
                OptionSpec(name="absorb", type=String, default="",
                           description="Comma-separated high-dimensional FE dimensions to absorb (entity, time, cohort, or a column name); --method fe only");
                OptionSpec(name="hdfe-tol", type=Float64, default=1e-8,
                           description="Absorption convergence tolerance");
                OptionSpec(name="hdfe-maxiter", type=Int, default=1000,
                           description="Maximum alternating-projection iterations");
                # W10/#131 Roodman/xtabond2 instrument-proliferation controls
                # (MEMs#549). --method ab|bb only: upstream IGNORES them silently
                # for every other method, so the handler refuses instead.
                OptionSpec(name="min-lag-endo", type=Int, default=2,
                           description="First instrument lag for endogenous regressors (--method ab|bb)");
                OptionSpec(name="max-lag-endo", type=Int, default=99,
                           description="Last instrument lag for endogenous regressors (--method ab|bb)")
            ],
            flags=[
                FlagSpec(name="twoway", description="Include time fixed effects"),
                FlagSpec(name="collapse",
                         description="Collapse the GMM instrument matrix (--method ab|bb)")
            ],
            tables=[TableSpec(name=:panel_regression_coefficients,
                              description="Panel coefficients with standard errors, t-statistics and p-values"),
                    TableSpec(name=:model_statistics,
                              description="Within/between/overall R-squared, F-statistic, and observation and group counts"),
                    TableSpec(name=:dynamic_panel_diagnostics,
                              description="AR(1)/AR(2) tests, Hansen J and the instrument count (--method ab|bb)"),
                    TableSpec(name=:hdfe_absorption,
                              description="Absorbed dimensions, level counts and the projection-loop convergence (--absorb)")],
            category="estimate",
            handler=wrap_legacy(_estimate_preg),
        ),
        CommandSpec(
            path=["estimate", "piv"],
            summary="Path to CSV panel data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV panel data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name"),
                OptionSpec(name="exog", type=String, default="", description="Exogenous variables (comma-separated)"),
                OptionSpec(name="endog", type=String, default="", description="Endogenous variables (comma-separated)"),
                OptionSpec(name="instruments", type=String, default="", description="Instruments (comma-separated)"),
                OptionSpec(name="method", short="m", type=String, default="fe", description="fe|re|fd|hausman-taylor"),
                OptionSpec(name="cov-type", type=String, default="cluster", description="ols|cluster|twoway|driscoll-kraay"),
                OptionSpec(name="id-col", type=String, default="", description="Panel group ID column"),
                OptionSpec(name="time-col", type=String, default="", description="Panel time column"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:panel_iv_coefficients,
                              description="Panel IV coefficients with standard errors, t-statistics and p-values"),
                    TableSpec(name=:weak_instrument_diagnostics,
                              description="First-stage F, Cragg-Donald and Kleibergen-Paap F, Stock-Yogo bound and Sargan test")],
            category="estimate",
            handler=wrap_legacy(_estimate_piv),
        ),
        CommandSpec(
            path=["estimate", "plogit"],
            summary="Path to CSV panel data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV panel data file")],
            options=with_default(PREG_OPTIONS, "method", "pooled"),
            flags=FlagSpec[],
            tables=[TableSpec(name=:panel_logit_coefficients,
                              description="Panel logit coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:model_statistics,
                              description="Pseudo R-squared, log-likelihood, AIC/BIC, convergence and observation/group counts")],
            category="estimate",
            handler=wrap_legacy(_estimate_plogit),
        ),
        CommandSpec(
            path=["estimate", "pprobit"],
            summary="Path to CSV panel data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV panel data file")],
            options=with_default(PREG_OPTIONS, "method", "pooled"),
            flags=FlagSpec[],
            tables=[TableSpec(name=:panel_probit_coefficients,
                              description="Panel probit coefficients with standard errors, z-statistics and p-values"),
                    TableSpec(name=:model_statistics,
                              description="Pseudo R-squared, log-likelihood, AIC/BIC, convergence and observation/group counts")],
            category="estimate",
            handler=wrap_legacy(_estimate_pprobit),
        ),
        CommandSpec(
            path=["estimate", "ologit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                REG_OPTIONS...
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ordered_logit_coefficients,
                              description="Ordered-logit slope coefficients with standard errors, p-values and CIs"),
                    TableSpec(name=:cutpoints, description="Estimated category cutpoints of the latent index"),
                    TableSpec(name=:fit_statistics,
                              description="Pseudo R-squared, log-likelihood, AIC/BIC and the category count")],
            category="estimate",
            handler=wrap_legacy(_estimate_ologit),
        ),
        CommandSpec(
            path=["estimate", "oprobit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                REG_OPTIONS...
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:ordered_probit_coefficients,
                              description="Ordered-probit slope coefficients with standard errors, p-values and CIs"),
                    TableSpec(name=:cutpoints, description="Estimated category cutpoints of the latent index"),
                    TableSpec(name=:fit_statistics,
                              description="Pseudo R-squared, log-likelihood, AIC/BIC and the category count")],
            category="estimate",
            handler=wrap_legacy(_estimate_oprobit),
        ),
        CommandSpec(
            path=["estimate", "mlogit"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column name"),
                OptionSpec(name="cov-type", type=String, default="ols", description="ols|hc0|hc1|hc2|hc3"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:multinomial_logit_coefficients,
                              description="Coefficients for every alternative in one tidy table, with standard errors and CIs"),
                    TableSpec(name=:fit_statistics,
                              description="Pseudo R-squared, log-likelihood, AIC/BIC and the category count")],
            category="estimate",
            handler=wrap_legacy(_estimate_mlogit),
        )
    ]
end

const _ESTIMATE_PANEL = Set(["pvar", "preg", "piv", "plogit", "pprobit", "pmg", "xtcointreg"])
const _ESTIMATE_XS = Set(["3sls", "elastic-net", "heckman", "iv", "kde", "kernel-reg",
    "lasso", "logit", "lowess", "ml", "mlogit", "nbreg", "ologit", "oprobit",
    "poisson", "probit", "qreg", "rdd", "reg", "ridge", "robust", "select",
    "sur", "tobit", "truncreg"])

function _data_kinds_for_estimator(leaf::AbstractString)
    leaf in _ESTIMATE_PANEL ? [:panel, :csv] :
    leaf in _ESTIMATE_XS    ? [:cross_section, :timeseries, :csv] :
                              [:timeseries, :csv]
end

function register_estimate_commands!()
    specs = with_config_ergonomics(with_save_model(estimate_specs()))
    out = CommandSpec[]
    for s in specs
        push!(out, _copy_spec(s; data_kinds=_data_kinds_for_estimator(s.path[end])))
    end
    out = with_default_csv_kinds(out)
    register!(out)
    return build_node("estimate", out; description="Model estimation")
end


# ── VAR ────────────────────────────────────────────────────

function _estimate_var(; data::String, lags=nothing, trend::String="constant",
                        output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    _status("Estimating VAR($p) with $n variables: $(join(varnames, ", "))")
    _status("Trend: $trend, Observations: $(size(Y, 1))")
    _status()

    model = estimate_var(Y, p)
    _status_report(() -> report(model))

    # C051: MEMs renders coefficient-bearing models as a tidy coef table via Tables.jl
    # (equation|term|estimate|std_error|stat|p_value|ci_lower|ci_upper).
    output_result(DataFrame(model); format=Symbol(format), output=output, title="VAR($p) Coefficients",
                  key="var_coefficients")

    _status()
    output_model_criteria(model; format=format, title="Information Criteria")
    return model
end

# ── BVAR ───────────────────────────────────────────────────

"""
    _bvar_select_hyper(Y, p, hyperopt) -> (hyper, table_or_nothing)

Run the Minnesota hyperparameter selection the CLI is about to hand to `estimate_bvar`.

Upstream does this internally and **discards the diagnostics** — `BVARPosterior` keeps only
the draws, so once `estimate_bvar` returns there is no way to report which hyperparameters
were chosen or whether the optimizer converged. Selecting here and passing `hyper=` gives
identical numbers (upstream's `:glp` branch is exactly `optimize_hyperparameters_glp(Y, p).hyper`
on the same un-lagged `Y`) while making the choice reportable. No extra work: supplying
`hyper=` skips upstream's own optimization.
"""
function _bvar_select_hyper(Y::AbstractMatrix, p::Int, hyperopt::String)
    if hyperopt == "glp"
        _status("Optimizing Minnesota hyperparameters (GLP joint)...")
        r = optimize_hyperparameters_glp(Y, p)
        h = r.hyper
        # at_bound matters more than converged: a hyperparameter pinned to its bound means
        # the optimum is outside the admissible region, so the "optimized" value is an
        # artefact of the box rather than a maximum.
        r.at_bound && _status("  warning: a hyperparameter is AT ITS BOUND — the optimum " *
                              "lies outside the search box, so treat these values with care")
        tbl = DataFrame(
            parameter = ["tau", "decay", "lambda", "mu", "omega",
                         "log_ml", "log_ml_default", "log_posterior",
                         "converged", "at_bound", "iterations"],
            value = Float64[h.tau, h.decay, h.lambda, h.mu, h.omega,
                            r.log_ml, r.log_ml_default, r.log_posterior,
                            r.converged ? 1.0 : 0.0, r.at_bound ? 1.0 : 0.0, r.iterations],
        )
        return h, tbl
    end
    _status("Selecting Minnesota tau by grid search...")
    h = optimize_hyperparameters(Y, p)
    # The grid path optimizes tau ONLY and returns no diagnostics, so the table is just the
    # resulting hyperparameters — deliberately the same columns, minus the GLP-only rows.
    tbl = DataFrame(
        parameter = ["tau", "decay", "lambda", "mu", "omega"],
        value = Float64[h.tau, h.decay, h.lambda, h.mu, h.omega],
    )
    return h, tbl
end

function _estimate_bvar(; data::String, lags::Int=4, prior::String="minnesota",
                         draws::Int=2000, sampler::String="direct", method::String="mean",
                         hyperopt::String="glp",
                         config::String="", output::String="", format::String="table")
    hopt = lowercase(strip(hyperopt))
    hopt in ("glp", "grid") || throw(CliError("usage/invalid-option",
        "invalid --hyperopt '$hyperopt'; must be glp or grid"))
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    p = lags

    _status("Estimating Bayesian VAR($p) with $n variables: $(join(varnames, ", "))")
    _status("Prior: $prior, Sampler: $sampler, Draws: $draws, Posterior: $method")
    _status()

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? Symbol(prior) : :minnesota

    # Precedence: an explicit config [prior] pins the hyperparameters and upstream then
    # ignores hyperopt entirely, so say so rather than letting the flag look effective.
    hyper_tbl = nothing
    if prior_obj !== nothing
        hopt == "glp" || _status("--hyperopt=$hopt ignored: [prior] in the config pins " *
                                 "the hyperparameters")
    elseif prior_sym === :minnesota
        prior_obj, hyper_tbl = _bvar_select_hyper(Y, p, hopt)
    else
        # :normal/:diffuse never consult Minnesota hyperparameters at all.
        hopt == "glp" || _status("--hyperopt=$hopt ignored: it applies to the minnesota " *
                                 "prior, not --prior $prior")
    end

    post = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_draws=draws,
        prior=prior_sym, hyper=prior_obj, seed=_SEED[])

    model = if method == "median"
        posterior_median_model(post)
    else
        posterior_mean_model(post)
    end

    _status_report(() -> report(model))

    coef_df = _build_var_coef_table(coef(model), varnames, p)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="BVAR($p) Posterior $(titlecase(method)) Coefficients",
                  key="bvar_coefficients")

    if hyper_tbl !== nothing
        output_result(hyper_tbl; format=Symbol(format),
                      output=_per_var_output_path(output, "hyper"),
                      title="Minnesota Hyperparameters ($hopt)",
                      key="minnesota_hyperparameters")
    end

    _status()
    output_model_criteria(model; format=format, title="Information Criteria (Posterior $(titlecase(method)))")
    return model
end

# ── LP ─────────────────────────────────────────────────────

function _estimate_lp(; data::String, method::String="standard", shock::Int=1,
                       horizons::Int=20, control_lags::Int=4, vcov::String="newey_west",
                       instruments::String="", knots::Int=3, lambda::Float64=0.0,
                       state_var=nothing, gamma::Float64=1.5, transition::String="logistic",
                       treatment::Int=1, score_method::String="logit",
                       mop_f::Bool=false, mop_tau::Float64=0.10, mop_bandwidth::Int=0,
                       ar_bands::Bool=false, ar_level::Float64=0.95, ar_grid::Int=401,
                       ar_span::Float64=20.0,
                       output::String="", format::String="table")
    validate_method(method, ["standard", "iv", "smooth", "state", "propensity", "robust"], "LP method")
    # W10/#112: the weak-IV riders only exist on the LP-IV path. Silently ignoring them under
    # another --method would let an agent believe it got robust bands it never received.
    if method != "iv"
        set = String[]
        mop_f && push!(set, "--mop-f")
        ar_bands && push!(set, "--ar-bands")
        mop_tau == 0.10 || push!(set, "--mop-tau")
        mop_bandwidth == 0 || push!(set, "--mop-bandwidth")
        ar_level == 0.95 || push!(set, "--ar-level")
        ar_grid == 401 || push!(set, "--ar-grid")
        ar_span == 20.0 || push!(set, "--ar-span")
        isempty(set) || throw(CliError("usage/invalid",
            "estimate lp: $(join(set, ", ")) applies only to --method iv (got --method $method)"))
    end
    if method == "standard"
        return _estimate_lp_standard(data, shock, horizons, control_lags, vcov, output, format)
    elseif method == "iv"
        return _estimate_lp_iv(data, shock, horizons, control_lags, vcov, instruments, output, format;
                               mop_f=mop_f, mop_tau=mop_tau, mop_bandwidth=mop_bandwidth,
                               ar_bands=ar_bands, ar_level=ar_level, ar_grid=ar_grid,
                               ar_span=ar_span)
    elseif method == "smooth"
        return _estimate_lp_smooth(data, shock, horizons, knots, lambda, output, format)
    elseif method == "state"
        return _estimate_lp_state(data, shock, horizons, state_var, gamma, transition, output, format)
    elseif method == "propensity"
        return _estimate_lp_propensity(data, treatment, horizons, score_method, output, format)
    elseif method == "robust"
        return _estimate_lp_robust(data, treatment, horizons, score_method, output, format)
    end
end

"""Output LP coefficient table: shock coefficients + SEs per horizon per response variable."""
function _output_lp_coef_table(irf_result, varnames, horizons;
                               title::String="", format::String="table", output::String="",
                               key::String="")
    n_h = size(irf_result.values, 1)
    n_resp = size(irf_result.values, 2)
    has_se = hasproperty(irf_result, :se) && !isnothing(irf_result.se) &&
             size(irf_result.se) == size(irf_result.values)

    coef_df = DataFrame()
    coef_df.Horizon = 0:(n_h - 1)
    for vi in 1:n_resp
        vname = vi <= length(varnames) ? varnames[vi] : "var_$vi"
        coeffs = irf_result.values[:, vi]
        coef_df[!, Symbol(vname)] = round.(coeffs; digits=6)
        if has_se
            ses = irf_result.se[:, vi]
            tstats = coeffs ./ ses
            coef_df[!, Symbol("$(vname)_se")] = round.(ses; digits=6)
            coef_df[!, Symbol("$(vname)_t")] = round.(tstats; digits=3)
        end
    end

    output_result(coef_df; format=Symbol(format), output=output, title=title, key=key)
end

function _estimate_lp_standard(data, shock, horizons, control_lags, vcov, output, format)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    shock_name = _shock_name(varnames, shock)

    _status("Estimating Local Projections: method=standard")
    _status("  Shock: $shock_name, Horizons: $horizons, Lags: $control_lags, VCov: $vcov")
    _status("  Variables: $(join(varnames, ", "))")
    _status()

    model = estimate_lp(Y, shock, horizons; lags=control_lags, cov_type=Symbol(vcov))
    irf_result = lp_irf(model)

    _output_lp_coef_table(irf_result, varnames, horizons;
        title="LP Coefficients ($shock_name → responses)", format=format, output=output,
        key="lp_coefficients")

    _status()
    # `LPModel.T_eff` is a Vector{Int} of length H+1 — the sample shrinks as h grows. The old
    # code nested the whole vector into one kv cell (the `data describe` defect: an agent
    # reading a scalar-looking key gets an array). Report the endpoints instead. Found while
    # fixing the LP-IV path in W10/#112.
    t_eff = collect(Int.(model.T_eff))
    output_kv(Pair{String,Any}[
        "Effective observations (h=0)" => first(t_eff),
        "Effective observations (min)" => minimum(t_eff),
        "Covariance estimator" => vcov,
        "Horizons" => horizons,
        "Control lags" => control_lags,
    ]; format=format, title="Estimation Summary", key="lp_estimation_summary")
    return model
end

function _estimate_lp_iv(data, shock, horizons, control_lags, vcov, instruments, output, format;
                         mop_f::Bool=false, mop_tau::Float64=0.10, mop_bandwidth::Int=0,
                         ar_bands::Bool=false, ar_level::Float64=0.95, ar_grid::Int=401,
                         ar_span::Float64=20.0)
    # was a bare error() → untyped exit 1 for an ordinary missing-option mistake
    isempty(instruments) && throw(CliError("usage/missing",
        "estimate lp --method iv requires --instruments <file.csv>";
        hint="the instruments live in their own CSV, one column per excluded instrument"))
    # MOP publishes only the four tabulated simplified critical values; anything else is an
    # untyped ArgumentError from inside the estimator.
    if mop_f
        mop_tau in (0.05, 0.10, 0.20, 0.30) || throw(CliError("usage/invalid",
            "estimate lp: --mop-tau must be one of 0.05, 0.10, 0.20, 0.30 (got $mop_tau)"))
        mop_bandwidth >= 0 || throw(CliError("usage/invalid",
            "estimate lp: --mop-bandwidth must be ≥ 0 (got $mop_bandwidth)"))
    end
    if ar_bands
        (0.0 < ar_level < 1.0) || throw(CliError("usage/invalid",
            "estimate lp: --ar-level must be in (0, 1) (got $ar_level)"))
        ar_grid >= 5 || throw(CliError("usage/invalid",
            "estimate lp: --ar-grid must be ≥ 5 (got $ar_grid)"))
        ar_span > 0.0 || throw(CliError("usage/invalid",
            "estimate lp: --ar-span must be > 0 (got $ar_span)"))
    end

    Y, varnames = load_multivariate_data(data)
    Z, _ = load_multivariate_data(instruments)
    n = size(Y, 2)
    shock_name = _shock_name(varnames, shock)

    _status("Estimating LP-IV: shock=$shock_name, horizons=$horizons, instruments=$(size(Z, 2))")
    _status()

    model = try
        estimate_lp_iv(Y, shock, Z, horizons; lags=control_lags, cov_type=Symbol(vcov),
                       varnames=varnames)
    catch e
        throw(_domain_or_data_error(e, "LP-IV estimation"))
    end

    # First-stage diagnostics.
    # W10/#112 — THIS LEAF WAS DEAD ON REAL MEMs. `weak_instrument_test(::LPIVModel)` returns
    # `(F_stats, weak_horizons, min_F, passes_threshold, threshold)`; the old code read
    # `wi.F_stat`, a name real does not have, so every `estimate lp --method iv` invocation
    # exited 1 with an untyped FieldError. The mock had INVENTED `(F_stat=…, is_weak=…)`,
    # which kept T1/T2 green, and `estimate lp` had T3 coverage only for --method standard.
    # Same class as #84: a mock more permissive than real hides a guaranteed production crash.
    # The F is per-horizon (LP-IV re-estimates the first stage at every h), so the summary
    # reports the MINIMUM — the binding one — not a single scalar that does not exist.
    wi = weak_instrument_test(model)
    f_stats = collect(Float64.(wi.F_stats))
    min_f = Float64(wi.min_F)
    _status("First-stage diagnostics:")
    _status("  F-statistic: min $(round(min_f; digits=2)) over $(length(f_stats)) horizon(s)")
    if !wi.passes_threshold
        _status_styled("  Warning: Weak instruments (F < 10) at horizon(s) " *
            "$(join(wi.weak_horizons .- 1, ", "))\n"; color=:yellow)
    else
        _status_styled("  Instruments appear strong (F >= 10 at every horizon)\n"; color=:green)
    end
    _status()

    irf_result = lp_iv_irf(model)

    _output_lp_coef_table(irf_result, varnames, horizons;
        title="LP-IV Coefficients ($shock_name → responses)", format=format, output=output,
        key="lp_coefficients")

    _status()
    # `T_eff` is also per-horizon (the sample shrinks as h grows), so nesting the whole
    # vector in one kv cell would repeat the `data describe` defect — report the endpoints.
    t_eff = collect(Int.(model.T_eff))
    output_kv(Pair{String,Any}[
        "Effective observations (h=0)" => first(t_eff),
        "Effective observations (min)" => minimum(t_eff),
        "Covariance estimator" => vcov,
        "First-stage F (min)" => round(min_f; digits=2),
        "First-stage F (h=0)" => round(first(f_stats); digits=2),
        "Weak horizons" => length(wi.weak_horizons),
        "Instruments" => size(Z, 2),
    ]; format=format, title="LP-IV Estimation Summary", key="lp_estimation_summary")

    # ── W10/#112: weak-instrument-robust LP-IV diagnostics ──────────────────
    if mop_f
        mop = try
            montiel_olea_pflueger_f(model; tau=mop_tau, bandwidth=mop_bandwidth)
        catch e
            throw(_domain_or_data_error(e, "Montiel Olea-Pflueger effective F"))
        end
        _status()
        output_kv(Pair{String,Any}[
            "f_effective"     => round(Float64(mop.f_effective); digits=4),
            "critical_value"  => round(Float64(mop.critical_value); digits=4),
            "tau"             => Float64(mop.tau),
            "weak"            => mop.weak,
            "n_instruments"   => mop.n_instruments,
            "bandwidth"       => mop.bandwidth,
            "f_naive"         => round(Float64(mop.f_naive); digits=4),
        ]; format=format, title="Montiel Olea-Pflueger Effective F")
        mop.weak && _status_styled(
            "  Effective F $(round(Float64(mop.f_effective); digits=2)) < MOP critical value " *
            "$(round(Float64(mop.critical_value); digits=2)): the 2SLS bands above are unreliable — " *
            "rerun with --ar-bands for weak-instrument-robust bands\n"; color=:yellow)
    end

    if ar_bands
        band = try
            lp_iv_ar_band(model; level=ar_level, n_grid=ar_grid, span=ar_span)
        catch e
            throw(_domain_or_data_error(e, "LP-IV Anderson-Rubin bands"))
        end
        # Second output_result on this handler → its own --output path, or the coefficient
        # table written above would be silently overwritten.
        output_result(_lp_ar_band_table(band);
            format=Symbol(format), output=_per_var_output_path(output, "ar_bands"),
            title="LP-IV Anderson-Rubin Bands ($(round(Int, 100 * ar_level))%)",
            key="lp_iv_anderson_rubin_bands")
        n_unb = count(!, band.bounded)
        n_unb > 0 && _status_styled(
            "  $n_unb of $(length(band.bounded)) AR cells are unbounded: at those horizons the " *
            "instrument cannot bound the response and the Wald band is over-confident\n";
            color=:yellow)
    end
    return model
end

"""
Tidy (C051) rendering of an `LPIVARBand`: one row per horizon × response.

`lower`/`upper` are the ENVELOPE of the AR set, which may be `±Inf` — an unbounded AR set
is the substantive finding under weak identification, so it is reported rather than
truncated (`_json_safe` renders the infinities as `"Inf"`/`"-Inf"` in the envelope). A set
that is a union of disjoint components carries `n_components > 1`; `is_empty` marks the
over-identification rejection where no value of θ is consistent with the instruments.
"""
function _lp_ar_band_table(band)
    H = band.horizon
    resp = band.response_names
    rows = NamedTuple[]
    for j in eachindex(resp), h in 0:H
        i = h + 1
        push!(rows, (horizon = h,
                     response = resp[j],
                     irf = Float64(band.point[i, j]),
                     ar_lower = Float64(band.lower[i, j]),
                     ar_upper = Float64(band.upper[i, j]),
                     n_components = length(band.sets[i, j]),
                     bounded = band.bounded[i, j],
                     is_empty = band.is_empty[i, j],
                     wald_lower = Float64(band.wald_lower[i, j]),
                     wald_upper = Float64(band.wald_upper[i, j]),
                     bandwidth = band.bandwidths[i, j]))
    end
    return DataFrame(rows)
end

function _estimate_lp_smooth(data, shock, horizons, knots, lambda, output, format)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    lam = if lambda == 0.0
        _status("Cross-validating smoothing parameter...")
        cross_validate_lambda(Y, shock, horizons)
    else
        lambda
    end

    shock_name = _shock_name(varnames, shock)
    _status("Estimating Smooth LP: shock=$shock_name, horizons=$horizons, knots=$knots, λ=$(round(lam; digits=4))")
    _status()

    model = estimate_smooth_lp(Y, shock, horizons; n_knots=knots, lambda=lam)
    irf_result = smooth_lp_irf(model)

    _output_lp_coef_table(irf_result, varnames, horizons;
        title="Smooth LP Coefficients ($shock_name → responses)", format=format, output=output,
        key="lp_coefficients")

    _status()
    output_kv(Pair{String,Any}[
        "Smoothing parameter (λ)" => round(lam; digits=6),
        "B-spline knots" => knots,
        "Horizons" => horizons,
    ]; format=format, title="Smooth LP Estimation Summary", key="lp_estimation_summary")
    return model
end

function _estimate_lp_state(data, shock, horizons, state_var, gamma, transition, output, format)
    isnothing(state_var) && error("state-dependent LP requires --state-var=<idx>")

    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    shock_name = _shock_name(varnames, shock)
    state_name = _var_name(varnames, state_var)

    _status("Estimating State-Dependent LP: shock=$shock_name, state=$state_name, γ=$gamma, transition=$transition")
    _status()

    state_vec = Y[:, state_var]
    model = estimate_state_lp(Y, shock, state_vec, horizons; gamma=gamma)
    results = state_irf(model)

    for (regime, label) in [(:expansion, "Expansion"), (:recession, "Recession")]
        irf_result = getfield(results, regime)
        # Both regimes are emitted in the SAME invocation, so the two tables get distinct —
        # but still statically known — keys (`lp_coefficients_expansion`/`_recession`).
        _output_lp_coef_table(irf_result, varnames, horizons;
            title="State LP Coefficients — $label ($shock_name → responses)",
            format=format, key=_table_key("lp_coefficients", label),
            output=isempty(output) ? "" : replace(output, "." => "_$(lowercase(label))."))
        _status()
    end

    diff_test = test_regime_difference(model)
    jt = diff_test.joint_test
    _status("Regime Difference Test:")
    _status("  Avg t-statistic: $(round(jt.avg_t_stat; digits=3))")
    _status("  p-value: $(round(jt.p_value; digits=4))")
    if jt.p_value < 0.05
        _status_styled("  → Significant regime differences at 5%\n"; color=:green)
    else
        _status_styled("  → No significant regime differences at 5%\n"; color=:yellow)
    end
    return model
end

function _estimate_lp_propensity(data, treatment, horizons, score_method, output, format)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    treat_name = _var_name(varnames, treatment)

    _status("Estimating Propensity Score LP: treatment=$treat_name, horizons=$horizons, method=$score_method")
    _status()

    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = estimate_propensity_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    # Propensity score diagnostics
    diag = propensity_diagnostics(model)
    ps = diag.propensity_summary
    _status("Propensity Score Diagnostics:")
    _status("  Treated mean score: $(round(ps.treated.mean; digits=4))")
    _status("  Control mean score: $(round(ps.control.mean; digits=4))")
    _status("  Max weighted SMD: $(round(diag.balance.max_weighted; digits=4))")
    _status()

    irf_result = propensity_irf(model)

    _output_lp_coef_table(irf_result, varnames, horizons;
        title="Propensity Score LP: ATE Estimates ($treat_name)", format=format, output=output,
        key="lp_coefficients")
    return model
end

function _estimate_lp_robust(data, treatment, horizons, score_method, output, format)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    treat_name = _var_name(varnames, treatment)

    _status("Estimating Doubly Robust LP: treatment=$treat_name, horizons=$horizons, method=$score_method")
    _status()

    treatment_bool = Bool.(Y[:, treatment] .> median(Y[:, treatment]))
    covariates = Y[:, setdiff(1:size(Y,2), [treatment])]

    model = doubly_robust_lp(Y, treatment_bool, covariates, horizons;
        ps_method=Symbol(score_method))

    # Diagnostics
    diag = propensity_diagnostics(model)
    ps = diag.propensity_summary
    _status("Doubly Robust Diagnostics:")
    _status("  Treated mean score: $(round(ps.treated.mean; digits=4))")
    _status("  Control mean score: $(round(ps.control.mean; digits=4))")
    _status("  Max weighted SMD: $(round(diag.balance.max_weighted; digits=4))")
    _status()

    irf_result = propensity_irf(model)

    _output_lp_coef_table(irf_result, varnames, horizons;
        title="Doubly Robust LP: ATE Estimates ($treat_name)", format=format, output=output,
        key="lp_coefficients")
    return model
end

# ── ARIMA ──────────────────────────────────────────────────

function _estimate_arima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          max_p::Int=5, max_d::Int=2, max_q::Int=5,
                          criterion::String="bic", method::String="css_mle",
                          format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    method_sym = Symbol(method)
    safe_method = method_sym == :css_mle ? :mle : method_sym

    model = if isnothing(p)
        crit_sym = Symbol(lowercase(criterion))
        _status("Auto ARIMA: variable=$vname, observations=$(length(y))")
        _status("  Search: p=0:$max_p, d=0:$max_d, q=0:$max_q, criterion=$criterion, method=$method")
        _status()
        m = auto_arima(y; max_p=max_p, max_q=max_q, max_d=max_d, criterion=crit_sym, method=safe_method)
        label = _model_label(ar_order(m), diff_order(m), ma_order(m))
        _status_styled("Selected model: $label\n"; bold=true)
        _status()
        m
    else
        label = _model_label(p, d, q)
        _status("Estimating $label: variable=$vname, observations=$(length(y)), method=$method")
        _status()
        _estimate_arima_model(y, p, d, q; method=method_sym)
    end

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    _arima_coef_table(model; format=format, output=output, title="$label Coefficients ($vname)",
                      key="arima_coefficients")

    _status()
    output_kv(Pair{String,Any}[
        "AIC" => round(aic(model); digits=4),
        "BIC" => round(bic(model); digits=4),
        "Log-likelihood" => round(loglikelihood(model); digits=4),
    ]; format=format, title="Information Criteria")
    return model
end

# ── W6/#108: multiplicative seasonal ARIMA ────────────────────────────────────
#
# MEMs#341. `SARIMAModel <: AbstractARIMAModel`, so residuals/predict come from the
# abstract StatsAPI block and `plot_result(::AbstractARIMAModel)` covers it — an ABSTRACT
# dispatch, which is exactly the case a by-name grep for `plot_result(::SARIMAModel)` would
# have missed. `forecast(::SARIMAModel, h)` returns an ARIMAForecast, which has its own
# recipe, so BOTH the estimate and forecast leaves are genuinely plot-capable.

"""Shared SARIMA fit for the four leaves. `--p` unset (or `--auto`) routes to `auto_sarima`,
which selects d/D itself unless they are pinned."""
function _sarima_fit(data::String, column::Int, p, d::Int, q::Int, P::Int, D::Int, Q::Int,
                     s::Int, auto::Bool, max_p::Int, max_q::Int, max_P::Int, max_Q::Int,
                     criterion::String, method::String, max_iter::Int, no_intercept::Bool,
                     leaf::String)
    s >= 1 || throw(CliError("usage/invalid", "$leaf: --s must be ≥ 1 (got $s)"))
    max_iter >= 1 || throw(CliError("usage/invalid", "$leaf: --max-iter must be ≥ 1 (got $max_iter)"))
    for (nm, v) in (("d", d), ("q", q), ("P", P), ("D", D), ("Q", Q))
        v >= 0 || throw(CliError("usage/invalid", "$leaf: --$nm must be ≥ 0 (got $v)"))
    end
    isnothing(p) || p >= 0 || throw(CliError("usage/invalid", "$leaf: --p must be ≥ 0 (got $p)"))
    for (nm, v) in (("max-p", max_p), ("max-q", max_q), ("max-P", max_P), ("max-Q", max_Q))
        v >= 0 || throw(CliError("usage/invalid", "$leaf: --$nm must be ≥ 0 (got $v)"))
    end
    crit = lowercase(criterion)
    crit in ("aic", "bic") || throw(CliError("usage/invalid",
        "$leaf: --criterion must be aic or bic (got '$criterion')"))
    y, vname = load_univariate_series(data, column)
    use_auto = auto || isnothing(p)
    model = try
        if use_auto
            auto_sarima(y, s; max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                        criterion=Symbol(crit), method=Symbol(method),
                        include_intercept=!no_intercept)
        else
            estimate_sarima(y, p, d, q, P, D, Q, s; method=Symbol(method),
                            include_intercept=!no_intercept, max_iter=max_iter)
        end
    catch e
        throw(_domain_or_data_error(e, "SARIMA estimation"))
    end
    return model, vname, use_auto
end

"""`SARIMA(p,d,q)(P,D,Q)[s]` label, the conventional Box-Jenkins notation."""
_sarima_label(m) = "SARIMA($(m.p),$(m.d),$(m.q))($(m.P),$(m.D),$(m.Q))[$(m.s)]"

function _sarima_coef_df(m)
    names = String[]; vals = Float64[]
    m.c != 0 && (push!(names, "intercept"); push!(vals, Float64(m.c)))
    for (i, v) in enumerate(m.phi);   push!(names, "ar$i");    push!(vals, Float64(v)); end
    for (i, v) in enumerate(m.theta); push!(names, "ma$i");    push!(vals, Float64(v)); end
    for (i, v) in enumerate(m.Phi);   push!(names, "sar$i");   push!(vals, Float64(v)); end
    for (i, v) in enumerate(m.Theta); push!(names, "sma$i");   push!(vals, Float64(v)); end
    push!(names, "sigma2"); push!(vals, Float64(m.sigma2))
    return DataFrame(parameter=names, estimate=round.(vals; digits=6))
end

function _estimate_sarima(; data::String, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                           P::Int=0, D::Int=0, Q::Int=0, s::Int=12, auto::Bool=false,
                           max_p::Int=2, max_q::Int=2, max_P::Int=1, max_Q::Int=1,
                           criterion::String="aic", method::String="css_mle",
                           max_iter::Int=500, no_intercept::Bool=false,
                           plot::Bool=false, plot_save::String="",
                           output::String="", format::String="table")
    model, vname, used_auto = _sarima_fit(data, column, p, d, q, P, D, Q, s, auto,
                                          max_p, max_q, max_P, max_Q, criterion, method,
                                          max_iter, no_intercept, "estimate sarima")
    lbl = _sarima_label(model)
    _status(used_auto ? "Auto SARIMA (s=$s): variable=$vname → $lbl" :
                        "Estimating $lbl: variable=$vname")
    _status()
    output_result(_sarima_coef_df(model); format=Symbol(format), output=output,
                  title="$lbl Coefficients ($vname)", key="sarima_coefficients")
    _status()
    output_kv(Pair{String,Any}[
        "AIC"            => round(Float64(aic(model)); digits=4),
        "BIC"            => round(Float64(bic(model)); digits=4),
        "Log-likelihood" => round(Float64(loglikelihood(model)); digits=4),
        "n_obs"          => length(model.y),
        "n_effective"    => length(model.residuals),
    ]; format=format, title="Information Criteria")
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    return model
end

function _forecast_sarima(; data::String="", result=nothing, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                           P::Int=0, D::Int=0, Q::Int=0, s::Int=12, auto::Bool=false,
                           max_p::Int=2, max_q::Int=2, max_P::Int=1, max_Q::Int=1,
                           criterion::String="aic", method::String="css_mle",
                           max_iter::Int=500, no_intercept::Bool=false,
                           horizons::Int=12, ci_level::Float64=0.95,
                           plot::Bool=false, plot_save::String="",
                           output::String="", format::String="table", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast sarima")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="SARIMA Forecast", key="sarima_forecast", plot, plot_save)
    horizons >= 1 || throw(CliError("usage/invalid",
        "forecast sarima: --horizons must be ≥ 1 (got $horizons)"))
    (0 < ci_level < 1) || throw(CliError("usage/invalid",
        "forecast sarima: --ci-level must satisfy 0 < level < 1 (got $ci_level)"))
    model, vname, _ = _sarima_fit(data, column, p, d, q, P, D, Q, s, auto,
                                  max_p, max_q, max_P, max_Q, criterion, method,
                                  max_iter, no_intercept, "forecast sarima")
    lbl = _sarima_label(model)
    _status("$lbl forecast (h=$horizons): variable=$vname, ci=$ci_level"); _status()
    fc = try
        forecast(model, horizons; conf_level=ci_level)
    catch e
        throw(_domain_or_data_error(e, "SARIMA forecast"))
    end
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="$lbl Forecast for $vname (h=$horizons, $(Int(round(ci_level*100)))% CI)",
                  key="sarima_forecast")
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    return (; model, result=fc)
end

function _predict_sarima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                          P::Int=0, D::Int=0, Q::Int=0, s::Int=12, auto::Bool=false,
                          max_p::Int=2, max_q::Int=2, max_P::Int=1, max_Q::Int=1,
                          criterion::String="aic", method::String="css_mle",
                          max_iter::Int=500, no_intercept::Bool=false,
                          output::String="", format::String="table", model=nothing)
    m, vname = if model === nothing
        mm, vn, _ = _sarima_fit(data, column, p, d, q, P, D, Q, s, auto, max_p, max_q,
                                max_P, max_Q, criterion, method, max_iter, no_intercept,
                                "predict sarima")
        (mm, vn)
    else
        (model, "model")
    end
    f = predict(m)
    _status("$(_sarima_label(m)) fitted values: variable=$vname, n=$(length(f))"); _status()
    return output_result(DataFrame(t=1:length(f), fitted=round.(Float64.(f); digits=6));
                         format=Symbol(format), output=output,
                         title="$(_sarima_label(m)) In-Sample Predictions for $vname",
                         key="sarima_predictions")
end

function _residuals_sarima(; data::String="", column::Int=1, p=nothing, d::Int=0, q::Int=0,
                            P::Int=0, D::Int=0, Q::Int=0, s::Int=12, auto::Bool=false,
                            max_p::Int=2, max_q::Int=2, max_P::Int=1, max_Q::Int=1,
                            criterion::String="aic", method::String="css_mle",
                            max_iter::Int=500, no_intercept::Bool=false,
                            output::String="", format::String="table", model=nothing)
    m, vname = if model === nothing
        mm, vn, _ = _sarima_fit(data, column, p, d, q, P, D, Q, s, auto, max_p, max_q,
                                max_P, max_Q, criterion, method, max_iter, no_intercept,
                                "residuals sarima")
        (mm, vn)
    else
        (model, "model")
    end
    r = residuals(m)
    _status("$(_sarima_label(m)) residuals: variable=$vname, n=$(length(r))"); _status()
    return output_result(DataFrame(t=1:length(r), residual=round.(Float64.(r); digits=6));
                         format=Symbol(format), output=output,
                         title="$(_sarima_label(m)) Residuals for $vname",
                         key="sarima_residuals")
end

# ARIMA helpers (from old arima.jl)
function _estimate_arima_model(y::Vector{Float64}, p::Int, d::Int, q::Int; method::Symbol=:css_mle)
    if d == 0 && q == 0
        ar_method = method in (:ols, :mle) ? method : :mle
        return estimate_ar(y, p; method=ar_method)
    elseif d == 0 && p == 0
        return estimate_ma(y, q; method=method)
    elseif d == 0
        return estimate_arma(y, p, q; method=method)
    else
        return estimate_arima(y, p, d, q; method=method)
    end
end

function _model_label(p::Int, d::Int, q::Int)
    if d == 0 && q == 0
        return "AR($p)"
    elseif d == 0 && p == 0
        return "MA($q)"
    elseif d == 0
        return "ARMA($p,$q)"
    else
        return "ARIMA($p,$d,$q)"
    end
end

function _arima_coef_table(model; format::String="table", output::String="", title::String="Coefficients",
                           key::String="")
    c = coef(model)
    p_order = ar_order(model)
    q_order = ma_order(model)
    param_names = String[]
    for i in 1:p_order
        push!(param_names, "ar$i")
    end
    for i in 1:q_order
        push!(param_names, "ma$i")
    end
    n_named = p_order + q_order
    for i in (n_named+1):length(c)
        push!(param_names, "const$i")
    end

    coef_df = try
        se = stderror(model)
        z = c ./ se
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        DataFrame(parameter=param_names, estimate=round.(c; digits=6),
                  std_error=round.(se; digits=6),
                  z_stat=round.(z; digits=3), p_value=round.(pv; digits=4))
    catch
        DataFrame(parameter=param_names, estimate=round.(c; digits=6))
    end

    output_result(coef_df; format=Symbol(format), output=output, title=title, key=key)
end

# ── ARFIMA (fractional integration / long memory) ──────────

# ── #73: arfima downstream verbs ─────────────────────────────────────────────
# `forecast(::ARFIMAModel, h; conf_level, trunc_lag)` returns an ARIMAForecast — the
# same type `forecast arima` renders — so the forecast leaf reuses that path. There are
# no StatsAPI predict/residuals methods for ARFIMAModel, but `fitted` and `residuals`
# are real FIELDS (arima/types.jl), so predict/residuals read them directly.
#
# All three mirror `estimate arfima`'s option set (--column/--p/--q/--method/--max-iter,
# plus --d0) so a non-default fit can be reproduced; the `:arima` kind in
# FITTED_MODEL_KINDS only supplies --column, which would silently pin p=q=0.

"""Refit an ARFIMA for the downstream verbs, mirroring `_estimate_arfima` exactly."""
function _arfima_refit(data, column, p, q, method, d0, max_iter, model)
    model === nothing || return model, "model"
    y, vname = load_univariate_series(data, column)
    d0v = isnothing(d0) ? nothing : Float64(d0)
    m = try
        estimate_arfima(y, p, q; method=Symbol(method), d0=d0v, max_iter=max_iter)
    catch e
        throw(_long_memory_error(e, "ARFIMA"))
    end
    return m, vname
end

function _forecast_arfima(; data::String="", result=nothing, column::Int=1, p::Int=0, q::Int=0,
        method::String="css", d0=nothing, max_iter::Int=500,
        horizons::Int=12, confidence::Float64=0.95, trunc_lag::Int=200,
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast arfima")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="ARFIMA Forecast", key="arfima_forecast", plot, plot_save)
    horizons >= 1 || throw(CliError("usage/invalid", "forecast arfima: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < confidence < 1.0) || throw(CliError("usage/invalid",
        "forecast arfima: --confidence must be in (0, 1) (got $confidence)"))
    trunc_lag >= 1 || throw(CliError("usage/invalid",
        "forecast arfima: --trunc-lag must be ≥ 1 (got $trunc_lag)"))
    m, vname = _arfima_refit(data, column, p, q, method, d0, max_iter, model)
    _status("ARFIMA($p,d,$q) Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=confidence, trunc_lag=trunc_lag)
    catch e
        throw(_long_memory_error(e, "ARFIMA forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    output_result(DataFrame(
            horizon = collect(1:horizons),
            forecast = round.(Float64.(collect(fc.forecast)); digits=6),
            lower = round.(Float64.(collect(fc.ci_lower)); digits=6),
            upper = round.(Float64.(collect(fc.ci_upper)); digits=6));
        format=Symbol(format), output=output,
        title="ARFIMA($p,d,$q) Forecast for $vname", key="arfima_forecast")
    return (; model=m, result=fc)
end

function _predict_arfima(; data::String="", column::Int=1, p::Int=0, q::Int=0,
        method::String="css", d0=nothing, max_iter::Int=500,
        output::String="", format::String="table", model=nothing)
    m, vname = _arfima_refit(data, column, p, q, method, d0, max_iter, model)
    _status("ARFIMA($p,d,$q) in-sample fitted values: variable=$vname"); _status()
    f = Float64.(collect(m.fitted))
    output_result(DataFrame(t=1:length(f), fitted=round.(f; digits=6));
        format=Symbol(format), output=output,
        title="ARFIMA($p,d,$q) In-Sample Predictions for $vname", key="arfima_predictions")
    return m.fitted
end

function _residuals_arfima(; data::String="", column::Int=1, p::Int=0, q::Int=0,
        method::String="css", d0=nothing, max_iter::Int=500,
        output::String="", format::String="table", model=nothing)
    m, vname = _arfima_refit(data, column, p, q, method, d0, max_iter, model)
    _status("ARFIMA($p,d,$q) residuals: variable=$vname"); _status()
    r = Float64.(collect(m.residuals))
    output_result(DataFrame(t=1:length(r), residual=round.(r; digits=6));
        format=Symbol(format), output=output,
        title="ARFIMA($p,d,$q) Residuals for $vname", key="arfima_residuals")
    return m.residuals
end

function _estimate_arfima(; data::String, column::Int=1, p::Int=0, q::Int=0,
                           method::String="css", d0=nothing, max_iter::Int=500,
                           format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    method_sym = Symbol(method)

    _status("Estimating ARFIMA($p,d,$q): variable=$vname, observations=$(length(y)), method=$method")
    _status()

    model = try
        d0v = isnothing(d0) ? nothing : Float64(d0)
        estimate_arfima(y, p, q; method=method_sym, d0=d0v, max_iter=max_iter)
    catch e
        throw(_long_memory_error(e, "ARFIMA estimation"))
    end

    label = "ARFIMA($(model.p),$(round(model.d; digits=4)),$(model.q))"

    # Coefficient table: hand-built. ARFIMAModel is not a MEMs coef-table type
    # (`DataFrame(model)`), so the tidy coefficient path does not apply — this is a
    # documented C051 exception (like ARIMA/volatility). Ordering is [c, d, phi.., theta..].
    _arfima_coef_table(model; format=format, output=output,
                       title="$label Coefficients ($vname)", key="arfima_coefficients")

    _status()
    output_kv(Pair{String,Any}[
        "d (frac. integ.)" => round(model.d; digits=6),
        "d std. error"     => round(model.d_se; digits=6),
        "Log-likelihood"   => round(model.loglik; digits=4),
        "AIC"              => round(model.aic; digits=4),
        "BIC"              => round(model.bic; digits=4),
        "Converged"        => model.converged,
    ]; format=format, title="ARFIMA Diagnostics ($vname)", key="arfima_diagnostics")
    return model
end

function _arfima_coef_table(model; format::String="table", output::String="", title::String="Coefficients",
                            key::String="")
    c = coef(model)                                 # [c, d, phi.., theta..]
    p_order = ar_order(model)
    q_order = ma_order(model)
    param_names = String["const", "d"]
    for i in 1:p_order
        push!(param_names, "ar$i")
    end
    for i in 1:q_order
        push!(param_names, "ma$i")
    end
    # Guard against any length mismatch (keep the table well-formed).
    for i in (length(param_names)+1):length(c)
        push!(param_names, "param$i")
    end
    param_names = param_names[1:length(c)]

    coef_df = try
        se = stderror(model)
        z = c ./ se
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        DataFrame(parameter=param_names, estimate=round.(c; digits=6),
                  std_error=round.(se; digits=6),
                  z_stat=round.(z; digits=3), p_value=round.(pv; digits=4))
    catch
        DataFrame(parameter=param_names, estimate=round.(c; digits=6))
    end

    output_result(coef_df; format=Symbol(format), output=output, title=title, key=key)
end

# ── GMM ────────────────────────────────────────────────────

function _estimate_gmm(; data::String, config::String="",
                        weighting::String="twostep",
                        output::String="", format::String="table")
    isempty(config) && error("GMM requires a --config=<file.toml> specifying moment conditions and instruments")

    Y, varnames = load_multivariate_data(data)

    cfg = load_config(config)
    gmm_cfg = get_gmm(cfg)

    weighting_map = Dict("identity" => :identity, "optimal" => :optimal,
                         "twostep" => :two_step, "iterated" => :iterated)
    w = get(weighting_map, lowercase(weighting), :two_step)

    _status("Estimating GMM: weighting=$weighting")
    _status("  Moment conditions: $(length(gmm_cfg["moment_conditions"]))")
    _status()

    moment_cols = gmm_cfg["moment_conditions"]
    shock_var = if !isempty(moment_cols)
        idx = findfirst(==(moment_cols[1]), varnames)
        isnothing(idx) ? 1 : idx
    else
        1
    end

    models = estimate_lp_gmm(Y, shock_var, 0; lags=4, weighting=w)

    if !isempty(models)
        model = models[1]
        summ = gmm_summary(model)
        jtest = j_test(model)
        _status()
        _status("Hansen's J-test for overidentification:")
        _status("  J-statistic: $(round(jtest.J_stat; digits=4))")
        # M-29 (MEMs#797): j_test(::GMMModel) shares the SMM NaN-under-identity
        # policy (stored NaN p-value) — render n/a with the reason, never a number.
        if isnan(jtest.p_value)
            _status("  p-value: n/a (identity weighting — χ² limit needs efficient weighting)")
        else
            _status("  p-value: $(round(jtest.p_value; digits=4))")
        end
        _status("  Degrees of freedom: $(jtest.df)")

        # NaN < 0.05 is false, so a bare comparison would misreport
        # "Cannot reject" on an undefined p-value — verdict stays silent on NaN.
        if isnan(jtest.p_value)
            _status_styled("  -> J-test n/a under identity weighting (χ² limit needs efficient weighting)\n"; color=:yellow)
        elseif jtest.p_value < 0.05
            _status_styled("  -> Reject valid moment conditions at 5%\n"; color=:yellow)
        else
            _status_styled("  -> Cannot reject valid moment conditions\n"; color=:green)
        end

        if !isempty(output)
            se = stderror(model)
            param_df = DataFrame(parameter=["theta$i" for i in 1:length(model.theta)],
                                 estimate=model.theta, std_error=se)
            output_result(param_df; format=Symbol(format), output=output, title="GMM Estimates")
        end
        return model
    end
end

# ── Factor Models ──────────────────────────────────────────

function _estimate_static(; data::String, nfactors=nothing, criterion::String="ic1",
                           output::String="", format::String="table",
                           plot::Bool=false, plot_save::String="")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        _status("Selecting number of factors via Bai-Ng information criteria...")
        ic = ic_criteria(X, min(20, size(X, 2)))
        r_sym = Symbol("r_", uppercase(criterion))
        optimal_r = getfield(ic, r_sym)
        _status("  $criterion suggests $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    _status("Estimating static factor model: $r factors, $(size(X, 2)) variables, $(size(X, 1)) observations")
    _status()

    model = estimate_factors(X, r; varnames=varnames)   # carried on the model since MEMs#538
    _maybe_plot(model; plot=plot, plot_save=plot_save)

    scree = scree_plot_data(model)
    scree_df = DataFrame(component=scree.factors, eigenvalue=scree.explained_variance,
                         cumulative=scree.cumulative_variance)
    output_result(scree_df; format=Symbol(format), title="Scree Data (Eigenvalues & Variance Shares)")
    _status()

    loadings = model.loadings
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format), output=output, title="Factor Loadings")
    return model
end

function _estimate_dynamic(; data::String, nfactors=nothing, factor_lags::Int=1,
                            method::String="twostep", output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    X, varnames = load_multivariate_data(data)

    r = if isnothing(nfactors)
        _status("Selecting number of factors...")
        ic = ic_criteria(X, min(10, size(X, 2)))
        optimal_r = ic.r_IC1
        _status("  Auto-selected $optimal_r factors")
        optimal_r
    else
        nfactors
    end

    _status("Estimating dynamic factor model: $r factors, $factor_lags lags, method=$method")
    _status()

    model = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    _maybe_plot(model; plot=plot, plot_save=plot_save)

    stable_result = is_stationary(model)
    stable = stable_result isa Bool ? stable_result : stable_result.is_stationary
    if stable
        _status_styled("Factor VAR is stationary\n"; color=:green)
    else
        _status_styled("Factor VAR is not stationary\n"; color=:yellow)
    end
    _status()

    loadings = model.loadings
    loading_df = DataFrame(loadings, ["F$i" for i in 1:r])
    insertcols!(loading_df, 1, :variable => varnames)
    output_result(loading_df; format=Symbol(format), output=output, title="Dynamic Factor Loadings")

    _status()
    _status("Factor VAR Companion Matrix eigenvalues:")
    comp = companion_matrix_factors(model)
    eig_moduli = abs.(eigvals(comp))
    for (i, ev) in enumerate(sort(eig_moduli; rev=true))
        _status("  lambda$i = $(round(ev; digits=6))")
    end
    return model
end

function _estimate_gdfm(; data::String, nfactors=nothing, dynamic_rank=nothing,
                         spectral::String="lag-window",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    X, varnames = load_multivariate_data(data)

    q = if isnothing(dynamic_rank)
        _status("Selecting dynamic rank...")
        ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
        q_opt = ic.q_ratio
        _status("  Auto-selected $q_opt dynamic factors")
        q_opt
    else
        dynamic_rank
    end

    r = if isnothing(nfactors)
        _status("Selecting static rank...")
        ic_static = ic_criteria(X, min(20, size(X, 2)))
        r_opt = ic_static.r_IC1
        _status("  Auto-selected $r_opt static factors")
        r_opt
    else
        nfactors
    end

    haskey(_GDFM_SPECTRAL, spectral) || throw(CliError("usage/invalid",
        "estimate gdfm: --spectral must be lag-window|smoothed-periodogram (got '$spectral')"))
    _status("Estimating GDFM: static rank=$r, dynamic rank=$q, spectral=$spectral")
    _status()

    model = estimate_gdfm(X, q; r=r, spectral=_GDFM_SPECTRAL[spectral])

    var_shares = common_variance_share(model)
    var_df = DataFrame(variable=varnames, common_variance_share=round.(var_shares; digits=4))
    output_result(var_df; format=Symbol(format), output=output,
                  title="GDFM Common Variance Shares")
    _maybe_plot(model; plot=plot, plot_save=plot_save)

    _status()
    _status("Average common variance share: $(round(mean(var_shares); digits=4))")
    return model
end

# Volatility estimate handlers live in shared.jl (VOL_MODELS / _VOL_ESTIMATE_HANDLERS).

# ── Non-Gaussian ICA ──────────────────────────────────────

function _estimate_fastica(; data::String, lags=nothing, method::String="fastica",
                             contrast::String="logcosh", output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    _status("Non-Gaussian SVAR: method=$method, contrast=$contrast, VAR($p), $n variables")
    _status()

    result = if method == "jade"
        identify_jade(model)
    elseif method == "sobi"
        identify_sobi(model)
    elseif method == "dcov"
        identify_dcov(model)
    elseif method == "hsic"
        identify_hsic(model; _fwd_seed()...)
    else
        identify_fastica(model; contrast=Symbol(contrast), _fwd_seed()...)
    end

    if hasproperty(result, :converged)
        if result.converged
            _status_styled("Converged in $(result.iterations) iterations\n"; color=:green)
        else
            _status_styled("Did not converge after $(result.iterations) iterations\n"; color=:yellow)
        end
    end
    _status()

    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    _status()

    shocks = result.shocks
    T_shocks = size(shocks, 1)
    n_show = min(T_shocks, 10)
    shock_df = DataFrame(shocks[1:n_show, :], ["shock_$i" for i in 1:n])
    insertcols!(shock_df, 1, :t => 1:n_show)
    output_result(shock_df; format=Symbol(format), output=output,
                  title="Structural Shocks (first $n_show observations)",
                  key="structural_shocks")
    return result
end

# ── Non-Gaussian ML ───────────────────────────────────────

function _estimate_ml(; data::String, lags=nothing, distribution::String="student_t",
                        output::String="", format::String="table")
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = length(varnames)

    _status("Non-Gaussian ML SVAR: distribution=$distribution, VAR($p), $n variables")
    _status()

    result = if distribution == "mixture_normal"
        identify_mixture_normal(model)
    elseif distribution == "pml"
        identify_pml(model)
    elseif distribution == "skew_normal"
        identify_skew_normal(model)
    else
        identify_nongaussian_ml(model; distribution=Symbol(distribution))
    end

    b0_df = DataFrame(result.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), title="Structural Impact Matrix (B0)")
    _status()

    output_kv(Pair{String,Any}[
        "Log-likelihood" => round(result.loglik; digits=4),
        "Log-likelihood (Gaussian)" => round(result.loglik_gaussian; digits=4),
        "AIC" => round(result.aic; digits=4),
        "BIC" => round(result.bic; digits=4),
        "Distribution" => string(result.distribution),
    ]; format=format, title="Model Fit")

    if !isempty(result.dist_params)
        _status()
        _status("Distribution parameters:")
        for (k, v) in pairs(result.dist_params)
            if v isa AbstractArray
                _status("  $k = $(round.(v; digits=4))")
            else
                _status("  $k = $(round(v; digits=4))")
            end
        end
    end

    if !isnothing(result.se) && length(result.se) > 0
        _status()
        se_df = DataFrame(
            parameter=["B0[$i,$j]" for i in 1:n for j in 1:n],
            estimate=vec(result.B0),
            std_error=result.se[1:min(length(result.se), n*n)]
        )
        output_result(se_df; format=Symbol(format), output=output,
                      title="Parameter Estimates with Standard Errors")
    end
    return result
end

# ── VECM ─────────────────────────────────────────────────

function _estimate_vecm(; data::String, lags::Int=2, rank::String="auto",
                          deterministic::String="constant", method::String="johansen",
                          significance::Float64=0.05,
                          output::String="", format::String="table")
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, method, significance)
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    _status("Estimating VECM($(p-1)) with $n variables: $(join(varnames, ", "))")
    _status("Cointegration rank: $r, Deterministic: $deterministic, Method: $method")
    _status("Observations: $(size(Y, 1))")
    _status()

    _status_report(() -> report(vecm))

    # Cointegrating vectors (beta)
    beta_df = DataFrame(vecm.beta, ["CV$i" for i in 1:r])
    insertcols!(beta_df, 1, :variable => varnames)
    output_result(beta_df; format=Symbol(format), output=output, title="Cointegrating Vectors (beta)")
    _status()

    # Adjustment coefficients (alpha)
    alpha_df = DataFrame(vecm.alpha, ["CV$i" for i in 1:r])
    insertcols!(alpha_df, 1, :equation => varnames)
    output_result(alpha_df; format=Symbol(format), output=output, title="Adjustment Coefficients (alpha)")
    _status()

    output_kv(Pair{String,Any}[
        "AIC" => round(vecm.aic; digits=4),
        "BIC" => round(vecm.bic; digits=4),
        "HQC" => round(vecm.hqic; digits=4),
        "Log-likelihood" => round(loglikelihood(vecm); digits=4),
    ]; format=format, title="Information Criteria")
    return vecm
end

# ── SVAR (AB-model ML) ─────────────────────────────────────

function _estimate_svar(; data::String, lags=nothing, pattern::String="recursive", config::String="",
                        n_starts::Int=5, max_iter::Int=400,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    n_starts >= 1 || throw(CliError("usage/invalid",
        "estimate svar: --n-starts must be ≥ 1 (got $n_starts)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "estimate svar: --max-iter must be ≥ 1 (got $max_iter)"))
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)
    pat = _load_svar_pattern(config, n, pattern, "estimate svar")

    _status("Estimating SVAR($p) with $n variables: $(join(varnames, ", "))")
    _status("Pattern: $pattern, Starts: $n_starts, Max iterations: $max_iter")
    _status()

    svar = try
        estimate_svar(model, pat; n_starts=n_starts, max_iter=max_iter)
    catch e
        throw(_domain_or_data_error(e, "SVAR estimation"))
    end
    _status_report(() -> report(svar))
    _status()

    a_df = DataFrame(svar.A, varnames)
    insertcols!(a_df, 1, :equation => varnames)
    output_result(a_df; format=Symbol(format), output=output,
                  title="SVAR Contemporaneous Matrix (A)", key="svar_a")
    _status()

    b_df = DataFrame(svar.B, varnames)
    insertcols!(b_df, 1, :equation => varnames)
    output_result(b_df; format=Symbol(format), output=_per_var_output_path(output, "b"),
                  title="SVAR Structural Matrix (B)", key="svar_b")
    _status()

    output_kv(Pair{String,Any}[
        "Log-likelihood" => round(Float64(svar.loglik); digits=4),
        "LR statistic" => round(Float64(svar.lr_stat); digits=4),
        "LR df" => svar.lr_df,
        "LR p-value" => round(Float64(svar.lr_pvalue); digits=6),
        "Identification" => string(svar.identification.status),
        "Overidentifying restrictions" => svar.identification.n_overidentifying,
    ]; format=format, output=_per_var_output_path(output, "summary"), title="SVAR Identification Summary",
        key="svar_summary")

    _maybe_plot(svar; plot=plot, plot_save=plot_save)
    return svar
end

# ── SVEC (VECM structural) ─────────────────────────────────

function _estimate_svec(; data::String, lags::Int=2, rank::String="auto",
                        deterministic::String="constant", method::String="johansen",
                        significance::Float64=0.05, config::String="",
                        n_starts::Int=5, max_iter::Int=400,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    n_starts >= 1 || throw(CliError("usage/invalid",
        "estimate svec: --n-starts must be ≥ 1 (got $n_starts)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "estimate svec: --max-iter must be ≥ 1 (got $max_iter)"))
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, method, significance)
    n = size(Y, 2)
    lr_zeros, sr_zeros = _load_svec_zeros(config, n, "estimate svec")

    _status("Estimating SVEC($(p-1)) with $n variables: $(join(varnames, ", "))")
    _status("Cointegration rank: $(cointegrating_rank(vecm)), Restrictions: " *
            (lr_zeros === nothing && sr_zeros === nothing ? "default (KPSW)" : "custom [svec]"))
    _status()

    svec = try
        identify_svec(vecm; long_run_zeros=lr_zeros, short_run_zeros=sr_zeros,
                      n_starts=n_starts, max_iter=max_iter)
    catch e
        throw(_domain_or_data_error(e, "SVEC identification"))
    end
    _status_report(() -> report(svec))
    _status()

    b0_df = DataFrame(svec.B0, varnames)
    insertcols!(b0_df, 1, :equation => varnames)
    output_result(b0_df; format=Symbol(format), output=output,
                  title="SVEC Contemporaneous Impact Matrix (B0)", key="svec_b0")
    _status()

    xi_df = DataFrame(svec.Xi, varnames)
    insertcols!(xi_df, 1, :equation => varnames)
    output_result(xi_df; format=Symbol(format), output=_per_var_output_path(output, "xi"),
                  title="SVEC Long-Run Impact Matrix (Xi)", key="svec_xi")
    _status()

    output_kv(Pair{String,Any}[
        "Permanent shocks" => svec.n_permanent,
        "Identification" => string(svec.identification.status),
        "Overidentifying restrictions" => svec.identification.n_overidentifying,
    ]; format=format, output=_per_var_output_path(output, "summary"), title="SVEC Identification Summary",
        key="svec_summary")

    _maybe_plot(svec; plot=plot, plot_save=plot_save)
    return svec
end

# ── Panel VAR ─────────────────────────────────────────────

function _estimate_pvar(; data::String, id_col::String="", time_col::String="",
                         lags::Int=1, dependent::String="", predet::String="", exog::String="",
                         transformation::String="fd", steps::String="twostep",
                         method::String="gmm", system::Bool=false, collapse::Bool=false,
                         min_lag_endo::Int=2, max_lag_endo::Int=99,
                         output::String="", format::String="table")
    validate_method(method, ["gmm", "feols"], "PVAR estimation method")
    validate_method(transformation, ["fd", "fod"], "PVAR transformation")
    validate_method(steps, ["onestep", "twostep"], "PVAR steps")

    model, panel, varnames = _load_and_estimate_pvar(data, id_col, time_col, lags;
        method=method, transformation=transformation, steps=steps,
        system=system, collapse=collapse,
        dependent=dependent, predet=predet, exog=exog,
        min_lag_endo=min_lag_endo, max_lag_endo=max_lag_endo)

    n = length(varnames)
    p = lags

    _status("Estimating Panel VAR($p) with $n variables: $(join(varnames, ", "))")
    _status("  Method: $method, Transformation: $transformation, Steps: $steps")
    _status("  Groups: $(panel.n_groups), Observations: $(panel.T_obs)")
    if system
        _status("  System GMM (level + difference equations)")
    end
    if collapse
        _status("  Instruments collapsed")
    end
    _status()

    _status_report(() -> report(model))

    coef_df = _build_pvar_coef_table(model, varnames, p)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="Panel VAR($p) Coefficients ($method)", key="panel_var_coefficients")

    _status()
    output_kv(Pair{String,Any}[
        "Groups" => panel.n_groups,
        "Total observations" => panel.T_obs,
        "Instruments" => model.n_instruments,
        "Method" => string(model.method),
        "Transformation" => string(model.transformation),
    ]; format=format, title="Panel Summary")
    return model
end

# ── SMM ───────────────────────────────────────────────────

# Map the CLI/config weighting string to the two symbols real MEMs' `estimate_smm`
# understands. `optimal`/`iterated`/`twostep` are aliases of the two-step optimal
# weighting matrix (Ω⁻¹); anything else is a usage/config error.
function _smm_weighting_symbol(w::AbstractString)
    wl = lowercase(w)
    wl == "identity" && return :identity
    wl in ("two_step", "twostep", "optimal", "iterated") && return :two_step
    throw(CliError("config/enum", "unknown smm weighting `$w` (identity|two_step)"))
end

# Optional bounds → ParameterTransform (both lower and upper, matching θ length).
function _smm_bounds(smm, n_params::Int)
    lower = smm["lower"]; upper = smm["upper"]
    (lower === nothing && upper === nothing) && return nothing
    (lower === nothing || upper === nothing) &&
        throw(CliError("config/shape", "[smm] bounds require BOTH `lower` and `upper`"))
    (length(lower) == n_params && length(upper) == n_params) ||
        throw(CliError("config/shape",
            "[smm] `lower`/`upper` must each have $n_params entries (one per θ), " *
            "got $(length(lower))/$(length(upper))"))
    ParameterTransform(lower, upper)
end

# Build a built-in SMM data-generating simulator. Returns (simulator_fn, param_names);
# simulator_fn(θ, T_periods, burn; rng) → a (T_periods × k) matrix (burn already discarded),
# matching real MEMs' `estimate_smm` simulator contract. θ layout depends on the model.
function _smm_simulator(model::AbstractString, k::Int, theta0, smm)
    m = lowercase(model)
    if m == "ar1"
        k == 1 || throw(CliError("config/shape", "smm model `ar1` needs univariate data (1 column), got $k"))
        length(theta0) == 2 || throw(CliError("config/shape",
            "ar1 `theta0` must be [phi, sigma] (2 params), got $(length(theta0))"))
        sim = (theta, Tp, burn; rng=Random.default_rng()) -> begin
            phi = theta[1]; sigma = abs(theta[2])
            n = Tp + burn
            y = zeros(Float64, n)
            @inbounds for t in 2:n
                y[t] = phi * y[t-1] + sigma * randn(rng)
            end
            reshape(y[(burn+1):end], :, 1)
        end
        return sim, ["phi", "sigma"]
    elseif m == "arp"
        k == 1 || throw(CliError("config/shape", "smm model `arp` needs univariate data (1 column), got $k"))
        p = smm["p"]
        p === nothing && throw(CliError("config/missing-key", "smm model `arp` requires `p` (AR order) in [smm]"))
        p >= 1 || throw(CliError("config/shape", "smm model `arp` requires p >= 1, got $p"))
        length(theta0) == p + 1 || throw(CliError("config/shape",
            "arp `theta0` must be [phi_1..phi_$p, sigma] ($(p+1) params), got $(length(theta0))"))
        sim = (theta, Tp, burn; rng=Random.default_rng()) -> begin
            phis = @view theta[1:p]; sigma = abs(theta[p+1])
            n = Tp + burn
            y = zeros(Float64, n)
            @inbounds for t in (p+1):n
                s = 0.0
                for j in 1:p
                    s += phis[j] * y[t-j]
                end
                y[t] = s + sigma * randn(rng)
            end
            reshape(y[(burn+1):end], :, 1)
        end
        return sim, String[["phi_$j" for j in 1:p]; "sigma"]
    elseif m == "var1"
        length(theta0) == k * k + k || throw(CliError("config/shape",
            "var1 `theta0` must be [vec(A) ($(k*k) entries, column-major); sigma_1..sigma_$k] " *
            "($(k*k + k) params), got $(length(theta0))"))
        sim = (theta, Tp, burn; rng=Random.default_rng()) -> begin
            A = reshape(theta[1:k*k], k, k)
            sigmas = abs.(theta[k*k+1:k*k+k])
            n = Tp + burn
            Ymat = zeros(Float64, n, k)
            @inbounds for t in 2:n
                Ymat[t, :] = A * Ymat[t-1, :] .+ sigmas .* randn(rng, k)
            end
            Ymat[(burn+1):end, :]
        end
        # column-major to match reshape(theta[1:k²], k, k): A[1,1],A[2,1],…,A[k,1],A[1,2],…
        names = String[]
        for j in 1:k, i in 1:k
            push!(names, "A[$i,$j]")
        end
        for i in 1:k
            push!(names, "sigma_$i")
        end
        return sim, names
    elseif m == "iid_normal"
        length(theta0) == k || throw(CliError("config/shape",
            "iid_normal `theta0` must be [sigma_1..sigma_$k] ($k params), got $(length(theta0))"))
        sim = (theta, Tp, burn; rng=Random.default_rng()) -> begin
            sigmas = abs.(theta)
            n = Tp + burn
            Ymat = randn(rng, n, k) .* sigmas'
            Ymat[(burn+1):end, :]
        end
        return sim, ["sigma_$i" for i in 1:k]
    else
        throw(CliError("config/enum", "unknown smm model `$model` (ar1|arp|var1|iid_normal)"))
    end
end

function _estimate_smm(; data::String, config::String="",
                        weighting::String="two_step", sim_ratio::Int=5,
                        burn::Int=100,
                        output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    k = size(Y, 2)

    # SMM needs a data-generating model to simulate from — that lives in the [smm] config,
    # so --config is required (unlike the other estimate leaves).
    isempty(config) && throw(CliError("config/missing",
        "estimate smm requires --config <toml> with an [smm] section specifying a " *
        "data-generating `model` (ar1|arp|var1|iid_normal) and `theta0`. SMM matches " *
        "simulated moments to sample moments, so it needs a model to simulate."))

    cfg = load_config(config)
    smm = get_smm(cfg)
    weighting = smm["weighting"]
    sim_ratio = smm["sim_ratio"]
    burn      = smm["burn"]
    lags      = smm["lags"]

    modelname = smm["model"]
    modelname === nothing && throw(CliError("config/missing-key",
        "[smm] must set `model` = one of ar1|arp|var1|iid_normal"))
    theta0 = smm["theta0"]
    theta0 === nothing && throw(CliError("config/missing-key",
        "[smm] must set `theta0` (initial parameter vector; layout depends on `model`)"))

    simulator_fn, param_names = _smm_simulator(modelname, k, theta0, smm)
    moments_fn       = d -> autocovariance_moments(d; lags=lags)
    contributions_fn = d -> autocovariance_moment_contributions(d; lags=lags)

    n_params  = length(theta0)
    n_moments = k * (k + 1) ÷ 2 + k * lags
    n_moments >= n_params || throw(CliError("config/underidentified",
        "SMM needs at least as many moments as parameters: $n_moments moments " *
        "(k(k+1)/2 + k·lags, k=$k, lags=$lags) < $n_params params. Increase `lags`."))

    bounds = _smm_bounds(smm, n_params)
    wsym   = _smm_weighting_symbol(weighting)
    # Determinism (C052/#243): the global Random.seed! already makes runs reproducible;
    # forward --seed as the estimator's own rng so simulation draws are pinned too.
    rng = _SEED[] === nothing ? Random.default_rng() : Random.MersenneTwister(_SEED[])

    _status("Estimating SMM: model=$modelname, $k variable(s), $n_params params, " *
            "$n_moments moments; weighting=$weighting, sim_ratio=$sim_ratio, burn=$burn")
    _status()

    model = try
        estimate_smm(simulator_fn, moments_fn, theta0, Y;
                     weighting=wsym, sim_ratio=sim_ratio, burn=burn,
                     contributions_fn=contributions_fn, bounds=bounds, rng=rng,
                     _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        (e isa ArgumentError || e isa AssertionError || e isa BoundsError ||
         e isa DimensionMismatch) ||
            rethrow()
        throw(CliError("model/error", "SMM estimation failed: $(sprint(showerror, e))"))
    end

    se = sqrt.(abs.(diag(model.vcov)))
    t_stats = model.theta ./ se
    p_vals = [2.0 * (1.0 - _normal_cdf(abs(t))) for t in t_stats]

    est_df = DataFrame(
        parameter = param_names,
        estimate = round.(model.theta; digits=6),
        std_error = round.(se; digits=6),
        t_stat = round.(t_stats; digits=4),
        p_value = round.(p_vals; digits=4),
    )
    output_result(est_df; format=Symbol(format), output=output,
                  title="SMM Estimation (model=$modelname, weighting=$weighting)",
                  key="smm_estimates")

    _status()
    _status_styled("  J-statistic: $(round(model.J_stat; digits=4))\n"; color=:cyan)
    # M-29 (MEMs#797): the χ² limit needs efficient weighting — under identity
    # model.J_pvalue is NaN, so render n/a with the reason instead of bare NaN.
    if isnan(model.J_pvalue)
        _status_styled("  J p-value:   n/a (identity weighting — χ² limit needs efficient weighting)\n"; color=:cyan)
    else
        _status_styled("  J p-value:   $(round(model.J_pvalue; digits=4))\n"; color=:cyan)
    end
    _status_styled("  Converged:   $(model.converged)\n";
                color = model.converged ? :green : :red)
    return model
end

# ── FAVAR ──────────────────────────────────────────────

function _estimate_favar(; data::String, factors=nothing, lags::Int=2,
                          key_vars::String="", method::String="two_step",
                          draws::Int=5000, output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, method, draws)

    if favar isa MacroEconometricModels.BayesianFAVAR
        _status("Bayesian FAVAR: $(favar.n_factors) factors, $(favar.n_key) key vars, $(size(favar.B_draws, 3)) draws")
        pairs = Pair{String,Any}[
            "Factors" => favar.n_factors,
            "Key variables" => favar.n_key,
            "Lags" => favar.p,
            "MCMC draws" => size(favar.B_draws, 3),
        ]
        output_kv(pairs; format=format, output=output, title="Bayesian FAVAR")
        return favar
    end

    var_model = to_var(favar)
    coef_df = _build_var_coef_table(coef(var_model), favar.varnames, favar.p)
    output_result(coef_df; format=Symbol(format), output=output, title="FAVAR($lags) Coefficients",
                  key="favar_coefficients")

    _status()
    _status_styled("  Factors: $(favar.n_factors), Key variables: $(favar.n_key)\n"; color=:cyan)
    _status_styled("  AIC: $(round(favar.aic; digits=2)), BIC: $(round(favar.bic; digits=2))\n"; color=:cyan)

    _maybe_plot(favar; plot=plot, plot_save=plot_save)
    return favar
end

# ── Structural DFM ────────────────────────────────────

function _estimate_sdfm(; data::String, factors=nothing, id::String="cholesky",
                         var_lags::Int=1, horizon::Int=40,
                         config::String="", bandwidth::Int=0,
                         kernel::String="bartlett", method::String="fglr",
                         spectral::String="lag-window", instrument::String="",
                         q_method::String="hallin-liska",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    # W1/#165: one shared data path (see `_load_and_estimate_sdfm`); --factors
    # omitted selects q via upstream `:auto` + `--q-method` (deterministic).
    sdfm, Y, varnames, q = _load_and_estimate_sdfm(data, factors, id, var_lags, horizon,
        config, method, spectral, instrument, q_method;
        bandwidth=bandwidth, kernel=kernel)
    n = length(varnames)

    _status("Estimating Structural DFM: $q factors, id=$id, method=$method, VAR lags=$var_lags, horizon=$horizon")

    _status("  Identification: $(sdfm.identification)")
    _status("  Factor VAR lags: $(sdfm.p_var)")
    _status("  Shocks: $(join(sdfm.shock_names, ", "))")

    # #147: the leaf was status-only — a success exit with nothing addressable.
    # The structural results live on `irf sdfm`/`fevd sdfm`; this table is the
    # estimation record (dimensions, identification, variance shares).
    summary_df = DataFrame(
        metric = ["n_vars", "T", "dynamic_factors", "static_rank", "identification",
                  "factor_var_lags", "horizon", "shocks", "avg_common_variance_share"],
        value = [string(n), string(size(Y, 1)), string(sdfm.gdfm.q), string(sdfm.gdfm.r),
                 String(sdfm.identification), string(sdfm.p_var), string(horizon),
                 join(sdfm.shock_names, ", "),
                 string(round(mean(sdfm.gdfm.variance_explained); digits=4))],
    )
    output_result(summary_df; format=Symbol(format), output=output,
                  title="Structural DFM Estimation Summary",
                  key="sdfm_estimation_summary")

    _maybe_plot(sdfm; plot=plot, plot_save=plot_save)
    return sdfm
end

# ── OLS/WLS Regression ────────────────────────────────

function _estimate_reg(; data::String, dep::String="", cov_type::String="hc1",
                        weights::String="", clusters::String="",
                        lat::String="", lon::String="", dist_cutoff::Float64=0.0,
                        conley_kernel::String="bartlett", conley_metric::String="euclidean",
                        time_col::String="", time_cutoff::Int=0,
                        output::String="", format::String="table")
    # ── W10/#112: Conley (1999) spatial HAC ─────────────────────────────────
    # Guard the whole option group up front and in BOTH directions: choosing `conley`
    # without coordinates, and supplying coordinates under another cov-type (which would
    # otherwise be a silent no-op that reads as spatially-corrected inference).
    is_conley = cov_type == "conley"
    conley_opts = [("--lat", !isempty(lat)), ("--lon", !isempty(lon)),
                   ("--dist-cutoff", dist_cutoff != 0.0),
                   ("--time-col", !isempty(time_col)), ("--time-cutoff", time_cutoff != 0)]
    if is_conley
        (!isempty(lat) && !isempty(lon)) || throw(CliError("usage/missing",
            "estimate reg --cov-type conley requires both --lat and --lon coordinate columns";
            hint="e.g. --cov-type conley --lat latitude --lon longitude --dist-cutoff 100"))
        dist_cutoff > 0.0 || throw(CliError("usage/invalid",
            "estimate reg: --dist-cutoff must be > 0 for --cov-type conley (got $dist_cutoff)";
            hint="km for --conley-metric haversine, coordinate units for euclidean"))
        time_cutoff >= 0 || throw(CliError("usage/invalid",
            "estimate reg: --time-cutoff must be ≥ 0 (got $time_cutoff)"))
        # A time cutoff without a time column silently degrades to spatial-only, and a time
        # column with cutoff 0 does nothing — both mean the user did not get what they asked.
        (isempty(time_col) == (time_cutoff == 0)) || throw(CliError("usage/invalid",
            "estimate reg: --time-col and --time-cutoff must be given together for " *
            "spatial+serial Conley (got --time-col '$time_col', --time-cutoff $time_cutoff)"))
    else
        set = [n for (n, given) in conley_opts if given]
        isempty(set) || throw(CliError("usage/invalid",
            "estimate reg: $(join(set, ", ")) applies only to --cov-type conley (got --cov-type $cov_type)"))
    end

    coords = is_conley ? _load_coords(data, lat, lon, conley_metric) : nothing
    tvec = nothing
    if is_conley && !isempty(time_col)
        df_t = load_data(data)
        time_col in names(df_t) || throw(CliError("data/column-range",
            "--time-col '$time_col' not found; available: $(join(names(df_t), ", "))"))
        any(ismissing, df_t[!, time_col]) && throw(CliError("data/missing-values",
            "--time-col '$time_col' contains missing values"))
        # Upstream forms the serial lag as `abs(t_i - t_j)` and gives any NON-INTEGER gap
        # zero weight. So a String time column is an untyped MethodError deep inside the
        # loop, and — worse — a Float column with fractional values is a SILENT no-op: the
        # serial correction runs and contributes nothing. Require an integral period index.
        traw = collect(df_t[!, time_col])
        tnum = try
            Vector{Float64}(traw)
        catch
            throw(CliError("data/invalid",
                "--time-col '$time_col' must be a numeric period index (e.g. a year or a " *
                "1..T counter); dates and labels are not supported"))
        end
        all(v -> isfinite(v) && v == round(v), tnum) || throw(CliError("data/invalid",
            "--time-col '$time_col' must hold whole-number periods — upstream gives any " *
            "non-integer time gap zero weight, so a fractional column silently disables " *
            "the serial correction instead of failing"))
        tvec = Int.(round.(tnum))
    end
    # The coordinate/time columns are numeric, so `_load_reg_data` would otherwise pull them
    # into X as regressors — a wrong point estimate that no error would reveal.
    drop_cols = is_conley ? filter(!isempty, String[lat, lon, time_col]) : String[]

    y, X, xcols = _load_reg_data(data, dep; weights_col=weights, clusters_col=clusters,
                                 exclude_cols=drop_cols)
    w = _load_weights(data, weights)
    cl = _load_clusters(data, clusters)
    if cov_type == "cluster" && cl === nothing
        throw(CliError("usage/missing",
            "estimate reg --cov-type cluster requires --clusters <column>"))
    end

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    wls_tag = isnothing(w) ? "OLS" : "WLS"
    _status("$wls_tag Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(y)), Regressors: $(length(xcols)), Cov type: $cov_type")
    if is_conley
        _status("  Conley HAC: coords=($lat, $lon), metric=$conley_metric, kernel=$conley_kernel, cutoff=$dist_cutoff")
        isempty(time_col) || _status("  Conley serial: time=$time_col, lag cutoff=$time_cutoff")
    end
    _status()

    model = try
        estimate_reg(y, X; cov_type=Symbol(cov_type), weights=w,
                     varnames=xcols, clusters=cl,
                     coords=coords, cutoff=dist_cutoff,
                     conley_kernel=Symbol(conley_kernel),
                     conley_metric=Symbol(conley_metric),
                     time=tvec, time_cutoff=time_cutoff)
    catch e
        # NOT `_garch_variant_error`: its hint talks about series length and p/q orders,
        # which is nonsense for a cross-section regression.
        throw(_domain_or_data_error(e, "OLS/WLS regression"))
    end

    coef_df = _reg_coef_table(model, xcols)
    # The OLS/WLS distinction is an option-driven label, so it stays in the human-facing
    # title only — the envelope address is fixed for both paths (leaf-stem key,
    # matching predict/residuals reg → reg_fitted_values/reg_residuals).
    output_result(coef_df; format=Symbol(format), output=output, title="$wls_tag Regression Coefficients",
                  key="reg_coefficients")

    _status()
    f_pv = hasproperty(model, :f_pval) ? model.f_pval :
           (hasproperty(model, :f_pvalue) ? model.f_pvalue : NaN)
    pairs = Pair{String,Any}[
        "R²"              => round(r2(model); digits=6),
        "Adj. R²"         => round(model.adj_r2; digits=6),
        "F-statistic"     => round(model.f_stat; digits=4),
        "F p-value"       => round(f_pv; digits=4),
        "Log-likelihood"  => round(loglikelihood(model); digits=4),
        "AIC"             => round(aic(model); digits=4),
        "BIC"             => round(bic(model); digits=4),
    ]
    output_kv(pairs; format=format, title="Fit Statistics")

    # Emitted ONLY under --cov-type conley, so the existing `estimate reg` envelope stays
    # byte-identical when the new options are not used. The settings belong in the data
    # because a Conley SE is meaningless without the cutoff/metric that produced it.
    if is_conley
        cpairs = Pair{String,Any}[
            "cov_type"      => "conley",
            "lat_column"    => lat,
            "lon_column"    => lon,
            "metric"        => conley_metric,
            "kernel"        => conley_kernel,
            "dist_cutoff"   => dist_cutoff,
            "cutoff_units"  => conley_metric == "haversine" ? "km" : "coordinate units",
        ]
        if !isempty(time_col)
            push!(cpairs, "time_column" => time_col)
            push!(cpairs, "time_cutoff" => time_cutoff)
        end
        output_kv(cpairs; format=format, title="Conley Spatial HAC Settings")
    end
    return model
end

# ── IV (2SLS) Regression ──────────────────────────────

# C067 (#72): general-to-specific / stepwise variable selection. A DEDICATED LEAF
# rather than an `estimate reg --select` flag, so `estimate reg`'s envelope tables stay
# fixed — a leaf whose table set changes with a flag forces every agent consuming it to
# branch. `select_variables` returns a SelectionResult that CARRIES the refitted
# `final::RegModel`, so this renders the same coefficient table as `estimate reg` plus
# the selection path.
function _estimate_select(; data::String, dep::String="", method::String="bidirectional",
                           criterion::String="pvalue", p_enter::Float64=0.05,
                           p_remove::Float64=0.10, keep::String="",
                           output::String="", format::String="table")
    (0.0 < p_enter < 1.0) || throw(CliError("usage/invalid",
        "estimate select: --p-enter must be in (0, 1) (got $p_enter)"))
    (0.0 < p_remove < 1.0) || throw(CliError("usage/invalid",
        "estimate select: --p-remove must be in (0, 1) (got $p_remove)"))
    # Upstream requires p_remove >= p_enter for the bidirectional p-value search; guard
    # it here so the common mistake is a usage error, not a bare ArgumentError.
    (method != "bidirectional" || criterion != "pvalue" || p_remove >= p_enter) ||
        throw(CliError("usage/invalid",
            "estimate select: bidirectional pvalue search needs --p-remove ≥ --p-enter (got $p_remove < $p_enter)"))
    y, X, xcols = _load_reg_data(data, dep)
    keep_idx = nothing
    if !isempty(keep)
        names = [strip(t) for t in split(keep, ",") if !isempty(strip(t))]
        isempty(names) && throw(CliError("usage/invalid", "estimate select: --keep is empty"))
        keep_idx = Int[]
        for nm in names
            i = findfirst(==(String(nm)), xcols)
            i === nothing && throw(CliError("data/column-range",
                "estimate select: --keep column '$nm' is not a regressor";
                hint="available: $(join(xcols, ", "))"))
            push!(keep_idx, i)
        end
    end
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Variable Selection ($method/$criterion): $dep_name ~ $(join(xcols, " + "))"); _status()
    res = try
        select_variables(y, X; method=Symbol(replace(method, '-' => '_')),
                         criterion=Symbol(criterion), p_enter=p_enter,
                         p_remove=p_remove, keep=keep_idx, varnames=xcols)
    catch e
        throw(_garch_variant_error(e, "variable selection"))
    end
    # The refitted final model renders exactly like `estimate reg`.
    output_result(_reg_coef_table(res.final, res.varnames[res.selected]);
        format=Symbol(format), output=output,
        title="Selected Model Coefficients ($dep_name)", key="selected_model_coefficients")
    # The search path is the audit trail: each step is (action, column, statistic).
    if !isempty(res.path)
        output_result(DataFrame(
                step = collect(1:length(res.path)),
                action = [String(p[1]) for p in res.path],
                variable = [res.varnames[p[2]] for p in res.path],
                statistic = [round(Float64(p[3]); digits=6) for p in res.path]);
            format=Symbol(format), output=_per_var_output_path(output, "path"),
            title="Selection Path ($dep_name)", key="selection_path")
    end
    pairs = Pair{String,Any}[
        "method" => String(res.method),
        "criterion" => String(res.criterion),
        "selected" => isempty(res.selected) ? "none" : join(res.varnames[res.selected], ", "),
        "n selected" => length(res.selected),
        "kept (forced)" => isempty(res.keep) ? "none" : join(res.varnames[res.keep], ", "),
        "candidates (GUM)" => res.n_gum,
        "terminal models" => length(res.terminal_models),
    ]
    res.encompassing_f === nothing || push!(pairs, "encompassing F" => _finite_or_str_reg(Float64(res.encompassing_f)))
    res.encompassing_pval === nothing || push!(pairs, "encompassing p-value" => _finite_or_str_reg(Float64(res.encompassing_pval)))
    output_kv(pairs; format=format, title="Selection Summary")
    return res
end

"""Round for display but render a non-finite value as a string (legacy JSON writer)."""
_finite_or_str_reg(x) = isfinite(x) ? round(Float64(x); digits=6) : string(Float64(x))

function _estimate_iv(; data::String, dep::String="", endogenous::String="",
                       instruments::String="", cov_type::String="hc1",
                       method::String="tsls", k::String="", fuller_a::Float64=1.0,
                       output::String="", format::String="table")
    # C067b: typed shared loader (`_load_iv_data`) replaces the old bare `error()` sites
    # (untyped exit-1) — a missing/unknown column is user input, not a CLI bug. Shared with
    # `test weak-instrument` so the two never drift.
    # k-class family (#72): :tsls | :liml | :fuller | :kclass. `k` is REQUIRED for
    # kclass and meaningless otherwise; guard both up front so a bad combination is a
    # usage error rather than a bare upstream ArgumentError.
    kval = nothing
    if method == "kclass"
        isempty(k) && throw(CliError("usage/missing",
            "estimate iv: --method kclass requires --k";
            hint="give the k-class scalar, e.g. --k 1 (k=0 is OLS, k=1 is 2SLS)"))
        kv = tryparse(Float64, k)
        (kv === nothing || !isfinite(kv)) && throw(CliError("usage/invalid",
            "estimate iv: --k must be a finite number, got '$k'"))
        kval = kv
    elseif !isempty(k)
        throw(CliError("usage/invalid",
            "estimate iv: --k applies only to --method kclass (got --method $method)"))
    end
    method == "fuller" || fuller_a == 1.0 || throw(CliError("usage/invalid",
        "estimate iv: --fuller-a applies only to --method fuller (got --method $method)"))
    method != "fuller" || fuller_a > 0 || throw(CliError("usage/invalid",
        "estimate iv: --fuller-a must be > 0 (got $fuller_a)"))

    d = _load_iv_data(data, dep, endogenous, instruments)
    y, X, Z, xcols, endog_idx = d.y, d.X, d.Z, d.xcols, d.endog_idx

    label = uppercase(method == "tsls" ? "2SLS" : method)
    _status("IV ($label) Regression: $(d.dep_col) ~ $(join(xcols, " + "))")
    _status("  Endogenous: $(join(d.endog_names, ", "))")
    _status("  Excluded instruments: $(join(d.inst_names, ", "))  (instrument set Z: $(join(d.zcols, ", ")))")
    _status("  Observations: $(length(y)), Cov type: $cov_type")
    _status()

    model = try
        estimate_iv(y, X, Z; endogenous=endog_idx, cov_type=Symbol(cov_type),
                    method=Symbol(method), k=kval, fuller_a=fuller_a, varnames=xcols)
    catch e
        throw(_garch_variant_error(e, "IV ($label) estimation"))
    end

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output, title="IV ($label) Regression Coefficients",
                  key="iv_coefficients")

    _status()
    pairs = Pair{String,Any}[
        "R²"              => round(r2(model); digits=6),
        "Adj. R²"         => round(model.adj_r2; digits=6),
    ]
    push!(pairs, "method" => method)
    isnothing(model.kclass_k)   || push!(pairs, "k-class k" => round(Float64(model.kclass_k); digits=6))
    isnothing(model.kappa_hat)  || push!(pairs, "kappa_hat" => round(Float64(model.kappa_hat); digits=6))
    if !isnothing(model.first_stage_f)
        push!(pairs, "First-stage F" => round(model.first_stage_f; digits=4))
    end
    if !isnothing(model.sargan_stat)
        push!(pairs, "Sargan statistic" => round(model.sargan_stat; digits=4))
        push!(pairs, "Sargan p-value"   => round(model.sargan_pval; digits=4))
    end
    output_kv(pairs; format=format, title="IV Diagnostics")
    return model
end

# ── Systems: SUR & 3SLS (C063, M5c) ─────────────────────────
# SURModel/ThreeSLSModel are not Tables.jl-registered upstream, so the coefficient
# table is hand-built (documented C051 exception, like the io family): one tidy row
# per (equation, term). Both are asymptotic (FGLS / 3SLS) → normal-approx p-values/CIs.

# Build a (y, X, names) equation tuple from column names; optionally prepend a constant.
function _system_eq_matrix(df, numcols::Vector{String}, eq::AbstractDict, intercept::Bool)
    eq["dep"] in numcols || throw(CliError("config/bad-column",
        "equation '$(eq["name"])': dependent column '$(eq["dep"])' not found in numeric columns: $(join(numcols, ", "))"))
    for c in eq["indep"]
        c in numcols || throw(CliError("config/bad-column",
            "equation '$(eq["name"])': regressor column '$c' not found in numeric columns: $(join(numcols, ", "))"))
    end
    y = Vector{Float64}(df[!, eq["dep"]])
    Xcols = [Vector{Float64}(df[!, c]) for c in eq["indep"]]
    X = intercept ? hcat(ones(Float64, length(y)), Xcols...) : reduce(hcat, Xcols)
    names = intercept ? String["const"; eq["indep"]...] : String[eq["indep"]...]
    return (y, X, names)   # (y, X, names) tuple for estimate_sur/estimate_3sls
end

# Per-equation instrument matrix (3SLS instruments=perequation); optional constant.
function _system_instr_matrix(df, numcols::Vector{String}, eq::AbstractDict, intercept::Bool)
    eq["instr"] === nothing && throw(CliError("config/missing-key",
        "instruments=perequation requires each [[equations]] block to set `instr`; equation '$(eq["name"])' has none"))
    for c in eq["instr"]
        c in numcols || throw(CliError("config/bad-column",
            "equation '$(eq["name"])': instrument '$c' not found in numeric columns"))
    end
    Zcols = [Vector{Float64}(df[!, c]) for c in eq["instr"]]
    intercept ? hcat(ones(Float64, size(df, 1)), Zcols...) : hcat(Zcols...)
end

# Tidy coefficient table for a SUR/ThreeSLS model: equation|term|estimate|std_error|stat|p_value|ci_lower|ci_upper.
function _system_coef_table(model)
    eqc = String[]; term = String[]; est = Float64[]; sec = Float64[]
    stat = Float64[]; pval = Float64[]; lo = Float64[]; hi = Float64[]
    z95 = 1.959964
    for j in eachindex(model.eqnames)
        b = model.betas[j]; s = model.ses[j]; vn = model.varnames[j]
        for i in eachindex(b)
            zi = s[i] == 0 ? 0.0 : b[i] / s[i]
            push!(eqc, model.eqnames[j]); push!(term, vn[i])
            push!(est, round(b[i]; digits=6)); push!(sec, round(s[i]; digits=6))
            push!(stat, round(zi; digits=4)); push!(pval, round(2.0 * (1.0 - _normal_cdf(abs(zi))); digits=4))
            push!(lo, round(b[i] - z95 * s[i]; digits=6)); push!(hi, round(b[i] + z95 * s[i]; digits=6))
        end
    end
    DataFrame(equation=eqc, term=term, estimate=est, std_error=sec,
              stat=stat, p_value=pval, ci_lower=lo, ci_upper=hi)
end

# Map an estimation failure to a typed CliError (never an uncaught exit-1 — io-family lesson).
function _system_estimation_error(e, what::String)
    e isa CliError && return e
    e isa ArgumentError && return CliError("config/system", sprint(showerror, e);
        hint="check equation specs: all equations need equal T, valid columns, and enough observations")
    return CliError("model/error", "$what estimation failed: $(sprint(showerror, e))";
        hint="near-singular system — drop collinear regressors or add observations")
end

# ── #68: sur / 3sls downstream verbs ─────────────────────────────────────────
# `SURModel` and `ThreeSLSModel` carry PER-EQUATION `fitted::Vector{Vector{T}}` and
# `residuals::Vector{Vector{T}}` as fields (system/types.jl) alongside `eqnames`. There
# are no StatsAPI predict/residuals methods for them, so these read the fields.
#
# OUTPUT SHAPE: one tidy LONG table `equation | t | fitted` (resp. `residual`) rather
# than N separate tables — it matches the C051 convention and `_heckman_coef_table`'s
# handling of a multi-equation model, and keeps one table per leaf so an agent does not
# have to discover a variable number of envelope keys.
#
# Both verbs mirror `estimate sur`/`estimate 3sls`'s options because the equation system
# lives in the --config TOML: without it the model cannot be refit at all.

"""Long per-equation table from a system model's `fitted`/`residuals` field."""
function _system_long_table(model, field::Symbol, valuecol::String)
    blocks = getfield(model, field)
    eqs = String[]; ts = Int[]; vals = Float64[]
    for (i, b) in enumerate(blocks)
        v = Float64.(collect(b))
        append!(eqs, fill(model.eqnames[i], length(v)))
        append!(ts, 1:length(v))
        append!(vals, v)
    end
    return DataFrame("equation" => eqs, "t" => ts, valuecol => round.(vals; digits=6))
end

"""Refit the SUR system for the downstream verbs, mirroring `_estimate_sur` exactly."""
function _sur_refit(data, config, iterate, no_intercept)
    isempty(config) && throw(CliError("config/missing",
        "requires --config <toml> with [[equations]] blocks (each `dep` + `indep`)"))
    df = load_data(data)
    numcols = variable_names(df)
    spec = get_system(load_config(config))
    intercept = !no_intercept
    eqs = [_system_eq_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    eqnames = String[eq["name"] for eq in spec["equations"]]
    return try
        estimate_sur(eqs; iterate=iterate, eqnames=eqnames)
    catch e
        throw(_system_estimation_error(e, "SUR"))
    end
end

"""Refit the 3SLS system, mirroring `_estimate_3sls` exactly (including the
common-vs-per-equation instrument construction)."""
function _3sls_refit(data, config, instruments, no_intercept)
    isempty(config) && throw(CliError("config/missing",
        "requires --config <toml> with [[equations]] and instruments"))
    df = load_data(data)
    numcols = variable_names(df)
    spec = get_system(load_config(config))
    intercept = !no_intercept
    imode = Symbol(instruments)
    eqs = [_system_eq_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    eqnames = String[eq["name"] for eq in spec["equations"]]
    Z = if imode == :common
        spec["common_instruments"] === nothing && throw(CliError("config/missing",
            "instruments=common requires an [instruments] section with a `common` list"))
        for c in spec["common_instruments"]
            c in numcols || throw(CliError("config/bad-column",
                "instrument '$c' not found in numeric columns: $(join(numcols, ", "))"))
        end
        Zcols = [Vector{Float64}(df[!, c]) for c in spec["common_instruments"]]
        intercept ? hcat(ones(Float64, size(df, 1)), Zcols...) : hcat(Zcols...)
    else
        [_system_instr_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    end
    return try
        estimate_3sls(eqs, Z; instruments=imode, eqnames=eqnames)
    catch e
        throw(_system_estimation_error(e, "3SLS"))
    end
end

function _predict_sur(; data::String, config::String="", iterate::Bool=false,
        no_intercept::Bool=false, output::String="", format::String="table", model=nothing)
    m = model === nothing ? _sur_refit(data, config, iterate, no_intercept) : model
    _status("SUR fitted values: $(length(m.eqnames)) equations"); _status()
    output_result(_system_long_table(m, :fitted, "fitted"); format=Symbol(format),
                  output=output, title="SUR Fitted Values (per equation)")
    return m.fitted
end

function _residuals_sur(; data::String, config::String="", iterate::Bool=false,
        no_intercept::Bool=false, output::String="", format::String="table", model=nothing)
    m = model === nothing ? _sur_refit(data, config, iterate, no_intercept) : model
    _status("SUR residuals: $(length(m.eqnames)) equations"); _status()
    output_result(_system_long_table(m, :residuals, "residual"); format=Symbol(format),
                  output=output, title="SUR Residuals (per equation)")
    return m.residuals
end

function _predict_3sls(; data::String, config::String="", instruments::String="common",
        no_intercept::Bool=false, output::String="", format::String="table", model=nothing)
    m = model === nothing ? _3sls_refit(data, config, instruments, no_intercept) : model
    _status("3SLS fitted values: $(length(m.eqnames)) equations"); _status()
    output_result(_system_long_table(m, :fitted, "fitted"); format=Symbol(format),
                  output=output, title="3SLS Fitted Values (per equation)")
    return m.fitted
end

function _residuals_3sls(; data::String, config::String="", instruments::String="common",
        no_intercept::Bool=false, output::String="", format::String="table", model=nothing)
    m = model === nothing ? _3sls_refit(data, config, instruments, no_intercept) : model
    _status("3SLS residuals: $(length(m.eqnames)) equations"); _status()
    output_result(_system_long_table(m, :residuals, "residual"); format=Symbol(format),
                  output=output, title="3SLS Residuals (per equation)")
    return m.residuals
end

function _estimate_sur(; data::String, config::String="", iterate::Bool=false,
                        no_intercept::Bool=false, output::String="", format::String="table")
    isempty(config) && throw(CliError("config/missing",
        "estimate sur requires --config <toml> with [[equations]] blocks (each `dep` + `indep`)"))
    df = load_data(data)
    numcols = variable_names(df)
    spec = get_system(load_config(config))
    intercept = !no_intercept
    eqs = [_system_eq_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    eqnames = String[eq["name"] for eq in spec["equations"]]

    _status("SUR: $(length(eqs)) equations, $(intercept ? "with" : "no") intercept, iterate=$iterate")
    _status()
    model = try
        estimate_sur(eqs; iterate=iterate, eqnames=eqnames)
    catch e
        throw(_system_estimation_error(e, "SUR"))
    end

    output_result(_system_coef_table(model); format=Symbol(format), output=output,
                  title="SUR Coefficients")
    kind = model.iterated ? "Iterated SUR" : "SUR"
    output_kv(Pair{String,Any}[
        "estimator"  => model.restricted ? "$kind (restricted)" : kind,
        "equations"  => length(model.eqnames),
        "obs_per_eq" => model.nobs,
        "det_sigma"  => round(model.det_sigma; digits=6),
        "mcelroy_r2" => round(model.mcelroy_r2; digits=6),
        "loglik"     => round(model.loglik; digits=4),
        "iterations" => model.iterations,
    ]; format=format, title="SUR System Statistics")
    return model
end

function _estimate_3sls(; data::String, config::String="", instruments::String="common",
                         no_intercept::Bool=false, output::String="", format::String="table")
    isempty(config) && throw(CliError("config/missing",
        "estimate 3sls requires --config <toml> with [[equations]] and instruments"))
    df = load_data(data)
    numcols = variable_names(df)
    spec = get_system(load_config(config))
    intercept = !no_intercept
    eqs = [_system_eq_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    eqnames = String[eq["name"] for eq in spec["equations"]]
    imode = Symbol(instruments)

    Z = if imode == :common
        spec["common_instruments"] === nothing && throw(CliError("config/missing",
            "instruments=common requires an [instruments] section with a `common` list"))
        for c in spec["common_instruments"]
            c in numcols || throw(CliError("config/bad-column",
                "instrument '$c' not found in numeric columns: $(join(numcols, ", "))"))
        end
        Zcols = [Vector{Float64}(df[!, c]) for c in spec["common_instruments"]]
        intercept ? hcat(ones(Float64, size(df, 1)), Zcols...) : hcat(Zcols...)
    else
        [_system_instr_matrix(df, numcols, eq, intercept) for eq in spec["equations"]]
    end

    _status("3SLS: $(length(eqs)) equations, instruments=$instruments, $(intercept ? "with" : "no") intercept")
    _status()
    model = try
        estimate_3sls(eqs, Z; instruments=imode, eqnames=eqnames)
    catch e
        throw(_system_estimation_error(e, "3SLS"))
    end

    output_result(_system_coef_table(model); format=Symbol(format), output=output,
                  title="3SLS Coefficients")
    output_kv(Pair{String,Any}[
        "estimator"          => "3SLS",
        "equations"          => length(model.eqnames),
        "obs_per_eq"         => model.nobs,
        "det_sigma"          => round(model.det_sigma; digits=6),
        "mcelroy_r2"         => round(model.mcelroy_r2; digits=6),
        "instruments_per_eq" => join(model.n_instruments, ", "),
    ]; format=format, title="3SLS System Statistics")
    return model
end

# ── Logit Regression ──────────────────────────────────

function _estimate_logit(; data::String, dep::String="", cov_type::String="hc1",
                          clusters::String="", maxiter::Int=100, tol::Float64=1e-8,
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
    cl = _load_clusters(data, clusters)

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Logit Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(y)), Regressors: $(length(xcols)), Cov type: $cov_type")
    _status()

    model = estimate_logit(y, X; cov_type=Symbol(cov_type), varnames=xcols,
                           clusters=cl, maxiter=maxiter, tol=tol)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output, title="Logit Regression Coefficients")

    _status()
    pr2 = hasproperty(model, :pseudo_r2) ? model.pseudo_r2 :
          (try r2(model) catch; NaN end)
    pairs = Pair{String,Any}[
        "Pseudo R²"       => round(pr2; digits=6),
        "Log-likelihood"  => round(loglikelihood(model); digits=4),
        "Log-lik (null)"  => round(model.loglik_null; digits=4),
        "AIC"             => round(aic(model); digits=4),
        "BIC"             => round(bic(model); digits=4),
        "Converged"       => model.converged,
        "Iterations"      => model.iterations,
    ]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

# ── Probit Regression ─────────────────────────────────

function _estimate_probit(; data::String, dep::String="", cov_type::String="hc1",
                           clusters::String="", maxiter::Int=100, tol::Float64=1e-8,
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
    cl = _load_clusters(data, clusters)

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Probit Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(y)), Regressors: $(length(xcols)), Cov type: $cov_type")
    _status()

    model = estimate_probit(y, X; cov_type=Symbol(cov_type), varnames=xcols,
                            clusters=cl, maxiter=maxiter, tol=tol)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output, title="Probit Regression Coefficients")

    _status()
    pr2 = hasproperty(model, :pseudo_r2) ? model.pseudo_r2 :
          (try r2(model) catch; NaN end)
    pairs = Pair{String,Any}[
        "Pseudo R²"       => round(pr2; digits=6),
        "Log-likelihood"  => round(loglikelihood(model); digits=4),
        "Log-lik (null)"  => round(model.loglik_null; digits=4),
        "AIC"             => round(aic(model); digits=4),
        "BIC"             => round(bic(model); digits=4),
        "Converged"       => model.converged,
        "Iterations"      => model.iterations,
    ]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

# ── Panel Regression ──────────────────────────────────

function _estimate_preg(; data::String, dep::String="", indep::String="",
                         method::String="fe", twoway::Bool=false,
                         cov_type::String="cluster", ar1::String="none",
                         pcse_unbalanced::String="casewise",
                         absorb::String="", hdfe_tol::Float64=1e-8, hdfe_maxiter::Int=1000,
                         collapse::Bool=false, min_lag_endo::Int=2, max_lag_endo::Int=99,
                         id_col::String="", time_col::String="",
                         output::String="", format::String="table")
    # was a bare error() -> internal/error exit 1 for an ordinary usage mistake
    isempty(dep) && throw(CliError("usage/missing", "--dep is required";
        hint="name the dependent variable column, e.g. --dep y"))
    # #75: Beck-Katz PCSE + Prais-Winsten AR(1). --pcse-unbalanced only affects the PCSE
    # covariance, so a mismatch is a usage error rather than a silent no-op.
    (cov_type == "pcse" || pcse_unbalanced == "casewise") || throw(CliError("usage/invalid",
        "estimate preg: --pcse-unbalanced applies only to --cov-type pcse (got --cov-type $cov_type)"))

    # ── W10/#112: high-dimensional fixed-effect absorption ──────────────────
    absorb_dims = String[strip(s) for s in split(absorb, ",") if !isempty(strip(s))]
    if !isempty(absorb_dims)
        method == "fe" || throw(CliError("usage/invalid",
            "estimate preg: --absorb is supported only for --method fe (the within estimator), got --method $method"))
        # NOT merely "mutually exclusive with --twoway": upstream computes `twoway=true` by
        # the SAME alternating projections, so pointing the user at `--absorb entity,time`
        # is the correct route on balanced AND unbalanced panels. The additive identity
        # y - ȳᵢ - ȳₜ + ȳ that a naive two-way demeaning uses is only valid when balanced.
        twoway && throw(CliError("usage/invalid",
            "estimate preg: --absorb and --twoway are mutually exclusive";
            hint="pass --absorb entity,time to absorb entity and time fixed effects"))
        length(unique(absorb_dims)) == length(absorb_dims) || throw(CliError("usage/invalid",
            "estimate preg: --absorb contains duplicate dimensions ($(join(absorb_dims, ", ")))"))
        hdfe_tol > 0 || throw(CliError("usage/invalid",
            "estimate preg: --hdfe-tol must be > 0 (got $hdfe_tol)"))
        hdfe_maxiter >= 1 || throw(CliError("usage/invalid",
            "estimate preg: --hdfe-maxiter must be ≥ 1 (got $hdfe_maxiter)"))
    else
        (hdfe_tol == 1e-8 && hdfe_maxiter == 1000) || throw(CliError("usage/invalid",
            "estimate preg: --hdfe-tol/--hdfe-maxiter apply only with --absorb"))
    end

    # ── W10/#131: instrument-proliferation controls (MEMs#549) ──────────────
    # Upstream accepts these on every method but only the :ab/:bb GMM path reads
    # them — a silent no-op elsewhere, so refuse rather than pass through.
    is_gmm = method in ("ab", "bb")
    if is_gmm
        min_lag_endo >= 1 || throw(CliError("usage/invalid",
            "estimate preg: --min-lag-endo must be ≥ 1 (got $min_lag_endo)"))
        max_lag_endo >= min_lag_endo || throw(CliError("usage/invalid",
            "estimate preg: --max-lag-endo must be ≥ --min-lag-endo (got $max_lag_endo < $min_lag_endo)"))
    elseif collapse || min_lag_endo != 2 || max_lag_endo != 99
        throw(CliError("usage/invalid",
            "estimate preg: --collapse/--min-lag-endo/--max-lag-endo apply only to --method ab|bb (got --method $method)";
            hint="these shape the dynamic-panel GMM instrument matrix; use --method ab or bb"))
    end

    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model_sym = _to_sym(method)
    cov_sym = _to_sym(cov_type)
    _status("Panel Regression ($method): $dep ~ $(join(indep_syms, " + "))")
    ar1 == "none" || _status("  Prais-Winsten AR(1): $ar1")
    cov_type == "pcse" && _status("  PCSE unbalanced handling: $pcse_unbalanced")
    isempty(absorb_dims) || _status("  Absorbing FE: $(join(absorb_dims, " × "))")
    is_gmm && _status("  GMM instruments: lags $min_lag_endo:$max_lag_endo" *
                      (collapse ? ", collapsed" : ""))
    _status()

    # Previously unwrapped: an upstream ArgumentError surfaced as exit 1.
    model = try
        estimate_xtreg(pd, Symbol(dep), indep_syms;
            model=model_sym, twoway=twoway, cov_type=cov_sym,
            # NOT `_to_sym`: these are CSV column names (or the reserved index aliases),
            # not hyphenated CLI enum values — rewriting `-`→`_` would corrupt a legal
            # column name like `region-code`.
            absorb=Symbol[Symbol(d) for d in absorb_dims],
            hdfe_tol=hdfe_tol, hdfe_maxiter=hdfe_maxiter,
            ar1=_to_sym(replace(ar1, '-' => '_')),
            pcse_unbalanced=_to_sym(pcse_unbalanced),
            collapse=collapse, min_lag_endo=min_lag_endo, max_lag_endo=max_lag_endo)
    catch e
        throw(_garch_variant_error(e, "panel regression"))
    end

    coef_df = _preg_coef_table(model, model.varnames)
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Regression Coefficients ($method)", key="panel_regression_coefficients")

    _status()
    pairs = Pair{String,Any}[
        "R2 (within)"  => round(model.r2_within; digits=6),
        "R2 (between)" => round(model.r2_between; digits=6),
        "R2 (overall)" => round(model.r2_overall; digits=6),
        "F-statistic"  => round(model.f_stat; digits=4),
        "F p-value"    => round(hasproperty(model, :f_pval) ? model.f_pval :
                                (hasproperty(model, :f_pvalue) ? model.f_pvalue : NaN); digits=4),
        "N obs"        => model.n_obs,
        "N groups"     => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")

    # Emitted ONLY for --method ab|bb (dynamic_diagnostics is nothing elsewhere),
    # so no other preg envelope changes. n_instruments is the headline: it is what
    # --collapse/--min-lag-endo/--max-lag-endo exist to control, and Hansen J is
    # unreliable exactly when it proliferates (Roodman 2009).
    dd = hasproperty(model, :dynamic_diagnostics) ? model.dynamic_diagnostics : nothing
    if dd !== nothing
        dpairs = Pair{String,Any}[
            "AR(1) statistic" => round(dd.ar1; digits=4),
            "AR(1) p-value"   => round(dd.ar1_p; digits=4),
            "AR(2) statistic" => round(dd.ar2; digits=4),
            "AR(2) p-value"   => round(dd.ar2_p; digits=4),
            "Hansen J"        => round(dd.hansen; digits=4),
            "Hansen df"       => dd.hansen_df,
            "Hansen p-value"  => round(dd.hansen_p; digits=4),
            "n_instruments"   => dd.n_instruments,
            "collapse"        => collapse,
            "instrument lag window" => "$min_lag_endo:$max_lag_endo",
        ]
        output_kv(dpairs; format=format, title="Dynamic Panel Diagnostics")
    end

    # Emitted ONLY under --absorb, so a plain `estimate preg` envelope is unchanged. The
    # absorbed-parameter count is not decoration: it is the degrees of freedom the within
    # transformation consumed, and `converged=false` means the reported coefficients came
    # from a TRUNCATED projection loop — a silent accuracy loss otherwise invisible.
    hd = hasproperty(model, :hdfe) ? model.hdfe : nothing
    if hd !== nothing
        hpairs = Pair{String,Any}[
            "absorb"          => join(string.(hd.absorb), ","),
            "n_absorbed"      => hd.n_absorbed,
            "n_levels"        => join(string.(hd.n_levels), ","),
            "n_components"    => hd.n_components,
            "converged"       => hd.converged,
            "iterations"      => hd.iterations,
            "final_change"    => hd.change,
            "tol"             => hd.tol,
        ]
        output_kv(hpairs; format=format, title="HDFE Absorption")
        hd.converged || _status_styled(
            "  Warning: HDFE absorption did NOT converge in $(hd.iterations) iterations " *
            "(change $(hd.change) > tol $(hd.tol)); raise --hdfe-maxiter or --hdfe-tol\n";
            color=:yellow)
    end
    return model
end

function _estimate_piv(; data::String, dep::String="", exog::String="",
                        endog::String="", instruments::String="",
                        method::String="fe", cov_type::String="cluster",
                        id_col::String="", time_col::String="",
                        output::String="", format::String="table")
    # were bare error() → internal exit 1 for ordinary usage mistakes
    isempty(dep) && throw(CliError("usage/missing", "--dep is required";
        hint="name the dependent variable column, e.g. --dep y"))
    isempty(endog) && throw(CliError("usage/missing", "--endog is required";
        hint="name the endogenous regressor(s), e.g. --endog investment"))
    pd = _load_panel_for_preg(data, id_col, time_col)

    exog_syms = isempty(exog) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(exog, ",")]
    endog_syms = Symbol[Symbol(strip(s)) for s in split(endog, ",")]
    inst_syms = isempty(instruments) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(instruments, ",")]

    _status("Panel IV ($method): $dep ~ $(join([exog_syms; endog_syms], " + "))")
    _status("  Instruments: $(join(inst_syms, ", "))")
    _status()

    model = estimate_xtiv(pd, Symbol(dep), exog_syms, endog_syms;
        instruments=inst_syms, model=_to_sym(method), cov_type=_to_sym(cov_type))

    all_vars = [string.(exog_syms); string.(endog_syms)]
    coef_df = _preg_coef_table(model, all_vars)
    output_result(coef_df; format=Symbol(format), output=output, title="Panel IV Coefficients")

    # ── W10/#131 (MEMs#553): weak-instrument diagnostics, populated since 0.7.3.
    # A `nothing` upstream means the computation FAILED or the model is
    # underidentified (bare try/catch in _panel_weak_iv_diagnostics; Sargan is
    # nothing when just-identified) — say so rather than a clean "N/A".
    _status()
    fmt_d(x) = x === nothing ? "unavailable (failed or underidentified)" :
               round(Float64(x); digits=4)
    dpairs = Pair{String,Any}[
        "first-stage F (min partial)"    => fmt_d(model.first_stage_f),
        "Cragg-Donald F"                 => fmt_d(model.cragg_donald_f),
        "Kleibergen-Paap F"              => fmt_d(model.kleibergen_paap_f),
        "Stock-Yogo 10% critical value"  => fmt_d(model.stock_yogo_10pct),
        "Sargan statistic"               => fmt_d(model.sargan_stat),
        "Sargan p-value"                 => fmt_d(model.sargan_pval),
    ]
    output_kv(dpairs; format=format, title="Weak-Instrument Diagnostics")
    # Upstream computes KP with HC1 even under --cov-type cluster (a
    # cluster-robust rk statistic is not implemented) — flag it, or the number
    # reads as cluster-robust when it is not.
    cov_type == "cluster" && _status_styled(
        "  Note: Kleibergen-Paap F ignores within-entity dependence (HC1, not cluster-robust)\n";
        color=:yellow)
    return model
end

function _estimate_plogit(; data::String, dep::String="", indep::String="",
                           method::String="pooled", cov_type::String="cluster",
                           id_col::String="", time_col::String="",
                           output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    _status("Panel Logit ($method): $dep ~ $(join(indep_syms, " + "))")
    _status()

    model = estimate_xtlogit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    coef_df = _preg_coef_table(model, model.varnames)
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Logit Coefficients ($method)", key="panel_logit_coefficients")

    _status()
    pairs = Pair{String,Any}[
        "Pseudo R2"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Converged"       => model.converged,
        "N obs"           => model.n_obs,
        "N groups"        => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")
    return model
end

function _estimate_pprobit(; data::String, dep::String="", indep::String="",
                            method::String="pooled", cov_type::String="cluster",
                            id_col::String="", time_col::String="",
                            output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    _status("Panel Probit ($method): $dep ~ $(join(indep_syms, " + "))")
    _status()

    model = estimate_xtprobit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    coef_df = _preg_coef_table(model, model.varnames)
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Probit Coefficients ($method)", key="panel_probit_coefficients")

    _status()
    pairs = Pair{String,Any}[
        "Pseudo R2"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Converged"       => model.converged,
        "N obs"           => model.n_obs,
        "N groups"        => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")
    return model
end

# ── Ordered Logit ──────────────────────────────────────

function _estimate_ologit(; data::String, dep::String="", cov_type::String="ols",
                           clusters::String="",
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    _status("Ordered Logit: $dep_name ~ $(join(xcols, " + "))")
    _status()

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    J = length(model.cutpoints)
    cut_df = DataFrame(
        Cutpoint = ["cut$i" for i in 1:J],
        Value = round.(model.cutpoints; digits=6),
    )
    output_result(cut_df; format=Symbol(format), output="", title="Cutpoints")

    _status()
    # C051: MEMs tidy coef table (term|estimate|std_error|stat|p_value|ci_lower|ci_upper).
    output_result(DataFrame(model); format=Symbol(format), output=output, title="Ordered Logit Coefficients")

    _status()
    pairs = Pair{String,Any}[
        "Pseudo R2"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Categories"      => length(model.categories),
    ]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

# ── Ordered Probit ─────────────────────────────────────

function _estimate_oprobit(; data::String, dep::String="", cov_type::String="ols",
                            clusters::String="",
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    _status("Ordered Probit: $dep_name ~ $(join(xcols, " + "))")
    _status()

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    J = length(model.cutpoints)
    cut_df = DataFrame(Cutpoint = ["cut$i" for i in 1:J], Value = round.(model.cutpoints; digits=6))
    output_result(cut_df; format=Symbol(format), output="", title="Cutpoints")

    _status()
    # C051: MEMs tidy coef table (term|estimate|std_error|stat|p_value|ci_lower|ci_upper).
    output_result(DataFrame(model); format=Symbol(format), output=output, title="Ordered Probit Coefficients")

    _status()
    pairs = Pair{String,Any}[
        "Pseudo R2" => round(model.pseudo_r2; digits=6),
        "Log-likelihood" => round(model.loglik; digits=4),
        "AIC" => round(model.aic; digits=4), "BIC" => round(model.bic; digits=4),
        "Categories" => length(model.categories)]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

# ── Multinomial Logit ──────────────────────────────────

function _estimate_mlogit(; data::String, dep::String="", cov_type::String="ols",
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    _status("Multinomial Logit: $dep_name ~ $(join(xcols, " + "))")
    _status()

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols)

    # C051: MEMs tidy coef table keyed by alternative — all categories in one table
    # (alternative|term|estimate|std_error|stat|p_value|ci_lower|ci_upper), replacing the
    # per-category loop of wide tables.
    output_result(DataFrame(model); format=Symbol(format), output=output,
                  title="Multinomial Logit Coefficients")
    _status()

    pairs = Pair{String,Any}[
        "Pseudo R2" => round(model.pseudo_r2; digits=6),
        "Log-likelihood" => round(model.loglik; digits=4),
        "AIC" => round(model.aic; digits=4), "BIC" => round(model.bic; digits=4),
        "Categories" => size(model.fitted, 2)]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

# ── W2/#107: count-data regression (Poisson / NB2) ────────────────────────────
#
# MEMs#427. Deliberate surface decisions, each checked against the 0.7.2 source rather
# than the issue text (which is wrong on two of them):
#   * `estimate_nbreg` DOES accept `exposure` — the issue said offset-only.
#   * `residuals(m)` is a bare field accessor with NO `kind` kwarg, so no `--kind` option
#     is advertised (a declared option its handler cannot honour is the #85 failure mode).
#   * `src/plotting/` has NO dispatch covering PoissonModel/NegBinModel, so neither leaf
#     declares `--plot`/`--plot-save` — the mock's generic plot_result would hide that.
#   * `--cov-type` on poisson defaults to `robust`, mirroring upstream's QMLE sandwich, and
#     its choice set is upstream's own (robust|mle|hc0..hc3|cluster) — NOT the shared
#     REG_OPTIONS set, which lacks `robust`/`mle` and defaults to hc1.
#   * `estimate_nbreg` takes no cov_type/clusters at all, so neither is offered there.

"""Load count data and guard the response BEFORE handing it to the estimator.

Upstream rejects negative or non-integer `y`, but through a raw `ArgumentError`; catching it
here keeps the class typed (`data/invalid`) instead of surfacing as internal/error (exit 1).
"""
function _load_count_data(data::String, dep::String, offset_col::String,
                          exposure_col::String, clusters_col::String)
    isempty(offset_col) || isempty(exposure_col) || throw(CliError("usage/invalid",
        "--offset and --exposure are mutually exclusive (exposure is log-transformed into an offset)";
        hint="pass --exposure for a population/time-at-risk column, --offset for an already-logged one"))
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters_col)
    any(v -> v < 0, y) && throw(CliError("data/invalid",
        "count regression requires a non-negative response, got a negative value";
        hint="check --dep — Poisson/NB2 model counts, not levels or differences"))
    all(v -> isinteger(v), y) || throw(CliError("data/invalid",
        "count regression requires an integer-valued response";
        hint="check --dep — Poisson/NB2 model event counts"))
    off = isempty(offset_col) ? nothing : _load_count_vector(data, offset_col, "offset", y)
    exp_ = isempty(exposure_col) ? nothing : _load_count_vector(data, exposure_col, "exposure", y)
    exp_ === nothing || all(v -> v > 0, exp_) || throw(CliError("data/invalid",
        "--exposure must be strictly positive (it is log-transformed into the offset)"))
    return y, X, xcols, off, exp_
end

"""Read one auxiliary numeric column (offset/exposure), guarding every raw conversion."""
function _load_count_vector(data::String, col::String, role::String, y::AbstractVector)
    df = load_data(data)
    col in names(df) || throw(CliError("data/missing-column",
        "--$role column '$col' not found"; hint="available: $(join(names(df), ", "))"))
    raw = df[!, col]
    any(v -> v === missing, raw) && throw(CliError("data/missing-value",
        "--$role column '$col' contains missing values"))
    v = try
        Vector{Float64}(raw)
    catch e
        throw(CliError("data/invalid", "--$role column '$col' must be numeric";
                       hint=sprint(showerror, e)))
    end
    length(v) == length(y) || throw(CliError("data/shape",
        "--$role column '$col' has $(length(v)) rows, expected $(length(y))"))
    return v
end

"""Shared IRR table for `--irr`. `incidence_rate_ratio` returns an `OddsRatio`, whose
point-estimate field is `or` (NOT `odds_ratio` — the #84 alias trap)."""
function _irr_table(model, conf_level::Float64)
    irr = incidence_rate_ratio(model; conf_level=conf_level)
    return DataFrame(term=copy(irr.varnames),
                     irr=round.(Float64.(irr.or); digits=6),
                     std_error=round.(Float64.(irr.se); digits=6),
                     ci_lower=round.(Float64.(irr.ci_lower); digits=6),
                     ci_upper=round.(Float64.(irr.ci_upper); digits=6))
end

"""Tidy coefficient table for the count models.

Hand-built on purpose: `_reg_coef_table` goes through `DataFrame(model)`, and MEMs 0.7.2
does NOT register `PoissonModel`/`NegBinModel` with Tables.jl — routing them there raises
"no default `Tables.columns` implementation" and exits 1. Column set matches the tidy C051
convention the registered models produce, so the envelope shape stays uniform.
"""
function _count_coef_table(model, conf_level::Float64)
    beta = Float64.(coef(model))
    se = Float64.(stderror(model))
    z = [s > 0 ? b / s : NaN for (b, s) in zip(beta, se)]
    # Distributions is not a direct CLI dep — use the shared A&S normal helpers, same as
    # the volatility renderers, rather than pulling `Normal()` in through MEMs.
    zc = _normal_quantile(1 - (1 - conf_level) / 2)
    return DataFrame(
        term=copy(model.varnames),
        estimate=round.(beta; digits=6),
        std_error=round.(se; digits=6),
        z_stat=round.(z; digits=4),
        p_value=round.([isnan(v) ? NaN : 2 * (1 - _normal_cdf(abs(v))) for v in z]; digits=6),
        ci_lower=round.(beta .- zc .* se; digits=6),
        ci_upper=round.(beta .+ zc .* se; digits=6),
    )
end

"""Fit statistics shared by the two count estimators."""
function _count_fit_pairs(model)
    return Pair{String,Any}[
        "Pseudo R²"      => round(Float64(model.pseudo_r2); digits=6),
        "Log-likelihood" => round(Float64(model.loglik); digits=4),
        "Log-lik (null)" => round(Float64(model.loglik_null); digits=4),
        "Deviance"       => round(Float64(model.deviance); digits=4),
        "AIC"            => round(Float64(model.aic); digits=4),
        "BIC"            => round(Float64(model.bic); digits=4),
        "Converged"      => model.converged,
        "Iterations"     => model.iterations,
    ]
end

function _count_guards(maxiter::Int, tol::Float64, conf_level::Float64, leaf::String)
    maxiter >= 1 || throw(CliError("usage/invalid", "$leaf: --maxiter must be ≥ 1 (got $maxiter)"))
    tol > 0 || throw(CliError("usage/invalid", "$leaf: --tol must be > 0 (got $tol)"))
    (0 < conf_level < 1) || throw(CliError("usage/invalid",
        "$leaf: --conf-level must satisfy 0 < level < 1 (got $conf_level)"))
end

function _fit_poisson(data, dep, offset, exposure, cov_type, clusters, maxiter, tol)
    y, X, xcols, off, exp_ = _load_count_data(data, dep, offset, exposure, clusters)
    cl = _load_clusters(data, clusters)
    model = try
        estimate_poisson(y, X; offset=off, exposure=exp_, cov_type=Symbol(cov_type),
                         varnames=xcols, clusters=cl, maxiter=maxiter, tol=tol)
    catch e
        throw(_domain_or_data_error(e, "Poisson regression"))
    end
    return model, xcols
end

function _fit_nbreg(data, dep, offset, exposure, maxiter, tol)
    y, X, xcols, off, exp_ = _load_count_data(data, dep, offset, exposure, "")
    model = try
        estimate_nbreg(y, X; offset=off, exposure=exp_, varnames=xcols,
                       maxiter=maxiter, tol=tol)
    catch e
        throw(_domain_or_data_error(e, "Negative binomial regression"))
    end
    return model, xcols
end

"""Map a raw estimator exception to a typed class. An `ArgumentError` from these estimators
is always a statement about the DATA (bad response, mismatched offset), so it maps to
`data/invalid` rather than the generic model class; everything else defers to
`_domain_error_class` via CliError so it can never surface as internal/error.

W4/#139: `_domain_error_class` is consulted BEFORE the generic fallbacks so the
direct-`Exception` domain types outside `MacroModelError` (`StochasticSingularityError`,
`DSGESolveError`) keep their specific classes on wrapped paths — the old fallback
collapsed them into `model/error`."""
function _domain_or_data_error(e, label::String)
    e isa CliError && return e
    _has_supertype_named(typeof(e), :MacroModelError) && return e
    cli = _domain_error_class(e)
    cli === nothing || return cli
    e isa ArgumentError && return CliError("data/invalid",
        "$label: $(sprint(showerror, e))")
    return CliError("model/error", "$label failed"; hint=sprint(showerror, e))
end

function _estimate_poisson(; data::String, dep::String="", offset::String="",
                            exposure::String="", cov_type::String="robust",
                            clusters::String="", maxiter::Int=100, tol::Float64=1e-10,
                            conf_level::Float64=0.95, irr::Bool=false,
                            output::String="", format::String="table")
    _count_guards(maxiter, tol, conf_level, "estimate poisson")
    model, xcols = _fit_poisson(data, dep, offset, exposure, cov_type, clusters, maxiter, tol)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Poisson Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(model.y)), Regressors: $(length(xcols)), Cov type: $cov_type")
    _status()
    output_result(_count_coef_table(model, conf_level); format=Symbol(format), output=output,
                  title="Poisson Regression Coefficients")
    if irr
        output_result(_irr_table(model, conf_level); format=Symbol(format),
                      output=_per_var_output_path(output, "irr"),
                      title="Incidence-Rate Ratios ($(Int(round(conf_level*100)))% CI)",
                      key="incidence_rate_ratios")
    end
    _status()
    output_kv(_count_fit_pairs(model); format=format, title="Fit Statistics")
    return model
end

function _estimate_nbreg(; data::String, dep::String="", offset::String="",
                          exposure::String="", maxiter::Int=1000, tol::Float64=1e-10,
                          conf_level::Float64=0.95, irr::Bool=false,
                          output::String="", format::String="table")
    _count_guards(maxiter, tol, conf_level, "estimate nbreg")
    model, xcols = _fit_nbreg(data, dep, offset, exposure, maxiter, tol)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Negative Binomial (NB2) Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(model.y)), Regressors: $(length(xcols))")
    _status()
    output_result(_count_coef_table(model, conf_level); format=Symbol(format), output=output,
                  title="Negative Binomial Regression Coefficients")
    # α and its delta-method SE are NOT in the coefficient table (upstream's vcov slice
    # stops at β), so they get their own table — with a distinct output path, or --output
    # would drop the coefficients above.
    z = model.alpha_se > 0 ? model.alpha / model.alpha_se : NaN
    output_result(DataFrame(parameter=["alpha"],
                            estimate=[round(Float64(model.alpha); digits=6)],
                            std_error=[round(Float64(model.alpha_se); digits=6)],
                            z_stat=[round(Float64(z); digits=4)]);
                  format=Symbol(format), output=_per_var_output_path(output, "alpha"),
                  title="Overdispersion Parameter")
    if irr
        output_result(_irr_table(model, conf_level); format=Symbol(format),
                      output=_per_var_output_path(output, "irr"),
                      title="Incidence-Rate Ratios ($(Int(round(conf_level*100)))% CI)",
                      key="incidence_rate_ratios")
    end
    _status()
    output_kv(_count_fit_pairs(model); format=format, title="Fit Statistics")
    return model
end

function _predict_poisson(; data::String="", dep::String="", offset::String="",
                           exposure::String="", cov_type::String="robust",
                           clusters::String="", maxiter::Int=100, tol::Float64=1e-10,
                           output::String="", format::String="table", model=nothing)
    _count_guards(maxiter, tol, 0.95, "predict poisson")
    m = model === nothing ?
        first(_fit_poisson(data, dep, offset, exposure, cov_type, clusters, maxiter, tol)) : model
    _status("Poisson conditional means: n=$(length(predict(m)))"); _status()
    mu = predict(m)   # 1-arg predict == m.fitted upstream; the (m, Xnew) form is out of scope
    return output_result(DataFrame(observation=1:length(mu),
                                   fitted=round.(Float64.(mu); digits=6));
                         format=Symbol(format), output=output,
                         title="Poisson Conditional Means")
end

function _predict_nbreg(; data::String="", dep::String="", offset::String="",
                         exposure::String="", maxiter::Int=1000, tol::Float64=1e-10,
                         output::String="", format::String="table", model=nothing)
    _count_guards(maxiter, tol, 0.95, "predict nbreg")
    m = model === nothing ? first(_fit_nbreg(data, dep, offset, exposure, maxiter, tol)) : model
    mu = predict(m)
    _status("Negative binomial conditional means: n=$(length(mu))"); _status()
    return output_result(DataFrame(observation=1:length(mu),
                                   fitted=round.(Float64.(mu); digits=6));
                         format=Symbol(format), output=output,
                         title="Negative Binomial Conditional Means")
end

function _residuals_poisson(; data::String="", dep::String="", offset::String="",
                             exposure::String="", cov_type::String="robust",
                             clusters::String="", maxiter::Int=100, tol::Float64=1e-10,
                             output::String="", format::String="table", model=nothing)
    _count_guards(maxiter, tol, 0.95, "residuals poisson")
    m = model === nothing ?
        first(_fit_poisson(data, dep, offset, exposure, cov_type, clusters, maxiter, tol)) : model
    r = residuals(m)
    _status("Poisson residuals: n=$(length(r))"); _status()
    return output_result(DataFrame(observation=1:length(r),
                                   residual=round.(Float64.(r); digits=6));
                         format=Symbol(format), output=output,
                         title="Poisson Residuals")
end

function _residuals_nbreg(; data::String="", dep::String="", offset::String="",
                           exposure::String="", maxiter::Int=1000, tol::Float64=1e-10,
                           output::String="", format::String="table", model=nothing)
    _count_guards(maxiter, tol, 0.95, "residuals nbreg")
    m = model === nothing ? first(_fit_nbreg(data, dep, offset, exposure, maxiter, tol)) : model
    r = residuals(m)
    _status("Negative binomial residuals: n=$(length(r))"); _status()
    return output_result(DataFrame(observation=1:length(r),
                                   residual=round.(Float64.(r); digits=6));
                         format=Symbol(format), output=output,
                         title="Negative Binomial Residuals")
end

"""`test dispersion` — Cameron & Trivedi (1990) overdispersion test.

`dispersion_test` takes a fitted POISSON model and returns both the NB1 and NB2 auxiliary
regressions, each a NamedTuple of (alpha, se, t_stat, p_value). A significantly positive α
means the Poisson equidispersion assumption fails and `estimate nbreg` is preferred.
"""
function _test_dispersion(; data::String="", dep::String="", offset::String="",
                           exposure::String="", cov_type::String="robust",
                           clusters::String="", maxiter::Int=100, tol::Float64=1e-10,
                           alpha::Float64=0.05, output::String="", format::String="table")
    _count_guards(maxiter, tol, 0.95, "test dispersion")
    (0 < alpha < 1) || throw(CliError("usage/invalid",
        "test dispersion: --alpha must satisfy 0 < alpha < 1 (got $alpha)"))
    m, _ = _fit_poisson(data, dep, offset, exposure, cov_type, clusters, maxiter, tol)
    dt = try
        dispersion_test(m)
    catch e
        throw(_domain_or_data_error(e, "Dispersion test"))
    end
    rows = [(form="NB2", nt=dt.nb2), (form="NB1", nt=dt.nb1)]
    df = DataFrame(
        form=[r.form for r in rows],
        alpha=[round(Float64(r.nt.alpha); digits=6) for r in rows],
        std_error=[round(Float64(r.nt.se); digits=6) for r in rows],
        t_stat=[round(Float64(r.nt.t_stat); digits=4) for r in rows],
        p_value=[round(Float64(r.nt.p_value); digits=6) for r in rows],
        # The p-value is two-sided, but the decision is DIRECTIONAL: alpha > 0 is
        # overdispersion (Poisson understates the variance → prefer nbreg), alpha < 0 is
        # UNDERdispersion, for which nbreg is not the remedy — NB2 cannot represent it.
        # Collapsing both into "reject equidispersion ⇒ use nbreg" would recommend the
        # wrong model on underdispersed counts.
        decision=[Float64(r.nt.p_value) < alpha ?
                  (Float64(r.nt.alpha) > 0 ? "overdispersed" : "underdispersed") :
                  "equidispersion not rejected" for r in rows],
    )
    _status("Cameron-Trivedi dispersion test on a Poisson fit (n=$(dt.n))"); _status()
    output_result(df; format=Symbol(format), output=output,
                  title="Overdispersion Test (Cameron & Trivedi 1990)")
    sig = Float64(dt.nb2.p_value) < alpha
    a2 = Float64(dt.nb2.alpha)
    preferred = if sig && a2 > 0
        "nbreg (overdispersion: alpha > 0 and significant)"
    elseif sig
        "poisson (significant UNDERdispersion: alpha < 0 — nbreg does not model this; " *
        "consider a generalised-Poisson or Conway-Maxwell-Poisson model)"
    else
        "poisson (equidispersion not rejected)"
    end
    output_kv(Pair{String,Any}[
        "n"               => dt.n,
        "alpha_level"     => alpha,
        "nb2_alpha"       => round(a2; digits=6),
        "preferred_model" => preferred,
    ]; format=format, title="Dispersion Summary")
    return dt
end

# ── C064a: univariate GARCH variants ─────────────────────────
# igarch / cgarch / aparch / figarch / fiegarch / garch-midas (MEMs 0.7.0 src/garch).
# None of these result types is registered in MEMs `_COEF_TABLE_TYPES`, so — per the
# C051 convention — their coefficient table is HAND-BUILT here (the same documented
# exception used by the SUR/3SLS systems leaves), not via `DataFrame(model)`.

"""Hand-built coefficient table for a GARCH-variant volatility model:
`parameter | estimate | std_error | z_stat | p_value`. Falls back to estimate-only
when the QMLE covariance is unavailable (non-finite SEs)."""
function _garch_variant_coef_table(model, param_names::Vector{String})
    c = Float64.(coef(model))
    names = param_names[1:length(c)]
    try
        se = Float64.(stderror(model))
        (length(se) == length(c) && all(isfinite, se)) || error("unavailable SEs")
        z = [se[i] == 0 ? 0.0 : c[i] / se[i] for i in eachindex(c)]
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        return DataFrame(parameter=names, estimate=round.(c; digits=6),
                         std_error=round.(se; digits=6),
                         z_stat=round.(z; digits=4), p_value=round.(pv; digits=4))
    catch
        return DataFrame(parameter=names, estimate=round.(c; digits=6))
    end
end

"""Diagnostics kv for a GARCH-variant fit: log-lik, AIC/BIC, persistence,
convergence, iterations, plus any model-specific `extra` rows."""
function _garch_variant_diag(model; extra::Vector{<:Pair}=Pair{String,Any}[])
    pairs = Pair{String,Any}[
        "log_likelihood" => round(Float64(loglikelihood(model)); digits=4),
        "aic"            => round(Float64(model.aic); digits=4),
        "bic"            => round(Float64(model.bic); digits=4),
        "persistence"    => round(Float64(persistence(model)); digits=6),
        "converged"      => model.converged,
        "iterations"     => model.iterations,
    ]
    append!(pairs, extra)
    return pairs
end

"""Map an untyped MEMs volatility-estimation failure to a typed CliError — never an
uncaught exit-1 (io-family lesson). Bad-input `ArgumentError`/`DomainError` →
`data/invalid` (3); shape mismatch → `data/shape` (3); anything else → `model/error` (5)."""
function _garch_variant_error(e, label::String)
    e isa CliError && return e
    if e isa ArgumentError || e isa DomainError
        return CliError("data/invalid", "$label: $(sprint(showerror, e))";
            hint="check series length, orders (p/q), and parameter bounds")
    elseif e isa DimensionMismatch
        return CliError("data/shape", "$label: $(sprint(showerror, e))")
    end
    return CliError("model/error", "$label estimation failed: $(sprint(showerror, e))";
        hint="the QMLE optimizer did not converge — try a longer or cleaner return series")
end

# ── C065 nonlinear-TS shared renderers/mappers (reused by SETAR/STAR/MS sub-waves) ──
# The three nonlinear model types are NOT in MEMs `_COEF_TABLE_TYPES`, so every
# coefficient table is hand-built (the documented C051 exception, like io/mgarch/sur).

"""Tidy per-regime coefficient table for a nonlinear-TS model regime block:
`regime | term | estimate | std_error | z_stat | p_value` (large-sample normal
z/p, matching the `_garch_variant`/`dsge` `_normal_cdf` convention). Callers `vcat`
the two regime blocks into one table (C065a SETAR, reused by C065b STAR)."""
function _threshold_coef_table(betas::AbstractVector, ses::AbstractVector,
                               terms::Vector{String}, regime_label::String)
    b = Float64.(betas); s = Float64.(ses)
    n = length(b)
    tm = length(terms) == n ? terms : String["term$i" for i in 1:n]
    z  = [s[i] == 0.0 ? 0.0 : b[i] / s[i] for i in 1:n]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    return DataFrame(regime=fill(regime_label, n), term=tm[1:n],
                     estimate=round.(b; digits=6), std_error=round.(s; digits=6),
                     z_stat=round.(z; digits=4), p_value=round.(pv; digits=4))
end

"""Map an untyped MEMs nonlinear-TS failure (SETAR/STAR/MS estimator, test, or
forecast) to a typed CliError — never an uncaught exit-1 on bad input (the standing
shared lesson). Same shape as `_garch_variant_error`: bad-input `ArgumentError`/
`DomainError` → `data/invalid` (3); external-series length mismatch → `data/shape`
(3); anything else (optimizer failure) → `model/error` (5); a `CliError` passes
through."""
function _nonlinear_error(e, label::String)
    e isa CliError && return e
    if e isa ArgumentError || e isa DomainError
        return CliError("data/invalid", "$label: $(sprint(showerror, e))";
            hint="check the series length, AR order p, delay d, trimming, and any external transition series")
    elseif e isa DimensionMismatch
        return CliError("data/shape", "$label: $(sprint(showerror, e))")
    end
    return CliError("model/error", "$label failed: $(sprint(showerror, e))";
        hint="the nonlinear optimizer did not converge — try a longer or cleaner series")
end

"""Parse a SETAR `--d` delay argument: the sentinel `"auto"` → `:auto` (=1:p grid
inside MEMs), else a positive integer. Junk or a non-positive integer → a typed
`usage/invalid` (never a raw parse throw). Shared by `estimate setar`/`forecast setar`."""
function _parse_setar_delay(d::AbstractString)
    lowercase(strip(d)) == "auto" && return :auto
    v = tryparse(Int, strip(d))
    v === nothing && throw(CliError("usage/invalid",
        "--d must be 'auto' or a positive integer (got '$d')"))
    v >= 1 || throw(CliError("usage/invalid", "--d must be ≥ 1 (got $v)"))
    return v
end

# ─────────────────────────────────────────────────────────────────────────────
# C064 remainder (#69): forecast / predict / residuals for the six univariate GARCH
# variants added by C064a.
#
# DESIGN: these are per-variant handlers rather than entries in `VOL_MODELS`. That
# factory (shared.jl) generates all four verbs from one descriptor, but its handler
# signature is fixed at (data, column, p, q, draws) — it cannot express aparch's
# --fix-delta/--fix-gamma, figarch/fiegarch's --d0/--truncation/--dist, or
# garch-midas's --m-freq/--k/--rv/--span. Each verb handler therefore mirrors its
# `estimate` sibling's option set, refits, and renders through the three shared
# helpers below, so only the fit call is repeated (2 lines), never the rendering.
#
# UPSTREAM SHAPES verified against MEMs garch/{forecast,figarch,midas,types}.jl:
#   * forecast(m, h; conf_level, n_sim) for igarch/cgarch/aparch/figarch/fiegarch
#   * forecast(m, h) for GarchMidasModel — NOTE the type is `GarchMidasModel`, not
#     `GARCHMIDASModel`, and it takes NO conf_level/n_sim, so it gets no --conf-level.
#   * residuals(m) for all six.
#   * predict(m) (in-sample conditional variance) exists for igarch/cgarch/aparch and
#     garch-midas but NOT for figarch/fiegarch, whose only `predict` is the h-argument
#     forecast form. Those two read the real `.conditional_variance` field instead.
# ─────────────────────────────────────────────────────────────────────────────

"""In-sample conditional variance table, matching `_make_predict_vol`'s output shape
so `predict garch` and `predict igarch` render identically."""
function _vol_variant_predict_output(cond_var, vname::String, title::String;
                                     format::String, output::String, key::String="")
    cv = Float64.(collect(cond_var))
    output_result(DataFrame(t=1:length(cv), variance=round.(cv; digits=6),
                            volatility=round.(sqrt.(abs.(cv)); digits=6));
        format=Symbol(format), output=output, title="$title ($vname)", key=key)
    return cond_var
end

"""Standardized-residual table, matching `_make_residuals_vol`'s output shape."""
function _vol_variant_residuals_output(resid, vname::String, title::String;
                                       format::String, output::String, key::String="")
    r = Float64.(collect(resid))
    output_result(DataFrame(t=1:length(r), residual=round.(r; digits=6));
        format=Symbol(format), output=output, title="$title ($vname)", key=key)
    return resid
end

"""Conditional variance for a fitted variant. figarch/fiegarch have no zero-argument
`predict` upstream, so read the real `.conditional_variance` field for them."""
_vol_variant_cond_var(m) = hasmethod(predict, Tuple{typeof(m)}) ? predict(m) :
                                                                  m.conditional_variance


# ── C064 remainder (#69): the 18 verb handlers. Each mirrors its `estimate`
# sibling's option set so the model can be refit, then renders through the shared
# helpers above.

function _forecast_igarch(; data::String="", result=nothing, column::Int=1, p::Int=1, q::Int=1, horizons::Int=10,
        conf_level::Float64=0.95, model=nothing, output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, leaf="forecast igarch")
    if loaded !== nothing
        h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
        _vol_forecast_output(loaded, "result", "IGARCH($p,$q)", h; format=format, output=output,
                             key="igarch_volatility_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "forecast igarch: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < conf_level < 1.0) || throw(CliError("usage/invalid",
        "forecast igarch: --conf-level must be in (0, 1) (got $conf_level)"))
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_igarch(y, p, q); catch e; throw(_garch_variant_error(e, "IGARCH")); end) : model
    _status("IGARCH Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=conf_level)
    catch e
        throw(_garch_variant_error(e, "IGARCH forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _vol_forecast_output(fc, vname, "IGARCH($p,$q)", horizons; format=format, output=output,
                         key="igarch_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_igarch(; data::String, column::Int=1, p::Int=1, q::Int=1, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_igarch(y, p, q); catch e; throw(_garch_variant_error(e, "IGARCH")); end) : model
    _status("IGARCH conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "IGARCH($p,$q)" * " Conditional Variance"; format=format, output=output,
        key="igarch_conditional_variance")
end

function _residuals_igarch(; data::String, column::Int=1, p::Int=1, q::Int=1, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_igarch(y, p, q); catch e; throw(_garch_variant_error(e, "IGARCH")); end) : model
    _status("IGARCH standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "IGARCH($p,$q)" * " Standardized Residuals"; format=format, output=output,
        key="igarch_standardized_residuals")
end

function _forecast_cgarch(; data::String="", result=nothing, column::Int=1, horizons::Int=10,
        conf_level::Float64=0.95, model=nothing, output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, leaf="forecast cgarch")
    if loaded !== nothing
        h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
        _vol_forecast_output(loaded, "result", "Component-GARCH(1,1)", h; format=format, output=output,
                             key="cgarch_volatility_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "forecast cgarch: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < conf_level < 1.0) || throw(CliError("usage/invalid",
        "forecast cgarch: --conf-level must be in (0, 1) (got $conf_level)"))
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_cgarch(y); catch e; throw(_garch_variant_error(e, "Component-GARCH")); end) : model
    _status("Component-GARCH Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=conf_level)
    catch e
        throw(_garch_variant_error(e, "Component-GARCH forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _vol_forecast_output(fc, vname, "Component-GARCH(1,1)", horizons; format=format, output=output,
                         key="cgarch_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_cgarch(; data::String, column::Int=1, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_cgarch(y); catch e; throw(_garch_variant_error(e, "Component-GARCH")); end) : model
    _status("Component-GARCH conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "Component-GARCH(1,1)" * " Conditional Variance"; format=format, output=output,
        key="cgarch_conditional_variance")
end

function _residuals_cgarch(; data::String, column::Int=1, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_cgarch(y); catch e; throw(_garch_variant_error(e, "Component-GARCH")); end) : model
    _status("Component-GARCH standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "Component-GARCH(1,1)" * " Standardized Residuals"; format=format, output=output,
        key="cgarch_standardized_residuals")
end

function _forecast_aparch(; data::String="", result=nothing, column::Int=1, p::Int=1, q::Int=1, fix_delta=nothing, fix_gamma=nothing, horizons::Int=10,
        conf_level::Float64=0.95, model=nothing, output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, leaf="forecast aparch")
    if loaded !== nothing
        h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
        _vol_forecast_output(loaded, "result", "APARCH($p,$q)", h; format=format, output=output,
                             key="aparch_volatility_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "forecast aparch: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < conf_level < 1.0) || throw(CliError("usage/invalid",
        "forecast aparch: --conf-level must be in (0, 1) (got $conf_level)"))
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_aparch(y, p, q; fix_delta=(fix_delta === nothing ? nothing : Float64(fix_delta)), fix_gamma=(fix_gamma === nothing ? nothing : Float64(fix_gamma))); catch e; throw(_garch_variant_error(e, "APARCH")); end) : model
    _status("APARCH Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=conf_level)
    catch e
        throw(_garch_variant_error(e, "APARCH forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _vol_forecast_output(fc, vname, "APARCH($p,$q)", horizons; format=format, output=output,
                         key="aparch_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_aparch(; data::String, column::Int=1, p::Int=1, q::Int=1, fix_delta=nothing, fix_gamma=nothing, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_aparch(y, p, q; fix_delta=(fix_delta === nothing ? nothing : Float64(fix_delta)), fix_gamma=(fix_gamma === nothing ? nothing : Float64(fix_gamma))); catch e; throw(_garch_variant_error(e, "APARCH")); end) : model
    _status("APARCH conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "APARCH($p,$q)" * " Conditional Variance"; format=format, output=output,
        key="aparch_conditional_variance")
end

function _residuals_aparch(; data::String, column::Int=1, p::Int=1, q::Int=1, fix_delta=nothing, fix_gamma=nothing, model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_aparch(y, p, q; fix_delta=(fix_delta === nothing ? nothing : Float64(fix_delta)), fix_gamma=(fix_gamma === nothing ? nothing : Float64(fix_gamma))); catch e; throw(_garch_variant_error(e, "APARCH")); end) : model
    _status("APARCH standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "APARCH($p,$q)" * " Standardized Residuals"; format=format, output=output,
        key="aparch_standardized_residuals")
end

function _forecast_figarch(; data::String="", result=nothing, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", horizons::Int=10,
        conf_level::Float64=0.95, model=nothing, output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, leaf="forecast figarch")
    if loaded !== nothing
        h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
        _vol_forecast_output(loaded, "result", "FIGARCH($p,d,$q)", h; format=format, output=output,
                             key="figarch_volatility_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "forecast figarch: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < conf_level < 1.0) || throw(CliError("usage/invalid",
        "forecast figarch: --conf-level must be in (0, 1) (got $conf_level)"))
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_figarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIGARCH")); end) : model
    _status("FIGARCH Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=conf_level)
    catch e
        throw(_garch_variant_error(e, "FIGARCH forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _vol_forecast_output(fc, vname, "FIGARCH($p,d,$q)", horizons; format=format, output=output,
                         key="figarch_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_figarch(; data::String, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_figarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIGARCH")); end) : model
    _status("FIGARCH conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "FIGARCH($p,d,$q)" * " Conditional Variance"; format=format, output=output,
        key="figarch_conditional_variance")
end

function _residuals_figarch(; data::String, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_figarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIGARCH")); end) : model
    _status("FIGARCH standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "FIGARCH($p,d,$q)" * " Standardized Residuals"; format=format, output=output,
        key="figarch_standardized_residuals")
end

function _forecast_fiegarch(; data::String="", result=nothing, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", horizons::Int=10,
        conf_level::Float64=0.95, model=nothing, output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, leaf="forecast fiegarch")
    if loaded !== nothing
        h = hasproperty(loaded, :horizon) ? Int(loaded.horizon) : horizons
        _vol_forecast_output(loaded, "result", "FIEGARCH($p,d,$q)", h; format=format, output=output,
                             key="fiegarch_volatility_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "forecast fiegarch: --horizons must be ≥ 1 (got $horizons)"))
    (0.0 < conf_level < 1.0) || throw(CliError("usage/invalid",
        "forecast fiegarch: --conf-level must be in (0, 1) (got $conf_level)"))
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_fiegarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIEGARCH")); end) : model
    _status("FIEGARCH Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons; conf_level=conf_level)
    catch e
        throw(_garch_variant_error(e, "FIEGARCH forecast"))
    end
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _vol_forecast_output(fc, vname, "FIEGARCH($p,d,$q)", horizons; format=format, output=output,
                         key="fiegarch_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_fiegarch(; data::String, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_fiegarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIEGARCH")); end) : model
    _status("FIEGARCH conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "FIEGARCH($p,d,$q)" * " Conditional Variance"; format=format, output=output,
        key="fiegarch_conditional_variance")
end

function _residuals_fiegarch(; data::String, column::Int=1, p::Int=1, q::Int=1, d0::Float64=0.4, truncation::Int=1000, dist::String="normal", model=nothing, output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    m = model === nothing ?
        (try; estimate_fiegarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist)); catch e; throw(_garch_variant_error(e, "FIEGARCH")); end) : model
    _status("FIEGARCH standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "FIEGARCH($p,d,$q)" * " Standardized Residuals"; format=format, output=output,
        key="fiegarch_standardized_residuals")
end

# garch-midas is the odd one out: forecast(m::GarchMidasModel, h) takes NO conf_level,
# so this leaf deliberately has no --conf-level (advertising one would be a lie).
function _garch_midas_refit(data, column, m_freq, k, rv, span, config, model=nothing)
    # Mirrors `_estimate_garch_midas`'s validation and fit exactly, so the verbs
    # describe the same model that leaf would produce.
    m_freq >= 1 || throw(CliError("usage/missing-option",
        "--m-freq ≥ 1 is required (high-frequency observations per low-frequency block)"))
    rv_sym = Symbol(rv); span_sym = Symbol(span)
    rv_sym in (:realized, :macro) || throw(CliError("usage/bad-value", "--rv must be realized|macro, got $rv"))
    span_sym in (:fixed, :rolling) || throw(CliError("usage/bad-value", "--span must be fixed|rolling, got $span"))
    y, vname = load_univariate_series(data, column)
    x_lf = Float64[]
    if rv_sym === :macro
        isempty(config) && throw(CliError("config/missing",
            "--rv macro requires --config <toml> with a [garch_midas] x_lf = [...] low-frequency driver"))
        x_lf = get_garch_midas(load_config(config))
    end
    m = model === nothing ?
        (try
            estimate_garch_midas(y, x_lf; K=k, m_freq=m_freq, rv=rv_sym, span=span_sym)
        catch e
            throw(_garch_variant_error(e, "GARCH-MIDAS"))
        end) : model
    return m, vname
end

function _forecast_garch_midas(; data::String="", result=nothing, column::Int=1, m_freq::Int=0, k::Int=12,
        rv::String="realized", span::String="fixed", config::String="", horizons::Int=10,
        model=nothing, output::String="", format::String="table")
    loaded = _loaded_result(result; data, model, leaf="forecast garch-midas")
    loaded === nothing || return loaded
    horizons >= 1 || throw(CliError("usage/invalid",
        "forecast garch-midas: --horizons must be ≥ 1 (got $horizons)"))
    m, vname = _garch_midas_refit(data, column, m_freq, k, rv, span, config, model)
    _status("GARCH-MIDAS Volatility Forecast: variable=$vname, horizons=$horizons"); _status()
    fc = try
        forecast(m, horizons)
    catch e
        throw(_garch_variant_error(e, "GARCH-MIDAS forecast"))
    end
    # forecast(::GarchMidasModel, h) returns a NamedTuple
    # (total, long_run, short_run, horizon) — NOT a VolatilityForecast — so it needs its
    # own renderer. The long-run/short-run split IS the point of GARCH-MIDAS, so surface
    # all three rather than flattening to a single path.
    tot = Float64.(collect(fc.total))
    output_result(DataFrame(
            horizon = collect(1:length(tot)),
            total_variance = round.(tot; digits=6),
            long_run = round.(Float64.(collect(fc.long_run)); digits=6),
            short_run = round.(Float64.(collect(fc.short_run)); digits=6),
            volatility = round.(sqrt.(abs.(tot)); digits=6));
        format=Symbol(format), output=output,
        title="GARCH-MIDAS Volatility Forecast ($vname)", key="garch_midas_volatility_forecast")
    return (; model=m, result=fc)
end

function _predict_garch_midas(; data::String, column::Int=1, m_freq::Int=0, k::Int=12,
        rv::String="realized", span::String="fixed", config::String="",
        model=nothing, output::String="", format::String="table")
    m, vname = _garch_midas_refit(data, column, m_freq, k, rv, span, config, model)
    _status("GARCH-MIDAS conditional variance: variable=$vname"); _status()
    return _vol_variant_predict_output(_vol_variant_cond_var(m), vname,
        "GARCH-MIDAS Conditional Variance"; format=format, output=output,
        key="garch_midas_conditional_variance")
end

function _residuals_garch_midas(; data::String, column::Int=1, m_freq::Int=0, k::Int=12,
        rv::String="realized", span::String="fixed", config::String="",
        model=nothing, output::String="", format::String="table")
    m, vname = _garch_midas_refit(data, column, m_freq, k, rv, span, config, model)
    _status("GARCH-MIDAS standardized residuals: variable=$vname"); _status()
    return _vol_variant_residuals_output(residuals(m), vname,
        "GARCH-MIDAS Standardized Residuals"; format=format, output=output,
        key="garch_midas_standardized_residuals")
end

function _estimate_igarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                           output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    _status("Estimating IGARCH($p,$q): variable=$vname, observations=$(length(y))")
    _status()
    model = try
        estimate_igarch(y, p, q)
    catch e
        throw(_garch_variant_error(e, "IGARCH"))
    end
    names = String["mu"; "omega"; ["alpha$i" for i in 1:q]; ["beta$i" for i in 1:p]]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="IGARCH($p,$q) Coefficients ($vname)",
                  key="igarch_coefficients")
    output_kv(_garch_variant_diag(model); format=format, title="IGARCH($p,$q) Diagnostics",
              key="igarch_diagnostics")
    return model
end

function _estimate_cgarch(; data::String, column::Int=1,
                           output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    _status("Estimating Component-GARCH(1,1): variable=$vname, observations=$(length(y))")
    _status()
    model = try
        estimate_cgarch(y)
    catch e
        throw(_garch_variant_error(e, "CGARCH"))
    end
    names = String["mu", "omega", "rho", "phi", "alpha", "beta"]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="Component-GARCH(1,1) Coefficients ($vname)",
                  key="cgarch_coefficients")
    output_kv(_garch_variant_diag(model; extra=Pair{String,Any}[
        "transitory_persistence" => round(Float64(model.alpha + model.beta); digits=6),
        "unconditional_variance" => round(Float64(unconditional_variance(model)); digits=6),
    ]); format=format, title="Component-GARCH Diagnostics", key="cgarch_diagnostics")
    return model
end

function _estimate_aparch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                           fix_delta=nothing, fix_gamma=nothing,
                           output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    fd = fix_delta === nothing ? nothing : Float64(fix_delta)
    fg = fix_gamma === nothing ? nothing : Float64(fix_gamma)
    _status("Estimating APARCH($p,$q): variable=$vname, observations=$(length(y))" *
            (fd === nothing ? "" : ", fix_delta=$fd") * (fg === nothing ? "" : ", fix_gamma=$fg"))
    _status()
    model = try
        estimate_aparch(y, p, q; fix_delta=fd, fix_gamma=fg)
    catch e
        throw(_garch_variant_error(e, "APARCH"))
    end
    names = String["mu"; "omega"; ["alpha$i" for i in 1:q]; ["gamma$i" for i in 1:q]; ["beta$i" for i in 1:p]; "delta"]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="APARCH($p,$q) Coefficients ($vname)",
                  key="aparch_coefficients")
    output_kv(_garch_variant_diag(model; extra=Pair{String,Any}[
        "delta" => round(Float64(model.delta); digits=6),
        "n_params" => model.n_params,
    ]); format=format, title="APARCH Diagnostics", key="aparch_diagnostics")
    return model
end

function _estimate_figarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                            d0::Float64=0.4, truncation::Int=1000, dist::String="normal",
                            output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    _status("Estimating FIGARCH($p,d,$q): variable=$vname, observations=$(length(y)), d0=$d0, truncation=$truncation")
    _status()
    model = try
        estimate_figarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist))
    catch e
        throw(_garch_variant_error(e, "FIGARCH"))
    end
    names = String["mu"; "omega"; ["phi$i" for i in 1:q]; ["beta$i" for i in 1:p]; "d"]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="FIGARCH($p,d,$q) Coefficients ($vname)",
                  key="figarch_coefficients")
    output_kv(_garch_variant_diag(model; extra=Pair{String,Any}[
        "d" => round(Float64(model.d); digits=6),
        "truncation" => model.truncation,
        "n_neg_lambda" => model.n_neg_lambda,
    ]); format=format, title="FIGARCH Diagnostics", key="figarch_diagnostics")
    return model
end

function _estimate_fiegarch(; data::String, column::Int=1, p::Int=1, q::Int=1,
                             d0::Float64=0.4, truncation::Int=1000, dist::String="normal",
                             output::String="", format::String="table")
    y, vname = load_univariate_series(data, column)
    _status("Estimating FIEGARCH($p,d,$q): variable=$vname, observations=$(length(y)), d0=$d0, truncation=$truncation")
    _status()
    model = try
        estimate_fiegarch(y; p=p, q=q, d0=d0, truncation=truncation, dist=Symbol(dist))
    catch e
        throw(_garch_variant_error(e, "FIEGARCH"))
    end
    names = String["mu"; "omega"; "theta"; "gamma"; ["phi$i" for i in 1:q]; ["beta$i" for i in 1:p]; "d"]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="FIEGARCH($p,d,$q) Coefficients ($vname)",
                  key="fiegarch_coefficients")
    output_kv(_garch_variant_diag(model; extra=Pair{String,Any}[
        "d" => round(Float64(model.d); digits=6),
        "truncation" => model.truncation,
    ]); format=format, title="FIEGARCH Diagnostics", key="fiegarch_diagnostics")
    return model
end

function _estimate_garch_midas(; data::String, column::Int=1, m_freq::Int=0, k::Int=12,
                                rv::String="realized", span::String="fixed", config::String="",
                                output::String="", format::String="table")
    m_freq >= 1 || throw(CliError("usage/missing-option",
        "estimate garch-midas requires --m-freq ≥ 1 (high-frequency observations per low-frequency block)"))
    rv_sym = Symbol(rv); span_sym = Symbol(span)
    rv_sym in (:realized, :macro) || throw(CliError("usage/bad-value", "--rv must be realized|macro, got $rv"))
    span_sym in (:fixed, :rolling) || throw(CliError("usage/bad-value", "--span must be fixed|rolling, got $span"))
    y, vname = load_univariate_series(data, column)
    x_lf = Float64[]
    if rv_sym === :macro
        isempty(config) && throw(CliError("config/missing",
            "--rv macro requires --config <toml> with a [garch_midas] x_lf = [...] low-frequency driver"))
        x_lf = get_garch_midas(load_config(config))
    end
    _status("Estimating GARCH-MIDAS: variable=$vname, observations=$(length(y)), K=$k, m_freq=$m_freq, rv=$rv, span=$span")
    _status()
    model = try
        estimate_garch_midas(y, x_lf; K=k, m_freq=m_freq, rv=rv_sym, span=span_sym)
    catch e
        throw(_garch_variant_error(e, "GARCH-MIDAS"))
    end
    names = String["mu", "alpha", "beta", "m", "theta", "w"]
    output_result(_garch_variant_coef_table(model, names); format=Symbol(format),
                  output=output, title="GARCH-MIDAS Coefficients ($vname)",
                  key="garch_midas_coefficients")
    output_kv(_garch_variant_diag(model; extra=Pair{String,Any}[
        "variance_ratio" => round(Float64(model.variance_ratio); digits=6),
        "K" => model.K, "m_freq" => model.m_freq, "n_blocks" => model.n_blocks,
        "rv" => String(model.rv), "span" => String(model.span),
    ]); format=format, title="GARCH-MIDAS Diagnostics", key="garch_midas_diagnostics")
    return model
end

# ── C064b: multivariate GARCH (MGARCH) ───────────────────────
# ccc / dcc (cDCC) / bekk (MEMs 0.7.0 src/mgarch). `MGARCHModel` is not registered in
# MEMs `_COEF_TABLE_TYPES`; the second-stage dynamics table is hand-built via the shared
# `_garch_variant_coef_table`, and the conditional-correlation matrix renders WIDE
# (sector×sector) — the documented C051 exception, same as the io family.

"""Wide sector×sector correlation DataFrame with a leading `series` label column.
`makeunique=true` guards the case where an input column is literally named `series`
(the label would otherwise collide → uncaught `ArgumentError` → exit-1) and any duplicate
input headers; the colliding label is renamed (`series_1`) rather than crashing."""
_mgarch_corr_df(R::AbstractMatrix, names::Vector{String}) =
    insertcols!(DataFrame(round.(R; digits=6), names; makeunique=true), 1,
                :series => names; makeunique=true)

"""Shared renderer for an MGARCH fit: headline conditional-correlation matrix (wide),
second-stage dynamics coefficients (skipped for CCC, which has none), and a diagnostics
kv block. Follows the repo convention of routing `--output` to the headline table only
(same as the C064a / mlogit handlers) so a `-o file` export is not overwritten by the
subsequent blocks."""
function _mgarch_output(model, varnames::Vector{String}, label::String;
                        format::String="table", output::String="", key_prefix::String="")
    R = correlations(model)[:, :, end]   # constant matrix for CCC/BEKK; last DCC slice
    n = size(R, 1)
    names = length(varnames) >= n ? varnames[1:n] : String["series_$i" for i in 1:n]
    # `label` carries the option-dependent variant (cDCC vs DCC, scalar vs diagonal BEKK),
    # so it stays in the title while the envelope key comes from the leaf's fixed prefix.
    output_result(_mgarch_corr_df(R, names); format=Symbol(format), output=output,
                  title="$label Conditional Correlation",
                  key="$(key_prefix)_conditional_correlation")
    if !isempty(coef(model))
        output_result(_garch_variant_coef_table(model, model.param_names);
                      format=Symbol(format), title="$label Dynamics Parameters",
                      key="$(key_prefix)_dynamics_parameters")
    end
    pairs = Pair{String,Any}[
        "loglik"       => round(Float64(loglikelihood(model)); digits=4),
        "aic"          => round(Float64(model.aic); digits=4),
        "bic"          => round(Float64(model.bic); digits=4),
        "series"       => model.n,
        "observations" => size(model.Y, 1),
        "converged"    => model.converged,
        "kind"         => string(model.kind),
    ]
    if model.kind === :dcc
        push!(pairs, "correction" => string(model.correction))
        push!(pairs, "persistence" => round(Float64(sum(coef(model))); digits=6))
    elseif model.kind === :bekk
        push!(pairs, "bekk_kind" => string(model.bekk_kind))
    end
    output_kv(pairs; format=format, title="$label Diagnostics", key="$(key_prefix)_diagnostics")
    return nothing
end

function _estimate_ccc(; data::String, p::Int=1, q::Int=1,
                        output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    _status("Estimating CCC-GARCH: series=$(size(Y,2)), observations=$(size(Y,1)), margins=GARCH($p,$q)")
    _status()
    model = try
        estimate_ccc(Y; p=p, q=q)
    catch e
        throw(_garch_variant_error(e, "CCC-GARCH"))
    end
    _mgarch_output(model, varnames, "CCC-GARCH"; format=format, output=output,
                   key_prefix="ccc_garch")
    return model
end

function _estimate_dcc(; data::String, p::Int=1, q::Int=1, correction::String="none",
                        output::String="", format::String="table")
    correction in ("none", "aielli") || throw(CliError("usage/invalid",
        "estimate dcc: --correction must be none|aielli, got $correction"))
    Y, varnames = load_multivariate_data(data)
    _status("Estimating DCC-GARCH: series=$(size(Y,2)), observations=$(size(Y,1)), margins=GARCH($p,$q), correction=$correction")
    _status()
    model = try
        estimate_dcc(Y; p=p, q=q, correction=Symbol(correction))
    catch e
        throw(_garch_variant_error(e, "DCC-GARCH"))
    end
    label = correction == "aielli" ? "cDCC-GARCH" : "DCC-GARCH"
    _mgarch_output(model, varnames, label; format=format, output=output,
                   key_prefix="dcc_garch")
    return model
end

function _estimate_bekk(; data::String, kind::String="scalar",
                         output::String="", format::String="table")
    kind in ("scalar", "diagonal") || throw(CliError("usage/invalid",
        "estimate bekk: --kind must be scalar|diagonal, got $kind"))
    Y, varnames = load_multivariate_data(data)
    _status("Estimating BEKK-$kind GARCH: series=$(size(Y,2)), observations=$(size(Y,1))")
    _status()
    model = try
        estimate_bekk(Y; kind=Symbol(kind))
    catch e
        throw(_garch_variant_error(e, "BEKK-GARCH"))
    end
    _mgarch_output(model, varnames, "BEKK-$kind"; format=format, output=output,
                   key_prefix="bekk")
    return model
end

# ── C067a: penalized & limited-dependent cross-section regression ─────────────
# lasso / ridge / elastic-net (→ PenalizedRegModel), robust (→ RobustRegModel),
# tobit (→ TobitModel), MEMs 0.7.0 src/reg/{penalized,robust,tobit}. All take (y, X)
# via the shared `_load_reg_data` loader (X = numeric columns except --dep; no constant
# prepended, mirroring `estimate reg`). None of the three result types is registered in
# MEMs `_COEF_TABLE_TYPES`, so their coefficient tables are HAND-BUILT (the documented
# C051 exception, same as the systems/volatility waves).

"""Hand-built coefficient table for a penalized (Lasso/Ridge/Elastic-Net) fit:
`term | estimate | nonzero`, with the intercept (`beta0`) prepended. Penalized fits
carry no standard errors (`stderror(::PenalizedRegModel)` is undefined upstream), so this
is estimate-only — the C051-documented hand-built exception."""
function _penalized_coef_table(model, xcols::Vector{String})
    names = String["(Intercept)"; xcols[1:length(model.beta)]]
    est = Float64[model.beta0; model.beta]
    DataFrame(term=names, estimate=round.(est; digits=6), nonzero=(est .!= 0.0))
end

"""Diagnostics kv for a penalized fit: mixing `alpha`, selected `lambda`, active-set
size, fit `r2`, information criteria, and the selection rule."""
function _penalized_diag(model)
    Pair{String,Any}[
        "alpha"    => round(Float64(model.alpha); digits=4),
        "lambda"   => round(Float64(model.lambda); digits=6),
        "n_active" => count(!=(0.0), model.beta),
        "r2"       => round(Float64(model.r2); digits=6),
        "aic"      => round(Float64(model.aic); digits=4),
        "bic"      => round(Float64(model.bic); digits=4),
        "ebic"     => round(Float64(model.ebic); digits=4),
        "select"   => string(model.select),
    ]
end

"""Parse a penalized-regression `--lambda`: `auto`/`cv` → `:cv` (auto-select a path);
otherwise a non-negative number. Junk or a negative value → a typed usage error (never a
raw MEMs `ArgumentError`)."""
function _parse_penalty_lambda(s::AbstractString)
    (s == "auto" || s == "cv") && return :cv
    v = tryparse(Float64, s)
    v === nothing && throw(CliError("usage/invalid",
        "--lambda must be `auto` or a non-negative number, got '$s'"))
    v < 0 && throw(CliError("usage/invalid",
        "--lambda must be non-negative, got $v"))
    return v
end

function _estimate_lasso(; data::String, dep::String="", lambda::String="auto",
                          select::String="cv", output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    lam = _parse_penalty_lambda(lambda)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("LASSO (L1) regression: $dep_name ~ $(join(xcols, " + ")), n=$(length(y)), lambda=$lambda")
    _status()
    model = try
        estimate_lasso(y, X; lambda=lam, select=Symbol(select), varnames=xcols)
    catch e
        throw(_garch_variant_error(e, "LASSO"))
    end
    output_result(_penalized_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="LASSO Coefficients")
    output_kv(_penalized_diag(model); format=format, title="LASSO Diagnostics")
    return model
end

function _estimate_ridge(; data::String, dep::String="", lambda::String="auto",
                          select::String="cv", output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    lam = _parse_penalty_lambda(lambda)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Ridge (L2) regression: $dep_name ~ $(join(xcols, " + ")), n=$(length(y)), lambda=$lambda")
    _status()
    model = try
        estimate_ridge(y, X; lambda=lam, select=Symbol(select), varnames=xcols)
    catch e
        throw(_garch_variant_error(e, "Ridge"))
    end
    output_result(_penalized_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="Ridge Coefficients")
    output_kv(_penalized_diag(model); format=format, title="Ridge Diagnostics")
    return model
end

function _estimate_elastic_net(; data::String, dep::String="", alpha::Float64=0.5,
                                lambda::String="auto", select::String="cv",
                                output::String="", format::String="table")
    (0.0 <= alpha <= 1.0) || throw(CliError("usage/invalid",
        "estimate elastic-net: --alpha must be in [0,1], got $alpha"))
    y, X, xcols = _load_reg_data(data, dep)
    lam = _parse_penalty_lambda(lambda)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Elastic-Net regression (alpha=$alpha): $dep_name ~ $(join(xcols, " + ")), n=$(length(y)), lambda=$lambda")
    _status()
    model = try
        estimate_elastic_net(y, X; alpha=alpha, lambda=lam, select=Symbol(select), varnames=xcols)
    catch e
        throw(_garch_variant_error(e, "Elastic-Net"))
    end
    output_result(_penalized_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="Elastic-Net Coefficients")
    output_kv(_penalized_diag(model); format=format, title="Elastic-Net Diagnostics")
    return model
end

function _estimate_robust(; data::String, dep::String="", psi::String="huber",
                           method::String="m", output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Robust regression ($psi $method-estimator): $dep_name ~ $(join(xcols, " + ")), n=$(length(y))")
    _status()
    model = try
        estimate_robust(y, X; psi=Symbol(psi), method=Symbol(method), _fwd_seed()...)
    catch e
        throw(_garch_variant_error(e, "Robust regression"))
    end
    output_result(_garch_variant_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="Robust Regression Coefficients")
    output_kv(Pair{String,Any}[
        "psi"        => string(model.psi),
        "method"     => string(model.method),
        "scale"      => round(Float64(model.scale); digits=6),
        "robust_r2"  => round(Float64(model.robust_r2); digits=6),
        "tuning"     => round(Float64(model.tuning); digits=6),
        "converged"  => model.converged,
        "iterations" => model.iterations,
    ]; format=format, title="Robust Regression Diagnostics")
    return model
end

function _estimate_tobit(; data::String, dep::String="", lower::Float64=0.0,
                          upper::Float64=Inf, output::String="", format::String="table")
    lower < upper || throw(CliError("usage/invalid",
        "estimate tobit: --lower ($lower) must be < --upper ($upper)"))
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Tobit regression (lower=$lower, upper=$upper): $dep_name ~ $(join(xcols, " + ")), n=$(length(y))")
    _status()
    model = try
        estimate_tobit(y, X; lower=lower, upper=upper)
    catch e
        throw(_garch_variant_error(e, "Tobit regression"))
    end
    output_result(_garch_variant_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="Tobit Coefficients")
    output_kv(Pair{String,Any}[
        "sigma"            => round(Float64(model.sigma); digits=6),
        "loglik"           => round(Float64(model.loglik); digits=4),
        "aic"              => round(Float64(model.aic); digits=4),
        "bic"              => round(Float64(model.bic); digits=4),
        # Render ±Inf bounds as strings: a raw Inf/-Inf Float crashes the legacy
        # (FRIEDMAN_LEGACY_OUTPUT) JSON writer, which — unlike the envelope path — does not
        # apply `_json_safe` ("Inf not allowed in JSON spec"). Matches the envelope output.
        "lower"            => isfinite(model.lower) ? Float64(model.lower) : string(model.lower),
        "upper"            => isfinite(model.upper) ? Float64(model.upper) : string(model.upper),
        "n_censored_left"  => model.n_censored_left,
        "n_censored_right" => model.n_censored_right,
        "converged"        => model.converged,
    ]; format=format, title="Tobit Diagnostics")
    return model
end

# ── C067b: truncated-normal regression (Hausman-Wise) ──────────
# TruncRegModel shares TobitModel's `coef`/`stderror` (StatsAPI) so the coefficient table
# reuses `_garch_variant_coef_table`. It is NOT in MEMs `_COEF_TABLE_TYPES` (same documented
# C051 hand-built exception as Tobit/robust). Bad input (any y outside (lower,upper), or a
# missing cell) is mapped to a typed data error via `_garch_variant_error`/`_load_reg_data`.
function _estimate_truncreg(; data::String, dep::String="", lower::Float64=0.0,
                             upper::Float64=Inf, output::String="", format::String="table")
    lower < upper || throw(CliError("usage/invalid",
        "estimate truncreg: --lower ($lower) must be < --upper ($upper)"))
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Truncated regression (lower=$lower, upper=$upper): $dep_name ~ $(join(xcols, " + ")), n=$(length(y))")
    _status()
    model = try
        estimate_truncreg(y, X; lower=lower, upper=upper)
    catch e
        throw(_garch_variant_error(e, "Truncated regression"))
    end
    output_result(_garch_variant_coef_table(model, xcols); format=Symbol(format), output=output,
                  title="Truncated Regression Coefficients")
    output_kv(Pair{String,Any}[
        "sigma"        => round(Float64(model.sigma); digits=6),
        "sigma_se"     => round(Float64(model.sigma_se); digits=6),
        "loglik"       => round(Float64(model.loglik); digits=4),
        "aic"          => round(Float64(model.aic); digits=4),
        "bic"          => round(Float64(model.bic); digits=4),
        # ±Inf bounds → strings (legacy JSON writer rejects non-finite floats; see Tobit).
        "lower"        => isfinite(model.lower) ? Float64(model.lower) : string(model.lower),
        "upper"        => isfinite(model.upper) ? Float64(model.upper) : string(model.upper),
        "n_truncated"  => model.n_truncated,
        "converged"    => model.converged,
    ]; format=format, title="Truncated Regression Diagnostics")
    return model
end

# ── C067b: Heckman sample-selection model (two-step / FIML) ─────
# HeckmanModel is a TWO-equation result (outcome β + selection γ) → the coefficient table is
# hand-built as one tidy `equation|term|estimate|std_error|z_stat|p_value` frame (C051 tidy
# convention: the `equation` column carries outcome/selection, like VAR/panel prepend one).
# SEs come from the stored vcov blocks; z/p are normal-approx (both estimators are asymptotic).
function _heckman_coef_table(model)
    eqc = String[]; term = String[]; est = Float64[]; sec = Float64[]
    _append! = (eqname, names, beta, vcov) -> begin
        se = sqrt.(max.(diag(vcov), 0.0))
        for i in eachindex(beta)
            push!(eqc, eqname); push!(term, names[i])
            push!(est, Float64(beta[i])); push!(sec, Float64(se[i]))
        end
    end
    _append!("outcome",   model.outcome_names, model.beta,  model.vcov_beta)
    _append!("selection", model.select_names,  model.gamma, model.vcov_gamma)
    z  = [sec[i] == 0.0 ? 0.0 : est[i] / sec[i] for i in eachindex(est)]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    DataFrame(equation=eqc, term=term, estimate=round.(est; digits=6),
              std_error=round.(sec; digits=6), z_stat=round.(z; digits=4),
              p_value=round.(pv; digits=4))
end

function _estimate_heckman(; data::String, dep::String="", select::String="",
                            outcome_vars::String="", select_vars::String="",
                            method::String="twostep", output::String="", format::String="table")
    method in ("twostep", "mle") || throw(CliError("usage/invalid",
        "estimate heckman: --method must be twostep or mle, got '$method'"))
    isempty(select) && throw(CliError("usage/missing",
        "--select is required (the binary 0/1 selection-indicator column)"))
    isempty(outcome_vars) && throw(CliError("usage/missing",
        "--outcome-vars is required (comma-separated outcome-equation regressor columns)"))
    isempty(select_vars) && throw(CliError("usage/missing",
        "--select-vars is required (comma-separated selection-equation regressor columns)"))

    df = load_data(data)
    numcols = variable_names(df)
    dep_col = isempty(dep) ? numcols[1] : dep
    !isempty(dep) && !(dep_col in numcols) && throw(CliError("data/column-range",
        "outcome variable '$dep_col' not found in numeric columns: $(join(numcols, ", "))"))
    select in numcols || throw(CliError("data/column-range",
        "selection indicator '$select' not found in numeric columns: $(join(numcols, ", "))"))

    ovars = String[strip(s) for s in split(outcome_vars, ",") if !isempty(strip(s))]
    svars = String[strip(s) for s in split(select_vars, ",") if !isempty(strip(s))]
    isempty(ovars) && throw(CliError("usage/missing", "--outcome-vars resolved to no columns"))
    isempty(svars) && throw(CliError("usage/missing", "--select-vars resolved to no columns"))
    for nm in ovars
        nm in numcols || throw(CliError("data/column-range",
            "outcome regressor '$nm' not found in numeric columns: $(join(numcols, ", "))"))
    end
    for nm in svars
        nm in numcols || throw(CliError("data/column-range",
            "selection regressor '$nm' not found in numeric columns: $(join(numcols, ", "))"))
    end
    # Guard missing cells in the selection indicator and BOTH regressor sets (needed for every
    # row) BEFORE the Matrix{Float64} conversion (untyped ArgumentError → exit-1). The OUTCOME
    # column is handled separately below: it is unobserved by design for non-selected rows.
    for c in unique(vcat([select], ovars, svars))
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them first"))
    end

    d = Vector{Float64}(df[!, select])
    # Binary-indicator guard up front → typed usage error (MEMs also checks, but this gives a
    # clearer message than the wrapped ArgumentError).
    all(v -> v == 0.0 || v == 1.0, d) || throw(CliError("data/invalid",
        "selection indicator '$select' must be binary 0/1"))
    sel = findall(==(1.0), d)

    # The outcome is observed ONLY for selected (d==1) rows — MEMs uses `y[sel]` and only
    # requires those to be finite (heckman.jl). So a blank/missing outcome for NON-selected
    # rows is the canonical incidental-truncation layout, not an error: require the outcome
    # present for selected rows, and fill unselected cells with a finite placeholder (0.0,
    # ignored downstream). This makes real selection CSVs usable without pre-imputing.
    yraw = df[!, dep_col]
    any(ismissing, yraw[sel]) && throw(CliError("data/missing-values",
        "outcome '$dep_col' has missing values among SELECTED ($select==1) rows; the outcome must be observed wherever the unit is selected"))
    y = Float64[ismissing(v) ? 0.0 : Float64(v) for v in yraw]
    X = Matrix{Float64}(df[!, ovars])
    Z = Matrix{Float64}(df[!, svars])

    n_sel = length(sel)
    _status("Heckman selection ($method): $dep_col ~ $(join(ovars, " + ")) | select $select ~ $(join(svars, " + "))")
    _status("  Observations: $(length(y)) ($n_sel selected), exclusion vars in selection: $(join(setdiff(svars, ovars), ", "))")
    _status()

    model = try
        estimate_heckman(y, X, d, Z; method=Symbol(method),
                         outcome_names=ovars, select_names=svars)
    catch e
        throw(_garch_variant_error(e, "Heckman selection model"))
    end

    output_result(_heckman_coef_table(model); format=Symbol(format), output=output,
                  title="Heckman Coefficients (outcome + selection)")
    output_kv(Pair{String,Any}[
        "method"      => string(model.method),
        "rho"         => round(Float64(model.rho); digits=6),
        "rho_se"      => isfinite(model.rho_se) ? round(Float64(model.rho_se); digits=6) : string(model.rho_se),
        "sigma"       => round(Float64(model.sigma); digits=6),
        "sigma_se"    => isfinite(model.sigma_se) ? round(Float64(model.sigma_se); digits=6) : string(model.sigma_se),
        "lambda"      => round(Float64(model.lambda); digits=6),
        "lambda_se"   => isfinite(model.lambda_se) ? round(Float64(model.lambda_se); digits=6) : string(model.lambda_se),
        "loglik"      => round(Float64(model.loglik); digits=4),
        "aic"         => round(Float64(model.aic); digits=4),
        "bic"         => round(Float64(model.bic); digits=4),
        "n_selected"  => model.n_selected,
        "n_total"     => model.n_total,
        "converged"   => model.converged,
    ]; format=format, title="Heckman Diagnostics")
    return model
end

# ── C066: state-space + nonparametric estimation (M5c) ─────────
# None of StateSpaceModel/KernelDensity/KernelRegression/LowessFit is registered in MEMs
# `_COEF_TABLE_TYPES`, so all tables are HAND-BUILT (the documented C051 exception, like
# io/mgarch/sur). MEMs calls are wrapped → typed CliError via `_garch_variant_error`.

"""Hand-built parameter table for a fitted `StateSpaceModel`: `parameter | estimate`,
pairing `param_names` with the estimated (natural-scale) hyper-parameters `theta`. No SEs —
theta standard errors are not exposed upstream."""
function _statespace_param_table(model)
    names = String.(model.param_names)
    th = Float64.(model.theta)
    m = min(length(names), length(th))
    DataFrame(parameter=names[1:m], estimate=round.(th[1:m]; digits=6))
end

"""Tidy LONG coefficient-path table for a TVP regression: one row per (period, coefficient)
from the `smoothed_state` (T×k) — `period | coefficient | estimate` — the time-varying βₜ."""
function _tvp_path_table(model, coefnames::Vector{String})
    S = model.smoothed_state
    Tn, k = size(S)
    kk = min(k, length(coefnames))
    period = Int[]; coefficient = String[]; estimate = Float64[]
    for t in 1:Tn, j in 1:kk
        push!(period, t); push!(coefficient, coefnames[j])
        push!(estimate, round(Float64(S[t, j]); digits=6))
    end
    DataFrame(period=period, coefficient=coefficient, estimate=estimate)
end

# estimate statespace — structural univariate state-space (local level / local linear trend)
# ── #71: statespace downstream verbs ─────────────────────────────────────────
# `StateSpaceModel` carries the filtered and smoothed state paths and the one-step
# prediction errors as FIELDS (statespace/types.jl):
#   filtered_state :: T_obs x n_state   (a_t|t)
#   smoothed_state :: T_obs x n_state   (a_t|T)
#   innovations    :: T_obs x n_obs     (v_t, the one-step prediction errors)
#   std_residuals  :: T_obs x n_obs     (v_t / sqrt(F_t))
# There are no StatsAPI predict/residuals methods, so these read the fields.
#
# `predict` renders a tidy LONG table `period | state | filtered | smoothed` rather than
# one column per state — the state count is model-dependent (1 for local-level, 2 for
# local-linear-trend), so a wide table would change shape with --kind. `--state` narrows
# it to one path when only that is wanted. This mirrors `estimate tvp`, which already
# emits its coefficient path long.

"""Refit the state-space model for the downstream verbs, mirroring `_estimate_statespace`."""
function _statespace_refit(data, column, model, init_mode, kappa)
    model in ("local-level", "local-linear-trend") || throw(CliError("usage/invalid",
        "--model must be local-level or local-linear-trend, got '$model'"))
    init_mode in ("kappa", "diffuse") || throw(CliError("usage/invalid",
        "--init-mode must be kappa or diffuse, got '$init_mode'"))
    y, vname = load_univariate_series(data, column)
    im = Symbol(init_mode)
    ssm = try
        model == "local-level" ? local_level(y; init_mode=im, kappa=kappa) :
                                 local_linear_trend(y; init_mode=im, kappa=kappa)
    catch e
        throw(_garch_variant_error(e, "State-space estimation"))
    end
    return ssm, vname
end

function _predict_statespace(; data::String="", column::Int=1, kind::String="local-level",
        init_mode::String="kappa", kappa::Float64=1e6, state::String="both",
        output::String="", format::String="table", model=nothing)
    state in ("filtered", "smoothed", "both") || throw(CliError("usage/invalid",
        "predict statespace: --state must be filtered|smoothed|both, got '$state'"))
    ssm, vname = model === nothing ?
        _statespace_refit(data, column, kind, init_mode, kappa) : (model, "model")
    fs = Float64.(ssm.filtered_state); sm = Float64.(ssm.smoothed_state)
    nT, nS = size(fs)
    periods = Int[]; states = String[]; filt = Float64[]; smoo = Float64[]
    for j in 1:nS, t in 1:nT
        push!(periods, t); push!(states, "state$j")
        push!(filt, fs[t, j]); push!(smoo, sm[t, j])
    end
    _status("State-space $kind states: variable=$vname, T=$nT, states=$nS"); _status()
    df = DataFrame("period" => periods, "state" => states)
    state in ("filtered", "both") && (df[!, "filtered"] = round.(filt; digits=6))
    state in ("smoothed", "both") && (df[!, "smoothed"] = round.(smoo; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="State-Space State Paths ($kind, $vname)", key="state_space_state_paths")
    return ssm
end

function _residuals_statespace(; data::String="", column::Int=1, kind::String="local-level",
        init_mode::String="kappa", kappa::Float64=1e6, standardized::Bool=false,
        output::String="", format::String="table", model=nothing)
    ssm, vname = model === nothing ?
        _statespace_refit(data, column, kind, init_mode, kappa) : (model, "model")
    # innovations are the one-step prediction errors v_t; --standardized gives v_t/sqrt(F_t)
    M = Float64.(standardized ? ssm.std_residuals : ssm.innovations)
    nT, nO = size(M)
    periods = Int[]; series = String[]; vals = Float64[]
    for j in 1:nO, t in 1:nT
        push!(periods, t); push!(series, nO == 1 ? vname : "obs$j"); push!(vals, M[t, j])
    end
    _status("State-space $kind $(standardized ? "standardized " : "")innovations: variable=$vname, T=$nT"); _status()
    output_result(DataFrame("period" => periods, "series" => series,
                            "residual" => round.(vals; digits=6));
        format=Symbol(format), output=output,
        title="State-Space $(standardized ? "Standardized " : "")Innovations ($kind, $vname)",
        key="state_space_innovations")
    return M
end

"""System-matrix summary for the GENERAL `--config` path.

A fixed-matrix system has NO estimated hyper-parameters: `estimate_statespace(ss, y)` returns
`method = :filter` with `theta` and `param_names` EMPTY, so `_statespace_param_table` would
render an empty table and silently look like a failed fit. This reports the system the user
declared instead — dimensions plus the matrices actually in force after MEMs applied its
defaults (R = I, d = c = 0), which is the part a user most often gets wrong."""
function _statespace_system_table(ssm)
    DataFrame(
        "matrix" => String["Z", "H", "T", "Q", "R", "d", "c"],
        "role" => String["observation", "obs. noise cov", "transition", "state noise cov",
                         "state-noise loading", "obs. intercept", "state intercept"],
        "rows" => Int[size(ssm.Z, 1), size(ssm.H, 1), size(ssm.Tt, 1), size(ssm.Q, 1),
                      size(ssm.R, 1), length(ssm.d), length(ssm.c)],
        "cols" => Int[size(ssm.Z, 2), size(ssm.H, 2), size(ssm.Tt, 2), size(ssm.Q, 2),
                      size(ssm.R, 2), 1, 1],
    )
end

function _estimate_statespace(; data::String, column::Int=1, model::String="local-level",
                               init_mode::String="kappa", kappa::Float64=1e6,
                               config::String="", output::String="", format::String="table")
    # ── GENERAL system from [statespace] in --config ────────────────────────────────────
    # This path is MULTIVARIATE (y is T_obs × n_obs), so it loads the whole numeric matrix
    # via the hardened `load_multivariate_data` rather than the canned path's single column.
    if !isempty(config)
        cfg = get_statespace(load_config(config))
        Y, vnames = load_multivariate_data(data)
        size(Y, 2) == cfg.n_obs || throw(CliError("data/shape",
            "estimate statespace: [statespace] Z implies n_obs=$(cfg.n_obs) observed series, " *
            "but $data has $(size(Y, 2)) numeric columns";
            hint="Z is n_obs×n_state — one ROW per observed series"))
        _status("State-space (general system from $config): series=$(join(vnames, ", ")), " *
                "T=$(size(Y, 1)), n_obs=$(cfg.n_obs), n_state=$(cfg.n_state), init_mode=$(cfg.init_mode)")
        _status()
        # The MEMs constructor re-checks dimensions and throws untyped ArgumentErrors;
        # `get_statespace` has already checked them so a throw here is a genuine internal
        # inconsistency, but map it anyway rather than let it escape as exit 1.
        spec = try
            StateSpaceModel(cfg.Z, cfg.H, cfg.T, cfg.Q; d=cfg.d, c=cfg.c, R=cfg.R,
                            a1=cfg.a1, P1=cfg.P1, init_mode=cfg.init_mode, kappa=kappa)
        catch e
            e isa ArgumentError && throw(CliError("config/shape",
                "estimate statespace: inconsistent [statespace] system — $(e.msg)"))
            throw(_garch_variant_error(e, "State-space system"))
        end
        fitted_ss = try
            estimate_statespace(spec, Y)
        catch e
            throw(_garch_variant_error(e, "State-space filtering"))
        end
        output_result(_statespace_system_table(fitted_ss); format=Symbol(format), output=output,
                      title="State-Space System (general)")
        output_kv(Pair{String,Any}[
            "model"     => "general",
            "loglik"    => isfinite(fitted_ss.loglik) ? round(Float64(fitted_ss.loglik); digits=4) :
                           string(fitted_ss.loglik),
            "init_mode" => string(fitted_ss.init_mode),
            "n_obs_series" => fitted_ss.n_obs,
            "n_state"   => fitted_ss.n_state,
            "n_periods" => fitted_ss.T_obs,
            # :filter, not :mle — a fixed-matrix system is filtered and log-likelihood
            # evaluated, never optimized, so there is nothing to converge.
            "method"    => string(fitted_ss.method),
        ]; format=format, title="State-Space Diagnostics")
        return fitted_ss
    end

    # ── Canned univariate systems ───────────────────────────────────────────────────────
    model in ("local-level", "local-linear-trend") || throw(CliError("usage/invalid",
        "estimate statespace: --model must be local-level or local-linear-trend, got '$model'"))
    init_mode in ("kappa", "diffuse") || throw(CliError("usage/invalid",
        "estimate statespace: --init-mode must be kappa or diffuse, got '$init_mode'"))
    y, vname = load_univariate_series(data, column)
    _status("State-space $model: variable=$vname, observations=$(length(y)), init_mode=$init_mode")
    _status()
    im = Symbol(init_mode)
    ssm = try
        model == "local-level" ? local_level(y; init_mode=im, kappa=kappa) :
                                 local_linear_trend(y; init_mode=im, kappa=kappa)
    catch e
        throw(_garch_variant_error(e, "State-space estimation"))
    end
    output_result(_statespace_param_table(ssm); format=Symbol(format), output=output,
                  title="State-Space Hyper-Parameters ($model, $vname)",
                  key="state_space_hyper_parameters")
    output_kv(Pair{String,Any}[
        "model"     => model,
        # loglik/theta can be non-finite if the optimizer stalls — string-render (legacy-JSON
        # writer rejects non-finite floats; see the C067a Inf gotcha).
        "loglik"    => isfinite(ssm.loglik) ? round(Float64(ssm.loglik); digits=4) : string(ssm.loglik),
        "converged" => ssm.converged,
        "n_state"   => ssm.n_state,
        "n_obs"     => ssm.T_obs,   # T_obs = number of time observations (the user-facing "n_obs")
        "method"    => string(ssm.method),
    ]; format=format, title="State-Space Diagnostics")
    return ssm
end

# estimate tvp — time-varying-parameter regression (random-walk coefficients)
function _estimate_tvp(; data::String, dep::String="", init_mode::String="kappa",
                        kappa::Float64=1e6, no_intercept::Bool=false,
                        output::String="", format::String="table")
    init_mode in ("kappa", "diffuse") || throw(CliError("usage/invalid",
        "estimate tvp: --init-mode must be kappa or diffuse, got '$init_mode'"))
    y, X, xcols = _load_reg_data(data, dep)   # X = numeric cols except --dep; estimate_tvp_reg adds its own intercept
    intercept = !no_intercept
    coefnames = intercept ? vcat(["intercept"], xcols) : xcols
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Time-varying-parameter regression: $dep_name ~ $(join(xcols, " + "))" *
            (intercept ? " (+ intercept)" : "") * ", n=$(length(y)), init_mode=$init_mode")
    _status()
    ssm = try
        estimate_tvp_reg(y, X; intercept=intercept, init_mode=Symbol(init_mode), kappa=kappa)
    catch e
        throw(_garch_variant_error(e, "TVP regression"))
    end
    output_result(_statespace_param_table(ssm); format=Symbol(format), output=output,
                  title="TVP Hyper-Parameters")
    output_result(_tvp_path_table(ssm, coefnames); format=Symbol(format),
                  output=_per_var_output_path(output, "path"), title="TVP Coefficient Paths")
    output_kv(Pair{String,Any}[
        "loglik"    => isfinite(ssm.loglik) ? round(Float64(ssm.loglik); digits=4) : string(ssm.loglik),
        "converged" => ssm.converged,
        "n_coef"    => ssm.n_state,
        "intercept" => intercept,
        "method"    => string(ssm.method),
    ]; format=format, title="TVP Diagnostics")
    return ssm
end

# estimate kde — univariate kernel density estimate
_kde_table(kd) = DataFrame(x=round.(Float64.(kd.x); digits=6),
                           density=round.(Float64.(kd.density); digits=6))

function _estimate_kde(; data::String, column::Int=1, kernel::String="gaussian",
                        bw::String="silverman", npoints::Int=512, cut::Float64=3.0,
                        output::String="", format::String="table")
    kernel in ("gaussian", "epanechnikov", "triangular", "uniform") || throw(CliError("usage/invalid",
        "estimate kde: --kernel must be gaussian|epanechnikov|triangular|uniform, got '$kernel'"))
    npoints >= 2 || throw(CliError("usage/invalid",
        "estimate kde: --npoints must be ≥ 2, got $npoints"))
    y, vname = load_univariate_series(data, column)
    bwv = _parse_bandwidth(bw, (:silverman, :sj))
    _status("Kernel density: variable=$vname, observations=$(length(y)), kernel=$kernel, bw=$bw")
    _status()
    kd = try
        kernel_density(y; kernel=Symbol(kernel), bw=bwv, npoints=npoints, cut=cut)
    catch e
        throw(_garch_variant_error(e, "Kernel density"))
    end
    output_result(_kde_table(kd); format=Symbol(format), output=output,
                  title="Kernel Density Estimate ($vname)", key="kernel_density_estimate")
    output_kv(Pair{String,Any}[
        "kernel"    => string(kd.kernel),
        "bw_method" => string(kd.bw_method),
        "bandwidth" => round(Float64(kd.bandwidth); digits=6),
        "nobs"      => kd.nobs,
    ]; format=format, title="Kernel Density Diagnostics")
    return kd
end

# estimate kernel-reg — Nadaraya-Watson / local-linear / local-polynomial regression
_kernel_reg_table(kr) = DataFrame(x=round.(Float64.(kr.x); digits=6),
                                  fitted=round.(Float64.(kr.fitted); digits=6),
                                  se=round.(Float64.(kr.se); digits=6))

function _estimate_kernel_reg(; data::String, dep::String="", indep::String="",
                               method::String="ll", degree::Int=1, bw::String="cv",
                               kernel::String="gaussian", output::String="", format::String="table")
    method in ("nw", "ll", "lp") || throw(CliError("usage/invalid",
        "estimate kernel-reg: --method must be nw|ll|lp, got '$method'"))
    kernel in ("gaussian", "epanechnikov", "triangular", "uniform") || throw(CliError("usage/invalid",
        "estimate kernel-reg: --kernel must be gaussian|epanechnikov|triangular|uniform, got '$kernel'"))
    degree >= 0 || throw(CliError("usage/invalid",
        "estimate kernel-reg: --degree must be ≥ 0, got $degree"))
    y, x, ynm, xnm = _load_xy_data(data, dep, indep)
    bwv = _parse_bandwidth(bw, (:cv, :rot))
    _status("Kernel regression ($method): $ynm ~ $xnm, n=$(length(y)), bw=$bw, kernel=$kernel")
    _status()
    kr = try
        kernel_reg(y, x; method=Symbol(method), degree=degree, bw=bwv, kernel=Symbol(kernel))
    catch e
        throw(_garch_variant_error(e, "Kernel regression"))
    end
    output_result(_kernel_reg_table(kr); format=Symbol(format), output=output,
                  title="Kernel Regression Fit ($ynm ~ $xnm)", key="kernel_regression_fit")
    output_kv(Pair{String,Any}[
        "method"    => string(kr.method),
        "degree"    => kr.degree,
        "kernel"    => string(kr.kernel),
        "bw_method" => string(kr.bw_method),
        "bandwidth" => round(Float64(kr.bandwidth); digits=6),
        "nobs"      => kr.nobs,
    ]; format=format, title="Kernel Regression Diagnostics")
    return kr
end

# estimate lowess — Cleveland (1979) locally-weighted scatterplot smoother
_lowess_table(lf) = DataFrame(x=round.(Float64.(lf.x); digits=6),
                              fitted=round.(Float64.(lf.fitted); digits=6))

function _estimate_lowess(; data::String, dep::String="", indep::String="",
                           frac::Float64=0.6667, iter::Int=3,
                           output::String="", format::String="table")
    (0.0 < frac <= 1.0) || throw(CliError("usage/invalid",
        "estimate lowess: --frac must be in (0,1], got $frac"))
    iter >= 0 || throw(CliError("usage/invalid",
        "estimate lowess: --iter must be ≥ 0, got $iter"))
    y, x, ynm, xnm = _load_xy_data(data, dep, indep)
    _status("LOWESS: $ynm ~ $xnm, n=$(length(y)), f=$frac, iter=$iter")
    _status()
    lf = try
        lowess(y, x; f=frac, iter=iter)
    catch e
        throw(_garch_variant_error(e, "LOWESS"))
    end
    output_result(_lowess_table(lf); format=Symbol(format), output=output,
                  title="LOWESS Fit ($ynm ~ $xnm)", key="lowess_fit")
    output_kv(Pair{String,Any}[
        "frac" => round(Float64(lf.span); digits=6),
        "iter" => lf.iter,
        "nobs" => lf.nobs,
    ]; format=format, title="LOWESS Diagnostics")
    return lf
end

# ── C062a: cointegrating regression (FMOLS / CCR / DOLS) ─────────────
# Single-equation `estimate cointreg` (`_load_reg_data`, y = --dep levels, X = other numeric
# columns, no intercept prepended — cointreg adds its own deterministics via --trend) and
# panel `estimate xtcointreg` (`_load_panel_reg`). `CointRegModel`/`PanelCointRegModel` are
# StatsAPI RegressionModels but NOT registered in MEMs `_COEF_TABLE_TYPES`, so the tidy coef
# table is hand-built (documented C051 exception, like io/mgarch/sur). Every MEMs call is
# wrapped → typed CliError via `_garch_variant_error`; enums are guarded up front (`choices=`
# on the spec). The cointreg trend vocabulary is `none|const|linear` (NOT shared with other
# leaves — PMG uses `:constant`, ARDL uses `:const`/`:trend`).

"""Parse a cointreg `--bandwidth`: `andrews`|`nw94` → Symbol, else a non-negative number
(a fixed HAC truncation lag). A bad token → typed `usage/invalid` (never a raw MEMs error)."""
function _parse_cointreg_bandwidth(s::AbstractString)
    (s == "andrews" || s == "nw94") && return Symbol(s)
    v = tryparse(Float64, s)
    (v === nothing || !isfinite(v) || v < 0) && throw(CliError("usage/invalid",
        "--bandwidth must be andrews|nw94 or a non-negative number, got '$s'"))
    return v
end

"""Parse a DOLS `--leads`/`--lags`: `auto` → `:auto`, else a non-negative integer.
A bad token → typed `usage/invalid` (0 is valid — do NOT route through `_parse_bandwidth`,
which rejects 0)."""
function _parse_cointreg_leadlag(s::AbstractString, flag::String)
    s == "auto" && return :auto
    v = tryparse(Int, s)
    (v === nothing || v < 0) && throw(CliError("usage/invalid",
        "$flag must be 'auto' or a non-negative integer, got '$s'"))
    return v
end

"""Hand-built coefficient table for a single-equation cointegrating regression
(`CointRegModel`): `term|estimate|std_error|stat|p_value|ci_lower|ci_upper`. p-values/CIs use
the large-sample normal approximation (the estimators are asymptotically mixed-normal; same
convention as `_garch_variant_coef_table`/`dsge` using `_normal_cdf`)."""
function _cointreg_coef_table(model)
    est = Float64.(coef(model))
    se = Float64.(stderror(model))
    names = String.(model.varnames)[1:length(est)]
    z = [se[i] == 0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    zc = 1.959963984540054   # Φ⁻¹(0.975)
    return DataFrame(term=names, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6), stat=round.(z; digits=4),
                     p_value=round.(pv; digits=4),
                     ci_lower=round.(est .- zc .* se; digits=6),
                     ci_upper=round.(est .+ zc .* se; digits=6))
end

"""Hand-built coefficient table for a panel cointegrating regression
(`PanelCointRegModel`): reads the PRECOMPUTED `coef`/`se`/`tstats`/`pvalues` fields directly.
The group-mean SE is a back-solved display SE and can be `Inf` — do NOT recompute from
`vcov`; non-finite entries are kept (round-safe) and rendered by the non-finite-safe output
path (C067a/C073)."""
function _panel_cointreg_coef_table(model)
    est = Float64.(model.coef)
    se = Float64.(model.se)
    names = String.(model.varnames)[1:length(est)]
    zc = 1.959963984540054
    return DataFrame(term=names, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6), stat=round.(Float64.(model.tstats); digits=4),
                     p_value=round.(Float64.(model.pvalues); digits=4),
                     ci_lower=round.(est .- zc .* se; digits=6),
                     ci_upper=round.(est .+ zc .* se; digits=6))
end

function _estimate_cointreg(; data::String, dep::String="", method::String="fmols",
                             trend::String="const", kernel::String="bartlett",
                             bandwidth::String="andrews", leads::String="auto",
                             lags::String="auto", ic::String="aic", dols_se::String="lrv",
                             output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    bw = _parse_cointreg_bandwidth(bandwidth)
    ld = _parse_cointreg_leadlag(leads, "--leads")
    lg = _parse_cointreg_leadlag(lags, "--lags")
    _status("Cointegrating regression ($(uppercase(method))): $dep_name ~ $(join(xcols, " + ")), n=$(length(y))")
    _status()
    model = try
        estimate_cointreg(y, X; method=Symbol(method), trend=Symbol(trend),
            kernel=Symbol(replace(kernel, '-' => '_')), bandwidth=bw,
            leads=ld, lags=lg, ic=Symbol(ic), dols_se=Symbol(dols_se))
    catch e
        throw(_garch_variant_error(e, "cointreg"))
    end
    output_result(_cointreg_coef_table(model); format=Symbol(format), output=output,
                  title="Cointegrating Regression Coefficients ($dep_name)",
                  key="cointegrating_regression_coefficients")
    pairs = Pair{String,Any}[
        "method"    => String(model.method),
        "trend"     => String(model.trend),
        "kernel"    => String(model.kernel),
        "bandwidth" => round(Float64(model.bandwidth); digits=4),
        "omega_uv"  => round(Float64(model.omega_uv); digits=6),
        "nobs"      => model.nobs,
        "d"         => model.d,
        "k"         => model.k,
    ]
    if model.method === :dols
        push!(pairs, "leads" => model.leads)
        push!(pairs, "lags"  => model.lags)
    end
    output_kv(pairs; format=format, title="Cointegrating Regression Diagnostics")
    return model
end

function _estimate_xtcointreg(; data::String, id_col::String="", time_col::String="",
                               dep::String="", indep::String="", method::String="fmols",
                               pooling::String="group", trend::String="const",
                               kernel::String="bartlett", bandwidth::String="andrews",
                               leads::String="auto", lags::String="auto", ic::String="aic",
                               dols_se::String="lrv", output::String="", format::String="table")
    pd, depsym, indepsyms, depc, indeps = _load_panel_reg(data, id_col, time_col, dep, indep)
    bw = _parse_cointreg_bandwidth(bandwidth)
    ld = _parse_cointreg_leadlag(leads, "--leads")
    lg = _parse_cointreg_leadlag(lags, "--lags")
    _status("Panel cointegrating regression ($(uppercase(method)), $pooling): $depc ~ $(join(indeps, " + ")), units=$(pd.n_groups)")
    _status()
    model = try
        estimate_xtcointreg(pd, depsym, indepsyms...; method=Symbol(method),
            pooling=Symbol(pooling), trend=Symbol(trend),
            kernel=Symbol(replace(kernel, '-' => '_')), bandwidth=bw,
            leads=ld, lags=lg, ic=Symbol(ic), dols_se=Symbol(dols_se))
    catch e
        throw(_garch_variant_error(e, "xtcointreg"))
    end
    title = model.pooling === :group ? "Group-mean" : "Pooled"
    # `--pooling` selects the estimator, not a different table: same columns either way, so
    # it stays in the title and the key is fixed.
    output_result(_panel_cointreg_coef_table(model); format=Symbol(format), output=output,
                  title="Panel Cointegrating Regression $title Coefficients ($depc)",
                  key="panel_cointegrating_regression_coefficients")
    tspan = model.balanced ? string(first(model.T_i)) :
            "$(minimum(model.T_i))-$(maximum(model.T_i))"
    output_kv(Pair{String,Any}[
        "method"   => String(model.method),
        "pooling"  => String(model.pooling),
        "trend"    => String(model.trend),
        "kernel"   => String(model.kernel),
        "N"        => model.N,
        "nobs"     => model.nobs,
        "T_i"      => tspan,
        "balanced" => model.balanced,
        "k"        => model.k,
        "d"        => model.d,
    ]; format=format, title="Panel Cointegrating Regression Diagnostics")
    return model
end

# ── C062b: single-equation ARDL / NARDL ──────────────────────────────
# `estimate ardl` / `estimate nardl`, plus `test ardl-bounds` / `test nardl-symmetry`
# (test.jl) and `multipliers nardl` (multipliers.jl) all fit via these shared helpers.
# ARDLModel/NARDLModel are StatsAPI RegressionModels but NOT in MEMs `_COEF_TABLE_TYPES`,
# so the tidy coef/long-run tables are hand-built (documented C051 exception, like
# io/mgarch/sur/cointreg). p-values use the normal approximation (`_normal_cdf`) — the CLI
# convention everywhere (the ARDL OLS t-ratios are asymptotically standard-normal, and the
# mock has no TDist); the ARDL/NARDL trend vocabulary is none|const|trend (ARDL-specific).

"""Hand-built levels-form coefficient table for a fitted `ARDLModel` (pass `m` for
`estimate ardl` or `m.ardl` for NARDL): `term|estimate|std_error|stat|p_value`."""
function _ardl_coef_table(a)
    est = Float64.(coef(a))
    se = Float64.(stderror(a))
    names = String.(a.coefnames)[1:length(est)]
    z = [se[i] == 0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    return DataFrame(term=names, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6), stat=round.(z; digits=4),
                     p_value=round.(pv; digits=4))
end

"""Hand-built long-run (level) multiplier table from an `ARDLLongRun`:
`term|estimate|std_error|stat|p_value` (delta-method SEs; normal-approx stat/p — MEMs uses
`dist=:z` for the long-run block). Serves `estimate ardl` (symmetric) and `estimate nardl`
(θ⁺/θ⁻ on the enlarged regressor set)."""
function _ardl_longrun_table(lr)
    est = Float64.(lr.theta)
    se = Float64.(lr.se)
    names = String.(lr.varnames)[1:length(est)]
    z = [se[i] == 0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    return DataFrame(term=names, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6), stat=round.(z; digits=4),
                     p_value=round.(pv; digits=4))
end

"""Fit an `ARDLModel` from a loaded `(y, X, xcols)` + a resolved dep name, with all
argument validation done up front (`--case`/`--max-p`/`--max-q` + `--p`/`--q` parse and the
per-regressor `--q` length check). Every MEMs call is wrapped → typed `CliError` via
`_garch_variant_error` (bad case/q → `data/invalid`, shape → `data/shape`). Shared by
`estimate ardl` and `test ardl-bounds`."""
function _fit_ardl(y, X, xcols, dep_name; p::String, q::String, max_p::Int, max_q::Int,
                   ic::String, case::Int, trend::String, label::String)
    (1 <= case <= 5) || throw(CliError("usage/invalid", "$label: --case must be in 1:5, got $case"))
    max_p >= 1 || throw(CliError("usage/invalid", "$label: --max-p must be ≥ 1, got $max_p"))
    max_q >= 0 || throw(CliError("usage/invalid", "$label: --max-q must be ≥ 0, got $max_q"))
    pp = _parse_lag_spec(p, "--p"; min=1)
    qq = _parse_lag_spec(q, "--q"; min=0)
    k = length(xcols)
    (qq isa Vector{Int} && length(qq) != k) && throw(CliError("usage/invalid",
        "$label: --q has $(length(qq)) entries but there are $k regressor(s); pass one per regressor, a single int, or auto"))
    return try
        estimate_ardl(y, X; p=pp, q=qq, max_p=max_p, max_q=max_q, ic=Symbol(ic),
                      case=case, trend=Symbol(trend), xnames=xcols, yname=dep_name)
    catch e
        throw(_garch_variant_error(e, label))
    end
end

"""Fit a `NARDLModel` from a loaded `(y, X, xcols)` + a resolved dep name. Validates
`--case`/orders, parses `--p`/`--q`, and resolves/bounds-checks `--asymmetric` (each index ∈
`1:k₀`). Wrapped → typed `CliError`. Shared by `estimate nardl`, `test nardl-symmetry`, and
`multipliers nardl`."""
function _fit_nardl(y, X, xcols, dep_name; asymmetric::String, p::String, q::String,
                    max_p::Int, max_q::Int, ic::String, case::Int, label::String)
    (1 <= case <= 5) || throw(CliError("usage/invalid", "$label: --case must be in 1:5, got $case"))
    max_p >= 1 || throw(CliError("usage/invalid", "$label: --max-p must be ≥ 1, got $max_p"))
    max_q >= 0 || throw(CliError("usage/invalid", "$label: --max-q must be ≥ 0, got $max_q"))
    k0 = length(xcols)
    asym = _parse_asym_spec(asymmetric)
    (asym isa Vector{Int} && !all(j -> 1 <= j <= k0, asym)) && throw(CliError("usage/invalid",
        "$label: --asymmetric indices must be in 1:$k0, got $asym"))
    pp = _parse_lag_spec(p, "--p"; min=1)
    qq = _parse_lag_spec(q, "--q"; min=0)
    # Up-front `--q` length check on the ENLARGED split design (each asymmetric regressor →
    # POS/NEG), mirroring `_fit_ardl`'s guard → a wrong-length list is usage/invalid (exit 2)
    # with a clean CLI message, not MEMs' data/invalid referencing the internal split count
    # (adversarial review C062b).
    kk = asym === :all ? 2 * k0 : k0 + length(asym)
    (qq isa Vector{Int} && length(qq) != kk) && throw(CliError("usage/invalid",
        "$label: --q has $(length(qq)) entries but the NARDL split design has $kk regressor(s) (each asymmetric regressor is split into POS/NEG); pass one per split regressor, a single int, or auto"))
    return try
        estimate_nardl(y, X; asymmetric=asym, p=pp, q=qq, max_p=max_p, max_q=max_q,
                       ic=Symbol(ic), case=case, xnames=xcols, yname=dep_name)
    catch e
        throw(_garch_variant_error(e, label))
    end
end

"""Round a scalar for kv output, string-rendering non-finite values (the legacy-JSON writer
rejects Inf/NaN — the C067a gotcha; `alpha_t` can be Inf when `alpha_se==0`)."""
_ardl_kv(x; digits::Int=6) = isfinite(x) ? round(Float64(x); digits=digits) : string(x)

function _estimate_ardl(; data::String, dep::String="", p::String="auto", q::String="auto",
                         max_p::Int=4, max_q::Int=4, ic::String="aic", case::Int=3,
                         trend::String="none", output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("ARDL: $dep_name ~ $(join(xcols, " + ")) (p=$p, q=$q, case=$case, ic=$ic), n=$(length(y))"); _status()
    m = _fit_ardl(y, X, xcols, dep_name; p=p, q=q, max_p=max_p, max_q=max_q,
                  ic=ic, case=case, trend=trend, label="ARDL")
    output_result(_ardl_coef_table(m); format=Symbol(format), output=output,
                  title="ARDL Coefficients (levels form) ($dep_name)", key="ardl_coefficients")
    lr = long_run(m)
    output_result(_ardl_longrun_table(lr); format=Symbol(format),
                  output=_per_var_output_path(output, "longrun"),
                  title="ARDL Long-Run Coefficients ($dep_name)",
                  key="ardl_long_run_coefficients")
    ecm = ecm_form(m)   # exported since MEMs 0.8.0
    output_kv(Pair{String,Any}[
        "p"             => m.p,
        "q"             => join(m.q, ","),
        "case"          => m.case,
        "trend"         => String(m.trend),
        "ic"            => String(m.ic),
        "selected"      => m.selected,
        "nobs"          => m.n,
        "K"             => m.K,
        "sigma2"        => _ardl_kv(m.sigma2),
        "loglik"        => _ardl_kv(m.loglik; digits=4),
        "aic"           => _ardl_kv(m.aic; digits=4),
        "bic"           => _ardl_kv(m.bic; digits=4),
        "alpha"         => _ardl_kv(ecm.alpha),           # ECM speed of adjustment (Σφ − 1)
        "alpha_se"      => _ardl_kv(ecm.alpha_se),
        "alpha_t"       => _ardl_kv(ecm.alpha_t; digits=4),
        "longrun_denom" => _ardl_kv(lr.denom),            # 1 − Σφ̂ = −alpha
    ]; format=format, title="ARDL Diagnostics")
    return m
end

function _estimate_nardl(; data::String, dep::String="", asymmetric::String="all",
                          p::String="auto", q::String="auto", max_p::Int=4, max_q::Int=4,
                          ic::String="aic", case::Int=3, output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("NARDL: $dep_name ~ $(join(xcols, " + ")) (asym=$asymmetric, p=$p, q=$q, case=$case), n=$(length(y))"); _status()
    m = _fit_nardl(y, X, xcols, dep_name; asymmetric=asymmetric, p=p, q=q,
                   max_p=max_p, max_q=max_q, ic=ic, case=case, label="NARDL")
    a = m.ardl
    output_result(_ardl_coef_table(a); format=Symbol(format), output=output,
                  title="NARDL Coefficients (levels form, split regressors) ($dep_name)",
                  key="nardl_coefficients")
    lr = long_run(m)   # ARDLLongRun on the enlarged (θ⁺/θ⁻) regressor set
    output_result(_ardl_longrun_table(lr); format=Symbol(format),
                  output=_per_var_output_path(output, "longrun"),
                  title="NARDL Asymmetric Long-Run Coefficients (θ⁺ / θ⁻) ($dep_name)",
                  key="nardl_asymmetric_long_run_coefficients")
    # Cached enlarged-k bounds decision (NO p-value — non-standard PSS test).
    b = m.bounds
    li = findfirst(x -> isapprox(x, b.level), b.levels)
    output_kv(Pair{String,Any}[
        "k_orig"       => m.k_orig,
        "k"            => m.k,                 # enlarged: each asymmetric regressor counts twice
        "asym"         => join(m.asym, ","),
        "p"            => a.p,
        "q"            => join(a.q, ","),
        "case"         => a.case,
        "ic"           => String(a.ic),
        "nobs"         => a.n,
        "f_stat"       => _ardl_kv(b.fstat; digits=4),
        "t_stat"       => _ardl_kv(b.tstat; digits=4),
        "bounds_level" => b.level,
        "f_lower"      => li === nothing ? "n/a" : _ardl_kv(b.f_lower[li]; digits=4),
        "f_upper"      => li === nothing ? "n/a" : _ardl_kv(b.f_upper[li]; digits=4),
        "f_decision"   => String(b.f_decision),
        "t_decision"   => String(b.t_decision),
    ]; format=format, title="NARDL Diagnostics + Bounds (enlarged k=$(m.k))",
       key="nardl_diagnostics")
    return m
end

# ── C062c: dynamic heterogeneous-panel ARDL (PMG / MG / DFE) ──────────────────
# `estimate pmg` (Pooled Mean Group + Mean Group + Dynamic Fixed Effects; Pesaran-Shin-Smith
# 1999) and `test pmg-hausman` (test.jl) fit via the shared hardened `_load_panel_reg` panel
# loader (typed dup-(id,time) / missing-cell / bad-column guards). `PMGModel` is a StatsAPI
# model but NOT in MEMs `_COEF_TABLE_TYPES` → hand-built long-run + short-run/EC tables
# (documented C051 exception, like io/mgarch/sur/cointreg/ardl). Long-run p-values use the
# normal approximation (`_normal_cdf`, the CLI convention — the pooled θ is asymptotically
# normal). NOTE the trend vocabulary is PMG-specific: none|constant|trend — `:constant` is
# SPELLED OUT (NOT ARDL's :const, NOT cointreg's :linear); do not copy another leaf's trend
# OptionSpec here. Display SEs (φ, θ) can be Inf → the kv path string-renders non-finite
# scalars (`_ardl_kv`) and the table path is non-finite-safe (C067a/C073).

"""Hand-built long-run coefficient table for a fitted `PMGModel`:
`term|estimate|std_error|stat|p_value` from the common (`:pmg`/`:dfe`) or averaged (`:mg`)
long-run block `theta`/`theta_se`; `term = m.xnames`. Normal-approx stat/p."""
function _pmg_longrun_table(m)
    est = Float64.(m.theta)
    se = Float64.(m.theta_se)
    names = String.(m.xnames)[1:length(est)]
    z = [se[i] == 0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
    pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    return DataFrame(term=names, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6), stat=round.(z; digits=4),
                     p_value=round.(pv; digits=4))
end

"""Hand-built short-run / error-correction table for a fitted `PMGModel`:
`term|estimate|std_error`. The error-correction speed φ (mean for `:pmg`/`:mg`, common for
`:dfe`) is the first row, followed by the averaged/common short-run block (`m.sr`/`m.sr_se`,
`term = m.srnames`). For a DFE fit the unit intercept is absorbed → `srnames` may be empty, in
which case only the φ row is emitted."""
function _pmg_shortrun_table(m)
    terms = String["EC speed (phi)"]
    est = Float64[m.phi]
    se = Float64[m.phi_se]
    srnames = String.(m.srnames)
    for j in eachindex(srnames)
        push!(terms, srnames[j]); push!(est, Float64(m.sr[j])); push!(se, Float64(m.sr_se[j]))
    end
    return DataFrame(term=terms, estimate=round.(est; digits=6), std_error=round.(se; digits=6))
end

function _estimate_pmg(; data::String, id_col::String="", time_col::String="",
                        dep::String="", indep::String="", method::String="pmg",
                        trend::String="constant", p::Int=1, q::Int=1,
                        maxiter::Int=100, tol::Float64=1e-8,
                        output::String="", format::String="table")
    p >= 1 || throw(CliError("usage/invalid", "--p must be ≥ 1, got $p"))
    q >= 0 || throw(CliError("usage/invalid", "--q must be ≥ 0, got $q"))
    maxiter >= 1 || throw(CliError("usage/invalid", "--maxiter must be ≥ 1, got $maxiter"))
    tol > 0 || throw(CliError("usage/invalid", "--tol must be > 0, got $tol"))
    pd, depsym, indepsyms, depc, indeps = _load_panel_reg(data, id_col, time_col, dep, indep)
    _status("Panel ARDL ($(uppercase(method)), EC form): $depc ~ $(join(indeps, " + ")) (p=$p, q=$q, trend=$trend), units=$(pd.n_groups)"); _status()
    m = try
        estimate_pmg(pd, depsym, indepsyms...; p=p, q=q, method=Symbol(method),
                     trend=Symbol(trend), maxiter=maxiter, tol=tol)
    catch e
        throw(_garch_variant_error(e, "PMG"))
    end
    output_result(_pmg_longrun_table(m); format=Symbol(format), output=output,
                  title="Panel ARDL Long-Run Coefficients (theta) ($depc)",
                  key="panel_ardl_long_run_coefficients")
    output_result(_pmg_shortrun_table(m); format=Symbol(format),
                  output=_per_var_output_path(output, "shortrun"),
                  title="Panel ARDL Short-Run + EC Coefficients ($depc)",
                  key="panel_ardl_short_run_ec_coefficients")
    tspan = length(unique(m.T_i)) == 1 ? string(first(m.T_i)) :
            "$(minimum(m.T_i))-$(maximum(m.T_i))"
    output_kv(Pair{String,Any}[
        "method"    => String(m.method),
        "N"         => m.N,
        "p"         => m.p,
        "q"         => m.q,
        "T_i"       => tspan,
        "phi"       => _ardl_kv(m.phi),
        "phi_se"    => _ardl_kv(m.phi_se),
        "loglik"    => _ardl_kv(m.loglik; digits=4),
        "converged" => m.converged,
        "iters"     => m.iters,
        "n_nonconv" => m.n_nonconv,
    ]; format=format, title="Panel ARDL Diagnostics")
    return m
end

# ── C062d: MIDAS mixed-frequency regression ──────────────────────────
# `estimate midas` fits a (restricted) MIDAS / ADL-MIDAS / U-MIDAS regression of a low-frequency
# target on `--k` high-frequency lags of a single indicator, aggregated through a weight function.
# The mixed-frequency inputs are loaded via the hardened `_load_midas_data` (both series through
# `load_univariate_series`; len(HF)>=m×len(LF), leading ragged edge dropped → typed data/shape). `MidasModel` is a
# StatsAPI RegressionModel but NOT in MEMs `_COEF_TABLE_TYPES` → the weight-curve + coefficient
# tables are hand-built (documented C051 exception, like io/mgarch/sur/cointreg/ardl). p-values use
# the normal approximation (`_normal_cdf`) — the CLI convention (the NLS t-ratios are asymptotically
# standard-normal). `forecast midas` is DEFERRED (fresh-HF-block UX + non-native save/load; v1.1).

"""Map an untyped MEMs MIDAS-estimation failure to a typed CliError — never an uncaught exit-1
(the standing shared-loader lesson). Bad-input `ArgumentError`/`DomainError` (bad m/K, no complete
HF blocks, no AR periods, K<2 for a Beta weight, wrong θ length) → `data/invalid` (3); shape
mismatch → `data/shape` (3); the NLS 'failed to converge from any start' → `model/convergence` (5);
anything else → `model/error` (5)."""
function _midas_error(e, label::String)
    e isa CliError && return e
    if e isa ArgumentError || e isa DomainError
        return CliError("data/invalid", "$label: $(sprint(showerror, e))";
            hint="check --m, --k, --p-ar and the series lengths (HF ≈ m×LF)")
    elseif e isa DimensionMismatch
        return CliError("data/shape", "$label: $(sprint(showerror, e))")
    elseif occursin("converge", lowercase(sprint(showerror, e)))
        return CliError("model/convergence", "$label failed to converge from any start";
            hint="try --weights umidas, a smaller --k, or a larger --max-iter")
    end
    return CliError("model/error", "$label estimation failed: $(sprint(showerror, e))")
end

"""Headline MIDAS weight-curve table `lag|weight` from `midas_weights(m)` (== `m.w`, length K,
most-recent-first). For `:umidas` these are the unrestricted lag coefficients (raw, not normalized)."""
function _midas_weight_table(model)
    w = Float64.(midas_weights(model))
    return DataFrame(lag=collect(1:length(w)), weight=round.(w; digits=6))
end

"""Hand-built MIDAS coefficient table (`term|estimate|std_error|stat|p_value`) over the full
parameter vector `[β; θ]` (`coef(m)`), with `term = m.varnames`. Falls back to estimate-only when
the Gauss-Newton SEs are unavailable/non-finite (mirrors `_garch_variant_coef_table`). For
`:umidas` the θ block is empty and the K lag coefficients enter as the linear block; θ labels are
padded if `varnames` is shorter than `coef` (defensive)."""
function _midas_coef_table(model)
    est = Float64.(coef(model))
    nm = String.(model.varnames)
    names = length(nm) == length(est) ? nm :
            String[i <= length(nm) ? nm[i] : "theta$(i - length(model.beta))" for i in eachindex(est)]
    try
        se = Float64.(stderror(model))
        (length(se) == length(est) && all(isfinite, se)) || error("unavailable SEs")
        z = [se[i] == 0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        return DataFrame(term=names, estimate=round.(est; digits=6),
                         std_error=round.(se; digits=6), stat=round.(z; digits=4),
                         p_value=round.(pv; digits=4))
    catch
        return DataFrame(term=names, estimate=round.(est; digits=6))
    end
end

# ── #67: `forecast midas` ────────────────────────────────────────────────────
# This leaf is shaped UNLIKE every other `forecast` leaf, and deliberately so:
#
#   forecast(m::MidasModel, X_new; y_lags=nothing, level=0.95) -> MidasForecast
#
# takes a fresh high-frequency BLOCK, not a horizon. The horizon is fixed at
# estimation time (`--horizon`, stored as `m.h`), so this leaf has **no --horizons** —
# advertising one would be a lie. The result is a single direct h-step point forecast
# with a Gaussian NLS prediction interval.
#
# CRITICAL: `X_new` must be **most-recent-first**. Passing the tail in chronological
# order does not error — it silently applies the weight curve backwards and returns a
# wrong number. Hence the explicit `reverse` below, and the T3 assertion that a
# reversed input changes the answer.
#
# `y_lags` is left to upstream: with `p_ar > 0` it defaults to the most recent in-sample
# target values, which is exactly what forecasting the next period means here.
function _forecast_midas(; data::String="", result=nothing, column::Int=1, hf_data::String="", hf_column::Int=1,
        m::Int=0, k::Int=0, weights::String="expalmon", p_ar::Int=0,
        poly_degree::Int=2, horizon::Int=1, max_iter::Int=500, level::Float64=0.95,
        output::String="", format::String="table", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast midas")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="MIDAS Forecast", key="midas_forecast")
    (0.0 < level < 1.0) || throw(CliError("usage/invalid",
        "forecast midas: --level must be in (0, 1) (got $level)"))
    poly_degree >= 0 || throw(CliError("usage/invalid",
        "forecast midas: --poly-degree must be ≥ 0 (got $poly_degree)"))
    horizon >= 1 || throw(CliError("usage/invalid",
        "forecast midas: --horizon must be ≥ 1 (got $horizon)"))
    (weights in ("beta2", "beta3") && k < 2) && throw(CliError("data/invalid",
        "--weights $weights requires --k ≥ 2 (the Beta weight grid needs ≥ 2 lags), got k=$k"))
    y_lf, x_hf, ynm, xnm = _load_midas_data(data, column, hf_data, hf_column; m=m)
    mdl = model === nothing ?
        (try
            estimate_midas(y_lf, x_hf; m=m, K=k, weights=Symbol(weights), p_ar=p_ar,
                           poly_degree=poly_degree, h=horizon, max_iter=max_iter)
        catch e
            throw(_midas_error(e, "MIDAS"))
        end) : model
    # The most recent K high-frequency observations, MOST-RECENT-FIRST.
    length(x_hf) >= mdl.K || throw(CliError("data/shape",
        "forecast midas: need at least K=$(mdl.K) high-frequency observations to condition on (got $(length(x_hf)))"))
    x_new = reverse(Float64.(x_hf[end - mdl.K + 1:end]))
    _status("MIDAS Forecast: target=$ynm, indicator=$xnm, h=$(mdl.h) (fixed at estimation), K=$(mdl.K)")
    _status()
    fc = try
        forecast(mdl, x_new; level=level)
    catch e
        throw(_midas_error(e, "MIDAS forecast"))
    end
    output_result(DataFrame(
            horizon = [mdl.h],
            forecast = round.(Float64.(collect(fc.forecast)); digits=6),
            lower = round.(Float64.(collect(fc.ci_lower)); digits=6),
            upper = round.(Float64.(collect(fc.ci_upper)); digits=6),
            se = round.(Float64.(collect(fc.se)); digits=6));
        format=Symbol(format), output=output,
        title="MIDAS Direct Forecast ($ynm)", key="midas_forecast")
    output_kv(Pair{String,Any}[
        "horizon (h)" => mdl.h,
        "K (HF lags)" => mdl.K,
        "m (HF per LF)" => mdl.m,
        "p_ar" => mdl.p_ar,
        "weights" => String(mdl.weights_kind),
        "level" => fc.conf_level];
        format=format, title="MIDAS Forecast Summary")
    return (; model=mdl, result=fc)
end

function _estimate_midas(; data::String, column::Int=1, hf_data::String="", hf_column::Int=1,
                          m::Int=0, k::Int=0, weights::String="expalmon", p_ar::Int=0,
                          poly_degree::Int=2, horizon::Int=1, max_iter::Int=500,
                          output::String="", format::String="table")
    # Up-front guards → usage/invalid (before any load or estimator call).
    isempty(hf_data) && throw(CliError("usage/missing-option",
        "estimate midas requires --hf-data <csv> (the high-frequency indicator series)"))
    m >= 1 || throw(CliError("usage/invalid", "--m (HF/LF frequency ratio) must be ≥ 1, got $m"))
    k >= 1 || throw(CliError("usage/invalid", "--k (number of high-frequency lags) must be ≥ 1, got $k"))
    p_ar >= 0 || throw(CliError("usage/invalid", "--p-ar must be ≥ 0, got $p_ar"))
    # h was INERT upstream at ≤0.7.2; implemented at 0.7.3 (MEMs#574) — the guard
    # keeps a bad value a usage error rather than upstream's ArgumentError.
    horizon >= 1 || throw(CliError("usage/invalid", "--horizon must be ≥ 1, got $horizon"))
    max_iter >= 1 || throw(CliError("usage/invalid", "--max-iter must be ≥ 1, got $max_iter"))
    # --poly-degree feeds the `:almon` polynomial (real `_n_theta(:almon, d)=d+1`); a negative degree
    # is never meaningful and, unguarded, reaches `_midas_theta_starts` OUTSIDE the estimator try/catch
    # → a bare `BoundsError` on real MEMs (`base[1]=…` on `zeros(0)`) mapped to model/error, while the
    # mock throws `ArgumentError` → data/invalid: a latent mock/real exit-class divergence. Reject up
    # front → usage/invalid (poly_degree=0 = constant/equal weights is valid on both).
    poly_degree >= 0 || throw(CliError("usage/invalid", "--poly-degree must be ≥ 0, got $poly_degree"))
    # Beta weights need a ≥2-point grid; pre-guard for a friendly message (real `_beta_grid` throws).
    (weights in ("beta2", "beta3") && k < 2) && throw(CliError("data/invalid",
        "--weights $weights requires --k ≥ 2 (the Beta weight grid needs ≥ 2 lags), got k=$k"))
    y_lf, x_hf, ynm, xnm = _load_midas_data(data, column, hf_data, hf_column; m=m)
    _status("Estimating MIDAS: target=$ynm (n=$(length(y_lf))), indicator=$xnm (HF n=$(length(x_hf))), m=$m, K=$k, weights=$weights, p_ar=$p_ar")
    _status()
    model = try
        estimate_midas(y_lf, x_hf; m=m, K=k, weights=Symbol(weights), p_ar=p_ar,
                       poly_degree=poly_degree, h=horizon, max_iter=max_iter)
    catch e
        throw(_midas_error(e, "MIDAS"))
    end
    output_result(_midas_weight_table(model); format=Symbol(format), output=output,
                  title="MIDAS Weight Curve ($xnm, most-recent-first)", key="midas_weight_curve")
    output_result(_midas_coef_table(model); format=Symbol(format),
                  output=_per_var_output_path(output, "coef"),
                  title="MIDAS Coefficients ($ynm)", key="midas_coefficients")
    output_kv(Pair{String,Any}[
        "weights_kind" => String(model.weights_kind),
        "m"            => model.m,
        "K"            => model.K,
        "p_ar"         => model.p_ar,
        "poly_degree"  => model.poly_degree,
        "h"            => model.h,
        "nobs"         => nobs(model),
        "r2"           => _ardl_kv(model.r2),
        "adj_r2"       => _ardl_kv(model.adj_r2),
        "ssr"          => _ardl_kv(model.ssr; digits=4),
        "sigma2"       => _ardl_kv(model.sigma2),
        "aic"          => _ardl_kv(model.aic; digits=4),
        "bic"          => _ardl_kv(model.bic; digits=4),
        "loglik"       => _ardl_kv(model.loglik; digits=4),
        "converged"    => model.converged,
    ]; format=format, title="MIDAS Diagnostics")
    return model
end

# ── C065a: SETAR handler ───────────────────────────────────────
# Every user option is guarded up-front → usage/invalid (before any MEMs call); the
# estimator is try-wrapped → typed CliError via `_nonlinear_error` (never exit-1). The
# two regime blocks render via the shared `_threshold_coef_table`; the Hansen (1996)
# linearity test, attached iff `linearity`, folds into the diagnostics kv.
# ── #70: `estimate threshold` — the GENERAL threshold regression ─────────────
# `estimate_threshold(y, X, q; …)` regresses y on X and splits the sample by a SEPARATE
# threshold variable q. That is genuinely distinct from `estimate setar`, which is the
# self-exciting special case where q = y[t-d] and X is the lag matrix — both return the
# same `ThresholdModel`, so the rendering helpers are shared.
#
# COLUMN PARTITION (a third shape, alongside _load_reg_data and _load_iv_data):
#   y = --dep, q = --threshold-col, X = every OTHER numeric column.
# q MUST be excluded from X — leaving it in makes the regressors collinear with the
# splitting variable, which is a different (and wrong) model, not an error.

"""Load `(y, X, q)` for `estimate threshold`. `--threshold-col` is required and is
EXCLUDED from the regressor matrix; every other numeric column becomes a regressor. No
intercept is prepended (include a `const` column), matching `estimate reg`."""
function _load_threshold_data(data::String, dep::String, threshold_col::String)
    isempty(threshold_col) && throw(CliError("usage/missing",
        "estimate threshold: --threshold-col is required";
        hint="name the variable that splits the sample, e.g. --threshold-col z"))
    df = load_data(data)
    numcols = _numeric_column_names(df)
    isempty(numcols) && throw(CliError("data/invalid",
        "no numeric columns found in $data"))
    depc = isempty(dep) ? numcols[1] : dep
    depc in numcols || throw(CliError("data/column-range",
        "estimate threshold: --dep '$depc' is not a numeric column";
        hint="available: $(join(numcols, ", "))"))
    threshold_col in numcols || throw(CliError("data/column-range",
        "estimate threshold: --threshold-col '$threshold_col' is not a numeric column";
        hint="available: $(join(numcols, ", "))"))
    threshold_col == depc && throw(CliError("usage/invalid",
        "estimate threshold: --threshold-col must differ from --dep (both '$depc')"))
    xcols = [c for c in numcols if c != depc && c != threshold_col]
    isempty(xcols) && throw(CliError("data/invalid",
        "estimate threshold: no regressor columns left after removing --dep and --threshold-col";
        hint="the CSV needs at least one column besides those two"))
    for c in vcat(depc, threshold_col, xcols)
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' has missing values"; hint="clean them with `friedman data dropna`"))
    end
    y = Vector{Float64}(df[!, depc])
    q = Vector{Float64}(df[!, threshold_col])
    X = Matrix{Float64}(df[!, xcols])
    return y, X, q, depc, xcols
end

function _estimate_threshold(; data::String, dep::String="", threshold_col::String="",
        trim::Float64=0.15, reps::Int=1000, ci_level::Float64=0.95,
        het::Bool=false, no_linearity::Bool=false,
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    (0.0 < trim < 0.5) || throw(CliError("usage/invalid",
        "estimate threshold: --trim must be in (0, 0.5) (got $trim)"))
    reps >= 1 || throw(CliError("usage/invalid", "estimate threshold: --reps must be ≥ 1 (got $reps)"))
    # Hansen (2000) critical values are tabulated ONLY for these three levels — anything
    # else makes `_hansen2000_crit` throw an uncaught ArgumentError (same as setar).
    (ci_level == 0.90 || ci_level == 0.95 || ci_level == 0.99) || throw(CliError("usage/invalid",
        "estimate threshold: --ci-level must be exactly 0.90, 0.95, or 0.99 (got $ci_level)"))
    y, X, q, depc, xcols = _load_threshold_data(data, dep, threshold_col)
    _status("Estimating threshold regression: $depc ~ $(join(xcols, " + ")), split by $threshold_col, n=$(length(y)), ci=$ci_level" *
            (het ? ", het bootstrap" : "") * (no_linearity ? ", linearity test skipped" : ""))
    _status()
    model = try
        estimate_threshold(y, X, q; trim=trim, linearity=!no_linearity, reps=reps,
                           ci_level=ci_level, het=het, xnames=xcols, qname=threshold_col,
                           _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "threshold regression"))
    end
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    coef = vcat(
        _threshold_coef_table(model.beta1, model.se1, model.xnames, "regime1 ($threshold_col≤γ)"),
        _threshold_coef_table(model.beta2, model.se2, model.xnames, "regime2 ($threshold_col>γ)"),
    )
    output_result(coef; format=Symbol(format), output=output,
                  title="Threshold Regression Coefficients ($depc)",
                  key="threshold_regression_coefficients")
    diag = Pair{String,Any}[
        "threshold_var"  => threshold_col,
        "gamma"          => round(Float64(model.gamma); digits=6),
        "gamma_ci_lower" => round(Float64(model.gamma_ci[1]); digits=6),
        "gamma_ci_upper" => round(Float64(model.gamma_ci[2]); digits=6),
        "gamma_ci_level" => Float64(model.gamma_ci_level),
        "n"              => model.n,
        "n1"             => model.n1,
        "n2"             => model.n2,
        "ssr"            => round(Float64(model.ssr); digits=6),
        "sigma2"         => round(Float64(model.sigma2); digits=6),
        "aic"            => round(Float64(model.aic); digits=4),
        "bic"            => round(Float64(model.bic); digits=4),
        "is_setar"       => model.is_setar,
    ]
    if model.linearity !== nothing
        lt = model.linearity
        append!(diag, Pair{String,Any}[
            "sup_lm"      => round(Float64(lt.sup_lm); digits=4),
            "pvalue_lm"   => round(Float64(lt.pvalue_lm); digits=4),
            "sup_wald"    => round(Float64(lt.sup_wald); digits=4),
            "pvalue_wald" => round(Float64(lt.pvalue_wald); digits=4),
            "gamma_sup"   => round(Float64(lt.gamma_sup); digits=6),
        ])
    end
    output_kv(diag; format=format, title="Threshold Regression Diagnostics")
    return model
end

function _estimate_setar(; data::String, column::Int=1, p::Int=1, d::String="1",
                          trim::Float64=0.15, reps::Int=1000, ci_level::Float64=0.95,
                          het::Bool=false, no_linearity::Bool=false,
                          plot::Bool=false, plot_save::String="",
                          output::String="", format::String="table")
    p >= 1 || throw(CliError("usage/invalid", "estimate setar: --p must be ≥ 1 (got $p)"))
    (0.0 < trim < 0.5) || throw(CliError("usage/invalid",
        "estimate setar: --trim must be in (0, 0.5) (got $trim)"))
    reps >= 1 || throw(CliError("usage/invalid", "estimate setar: --reps must be ≥ 1 (got $reps)"))
    # Hansen (2000) critical values are tabulated ONLY for these three levels — anything
    # else makes `_hansen2000_crit` throw an uncaught ArgumentError, so guard exactly.
    (ci_level == 0.90 || ci_level == 0.95 || ci_level == 0.99) || throw(CliError("usage/invalid",
        "estimate setar: --ci-level must be exactly 0.90, 0.95, or 0.99 (got $ci_level)"))
    d_arg = _parse_setar_delay(d)
    y, vname = load_univariate_series(data, column)
    _status("Estimating SETAR(2;$p,$p): variable=$vname, obs=$(length(y)), d=$d, ci=$ci_level" *
            (het ? ", het bootstrap" : "") * (no_linearity ? ", linearity test skipped" : ""))
    _status()
    model = try
        estimate_setar(y, p, d_arg; trim=trim, linearity=!no_linearity, reps=reps,
                       ci_level=ci_level, het=het, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "SETAR"))
    end
    coef = vcat(
        _threshold_coef_table(model.beta1, model.se1, model.xnames, "regime1 (q≤γ)"),
        _threshold_coef_table(model.beta2, model.se2, model.xnames, "regime2 (q>γ)"),
    )
    output_result(coef; format=Symbol(format), output=output,
                  title="SETAR(2;$p,$p) Coefficients ($vname)", key="setar_coefficients")
    diag = Pair{String,Any}[
        "gamma"          => round(Float64(model.gamma); digits=6),
        "gamma_ci_lower" => round(Float64(model.gamma_ci[1]); digits=6),
        "gamma_ci_upper" => round(Float64(model.gamma_ci[2]); digits=6),
        "gamma_ci_level" => Float64(model.gamma_ci_level),
        "n"              => model.n,
        "n1"             => model.n1,
        "n2"             => model.n2,
        "ssr"            => round(Float64(model.ssr); digits=6),
        "sigma2"         => round(Float64(model.sigma2); digits=6),
        "aic"            => round(Float64(model.aic); digits=4),
        "bic"            => round(Float64(model.bic); digits=4),
        "p"              => model.p,
        "d"              => model.d,
        "is_setar"       => model.is_setar,
    ]
    if model.linearity !== nothing
        lt = model.linearity
        append!(diag, Pair{String,Any}[
            "sup_lm"      => round(Float64(lt.sup_lm); digits=4),
            "pvalue_lm"   => round(Float64(lt.pvalue_lm); digits=4),
            "sup_wald"    => round(Float64(lt.sup_wald); digits=4),
            "pvalue_wald" => round(Float64(lt.pvalue_wald); digits=4),
            "gamma_sup"   => round(Float64(lt.gamma_sup); digits=6),
        ])
    end
    output_kv(diag; format=format, title="SETAR(2;$p,$p) Diagnostics", key="setar_diagnostics")
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    return model
end

# ── C065b: STAR handler ─────────────────────────────────────────
# Tidy transition-parameters table for a STARModel: `parameter | estimate | std_error |
# z_stat | p_value` (γ then the location(s) c — length 1 for lstr1/estr, 2 for lstr2), with
# large-sample normal z/p (the `_garch_variant`/`dsge` `_normal_cdf` convention). Rendered
# as its own table, distinct from the two regime-weight blocks.
function _star_transition_table(model)
    cnames = length(model.c) == 1 ? String["c"] : String["c$i" for i in 1:length(model.c)]
    params = vcat("γ", cnames)
    est = Float64.(vcat(model.gamma, model.c))
    se  = Float64.(vcat(model.se_gamma, model.se_c))
    z   = [se[i] == 0.0 ? 0.0 : est[i] / se[i] for i in eachindex(est)]
    pv  = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    return DataFrame(parameter=params, estimate=round.(est; digits=6),
                     std_error=round.(se; digits=6),
                     z_stat=round.(z; digits=4), p_value=round.(pv; digits=4))
end

# Every user option is guarded up-front → usage/invalid (before any MEMs call); the estimator
# is try-wrapped → typed CliError via the shared `_nonlinear_error` (never uncaught exit-1). The
# two regime-weight blocks (1−G / G) render via the shared `_threshold_coef_table`; the transition
# parameters (γ, c) via `_star_transition_table`; the LM3 linearity statistics + the optional
# Teräsvirta sequential-selection triple (only for `--type auto`) fold into the diagnostics kv.
function _estimate_star(; data::String, column::Int=1, p::Int=1, d::Int=1,
                         type::String="auto", n_gamma::Int=15, n_c::Int=15,
                         transition_col::Int=0, plot::Bool=false, plot_save::String="",
                         output::String="", format::String="table")
    p >= 1 || throw(CliError("usage/invalid", "estimate star: --p must be ≥ 1 (got $p)"))
    d >= 1 || throw(CliError("usage/invalid", "estimate star: --d must be ≥ 1 (got $d)"))
    n_gamma >= 2 || throw(CliError("usage/invalid", "estimate star: --n-gamma must be ≥ 2 (got $n_gamma)"))
    n_c >= 2 || throw(CliError("usage/invalid", "estimate star: --n-c must be ≥ 2 (got $n_c)"))
    ttype = Symbol(type)
    ttype in (:lstr1, :lstr2, :estr, :auto) || throw(CliError("usage/invalid",
        "estimate star: --type must be one of lstr1|lstr2|estr|auto (got '$type')"))
    y, vname = load_univariate_series(data, column)
    s = nothing
    if transition_col > 0
        s, _ = load_univariate_series(data, transition_col)
        length(s) == length(y) || throw(CliError("data/shape",
            "estimate star: transition variable (column $transition_col) has length $(length(s)), " *
            "but the series has length $(length(y))"))
        std(s) > 0 || throw(CliError("data/invalid",
            "estimate star: transition variable (column $transition_col) is constant; cannot scale γ"))
    end
    _status("Estimating STAR($p) [$type]: variable=$vname, obs=$(length(y)), d=$d" *
            (transition_col > 0 ? ", external transition col=$transition_col" : ", self-exciting")); _status()
    model = try
        estimate_star(y, p; s=s, d=d, type=ttype, n_gamma=n_gamma, n_c=n_c)
    catch e
        throw(_nonlinear_error(e, "STAR"))
    end
    coef = vcat(
        _threshold_coef_table(model.phi1, model.se_phi1, model.znames, "regime1 (G→0)"),
        _threshold_coef_table(model.phi2, model.se_phi2, model.znames, "regime2 (G→1)"),
    )
    output_result(coef; format=Symbol(format), output=output,
                  title="STAR($p) Regime Coefficients ($vname)", key="star_regime_coefficients")
    # Second table to a DISTINCT --output path (mirrors ardl/pmg): a shared file path would let
    # the transition table overwrite the regime-coefficient file. No-op for stdout/envelope.
    output_result(_star_transition_table(model); format=Symbol(format),
                  output=_per_var_output_path(output, "transition"),
                  title="STAR($p) Transition Parameters", key="star_transition_parameters")
    diag = Pair{String,Any}[
        "trans_type"  => string(model.trans_type),
        "sname"       => model.sname,
        "sigma_s"     => round(Float64(model.sigma_s); digits=6),
        "n"           => model.n,
        "p"           => model.p,
        "d"           => model.d,
        "ssr"         => round(Float64(model.ssr); digits=6),
        "sigma2"      => round(Float64(model.sigma2); digits=6),
        "aic"         => round(Float64(model.aic); digits=4),
        "bic"         => round(Float64(model.bic); digits=4),
        "lm3_stat"    => round(Float64(model.lm3_stat); digits=4),
        "lm3_pvalue"  => round(Float64(model.lm3_pvalue); digits=4),
        "lm3_fstat"   => round(Float64(model.lm3_fstat); digits=4),
        "lm3_fpvalue" => round(Float64(model.lm3_fpvalue); digits=4),
        "converged"   => model.converged,
    ]
    if model.sel_pvalues !== nothing
        sp = model.sel_pvalues
        append!(diag, Pair{String,Any}[
            "sel_H04" => round(Float64(sp[1]); digits=4),
            "sel_H03" => round(Float64(sp[2]); digits=4),
            "sel_H02" => round(Float64(sp[3]); digits=4),
        ])
    end
    output_kv(diag; format=format, title="STAR($p) Diagnostics", key="star_diagnostics")
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    return model
end

# ── C065c: Markov-switching (MSRegModel) shared renderer ─────────
# `estimate_ms_ar` (Hamilton mean-switching) and `estimate_ms` (K-state regression) both
# return an MSRegModel, which is NOT in MEMs `_COEF_TABLE_TYPES` → every table is hand-built
# (the documented C051 exception, like io/mgarch/sur). The transition matrix P renders WIDE
# (regime×regime), the parallel of the MGARCH conditional-correlation matrix.

"""Wide K×K transition-matrix DataFrame with a leading `from_regime` label column and
`to_regime\$j` value columns (`P[i,j] = Pr(sₜ=j | s_{t−1}=i)`, rows sum to 1). `makeunique=true`
on BOTH the matrix constructor AND the `insertcols!` guards the recurring wide-matrix
header/label collision (a duplicate/colliding header would otherwise crash the UNWRAPPED
renderer → exit-1), exactly as `_mgarch_corr_df` does for the MGARCH correlation matrix."""
function _ms_transition_df(P::AbstractMatrix)
    K = size(P, 1)
    df = DataFrame(round.(Float64.(P); digits=6), String["to_regime$j" for j in 1:K]; makeunique=true)
    insertcols!(df, 1, :from_regime => String["regime$i" for i in 1:K]; makeunique=true)
    return df
end

"""Per-period regime probabilities as ONE tidy LONG table
(`period | regime | filtered | smoothed`).

`MSRegModel` carries both `filtered_prob` (P(Sₜ=k | y₁..ₜ), real time) and `smoothed_prob`
(P(Sₜ=k | y₁..ₙ), full sample) as `n × K` matrices. This is the headline output of a
Markov-switching fit — "which regime were we in at time t" — so it is emitted alongside the
parameters rather than hidden behind a flag.

LONG rather than one column per regime, for the same reason `predict statespace` is long:
K comes from `--k-regimes`, so a wide table would make the COLUMN SET depend on the user's
option and every consumer would have to discover K before reading. Long keeps the columns
fixed and grows the ROW COUNT instead. Row order matches the state-space renderer
(regime-major, then period)."""
function _ms_prob_table(model)
    K = model.k_regimes
    fp = Float64.(model.filtered_prob)
    sp = Float64.(model.smoothed_prob)
    n = size(sp, 1)
    periods = Int[]; regimes = String[]; filt = Float64[]; smoo = Float64[]
    for k in 1:K, t in 1:n
        push!(periods, t); push!(regimes, "regime$k")
        # filtered_prob can be shorter/empty on a degenerate fit — fall back to the smoothed
        # path rather than indexing out of bounds (an uncaught BoundsError would be exit 1).
        push!(filt, (size(fp, 1) >= t && size(fp, 2) >= k) ? fp[t, k] : sp[t, k])
        push!(smoo, sp[t, k])
    end
    DataFrame("period" => periods, "regime" => regimes,
              "filtered" => round.(filt; digits=6), "smoothed" => round.(smoo; digits=6))
end

"""Shared renderer for a Markov-switching fit (`estimate ms-ar` / `estimate ms`). Emits, in
order: a tidy per-regime coefficient table, a per-regime variance table, the WIDE K×K
transition matrix, the tidy LONG per-period regime-probability table, and a diagnostics kv.
Branches on `model.model_type`:

- `:ms_ar` (Hamilton mean-switching) — one switching-`mu` row per regime (from `model.mu` /
  `model.se_coefs[1,k]`) PLUS a single `common-AR` block for the shared AR coefficients φ
  (from `model.ar` / `model.se_ar`), since only the level switches.
- `:regression` — the full per-regime switching coefficients over `model.xnames` (from
  `model.coefs[:,k]` / `model.se_coefs[:,k]`).

W3/#138: the envelope keys come from `model_type`, not from `label` — the MS-AR label embeds
the AR order, which would put `p` in the address. Deriving the prefix here (rather than taking
it from the caller) keeps `estimate ms-ar` and `estimate ms` from drifting apart.
The 2nd/3rd tables route `--output` to a distinct `_per_var_output_path` so a `-o file` export
is not overwritten by the subsequent table (the C065b multi-table lesson; no-op for
stdout/envelope)."""
function _ms_render(model, format::String, output::String)
    K = model.k_regimes
    label = model.model_type == :ms_ar ? "MS-AR($(model.p))" : "MS Regression"
    kp = model.model_type == :ms_ar ? "ms_ar" : "ms_regression"
    yname = String(model.yname)
    blocks = DataFrame[]
    if model.model_type == :ms_ar
        for k in 1:K
            push!(blocks, _threshold_coef_table(
                Float64[model.mu[k]], Float64[model.se_coefs[1, k]],
                String["mu"], "regime$k"))
        end
        # AR coefficients φ are COMMON across regimes → one separate block
        if !isempty(model.ar)
            push!(blocks, _threshold_coef_table(
                Float64.(model.ar), Float64.(model.se_ar),
                String["φ$i" for i in 1:length(model.ar)], "common-AR"))
        end
    else
        for k in 1:K
            push!(blocks, _threshold_coef_table(
                Float64.(model.coefs[:, k]), Float64.(model.se_coefs[:, k]),
                model.xnames, "regime$k"))
        end
    end
    output_result(vcat(blocks...); format=Symbol(format), output=output,
                  title="$label Regime Coefficients ($yname)",
                  key="$(kp)_regime_coefficients")
    vardf = DataFrame(regime=String["regime$k" for k in 1:K],
                      sigma2=round.(Float64.(model.sigma2); digits=6),
                      std_error=round.(Float64.(model.se_sigma2); digits=6))
    output_result(vardf; format=Symbol(format),
                  output=_per_var_output_path(output, "variance"),
                  title="$label Regime Variances", key="$(kp)_regime_variances")
    output_result(_ms_transition_df(model.P); format=Symbol(format),
                  output=_per_var_output_path(output, "transition"),
                  title="$label Transition Matrix", key="$(kp)_transition_matrix")
    output_result(_ms_prob_table(model); format=Symbol(format),
                  output=_per_var_output_path(output, "probabilities"),
                  title="$label Regime Probabilities", key="$(kp)_regime_probabilities")
    diag = Pair{String,Any}[
        "loglik"   => round(Float64(model.loglik); digits=4),
        "n_params" => model.n_params,
        "aic"      => round(Float64(model.aic); digits=4),
        "bic"      => round(Float64(model.bic); digits=4),
    ]
    for k in 1:K
        push!(diag, "ergodic_$k" => round(Float64(model.ergodic[k]); digits=6))
    end
    for k in 1:K
        push!(diag, "expected_duration_$k" => round(Float64(model.expected_durations[k]); digits=4))
    end
    append!(diag, Pair{String,Any}[
        "switching_var" => model.switching_var,
        "switching_ar"  => model.switching_ar,
        "converged"     => model.converged,
        "iterations"    => model.iterations,
    ])
    output_kv(diag; format=format, title="$label Diagnostics", key="$(kp)_diagnostics")
    return model
end

# ── C065c: MS-AR handler ─────────────────────────────────────────
# Every option guarded up-front → usage/invalid (before any MEMs call); the estimator is
# try-wrapped → typed CliError via the shared `_nonlinear_error` (never uncaught exit-1).
# `switching_variance` default FALSE (Hamilton) — the `--switching-variance` flag turns it on.
function _estimate_ms_ar(; data::String, column::Int=1, p::Int=1, k_regimes::Int=2,
                          switching_variance::Bool=false, max_iter::Int=1000,
                          plot::Bool=false, plot_save::String="",
                          output::String="", format::String="table")
    p >= 1 || throw(CliError("usage/invalid", "estimate ms-ar: --p must be ≥ 1 (got $p)"))
    k_regimes >= 2 || throw(CliError("usage/invalid",
        "estimate ms-ar: --k-regimes must be ≥ 2 (got $k_regimes)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "estimate ms-ar: --max-iter must be ≥ 1 (got $max_iter)"))
    y, vname = load_univariate_series(data, column)
    _status("Estimating MS-AR($p) [$k_regimes regimes]: variable=$vname, obs=$(length(y))" *
            (switching_variance ? ", switching variance" : ", common variance")); _status()
    model = try
        estimate_ms_ar(y, p; k_regimes=k_regimes, switching_variance=switching_variance,
                       max_iter=max_iter, yname=vname)
    catch e
        throw(_nonlinear_error(e, "MS-AR"))
    end
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    return _ms_render(model, format, output)
end

# ── C065c: MS regression handler ─────────────────────────────────
# Loaded via the hardened `_load_reg_data` (y + numeric regressors, NO auto-intercept). When
# the dependent variable is the ONLY numeric column, `_load_reg_data` raises data/invalid
# ("no regressor columns remaining") — that specific case routes to the single-arg intercept-only
# dispatch `estimate_ms(y; …)` (X === nothing). Every option guarded up-front → usage/invalid;
# the estimator is try-wrapped → typed CliError. `switching_variance` default TRUE — the
# `--no-switching-variance` flag turns it off.
"""Load `(y, X, xcols, intercept_only)` for the Markov-switching REGRESSION leaves.

Routes to the single-argument intercept-only `estimate_ms(y; …)` when the dependent variable
is the only numeric column. The catch is deliberately NARROW — only `_load_reg_data`'s
`data/invalid` "no regressor columns" is converted; every other class rethrows, so a dep-only
CSV with a missing cell still surfaces as `data/missing-values` and an all-non-numeric CSV
still surfaces as `data/invalid`. Shared by `estimate ms` and `residuals ms` so the two cannot
drift into fitting different models from the same arguments."""
function _ms_load_data(data::String, dep::String)
    try
        y, X, xcols = _load_reg_data(data, dep)
        return y, X, xcols, false
    catch e
        (e isa CliError && e.code == "data/invalid" &&
            occursin("no regressor columns", e.message)) || rethrow(e)
        df = load_data(data)
        numcols = variable_names(df)
        dep_col = isempty(dep) ? numcols[1] : dep
        idx = findfirst(==(dep_col), numcols)
        idx === nothing && rethrow(e)
        y, _ = load_univariate_series(data, idx)
        return y, nothing, String["const"], true
    end
end

function _estimate_ms(; data::String, dep::String="", k_regimes::Int=2,
                       no_switching_variance::Bool=false, max_iter::Int=500,
                       tol::Float64=1e-8, plot::Bool=false, plot_save::String="",
                       output::String="", format::String="table")
    k_regimes >= 2 || throw(CliError("usage/invalid",
        "estimate ms: --k-regimes must be ≥ 2 (got $k_regimes)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "estimate ms: --max-iter must be ≥ 1 (got $max_iter)"))
    tol > 0 || throw(CliError("usage/invalid", "estimate ms: --tol must be > 0 (got $tol)"))
    sv = !no_switching_variance
    y, X, xcols, intercept_only = _ms_load_data(data, dep)
    _status("Estimating MS regression [$k_regimes regimes]: obs=$(length(y)), " *
            (intercept_only ? "intercept-only" : "regressors=$(length(xcols))") *
            (sv ? ", switching variance" : ", common variance")); _status()
    model = try
        X === nothing ?
            estimate_ms(y; k_regimes=k_regimes, switching_variance=sv, max_iter=max_iter, tol=tol) :
            estimate_ms(y, X; k_regimes=k_regimes, switching_variance=sv, max_iter=max_iter,
                        tol=tol, xnames=xcols)
    catch e
        throw(_nonlinear_error(e, "MS regression"))
    end
    _maybe_plot(model; plot=plot, plot_save=plot_save)
    return _ms_render(model, format, output)
end

# ── #70 remainder: `residuals` for the nonlinear time-series models ──────────
# `StatsAPI.residuals` EXISTS upstream for all three nonlinear types (nonlinear/types.jl:256
# ThresholdModel, :450 STARModel, :618 MSRegModel), which is why these four leaves are in
# scope. There is deliberately NO matching `predict`: none of the three EXPOSES
# `predict`/`fitted`. Note this is an availability gap, NOT an ill-defined quantity — for the MS
# models the fitted series is the regime-probability-weighted conditional mean, which MEMs
# already computes internally (markov_switching.jl:387, :617) to build these very residuals and
# then discards. Filed upstream as MEMs#510 (which also covers the missing MS `forecast`);
# recomputing it here would risk diverging from whatever definition upstream publishes.
#
# Each handler mirrors its `estimate` sibling's option set so any fit that changes the
# residuals can be reproduced (the #73 lesson: a verb that cannot express `--p` silently pins
# a different model). Options that affect ONLY the attached inference — SETAR's `--reps`,
# `--ci-level`, `--het` drive the Hansen bootstrap and threshold CI, never the residuals — are
# omitted, and the refit passes `linearity=false` to skip the bootstrap entirely.

"""One tidy `period | residual` table for a nonlinear-TS fit.

`period` is the EFFECTIVE-sample index (1..n_eff): SETAR/STAR/MS-AR all drop the first `p`
(or `max(p,d)`) observations to build their lag matrix, so residual `t` is not calendar `t`."""
function _nonlinear_resid_output(resid, label::String, vname::String,
                                 format::String, output::String; key::String="")
    r = Float64.(collect(resid))
    output_result(DataFrame("period" => collect(1:length(r)),
                            "residual" => round.(r; digits=6));
        format=Symbol(format), output=output,
        title="$label Residuals ($vname)", key=key)
    return r
end

"""Refit a SETAR for `residuals setar`, mirroring `_estimate_setar`'s guards and load path.
`linearity=false`: the Hansen (1996) bootstrap does not affect the residuals and is the
single most expensive part of the fit."""
function _setar_refit(data::String, column::Int, p::Int, d::String, trim::Float64)
    p >= 1 || throw(CliError("usage/invalid", "residuals setar: --p must be ≥ 1 (got $p)"))
    (0.0 < trim < 0.5) || throw(CliError("usage/invalid",
        "residuals setar: --trim must be in (0, 0.5) (got $trim)"))
    d_arg = _parse_setar_delay(d)
    y, vname = load_univariate_series(data, column)
    model = try
        estimate_setar(y, p, d_arg; trim=trim, linearity=false, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "SETAR"))
    end
    return model, vname
end

function _residuals_setar(; data::String="", column::Int=1, p::Int=1, d::String="1",
        trim::Float64=0.15, output::String="", format::String="table", model=nothing)
    m, vname = model === nothing ? _setar_refit(data, column, p, d, trim) : (model, "model")
    _status("SETAR(2;$p,$p) residuals: variable=$vname, effective obs=$(length(m.residuals))"); _status()
    return _nonlinear_resid_output(residuals(m), "SETAR(2;$p,$p)", vname, format, output;
                                   key="setar_residuals")
end

"""Refit a STAR for `residuals star`, mirroring `_estimate_star` exactly (including the
external-transition-variable path and its length/constant guards)."""
function _star_refit(data::String, column::Int, p::Int, d::Int, type::String,
                     n_gamma::Int, n_c::Int, transition_col::Int)
    p >= 1 || throw(CliError("usage/invalid", "residuals star: --p must be ≥ 1 (got $p)"))
    d >= 1 || throw(CliError("usage/invalid", "residuals star: --d must be ≥ 1 (got $d)"))
    n_gamma >= 2 || throw(CliError("usage/invalid", "residuals star: --n-gamma must be ≥ 2 (got $n_gamma)"))
    n_c >= 2 || throw(CliError("usage/invalid", "residuals star: --n-c must be ≥ 2 (got $n_c)"))
    ttype = Symbol(type)
    ttype in (:lstr1, :lstr2, :estr, :auto) || throw(CliError("usage/invalid",
        "residuals star: --type must be one of lstr1|lstr2|estr|auto (got '$type')"))
    y, vname = load_univariate_series(data, column)
    s = nothing
    if transition_col > 0
        s, _ = load_univariate_series(data, transition_col)
        length(s) == length(y) || throw(CliError("data/shape",
            "residuals star: transition variable (column $transition_col) has length $(length(s)), " *
            "but the series has length $(length(y))"))
        std(s) > 0 || throw(CliError("data/invalid",
            "residuals star: transition variable (column $transition_col) is constant; cannot scale γ"))
    end
    model = try
        estimate_star(y, p; s=s, d=d, type=ttype, n_gamma=n_gamma, n_c=n_c)
    catch e
        throw(_nonlinear_error(e, "STAR"))
    end
    return model, vname
end

function _residuals_star(; data::String="", column::Int=1, p::Int=1, d::Int=1,
        type::String="auto", n_gamma::Int=15, n_c::Int=15, transition_col::Int=0,
        output::String="", format::String="table", model=nothing)
    m, vname = model === nothing ?
        _star_refit(data, column, p, d, type, n_gamma, n_c, transition_col) : (model, "model")
    _status("STAR($p) residuals: variable=$vname, effective obs=$(length(m.residuals))"); _status()
    return _nonlinear_resid_output(residuals(m), "STAR($p)", vname, format, output;
                                   key="star_residuals")
end

"""Refit an MS-AR for `residuals ms-ar`, mirroring `_estimate_ms_ar`."""
function _ms_ar_refit(data::String, column::Int, p::Int, k_regimes::Int,
                      switching_variance::Bool, max_iter::Int)
    p >= 1 || throw(CliError("usage/invalid", "residuals ms-ar: --p must be ≥ 1 (got $p)"))
    k_regimes >= 2 || throw(CliError("usage/invalid",
        "residuals ms-ar: --k-regimes must be ≥ 2 (got $k_regimes)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "residuals ms-ar: --max-iter must be ≥ 1 (got $max_iter)"))
    y, vname = load_univariate_series(data, column)
    model = try
        estimate_ms_ar(y, p; k_regimes=k_regimes, switching_variance=switching_variance,
                       max_iter=max_iter, yname=vname)
    catch e
        throw(_nonlinear_error(e, "MS-AR"))
    end
    return model, vname
end

function _residuals_ms_ar(; data::String="", column::Int=1, p::Int=1, k_regimes::Int=2,
        switching_variance::Bool=false, max_iter::Int=1000,
        output::String="", format::String="table", model=nothing)
    m, vname = model === nothing ?
        _ms_ar_refit(data, column, p, k_regimes, switching_variance, max_iter) : (model, "model")
    _status("MS-AR($p) [$k_regimes regimes] residuals: variable=$vname, effective obs=$(length(m.residuals))"); _status()
    return _nonlinear_resid_output(residuals(m), "MS-AR($p)", vname, format, output;
                                   key="ms_ar_residuals")
end

"""Refit an MS regression for `residuals ms`, reusing `_ms_load_data` so the intercept-only
dispatch matches `estimate ms` exactly."""
function _ms_refit(data::String, dep::String, k_regimes::Int, sv::Bool,
                   max_iter::Int, tol::Float64)
    k_regimes >= 2 || throw(CliError("usage/invalid",
        "residuals ms: --k-regimes must be ≥ 2 (got $k_regimes)"))
    max_iter >= 1 || throw(CliError("usage/invalid",
        "residuals ms: --max-iter must be ≥ 1 (got $max_iter)"))
    tol > 0 || throw(CliError("usage/invalid", "residuals ms: --tol must be > 0 (got $tol)"))
    y, X, xcols, intercept_only = _ms_load_data(data, dep)
    model = try
        X === nothing ?
            estimate_ms(y; k_regimes=k_regimes, switching_variance=sv, max_iter=max_iter, tol=tol) :
            estimate_ms(y, X; k_regimes=k_regimes, switching_variance=sv, max_iter=max_iter,
                        tol=tol, xnames=xcols)
    catch e
        throw(_nonlinear_error(e, "MS regression"))
    end
    # `intercept_only` is returned alongside the model because it CANNOT be recovered from
    # the fit: a one-regressor design also has size(coefs, 1) == 1, so keying off the
    # coefficient count would silently treat "one real regressor" as "intercept only" and
    # let `forecast ms` invent a column of ones instead of demanding future values.
    return model, intercept_only
end

function _residuals_ms(; data::String="", dep::String="", k_regimes::Int=2,
        no_switching_variance::Bool=false, max_iter::Int=500, tol::Float64=1e-8,
        output::String="", format::String="table", model=nothing)
    m = model === nothing ?
        first(_ms_refit(data, dep, k_regimes, !no_switching_variance, max_iter, tol)) : model
    vname = model === nothing ? String(m.yname) : "model"
    _status("MS regression [$k_regimes regimes] residuals: obs=$(length(m.residuals))"); _status()
    return _nonlinear_resid_output(residuals(m), "MS Regression", vname, format, output;
                                   key="ms_regression_residuals")
end

# ── W3/#101: MS predict + forecast (un-gated by MEMs#510) ────────────────────
#
# `predict(m; probs=:smoothed|:filtered)` returns the regime-weighted fitted mean.
# Upstream warns that `y - predict(m; probs=:filtered)` is NOT `residuals(m)`: the
# published residuals are smoothed-weighted, and the filtered mean uses strictly less
# information. The two verbs are therefore genuinely different outputs, not aliases.

"""Shared renderer for `predict ms|ms-ar` — regime-weighted fitted values."""
function _ms_predict_output(m, probs::String, label::String, vname::String,
                            format::String, output::String; key_prefix::String="")
    fitted = try
        predict(m; probs=Symbol(probs))
    catch e
        throw(_nonlinear_error(e, "$label predict"))
    end
    df = DataFrame(t=1:length(fitted), fitted=round.(Float64.(fitted); digits=6))
    # `--probs` selects which weighting produced the same column set, so it stays in the
    # title and out of the key.
    return output_result(df; format=Symbol(format), output=output,
                         title="$label Fitted Values ($probs) for $vname",
                         key="$(key_prefix)_fitted_values")
end

"""Shared renderer for `forecast ms|ms-ar`.

Emits the tidy forecast path plus the `h x K` predicted regime probabilities. The second
table MUST take a distinct `_per_var_output_path`, or `--output <file>` silently drops the
first one. NO `_maybe_plot`: MEMs 0.7.2 ships no `plot_result(::MSForecast)` recipe — the
same gap that keeps the flags off `forecast setar|star` — so the leaves declare no plot flags.
"""
function _ms_forecast_output(fc, label::String, vname::String, horizons::Int,
                             ci_level::Float64, format::String, output::String;
                             key_prefix::String="")
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="$label Forecast for $vname (h=$horizons, $(Int(round(ci_level*100)))% CI)",
                  key="$(key_prefix)_forecast")
    K = size(fc.regime_prob, 2)
    rp = DataFrame(horizon=1:size(fc.regime_prob, 1))
    for k in 1:K
        rp[!, "regime$k"] = round.(Float64.(fc.regime_prob[:, k]); digits=6)
    end
    return output_result(rp; format=Symbol(format),
                         output=_per_var_output_path(output, "regime-probabilities"),
                         title="$label Predicted Regime Probabilities",
                         key="$(key_prefix)_predicted_regime_probabilities")
end

function _predict_ms_ar(; data::String="", column::Int=1, p::Int=1, k_regimes::Int=2,
        switching_variance::Bool=false, max_iter::Int=1000, probs::String="smoothed",
        output::String="", format::String="table", model=nothing)
    m, vname = model === nothing ?
        _ms_ar_refit(data, column, p, k_regimes, switching_variance, max_iter) : (model, "model")
    _status("MS-AR($p) [$k_regimes regimes] fitted ($probs): variable=$vname"); _status()
    return _ms_predict_output(m, probs, "MS-AR($p)", vname, format, output;
                              key_prefix="ms_ar")
end

function _predict_ms(; data::String="", dep::String="", k_regimes::Int=2,
        no_switching_variance::Bool=false, max_iter::Int=500, tol::Float64=1e-8,
        probs::String="smoothed", output::String="", format::String="table", model=nothing)
    m = model === nothing ?
        first(_ms_refit(data, dep, k_regimes, !no_switching_variance, max_iter, tol)) : model
    vname = model === nothing ? String(m.yname) : "model"
    _status("MS regression [$k_regimes regimes] fitted ($probs)"); _status()
    return _ms_predict_output(m, probs, "MS Regression", vname, format, output;
                              key_prefix="ms_regression")
end

function _forecast_ms_ar(; data::String="", result=nothing, column::Int=1, p::Int=1, k_regimes::Int=2,
        switching_variance::Bool=false, max_iter::Int=1000, horizons::Int=12,
        reps::Int=1000, ci_level::Float64=0.90,
        output::String="", format::String="table", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast ms-ar")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="MS-AR Forecast", key="ms_ar_forecast")
    horizons >= 1 || throw(CliError("usage/invalid",
        "forecast ms-ar: --horizons must be ≥ 1 (got $horizons)"))
    reps >= 1 || throw(CliError("usage/invalid", "forecast ms-ar: --reps must be ≥ 1 (got $reps)"))
    (0 < ci_level < 1) || throw(CliError("usage/invalid",
        "forecast ms-ar: --ci-level must satisfy 0 < level < 1 (got $ci_level)"))
    m, vname = model === nothing ?
        _ms_ar_refit(data, column, p, k_regimes, switching_variance, max_iter) : (model, "model")
    _status("MS-AR($p) forecast (h=$horizons): variable=$vname, ci=$ci_level"); _status()
    fc = try
        forecast(m, horizons; reps=reps, level=ci_level, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "MS-AR forecast"))
    end
    _ms_forecast_output(fc, "MS-AR($p)", vname, horizons, ci_level, format, output;
                        key_prefix="ms_ar")
    return (; model=m, result=fc)
end

function _forecast_ms(; data::String="", result=nothing, dep::String="", k_regimes::Int=2,
        no_switching_variance::Bool=false, max_iter::Int=500, tol::Float64=1e-8,
        horizons::Int=12, x_future::String="", reps::Int=1000, ci_level::Float64=0.90,
        output::String="", format::String="table", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast ms")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="MS Forecast", key="ms_forecast")
    horizons >= 1 || throw(CliError("usage/invalid",
        "forecast ms: --horizons must be ≥ 1 (got $horizons)"))
    reps >= 1 || throw(CliError("usage/invalid", "forecast ms: --reps must be ≥ 1 (got $reps)"))
    (0 < ci_level < 1) || throw(CliError("usage/invalid",
        "forecast ms: --ci-level must satisfy 0 < level < 1 (got $ci_level)"))
    m, intercept_only = if model === nothing
        _ms_refit(data, dep, k_regimes, !no_switching_variance, max_iter, tol)
    else
        # From a saved handle there is no CSV to re-inspect. `_ms_load_data` labels the
        # intercept-only design "const" while the regressor path passes the real column
        # names through, so that label is the only signal left.
        (model, model.xnames == String["const"])
    end
    vname = model === nothing ? String(m.yname) : "model"

    # A switching REGRESSION cannot project itself: upstream requires an h x k matrix of
    # FUTURE regressors. The one case that needs no input is the intercept-only fit —
    # `estimate_ms(y)` delegates to `estimate_ms(y, ones(n,1))` — where the future design
    # is just a column of ones, so `--horizons` alone is enough there. NOTE this keys off
    # `intercept_only`, NOT `size(coefs,1) == 1`: a single real regressor also has k == 1,
    # and treating that as intercept-only would forecast against a fabricated ones-column.
    k = size(m.coefs, 1)
    X_new = if !isempty(x_future)
        Xf, _ = _load_ms_future_x(x_future, k)
        Xf
    elseif intercept_only
        ones(Float64, horizons, 1)
    else
        throw(CliError("usage/missing",
            "forecast ms: this model has $k regressor$(k == 1 ? "" : "s"), so future values are required";
            hint="pass --x-future <csv> with $horizons row$(horizons == 1 ? "" : "s") and " *
                 "$k column$(k == 1 ? "" : "s") " *
                 "(only an intercept-only model can be projected from --horizons alone)"))
    end

    _status("MS regression forecast (h=$(size(X_new,1))): ci=$ci_level"); _status()
    fc = try
        forecast(m, X_new; reps=reps, level=ci_level, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "MS regression forecast"))
    end
    _ms_forecast_output(fc, "MS Regression", vname, size(X_new, 1), ci_level,
                        format, output; key_prefix="ms_regression")
    return (; model=m, result=fc)
end

"""Load the future-regressor matrix for `forecast ms`.

Every raw conversion is guarded BEFORE `Matrix{Float64}`: a shared loader that hands a
missing cell straight to the converter is the recurring untyped-exit-1 site in this repo.
"""
function _load_ms_future_x(path::String, k::Int)
    df = load_data(path)
    ncol(df) == 0 && throw(CliError("data/empty", "forecast ms: --x-future has no columns: $path"))
    nrow(df) == 0 && throw(CliError("data/empty", "forecast ms: --x-future has no rows: $path"))
    for c in names(df), v in df[!, c]
        (v === missing || (v isa Real && !isfinite(v))) && throw(CliError("data/missing-value",
            "forecast ms: --x-future contains a missing or non-finite value in column '$c'"))
    end
    X = try
        Matrix{Float64}(df)
    catch e
        throw(CliError("data/invalid",
            "forecast ms: --x-future must be all-numeric"; hint=sprint(showerror, e)))
    end
    size(X, 2) == k || throw(CliError("data/shape",
        "forecast ms: --x-future must have $k column$(k == 1 ? "" : "s") to match the fitted model, got $(size(X,2))";
        hint=k == 1 ? "an intercept-only model expects a single column of ones" : ""))
    return X, names(df)
end

# ── W7/#109: TVP-VAR-SV (Primiceri 2005) and MF-VAR (Schorfheide-Song 2015) ──
#
# Neither TVPVARPosterior nor MFVARPosterior has a plot_result recipe on the 0.7.2 tag
# (checked against src/plotting/), so NEITHER leaf advertises --plot. Both estimators take
# `rng`, not `seed`, so per-estimator seeding is unavailable; the global Random.seed! in
# run_cli still makes runs reproducible, and the docs say so rather than implying a
# manifest-recorded seed the result cannot carry.

"""
    _band_long_table(mu, qs, varnames, levels; index_name) -> DataFrame

Tidy long form for the `(mean, quantiles)` pairs returned by `volatility_path`/`latent_path`:
`mu` is `T × n` and `qs` is `T × n × n_q`, so one row per (period, variable) with a column
per requested quantile. C051 tidy convention — never a wide T × n block.
"""
function _band_long_table(mu::AbstractMatrix, qs::AbstractArray{<:Real,3},
                          varnames::Vector{String}, levels::Vector{Float64};
                          index_name::Symbol=:period)
    T_, n = size(mu)
    rows = T_ * n
    df = DataFrame(index_name => repeat(1:T_, outer=n),
                   :variable => repeat(varnames; inner=T_),
                   :mean => vec(Float64.(mu)))
    for (q, lv) in enumerate(levels)
        # `q16`/`q50`/`q84` — the level, not the index, so a reader can tell which band a
        # column is without consulting the invocation.
        df[!, Symbol("q", string(round(Int, lv * 100)))] = vec(Float64.(@view qs[:, :, q]))
    end
    return df
end

const _TVP_QUANTILES = [0.16, 0.5, 0.84]

function _load_and_estimate_tvpvar(data::String, lags::Int, draws::Int, burnin::Int,
                                   thin::Int, n_train::Int, k_q::Float64, k_s::Float64,
                                   k_w::Float64, no_tvp::Bool, no_sv::Bool)
    lags >= 1 || throw(CliError("usage/invalid", "--lags must be ≥ 1 (got $lags)"))
    draws >= 1 || throw(CliError("usage/invalid", "--draws must be ≥ 1 (got $draws)"))
    burnin >= 0 || throw(CliError("usage/invalid", "--burnin must be ≥ 0 (got $burnin)"))
    thin >= 1 || throw(CliError("usage/invalid", "--thin must be ≥ 1 (got $thin)"))
    n_train >= 0 || throw(CliError("usage/invalid", "--n-train must be ≥ 0 (got $n_train)"))
    for (nm, v) in (("--k-q", k_q), ("--k-s", k_s), ("--k-w", k_w))
        v > 0 || throw(CliError("usage/invalid", "$nm must be > 0 (got $v)"))
    end

    Y, varnames = load_multivariate_data(data)
    size(Y, 2) >= 2 || throw(CliError("data/shape",
        "TVP-VAR requires at least 2 variables, got $(size(Y, 2))"))

    _status("Estimating TVP-VAR($lags): $(size(Y,2)) variables, $draws draws " *
            "($burnin burn-in, thin $thin)")
    post = try
        estimate_tvpvar(Y, lags; tvp=!no_tvp, sv=!no_sv, n_draws=draws, n_burn=burnin,
                        thin=thin, n_train=n_train, k_Q=k_q, k_S=k_s, k_W=k_w,
                        varnames=varnames, _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "estimate tvpvar"))
    end
    return post, varnames
end

function _estimate_tvpvar(; data::String, lags::Int=2, draws::Int=2000, burnin::Int=1000,
                           thin::Int=1, n_train::Int=0,
                           k_q::Float64=0.01, k_s::Float64=0.1, k_w::Float64=0.01,
                           no_tvp::Bool=false, no_sv::Bool=false,
                           output::String="", format::String="table")
    post, varnames = _load_and_estimate_tvpvar(data, lags, draws, burnin, thin, n_train,
                                               k_q, k_s, k_w, no_tvp, no_sv)
    _status_report(() -> report(post))

    mu, qs = volatility_path(post; quantile_levels=_TVP_QUANTILES)
    # σ_{i,t} = exp(h_{i,t}/2): volatility_path already converts the stored log-VARIANCE
    # state to a standard deviation, so these are directly comparable with the data's units.
    output_result(_band_long_table(mu, qs, varnames, _TVP_QUANTILES);
                  format=Symbol(format), output=output,
                  title="TVP-VAR Stochastic Volatility Path (posterior sd, 68% band)")

    output_kv(Pair{String,Any}[
        "lags"      => post.p,
        "variables" => post.n,
        "T_eff"     => post.T_eff,
        "n_train"   => post.n_train,
        "draws"     => size(post.B_draws, 1),
        "tvp"       => post.tvp,
        "sv"        => post.sv,
    ]; format=format, title="TVP-VAR Specification")
    return post
end

function _irf_tvpvar(; data::String="", result=nothing, date::Int=0, horizons::Int=20, lags::Int=2,
                      draws::Int=2000, burnin::Int=1000, thin::Int=1, n_train::Int=0,
                      k_q::Float64=0.01, k_s::Float64=0.1, k_w::Float64=0.01,
                      no_tvp::Bool=false, no_sv::Bool=false,
                      irf_draws::Int=500, shock::Int=1, no_stationary_only::Bool=false,
                      output::String="", format::String="table", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="irf tvpvar",
                            horizons, horizons_default=20)
    loaded === nothing || return _rerender_irf_result(loaded; format, output,
        title="Impulse Responses", key="tvpvar_irf", shock)
    horizons >= 1 || throw(CliError("usage/invalid", "--horizons must be ≥ 1 (got $horizons)"))
    irf_draws >= 1 || throw(CliError("usage/invalid", "--irf-draws must be ≥ 1 (got $irf_draws)"))
    # The date-t IRF is the entire point of a TVP model, so --date is required rather than
    # silently defaulting to the end of the sample. Check it BEFORE the Gibbs sampler runs —
    # the upper bound needs T_eff and so cannot be known until after estimation, but a
    # missing --date can, and making the user sit through a full MCMC to be told they forgot
    # an argument is indefensible.
    date == 0 && throw(CliError("usage/missing",
        "irf tvpvar: --date is required — a TVP-VAR has a different IRF at every date";
        hint="--date indexes the effective sample (1:T_eff, after lags and any training " *
             "observations); run `estimate tvpvar` to see T_eff"))
    date >= 1 || throw(CliError("usage/invalid",
        "irf tvpvar: --date must be ≥ 1, got $date"))

    if isnothing(model)
        post, varnames = _load_and_estimate_tvpvar(data, lags, draws, burnin, thin, n_train,
                                                   k_q, k_s, k_w, no_tvp, no_sv)
    else
        post = model
        varnames = post.varnames
    end

    date <= post.T_eff || throw(CliError("usage/invalid",
        "irf tvpvar: --date must be in 1:$(post.T_eff), got $date"))
    1 <= shock <= post.n || throw(CliError("usage/invalid",
        "irf tvpvar: --shock must be in 1:$(post.n), got $shock"))

    _status("Computing TVP-VAR IRF at date $date (horizon $horizons)")
    birf = try
        irf(post, horizons; t=date, n_draws=irf_draws,
            stationary_only=!no_stationary_only)
    catch e
        e isa CliError && rethrow()
        # Every draw explosive at this date is a bare error() upstream. It is a real
        # modelling outcome, not a bug, so name the escape hatch.
        throw(CliError("model/error",
            "TVP-VAR IRF failed at date $date: $(sprint(showerror, e))";
            hint="all posterior draws were explosive at this date; try another --date, " *
                 "or --no-stationary-only to include explosive draws"))
    end

    shock_name = birf.shocks[shock]
    df = long_table(birf)
    df = df[df.shock .== shock_name, :]
    output_result(df; format=Symbol(format), output=output,
                  title="TVP-VAR IRF at date $date to $shock_name (68% credible interval)",
                  key="tvpvar_irf")
    return (; model=post, result=birf)
end

const _MF_AGGREGATIONS = ("stock", "flow", "average", "growth")

"""
    _load_mfvar_data(path) -> (Matrix{Float64}, varnames)

Mixed-frequency loader. Unlike every other multivariate loader, MISSING CELLS ARE THE POINT:
a low-frequency series is observed once per `--freq-ratio` high-frequency periods and blank
in between, and `estimate_mfvar` expects those gaps as `NaN`. So this deliberately does NOT
reject missing values the way `load_multivariate_data` does — it converts them.

`estimate_mfvar` still requires the high-frequency columns to be complete, and validates that
itself; the CLI maps the resulting ArgumentError to a typed data error.
"""
function _load_mfvar_data(path::String)
    df = load_data(path)
    isempty(names(df)) && throw(CliError("data/empty", "no columns in $path"))
    numeric = [nm for nm in names(df) if eltype(df[!, nm]) <: Union{Missing,Real}]
    isempty(numeric) && throw(CliError("data/invalid",
        "no numeric columns in $path"))
    m = Matrix{Float64}(undef, nrow(df), length(numeric))
    for (j, nm) in enumerate(numeric)
        col = df[!, nm]
        for i in 1:nrow(df)
            v = col[i]
            m[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end
    return m, String.(numeric)
end

function _estimate_mfvar(; data::String, lags::Int=2, low_freq::String="",
                          freq_ratio::Int=3, aggregation::String="growth",
                          draws::Int=1000, burnin::Int=500, prior::String="minnesota",
                          output::String="", format::String="table")
    lags >= 1 || throw(CliError("usage/invalid", "--lags must be ≥ 1 (got $lags)"))
    draws >= 1 || throw(CliError("usage/invalid", "--draws must be ≥ 1 (got $draws)"))
    burnin >= 0 || throw(CliError("usage/invalid", "--burnin must be ≥ 0 (got $burnin)"))
    freq_ratio >= 1 || throw(CliError("usage/invalid",
        "--freq-ratio must be ≥ 1 (got $freq_ratio)"))
    pr = lowercase(strip(prior))
    pr in ("minnesota", "diffuse") || throw(CliError("usage/invalid-option",
        "invalid --prior '$prior'; must be minnesota or diffuse"))

    Y, varnames = _load_mfvar_data(data)
    n = size(Y, 2)

    # --low-freq is 1-based column indices; default to every column that actually has gaps,
    # which is what a mixed-frequency CSV looks like in practice.
    lf = if isempty(strip(low_freq))
        found = [j for j in 1:n if any(isnan, @view Y[:, j])]
        isempty(found) && throw(CliError("usage/missing",
            "no low-frequency series: --low-freq was not given and no column has gaps";
            hint="a mixed-frequency CSV leaves the low-frequency series blank between " *
                 "observations, or name the columns' indices with --low-freq 2,3"))
        _status("--low-freq not given; inferred from gaps: $(join(found, ","))")
        found
    else
        idx = Int[]
        for tok in split(low_freq, ',')
            s = strip(tok)
            isempty(s) && continue
            v = tryparse(Int, s)
            v === nothing && throw(CliError("usage/invalid",
                "--low-freq must be comma-separated column indices, got '$low_freq'"))
            push!(idx, v)
        end
        isempty(idx) && throw(CliError("usage/invalid", "--low-freq is empty"))
        all(1 .<= idx .<= n) || throw(CliError("usage/invalid",
            "--low-freq indices must be in 1:$n, got $(idx)"))
        length(unique(idx)) == length(idx) || throw(CliError("usage/invalid",
            "--low-freq must not repeat an index"))
        idx
    end

    aggs = String[]
    for tok in split(aggregation, ',')
        s = lowercase(strip(tok))
        isempty(s) && continue
        s in _MF_AGGREGATIONS || throw(CliError("usage/invalid-option",
            "invalid --aggregation '$s'; must be one of $(join(_MF_AGGREGATIONS, "|"))"))
        push!(aggs, s)
    end
    isempty(aggs) && throw(CliError("usage/invalid", "--aggregation is empty"))
    length(aggs) == 1 || length(aggs) == length(lf) || throw(CliError("usage/invalid",
        "--aggregation takes one value or one per --low-freq series " *
        "($(length(lf)) expected, got $(length(aggs)))"))
    agg_arg = length(aggs) == 1 ? Symbol(aggs[1]) : [Symbol(a) for a in aggs]

    _status("Estimating MF-VAR($lags): $n series, low-frequency $(lf), " *
            "ratio $freq_ratio, aggregation $(join(aggs, ","))")
    post = try
        estimate_mfvar(Y, lags; low_freq=lf, freq_ratio=freq_ratio, aggregation=agg_arg,
                       n_draws=draws, n_burn=burnin, prior=Symbol(pr), varnames=varnames,
                       _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "estimate mfvar"))
    end
    _status_report(() -> report(post))

    mu, qs = latent_path(post; quantile_levels=_TVP_QUANTILES)
    # The latent path is at the HIGH frequency for every series, including the ones observed
    # only every freq_ratio periods — interpolating them is what the model is for.
    output_result(_band_long_table(mu, qs, varnames, _TVP_QUANTILES);
                  format=Symbol(format), output=output,
                  title="MF-VAR Latent High-Frequency Path (68% credible band)")

    output_kv(Pair{String,Any}[
        "lags"        => post.p,
        "variables"   => post.n,
        "T_hf"        => post.T_hf,
        "low_freq"    => join(post.low_freq, ","),
        "freq_ratio"  => post.freq_ratio,
        "aggregation" => join(String.(post.aggregation), ","),
        "draws"       => size(post.B_draws, 1),
    ]; format=format, title="MF-VAR Specification")
    return post
end

# ── W9/#111: quantile regression (Koenker-Bassett) + RDD (CCT) ───────────────
#
# NEITHER QuantileRegModel NOR RDDResult is in MEMs' `_COEF_TABLE_TYPES`, so `DataFrame(model)`
# raises "no default Tables.columns implementation" -- the `_reg_coef_table` trap. Both tables
# are hand-built to the same C051 column set. Neither type has a `plot_result` recipe on the
# 0.7.2 tag (checked src/plotting/), so neither leaf advertises --plot.

const _QREG_SE_TYPES = ("iid", "robust", "boot")

"""Parse `--tau`: one value or a comma-list. Upstream fits a whole vector in one call."""
function _parse_taus(tau::String)
    taus = Float64[]
    for tok in split(tau, ',')
        s = strip(tok)
        isempty(s) && continue
        v = tryparse(Float64, s)
        v === nothing && throw(CliError("usage/invalid",
            "--tau must be a number or comma-separated numbers in (0, 1), got '$tau'"))
        0 < v < 1 || throw(CliError("usage/invalid",
            "--tau must lie strictly in (0, 1), got $v";
            hint="0 and 1 are not estimable quantiles"))
        push!(taus, v)
    end
    isempty(taus) && throw(CliError("usage/invalid", "--tau is empty"))
    length(unique(taus)) == length(taus) || throw(CliError("usage/invalid",
        "--tau repeats a quantile: $tau"))
    return taus
end

function _estimate_qreg(; data::String, dep::String="", tau::String="0.5",
                         se::String="iid", n_boot::Int=500, alpha::Float64=0.05,
                         output::String="", format::String="table")
    taus = _parse_taus(tau)
    se_l = lowercase(strip(se))
    se_l in _QREG_SE_TYPES || throw(CliError("usage/invalid-option",
        "invalid --se '$se'; must be one of $(join(_QREG_SE_TYPES, "|"))"))
    n_boot >= 1 || throw(CliError("usage/invalid", "--n-boot must be ≥ 1 (got $n_boot)"))
    0 < alpha < 1 || throw(CliError("usage/invalid",
        "--alpha must lie in (0, 1) (got $alpha)"))

    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    _status("Quantile Regression: $dep_name ~ $(join(xcols, " + "))")
    _status("  Observations: $(length(y)), τ: $(join(taus, ", ")), SE: $se_l")
    _status()

    model = try
        estimate_qreg(y, X, length(taus) == 1 ? taus[1] : taus;
                      se=Symbol(se_l), varnames=xcols, n_boot=n_boot, alpha=alpha,
                      _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "estimate qreg"))
    end

    # beta/stderr are k × n_tau, so the tidy table carries a `tau` column and one row per
    # (tau, term) -- never a wide per-quantile block.
    k, ntau = size(model.beta)
    z = _normal_quantile(1 - alpha / 2)
    rows_tau = Float64[]; rows_term = String[]
    est = Float64[]; ses = Float64[]; stat = Float64[]; pval = Float64[]
    lo = Float64[]; hi = Float64[]
    for j in 1:ntau, i in 1:k
        b = Float64(model.beta[i, j]); s = Float64(model.stderr[i, j])
        t = s > 0 ? b / s : NaN
        push!(rows_tau, Float64(model.taus[j])); push!(rows_term, model.varnames[i])
        push!(est, b); push!(ses, s); push!(stat, t)
        push!(pval, isfinite(t) ? 2 * (1 - _normal_cdf(abs(t))) : NaN)
        push!(lo, b - z * s); push!(hi, b + z * s)
    end
    output_result(DataFrame(tau=rows_tau, term=rows_term, estimate=est, std_error=ses,
                            stat=stat, p_value=pval, ci_lower=lo, ci_upper=hi);
                  format=Symbol(format), output=output,
                  title="Quantile Regression Coefficients ($(se_l) SE)",
                  key="quantile_regression_coefficients")

    # Pseudo-R² is per-quantile and NOT comparable with an OLS R²: it compares the check-
    # function objective against an intercept-only quantile fit, so it measures fit at that
    # quantile only.
    output_result(DataFrame(tau=Float64.(model.taus),
                            objective=round.(Float64.(model.objective); digits=6),
                            pseudo_r2=round.(Float64.(model.pseudo_r2); digits=6),
                            converged=[c ? 1.0 : 0.0 for c in model.converged]);
                  format=Symbol(format), output=_per_var_output_path(output, "fit"),
                  title="Quantile Fit Diagnostics")
    any(!, model.converged) && _status_styled(
        "  warning: at least one quantile did NOT converge\n"; color=:yellow)
    return model
end

const _RDD_KERNELS = ("triangular", "epanechnikov", "uniform")

function _estimate_rdd(; data::String, outcome::String="", running::String="",
                        fuzzy::String="", cutoff::Float64=0.0,
                        bandwidth::Float64=0.0, bias_bandwidth::Float64=0.0,
                        kernel::String="triangular", order::Int=1,
                        level::Float64=0.95,
                        output::String="", format::String="table")
    kern = lowercase(strip(kernel))
    kern in _RDD_KERNELS || throw(CliError("usage/invalid-option",
        "invalid --kernel '$kernel'; must be one of $(join(_RDD_KERNELS, "|"))"))
    order >= 1 || throw(CliError("usage/invalid", "--order must be ≥ 1 (got $order)"))
    0 < level < 1 || throw(CliError("usage/invalid",
        "--level must lie in (0, 1) (got $level)"))
    bandwidth >= 0 || throw(CliError("usage/invalid",
        "--bandwidth must be > 0, or 0 for CCT auto-selection (got $bandwidth)"))
    bias_bandwidth >= 0 || throw(CliError("usage/invalid",
        "--bias-bandwidth must be > 0, or 0 for CCT auto-selection (got $bias_bandwidth)"))

    df = load_data(data)
    numcols = variable_names(df)
    isempty(numcols) && throw(CliError("data/invalid", "no numeric columns found in the data"))
    ycol = isempty(outcome) ? numcols[1] : outcome
    ycol in numcols || throw(CliError("data/column-range",
        "outcome '$ycol' not found in numeric columns: $(join(numcols, ", "))"))
    rcol = if isempty(running)
        cand = filter(!=(ycol), numcols)
        isempty(cand) && throw(CliError("usage/missing",
            "--running is required: only one numeric column ('$ycol') is present"))
        cand[1]
    else
        running
    end
    rcol in numcols || throw(CliError("data/column-range",
        "running variable '$rcol' not found in numeric columns: $(join(numcols, ", "))"))
    ycol == rcol && throw(CliError("usage/invalid",
        "--outcome and --running must be different columns (both '$ycol')"))

    cols = [ycol, rcol]
    isempty(fuzzy) || push!(cols, fuzzy)
    isempty(fuzzy) || fuzzy in numcols || throw(CliError("data/column-range",
        "fuzzy treatment '$fuzzy' not found in numeric columns: $(join(numcols, ", "))"))
    for c in cols
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them first"))
    end
    y = Vector{Float64}(df[!, ycol])
    run = Vector{Float64}(df[!, rcol])

    # Both sides of the cutoff must carry mass, or the local regressions have nothing to
    # fit on one side. Upstream's failure here is not a CLI-level message, and this is
    # squarely user input.
    nl = count(<(cutoff), run); nr = count(>=(cutoff), run)
    (nl > 0 && nr > 0) || throw(CliError("data/invalid",
        "running variable '$rcol' has no observations on " *
        (nl == 0 ? "the LEFT" : "the RIGHT") * " of cutoff $cutoff " *
        "($nl below, $nr at or above)";
        hint="check --cutoff is on the scale of '$rcol' (range " *
             "$(round(minimum(run); digits=4))…$(round(maximum(run); digits=4)))"))

    fz = isempty(fuzzy) ? nothing : Vector{Float64}(df[!, fuzzy])
    _status("RDD: outcome=$ycol, running=$rcol, cutoff=$cutoff" *
            (isempty(fuzzy) ? " (sharp)" : ", fuzzy=$fuzzy"))
    _status("  $nl observations below the cutoff, $nr at or above; kernel=$kern, p=$order")
    _status()

    res = try
        estimate_rdd(y, run; cutoff=cutoff, fuzzy=fz, kernel=Symbol(kern), p=order,
                     h=bandwidth > 0 ? bandwidth : nothing,
                     b=bias_bandwidth > 0 ? bias_bandwidth : nothing,
                     level=level)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "estimate rdd"))
    end

    # The CCT triple. All three share ONE point estimate per row-pair by construction:
    # "robust" is the bias-corrected estimate with a wider SE that accounts for having
    # estimated the bias, NOT a third estimate. Reporting them as three rows makes that
    # explicit rather than leaving a reader to infer it.
    output_result(DataFrame(
        method   = ["conventional", "bias-corrected", "robust"],
        estimate = round.([Float64(res.tau_conventional), Float64(res.tau_bias_corrected),
                           Float64(res.tau_bias_corrected)]; digits=6),
        std_error = round.([Float64(res.se_conventional), Float64(res.se_conventional),
                            Float64(res.se_robust)]; digits=6),
        ci_lower = round.([Float64(res.ci_conventional[1]), NaN,
                           Float64(res.ci_robust[1])]; digits=6),
        ci_upper = round.([Float64(res.ci_conventional[2]), NaN,
                           Float64(res.ci_robust[2])]; digits=6),
    ); format=Symbol(format), output=output,
       title="RDD Treatment Effect ($(res.design), $(round(Int, 100*res.level))% CI)",
       key="rdd_treatment_effect")

    settings = Pair{String,Any}[
        "design"          => String(res.design),
        "cutoff"          => res.cutoff,
        "kernel"          => String(res.kernel),
        "order"           => res.p,
        "bandwidth_h"     => round(Float64(res.h); digits=6),
        "bias_bandwidth_b" => round(Float64(res.b); digits=6),
        "n_left"          => res.n_left,
        "n_right"         => res.n_right,
        "z_robust"        => round(Float64(res.z_robust); digits=6),
        "pvalue_robust"   => round(Float64(res.pvalue_robust); digits=6),
    ]
    res.first_stage === nothing ||
        push!(settings, "first_stage" => round(Float64(res.first_stage); digits=6))
    output_kv(settings; format=format, title="RDD Settings & Diagnostics")

    # n_left/n_right are the EFFECTIVE counts inside the bandwidth, not the whole sample.
    # A handful of effective observations makes the estimate unreliable however tight the
    # nominal CI looks.
    min(res.n_left, res.n_right) < 10 && _status_styled(
        "  warning: only $(res.n_left)/$(res.n_right) effective observations within the " *
        "bandwidth on the left/right — the estimate rests on very few points\n";
        color=:yellow)
    return res
end
