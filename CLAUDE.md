# CLAUDE.md — Friedman-cli

## Project Overview

Friedman-cli (v0.1.3) is a Julia CLI for macroeconometric analysis, wrapping [MacroEconometricModels.jl](https://github.com/chung9207/MacroEconometricModels.jl) (v0.1.3, 160+ exports, 30+ source files). It provides terminal-based VAR/BVAR estimation, impulse response analysis, factor models, local projections, unit root/cointegration tests, GMM estimation, ARIMA modeling/forecasting, and non-Gaussian SVAR identification. Pipeline-based CLI: post-estimation commands (irf, fevd, hd, forecast) nest under their model type (var, bvar, or lp). MIT licensed. ~3,100 lines across 22 source files.

## Quick Reference

```bash
# Install (uses MacroEconometricModels.jl from GitHub by default)
git clone https://github.com/chung9207/Friedman-cli.git
cd Friedman-cli
julia --project -e '
using Pkg
Pkg.rm("MacroEconometricModels")
Pkg.add(url="https://github.com/chung9207/MacroEconometricModels.jl.git")
'

# Or, to use a specific registry version instead:
# julia --project -e 'using Pkg; Pkg.instantiate()'

# Run
julia --project bin/friedman [command] [subcommand] [args] [options]

# Test (CLI engine + IO + config; no MacroEconometricModels needed)
julia --project test/runtests.jl
```

## Project Structure

```
bin/
  friedman                # Entry point — activates project, calls Friedman.main(ARGS)
src/
  Friedman.jl             # Main module — imports deps, includes all files, builds CLI tree via build_app()
  cli/
    types.jl              # 6 structs: Argument, Option, Flag, LeafCommand, NodeCommand, Entry
    parser.jl             # tokenize() → ParsedArgs, bind_args(), convert_value()
    dispatch.jl           # dispatch() walks Entry→NodeCommand→LeafCommand, calls handler
    help.jl               # print_help() generates colored, column-aligned help text
    COMONICON_LICENSE      # License for adapted Comonicon.jl code
  commands/
    shared.jl             # Shared utilities: ID_METHOD_MAP, _load_and_estimate_var/bvar, _load_and_structural_lp, _build_prior, _build_check_func
    var.jl                # VAR pipeline: estimate, lagselect, stability, irf, fevd, hd, forecast
    bvar.jl               # BVAR pipeline: estimate, posterior, irf, fevd, hd, forecast
    lp.jl                 # LP pipeline: estimate (standard|iv|smooth|state|propensity|robust), irf, fevd, hd, forecast
    factor.jl             # Factor Models: estimate (static|dynamic|gdfm), forecast
    test_cmd.jl           # Unit Root/Cointegration: adf, kpss, pp, za, np, johansen
    gmm.jl                # GMM: estimate
    arima.jl              # ARIMA: estimate (explicit or auto), forecast
    nongaussian.jl        # Non-Gaussian SVAR: fastica, ml, heteroskedasticity, normality, identifiability
  config.jl               # TOML loader: load_config, get_prior, get_identification, get_gmm, get_nongaussian
  io.jl                   # Data I/O: load_data, df_to_matrix, variable_names, output_result, output_kv
test/
  runtests.jl             # Tests CLI engine: types, tokenizer, arg binding, help, dispatch
Project.toml              # Julia project — deps and compat
LICENSE                   # MIT
README.md                 # Usage documentation with examples
```

## Dependencies (Project.toml)

Direct: `CSV`, `DataFrames`, `JSON3`, `MacroEconometricModels`, `PrettyTables`
Stdlib (imported in Friedman.jl): `TOML`, `LinearAlgebra` (eigvals), `Statistics` (mean)
Julia compat: `≥ 1.10`

**Note:** Always install `MacroEconometricModels` from GitHub (`https://github.com/chung9207/MacroEconometricModels.jl.git`) unless a specific registry version is requested. The Project.toml UUID (`5b366557`) is a local/dev UUID that differs from the registry UUID (`14a6ec33`), so `Pkg.instantiate()` alone will fail — use `Pkg.rm` + `Pkg.add(url=...)` as shown in Quick Reference.

## Command Hierarchy

```
friedman
├── var          estimate | lagselect | stability | irf | fevd | hd | forecast
├── bvar         estimate | posterior | irf | fevd | hd | forecast
├── lp           estimate | irf | fevd | hd | forecast
├── factor       estimate (static | dynamic | gdfm) | forecast
├── test         adf | kpss | pp | za | np | johansen
├── gmm          estimate
├── arima        estimate | forecast
└── nongaussian  fastica | ml | heteroskedasticity | normality | identifiability
```

Total: 8 top-level commands, 37 subcommands. Post-estimation (irf/fevd/hd/forecast) nests under model type (var, bvar, lp) — no `--bayesian` flags.

## Architecture

### Execution Flow

```
bin/friedman ARGS
  → Pkg.activate(project_dir)
  → Friedman.main(ARGS)
    → build_app()                          # constructs Entry with full command tree
      → register_var_commands!()           # each returns NodeCommand with LeafCommand children
      → register_bvar_commands!()
      → ... (8 register functions)
    → dispatch(entry, ARGS)
      → dispatch_node()                    # walks NodeCommand tree by matching arg tokens
      → dispatch_leaf()                    # tokenize → bind_args → leaf.handler(; bound...)
```

### Data Flow

```
CSV file → load_data(path)                 # → DataFrame, validates exists & non-empty
         → df_to_matrix(df)                # → Matrix{Float64}, selects numeric columns only
         → variable_names(df)              # → Vector{String}, numeric column names
                ↓
    MacroEconometricModels.jl functions     # estimate_var, estimate_bvar, irf, etc.
                ↓
    Results → DataFrame                     # command builds result DataFrame
           → output_result(df; format, output, title)
                ↓
              :table → PrettyTables (tf_unicode_rounded, centered)
              :csv   → CSV.write
              :json   → JSON3.write (array of row dicts)
```

### CLI Framework

Custom-built, adapted from Comonicon.jl. Key types:
- `Entry` — top-level: name + root NodeCommand + version
- `NodeCommand` — command group: name + `Dict{String, Union{NodeCommand, LeafCommand}}`
- `LeafCommand` — executable: name + handler function + args/options/flags
- `Argument` — positional (name, type, required, default)
- `Option` — named `--opt=val` or `-o val` (name, short, type, default)
- `Flag` — boolean `--flag` or `-f` (name, short)

Parser features: `--opt=val`, `--opt val`, `-o val`, bundled `-abc`, `--` stops parsing.

## Code Conventions

- **Naming:** functions `snake_case`, types `PascalCase`, internal handlers prefixed `_` (e.g. `_bvar_estimate`)
- **Command pattern:** Each `src/commands/X.jl` defines `register_X_commands!()` → `NodeCommand`
- **Handler signature:** `_command_subcommand(; data::String, option1=default, ...)` — keyword args match declared Options
- **Option hyphen→underscore:** `--control-lags` binds to `control_lags` kwarg (parser replaces `-` with `_`)
- **Config-driven complexity:** TOML files for priors, restrictions, GMM specs — keeps CLI flags clean
- **Auto-selection:** lag orders via `select_lag_order(...; criterion=:aic)`, factor counts via `ic_criteria()`, smoothing λ via cross-validation — when user doesn't specify
- **Output:** `println()` for status, `printstyled(; color=:green/:yellow/:red)` for diagnostics, `output_result()` for data tables
- **Error handling:** `error()` for missing required config, `ParseError` for CLI parsing failures

## Command Details

### shared (shared.jl, ~255 lines)
- `ID_METHOD_MAP` — maps CLI id strings ("cholesky"/"sign"/"narrative"/"longrun") to library symbols
- `_load_and_estimate_var(data, lags)` — load CSV, auto lag selection, estimate frequentist VAR
- `_load_and_estimate_bvar(data, lags, config, draws, sampler)` — load CSV, build prior, estimate Bayesian VAR via MCMC
- `_load_and_structural_lp(data, horizons, lags, var_lags, id, vcov, config)` — load CSV, build identification, call `structural_lp()`, returns `(slp, Y, varnames)`
- `_build_prior(config, Y, p)` — builds `MinnesotaHyperparameters` from TOML or auto-estimates AR(1) residual σ
- `_build_check_func(config)` — constructs sign_matrix and narrative check closures from TOML
- `_build_identification_kwargs(id, config)` — assembles kwargs dict (method, check_func, narrative_check)
- `_var_forecast_point(B, Y, p, horizons)` — iterates VAR equation h steps ahead for point forecasts

### var (var.jl, 447 lines)
- **estimate** — OLS VAR(p), auto lag selection, outputs coefficients + AIC/BIC/HQC/loglik
- **lagselect** — estimates VAR for each p=1..max, prints IC table, reports optimal
- **stability** — companion matrix eigenvalues, stationarity check (all |λ| < 1)
- **irf** — frequentist IRFs with 5 identification schemes: cholesky, sign, narrative, longrun, arias. Bootstrap/theoretical CIs
- **fevd** — FEVD proportions (n_vars × n_shocks × horizons), per-variable output tables
- **hd** — historical decomposition: actual values, initial conditions, shock contributions. Calls `verify_decomposition()` for validation
- **forecast** — h-step ahead VAR point forecasts with analytical confidence intervals (MA(∞) MSE)
- Helper: `_var_irf_arias()` — Arias identification via `identify_arias()` with zero/sign `SVARRestrictions`
- Helper: `quantile_normal()` — normal quantile without Distributions.jl dependency
- Uses shared: `_load_and_estimate_var()`, `_build_identification_kwargs()`, `_var_forecast_point()`

### bvar (bvar.jl, 386 lines)
- **estimate** — Bayesian VAR with MCMC (nuts/hmc/smc), Minnesota prior (optional TOML config with hyperparameter optimization)
- **posterior** — posterior mean or median extraction, outputs coefficients + IC
- **irf** — Bayesian IRFs with 68% credible intervals (16th/50th/84th percentiles), sign/narrative identification
- **fevd** — Bayesian FEVD, posterior mean proportions per variable
- **hd** — Bayesian historical decomposition, posterior mean contributions + initial conditions
- **forecast** — Bayesian h-step ahead forecasts with 68% credible intervals from posterior draws; uses `MacroEconometricModels.extract_chain_parameters`/`parameters_to_model` (internal, not exported in v0.1.3)
- Uses shared: `_load_and_estimate_bvar()`, `_build_prior()`, `_build_check_func()`, `ID_METHOD_MAP`, `_var_forecast_point()`

### lp (lp.jl, ~550 lines)
- **estimate** — unified estimation with `--method=standard|iv|smooth|state|propensity|robust`; dispatches to internal helpers per method
  - standard: Jorda (2005) LP with Newey-West/White/Driscoll-Kraay HAC
  - iv: LP-IV (Stock & Watson 2018) with external instruments, weak instrument F-test
  - smooth: Barnichon & Brownlees (2019) B-spline smoothed LP, auto λ via cross-validation
  - state: Auerbach & Gorodnichenko (2013) state-dependent LP with logistic/exponential/indicator transition, Wald test
  - propensity: Angrist et al. (2018) propensity score LP with logit/probit, reports ATE + diagnostics
  - robust: doubly robust LP, combines propensity score + outcome regression
- **irf** — structural LP IRFs via `structural_lp()`, supports cholesky/sign/narrative/longrun identification, multi-shock via `--shocks=1,2,3`
- **fevd** — native LP-FEVD via `lp_fevd(slp, horizons)` with bias-corrected proportions (Gorodnichenko & Lee 2019)
- **hd** — native LP HD via `historical_decomposition(slp, T_hd)`, verify decomposition
- **forecast** — direct LP forecast via `forecast(LPModel, shock_path)`, unit impulse with `--shock-size`, analytical/bootstrap CIs
- Helpers: `_lp_estimate_standard()`, `_lp_estimate_iv()`, `_lp_estimate_smooth()`, `_lp_estimate_state()`, `_lp_estimate_propensity()`, `_lp_estimate_robust()`
- Uses shared: `_load_and_structural_lp()`, `_build_check_func()`, `ID_METHOD_MAP`

### factor (factor.jl, ~330 lines)
- **estimate static** — PCA factor model, Bai-Ng IC (ic1/ic2/ic3) for factor count, scree data + loadings
- **estimate dynamic** — dynamic factor model with factor VAR, stationarity check, companion eigenvalues
- **estimate gdfm** — generalized dynamic factor model, common variance shares
- **forecast** — forecast observables using factor model; `--model=static|dynamic|gdfm` selects model type (default: static); static supports bootstrap/parametric CIs, dynamic uses factor VAR extrapolation, GDFM uses common component AR(1) projection
- Helpers: `_factor_forecast_static()`, `_factor_forecast_dynamic()`, `_factor_forecast_gdfm()`

### test (test_cmd.jl, 267 lines)
- **adf** — Augmented Dickey-Fuller (auto lag via AIC, trend options)
- **kpss** — KPSS stationarity test (H₀: stationary)
- **pp** — Phillips-Perron
- **za** — Zivot-Andrews structural break test (reports break date)
- **np** — Ng-Perron (MZa, MZt, MSB, MPT statistics)
- **johansen** — Johansen cointegration (trace + max eigenvalue, reports cointegration rank)
- Helper: `_extract_series()` — extracts single column for univariate tests

### gmm (gmm.jl, 90 lines)
- **estimate** — LP-GMM with identity/optimal/twostep/iterated weighting
- Hansen's J-test for overidentification, parameter estimates with standard errors
- Requires TOML config specifying moment conditions

### arima (arima.jl, ~185 lines)
- **estimate** — Estimate ARIMA(p,d,q): explicit orders when `--p` given, auto-selects via `auto_arima()` when `--p` omitted; auto-dispatches to `estimate_ar`/`estimate_ma`/`estimate_arma`/`estimate_arima` based on (p,d,q); reports coefficients + AIC/BIC/loglik
- **forecast** — Estimate + forecast in one step; uses `auto_arima` if `--p` omitted, outputs horizon/forecast/CI/SE table
- Helpers: `_estimate_arima_model()`, `_model_label()`, `_arima_coef_table()`
- Reuses `_extract_series()` from test_cmd.jl for single-column extraction

### nongaussian (nongaussian.jl, ~310 lines)
- **fastica** — ICA-based SVAR identification (FastICA/Infomax/JADE/SOBI/dCov/HSIC), outputs B0 matrix + structural shocks
- **ml** — Maximum likelihood non-Gaussian SVAR (Student-t/skew-t/GHD/mixture-normal/PML/skew-normal), outputs B0 + log-likelihood + AIC/BIC
- **heteroskedasticity** — Heteroskedasticity-based identification (Markov-switching/GARCH/smooth-transition/external volatility)
- **normality** — Normality test suite for VAR residuals, reports which tests reject Gaussianity
- **identifiability** — Tests for identification strength, shock Gaussianity, independence, overidentification, Gaussian vs non-Gaussian comparison
- Helper: `_ng_estimate_var()` — shared VAR estimation with auto lag selection

## TOML Configuration

```toml
# Minnesota prior (for bvar)
[prior]
type = "minnesota"
[prior.hyperparameters]
lambda1 = 0.2      # tau (tightness)
lambda2 = 0.5      # cross-variable shrinkage
lambda3 = 1.0      # lag decay
lambda4 = 100000.0  # constant term variance
[prior.optimization]
enabled = true      # auto-optimize hyperparameters

# Sign restrictions (for var irf/fevd/hd, bvar irf/fevd/hd)
[identification]
method = "sign"
[identification.sign_matrix]
matrix = [[1, -1, 1], [0, 1, -1], [0, 0, 1]]
horizons = [0, 1, 2, 3]

# Narrative restrictions
[identification.narrative]
shock_index = 1
periods = [10, 15, 20]
signs = [1, -1, 1]

# Arias identification
[[identification.zero_restrictions]]
var = 1
shock = 1
horizon = 0
[[identification.sign_restrictions]]
var = 2
shock = 1
sign = "positive"
horizon = 0

# Non-Gaussian SVAR (for nongaussian)
[nongaussian]
method = "fastica"          # fastica/infomax/jade/ml/markov/garch/smooth_transition/external
contrast = "logcosh"        # for ICA methods: logcosh/exp/kurtosis
distribution = "student_t"  # for ML method: student_t/skew_t/ghd
n_regimes = 2               # for markov-switching/external
transition_variable = "spread"  # column name for smooth-transition
regime_variable = "nber"        # column name for external volatility

# GMM specification
[gmm]
moment_conditions = ["output", "inflation"]
instruments = ["lag_output", "lag_inflation"]
weighting = "twostep"
```

## Testing

Tests (`test/runtests.jl`) cover CLI engine, IO, and config — no MacroEconometricModels dependency needed:
- **Types** — construction of Argument, Option, Flag, LeafCommand, NodeCommand, Entry
- **Tokenizer** — positional args, `--long=val`, `--long val`, `-s val`, flags, bundled `-abc`, `--` separator
- **Argument binding** — type conversion, defaults, short aliases, required arg validation, excess arg rejection
- **Help generation** — output contains expected command names, argument labels, option flags
- **Dispatch** — walks node→leaf correctly, passes bound args to handler, help flags don't error
- **VAR pipeline structure** — var node has irf/fevd/hd/forecast as LeafCommands, dispatch, help text, arg binding
- **BVAR pipeline structure** — bvar node has irf/fevd/hd/forecast as LeafCommands with draws/sampler options, no flags
- **LP pipeline structure** — lp node has 5 subcmds (estimate/irf/fevd/hd/forecast), no old flat cmds (iv/smooth/state/propensity/multi/robust), option counts, help text, dispatch
- **Pipeline cleanup** — verifies top-level root no longer has irf/fevd/hd commands
- **Non-Gaussian SVAR structure** — 5 subcommands with correct options, help text, dispatch
- **Factor command structure** — estimate NodeCommand (static/dynamic/gdfm) + forecast LeafCommand, dispatch through 3 levels
- **IO utilities** — load_data, df_to_matrix, variable_names, output_result (CSV/JSON/table), output_kv
- **Config parsing** — load_config, get_identification, get_prior, get_gmm, get_nongaussian

Run: `julia --project -e 'using Pkg; Pkg.test()'`

## Adding a New Command

1. Create `src/commands/mycommand.jl`
2. Define `register_mycommand_commands!()`:
   - Create `LeafCommand` for each subcommand with args, options, handler
   - Return `NodeCommand("mycommand", subcmds_dict, "Description")`
3. Define handler `_mycommand_subcommand(; data::String, ...options)`:
   - `df = load_data(data)` → `Y = df_to_matrix(df)` → `varnames = variable_names(df)`
   - Call MacroEconometricModels functions
   - Build result `DataFrame`
   - `output_result(df; format=Symbol(format), output=output, title="...")`
4. In `src/Friedman.jl`: add `include("commands/mycommand.jl")` and register in `build_app()`
5. Add tests if needed in `test/runtests.jl`

## Common Options (all commands)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `"table"` | table\|csv\|json |
| `--output` | `-o` | String | `""` | Export file path (empty = stdout) |
| `--lags` | `-p` | Int | varies | Lag order (often auto via AIC) |
| `--config` | — | String | `""` | TOML config file path |

## MacroEconometricModels.jl API Reference

Upstream library (v0.1.2): 144+ exports, 27+ source files, 24+ test files.
Dependencies: DataFrames, Distributions, FFTW, LinearAlgebra, Optim, MCMCChains, PrettyTables, Random, SpecialFunctions, Statistics, StatsAPI, Turing.

### Key Types

```
VARModel{T}              # Y, p, B, U, Sigma, aic, bic, hqic
MinnesotaHyperparameters  # tau, decay, lambda, mu, omega
ImpulseResponse{T}       # values::Array{T,3}, ci_lower, ci_upper, horizon, variables, shocks, ci_type
BayesianImpulseResponse{T}  # quantiles::Array{T,4}, mean::Array{T,3}, quantile_levels
FEVD{T}                  # decomposition::Array{T,3}, proportions::Array{T,3}
BayesianFEVD{T}          # quantiles, mean, quantile_levels
HistoricalDecomposition{T}  # contributions::Array{T,3}, initial_conditions, actual, shocks, T_eff
BayesianHistoricalDecomposition{T}  # quantiles, mean, initial_quantiles, initial_mean
SVARRestrictions          # zeros::Vector{ZeroRestriction}, signs::Vector{SignRestriction}, n_vars, n_shocks
AriasSVARResult{T}       # Q_draws, irf_draws, weights, acceptance_rate, restrictions
FactorModel{T}           # data, factors, loadings, eigenvalues
DynamicFactorModel{T}    # factors, loadings, VAR coefficients
GeneralizedDynamicFactorModel{T}  # common component, spectral densities, complex loadings
LPModel{T}               # Y, shock_var, horizon, lags, B, residuals, vcov, cov_estimator
LPIVModel{T}             # + instruments, first_stage_F, first_stage_coef
SmoothLPModel{T}         # + spline_basis, theta, lambda, irf_values
StateLPModel{T}          # + state, B_expansion, B_recession, vcov_diff
PropensityLPModel{T}     # + treatment, propensity_scores, ipw_weights, ate, ate_se
LPImpulseResponse{T}     # values, ci_lower, ci_upper, se, horizon
PropensityScoreConfig{T}  # method, trimming, normalize
GMMModel{T}              # theta, vcov, n_moments, n_params, W, g_bar, J_stat, J_pvalue
GMMWeighting{T}          # method, max_iter, tol
ARModel{T}, MAModel{T}, ARMAModel{T}, ARIMAModel{T}  # ARIMA family
ARIMAForecast{T}         # forecast, ci_lower, ci_upper, se, horizon
NeweyWestEstimator, WhiteEstimator, DriscollKraayEstimator  # HAC covariance
ICASVARResult{T}         # B0, W, Q, shocks, method, converged, iterations, objective
NonGaussianMLResult{T}   # B0, Q, shocks, distribution, loglik, loglik_gaussian, dist_params, vcov, se, aic, bic
MarkovSwitchingSVARResult{T}  # B0, regime info
GARCHSVARResult{T}       # B0, GARCH parameters
SmoothTransitionSVARResult{T}  # B0, transition parameters
ExternalVolatilitySVARResult{T}  # B0, regime indicator
NormalityTestSuite{T}    # results::Vector{NormalityTestResult{T}}
NormalityTestResult{T}   # test_name, statistic, pvalue, df
StructuralLP{T}          # irf, var_model, Q, method, se, lp_models
LPForecast{T}            # forecasts, ci_lower, ci_upper, se, horizon, response_vars, shock_var, shock_path
LPFEVD{T}                # R2, lp_a, lp_b, bias_corrected, bootstrap_se, horizons, variables, shocks
FactorForecast{T}        # factors, observables, *_lower, *_upper, observables_se, horizon, conf_level, ci_method
```

### VAR Estimation

```julia
estimate_var(Y, p; check_stability=true) → VARModel
estimate_var(df::DataFrame, p; vars=Symbol[]) → VARModel
select_lag_order(Y, max_p; criterion=:bic) → Int          # :aic, :bic, :hqic
companion_matrix(model) → Matrix
is_stationary(model) → VARStationarityResult              # .is_stationary, .eigenvalues
nvars(model), nlags(model), ncoefs(model), effective_nobs(model)
```

### Bayesian VAR

```julia
estimate_bvar(Y, p; n_samples=1000, n_adapts=500, prior=:normal,
    hyper=nothing, sampler=:nuts) → Chains                 # :nuts, :hmc, :hmcda, :smc, :pg
posterior_mean_model(chain, p, n; data=Y) → VARModel
posterior_median_model(chain, p, n; data=Y) → VARModel
extract_chain_parameters(chain) → (b_vecs, sigmas)
parameters_to_model(b_vec, sigma_vec, p, n; data) → VARModel
```

### Minnesota Prior

```julia
MinnesotaHyperparameters(; tau, decay, lambda, mu, omega)
gen_dummy_obs(Y, p, hyper) → (Y_dummy, X_dummy)
log_marginal_likelihood(Y, p, hyper) → T
optimize_hyperparameters(Y, p) → MinnesotaHyperparameters      # grid over tau
optimize_hyperparameters_full(Y, p) → MinnesotaHyperparameters  # grid over tau, lambda, mu
```

### Structural Identification

```julia
identify_cholesky(model) → Matrix                          # lower-triangular
identify_sign(model, horizon, check_func; max_draws=1000) → (Q, irf)
identify_narrative(model, horizon, sign_check, narrative_check; max_draws=1000) → (Q, irf, shocks)
identify_long_run(model) → Matrix                          # Blanchard-Quah

# Arias et al. (2018)
zero_restriction(variable, shock; horizon=0) → ZeroRestriction
sign_restriction(variable, shock, sign::Symbol; horizon=0) → SignRestriction
SVARRestrictions(n; zeros, signs)
identify_arias(model, restrictions, horizon; n_draws=1000, n_rotations=1000) → AriasSVARResult
identify_arias_bayesian(chain, p, n, restrictions, horizon; data, n_rotations=100,
    quantiles=[0.16, 0.5, 0.84]) → NamedTuple

generate_Q(n) → Matrix           # random orthogonal
compute_irf(model, Q, horizon) → Array{T,3}
compute_structural_shocks(model, Q) → Matrix
irf_percentiles(result; probs=[0.16, 0.5, 0.84]) → Array
irf_mean(result) → Array
```

### Non-Gaussian SVAR Identification (v0.1.3)

```julia
# ICA methods
identify_fastica(model::VARModel{T}; contrast=:logcosh, max_iter=200, tol=1e-6) → ICASVARResult{T}
identify_infomax(model::VARModel{T}; max_iter=200, tol=1e-6, learning_rate=0.01) → ICASVARResult{T}
identify_jade(model::VARModel{T}) → ICASVARResult{T}
identify_sobi(model::VARModel{T}) → ICASVARResult{T}
identify_dcov(model::VARModel{T}) → ICASVARResult{T}
identify_hsic(model::VARModel{T}) → ICASVARResult{T}

# ML methods
identify_nongaussian_ml(model::VARModel{T}; distribution=:student_t, max_iter=500, tol=1e-6) → NonGaussianMLResult{T}
identify_student_t(model::VARModel{T}) → NonGaussianMLResult{T}
identify_mixture_normal(model::VARModel{T}) → NonGaussianMLResult{T}
identify_pml(model::VARModel{T}) → NonGaussianMLResult{T}
identify_skew_normal(model::VARModel{T}) → NonGaussianMLResult{T}

# Heteroskedasticity-based
identify_markov_switching(model::VARModel{T}; n_regimes=2, max_iter=200, tol=1e-6) → MarkovSwitchingSVARResult{T}
identify_garch(model::VARModel{T}; max_iter=200, tol=1e-6) → GARCHSVARResult{T}
identify_smooth_transition(model::VARModel{T}, transition_var::AbstractVector; gamma=1.0, c=0.0) → SmoothTransitionSVARResult{T}
identify_external_volatility(model::VARModel{T}, regime_indicator::AbstractVector; regimes=2) → ExternalVolatilitySVARResult{T}

# Tests
normality_test_suite(model::VARModel) → NormalityTestSuite{T}
test_identification_strength(model::VARModel) → NamedTuple
test_shock_gaussianity(result::ICASVARResult) → NamedTuple
test_gaussian_vs_nongaussian(model::VARModel) → NamedTuple
test_shock_independence(result::ICASVARResult) → NamedTuple
test_overidentification(result::ICASVARResult) → NamedTuple
```

### Impulse Response Functions

```julia
# Frequentist
irf(model::VARModel, horizon; method=:cholesky, check_func=nothing,
    narrative_check=nothing, ci_type=:none, reps=200, conf_level=0.95) → ImpulseResponse
# method: :cholesky, :sign, :narrative, :long_run
# ci_type: :none, :bootstrap, :theoretical

# Bayesian
irf(chain::Chains, p, n, horizon; method=:cholesky, data=...,
    check_func=nothing, narrative_check=nothing,
    quantiles=[0.16, 0.5, 0.84], threaded=false) → BayesianImpulseResponse

# LP-based
lp_irf(model::LPModel; conf_level=0.95) → LPImpulseResponse
cumulative_irf(irf::LPImpulseResponse) → LPImpulseResponse
```

### FEVD

```julia
fevd(model::VARModel, horizon; method=:cholesky, ...) → FEVD
fevd(chain::Chains, p, n, horizon; quantiles=[0.16, 0.5, 0.84], ...) → BayesianFEVD
```

### Historical Decomposition

```julia
historical_decomposition(model::VARModel, horizon; method=:cholesky,
    check_func=nothing, narrative_check=nothing) → HistoricalDecomposition
historical_decomposition(chain::Chains, p, n, horizon; data, method=:cholesky,
    quantiles=[0.16, 0.5, 0.84]) → BayesianHistoricalDecomposition
historical_decomposition(model, restrictions::SVARRestrictions, horizon;
    n_draws=1000, n_rotations=1000) → BayesianHistoricalDecomposition

contribution(hd, var, shock) → Vector
total_shock_contribution(hd, var) → Vector
verify_decomposition(hd; tol=...) → Bool
```

### Factor Models

```julia
# Static (PCA)
estimate_factors(X, r; standardize=true) → FactorModel
ic_criteria(X, max_factors; standardize=true) → NamedTuple  # .ic1/.ic2/.ic3, .r_IC1/.r_IC2/.r_IC3
scree_plot_data(model) → NamedTuple                         # .eigenvalues, .variance_shares, .cumulative_shares

# Dynamic
estimate_dynamic_factors(X, r, p; method=:twostep, max_iter=100, tol=1e-6) → DynamicFactorModel
ic_criteria_dynamic(X, max_r, max_p; method=:twostep) → NamedTuple  # .r_opt
companion_matrix_factors(model) → Matrix
is_stationary(model::DynamicFactorModel) → Bool
forecast(model::DynamicFactorModel, h; ci=false) → ...
forecast(model::FactorModel, h; ci_method=:none, conf_level=0.95) → FactorForecast{T}

# Generalized Dynamic (spectral)
estimate_gdfm(X, q; standardize=true, bandwidth=0, kernel=:bartlett, r=0) → GeneralizedDynamicFactorModel
ic_criteria_gdfm(X, max_q; standardize=true) → NamedTuple   # .q_opt, .r_opt
common_variance_share(model) → Vector
spectral_eigenvalue_plot_data(model) → NamedTuple
```

### Local Projections

```julia
# Core LP (Jorda 2005)
estimate_lp(Y, shock_var, horizon; lags=4, cov_type=:newey_west) → LPModel
estimate_lp_multi(Y, shock_vars, horizon; ...) → Vector{LPModel}
estimate_lp_cholesky(Y, horizon; lags=4) → ...
lp_irf(model; conf_level=0.95) → LPImpulseResponse
compare_var_lp(Y, horizon; lags=4) → NamedTuple

# Structural LP (v0.1.3)
structural_lp(Y, horizon; method, lags, var_lags, cov_type, ci_type, reps,
    conf_level, check_func, narrative_check, max_draws) → StructuralLP{T}
# Fields: irf::ImpulseResponse{T}, var_model::VARModel{T}, Q::Matrix{T},
#         method::Symbol, se::Array{T,3}, lp_models::Vector{LPModel{T}}
irf(slp::StructuralLP) → ImpulseResponse{T}
lp_fevd(slp::StructuralLP, horizons; estimator=:R2, n_boot=200, conf_level=0.95) → LPFEVD{T}
# LPFEVD fields: R2, lp_a, lp_b, bias_corrected, bootstrap_se (Gorodnichenko & Lee 2019)
historical_decomposition(slp::StructuralLP, T_hd) → HistoricalDecomposition{T}

# LP Forecast (v0.1.3)
forecast(model::LPModel, shock_path; ci_method=:analytical, conf_level=0.95, n_boot=500) → LPForecast{T}
forecast(slp::StructuralLP, shock_idx, shock_path; ...) → LPForecast{T}
# LPForecast fields: forecasts, ci_lower, ci_upper, se, horizon, response_vars, shock_var, shock_path

# LP-IV (Stock & Watson 2018)
estimate_lp_iv(Y, shock_var, instruments, horizon; lags=4, cov_type=:newey_west) → LPIVModel
lp_iv_irf(model; conf_level=0.95) → LPImpulseResponse
weak_instrument_test(model; threshold=10.0) → NamedTuple    # .F_stat
sargan_test(model, h) → NamedTuple

# Smooth LP (Barnichon & Brownlees 2019)
estimate_smooth_lp(Y, shock_var, horizon; degree=3, n_knots=4, lambda=0.0) → SmoothLPModel
smooth_lp_irf(model; conf_level=0.95) → LPImpulseResponse
cross_validate_lambda(Y, shock_var, horizon; k_folds=5) → T
compare_smooth_lp(Y, shock_var, horizon; lambda=1.0) → NamedTuple

# State-Dependent LP (Auerbach & Gorodnichenko 2013)
estimate_state_lp(Y, shock_var, state_var, horizon; gamma=:estimate, lags=4) → StateLPModel
state_irf(model; regime=:both, conf_level=0.95) → NamedTuple  # .expansion, .recession
test_regime_difference(model; h=nothing) → NamedTuple          # .wald_stat, .p_value
logistic_transition(z, gamma, c), exponential_transition, indicator_transition

# Propensity Score LP (Angrist et al. 2018)
estimate_propensity_lp(Y, treatment, covariates, horizon; ps_method=:logit,
    trimming=(0.01, 0.99)) → PropensityLPModel
doubly_robust_lp(Y, treatment, covariates, horizon; ...) → PropensityLPModel
propensity_irf(model; conf_level=0.95) → LPImpulseResponse
propensity_diagnostics(model) → NamedTuple                    # .mean_score, .effective_n
PropensityScoreConfig(; method=:logit, trimming, normalize)
```

### GMM

```julia
estimate_gmm(moment_fn, theta0, data; weighting=:two_step, max_iter=100,
    hac=true) → GMMModel                                      # :identity, :optimal, :two_step, :iterated
estimate_lp_gmm(Y, shock_var, horizon; lags=4, weighting=:two_step) → Vector{GMMModel}
j_test(model) → NamedTuple                                    # .J_stat, .p_value, .df
gmm_summary(model) → NamedTuple
gmm_objective(theta, moment_fn, data, W) → T
optimal_weighting_matrix(moment_fn, theta, data; hac=true) → Matrix
```

### ARIMA

```julia
estimate_ar(y, p; method=:ols) → ARModel
estimate_ma(y, q; method=:css_mle) → MAModel
estimate_arma(y, p, q; method=:css_mle) → ARMAModel
estimate_arima(y, p, d, q; method=:css_mle) → ARIMAModel
auto_arima(y; max_p=5, max_q=5, max_d=2, criterion=:bic) → ARIMAModel
select_arima_order(y, max_p, max_q; criterion=:bic) → ARIMAOrderSelection
forecast(model, h; conf_level=0.95) → ARIMAForecast
ar_order(m), ma_order(m), diff_order(m)
```

### Unit Root & Cointegration Tests

```julia
adf_test(y; lags=:aic, regression=:constant) → ADFResult     # .statistic, .pvalue, .lags
kpss_test(y; regression=:constant) → KPSSResult               # H₀: stationary
pp_test(y; regression=:constant) → PPResult
za_test(y; regression=:both, trim=0.15) → ZAResult            # .break_point
ngperron_test(y; regression=:constant) → NgPerronResult        # .MZa, .MZt, .MSB, .MPT
johansen_test(Y, p; deterministic=:constant) → JohansenResult  # .trace_stat, .trace_cv_5, .max_stat, .max_cv_5

unit_root_summary(y; tests=[:adf, :kpss, :pp]) → NamedTuple
test_all_variables(Y; test=:adf) → Vector
```

### Covariance Estimators

```julia
newey_west(X, residuals; bandwidth=0, kernel=:bartlett) → Matrix
white_vcov(X, residuals; variant=:hc0) → Matrix               # :hc0, :hc1, :hc2, :hc3
driscoll_kraay(X, u; bandwidth=0, kernel=:bartlett) → Matrix
optimal_bandwidth_nw(residuals) → Int
robust_vcov(X, residuals, estimator) → Matrix
# Kernels: :bartlett, :parzen, :quadratic_spectral, :tukey_hanning
```

### StatsAPI Interface

VARModel and ARIMA models implement the full StatsAPI interface:
`coef`, `vcov`, `residuals`, `predict`, `r2`, `aic`, `bic`, `dof`, `dof_residual`, `nobs`, `loglikelihood`, `confint`, `stderror`, `islinear`, `fit`

### Summary & Display

```julia
summary(model)           # VARModel, ImpulseResponse, FEVD, HistoricalDecomposition, Bayesian variants
table(result, var, shock; horizons=nothing) → Matrix
print_table(result, var, shock)
point_estimate(result), has_uncertainty(result), uncertainty_bounds(result)
```

## CLI ↔ Library Coverage

| Library Feature | CLI Command | Status |
|---|---|---|
| VAR estimation | `var estimate` | Wrapped |
| Lag selection | `var lagselect` | Wrapped |
| Stationarity check | `var stability` | Wrapped |
| Frequentist IRF (5 methods) | `var irf` | Wrapped |
| Frequentist FEVD | `var fevd` | Wrapped |
| Frequentist HD | `var hd` | Wrapped |
| VAR forecast | `var forecast` | Wrapped |
| Bayesian VAR | `bvar estimate/posterior` | Wrapped |
| Minnesota prior | `bvar --config` | Wrapped |
| Bayesian IRF | `bvar irf` | Wrapped |
| Bayesian FEVD | `bvar fevd` | Wrapped |
| Bayesian HD | `bvar hd` | Wrapped |
| Bayesian forecast | `bvar forecast` | Wrapped |
| Local projections (6 methods) | `lp estimate --method=standard\|iv\|smooth\|state\|propensity\|robust` | Wrapped |
| Structural LP IRFs | `lp irf` | Wrapped |
| LP FEVD (native, bias-corrected) | `lp fevd` | Wrapped |
| LP HD (native, structural LP) | `lp hd` | Wrapped |
| LP forecast | `lp forecast` | Wrapped |
| Static factor model | `factor estimate static` | Wrapped |
| Dynamic factor model | `factor estimate dynamic` | Wrapped |
| Generalized DFM | `factor estimate gdfm` | Wrapped |
| Unit root tests (5 types) | `test adf/kpss/pp/za/np` | Wrapped |
| Johansen cointegration | `test johansen` | Wrapped |
| GMM | `gmm estimate` | Wrapped |
| ARIMA (AR/MA/ARMA/ARIMA/auto) | `arima estimate/forecast` | Wrapped |
| Factor model forecasting | `factor forecast` | Wrapped |
| ICA SVAR (FastICA/Infomax/JADE/SOBI/dCov/HSIC) | `nongaussian fastica` | Wrapped |
| Non-Gaussian ML SVAR (+ mixture normal/PML/skew-normal) | `nongaussian ml` | Wrapped |
| Heteroskedasticity SVAR | `nongaussian heteroskedasticity` | Wrapped |
| Normality test suite | `nongaussian normality` | Wrapped |
| Identifiability tests | `nongaussian identifiability` | Wrapped |
| VAR-LP comparison | — | Not wrapped (convenience utility) |
| unit_root_summary / test_all_variables | — | Not wrapped (convenience utility) |

## MacroEconometricModels.jl — Upstream Documentation

Full docs deployed at: https://chung9207.github.io/MacroEconometricModels.jl/dev/
DOI: 10.5281/zenodo.18439170

### Documentation Structure (docs/src/)

```
index.md                  # Home: overview, features, installation, quick start, package structure, notation
manual.md                 # VAR theory: OLS estimation, stability, IC, SVAR identification, HAC estimators
bayesian.md               # BVAR: Minnesota prior, dummy observations, hyperparameter optimization, MCMC via Turing.jl
lp.md                     # LP: standard, LP-IV, smooth (B-spline), state-dependent, propensity score, LP vs VAR
factormodels.md           # Factor models: static PCA, Bai-Ng IC, dynamic (two-step/EM), GDFM (spectral)
innovation_accounting.md  # IRF (recursive, cumulative, bootstrap/Bayesian CI), FEVD, HD, summary tables
hypothesis_tests.md       # ADF, KPSS, PP, ZA, Ng-Perron, Johansen; decision matrix, practical workflow
examples.md               # 7 worked examples: VAR, BVAR, LP, factors, GMM, integrated workflow, unit root
api.md                    # Quick reference tables: 12 estimation, 6 structural, 9 test, 5 LP-IRF, etc.
api_types.md              # Type hierarchy with Documenter.jl @docs directives
api_functions.md          # Function docs with @autodocs directives by domain
```

### Example Scripts (examples/)

- `example_analysis.jl` — frequentist + Bayesian VAR with 4 identification schemes
- `factor_model_example.jl` — 7 sub-examples: estimation, diagnostics, Bai-Ng IC, scree, standardization, forecasting, macro scenario
- `local_projections_example.jl` — standard LP, VAR vs LP comparison, cumulative IRFs
- `macro_data.csv` — synthetic 3-variable CSV for examples

### Mathematical Foundations (from docs)

**VAR(p):** Y_t = A_1 Y_{t-1} + ... + A_p Y_{t-p} + u_t, u_t ~ N(0, Σ)
- OLS: B = (X'X)^{-1} X'Y, Σ = U'U / (T - np - 1)
- Stability: all eigenvalues of companion matrix inside unit circle
- IC: AIC = log|Σ| + 2k/T, BIC = log|Σ| + k log(T)/T, HQ = log|Σ| + 2k log(log(T))/T

**Minnesota Prior:** shrinks VAR toward n independent random walks
- Hyperparameters: τ (overall tightness), λ (cross-variable), d (lag decay), μ (sum-of-coefficients)
- Dummy observations approach: augmented regression Y* = [Y; Y_d], X* = [X; X_d]
- Optimization: maximize log marginal likelihood p(Y|τ) over grid

**SVAR Identification:** Y_t structural form with B_0 y_t = ... + ε_t
- Cholesky: P = chol(Σ), recursive ordering
- Sign restrictions: draw Q from uniform orthogonal, check sign(PQ * e_j) at horizons
- Narrative: additionally check structural shock signs at specific periods
- Long-run: Blanchard-Quah, C(1) = (I - A_1 - ... - A_p)^{-1}, P_LR = chol(C(1) Σ C(1)')
- Arias et al. (2018): zero + sign restrictions with importance sampling

**Local Projections:** y_{t+h} = α_h + β_h x_t + γ_h controls + u_{t+h}
- HAC: Newey-West with Bartlett/Parzen/QS kernels, auto bandwidth = floor(4(T/100)^{2/9})
- LP-IV: two-stage, first-stage F > 10 rule
- Smooth LP: β(h) = Σ θ_j B_j(h), B-spline basis, penalized min Σ(y - Xθ)² + λ θ'Ω θ
- State-dependent: F(z_t) logistic transition, regime-specific β_expansion / β_recession
- Propensity: IPW weights w_i = D_i/p(X_i) + (1-D_i)/(1-p(X_i))

**Factor Models:**
- Static: X = FΛ' + e, PCA on X'X/(NT), Bai-Ng IC for r selection
- Dynamic: X_t = Λ f_t + e_t, f_t = A_1 f_{t-1} + ... + A_p f_{t-p} + η_t
- GDFM: spectral density Σ_X(ω) = Λ(ω) Σ_f(ω) Λ(ω)* + Σ_e(ω)

**Unit Root Tests:**
- ADF: Δy_t = α + βt + γ y_{t-1} + Σ δ_j Δy_{t-j} + ε_t, H₀: γ=0 (unit root)
- KPSS: y_t = ξt + r_t + ε_t, H₀: σ²_η=0 (stationary) — reversed null
- PP: nonparametric correction to Dickey-Fuller
- ZA: endogenous structural break detection
- Ng-Perron: GLS-detrended, 4 statistics (MZa, MZt, MSB, MPT)
- Johansen: VECM rank test via trace λ_trace(r) = -T Σ log(1-λ̂_i) and max-eigenvalue

**ADF/KPSS Decision Matrix:**
| ADF | KPSS | Conclusion |
|-----|------|------------|
| Reject | Don't reject | Stationary |
| Don't reject | Reject | Unit root |
| Reject | Reject | Inconclusive (trend-stationary?) |
| Don't reject | Don't reject | Inconclusive (low power?) |

### Key References

- Antolín-Díaz & Rubio-Ramírez (2018). Narrative Sign Restrictions for SVARs. *AER*.
- Auerbach & Gorodnichenko (2013). State-dependent fiscal multipliers. *AER*.
- Bai & Ng (2002). Determining the number of factors. *Econometrica*.
- Bańbura, Giannone & Reichlin (2010). Large Bayesian VARs. *JAE*.
- Barnichon & Brownlees (2019). Impulse Response Estimation by Smooth LP. *RESTUD*.
- Blanchard & Quah (1989). Dynamic effects of demand and supply disturbances. *AER*.
- Forni, Hallin, Lippi & Reichlin (2000). The Generalized Dynamic Factor Model. *RESTUD*.
- Giannone, Lenza & Primiceri (2015). Prior selection for VARs. *RESTAT*.
- Jordà (2005). Estimation and inference of IRFs by LP. *AER*.
- Kilian & Lütkepohl (2017). *Structural VAR Analysis*. Cambridge.
- Lütkepohl (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Rubio-Ramírez, Waggoner & Zha (2010). Structural VARs. *RESTUD*.
- Stock & Watson (2002). Forecasting using principal components. *JASA*.
- Stock & Watson (2018). Identification and estimation of dynamic causal effects. *EJ*.
- Hyvärinen, Karhunen & Oja (2001). *Independent Component Analysis*. Wiley.
- Lanne, Meitz & Saikkonen (2017). Identification and estimation of non-Gaussian SVARs. *JBES*.
- Gouriéroux, Monfort & Renne (2017). Statistical inference for independent component analysis. *JoE*.
- Lütkepohl & Netšunajev (2017). Structural vector autoregressions with heteroskedasticity. *JoE*.

### CI/CD

- **CI:** GitHub Actions on push/PR — Julia 1.10 + latest on Ubuntu/macOS/Windows
- **Docs:** auto-deployed to GitHub Pages via Documenter.jl on push to main
- **Coverage:** Codecov integration, threshold 1%
- **Quality:** Aqua.jl package quality checks
