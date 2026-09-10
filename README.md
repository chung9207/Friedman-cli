# Friedman-cli

[![CI](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/CI.yml/badge.svg)](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/CI.yml)
[![Nightly](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/nightly.yml/badge.svg)](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/nightly.yml)
[![codecov](https://codecov.io/gh/FriedmanJP/Friedman-cli/graph/badge.svg?token=TIYTWTJG36)](https://codecov.io/gh/FriedmanJP/Friedman-cli)
[![Documentation](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/Documentation.yml/badge.svg)](https://friedmanjp.github.io/Friedman-cli/dev/)

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/FriedmanJP/MacroEconometricModels.jl) (v0.9.0).

> **v0.5.0+ agent contract:** with `--format=json`, stdout is **exactly one** versioned JSON envelope; status goes to **stderr**. Exit codes: 0 ok · 2 usage · 3 data · 4 config · 5 model · 6 env · 1 internal. Since v0.10.0 **every failure under JSON also emits exactly one error envelope** — including usage/parse errors that fail before a command resolves — with `error.code` (`class/code`) and `error.exit_code` always matching the process exit. Also since v0.10.0, envelope `data` keys are **stable, registry-declared addresses** (`var_coefficients`, never `var_2_coefficients`): predictable before the command runs, with per-variable families keyed `<name>_<your-variable>`. See the [Agent Guide](https://friedmanjp.github.io/Friedman-cli/dev/agent-guide/). `FRIEDMAN_LEGACY_OUTPUT=1` restores pre-0.5 JSON for one minor release.
>
> **v0.12.3 (MEMs 0.9.6 adoption):** targets MacroEconometricModels **v0.9.6** from the Julia General registry. Pin rule: CI/release/integration resolve against an **exact** pin, `[compat] MacroEconometricModels = "=0.9.6"` as of CLI v0.12.3 (canary/nightly-dev may track MEMs `dev`). The pin is exact because the root `Manifest.toml` is gitignored, so a release re-resolves deps — a caret range once floated an unvetted patch release into a shipped build. At this baseline JuMP + Ipopt remain required upstream deps (bundled — see the solver note below).

21 top-level commands (incl. `policy`, `multipliers`, `model`, `serve`, `show`, `completions`), 456 subcommands. Action-first CLI: commands are organized by action (`estimate`, `irf`, `forecast`, `dsge`, `did`, `spectral`, `io`, `policy`, `multipliers`, ...) rather than by model type. Features include VAR/BVAR/Panel VAR, time-varying-parameter VAR with stochastic volatility (Primiceri) and mixed-frequency VAR (Schorfheide-Song), Waggoner-Zha conditional/scenario forecasts, wild/block bootstrap and Kilian bias-corrected IRF bands, Pesaran-Shin generalized FEVD, FAVAR, structural DFM, cross-sectional regression (OLS/WLS/IV/Logit/Probit/ordered logit/ordered probit/multinomial logit), Koenker-Bassett quantile regression, regression discontinuity with CCT robust bias correction (sharp and fuzzy), count-data regression (Poisson with QMLE-sandwich SEs and offset/exposure, negative binomial NB2, incidence-rate ratios, and a Cameron-Trivedi overdispersion test), penalized regression (Lasso/Ridge/Elastic-Net), robust (Huber/bisquare M/MM), Tobit censored, truncated-normal and Heckman sample-selection regression, single-equation and panel cointegrating regression (FMOLS/CCR/DOLS), single-equation ARDL and nonlinear/asymmetric NARDL (with PSS bounds tests, symmetry Wald tests, and cumulative dynamic multipliers), dynamic heterogeneous-panel ARDL (Pooled Mean Group / Mean Group / Dynamic Fixed Effects, with a PMG Hausman selection test), mixed-frequency MIDAS regression (exp-Almon / Beta / Almon / U-MIDAS weights, with ADL-MIDAS autoregressive augmentation), k-class instrumental-variables estimation (2SLS/LIML/Fuller/generic k-class) with Stock-Yogo weak-instrument diagnostics and Anderson-Rubin weak-instrument-robust tests and confidence sets (unbounded, disjoint and empty sets all represented), Conley (1999) spatial-HAC standard errors, reghdfe-style high-dimensional fixed-effect absorption, wild cluster bootstrap for few-cluster inference (Cameron-Gelbach-Miller, with exact enumeration of the sign space), Montiel Olea-Pflueger effective-F and Anderson-Rubin bands for LP-IV, general-to-specific and stepwise variable selection (forward/backward/bidirectional/best-subset/GETS), structural state-space models (local level/local linear trend) and time-varying-parameter regression, nonparametric estimation (kernel density, kernel/local-polynomial regression, LOWESS), two-regime threshold regression split by an external variable (Hansen 1996/2000, with a bootstrap threshold confidence interval), self-exciting threshold autoregression (SETAR) and smooth-transition autoregression (STAR: LSTR1/LSTR2/ESTR) with Hansen (1996) and Teräsvirta LM3 linearity testing and bootstrap-simulation forecasts, Markov-switching autoregression (MS-AR, Hamilton mean-switching) and K-state Markov-switching regression (per-regime coefficients/variances and a wide regime-transition matrix), panel regression (POLS/FE/RE/FD/IV), local projections, policy counterfactuals and optimal policy (McKay-Wolf policy-rule counterfactuals from empirical causal-effect menus, optimal policy rules and counterfactual moments under quadratic loss, Barnichon-Mesters optimal policy perturbations with forecast-gap targeting including constrained and sequential variants, DSGE/HA news-shock menus and heterogeneous-agent Jacobians, historical counterfactuals, and spanning/sufficiency diagnostics), DSGE (including full Bayesian workflow with convergence/identification diagnostics -- R-hat/ESS/Geweke, Iskrev rank test, learning-rate, prior/posterior overlap, bridge-sampling marginal likelihood -- historical decomposition, and 3rd-order perturbation), DID/event study/LP-DiD, factor models, ARIMA, ARFIMA long memory (with GPH and local-Whittle `d` estimators), volatility models (ARCH/GARCH/EGARCH/GJR-GARCH/SV plus IGARCH/Component-GARCH/APARCH/FIGARCH/FIEGARCH/GARCH-MIDAS), multivariate GARCH (CCC/DCC/cDCC/BEKK) with sign-bias and Nyblom-stability residual diagnostics, non-Gaussian SVAR, GMM/SMM, forecast evaluation & combination (Diebold-Mariano, Clark-West, Mincer-Zarnowitz, encompassing, accuracy metrics, forecast combination), input-output analysis (Leontief/Ghosh multipliers, linkages, SDA, environmental footprints, Baqaee-Farhi), time series filtering, nowcasting, spectral analysis (ACF, periodogram, spectral density, cross-spectrum, transfer function), advanced unit root tests (Fourier ADF/KPSS, DF-GLS, LM with breaks, ADF 2-break, Gregory-Hansen), structural break tests (Andrews, Bai-Perron), panel unit root tests (PANIC, CIPS, Moon-Perron, factor break), Hadri panel stationarity, first-generation panel unit-root tests (Levin-Lin-Chu, Im-Pesaran-Shin, Breitung), Pedroni/Kao/Westerlund and Fisher-Johansen panel cointegration tests, Dumitrescu-Hurlin panel Granger causality, Beck-Katz panel-corrected standard errors and Prais-Winsten AR(1) panel regression, variance-ratio (Lo-MacKinlay/Chow-Denning) and BDS randomness/nonlinearity tests, HEGY seasonal unit-root and ERS point-optimal tests, SADF/GSADF (Phillips-Shi-Yu) explosive-bubble detection with dated episodes, EDF goodness-of-fit tests (KS/Lilliefors/Cramer-von Mises/Anderson-Darling/Watson), Engle-Granger and Phillips-Ouliaris residual-based cointegration tests with Hansen (1992) instability and Park (1990) added-variables diagnostics, OLS regression diagnostics (White/Glejser/Harvey heteroskedasticity, Chow structural-break, Brown-Durbin-Evans CUSUM/CUSUMSQ parameter stability, recursive residuals, and leverage/DFFITS/Cook's-distance influence statistics), Hansen (1996) SETAR and Teräsvirta LM3 STAR linearity tests, VECM cointegration restriction tests (β/α/weak-exogeneity/known-β/joint), VIF multicollinearity diagnostics, and data management.

### Agent quick start

```bash
# install (see below), then:
friedman estimate var data.csv --lags 1 --format json | jq .
friedman schema estimate var | jq '.input_schema'   # draft-07 invocation schema
friedman schema estimate var | jq '.tables'         # stable result-table keys
friedman schema | jq '.contract.exit_codes'         # the whole output contract
friedman schema --docs | jq -r '.docs'              # the full agent guide
echo $?   # 0 ok · 2 usage · 3 data · 4 config · 5 model · 6 env · 1 internal
```

## Stability and versioning

Friedman-cli follows Semantic Versioning, and — because the primary user is an
agent — the machine surface IS the API:

- **The API surface** = the command tree, the option/flag surface, the envelope
  schema (`schema/envelope-v1.json`), the stable `data` table keys, the
  error-code taxonomy (`class/code`), and the process exit codes.
- **Envelope schema v1 is additive-only from v0.10.0**: within
  `schema_version: 1` no key is removed or retyped; a breaking envelope change
  bumps `schema_version` to 2 and is a **major** release. The formal freeze
  declaration lands at v1.0.0.
- **0.x minors are additive-only**: new commands, options, tables, and error
  codes may appear; nothing is removed or renamed. Removals/renames happen only
  at majors, after at least one minor of deprecation alias.
- **Scheduled for v1.0.0**: removal of the hidden snake_case aliases (they warn
  on stderr today) and of `FRIEDMAN_LEGACY_OUTPUT`.

## Installation

### Quick Install

**macOS and Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.sh | bash
```

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.ps1 | iex
```

The installer checks for Julia 1.12 (installs [juliaup](https://github.com/JuliaLang/juliaup) if needed, without changing your default Julia version), downloads a precompiled sysimage, and installs to `~/.friedman-cli/`.

**Supported platforms:** macOS ARM64 (Apple Silicon), Linux x86_64, Windows x86_64.

> **Solver dependencies:** From the v0.7.0 line onward (the MacroEconometricModels 0.7.0 adoption), JuMP (MPL-2.0) and Ipopt (MIT Julia wrapper over the EPL-2.0 Ipopt library, linked dynamically as a separate work) are required upstream dependencies and **are bundled** in the precompiled release — DSGE constrained optimization (OccBin, etc.) works out of the box, with no separate install. PATHSolver remains optional and is not bundled — add it with `Pkg.add("PATHSolver")` if a model needs the PATH solver. (Older v0.6.x releases bundle none of the three; on those, install JuMP and Ipopt yourself with `julia -e 'using Pkg; Pkg.add(["JuMP", "Ipopt"])'`.)

### Install from Source

```bash
git clone https://github.com/FriedmanJP/Friedman-cli.git
cd Friedman-cli
julia --project -e 'using Pkg; Pkg.instantiate()'
```

See [Installation docs](https://friedmanjp.github.io/Friedman-cli/dev/installation/) for specific version install, manual install from GitHub Releases, upgrading, and uninstalling.

## Usage

```bash
friedman [command] [subcommand] [args...] [options...]
```

### Commands

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `estimate` | `var` `bvar` `tvpvar` `mfvar` `qreg` `rdd` `lp` `arima` `arfima` `gmm` `smm` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr-garch` `sv` `igarch` `cgarch` `aparch` `figarch` `fiegarch` `garch-midas` `ccc` `dcc` `bekk` `lasso` `ridge` `elastic-net` `robust` `tobit` `truncreg` `heckman` `cointreg` `xtcointreg` `ardl` `nardl` `pmg` `midas` `threshold` `setar` `star` `ms-ar` `ms` `statespace` `tvp` `kde` `kernel-reg` `lowess` `fastica` `ml` `vecm` `pvar` `favar` `sdfm` `reg` `iv` `select` `sur` `3sls` `logit` `probit` `ologit` `oprobit` `mlogit` `preg` `pols` `pfe` `pre` | Estimate models (66 model types, incl. ARFIMA long memory, 6 univariate GARCH variants, multivariate GARCH CCC/DCC/BEKK, penalized (Lasso/Ridge/Elastic-Net) + robust + Tobit + truncated + Heckman sample-selection regression, single-equation (FMOLS/CCR/DOLS) + panel (group/pooled) cointegrating regression, single-equation ARDL + nonlinear/asymmetric NARDL (with folded long-run + ECM/bounds), dynamic heterogeneous-panel ARDL (PMG/MG/DFE), mixed-frequency MIDAS/ADL-MIDAS/U-MIDAS, two-regime threshold regression split by an external variable, self-exciting threshold autoregression (SETAR, with attached Hansen 1996 linearity test), smooth-transition autoregression (STAR: LSTR1/LSTR2/ESTR, Teräsvirta NLS), Markov-switching (MS-AR mean-switching + K-state MS regression, with a wide regime-transition matrix), structural state-space (local level/linear trend) + TVP regression, nonparametric (KDE/kernel-reg/LOWESS), SUR/3SLS systems, panel reg, ordered/multinomial choice) |
| `test` | `adf` `kpss` `pp` `za` `np` `ers` `hegy` `sadf` `gsadf` `edf` `gph` `local-whittle` `variance-ratio` `bds` `hansen-linearity` `star-linearity` `weak-instrument` `engle-granger` `phillips-ouliaris` `hansen-instability` `park-added` `white` `glejser` `harvey` `chow` `cusum` `cusumsq` `recursive-residuals` `influence` `llc` `ips` `breitung` `fisher-johansen` `dh-causality` `johansen` `normality` `identifiability` `heteroskedasticity` `arch-lm` `ljung-box` `sign-bias` `nyblom` `ardl-bounds` `nardl-symmetry` `pmg-hausman` `granger` `lr` `lm` `andrews` `bai-perron` `panic` `cips` `moon-perron` `factor-break` `hadri` `pedroni` `kao` `westerlund` `fourier-adf` `fourier-kpss` `dfgls` `lm-unitroot` `adf-2break` `gregory-hansen` `vif` + panel spec tests + spectral diagnostics + discrete choice tests + `var` (`lagselect` `stability`) + `pvar` (`hansen-j` `mmsc` `lagselect` `stability`) + `vecm` (`beta` `alpha` `weak-exog` `known-beta` `joint`) | Statistical tests (77+ leaves + nested; incl. GPH & local-Whittle long-memory `d`, variance-ratio & BDS randomness, Hansen (1996) SETAR & Teräsvirta LM3 STAR linearity, Stock-Yogo weak-instrument diagnostics, sign-bias & Nyblom volatility diagnostics, ARDL PSS bounds & NARDL symmetry tests, PMG Hausman panel selection test, VECM cointegration restriction tests, Hadri panel stationarity + Pedroni/Kao/Westerlund panel cointegration, HEGY seasonal unit roots, ERS point-optimal, SADF/GSADF explosive-bubble detection, EDF goodness-of-fit, Engle-Granger/Phillips-Ouliaris residual cointegration with Hansen instability and Park added-variables diagnostics, and cross-section OLS diagnostics -- White/Glejser/Harvey heteroskedasticity, Chow structural break, CUSUM/CUSUMSQ stability, recursive residuals and influence statistics, first-generation panel unit-root tests (LLC/IPS/Breitung), Fisher-Johansen panel cointegration and Dumitrescu-Hurlin panel causality) |
| `irf` | `var` `bvar` `tvpvar` `lp` `vecm` `pvar` `favar` `sdfm` | Impulse response functions (`tvpvar` is date-specific: `--date`) |
| `fevd` | `var` `bvar` `lp` `vecm` `pvar` `favar` `sdfm` | Forecast error variance decomposition (`var` adds `--generalized` Pesaran-Shin) |
| `hd` | `var` `bvar` `lp` `vecm` `favar` | Historical decomposition |
| `forecast` | `var` `bvar` `scenario` `lp` `arima` `setar` `star` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr-garch` `sv` `vecm` `favar` · `evaluate metrics/dm/clark-west/mincer-zarnowitz/encompassing/combine` | Forecasting (16 model types, incl. SETAR/STAR bootstrap-simulation forecasts) + evaluation & combination sub-family |
| `predict` | `3sls` `aparch` `arch` `arfima` `arima` `bvar` `cgarch` `dynamic` `egarch` `favar` `fiegarch` `figarch` `garch` `garch-midas` `gjr-garch` `igarch` `logit` `mlogit` `ologit` `oprobit` `piv` `plogit` `pprobit` `preg` `probit` `reg` `statespace` `static` `sur` `sv` `var` `vecm` | In-sample fitted values (32 model types; `statespace` emits the filtered/smoothed state paths, `sur`/`3sls` one long per-equation table) |
| `residuals` | `3sls` `aparch` `arch` `arfima` `arima` `bvar` `cgarch` `dynamic` `egarch` `favar` `fiegarch` `figarch` `garch` `garch-midas` `gjr-garch` `igarch` `logit` `mlogit` `ms` `ms-ar` `ologit` `oprobit` `piv` `plogit` `pprobit` `preg` `probit` `reg` `setar` `star` `statespace` `static` `sur` `sv` `var` `vecm` | Model residuals (36 model types; `statespace` emits Kalman innovations, `--standardized` for `v_t/√F_t`; `setar`/`star`/`ms`/`ms-ar` have no `predict` counterpart -- MEMs defines no `predict`/`fitted` for them) |
| `filter` | `hp` `hamilton` `bn` `bk` `bhp` `x13` | Time series filters (+ X-13ARIMA-SEATS) |
| `data` | `list` `load` `import` `export` `describe` `diagnose` `fix` `transform` `filter` `validate` `balance` `dropna` `keeprows` | Data management (13 leaves) |
| `io` | `sources` `download` `load` `leontief` `ghosh` `multipliers` `linkages` `key-sectors` `sda` `extract` `footprint` `baqaee-farhi` | Input-output analysis (12 leaves) |
| `nowcast` | `dfm` `bvar` `bridge` `news` `forecast` | Nowcasting (DFM, BVAR, bridge equations) |
| `dsge` | RA leaves + `bayes` + `ha` + `ct` (solve, transition) + `olg` (solve, simulate) | RA, Bayesian, HA, continuous-time Aiyagari, Blanchard OLG (MEMs 0.7.0) |
| `spectral` | `acf` `periodogram` `density` `cross` `transfer` | Spectral analysis (ACF, periodogram, spectral density, cross-spectrum, transfer function) |
| `did` | `estimate` `event-study` `lp-did` + `test` (`bacon` `pretrend` `negweight` `honest`) | Difference-in-differences (3 + 4 nested) |
| `multipliers` | `nardl` | NARDL cumulative asymmetric dynamic multipliers (m⁺/m⁻ response curves + bootstrap bands) |
| `policy` | `effects` (`var` `bvar` `lp` `sign`) + `counterfactual` (`var` `bvar` `lp`) + `optimal` (`var` `bvar` `lp`) + `moments` (`var` `bvar`) + `opp` / `opp-sequence` (`var` `bvar`) + `history` (`var` `bvar`) + `news` (`dsge` `ha`) + `jacobian` (`ha`) + `spanning` (`var`) + `sufficiency` (`dsge`) | Policy counterfactuals & optimal policy (McKay-Wolf causal-effect menus, Barnichon-Mesters OPP, rule counterfactuals, structural news/Jacobian routes; 23 leaves) |
| `schema` | — | Machine-readable self-description (raw JSON): per-leaf draft-07 `input_schema` + declared result-table keys; root `contract` (embedded envelope schema + exit codes); `--docs` embeds the agent guide |
| `serve` | — | MCP server over stdio (`--mcp`): every command as a Model Context Protocol tool — `tools/call` returns the JSON envelope verbatim; in-memory `model://` handles let one session estimate once and reuse the fit |

All commands support `--format` (`table`|`csv`|`json`) and `--output` (file path) options.

**Global flags (leading only):** `--version`/`-V`, `--warranty`, `--conditions`, plus agent globals `--quiet`/`-q`, `--no-color`, `--seed N`, `--json`. Mid-argv they are ordinary tokens (an unknown option is a usage error, exit 2) — they are never swallowed out of a command line. `--help`/`-h` prints help at any position; `-h` is reserved for help on every leaf, and the registry refuses any option/flag that tries to claim a reserved name or short (`help`, `version`, `warranty`, `conditions`, `-h`, `-V`).

### Interactive REPL

Launch an interactive session with `friedman repl`:

```bash
friedman repl
```

The REPL provides:
- **Session data** -- Load data once, use across commands: `data use mydata.csv` or `data use :fred-md`
- **Result caching** -- Estimation results cached automatically, reused by downstream commands
- **Tab completion** -- Commands, subcommands, and options
- **REPL-only commands** -- `data use`, `data current`, `data clear`, `exit`/`quit`

```
friedman> data use :fred-md
Loaded :fred-md (804x126, vars: INDPRO, CPIAUCSL, ...)

friedman> estimate var --lags 4
[estimation output]
Result cached as :var

friedman> irf var --horizons 20
[uses cached VAR model -- no re-estimation]

friedman> data current
:fred-md (804x126)
Cached results: var
```

### Estimation

```bash
# VAR(2)
friedman estimate var data.csv --lags=2

# Bayesian VAR with NUTS sampler
friedman estimate bvar data.csv --lags=4 --draws=2000 --sampler=nuts

# Bayesian VAR with Minnesota prior config
friedman estimate bvar data.csv --config=prior.toml

# Bayesian posterior summary (mean or median)
friedman estimate bvar data.csv --lags=4 --method=mean

# Local Projections (Jorda 2005)
friedman estimate lp data.csv --shock=1 --horizons=20 --vcov=newey_west

# LP-IV (Stock & Watson 2018)
friedman estimate lp data.csv --method=iv --shock=1 --instruments=instruments.csv

# Smooth LP (Barnichon & Brownlees 2019) — auto-selects lambda via CV
friedman estimate lp data.csv --method=smooth --shock=1 --horizons=20

# State-dependent LP (Auerbach & Gorodnichenko 2013)
friedman estimate lp data.csv --method=state --shock=1 --state-var=2 --gamma=1.5

# Propensity score LP (Angrist et al. 2018)
friedman estimate lp data.csv --method=propensity --treatment=1 --score-method=logit

# Doubly robust LP
friedman estimate lp data.csv --method=robust --treatment=1 --score-method=logit

# ARIMA — explicit or auto order selection
friedman estimate arima data.csv --p=1 --d=1 --q=1
friedman estimate arima data.csv --criterion=bic   # auto-select

# GMM
friedman estimate gmm data.csv --config=gmm_spec.toml --weighting=twostep

# Simulated Method of Moments (SMM) — --config required (names a built-in simulator + theta0)
friedman estimate smm gdp.csv --config=smm_ar1.toml

# Static factor model (PCA) — auto-selects factors via Bai-Ng IC
friedman estimate static data.csv
friedman estimate static data.csv --nfactors=3 --criterion=ic2

# Dynamic factor model
friedman estimate dynamic data.csv --nfactors=2 --factor-lags=1

# Generalized dynamic factor model (spectral)
friedman estimate gdfm data.csv --dynamic-rank=2

# ICA-based SVAR identification (FastICA, JADE, SOBI, dCov, HSIC)
friedman estimate fastica data.csv --method=fastica --contrast=logcosh
friedman estimate fastica data.csv --method=jade

# Maximum likelihood non-Gaussian SVAR
friedman estimate ml data.csv --distribution=student_t
friedman estimate ml data.csv --distribution=mixture_normal

# Vector Error Correction Model (Johansen)
friedman estimate vecm data.csv --lags=2
friedman estimate vecm data.csv --rank=1 --deterministic=constant

# Panel VAR (GMM or FE-OLS)
friedman estimate pvar data.csv --id-col=country --time-col=year --lags=2
friedman estimate pvar data.csv --id-col=country --time-col=year --method=feols

# Factor-Augmented VAR (FAVAR)
friedman estimate favar data.csv --lags=4 --nfactors=3
friedman estimate favar data.csv --lags=4 --nfactors=3 --slow-vars=1,2,3

# Structural Dynamic Factor Model (SDFM)
friedman estimate sdfm data.csv --nfactors=3 --factor-lags=2
friedman estimate sdfm data.csv --nfactors=3 --id=cholesky
```

### Cross-Sectional Regression

```bash
# OLS regression (first column = dependent, rest = regressors)
friedman estimate reg data.csv --dep=wage --cov-type=hc1

# Weighted Least Squares
friedman estimate reg data.csv --dep=wage --weights=pop_weight

# IV (2SLS) regression
friedman estimate iv data.csv --dep=wage --endogenous=educ --instruments=father_educ,mother_educ

# Logit (binary choice)
friedman estimate logit data.csv --dep=employed --cov-type=hc1

# Probit
friedman estimate probit data.csv --dep=employed --clusters=state

# Predictions with marginal effects
friedman predict logit data.csv --dep=employed --marginal-effects

# VIF multicollinearity check
friedman test vif data.csv --dep=wage
```

### ARDL / NARDL (single-equation cointegration)

```bash
# ARDL(p, q): coefficients + long-run multipliers + ECM speed of adjustment (folded in)
friedman estimate ardl data.csv --dep=y --p=1 --q=1 --case=3

# Pesaran-Shin-Smith bounds test for a level relationship (decision symbols, NO p-value)
friedman test ardl-bounds data.csv --dep=y --p=1 --q=1 --case=3 --level=0.05

# Nonlinear (asymmetric) ARDL: split every regressor into +/- partial sums
friedman estimate nardl data.csv --dep=y --asymmetric=all --p=1 --q=1

# Long- and short-run symmetry Wald tests (H0: theta+ = theta-)
friedman test nardl-symmetry data.csv --dep=y --asymmetric=all

# Cumulative asymmetric dynamic multipliers with bootstrap bands
friedman multipliers nardl data.csv --dep=y --horizon=24 --nreps=500
```

### PMG / MG / DFE (dynamic heterogeneous-panel ARDL)

```bash
# Pooled Mean Group: common long-run theta + heterogeneous short-run/EC speeds (long panel)
friedman estimate pmg panel.csv --id-col=id --time-col=time --dep=y --indep=x1,x2 --method=pmg

# Mean Group (per-unit averaged) and Dynamic Fixed Effects alternatives
friedman estimate pmg panel.csv --dep=y --indep=x1,x2 --method=mg
friedman estimate pmg panel.csv --dep=y --indep=x1,x2 --method=dfe

# PMG Hausman selection test (H0: long-run homogeneity; low p favours MG)
friedman test pmg-hausman panel.csv --dep=y --indep=x1,x2 --efficient=pmg
```

### MIDAS (mixed-frequency regression)

```bash
# Nowcast a quarterly target from a monthly indicator (m=3, 6 HF lags, exp-Almon weights)
# --data = low-frequency target CSV; --hf-data = high-frequency indicator CSV (aligned: len(HF)=m*len(LF))
friedman estimate midas gdp_q.csv --hf-data ip_m.csv --m 3 --k 6 --weights expalmon

# ADL-MIDAS (one AR lag of the target) with Beta weights; unrestricted U-MIDAS (OLS on K lags)
friedman estimate midas gdp_q.csv --hf-data ip_m.csv --m 3 --k 6 --weights beta2 --p-ar 1
friedman estimate midas gdp_q.csv --hf-data ip_m.csv --m 3 --k 6 --weights umidas
```

### Nonlinear Time Series (threshold regression / SETAR / STAR)

```bash
# Two-regime threshold regression: y on the remaining numeric columns, sample split by z
# (z is the splitting variable and is EXCLUDED from the regressors)
friedman estimate threshold data.csv --dep y --threshold-col z

# 90% Hansen (2000) threshold CI, heteroskedastic bootstrap, skip the linearity test
friedman estimate threshold data.csv --dep y --threshold-col z --ci-level 0.90 --het --no-linearity

# Self-exciting threshold AR: SETAR(2; 1, 1), delay d=1, with an attached Hansen (1996) linearity test
friedman estimate setar y.csv --p 1 --d 1

# Auto-select the delay over the 1:p grid; standalone Hansen linearity test; SETAR bootstrap forecast
friedman estimate setar y.csv --p 2 --d auto
friedman test hansen-linearity y.csv --p 1 --d 1
friedman forecast setar y.csv --p 1 --d 1 --horizons 12

# Smooth-transition AR (STAR): Teräsvirta auto-select the transition shape (LSTR1/LSTR2/ESTR)
friedman estimate star y.csv --p 1 --d 1 --type auto

# Fix the shape (e.g. logistic one-location LSTR1); Teräsvirta LM3 linearity test; STAR bootstrap forecast
friedman estimate star y.csv --p 2 --type lstr1
friedman test star-linearity y.csv --p 1 --d 1
friedman forecast star y.csv --p 1 --d 1 --horizons 12

# Markov-switching AR (Hamilton mean-switching, 2 regimes, common variance)
friedman estimate ms-ar y.csv --p 1 --k-regimes 2
# ... with per-regime (switching) variances
friedman estimate ms-ar y.csv --p 1 --switching-variance

# K-state Markov-switching regression: dep = y, regressors = the other numeric columns (add a const)
friedman estimate ms data.csv --dep y
# Intercept-only switching-mean model (dep is the only numeric column), common variance
friedman estimate ms y.csv --no-switching-variance
```

### Volatility Models

```bash
# ARCH(q)
friedman estimate arch data.csv --column=1 --q=1

# GARCH(p,q)
friedman estimate garch data.csv --column=1 --p=1 --q=1

# EGARCH(p,q)
friedman estimate egarch data.csv --column=1 --p=1 --q=1

# GJR-GARCH(p,q)
friedman estimate gjr-garch data.csv --column=1 --p=1 --q=1

# Stochastic volatility
friedman estimate sv data.csv --column=1 --draws=2000

# Multivariate GARCH (T×n returns matrix; conditional correlation matrix)
friedman estimate ccc returns.csv --p=1 --q=1
friedman estimate dcc returns.csv --correction=aielli   # cDCC
friedman estimate bekk returns.csv --kind=diagonal

# Volatility residual diagnostics (fit a model, test its residuals)
friedman test sign-bias data.csv --column=1 --model=garch
friedman test nyblom data.csv --column=1 --model=gjr-garch
```

### Testing

```bash
# Unit root tests
friedman test adf data.csv --column=1 --trend=constant
friedman test kpss data.csv --column=1 --trend=constant
friedman test pp data.csv --column=1
friedman test za data.csv --column=1 --trend=both --trim=0.15
friedman test np data.csv --column=1

# Cointegration
friedman test johansen data.csv --lags=2 --trend=constant

# VAR diagnostics (nested under test var)
friedman test var lagselect data.csv --max-lags=12 --criterion=aic
friedman test var stability data.csv --lags=2

# Non-Gaussian SVAR diagnostics
friedman test normality data.csv --lags=4
friedman test identifiability data.csv --test=all
friedman test heteroskedasticity data.csv --method=markov --regimes=2

# Residual diagnostics
friedman test arch-lm data.csv --lags=4
friedman test ljung-box data.csv --lags=10

# Granger causality
friedman test granger data.csv --cause=1 --effect=2 --lags=4
friedman test granger data.csv --all --lags=4

# Model comparison (LR and LM tests)
friedman test lr data.csv data.csv --lags1=2 --lags2=4
friedman test lm data.csv data.csv --lags1=2 --lags2=4

# Panel VAR diagnostics
friedman test pvar hansen-j data.csv --id-col=country --time-col=year --lags=2
friedman test pvar mmsc data.csv --id-col=country --time-col=year --max-lags=8
friedman test pvar lagselect data.csv --id-col=country --time-col=year --max-lags=8
friedman test pvar stability data.csv --id-col=country --time-col=year --lags=2

# Structural break tests
friedman test andrews data.csv --column=1 --trim=0.15
friedman test bai-perron data.csv --column=1 --max-breaks=5

# Panel unit root tests
friedman test panic data.csv --id-col=country --time-col=year --column=gdp
friedman test cips data.csv --id-col=country --time-col=year --column=gdp --lags=1
friedman test moon-perron data.csv --id-col=country --time-col=year --column=gdp
friedman test factor-break data.csv --id-col=country --time-col=year --column=gdp

# Advanced unit root tests
friedman test fourier-adf data.csv --column=1 --regression=constant --fmax=3
friedman test fourier-kpss data.csv --column=1 --regression=constant --fmax=3
friedman test dfgls data.csv --column=1 --regression=constant
friedman test lm-unitroot data.csv --column=1 --breaks=1 --regression=level
friedman test adf-2break data.csv --column=1 --model=level --trim=0.10
friedman test gregory-hansen data.csv --model=C --lags=aic

# Multicollinearity diagnostics
friedman test vif data.csv --dep=wage --cov-type=hc1
```

### Impulse Response Functions

```bash
# Frequentist IRF (Cholesky identification)
friedman irf var data.csv --shock=1 --horizons=20 --id=cholesky

# Sign restrictions (requires config)
friedman irf var data.csv --id=sign --config=sign_restrictions.toml

# Sign identified set (full set of accepted rotations)
friedman irf var data.csv --id=sign --config=sign_restrictions.toml --identified-set

# Cumulative IRFs (for differenced data)
friedman irf var data.csv --shock=1 --horizons=20 --cumulative

# Narrative sign restrictions
friedman irf var data.csv --id=narrative --config=narrative.toml

# Long-run (Blanchard-Quah) identification
friedman irf var data.csv --id=longrun --horizons=40

# Arias et al. (2018) zero/sign restrictions
friedman irf var data.csv --id=arias --config=arias_restrictions.toml

# Uhlig (Mountford & Uhlig 2009) penalty-based identification
friedman irf var data.csv --id=uhlig --config=uhlig_restrictions.toml

# With bootstrap confidence intervals
friedman irf var data.csv --shock=1 --ci=bootstrap --replications=1000

# Bayesian IRFs (posterior credible intervals)
friedman irf bvar data.csv --shock=1 --horizons=20
friedman irf bvar data.csv --draws=5000 --sampler=gibbs --config=prior.toml

# Structural LP IRFs
friedman irf lp data.csv --id=cholesky --shock=1 --horizons=20
friedman irf lp data.csv --shocks=1,2,3 --id=cholesky --horizons=30

# VECM IRFs
friedman irf vecm data.csv --shock=1 --horizons=20 --rank=2

# Panel VAR IRFs (OIRF or GIRF)
friedman irf pvar data.csv --id-col=country --time-col=year --horizons=20
friedman irf pvar data.csv --irf-type=girf --horizons=12

# FAVAR IRFs
friedman irf favar data.csv --shock=1 --horizons=20 --nfactors=3

# Structural DFM IRFs
friedman irf sdfm data.csv --shock=1 --horizons=20 --nfactors=3
```

### FEVD

```bash
# Frequentist FEVD
friedman fevd var data.csv --horizons=20 --id=cholesky
friedman fevd var data.csv --id=sign --config=sign_restrictions.toml

# Bayesian FEVD
friedman fevd bvar data.csv --horizons=20

# LP FEVD (bias-corrected, Gorodnichenko & Lee 2019)
friedman fevd lp data.csv --horizons=20 --id=cholesky

# VECM FEVD
friedman fevd vecm data.csv --horizons=20 --rank=2

# Panel VAR FEVD
friedman fevd pvar data.csv --id-col=country --time-col=year --horizons=20

# FAVAR FEVD
friedman fevd favar data.csv --horizons=20 --nfactors=3

# Structural DFM FEVD
friedman fevd sdfm data.csv --horizons=20 --nfactors=3
```

### Historical Decomposition

```bash
friedman hd var data.csv --id=cholesky
friedman hd var data.csv --id=longrun --lags=4
friedman hd bvar data.csv --draws=2000
friedman hd lp data.csv --id=cholesky

# VECM historical decomposition
friedman hd vecm data.csv --id=cholesky --rank=2

# FAVAR historical decomposition
friedman hd favar data.csv --id=cholesky --nfactors=3
```

### Forecasting

```bash
# VAR forecast with confidence intervals
friedman forecast var data.csv --horizons=12 --confidence=0.95

# VAR forecast with bootstrap confidence intervals
friedman forecast var data.csv --horizons=12 --ci=bootstrap --replications=500

# Bayesian forecast (posterior credible intervals)
friedman forecast bvar data.csv --horizons=12 --draws=2000

# Direct LP forecast
friedman forecast lp data.csv --shock=1 --horizons=12 --shock-size=1.0

# ARIMA forecast (auto model selection + h-step forecast)
friedman forecast arima data.csv --horizons=12 --confidence=0.95

# Factor model forecasting
friedman forecast static data.csv --horizon=12
friedman forecast dynamic data.csv --nfactors=2 --factor-lags=1 --horizon=12
friedman forecast gdfm data.csv --dynamic-rank=2 --horizon=12

# Volatility model forecasting
friedman forecast arch data.csv --column=1 --horizons=12
friedman forecast garch data.csv --column=1 --horizons=12
friedman forecast egarch data.csv --column=1 --horizons=12
friedman forecast gjr-garch data.csv --column=1 --horizons=12
friedman forecast sv data.csv --column=1 --horizons=12

# VECM forecast (bootstrap CIs)
friedman forecast vecm data.csv --horizons=12 --rank=2

# FAVAR forecast
friedman forecast favar data.csv --horizons=12 --nfactors=3
```

### Predict & Residuals

```bash
# In-sample fitted values
friedman predict var data.csv --lags=2
friedman predict bvar data.csv --lags=4 --draws=2000
friedman predict arima data.csv --p=1 --d=1 --q=1
friedman predict vecm data.csv --rank=1
friedman predict static data.csv --nfactors=3
friedman predict garch data.csv --column=1 --p=1 --q=1
friedman predict favar data.csv --lags=4 --nfactors=3
friedman predict reg data.csv --dep=wage
friedman predict logit data.csv --dep=employed --marginal-effects
friedman predict probit data.csv --dep=employed --classification-table

# Model residuals
friedman residuals var data.csv --lags=2
friedman residuals bvar data.csv --lags=4 --draws=2000
friedman residuals arima data.csv --p=1 --d=1 --q=1
friedman residuals vecm data.csv --rank=1
friedman residuals static data.csv --nfactors=3
friedman residuals garch data.csv --column=1 --p=1 --q=1
friedman residuals favar data.csv --lags=4 --nfactors=3
friedman residuals reg data.csv --dep=wage
friedman residuals logit data.csv --dep=employed
friedman residuals probit data.csv --dep=employed
```

### Filters

```bash
# Hodrick-Prescott filter
friedman filter hp data.csv --column=1 --lambda=1600

# Hamilton (2018) filter
friedman filter hamilton data.csv --column=1 --h=8 --p=4

# Beveridge-Nelson decomposition (ARIMA or state-space method)
friedman filter bn data.csv --column=1
friedman filter bn data.csv --column=1 --method=statespace

# Baxter-King band-pass filter
friedman filter bk data.csv --column=1 --pl=6 --pu=32

# Boosted HP filter (Phillips & Shi 2021)
friedman filter bhp data.csv --column=1 --lambda=1600 --stopping=BIC

# X-13ARIMA-SEATS seasonal adjustment
friedman filter x13 monthly.csv --frequency=12 --method=x11
```

### Data Management

```bash
# List available example datasets
friedman data list

# Load an example dataset (see `data list` for all 11)
friedman data load fred_md --output=fred_md.csv
friedman data load fred_md --vars=INDPRO,CPIAUCSL --transform

# Load your own CSV (no dataset name needed)
friedman data load --path=mydata.csv --output=loaded.csv

# Any <data> argument also accepts a `:name` reference to a bundled dataset
friedman data describe :fred_md

# Import CSV to a typed handle (stem; .jld2 is storage, not argv)
friedman data import data.csv --kind timeseries --frequency quarterly -o macro
friedman estimate var macro --lags 2 --save-model var   # stem → var.jld2
friedman irf var --model var --horizons 12 --save-result irf
friedman show irf                                # re-render the saved ImpulseResponse
friedman data export macro -o macro.csv          # inverse of import

# Describe data (summary statistics; CSV, :example, or typed handle stem)
friedman data describe data.csv
friedman data describe macro

# Diagnose data quality (NaN, Inf, constant columns)
friedman data diagnose data.csv
friedman data diagnose panel

# Fix data issues
friedman data fix data.csv --method=interpolate --output=cleaned.csv

# Apply transformation codes
friedman data transform data.csv --tcodes=1,5,5,2 --output=transformed.csv

# Filter data (unified interface)
friedman data filter data.csv --method=hp --lambda=1600

# Validate data for a specific model type (--model is a type string, not a handle)
friedman data validate data.csv --model=var
friedman data validate panel --model=var

# Balance panel with missing data via DFM imputation
friedman data balance data.csv --method=dfm --factors=3
```

### Nowcasting

```bash
# Dynamic Factor Model nowcast (EM algorithm)
friedman nowcast dfm data.csv --monthly-vars=4 --quarterly-vars=1 --factors=2

# Bayesian VAR nowcast
friedman nowcast bvar data.csv --monthly-vars=4 --quarterly-vars=1 --lags=5

# Bridge equation nowcast
friedman nowcast bridge data.csv --monthly-vars=4 --quarterly-vars=1

# News decomposition (Banbura & Modugno 2014)
friedman nowcast news --data-new=new.csv --data-old=old.csv --monthly-vars=4 --quarterly-vars=1

# Forecast from a nowcasting model
friedman nowcast forecast data.csv --method=dfm --horizons=4
```

### Input-Output Analysis

```bash
# Inspect the bundled Miller & Blair example (no data needed)
friedman io load

# Leontief inverse (total requirements), Ghosh inverse, multipliers
friedman io leontief --matrix both
friedman io multipliers --kind output --type I

# Linkages / key sectors, structural decomposition, extraction
friedman io linkages --forward ghosh
friedman io sda --data period0.csv --data2 period1.csv --n-sectors 35
friedman io extract --sectors-extract Manufacturing

# Environmental footprint + Baqaee-Farhi decomposition
friedman io footprint --account CO2 --detail
friedman io baqaee-farhi --second-order

# Download MRIO tables (network; --offline refuses with exit 6)
friedman io sources
friedman io download --source oecd --storage ./io_data --version v2023
```

### DSGE Models

```bash
# Solve a DSGE model (from TOML specification)
friedman dsge solve model.toml --method=gensys
friedman dsge solve model.toml --method=perturbation --order=2
friedman dsge solve model.toml --method=perturbation --order=3
friedman dsge solve model.toml --method=projection --degree=5 --grid=chebyshev

# HA-DSGE builtins (MEMs 0.7.0): huggett | krusell-smith | one-asset-hank | two-asset-hank
friedman dsge ha steady-state huggett
friedman dsge ha solve huggett --method=reiter --n-reduced=20
friedman dsge ha irf huggett --method=reiter --horizon=40
friedman dsge ha inequality-irf huggett --method=reiter
# Bayesian HA estimation (RWMH; un-deferred in v0.6.0 after MEMs#228)
friedman dsge ha estimate krusell-smith --data aggregates.csv --priors priors.toml --observables K

# Solve with OccBin occasionally binding constraints (e.g., ZLB)
friedman dsge solve model.toml --method=gensys --constraints=zlb.toml --periods=40

# Solve from Julia model file
friedman dsge solve model.jl --method=klein

# Impulse response functions
friedman dsge irf model.toml --horizon=40 --shock-size=1.0
friedman dsge irf model.toml --method=perturbation --order=2 --n-sim=500

# OccBin piecewise-linear IRFs
friedman dsge irf model.toml --constraints=zlb.toml --horizon=40

# Forecast error variance decomposition
friedman dsge fevd model.toml --horizon=40

# Simulate time series from solved model
friedman dsge simulate model.toml --periods=200 --burn=100 --seed=42
friedman dsge simulate model.toml --method=perturbation --antithetic

# Estimate DSGE parameters
friedman dsge estimate model.toml --data=macro.csv --method=irf_matching --params=rho,sigma
friedman dsge estimate model.toml --data=macro.csv --method=smm --params=alpha,beta --sim-ratio=5

# Perfect foresight transition path
friedman dsge perfect-foresight model.toml --shocks=shock_path.csv --periods=100

# Bayesian DSGE estimation (posterior sampling)
friedman dsge bayes estimate model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --method=smc
friedman dsge bayes estimate model.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --method=rwmh --n-draws=10000

# Bayesian DSGE post-estimation
friedman dsge bayes irf model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --horizon=40
friedman dsge bayes fevd model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --horizon=40
friedman dsge bayes simulate model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --periods=200
friedman dsge bayes summary model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml
friedman dsge bayes compare model1.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --model2=model2.toml --params2=rho,sigma --priors2=priors2.toml
friedman dsge bayes predictive model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml --n-sim=100

# Bayesian DSGE diagnostics (convergence + identification)
friedman dsge bayes mcmc-diag model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml       # R-hat / ESS / Geweke
friedman dsge bayes identification model.toml --params=rho,sigma --observables=Y                          # Iskrev (2010) rank test (no MCMC)
friedman dsge bayes learning-rate model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml     # Koop-Pesaran-Smith (2013)
friedman dsge bayes overlap model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml           # prior/posterior overlap
friedman dsge bayes marginal-lik model.toml --data=macro.csv --params=rho,sigma --priors=priors.toml      # bridge-sampling log-ML

# Compute steady state
friedman dsge steady-state model.toml
friedman dsge steady-state model.toml --constraints=zlb.toml
```

### Difference-in-Differences

```bash
# TWFE DID estimation
friedman did estimate panel.csv --outcome=y --treatment=treat --method=twfe

# Callaway-Sant'Anna (2021) with group-time ATT
friedman did estimate panel.csv --outcome=y --treatment=treat --method=cs --control-group=never_treated

# Sun-Abraham (2021)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=sa

# Borusyak-Jaravel-Spiess (2024) imputation estimator
friedman did estimate panel.csv --outcome=y --treatment=treat --method=bjs

# de Chaisemartin-D'Haultfoeuille (2020)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=dcdh --n-boot=500

# Panel event study LP (Jordà 2005 + panel FE)
friedman did event-study panel.csv --outcome=y --treatment=treat --leads=3 --horizon=5

# LP-DiD (Dube, Girardi, Jorda & Taylor 2025)
friedman did lp-did panel.csv --outcome=y --treatment=treat --horizon=5
friedman did lp-did panel.csv --outcome=y --treatment=treat --horizon=5 --reweight --pmd=ipw
friedman did lp-did panel.csv --outcome=y --treatment=treat --horizon=5 --notyet --only-pooled

# DID with base period control (CS method)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=cs --base-period=universal

# Bacon decomposition (Goodman-Bacon 2021) — diagnose TWFE bias
friedman did test bacon panel.csv --outcome=y --treatment=treat

# Pre-trend test for parallel trends assumption
friedman did test pretrend panel.csv --outcome=y --treatment=treat
friedman did test pretrend panel.csv --outcome=y --treatment=treat --method=event-study

# Negative weight check (de Chaisemartin-D'Haultfoeuille 2020)
friedman did test negweight panel.csv --treatment=treat

# HonestDiD sensitivity analysis (Rambachan-Roth 2023)
friedman did test honest panel.csv --outcome=y --treatment=treat --mbar=1.0
```

## Output Formats

All commands support `--format` and `--output`:

```bash
# Terminal table (default)
friedman estimate var data.csv

# CSV export
friedman estimate var data.csv --format=csv --output=results.csv

# JSON export
friedman estimate var data.csv --format=json --output=results.json
```

### Tidy result tables

Array-valued results (`irf`, `fevd`, `forecast`) render through MacroEconometricModels.jl's
tidy `long_table(result)`: one row per `(horizon, variable[, shock])` cell, columns
`horizon | variable | shock | value | lower | upper` (`fevd` drops `lower`/`upper`;
`forecast` drops `shock`). Coefficient-bearing models (`estimate var`/`reg`/`iv`/`logit`/
`probit`/panel/ordered/multinomial) render through `DataFrame(model)`: one row per term,
columns `term | estimate | std_error | stat | p_value | ci_lower | ci_upper`. A handful of
leaves are deliberate exceptions and keep a domain-specific wide table instead — volatility
`forecast` (`variance`/`volatility` columns), `did estimate` (an ATT summary), `irf`/`fevd
pvar`, `hd`, and `predict`/`residuals` — either because MEMs has no matching tidy result
type for them yet or because collapsing to the generic schema would drop information.

### Model handles & reproducibility

Save a fitted model and reuse it later without re-estimation:

```bash
friedman estimate var macro --lags 2 --save-model var   # stem → var.jld2
friedman irf var --model var --horizons 12 --save-result irf
friedman irf var --result irf                           # skip compute, re-render
friedman show irf
friedman model info var.jld2                            # inspect type / dims / versions
```

`--save-model`, `--save-result`, data positionals, `--model` (typed leaves),
`--result`, and `friedman show` accept suffix-less stems (append `.jld2`).
`model info` still wants an explicit handle path (`.jld2` / `.fmod` /
`model://`). A data handle whose type is not in the leaf's `data_kinds` is
`data/wrong-kind` (exit 3) — e.g. a panel handle on `estimate var`. A
`--result` handle of the wrong result type is `data/wrong-result` (exit 3);
a `--model` handle of the wrong model type is `model/wrong-kind` (exit 5).

Two on-disk formats, chosen by suffix + model type:

- **`.jld2`** — native MacroEconometricModels.jl `save_model`/`load_model` (JLD2-backed,
  versioned, portable across a package upgrade). Covers the upstream serialization
  registry — **350 types** at MacroEconometricModels 0.9.3: everything above plus SVAR
  identification results, BVAR posteriors, innovation accounting (IRF/FEVD/HD), forecasts,
  nowcasts, micro/DiD results, filters, test statistics, policy/counterfactual objects,
  IO leftovers, and the DSGE/HA solutions (RA solutions, Bayesian DSGE, HA steady
  states, SSJ blocks, DCEGM/firm/intermediary, OLG, continuous-time — their equations
  recompile at load under an AST allowlist). A file whose format version is
  incompatible refuses to load (exit 3) rather than mis-read.
- **`.fmod`** — the interim `Serialization` handle, kept for genuinely unregistered
  payloads and old handles. Verify a saved result against a fresh re-run with
  `friedman model reproduce HANDLE` (needs `--seed` at estimation time).

Every `--format json` envelope carries a **reproducibility manifest** under `meta.manifest`
(RNG seed, thread count, OS, Julia + package versions, dependency versions, git, timestamp) —
provenance enough to reproduce and audit a published result. `--seed N` (a leading global) is
echoed at `meta.seed` and `meta.manifest.seed`, and is forwarded to the BVAR family and VAR/VECM
IRF estimators so their draws — and the resulting posteriors/IRFs — reproduce bit-for-bit.

## Plots

Plot-capable leaves accept `--plot` (open in browser) and `--plot-save PATH` (write a
self-contained HTML file). Both are inline flags — there is no separate `plot` command.

**A leaf advertises them only when MacroEconometricModels defines a real plot recipe for that
leaf's result type.** This is deliberate: the flags are not universal, and a leaf that offered
them without an upstream recipe would fail at the point of plotting rather than refuse up front.
As of CLI v0.12.3 (MacroEconometricModels 0.9.6) 145 of 456 leaves are plot-capable (`friedman show` plots when a recipe exists).

Notable gaps, all upstream rather than CLI choices: `forecast setar` and `forecast star` have no
recipe for their forecast result types (the models themselves plot fine via `estimate setar` /
`estimate star`), and the same holds for Markov-switching forecasts — `estimate ms` / `estimate
ms-ar` plot, `forecast ms` / `forecast ms-ar` do not.

If a recipe is missing for whatever you asked to plot, the CLI now fails with a typed
`model/unsupported` (exit 5) naming the type, rather than an internal error. Use `--format json`
and plot the returned tables yourself in that case.

## TOML Configuration

Complex model specs use TOML config files.

**Minnesota prior:**

```toml
[prior]
type = "minnesota"

[prior.hyperparameters]
lambda1 = 0.2
lambda2 = 0.5
lambda3 = 1.0
lambda4 = 100000.0

[prior.optimization]
enabled = true
```

**Sign restrictions:**

```toml
[identification]
method = "sign"

[identification.sign_matrix]
matrix = [
  [1, -1, 1],
  [0, 1, -1],
  [0, 0, 1]
]
horizons = [0, 1, 2, 3]
```

**Narrative restrictions:**

```toml
[identification.narrative]
shock_index = 1
periods = [10, 15, 20]
signs = [1, -1, 1]
```

**Arias identification (zero + sign):**

```toml
[[identification.zero_restrictions]]
var = 1
shock = 1
horizon = 0

[[identification.sign_restrictions]]
var = 2
shock = 1
sign = "positive"
horizon = 0
```

**Uhlig identification (penalty-based, same restriction format as Arias):**

```toml
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
n_starts = 100
n_refine = 20
```

**Non-Gaussian SVAR:**

```toml
[nongaussian]
method = "smooth_transition"
transition_variable = "spread"
n_regimes = 2
```

**GMM specification:**

```toml
[gmm]
moment_conditions = ["output", "inflation"]
instruments = ["lag_output", "lag_inflation"]
weighting = "twostep"
```

**DSGE model (TOML format):**

```toml
[model]
parameters = { rho = 0.9, sigma = 0.01, beta = 0.99, alpha = 0.36, delta = 0.025 }
endogenous = ["C", "K", "Y", "A"]
exogenous = ["e_A"]

[[model.equations]]
expr = "C[t] + K[t] = (1-delta)*K[t-1] + Y[t]"
[[model.equations]]
expr = "Y[t] = A[t] * K[t-1]^alpha"
[[model.equations]]
expr = "1/C[t] = beta * (1/C[t+1] * (alpha*A[t+1]*K[t]^(alpha-1) + 1-delta))"
[[model.equations]]
expr = "A[t] = rho * A[t-1] + sigma * e_A[t]"

[solver]
method = "gensys"
```

**OccBin constraints (for ZLB, etc.):**

```toml
[constraints]
[[constraints.bounds]]
variable = "i"
lower = 0.0
```

**Nonlinear constraints (MEMs v0.4.1):**

```toml
[[constraints.nonlinear]]
expr = "K[t] + C[t] <= Y[t]"
label = "resource constraint"
```

**SMM specification** (`--config` required — SMM matches simulated to sample moments, so it
needs a data-generating `model` and `theta0`; built-ins: `ar1`, `arp`, `var1`, `iid_normal`):

```toml
[smm]
model     = "ar1"          # ar1 | arp | var1 | iid_normal
theta0    = [0.4, 0.5]     # layout depends on model (ar1: [phi, sigma])
lags      = 2              # autocovariance-moment lags
weighting = "two_step"     # identity | two_step
sim_ratio = 5
burn      = 100
lower     = [-0.99, 1.0e-4]  # optional bounds (both together, length = theta0)
upper     = [0.99, 10.0]
```

## License

GPL-3.0-or-later
