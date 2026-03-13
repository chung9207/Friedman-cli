# Friedman-cli

[![CI](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/CI.yml/badge.svg)](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/FriedmanJP/Friedman-cli/graph/badge.svg?token=TIYTWTJG36)](https://codecov.io/gh/FriedmanJP/Friedman-cli)
[![Documentation](https://github.com/FriedmanJP/Friedman-cli/actions/workflows/Documentation.yml/badge.svg)](https://friedmanjp.github.io/Friedman-cli/dev/)

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/FriedmanJP/MacroEconometricModels.jl) (v0.4.0).

14 top-level commands, ~199 subcommands. Action-first CLI: commands are organized by action (`estimate`, `irf`, `forecast`, `dsge`, `did`, `spectral`, ...) rather than by model type. Features include VAR/BVAR/Panel VAR, FAVAR, structural DFM, cross-sectional regression (OLS/WLS/IV/Logit/Probit/ordered logit/ordered probit/multinomial logit), panel regression (POLS/FE/RE/FD/IV), local projections, DSGE (including full Bayesian workflow, historical decomposition, and 3rd-order perturbation), DID/event study/LP-DiD, factor models, ARIMA, volatility models (ARCH/GARCH/EGARCH/GJR-GARCH/SV), non-Gaussian SVAR, GMM/SMM, time series filtering, nowcasting, spectral analysis (ACF, periodogram, spectral density, cross-spectrum, transfer function), advanced unit root tests (Fourier ADF/KPSS, DF-GLS, LM with breaks, ADF 2-break, Gregory-Hansen), structural break tests (Andrews, Bai-Perron), panel unit root tests (PANIC, CIPS, Moon-Perron, factor break), VIF multicollinearity diagnostics, and data management.

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

### Install from Source

```bash
git clone https://github.com/FriedmanJP/Friedman-cli.git
cd Friedman-cli
julia --project -e '
using Pkg
Pkg.rm("MacroEconometricModels")
Pkg.add(url="https://github.com/FriedmanJP/MacroEconometricModels.jl.git")
'
```

See [Installation docs](https://friedmanjp.github.io/Friedman-cli/dev/installation/) for specific version install, manual install from GitHub Releases, upgrading, and uninstalling.

## Usage

```bash
friedman [command] [subcommand] [args...] [options...]
```

### Commands

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `estimate` | `var` `bvar` `lp` `arima` `gmm` `smm` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` `fastica` `ml` `vecm` `pvar` `favar` `sdfm` `reg` `iv` `logit` `probit` `ologit` `oprobit` `mlogit` `preg` `pols` `pfe` `pre` | Estimate models (31 model types, incl. panel reg, ordered/multinomial choice) |
| `test` | `adf` `kpss` `pp` `za` `np` `johansen` `normality` `identifiability` `heteroskedasticity` `arch_lm` `ljung_box` `granger` `lr` `lm` `andrews` `bai-perron` `panic` `cips` `moon-perron` `factor-break` `fourier-adf` `fourier-kpss` `dfgls` `lm-unitroot` `adf-2break` `gregory-hansen` `vif` + panel spec tests + spectral diagnostics + discrete choice tests + `var` (`lagselect` `stability`) + `pvar` (`hansen_j` `mmsc` `lagselect` `stability`) | Statistical tests (41+ leaves + nested) |
| `irf` | `var` `bvar` `lp` `vecm` `pvar` `favar` `sdfm` | Impulse response functions |
| `fevd` | `var` `bvar` `lp` `vecm` `pvar` `favar` `sdfm` | Forecast error variance decomposition |
| `hd` | `var` `bvar` `lp` `vecm` `favar` | Historical decomposition |
| `forecast` | `var` `bvar` `lp` `arima` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` `vecm` `favar` | Forecasting (14 model types) |
| `predict` | `var` `bvar` `arima` `vecm` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` `favar` `reg` `logit` `probit` `ologit` `oprobit` `mlogit` `preg` `pols` `pfe` `pre` | In-sample fitted values (23 model types) |
| `residuals` | `var` `bvar` `arima` `vecm` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` `favar` `reg` `logit` `probit` `ologit` `oprobit` `mlogit` `preg` `pols` `pfe` `pre` | Model residuals (23 model types) |
| `filter` | `hp` `hamilton` `bn` `bk` `bhp` | Time series filters |
| `data` | `list` `load` `describe` `diagnose` `fix` `transform` `filter` `validate` `balance` `dropna` `keeprows` | Data management (11 leaves) |
| `nowcast` | `dfm` `bvar` `bridge` `news` `forecast` | Nowcasting (DFM, BVAR, bridge equations) |
| `dsge` | `solve` `irf` `fevd` `hd` `simulate` `estimate` `perfect-foresight` `steady-state` + `bayes` (`estimate` `irf` `fevd` `simulate` `summary` `compare` `predictive`) | DSGE models (8 + 7 nested bayes, incl. historical decomposition) |
| `spectral` | `acf` `periodogram` `density` `cross` `transfer` | Spectral analysis (ACF, periodogram, spectral density, cross-spectrum, transfer function) |
| `did` | `estimate` `event-study` `lp-did` + `test` (`bacon` `pretrend` `negweight` `honest`) | Difference-in-differences (3 + 4 nested) |

All commands support `--format` (`table`|`csv`|`json`) and `--output` (file path) options.

**Global flags:** `--help`, `--version`, `--warranty` (GPL warranty disclaimer), `--conditions` (GPL distribution conditions).

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

# Simulated Method of Moments (SMM)
friedman estimate smm data.csv --config=smm_spec.toml --weighting=two_step --sim-ratio=5

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

### Volatility Models

```bash
# ARCH(q)
friedman estimate arch data.csv --column=1 --q=1

# GARCH(p,q)
friedman estimate garch data.csv --column=1 --p=1 --q=1

# EGARCH(p,q)
friedman estimate egarch data.csv --column=1 --p=1 --q=1

# GJR-GARCH(p,q)
friedman estimate gjr_garch data.csv --column=1 --p=1 --q=1

# Stochastic volatility
friedman estimate sv data.csv --column=1 --draws=2000
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
friedman test arch_lm data.csv --lags=4
friedman test ljung_box data.csv --lags=10

# Granger causality
friedman test granger data.csv --cause=1 --effect=2 --lags=4
friedman test granger data.csv --all --lags=4

# Model comparison (LR and LM tests)
friedman test lr data.csv data.csv --lags1=2 --lags2=4
friedman test lm data.csv data.csv --lags1=2 --lags2=4

# Panel VAR diagnostics
friedman test pvar hansen_j data.csv --id-col=country --time-col=year --lags=2
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
friedman forecast gjr_garch data.csv --column=1 --horizons=12
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
```

### Data Management

```bash
# List available example datasets
friedman data list

# Load example dataset (FRED-MD, FRED-QD, PWT, mpdta, ddcg)
friedman data load fred_md --output=fred_md.csv
friedman data load fred_md --vars=INDPRO,CPIAUCSL --transform

# Describe data (summary statistics)
friedman data describe data.csv

# Diagnose data quality (NaN, Inf, constant columns)
friedman data diagnose data.csv

# Fix data issues
friedman data fix data.csv --method=interpolate --output=cleaned.csv

# Apply transformation codes
friedman data transform data.csv --tcodes=1,5,5,2 --output=transformed.csv

# Filter data (unified interface)
friedman data filter data.csv --method=hp --lambda=1600

# Validate data for a specific model type
friedman data validate data.csv --model=var

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

### DSGE Models

```bash
# Solve a DSGE model (from TOML specification)
friedman dsge solve model.toml --method=gensys
friedman dsge solve model.toml --method=perturbation --order=2
friedman dsge solve model.toml --method=perturbation --order=3
friedman dsge solve model.toml --method=projection --degree=5 --grid=chebyshev

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
expr = "1/C[t] = beta * E[t](1/C[t+1] * (alpha*A[t+1]*K[t]^(alpha-1) + 1-delta))"
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

**SMM specification:**

```toml
[smm]
weighting = "two_step"
sim_ratio = 5
burn = 100
```

## What's New

### v0.4.0
- **Spectral analysis** — new `spectral` top-level command with `acf`, `periodogram`, `density`, `cross`, and `transfer` subcommands
- **Panel regression** — `estimate preg/pols/pfe/pre` for pooled OLS, fixed effects, random effects, and first-difference panel estimators
- **Ordered and multinomial choice models** — `estimate ologit`, `estimate oprobit`, `estimate mlogit` with marginal effects and prediction support
- **DSGE historical decomposition** — `dsge hd` for shock-contribution decomposition of DSGE model simulations
- **Extended predict/residuals** — 23 model types each (up from 16), covering all new estimators
- **Data management additions** — `data dropna` and `data keeprows` for row-level filtering

## License

GPL-3.0-or-later
