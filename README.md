# Friedman-cli

[![CI](https://github.com/chung9207/Friedman-cli/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/Friedman-cli/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/Friedman-cli/branch/master/graph/badge.svg)](https://codecov.io/gh/chung9207/Friedman-cli)
[![Documentation](https://github.com/chung9207/Friedman-cli/actions/workflows/Documentation.yml/badge.svg)](https://chung9207.github.io/Friedman-cli/dev/)

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/chung9207/MacroEconometricModels.jl) (v0.2.1).

13 top-level commands, ~85 subcommands. Action-first CLI: commands are organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

## Installation

Requires Julia 1.10+.

```bash
git clone https://github.com/chung9207/Friedman-cli.git
cd Friedman-cli
julia --project -e '
using Pkg
Pkg.rm("MacroEconometricModels")
Pkg.add(url="https://github.com/chung9207/MacroEconometricModels.jl.git")
'
```

## Usage

```bash
julia --project bin/friedman [command] [subcommand] [args...] [options...]
```

### Commands

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `estimate` | `var` `bvar` `lp` `arima` `gmm` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` `fastica` `ml` | Estimate models (15 model types) |
| `test` | `adf` `kpss` `pp` `za` `np` `johansen` `normality` `identifiability` `heteroskedasticity` `arch_lm` `ljung_box` + `var` (`lagselect` `stability`) | Statistical tests (12 + nested) |
| `irf` | `var` `bvar` `lp` | Impulse response functions |
| `fevd` | `var` `bvar` `lp` | Forecast error variance decomposition |
| `hd` | `var` `bvar` `lp` | Historical decomposition |
| `forecast` | `var` `bvar` `lp` `arima` `static` `dynamic` `gdfm` `arch` `garch` `egarch` `gjr_garch` `sv` | Forecasting (12 model types) |
| `list` | `models` `results` | List stored models and results |
| `rename` | — | Rename stored tags |
| `project` | `list` `show` | Manage project registry |

All commands support `--format` (`table`|`csv`|`json`) and `--output` (file path) options.

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
```

### Impulse Response Functions

```bash
# Frequentist IRF (Cholesky identification)
friedman irf var data.csv --shock=1 --horizons=20 --id=cholesky

# Sign restrictions (requires config)
friedman irf var data.csv --id=sign --config=sign_restrictions.toml

# Narrative sign restrictions
friedman irf var data.csv --id=narrative --config=narrative.toml

# Long-run (Blanchard-Quah) identification
friedman irf var data.csv --id=longrun --horizons=40

# Arias et al. (2018) zero/sign restrictions
friedman irf var data.csv --id=arias --config=arias_restrictions.toml

# With bootstrap confidence intervals
friedman irf var data.csv --shock=1 --ci=bootstrap --replications=1000

# Bayesian IRFs (posterior credible intervals)
friedman irf bvar data.csv --shock=1 --horizons=20
friedman irf bvar data.csv --draws=5000 --sampler=hmc --config=prior.toml

# Structural LP IRFs
friedman irf lp data.csv --id=cholesky --shock=1 --horizons=20
friedman irf lp data.csv --shocks=1,2,3 --id=cholesky --horizons=30

# From stored model tag
friedman irf var001
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

# From stored model tag
friedman fevd var001
```

### Historical Decomposition

```bash
friedman hd var data.csv --id=cholesky
friedman hd var data.csv --id=longrun --lags=4
friedman hd bvar data.csv --draws=2000
friedman hd lp data.csv --id=cholesky

# From stored model tag
friedman hd bvar001
```

### Forecasting

```bash
# VAR forecast with confidence intervals
friedman forecast var data.csv --horizons=12 --confidence=0.95

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

# From stored model tag
friedman forecast var001
```

### Storage & Projects

```bash
# Models are auto-saved with tags (var001, bvar001, etc.)
friedman estimate var data.csv --lags=2
#   Saved as: var001

# List stored models and results
friedman list models
friedman list results

# Rename a tag
friedman rename var001 gdp_model

# Use stored tag for post-estimation
friedman irf var001
friedman forecast gdp_model

# Project management (auto-registered on first save)
friedman project list
friedman project show
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

## License

MIT
