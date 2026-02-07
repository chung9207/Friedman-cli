# Friedman

[![CI](https://github.com/chung9207/Friedman-cli/actions/workflows/CI.yml/badge.svg)](https://github.com/chung9207/Friedman-cli/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/chung9207/Friedman-cli/branch/master/graph/badge.svg)](https://codecov.io/gh/chung9207/Friedman-cli)

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/chung9207/MacroEconometricModels.jl).

## Installation

Requires Julia 1.10+.

```bash
git clone https://github.com/chung9207/Friedman-cli.git
cd Friedman-cli
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Usage

```bash
julia --project bin/friedman [command] [subcommand] [args...] [options...]
```

### Commands

| Command | Subcommands | Description |
|---------|-------------|-------------|
| `var`    | `estimate` `lagselect` `stability` `irf` `fevd` `hd` `forecast` | Vector Autoregression (frequentist) |
| `bvar`   | `estimate` `posterior` `irf` `fevd` `hd` `forecast` | Bayesian VAR |
| `lp`     | `estimate` `irf` `fevd` `hd` `forecast` | Local Projections |
| `factor` | `estimate` (`static` `dynamic` `gdfm`) `forecast` | Factor Models |
| `test`   | `adf` `kpss` `pp` `za` `np` `johansen` | Unit Root & Cointegration Tests |
| `gmm`    | `estimate` | Generalized Method of Moments |
| `arima`  | `estimate` `forecast` | ARIMA Models |
| `nongaussian` | `fastica` `ml` `heteroskedasticity` `normality` `identifiability` | Non-Gaussian SVAR |

All commands support `--format` (`table`|`csv`|`json`) and `--output` (file path) options.

### VAR

```bash
# Estimate VAR(2)
friedman var estimate data.csv --lags=2

# Automatic lag selection (AIC/BIC/HQC)
friedman var lagselect data.csv --max-lags=12 --criterion=aic

# Stationarity check (companion matrix eigenvalues)
friedman var stability data.csv --lags=2

# Frequentist IRF (Cholesky identification)
friedman var irf data.csv --shock=1 --horizons=20 --id=cholesky

# Sign restrictions (requires config)
friedman var irf data.csv --id=sign --config=sign_restrictions.toml

# Narrative sign restrictions
friedman var irf data.csv --id=narrative --config=narrative.toml

# Long-run (Blanchard-Quah) identification
friedman var irf data.csv --id=longrun --horizons=40

# Arias et al. (2018) zero/sign restrictions
friedman var irf data.csv --id=arias --config=arias_restrictions.toml

# With bootstrap confidence intervals
friedman var irf data.csv --shock=1 --ci=bootstrap --replications=1000

# Theoretical CIs or no CIs
friedman var irf data.csv --ci=theoretical
friedman var irf data.csv --ci=none

# FEVD with Cholesky identification
friedman var fevd data.csv --horizons=20 --id=cholesky

# FEVD with sign restrictions
friedman var fevd data.csv --id=sign --config=sign_restrictions.toml

# Historical decomposition
friedman var hd data.csv --id=cholesky
friedman var hd data.csv --id=longrun --lags=4

# h-step ahead forecast with confidence intervals
friedman var forecast data.csv --horizons=12 --confidence=0.95
friedman var forecast data.csv --lags=4 --horizons=24
```

### Bayesian VAR

```bash
# Estimate with NUTS sampler
friedman bvar estimate data.csv --lags=4 --draws=2000 --sampler=nuts

# With Minnesota prior config
friedman bvar estimate data.csv --config=prior.toml

# Posterior summary (mean or median)
friedman bvar posterior data.csv --lags=4 --method=mean
friedman bvar posterior data.csv --method=median --sampler=hmc

# Bayesian IRFs (posterior credible intervals)
friedman bvar irf data.csv --shock=1 --horizons=20
friedman bvar irf data.csv --draws=5000 --sampler=hmc --config=prior.toml

# Bayesian FEVD
friedman bvar fevd data.csv --horizons=20

# Bayesian historical decomposition
friedman bvar hd data.csv --draws=2000

# Bayesian forecast (posterior credible intervals)
friedman bvar forecast data.csv --horizons=12 --draws=2000
friedman bvar forecast data.csv --horizons=24 --sampler=hmc --config=prior.toml
```

### Local Projections

```bash
# Basic LP (Jorda 2005)
friedman lp estimate data.csv --shock=1 --horizons=20 --vcov=newey_west

# LP-IV (Stock & Watson 2018)
friedman lp estimate data.csv --method=iv --shock=1 --instruments=instruments.csv

# Smooth LP (Barnichon & Brownlees 2019) — auto-selects lambda via CV
friedman lp estimate data.csv --method=smooth --shock=1 --horizons=20 --knots=3
friedman lp estimate data.csv --method=smooth --shock=1 --lambda=0.5

# State-dependent LP (Auerbach & Gorodnichenko 2013)
friedman lp estimate data.csv --method=state --shock=1 --state-var=2 --gamma=1.5
friedman lp estimate data.csv --method=state --shock=1 --state-var=3 --transition=exponential

# Propensity score LP (Angrist et al. 2018)
friedman lp estimate data.csv --method=propensity --treatment=1 --score-method=logit

# Doubly robust LP (propensity score + outcome regression)
friedman lp estimate data.csv --method=robust --treatment=1 --score-method=logit

# Structural LP IRFs (with identification)
friedman lp irf data.csv --id=cholesky --shock=1 --horizons=20
friedman lp irf data.csv --id=sign --config=sign_restrictions.toml
friedman lp irf data.csv --shocks=1,2,3 --id=cholesky --horizons=30

# LP FEVD (via structural LP's VAR model)
friedman lp fevd data.csv --horizons=20 --id=cholesky

# LP Historical Decomposition
friedman lp hd data.csv --id=cholesky

# Direct LP forecast
friedman lp forecast data.csv --shock=1 --horizons=12 --shock-size=1.0
friedman lp forecast data.csv --shock=1 --horizons=24 --ci-method=bootstrap --n-boot=500
```

### Factor Models

```bash
# Static (PCA) — auto-selects factors via Bai-Ng IC
friedman factor estimate static data.csv
friedman factor estimate static data.csv --nfactors=3 --criterion=ic2

# Dynamic factor model
friedman factor estimate dynamic data.csv --nfactors=2 --factor-lags=1 --method=twostep

# Generalized DFM (spectral)
friedman factor estimate gdfm data.csv --dynamic-rank=2

# Factor model forecasting (static, dynamic, or GDFM)
friedman factor forecast data.csv --horizon=12
friedman factor forecast data.csv --model=static --nfactors=3 --horizon=24 --ci-method=bootstrap
friedman factor forecast data.csv --model=dynamic --nfactors=2 --factor-lags=1 --horizon=12
friedman factor forecast data.csv --model=gdfm --dynamic-rank=2 --horizon=12
```

### Unit Root & Cointegration Tests

```bash
# Augmented Dickey-Fuller (auto lag via AIC)
friedman test adf data.csv --column=1 --trend=constant

# KPSS stationarity test (reversed null: H0 = stationary)
friedman test kpss data.csv --column=1 --trend=constant

# Phillips-Perron
friedman test pp data.csv --column=1

# Zivot-Andrews (structural break)
friedman test za data.csv --column=1 --trend=both --trim=0.15

# Ng-Perron (MZa, MZt, MSB, MPT)
friedman test np data.csv --column=1

# Johansen cointegration (trace + max eigenvalue)
friedman test johansen data.csv --lags=2 --trend=constant
```

### GMM

```bash
friedman gmm estimate data.csv --config=gmm_spec.toml --weighting=twostep
friedman gmm estimate data.csv --config=gmm_spec.toml --weighting=iterated
```

### ARIMA

```bash
# Estimate ARIMA(p,d,q) — dispatches to AR/MA/ARMA/ARIMA as needed
friedman arima estimate data.csv --p=1 --d=1 --q=1
friedman arima estimate data.csv --p=2                   # AR(2)
friedman arima estimate data.csv --p=0 --q=3             # MA(3)
friedman arima estimate data.csv --p=2 --q=1             # ARMA(2,1)
friedman arima estimate data.csv --p=1 --d=1 --q=1 --method=mle --column=2

# Automatic order selection (omit --p to auto-select)
friedman arima estimate data.csv --criterion=bic
friedman arima estimate data.csv --max-p=5 --max-d=2 --max-q=5 --criterion=aic

# Forecast (auto model selection + h-step forecast with confidence intervals)
friedman arima forecast data.csv --horizons=12 --confidence=0.95

# Forecast with explicit model
friedman arima forecast data.csv --p=2 --d=1 --q=1 --horizons=24
```

### Non-Gaussian SVAR

```bash
# ICA-based identification (FastICA, Infomax, JADE, SOBI, dCov, HSIC)
friedman nongaussian fastica data.csv --method=fastica --contrast=logcosh
friedman nongaussian fastica data.csv --method=infomax --lags=4
friedman nongaussian fastica data.csv --method=jade
friedman nongaussian fastica data.csv --method=sobi
friedman nongaussian fastica data.csv --method=dcov
friedman nongaussian fastica data.csv --method=hsic

# Maximum likelihood (Student-t, skew-t, GHD, mixture-normal, PML, skew-normal)
friedman nongaussian ml data.csv --distribution=student_t
friedman nongaussian ml data.csv --distribution=skew_t --lags=2
friedman nongaussian ml data.csv --distribution=mixture_normal
friedman nongaussian ml data.csv --distribution=pml
friedman nongaussian ml data.csv --distribution=skew_normal

# Heteroskedasticity-based identification
friedman nongaussian heteroskedasticity data.csv --method=markov --regimes=2
friedman nongaussian heteroskedasticity data.csv --method=garch
friedman nongaussian heteroskedasticity data.csv --method=smooth_transition --config=ng.toml
friedman nongaussian heteroskedasticity data.csv --method=external --config=ng.toml

# Normality test suite (checks if non-Gaussian methods are applicable)
friedman nongaussian normality data.csv --lags=4

# Identifiability tests
friedman nongaussian identifiability data.csv --test=all
friedman nongaussian identifiability data.csv --test=strength
friedman nongaussian identifiability data.csv --test=gaussianity --method=fastica
friedman nongaussian identifiability data.csv --test=overidentification --method=sobi
```

## Output Formats

All commands support `--format` and `--output`:

```bash
# Terminal table (default)
friedman var estimate data.csv

# CSV export
friedman var estimate data.csv --format=csv --output=results.csv

# JSON export
friedman var estimate data.csv --format=json --output=results.json
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
