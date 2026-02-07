# Friedman-cli

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/chung9207/MacroEconometricModels.jl).

## Features

| Category | Models / Tests | Commands |
|----------|---------------|----------|
| **VAR** | Frequentist VAR, Bayesian VAR (Minnesota prior, MCMC) | `estimate var`, `estimate bvar` |
| **Local Projections** | Standard, IV, Smooth, State-dependent, Propensity score, Doubly robust | `estimate lp --method=...` |
| **Factor Models** | Static (PCA), Dynamic, Generalized Dynamic (spectral) | `estimate static`, `dynamic`, `gdfm` |
| **ARIMA** | AR, MA, ARMA, ARIMA with auto order selection | `estimate arima` |
| **Volatility** | ARCH, GARCH, EGARCH, GJR-GARCH, Stochastic Volatility | `estimate arch`, `garch`, ... |
| **Non-Gaussian SVAR** | FastICA, JADE, SOBI, dCov, HSIC, ML (Student-t, mixture, PML, skew-normal) | `estimate fastica`, `estimate ml` |
| **GMM** | Identity, optimal, two-step, iterated weighting | `estimate gmm` |
| **IRF** | Cholesky, sign, narrative, long-run, Arias, non-Gaussian methods | `irf var`, `irf bvar`, `irf lp` |
| **FEVD** | Frequentist, Bayesian, LP (bias-corrected) | `fevd var`, `fevd bvar`, `fevd lp` |
| **Historical Decomposition** | Frequentist, Bayesian, LP-based | `hd var`, `hd bvar`, `hd lp` |
| **Forecasting** | VAR, BVAR, LP, ARIMA, factor models, volatility models | `forecast var`, `forecast arima`, ... |
| **Unit Root Tests** | ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron | `test adf`, `test kpss`, ... |
| **Cointegration** | Johansen trace and max eigenvalue | `test johansen` |
| **Diagnostics** | Normality, identifiability, ARCH-LM, Ljung-Box, heteroskedasticity | `test normality`, ... |

**9 top-level commands, ~54 subcommands.** Action-first CLI: commands organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

## Quick Start

```bash
# Install
git clone https://github.com/chung9207/Friedman-cli.git
cd Friedman-cli
julia --project -e '
using Pkg
Pkg.rm("MacroEconometricModels")
Pkg.add(url="https://github.com/chung9207/MacroEconometricModels.jl.git")
'

# Estimate a VAR(2) model
julia --project bin/friedman estimate var data.csv --lags=2

# Compute impulse responses
julia --project bin/friedman irf var data.csv --shock=1 --horizons=20

# Forecast 12 steps ahead
julia --project bin/friedman forecast var data.csv --horizons=12

# Run unit root test
julia --project bin/friedman test adf data.csv --column=1
```

All commands support `--format=table|csv|json` and `--output=file.csv` for flexible output.

Models are automatically saved with tags (`var001`, `bvar001`, ...) and can be reused in post-estimation:

```bash
julia --project bin/friedman estimate var data.csv --lags=2
#   Saved as: var001

julia --project bin/friedman irf var001    # uses stored model
```

## Contents

```@contents
Pages = [
    "installation.md",
    "commands/overview.md",
    "commands/estimate.md",
    "commands/test.md",
    "commands/irf.md",
    "commands/fevd.md",
    "commands/hd.md",
    "commands/forecast.md",
    "commands/storage.md",
    "configuration.md",
    "api.md",
    "architecture.md",
]
Depth = 2
```
