# Friedman-cli

Macroeconometric analysis from the terminal. A Julia CLI wrapping [MacroEconometricModels.jl](https://github.com/chung9207/MacroEconometricModels.jl).

## Features

| Category | Models / Tests | Commands |
|----------|---------------|----------|
| **VAR** | Frequentist VAR, Bayesian VAR (Minnesota prior, MCMC) | `estimate var`, `estimate bvar` |
| **VECM** | Vector Error Correction Model (Johansen) | `estimate vecm` |
| **Panel VAR** | GMM/FE-OLS estimation, OIRF/GIRF | `estimate pvar` |
| **Local Projections** | Standard, IV, Smooth, State-dependent, Propensity score, Doubly robust | `estimate lp --method=...` |
| **Factor Models** | Static (PCA), Dynamic, Generalized Dynamic (spectral) | `estimate static`, `dynamic`, `gdfm` |
| **ARIMA** | AR, MA, ARMA, ARIMA with auto order selection | `estimate arima` |
| **Volatility** | ARCH, GARCH, EGARCH, GJR-GARCH, Stochastic Volatility | `estimate arch`, `garch`, ... |
| **Non-Gaussian SVAR** | FastICA, JADE, SOBI, dCov, HSIC, ML (Student-t, mixture, PML, skew-normal) | `estimate fastica`, `estimate ml` |
| **GMM** | Identity, optimal, two-step, iterated weighting | `estimate gmm` |
| **IRF** | Cholesky, sign, narrative, long-run, Arias, Uhlig, non-Gaussian methods | `irf var`, `irf bvar`, `irf lp`, `irf vecm`, `irf pvar` |
| **FEVD** | Frequentist, Bayesian, LP (bias-corrected), VECM, Panel VAR | `fevd var`, `fevd bvar`, `fevd lp`, `fevd vecm`, `fevd pvar` |
| **Historical Decomposition** | Frequentist, Bayesian, LP-based, VECM | `hd var`, `hd bvar`, `hd lp`, `hd vecm` |
| **Forecasting** | VAR, BVAR, LP, ARIMA, factor models, volatility models, VECM | `forecast var`, `forecast arima`, ... |
| **Predict / Residuals** | In-sample fitted values and model residuals | `predict var`, `residuals var`, ... |
| **Filters** | HP, Hamilton, Beveridge-Nelson, Baxter-King, Boosted HP | `filter hp`, `filter hamilton`, ... |
| **Nowcasting** | DFM, BVAR, bridge equations, news decomposition | `nowcast dfm`, `nowcast bvar`, ... |
| **Data Management** | Example datasets, diagnostics, transformations, validation, balancing | `data list`, `data load`, `data describe`, ... |
| **Unit Root Tests** | ADF, KPSS, Phillips-Perron, Zivot-Andrews, Ng-Perron | `test adf`, `test kpss`, ... |
| **Cointegration** | Johansen trace and max eigenvalue | `test johansen` |
| **Diagnostics** | Normality, identifiability, ARCH-LM, Ljung-Box, heteroskedasticity | `test normality`, ... |
| **Model Comparison** | Granger causality, LR test, LM test | `test granger`, `test lr`, `test lm` |

**11 top-level commands, ~103 subcommands.** Action-first CLI: commands organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

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

# Nowcast GDP
julia --project bin/friedman nowcast dfm mixed_freq.csv --factors=3
```

All commands support `--format=table|csv|json` and `--output=file.csv` for flexible output.

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
    "commands/predict_residuals.md",
    "commands/filter.md",
    "commands/data.md",
    "commands/nowcast.md",
    "configuration.md",
    "api.md",
    "architecture.md",
]
Depth = 2
```
