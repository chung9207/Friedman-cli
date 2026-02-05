# Friedman

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

| Command | Description |
|---------|-------------|
| `var`    | Vector Autoregression |
| `bvar`   | Bayesian VAR |
| `irf`    | Impulse Response Functions |
| `fevd`   | Forecast Error Variance Decomposition |
| `hd`     | Historical Decomposition |
| `lp`     | Local Projections |
| `factor` | Factor Models |
| `test`   | Unit Root & Cointegration Tests |
| `gmm`    | Generalized Method of Moments |

### VAR

```bash
# Estimate VAR(2)
friedman var estimate data.csv --lags=2

# Automatic lag selection
friedman var lagselect data.csv --max-lags=12 --criterion=aic

# Stationarity check
friedman var stability data.csv --lags=2
```

### Bayesian VAR

```bash
# Estimate with NUTS sampler
friedman bvar estimate data.csv --lags=4 --draws=2000 --sampler=nuts

# With Minnesota prior config
friedman bvar estimate data.csv --config=prior.toml

# Posterior summary
friedman bvar posterior data.csv --method=mean
```

### Impulse Response Functions

```bash
# Cholesky identification
friedman irf compute data.csv --shock=1 --horizons=20 --id=cholesky

# Sign restrictions (requires config)
friedman irf compute data.csv --id=sign --config=sign_restrictions.toml

# With bootstrap confidence intervals
friedman irf compute data.csv --shock=1 --ci=bootstrap --replications=1000
```

### FEVD & Historical Decomposition

```bash
friedman fevd compute data.csv --horizons=20 --id=cholesky
friedman hd compute data.csv --id=cholesky
```

### Local Projections

```bash
# Basic LP (Jorda 2005)
friedman lp estimate data.csv --shock=1 --horizons=20 --vcov=newey_west

# LP-IV (Stock & Watson 2018)
friedman lp iv data.csv --shock=1 --instruments=instruments.csv

# Smooth LP (Barnichon & Brownlees 2019)
friedman lp smooth data.csv --shock=1 --knots=3

# State-dependent LP (Auerbach & Gorodnichenko 2013)
friedman lp state data.csv --shock=1 --state-var=2 --gamma=1.5

# Propensity score LP (Angrist et al. 2018)
friedman lp propensity data.csv --treatment=1
```

### Factor Models

```bash
# Static (PCA)
friedman factor static data.csv --nfactors=3

# Dynamic factor model
friedman factor dynamic data.csv --nfactors=2 --factor-lags=1 --method=twostep

# Generalized DFM
friedman factor gdfm data.csv --dynamic-rank=2
```

### Unit Root & Cointegration Tests

```bash
friedman test adf data.csv --column=1 --trend=constant
friedman test kpss data.csv --column=1
friedman test pp data.csv --column=1
friedman test za data.csv --column=1        # Zivot-Andrews (structural break)
friedman test np data.csv --column=1        # Ng-Perron
friedman test johansen data.csv --lags=2    # Cointegration
```

### GMM

```bash
friedman gmm estimate data.csv --config=gmm_spec.toml --weighting=twostep
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

**GMM specification:**

```toml
[gmm]
moment_conditions = ["output", "inflation"]
weighting = "twostep"
```

## License

MIT
