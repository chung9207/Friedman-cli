# predict & residuals

In-sample fitted values (`predict`) and model residuals (`residuals`). 16 subcommands each, covering time series, volatility, factor, and cross-sectional regression models.

Both commands share identical subcommand structure and options. Each subcommand estimates the model and extracts fitted values or residuals.

## Supported Models

| Subcommand | Model |
|------------|-------|
| `var` | Frequentist VAR |
| `bvar` | Bayesian VAR |
| `arima` | ARIMA |
| `vecm` | Vector Error Correction Model |
| `static` | Static factor model (PCA) |
| `dynamic` | Dynamic factor model |
| `gdfm` | Generalized dynamic factor model |
| `arch` | ARCH volatility |
| `garch` | GARCH volatility |
| `egarch` | EGARCH volatility |
| `gjr_garch` | GJR-GARCH volatility |
| `sv` | Stochastic volatility |
| `favar` | Factor-Augmented VAR |
| `reg` | OLS/WLS regression |
| `logit` | Logit regression |
| `probit` | Probit regression |

## predict

```bash
friedman predict var data.csv --lags=2
friedman predict arima data.csv --p=1 --d=1 --q=1
friedman predict garch data.csv --column=1 --p=1 --q=1
friedman predict vecm data.csv --lags=4 --rank=2
```

## residuals

```bash
friedman residuals var data.csv --lags=2
friedman residuals arima data.csv --p=1 --d=1 --q=1
friedman residuals garch data.csv --column=1 --p=1 --q=1
friedman residuals vecm data.csv --lags=4 --rank=2
```

## Common Options

Options match those in the corresponding `estimate` command for each model type. All subcommands support:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### VAR / BVAR

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto/4 | Lag order |

### ARIMA

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | auto | AR order |
| `--d` | | Int | 0 | Differencing order |
| `--q` | | Int | 0 | MA order |

### VECM

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--rank` | `-r` | Int | auto | Cointegration rank |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |

### Volatility Models (arch, garch, egarch, gjr\_garch, sv)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | 1 | GARCH order (not for arch) |
| `--q` | | Int | 1 | ARCH order |

### Factor Models (static, dynamic, gdfm)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of factors |

### Regression Models (reg, logit, probit)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--dep` | | String | (1st col) | Dependent variable column name |
| `--cov-type` | | String | `hc1` | Covariance type |
| `--clusters` | | String | | Cluster variable column name |

### Logit/Probit Predict Flags

`predict logit` and `predict probit` support additional flags for alternative output:

| Flag | Description |
|------|-------------|
| `--marginal-effects` | Output average marginal effects instead of fitted probabilities |
| `--odds-ratio` | Output odds ratio table (logit only) |
| `--classification-table` | Output confusion matrix |
| `--threshold` | Classification threshold (default: 0.5, with `--classification-table`) |

These flags are mutually exclusive with default fitted-values output.
