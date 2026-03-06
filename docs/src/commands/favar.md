# favar & sdfm

Factor-Augmented VAR (FAVAR) and Structural Dynamic Factor Model (SDFM) commands. FAVAR spans the full analysis pipeline with 7 commands across `estimate`, `irf`, `fevd`, `hd`, `forecast`, `predict`, and `residuals`. SDFM provides 3 commands: `estimate`, `irf`, `fevd`.

## FAVAR

FAVAR (Bernanke, Boivin & Eliasz 2005) augments a standard VAR with latent factors extracted from a large panel of macroeconomic variables. A small number of "key" variables (e.g., the federal funds rate) enter the VAR directly alongside the extracted factors.

### estimate favar

Estimate a FAVAR model. Supports two-step (PCA + VAR) and Bayesian (one-step MCMC) estimation.

```bash
# Two-step estimation with 3 factors
friedman estimate favar macro.csv --key-vars=ffr,cpi --factors=3 --lags=4

# Auto-select factor count via information criteria
friedman estimate favar macro.csv --key-vars=ffr,cpi

# Bayesian estimation
friedman estimate favar macro.csv --key-vars=ffr,cpi --method=bayesian --draws=10000

# Key vars by column index
friedman estimate favar macro.csv --key-vars=1,3,5 --factors=4
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto (IC) | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices (comma-separated) |
| `--method` | | String | `two_step` | `two_step`, `bayesian` |
| `--draws` | `-n` | Int | 5000 | MCMC draws (bayesian only) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Coefficient matrix, factor count, key variable count, AIC/BIC.

### irf favar

FAVAR impulse response functions. Supports all standard identification methods.

```bash
friedman irf favar macro.csv --key-vars=ffr,cpi --horizons=20

# Panel-wide IRFs (trace responses for all N original variables)
friedman irf favar macro.csv --key-vars=ffr,cpi --horizons=20 --panel-irf

# With sign restrictions
friedman irf favar macro.csv --key-vars=ffr,cpi --id=sign --config=restrictions.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--id` | | String | `cholesky` | Identification method |
| `--config` | | String | | TOML config for identification restrictions |
| `--panel-irf` | | Flag | | Output panel-wide IRFs (N variables) instead of factor-level |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-shock IRF tables. With `--panel-irf`, responses are mapped back to all original panel variables.

### fevd favar

FAVAR forecast error variance decomposition.

```bash
friedman fevd favar macro.csv --key-vars=ffr,cpi --horizons=20
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--horizons` | `-h` | Int | 20 | FEVD horizon |
| `--id` | | String | `cholesky` | Identification method |
| `--config` | | String | | TOML config for identification restrictions |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable FEVD proportions table (columns = shocks, rows = horizons).

### hd favar

FAVAR historical decomposition.

```bash
friedman hd favar macro.csv --key-vars=ffr,cpi --id=cholesky
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--horizons` | `-h` | Int | 20 | HD horizon |
| `--id` | | String | `cholesky` | Identification method |
| `--config` | | String | | TOML config for identification restrictions |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable shock contribution tables + initial conditions.

### forecast favar

FAVAR forecasting with optional panel-wide output.

```bash
friedman forecast favar macro.csv --key-vars=ffr,cpi --horizons=12

# Panel-wide forecast (all N original variables)
friedman forecast favar macro.csv --key-vars=ffr,cpi --horizons=12 --panel-forecast
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--panel-forecast` | | Flag | | Forecast all N panel variables (via factor loadings) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable forecast table. With `--panel-forecast`, includes forecasts for all original panel variables.

### predict favar

FAVAR in-sample fitted values.

```bash
friedman predict favar macro.csv --key-vars=ffr,cpi --factors=3
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** In-sample fitted values for each variable.

### residuals favar

FAVAR model residuals.

```bash
friedman residuals favar macro.csv --key-vars=ffr,cpi --factors=3
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-r` | Int | auto | Number of factors |
| `--lags` | `-p` | Int | 2 | VAR lag order |
| `--key-vars` | | String | (required) | Key variable names or column indices |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Model residuals for each variable.

## Structural DFM

Structural Dynamic Factor Model (Forni et al. 2009) identifies structural shocks in a dynamic factor framework using Cholesky or sign restrictions.

### estimate sdfm

Estimate a Structural DFM.

```bash
# Cholesky identification (default)
friedman estimate sdfm macro.csv --factors=3

# Sign restrictions
friedman estimate sdfm macro.csv --factors=3 --id=sign --config=restrictions.toml

# Custom bandwidth and kernel
friedman estimate sdfm macro.csv --factors=3 --bandwidth=10 --kernel=parzen
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-q` | Int | auto | Number of dynamic factors |
| `--id` | | String | `cholesky` | `cholesky`, `sign` |
| `--var-lags` | | Int | 1 | Factor VAR lag order |
| `--horizon` | `-h` | Int | 40 | Structural IRF horizon |
| `--config` | | String | | TOML config for sign restrictions |
| `--bandwidth` | | Int | 0 | Spectral bandwidth (0 = auto) |
| `--kernel` | | String | `bartlett` | `bartlett`, `parzen`, `quadratic_spectral` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Identification method, factor VAR lags, shock names.

### irf sdfm

Structural DFM impulse response functions. Outputs panel-wide responses (all original variables).

```bash
friedman irf sdfm macro.csv --factors=3 --horizons=40
friedman irf sdfm macro.csv --id=sign --config=restrictions.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-q` | Int | auto | Number of dynamic factors |
| `--id` | | String | `cholesky` | `cholesky`, `sign` |
| `--var-lags` | | Int | 1 | Factor VAR lag order |
| `--horizons` | `-h` | Int | 40 | IRF horizon |
| `--config` | | String | | TOML config for sign restrictions |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-shock IRF tables with panel-wide variable responses.

### fevd sdfm

Structural DFM forecast error variance decomposition.

```bash
friedman fevd sdfm macro.csv --factors=3 --horizons=20
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | `-q` | Int | auto | Number of dynamic factors |
| `--id` | | String | `cholesky` | `cholesky`, `sign` |
| `--var-lags` | | Int | 1 | Factor VAR lag order |
| `--horizons` | `-h` | Int | 20 | FEVD horizon |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-factor FEVD proportions table.

## Key Concepts

### FAVAR vs Standard VAR

A standard VAR includes a small number of variables. FAVAR extracts latent factors from a large dataset (potentially hundreds of series) and includes them alongside key policy variables in the VAR. This allows:

- Information from a large cross-section without running into dimensionality issues
- Policy analysis (e.g., monetary policy shocks) while controlling for the broad state of the economy
- Panel-wide impulse responses: track how every series in the panel responds to structural shocks

### Two-Step vs Bayesian Estimation

- **Two-step** (default): Extract factors via PCA, then estimate the VAR on factors + key variables. Fast and straightforward.
- **Bayesian**: Joint estimation of factors and VAR parameters via MCMC. More coherent but computationally intensive.

### Panel-Wide Output

The `--panel-irf` and `--panel-forecast` flags map factor-space results back to the original variable space using the estimated factor loadings. This produces responses/forecasts for all N variables in the panel, not just the factors and key variables.
