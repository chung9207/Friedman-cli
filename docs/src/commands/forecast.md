# forecast

Compute forecasts. 13 subcommands covering VAR, BVAR, LP, ARIMA, factor models, volatility models, and VECM. All support `--from-tag` for stored model reuse.

## forecast var

H-step ahead VAR point forecasts with analytical confidence intervals.

```bash
friedman forecast var data.csv --horizons=12 --confidence=0.95
friedman forecast var data.csv --lags=4 --horizons=24
friedman forecast var001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--confidence` | | Float64 | 0.95 | Confidence level for intervals |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable forecasts with lower/upper bounds and standard errors.

## forecast bvar

Bayesian forecasts with posterior credible intervals (16th/50th/84th percentiles).

```bash
friedman forecast bvar data.csv --horizons=12 --draws=2000
friedman forecast bvar data.csv --sampler=gibbs --config=prior.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--config` | | String | | TOML config for prior |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast lp

Direct LP forecasts with configurable impulse path and confidence intervals.

```bash
friedman forecast lp data.csv --shock=1 --horizons=12 --shock-size=1.0
friedman forecast lp data.csv --ci-method=bootstrap --n-boot=500
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--shock` | | Int | 1 | Shock variable index |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--shock-size` | | Float64 | 1.0 | Impulse shock size |
| `--lags` | `-p` | Int | 4 | LP control lags |
| `--vcov` | | String | `newey_west` | `newey_west`, `white`, `driscoll_kraay` |
| `--ci-method` | | String | `analytical` | `analytical`, `bootstrap`, `none` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--n-boot` | | Int | 500 | Bootstrap replications |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast arima

ARIMA forecast with auto model selection when `--p` is omitted.

```bash
friedman forecast arima data.csv --horizons=12 --confidence=0.95
friedman forecast arima data.csv --p=1 --d=1 --q=1 --horizons=24
friedman forecast arima data.csv --column=2 --criterion=aic
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | auto | AR order |
| `--d` | | Int | 0 | Differencing order |
| `--q` | | Int | 0 | MA order |
| `--max-p` | | Int | 5 | Max AR order for auto selection |
| `--max-d` | | Int | 2 | Max differencing order |
| `--max-q` | | Int | 5 | Max MA order |
| `--criterion` | | String | `bic` | `aic`, `bic` |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--confidence` | | Float64 | 0.95 | Confidence level |
| `--method` | `-m` | String | `css_mle` | `ols`, `css`, `mle`, `css_mle` |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast static

Forecast observables using a static factor model (PCA).

```bash
friedman forecast static data.csv --horizons=12
friedman forecast static data.csv --nfactors=3 --ci-method=bootstrap
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto (IC) | Number of factors |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--ci-method` | | String | `none` | `none`, `bootstrap`, `parametric` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast dynamic

Forecast observables using a dynamic factor model.

```bash
friedman forecast dynamic data.csv --nfactors=2 --factor-lags=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of factors |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--factor-lags` | `-p` | Int | 1 | Factor VAR lag order |
| `--method` | | String | `twostep` | `twostep`, `em` |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast gdfm

Forecast observables using a Generalized Dynamic Factor Model.

```bash
friedman forecast gdfm data.csv --dynamic-rank=2 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of static factors |
| `--dynamic-rank` | `-q` | Int | auto | Dynamic rank |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## Volatility Model Forecasts

All volatility forecast commands produce a table with `horizon`, `variance`, and `volatility` (= sqrt of variance) columns.

### forecast arch

```bash
friedman forecast arch data.csv --column=1 --q=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--q` | | Int | 1 | ARCH order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### forecast garch

```bash
friedman forecast garch data.csv --column=1 --p=1 --q=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | 1 | GARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### forecast egarch

```bash
friedman forecast egarch data.csv --column=1 --p=1 --q=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | 1 | EGARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### forecast gjr\_garch

```bash
friedman forecast gjr_garch data.csv --column=1 --p=1 --q=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--p` | | Int | 1 | GARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### forecast sv

```bash
friedman forecast sv data.csv --column=1 --draws=5000 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--draws` | `-n` | Int | 5000 | MCMC draws |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## forecast vecm

VECM forecasts with bootstrap confidence intervals.

```bash
friedman forecast vecm data.csv --horizons=12
friedman forecast vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman forecast vecm data.csv --confidence=0.90 --replications=1000
friedman forecast vecm001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--rank` | `-r` | Int | auto | Cointegration rank (auto via Johansen) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--confidence` | | Float64 | 0.95 | Confidence level |
| `--replications` | | Int | 500 | Bootstrap replications |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable forecasts with bootstrap confidence bands.
