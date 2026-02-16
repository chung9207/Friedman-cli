# test

Statistical tests: unit root, cointegration, diagnostics, identification, and model comparison tests. 16 subcommands plus nested `var` (2) and `pvar` (4) nodes.

## Unit Root Tests

### test adf

Augmented Dickey-Fuller unit root test. H0: series has a unit root.

```bash
friedman test adf data.csv --column=1 --trend=constant
friedman test adf data.csv --column=2 --max-lags=8 --trend=trend
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--max-lags` | | Int | auto (AIC) | Maximum lag order |
| `--trend` | | String | `constant` | `none`, `constant`, `trend`, `both` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Test statistic, lags, p-value, rejection decision at 5%.

### test kpss

KPSS stationarity test. H0: series is stationary (reversed null compared to ADF).

```bash
friedman test kpss data.csv --column=1 --trend=constant
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--trend` | | String | `constant` | `constant`, `trend` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### test pp

Phillips-Perron unit root test. H0: series has a unit root.

```bash
friedman test pp data.csv --column=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--trend` | | String | `constant` | `none`, `constant`, `trend` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### test za

Zivot-Andrews unit root test with endogenous structural break.

```bash
friedman test za data.csv --column=1 --trend=both --trim=0.15
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--trend` | | String | `both` | `intercept`, `trend`, `both` |
| `--trim` | | Float64 | 0.15 | Trimming proportion |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Test statistic, estimated break date.

### test np

Ng-Perron unit root test (MZa, MZt, MSB, MPT statistics).

```bash
friedman test np data.csv --column=1 --trend=constant
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--trend` | | String | `constant` | `constant`, `trend` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## Cointegration

### test johansen

Johansen cointegration test with trace and max eigenvalue statistics.

```bash
friedman test johansen data.csv --lags=2 --trend=constant
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 2 | Lag order |
| `--trend` | | String | `constant` | `none`, `constant`, `trend` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Trace statistics table, max eigenvalue statistics table, estimated cointegration rank.

## VAR Diagnostics

### test var lagselect

Select optimal lag order for a VAR model.

```bash
friedman test var lagselect data.csv --max-lags=12 --criterion=aic
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--max-lags` | | Int | 12 | Maximum lag order to test |
| `--criterion` | | String | `aic` | `aic`, `bic`, `hqc` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Table of AIC/BIC/HQC for each lag order, optimal lag.

### test var stability

Check VAR stationarity via companion matrix eigenvalues.

```bash
friedman test var stability data.csv --lags=2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | Lag order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Companion matrix eigenvalues with moduli, stability verdict, max modulus.

## Non-Gaussian SVAR Diagnostics

### test normality

Normality test suite for VAR residuals. Useful as a pre-test for non-Gaussian SVAR methods.

```bash
friedman test normality data.csv --lags=4
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | VAR lag order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Multiple normality test results (statistic, p-value, df), rejection count.

### test identifiability

Test identifiability conditions for non-Gaussian SVAR. Runs up to 5 tests: identification strength, shock Gaussianity, shock independence, overidentification, and Gaussian vs non-Gaussian comparison.

```bash
friedman test identifiability data.csv --test=all
friedman test identifiability data.csv --test=strength
friedman test identifiability data.csv --test=gaussianity --method=jade
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | VAR lag order |
| `--test` | `-t` | String | `all` | `strength`, `gaussianity`, `independence`, `overidentification`, `all` |
| `--method` | | String | `fastica` | `fastica`, `jade`, `sobi`, `dcov`, `hsic` |
| `--contrast` | | String | `logcosh` | `logcosh`, `exp`, `kurtosis` (FastICA only) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### test heteroskedasticity

Heteroskedasticity-based SVAR identification. Estimates structural impact matrix B0 using variance changes across regimes.

```bash
friedman test heteroskedasticity data.csv --method=markov --regimes=2
friedman test heteroskedasticity data.csv --method=garch
friedman test heteroskedasticity data.csv --method=smooth_transition --config=config.toml
friedman test heteroskedasticity data.csv --method=external --config=config.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | VAR lag order |
| `--method` | | String | `markov` | `markov`, `garch`, `smooth_transition`, `external` |
| `--config` | | String | | TOML config (required for `smooth_transition` and `external`) |
| `--regimes` | | Int | 2 | Number of regimes |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Structural impact matrix (B0).

See [Configuration](../configuration.md) for the TOML format specifying transition/regime variables.

## Residual Diagnostics

### test arch\_lm

ARCH-LM test for conditional heteroskedasticity in a series. H0: no ARCH effects.

```bash
friedman test arch_lm data.csv --column=1 --lags=4
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--lags` | `-p` | Int | 4 | Number of lags |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### test ljung\_box

Ljung-Box test on squared residuals for serial autocorrelation. H0: no serial correlation in squared residuals.

```bash
friedman test ljung_box data.csv --column=1 --lags=10
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--lags` | `-p` | Int | 10 | Number of lags |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## Model Comparison Tests

### test granger

Granger causality test for VAR or VECM models.

```bash
friedman test granger data.csv --cause=1 --effect=2 --lags=4
friedman test granger data.csv --cause=1 --effect=2 --model=vecm --rank=1
friedman test granger data.csv --all --lags=4
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--cause` | | Int | 1 | Cause variable index (1-based) |
| `--effect` | | Int | 2 | Effect variable index (1-based) |
| `--model` | | String | `var` | `var`, `vecm` |
| `--rank` | `-r` | Int | auto | Cointegration rank (VECM only) |
| `--all` | | Flag | | Test all pairwise Granger causality |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Test statistic, p-value, rejection decision.

### test lr

Likelihood ratio test comparing two nested models by tag.

```bash
friedman test lr --restricted=var001 --unrestricted=var002
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--restricted` | | String | (required) | Tag for the restricted model |
| `--unrestricted` | | String | (required) | Tag for the unrestricted model |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** LR statistic, degrees of freedom, p-value, rejection decision.

### test lm

Lagrange multiplier test comparing two nested models by tag.

```bash
friedman test lm --restricted=var001 --unrestricted=var002
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--restricted` | | String | (required) | Tag for the restricted model |
| `--unrestricted` | | String | (required) | Tag for the unrestricted model |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** LM statistic, degrees of freedom, p-value, rejection decision.

## Panel VAR Diagnostics

Nested under `test pvar`. 4 subcommands for Panel VAR model diagnostics.

### test pvar hansen\_j

Hansen's J overidentification test for Panel VAR.

```bash
friedman test pvar hansen_j data.csv --id-col=country --time-col=year --lags=2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--id-col` | | String | (required) | Panel group identifier column |
| `--time-col` | | String | (required) | Panel time identifier column |
| `--vars` | | String | | Comma-separated dependent variables |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** J statistic, p-value, degrees of freedom.

### test pvar mmsc

Andrews-Lu MMSC model and moment selection criteria for optimal lag order.

```bash
friedman test pvar mmsc data.csv --id-col=country --time-col=year --max-lags=8
friedman test pvar mmsc data.csv --id-col=country --time-col=year --criterion=bic
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--max-lags` | | Int | 8 | Maximum lag order to test |
| `--criterion` | | String | `bic` | `aic`, `bic`, `hqc` |
| `--id-col` | | String | (required) | Panel group identifier column |
| `--time-col` | | String | (required) | Panel time identifier column |
| `--vars` | | String | | Comma-separated dependent variables |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** MMSC criteria table for each lag order, optimal lag.

### test pvar lagselect

Select optimal lag order for a Panel VAR model.

```bash
friedman test pvar lagselect data.csv --id-col=country --time-col=year --max-lags=8
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--max-lags` | | Int | 8 | Maximum lag order to test |
| `--criterion` | | String | `aic` | `aic`, `bic`, `hqc` |
| `--id-col` | | String | (required) | Panel group identifier column |
| `--time-col` | | String | (required) | Panel time identifier column |
| `--vars` | | String | | Comma-separated dependent variables |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Information criteria table for each lag order, optimal lag.

### test pvar stability

Check Panel VAR stationarity via companion matrix eigenvalues.

```bash
friedman test pvar stability data.csv --id-col=country --time-col=year --lags=2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--id-col` | | String | (required) | Panel group identifier column |
| `--time-col` | | String | (required) | Panel time identifier column |
| `--vars` | | String | | Comma-separated dependent variables |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Companion matrix eigenvalues with moduli, stability verdict, max modulus.
