# forecast

Compute forecasts. 16 model subcommands covering VAR, BVAR, LP, ARIMA, SETAR, STAR, factor models, volatility models, VECM, and FAVAR, plus a nested [`forecast evaluate`](#forecast-evaluate) sub-family (6 leaves) for post-hoc forecast evaluation and combination.

## Output format (C051)

`var`, `bvar`, `lp`, `arima`, `static`, `dynamic`, `gdfm`, `vecm`, `favar`, and `sdfm` (see
[favar & sdfm](favar.md)) all render through MEMs' tidy `long_table(result)`: one row per
`(horizon, variable)` cell, columns `horizon | variable | value | lower | upper`
(`lower`/`upper` are `missing` when the forecast carries no CI, e.g. `--ci-method=none`).
Univariate forecasts (ARIMA, volatility) reshape to the same schema with a single
`variable` value. `bvar`/`dynamic`/`gdfm`/`favar` previously hand-computed their forecasts
directly from posterior draws / factor loadings; they now route through MEMs'
`forecast(...)` first so the result is a typed `*Forecast` object `long_table` can render.
**Left wide on purpose:** the volatility leaves (`arch`/`garch`/`egarch`/`gjr_garch`/`sv`)
share a domain-specific `horizon | variance | volatility` table — collapsing it into the
generic tidy schema would drop the `volatility` (= √variance) column, so it stays a
principled exception (see [Volatility Model Forecasts](#volatility-model-forecasts) below).

## forecast var

H-step ahead VAR point forecasts with analytical or bootstrap confidence intervals.

```bash
friedman forecast var data.csv --horizons=12 --confidence=0.95
friedman forecast var data.csv --lags=4 --horizons=24
friedman forecast var data.csv --ci-method=bootstrap
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--confidence` | | Float64 | 0.95 | Confidence level for intervals |
| `--ci-method` | | String | `analytical` | `analytical`, `bootstrap` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`) — `lower`/`upper` are `missing` when the forecast has no CI.

!!! note "v0.3.0"
    VAR forecasts now return typed `VARForecast` objects with accessor functions: `point_forecast()`, `lower_bound()`, `upper_bound()`, `forecast_horizon()`.

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Tidy table (`horizon|variable|value|lower|upper`); `--conf-level=0.68` band from the BVAR posterior.

!!! note "v0.3.0"
    BVAR forecasts now return typed `BVARForecast` objects with the same accessor interface as `VARForecast`.

## forecast scenario

Waggoner–Zha conditional (scenario) forecasts: pin some variables to chosen paths and let
the model work out everything else, together with the structural shocks that would deliver
the scenario.

```bash
friedman forecast scenario data.csv --conditions-file=scenario.csv --horizons=12
friedman forecast scenario data.csv --conditions-file=scenario.csv --method=bvar --draws=2000
```

**Conditions file.** A long-format CSV, one condition per row:

```csv
variable,period,value,sd
gdp,1,2.5,0
gdp,2,2.0,0
inflation,4,3.1,0.5
```

- `variable` — a column name from your data, or a 1-based index.
- `period` — the forecast horizon the condition applies to (1 = first forecast period).
- `value` — the level the variable is pinned to.
- `sd` — optional. **0 (the default) is a hard condition**: the path is pinned exactly and
  the interval collapses to a point at that horizon. A positive `sd` makes it soft, so the
  model may deviate at a cost.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--conditions-file` | | String | | **Required.** Path to the conditions CSV |
| `--method` | | String | `var` | Model to condition: `var`, `bvar` |
| `--lags` | `-p` | Int | auto | Lag order (4 for `bvar`) |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--replications` | | Int | 1000 | Draws used for the conditional bands |
| `--confidence` | | Float64 | 0.95 | Confidence level in (0, 1) |
| `--draws` | `-n` | Int | 2000 | MCMC draws (`--method bvar`) |
| `--sampler` | | String | `direct` | `direct`, `gibbs` (`--method bvar`) |
| `--plot` / `--plot-save` | | Flag/String | | Plot the conditional path |

**Output:** the conditional path (`horizon`, `variable`, `value`, `lower`, `upper`,
`unconditional`), the implied structural shocks as a second table, and the settings.

The `unconditional` column is the baseline forecast from the same fit. A scenario is only
interpretable against the baseline it departs from, and computing it here rather than in a
second command guarantees both come from the same draws.

Check the implied shocks before believing a scenario. A path that requires structural shocks
far outside their historical range is arithmetically consistent but not economically
credible, and the shock table is what reveals that.

!!! note "The option is `--conditions-file`, not `--conditions`"
    `--conditions` is reserved: it prints the GPL conditions notice, and it is matched
    anywhere in the command line.

## forecast lp

Direct LP forecasts with configurable impulse path and confidence intervals.

!!! note "v0.3.0"
    `LPForecast` field renamed: `.forecast` (was `.forecasts` in earlier versions).

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`).

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`) with a single `variable` (univariate).

## forecast setar

Bootstrap-simulation forecast from a **self-exciting threshold autoregression (SETAR)**. The handler re-estimates the SETAR (see [`estimate setar`](estimate.md#estimate-setar)) and then simulates forward paths through the fitted two-regime dynamics, reporting the mean path and percentile bands. `--d` accepts an integer delay or `auto` (=`1:p` grid); `--ci-level` must be **exactly** `0.90`, `0.95`, or `0.99` (the re-estimated threshold CI uses the Hansen 2000 tabulation). Only self-exciting SETAR models are forecastable, so no external transition-variable option is offered.

```bash
friedman forecast setar y.csv --p=1 --d=1 --horizons=12
friedman forecast setar y.csv --p=2 --d=auto --horizons=6 --ci-level=0.90 --reps=2000
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | 1 | AR order (≥ 1) |
| `--d` | | String | `1` | Delay lag: an integer ≥ 1, or `auto` (=`1:p` grid) |
| `--horizons` | `-h` | Int | 12 | Forecast horizon (≥ 1) |
| `--reps` | | Int | 1000 | Bootstrap simulation paths (≥ 1) |
| `--ci-level` | | Float64 | 0.95 | Band coverage: `0.90`, `0.95`, or `0.99` (exact) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Tidy table (`horizon|variable|value|lower|upper`) with a single `variable` (univariate). `ThresholdForecast` is an `AbstractForecastResult`, so it renders through the shared `long_table` path. Unlike the other `forecast` leaves, `forecast setar` offers **no** `--plot`/`--plot-save`: MacroEconometricModels 0.7.0 ships no plot recipe for `ThresholdForecast` (only the fitted `ThresholdModel` from `estimate setar` is plottable), so per the plot-capable-leaves-only convention the flags are omitted rather than advertised-but-broken.

## forecast star

Bootstrap-simulation forecast from a **smooth-transition autoregression (STAR)**. The handler re-estimates a *self-exciting* STAR (see [`estimate star`](estimate.md#estimate-star)) and simulates forward paths through the fitted smooth-transition dynamics, drawing residuals with replacement and reporting the mean path and percentile bands. `--type` selects the transition shape (or `auto`). Only self-exciting STAR models are forecastable (a future path of an external transition variable would be required), so no `--transition-col` option is offered.

```bash
friedman forecast star y.csv --p=1 --d=1 --horizons=12
friedman forecast star y.csv --p=2 --type=lstr1 --horizons=6 --ci-level=0.90 --reps=2000
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | 1 | AR order (≥ 1) |
| `--d` | | Int | 1 | Delay lag for the self-exciting transition var (≥ 1) |
| `--type` | | String | `auto` | Transition shape: `lstr1`, `lstr2`, `estr`, `auto` |
| `--horizons` | `-h` | Int | 12 | Forecast horizon (≥ 1) |
| `--reps` | | Int | 1000 | Bootstrap simulation paths (≥ 1) |
| `--ci-level` | | Float64 | 0.95 | Band coverage: `0.90`, `0.95`, or `0.99` (exact) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Tidy table (`horizon|variable|value|lower|upper`) with a single `variable` (univariate). `STARForecast` is an `AbstractForecastResult`, so it renders through the shared `long_table` path. Like `forecast setar`, `forecast star` offers **no** `--plot`/`--plot-save`: MacroEconometricModels 0.7.0 ships no plot recipe for `STARForecast` (only the fitted `STARModel` from `estimate star` is plottable), so per the plot-capable-leaves-only convention the flags are omitted rather than advertised-but-broken.

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`).

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`).

## forecast gdfm

Forecast observables using a Generalized Dynamic Factor Model. `--method` selects the factor projection: `ar` fits an AR(1) on each two-sided factor (default); `one-sided` and `spectral` are the FHLR (2005) one-sided projection.

```bash
friedman forecast gdfm data.csv --dynamic-rank=2 --horizons=12
friedman forecast gdfm data.csv --dynamic-rank=2 --horizons=12 --method=one-sided
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of static factors |
| `--dynamic-rank` | `-q` | Int | auto | Dynamic rank |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--method` | | String | `ar` | Factor projection: `ar` (two-sided), `one-sided`, `spectral` |
| `--spectral` | | String | `lag-window` | GDFM spectrum: `lag-window` (FHLR), `smoothed-periodogram` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`).

## Volatility Model Forecasts

All volatility forecast commands produce a table with `horizon`, `variance`, and `volatility` (= sqrt of variance) columns. This is a deliberate exception to the [tidy `long_table` schema](#output-format-c051) used elsewhere in `forecast` — collapsing into the generic `horizon|variable|value|lower|upper` shape would drop the `volatility` column, so the shared `_vol_forecast_output` helper keeps its own domain-specific table.

### forecast arch

```bash
friedman forecast arch data.csv --column=1 --q=1 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--q` | | Int | 1 | ARCH order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

### forecast sv

```bash
friedman forecast sv data.csv --column=1 --draws=5000 --horizons=12
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index |
| `--draws` | `-n` | Int | 5000 | MCMC draws |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

## forecast vecm

VECM forecasts with bootstrap confidence intervals.

```bash
friedman forecast vecm data.csv --horizons=12
friedman forecast vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman forecast vecm data.csv --confidence=0.90 --replications=1000
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 12 | Forecast horizon |
| `--rank` | `-r` | Int | auto | Cointegration rank (auto via Johansen) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--confidence` | | Float64 | 0.95 | Confidence level |
| `--replications` | | Int | 500 | Bootstrap replications |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|value|lower|upper`); bootstrap confidence bands.

## forecast evaluate

Post-hoc evaluation and combination of **already-computed** forecasts (C072). These
leaves are model-agnostic: they take realized values plus competing forecasts, and
wrap the MEMs `fceval` toolkit (Diebold–Mariano, Clark–West, Mincer–Zarnowitz,
forecast encompassing, accuracy metrics, combination).

**Uniform input convention** — every leaf takes:

- `data` — realized values (CSV or data handle; positional). Required even with `--result`.
- `--actual <col>` — the realized-values column name (required).
- `--forecasts <col1,col2,...>` — forecast column names in `data` (CSV path).
- `--result <stem1,stem2,...>` — comma-separated forecast-result handle stems
  (`VARForecast` / `BVARForecast` / `ARIMAForecast` / `VECMForecast` /
  `FactorForecast` / … from `forecast * --save-result`).
  Point forecasts become the columns `--forecasts` would name; model names default to
  the stem basenames. An `H×n` forecast matrix is **not** flattened: the column
  matching `--actual` (via `varnames`) is used, else column 1. Do not combine
  with `--forecasts`. This is a **string** option (`handle=false`) so a comma
  list is not loaded as one path.

The handler forms whatever the underlying statistic needs from those series:
forecast **errors** `e = actual − forecast` (Diebold–Mariano), the forecast
**difference** `f_adj = f_small − f_big` (Clark–West squares it internally), or the
`T×M` forecast matrix (accuracy metrics, combination). Forecast-count arity is
validated per leaf (e.g. `dm` requires exactly 2) → usage error; unknown columns →
`data/bad-column`; result length mismatch → `data/shape`.

These result types are not Tables.jl-registered upstream, so their tables are
hand-built (a documented C051 exception, like the `io` and `estimate sur/3sls`
families): the test leaves emit a `metric | value` key–value table, `metrics` emits
a wide accuracy table plus the Theil decomposition, and `combine` emits a
`model | weight | mse` table.

| Leaf | Forecasts | Purpose |
|------|-----------|---------|
| `metrics` | ≥1 | Point-accuracy metrics (ME, MAE, RMSE, MAPE, sMAPE, MASE, Theil U1/U2) + Theil MSE bias/variance/covariance decomposition |
| `dm` | exactly 2 | Diebold–Mariano (1995) equal-predictive-accuracy test |
| `clark-west` | exactly 2 (small, big) | Clark–West (2007) adjusted-MSPE test for nested models |
| `mincer-zarnowitz` | exactly 1 | Mincer–Zarnowitz (1969) forecast-efficiency regression |
| `encompassing` | exactly 2 | Harvey–Leybourne–Newbold (1998) forecast-encompassing test |
| `combine` | ≥2 | Forecast combination (equal / Bates–Granger / Granger–Ramanathan weights) |

```bash
# Accuracy metrics + Theil decomposition for two competing forecasts
friedman forecast evaluate metrics data.csv --actual y --forecasts f1,f2

# Diebold–Mariano test (squared-error loss, 1-step); errors formed as y - f
friedman forecast evaluate dm data.csv --actual y --forecasts f1,f2 --loss se --horizon 1

# Clark–West test for a nested pair (f1 restricted, f2 unrestricted)
friedman forecast evaluate clark-west data.csv --actual y --forecasts f1,f2

# Mincer–Zarnowitz efficiency regression with HAC(4) covariance
friedman forecast evaluate mincer-zarnowitz data.csv --actual y --forecasts f1 --lags 4

# Combine three forecasts with inverse-MSE (Bates–Granger) weights
friedman forecast evaluate combine data.csv --actual y --forecasts f1,f2,f3 --method bates-granger

# Same metrics from saved forecast-result handles (names = stem basenames)
friedman forecast evaluate metrics macro --actual y --result fcst_var,fcst_bvar
```

### forecast evaluate metrics

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--actual` | String | (required) | Realized-values column name |
| `--forecasts` | String | (required unless `--result`) | Forecast column names, comma-separated (≥1) |
| `--result` | String | | Comma-separated forecast-result handle stems (alternative to `--forecasts`) |
| `--seasonal-period` | Int | 1 | Seasonal lag for the MASE naive-forecast scaling |
| `--format` | String | `table` | `table`, `csv`, `json` |
| `--output` | String | | Export file path |

**Output:** a wide accuracy table `model | ME | MAE | RMSE | MAPE | sMAPE | MASE | U1 | U2` (one row per forecast) and a `model | bias | variance | covariance` Theil MSE decomposition (proportions sum to 1).

### forecast evaluate dm

Diebold–Mariano test of equal predictive accuracy. Errors are formed internally as `e = actual − forecast`. A positive statistic means forecast 1 has the larger average loss (is worse). Invalid for nested models — use `clark-west` there.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--actual` | | String | (required) | Realized-values column name |
| `--forecasts` | | String | (required unless `--result`) | Exactly two forecast columns |
| `--result` | | String | | Two forecast-result handle stems (alternative to `--forecasts`) |
| `--loss` | | String | `se` | Loss function: `se` (squared) or `ad` (absolute) |
| `--horizon` | `-h` | Int | 1 | Forecast horizon (sets the truncation lag `h−1`) |
| `--alternative` | | String | `two-sided` | `two-sided`, `less`, `greater` |
| `--no-hln` | | Flag | | Disable the Harvey–Leybourne–Newbold small-sample correction (reference `N(0,1)` instead of `t_{T−1}`) |

**Output:** a `metric | value` table (statistic, p-value, mean loss differential, long-run variance, horizon, HLN flag, alternative, n).

### forecast evaluate clark-west

Clark–West adjusted-MSPE test for nested models. Give the two forecasts as **small (restricted) then big (unrestricted)**. Internally uses `e_small = y − f_small`, `e_big = y − f_big`, and `f_adj = f_small − f_big` (the library squares `f_adj`).

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--actual` | | String | (required) | Realized-values column name |
| `--forecasts` | | String | (required unless `--result`) | Exactly two columns: small, then big |
| `--result` | | String | | Two forecast-result handle stems: small, then big |
| `--horizon` | `-h` | Int | 1 | Forecast horizon (sets the truncation lag `h−1`) |
| `--alternative` | | String | `greater` | `two-sided`, `less`, `greater` |

**Output:** a `metric | value` table (one-sided CW statistic, p-value, mean adjusted differential, long-run variance, horizon, alternative, n).

### forecast evaluate mincer-zarnowitz

Mincer–Zarnowitz forecast-efficiency regression `actual = a + b·fc + u`, jointly testing `(a, b) = (0, 1)` with a Newey–West HAC covariance.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--actual` | String | (required) | Realized-values column name |
| `--forecasts` | String | (required unless `--result`) | Exactly one forecast column |
| `--result` | String | | One forecast-result handle stem (alternative to `--forecasts`) |
| `--lags` | Int | 0 | Newey–West HAC truncation lag (0 = White) |
| `--kernel` | String | `bartlett` | `bartlett`, `parzen`, `quadratic_spectral`, `tukey_hanning` |

**Output:** a `metric | value` table (`a`, `b`, HAC `se_a`/`se_b`, Wald χ²(2) and its p-value, the equivalent `F(2, T−2)` and its p-value, HAC lags, kernel, n).

### forecast evaluate encompassing

Regression-based forecast-encompassing test `actual = a + b₁·fc1 + b₂·fc2 + u`, testing `b₂ = 0`. Non-rejection means forecast 1 encompasses forecast 2.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--actual` | String | (required) | Realized-values column name |
| `--forecasts` | String | (required unless `--result`) | Exactly two forecast columns |
| `--result` | String | | Two forecast-result handle stems (alternative to `--forecasts`) |
| `--lags` | Int | 0 | Newey–West HAC truncation lag (0 = White) |
| `--kernel` | String | `bartlett` | `bartlett`, `parzen`, `quadratic_spectral`, `tukey_hanning` |

**Output:** a `metric | value` table (`b1`, `b2`, `se(b2)`, t-statistic on `b₂`, two-sided p-value, HAC lags, kernel, n).

### forecast evaluate combine

Combine ≥2 forecasts into one series.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--actual` | String | (required) | Realized-values column name |
| `--forecasts` | String | (required unless `--result`) | Forecast column names, comma-separated (≥2) |
| `--result` | String | | Comma-separated forecast-result handle stems (≥2; alternative to `--forecasts`) |
| `--method` | String | `equal` | `equal`, `bates-granger` (inverse-MSE), `granger-ramanathan` (constrained least squares) |
| `--emit-series` | Flag | | Also emit the combined forecast series (`index | combined`) |

**Output:** a `model | weight | mse` table (weights sum to 1; Granger–Ramanathan weights may be negative). With `--emit-series`, an additional `index | combined` table carries the combined series.

## See Also

For FAVAR forecasting, see [favar & sdfm](favar.md#forecast-favar); for Structural DFM forecasting, see [forecast sdfm](favar.md#forecast-sdfm). For DSGE model forecasting via simulation, see [dsge simulate](dsge.md#dsge-simulate).
