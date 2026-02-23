# nowcast

Real-time nowcasting with mixed-frequency data. 5 subcommands: `dfm`, `bvar`, `bridge`, `news`, `forecast`.

All nowcast methods accept `--monthly-vars` and `--quarterly-vars` to specify the frequency split. When omitted, defaults to all-but-last as monthly and last column as the quarterly target.

## nowcast dfm

Nowcast via Dynamic Factor Model using the EM algorithm (Banbura, Giannone, & Reichlin 2011).

```bash
friedman nowcast dfm data.csv --factors=3 --lags=2
friedman nowcast dfm data.csv --monthly-vars=10 --quarterly-vars=2 --target-var=12
friedman nowcast dfm data.csv --idio=iid --max-iter=200 --plot
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--monthly-vars` | | Int | auto | Number of monthly variables (first N columns) |
| `--quarterly-vars` | | Int | auto | Number of quarterly variables (remaining columns) |
| `--factors` | `-r` | Int | 2 | Number of factors |
| `--lags` | `-p` | Int | 1 | Factor VAR lags |
| `--idio` | | String | `ar1` | Idiosyncratic component: `ar1`, `iid` |
| `--max-iter` | | Int | 100 | Maximum EM iterations |
| `--target-var` | | Int | 0 | Target variable index (0 = last) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Nowcast value, forecast value, log-likelihood, EM iterations.

## nowcast bvar

Nowcast via Bayesian VAR for mixed-frequency data.

```bash
friedman nowcast bvar data.csv --lags=5
friedman nowcast bvar data.csv --monthly-vars=8 --quarterly-vars=1 --target-var=9
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--monthly-vars` | | Int | auto | Number of monthly variables |
| `--quarterly-vars` | | Int | auto | Number of quarterly variables |
| `--lags` | `-p` | Int | 5 | VAR lags |
| `--target-var` | | Int | 0 | Target variable index (0 = last) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Nowcast value, forecast value, log-likelihood.

## nowcast bridge

Nowcast via bridge equations linking monthly indicators to the quarterly target.

```bash
friedman nowcast bridge data.csv
friedman nowcast bridge data.csv --lag-m=2 --lag-q=1 --lag-y=1
friedman nowcast bridge data.csv --monthly-vars=5 --quarterly-vars=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--monthly-vars` | | Int | auto | Number of monthly variables |
| `--quarterly-vars` | | Int | auto | Number of quarterly variables |
| `--lag-m` | | Int | 1 | Monthly indicator lags |
| `--lag-q` | | Int | 1 | Quarterly indicator lags |
| `--lag-y` | | Int | 1 | Dependent variable lags |
| `--target-var` | | Int | 0 | Target variable index (0 = last) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Nowcast value, forecast value, number of bridge equations.

## nowcast news

News decomposition for nowcast revisions (Banbura & Modugno 2014). Compares old and new data vintages to attribute nowcast revisions to individual data releases.

```bash
friedman nowcast news --data-new=new_vintage.csv --data-old=old_vintage.csv
friedman nowcast news --data-new=v2.csv --data-old=v1.csv --method=bvar
friedman nowcast news --data-new=v2.csv --data-old=v1.csv --target-period=50 --plot
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--data-new` | | String | (required) | Path to new vintage CSV |
| `--data-old` | | String | (required) | Path to old vintage CSV |
| `--monthly-vars` | | Int | auto | Number of monthly variables |
| `--quarterly-vars` | | Int | auto | Number of quarterly variables |
| `--method` | | String | `dfm` | `dfm`, `bvar` |
| `--factors` | `-r` | Int | 2 | Number of factors (DFM only) |
| `--lags` | `-p` | Int | 1 | Factor VAR lags |
| `--target-period` | | Int | 0 | Target period (0 = last) |
| `--target-var` | | Int | 0 | Target variable index (0 = last) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Old nowcast, new nowcast, revision, per-variable news impact table.

## nowcast forecast

Multi-step ahead forecast from a nowcasting model.

```bash
friedman nowcast forecast data.csv --method=dfm --horizons=8
friedman nowcast forecast data.csv --method=bvar --horizons=4
friedman nowcast forecast data.csv --method=bridge --horizons=4 --plot
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--monthly-vars` | | Int | auto | Number of monthly variables |
| `--quarterly-vars` | | Int | auto | Number of quarterly variables |
| `--method` | | String | `dfm` | `dfm`, `bvar`, `bridge` |
| `--factors` | `-r` | Int | 2 | Number of factors (DFM only) |
| `--lags` | `-p` | Int | 1 | Factor VAR lags |
| `--horizons` | `-h` | Int | 4 | Forecast horizon |
| `--target-var` | | Int | 0 | Target variable index (0 = last) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable forecast table across horizons.
