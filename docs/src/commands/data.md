# data

Data management commands: load example datasets, inspect, clean, transform, and validate data. 9 subcommands.

## data list

List available example datasets.

```bash
friedman data list
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Available datasets:**

| Name | Type | Dimensions | Description |
|------|------|------------|-------------|
| `fred_md` | Time Series | 804 x 126 | FRED-MD Monthly Database (126 macroeconomic indicators) |
| `fred_qd` | Time Series | 268 x 245 | FRED-QD Quarterly Database (245 macroeconomic indicators) |
| `pwt` | Panel | 38 x 74 x 42 | Penn World Table (38 OECD countries, 74 years, 42 variables) |

## data load

Load an example dataset or CSV file and export.

```bash
# Named example datasets
friedman data load fred_md --output=macro.csv
friedman data load fred_md --vars=INDPRO,UNRATE,CPIAUCSL --transform
friedman data load pwt --country=USA --output=us_data.csv

# From CSV file with date labels
friedman data load mydata --path=data.csv --dates=date_column
```

| Argument | Description |
|----------|-------------|
| `<name>` | Dataset name (`fred_md`, `fred_qd`, `pwt`) or label for `--path` |

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--path` | | String | | Path to CSV file (alternative to named dataset) |
| `--vars` | | String | | Comma-separated variable subset |
| `--country` | | String | | Country filter (PWT panel data only) |
| `--dates` | | String | | Column name for date labels |
| `--output` | `-o` | String | auto | Output CSV file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--transform` | `-t` | Flag | | Apply FRED transformation codes |

## data describe

Summary statistics for a dataset.

```bash
friedman data describe data.csv
friedman data describe data.csv --format=csv --output=stats.csv
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable: n, mean, std, min, p25, median, p75, max, skewness, kurtosis.

## data diagnose

Data quality diagnostics.

```bash
friedman data diagnose data.csv
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable: NaN count, Inf count, is-constant flag. Colored verdict (clean / issues found).

## data fix

Clean data by handling NaN, Inf, and constant columns.

```bash
friedman data fix data.csv --method=listwise
friedman data fix data.csv --method=interpolate --output=data_clean.csv
friedman data fix data.csv --method=mean
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | `-m` | String | `listwise` | `listwise`, `interpolate`, `mean` |
| `--output` | `-o` | String | auto | Output CSV file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

## data transform

Apply FRED transformation codes.

```bash
friedman data transform data.csv --tcodes=5,5,1,6
```

| Code | Transformation |
|------|---------------|
| 1 | Level (no transformation) |
| 2 | First difference |
| 3 | Second difference |
| 4 | Log |
| 5 | First difference of log |
| 6 | Second difference of log |
| 7 | First difference of percent change |

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--tcodes` | | String | (required) | Comma-separated codes, one per variable |
| `--output` | `-o` | String | auto | Output CSV file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

## data filter

Apply a time series filter (unified interface).

```bash
friedman data filter data.csv --method=hp --component=cycle
friedman data filter data.csv --method=hamilton --horizon=8 --lags=4
friedman data filter data.csv --method=bn --columns=1,3,5
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | `-m` | String | `hp` | `hp`, `hamilton`, `bn`, `bk`, `bhp` |
| `--component` | | String | `cycle` | `cycle`, `trend` |
| `--lambda` | `-l` | Float64 | 1600.0 | Smoothing parameter (HP/BHP) |
| `--horizon` | | Int | 8 | Forecast horizon (Hamilton) |
| `--lags` | `-p` | Int | 4 | Number of lags (Hamilton/BN) |
| `--columns` | `-c` | String | all | Column indices, comma-separated |
| `--output` | `-o` | String | | Export file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

## data validate

Validate data suitability for a model type.

```bash
friedman data validate data.csv --model=var
friedman data validate data.csv --model=garch
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--model` | | String | (required) | `var`, `bvar`, `vecm`, `arima`, `garch`, `sv`, `lp`, `gmm`, `factor`, `arch`, `egarch`, `gjr_garch`, `static`, `dynamic`, `gdfm` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## data balance

Balance a panel dataset with missing observations via DFM imputation.

```bash
friedman data balance data.csv --factors=3 --lags=2
friedman data balance data.csv --output=balanced.csv
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `dfm` | Imputation method |
| `--factors` | `-r` | Int | 3 | Number of factors |
| `--lags` | `-p` | Int | 2 | Factor VAR lags |
| `--output` | `-o` | String | | Export file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

**Output:** Balanced panel with imputed values.
