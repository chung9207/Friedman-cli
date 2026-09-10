# data

Data management commands: import/export typed handles, load example datasets, inspect, clean, transform, and validate data. 13 subcommands.

## Typed handles and stems

The preferred unit of work is a MEMs typed container (`TimeSeriesData`,
`PanelData`, `CrossSectionData`) persisted as a **stem** — pass `macro`, not
`macro.jld2`. `.jld2` is storage. CSV remains legal on every leaf that accepted
it before (additive 0.x).

- **Load (data slots):** `macro.jld2` if it exists (preferred), else
  `macro.csv`, else the exact path. Explicit suffixes skip the search
  (`macro.csv` is always CSV). `--model` on typed leaves (nonempty
  `model_types`) stem-resolves like `--result` (`var` → `var.jld2`, no CSV
  fallback). `model info` still wants `var.jld2` (or `.fmod` / `model://`).
- **Save:** a suffix-less `-o` / `--save-model` becomes `.jld2`.
  `data import … -o out.jld2` (or a stem) **is** the CSV → typed conversion.
  An **edit** of a CSV with `-o out.jld2` is `usage/invalid` (import first).
  Omitting `--save-model` means do not save (no default stem).
- **Wrong kind:** a handle whose type is not in the leaf's `data_kinds` is
  `data/wrong-kind` (exit 3) — e.g. a panel handle on `estimate var`, a raw CSV
  on `data export`, or `apply_tcode` on `CrossSectionData`.

| Input | `-o` | Writes |
|-------|------|--------|
| handle | omitted / stem | same type, default `*_clean.jld2` / `*_transformed.jld2` / … |
| handle | `out.csv` | CSV export; stderr warns metadata dropped |
| `.csv` | omitted / stem | CSV (today’s path, default `*_clean.csv`) |
| `.csv` edit | `out.jld2` | `usage/invalid` — run `data import` first |
| `data import` CSV | stem / `out.jld2` | typed handle (intended conversion) |

No implicit in-place overwrite. `data fix macro -o macro` is allowed (explicit
same stem); stderr notes the replace. See [Architecture](../architecture.md)
for the full pipeline.

`friedman show STEM` renders any loadable handle (data, model, or result).
Data containers emit the same descriptive-stats table as `data describe`;
bundles list keys only and are not unpacked. Stem resolution uses the result
slot (`.jld2`, no CSV fallback).

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
| `mpdta` | Panel | 500 x 5 x 3 | Callaway-Sant'Anna (2021) minimum wage panel |
| `ddcg` | Panel | 184 x 51 x 2 | Acemoglu et al. democracy-GDP panel |
| `denmark` | Time Series | 55 x 5 | Danish money-demand data (Johansen-Juselius cointegration) |
| `gnp_hamilton` | Time Series | 135 x 1 | US GNP growth (Hamilton 1989 Markov-switching example) |
| `grunfeld` | Panel | 10 x 20 x 3 | Grunfeld investment panel (10 firms, 20 years) |
| `mp_shocks` | Time Series | 240 x 8 | US monetary panel + policy-shock series (McKay-Wolf 2023; NaN outside published samples) |
| `mroz` | Cross Section | 753 x 22 | Mroz (1987) female labour supply |
| `nile` | Time Series | 100 x 1 | Nile river annual flow (local-level state space) |
| `stackloss` | Cross Section | 21 x 4 | Brownlee stack-loss plant data (robust regression) |

The `wiot` input-output table is not listed here: it is an IO archive served by the
[`io`](io.md) family rather than a rectangular dataset.

!!! warning "NaN-padded series: NaN is not zero"
    `mp_shocks` (quarterly, 1960Q1–2019Q4) keeps each policy-shock series `NaN` outside
    its published sample: `rr` (Romer–Romer/Wieland–Yang) covers 1969Q1–2007Q4, `mp1`
    (Gertler–Karadi) 1988Q4–2012Q2, `ad` (Aruoba–Drechsel) 1982Q4–2008Q3, `bzk_ist`
    (Ben Zeev–Khan) through 2012Q1, and `ygap` starts in 1969Q1. Loading preserves the
    `NaN` cells — zero is a valid shock value, so they are never zero-filled (zero-filling
    is an estimation-time IV convention, not a property of the data). Use
    `data describe` to see each column's valid window (`first_valid`/`last_valid`), then
    `data dropna --vars ...` or `data keeprows --rows ...` to cut a finite sample before
    estimation.

### Referring to a dataset

Every command that takes a `<data>` path also accepts a `:name` reference to a bundled
dataset, and both separator spellings resolve to the same set:

```bash
friedman data describe :fred_md      # equivalently :fred-md, fred_md, fred-md
friedman test cips :grunfeld --id-col=group --time-col=time
```

Panel datasets expose their identifiers as leading `group` and `time` columns, so panel
commands can bind `--id-col`/`--time-col` directly to a bundled panel.

## data load

Load an example dataset or CSV file and export.

```bash
# Named example datasets
friedman data load fred_md --output=macro.csv
friedman data load fred_md --vars=INDPRO,UNRATE,CPIAUCSL --transform
friedman data load pwt --country=USA --output=us_data.csv
friedman data load :fred-md --output=macro.csv     # ':' reference also accepted

# From a CSV file — no dataset name needed
friedman data load --path=data.csv --dates=date_column
```

Give either a dataset `<name>` or `--path`; supplying neither is a usage error. When both
are given, `--path` wins and a note is written to stderr.

| Argument | Description |
|----------|-------------|
| `<name>` | Example dataset name (optional; see `data list`). Omit when using `--path` |

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--path` | | String | | Path to CSV file (alternative to named dataset) |
| `--vars` | | String | | Comma-separated variable subset |
| `--country` | | String | | Country filter (PWT panel data only) |
| `--dates` | | String | | Column name for date labels |
| `--output` | `-o` | String | auto | Output CSV file path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--transform` | `-t` | Flag | | Apply FRED transformation codes |

## data import

Import a CSV or a bundled `:example` dataset to a typed `.jld2` handle (`TimeSeriesData`, `PanelData`, or `CrossSectionData`). `--kind` is required for CSV and optional for `:example` (inferred; a mismatch is `data/wrong-kind`).

```bash
friedman data import macro.csv --kind timeseries --frequency quarterly -o macro
friedman data import panel.csv --kind panel --id-col group --time-col time -o panel
friedman data import xs.csv --kind cross-section -o xs
friedman data import :fred_md -o fred_md
```

Default `-o` is the input basename next to the source (or `fred_md.jld2` for `:fred_md`). A suffix-less stem becomes `.jld2`.

| Argument | Description |
|----------|-------------|
| `<data>` | CSV path, stem, or `:example` dataset |

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--kind` | | String | | `timeseries`, `panel`, `cross-section` (required for CSV) |
| `--frequency` | | String | `other` | `daily`, `monthly`, `quarterly`, `annual`, `mixed`, `other` (illegal on cross-section) |
| `--dates` | | String | | CSV column of date labels (timeseries) |
| `--id-col` | | String | | Panel group column (required for `--kind panel`) |
| `--time-col` | | String | | Panel time column (required for `--kind panel`) |
| `--vars` | | String | | Comma-separated variable subset |
| `--tcodes` | | String | | Comma-separated FRED tcode per variable |
| `--note` | | String | | Free-form note stored in the handle header |
| `--output` | `-o` | String | input basename | Output stem or path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

**Output table:** `imported_data` — `kind`, `n_obs`, `n_vars`, `path`, `frequency`.

## data export

Write a typed handle to CSV (the inverse of `data import`). Panel handles include `group`/`time` columns. Frequency, tcode, and dates are dropped (stderr note). Input must be a data container (`TimeSeriesData` / `PanelData` / `CrossSectionData`); a raw CSV is `data/wrong-kind`.

```bash
friedman data export macro -o macro.csv
friedman data export panel
```

Default `-o` is `<stem>.csv` next to the handle.

| Argument | Description |
|----------|-------------|
| `<data>` | Handle stem or path |

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--output` | `-o` | String | `<stem>.csv` | Output CSV path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

No envelope table — writes the CSV directly.

## data describe

Summary statistics for a dataset. Accepts a CSV path, a `:example` name, or a typed handle stem. A `PanelData` handle is described as panel data (not wrapped as `TimeSeriesData`).

```bash
friedman data describe data.csv
friedman data describe data.csv --format=csv --output=stats.csv
friedman data describe panel
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable: n, first\_valid, last\_valid, mean, std, min, p25, median, p75,
max, skewness, kurtosis. `n` counts finite observations and the statistics exclude
NaN/Inf; `first_valid`/`last_valid` are the row indices of the first and last finite
value (0 if none), which locate the usable window of NaN-padded series such as the
`mp_shocks` shock columns.

## data diagnose

Data quality diagnostics. Same inputs as `data describe` (CSV, `:example`, or typed handle).

```bash
friedman data diagnose data.csv
friedman data diagnose panel
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable: NaN count, Inf count, is-constant flag. Colored verdict (clean / issues found).

## data fix

Clean data by handling NaN, Inf, and constant columns. A typed handle is cleaned in place as the same type (`*_clean.jld2` by default). CSV input still writes CSV; CSV `-o out.jld2` is `usage/invalid` (run `data import` first). Handle `-o out.csv` writes CSV and notes that metadata was dropped.

```bash
friedman data fix data.csv --method=listwise
friedman data fix macro --method=listwise -o macro_clean
friedman data fix data.csv --method=interpolate --output=data_clean.csv
friedman data fix data.csv --method=mean
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | `-m` | String | `listwise` | `listwise`, `interpolate`, `mean` |
| `--output` | `-o` | String | auto | Output stem (handle) or CSV path |
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
| `--output` | `-o` | String | auto | Output stem (handle) or CSV path |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

`apply_tcode` is defined for time-series and panel data only. A `CrossSectionData` handle is `data/wrong-kind`.

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

Validate data suitability for a model type. `--model` is a **model-type string** (`var`, `arima`, …), not a saved-model handle.

```bash
friedman data validate data.csv --model=var
friedman data validate data.csv --model=garch
friedman data validate panel --model=var
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

**Output:** Balanced panel with imputed values (envelope table `balanced_panel`). Also persists the cleaned object (`*_balanced.jld2` from a handle, `*_balanced.csv` from CSV).

## data dropna

Drop all rows containing NaN or missing values from a dataset.

```bash
friedman data dropna data.csv
friedman data dropna data.csv --output=data_clean.csv
friedman data dropna macro.csv --vars=ygap,infl,ffr   # check only these columns
```

With `--vars`, only the listed columns are checked for NaN/Inf (a row with NaN in an
*unlisted* column is kept). This is the prep step for NaN-padded datasets like
`mp_shocks`: subset the variables of interest, drop their jointly-invalid rows, and
estimate on the result. An unknown column name is a typed data error; if every row
contains NaN/Inf the command fails with `data/invalid` rather than emitting an empty
dataset.

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--vars` | | String | (all) | Column names to check, comma-separated |
| `--output` | `-o` | String | (stdout) | Export results to file |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

**Output:** Cleaned dataset with missing rows removed, plus a summary of rows dropped (envelope table `cleaned_data`). Also persists the cleaned object (`*_dropna.jld2` from a handle, `*_dropna.csv` from CSV).

## data keeprows

Keep only rows at the given index positions (range or comma-separated list).

```bash
friedman data keeprows data.csv --rows=1:100
friedman data keeprows data.csv --rows=1,5,10 --output=subset.csv
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--rows` | | String | (all) | Row indices to keep (e.g. `1:100`, `1,5,10`) |
| `--output` | `-o` | String | (stdout) | Export results to file |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |

**Output:** Filtered dataset containing only the selected rows (envelope table `filtered_data`). Also persists the selected object (`*_rows.jld2` from a handle, `*_rows.csv` from CSV).
