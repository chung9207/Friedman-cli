# did

Difference-in-differences estimation, event study local projections, and diagnostics. 7 subcommands organized as 3 estimation commands + nested `test` group with 4 diagnostic commands.

Supports panel data with staggered treatment adoption. All commands accept panel CSV data as a positional argument with `--id-col` and `--time-col` options (default: first and second columns respectively).

## did estimate

Estimate treatment effects using difference-in-differences. 5 estimators: TWFE, Callaway-Sant'Anna, Sun-Abraham, Borusyak-Jaravel-Spiess, de Chaisemartin-D'Haultfoeuille.

```bash
# Two-way fixed effects (default)
friedman did estimate panel.csv --outcome=y --treatment=treat

# Callaway-Sant'Anna (2021)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=cs --control-group=never_treated

# Sun-Abraham (2021)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=sa

# Borusyak-Jaravel-Spiess (2024) imputation estimator
friedman did estimate panel.csv --outcome=y --treatment=treat --method=bjs

# de Chaisemartin-D'Haultfoeuille (2020)
friedman did estimate panel.csv --outcome=y --treatment=treat --method=dcdh --n-boot=500

# With covariates and explicit panel columns
friedman did estimate panel.csv --outcome=y --treatment=treat --method=cs \
    --id-col=state --time-col=year --covariates=x1,x2 --cluster=twoway
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--outcome` | | String | (required) | Outcome variable column name |
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--method` | | String | `twfe` | `twfe`, `cs`, `sa`, `bjs`, `dcdh` |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--leads` | | Int | 0 | Pre-treatment periods |
| `--horizon` | | Int | 5 | Post-treatment periods |
| `--covariates` | | String | | Comma-separated covariate column names |
| `--control-group` | | String | `never_treated` | `never_treated` or `not_yet_treated` |
| `--cluster` | | String | `unit` | `unit`, `time`, or `twoway` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--n-boot` | | Int | 200 | Bootstrap replications (dCdH only) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** ATT table (event time, ATT, SE, CI bounds) + overall ATT summary. Callaway-Sant'Anna (`cs`) additionally outputs group-time ATT matrix.

### DID Methods

| Method | `--method` | Reference |
|--------|-----------|-----------|
| Two-way Fixed Effects | `twfe` | Standard TWFE regression |
| Callaway-Sant'Anna | `cs` | Callaway & Sant'Anna (2021) |
| Sun-Abraham | `sa` | Sun & Abraham (2021) |
| Borusyak-Jaravel-Spiess | `bjs` | Borusyak, Jaravel & Spiess (2024) |
| de Chaisemartin-D'Haultfoeuille | `dcdh` | de Chaisemartin & D'Haultfoeuille (2020) |

## did event-study

Panel event study via local projections (Jorda 2005 + panel fixed effects).

```bash
friedman did event-study panel.csv --outcome=y --treatment=treat --leads=3 --horizon=5
friedman did event-study panel.csv --outcome=y --treatment=treat --lags=6 --cluster=twoway
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--outcome` | | String | (required) | Outcome variable column name |
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--leads` | | Int | 3 | Pre-treatment leads |
| `--horizon` | | Int | 5 | Post-treatment horizon |
| `--lags` | `-p` | Int | 4 | Control lags |
| `--covariates` | | String | | Comma-separated covariate column names |
| `--cluster` | | String | `unit` | `unit`, `time`, or `twoway` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Coefficient table (event time, coefficient, SE, CI bounds) + panel summary.

## did lp-did

LP-DiD with clean controls (Dube, Girardi, Jorda & Taylor 2023). Uses not-yet-treated units as the control group to avoid forbidden comparisons.

```bash
friedman did lp-did panel.csv --outcome=y --treatment=treat --leads=3 --horizon=5
```

Options are identical to `did event-study`.

**Output:** Coefficient table with clean-controls indicator + panel summary.

## did test bacon

Bacon decomposition (Goodman-Bacon 2021). Decomposes the TWFE estimator into weighted 2x2 DID comparisons to diagnose heterogeneity bias.

```bash
friedman did test bacon panel.csv --outcome=y --treatment=treat
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--outcome` | | String | (required) | Outcome variable column name |
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Decomposition table (comparison type, cohort pair, estimate, weight) + overall TWFE ATT.

## did test pretrend

Test the parallel trends assumption using pre-treatment coefficients.

```bash
# Test from DID estimation
friedman did test pretrend panel.csv --outcome=y --treatment=treat

# Test from event study
friedman did test pretrend panel.csv --outcome=y --treatment=treat --method=event-study
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--outcome` | | String | (required) | Outcome variable column name |
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--leads` | | Int | 3 | Pre-treatment leads |
| `--horizon` | | Int | 5 | Post-treatment horizon |
| `--lags` | `-p` | Int | 4 | Control lags (event-study only) |
| `--cluster` | | String | `unit` | `unit`, `time`, or `twoway` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--method` | | String | `did` | `did` or `event-study` |
| `--did-method` | | String | `twfe` | DID method (when `--method=did`) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** F-statistic, p-value, degrees of freedom, and verdict on parallel trends.

## did test negweight

Check for negative weights in TWFE estimation (de Chaisemartin & D'Haultfoeuille 2020).

```bash
friedman did test negweight panel.csv --treatment=treat
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Whether negative weights exist, count, total negative weight. If found, details table of affected cohort-time pairs.

## did test honest

HonestDiD sensitivity analysis (Rambachan & Roth 2023). Computes robust confidence intervals allowing for bounded violations of parallel trends.

```bash
# From DID estimation
friedman did test honest panel.csv --outcome=y --treatment=treat --mbar=1.0

# From event study
friedman did test honest panel.csv --outcome=y --treatment=treat --method=event-study --mbar=0.5
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--outcome` | | String | (required) | Outcome variable column name |
| `--treatment` | | String | (required) | Treatment indicator column name |
| `--id-col` | | String | (1st col) | Panel unit ID column |
| `--time-col` | | String | (2nd col) | Time column |
| `--mbar` | | Float64 | 1.0 | Violation bound M-bar |
| `--leads` | | Int | 3 | Pre-treatment leads |
| `--horizon` | | Int | 5 | Post-treatment horizon |
| `--lags` | `-p` | Int | 4 | Control lags (event-study only) |
| `--cluster` | | String | `unit` | `unit`, `time`, or `twoway` |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--method` | | String | `did` | `did` or `event-study` |
| `--did-method` | | String | `twfe` | DID method (when `--method=did`) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Post-treatment ATT with robust and original CI bounds + breakdown value (maximum M-bar for which zero is excluded from CI).
