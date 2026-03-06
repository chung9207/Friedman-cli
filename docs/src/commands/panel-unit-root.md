# Panel Unit Root Tests

Panel unit root tests that account for cross-sectional dependence via common factors. 4 subcommands under `test`.

All panel unit root tests accept CSV data in wide format (rows = time periods, columns = cross-sectional units) or panel format with `--id-col` and `--time-col` options.

## test panic

PANIC (Panel Analysis of Nonstationarity in Idiosyncratic and Common components) test by Bai & Ng (2004). Decomposes panel data into common factors and idiosyncratic components, then tests each for unit roots separately.

```bash
friedman test panic panel.csv --factors=auto
friedman test panic panel.csv --factors=3 --method=individual
friedman test panic panel.csv --id-col=country --time-col=year
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | | String | `auto` | Number of factors (`auto` or integer) |
| `--method` | | String | `pooled` | `pooled` (pooled ADF on defactored data), `individual` (unit-by-unit) |
| `--id-col` | | String | | Panel unit ID column (optional) |
| `--time-col` | | String | | Time column (optional) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Test statistic, p-value, number of factors, and verdict on panel-wide stationarity.

## test cips

Pesaran (2007) Cross-sectionally Augmented IPS (CIPS) test. Augments individual ADF regressions with cross-sectional averages to account for common factors without explicitly estimating them.

```bash
friedman test cips panel.csv
friedman test cips panel.csv --lags=4 --deterministic=trend
friedman test cips panel.csv --id-col=country --time-col=year
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | | String | `auto` | Lag order (`auto` or integer) |
| `--deterministic` | | String | `constant` | `constant`, `trend` |
| `--id-col` | | String | | Panel unit ID column (optional) |
| `--time-col` | | String | | Time column (optional) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** CIPS statistic, p-value, and rejection decision. CIPS is the average of individual CADF statistics.

## test moon-perron

Moon & Perron (2004) panel unit root test. Uses a factor-based approach where common factors are estimated and removed before applying modified t-statistics.

```bash
friedman test moon-perron panel.csv
friedman test moon-perron panel.csv --factors=2
friedman test moon-perron panel.csv --id-col=country --time-col=year
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | | String | `auto` | Number of factors (`auto` or integer) |
| `--id-col` | | String | | Panel unit ID column (optional) |
| `--time-col` | | String | | Time column (optional) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Modified t-bar and t-star statistics with p-values and number of factors.

## test factor-break

Factor break test for structural change in the factor structure of a panel. Tests whether the factor loadings or factor structure has changed at an unknown break point.

```bash
friedman test factor-break panel.csv --factors=2
friedman test factor-break panel.csv --factors=3 --method=chen_dolado_gonzalo
friedman test factor-break panel.csv --method=han_inoue --id-col=country --time-col=year
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--factors` | | Int | 2 | Number of factors |
| `--method` | | String | `breitung_eickmeier` | `breitung_eickmeier`, `chen_dolado_gonzalo`, `han_inoue` |
| `--id-col` | | String | | Panel unit ID column (optional) |
| `--time-col` | | String | | Time column (optional) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Test statistic, p-value, estimated break date, and method used.

### Methods

| Method | Reference |
|--------|-----------|
| `breitung_eickmeier` | Breitung & Eickmeier (2011) |
| `chen_dolado_gonzalo` | Chen, Dolado & Gonzalo (2014) |
| `han_inoue` | Han & Inoue (2015) |

## References

- Bai, J., & Ng, S. (2004). "A PANIC Attack on Unit Roots and Cointegration." *Econometrica*, 72(4), 1127--1177.
- Pesaran, M. H. (2007). "A Simple Panel Unit Root Test in the Presence of Cross-Section Dependence." *Journal of Applied Econometrics*, 22(2), 265--312.
- Moon, H. R., & Perron, B. (2004). "Testing for a Unit Root in Panels with Dynamic Factors." *Journal of Econometrics*, 122(1), 81--126.
- Breitung, J., & Eickmeier, S. (2011). "Testing for Structural Breaks in Dynamic Factor Models." *Journal of Econometrics*, 163(1), 71--84.
