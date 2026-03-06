# Structural Break Tests

Tests for structural breaks in time series regression models. 2 subcommands under `test`.

## test andrews

Andrews (1993) supremum-type structural break test. Tests for a single unknown break point in a linear regression by computing the sup, exponential, or mean of Wald/LR/LM statistics over candidate break dates.

```bash
friedman test andrews data.csv --response=1 --test=supwald
friedman test andrews data.csv --response=2 --test=explr --trimming=0.20
friedman test andrews data.csv --response=1 --test=meanlm
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--response` | | Int | 1 | Response variable column index (1-based) |
| `--test` | | String | `supwald` | Test type (see below) |
| `--trimming` | | Float64 | 0.15 | Trimming proportion (fraction of endpoints excluded) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

### Test Types

| Value | Description |
|-------|-------------|
| `supwald` | Supremum Wald statistic |
| `suplr` | Supremum LR statistic |
| `suplm` | Supremum LM statistic |
| `expwald` | Exponential Wald statistic |
| `explr` | Exponential LR statistic |
| `explm` | Exponential LM statistic |
| `meanwald` | Mean Wald statistic |
| `meanlr` | Mean LR statistic |
| `meanlm` | Mean LM statistic |

**Output:** Test statistic, p-value, estimated break date, and rejection decision.

## test bai-perron

Bai-Perron (1998) multiple structural break test. Estimates the number and location of multiple break points in a linear regression using a sequential or global optimization approach.

```bash
friedman test bai-perron data.csv --response=1 --max-breaks=5
friedman test bai-perron data.csv --response=2 --max-breaks=3 --criterion=lwz
friedman test bai-perron data.csv --response=1 --trimming=0.20
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--response` | | Int | 1 | Response variable column index (1-based) |
| `--max-breaks` | | Int | 5 | Maximum number of breaks to test |
| `--trimming` | | Float64 | 0.15 | Trimming proportion |
| `--criterion` | | String | `bic` | `bic`, `lwz` (Liu-Wu-Zidek) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Number of estimated breaks, break dates with confidence intervals, and segment-specific coefficient estimates. Sequential and global test statistics for each potential number of breaks.

## References

- Andrews, D. W. K. (1993). "Tests for Parameter Instability and Structural Change with Unknown Change Point." *Econometrica*, 61(4), 821--856.
- Bai, J., & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple Structural Changes." *Econometrica*, 66(1), 47--78.
