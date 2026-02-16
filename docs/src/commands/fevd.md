# fevd

Compute forecast error variance decomposition. 5 subcommands: `var`, `bvar`, `lp`, `vecm`, `pvar`. All support `--from-tag`.

## fevd var

Frequentist FEVD with configurable identification.

```bash
friedman fevd var data.csv --horizons=20 --id=cholesky
friedman fevd var data.csv --id=sign --config=sign_restrictions.toml
friedman fevd var001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 20 | Forecast horizon |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun`, `arias`, `uhlig` |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** FEVD proportions table per variable (columns = shocks, rows = horizons).

## fevd bvar

Bayesian FEVD with posterior mean proportions.

```bash
friedman fevd bvar data.csv --horizons=20
friedman fevd bvar data.csv --draws=5000 --sampler=gibbs
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--horizons` | `-h` | Int | 20 | Forecast horizon |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--config` | | String | | TOML config for identification/prior |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## fevd lp

LP-based FEVD with bias-corrected proportions (Gorodnichenko & Lee 2019).

```bash
friedman fevd lp data.csv --horizons=20 --id=cholesky
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--horizons` | `-h` | Int | 20 | Forecast horizon |
| `--lags` | `-p` | Int | 4 | LP control lags |
| `--var-lags` | | Int | same as `--lags` | VAR lag order for identification |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--vcov` | | String | `newey_west` | `newey_west`, `white`, `driscoll_kraay` |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## fevd vecm

VECM-based FEVD. The VECM is converted to its VAR representation for decomposition.

```bash
friedman fevd vecm data.csv --horizons=20
friedman fevd vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman fevd vecm001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 20 | Forecast horizon |
| `--rank` | `-r` | Int | auto | Cointegration rank (auto via Johansen) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--id` | | String | `cholesky` | Identification method |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## fevd pvar

Panel VAR forecast error variance decomposition.

```bash
friedman fevd pvar data.csv --id-col=country --time-col=year --horizons=20
friedman fevd pvar001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--horizons` | `-h` | Int | 20 | Forecast horizon |
| `--id-col` | | String | | Panel group identifier column |
| `--time-col` | | String | | Panel time identifier column |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
