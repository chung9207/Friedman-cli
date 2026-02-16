# hd

Compute historical decomposition of shocks. 4 subcommands: `var`, `bvar`, `lp`, `vecm`. All support `--from-tag`.

Historical decomposition decomposes observed data into contributions from each structural shock plus initial conditions.

## hd var

Frequentist historical decomposition.

```bash
friedman hd var data.csv --id=cholesky
friedman hd var data.csv --id=longrun --lags=4
friedman hd var data.csv --id=sign --config=sign_restrictions.toml
friedman hd var001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun`, `arias`, `uhlig` |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable table with columns: period, actual value, initial conditions, contribution from each shock. Includes decomposition verification.

## hd bvar

Bayesian historical decomposition with posterior mean contributions.

```bash
friedman hd bvar data.csv --draws=2000
friedman hd bvar data.csv --id=sign --config=sign_restrictions.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--config` | | String | | TOML config for identification/prior |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## hd lp

Historical decomposition via structural local projections.

```bash
friedman hd lp data.csv --id=cholesky
friedman hd lp data.csv --id=sign --config=sign_restrictions.toml --vcov=white
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | LP control lags |
| `--var-lags` | | Int | same as `--lags` | VAR lag order for identification |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--vcov` | | String | `newey_west` | `newey_west`, `white`, `driscoll_kraay` |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## hd vecm

Historical decomposition for Vector Error Correction Models. The VECM is converted to its VAR representation for decomposition.

```bash
friedman hd vecm data.csv --id=cholesky
friedman hd vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman hd vecm001    # from stored tag
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--rank` | `-r` | Int | auto | Cointegration rank (auto via Johansen) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--id` | | String | `cholesky` | Identification method |
| `--config` | | String | | TOML config for identification |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Per-variable table with columns: period, actual value, initial conditions, contribution from each shock.
