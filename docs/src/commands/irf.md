# irf

Compute impulse response functions. 3 subcommands: `var` (frequentist), `bvar` (Bayesian), `lp` (local projections). All support `--from-tag` to reuse stored models.

## irf var

Frequentist IRFs with multiple identification schemes and confidence intervals.

```bash
# Cholesky identification (default)
friedman irf var data.csv --shock=1 --horizons=20

# Sign restrictions (requires config)
friedman irf var data.csv --id=sign --config=sign_restrictions.toml

# Narrative sign restrictions
friedman irf var data.csv --id=narrative --config=narrative.toml

# Long-run (Blanchard-Quah) identification
friedman irf var data.csv --id=longrun --horizons=40

# Arias et al. (2018) zero/sign restrictions
friedman irf var data.csv --id=arias --config=arias_restrictions.toml

# Non-Gaussian identification methods
friedman irf var data.csv --id=fastica
friedman irf var data.csv --id=jade

# With bootstrap confidence intervals
friedman irf var data.csv --shock=1 --ci=bootstrap --replications=1000

# From stored model tag
friedman irf var001
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--id` | | String | `cholesky` | Identification method (see below) |
| `--ci` | | String | `bootstrap` | `none`, `bootstrap`, `theoretical` |
| `--replications` | | Int | 1000 | Bootstrap replications |
| `--config` | | String | | TOML config for identification restrictions |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### Identification Methods

| ID String | Method | Config Required |
|-----------|--------|----------------|
| `cholesky` | Cholesky decomposition (recursive) | No |
| `sign` | Sign restrictions | Yes (sign matrix) |
| `narrative` | Narrative sign restrictions | Yes (narrative block) |
| `longrun` | Long-run (Blanchard-Quah) | No |
| `arias` | Arias et al. zero + sign | Yes (restrictions) |
| `fastica` | FastICA | No |
| `jade` | JADE | No |
| `sobi` | SOBI | No |
| `dcov` | Distance covariance | No |
| `hsic` | Hilbert-Schmidt independence | No |
| `student_t` | Student-t ML | No |
| `mixture_normal` | Mixture normal ML | No |
| `pml` | Pseudo-ML | No |
| `skew_normal` | Skew-normal ML | No |
| `markov_switching` | Markov-switching heteroskedasticity | No |
| `garch_id` | GARCH-based heteroskedasticity | No |

See [Configuration](../configuration.md) for restriction TOML formats.

## irf bvar

Bayesian IRFs with 68% credible intervals (16th/50th/84th percentiles).

```bash
friedman irf bvar data.csv --shock=1 --horizons=20
friedman irf bvar data.csv --draws=5000 --sampler=gibbs --config=prior.toml
friedman irf bvar data.csv --id=sign --config=sign_restrictions.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--config` | | String | | TOML config for identification/prior |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Median IRFs with 16th/84th percentile bands per variable.

## irf lp

Structural LP impulse response functions. Supports multi-shock analysis.

```bash
# Single shock
friedman irf lp data.csv --id=cholesky --shock=1 --horizons=20

# Multiple shocks
friedman irf lp data.csv --shocks=1,2,3 --id=cholesky --horizons=30

# With bootstrap CI
friedman irf lp data.csv --id=cholesky --ci=bootstrap --replications=500
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--shock` | | Int | 1 | Single shock index (1-based) |
| `--shocks` | | String | | Comma-separated shock indices (e.g. `1,2,3`) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--lags` | `-p` | Int | 4 | LP control lags |
| `--var-lags` | | Int | same as `--lags` | VAR lag order for identification |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun` |
| `--ci` | | String | `none` | `none`, `bootstrap` |
| `--replications` | | Int | 200 | Bootstrap replications |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--vcov` | | String | `newey_west` | `newey_west`, `white`, `driscoll_kraay` |
| `--config` | | String | | TOML config for sign/narrative restrictions |
| `--from-tag` | | String | | Load model from stored tag |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
