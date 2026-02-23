# irf

Compute impulse response functions. 5 subcommands: `var`, `bvar`, `lp`, `vecm`, `pvar`.

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

# Cumulative IRFs (for differenced data)
friedman irf var data.csv --shock=1 --cumulative

# Full identified set for sign restrictions
friedman irf var data.csv --id=sign --config=sign.toml --identified-set

# Filter non-stationary draws
friedman irf var data.csv --shock=1 --ci=bootstrap --stationary-only
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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |
| `--cumulative` | | Flag | | Compute cumulative IRFs (for differenced data) |
| `--identified-set` | | Flag | | Return full identified set (sign restrictions only) |
| `--stationary-only` | | Flag | | Filter non-stationary bootstrap draws |

### Identification Methods

| ID String | Method | Config Required |
|-----------|--------|----------------|
| `cholesky` | Cholesky decomposition (recursive) | No |
| `sign` | Sign restrictions | Yes (sign matrix) |
| `narrative` | Narrative sign restrictions | Yes (narrative block) |
| `longrun` | Long-run (Blanchard-Quah) | No |
| `arias` | Arias et al. zero + sign | Yes (restrictions) |
| `uhlig` | Uhlig (Mountford & Uhlig 2009) penalty-based | Yes (restrictions + uhlig params) |
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
friedman irf bvar data.csv --shock=1 --cumulative
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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |
| `--cumulative` | | Flag | | Compute cumulative IRFs (for differenced data) |

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

# Cumulative IRFs
friedman irf lp data.csv --id=cholesky --shock=1 --cumulative
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
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |
| `--cumulative` | | Flag | | Compute cumulative IRFs (for differenced data) |

## irf vecm

IRFs for Vector Error Correction Models. The VECM is converted to its VAR representation and then IRFs are computed.

```bash
friedman irf vecm data.csv --shock=1 --horizons=20
friedman irf vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman irf vecm data.csv --id=cholesky --ci=bootstrap --replications=500
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 2 | Lag order (in levels) |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--rank` | `-r` | String | `auto` | Cointegration rank (auto via Johansen, or explicit) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--id` | | String | `cholesky` | Identification method |
| `--ci` | | String | `bootstrap` | `none`, `bootstrap`, `theoretical` |
| `--replications` | | Int | 1000 | Bootstrap replications |
| `--config` | | String | | TOML config for identification restrictions |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** IRFs per variable with confidence bands.

## irf pvar

Panel VAR impulse response functions. Supports orthogonalized (OIRF) and generalized (GIRF) impulse responses.

```bash
friedman irf pvar data.csv --id-col=country --time-col=year --horizons=20
friedman irf pvar data.csv --irf-type=girf --horizons=12
friedman irf pvar data.csv --ci=bootstrap --replications=500
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto | Lag order |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--id-col` | | String | | Panel group identifier column |
| `--time-col` | | String | | Panel time identifier column |
| `--irf-type` | | String | `oirf` | `oirf` (orthogonalized), `girf` (generalized) |
| `--ci` | | String | `bootstrap` | `none`, `bootstrap` |
| `--replications` | | Int | 500 | Bootstrap replications |
| `--conf-level` | | Float64 | 0.95 | Confidence level |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** IRFs per variable with bootstrap confidence bands.
