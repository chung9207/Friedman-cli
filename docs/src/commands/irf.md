# irf

Compute impulse response functions. 7 subcommands: `var`, `bvar`, `lp`, `vecm`, `pvar`, `favar`, `sdfm`.

## Output format (C051)

`irf var`, `bvar`, `vecm`, `lp`, `favar`, and `sdfm` render through MEMs' tidy
`long_table(result)`: one row per `(horizon, variable, shock)` cell, columns
`horizon | variable | shock | value | lower | upper` (`lower`/`upper` are `missing` when
the result carries no uncertainty band, e.g. `--ci=none`). `var`/`bvar`/`vecm` filter the
tidy rows to the single selected `--shock`; `lp` filters to every shock named by
`--shock`/`--shocks` (one table covers all of them); `favar`/`sdfm` take no shock filter
and return every shock in one table. `irf pvar` and the Arias/Uhlig/sign-restriction paths
build their arrays by hand (no MEMs result type to route through `long_table`) and stay on
the older wide, per-shock-file layout.

## irf var

Frequentist IRFs with multiple identification schemes and confidence intervals.

```bash
# From a saved model stem (Wave 2); persist the IRF and re-render later
friedman irf var --model var --horizons=20 --save-result irf
friedman irf var --result irf
friedman show irf

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

# Narrative ADRR (Arias pipeline + narrative contribution restrictions)
friedman irf var data.csv --id=narrative-adrr --config=adrr_restrictions.toml

# Non-Gaussian identification methods
friedman irf var data.csv --id=fastica
friedman irf var data.csv --id=jade

# With bootstrap confidence intervals
friedman irf var data.csv --shock=1 --ci=bootstrap --replications=1000

# Wild bootstrap (heteroskedastic residuals) / moving-block (serial dependence)
friedman irf var data.csv --ci=bootstrap --bootstrap=wild --wild-dist=mammen
friedman irf var data.csv --ci=bootstrap --bootstrap=block --block-length=12

# Kilian (1998) bias-corrected bands
friedman irf var data.csv --ci=bootstrap --bias-correct --bias-reps=250

# Cumulative IRFs (for differenced data)
friedman irf var data.csv --shock=1 --cumulative

# Full identified set for sign restrictions
friedman irf var data.csv --id=sign --config=sign.toml --identified-set

# Set-identified summaries (Fry-Pagan median target, modal model, joint bands)
friedman irf var data.csv --id=sign --config=sign.toml --identified-set --summary=median-target
friedman irf var data.csv --id=sign --config=sign.toml --identified-set --summary=joint-band

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
| `--summary` | | String | `none` | Set-identified summary: `median-target`, `modal-model`, `joint-band`, `sup-t-band` (only with `--identified-set`) |
| `--stationary-only` | | Flag | | Filter non-stationary bootstrap draws |

### Identification Methods

| ID String | Method | Config Required |
|-----------|--------|----------------|
| `cholesky` | Cholesky decomposition (recursive) | No |
| `sign` | Sign restrictions | Yes (sign matrix) |
| `narrative` | Narrative sign restrictions | Yes (narrative block) |
| `longrun` | Long-run (Blanchard-Quah) | No |
| `arias` | Arias et al. zero + sign | Yes (restrictions) |
| `narrative-adrr` | Arias pipeline + Antolín-Díaz/Rubio-Ramírez narrative contributions | Yes (restrictions + narrative_contributions) |
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

**Output:** Tidy table (`horizon|variable|shock|value|lower|upper`) filtered to `--shock` (the Arias/Uhlig/`--identified-set` paths stay wide — see [Output format](#output-format-c051) above).

Sign/narrative IRFs report the identified-set median with set-robust bands by default (MEMs 0.9.2); `--summary` selects an alternative set summary (Fry–Pagan median target, modal model, joint or sup-t bands) on the `--identified-set` path.

### Bootstrap schemes and bias correction

These apply under `--ci bootstrap` only; with any other `--ci` they are ignored and the CLI
says so on stderr rather than letting the flag look effective.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--bootstrap` | String | `iid` | `iid`, `wild`, `block` |
| `--wild-dist` | String | `rademacher` | Multiplier for `--bootstrap wild`: `rademacher`, `mammen` |
| `--block-length` | Int | 0 | Block length for `--bootstrap block` (0 = library default) |
| `--bias-correct` | Flag | | Kilian (1998) bias-corrected bands |
| `--bias-reps` | Int | 0 | Inner reps for `--bias-correct` (0 = same as `--replications`) |

Which scheme to use follows from what the residuals violate. `iid` resamples them
independently and assumes both homoskedasticity and no serial dependence. `wild` multiplies
each residual by a mean-zero draw, preserving conditional heteroskedasticity — the usual
choice for macro data. `block` resamples contiguous blocks, preserving short-range
dependence the VAR has not captured.

!!! warning "`--bias-correct` is bootstrap-after-bootstrap"
    OLS VAR coefficients are biased toward stationarity in small samples, which biases the
    IRFs. Kilian's correction estimates that bias by an *inner* bootstrap, re-centres the
    DGP, and corrects every outer draw.

    The cost is **multiplicative, not additive**: the inner loop runs `--bias-reps` (default:
    `--replications`) re-estimations *per outer draw*. With the defaults that is 1000 × 1000
    VAR fits. Set `--bias-reps` well below `--replications` — a few hundred is normally
    enough to pin the bias down — before running this on anything large.

### Arias identification diagnostics

`--id arias` now reports an importance-sampling diagnostics table alongside the IRF:

| Field | Meaning |
|---|---|
| `acceptance_rate` | Fraction of candidate draws satisfying the restrictions |
| `n_draws` | Nominal draw count |
| `ess` | Kish's effective sample size of the importance weights |
| `ess_fraction` | `ess / n_draws` |

Read `ess_fraction`, not `n_draws`. Under pure sign restrictions the weights are uniform and
the fraction is 1. **With zero restrictions it can be far below 1**, meaning a handful of
draws carry most of the posterior mass and the summary rests on far fewer effective draws
than the nominal count suggests. Below 0.1 the CLI warns on stderr.

This matters from CLI v0.9.1 specifically: the importance weights were inoperative before
MEMs 0.7.2, so Arias results changed at the bump and the diagnostic is what tells you
whether to trust the new ones.

## irf bvar

Bayesian IRFs with 68% credible intervals (16th/50th/84th percentiles).

```bash
friedman irf bvar data.csv --shock=1 --horizons=20
friedman irf bvar data.csv --draws=5000 --sampler=gibbs --config=prior.toml
friedman irf bvar data.csv --id=sign --config=sign_restrictions.toml
friedman irf bvar data.csv --id=robust-bayes --config=bvar_restrictions.toml
friedman irf bvar data.csv --shock=1 --cumulative
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--id` | | String | `cholesky` | `cholesky`, `sign`, `narrative`, `longrun`, `robust-bayes` (Giacomini–Kitagawa bands; config must carry both `[prior]` and `[identification]`) |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--config` | | String | | TOML config for identification/prior |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |
| `--cumulative` | | Flag | | Compute cumulative IRFs (for differenced data) |

**Output:** Tidy table (`horizon|variable|shock|value|lower|upper`), filtered to `--shock`; `value` = posterior mean, `lower`/`upper` = the 16th/84th percentile credible band.

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

**Output:** Tidy table (`horizon|variable|shock|value|lower|upper`) filtered to every shock named by `--shock`/`--shocks` — one table even with multiple shocks selected.

## irf vecm

IRFs for Vector Error Correction Models. The VECM is converted to its VAR representation and then IRFs are computed.

```bash
friedman irf vecm data.csv --shock=1 --horizons=20
friedman irf vecm data.csv --rank=2 --deterministic=constant --lags=4
friedman irf vecm data.csv --id=cholesky --ci=bootstrap --replications=500
friedman irf vecm data.csv --id=svec --ci=none --rank=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 2 | Lag order (in levels) |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--rank` | `-r` | String | `auto` | Cointegration rank (auto via Johansen, or explicit) |
| `--deterministic` | | String | `constant` | `none`, `constant`, `trend` |
| `--id` | | String | `cholesky` | Identification method (`cholesky`, `sign`, `narrative`, `longrun`, `svec`; `svec` requires `--ci none` and accepts an optional `[svec]` config) |
| `--ci` | | String | `bootstrap` | `none`, `bootstrap`, `theoretical` |
| `--replications` | | Int | 1000 | Bootstrap replications |
| `--config` | | String | | TOML config for identification restrictions |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Tidy table (`horizon|variable|shock|value|lower|upper`) filtered to `--shock` (VECM → VAR representation, same schema as `irf var`).

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

**Output:** Wide, per-shock-file table (columns = variables, rows = horizons) — `pvar_bootstrap_irf` returns a plain NamedTuple, not a MEMs result type, so this leaf stays outside the `long_table` conversion.

## See Also

For FAVAR and Structural DFM IRFs, see [favar & sdfm](favar.md). For DSGE model IRFs, see [dsge irf](dsge.md#dsge-irf).
