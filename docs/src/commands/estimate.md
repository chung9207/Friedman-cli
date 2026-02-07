# estimate

Estimate econometric models. 15 subcommands covering VAR, BVAR, local projections, ARIMA, GMM, factor models, volatility models, and non-Gaussian SVAR identification.

All estimation commands auto-save results with tags (e.g., `var001`, `bvar001`).

## estimate var

Estimate a VAR(p) model via OLS. Lag order is auto-selected via AIC when `--lags` is omitted.

```bash
friedman estimate var data.csv
friedman estimate var data.csv --lags=2
friedman estimate var data.csv --lags=4 --format=csv --output=var_results.csv
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | Lag order |
| `--trend` | | String | `constant` | `none`, `constant`, `trend`, `both` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficient matrix, AIC/BIC/HQC/log-likelihood.

## estimate bvar

Estimate a Bayesian VAR with MCMC sampling and posterior extraction.

```bash
friedman estimate bvar data.csv --lags=4 --draws=2000
friedman estimate bvar data.csv --config=prior.toml --method=median
friedman estimate bvar data.csv --sampler=gibbs --draws=5000
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | 4 | Lag order |
| `--prior` | | String | `minnesota` | Prior type |
| `--draws` | `-n` | Int | 2000 | MCMC draws |
| `--sampler` | | String | `direct` | `direct`, `gibbs` |
| `--method` | | String | `mean` | `mean`, `median` (posterior extraction) |
| `--config` | | String | | TOML config for prior hyperparameters |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Posterior mean/median coefficient matrix, AIC/BIC/HQC.

See [Configuration](../configuration.md) for Minnesota prior TOML format.

## estimate lp

Estimate local projections with 6 method variants.

### Standard LP (Jorda 2005)

```bash
friedman estimate lp data.csv --shock=1 --horizons=20 --vcov=newey_west
```

### LP-IV (Stock & Watson 2018)

```bash
friedman estimate lp data.csv --method=iv --shock=1 --instruments=instruments.csv
```

### Smooth LP (Barnichon & Brownlees 2019)

```bash
friedman estimate lp data.csv --method=smooth --shock=1 --horizons=20
friedman estimate lp data.csv --method=smooth --lambda=0.5 --knots=4
```

When `--lambda=0` (default), the smoothing parameter is auto-selected via cross-validation.

### State-Dependent LP (Auerbach & Gorodnichenko 2013)

```bash
friedman estimate lp data.csv --method=state --shock=1 --state-var=2 --gamma=1.5
```

### Propensity Score LP (Angrist et al. 2018)

```bash
friedman estimate lp data.csv --method=propensity --treatment=1 --score-method=logit
```

### Doubly Robust LP

```bash
friedman estimate lp data.csv --method=robust --treatment=1 --score-method=logit
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `standard` | `standard`, `iv`, `smooth`, `state`, `propensity`, `robust` |
| `--shock` | | Int | 1 | Shock variable index (1-based) |
| `--horizons` | `-h` | Int | 20 | IRF horizon |
| `--control-lags` | | Int | 4 | Number of control lags |
| `--vcov` | | String | `newey_west` | `newey_west`, `white`, `driscoll_kraay` |
| `--instruments` | | String | | Path to instruments CSV (iv only) |
| `--knots` | | Int | 3 | B-spline knots (smooth only) |
| `--lambda` | | Float64 | 0.0 | Smoothing penalty, 0=auto CV (smooth only) |
| `--state-var` | | Int | | State variable index (state only, required) |
| `--gamma` | | Float64 | 1.5 | Transition steepness (state only) |
| `--transition` | | String | `logistic` | `logistic`, `exponential`, `indicator` (state only) |
| `--treatment` | | Int | 1 | Treatment variable index (propensity/robust only) |
| `--score-method` | | String | `logit` | `logit`, `probit` (propensity/robust only) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## estimate arima

Estimate ARIMA(p,d,q) models. Auto-selects order via information criteria when `--p` is omitted.

```bash
# Auto-selection
friedman estimate arima data.csv --criterion=bic

# Explicit order
friedman estimate arima data.csv --p=1 --d=1 --q=1

# Specific column
friedman estimate arima data.csv --column=2 --p=2 --d=0 --q=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | auto | AR order |
| `--d` | | Int | 0 | Differencing order |
| `--q` | | Int | 0 | MA order |
| `--max-p` | | Int | 5 | Max AR order for auto selection |
| `--max-d` | | Int | 2 | Max differencing order for auto selection |
| `--max-q` | | Int | 5 | Max MA order for auto selection |
| `--criterion` | | String | `bic` | `aic`, `bic` |
| `--method` | `-m` | String | `css_mle` | `ols`, `css`, `mle`, `css_mle` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** AR/MA coefficients, AIC/BIC/log-likelihood.

## estimate gmm

Estimate a GMM model. Requires a TOML config specifying moment conditions and instruments.

```bash
friedman estimate gmm data.csv --config=gmm_spec.toml --weighting=twostep
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | | String | (required) | TOML config file |
| `--weighting` | `-w` | String | `twostep` | `identity`, `optimal`, `twostep`, `iterated` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Parameter estimates, J-test for overidentification.

See [Configuration](../configuration.md) for GMM TOML format.

## estimate static

Estimate a static factor model via PCA. Factor count is auto-selected via Bai-Ng information criteria when `--nfactors` is omitted.

```bash
friedman estimate static data.csv
friedman estimate static data.csv --nfactors=3 --criterion=ic2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto (IC) | Number of factors |
| `--criterion` | | String | `ic1` | `ic1`, `ic2`, `ic3` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Scree data (eigenvalues, variance shares), factor loadings.

## estimate dynamic

Estimate a dynamic factor model with a factor VAR.

```bash
friedman estimate dynamic data.csv --nfactors=2 --factor-lags=1
friedman estimate dynamic data.csv --method=em
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of factors |
| `--factor-lags` | `-p` | Int | 1 | Factor VAR lag order |
| `--method` | | String | `twostep` | `twostep`, `em` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Factor loadings, companion matrix eigenvalues, stationarity check.

## estimate gdfm

Estimate a generalized dynamic factor model (spectral method).

```bash
friedman estimate gdfm data.csv --dynamic-rank=2
friedman estimate gdfm data.csv --nfactors=5 --dynamic-rank=3
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--nfactors` | `-r` | Int | auto | Number of static factors |
| `--dynamic-rank` | `-q` | Int | auto | Dynamic rank |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Common variance shares per variable, average common variance share.

## estimate arch

Estimate an ARCH(q) volatility model.

```bash
friedman estimate arch data.csv --column=1 --q=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--q` | | Int | 1 | ARCH order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficients (mu, omega, alpha), persistence, unconditional variance.

## estimate garch

Estimate a GARCH(p,q) volatility model.

```bash
friedman estimate garch data.csv --column=1 --p=1 --q=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | 1 | GARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficients (mu, omega, alpha, beta), persistence, half-life, unconditional variance.

## estimate egarch

Estimate an EGARCH(p,q) volatility model.

```bash
friedman estimate egarch data.csv --column=1 --p=1 --q=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | 1 | EGARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficients (mu, omega, alpha, gamma, beta), persistence.

## estimate gjr\_garch

Estimate a GJR-GARCH(p,q) volatility model with asymmetric leverage effects.

```bash
friedman estimate gjr_garch data.csv --column=1 --p=1 --q=1
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--p` | | Int | 1 | GARCH order |
| `--q` | | Int | 1 | ARCH order |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficients (mu, omega, alpha, gamma, beta), persistence, half-life.

## estimate sv

Estimate a Stochastic Volatility model via MCMC.

```bash
friedman estimate sv data.csv --column=1 --draws=5000
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--column` | `-c` | Int | 1 | Column index (1-based) |
| `--draws` | `-n` | Int | 5000 | MCMC draws |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Coefficients (mu, phi, sigma_eta), persistence (phi).

## estimate fastica

ICA-based non-Gaussian SVAR identification. Supports 5 ICA methods.

```bash
friedman estimate fastica data.csv --method=fastica --contrast=logcosh
friedman estimate fastica data.csv --method=jade
friedman estimate fastica data.csv --method=sobi
friedman estimate fastica data.csv --method=dcov
friedman estimate fastica data.csv --method=hsic
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | VAR lag order |
| `--method` | | String | `fastica` | `fastica`, `jade`, `sobi`, `dcov`, `hsic` |
| `--contrast` | | String | `logcosh` | `logcosh`, `exp`, `kurtosis` (FastICA only) |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Structural impact matrix (B0), structural shocks (first 10 observations).

## estimate ml

Maximum likelihood non-Gaussian SVAR identification.

```bash
friedman estimate ml data.csv --distribution=student_t
friedman estimate ml data.csv --distribution=mixture_normal
friedman estimate ml data.csv --distribution=pml
friedman estimate ml data.csv --distribution=skew_normal
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--lags` | `-p` | Int | auto (AIC) | VAR lag order |
| `--distribution` | `-d` | String | `student_t` | `student_t`, `skew_t`, `ghd`, `mixture_normal`, `pml`, `skew_normal` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Structural impact matrix (B0), model fit (log-likelihood, AIC, BIC), distribution parameters, parameter estimates with standard errors.
