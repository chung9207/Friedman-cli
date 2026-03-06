# Friedman-cli v0.3.3 — MEMs v0.3.4 Integration Design

## Scope

Friedman-cli v0.3.3 wraps MacroEconometricModels.jl v0.3.4. Full parity with all 4 new feature areas plus bug fixes (automatic via version bump).

| Area | New Leaves | Modified |
|------|-----------|----------|
| estimate | +4 (reg, iv, logit, probit) | — |
| test | +7 (fourier-adf, fourier-kpss, dfgls, lm-unitroot, adf-2break, gregory-hansen, vif) | — |
| predict | +3 (reg, logit, probit) | — |
| residuals | +3 (reg, logit, probit) | — |
| dsge bayes | +6 (leaf → node with 7 leaves) | — |
| did | — | lp-did (replace with LPDiDResult API), estimate (add --base-period) |
| data | — | list/load (add mpdta, ddcg datasets) |

**Total: ~23 new leaves → ~164 subcommands**

---

## 1. Estimate Commands (+4 leaves)

```
estimate reg    <data>  [--dep=col] [--cov-type=hc1] [--weights=col] [--clusters=col]
estimate iv     <data>  --dep=col --endogenous=col1,col2 --instruments=z1,z2,z3 [--cov-type=hc1]
estimate logit  <data>  [--dep=col] [--cov-type=ols] [--clusters=col] [--maxiter=100] [--tol=1e-8]
estimate probit <data>  [--dep=col] [--cov-type=ols] [--clusters=col] [--maxiter=100] [--tol=1e-8]
```

- `--dep` specifies dependent variable column; remaining numeric columns → X matrix
- `--instruments` comma-separated column names from same data file (no separate CSV)
- `--endogenous` comma-separated column names of endogenous regressors
- `--cov-type`: ols, hc0, hc1, hc2, hc3, cluster
- Output: coefficient table (beta, SE, t-stat, p-value, CI) + fit stats. IV adds first-stage F and Sargan test.

---

## 2. Test Commands (+7 leaves)

```
test fourier-adf    <data> [--regression=constant] [--fmax=3] [--lags=aic] [--max-lags=] [--trim=0.15]
test fourier-kpss   <data> [--regression=constant] [--fmax=3] [--bandwidth=]
test dfgls          <data> [--regression=constant] [--lags=aic] [--max-lags=]
test lm-unitroot    <data> [--breaks=0] [--regression=level] [--lags=aic] [--max-lags=] [--trim=0.15]
test adf-2break     <data> [--model=level] [--lags=aic] [--max-lags=] [--trim=0.10]
test gregory-hansen <data> [--model=C] [--lags=aic] [--max-lags=] [--trim=0.15]
test vif            <data> --dep=col [--cov-type=hc1]
```

- Unit root tests: single-variable (first numeric column), same pattern as adf/kpss/pp/za/np
- gregory-hansen: multivariate (matrix), like johansen
- test vif: re-estimates OLS internally, outputs VIF per regressor
- Output: test statistic, p-value, critical values, break dates (where applicable)

---

## 3. Predict & Residuals (+3+3 leaves)

```
predict reg    <data> --dep=col [--cov-type=hc1] [--weights=col] [--clusters=col]
predict logit  <data> --dep=col [--cov-type=ols] [--clusters=col] [--marginal-effects] [--odds-ratio] [--classification-table] [--threshold=0.5]
predict probit <data> --dep=col [--cov-type=ols] [--clusters=col] [--marginal-effects] [--classification-table] [--threshold=0.5]

residuals reg    <data> --dep=col [--cov-type=hc1] [--weights=col] [--clusters=col]
residuals logit  <data> --dep=col [--cov-type=ols] [--clusters=col]
residuals probit <data> --dep=col [--cov-type=ols] [--clusters=col]
```

- Re-estimate internally, output fitted values / residuals
- `--marginal-effects` outputs AME table instead of fitted probabilities
- `--odds-ratio` outputs odds ratio table (logit only)
- `--classification-table` + `--threshold=0.5` outputs confusion matrix
- Flags are mutually exclusive with default fitted-values output

---

## 4. DSGE Bayes Node (leaf → NodeCommand, 7 leaves)

```
dsge bayes estimate   <model> --data=file --params=p1,p2 --priors=file [--method=smc] [--n-draws=10000] [--burnin=5000] [--n-particles=500] [--solver=gensys]
dsge bayes irf        <model> --data=file --params=p1,p2 --priors=file [--horizon=40] [--n-draws=200] [--solver=gensys] [--method=smc] [--plot] [--plot-save=]
dsge bayes fevd       <model> --data=file --params=p1,p2 --priors=file [--horizon=40] [--n-draws=200] [--solver=gensys] [--method=smc] [--plot] [--plot-save=]
dsge bayes simulate   <model> --data=file --params=p1,p2 --priors=file [--periods=100] [--n-draws=200] [--solver=gensys] [--method=smc] [--plot] [--plot-save=]
dsge bayes summary    <model> --data=file --params=p1,p2 --priors=file [--method=smc] [--n-draws=10000]
dsge bayes compare    <model> --data=file --params=p1,p2 --priors=file --model2=file --params2=p1,p2 --priors2=file [--method=smc]
dsge bayes predictive <model> --data=file --params=p1,p2 --priors=file [--n-sim=100] [--periods=100] [--method=smc]
```

- All leaves re-estimate internally (estimate → call post-estimation function)
- `--method`: smc, rwmh, csmc, smc2, importance
- `irf`/`fevd`/`simulate` support `--plot`/`--plot-save`
- `summary` outputs posterior table (mean, median, std, 68%/90% CI)
- `compare` outputs Bayes factor + marginal likelihoods for two models
- Common options via `_DSGE_BAYES_OPTIONS` const pattern

---

## 5. DID Enhancements

### did lp-did — replace with LPDiDResult API

```
did lp-did <data> --outcome=col --treatment=col [--horizon=5]
    [--pre-window=3] [--post-window=H] [--ylags=0] [--dylags=0]
    [--covariates=c1,c2] [--cluster=unit]
    [--pmd=] [--reweight] [--nocomp]
    [--nonabsorbing=] [--notyet] [--nevertreated] [--firsttreat] [--oneoff]
    [--only-pooled] [--only-event]
    [--id-col=] [--time-col=] [--conf-level=0.95]
    [--plot] [--plot-save=]
```

New options: `--pmd`, `--reweight`, `--nocomp`, `--nonabsorbing`, `--notyet`, `--nevertreated`, `--firsttreat`, `--oneoff`, `--ylags`, `--dylags`, `--pre-window`, `--post-window`, `--only-pooled`, `--only-event`.
Output: event-time coefficient table + optional pooled pre/post effects.
Returns `LPDiDResult` internally.

### did estimate — add --base-period

```
did estimate <data> ... [--base-period=varying]
```

Only applies when `--method=cs`. Values: varying (default), universal.

### data list/load — add datasets

- `:mpdta` — Callaway & Sant'Anna (2021) minimum wage panel
- `:ddcg` — Acemoglu et al. democracy-GDP panel

---

## 6. File Changes

| File | Changes |
|------|---------|
| `Project.toml` | version → 0.3.3, MEMs compat → 0.3.4 |
| `src/Friedman.jl` | FRIEDMAN_VERSION → 0.3.3 |
| `src/commands/estimate.jl` | +4 leaves, +4 handlers |
| `src/commands/test.jl` | +7 leaves, +7 handlers |
| `src/commands/predict.jl` | +3 leaves, +3 handlers |
| `src/commands/residuals.jl` | +3 leaves, +3 handlers |
| `src/commands/dsge.jl` | bayes leaf → NodeCommand (7 leaves), +7 handlers |
| `src/commands/did.jl` | replace lp-did handler, add --base-period to estimate |
| `src/commands/data.jl` | add mpdta/ddcg to list + load |
| `src/commands/shared.jl` | add _REG_OPTIONS, _parse_dep_var(), _DSGE_BAYES_OPTIONS |
| `test/mocks.jl` | +mock types for all new MEMs types |
| `test/test_commands.jl` | +handler tests for all 23 new leaves |
| `test/runtests.jl` | version refs → 0.3.3, structure counts |

No new source files.
