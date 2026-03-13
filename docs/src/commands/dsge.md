# dsge

DSGE modeling from the terminal. 8 direct subcommands (`solve`, `irf`, `fevd`, `hd`, `simulate`, `estimate`, `perfect-foresight`, `steady-state`) plus a `bayes` node with 8 sub-leaves for the full Bayesian DSGE workflow.

Friedman supports DSGE models specified as TOML files or Julia scripts. See [Configuration](../configuration.md#dsge-model) for TOML format details.

## Model Input Formats

### TOML (`.toml`)

The model is defined in the `[model]` section with `endogenous`, `exogenous`, `parameters`, and `[[model.equations]]` entries. An optional `[solver]` section specifies the solution method.

```toml
[model]
endogenous = ["y", "c", "k", "n"]
exogenous = ["eps_a"]

[model.parameters]
alpha = 0.36
beta = 0.99
delta = 0.025
sigma = 1.0
phi_n = 1.0

[[model.equations]]
expr = "c^(-sigma) = beta * c(+1)^(-sigma) * (alpha * exp(eps_a(+1)) * k^(alpha-1) * n(+1)^(1-alpha) + 1 - delta)"

[[model.equations]]
expr = "phi_n * n^phi_n = c^(-sigma) * (1-alpha) * exp(eps_a) * k(-1)^alpha * n^(-alpha)"

[[model.equations]]
expr = "k = (1-delta)*k(-1) + y - c"

[[model.equations]]
expr = "y = exp(eps_a) * k(-1)^alpha * n^(1-alpha)"

[solver]
method = "gensys"
order = 1
```

### Julia Script (`.jl`)

The file must define a `model` variable of type `DSGESpec`:

```julia
using MacroEconometricModels
model = DSGESpec(...)
```

The CLI auto-detects the format by file extension.

## dsge solve

Solve a DSGE model. Supports 5 solution methods and OccBin occasionally binding constraints.

```bash
friedman dsge solve rbc.toml
friedman dsge solve rbc.toml --method=perturbation --order=2
friedman dsge solve rbc.toml --method=projection --degree=7 --grid=chebyshev
friedman dsge solve rbc.toml --constraints=occbin.toml --periods=60
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `gensys` | `gensys`, `klein`, `perturbation`, `projection`, `pfi` |
| `--order` | | Int | 1 | Perturbation order (1 or 2) |
| `--degree` | | Int | 5 | Polynomial degree (projection/pfi) |
| `--grid` | | String | `auto` | Grid type: `auto`, `chebyshev`, `smolyak` |
| `--constraints` | | String | | Path to OccBin constraints TOML |
| `--periods` | | Int | 40 | Number of periods for OccBin simulation |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output (standard):** Policy function matrices. Format depends on solution method — `DSGESolution` shows G1 policy matrix, `PerturbationSolution` shows gx control-state policy, `ProjectionSolution` shows coefficients with convergence diagnostics.

**Output (OccBin):** Piecewise-linear transition path for all endogenous variables.

See [Configuration](../configuration.md#occbin-constraints) for the OccBin constraints TOML format.

## dsge irf

Impulse response functions from a solved DSGE model.

```bash
friedman dsge irf rbc.toml --horizon=40
friedman dsge irf rbc.toml --shock-size=0.5 --n-sim=1000
friedman dsge irf rbc.toml --constraints=occbin.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `gensys` | Solution method |
| `--order` | | Int | 1 | Perturbation order |
| `--horizon` | `-h` | Int | 40 | IRF horizon |
| `--shock-size` | | Float64 | 1.0 | Shock size (std devs) |
| `--n-sim` | | Int | 0 | Simulation-based IRF draws (0 = analytical) |
| `--constraints` | | String | | Path to OccBin constraints TOML |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output (standard):** Per-shock IRF tables with columns for each endogenous variable.

**Output (OccBin):** Per-variable tables comparing linear vs piecewise-linear IRFs.

## dsge fevd

Forecast error variance decomposition from a solved DSGE model.

```bash
friedman dsge fevd rbc.toml --horizon=40
friedman dsge fevd rbc.toml --method=perturbation --order=2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `gensys` | Solution method |
| `--order` | | Int | 1 | Perturbation order |
| `--horizon` | `-h` | Int | 40 | FEVD horizon |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable FEVD proportions table (columns = shocks, rows = horizons).

## dsge hd

Historical decomposition from a solved DSGE model.

```bash
friedman dsge hd rbc.toml --horizon=40
friedman dsge hd rbc.toml --method=perturbation --order=2
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `gensys` | Solution method |
| `--order` | | Int | 1 | Perturbation order |
| `--horizon` | `-h` | Int | 40 | HD horizon |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Per-variable historical decomposition tables (columns = shocks, rows = time periods).

## dsge simulate

Simulate from a solved DSGE model.

```bash
friedman dsge simulate rbc.toml --periods=500 --burn=200
friedman dsge simulate rbc.toml --seed=42 --antithetic
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--method` | | String | `gensys` | Solution method |
| `--order` | | Int | 1 | Perturbation order |
| `--periods` | | Int | 200 | Simulation periods (after burn-in) |
| `--burn` | | Int | 100 | Burn-in periods to discard |
| `--seed` | | Int | 0 | Random seed (0 = no seed) |
| `--antithetic` | | Flag | | Use antithetic sampling for variance reduction |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

**Output:** Simulated data table with a column per endogenous variable, periods after burn-in.

## dsge estimate

Estimate DSGE model parameters from data. 4 estimation methods.

```bash
friedman dsge estimate rbc.toml --data=macro.csv --params=alpha,beta --method=irf_matching
friedman dsge estimate rbc.toml --data=macro.csv --params=alpha,beta --method=smm --sim-ratio=10
friedman dsge estimate rbc.toml --data=macro.csv --params=alpha,beta --method=likelihood
friedman dsge estimate rbc.toml --data=macro.csv --params=alpha,beta --bounds=bounds.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--data` | `-d` | String | (required) | Path to CSV data file |
| `--method` | | String | `irf_matching` | `irf_matching`, `likelihood`, `bayesian`, `smm` |
| `--params` | | String | (required) | Comma-separated parameter names to estimate |
| `--solve-method` | | String | `gensys` | DSGE solution method |
| `--solve-order` | | Int | 1 | Perturbation order for solution |
| `--weighting` | | String | `optimal` | `identity`, `optimal`, `diagonal` |
| `--irf-horizon` | | Int | 20 | IRF horizon for matching |
| `--var-lags` | | Int | 4 | VAR lags for empirical IRF |
| `--sim-ratio` | | Int | 5 | Simulation-to-data ratio (SMM) |
| `--bounds` | | String | | Path to parameter bounds TOML |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Parameter estimates with standard errors, t-statistics, and p-values. Includes J-statistic and convergence status.

## dsge perfect-foresight

Perfect foresight (deterministic) simulation for transition paths.

```bash
friedman dsge perfect-foresight rbc.toml --shocks=shocks.csv --periods=200
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--shocks` | | String | (required) | Path to shock sequence CSV |
| `--periods` | | Int | 100 | Simulation periods |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |
| `--plot` | | Flag | | Open interactive plot in browser |
| `--plot-save` | | String | | Save plot to HTML file |

The shock CSV must have columns matching the model's exogenous variables and rows for each shock period.

**Output:** Transition path for all endogenous variables, with convergence status.

## dsge steady-state

Compute the steady state of a DSGE model.

```bash
friedman dsge steady-state rbc.toml
friedman dsge steady-state rbc.toml --constraints=occbin.toml
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--constraints` | | String | | Path to OccBin constraints TOML |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Variable names and steady-state values.

## dsge bayes

Bayesian DSGE workflow. `bayes` is a **nested command group** with 8 sub-leaves: `estimate`, `irf`, `fevd`, `hd`, `simulate`, `summary`, `compare`, `predictive`. All share common options for model specification, data, parameters, and priors.

### Common Options (all dsge bayes sub-commands)

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--data` | `-d` | String | (required) | Path to CSV data file |
| `--params` | | String | (required) | Comma-separated parameter names to estimate |
| `--priors` | | String | (required) | Path to priors TOML file |
| `--method` | | String | `smc` | `smc`, `rwmh`, `csmc`, `smc2`, `importance` |
| `--n-draws` | | Int | 10000 | Posterior draws |
| `--burnin` | | Int | 5000 | Burn-in draws |
| `--n-particles` | | Int | 500 | Particle filter particles (smc2) |
| `--solver` | | String | `gensys` | `gensys`, `klein`, `perturbation` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### dsge bayes estimate

Bayesian DSGE posterior estimation.

```bash
friedman dsge bayes estimate rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml
friedman dsge bayes estimate rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --method=rwmh --n-draws=20000
```

**Output:** Posterior summary table (mean, median, std, CI per parameter) + log marginal likelihood.

### dsge bayes irf

Bayesian DSGE impulse responses with posterior uncertainty.

```bash
friedman dsge bayes irf rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --horizon=40 --n-draws=200
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--horizon` | Int | 40 | IRF horizon |
| `--plot` | Flag | | Open interactive plot |
| `--plot-save` | String | | Save plot to HTML |

### dsge bayes fevd

Bayesian DSGE forecast error variance decomposition.

```bash
friedman dsge bayes fevd rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --horizon=40
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--horizon` | Int | 40 | FEVD horizon |
| `--plot` | Flag | | Open interactive plot |
| `--plot-save` | String | | Save plot to HTML |

### dsge bayes hd

Bayesian DSGE historical decomposition with posterior uncertainty.

```bash
friedman dsge bayes hd rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --horizon=40
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--horizon` | Int | 40 | HD horizon |
| `--plot` | Flag | | Open interactive plot |
| `--plot-save` | String | | Save plot to HTML |

### dsge bayes simulate

Simulate from the posterior of a Bayesian DSGE.

```bash
friedman dsge bayes simulate rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --periods=200
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--periods` | Int | 100 | Simulation periods |
| `--plot` | Flag | | Open interactive plot |
| `--plot-save` | String | | Save plot to HTML |

### dsge bayes summary

Detailed posterior summary statistics.

```bash
friedman dsge bayes summary rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml
```

**Output:** Posterior table (mean, median, std, 68%/90% CI).

### dsge bayes compare

Compare two Bayesian DSGE models via Bayes factors and marginal likelihoods.

```bash
friedman dsge bayes compare model1.toml --data=macro.csv --params=rho,sigma --priors=priors.toml \
    --model2=model2.toml --params2=rho2,sigma2 --priors2=priors2.toml
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--model2` | String | (required) | Path to second model file |
| `--params2` | String | (required) | Parameters for second model |
| `--priors2` | String | (required) | Priors TOML for second model |

**Output:** Bayes factor, marginal likelihoods for both models, posterior odds.

### dsge bayes predictive

Posterior predictive checks.

```bash
friedman dsge bayes predictive rbc.toml --data=macro.csv --params=alpha,beta --priors=priors.toml --n-sim=100
```

| Additional Option | Type | Default | Description |
|-------------------|------|---------|-------------|
| `--n-sim` | Int | 100 | Predictive simulations |
| `--periods` | Int | 100 | Simulation periods |

**Output:** Predictive summary (mean, std) vs observed data moments.

### Priors TOML Format

```toml
[priors.alpha]
dist = "beta"
a = 2.0
b = 5.0

[priors.beta]
dist = "normal"
a = 0.99    # mean
b = 0.01    # std

[priors.sigma]
dist = "inverse_gamma"
a = 2.0
b = 0.1
```

Each parameter must have a `dist` key (distribution name) and shape parameters `a`, `b`.

## Solution Methods

| Method | `--method` value | When to use |
|--------|-----------------|-------------|
| Gensys (Sims 2002) | `gensys` | Default. Linear rational expectations models |
| Klein (2000) | `klein` | Alternative generalized Schur decomposition solver |
| Perturbation | `perturbation` | Higher-order approximations (order 1, 2, or 3) |
| Projection | `projection` | Global solutions, nonlinear models, accuracy matters |
| Policy Function Iteration | `pfi` | Global solutions, value function problems |

Projection and PFI methods support `--degree` (polynomial degree) and `--grid` (grid type) options.
