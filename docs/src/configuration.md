# Configuration

Friedman uses TOML configuration files for complex model specifications. Pass them via the `--config` option.

## Minnesota Prior

Used by `estimate bvar`, `irf bvar`, `fevd bvar`, `hd bvar`, `forecast bvar`.

```toml
[prior]
type = "minnesota"

[prior.hyperparameters]
lambda1 = 0.2       # tau (overall tightness)
lambda2 = 0.5       # cross-variable shrinkage
lambda3 = 1.0       # lag decay
lambda4 = 100000.0  # constant term variance

[prior.optimization]
enabled = true       # auto-optimize hyperparameters via grid search
```

### Hyperparameters

| Parameter | Field | Description | Typical Range |
|-----------|-------|-------------|---------------|
| `lambda1` | `tau` | Overall tightness of the prior | 0.01 -- 1.0 |
| `lambda2` | `lambda` | Cross-variable shrinkage (how much other variables' lags matter) | 0.1 -- 1.0 |
| `lambda3` | `decay` | Lag decay rate (higher = faster decay of lag importance) | 0.5 -- 2.0 |
| `lambda4` | `omega` | Constant term variance (large = uninformative) | 10000 -- 1e6 |

When `optimization.enabled = true`, Friedman ignores the manual hyperparameters and uses `optimize_hyperparameters()` to find optimal values via grid search over tau.

When optimization is disabled, AR(1) residual standard deviations are estimated from the data and used for the `omega` parameter (per-variable scale).

## Sign Restrictions

Used by `irf var`, `irf bvar`, `irf lp`, `fevd`, `hd` with `--id=sign`.

```toml
[identification]
method = "sign"

[identification.sign_matrix]
# Each row = variable, each column = shock
# 1 = positive, -1 = negative, 0 = unrestricted
matrix = [
  [1, -1, 1],
  [0, 1, -1],
  [0, 0, 1]
]
horizons = [0, 1, 2, 3]  # horizons at which restrictions apply (0-based)
```

The sign matrix dimensions must match the number of variables. Restrictions are checked at each specified horizon.

## Narrative Restrictions

Used with `--id=narrative`. Can be combined with sign restrictions.

```toml
[identification]
method = "narrative"

[identification.sign_matrix]
matrix = [[1, -1], [0, 1]]
horizons = [0]

[identification.narrative]
shock_index = 1              # which shock to constrain (1-based)
periods = [10, 15, 20]      # time periods where restrictions apply
signs = [1, -1, 1]          # required sign at each period
```

## Arias Identification

Arias et al. (2018) zero and sign restrictions. Used with `--id=arias`.

```toml
# Zero restrictions: variable-shock-horizon triples forced to zero
[[identification.zero_restrictions]]
var = 1
shock = 1
horizon = 0

[[identification.zero_restrictions]]
var = 1
shock = 2
horizon = 0

# Sign restrictions: variable-shock pairs with sign constraints
[[identification.sign_restrictions]]
var = 2
shock = 1
sign = "positive"
horizon = 0

[[identification.sign_restrictions]]
var = 3
shock = 2
sign = "negative"
horizon = 0
```

Multiple `[[identification.zero_restrictions]]` and `[[identification.sign_restrictions]]` blocks can be specified (TOML array of tables syntax).

## Non-Gaussian SVAR

Used by `test heteroskedasticity` with `--method=smooth_transition` or `--method=external`.

```toml
[nongaussian]
method = "smooth_transition"
transition_variable = "spread"   # column name in data CSV
n_regimes = 2
```

For external volatility:

```toml
[nongaussian]
method = "external"
regime_variable = "nber"         # column name for regime indicator
n_regimes = 2
```

## GMM Specification

Used by `estimate gmm`.

```toml
[gmm]
moment_conditions = ["output", "inflation"]
instruments = ["lag_output", "lag_inflation"]
weighting = "twostep"
```

| Field | Description |
|-------|-------------|
| `moment_conditions` | Column names used as moment condition variables |
| `instruments` | Column names used as instruments |
| `weighting` | Weighting matrix method (overridden by `--weighting` flag) |

## Output Formats

All commands support three output formats:

### Table (default)

Terminal-formatted table using PrettyTables with center-aligned columns.

```bash
friedman estimate var data.csv
```

### CSV

Standard CSV output, either to stdout or file.

```bash
friedman estimate var data.csv --format=csv
friedman estimate var data.csv --format=csv --output=results.csv
```

### JSON

Array of row dictionaries.

```bash
friedman estimate var data.csv --format=json
friedman estimate var data.csv --format=json --output=results.json
```

Example JSON output:

```json
[
  {"equation": "gdp", "inflation_L1": 0.123, "gdp_L1": 0.456},
  {"equation": "inflation", "inflation_L1": 0.789, "gdp_L1": 0.012}
]
```
