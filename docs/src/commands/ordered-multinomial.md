# Ordered & Multinomial Choice Models

## Estimation

| Command | Description |
|---------|-------------|
| `estimate ologit` | Ordered logit regression |
| `estimate oprobit` | Ordered probit regression |
| `estimate mlogit` | Multinomial logit regression |

## Diagnostics

| Command | Description |
|---------|-------------|
| `predict ologit/oprobit/mlogit` | Predicted probabilities |
| `residuals ologit/oprobit/mlogit` | Model residuals |

## Tests

| Command | Description |
|---------|-------------|
| `test brant` | Brant test for parallel regression assumption (ordered models) |
| `test hausman-iia` | Hausman-McFadden IIA test (multinomial logit) |

## Usage

```bash
# Ordered logit
friedman estimate ologit data.csv --dep satisfaction

# Brant test
friedman test brant data.csv --dep satisfaction

# Multinomial logit
friedman estimate mlogit data.csv --dep choice

# Hausman IIA test
friedman test hausman-iia data.csv --dep choice --omit-category 3
```
