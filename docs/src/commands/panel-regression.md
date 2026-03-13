# Panel Regression

Panel regression commands are available under `estimate`, `predict`, `residuals`, and `test`.

## Estimation

| Command | Description |
|---------|-------------|
| `estimate preg` | Panel regression (FE/RE/BE/pooled, with twoway option) |
| `estimate piv` | Panel IV (2SLS) regression |
| `estimate plogit` | Panel logit (pooled/RE) |
| `estimate pprobit` | Panel probit (pooled/RE) |

## Diagnostics

| Command | Description |
|---------|-------------|
| `predict preg/piv/plogit/pprobit` | In-sample fitted values |
| `residuals preg/piv/plogit/pprobit` | Model residuals |

## Specification Tests

| Command | Description |
|---------|-------------|
| `test hausman` | Hausman specification test (FE vs RE) |
| `test breusch-pagan` | Breusch-Pagan LM test for random effects |
| `test f-fe` | F-test for individual fixed effects |
| `test pesaran-cd` | Pesaran CD test for cross-sectional dependence |
| `test wooldridge-ar` | Wooldridge test for serial correlation |
| `test modified-wald` | Modified Wald test for groupwise heteroskedasticity |

## Usage

```bash
# Fixed effects regression
friedman estimate preg panel.csv --dep gdp --indep investment,trade --method fe

# Hausman test
friedman test hausman panel.csv --dep gdp --indep investment,trade

# Panel IV
friedman estimate piv panel.csv --dep gdp --exog trade --endog investment --instruments lag_inv
```
