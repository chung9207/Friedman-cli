# Spectral Analysis

The `spectral` command provides frequency-domain analysis tools.

## Subcommands

| Command | Description |
|---------|-------------|
| `spectral acf` | Autocorrelation and partial autocorrelation function |
| `spectral periodogram` | Compute periodogram |
| `spectral density` | Spectral density estimation (Welch/Bartlett/Daniell) |
| `spectral cross` | Cross-spectrum and coherence between two series |
| `spectral transfer` | Transfer function analysis for filters |

## Usage

```bash
# ACF with 20 lags
friedman spectral acf data.csv --max-lag 20

# Spectral density (Welch method)
friedman spectral density data.csv --method welch

# Cross-spectrum between variables 1 and 2
friedman spectral cross data.csv --var1 1 --var2 2

# Transfer function for HP filter
friedman spectral transfer --filter hp --lambda 1600 --nobs 200
```
