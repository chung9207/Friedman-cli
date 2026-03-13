# CLI Reference Overview

Friedman-cli uses an **action-first** command hierarchy: commands are organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

## Command Tree

```
friedman
├── estimate     var | bvar | lp | arima | gmm | smm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | fastica | ml | vecm | pvar |
│                favar | sdfm | reg | iv | logit | probit |
│                preg | piv | plogit | pprobit |
│                ologit | oprobit | mlogit
├── test         adf | kpss | pp | za | np | johansen | normality |
│                identifiability | heteroskedasticity | arch_lm | ljung_box |
│                granger | lr | lm | andrews | bai-perron |
│                panic | cips | moon-perron | factor-break |
│                fourier-adf | fourier-kpss | dfgls | lm-unitroot |
│                adf-2break | gregory-hansen | vif |
│                hausman | breusch-pagan | f-fe | pesaran-cd |
│                wooldridge-ar | modified-wald |
│                brant | hausman-iia |
│                var (lagselect | stability) |
│                pvar (hansen_j | mmsc | lagselect | stability)
├── irf          var | bvar | lp | vecm | pvar | favar | sdfm
├── fevd         var | bvar | lp | vecm | pvar | favar | sdfm
├── hd           var | bvar | lp | vecm | favar | dsge | dsge-bayes
├── forecast     var | bvar | lp | arima | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | vecm | favar
├── predict      var | bvar | arima | vecm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | favar | reg | logit | probit |
│                preg | piv | plogit | pprobit | ologit | oprobit | mlogit
├── residuals    var | bvar | arima | vecm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | favar | reg | logit | probit |
│                preg | piv | plogit | pprobit | ologit | oprobit | mlogit
├── filter       hp | hamilton | bn | bk | bhp
├── data         list | load | describe | diagnose | fix | transform | filter |
│                validate | balance | dropna | keeprows
├── nowcast      dfm | bvar | bridge | news | forecast
├── dsge         solve | irf | fevd | hd | simulate | estimate |
│                perfect-foresight | steady-state |
│                bayes (estimate | irf | fevd | hd | simulate | summary | compare | predictive)
├── did          estimate | event-study | lp-did |
│                test (bacon | pretrend | negweight | honest)
└── spectral     acf | periodogram | density | cross | transfer
```

**Total: 14 top-level commands, ~200 subcommands.**

Additionally, `friedman repl` launches an [interactive REPL session](@ref Interactive-REPL) with persistent data loading, result caching, and tab completion.

## Common Options

All commands that produce output support these options:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | Output format: `table`, `csv`, or `json` |
| `--output` | `-o` | String | (stdout) | Export results to a file path |

## Output Formats

```bash
# Terminal table (default) -- uses PrettyTables
friedman estimate var data.csv

# CSV export
friedman estimate var data.csv --format=csv --output=results.csv

# JSON export
friedman estimate var data.csv --format=json --output=results.json
```

## Help

Every command and subcommand supports `--help`:

```bash
friedman --help                  # top-level help
friedman --version               # show version
friedman --warranty              # display GPL warranty disclaimer
friedman --conditions            # display GPL distribution conditions
friedman estimate --help         # list estimate subcommands
friedman estimate var --help     # detailed var estimation help
friedman irf var --help          # IRF options
```

## Option Syntax

The parser supports several formats for passing options:

```bash
--lags=4           # long option with =
--lags 4           # long option with space
-p 4               # short alias
-p=4               # short alias with =
```

Flags (boolean options) are toggled by presence:

```bash
--verbose          # sets verbose=true
-v                 # short alias
```

Bundled short flags are supported:

```bash
-abc               # equivalent to -a -b -c
```

The `--` separator stops option parsing:

```bash
friedman estimate var -- --data-that-starts-with-dash.csv
```

## Hyphen-to-Underscore Mapping

CLI option names use hyphens (`--control-lags`, `--monthly-vars`) which are automatically converted to underscores (`control_lags`, `monthly_vars`) when passed to handler functions.
