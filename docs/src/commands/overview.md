# CLI Reference Overview

Friedman-cli uses an **action-first** command hierarchy: commands are organized by action (`estimate`, `irf`, `forecast`, ...) rather than by model type.

## Command Tree

```
friedman
├── estimate     var | bvar | lp | arima | gmm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | fastica | ml | vecm | pvar
├── test         adf | kpss | pp | za | np | johansen | normality |
│                identifiability | heteroskedasticity | arch_lm | ljung_box |
│                granger | lr | lm | var (lagselect | stability) |
│                pvar (hansen_j | mmsc | lagselect | stability)
├── irf          var | bvar | lp | vecm | pvar
├── fevd         var | bvar | lp | vecm | pvar
├── hd           var | bvar | lp | vecm
├── forecast     var | bvar | lp | arima | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv | vecm
├── predict      var | bvar | arima | vecm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv
├── residuals    var | bvar | arima | vecm | static | dynamic | gdfm |
│                arch | garch | egarch | gjr_garch | sv
├── filter       hp | hamilton | bn | bk | bhp
├── data         list | load | describe | diagnose | fix | transform | filter | validate
├── list         models | results
├── rename       (renames stored tags)
└── project      list | show
```

**Total: 13 top-level commands, ~97 subcommands.**

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
friedman estimate --help         # list estimate subcommands
friedman estimate var --help     # detailed var estimation help
friedman irf var --help          # IRF options
```

## Tag Resolution

Friedman automatically stores models with auto-generated tags (`var001`, `bvar001`, ...). You can use these tags directly as subcommand arguments for post-estimation commands:

```bash
# These are equivalent:
friedman irf var001
friedman irf var data.csv --from-tag=var001
```

The tag resolution system (`resolve_stored_tags`) runs before dispatch and rewrites the arguments:
- `["irf", "var001"]` becomes `["irf", "var", "--from-tag=var001"]`

This works for `irf`, `fevd`, `hd`, `forecast`, `predict`, and `residuals` commands.

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

CLI option names use hyphens (`--from-tag`, `--control-lags`) which are automatically converted to underscores (`from_tag`, `control_lags`) when passed to handler functions.
