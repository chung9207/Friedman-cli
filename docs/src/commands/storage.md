# Storage & Projects

Commands for managing stored models/results and project registry.

## Storage System

Friedman automatically saves estimation and analysis results to a BSON file (`.friedmanlog.bson`) in the current working directory. Each entry is assigned an auto-generated tag:

```
var001, var002, ...       # VAR estimates
bvar001, bvar002, ...     # BVAR estimates
irf001, irf002, ...       # IRF results
forecast001, ...          # Forecasts
```

Tags are auto-incremented per type prefix. The tag is printed after each command:

```
Saved as: var001
```

### Using Stored Tags

Stored tags can be used directly with post-estimation commands (`irf`, `fevd`, `hd`, `forecast`):

```bash
# These are equivalent:
friedman irf var001
friedman irf var data.csv --from-tag=var001
```

The tag resolution system runs before dispatch and automatically maps `var001` to the correct model type and `--from-tag` option.

### Storage File

The storage file `.friedmanlog.bson` is created in the current working directory. Each entry contains:

- **tag**: Unique identifier (e.g., `var001`)
- **type**: Model/result type (e.g., `var`, `irf`, `forecast`)
- **data**: Serialized model data (Dict of primitives)
- **meta**: Metadata (command, data path, options used)
- **timestamp**: ISO 8601 timestamp

## list

### list models

List all stored models.

```bash
friedman list models
friedman list models --type=var
friedman list models --format=json --output=models.json
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--type` | `-t` | String | (all) | Filter by type: `var`, `bvar`, `lp`, `arima`, `gmm`, `static`, `dynamic`, `gdfm`, `arch`, `garch`, `egarch`, `gjr_garch`, `sv`, `fastica`, `ml` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

**Output:** Table with tag, type, timestamp, and command info.

### list results

List all stored analysis results.

```bash
friedman list results
friedman list results --type=irf
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--type` | `-t` | String | (all) | Filter by type: `irf`, `fevd`, `hd`, `forecast` |
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

## rename

Rename a stored model or result tag.

```bash
friedman rename var001 gdp_model
friedman rename irf001 baseline_irf
```

| Argument | Description |
|----------|-------------|
| `<old_tag>` | Current tag name (e.g., `var001`) |
| `<new_tag>` | New tag name |

## project

### project list

List all registered projects. Projects are auto-registered in `~/.friedman/projects.json` on first save.

```bash
friedman project list
```

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--format` | `-f` | String | `table` | `table`, `csv`, `json` |
| `--output` | `-o` | String | | Export file path |

### project show

Show information about the current project (working directory).

```bash
friedman project show
friedman project          # shortcut (bare "project" defaults to show)
```

**Output:** Project name (directory basename), path, number of stored entries, entries by type.

## Global Settings

Friedman stores global settings in `~/.friedman/`:

- **`settings.json`**: User preferences
- **`projects.json`**: Registry of project paths

The `FRIEDMAN_HOME` environment variable can override the default location.
