# Architecture

Friedman-cli is a ~5,400 line Julia CLI application with a custom command-line framework adapted from Comonicon.jl.

## Execution Flow

```
bin/friedman ARGS
  → Pkg.activate(project_dir)
  → Friedman.main(ARGS)
    → init_settings!()                     # ensure ~/.friedman/ exists
    → resolve_stored_tags(args)            # tag resolution: ["irf", "var001"]
                                           #   → ["irf", "var", "--from-tag=var001"]
    → build_app()                          # constructs Entry with full command tree
      → register_estimate_commands!()      # 9 register functions
      → register_test_commands!()
      → register_irf_commands!()
      → register_fevd_commands!()
      → register_hd_commands!()
      → register_forecast_commands!()
      → register_list_commands!()
      → register_rename_commands!()
      → register_project_commands!()
    → dispatch(entry, args)
      → dispatch_node()                    # walks NodeCommand tree by matching tokens
      → dispatch_leaf()                    # tokenize → bind_args → leaf.handler(; bound...)
```

## Data Flow

```
CSV file → load_data(path)                 # → DataFrame, validates exists & non-empty
         → df_to_matrix(df)                # → Matrix{Float64}, selects numeric columns
         → variable_names(df)              # → Vector{String}, numeric column names
                ↓
    MacroEconometricModels.jl functions     # estimate_var, irf, forecast, etc.
                ↓
    Results → storage_save_auto!(prefix, data)   # BSON auto-tagging
           → DataFrame                     # command builds result DataFrame
           → output_result(df; format, output, title)
                ↓
              :table → PrettyTables (center-aligned)
              :csv   → CSV.write
              :json  → JSON3.write (array of row dicts)
```

## CLI Framework

The CLI framework is custom-built (adapted from Comonicon.jl). Key types:

### Type Hierarchy

- **`Entry`** -- Top-level: name + root `NodeCommand` + version
- **`NodeCommand`** -- Command group: name + `Dict{String, Union{NodeCommand, LeafCommand}}`
- **`LeafCommand`** -- Executable: name + handler function + args/options/flags
- **`Argument`** -- Positional parameter (name, type, required, default)
- **`Option`** -- Named `--opt=val` or `-o val` (name, short, type, default)
- **`Flag`** -- Boolean `--flag` or `-f` (name, short)

### Parser

The `tokenize()` function converts raw argument strings into `ParsedArgs`:

```
--opt=val     → options["opt"] = "val"
--opt val     → options["opt"] = "val"
-o val        → options["o"] = "val"
--flag        → flags = Set(["flag"])
-abc          → flags = Set(["a", "b", "c"])     # bundled
--            → stops option parsing
other         → positional arguments
```

Then `bind_args()` maps parsed tokens to the `LeafCommand`'s declared arguments, options, and flags, with type conversion via `convert_value()`.

### Dispatch

`dispatch()` walks the command tree:

1. Entry-level: check `--version` / `--help`, then delegate to root node
2. Node-level: match first arg token as subcommand name, recurse into child
3. Leaf-level: tokenize remaining args, bind to declared params, call `handler(; bound...)`

Unknown subcommands print an error and show help. `--help` at any level prints context-appropriate help.

## Storage System

BSON-based model persistence in `.friedmanlog.bson`:

- **`storage_save_auto!(prefix, data, meta)`**: Auto-generate tag (`var001`, `var002`, ...) and save
- **`storage_save!(tag, prefix, data, meta)`**: Save under explicit tag
- **`storage_load(tag)`**: Load stored entry by tag
- **`storage_list(; type_filter)`**: List entries, optionally filtered
- **`storage_rename!(old, new)`**: Rename a tag
- **`serialize_model(model)`**: Convert model struct to Dict of primitives for BSON

### Tag Resolution

`resolve_stored_tags()` runs before dispatch as a pre-processing step:

1. Checks if `args[2]` looks like a stored tag (e.g., `var001`, `bvar003`)
2. Loads the tag's metadata to determine the model type
3. Rewrites args: `["irf", "var001"]` → `["irf", "var", "--from-tag=var001"]`

This enables the shorthand `friedman irf var001` syntax.

## Settings

Global settings in `~/.friedman/` (overridable via `FRIEDMAN_HOME`):

- **`settings.json`**: User preferences
- **`projects.json`**: Project registry (auto-populated on first `storage_save!`)

## Module Structure

```
src/
  Friedman.jl             # Main module: imports, includes, build_app(), main()
  cli/
    types.jl              # 6 CLI structs (Argument, Option, Flag, Leaf/Node/Entry)
    parser.jl             # tokenize(), bind_args(), convert_value()
    dispatch.jl           # dispatch() → dispatch_node() → dispatch_leaf()
    help.jl               # print_help() with colored, column-aligned output
  io.jl                   # load_data, df_to_matrix, variable_names, output_result
  config.jl               # TOML loader for priors, identification, GMM, non-Gaussian
  storage.jl              # BSON storage with auto-tagging
  settings.jl             # Global ~/.friedman/ management
  commands/
    shared.jl             # ID_METHOD_MAP, shared estimation helpers
    estimate.jl           # 15 estimation subcommands
    test.jl               # 12+ test subcommands
    irf.jl                # 3 IRF subcommands
    fevd.jl               # 3 FEVD subcommands
    hd.jl                 # 3 HD subcommands
    forecast.jl           # 12 forecast subcommands
    list.jl               # 2 list subcommands
    rename.jl             # 1 rename command
    project.jl            # 2 project subcommands
```

## Handler Conventions

- **Naming**: `_action_model(; kwargs...)` (e.g., `_estimate_var`, `_irf_bvar`, `_forecast_arch`)
- **Signature**: keyword arguments match declared `Option` names (with hyphen-to-underscore)
- **Pattern**: load data → call library → build DataFrame → `output_result()` → `storage_save_auto!()`
- **Registration**: each command file defines `register_X_commands!()` returning a `NodeCommand`

## Dependencies

| Package | Purpose |
|---------|---------|
| `MacroEconometricModels` | Core econometric library (229 exports) |
| `CSV` | Data loading |
| `DataFrames` | Tabular data manipulation |
| `PrettyTables` | Terminal table formatting |
| `JSON3` | JSON output format |
| `BSON` | Model storage serialization |
| `Dates` | Timestamps for storage entries |
| `TOML` (stdlib) | Configuration file parsing |
| `LinearAlgebra` (stdlib) | Matrix operations |
| `Statistics` (stdlib) | Mean, median calculations |
