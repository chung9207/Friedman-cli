# Architecture

Friedman-cli is a Julia CLI application with a custom command-line framework adapted from Comonicon.jl.

## Execution Flow

```
bin/friedman ARGS
  → Pkg.activate(project_dir)
  → Friedman.main(ARGS)
    → build_app()                          # constructs Entry with full command tree
      → register_estimate_commands!()      # 11 register functions, one per top-level command
      → register_test_commands!()
      → register_irf_commands!()
      → register_fevd_commands!()
      → register_hd_commands!()
      → register_forecast_commands!()
      → register_predict_commands!()
      → register_residuals_commands!()
      → register_filter_commands!()
      → register_data_commands!()
      → register_nowcast_commands!()
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
    Results → DataFrame                     # command builds result DataFrame
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
  commands/
    shared.jl             # ID_METHOD_MAP, shared estimation/output helpers
    estimate.jl           # 17 estimation subcommands
    test.jl               # 16+ test subcommands (+ nested var 2, pvar 4)
    irf.jl                # 5 IRF subcommands
    fevd.jl               # 5 FEVD subcommands
    hd.jl                 # 4 HD subcommands
    forecast.jl           # 13 forecast subcommands
    predict.jl            # 12 predict subcommands
    residuals.jl          # 12 residuals subcommands
    filter.jl             # 5 filter subcommands
    data.jl               # 9 data subcommands
    nowcast.jl            # 5 nowcast subcommands
```

## Handler Conventions

- **Naming**: `_action_model(; kwargs...)` (e.g., `_estimate_var`, `_irf_bvar`, `_forecast_arch`, `_nowcast_dfm`)
- **Signature**: keyword arguments match declared `Option` names (with hyphen-to-underscore)
- **Pattern**: load data → call library → build DataFrame → `output_result()`
- **Registration**: each command file defines `register_X_commands!()` returning a `NodeCommand`

## Dependencies

| Package | Purpose |
|---------|---------|
| `MacroEconometricModels` | Core econometric library |
| `CSV` | Data loading |
| `DataFrames` | Tabular data manipulation |
| `PrettyTables` | Terminal table formatting |
| `JSON3` | JSON output format |
| `TOML` (stdlib) | Configuration file parsing |
| `LinearAlgebra` (stdlib) | Matrix operations |
| `Statistics` (stdlib) | Mean, median calculations |
