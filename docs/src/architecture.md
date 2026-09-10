# Architecture

Friedman-cli is a Julia CLI application with a custom command-line framework adapted from Comonicon.jl.

## Execution Flow

```
bin/friedman ARGS
  → Pkg.activate(project_dir)
  → Friedman.main(ARGS) → run_cli(ARGS)
    → Friedman.APP                         # memoized Entry (const APP = build_app() at precompile)
      # build_app() registers all top-level command groups once
    → dispatch(APP, args)
      → dispatch_node()                    # walks NodeCommand tree by matching tokens
      → dispatch_leaf()                    # tokenize → bind_args → leaf.handler(; bound...)
```

## Data Flow

CSV is the **import** format, not the working format. Commands take a **stem**;
`.jld2` is native storage (MEMs `save_model` / `load_model`), not part of the
argv contract. Wave 2 ships **result** handles (`--result` / `--save-result`)
and `friedman show STEM` (render any loadable handle).

```
CSV | :example
        │
        ▼
data import --kind timeseries|panel|cross-section [-o STEM]
        │
        ▼
STEM.jld2     TimeSeriesData | PanelData | CrossSectionData
        │
        ├─ data describe|diagnose|validate     (read, real type)
        ├─ data fix|transform|dropna|keeprows|balance
        │       -o STEM'     →  same type, STEM'.jld2
        │       -o file.csv  →  CSV export; stderr: metadata dropped
        ├─ data export STEM  →  CSV (inverse of import)
        │
        ▼
estimate var STEM --save-model var         # stem → var.jld2
        │
        ▼
irf var --model var --save-result irf      # --model stem → var.jld2
friedman show irf                          # stem → irf.jld2 (no CSV fallback)
friedman show var                          # fitted model table / fields
forecast evaluate metrics STEM --actual gdp --result fcst_var,fcst_bvar
# evaluate --result is a comma-separated string (not RESULT_OPTION)

CSV shortcut (unchanged, additive 0.x):
estimate var macro.csv --lags 2
```

### Stem resolution

**Save** (`data import -o`, data-edit `-o`, and a *present* `--save-model`):

- **Data-edit `-o` only:** empty / omitted → leaf-specific default stem, then
  the rule below. (`--save-model` omitted means do not save — it is not a
  default stem.)
- No suffix → append `.jld2` and use native `save_model`.
- `.jld2` → native `save_model` (including `data import … -o out.jld2`, the
  intended CSV→typed conversion).
- `.fmod` → interim Serialization handle (unregistered types).
- `.csv` on a **data-edit** output → CSV export (frequency/tcode/dates dropped).
- **Data-edit** of a CSV with `-o out.jld2` → `usage/invalid` (run
  `data import` first). This edit refusal does **not** apply to `data import`
  itself.

**Load** (data positional / `data export`; `--model` / `--result` / `show`):

1. **Data slots:** resolve stem — `path.jld2` if that file exists (preferred),
   else `path.csv`, else exact `path` (`.fmod`, `.toml` DSGE specs,
   extensionless files). Else `data/file-not-found` (exit 3).
2. **`--model` (Wave 2):** when the leaf's `model_types` is nonempty,
   `resolve_stem(; slot=:result)` — `STEM.jld2` if that file exists (no CSV
   fallback), then type-check. Empty `model_types` (DSGE builtins,
   `data validate --model`) still requires an explicit suffix / URI.
   `model info` is header-only and still wants `.jld2` / `.fmod` / `model://`.
3. **`--result` / `friedman show STEM` (Wave 2):** `resolve_stem(; slot=:result)` —
   `STEM.jld2` if that file exists, else the exact path. No CSV fallback
   (show is for loadable handles, not import). Bundles emit a keys-only
   table (`show_payload`); `:timeseries`/`:panel`/`:cross_section` emit
   descriptive stats; `:io` and other kinds fall through to `long_table` /
   `DataFrame` / field dump (never `to_matrix` an IOData). `--plot` /
   `--plot-save` call `_maybe_plot` on every path; missing recipe →
   `model/unsupported` (exit 5).

If both `macro.jld2` and `macro.csv` exist, the handle wins on data slots.
Explicit suffixes skip the search (`macro.csv` is CSV, `var.jld2` is a
handle). `model://name` is the serve-session URI and is not stem-expanded.
`:fred_md` example names are unchanged.

`wrap_legacy` type-checks a loaded data handle against the leaf's
registry-declared `data_kinds` **before** the handler runs. A mismatch is
`data/wrong-kind` (exit 3) — e.g. a `PanelData` handle on `estimate var`. CSV
remains legal on every leaf that lists `:csv`. `--result` of a type not in
`result_types` is `data/wrong-result` (exit 3); `--model` of a type not in
`model_types` is `model/wrong-kind` (exit 5). `--result` cannot be combined
with `--model` or a data path (`usage/invalid`).

Central resolver: `src/handles.jl`. Native persist: `src/model_handle.jl`.

### Rendering

After the library call, results still go through `output_result` (`:table` →
PrettyTables, `:csv` → CSV.write, `:json` → the versioned envelope).

**Rendering the result to a DataFrame (C051)** goes through one of three paths, in order of
preference:

1. **`long_table(result)`** — MEMs' tidy renderer for array-valued results (IRF, FEVD,
   forecasts): one row per `(horizon, variable[, shock])` cell. Used by `irf`/`fevd`
   var/vecm/bvar/lp/favar/sdfm and `forecast` var/vecm/lp/arima/static/bvar/dynamic/gdfm/favar.
2. **`DataFrame(model)`** — MEMs' tidy renderer for coefficient-bearing models: one row per
   term, columns `term|estimate|std_error|stat|p_value|ci_lower|ci_upper` (plus an
   `equation`/`alternative`/`block` prefix for VAR/multinomial/ordered models). Used by
   `estimate` var/reg/iv/logit/probit/preg/piv/plogit/pprobit/ologit/oprobit/mlogit.
3. **Hand-built `DataFrame(...)`** — the pre-C051 fallback, kept only where MEMs has no
   matching result type (`irf`/`fevd pvar`, `hd`, `predict`/`residuals`, Arias/Uhlig/sign
   IRF paths, the whole `io` family, the SUR/3SLS systems and MGARCH (CCC/DCC/BEKK) leaves,
   the penalized/robust/Tobit/truncated/Heckman regression leaves, the state-space/TVP and
   nonparametric (KDE/kernel-reg/LOWESS) leaves, the single-equation/panel cointegrating
   regression leaves (`CointRegModel`/`PanelCointRegModel`), the ARDL/NARDL family
   (`ARDLModel`/`NARDLModel`/`ARDLLongRun`/`ARDLBoundsTest`/`NARDLSymmetryTest`/`NARDLMultipliers`
   — `estimate ardl`/`nardl`, `test ardl-bounds`/`nardl-symmetry`, `multipliers nardl`), and the
   dynamic heterogeneous-panel ARDL family (`PMGModel` — `estimate pmg`, `test pmg-hausman`), and the
   nonlinear-TS family (`ThresholdModel`/`STARModel`/`MSRegModel` — `estimate setar`/`star`/`ms-ar`/`ms`;
   the two `*Forecast` types ARE registered and render via `long_table`) — none of
   these result types are Tables.jl-registered upstream) or where the tidy schema would lose information the command
   needs to convey (volatility `forecast`'s `variance|volatility` table, `did estimate`'s ATT
   summary). The `io` matrices (Leontief/Ghosh inverses, coefficients), MGARCH conditional
   correlations, and the Markov-switching K×K regime-transition matrix (`estimate ms-ar`/`ms`) render
   **wide** (sector×sector / series×series / regime×regime); vector results render one row
   per sector/term.

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

1. Entry-level: check `--version` / `--help` / `--warranty` / `--conditions`, then delegate to root node
2. Node-level: match first arg token as subcommand name, recurse into child
3. Leaf-level: tokenize remaining args, bind to declared params, call `handler(; bound...)`

Unknown subcommands print an error and show help. `--help` at any level prints context-appropriate help.

## Module Structure

```
src/
  Friedman.jl             # Main module: imports, includes, build_app(), const APP, run_cli, main()
  cli/
    types.jl              # 6 CLI structs (Argument, Option, Flag, Leaf/Node/Entry)
    parser.jl             # tokenize(), bind_args(), convert_value()
    dispatch.jl           # dispatch() → dispatch_node() → dispatch_leaf()
    help.jl               # print_help() with colored, column-aligned output
  io.jl                   # load_data, df_to_matrix, variable_names, output_result
  model_handle.jl         # save_model_dispatch / load_model_dispatch (.jld2 | .fmod | model://)
  handles.jl              # stem resolver, data-kind check, typed persist (after model_handle.jl)
  registry/
    spec.jl               # CommandSpec (data_kinds / model_types / result_types)
    adapter.jl            # wrap_legacy: stem-resolve + type-check + save
  config.jl               # TOML loader for priors, identification, GMM, non-Gaussian
  commands/
    shared.jl             # ID_METHOD_MAP, shared estimation/output helpers
    estimate.jl           # 24 estimation subcommands
    test.jl               # 29+ test subcommands (+ nested var 2, pvar 4)
    irf.jl                # 7 IRF subcommands
    fevd.jl               # 7 FEVD subcommands
    hd.jl                 # 5 HD subcommands
    forecast.jl           # 14 forecast subcommands
    predict.jl            # 16 predict subcommands
    residuals.jl          # 16 residuals subcommands
    filter.jl             # 5 filter subcommands
    data.jl               # 13 data subcommands (incl. import / export)
    nowcast.jl            # 5 nowcast subcommands
    dsge.jl               # DSGE subcommands + bayes node (13 sub-leaves) + HA/CT/OLG nodes
    did.jl                # 7 DID subcommands (3 estimation + 4 test)
    multipliers.jl        # multipliers nardl — new top-level (C062b, action-first)
    policy.jl             # policy counterfactuals — new top-level (W4/#126, MEMs 0.8.0 CF module)
    serve.jl              # serve --mcp
    show.jl               # show HANDLE (Wave 2)
```

The ARDL/NARDL family (`estimate ardl`/`nardl` in `estimate.jl`, `test ardl-bounds`/`nardl-symmetry`
in `test.jl`, and `multipliers nardl` in the new top-level `multipliers.jl`) all fit via the shared
`_load_reg_data` (`y` + `X`) loader and the `_fit_ardl`/`_fit_nardl` wrappers in `estimate.jl`, so
the four leaves share one estimation path and one set of hand-built renderers. The dynamic
heterogeneous-panel ARDL family (`estimate pmg` in `estimate.jl`, `test pmg-hausman` in `test.jl`)
similarly shares the hardened `_load_panel_reg` panel loader (`shared.jl`): both resolve `--dep`/`--indep`
to `Symbol`s over a `PanelData` and splat the regressors into `estimate_pmg(pd, dep, xs...)`; the test
leaf fits the panel twice (efficient vs Mean Group) and runs the PMG-typed `hausman_test`.

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
| `SparseArrays` (stdlib) | Sparse matrix operations |
| `Random` (stdlib) | Random number generation (DSGE simulation) |
| `Logging` (stdlib) | Route MEMs `@info`/`@warn` to stderr; `--quiet` drops info (C050) |

## Compatibility

| | Version |
|---|---------|
| Julia | `>= 1.12` |
| MacroEconometricModels | `0.7.0` |

## Stability policy

Declared at v0.10.0: the machine surface is the API — the command tree, the
option/flag surface, the envelope schema (`schema/envelope-v1.json`), the
stable `data` table keys, the error-code taxonomy, and the exit codes.
Envelope schema v1 is **additive-only from v0.10.0** (a breaking envelope
change bumps `schema_version` to 2 and is a major release; the formal freeze
declaration lands at v1.0.0). 0.x minors are additive-only; removals and
renames happen only at majors after at least one minor of deprecation alias.
The hidden snake_case aliases and `FRIEDMAN_LEGACY_OUTPUT` are scheduled for
removal at v1.0.0.

## Totals

21 top-level commands, 456 subcommands (registry-generated — see the inventory at the bottom of `CLAUDE.md`).
