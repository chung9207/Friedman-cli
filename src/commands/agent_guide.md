# Agent Guide

Contract for agents driving Friedman-cli. This document is the single source: it
ships inside the binary and is served verbatim by `friedman schema --docs`, and
the documentation site renders the same file.

## One envelope on stdout

With `--format=json`, **stdout is exactly one JSON document** (the result envelope). Status and diagnostics go to **stderr**.

```bash
friedman estimate var data.csv --lags 1 --format json | jq .
```

Example shape (fields abbreviated):

```json
{
  "schema_version": 1,
  "command": "friedman estimate var",
  "status": "ok",
  "meta": {
    "cli_version": "0.9.2",
    "mems_version": "0.8.0",
    "julia": "1.12.x",
    "seed": null,
    "argv": ["estimate", "var", "data.csv", "--lags", "1", "--format", "json"],
    "elapsed_ms": 12.3
  },
  "data": {
    "var_coefficients": {
      "columns": ["equation", "term", "estimate", "..."],
      "rows": [["y1", "y1.l1", 0.5]]
    }
  },
  "warnings": [],
  "artifacts": [],
  "error": null
}
```

The shape is strict (v0.10.0, W1/#136):

- **Every `data` value is a table** — an object with exactly `columns` (array of
  string) and `rows` (array of arrays). Cell values are number, string, boolean,
  or null; non-finite floats appear as the strings `"NaN"`/`"Inf"`/`"-Inf"`,
  never silent JSON `null`; `missing` cells appear as `null`.
- `meta` always carries `cli_version`, `julia`, and `mems_version`; `seed`,
  `argv`, `elapsed_ms`, and `manifest` are typed-optional, and new meta keys may
  be added over time (additive).
- `status` and `error` co-occur: `"ok"` implies `error: null`; `"error"` implies
  an error object whose `code` matches `class/code` from the exit-code taxonomy
  below.

Envelope schema v1 is **draft until CLI v1.0** and **additive-only from
v0.10.0**: no key is removed or retyped within `schema_version` 1; a breaking
change bumps `schema_version` to 2 and is a major release. Normative JSON
Schema: `schema/envelope-v1.json` — it validates under any conformant draft-07
validator (CI cross-checks every golden and every T3-captured envelope with
python-jsonschema), so you can validate responses with ajv / jsonschema
directly.

## Stable table keys (v0.10.0)

`data` keys are **predictable before you run the command**: they come from each
leaf's registry-declared table names, never from runtime values.

- **Singleton tables** use the declared name verbatim: `estimate var` always
  answers under `var_coefficients` + `information_criteria` — regardless of
  `--lags`, your column names, or anything estimated. (Before v0.10.0 the same
  table was `var_2_coefficients` — the lag order baked into the address.)
- **Family tables** appear when one invocation emits several sibling tables
  (per-shock IRFs, per-variable historical decompositions). Their keys are
  `<declared-name>_<variable-slug>` — e.g. `irf var` on columns `gdp,cpi`
  answers under `irf_gdp` and `irf_cpi`. The declared name is the stable
  prefix; the suffix is a slug of *your own* variable name, so you can still
  compute every key in advance.
- Option values, horizons, CI levels, method names, and estimated parameters
  never appear in keys — they stay in the human-readable table titles.
- A CI drift gate (`check_table_keys`) validates every emitted key against the
  registry declarations, so this contract cannot silently rot.

## Every failure is an envelope too (v0.10.0)

When the argv asks for JSON (`--format json` / `-f json`, or the leading
`--json` global), **every failure also emits exactly one envelope on stdout** —
including usage/parse errors that fail before a command resolves, which used to
leave stdout empty. `status` is `"error"`, `data` is `{}`, and the `error`
object carries the machine-readable failure:

```json
{
  "schema_version": 1,
  "command": "friedman estimate var",
  "status": "error",
  "data": {},
  "error": {
    "code": "usage/parse",
    "message": "friedman estimate var: unknown option --lgas — did you mean --lags?",
    "exit_code": 2
  }
}
```

- `error.code` is `class/code` from the taxonomy below; `error.exit_code`
  **always equals the process exit code** — both derive from the same class
  mapping, so they cannot disagree.
- `error.hint` is present only when there is something to say; it is omitted
  rather than emitted empty.
- Under `--format table`/`csv`, failures keep stdout **empty** — human-readable
  error text goes to stderr, as always.
- The interactive REPL dispatches outside this path and is not part of the
  agent contract.

## Exit codes

| Code | Class | Example |
|------|-------|---------|
| 0 | ok | successful command |
| 2 | usage | unknown command/option, bad `--format` |
| 3 | data | file not found, empty CSV, bad path |
| 4 | config | missing/malformed TOML config |
| 5 | model | domain/estimation failures (typed when available) |
| 6 | env | network/environment failures |
| 1 | internal | unexpected errors (report as bugs) |

```bash
friedman nosuchcmd; echo $?          # 2
friedman estimate var /nope.csv; echo $?   # 3
```

Domain failures carry **typed codes** where the underlying failure mode is
recognized (all are stable identifiers; the set only grows):

| `error.code` | Exit | Meaning |
|--------------|------|---------|
| `model/convergence` | 5 | estimator failed to converge |
| `model/identification` | 5 | identifying restrictions/instruments insufficient |
| `model/singular` | 5 | near-singular system |
| `model/stochastic-singularity` | 5 | more observables than shocks with no measurement error (DSGE likelihood) |
| `model/solve` | 5 | DSGE steady state / solver failure |
| `model/error` | 5 | other recognized domain failure |
| `data/serialization` | 3 | saved model handle unreadable or version-incompatible |
| `data/orientation` | 3 | data matrix transposed relative to the observables |
| `data/wrong-kind` | 3 | data slot loaded a container not in the leaf's `data_kinds` (e.g. `PanelData` on `estimate var`) |
| `data/wrong-result` | 3 | `--result` loaded a type not in the leaf's `result_types` (e.g. a `VARModel` on `irf var --result`) |
| `model/wrong-kind` | 5 | `--model` loaded a type not in the leaf's `model_types` (e.g. an `ImpulseResponse` on `irf var --model`) |

Anything else surfaces as `usage/*`, `data/*`, `config/*`, or `env/*` per the
class table above; `internal/error` (exit 1) means a CLI bug — report it.

## Strict parsing & self-correction

Unknown options throw with a suggestion when the edit distance is small:

```text
Error: friedman estimate var: unknown option --lgas — did you mean --lags?
```

`--format` is restricted to `table|csv|json`. Negative numerics bind: `--threshold -0.5`.

## Self-description: `friedman schema`

```bash
friedman schema | jq '.commands | length'            # top-level command count
friedman schema estimate var | jq '.options[].name'  # leaf options
friedman schema estimate var | jq '.input_schema'    # draft-07 invocation schema
friedman schema estimate var | jq '.tables'          # declared result-table keys
friedman schema | jq '.contract.exit_codes'          # exit-code taxonomy
friedman schema | jq -r '.docs'                      # this guide (--docs)
```

Output is **raw JSON** (not wrapped in an envelope). Since v0.10.0 (W5/#140)
the document is fully machine-actionable:

- **`input_schema`** (leaf docs): a draft-07 JSON Schema over the invocation
  surface — one property per argument/option/flag under the CLI's kebab-case
  names (`string`/`integer`/`number`/`boolean`, `enum` from declared choices,
  defaults, `required` = required positionals, `additionalProperties: false`).
  Each property carries an **`x-cli`** annotation (`kind`:
  `argument|option|flag`, `position` for positionals, `long`/`short` spellings)
  so an exact argv can be reconstructed from a validated object. Handle slots
  also carry **`x-handle`** (`role`: `data|model|result`, plus `kinds` /
  `types` from the registry) — see *Typed handles* below.
- **`tables`** (leaf docs): the registry-declared result-table keys — `name`,
  `description`, and `family` (`true` means keys are `<name>_<variable-slug>`,
  one per variable/shock; see *Stable table keys*). This is the same
  declaration set the CI drift gate enforces, so it is exactly what the
  envelope's `data` will use.
- **`contract`** (root doc): `envelope_schema` embeds the full normative
  `envelope-v1.json`, and `exit_codes` lists the class taxonomy — an agent can
  bootstrap the entire output contract from one call.
- **`--docs`**: adds this guide verbatim as a `docs` markdown string (works on
  the root and on any command path).

The `schema` command itself is deliberately absent from the command inventory
(its variable-length path does not fit the leaf model); it is discoverable from
the top-level help and from this guide.

## MCP server: `friedman serve --mcp`

```bash
friedman serve --mcp    # JSON-RPC 2.0 / Model Context Protocol on stdio
```

Every command becomes an MCP **tool** — one process, no per-call spawn:

- **`tools/list`** mirrors the registry: tool name = command path joined with
  `_` (`estimate_var`, `dsge_bayes_estimate`); `inputSchema` is the same
  draft-07 schema `friedman schema` reports.
- **`tools/call`** reconstructs the exact argv from your arguments object,
  forces `--format json`, and returns the **envelope verbatim** as text
  content — same bytes as the CLI, same stable `data` keys, same typed
  `error.code`/`exit_code` on failure (`isError` mirrors a nonzero exit
  class). Everything in this guide about envelopes applies unchanged.
- **`model://` handles**: within a serve session, `--save-model model://name`
  stores the fitted model **in memory** and `--model model://name` reuses it —
  estimate once, then run irf/fevd/forecast against the handle with no
  re-estimation and no files. Handles live exactly as long as the session;
  file handles (`.jld2`/`.fmod`) also work as usual.
- Requests are handled **serially** (handlers are not thread-audited); stdout
  carries only the JSON-RPC stream — status and library logs stay on stderr.

## Determinism & reproducibility

```bash
friedman --seed 42 estimate var data.csv --format json
```

`meta.seed` echoes the seed; use the same seed for reproducible stochastic paths. Every JSON
envelope also carries `meta.manifest` — the MacroEconometricModels.jl reproducibility manifest
(seed, threads, OS, Julia + package + dependency versions, git, timestamp) — for provenance.
`--seed` is additionally forwarded as the estimator's own `seed=` everywhere upstream
supports it (BVAR/IRF plus SV, MFVAR/TVPVAR, FAVAR/SDFM, SMM, quantile/robust/nonlinear,
DiD, LP, PVAR bootstrap, conditional forecasts, set-identification, policy/OPP, DSGE Bayes
and Krusell–Smith), so their `ReproManifest` records it and the draws reproduce bit-for-bit.
`friedman model reproduce HANDLE` re-runs the recorded estimator and reports a match verdict
plus per-field diffs (`unverifiable` when no seed was recorded — not a pass).

## Typed handles (data, model, result)

Three object kinds, each with its own slot. **Wave 2 ships result handles and
`friedman show`.** Data + model shipped in Wave 1; `--result` / `--save-result`
skip compute and re-render a saved result; `friedman show STEM` renders any
loadable handle (data, model, result, or a keys-only bundle listing).

| Kind | Argv slot | Native persist | Wave |
|------|-----------|----------------|------|
| data | positional `<data>` / `--data` | `data import -o STEM` → `STEM.jld2` | 1 |
| model | `--model` / `--save-model` | `--save-model STEM` → `STEM.jld2` | 1 |
| result | `--result` / `--save-result` | `--save-result STEM` | 2 |

**Stems vs suffixes.** Data positionals, `--save-model`, `--save-result`,
`--model` (when the leaf declares `model_types`), `--result`, and
`friedman show` accept suffix-less stems (`macro`, `--save-model var` →
`var.jld2`, `--model var` loads `var.jld2`). `model info` still wants an
explicit handle path (`.jld2` / `.fmod` / `model://`). Data load prefers
`path.jld2` over `path.csv` when both exist; `--model` / `--result` / `show`
have no CSV fallback. An explicit suffix skips the search. `model://name` is
the in-session URI (serve) and is not stem-expanded. `:fred_md` example names
are unchanged.

```bash
friedman data import macro.csv --kind timeseries -o macro
friedman estimate var macro --lags 2 --save-model var   # stem → var.jld2
friedman irf var --model var --horizons 12 --save-result irf
friedman irf var --result irf                            # re-render; no data
friedman show macro          # TimeSeriesData descriptive stats
friedman show var            # fitted model table
friedman show irf            # re-render the saved ImpulseResponse
friedman forecast evaluate metrics macro --actual gdp --result fcst_var,fcst_bvar
friedman model info var.jld2
# CSV shortcut still works:
friedman estimate var macro.csv --lags 2
```

`data import --kind` is required for CSV (no autodetection). Edits do not
promote CSV to `.jld2` (`usage/invalid` — import first; `data import` itself
is the intended CSV→`.jld2` conversion). A handle whose type is not in the
leaf's `data_kinds` is `data/wrong-kind` (exit 3). `--result` of the wrong
result type is `data/wrong-result` (exit 3); `--model` of the wrong model
type is `model/wrong-kind` (exit 5). `--result` cannot be combined with
`--model` or a data path (`usage/invalid`).

`friedman schema <leaf>` annotates handle slots with **`x-handle`**:
`{role: "data"|"model"|"result", kinds: [...], types: [...]}` next to the
existing `x-cli` argv annotations. **Presence vs absence of `x-handle` is the
signal** — not whether `kinds`/`types` are empty. Empty `types` on a data slot
is normal (data uses `kinds`); producing-leaf `--model` / `--result` slots
carry `types` from the registry. `data validate --model` has **no**
`x-handle` (it is a type *string*, not a model handle). Evaluate `--result`
is a comma-separated string (`handle=false`, no `x-handle`).

### Model handles

`--save-model PATH` persists a fitted model (suffix-less stem → `.jld2`);
`--model STEM` (or `.jld2` / `.fmod` / `model://`) reloads it (skipping
re-estimation) on leaves that declare `model_types`. `.jld2` is the native, versioned format covering the full
upstream serialization registry (350 types at MacroEconometricModels 0.9.3) —
every model `estimate` can fit, including DSGE/HA solutions (`dsge solve`,
`dsge ha solve`, `dsge ha steady-state`, `dsge bayes estimate` all take
`--save-model`). `.fmod` remains as the interim handle for unregistered
payloads. `friedman model info PATH.jld2` reads the container header (writing
versions, note, bundle layout) without re-running estimation — header-only, it
never executes stored code.
Trust caveat (mirrors upstream): a `--model` handle carrying DSGE/HA equations recompiles
them at load through an AST allowlist (`Core.eval`), the same risk class as
`Serialization.deserialize` — only load files you trust. Programmatic payloads with
anonymous closures (household utilities, `ss_fn`) fail at `--save-model` time with
`data/serialization`; persist named functions or callable structs (`CRRAUtility`) instead.

## Quiet / no-color / json alias

| Flag | Effect |
|------|--------|
| `--quiet` / `-q` | suppress CLI status on stderr |
| `--no-color` | disable ANSI (also honors `NO_COLOR`) |
| `--json` | alias that injects `--format json` if missing |

Leading globals only (before the first subcommand token).

## Legacy output

```bash
FRIEDMAN_LEGACY_OUTPUT=1 friedman estimate var data.csv --format json
```

Restores pre-0.5 multi-document / non-envelope JSON for one minor release.

## Handler rules (for contributors)

- Status/progress: `_status` / `_status_styled` (stderr), never bare `println` for status
- Data tables: `output_result` / `output_kv`
- Typed failures: `throw(CliError("class/code", "message"; hint="…"))`
