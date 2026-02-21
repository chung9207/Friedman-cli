# Friedman-cli v0.2.2 Design: Storage Removal + MEMs v0.2.4 Feature Adoption

**Date:** 2026-02-22
**Status:** Approved

## Overview

Two changes in one release:
1. Remove BSON-based model/result storage, project registry, and all dependent commands
2. Wrap all new MacroEconometricModels.jl v0.2.3-v0.2.4 features

## Part 1: Storage Removal

### Files Deleted
- `src/storage.jl` (246 lines)
- `src/settings.jl` (146 lines)
- `src/commands/list.jl` (88 lines)
- `src/commands/rename.jl` (37 lines)
- `src/commands/project.jl` (87 lines)
- `src/commands/plot.jl` (342 lines)

### Top-Level Commands Removed
- `list` (models | results)
- `rename`
- `project` (list | show)
- `plot` (tag-based re-plotting)

### Code Stripped From Remaining Files

**src/Friedman.jl:**
- Remove `using BSON`
- Remove `include()` for deleted files
- Remove `init_settings!()` call in `main()`
- Remove `resolve_stored_tags()` call in `main()`
- Remove `build_app()` registrations for list, rename, project, plot

**src/commands/shared.jl:**
- Remove `_resolve_from_tag()` function
- Remove `serialize_model()` function (if defined here)

**All command files (estimate, test, irf, fevd, hd, forecast, predict, residuals, filter):**
- Remove all `storage_save_auto!()` calls (~80 total)
- Remove all `serialize_model()` calls (~45 total)
- Remove `--from-tag` Option declarations from LeafCommands
- Remove `from_tag` kwargs and conditional branches in handlers
- Remove `_resolve_from_tag()` calls (~57 total)

**Project.toml:**
- Remove BSON from `[deps]`, `[compat]`, `[extras]`, `[targets]`

### What Stays
- `--plot` / `--plot-save` flags on individual commands (these call MEMs `plot_result()` / `save_plot()` directly)
- `_maybe_plot()` helper in shared.jl (inline plotting, no storage)

## Part 2: New Feature Adoption (MEMs v0.2.3-v0.2.4)

### 2a. Nowcasting — New Top-Level Command

New file: `src/commands/nowcast.jl`

```
nowcast
├── dfm       --data, --monthly-vars, --quarterly-vars, --factors, --lags, --max-iter, --plot, --plot-save
├── bvar      --data, --monthly-vars, --quarterly-vars, --lags
├── bridge    --data, --monthly-vars, --quarterly-vars, --lag-m, --lag-q, --lag-y
├── news      --data-new, --data-old, --method, --target-period, --target-var, --plot, --plot-save
└── forecast  --data, --method, --horizons, --monthly-vars, --quarterly-vars, --plot, --plot-save
```

### 2b. Cumulative IRF
- `--cumulative` flag on `irf var`, `irf bvar`, `irf lp`
- Calls `cumulative_irf()` on the ImpulseResponse/BayesianImpulseResponse/LPImpulseResponse result

### 2c. Sign Identified Set
- `--identified-set` flag on `irf var` when `--id=sign`
- Returns `SignIdentifiedSet` with bounds and median instead of point estimate

### 2d. State-Space Beveridge-Nelson
- `--method` option on `filter bn` (values: `arima` (default), `statespace`)

### 2e. Bootstrap VAR Forecast CI
- `--ci-method` option on `forecast var` (values: `analytical` (default), `bootstrap`)

### 2f. Stationary-Only Filter
- `--stationary-only` flag on `irf var/bvar`, `fevd var/bvar`, `hd var/bvar` (bootstrap/Bayesian modes)

### 2g. Data Balance
- `data balance` subcommand using `balance_panel()`
- Options: `--data`, `--method` (dfm), `--factors`, `--lags`, `--output`

### 2h. Date Indexing
- `data load --dates=<column>` option — calls `set_dates!(ts, dates)` after loading

### 2i. Internal Type Adoption
- Use `VARForecast` / `BVARForecast` structured types in `forecast var/bvar` handlers
- Simplify `data transform` to use 1-arg `apply_tcode(data)` when tcodes are stored

## Updated Command Tree

```
friedman (v0.2.2, wraps MEMs v0.2.4)
├── estimate     17 leaves (unchanged)
├── test         16+4 leaves (unchanged)
├── irf          5 leaves (+--cumulative, +--identified-set, +--stationary-only)
├── fevd         5 leaves (+--stationary-only)
├── hd           4 leaves (+--stationary-only)
├── forecast     13 leaves (+--ci-method on var)
├── predict      12 leaves (unchanged)
├── residuals    12 leaves (unchanged)
├── filter       5 leaves (+--method on bn)
├── data         9 leaves (+balance)
└── nowcast      5 leaves (NEW)

11 top-level commands, ~103 subcommands
```

## Testing

### Removed
- All storage tests (save/load/list/rename/resolve_stored_tags/serialize_model)
- All `--from-tag` handler tests
- Tests for list, rename, project, plot commands
- Storage-related mocks

### Added
- Mock types: NowcastDFM, NowcastBVAR, NowcastBridge, NowcastResult, NowcastNews, SignIdentifiedSet, VARForecast, BVARForecast
- Mock functions: nowcast_dfm, nowcast_bvar, nowcast_bridge, nowcast, nowcast_news, balance_panel, cumulative_irf, irf_bounds, irf_median, set_dates!, dates
- Handler tests for: nowcast (5 subcommands), data balance, --cumulative, --identified-set, --stationary-only, --ci-method, --method=statespace
- Updated existing tests that referenced removed options/flags
