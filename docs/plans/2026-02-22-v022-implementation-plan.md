# Friedman-cli v0.2.2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove BSON storage system and adopt all MacroEconometricModels.jl v0.2.4 features.

**Architecture:** Two-phase approach — first strip storage (reducing 14→11 commands), then add new features (nowcasting, cumulative IRF, sign identified set, statespace BN, bootstrap forecast CI, data balance/dates). Storage removal is mechanical deletion; feature adoption adds one new command file and flags to existing commands.

**Tech Stack:** Julia 1.12, MacroEconometricModels.jl v0.2.4, existing CLI framework (types.jl/parser.jl/dispatch.jl/help.jl)

---

## Phase 1: Storage & BSON Removal

### Task 1: Delete storage-only files and strip Friedman.jl

**Files:**
- Delete: `src/storage.jl`, `src/settings.jl`
- Delete: `src/commands/list.jl`, `src/commands/rename.jl`, `src/commands/project.jl`, `src/commands/plot.jl`
- Modify: `src/Friedman.jl`
- Modify: `Project.toml`

**Step 1: Delete the 6 storage/settings/command files**

```bash
rm src/storage.jl src/settings.jl src/commands/list.jl src/commands/rename.jl src/commands/project.jl src/commands/plot.jl
```

**Step 2: Edit `src/Friedman.jl`**

Remove these lines:
- Line 19: remove `BSON` from the `using` statement (change `using CSV, DataFrames, PrettyTables, JSON3, TOML, BSON, Dates` → `using CSV, DataFrames, PrettyTables, JSON3, TOML, Dates`)
- Lines 35-36: remove `include("storage.jl")` and `include("settings.jl")`
- Lines 51-54: remove `include("commands/list.jl")`, `include("commands/rename.jl")`, `include("commands/project.jl")`, `include("commands/plot.jl")`
- Lines 76-79 in `build_app()`: remove `"list"`, `"rename"`, `"project"`, `"plot"` entries from `root_cmds` Dict
- Lines 95-104 in `main()`: remove `init_settings!()` call, `resolve_stored_tags(args)` call, and the `"project"` → `"show"` default handling block

The cleaned `main()` should be:
```julia
function main(args::Vector{String}=ARGS)
    app = build_app()
    try
        dispatch(app, args)
    catch e
        if e isa ParseError || e isa DispatchError
            printstyled(stderr, "Error: "; bold=true, color=:red)
            println(stderr, e.message)
            exit(1)
        else
            rethrow()
        end
    end
end
```

**Step 3: Edit `Project.toml`**

Remove BSON from `[deps]`, `[compat]`, `[extras]`, `[targets]`:
- `[deps]`: remove `BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"` line
- `[compat]`: remove `BSON = "0.3.9"` line
- `[extras]`: remove `BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"` line
- `[targets]`: remove `"BSON"` from `test = [...]` list

**Step 4: Commit**

```bash
git add -A && git commit -m "Remove storage infrastructure: delete storage.jl, settings.jl, list/rename/project/plot commands, strip BSON dep"
```

---

### Task 2: Strip shared.jl — remove _resolve_from_tag and serialize_model references

**Files:**
- Modify: `src/commands/shared.jl`

**Step 1: Remove `_resolve_from_tag()` function (lines 617-634)**

Delete the entire function definition from the `_resolve_from_tag` docstring through the closing `end` and trailing blank line.

**Step 2: Verify `_maybe_plot()` stays (lines 596-615)**

This function does NOT use storage — it only calls `plot_result()`, `save_plot()`, `display_plot()`. Keep it unchanged.

**Step 3: Commit**

```bash
git add src/commands/shared.jl && git commit -m "Remove _resolve_from_tag from shared.jl"
```

---

### Task 3: Strip estimate.jl — remove all storage_save_auto! calls

**Files:**
- Modify: `src/commands/estimate.jl`

**Step 1: Remove all `storage_save_auto!()` calls**

Pattern: Each handler ends with a `storage_save_auto!(...)` call spanning 2-3 lines. Remove these calls and the surrounding blank line before them. There are approximately 15+ such calls across all estimate handlers: `_estimate_var`, `_estimate_bvar`, all LP variants, `_estimate_arima`, `_estimate_gmm`, all factor model handlers, all volatility handlers, `_estimate_fastica`, `_estimate_ml`, `_estimate_vecm`, `_estimate_pvar`.

For example, in `_estimate_bvar` (around lines 334-336), remove:
```julia
    storage_save_auto!("bvar", serialize_model(model),
        Dict{String,Any}("command" => "estimate bvar", "data" => data, "lags" => p,
                          "draws" => draws, "sampler" => sampler, "method" => method))
```

Apply the same pattern to every handler in the file. Search for `storage_save_auto!` and remove every occurrence plus its Dict arguments.

**Step 2: Commit**

```bash
git add src/commands/estimate.jl && git commit -m "Strip storage_save_auto! from all estimate handlers"
```

---

### Task 4: Strip irf.jl, fevd.jl, hd.jl — remove --from-tag, storage_save_auto!

**Files:**
- Modify: `src/commands/irf.jl`, `src/commands/fevd.jl`, `src/commands/hd.jl`

**For each file, apply these 3 patterns:**

**Pattern A: Remove `--from-tag` Option from LeafCommand declarations.**
Find and delete lines like:
```julia
            Option("from-tag"; type=String, default="", description="Load model from stored tag"),
```

**Pattern B: Remove `from_tag` from handler signatures and the guard/resolve block.**
In each handler function signature, remove `, from_tag::String=""`.
Remove the 2 guard blocks:
```julia
    if isempty(data) && isempty(from_tag)
        error("Either <data> argument or --from-tag option is required")
    end
    if !isempty(from_tag) && isempty(data)
        data, _ = _resolve_from_tag(from_tag)
    end
```
The `data` argument becomes required again (it was `required=false` only to support `--from-tag`). Change the Argument declaration from `required=false, default=""` to `required=true` (or remove the `required` and `default` kwargs since `required=true` is the default).

**Pattern C: Remove `storage_save_auto!()` calls.**
Same as Task 3 — find and remove every `storage_save_auto!(...)` block.

**File-specific counts:**
- `irf.jl`: 5 `--from-tag` Options, 5 handler `from_tag` blocks, ~7 `storage_save_auto!` calls (var has 3 for arias/uhlig/main, bvar/lp/vecm/pvar have 1 each)
- `fevd.jl`: 5 `--from-tag` Options, 5 handler blocks, ~7 `storage_save_auto!` calls
- `hd.jl`: 4 `--from-tag` Options, 4 handler blocks, ~6 `storage_save_auto!` calls

**Make `data` required in the Argument declarations:**
Change from:
```julia
        args=[Argument("data"; required=false, default="", description="Path to CSV data file")],
```
To:
```julia
        args=[Argument("data"; description="Path to CSV data file")],
```

**Step 4: Commit**

```bash
git add src/commands/irf.jl src/commands/fevd.jl src/commands/hd.jl && git commit -m "Strip --from-tag and storage_save_auto! from irf/fevd/hd"
```

---

### Task 5: Strip forecast.jl, predict.jl, residuals.jl, filter.jl, test.jl

**Files:**
- Modify: `src/commands/forecast.jl`, `src/commands/predict.jl`, `src/commands/residuals.jl`, `src/commands/filter.jl`, `src/commands/test.jl`

**Apply the same 3 patterns as Task 4:**

**forecast.jl:** 13 `--from-tag` Options, 13 handler blocks, 13 `storage_save_auto!` calls. Make `data` required in all 13 LeafCommand Argument declarations.

**predict.jl:** 12 `--from-tag` Options, 12 handler blocks, 12 `storage_save_auto!` calls (these use Dict-based storage, not serialize_model). Make `data` required.

**residuals.jl:** 12 `--from-tag` Options, 12 handler blocks, 12 `storage_save_auto!` calls. Make `data` required.

**filter.jl:** No `--from-tag`, but 5 `storage_save_auto!` calls (one per filter handler: hp, hamilton, bn, bk, bhp). Remove them.

**test.jl:** 2 `_resolve_from_tag` calls in `_test_lr` and `_test_lm` handlers (for `--from-tag` support). These handlers take stored model tags for comparison. Remove the `--from-tag` Option, `from_tag` kwarg, and the resolve block from both. The `lr` and `lm` test commands should require explicit re-estimation — read the test handler code to confirm what changes are needed.

**Commit:**

```bash
git add src/commands/forecast.jl src/commands/predict.jl src/commands/residuals.jl src/commands/filter.jl src/commands/test.jl && git commit -m "Strip storage from forecast/predict/residuals/filter/test"
```

---

### Task 6: Update tests — remove storage/settings/plot tests

**Files:**
- Modify: `test/runtests.jl`
- Modify: `test/test_commands.jl`
- Modify: `test/mocks.jl`

**Step 1: Edit `test/runtests.jl`**

Remove these @testset blocks:
- "List, rename, project structure (action-first)" (lines ~2535-2671)
- "Tag resolution" (lines ~2673-2728)

Update any remaining tests that reference `--from-tag` or storage functions. The command structure tests for irf/fevd/hd/forecast/predict/residuals may test `--from-tag` option parsing — remove those specific test cases.

**Step 2: Edit `test/test_commands.jl`**

Remove these @testset blocks:
- "List, rename, project handlers" (lines ~2727-2816)
- "Storage operations" (lines ~2822-2965)
- "Settings operations" (lines ~2971-3066)
- "Plot Support" (lines ~4897-5559) — entire block including all `_deserialize_for_plot` tests and round-trip storage tests
- Any individual `--from-tag` test cases within handler tests

**Step 3: Edit `test/mocks.jl`**

Remove:
- `PlotOutput` struct (line ~1314-1316)
- `plot_result`, `save_plot`, `display_plot` mock functions (lines ~1318-1320)
- Exports for plot functions (line ~1322)

Note: Keep ALL filter/volatility/factor result type mocks — they're still used by the `--plot`/`--plot-save` handler tests that remain.

**Step 4: Run tests and fix any breakage**

```bash
julia --project test/runtests.jl
```

Fix any remaining references to deleted functions (`storage_save_auto!`, `serialize_model`, `_resolve_from_tag`, `storage_load`, `storage_list`, `storage_rename!`, `resolve_stored_tags`, `init_settings!`, `friedman_home`, `register_project!`, etc.) in test files.

**Step 5: Commit**

```bash
git add test/runtests.jl test/test_commands.jl test/mocks.jl && git commit -m "Remove storage/settings/plot tests, update command tests"
```

---

### Task 7: Verify clean compilation and all tests pass

**Step 1: Run full test suite**

```bash
julia --project test/runtests.jl
```

Expected: All remaining tests pass. No references to deleted storage functions.

**Step 2: Grep for any remaining storage references**

```bash
grep -rn "storage_save\|storage_load\|storage_list\|storage_rename\|serialize_model\|_resolve_from_tag\|resolve_stored_tags\|init_settings\|register_project\|friedman_home\|\.friedmanlog" src/ test/
```

Expected: Zero matches (except possibly in comments, which should also be cleaned up).

**Step 3: Grep for remaining BSON references**

```bash
grep -rn "BSON\|bson" src/ test/ Project.toml
```

Expected: Zero matches.

**Step 4: Commit if any fixes were needed**

```bash
git add -A && git commit -m "Fix remaining storage references after removal"
```

---

## Phase 2: New Features

### Task 8: Add `--cumulative` flag to `irf var`, `irf bvar`, `irf lp`

**Files:**
- Modify: `src/commands/irf.jl`
- Modify: `test/mocks.jl` (add `cumulative_irf` mock)
- Modify: `test/test_commands.jl` (add tests)

**Step 1: Add mock `cumulative_irf` to `test/mocks.jl`**

Add near the existing IRF mocks:
```julia
# Cumulative IRF — cumsum along horizon dimension
cumulative_irf(irf_result::ImpulseResponse) = ImpulseResponse(
    cumsum(irf_result.values; dims=1), irf_result.ci_lower, irf_result.ci_upper,
    irf_result.horizon, irf_result.variables, irf_result.shocks)
cumulative_irf(irf_result::BayesianImpulseResponse) = BayesianImpulseResponse(
    cumsum(irf_result.quantiles; dims=1), cumsum(irf_result.mean; dims=1),
    irf_result.quantile_levels)
cumulative_irf(irf_result::LPImpulseResponse) = LPImpulseResponse(
    cumsum(irf_result.values; dims=1), irf_result.ci_lower, irf_result.ci_upper,
    irf_result.se, irf_result.horizon)
```

Export `cumulative_irf`.

**Step 2: Write failing tests in `test/test_commands.jl`**

Add tests for `--cumulative` flag:
```julia
@testset "_irf_var — --cumulative flag" begin
    cd(mktempdir()) do
        write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        output = capture_stdout(() -> Friedman._irf_var(data="data.csv", cumulative=true))
        @test contains(output, "Cumulative")
    end
end
```

Similar tests for `_irf_bvar` and `_irf_lp` with `cumulative=true`.

**Step 3: Run tests to verify they fail**

```bash
julia --project test/runtests.jl
```

**Step 4: Add `--cumulative` flag to LeafCommand declarations in `register_irf_commands!()`**

For `irf_var`, `irf_bvar`, `irf_lp`, add to the `flags` array:
```julia
        flags=[
            Flag("plot"; description="Open interactive plot in browser"),
            Flag("cumulative"; description="Compute cumulative IRFs (for differenced data)"),
        ],
```

**Step 5: Add `cumulative` kwarg to handler functions**

In `_irf_var`, `_irf_bvar`, `_irf_lp`:
- Add `cumulative::Bool=false` to the function signature
- After computing the IRF result, add:
```julia
    if cumulative
        irf_result = cumulative_irf(irf_result)
        println("  (Cumulated)")
    end
```
- Update the title to include "(cumulative)" when the flag is set

**Step 6: Run tests**

```bash
julia --project test/runtests.jl
```

**Step 7: Commit**

```bash
git add src/commands/irf.jl test/mocks.jl test/test_commands.jl && git commit -m "Add --cumulative flag to irf var/bvar/lp"
```

---

### Task 9: Add `--identified-set` flag to `irf var` (sign restrictions)

**Files:**
- Modify: `src/commands/irf.jl`
- Modify: `test/mocks.jl` (add `SignIdentifiedSet`, `irf_bounds`, `irf_median` mocks)
- Modify: `test/test_commands.jl`

**Step 1: Add mocks to `test/mocks.jl`**

```julia
struct SignIdentifiedSet{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    n_accepted::Int
    n_total::Int
    acceptance_rate::T
    variables::Vector{String}
    shocks::Vector{String}
end

irf_bounds(s::SignIdentifiedSet; quantiles=[0.16, 0.84]) = (
    zeros(size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4)),
    ones(size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4))
)
irf_median(s::SignIdentifiedSet) = ones(size(s.irf_draws, 2), size(s.irf_draws, 3), size(s.irf_draws, 4)) .* 0.5
```

Update the mock `identify_sign` to accept `store_all` kwarg:
```julia
function identify_sign(model::VARModel, horizon, check_func; max_draws=1000, store_all=false)
    n = size(model.Y, 2)
    Q = Matrix{Float64}(I(n))
    irf_vals = ones(horizon + 1, n, n) * 0.1
    if store_all
        return SignIdentifiedSet{Float64}([Q], reshape(irf_vals, 1, size(irf_vals)...),
            1, max_draws, 1.0 / max_draws, ["y$i" for i in 1:n], ["s$i" for i in 1:n])
    end
    return Q, irf_vals
end
```

Export: `SignIdentifiedSet`, `irf_bounds`, `irf_median`.

**Step 2: Write failing test**

```julia
@testset "_irf_var — --identified-set with sign id" begin
    cd(mktempdir()) do
        write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        # Write a TOML config with sign restrictions
        write("config.toml", "[identification]\nmethod = \"sign\"\n[identification.sign_matrix]\nmatrix = [[1, -1], [0, 1]]\nhorizons = [0, 1]\n")
        output = capture_stdout(() -> Friedman._irf_var(
            data="data.csv", id="sign", config="config.toml", identified_set=true))
        @test contains(output, "Identified Set")
    end
end
```

**Step 3: Implement in `_irf_var`**

Add `identified_set::Bool=false` to handler signature and `Flag("identified-set"; ...)` to LeafCommand.

In the sign identification branch of `_irf_var`, when `identified_set` is true:
```julia
    if identified_set && id == "sign"
        set = identify_sign(model, horizons, check_func; max_draws=replications, store_all=true)
        lower, upper = irf_bounds(set)
        med = irf_median(set)
        println("Sign-Identified Set: $(set.n_accepted)/$(set.n_total) accepted ($(round(set.acceptance_rate*100; digits=1))%)")
        # Build output table with median + bounds
        irf_df = DataFrame()
        irf_df.horizon = 0:horizons
        for (vi, vname) in enumerate(varnames)
            irf_df[!, vname] = med[:, vi, shock]
            irf_df[!, "$(vname)_lower"] = lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = upper[:, vi, shock]
        end
        output_result(irf_df; format=Symbol(format), output=output,
                      title="IRF Identified Set (sign, shock=$shock_name)")
        return
    end
```

**Step 4: Run tests, commit**

```bash
git add src/commands/irf.jl test/mocks.jl test/test_commands.jl && git commit -m "Add --identified-set flag for sign-restricted IRF"
```

---

### Task 10: Add `--stationary-only` flag to `irf var`

**Files:**
- Modify: `src/commands/irf.jl`
- Modify: `test/test_commands.jl`

**Note:** `stationary_only` is ONLY supported on `irf(::VARModel)` with `ci_type=:bootstrap` in MEMs. Do NOT add it to fevd/hd/bvar.

**Step 1: Write failing test**

```julia
@testset "_irf_var — --stationary-only flag" begin
    cd(mktempdir()) do
        write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        output = capture_stdout(() -> Friedman._irf_var(
            data="data.csv", ci="bootstrap", stationary_only=true))
        @test contains(output, "stationary")  # or just verify no error
    end
end
```

**Step 2: Add flag to LeafCommand and handler**

In `register_irf_commands!()`, add to `irf_var` flags:
```julia
    Flag("stationary-only"; description="Filter non-stationary bootstrap draws"),
```

In `_irf_var` handler signature, add `stationary_only::Bool=false`.

Pass it through to the `irf()` call:
```julia
    irf_result = irf(model, horizons; method=method,
        check_func=check_func, narrative_check=narrative_check,
        ci_type=Symbol(ci), reps=replications, conf_level=0.95,
        stationary_only=stationary_only)
```

**Step 3: Run tests, commit**

```bash
git add src/commands/irf.jl test/test_commands.jl && git commit -m "Add --stationary-only flag to irf var"
```

---

### Task 11: Add `--method` option to `filter bn` (statespace BN)

**Files:**
- Modify: `src/commands/filter.jl`
- Modify: `test/test_commands.jl`

**Step 1: Write failing test**

```julia
@testset "_filter_bn — --method=statespace" begin
    cd(mktempdir()) do
        write("data.csv", "y\n" * join(["$(i + randn())" for i in 1:100], "\n") * "\n")
        output = capture_stdout(() -> Friedman._filter_bn(data="data.csv", method="statespace"))
        @test contains(output, "statespace") || contains(output, "BN")
    end
end
```

**Step 2: Add `--method` option to `filter bn` LeafCommand**

```julia
    Option("method"; type=String, default="arima", description="arima|statespace"),
```

**Step 3: Add `method` kwarg to `_filter_bn` handler**

Add `method::String="arima"` to the handler signature. Pass it through:
```julia
    result = beveridge_nelson(y; method=Symbol(method), p=p_order, q=q_order)
```

If `method == "statespace"`, may need to skip p/q options (they're for ARIMA only). Add guard:
```julia
    if method == "statespace"
        result = beveridge_nelson(y; method=:statespace)
    else
        result = beveridge_nelson(y; method=:arima, p=p_order, q=q_order)
    end
```

**Step 4: Run tests, commit**

```bash
git add src/commands/filter.jl test/test_commands.jl && git commit -m "Add --method=statespace to filter bn"
```

---

### Task 12: Add `--ci-method` option to `forecast var` (bootstrap CI)

**Files:**
- Modify: `src/commands/forecast.jl`
- Modify: `test/mocks.jl` (add `VARForecast` mock)
- Modify: `test/test_commands.jl`

**Step 1: Add `VARForecast` mock to `test/mocks.jl`**

```julia
struct VARForecast{T<:AbstractFloat}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
    conf_level::T
    varnames::Vector{String}
end
```

Update mock `forecast(::VARModel, h; ...)` to accept `ci_method` kwarg and return `VARForecast` when `ci_method=:bootstrap`.

Export `VARForecast`.

**Step 2: Write failing test**

```julia
@testset "_forecast_var — --ci-method=bootstrap" begin
    cd(mktempdir()) do
        write("data.csv", "a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n7.0,8.0\n9.0,10.0\n")
        output = capture_stdout(() -> Friedman._forecast_var(
            data="data.csv", ci_method="bootstrap"))
        @test contains(output, "bootstrap") || contains(output, "Forecast")
    end
end
```

**Step 3: Add `--ci-method` option to `forecast var` LeafCommand**

```julia
    Option("ci-method"; type=String, default="analytical", description="analytical|bootstrap"),
```

**Step 4: Modify `_forecast_var` handler**

Add `ci_method::String="analytical"` to signature.

When `ci_method == "bootstrap"`, use MEMs `forecast()` function instead of manual analytical CIs:
```julia
    if ci_method == "bootstrap"
        fc_result = forecast(model, horizons; ci_method=:bootstrap,
                             reps=500, conf_level=confidence)
        fc_df = DataFrame()
        fc_df.horizon = 1:horizons
        for (vi, vname) in enumerate(varnames)
            fc_df[!, vname] = fc_result.forecast[:, vi]
            fc_df[!, "$(vname)_lower"] = fc_result.ci_lower[:, vi]
            fc_df[!, "$(vname)_upper"] = fc_result.ci_upper[:, vi]
        end
        output_result(fc_df; format=Symbol(format), output=output,
                      title="VAR($p) Forecast (h=$horizons, bootstrap $(Int(round(confidence*100)))% CI)")
        return
    end
    # ... existing analytical CI code follows ...
```

**Step 5: Run tests, commit**

```bash
git add src/commands/forecast.jl test/mocks.jl test/test_commands.jl && git commit -m "Add --ci-method=bootstrap to forecast var"
```

---

### Task 13: Add `data balance` subcommand and `data load --dates`

**Files:**
- Modify: `src/commands/data.jl`
- Modify: `test/mocks.jl` (add `balance_panel`, `set_dates!`, `dates` mocks)
- Modify: `test/test_commands.jl`

**Step 1: Add mocks**

In `test/mocks.jl`, add:
```julia
function balance_panel(ts::TimeSeriesData; method::Symbol=:dfm, r::Int=3, p::Int=2)
    return ts  # mock: return unchanged
end

function set_dates!(ts::TimeSeriesData, dt::Vector{String})
    return ts  # mock: return unchanged
end

dates(ts::TimeSeriesData) = String[]
```

Export `balance_panel`, `set_dates!`, `dates`.

**Step 2: Write failing tests**

```julia
@testset "_data_balance" begin
    cd(mktempdir()) do
        write("data.csv", "a,b\n1.0,2.0\n3.0,NaN\n5.0,6.0\n")
        output = capture_stdout(() -> Friedman._data_balance(data="data.csv"))
        @test contains(output, "Balanc")
    end
end

@testset "_data_load — --dates option" begin
    cd(mktempdir()) do
        write("data.csv", "date,a,b\n2020Q1,1.0,2.0\n2020Q2,3.0,4.0\n")
        output = capture_stdout(() -> Friedman._data_load(
            name="", path="data.csv", dates="date"))
        @test contains(output, "date") || contains(output, "Date")
    end
end
```

**Step 3: Implement `_data_balance` handler**

In `src/commands/data.jl`, add handler:
```julia
function _data_balance(; data::String, method::String="dfm", factors::Int=3,
                        lags::Int=2, output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)

    println("Balancing panel via $(method): $(length(varnames)) variables")
    println()

    ts = TimeSeriesData(Y; varnames=varnames)
    balanced = balance_panel(ts; method=Symbol(method), r=factors, p=lags)

    result_df = DataFrame(to_matrix(balanced), varnames(balanced))
    output_result(result_df; format=Symbol(format), output=output,
                  title="Balanced Panel (method=$method, r=$factors, p=$lags)")
end
```

**Step 4: Register in `register_data_commands!()`**

Add a new LeafCommand `"balance"`:
```julia
    data_balance = LeafCommand("balance", _data_balance;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("method"; type=String, default="dfm", description="dfm"),
            Option("factors"; short="r", type=Int, default=3, description="Number of factors"),
            Option("lags"; short="p", type=Int, default=2, description="Factor VAR lags"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Balance panel with missing data via DFM imputation")
```

Add `"balance" => data_balance` to the subcmds Dict.

**Step 5: Add `--dates` option to `data load`**

In the `data_load` LeafCommand, add:
```julia
    Option("dates"; type=String, default="", description="Column name for date labels"),
```

In `_data_load` handler, add `dates::String=""` to the signature. After creating `TimeSeriesData`, call:
```julia
    if !isempty(dates_col)
        date_values = string.(df[!, dates_col])
        set_dates!(ts, date_values)
    end
```

Note: The kwarg name will be `dates` (since `--dates` maps to `dates` kwarg after hyphen-to-underscore). Check that it doesn't conflict with existing kwargs.

**Step 6: Run tests, commit**

```bash
git add src/commands/data.jl test/mocks.jl test/test_commands.jl && git commit -m "Add data balance subcommand and data load --dates option"
```

---

### Task 14: Add nowcast command group

**Files:**
- Create: `src/commands/nowcast.jl`
- Modify: `src/Friedman.jl` (add include + register)
- Modify: `test/mocks.jl` (add nowcast mocks)
- Modify: `test/test_commands.jl` (add tests)

**Step 1: Add nowcast mocks to `test/mocks.jl`**

```julia
# ── Nowcast Types ────────────────────────────────────────

abstract type AbstractNowcastModel end

struct NowcastDFM{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}; F::Matrix{T}; C::Matrix{T}; A::Matrix{T}; Q::Matrix{T}; R::Matrix{T}
    Mx::Vector{T}; Wx::Vector{T}; Z_0::Vector{T}; V_0::Matrix{T}
    r::Int; p::Int; blocks::Matrix{Int}; loglik::T; n_iter::Int
    nM::Int; nQ::Int; idio::Symbol; data::Matrix{T}
end

struct NowcastBVAR{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}; beta::Matrix{T}; sigma::Matrix{T}
    lambda::T; theta::T; miu::T; alpha::T; lags::Int; loglik::T
    nM::Int; nQ::Int; data::Matrix{T}
end

struct NowcastBridge{T<:AbstractFloat} <: AbstractNowcastModel
    X_sm::Matrix{T}; Y_nowcast::Vector{T}; Y_individual::Matrix{T}; n_equations::Int
    coefficients::Vector{Vector{T}}; nM::Int; nQ::Int; lagM::Int; lagQ::Int; lagY::Int
    data::Matrix{T}
end

struct NowcastResult{T<:AbstractFloat}
    model::AbstractNowcastModel; X_sm::Matrix{T}; target_index::Int
    nowcast::T; forecast::T; method::Symbol
end

struct NowcastNews{T<:AbstractFloat}
    old_nowcast::T; new_nowcast::T; impact_news::Vector{T}; impact_revision::T
    impact_reestimation::T; group_impacts::Vector{T}; variable_names::Vector{String}
end

# Convenience constructors for mocks
function _mock_nowcast_dfm(Y, nM, nQ; r=2, p=1, kwargs...)
    T_obs, N = size(Y)
    state_dim = r * p
    NowcastDFM{Float64}(
        copy(Y), randn(T_obs, state_dim), randn(N, state_dim), randn(state_dim, state_dim),
        Matrix{Float64}(I(state_dim)), Matrix{Float64}(I(N)),
        zeros(N), ones(N), zeros(state_dim), Matrix{Float64}(I(state_dim)),
        r, p, ones(Int, N, 1), -100.0, 50, nM, nQ, :ar1, copy(Y))
end

function nowcast_dfm(Y, nM, nQ; r=2, p=1, idio=:ar1, blocks=nothing, max_iter=100, thresh=1e-4)
    _mock_nowcast_dfm(Y, nM, nQ; r=r, p=p)
end

function nowcast_bvar(Y, nM, nQ; lags=5, kwargs...)
    T_obs, N = size(Y)
    NowcastBVAR{Float64}(copy(Y), randn(N*lags+1, N), Matrix{Float64}(I(N)),
        0.2, 1.0, 1.0, 2.0, lags, -100.0, nM, nQ, copy(Y))
end

function nowcast_bridge(Y, nM, nQ; lagM=1, lagQ=1, lagY=1)
    T_obs, N = size(Y)
    nQ_actual = max(nQ, 1)
    NowcastBridge{Float64}(copy(Y), randn(nQ_actual), randn(nQ_actual, nM),
        nM, [randn(3) for _ in 1:nM], nM, nQ, lagM, lagQ, lagY, copy(Y))
end

function nowcast(model::AbstractNowcastModel; target_var=nothing)
    idx = isnothing(target_var) ? size(model.data, 2) : target_var
    NowcastResult{Float64}(model, model.X_sm, idx, 1.5, 1.2, :dfm)
end

function nowcast_news(X_new, X_old, model, target_period; target_var=size(X_new, 2), groups=nothing)
    N = size(X_new, 2)
    NowcastNews{Float64}(1.0, 1.5, randn(N), 0.1, 0.05,
        isnothing(groups) ? randn(1) : randn(length(unique(groups))),
        ["var$i" for i in 1:N])
end

function forecast(model::AbstractNowcastModel, h; target_var=nothing)
    N = size(model.data, 2)
    return randn(h, N)
end
```

Export all: `AbstractNowcastModel`, `NowcastDFM`, `NowcastBVAR`, `NowcastBridge`, `NowcastResult`, `NowcastNews`, `nowcast_dfm`, `nowcast_bvar`, `nowcast_bridge`, `nowcast`, `nowcast_news`.

**Step 2: Write failing tests in `test/test_commands.jl`**

```julia
@testset "Nowcast commands" begin
    @testset "register_nowcast_commands!" begin
        node = Friedman.register_nowcast_commands!()
        @test isa(node, Friedman.NodeCommand)
        @test haskey(node.subcmds, "dfm")
        @test haskey(node.subcmds, "bvar")
        @test haskey(node.subcmds, "bridge")
        @test haskey(node.subcmds, "news")
        @test haskey(node.subcmds, "forecast")
    end

    @testset "_nowcast_dfm" begin
        cd(mktempdir()) do
            write("data.csv", join(["m1,m2,m3,q1\n"; ["$(rand()),$(rand()),$(rand()),$(rand())\n" for _ in 1:50]], ""))
            output = capture_stdout(() -> Friedman._nowcast_dfm(
                data="data.csv", monthly_vars=3, quarterly_vars=1))
            @test contains(output, "Nowcast") || contains(output, "DFM")
        end
    end

    @testset "_nowcast_bvar" begin
        cd(mktempdir()) do
            write("data.csv", join(["m1,m2,q1\n"; ["$(rand()),$(rand()),$(rand())\n" for _ in 1:50]], ""))
            output = capture_stdout(() -> Friedman._nowcast_bvar(
                data="data.csv", monthly_vars=2, quarterly_vars=1))
            @test contains(output, "Nowcast") || contains(output, "BVAR")
        end
    end

    @testset "_nowcast_bridge" begin
        cd(mktempdir()) do
            write("data.csv", join(["m1,m2,q1\n"; ["$(rand()),$(rand()),$(rand())\n" for _ in 1:50]], ""))
            output = capture_stdout(() -> Friedman._nowcast_bridge(
                data="data.csv", monthly_vars=2, quarterly_vars=1))
            @test contains(output, "Nowcast") || contains(output, "Bridge")
        end
    end

    @testset "_nowcast_news" begin
        cd(mktempdir()) do
            write("data_old.csv", join(["m1,m2,q1\n"; ["$(rand()),$(rand()),$(rand())\n" for _ in 1:50]], ""))
            write("data_new.csv", join(["m1,m2,q1\n"; ["$(rand()),$(rand()),$(rand())\n" for _ in 1:52]], ""))
            output = capture_stdout(() -> Friedman._nowcast_news(
                data_new="data_new.csv", data_old="data_old.csv",
                method="dfm", monthly_vars=2, quarterly_vars=1, target_period=52))
            @test contains(output, "News") || contains(output, "impact")
        end
    end

    @testset "_nowcast_forecast" begin
        cd(mktempdir()) do
            write("data.csv", join(["m1,m2,q1\n"; ["$(rand()),$(rand()),$(rand())\n" for _ in 1:50]], ""))
            output = capture_stdout(() -> Friedman._nowcast_forecast(
                data="data.csv", method="dfm", monthly_vars=2, quarterly_vars=1, horizons=4))
            @test contains(output, "Forecast") || contains(output, "nowcast")
        end
    end
end
```

**Step 3: Create `src/commands/nowcast.jl`**

```julia
# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# Nowcast commands: dfm, bvar, bridge, news, forecast

function register_nowcast_commands!()
    nc_dfm = LeafCommand("dfm", _nowcast_dfm;
        args=[Argument("data"; description="Path to CSV data file (monthly+quarterly, NaN for missing)")],
        options=[
            Option("monthly-vars"; short="m", type=Int, default=0, description="Number of monthly variables (first nM columns)"),
            Option("quarterly-vars"; short="q", type=Int, default=0, description="Number of quarterly variables (last nQ columns)"),
            Option("factors"; short="r", type=Int, default=2, description="Number of factors"),
            Option("lags"; short="p", type=Int, default=1, description="Factor VAR lags"),
            Option("max-iter"; type=Int, default=100, description="Max EM iterations"),
            Option("idio"; type=String, default="ar1", description="ar1|iid"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Nowcast via Dynamic Factor Model (Banbura & Modugno 2014)")

    nc_bvar = LeafCommand("bvar", _nowcast_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; short="m", type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; short="q", type=Int, default=0, description="Number of quarterly variables"),
            Option("lags"; short="p", type=Int, default=5, description="Number of lags"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Nowcast via large Bayesian VAR (Cimadomo et al. 2022)")

    nc_bridge = LeafCommand("bridge", _nowcast_bridge;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; short="m", type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; short="q", type=Int, default=0, description="Number of quarterly variables"),
            Option("lag-m"; type=Int, default=1, description="Monthly indicator lags"),
            Option("lag-q"; type=Int, default=1, description="Quarterly indicator lags"),
            Option("lag-y"; type=Int, default=1, description="Autoregressive lags"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Nowcast via bridge equations (Banbura et al. 2023)")

    nc_news = LeafCommand("news", _nowcast_news;
        args=[],
        options=[
            Option("data-new"; type=String, default="", description="Path to updated data CSV"),
            Option("data-old"; type=String, default="", description="Path to previous data CSV"),
            Option("method"; type=String, default="dfm", description="dfm|bvar|bridge"),
            Option("monthly-vars"; short="m", type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; short="q", type=Int, default=0, description="Number of quarterly variables"),
            Option("target-period"; type=Int, default=0, description="Target time period index"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="News decomposition: attribute nowcast revision to data releases")

    nc_forecast = LeafCommand("forecast", _nowcast_forecast;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("method"; type=String, default="dfm", description="dfm|bvar"),
            Option("monthly-vars"; short="m", type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; short="q", type=Int, default=0, description="Number of quarterly variables"),
            Option("horizons"; short="h", type=Int, default=4, description="Forecast horizon"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=all)"),
            Option("output"; short="o", type=String, default="", description="Export file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Multi-step nowcast forecast")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "dfm"      => nc_dfm,
        "bvar"     => nc_bvar,
        "bridge"   => nc_bridge,
        "news"     => nc_news,
        "forecast" => nc_forecast,
    )
    return NodeCommand("nowcast", subcmds, "Nowcasting (mixed-frequency real-time estimation)")
end

# ── DFM Nowcast ─────────────────────────────────────────

function _nowcast_dfm(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                       factors::Int=2, lags::Int=1, max_iter::Int=100,
                       idio::String="ar1",
                       output::String="", format::String="table",
                       plot::Bool=false, plot_save::String="")
    Y, varnames = load_multivariate_data(data)
    nM = monthly_vars > 0 ? monthly_vars : size(Y, 2) - quarterly_vars
    nQ = quarterly_vars > 0 ? quarterly_vars : size(Y, 2) - nM
    (nM + nQ != size(Y, 2)) && error("monthly-vars ($nM) + quarterly-vars ($nQ) must equal total variables ($(size(Y, 2)))")

    println("Nowcast DFM: nM=$nM, nQ=$nQ, r=$factors, p=$lags, idio=$idio")
    println("  Data: $(size(Y, 1)) periods, $(size(Y, 2)) variables")
    println()

    model = nowcast_dfm(Y, nM, nQ; r=factors, p=lags, idio=Symbol(idio), max_iter=max_iter)
    result = nowcast(model)

    _maybe_plot(result; plot=plot, plot_save=plot_save)

    println("  Current-quarter nowcast: $(round(result.nowcast; digits=4))")
    println("  Next-quarter forecast:   $(round(result.forecast; digits=4))")
    println("  Log-likelihood: $(round(model.loglik; digits=2))")
    println("  EM iterations: $(model.n_iter)")

    sm_df = DataFrame(model.X_sm, varnames)
    output_result(sm_df; format=Symbol(format), output=output,
                  title="Nowcast DFM — Smoothed Data (NaN filled)")
end

# ── BVAR Nowcast ────────────────────────────────────────

function _nowcast_bvar(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                        lags::Int=5, output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    nM = monthly_vars > 0 ? monthly_vars : size(Y, 2) - quarterly_vars
    nQ = quarterly_vars > 0 ? quarterly_vars : size(Y, 2) - nM
    (nM + nQ != size(Y, 2)) && error("monthly-vars ($nM) + quarterly-vars ($nQ) must equal total variables ($(size(Y, 2)))")

    println("Nowcast BVAR: nM=$nM, nQ=$nQ, lags=$lags")
    println()

    model = nowcast_bvar(Y, nM, nQ; lags=lags)
    result = nowcast(model)

    println("  Current-quarter nowcast: $(round(result.nowcast; digits=4))")
    println("  Next-quarter forecast:   $(round(result.forecast; digits=4))")
    println("  Log-likelihood: $(round(model.loglik; digits=2))")

    sm_df = DataFrame(model.X_sm, varnames)
    output_result(sm_df; format=Symbol(format), output=output,
                  title="Nowcast BVAR — Smoothed Data")
end

# ── Bridge Nowcast ──────────────────────────────────────

function _nowcast_bridge(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                          lag_m::Int=1, lag_q::Int=1, lag_y::Int=1,
                          output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    nM = monthly_vars > 0 ? monthly_vars : size(Y, 2) - quarterly_vars
    nQ = quarterly_vars > 0 ? quarterly_vars : size(Y, 2) - nM
    (nM + nQ != size(Y, 2)) && error("monthly-vars ($nM) + quarterly-vars ($nQ) must equal total variables ($(size(Y, 2)))")

    println("Nowcast Bridge: nM=$nM, nQ=$nQ, lagM=$lag_m, lagQ=$lag_q, lagY=$lag_y")
    println()

    model = nowcast_bridge(Y, nM, nQ; lagM=lag_m, lagQ=lag_q, lagY=lag_y)
    result = nowcast(model)

    println("  Current-quarter nowcast: $(round(result.nowcast; digits=4))")
    println("  Next-quarter forecast:   $(round(result.forecast; digits=4))")
    println("  Number of bridge equations: $(model.n_equations)")

    sm_df = DataFrame(model.X_sm, varnames)
    output_result(sm_df; format=Symbol(format), output=output,
                  title="Nowcast Bridge — Smoothed Data")
end

# ── News Decomposition ──────────────────────────────────

function _nowcast_news(; data_new::String="", data_old::String="",
                        method::String="dfm", monthly_vars::Int=0, quarterly_vars::Int=0,
                        target_period::Int=0, target_var::Int=0,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    isempty(data_new) && error("--data-new is required")
    isempty(data_old) && error("--data-old is required")

    Y_new, varnames = load_multivariate_data(data_new)
    Y_old, _ = load_multivariate_data(data_old)
    nM = monthly_vars > 0 ? monthly_vars : size(Y_new, 2) - quarterly_vars
    nQ = quarterly_vars > 0 ? quarterly_vars : size(Y_new, 2) - nM

    tp = target_period > 0 ? target_period : size(Y_new, 1)
    tv = target_var > 0 ? target_var : size(Y_new, 2)

    println("News Decomposition: method=$method, target_period=$tp, target_var=$tv")
    println()

    # Estimate model on old data
    est_model = if method == "dfm"
        nowcast_dfm(Y_old, nM, nQ)
    elseif method == "bvar"
        nowcast_bvar(Y_old, nM, nQ)
    else
        nowcast_bridge(Y_old, nM, nQ)
    end

    news = nowcast_news(Y_new, Y_old, est_model, tp; target_var=tv)

    _maybe_plot(news; plot=plot, plot_save=plot_save)

    println("  Old nowcast: $(round(news.old_nowcast; digits=4))")
    println("  New nowcast: $(round(news.new_nowcast; digits=4))")
    println("  Revision:    $(round(news.new_nowcast - news.old_nowcast; digits=4))")
    println()

    news_df = DataFrame(
        variable = news.variable_names,
        impact = round.(news.impact_news; digits=6),
    )
    output_result(news_df; format=Symbol(format), output=output,
                  title="News Decomposition — Per-Release Impact")
end

# ── Nowcast Forecast ────────────────────────────────────

function _nowcast_forecast(; data::String, method::String="dfm",
                            monthly_vars::Int=0, quarterly_vars::Int=0,
                            horizons::Int=4, target_var::Int=0,
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    Y, varnames = load_multivariate_data(data)
    nM = monthly_vars > 0 ? monthly_vars : size(Y, 2) - quarterly_vars
    nQ = quarterly_vars > 0 ? quarterly_vars : size(Y, 2) - nM

    println("Nowcast Forecast: method=$method, horizons=$horizons")
    println()

    est_model = if method == "dfm"
        nowcast_dfm(Y, nM, nQ)
    elseif method == "bvar"
        nowcast_bvar(Y, nM, nQ)
    else
        error("Forecast not supported for bridge method — use dfm or bvar")
    end

    tv = target_var > 0 ? target_var : nothing
    fc = forecast(est_model, horizons; target_var=tv)

    if fc isa AbstractVector
        fc_df = DataFrame(horizon=1:horizons, forecast=round.(fc; digits=6))
        output_result(fc_df; format=Symbol(format), output=output,
                      title="Nowcast Forecast (method=$method, h=$horizons, var=$(tv))")
    else
        fc_df = DataFrame()
        fc_df.horizon = 1:horizons
        for (vi, vname) in enumerate(varnames)
            fc_df[!, vname] = round.(fc[:, vi]; digits=6)
        end
        output_result(fc_df; format=Symbol(format), output=output,
                      title="Nowcast Forecast (method=$method, h=$horizons)")
    end
end
```

**Step 4: Register in `src/Friedman.jl`**

Add `include("commands/nowcast.jl")` after the other command includes (after `include("commands/data.jl")`).

Add `"nowcast" => register_nowcast_commands!()` to the `root_cmds` Dict in `build_app()`.

**Step 5: Run tests**

```bash
julia --project test/runtests.jl
```

**Step 6: Commit**

```bash
git add src/commands/nowcast.jl src/Friedman.jl test/mocks.jl test/test_commands.jl && git commit -m "Add nowcast command group: dfm, bvar, bridge, news, forecast"
```

---

## Phase 3: Finalize

### Task 15: Full test run, fix remaining issues

**Step 1: Run full test suite**

```bash
julia --project test/runtests.jl
```

**Step 2: Fix any failures**

Common issues to watch for:
- Tests that counted expected number of commands (14 → 11, but +nowcast = 12 if counted differently)
- Tests that listed expected subcommand names (list/rename/project/plot should be gone, nowcast should be present)
- Tests that checked `--from-tag` option parsing
- Tests that referenced `BSON` module
- Mock function signature mismatches

**Step 3: Grep for any leftover references**

Search for: `storage_save`, `storage_load`, `storage_list`, `storage_rename`, `serialize_model`, `_resolve_from_tag`, `resolve_stored_tags`, `init_settings`, `register_project`, `friedman_home`, `.friedmanlog`, `BSON`, `from_tag`, `from-tag`

**Step 4: Commit fixes**

```bash
git add -A && git commit -m "Fix test failures after v0.2.2 changes"
```

---

### Task 16: Update README and documentation

**Files:**
- Modify: `README.md`

**Step 1: Update command count and tree**

Update the command tree in README to show 11 top-level commands (estimate, test, irf, fevd, hd, forecast, predict, residuals, filter, data, nowcast) and ~103 subcommands. Remove references to list, rename, project, plot commands. Add nowcast to the tree.

**Step 2: Update feature list**

Add mentions of: nowcasting, cumulative IRF, sign identified set, statespace BN, bootstrap VAR forecast CI, data balance. Remove mentions of BSON storage, model persistence, tag-based workflows.

**Step 3: Commit**

```bash
git add README.md && git commit -m "Update README for v0.2.2: remove storage refs, add new features"
```

---

### Task 17: Update MEMORY.md

**Files:**
- Modify: `/Users/chung/.claude/projects/-Users-chung-Desktop-CODES-Friedman-cli/memory/MEMORY.md`

Update architecture section to reflect v0.2.2 changes:
- 11 top-level commands (removed list, rename, project, plot; added nowcast)
- No BSON storage, no settings, no tag resolution
- New flags: --cumulative, --identified-set, --stationary-only, --ci-method, --method on bn
- data: 9 leaves (+balance)
- nowcast: 5 leaves (dfm, bvar, bridge, news, forecast)
- MEMs v0.2.4 compat
