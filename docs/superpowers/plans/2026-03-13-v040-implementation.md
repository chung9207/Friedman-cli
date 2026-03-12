# Friedman-cli v0.4.0 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap MacroEconometricModels.jl v0.4.0 in Friedman-cli, adding DSGE HD, panel regression, spectral analysis, ordered/multinomial choice, data enhancements, and dependency updates.

**Architecture:** 6 independent feature areas added to the existing action-first CLI. Each area follows the established pattern: mock types → handler tests → LeafCommand registration → handler implementation → documentation. New `spectral` top-level command; all others extend existing commands.

**Tech Stack:** Julia 1.12, MacroEconometricModels.jl v0.4.0, CSV, DataFrames, PrettyTables v3, JSON3

**Spec:** `docs/superpowers/specs/2026-03-13-v040-mems-v040-design.md`

---

## Chunk 1: Version Bump & Dependencies

### Task 1: Version & Dependency Updates

**Files:**
- Modify: `Project.toml`
- Modify: `src/Friedman.jl:56`
- Modify: `src/cli/types.jl:111`
- Modify: `test/runtests.jl:394,513,1503`

- [ ] **Step 1: Update `Project.toml`**

Change version, MEMs compat, move FFTW from weakdeps to deps, add NonlinearSolve, fix PATHSolver UUID:

```toml
name = "Friedman"
uuid = "f7a1e3d0-2b4c-4e8f-9a6d-1c3b5e7f9a2d"
version = "0.4.0"

[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MacroEconometricModels = "14a6ec33-bcac-448e-845f-2fb6769698f1"
NonlinearSolve = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[weakdeps]
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
PATHSolver = "f5f7c340-0bb3-5c69-969a-41884d311d1b"

[compat]
FFTW = "1.5"
MacroEconometricModels = "0.4.0"
NonlinearSolve = "4"
julia = "1.12"

[extras]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
PrettyTables = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["Test", "CSV", "DataFrames", "JSON3", "PrettyTables"]
```

- [ ] **Step 2: Update version constant in `src/Friedman.jl:56`**

```julia
const FRIEDMAN_VERSION = v"0.4.0"
```

- [ ] **Step 3: Update Entry default version in `src/cli/types.jl:111`**

```julia
Entry(name::String, root::NodeCommand; version::VersionNumber=v"0.4.0") =
    Entry(name, root, version)
```

- [ ] **Step 4: Update version strings in `test/runtests.jl`**

Find all `v"0.3.5"` references (lines 394, 513, 1503) and replace with `v"0.4.0"`.

- [ ] **Step 5: Run tests to verify nothing broke**

Run: `julia --project test/runtests.jl`
Expected: All existing tests pass (version tests use new v"0.4.0").

- [ ] **Step 6: Commit**

```bash
git add Project.toml src/Friedman.jl src/cli/types.jl test/runtests.jl
git commit -m "chore: bump version to v0.4.0, update deps (FFTW, NonlinearSolve, PATHSolver UUID)"
```

---

## Chunk 2: Mock Types — DSGE HD & Spectral

All new mock types and functions go in `test/mocks.jl`. Add them **before** any handler tests.

### Task 2: DSGE HD Mock Dispatches

**Files:**
- Modify: `test/mocks.jl` (append near existing DSGE mocks section)

- [ ] **Step 1: Add `KalmanSmootherResult` mock type and DSGE HD dispatches**

Find the existing DSGE mock section (near `BayesianDSGE` definition) and append:

```julia
# --- DSGE Historical Decomposition (v0.4.0) ---

struct KalmanSmootherResult{T<:Real}
    smoothed_states::Matrix{T}
    smoothed_covariances::Array{T,3}
    smoothed_shocks::Matrix{T}
    filtered_states::Matrix{T}
    filtered_covariances::Array{T,3}
    predicted_states::Matrix{T}
    predicted_covariances::Array{T,3}
    log_likelihood::T
end

function dsge_smoother(sol::DSGESolution, data::AbstractMatrix,
                       observables::Vector{Symbol}; kwargs...)
    T_obs, _ = size(data)
    n_states = sol.spec.n_endog
    n_shocks = sol.spec.n_exog
    KalmanSmootherResult{Float64}(
        randn(T_obs, n_states), randn(T_obs, n_states, n_states),
        randn(T_obs, n_shocks), randn(T_obs, n_states), randn(T_obs, n_states, n_states),
        randn(T_obs, n_states), randn(T_obs, n_states, n_states), -100.0)
end

function historical_decomposition(sol::DSGESolution{T}, data::AbstractMatrix,
        observables::Vector{Symbol}; states::Symbol=:observables,
        measurement_error=nothing) where {T}
    T_obs = size(data, 1)
    n_obs = length(observables)
    n_shocks = sol.spec.n_exog
    n_vars = states == :all ? sol.spec.n_endog : n_obs
    varnames = states == :all ? sol.spec.varnames : [string(s) for s in observables]
    shock_names = sol.spec.exog
    HistoricalDecomposition{T}(
        randn(T_obs, n_vars, n_shocks), randn(T_obs, n_vars), randn(T_obs, n_vars),
        randn(T_obs, n_shocks), T_obs, varnames, shock_names, :dsge_linear)
end

function historical_decomposition(bd::BayesianDSGE{T}, data::AbstractMatrix,
        observables::Vector{Symbol}; mode_only::Bool=false, n_draws::Int=200,
        quantiles::Vector{<:Real}=T[0.16, 0.5, 0.84],
        measurement_error=nothing) where {T}
    T_obs = size(data, 1)
    n_obs = length(observables)
    n_shocks = bd.spec.n_exog
    n_q = length(quantiles)
    varnames = [string(s) for s in observables]
    shock_names = bd.spec.exog
    BayesianHistoricalDecomposition{T}(
        randn(T_obs, n_obs, n_shocks, n_q), randn(T_obs, n_obs, n_shocks),
        randn(T_obs, n_obs, n_q), randn(T_obs, n_obs), randn(T_obs, n_shocks),
        randn(T_obs, n_obs), T_obs, varnames, shock_names, T.(quantiles), :dsge_bayes)
end

contribution(hd::HistoricalDecomposition, var::Int, shock::Int) = hd.contributions[:, var, shock]
contribution(hd::BayesianHistoricalDecomposition, var::Int, shock::Int) = hd.point_estimate[:, var, shock]
total_shock_contribution(hd::HistoricalDecomposition, var::Int) = dropdims(sum(hd.contributions[:, var, :]; dims=2); dims=2)
verify_decomposition(hd::AbstractHistoricalDecomposition) = true

function dsge_particle_smoother(args...; kwargs...)
    # placeholder — not directly called by CLI
    nothing
end
```

- [ ] **Step 2: Verify mocks.jl still loads**

Run: `julia --project -e 'include("test/mocks.jl"); println("OK")'`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add test/mocks.jl
git commit -m "test: add DSGE HD mock types (KalmanSmootherResult, HD dispatches)"
```

### Task 3: Spectral Mock Types

**Files:**
- Modify: `test/mocks.jl` (append at end, before closing `end` of module)

- [ ] **Step 1: Add spectral mock types and functions**

```julia
# --- Spectral Analysis (v0.4.0) ---

struct ACFResult{T<:Real}
    lags::Vector{Int}
    acf::Vector{T}
    pacf::Vector{T}
    ci::T
    ccf::Union{Nothing,Vector{T}}
    q_stats::Vector{T}
    q_pvalues::Vector{T}
    nobs::Int
end

struct SpectralDensityResult{T<:Real}
    freq::Vector{T}
    density::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    method::Symbol
    bandwidth::T
    nobs::Int
end

struct CrossSpectrumResult{T<:Real}
    freq::Vector{T}
    co_spectrum::Vector{T}
    quad_spectrum::Vector{T}
    coherence::Vector{T}
    phase::Vector{T}
    gain::Vector{T}
    nobs::Int
end

struct TransferFunctionResult{T<:Real}
    freq::Vector{T}
    gain::Vector{T}
    phase::Vector{T}
    filter::Symbol
end

struct FisherTestResult{T<:Real}
    statistic::T
    pvalue::T
    peak_freq::T
    nobs::Int
end

struct BartlettWhiteNoiseResult{T<:Real}
    statistic::T
    pvalue::T
    nobs::Int
end

struct BoxPierceResult{T<:Real}
    statistic::T
    pvalue::T
    lags::Int
    nobs::Int
end

struct DurbinWatsonResult{T<:Real}
    statistic::T
    pvalue::T
    nobs::Int
end

function acf(y::AbstractVector{T}; maxlag::Int=min(20, length(y)-1)) where {T<:Real}
    n = length(y)
    lags = collect(1:maxlag)
    ACFResult{Float64}(lags, randn(maxlag), randn(maxlag), 1.96/sqrt(n),
        nothing, abs.(randn(maxlag)), rand(maxlag), n)
end

function pacf(y::AbstractVector; maxlag::Int=min(20, length(y)-1))
    acf(y; maxlag=maxlag)
end

function ccf(y::AbstractVector{T}, z::AbstractVector{T}; maxlag::Int=min(20, length(y)-1)) where {T<:Real}
    n = length(y)
    lags = collect(-maxlag:maxlag)
    nl = length(lags)
    ACFResult{Float64}(lags, randn(nl), randn(nl), 1.96/sqrt(n),
        randn(nl), abs.(randn(nl)), rand(nl), n)
end

function periodogram(y::AbstractVector{T}) where {T<:Real}
    n = length(y)
    nf = div(n, 2) + 1
    freq = collect(range(0.0, π; length=nf))
    SpectralDensityResult{Float64}(freq, abs.(randn(nf)), abs.(randn(nf)),
        abs.(randn(nf)) .+ 1.0, :periodogram, 1.0, n)
end

function spectral_density(y::AbstractVector{T}; method::Symbol=:welch,
                          bandwidth=nothing) where {T<:Real}
    n = length(y)
    nf = div(n, 2) + 1
    freq = collect(range(0.0, π; length=nf))
    SpectralDensityResult{Float64}(freq, abs.(randn(nf)), abs.(randn(nf)),
        abs.(randn(nf)) .+ 1.0, method, 1.0, n)
end

function cross_spectrum(y::AbstractVector{T}, z::AbstractVector{T}) where {T<:Real}
    n = length(y)
    nf = div(n, 2) + 1
    freq = collect(range(0.0, π; length=nf))
    CrossSpectrumResult{Float64}(freq, randn(nf), randn(nf), abs.(randn(nf)),
        randn(nf), abs.(randn(nf)), n)
end

function transfer_function(filt::Symbol; lambda::Float64=1600.0, nobs::Int=200)
    nf = 100
    freq = collect(range(0.0, π; length=nf))
    TransferFunctionResult{Float64}(freq, abs.(randn(nf)), randn(nf), filt)
end

fisher_test(y::AbstractVector) = FisherTestResult{Float64}(0.3, 0.05, 0.25, length(y))
bartlett_white_noise_test(y::AbstractVector) = BartlettWhiteNoiseResult{Float64}(1.2, 0.15, length(y))
box_pierce_test(y::AbstractVector; lags::Int=10) = BoxPierceResult{Float64}(15.3, 0.12, lags, length(y))
durbin_watson_test(y::AbstractVector) = DurbinWatsonResult{Float64}(1.95, 0.45, length(y))
```

- [ ] **Step 2: Verify mocks.jl still loads**

Run: `julia --project -e 'include("test/mocks.jl"); println("OK")'`

- [ ] **Step 3: Commit**

```bash
git add test/mocks.jl
git commit -m "test: add spectral analysis mock types (ACF, spectral density, cross-spectrum, tests)"
```

### Task 4: Panel Regression Mock Types

**Files:**
- Modify: `test/mocks.jl`

- [ ] **Step 1: Add panel regression mock types and functions**

```julia
# --- Panel Regression (v0.4.0) ---

struct PanelRegModel{T<:Real}
    beta::Vector{T}; vcov_mat::Matrix{T}; residuals::Vector{T}; fitted::Vector{T}
    y::Vector{T}; X::Matrix{T}
    r2_within::T; r2_between::T; r2_overall::T
    sigma_u::T; sigma_e::T; rho::T; theta::Union{Nothing,T}
    f_stat::T; f_pval::T; loglik::T; aic::T; bic::T
    varnames::Vector{String}; method::Symbol; twoway::Bool; cov_type::Symbol
    n_obs::Int; n_groups::Int; n_periods_avg::T
    group_effects::Union{Nothing,Vector{T}}
    data::PanelData{T}
end

struct PanelIVModel{T<:Real}
    beta::Vector{T}; vcov_mat::Matrix{T}; residuals::Vector{T}; fitted::Vector{T}
    y::Vector{T}; X::Matrix{T}; Z::Matrix{T}
    r2_within::T; r2_between::T; r2_overall::T
    sigma_u::T; sigma_e::T; rho::T
    f_stat::T; f_pval::T; loglik::T; aic::T; bic::T
    first_stage_f::T; sargan_stat::T; sargan_pval::T
    varnames::Vector{String}; method::Symbol; cov_type::Symbol
    n_obs::Int; n_groups::Int; n_periods_avg::T
    data::PanelData{T}
end

struct PanelLogitModel{T<:Real}
    beta::Vector{T}; vcov_mat::Matrix{T}; fitted::Vector{T}
    y::Vector{T}; X::Matrix{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    varnames::Vector{String}; method::Symbol; cov_type::Symbol
    converged::Bool; iterations::Int
    n_obs::Int; n_groups::Int; n_periods_avg::T
    data::PanelData{T}
end

struct PanelProbitModel{T<:Real}
    beta::Vector{T}; vcov_mat::Matrix{T}; fitted::Vector{T}
    y::Vector{T}; X::Matrix{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    varnames::Vector{String}; method::Symbol; cov_type::Symbol
    converged::Bool; iterations::Int
    n_obs::Int; n_groups::Int; n_periods_avg::T
    data::PanelData{T}
end

struct PanelTestResult{T<:Real}
    test_name::String
    statistic::T
    pvalue::T
    df::Int
    description::String
end

# StatsAPI for panel regression
StatsAPI.coef(m::PanelRegModel) = m.beta
StatsAPI.vcov(m::PanelRegModel) = m.vcov_mat
StatsAPI.residuals(m::PanelRegModel) = m.residuals
StatsAPI.predict(m::PanelRegModel) = m.fitted
StatsAPI.stderror(m::PanelRegModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.nobs(m::PanelRegModel) = m.n_obs
StatsAPI.coef(m::PanelIVModel) = m.beta
StatsAPI.vcov(m::PanelIVModel) = m.vcov_mat
StatsAPI.residuals(m::PanelIVModel) = m.residuals
StatsAPI.predict(m::PanelIVModel) = m.fitted
StatsAPI.stderror(m::PanelIVModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.nobs(m::PanelIVModel) = m.n_obs
StatsAPI.coef(m::PanelLogitModel) = m.beta
StatsAPI.vcov(m::PanelLogitModel) = m.vcov_mat
StatsAPI.predict(m::PanelLogitModel) = m.fitted
StatsAPI.stderror(m::PanelLogitModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.nobs(m::PanelLogitModel) = m.n_obs
StatsAPI.coef(m::PanelProbitModel) = m.beta
StatsAPI.vcov(m::PanelProbitModel) = m.vcov_mat
StatsAPI.predict(m::PanelProbitModel) = m.fitted
StatsAPI.stderror(m::PanelProbitModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.nobs(m::PanelProbitModel) = m.n_obs

function _mock_panel_reg(pd::PanelData{T}, k::Int; method::Symbol=:fe) where {T}
    n = size(pd.data, 1)
    b = randn(k); V = Matrix{Float64}(I, k, k) * 0.01
    r = randn(n); f = randn(n)
    PanelRegModel{Float64}(b, V, r, f, randn(n), randn(n, k),
        0.85, 0.80, 0.82, 0.5, 0.3, 0.7, method == :re ? 0.5 : nothing,
        25.0, 0.001, -100.0, 210.0, 220.0,
        ["x$i" for i in 1:k], method, false, :cluster,
        n, pd.n_groups, Float64(div(n, pd.n_groups)),
        method == :fe ? randn(pd.n_groups) : nothing, pd)
end

function estimate_xtreg(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
        model::Symbol=:fe, twoway::Bool=false, cov_type::Symbol=:cluster,
        bandwidth=nothing) where {T}
    k = length(indepvars)
    _mock_panel_reg(pd, k; method=model)
end

function estimate_xtiv(pd::PanelData{T}, depvar::Symbol, exog::Vector{Symbol},
        endog::Vector{Symbol}; instruments::Vector{Symbol}=Symbol[],
        model::Symbol=:fe, cov_type::Symbol=:cluster) where {T}
    k = length(exog) + length(endog)
    n = size(pd.data, 1)
    b = randn(k); V = Matrix{Float64}(I, k, k) * 0.01
    PanelIVModel{Float64}(b, V, randn(n), randn(n), randn(n), randn(n, k),
        randn(n, length(instruments) + length(exog)),
        0.85, 0.80, 0.82, 0.5, 0.3, 0.7, 25.0, 0.001, -100.0, 210.0, 220.0,
        15.0, 2.0, 0.35,
        [string.(exog); string.(endog)], model, cov_type,
        n, pd.n_groups, Float64(div(n, pd.n_groups)), pd)
end

function estimate_xtlogit(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
        model::Symbol=:pooled, cov_type::Symbol=:cluster) where {T}
    k = length(indepvars); n = size(pd.data, 1)
    PanelLogitModel{Float64}(randn(k), Matrix{Float64}(I, k, k)*0.01,
        rand(n), Float64.(rand(0:1, n)), randn(n, k),
        -50.0, -70.0, 0.28, 110.0, 120.0,
        string.(indepvars), model, cov_type, true, 10,
        n, pd.n_groups, Float64(div(n, pd.n_groups)), pd)
end

function estimate_xtprobit(pd::PanelData{T}, depvar::Symbol, indepvars::Vector{Symbol};
        model::Symbol=:pooled, cov_type::Symbol=:cluster) where {T}
    k = length(indepvars); n = size(pd.data, 1)
    PanelProbitModel{Float64}(randn(k), Matrix{Float64}(I, k, k)*0.01,
        rand(n), Float64.(rand(0:1, n)), randn(n, k),
        -50.0, -70.0, 0.28, 110.0, 120.0,
        string.(indepvars), model, cov_type, true, 10,
        n, pd.n_groups, Float64(div(n, pd.n_groups)), pd)
end

# Panel specification tests
function hausman_test(fe::PanelRegModel{T}, re::PanelRegModel{T}) where {T}
    PanelTestResult{Float64}("Hausman", 12.5, 0.014, length(fe.beta), "FE vs RE")
end
breusch_pagan_test(m::PanelRegModel) = PanelTestResult{Float64}("Breusch-Pagan LM", 45.2, 0.001, 1, "RE vs Pooled OLS")
f_test_fe(m::PanelRegModel) = PanelTestResult{Float64}("F-test FE", 8.3, 0.001, m.n_groups - 1, "Joint significance of FE")
pesaran_cd_test(m::PanelRegModel) = PanelTestResult{Float64}("Pesaran CD", 1.8, 0.072, 0, "Cross-sectional dependence")
wooldridge_ar_test(m::PanelRegModel) = PanelTestResult{Float64}("Wooldridge AR(1)", 5.2, 0.025, 1, "Serial correlation")
modified_wald_test(m::PanelRegModel) = PanelTestResult{Float64}("Modified Wald", 150.0, 0.001, m.n_groups, "Groupwise heteroskedasticity")
```

- [ ] **Step 2: Verify mocks.jl still loads**

Run: `julia --project -e 'include("test/mocks.jl"); println("OK")'`

- [ ] **Step 3: Commit**

```bash
git add test/mocks.jl
git commit -m "test: add panel regression mock types (PanelRegModel, PanelIVModel, PanelLogit/Probit, tests)"
```

### Task 5: Ordered/Multinomial & Data Mock Types

**Files:**
- Modify: `test/mocks.jl`

- [ ] **Step 1: Add ordered/multinomial mock types**

```julia
# --- Ordered/Multinomial Choice (v0.4.0) ---

struct OrderedLogitModel{T<:Real}
    y::Vector{Int}; X::Matrix{T}; beta::Vector{T}; cutpoints::Vector{T}
    vcov_mat::Matrix{T}; fitted::Matrix{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    varnames::Vector{String}; categories::Vector; converged::Bool; iterations::Int
    cov_type::Symbol
end

struct OrderedProbitModel{T<:Real}
    y::Vector{Int}; X::Matrix{T}; beta::Vector{T}; cutpoints::Vector{T}
    vcov_mat::Matrix{T}; fitted::Matrix{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    varnames::Vector{String}; categories::Vector; converged::Bool; iterations::Int
    cov_type::Symbol
end

struct MultinomialLogitModel{T<:Real}
    y::Vector{Int}; X::Matrix{T}; beta::Matrix{T}; vcov_mat::Matrix{T}
    fitted::Matrix{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    varnames::Vector{String}; categories::Vector; converged::Bool; iterations::Int
    cov_type::Symbol
end

StatsAPI.coef(m::OrderedLogitModel) = [m.beta; m.cutpoints]
StatsAPI.vcov(m::OrderedLogitModel) = m.vcov_mat
StatsAPI.nobs(m::OrderedLogitModel) = length(m.y)
StatsAPI.stderror(m::OrderedLogitModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.coef(m::OrderedProbitModel) = [m.beta; m.cutpoints]
StatsAPI.vcov(m::OrderedProbitModel) = m.vcov_mat
StatsAPI.nobs(m::OrderedProbitModel) = length(m.y)
StatsAPI.stderror(m::OrderedProbitModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.coef(m::MultinomialLogitModel) = vec(m.beta)
StatsAPI.vcov(m::MultinomialLogitModel) = m.vcov_mat
StatsAPI.nobs(m::MultinomialLogitModel) = length(m.y)
StatsAPI.stderror(m::MultinomialLogitModel) = sqrt.(diag(m.vcov_mat))
StatsAPI.predict(m::OrderedLogitModel) = m.fitted
StatsAPI.predict(m::OrderedProbitModel) = m.fitted
StatsAPI.predict(m::MultinomialLogitModel) = m.fitted

function estimate_ologit(y::AbstractVector, X::AbstractMatrix{T};
        cov_type::Symbol=:ols, varnames=nothing, clusters=nothing,
        maxiter::Int=200, tol=1e-8) where {T}
    n, k = size(X); J = length(unique(y))
    cuts = collect(range(-1.0, 1.0; length=J-1))
    total = k + J - 1
    OrderedLogitModel{Float64}(collect(1:n), Float64.(X), randn(k), cuts,
        Matrix{Float64}(I, total, total)*0.01, rand(n, J),
        -80.0, -100.0, 0.2, 170.0, 180.0,
        isnothing(varnames) ? ["x$i" for i in 1:k] : varnames,
        collect(1:J), true, 10, cov_type)
end

function estimate_oprobit(y::AbstractVector, X::AbstractMatrix{T};
        cov_type::Symbol=:ols, varnames=nothing, clusters=nothing,
        maxiter::Int=200, tol=1e-8) where {T}
    n, k = size(X); J = length(unique(y))
    cuts = collect(range(-1.0, 1.0; length=J-1))
    total = k + J - 1
    OrderedProbitModel{Float64}(collect(1:n), Float64.(X), randn(k), cuts,
        Matrix{Float64}(I, total, total)*0.01, rand(n, J),
        -80.0, -100.0, 0.2, 170.0, 180.0,
        isnothing(varnames) ? ["x$i" for i in 1:k] : varnames,
        collect(1:J), true, 10, cov_type)
end

function estimate_mlogit(y::AbstractVector, X::AbstractMatrix{T};
        cov_type::Symbol=:ols, varnames=nothing, clusters=nothing,
        maxiter::Int=200, tol=1e-8) where {T}
    n, k = size(X); J = length(unique(y))
    beta = randn(k, J-1)
    total = k * (J - 1)
    MultinomialLogitModel{Float64}(collect(1:n), Float64.(X), beta,
        Matrix{Float64}(I, total, total)*0.01, rand(n, J),
        -80.0, -100.0, 0.2, 170.0, 180.0,
        isnothing(varnames) ? ["x$i" for i in 1:k] : varnames,
        collect(1:J), true, 10, cov_type)
end

function brant_test(m::OrderedLogitModel)
    PanelTestResult{Float64}("Brant", 8.5, 0.07, length(m.beta), "Parallel regression assumption")
end

function hausman_iia(m::MultinomialLogitModel; omit_category::Int)
    PanelTestResult{Float64}("Hausman-McFadden IIA", 5.2, 0.15, length(m.beta[:,1]), "IIA for category $omit_category")
end
```

- [ ] **Step 2: Add data enhancement mock dispatches**

```julia
# --- Data Enhancements (v0.4.0) ---

function dropna(d::TimeSeriesData{T}; vars::Union{Vector{String},Nothing}=nothing) where {T}
    # Return same data (mock has no NaN)
    d
end

function dropna(d::PanelData{T}; vars::Union{Vector{String},Nothing}=nothing) where {T}
    d
end

function keeprows(d::TimeSeriesData{T}, idx::Vector{Int}) where {T}
    new_data = d.data[idx, :]
    TimeSeriesData(new_data; varnames=d.varnames)
end

function keeprows(d::PanelData{T}, idx::Vector{Int}) where {T}
    d  # simplified mock
end
```

- [ ] **Step 3: Verify mocks.jl still loads**

Run: `julia --project -e 'include("test/mocks.jl"); println("OK")'`

- [ ] **Step 4: Commit**

```bash
git add test/mocks.jl
git commit -m "test: add ordered/multinomial and data enhancement mock types"
```

---

## Chunk 3: Shared Helpers & Spectral Command (New File)

### Task 6: Panel Regression Shared Helpers

**Files:**
- Modify: `src/commands/shared.jl` (append after existing `_REG_COMMON_OPTIONS` section)

- [ ] **Step 1: Add panel regression shared helpers**

Append after `_reg_coef_table` function (around line 860):

```julia
# --- Panel Regression Shared Helpers (v0.4.0) ---

const _PREG_COMMON_OPTIONS = [
    Option("dep"; type=String, default="", description="Dependent variable column name"),
    Option("indep"; type=String, default="", description="Independent variables (comma-separated)"),
    Option("id-col"; type=String, default="", description="Panel group ID column (default: first column)"),
    Option("time-col"; type=String, default="", description="Panel time column (default: second column)"),
    Option("cov-type"; type=String, default="cluster", description="ols|cluster|twoway|driscoll-kraay"),
    Option("method"; short="m", type=String, default="fe", description="Estimation method"),
    Option("output"; short="o", type=String, default="", description="Export results to file"),
    Option("format"; short="f", type=String, default="table", description="table|csv|json"),
]

"""Load panel CSV for panel regression. Returns PanelData."""
function _load_panel_for_preg(data::String, id_col::String, time_col::String)
    df = load_data(data)
    cols = names(df)
    id = isempty(id_col) ? cols[1] : id_col
    tc = isempty(time_col) ? cols[2] : time_col
    pd = load_panel_data(data, id, tc)
    printstyled("  Panel: $(pd.n_groups) groups, $(pd.n_vars) variables"; color=:cyan)
    pd.balanced && printstyled(" (balanced)"; color=:cyan)
    println()
    return pd
end

"""Parse indep vars from comma-separated string. If empty, infer from all non-dep numeric cols."""
function _parse_indep_vars(pd, dep::String, indep_str::String)
    if isempty(indep_str)
        all_vars = pd.varnames
        return Symbol[Symbol(v) for v in all_vars if v != dep]
    else
        return Symbol[Symbol(strip(s)) for s in split(indep_str, ",")]
    end
end

"""Convert CLI option value with hyphens to MEMs Symbol with underscores."""
_to_sym(s::String) = Symbol(replace(s, "-" => "_"))

"""Build coefficient table from panel regression model."""
function _preg_coef_table(model, varnames::Vector{String})
    b = coef(model)
    se = stderror(model)
    t = b ./ se
    p = [2.0 * (1.0 - _normal_cdf(abs(ti))) for ti in t]
    ci_lo = b .- 1.96 .* se
    ci_hi = b .+ 1.96 .* se
    DataFrame(
        Variable = varnames,
        Coefficient = round.(b; digits=6),
        Std_Error = round.(se; digits=6),
        t_stat = round.(t; digits=4),
        p_value = round.(p; digits=4),
        CI_Lower = round.(ci_lo; digits=6),
        CI_Upper = round.(ci_hi; digits=6),
    )
end
```

- [ ] **Step 2: Commit**

```bash
git add src/commands/shared.jl
git commit -m "feat: add panel regression shared helpers (_PREG_COMMON_OPTIONS, _load_panel_for_preg, etc.)"
```

### Task 7: Spectral Command File (New)

**Files:**
- Create: `src/commands/spectral.jl`
- Modify: `src/Friedman.jl` (add include + register)

- [ ] **Step 1: Write `src/commands/spectral.jl`**

```julia
# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Spectral analysis commands: acf, periodogram, density, cross, transfer

function register_spectral_commands!()
    spec_acf = LeafCommand("acf", _spectral_acf;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("max-lag"; type=Int, default=nothing, description="Maximum lag (default: min(20, T-1))"),
            Option("ccf-with"; type=Int, default=nothing, description="Column index for cross-correlation"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Autocorrelation / partial autocorrelation / cross-correlation")

    spec_periodogram = LeafCommand("periodogram", _spectral_periodogram;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Raw periodogram")

    spec_density = LeafCommand("density", _spectral_density;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("method"; short="m", type=String, default="welch", description="periodogram|welch|smoothed|ar"),
            Option("bandwidth"; type=Float64, default=nothing, description="Smoothing bandwidth"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Spectral density estimation")

    spec_cross = LeafCommand("cross", _spectral_cross;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("var1"; type=Int, default=1, description="First variable column index"),
            Option("var2"; type=Int, default=2, description="Second variable column index"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Cross-spectral analysis (coherence, phase, gain)")

    spec_transfer = LeafCommand("transfer", _spectral_transfer;
        args=Argument[],
        options=[
            Option("filter"; type=String, default="hp", description="hp|bk|hamilton|ideal"),
            Option("lambda"; type=Float64, default=1600.0, description="Filter parameter (e.g. HP lambda)"),
            Option("nobs"; type=Int, default=200, description="Number of observations (for frequency grid)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Filter transfer function (theoretical frequency response)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "acf"         => spec_acf,
        "periodogram" => spec_periodogram,
        "density"     => spec_density,
        "cross"       => spec_cross,
        "transfer"    => spec_transfer,
    )
    return NodeCommand("spectral", subcmds,
        "Spectral analysis: ACF/PACF, periodogram, spectral density, cross-spectrum, transfer function")
end

# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

function _spectral_acf(; data::String, column::Int=1,
                        max_lag::Union{Int,Nothing}=nothing,
                        ccf_with::Union{Int,Nothing}=nothing,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    Y = df_to_matrix(df)
    y = Y[:, column]

    kwargs = isnothing(max_lag) ? (;) : (; maxlag=max_lag)
    result = acf(y; kwargs...)

    println("ACF/PACF: $(vnames[column])  (T = $(length(y)))")
    println()

    acf_df = DataFrame(
        Lag     = result.lags,
        ACF     = round.(result.acf; digits=6),
        PACF    = round.(result.pacf; digits=6),
        Q_stat  = round.(result.q_stats; digits=4),
        p_value = round.(result.q_pvalues; digits=4),
    )
    output_result(acf_df; format=Symbol(format), output=output, title="ACF / PACF")

    if !isnothing(ccf_with)
        z = Y[:, ccf_with]
        ccf_result = ccf(y, z; kwargs...)
        ccf_df = DataFrame(
            Lag = ccf_result.lags,
            CCF = round.(ccf_result.ccf; digits=6),
        )
        println()
        output_result(ccf_df; format=Symbol(format), output="",
            title="CCF: $(vnames[column]) × $(vnames[ccf_with])")
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_periodogram(; data::String, column::Int=1,
                                output::String="", format::String="table",
                                plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    y = df_to_matrix(df)[:, column]

    result = periodogram(y)

    println("Periodogram: $(vnames[column])  (T = $(length(y)))")
    println()

    peri_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Power     = round.(result.density; digits=6),
    )
    output_result(peri_df; format=Symbol(format), output=output, title="Periodogram")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_density(; data::String, column::Int=1,
                            method::String="welch",
                            bandwidth::Union{Float64,Nothing}=nothing,
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    y = df_to_matrix(df)[:, column]

    kwargs = Dict{Symbol,Any}(:method => Symbol(method))
    !isnothing(bandwidth) && (kwargs[:bandwidth] = bandwidth)
    result = spectral_density(y; kwargs...)

    println("Spectral Density ($(method)): $(vnames[column])  (T = $(length(y)))")
    println()

    sd_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Density   = round.(result.density; digits=6),
        CI_Lower  = round.(result.ci_lower; digits=6),
        CI_Upper  = round.(result.ci_upper; digits=6),
    )
    output_result(sd_df; format=Symbol(format), output=output, title="Spectral Density")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_cross(; data::String, var1::Int=1, var2::Int=2,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    Y = df_to_matrix(df)
    y = Y[:, var1]; z = Y[:, var2]

    result = cross_spectrum(y, z)

    println("Cross-Spectrum: $(vnames[var1]) × $(vnames[var2])  (T = $(length(y)))")
    println()

    cs_df = DataFrame(
        Frequency     = round.(result.freq; digits=6),
        Co_spectrum   = round.(result.co_spectrum; digits=6),
        Quad_spectrum = round.(result.quad_spectrum; digits=6),
        Coherence     = round.(result.coherence; digits=6),
        Phase         = round.(result.phase; digits=6),
        Gain          = round.(result.gain; digits=6),
    )
    output_result(cs_df; format=Symbol(format), output=output, title="Cross-Spectral Analysis")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_transfer(; filter::String="hp", lambda::Float64=1600.0,
                             nobs::Int=200,
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    result = transfer_function(Symbol(filter); lambda=lambda, nobs=nobs)

    println("Transfer Function: $(filter) filter  (λ = $lambda, T = $nobs)")
    println()

    tf_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Gain      = round.(result.gain; digits=6),
        Phase     = round.(result.phase; digits=6),
    )
    output_result(tf_df; format=Symbol(format), output=output, title="Filter Transfer Function")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end
```

- [ ] **Step 2: Add include and register in `src/Friedman.jl`**

After `include("commands/did.jl")` add:
```julia
include("commands/spectral.jl")
```

In `build_app()`, add to the root_cmds Dict:
```julia
"spectral" => register_spectral_commands!(),
```

- [ ] **Step 3: Run tests to verify compilation**

Run: `julia --project -e 'using Friedman; app = Friedman.build_app(); println("OK")'`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/commands/spectral.jl src/Friedman.jl
git commit -m "feat: add spectral top-level command (acf, periodogram, density, cross, transfer)"
```

---

## Chunk 4: DSGE HD & Data Commands

### Task 8: DSGE HD Command Leaves

**Files:**
- Modify: `src/commands/dsge.jl`

- [ ] **Step 1: Add `dsge hd` LeafCommand**

In `register_dsge_commands!()`, after the existing `dsge_ss` leaf (steady-state), add:

```julia
    dsge_hd = LeafCommand("hd", _dsge_hd;
        args=[Argument("model"; description="Path to DSGE model file (.toml or .jl)")],
        options=[
            Option("data"; short="d", type=String, default="", description="Path to CSV data file"),
            Option("observables"; type=String, default="", description="Observable variable names (comma-separated)"),
            Option("states"; type=String, default="observables", description="observables|all"),
            Option("measurement-error"; type=String, default="", description="Measurement error std devs (comma-separated) or auto"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Historical decomposition of DSGE model via Kalman smoother")
```

Add `"hd" => dsge_hd` to the `subcmds` Dict.

- [ ] **Step 2: Add `dsge bayes hd` LeafCommand**

In the `bayes_subcmds` Dict (where bayes_estimate, bayes_irf, etc. are), add:

```julia
    bayes_hd = LeafCommand("hd", _dsge_bayes_hd;
        args=[Argument("model"; description="Path to DSGE model file (.toml or .jl)")],
        options=[_bayes_common_options...,
            Option("observables"; type=String, default="", description="Observable variable names (comma-separated)"),
            Option("n-draws"; type=Int, default=200, description="Number of posterior draws for HD"),
            Option("quantiles"; type=String, default="0.16,0.5,0.84", description="Quantile levels"),
            Option("horizon"; short="h", type=Int, default=40, description="IRF horizon"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[
            Flag("mode-only"; description="Use posterior mode only (no full posterior)"),
            Flag("delayed-acceptance"; description="Use delayed acceptance for MH"),
            Flag("plot"; description="Open interactive plot in browser"),
        ],
        description="Historical decomposition from Bayesian DSGE posterior")
```

Add `"hd" => bayes_hd` to the `bayes_subcmds` Dict.

- [ ] **Step 3: Write `_dsge_hd` handler**

Append to dsge.jl:

```julia
function _dsge_hd(; model::String, data::String="", observables::String="",
                   states::String="observables",
                   measurement_error::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="")
    isempty(data) && error("--data is required for DSGE historical decomposition")
    isempty(observables) && error("--observables is required (comma-separated variable names)")

    spec = _load_dsge_model(model)
    sol = _solve_dsge(spec)

    df = load_data(data)
    Y = df_to_matrix(df)
    obs_syms = Symbol[Symbol(strip(s)) for s in split(observables, ",")]

    println("DSGE Historical Decomposition")
    println("  Model: $model")
    println("  Observations: $(size(Y, 1)), Observable variables: $(length(obs_syms))")
    println("  States: $states")
    println()

    # Parse measurement error
    me = if isempty(measurement_error)
        nothing
    elseif measurement_error == "auto"
        :auto
    else
        [parse(Float64, strip(s)) for s in split(measurement_error, ",")]
    end

    hd = historical_decomposition(sol, Y, obs_syms;
        states=Symbol(states), measurement_error=me)

    # Verify decomposition
    ok = verify_decomposition(hd)
    ok && printstyled("  Decomposition verified ✓\n"; color=:green)

    # Output contributions per shock
    for (si, sname) in enumerate(hd.shock_names)
        contrib = hd.contributions[:, :, si]
        contrib_df = DataFrame(contrib, hd.variables)
        insertcols!(contrib_df, 1, :t => 1:hd.T_eff)
        output_result(contrib_df; format=Symbol(format), output=output,
            title="Shock: $sname contributions")
    end

    _maybe_plot(hd; plot=plot, plot_save=plot_save)
    return hd
end
```

- [ ] **Step 4: Write `_dsge_bayes_hd` handler**

```julia
function _dsge_bayes_hd(; model::String, data::String="", params::String="",
                         priors::String="", observables::String="",
                         sampler::String="smc", n_smc::Int=5000, n_mh::Int=10000,
                         n_blocks::Int=1, conf_level::Float64=0.9,
                         n_draws::Int=200, quantiles::String="0.16,0.5,0.84",
                         mode_only::Bool=false,
                         delayed_acceptance::Bool=false,
                         horizon::Int=40,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    isempty(data) && error("--data is required")
    isempty(observables) && error("--observables is required (comma-separated variable names)")
    isempty(params) && error("--params is required (comma-separated parameter names)")
    isempty(priors) && error("--priors is required (path to priors TOML)")

    spec = _load_dsge_model(model)
    df = load_data(data)
    Y = df_to_matrix(df)
    obs_syms = Symbol[Symbol(strip(s)) for s in split(observables, ",")]
    q_levels = [parse(Float64, strip(s)) for s in split(quantiles, ",")]

    theta0 = zeros(length(split(params, ",")))
    param_names = [strip(s) for s in split(params, ",")]
    prior_config = load_config(priors)
    prior_dists = get_dsge_priors(prior_config)

    println("Bayesian DSGE Historical Decomposition")
    println("  Model: $model, Sampler: $sampler, Draws: $(mode_only ? "mode only" : n_smc)")
    println("  Observables: $observables")
    println()

    bd = estimate_dsge_bayes(spec, Y, theta0;
        priors=prior_dists, method=Symbol(sampler),
        n_smc=n_smc, n_mh=n_mh, n_blocks=n_blocks, conf_level=conf_level)

    hd = historical_decomposition(bd, Y, obs_syms;
        mode_only=mode_only, n_draws=n_draws, quantiles=q_levels)

    # Output point estimates per shock
    for (si, sname) in enumerate(hd.shock_names)
        pe = hd.point_estimate[:, :, si]
        pe_df = DataFrame(pe, hd.variables)
        insertcols!(pe_df, 1, :t => 1:hd.T_eff)
        output_result(pe_df; format=Symbol(format), output=output,
            title="Shock: $sname (posterior mean)")
    end

    _maybe_plot(hd; plot=plot, plot_save=plot_save)
    return hd
end
```

- [ ] **Step 5: Commit**

```bash
git add src/commands/dsge.jl
git commit -m "feat: add dsge hd and dsge bayes hd commands"
```

### Task 9: Data Enhancement Commands

**Files:**
- Modify: `src/commands/data.jl`

- [ ] **Step 1: Add `data dropna` and `data keeprows` leaves**

In `register_data_commands!()`, add leaf definitions:

```julia
    data_dropna = LeafCommand("dropna", _data_dropna;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("vars"; type=String, default="", description="Column names to check (comma-separated; default: all)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Drop rows with missing/NaN values")

    data_keeprows = LeafCommand("keeprows", _data_keeprows;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("rows"; type=String, default="", description="Row indices (e.g. 1:100, 1,5,10)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Filter rows by index range")
```

Add `"dropna" => data_dropna, "keeprows" => data_keeprows` to the subcmds Dict.

- [ ] **Step 2: Write handlers**

```julia
function _data_dropna(; data::String, vars::String="",
                       output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    vnames = variable_names(df)
    ts = TimeSeriesData(Y; varnames=vnames)

    n_before = size(Y, 1)
    var_list = isempty(vars) ? nothing : [strip(s) for s in split(vars, ",")]
    cleaned = dropna(ts; vars=var_list)
    n_after = size(cleaned.data, 1)

    println("Drop NA: $data")
    println("  Rows before: $n_before, after: $n_after, dropped: $(n_before - n_after)")
    println()

    result_df = DataFrame(cleaned.data, cleaned.varnames)
    output_result(result_df; format=Symbol(format), output=output, title="Cleaned Data")
    return cleaned
end

function _data_keeprows(; data::String, rows::String="",
                         output::String="", format::String="table")
    isempty(rows) && error("--rows is required (e.g. 1:100, 1,5,10)")

    df = load_data(data)
    Y = df_to_matrix(df)
    vnames = variable_names(df)
    ts = TimeSeriesData(Y; varnames=vnames)
    n_total = size(Y, 1)

    # Parse row indices
    indices = if occursin(":", rows)
        parts = split(rows, ":")
        lo = parse(Int, strip(parts[1]))
        hi_str = strip(parts[2])
        hi = hi_str == "end" ? n_total : parse(Int, hi_str)
        collect(lo:hi)
    else
        [parse(Int, strip(s)) for s in split(rows, ",")]
    end

    filtered = keeprows(ts, indices)

    println("Keep Rows: $data")
    println("  Selected $(length(indices)) of $n_total rows")
    println()

    result_df = DataFrame(filtered.data, filtered.varnames)
    output_result(result_df; format=Symbol(format), output=output, title="Filtered Data")
    return filtered
end
```

- [ ] **Step 3: Commit**

```bash
git add src/commands/data.jl
git commit -m "feat: add data dropna and data keeprows commands"
```

---

## Chunk 5: Panel Regression & Ordered/Multinomial Estimation

### Task 10: Panel Regression Estimation Leaves

**Files:**
- Modify: `src/commands/estimate.jl`

- [ ] **Step 1: Add 4 panel regression LeafCommands**

In `register_estimate_commands!()`, add:

```julia
    est_preg = LeafCommand("preg", _estimate_preg;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS...],
        flags=[Flag("twoway"; description="Include time fixed effects")],
        description="Panel regression (FE/RE/FD/Between/CRE/AB/BB)")

    est_piv = LeafCommand("piv", _estimate_piv;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[
            Option("dep"; type=String, default="", description="Dependent variable column name"),
            Option("exog"; type=String, default="", description="Exogenous variables (comma-separated)"),
            Option("endog"; type=String, default="", description="Endogenous variables (comma-separated)"),
            Option("instruments"; type=String, default="", description="Instruments (comma-separated)"),
            Option("method"; short="m", type=String, default="fe", description="fe|re|fd|hausman-taylor"),
            Option("cov-type"; type=String, default="cluster", description="ols|cluster|twoway|driscoll-kraay"),
            Option("id-col"; type=String, default="", description="Panel group ID column"),
            Option("time-col"; type=String, default="", description="Panel time column"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Panel IV/2SLS regression")

    est_plogit = LeafCommand("plogit", _estimate_plogit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[
            _PREG_COMMON_OPTIONS[1:2]...,  # dep, indep
            Option("method"; short="m", type=String, default="pooled", description="pooled|fe|re|cre"),
            Option("cov-type"; type=String, default="cluster", description="ols|cluster"),
            _PREG_COMMON_OPTIONS[4:end]...,  # id-col, time-col, output, format
        ],
        description="Panel logistic regression")

    est_pprobit = LeafCommand("pprobit", _estimate_pprobit;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[
            _PREG_COMMON_OPTIONS[1:2]...,  # dep, indep
            Option("method"; short="m", type=String, default="pooled", description="pooled|re|cre"),
            Option("cov-type"; type=String, default="cluster", description="ols|cluster"),
            _PREG_COMMON_OPTIONS[4:end]...,  # id-col, time-col, output, format
        ],
        description="Panel probit regression (no FE — incidental parameters problem)")
```

Add `"preg" => est_preg, "piv" => est_piv, "plogit" => est_plogit, "pprobit" => est_pprobit` to the subcmds Dict.

- [ ] **Step 2: Write the 4 handlers**

```julia
function _estimate_preg(; data::String, dep::String="", indep::String="",
                         method::String="fe", twoway::Bool=false,
                         cov_type::String="cluster",
                         id_col::String="", time_col::String="",
                         output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    model_sym = _to_sym(method)
    cov_sym = _to_sym(cov_type)
    println("Panel Regression ($method): $dep ~ $(join(indep_syms, " + "))")
    println()

    model = estimate_xtreg(pd, Symbol(dep), indep_syms;
        model=model_sym, twoway=twoway, cov_type=cov_sym)

    coef_df = _preg_coef_table(model, string.(indep_syms))
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Regression Coefficients ($method)")

    println()
    pairs = Pair{String,Any}[
        "R² (within)"  => round(model.r2_within; digits=6),
        "R² (between)" => round(model.r2_between; digits=6),
        "R² (overall)" => round(model.r2_overall; digits=6),
        "sigma_u"      => round(model.sigma_u; digits=6),
        "sigma_e"      => round(model.sigma_e; digits=6),
        "rho"          => round(model.rho; digits=6),
        "F-statistic"  => round(model.f_stat; digits=4),
        "F p-value"    => round(model.f_pval; digits=4),
        "N obs"        => model.n_obs,
        "N groups"     => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")
    return model
end

function _estimate_piv(; data::String, dep::String="", exog::String="",
                        endog::String="", instruments::String="",
                        method::String="fe", cov_type::String="cluster",
                        id_col::String="", time_col::String="",
                        output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    isempty(endog) && error("--endog is required")
    pd = _load_panel_for_preg(data, id_col, time_col)

    exog_syms = isempty(exog) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(exog, ",")]
    endog_syms = Symbol[Symbol(strip(s)) for s in split(endog, ",")]
    inst_syms = isempty(instruments) ? Symbol[] : Symbol[Symbol(strip(s)) for s in split(instruments, ",")]

    println("Panel IV ($method): $dep ~ $(join([exog_syms; endog_syms], " + "))")
    println("  Instruments: $(join(inst_syms, ", "))")
    println()

    model = estimate_xtiv(pd, Symbol(dep), exog_syms, endog_syms;
        instruments=inst_syms, model=_to_sym(method), cov_type=_to_sym(cov_type))

    all_vars = [string.(exog_syms); string.(endog_syms)]
    coef_df = _preg_coef_table(model, all_vars)
    output_result(coef_df; format=Symbol(format), output=output, title="Panel IV Coefficients")
    return model
end

function _estimate_plogit(; data::String, dep::String="", indep::String="",
                           method::String="pooled", cov_type::String="cluster",
                           id_col::String="", time_col::String="",
                           output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    println("Panel Logit ($method): $dep ~ $(join(indep_syms, " + "))")
    println()

    model = estimate_xtlogit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    coef_df = _preg_coef_table(model, string.(indep_syms))
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Logit Coefficients ($method)")

    println()
    pairs = Pair{String,Any}[
        "Pseudo R²"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Converged"       => model.converged,
        "N obs"           => model.n_obs,
        "N groups"        => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")
    return model
end

function _estimate_pprobit(; data::String, dep::String="", indep::String="",
                            method::String="pooled", cov_type::String="cluster",
                            id_col::String="", time_col::String="",
                            output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    println("Panel Probit ($method): $dep ~ $(join(indep_syms, " + "))")
    println()

    model = estimate_xtprobit(pd, Symbol(dep), indep_syms;
        model=_to_sym(method), cov_type=_to_sym(cov_type))

    coef_df = _preg_coef_table(model, string.(indep_syms))
    output_result(coef_df; format=Symbol(format), output=output,
        title="Panel Probit Coefficients ($method)")

    println()
    pairs = Pair{String,Any}[
        "Pseudo R²"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Converged"       => model.converged,
        "N obs"           => model.n_obs,
        "N groups"        => model.n_groups,
    ]
    output_kv(pairs; format=format, title="Model Statistics")
    return model
end
```

- [ ] **Step 3: Commit**

```bash
git add src/commands/estimate.jl
git commit -m "feat: add estimate preg/piv/plogit/pprobit panel regression commands"
```

### Task 11: Ordered & Multinomial Estimation Leaves

**Files:**
- Modify: `src/commands/estimate.jl`

- [ ] **Step 1: Add 3 ordered/multinomial LeafCommands**

```julia
    est_ologit = LeafCommand("ologit", _estimate_ologit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Ordered logistic regression")

    est_oprobit = LeafCommand("oprobit", _estimate_oprobit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...],
        description="Ordered probit regression")

    est_mlogit = LeafCommand("mlogit", _estimate_mlogit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("dep"; type=String, default="", description="Dependent variable column name"),
            Option("cov-type"; type=String, default="ols", description="ols|hc0|hc1|hc2|hc3"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Multinomial logistic regression")
```

Add `"ologit" => est_ologit, "oprobit" => est_oprobit, "mlogit" => est_mlogit` to subcmds Dict.

- [ ] **Step 2: Write the 3 handlers**

```julia
function _estimate_ologit(; data::String, dep::String="", cov_type::String="ols",
                           clusters::String="",
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    println("Ordered Logit: $dep_name ~ $(join(xcols, " + "))")
    println()

    model = estimate_ologit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    # Cutpoints table
    J = length(model.cutpoints)
    cut_df = DataFrame(
        Cutpoint = ["cut$i" for i in 1:J],
        Value = round.(model.cutpoints; digits=6),
    )
    output_result(cut_df; format=Symbol(format), output="", title="Cutpoints")

    # Coefficients table
    b = model.beta; se = stderror(model)[1:length(b)]
    z = b ./ se
    p = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    coef_df = DataFrame(
        Variable = xcols,
        Coefficient = round.(b; digits=6),
        Std_Error = round.(se; digits=6),
        z_stat = round.(z; digits=4),
        p_value = round.(p; digits=4),
    )
    println()
    output_result(coef_df; format=Symbol(format), output=output, title="Ordered Logit Coefficients")

    println()
    pairs = Pair{String,Any}[
        "Pseudo R²"       => round(model.pseudo_r2; digits=6),
        "Log-likelihood"  => round(model.loglik; digits=4),
        "AIC"             => round(model.aic; digits=4),
        "BIC"             => round(model.bic; digits=4),
        "Categories"      => length(model.categories),
    ]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

function _estimate_oprobit(; data::String, dep::String="", cov_type::String="ols",
                            clusters::String="",
                            output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    cl = _load_clusters(data, clusters)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    println("Ordered Probit: $dep_name ~ $(join(xcols, " + "))")
    println()

    model = estimate_oprobit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    J = length(model.cutpoints)
    cut_df = DataFrame(Cutpoint = ["cut$i" for i in 1:J], Value = round.(model.cutpoints; digits=6))
    output_result(cut_df; format=Symbol(format), output="", title="Cutpoints")

    b = model.beta; se = stderror(model)[1:length(b)]
    z = b ./ se
    p = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
    coef_df = DataFrame(
        Variable = xcols, Coefficient = round.(b; digits=6),
        Std_Error = round.(se; digits=6), z_stat = round.(z; digits=4), p_value = round.(p; digits=4))
    println()
    output_result(coef_df; format=Symbol(format), output=output, title="Ordered Probit Coefficients")

    println()
    pairs = Pair{String,Any}[
        "Pseudo R²" => round(model.pseudo_r2; digits=6),
        "Log-likelihood" => round(model.loglik; digits=4),
        "AIC" => round(model.aic; digits=4), "BIC" => round(model.bic; digits=4),
        "Categories" => length(model.categories)]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end

function _estimate_mlogit(; data::String, dep::String="", cov_type::String="ols",
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep)
    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep

    println("Multinomial Logit: $dep_name ~ $(join(xcols, " + "))")
    println()

    model = estimate_mlogit(y, X; cov_type=Symbol(cov_type), varnames=xcols)

    # Per-category coefficient tables
    J = size(model.fitted, 2)
    for j in 2:J
        cat_beta = model.beta[:, j-1]
        k = length(cat_beta)
        total_params = k * (J - 1)
        se_all = stderror(model)
        se_j = se_all[((j-2)*k+1):((j-1)*k)]
        z = cat_beta ./ se_j
        p = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        cat_df = DataFrame(
            Variable = xcols, Coefficient = round.(cat_beta; digits=6),
            Std_Error = round.(se_j; digits=6), z_stat = round.(z; digits=4),
            p_value = round.(p; digits=4))
        output_result(cat_df; format=Symbol(format), output=output,
            title="Category $(model.categories[j]) vs $(model.categories[1])")
        println()
    end

    pairs = Pair{String,Any}[
        "Pseudo R²" => round(model.pseudo_r2; digits=6),
        "Log-likelihood" => round(model.loglik; digits=4),
        "AIC" => round(model.aic; digits=4), "BIC" => round(model.bic; digits=4),
        "Categories" => J]
    output_kv(pairs; format=format, title="Fit Statistics")
    return model
end
```

- [ ] **Step 3: Commit**

```bash
git add src/commands/estimate.jl
git commit -m "feat: add estimate ologit/oprobit/mlogit ordered and multinomial choice commands"
```

---

## Chunk 6: Predict, Residuals, Test Command Additions

### Task 12: Predict Leaves (7 new)

**Files:**
- Modify: `src/commands/predict.jl`

- [ ] **Step 1: Add 7 new LeafCommands and handlers for predict**

Add leaves `predict preg`, `predict piv`, `predict plogit`, `predict pprobit`, `predict ologit`, `predict oprobit`, `predict mlogit`.

Each follows the existing `_predict_reg` pattern: load data → estimate model → `predict(model)` → output fitted values DataFrame.

Panel predict handlers use `_load_panel_for_preg` + `estimate_xtreg/xtiv/xtlogit/xtprobit`.
Ordered/multinomial predict handlers use `_load_reg_data` + `estimate_ologit/oprobit/mlogit`; output is a matrix of predicted probabilities per category.

- [ ] **Step 2: Register leaves in subcmds Dict**

Add `"preg" => pred_preg, "piv" => pred_piv, "plogit" => pred_plogit, "pprobit" => pred_pprobit, "ologit" => pred_ologit, "oprobit" => pred_oprobit, "mlogit" => pred_mlogit` to the subcmds Dict.

- [ ] **Step 3: Commit**

```bash
git add src/commands/predict.jl
git commit -m "feat: add predict preg/piv/plogit/pprobit/ologit/oprobit/mlogit"
```

### Task 13: Residuals Leaves (7 new)

**Files:**
- Modify: `src/commands/residuals.jl`

- [ ] **Step 1: Add 7 new LeafCommands and handlers for residuals**

Same pattern as predict. Panel reg/IV have `residuals(model)`. Panel logit/probit compute deviance residuals manually:
```julia
fitted_p = predict(model)
y_vals = model.y
resid = [sign(y_vals[i] - fitted_p[i]) *
    sqrt(-2.0 * (y_vals[i] * log(max(fitted_p[i], 1e-10)) +
                 (1.0 - y_vals[i]) * log(max(1.0 - fitted_p[i], 1e-10))))
    for i in eachindex(y_vals)]
```

Ordered/multinomial residuals also use deviance residuals.

- [ ] **Step 2: Register leaves in subcmds Dict**

- [ ] **Step 3: Commit**

```bash
git add src/commands/residuals.jl
git commit -m "feat: add residuals preg/piv/plogit/pprobit/ologit/oprobit/mlogit"
```

### Task 14: Test Command Additions (12 new leaves)

**Files:**
- Modify: `src/commands/test.jl`

- [ ] **Step 1: Add 6 panel specification test leaves**

```julia
    test_hausman = LeafCommand("hausman", _test_hausman;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]...,  # dep, indep
                 _PREG_COMMON_OPTIONS[4:5]...,  # id-col, time-col
                 Option("format"; short="f", type=String, default="table", description="table|csv|json"),
                 Option("output"; short="o", type=String, default="", description="Export results to file")],
        description="Hausman specification test (FE vs RE)")

    test_breusch_pagan = LeafCommand("breusch-pagan", _test_breusch_pagan;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[_PREG_COMMON_OPTIONS[1:2]..., _PREG_COMMON_OPTIONS[4:5]...,
                 Option("format"; short="f", type=String, default="table", description="table|csv|json"),
                 Option("output"; short="o", type=String, default="", description="Export results to file")],
        description="Breusch-Pagan LM test for random effects")
```

Similarly for `f-fe`, `pesaran-cd`, `wooldridge-ar`, `modified-wald`.

- [ ] **Step 2: Add 4 spectral/portmanteau test leaves**

```julia
    test_fisher_spec = LeafCommand("fisher", _test_fisher;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[Option("column"; short="c", type=Int, default=1, description="Column index"),
                 Option("format"; short="f", type=String, default="table", description="table|csv|json"),
                 Option("output"; short="o", type=String, default="", description="Export results to file")],
        description="Fisher's test for periodicity")
```

Similarly for `bartlett-wn`, `box-pierce`, `durbin-watson`.

- [ ] **Step 3: Add 2 discrete choice test leaves**

```julia
    test_brant = LeafCommand("brant", _test_brant;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[Option("dep"; type=String, default="", description="Dependent variable"),
                 Option("format"; short="f", type=String, default="table", description="table|csv|json"),
                 Option("output"; short="o", type=String, default="", description="Export results to file")],
        description="Brant test for parallel regression (ordered models)")

    test_hausman_iia = LeafCommand("hausman-iia", _test_hausman_iia;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[Option("dep"; type=String, default="", description="Dependent variable"),
                 Option("omit-category"; type=Int, default=nothing, description="Category to omit for IIA test (required)"),
                 Option("format"; short="f", type=String, default="table", description="table|csv|json"),
                 Option("output"; short="o", type=String, default="", description="Export results to file")],
        description="Hausman-McFadden IIA test for multinomial logit")
```

- [ ] **Step 4: Register all 12 leaves in subcmds Dict**

- [ ] **Step 5: Write all 12 handlers**

Each panel test handler: load panel → estimate model → call test function → output key-value result.
Each spectral test handler: load CSV → extract column → call test function → output key-value result.
Each discrete choice test handler: load CSV → estimate model → call test function → output result.

Handler pattern for test output:
```julia
function _test_hausman(; data::String, dep::String="", indep::String="",
                        id_col::String="", time_col::String="",
                        output::String="", format::String="table")
    isempty(dep) && error("--dep is required")
    pd = _load_panel_for_preg(data, id_col, time_col)
    indep_syms = _parse_indep_vars(pd, dep, indep)

    fe_model = estimate_xtreg(pd, Symbol(dep), indep_syms; model=:fe)
    re_model = estimate_xtreg(pd, Symbol(dep), indep_syms; model=:re)
    result = hausman_test(fe_model, re_model)

    println("Hausman Test: FE vs RE")
    println()
    pairs = Pair{String,Any}[
        "Test" => result.test_name,
        "χ² statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
        "df" => result.df,
        "Decision" => result.pvalue < 0.05 ? "Reject H0 (use FE)" : "Fail to reject H0 (RE consistent)",
    ]
    output_kv(pairs; format=format, title="Hausman Specification Test")
    return result
end
```

- [ ] **Step 6: Commit**

```bash
git add src/commands/test.jl
git commit -m "feat: add 12 test leaves (panel spec, spectral diagnostics, discrete choice)"
```

---

## Chunk 7: Handler Tests & Structure Tests

### Task 15: Handler Tests for New Commands

**Files:**
- Modify: `test/test_commands.jl`

- [ ] **Step 1: Add handler tests for spectral commands**

```julia
@testset "Spectral Commands" begin
    @testset "_spectral_acf" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_acf(; data=csv, column=1, format="table", output="")
            end
            @test occursin("ACF", out)
            @test occursin("PACF", out)
            @test occursin("Lag", out)
        end
    end

    @testset "_spectral_periodogram" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_periodogram(; data=csv, column=1, format="table", output="")
            end
            @test occursin("Periodogram", out)
            @test occursin("Frequency", out)
        end
    end

    @testset "_spectral_density" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_density(; data=csv, column=1, method="welch", format="table", output="")
            end
            @test occursin("Spectral Density", out)
        end
    end

    @testset "_spectral_cross" begin
        mktempdir() do dir
            csv = _make_csv(dir; T=100, n=3)
            out = _capture() do
                _spectral_cross(; data=csv, var1=1, var2=2, format="table", output="")
            end
            @test occursin("Cross-Spectrum", out)
            @test occursin("Coherence", out)
        end
    end

    @testset "_spectral_transfer" begin
        out = _capture() do
            _spectral_transfer(; filter="hp", lambda=1600.0, nobs=200, format="table", output="")
        end
        @test occursin("Transfer Function", out)
        @test occursin("Gain", out)
    end
end
```

- [ ] **Step 2: Add handler tests for panel regression**

```julia
@testset "Panel Regression Commands" begin
    @testset "_estimate_preg" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = _capture() do
                _estimate_preg(; data=csv, dep="var1", indep="var2,var3",
                    method="fe", cov_type="cluster",
                    id_col="group", time_col="time", format="table", output="")
            end
            @test occursin("Panel Regression", out)
            @test occursin("Coefficient", out)
            @test occursin("R²", out) || occursin("R²", out)
        end
    end

    @testset "_estimate_piv" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=4)
            out = _capture() do
                _estimate_piv(; data=csv, dep="var1", exog="var2", endog="var3",
                    instruments="var4", method="fe", cov_type="cluster",
                    id_col="group", time_col="time", format="table", output="")
            end
            @test occursin("Panel IV", out)
        end
    end

    @testset "_estimate_plogit" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = _capture() do
                _estimate_plogit(; data=csv, dep="var1", indep="var2,var3",
                    method="pooled", cov_type="cluster",
                    id_col="group", time_col="time", format="table", output="")
            end
            @test occursin("Panel Logit", out)
        end
    end

    @testset "_estimate_pprobit" begin
        mktempdir() do dir
            csv = _make_panel_csv(dir; G=5, T_per=20, n=3)
            out = _capture() do
                _estimate_pprobit(; data=csv, dep="var1", indep="var2,var3",
                    method="pooled", cov_type="cluster",
                    id_col="group", time_col="time", format="table", output="")
            end
            @test occursin("Panel Probit", out)
        end
    end
end
```

- [ ] **Step 3: Add handler tests for ordered/multinomial, DSGE HD, data enhancements, and test commands**

Follow the same pattern: `mktempdir`, `_make_csv`/`_make_panel_csv`, `_capture`, `@test occursin`.

For ordered/multinomial: use `_make_csv` with categorical-like data.
For DSGE HD: use `_make_dsge_model` (existing helper) + `_make_csv`.
For data dropna/keeprows: use `_make_csv`.
For panel spec tests: use `_make_panel_csv`.
For spectral diagnostic tests: use `_make_csv`.
For discrete choice tests: use `_make_csv`.

- [ ] **Step 4: Run all tests**

Run: `julia --project test/runtests.jl`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add test/test_commands.jl
git commit -m "test: add handler tests for all v0.4.0 commands (~42 new tests)"
```

### Task 16: Structure Tests for `spectral` Top-Level Command

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Add structure tests for spectral command**

Find the existing command structure test section (near DSGE/DID structure tests) and add:

```julia
@testset "Spectral command structure" begin
    app = Friedman.build_app()
    spectral_node = app.root.subcmds["spectral"]
    @test spectral_node isa Friedman.NodeCommand
    @test haskey(spectral_node.subcmds, "acf")
    @test haskey(spectral_node.subcmds, "periodogram")
    @test haskey(spectral_node.subcmds, "density")
    @test haskey(spectral_node.subcmds, "cross")
    @test haskey(spectral_node.subcmds, "transfer")
    @test length(spectral_node.subcmds) == 5
end
```

- [ ] **Step 2: Run tests**

Run: `julia --project test/runtests.jl`

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: add spectral command structure tests"
```

---

## Chunk 8: Documentation Updates

### Task 17: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update version references, command hierarchy, command details, deps, testing sections**

Key changes:
- Version: v0.3.5 → v0.4.0, MEMs v0.3.5 → v0.4.0
- Project Overview: update line counts, subcommand count (~199)
- Command Hierarchy: add `spectral` top-level, add new leaves under estimate/test/predict/residuals/dsge/data
- Dependencies: FFTW now regular dep, add NonlinearSolve
- Command Details: add sections for spectral.jl, panel reg additions, ordered/multinomial, DSGE HD, data enhancements
- Testing: update test counts
- Adding a New Command: mention spectral as example of new top-level

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for v0.4.0"
```

### Task 18: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update version, feature list, commands table**

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README.md for v0.4.0"
```

### Task 19: Update Documenter.jl Docs

**Files:**
- Modify: `docs/make.jl`
- Modify: `docs/src/index.md`
- Modify: `docs/src/commands/overview.md`
- Modify: `docs/src/architecture.md`
- Modify: `docs/src/api.md`
- Create: `docs/src/commands/spectral.md`
- Create: `docs/src/commands/panel-regression.md`
- Create: `docs/src/commands/ordered-multinomial.md`
- Create: `docs/src/commands/dsge-hd.md`

- [ ] **Step 1: Add new pages to `docs/make.jl`**
- [ ] **Step 2: Create new command documentation pages**
- [ ] **Step 3: Update existing pages with new command references**
- [ ] **Step 4: Commit**

```bash
git add docs/
git commit -m "docs: add Documenter.jl pages for v0.4.0 features"
```

### Task 20: Update `build_app.jl`

**Files:**
- Modify: `build_app.jl`

- [ ] **Step 1: Add `spectral` to precompile dispatch list**

Find the precompile script section and add:
```julia
Friedman.dispatch(app, ["spectral", "--help"])
```

- [ ] **Step 2: FFTW is now a regular dep — verify build script handles it**

FFTW no longer appears in `[weakdeps]`, so the generic weak→real migration loop simply won't see it. No code change needed unless there's a hardcoded FFTW check.

- [ ] **Step 3: Commit**

```bash
git add build_app.jl
git commit -m "chore: update build_app.jl for v0.4.0 (spectral precompile, FFTW now regular dep)"
```

---

## Chunk 9: Final Verification

### Task 21: Full Test Suite Run

- [ ] **Step 1: Run full test suite**

Run: `julia --project test/runtests.jl`
Expected: All tests pass, no regressions.

- [ ] **Step 2: Verify build still works**

Run: `julia --project -e 'using Friedman; app = Friedman.build_app(); println(length(app.root.subcmds), " top-level commands")'`
Expected: `14 top-level commands`

- [ ] **Step 3: Smoke test new commands**

```bash
julia --project bin/friedman spectral --help
julia --project bin/friedman estimate preg --help
julia --project bin/friedman dsge hd --help
julia --project bin/friedman data dropna --help
julia --project bin/friedman test hausman --help
julia --project bin/friedman estimate ologit --help
```

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address final test/compilation issues for v0.4.0"
```
