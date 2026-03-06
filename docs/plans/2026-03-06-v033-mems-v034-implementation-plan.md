# Friedman-cli v0.3.3 — MEMs v0.3.4 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wrap MacroEconometricModels.jl v0.3.4 in Friedman-cli v0.3.3 — adding cross-sectional regression (reg/iv/logit/probit), 6 advanced unit root tests, LP-DiD engine replacement, Bayesian DSGE node expansion, and mpdta/ddcg datasets.

**Architecture:** Follows existing action-first CLI pattern: add LeafCommand registrations + handler functions in existing `src/commands/*.jl` files, mock types in `test/mocks.jl`, handler tests in `test/test_commands.jl`. No new source files. Shared helpers in `shared.jl` for regression data loading.

**Tech Stack:** Julia 1.12, MacroEconometricModels.jl v0.3.4, CSV, DataFrames, PrettyTables, JSON3

**Design doc:** `docs/plans/2026-03-06-v033-mems-v034-integration-design.md`

---

## Task 1: Version Bump

**Files:**
- Modify: `Project.toml:3` (version), `Project.toml:25` (compat)
- Modify: `src/Friedman.jl:53` (FRIEDMAN_VERSION)

**Step 1: Update version references**

In `Project.toml`, change:
```toml
version = "0.3.3"
# ...
[compat]
MacroEconometricModels = "0.3.4"
```

In `src/Friedman.jl`, change line 53:
```julia
const FRIEDMAN_VERSION = v"0.3.3"
```

**Step 2: Verify project loads**

Run: `julia --project -e 'using Friedman; println(Friedman.FRIEDMAN_VERSION)'`
Expected: `0.3.3`

**Step 3: Commit**
```bash
git add Project.toml src/Friedman.jl
git commit -m "chore: bump version to v0.3.3, MEMs compat to v0.3.4"
```

---

## Task 2: Shared Helpers for Regression

**Files:**
- Modify: `src/commands/shared.jl` (append after existing helpers, before EOF)

**Step 1: Add `_REG_COMMON_OPTIONS` const and `_load_reg_data` helper**

Append to `src/commands/shared.jl`:
```julia
# ── Regression Helpers ────────────────────────────────────

const _REG_COMMON_OPTIONS = [
    Option("dep"; type=String, default="", description="Dependent variable column name (default: first numeric column)"),
    Option("cov-type"; type=String, default="hc1", description="ols|hc0|hc1|hc2|hc3|cluster"),
    Option("clusters"; type=String, default="", description="Cluster variable column name"),
    Option("output"; short="o", type=String, default="", description="Export results to file"),
    Option("format"; short="f", type=String, default="table", description="table|csv|json"),
]

"""
    _load_reg_data(data, dep; weights_col="") → (y, X, varnames)

Load CSV, split into dependent variable y and regressor matrix X.
If dep is empty, uses first numeric column as y.
Returns (y::Vector, X::Matrix, varnames::Vector{String}) where varnames
are the regressor column names (excludes dep, weights, clusters).
"""
function _load_reg_data(data::String, dep::String; weights_col::String="", clusters_col::String="")
    df = load_data(data)
    numcols = variable_names(df)

    # Determine dependent variable
    if isempty(dep)
        dep_col = numcols[1]
    else
        dep_col = dep
        dep_col in numcols || error("dependent variable '$dep_col' not found in numeric columns: $numcols")
    end

    # Exclude dep, weights, clusters from regressors
    exclude = Set([dep_col])
    !isempty(weights_col) && push!(exclude, weights_col)
    !isempty(clusters_col) && push!(exclude, clusters_col)
    xcols = filter(c -> !(c in exclude), numcols)
    isempty(xcols) && error("no regressor columns remaining after excluding dep='$dep_col'")

    y = Vector{Float64}(df[!, dep_col])
    X = Matrix{Float64}(df[!, xcols])

    return y, X, xcols
end

"""
    _load_clusters(data, clusters_col) → Union{Vector{Int}, Nothing}

Load cluster assignments from a CSV column, or return nothing.
"""
function _load_clusters(data::String, clusters_col::String)
    isempty(clusters_col) && return nothing
    df = load_data(data)
    clusters_col in names(df) || error("cluster column '$clusters_col' not found")
    return Vector{Int}(df[!, clusters_col])
end

"""
    _load_weights(data, weights_col) → Union{Vector{Float64}, Nothing}

Load observation weights from a CSV column, or return nothing.
"""
function _load_weights(data::String, weights_col::String)
    isempty(weights_col) && return nothing
    df = load_data(data)
    weights_col in names(df) || error("weights column '$weights_col' not found")
    return Vector{Float64}(df[!, weights_col])
end

"""
    _reg_coef_table(model, varnames) → DataFrame

Build coefficient table from a regression model (RegModel/LogitModel/ProbitModel).
"""
function _reg_coef_table(model, varnames::Vector{String})
    b = coef(model)
    se = stderror(model)
    t = b ./ se
    p = [2.0 * (1.0 - _normal_cdf(abs(ti))) for ti in t]
    ci = confint(model)
    labels = length(b) == length(varnames) + 1 ? ["_cons"; varnames] : varnames
    DataFrame(
        Variable = labels,
        Coefficient = round.(b; digits=6),
        Std_Error = round.(se; digits=6),
        t_stat = round.(t; digits=4),
        p_value = round.(p; digits=4),
        CI_Lower = round.(ci[:, 1]; digits=6),
        CI_Upper = round.(ci[:, 2]; digits=6),
    )
end
```

**Step 2: Verify no syntax errors**

Run: `julia --project -e 'using Friedman'`
Expected: No errors

**Step 3: Commit**
```bash
git add src/commands/shared.jl
git commit -m "feat: add regression shared helpers (_load_reg_data, _reg_coef_table)"
```

---

## Task 3: Mock Types for Regression

**Files:**
- Modify: `test/mocks.jl` (append before final `end` of module)

**Step 1: Add mock regression types and functions**

Append to `test/mocks.jl` before the final `end`:
```julia
# ── Regression Module ──────────────────────────────────────

struct RegModel{T<:AbstractFloat}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}
    ssr::T; tss::T; r2::T; adj_r2::T; f_stat::T; f_pvalue::T
    loglik::T; aic::T; bic::T; nobs::Int; rank::Int; dof_resid::Int
    cov_type::Symbol; weights::Union{Vector{T},Nothing}; varnames::Vector{String}
    clusters::Union{Vector{Int},Nothing}
    # IV-specific fields
    method::Symbol; Z::Union{Matrix{T},Nothing}
    endogenous::Union{Vector{Int},Nothing}
    first_stage_f::Union{T,Nothing}; sargan_stat::Union{T,Nothing}; sargan_pval::Union{T,Nothing}
end

struct LogitModel{T<:AbstractFloat}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    nobs::Int; varnames::Vector{String}; converged::Bool; iterations::Int
    cov_type::Symbol
end

struct ProbitModel{T<:AbstractFloat}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; loglik_null::T; pseudo_r2::T; aic::T; bic::T
    nobs::Int; varnames::Vector{String}; converged::Bool; iterations::Int
    cov_type::Symbol
end

struct MarginalEffects{T<:AbstractFloat}
    effects::Vector{T}; se::Vector{T}; z_stat::Vector{T}; p_values::Vector{T}
    ci_lower::Vector{T}; ci_upper::Vector{T}; varnames::Vector{String}
    type::Symbol; conf_level::T
end

# StatsAPI for regression
import StatsAPI: coef, vcov, residuals, predict, confint, stderror, nobs, loglikelihood, aic, bic, r2

coef(m::RegModel) = m.beta
vcov(m::RegModel) = m.var_beta
residuals(m::RegModel) = m.residuals
predict(m::RegModel) = m.fitted
stderror(m::RegModel) = sqrt.(diag(m.var_beta))
nobs(m::RegModel) = m.nobs
loglikelihood(m::RegModel) = m.loglik
aic(m::RegModel) = m.aic
bic(m::RegModel) = m.bic
r2(m::RegModel) = m.r2
confint(m::RegModel) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

coef(m::LogitModel) = m.beta
vcov(m::LogitModel) = m.var_beta
residuals(m::LogitModel) = m.residuals
predict(m::LogitModel) = m.fitted
stderror(m::LogitModel) = sqrt.(diag(m.var_beta))
nobs(m::LogitModel) = m.nobs
loglikelihood(m::LogitModel) = m.loglik
aic(m::LogitModel) = m.aic
bic(m::LogitModel) = m.bic
confint(m::LogitModel) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

coef(m::ProbitModel) = m.beta
vcov(m::ProbitModel) = m.var_beta
residuals(m::ProbitModel) = m.residuals
predict(m::ProbitModel) = m.fitted
stderror(m::ProbitModel) = sqrt.(diag(m.var_beta))
nobs(m::ProbitModel) = m.nobs
loglikelihood(m::ProbitModel) = m.loglik
aic(m::ProbitModel) = m.aic
bic(m::ProbitModel) = m.bic
confint(m::ProbitModel) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# Mock estimation functions
function estimate_reg(y::AbstractVector{T}, X::AbstractMatrix; cov_type=:hc1,
        weights=nothing, varnames=nothing, clusters=nothing) where T
    n, k = size(X)
    beta = randn(T, k + 1)  # +1 for constant
    vb = Matrix{T}(I(k + 1)) * T(0.01)
    res = randn(T, n); fit = randn(T, n)
    vn = isnothing(varnames) ? ["x$i" for i in 1:k] : varnames
    cl = isnothing(clusters) ? nothing : Vector{Int}(clusters)
    RegModel{T}(y, X, beta, vb, res, fit,
        T(1.0), T(10.0), T(0.9), T(0.88), T(45.0), T(0.001),
        T(-100.0), T(210.0), T(220.0), n, k + 1, n - k - 1,
        cov_type, isnothing(weights) ? nothing : Vector{T}(weights),
        vn, cl, :ols, nothing, nothing, nothing, nothing, nothing)
end

function estimate_iv(y::AbstractVector{T}, X::AbstractMatrix, Z::AbstractMatrix;
        endogenous=Int[], cov_type=:hc1, varnames=nothing) where T
    n, k = size(X)
    beta = randn(T, k + 1)
    vb = Matrix{T}(I(k + 1)) * T(0.01)
    res = randn(T, n); fit = randn(T, n)
    vn = isnothing(varnames) ? ["x$i" for i in 1:k] : varnames
    RegModel{T}(y, X, beta, vb, res, fit,
        T(1.0), T(10.0), T(0.85), T(0.83), T(30.0), T(0.005),
        T(-110.0), T(230.0), T(240.0), n, k + 1, n - k - 1,
        cov_type, nothing, vn, nothing,
        :iv, Z, endogenous, T(25.0), T(1.5), T(0.47))
end

function estimate_logit(y::AbstractVector{T}, X::AbstractMatrix; cov_type=:ols,
        varnames=nothing, clusters=nothing, maxiter=100, tol=1e-8) where T
    n, k = size(X)
    beta = randn(T, k + 1)
    vb = Matrix{T}(I(k + 1)) * T(0.01)
    res = randn(T, n); fit = rand(T, n)  # probabilities
    vn = isnothing(varnames) ? ["x$i" for i in 1:k] : varnames
    LogitModel{T}(y, X, beta, vb, res, fit,
        T(-50.0), T(-70.0), T(0.286), T(110.0), T(120.0),
        n, vn, true, 5, cov_type)
end

function estimate_probit(y::AbstractVector{T}, X::AbstractMatrix; cov_type=:ols,
        varnames=nothing, clusters=nothing, maxiter=100, tol=1e-8) where T
    n, k = size(X)
    beta = randn(T, k + 1)
    vb = Matrix{T}(I(k + 1)) * T(0.01)
    res = randn(T, n); fit = rand(T, n)
    vn = isnothing(varnames) ? ["x$i" for i in 1:k] : varnames
    ProbitModel{T}(y, X, beta, vb, res, fit,
        T(-55.0), T(-75.0), T(0.267), T(120.0), T(130.0),
        n, vn, true, 6, cov_type)
end

function marginal_effects(m::Union{LogitModel{T},ProbitModel{T}};
        type=:ame, at=nothing, conf_level=0.95) where T
    k = length(m.beta)
    eff = randn(T, k); se = abs.(randn(T, k)) .* T(0.1)
    z = eff ./ se; p = abs.(randn(T, k)) .* T(0.1)
    MarginalEffects{T}(eff, se, z, p, eff .- T(1.96) .* se, eff .+ T(1.96) .* se,
        ["_cons"; m.varnames], type, T(conf_level))
end

function odds_ratio(m::LogitModel{T}; conf_level=0.95) where T
    or = exp.(m.beta)
    se = or .* stderror(m)
    (odds_ratio=or, se=se, ci_lower=or .- T(1.96) .* se, ci_upper=or .+ T(1.96) .* se,
     varnames=["_cons"; m.varnames])
end

function vif(m::RegModel{T}) where T
    k = length(m.beta) - 1  # exclude constant
    fill(T(2.5), k)
end

function classification_table(m::Union{LogitModel,ProbitModel}; threshold=0.5)
    Dict{String,Any}("accuracy" => 0.85, "sensitivity" => 0.80,
        "specificity" => 0.90, "true_pos" => 40, "true_neg" => 45,
        "false_pos" => 5, "false_neg" => 10, "n" => nobs(m))
end

export RegModel, LogitModel, ProbitModel, MarginalEffects
export estimate_reg, estimate_iv, estimate_logit, estimate_probit
export marginal_effects, odds_ratio, vif, classification_table
```

**Step 2: Verify mocks compile**

Run: `julia --project -e 'include("test/mocks.jl"); using .MacroEconometricModels; println("OK")'`
Expected: `OK`

**Step 3: Commit**
```bash
git add test/mocks.jl
git commit -m "test: add mock types for regression module (RegModel, LogitModel, ProbitModel)"
```

---

## Task 4: Estimate Regression Commands (4 leaves)

**Files:**
- Modify: `src/commands/estimate.jl` — add 4 leaf registrations in `register_estimate_commands!()` + 4 handlers at EOF

**Step 1: Add leaf registrations**

In `register_estimate_commands!()`, add before the `subcmds` dict (before line 292):
```julia
    est_reg = LeafCommand("reg", _estimate_reg;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("weights"; type=String, default="", description="Weight column name (WLS)"),
        ],
        description="OLS/WLS linear regression")

    est_iv = LeafCommand("iv", _estimate_iv;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("dep"; type=String, default="", description="Dependent variable column name"),
            Option("endogenous"; type=String, default="", description="Endogenous regressor column names (comma-separated, required)"),
            Option("instruments"; type=String, default="", description="Instrument column names (comma-separated, required)"),
            Option("cov-type"; type=String, default="hc1", description="ols|hc0|hc1|hc2|hc3"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Instrumental variables (2SLS) regression")

    est_logit = LeafCommand("logit", _estimate_logit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("maxiter"; type=Int, default=100, description="Max IRLS iterations"),
            Option("tol"; type=Float64, default=1e-8, description="Convergence tolerance"),
        ],
        description="Binary logit regression (MLE via IRLS)")

    est_probit = LeafCommand("probit", _estimate_probit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            _REG_COMMON_OPTIONS...,
            Option("maxiter"; type=Int, default=100, description="Max IRLS iterations"),
            Option("tol"; type=Float64, default=1e-8, description="Convergence tolerance"),
        ],
        description="Binary probit regression (MLE via IRLS)")
```

Add to `subcmds` dict (after `"sdfm" => est_sdfm,`):
```julia
        "reg"       => est_reg,
        "iv"        => est_iv,
        "logit"     => est_logit,
        "probit"    => est_probit,
```

**Step 2: Add handlers at EOF of estimate.jl**

```julia
# ── OLS/WLS Regression ────────────────────────────────────

function _estimate_reg(; data::String, dep::String="", cov_type::String="hc1",
                        weights::String="", clusters::String="",
                        output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; weights_col=weights, clusters_col=clusters)
    w = _load_weights(data, weights)
    cl = _load_clusters(data, clusters)
    n, k = size(X)

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    println("OLS/WLS Regression: $dep_name ~ $(join(xcols, " + "))")
    println("  N=$n, K=$k, cov_type=$cov_type")
    println()

    model = estimate_reg(y, X; cov_type=Symbol(cov_type), weights=w,
                         varnames=xcols, clusters=cl)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="OLS/WLS Regression")
    println()
    printstyled("  R²: $(round(r2(model); digits=4))"; color=:cyan)
    println("  Adj R²: $(round(model.adj_r2; digits=4))")
    printstyled("  F-stat: $(round(model.f_stat; digits=4))"; color=:cyan)
    println("  (p=$(round(model.f_pvalue; digits=4)))")
    printstyled("  Log-lik: $(round(loglikelihood(model); digits=2))"; color=:cyan)
    println("  AIC: $(round(aic(model); digits=2))  BIC: $(round(bic(model); digits=2))")
end

# ── IV/2SLS ───────────────────────────────────────────────

function _estimate_iv(; data::String, dep::String="", endogenous::String="",
                       instruments::String="", cov_type::String="hc1",
                       output::String="", format::String="table")
    isempty(endogenous) && error("--endogenous is required (comma-separated column names)")
    isempty(instruments) && error("--instruments is required (comma-separated column names)")

    df = load_data(data)
    numcols = variable_names(df)

    dep_col = isempty(dep) ? numcols[1] : dep
    dep_col in numcols || error("dependent variable '$dep_col' not found")

    endog_names = String.(strip.(split(endogenous, ",")))
    inst_names = String.(strip.(split(instruments, ",")))
    for e in endog_names
        e in numcols || error("endogenous variable '$e' not found")
    end
    for z in inst_names
        z in numcols || error("instrument '$z' not found")
    end

    # Build X (regressors excl dep) and Z (instruments)
    xcols = filter(c -> c != dep_col, numcols)
    endog_idx = [findfirst(==(e), xcols) for e in endog_names]

    y = Vector{Float64}(df[!, dep_col])
    X = Matrix{Float64}(df[!, xcols])
    Z = Matrix{Float64}(df[!, inst_names])

    println("IV/2SLS Regression: $dep_col ~ $(join(xcols, " + "))")
    println("  Endogenous: $(join(endog_names, ", "))")
    println("  Instruments: $(join(inst_names, ", "))")
    println()

    model = estimate_iv(y, X, Z; endogenous=endog_idx,
                        cov_type=Symbol(cov_type), varnames=xcols)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="IV/2SLS Regression")
    println()
    printstyled("  R²: $(round(r2(model); digits=4))"; color=:cyan)
    println("  Adj R²: $(round(model.adj_r2; digits=4))")
    if !isnothing(model.first_stage_f)
        printstyled("  First-stage F: $(round(model.first_stage_f; digits=2))\n"; color=:cyan)
    end
    if !isnothing(model.sargan_stat)
        printstyled("  Sargan stat: $(round(model.sargan_stat; digits=4))"; color=:cyan)
        println("  (p=$(round(model.sargan_pval; digits=4)))")
    end
end

# ── Logit ─────────────────────────────────────────────────

function _estimate_logit(; data::String, dep::String="", cov_type::String="ols",
                          clusters::String="", maxiter::Int=100, tol::Float64=1e-8,
                          output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
    cl = _load_clusters(data, clusters)

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    println("Logit Regression: $dep_name ~ $(join(xcols, " + "))")
    println("  N=$(length(y)), cov_type=$cov_type")
    println()

    model = estimate_logit(y, X; cov_type=Symbol(cov_type), varnames=xcols,
                           clusters=cl, maxiter=maxiter, tol=tol)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="Logit Regression")
    println()
    printstyled("  Pseudo R²: $(round(model.pseudo_r2; digits=4))\n"; color=:cyan)
    printstyled("  Log-lik: $(round(loglikelihood(model); digits=2))"; color=:cyan)
    println("  (null: $(round(model.loglik_null; digits=2)))")
    printstyled("  AIC: $(round(aic(model); digits=2))  BIC: $(round(bic(model); digits=2))\n"; color=:cyan)
    printstyled("  Converged: $(model.converged) ($(model.iterations) iterations)\n"; color=:cyan)
end

# ── Probit ────────────────────────────────────────────────

function _estimate_probit(; data::String, dep::String="", cov_type::String="ols",
                           clusters::String="", maxiter::Int=100, tol::Float64=1e-8,
                           output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
    cl = _load_clusters(data, clusters)

    dep_name = isempty(dep) ? variable_names(load_data(data))[1] : dep
    println("Probit Regression: $dep_name ~ $(join(xcols, " + "))")
    println("  N=$(length(y)), cov_type=$cov_type")
    println()

    model = estimate_probit(y, X; cov_type=Symbol(cov_type), varnames=xcols,
                            clusters=cl, maxiter=maxiter, tol=tol)

    coef_df = _reg_coef_table(model, xcols)
    output_result(coef_df; format=Symbol(format), output=output,
                  title="Probit Regression")
    println()
    printstyled("  Pseudo R²: $(round(model.pseudo_r2; digits=4))\n"; color=:cyan)
    printstyled("  Log-lik: $(round(loglikelihood(model); digits=2))"; color=:cyan)
    println("  (null: $(round(model.loglik_null; digits=2)))")
    printstyled("  AIC: $(round(aic(model); digits=2))  BIC: $(round(bic(model); digits=2))\n"; color=:cyan)
    printstyled("  Converged: $(model.converged) ($(model.iterations) iterations)\n"; color=:cyan)
end
```

**Step 3: Verify compilation**

Run: `julia --project -e 'using Friedman'`

**Step 4: Add handler tests**

Append to `test/test_commands.jl` inside the handler tests section:
```julia
@testset "_estimate_reg" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=4)
        out = _capture() do
            _estimate_reg(; data=csv, dep="var1", cov_type="hc1", format="table")
        end
        @test occursin("OLS/WLS", out)
        @test occursin("R²", out)
    end
end

@testset "_estimate_iv" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=5, colnames=["y", "x1", "x2", "z1", "z2"])
        out = _capture() do
            _estimate_iv(; data=csv, dep="y", endogenous="x1", instruments="z1,z2", format="table")
        end
        @test occursin("IV/2SLS", out)
        @test occursin("First-stage F", out) || occursin("Sargan", out)
    end
end

@testset "_estimate_logit" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=4)
        out = _capture() do
            _estimate_logit(; data=csv, dep="var1", cov_type="ols", format="table")
        end
        @test occursin("Logit", out)
        @test occursin("Pseudo R²", out)
    end
end

@testset "_estimate_probit" begin
    mktempdir() do dir
        csv = _make_csv(dir; T=100, n=4)
        out = _capture() do
            _estimate_probit(; data=csv, dep="var1", cov_type="ols", format="table")
        end
        @test occursin("Probit", out)
        @test occursin("Pseudo R²", out)
    end
end
```

**Step 5: Run tests**

Run: `julia --project test/test_commands.jl`

**Step 6: Commit**
```bash
git add src/commands/estimate.jl test/test_commands.jl
git commit -m "feat: add estimate reg/iv/logit/probit commands"
```

---

## Task 5: Test Commands — 6 Unit Root Tests + VIF (7 leaves)

**Files:**
- Modify: `test/mocks.jl` — add 6 unit root result types + mock functions
- Modify: `src/commands/test.jl` — add 7 leaf registrations + 7 handlers
- Modify: `test/test_commands.jl` — add 7 handler tests

**Step 1: Add mock unit root types to `test/mocks.jl`**

```julia
# ── Advanced Unit Root Tests ───────────────────────────────

struct FourierADFResult{T<:AbstractFloat}
    statistic::T; pvalue::T; frequency::Int; f_statistic::T; f_pvalue::T
    lags::Int; regression::Symbol
    critical_values::Dict{Int,T}; f_critical_values::Dict{Int,T}; nobs::Int
end

struct FourierKPSSResult{T<:AbstractFloat}
    statistic::T; pvalue::T; frequency::Int; f_statistic::T; f_pvalue::T
    regression::Symbol; critical_values::Dict{Int,T}; f_critical_values::Dict{Int,T}
    bandwidth::Int; nobs::Int
end

struct DFGLSResult{T<:AbstractFloat}
    tau_statistic::T; pt_statistic::T; mgls_statistics::Dict{Symbol,T}
    pvalue::T; lags::Int; regression::Symbol; critical_values::Dict{Int,T}; nobs::Int
end

struct LMUnitRootResult{T<:AbstractFloat}
    statistic::T; pvalue::T; break_indices::Union{Nothing,Vector{Int}}
    break_fractions::Union{Nothing,Vector{T}}; breaks::Int; regression::Symbol
    critical_values::Dict{Int,T}; lags::Int; nobs::Int
end

struct ADF2BreakResult{T<:AbstractFloat}
    statistic::T; pvalue::T; break_index1::Int; break_index2::Int
    break_fraction1::T; break_fraction2::T; lags::Int; model::Symbol
    critical_values::Dict{Int,T}; nobs::Int
end

struct GregoryHansenResult{T<:AbstractFloat}
    adf_statistic::T; adf_pvalue::T; adf_break_index::Int
    zt_statistic::T; zt_pvalue::T; zt_break_index::Int
    za_statistic::T; za_pvalue::T; za_break_index::Int
    model::Symbol; critical_values::Dict{Int,T}; nobs::Int
end

function fourier_adf_test(y::AbstractVector{T}; regression=:constant,
        fmax=3, lags=:aic, max_lags=nothing, trim=0.15) where T
    n = length(y)
    cvs = Dict(1 => T(-4.42), 5 => T(-3.81), 10 => T(-3.49))
    FourierADFResult{T}(T(-3.95), T(0.03), 1, T(8.5), T(0.001),
        4, regression, cvs, cvs, n)
end

function fourier_kpss_test(y::AbstractVector{T}; regression=:constant,
        fmax=3, bandwidth=nothing) where T
    n = length(y)
    cvs = Dict(1 => T(0.74), 5 => T(0.46), 10 => T(0.35))
    FourierKPSSResult{T}(T(0.12), T(0.42), 1, T(6.2), T(0.005),
        regression, cvs, cvs, 4, n)
end

function dfgls_test(y::AbstractVector{T}; regression=:constant,
        lags=:aic, max_lags=nothing) where T
    n = length(y)
    cvs = Dict(1 => T(-2.58), 5 => T(-1.94), 10 => T(-1.62))
    mgls = Dict{Symbol,T}(:mza => T(-15.0), :msb => T(0.18), :mzt => T(-2.7))
    DFGLSResult{T}(T(-2.15), T(-12.0), mgls, T(0.08), 3, regression, cvs, n)
end

function lm_unitroot_test(y::AbstractVector{T}; breaks=0, regression=:level,
        lags=:aic, max_lags=nothing, trim=0.15) where T
    n = length(y)
    cvs = Dict(1 => T(-3.56), 5 => T(-3.06), 10 => T(-2.75))
    bi = breaks == 0 ? nothing : collect(div(n, 3):(div(n, 3)):(breaks * div(n, 3)))
    bf = isnothing(bi) ? nothing : T.(bi ./ n)
    LMUnitRootResult{T}(T(-3.2), T(0.04), bi, bf, breaks, regression, cvs, 4, n)
end

function adf_2break_test(y::AbstractVector{T}; model=:level,
        lags=:aic, max_lags=nothing, trim=0.10) where T
    n = length(y)
    b1, b2 = div(n, 3), 2 * div(n, 3)
    cvs = Dict(1 => T(-5.65), 5 => T(-5.07), 10 => T(-4.78))
    ADF2BreakResult{T}(T(-5.3), T(0.02), b1, b2, T(b1/n), T(b2/n),
        4, model, cvs, n)
end

function gregory_hansen_test(Y::AbstractMatrix{T}; model=:C,
        lags=:aic, max_lags=nothing, trim=0.15) where T
    n = size(Y, 1)
    bp = div(n, 2)
    cvs = Dict(1 => T(-5.77), 5 => T(-5.28), 10 => T(-5.02))
    GregoryHansenResult{T}(T(-5.4), T(0.03), bp,
        T(-5.1), T(0.05), bp, T(-48.0), T(0.04), bp,
        model, cvs, n)
end

export FourierADFResult, FourierKPSSResult, DFGLSResult
export LMUnitRootResult, ADF2BreakResult, GregoryHansenResult
export fourier_adf_test, fourier_kpss_test, dfgls_test
export lm_unitroot_test, adf_2break_test, gregory_hansen_test
```

**Step 2: Add leaf registrations + handlers in `src/commands/test.jl`**

Add 7 leaf registrations before the `subcmds` dict. Add to `subcmds`:
```julia
        "fourier-adf"    => test_fourier_adf,
        "fourier-kpss"   => test_fourier_kpss,
        "dfgls"          => test_dfgls,
        "lm-unitroot"    => test_lm_unitroot,
        "adf-2break"     => test_adf_2break,
        "gregory-hansen" => test_gregory_hansen,
        "vif"            => test_vif,
```

Leaf registration pattern (repeat for each test — use same pattern as `test_adf`/`test_za` with appropriate options per the design doc).

Handler pattern (example for `_test_fourier_adf`):
```julia
function _test_fourier_adf(; data::String, column::Int=1,
        regression::String="constant", fmax::Int=3,
        lags::String="aic", max_lags=nothing, trim::Float64=0.15,
        format::String="table", output::String="")
    y, vname = load_univariate_series(data, column)
    lags_arg = tryparse(Int, lags) !== nothing ? parse(Int, lags) : Symbol(lags)
    println("Fourier ADF Test: $vname, regression=$regression, fmax=$fmax")
    println()

    result = fourier_adf_test(y; regression=Symbol(regression), fmax=fmax,
        lags=lags_arg, max_lags=max_lags, trim=trim)

    pairs = Pair{String,Any}[
        "Test statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
        "Selected frequency" => result.frequency,
        "F-statistic (Fourier)" => round(result.f_statistic; digits=4),
        "F p-value" => round(result.f_pvalue; digits=4),
        "Lags" => result.lags,
        "Observations" => result.nobs,
    ]
    output_kv(pairs; format=format, output=output, title="Fourier ADF Test")

    interpret_test_result(result.pvalue,
        "Reject H0: series is stationary (with Fourier components)",
        "Cannot reject H0: unit root (Fourier ADF)")
end
```

Follow same pattern for fourier-kpss, dfgls, lm-unitroot, adf-2break, gregory-hansen. Each outputs key-value pairs via `output_kv`.

For `_test_vif`:
```julia
function _test_vif(; data::String, dep::String="", cov_type::String="hc1",
                    format::String="table", output::String="")
    y, X, xcols = _load_reg_data(data, dep)
    model = estimate_reg(y, X; cov_type=Symbol(cov_type), varnames=xcols)
    vif_values = vif(model)

    vif_df = DataFrame(
        Variable = xcols,
        VIF = round.(vif_values; digits=4),
    )
    output_result(vif_df; format=Symbol(format), output=output,
                  title="Variance Inflation Factors")

    max_vif = maximum(vif_values)
    if max_vif > 10.0
        printstyled("  Warning: VIF > 10 detected — severe multicollinearity\n"; color=:red)
    elseif max_vif > 5.0
        printstyled("  Warning: VIF > 5 detected — moderate multicollinearity\n"; color=:yellow)
    else
        printstyled("  All VIF < 5 — no concerning multicollinearity\n"; color=:green)
    end
end
```

**Step 3: Add handler tests in `test/test_commands.jl`**

One test per handler (same pattern as `_test_andrews` tests).

**Step 4: Run tests**

Run: `julia --project test/test_commands.jl`

**Step 5: Commit**
```bash
git add src/commands/test.jl test/mocks.jl test/test_commands.jl
git commit -m "feat: add test fourier-adf/fourier-kpss/dfgls/lm-unitroot/adf-2break/gregory-hansen/vif commands"
```

---

## Task 6: Predict & Residuals for Regression (3+3 leaves)

**Files:**
- Modify: `src/commands/predict.jl` — add 3 leaf registrations + 3 handlers
- Modify: `src/commands/residuals.jl` — add 3 leaf registrations + 3 handlers
- Modify: `test/test_commands.jl` — add 6 handler tests

**Step 1: Add predict leaves + handlers**

In `register_predict_commands!()`, add registrations before `subcmds` dict:
```julia
    pred_reg = LeafCommand("reg", _predict_reg;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...,
            Option("weights"; type=String, default="", description="Weight column name (WLS)"),
        ],
        description="OLS/WLS in-sample fitted values")

    pred_logit = LeafCommand("logit", _predict_logit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...,
            Option("threshold"; type=Float64, default=0.5, description="Classification threshold"),
        ],
        flags=[
            Flag("marginal-effects"; description="Output marginal effects instead of fitted values"),
            Flag("odds-ratio"; description="Output odds ratios (logit only)"),
            Flag("classification-table"; description="Output classification table"),
        ],
        description="Logit in-sample fitted probabilities")

    pred_probit = LeafCommand("probit", _predict_probit;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[_REG_COMMON_OPTIONS...,
            Option("threshold"; type=Float64, default=0.5, description="Classification threshold"),
        ],
        flags=[
            Flag("marginal-effects"; description="Output marginal effects instead of fitted values"),
            Flag("classification-table"; description="Output classification table"),
        ],
        description="Probit in-sample fitted probabilities")
```

Add to `subcmds`: `"reg" => pred_reg, "logit" => pred_logit, "probit" => pred_probit,`

Handler pattern for `_predict_reg`:
```julia
function _predict_reg(; data::String, dep::String="", cov_type::String="hc1",
                       weights::String="", clusters::String="",
                       output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; weights_col=weights, clusters_col=clusters)
    w = _load_weights(data, weights)
    cl = _load_clusters(data, clusters)
    model = estimate_reg(y, X; cov_type=Symbol(cov_type), weights=w,
                         varnames=xcols, clusters=cl)
    fitted = predict(model)
    pred_df = DataFrame(observation=1:length(fitted), fitted_value=round.(fitted; digits=6))
    output_result(pred_df; format=Symbol(format), output=output,
                  title="OLS/WLS Fitted Values")
end
```

Handler for `_predict_logit` (handles `--marginal-effects`, `--odds-ratio`, `--classification-table` flags):
```julia
function _predict_logit(; data::String, dep::String="", cov_type::String="ols",
                         clusters::String="", threshold::Float64=0.5,
                         marginal_effects::Bool=false, odds_ratio::Bool=false,
                         classification_table::Bool=false,
                         output::String="", format::String="table")
    y, X, xcols = _load_reg_data(data, dep; clusters_col=clusters)
    cl = _load_clusters(data, clusters)
    model = estimate_logit(y, X; cov_type=Symbol(cov_type), varnames=xcols, clusters=cl)

    if marginal_effects
        me = MacroEconometricModels.marginal_effects(model)
        me_df = DataFrame(Variable=me.varnames, Effect=round.(me.effects; digits=6),
            SE=round.(me.se; digits=6), z=round.(me.z_stat; digits=4),
            p_value=round.(me.p_values; digits=4),
            CI_Lower=round.(me.ci_lower; digits=6), CI_Upper=round.(me.ci_upper; digits=6))
        output_result(me_df; format=Symbol(format), output=output,
                      title="Average Marginal Effects (Logit)")
    elseif odds_ratio
        or = MacroEconometricModels.odds_ratio(model)
        or_df = DataFrame(Variable=or.varnames, Odds_Ratio=round.(or.odds_ratio; digits=6),
            SE=round.(or.se; digits=6),
            CI_Lower=round.(or.ci_lower; digits=6), CI_Upper=round.(or.ci_upper; digits=6))
        output_result(or_df; format=Symbol(format), output=output,
                      title="Odds Ratios (Logit)")
    elseif classification_table
        ct = MacroEconometricModels.classification_table(model; threshold=threshold)
        println("Classification Table (threshold=$threshold):")
        println("  Accuracy:    $(round(ct["accuracy"]; digits=4))")
        println("  Sensitivity: $(round(ct["sensitivity"]; digits=4))")
        println("  Specificity: $(round(ct["specificity"]; digits=4))")
        println("  True Pos: $(ct["true_pos"]), True Neg: $(ct["true_neg"])")
        println("  False Pos: $(ct["false_pos"]), False Neg: $(ct["false_neg"])")
    else
        fitted = predict(model)
        pred_df = DataFrame(observation=1:length(fitted), fitted_prob=round.(fitted; digits=6))
        output_result(pred_df; format=Symbol(format), output=output,
                      title="Logit Fitted Probabilities")
    end
end
```

Follow same pattern for `_predict_probit` (no odds-ratio flag).

**Step 2: Add residuals leaves + handlers** — same pattern, call `residuals(model)` instead of `predict(model)`.

**Step 3: Add tests, run, commit**
```bash
git add src/commands/predict.jl src/commands/residuals.jl test/test_commands.jl
git commit -m "feat: add predict/residuals reg/logit/probit commands"
```

---

## Task 7: DSGE Bayes Node Expansion (leaf → 7-leaf NodeCommand)

**Files:**
- Modify: `test/mocks.jl` — add BayesianDSGESimulation mock, irf/fevd/simulate dispatches on BayesianDSGE, posterior_summary, marginal_likelihood, bayes_factor, prior_posterior_table, posterior_predictive
- Modify: `src/commands/dsge.jl` — convert `bayes` from LeafCommand to NodeCommand with 7 leaves, add 6 new handlers
- Modify: `test/test_commands.jl` — add 7 handler tests (replace existing _dsge_bayes test)

**Step 1: Add mock types/functions to `test/mocks.jl`**

```julia
# ── Bayesian DSGE Enhancements ─────────────────────────────

struct BayesianDSGESimulation{T<:AbstractFloat}
    quantiles::Array{T,3}; point_estimate::Matrix{T}; all_paths::Array{T,3}
    variables::Vector{String}; quantile_levels::Vector{T}
end

# Add dispatches on BayesianDSGE
function irf(result::BayesianDSGE{T}, horizon::Int;
        n_draws=200, quantiles=[0.05, 0.16, 0.84, 0.95],
        solver=:gensys, solver_kwargs=NamedTuple(), rng=nothing) where T
    nv = length(result.param_names)
    ns = max(1, nv)
    q = Array{T,4}(undef, horizon, nv, ns, length(quantiles))
    fill!(q, T(0.1))
    m = zeros(T, horizon, nv, ns)
    BayesianImpulseResponse{T}(q, m, horizon,
        String.(result.param_names), ["shock$i" for i in 1:ns], T.(quantiles))
end

function fevd(result::BayesianDSGE{T}, horizon::Int;
        n_draws=200, quantiles=[0.05, 0.16, 0.84, 0.95],
        solver=:gensys, solver_kwargs=NamedTuple(), rng=nothing) where T
    nv = length(result.param_names)
    ns = max(1, nv)
    q = Array{T,4}(undef, horizon, nv, ns, length(quantiles))
    fill!(q, T(1.0 / ns))
    m = fill(T(1.0 / ns), horizon, nv, ns)
    BayesianFEVD{T}(q, m, horizon,
        String.(result.param_names), ["shock$i" for i in 1:ns], T.(quantiles))
end

function simulate(result::BayesianDSGE{T}, T_periods::Int;
        n_draws=200, quantiles=[0.05, 0.16, 0.84, 0.95],
        solver=:gensys, solver_kwargs=NamedTuple(), rng=nothing) where T
    nv = length(result.param_names)
    nq = length(quantiles)
    q = randn(T, T_periods, nv, nq)
    pe = randn(T, T_periods, nv)
    ap = randn(T, n_draws, T_periods, nv)
    BayesianDSGESimulation{T}(q, pe, ap, String.(result.param_names), T.(quantiles))
end

function posterior_summary(result::BayesianDSGE{T}) where T
    Dict(p => Dict(:mean => T(0.5), :median => T(0.49), :std => T(0.1),
        :q05 => T(0.3), :q95 => T(0.7)) for p in result.param_names)
end

function bayes_factor(r1::BayesianDSGE, r2::BayesianDSGE)
    exp(r1.log_marginal_likelihood - r2.log_marginal_likelihood)
end

function prior_posterior_table(result::BayesianDSGE{T}) where T
    [(param=p, prior_mean=T(0.5), prior_std=T(0.2),
      post_mean=T(0.5), post_std=T(0.1), post_q05=T(0.3), post_q95=T(0.7))
     for p in result.param_names]
end

function posterior_predictive(result::BayesianDSGE{T}, n_sim::Int;
        T_periods=100, rng=nothing) where T
    nv = length(result.param_names)
    randn(T, n_sim, T_periods, nv)
end

export BayesianDSGESimulation
export posterior_summary, bayes_factor, prior_posterior_table, posterior_predictive
```

**Step 2: Modify `src/commands/dsge.jl`**

Replace the `dsge_bayes` LeafCommand with a NodeCommand containing 7 leaves. In the `subcmds` dict, replace `"bayes" => dsge_bayes` with `"bayes" => bayes_node`.

Build `bayes_node` from 7 leaf registrations sharing common options via a `_DSGE_BAYES_COMMON_OPTIONS` const. Each leaf: estimate, irf, fevd, simulate, summary, compare, predictive.

The existing `_dsge_bayes` handler becomes `_dsge_bayes_estimate`. Add 6 new handlers: `_dsge_bayes_irf`, `_dsge_bayes_fevd`, `_dsge_bayes_simulate`, `_dsge_bayes_summary`, `_dsge_bayes_compare`, `_dsge_bayes_predictive`.

Each handler (except estimate) internally calls `_dsge_bayes_estimate_internal` to get the `BayesianDSGE` result, then calls the appropriate post-estimation function.

Add a shared internal helper:
```julia
function _dsge_bayes_estimate_internal(; model, data, params, priors, sampler,
        n_smc, n_particles, n_draws, burnin, ess_target, observables,
        solver, order, delayed_acceptance)
    # ... same logic as current _dsge_bayes, but returns the result instead of printing
end
```

**Step 3: Add tests, run, commit**
```bash
git add src/commands/dsge.jl test/mocks.jl test/test_commands.jl
git commit -m "feat: expand dsge bayes to NodeCommand with estimate/irf/fevd/simulate/summary/compare/predictive"
```

---

## Task 8: DID Enhancements (lp-did replacement + base-period)

**Files:**
- Modify: `test/mocks.jl` — add LPDiDResult mock type, update estimate_lp_did mock, add base_period to estimate_did
- Modify: `src/commands/did.jl` — replace lp-did handler, add --base-period option to estimate leaf
- Modify: `test/test_commands.jl` — update lp-did tests, add base-period test

**Step 1: Add LPDiDResult mock and update estimate_lp_did**

In `test/mocks.jl`:
```julia
struct LPDiDResult{T<:AbstractFloat}
    coefficients::Vector{T}; se_vec::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}
    event_times::Vector{Int}; reference_period::Int; nobs_h::Vector{Int}
    pooled_post_result::Union{NamedTuple,Nothing}; pooled_pre_result::Union{NamedTuple,Nothing}
    vcov_all::Vector; outcome_name::String; treatment_name::String
    T_obs::Int; n_groups::Int; spec_type::Symbol
    pmd::Union{Nothing,Symbol,Int}; reweight::Bool; nocomp::Bool
    ylags::Int; dylags::Int; pre_window::Int; post_window::Int
    cluster::Symbol; conf_level::T; pd::PanelData{T}
end

# Update estimate_lp_did to return LPDiDResult
function estimate_lp_did(pd::PanelData{T}, outcome, treatment, H::Int;
        pre_window=3, post_window=H, ylags=0, dylags=0,
        covariates=String[], nonabsorbing=nothing, notyet=false,
        nevertreated=false, firsttreat=false, oneoff=false,
        pmd=nothing, reweight=false, nocomp=false,
        cluster=:unit, conf_level=0.95,
        only_pooled=false, only_event=false,
        post_pooled=nothing, pre_pooled=nothing) where T
    nt = pre_window + post_window + 1
    et = collect(-pre_window:post_window)
    c = randn(T, nt); se = abs.(randn(T, nt)) .* T(0.1)
    pp = (coef=T(0.5), se=T(0.1), ci_lower=T(0.3), ci_upper=T(0.7), nobs=100)
    spec = oneoff ? :oneoff : (isnothing(nonabsorbing) ? :absorbing : :nonabsorbing)
    LPDiDResult{T}(c, se, c .- T(1.96) .* se, c .+ T(1.96) .* se,
        et, -1, fill(100, nt), pp, pp, Matrix{T}[],
        String(outcome), String(treatment), pd.T_obs, pd.n_groups,
        spec, pmd, reweight, nocomp, ylags, dylags, pre_window, post_window,
        cluster, T(conf_level), pd)
end

export LPDiDResult
```

Also add `base_period` kwarg to the existing `estimate_did` mock.

**Step 2: Update `src/commands/did.jl`**

Replace `did_lp_did` leaf registration with expanded options per design doc. Replace `_did_lp_did` handler:
```julia
function _did_lp_did(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        horizon::Int=5, pre_window::Int=3, post_window::Int=0,
        ylags::Int=0, dylags::Int=0,
        covariates::String="", cluster::String="unit",
        pmd::String="", reweight::Bool=false, nocomp::Bool=false,
        nonabsorbing::String="", notyet::Bool=false, nevertreated::Bool=false,
        firsttreat::Bool=false, oneoff::Bool=false,
        only_pooled::Bool=false, only_event::Bool=false,
        conf_level::Float64=0.95,
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")
    # ... validation, load panel, call estimate_lp_did with new API ...
end
```

Add `--base-period` option to `did_estimate` leaf. In `_did_estimate` handler, pass `base_period=Symbol(base_period)` to `estimate_did`.

**Step 3: Add tests, run, commit**
```bash
git add src/commands/did.jl test/mocks.jl test/test_commands.jl
git commit -m "feat: replace did lp-did with LPDiDResult API, add --base-period to estimate"
```

---

## Task 9: Data List/Load — mpdta + ddcg

**Files:**
- Modify: `src/commands/data.jl` — add mpdta/ddcg to _data_list, update _data_load
- Modify: `test/mocks.jl` — add :mpdta and :ddcg to load_example mock
- Modify: `test/test_commands.jl` — add test for new datasets

**Step 1: Update `_data_list` handler**

Add two entries to the `datasets` array in `_data_list`:
```julia
        ("mpdta",    "Panel",       "500 × 5 × 3",  "Callaway-Sant'Anna (2021) minimum wage panel"),
        ("ddcg",     "Panel",       "184 × 51",      "Acemoglu et al. democracy-GDP panel"),
```

**Step 2: Update mock `load_example` to handle :mpdta and :ddcg**

In `test/mocks.jl`, update the `load_example` function to accept `:mpdta` and `:ddcg` and return PanelData mocks.

**Step 3: Add test, run, commit**
```bash
git add src/commands/data.jl test/mocks.jl test/test_commands.jl
git commit -m "feat: add mpdta and ddcg datasets to data list/load"
```

---

## Task 10: Update Test Structure Counts

**Files:**
- Modify: `test/runtests.jl` — update version refs to v0.3.3, update structure test counts

**Step 1: Update version refs**

Search for `v"0.3.2"` and `"0.3.2"` in runtests.jl and update to `v"0.3.3"` / `"0.3.3"`.

**Step 2: Update structure counts**

Update leaf/node counts in structure tests to reflect new commands:
- estimate: 20 → 24 leaves
- test: 22 leaves + 4 pvar + 2 var → 29 leaves + 4 pvar + 2 var
- predict: 13 → 16 leaves
- residuals: 13 → 16 leaves
- dsge: 8 leaves → 7 leaves + 1 node (bayes with 7 sub-leaves)
- Total subcommands: ~141 → ~164

**Step 3: Run full test suite**

Run: `julia --project test/runtests.jl`
Expected: All tests pass

**Step 4: Commit**
```bash
git add test/runtests.jl
git commit -m "test: update version refs to v0.3.3, adjust structure counts for new commands"
```

---

## Task 11: Documentation Update

**Files:**
- Modify: `CLAUDE.md` — version, command hierarchy, command details, API reference, testing
- Modify: `README.md` — commands table, version, examples
- Modify: `docs/make.jl`, `docs/src/index.md`, `docs/src/commands/overview.md`, `docs/src/architecture.md`, `docs/src/api.md`
- Create: `docs/src/commands/regression.md`, `docs/src/commands/advanced-unit-root.md`

Update all documentation per the MANDATORY rule in CLAUDE.md. Key changes:
- Version: 0.3.2 → 0.3.3, MEMs 0.3.3 → 0.3.4
- Command hierarchy: add reg/iv/logit/probit under estimate, 7 new tests, reg/logit/probit under predict/residuals, dsge bayes node, lp-did enhancements
- Subcommand count: ~141 → ~164
- Line counts: update for all modified source files
- API reference: add regression types/functions, unit root test types, LPDiDResult, BayesianDSGESimulation, posterior functions

**Commit:**
```bash
git add CLAUDE.md README.md docs/
git commit -m "docs: full v0.3.3 documentation update — regression, advanced unit root, LP-DiD, Bayesian DSGE"
```

---

## Task 12: Final Verification

**Step 1: Run full test suite**
```bash
julia --project test/runtests.jl
```

**Step 2: Verify all 23 new handler tests pass**
```bash
julia --project test/test_commands.jl
```

**Step 3: Smoke test key commands**
```bash
julia --project bin/friedman estimate reg --help
julia --project bin/friedman test fourier-adf --help
julia --project bin/friedman predict logit --help
julia --project bin/friedman dsge bayes estimate --help
julia --project bin/friedman did lp-did --help
julia --project bin/friedman data list
```

**Step 4: Verify version**
```bash
julia --project bin/friedman --version
```
Expected: `friedman version 0.3.3`
