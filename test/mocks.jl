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

# Mock MacroEconometricModels module for testing command handlers
# Provides minimal types and functions that src/commands/ files reference.

module MacroEconometricModels

using LinearAlgebra: I, diagm, Diagonal, dot, pinv, norm, diag
using Statistics: mean, var, std, quantile
using Random
import Serialization
import DataFrames
using DataFrames: DataFrame   # for long_table (Tables.jl tidy exports, real MEMs #346)

# ─── Distributions re-export (real MEMs re-exports Distributions) ──────────
# Minimal stand-in so the CLI's prior bridge (_dsge_priors_distributions →
# MacroEconometricModels.Distributions.Beta/Normal/...) resolves under the mock.
# Stores constructor args and supports Statistics.mean (used to seed theta0).
module Distributions
    import Statistics
    abstract type VariateForm end
    abstract type Univariate <: VariateForm end
    abstract type ValueSupport end
    abstract type Continuous <: ValueSupport end
    abstract type Distribution{F<:VariateForm,S<:ValueSupport} end
    const ContinuousUnivariateDistribution = Distribution{Univariate,Continuous}
    struct Beta{T<:Real} <: ContinuousUnivariateDistribution; α::T; β::T; end
    struct Normal{T<:Real} <: ContinuousUnivariateDistribution; μ::T; σ::T; end
    struct InverseGamma{T<:Real} <: ContinuousUnivariateDistribution; α::T; θ::T; end
    struct Gamma{T<:Real} <: ContinuousUnivariateDistribution; α::T; θ::T; end
    struct Uniform{T<:Real} <: ContinuousUnivariateDistribution; a::T; b::T; end
    Beta(a, b) = Beta{Float64}(Float64(a), Float64(b))
    Normal(a, b) = Normal{Float64}(Float64(a), Float64(b))
    InverseGamma(a, b) = InverseGamma{Float64}(Float64(a), Float64(b))
    Gamma(a, b) = Gamma{Float64}(Float64(a), Float64(b))
    Uniform(a, b) = Uniform{Float64}(Float64(a), Float64(b))
    Statistics.mean(d::Beta) = d.α / (d.α + d.β)
    Statistics.mean(d::Normal) = d.μ
    Statistics.mean(d::InverseGamma) = d.α > 1 ? d.θ / (d.α - 1) : d.θ
    Statistics.mean(d::Gamma) = d.α * d.θ
    Statistics.mean(d::Uniform) = (d.a + d.b) / 2
    export Distribution, Beta, Normal, InverseGamma, Gamma, Uniform
end
using .Distributions: Distribution, Beta, Normal, InverseGamma, Gamma, Uniform

# ─── Typed exception hierarchy (MEMs 0.7.0 / #245; mocked for C050) ─────────
# Real MEMs roots all domain errors at `MacroModelError <: Exception`. The CLI's
# `_domain_error_class` (src/output/errors.jl) matches these by type NAME, so the
# mock only needs same-named throwable types with a `.msg` field.
abstract type MacroModelError <: Exception end
struct ConvergenceError <: MacroModelError
    msg::String; iters::Int; residual::Float64
end
ConvergenceError(msg::AbstractString) = ConvergenceError(String(msg), 0, NaN)
struct IdentificationError <: MacroModelError
    msg::String
end
struct SingularSystemError <: MacroModelError
    msg::String; cond::Float64
end
SingularSystemError(msg::AbstractString) = SingularSystemError(String(msg), Inf)
struct SerializationError <: MacroModelError
    msg::String
end
Base.showerror(io::IO, e::MacroModelError) = print(io, nameof(typeof(e)), ": ", e.msg)

# W4/#139 (#81): the two direct-Exception domain types OUTSIDE the MacroModelError
# hierarchy on real MEMs 0.8.0 (dsge/bayes_types.jl, dsge/types.jl). Fields match
# real exactly (msg::String only); showerror mirrors real's format.
struct StochasticSingularityError <: Exception
    msg::String
end
Base.showerror(io::IO, e::StochasticSingularityError) =
    print(io, "StochasticSingularityError: ", e.msg)
struct DSGESolveError <: Exception
    msg::String
end
Base.showerror(io::IO, e::DSGESolveError) = print(io, "DSGESolveError: ", e.msg)

# ─── Repro manifest + versioned save/load (MEMs 0.7.0 #345/#347; CLI C052) ────
# JLD2 is not a test dep, so the mock persists a version-tagged container via
# Serialization, mirroring real save_model/load_model version + type semantics.
const SERIALIZATION_FORMAT_VERSION = 1

struct ReproManifest
    seed::Union{Int,Nothing}
    n_threads::Int
    julia_version::String
    package_version::String
    dependency_versions::Dict{String,String}
    os::String
    machine::String
    timestamp::String
    git_sha::String
    git_dirty::Bool
    settings::Dict{String,Any}
end

# reproduce() report types (W3/#167 — exact field mirror of real ReproFieldDiff /
# ReproReport, so the mock-surface struct gate holds).
struct ReproFieldDiff
    name::String
    matched::Bool
    max_abs_diff::Float64
end

struct ReproReport
    matched::Union{Bool,Missing}
    fields::Vector{ReproFieldDiff}
    seed::Union{Int,Nothing}
    threads_captured::Int
    threads_current::Int
    note::String
end

capture_manifest(; seed::Union{Integer,Nothing}=nothing,
                   settings::AbstractDict=Dict{String,Any}()) =
    ReproManifest(seed === nothing ? nothing : Int(seed), Threads.nthreads(),
                  string(VERSION), "0.7.2",
                  Dict("Distributions" => "0.25", "StatsAPI" => "1.7"),
                  string(Sys.KERNEL), string(Sys.MACHINE),
                  "2026-01-01T00:00:00Z", "unknown", false,
                  Dict{String,Any}(settings))

function save_model(model, path::AbstractString)
    tname = string(nameof(typeof(model)))
    # `_SERIALIZABLE_TYPES` mirrors the real registry and is defined at the BOTTOM of this
    # file — every type in it must already exist, and mocks.jl is one flat top-to-bottom
    # module. A function body resolves at call time, so referring to it here is fine.
    haskey(_SERIALIZABLE_TYPES, tname) || throw(SerializationError(
        "save_model does not support $(typeof(model))"))
    open(p -> Serialization.serialize(p,
        Dict{String,Any}("format_version" => SERIALIZATION_FORMAT_VERSION,
                         "type" => tname, "payload" => model)), path, "w")
    return path
end

function load_model(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    c = open(Serialization.deserialize, path)
    (c isa AbstractDict && haskey(c, "format_version")) ||
        throw(SerializationError("file '$path' is not a model container"))
    c["format_version"] == SERIALIZATION_FORMAT_VERSION || throw(SerializationError(
        "unsupported serialization format_version $(c["format_version"]): " *
        "this build reads $SERIALIZATION_FORMAT_VERSION"))
    return c["payload"]
end

# Header inspection without reconstruction (W3/#167 — mirrors real model_info keys).
function model_info(path::AbstractString)
    isfile(path) || throw(SerializationError("no such model file: $path"))
    c = open(Serialization.deserialize, path)
    (c isa AbstractDict && haskey(c, "format_version")) ||
        throw(SerializationError("file '$path' is not a model container"))
    return Dict{String,Any}(
        "format_version" => c["format_version"],
        "package_version" => "0.7.2",
        "julia_version" => string(VERSION),
        "created" => "2026-01-01T00:00:00Z",
        "note" => "",
        "manifest" => nothing,
        "bundle" => false,
        "type" => c["type"],
    )
end

# ─── New abstract model supertypes (MEMs 0.7.0 modules; wrapped in C062–C073) ─
abstract type AbstractMGARCHModel end
abstract type AbstractNonlinearTSModel end
abstract type AbstractStateSpaceModel end

# ─── Core Types ───────────────────────────────────────────

struct VARModel{T<:Real}
    Y::Matrix{T}; p::Int; B::Matrix{T}; U::Matrix{T}; Sigma::Matrix{T}
    aic::T; bic::T; hqic::T
    varnames::Vector{String}
end

# Real MEMs VARModel carries varnames with this default (var/types.jl:31-35)
VARModel(Y::Matrix{T}, p::Int, B::Matrix{T}, U::Matrix{T}, Sigma::Matrix{T},
         aic::T, bic::T, hqic::T) where {T<:Real} =
    VARModel(Y, p, B, U, Sigma, aic, bic, hqic, ["y$i" for i in 1:size(Y, 2)])

(::Type{VARModel{T}})(Y::Matrix{T}, p::Int, B::Matrix{T}, U::Matrix{T}, Sigma::Matrix{T},
                      aic::T, bic::T, hqic::T) where {T<:Real} =
    VARModel{T}(Y, p, B, U, Sigma, aic, bic, hqic, ["y$i" for i in 1:size(Y, 2)])

struct MockChains end

struct BVARPosterior{T}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    n_draws::Int
    p::Int
    n::Int
    data::Matrix{T}
    # Real carries varnames and threads them into posterior_mean/median_model (#119).
    varnames::Vector{String}
    # Real records a ReproManifest when estimate_bvar was called with seed= (#769).
    manifest::Union{Nothing,ReproManifest}
end
BVARPosterior(B::Array{T,3}, S::Array{T,3}, nd::Int, p::Int, n::Int, data::Matrix{T}) where {T} =
    BVARPosterior(B, S, nd, p, n, data, ["y$i" for i in 1:n], nothing)
BVARPosterior(B::Array{T,3}, S::Array{T,3}, nd::Int, p::Int, n::Int, data::Matrix{T},
              varnames::Vector{String}) where {T} =
    BVARPosterior(B, S, nd, p, n, data, varnames, nothing)

struct MinnesotaHyperparameters
    # `omega` is a SCALAR weight on the residual-covariance prior in real MEMs. It used to
    # be Vector{Float64} here, which let `_build_prior` pass a length-n vector of AR
    # residual s.d.s — real raises `TypeError: in keyword argument omega, expected Real`,
    # i.e. exit 1 on every `--config` minnesota run. Keep the keyword `::Real` so the mock
    # rejects exactly what real rejects.
    tau::Float64; decay::Float64; lambda::Float64; mu::Float64; omega::Float64
end
MinnesotaHyperparameters(; tau::Real=3.0, decay::Real=0.5, lambda::Real=5.0,
                           mu::Real=2.0, omega::Real=2.0) =
    MinnesotaHyperparameters(tau, decay, lambda, mu, omega)

struct ImpulseResponse{T}
    values::Array{T,3}; ci_lower::Union{Array{T,3},Nothing}; ci_upper::Union{Array{T,3},Nothing}
    horizon::Int; variables::Vector{String}; shocks::Vector{String}; ci_type::Symbol
    # Real trailing field (var/types.jl): bootstrap draws in package layout
    # reps × horizon × variable × shock; nothing on deterministic paths. The CF
    # adapters (policy_causal_effects / baseline_path, W4/#126) read it directly.
    _draws::Union{Nothing,Array{T,4}}
end
# 7-arg form mirrors real's backward-compatible constructor (draws default nothing)
ImpulseResponse(v::Array{T,3}, cl, cu, h::Int, vars, shocks, ci::Symbol) where T =
    ImpulseResponse{T}(v, cl, cu, h, vars, shocks, ci, nothing)
# Convenience 3-arg constructor for backward compat with existing handler tests
ImpulseResponse(v::Array{T,3}, cl, cu) where T = ImpulseResponse(v, cl, cu, size(v,1),
    ["var$i" for i in 1:size(v,2)], ["shock$i" for i in 1:size(v,3)], :cholesky)

struct BayesianImpulseResponse{T}
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
    _draws::Array{T,4}
    n_requested::Int
    n_effective::Int
    n_failed::Int
end
# Convenience 3-arg constructor for backward compat
BayesianImpulseResponse(m::Array{T,3}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianImpulseResponse(q, m, size(m,1),
    ["var$i" for i in 1:size(m,2)], ["shock$i" for i in 1:size(m,3)], ql,
    zeros(T, size(q)...), 0, size(q, 4), 0)

struct FEVD{T}
    decomposition::Array{T,3}; proportions::Array{T,3}
    variables::Vector{String}; shocks::Vector{String}
end
# 2-arg backward-compat constructor
FEVD(d::Array{T,3}, p::Array{T,3}) where T =
    FEVD(d, p, ["var$i" for i in 1:size(p,1)], ["shock$i" for i in 1:size(p,2)])
struct BayesianFEVD{T}
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    horizon::Int
    variables::Vector{String}
    shocks::Vector{String}
    quantile_levels::Vector{T}
    n_requested::Int
    n_effective::Int
    n_failed::Int
end
# Convenience 3-arg constructor for backward compat.
# Real point_estimate is (variable, shock, horizon) since MEMs 0.7.3 (#527).
BayesianFEVD(m::Array{T,3}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianFEVD(q, m, size(m,3),
    ["var$i" for i in 1:size(m,1)], ["shock$i" for i in 1:size(m,2)], ql,
    0, size(q, 4), 0)

struct HistoricalDecomposition{T}
    contributions::Array{T,3}; initial_conditions::Matrix{T}; actual::Matrix{T}
    shocks::Matrix{T}; T_eff::Int; variables::Vector{String}; shock_names::Vector{String}
    method::Symbol
end
# Convenience 5-arg constructor for backward compat
HistoricalDecomposition(c::Array{T,3}, ic, a, s, te::Int) where T =
    HistoricalDecomposition(c, ic, a, s, te,
    ["var$i" for i in 1:size(c,2)], ["shock$i" for i in 1:size(c,3)], :cholesky)

struct BayesianHistoricalDecomposition{T}
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    initial_quantiles::Array{T,3}
    initial_point_estimate::Matrix{T}
    shocks_point_estimate::Matrix{T}
    actual::Matrix{T}
    T_eff::Int
    variables::Vector{String}
    shock_names::Vector{String}
    quantile_levels::Vector{T}
    method::Symbol
    n_requested::Int
    n_effective::Int
    n_failed::Int
end
# Convenience 4-arg constructor for backward compat
BayesianHistoricalDecomposition(m::Array{T,3}, im::Matrix{T}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianHistoricalDecomposition(q, m, zeros(T, 0, 0, 0), im, zeros(T, size(m,1), size(m,3)), zeros(T, size(m,1), size(m,2)),
    size(m,1), ["var$i" for i in 1:size(m,2)], ["shock$i" for i in 1:size(m,3)], ql, :cholesky,
    0, size(q, 4), 0)

struct ZeroRestriction
    variable::Int; shock::Int; horizon::Int
end
struct SignRestriction
    variable::Int; shock::Int; sign::Symbol; horizon::Int
end
struct SVARRestrictions
    # Untyped vectors: real holds Vector{AbstractSVARRestriction} mixing zero/sign/
    # long-run/A0/bound/cumulative/narrative restrictions; the mock must too (W2).
    n_vars::Int; zeros::Vector; signs::Vector
end
SVARRestrictions(n::Int; zeros=[], signs=[]) =
    SVARRestrictions(n, collect(Any, zeros), collect(Any, signs))
struct AriasSVARResult{T}
    Q_draws::Vector{Matrix{T}}; irf_draws::Array{T,4}; weights::Vector{T}; acceptance_rate::T
    restrictions::SVARRestrictions
    # MEMs#372: the importance weights became operative in 0.7.2, so ESS is now a real
    # diagnostic rather than a formality. Kish's ESS and its fraction of n_draws.
    ess::T
    ess_fraction::T
end
# Back-compatible arity, matching real: derive the diagnostics from the weights.
# Both the parametric and the inferred-parameter forms, since existing call sites use each.
AriasSVARResult(Q_draws::Vector{Matrix{T}}, irf_draws::Array{T,4}, weights::Vector{T},
                acceptance_rate::T, restrictions::SVARRestrictions) where {T} =
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions)
function AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate,
                            restrictions) where {T}
    sw = sum(weights); sw2 = sum(w -> w^2, weights)
    ess = sw2 > 0 ? T(sw^2 / sw2) : zero(T)
    n = length(weights)
    AriasSVARResult{T}(Q_draws, irf_draws, weights, acceptance_rate, restrictions,
                       ess, n > 0 ? ess / T(n) : zero(T))
end

struct UhligSVARResult{T}
    Q::Matrix{T}; irf::Array{T,3}; penalty::T; shock_penalties::Vector{T}
    restrictions::SVARRestrictions; converged::Bool
end

# ─── LP Types ─────────────────────────────────────────────

struct LPModel{T}
    Y::Matrix{T}; shock_var::Int; horizon::Int; lags::Int
    B::Matrix{T}; residuals::Matrix{T}; vcov::Matrix{T}; T_eff::Vector{Int}
end
# W10/#112: `first_stage_F` and `T_eff` are per-horizon VECTORS in real MEMs (the first
# stage is re-estimated at every h and the sample shrinks with h). The mock stored scalars,
# which let the CLI read a `wi.F_stat` that real never had — exit 1 in production, green
# suite. Mock array SHAPE must match real, not just the field names (#84, `fevd bvar`).
struct LPIVModel{T}
    Y::Matrix{T}; instruments::Matrix{T}; first_stage_F::Vector{T}; horizon::Int
    T_eff::Vector{Int}; shock_var::Int; response_vars::Vector{Int}; varnames::Vector{String}
end
struct SmoothLPModel{T}
    Y::Matrix{T}; lambda::T; horizon::Int
end
struct StateLPModel{T}
    Y::Matrix{T}; B_expansion::Matrix{T}; B_recession::Matrix{T}; horizon::Int
end
struct PropensityLPModel{T}
    # Real ate/ate_se are (H+1)×n_response MATRICES (#118 shape gate).
    Y::Matrix{T}; ate::Matrix{T}; ate_se::Matrix{T}; horizon::Int
end
struct LPImpulseResponse{T}
    values::Matrix{T}; ci_lower::Matrix{T}; ci_upper::Matrix{T}; se::Matrix{T}
end
struct StructuralLP{T}
    irf::ImpulseResponse{T}; var_model::VARModel{T}; Q::Matrix{T}
    method::Symbol; se::Array{T,3}; lp_models::Vector{LPModel{T}}
end
struct LPFEVD{T}
    proportions::Array{T,3}; bias_corrected::Array{T,3}; se::Array{T,3}
    ci_lower::Array{T,3}; ci_upper::Array{T,3}
    method::Symbol; horizon::Int; n_boot::Int; conf_level::T; bias_correction::Bool
end
# Convenience constructor for backward compat (old tests use R2/lp_a/lp_b fields)
LPFEVD(R2::Array{T,3}, lp_a, lp_b, bc, bse, h::Int, v::Int, s::Int) where T =
    LPFEVD(R2, bc, bse, R2, R2, :R2, h, 200, T(0.95), true)

struct LPForecast{T}
    forecast::Matrix{T}; ci_lower::Matrix{T}; ci_upper::Matrix{T}
    se::Matrix{T}; horizon::Int; response_vars::Vector{Int}; shock_var::Int
    shock_path::Vector{T}; conf_level::T; ci_method::Symbol
end
# Convenience 5-arg constructor for backward compat
LPForecast(f::Matrix{T}, cl, cu, se, h::Int) where T =
    LPForecast(f, cl, cu, se, h, collect(1:size(f,2)), 1, T[1.0], T(0.95), :analytical)

# ─── Factor Types ─────────────────────────────────────────

struct FactorModel{T}
    X::Matrix{T}; factors::Matrix{T}; loadings::Matrix{T}; eigenvalues::Vector{T}
    explained_variance::Vector{T}; cumulative_variance::Vector{T}; r::Int; standardized::Bool
end
# Convenience 4-arg constructor for backward compat
FactorModel(d::Matrix{T}, f, l, e) where T =
    FactorModel(d, f, l, e, fill(T(0.5), length(e)), cumsum(fill(T(0.5), length(e))),
    size(f, 2), true)

struct DynamicFactorModel{T}
    X::Matrix{T}; factors::Matrix{T}; loadings::Matrix{T}; A::Vector{Matrix{T}}
    factor_residuals::Matrix{T}; Sigma_eta::Matrix{T}; Sigma_e::Matrix{T}
    eigenvalues::Vector{T}; explained_variance::Vector{T}; cumulative_variance::Vector{T}
    r::Int; p::Int; method::Symbol; standardized::Bool; converged::Bool
    iterations::Int; loglik::T
end
# Convenience 3-arg constructor for backward compat
DynamicFactorModel(f::Matrix{T}, l, c) where T =
    DynamicFactorModel(zeros(T, 100, size(l,1)), f, l, Matrix{T}[c],
    zeros(T, 99, size(f,2)), Matrix{T}(I(size(f,2))), Matrix{T}(I(size(l,1))),
    ones(T, size(f,2)), fill(T(0.5), size(f,2)), cumsum(fill(T(0.5), size(f,2))),
    size(f, 2), 1, :twostep, true, true, 100, T(-250.0))
struct GeneralizedDynamicFactorModel{T}
    X::Matrix{T}
    factors::Matrix{T}
    common_component::Matrix{T}
    idiosyncratic::Matrix{T}
    loadings_spectral::Array{T,3}
    spectral_density_X::Array{T,3}
    spectral_density_chi::Array{T,3}
    eigenvalues_spectral::Matrix{T}
    frequencies::Vector{T}
    q::Int
    r::Int
    bandwidth::Int
    kernel::Symbol
    standardized::Bool
    variance_explained::Vector{T}
end
struct FactorForecast{T}
    factors::Matrix{T}; observables::Matrix{T}
    factors_lower::Matrix{T}; factors_upper::Matrix{T}
    observables_lower::Matrix{T}; observables_upper::Matrix{T}
    factors_se::Matrix{T}; observables_se::Matrix{T}
    horizon::Int; conf_level::T; ci_method::Symbol
end
# Convenience constructor for backward compat (some old tests may use 7-arg form)
FactorForecast(f::Matrix{T}, o, ol, ou, ose, h::Int, cl::T) where T =
    FactorForecast(f, o, f, f, isnothing(ol) ? o : ol, isnothing(ou) ? o : ou,
    f, isnothing(ose) ? o : ose, h, cl, :analytical)

# ─── ARIMA Types ──────────────────────────────────────────

struct ARModel{T}
    y::Vector{T}
    p::Int
    c::T
    phi::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end
struct MAModel{T}
    y::Vector{T}
    q::Int
    c::T
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end
struct ARMAModel{T}
    y::Vector{T}
    p::Int
    q::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end
struct ARIMAModel{T}
    y::Vector{T}
    y_diff::Vector{T}
    p::Int
    d::Int
    q::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end
struct ARIMAForecast{T}
    forecast::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}; se::Vector{T}
    horizon::Int; conf_level::T
end

# W6/#108 — multiplicative seasonal ARIMA. Field names/order mirror real
# `SARIMAModel <: AbstractARIMAModel` (mock ⊆ real). `phi_expanded`/`theta_expanded` are the
# multiplied-out polynomials real carries; kept so the subset check stays honest.
struct SARIMAModel{T}
    y::Vector{T}
    y_diff::Vector{T}
    p::Int
    d::Int
    q::Int
    P::Int
    D::Int
    Q::Int
    s::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    Phi::Vector{T}
    Theta::Vector{T}
    phi_expanded::Vector{T}
    theta_expanded::Vector{T}
    sigma2::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end
# Convenience 5-arg constructor for backward compat
ARIMAForecast(f::Vector{T}, cl, cu, se, h::Int) where T =
    ARIMAForecast(f, cl, cu, se, h, T(0.95))

# ARFIMA (fractional integration / long memory) — fields mirror real MEMs
struct ARFIMAModel{T}
    y::Vector{T}
    p::Int
    d::T
    q::Int
    c::T
    phi::Vector{T}
    theta::Vector{T}
    sigma2::T
    d_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    method::Symbol
    converged::Bool
    iterations::Int
end

# ─── VAR Forecast Type ──────────────────────────────────

struct VARForecast{T<:AbstractFloat}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
    conf_level::T
    varnames::Vector{String}
    # Real trailing field (0.8.0): forecast draws, layout n_draws × h × n.
    _draws::Union{Nothing,Array{T,3}}
end
VARForecast(f::Matrix{T}, cl, cu, h::Int, cm::Symbol, conf::T, vn) where T =
    VARForecast{T}(f, cl, cu, h, cm, conf, vn, nothing)

# ─── Non-Gaussian Types ──────────────────────────────────

struct ICASVARResult{T}
    B0::Matrix{T}; W::Matrix{T}; Q::Matrix{T}; shocks::Matrix{T}
    method::Symbol; converged::Bool; iterations::Int; objective::T
end
struct NonGaussianMLResult{T}
    B0::Matrix{T}; Q::Matrix{T}; shocks::Matrix{T}
    distribution::Symbol; loglik::T; loglik_gaussian::T
    dist_params::Dict{Symbol,Any}; vcov::Matrix{T}; se::Vector{T}; aic::T; bic::T
end
struct MarkovSwitchingSVARResult{T}
    B0::Matrix{T}
end
struct GARCHSVARResult{T}
    B0::Matrix{T}
end
struct SmoothTransitionSVARResult{T}
    B0::Matrix{T}
end
struct ExternalVolatilitySVARResult{T}
    B0::Matrix{T}
end

# ─── SVAR Identification Types (0.9.2) ────────────────────
# Field names mirror real MEMs (mock ⊆ real gate); sign-carrying fields stay
# Symbols on the mock (CLI-facing convention) — the gate compares names, and the
# CLI builds restrictions through the builder functions below, never positionally.

struct LongRunZeroRestriction
    variable::Int; shock::Int
end
struct A0ZeroRestriction
    variable::Int; shock::Int
end
struct A0SignRestriction
    variable::Int; shock::Int; sign::Symbol
end
struct ElasticityBound
    numerator_var::Int; denominator_var::Int; shock::Int; horizon::Int
    lower::Float64; upper::Float64
end
struct MagnitudeBound
    variable::Int; shock::Int; horizon::Int; lower::Float64; upper::Float64
end
struct CumulativeRestriction
    variable::Int; shock::Int; horizons::UnitRange{Int}; sign::Symbol
end
struct NarrativeShockRestriction
    shock::Int; dates::Vector{Int}; sign::Symbol
end
struct NarrativeContributionRestriction
    variable::Int; shock::Int; window::UnitRange{Int}; kind::Symbol
end
struct IdentificationStatus
    status::Symbol; ranks::Vector{Int}; orders::Vector{Int}; n_overidentifying::Int
end
struct SVARPattern{T}
    A::Matrix{T}; B::Matrix{T}; long_run::Union{Nothing,Matrix{T}}
    function SVARPattern(A::AbstractMatrix, B::AbstractMatrix; long_run=nothing)
        size(A, 1) == size(A, 2) || throw(ArgumentError("A must be square"))
        size(A) == size(B) || throw(ArgumentError("A and B must have the same size"))
        T = promote_type(float(eltype(A)), float(eltype(B)))
        lr = if long_run === nothing
            nothing
        else
            size(long_run) == size(A) ||
                throw(ArgumentError("long_run must be n×n, same as A and B"))
            Matrix{T}(long_run)
        end
        new{T}(Matrix{T}(A), Matrix{T}(B), lr)
    end
end
struct ProxySVARResult{T}
    Q::Matrix{T}; B0::Matrix{T}; k::Int; first_stage_F::T; reliability::T
    instruments_names::Vector{String}; varnames::Vector{String}
    shock_names::Vector{String}; is_partial::Bool
end
struct MaxShareResult{T}
    Q::Vector{T}; q::Vector{T}; target::Int; horizons::Vector{Int}; band::Symbol
    share::T; eigvals::Vector{T}; varnames::Vector{String}
    shock_names::Vector{String}; is_partial::Bool
end
struct NonGaussianGMMResult{T}
    B0::Matrix{T}; Q::Matrix{T}; theta::Vector{T}; vcov::Matrix{T}; se::Vector{T}
    J::T; J_pvalue::T; moments::Symbol; weighting::Symbol
    shocks::Matrix{T}; varnames::Vector{String}; shock_names::Vector{String}
end
struct SVARModel{T}
    A::Matrix{T}; B::Matrix{T}; Q::Matrix{T}
    vcov::Union{Nothing,Matrix{T}}; se::Union{Nothing,Matrix{T}}
    loglik::T; lr_stat::T; lr_df::Int; lr_pvalue::T
    pattern::SVARPattern{T}; identification::IdentificationStatus
    varnames::Vector{String}
end
struct RobustBayesResult{T}
    lower::Array{T,3}; upper::Array{T,3}
    robust_lower::Array{T,3}; robust_upper::Array{T,3}
    single_prior_lower::Array{T,3}; single_prior_upper::Array{T,3}
    informativeness::T; empty_set_prob::T; level::T
end
struct NormalityTestResult{T}
    test_name::Symbol; statistic::T; pvalue::T; df::Int
end
struct NormalityTestSuite{T}
    results::Vector{NormalityTestResult{T}}
end

# ─── Test Types ───────────────────────────────────────────

struct ADFResult{T}
    statistic::T; pvalue::T; lags::Int
end
struct KPSSResult{T}
    statistic::T
    pvalue::T
end
struct PPResult{T}
    statistic::T; pvalue::T
end
struct ZAResult{T}
    statistic::T; break_index::Int
end
struct NgPerronResult{T}
    MZa::T; MZt::T; MSB::T; MPT::T
end
# Long-memory d estimators (fields mirror real MEMs)
struct GPHResult{T}
    d::T; se::T; tstat::T; pval::T; m::Int; n::Int; trim::Int
end
struct LocalWhittleResult{T}
    d::T; se::T; tstat::T; pval::T; m::Int; n::Int; objective::T
end
struct JohansenResult{T}
    trace_stats::Vector{T}; trace_pvalues::Vector{T}
    max_eigen_stats::Vector{T}; max_eigen_pvalues::Vector{T}
end

# ─── GMM Types ────────────────────────────────────────────

struct GMMModel{T}
    theta::Vector{T}; vcov::Matrix{T}; n_moments::Int; n_params::Int
    W::Matrix{T}; g_bar::Vector{T}; J_stat::T; J_pvalue::T
end

# ─── Volatility Types ────────────────────────────────────

struct ARCHModel{T<:Real}
    y::Vector{T}; q::Int; mu::T; omega::T; alpha::Vector{T}
    conditional_variance::Vector{T}; standardized_residuals::Vector{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; aic::T; bic::T; method::Symbol; converged::Bool; iterations::Int
end
# Convenience 1-arg constructor for backward compat
# Convenience: c has length q+2 (mu, omega, alpha_1...alpha_q)
ARCHModel(c::Vector{T}) where T = let q = length(c) - 2
    ARCHModel(zeros(T, 100), q, c[1], c[2], c[3:end],
        ones(T, 100), zeros(T, 100), zeros(T, 100), zeros(T, 100),
        T(-150.0), T(310.0), T(320.0), :mle, true, 50)
end

struct GARCHModel{T<:Real}
    y::Vector{T}; p::Int; q::Int; mu::T; omega::T; alpha::Vector{T}; beta::Vector{T}
    conditional_variance::Vector{T}; standardized_residuals::Vector{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; aic::T; bic::T; method::Symbol; converged::Bool; iterations::Int
end
# Convenience: c has length p+q+2 (mu, omega, alpha_1...alpha_q, beta_1...beta_p)
GARCHModel(c::Vector{T}) where T = let np = length(c) - 2; q = div(np, 2); p = np - q
    GARCHModel(zeros(T, 100), p, q, c[1], c[2], c[3:3+q-1], c[3+q:end],
        ones(T, 100), zeros(T, 100), zeros(T, 100), zeros(T, 100),
        T(-150.0), T(310.0), T(320.0), :mle, true, 50)
end

struct EGARCHModel{T<:Real}
    y::Vector{T}; p::Int; q::Int; mu::T; omega::T; alpha::Vector{T}; gamma::Vector{T}; beta::Vector{T}
    conditional_variance::Vector{T}; standardized_residuals::Vector{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; aic::T; bic::T; method::Symbol; converged::Bool; iterations::Int
end
# Convenience: c has length 2*q+p+2 (mu, omega, alpha_1...alpha_q, gamma_1...gamma_q, beta_1...beta_p)
EGARCHModel(c::Vector{T}) where T = let np = length(c) - 2; q = div(np, 3); p = np - 2*q
    EGARCHModel(zeros(T, 100), p, q, c[1], c[2], c[3:3+q-1], c[3+q:3+2*q-1], c[3+2*q:end],
        ones(T, 100), zeros(T, 100), zeros(T, 100), zeros(T, 100),
        T(-150.0), T(310.0), T(320.0), :mle, true, 50)
end

struct GJRGARCHModel{T<:Real}
    y::Vector{T}; p::Int; q::Int; mu::T; omega::T; alpha::Vector{T}; gamma::Vector{T}; beta::Vector{T}
    conditional_variance::Vector{T}; standardized_residuals::Vector{T}
    residuals::Vector{T}; fitted::Vector{T}
    loglik::T; aic::T; bic::T; method::Symbol; converged::Bool; iterations::Int
end
# Convenience: c has length 2*q+p+2 (mu, omega, alpha_1...alpha_q, gamma_1...gamma_q, beta_1...beta_p)
GJRGARCHModel(c::Vector{T}) where T = let np = length(c) - 2; q = div(np, 3); p = np - 2*q
    GJRGARCHModel(zeros(T, 100), p, q, c[1], c[2], c[3:3+q-1], c[3+q:3+2*q-1], c[3+2*q:end],
        ones(T, 100), zeros(T, 100), zeros(T, 100), zeros(T, 100),
        T(-150.0), T(310.0), T(320.0), :mle, true, 50)
end

struct SVModel{T<:Real}
    y::Vector{T}; h_draws::Matrix{T}
    mu_post::Vector{T}; phi_post::Vector{T}; sigma_eta_post::Vector{T}
    volatility_mean::Vector{T}; volatility_quantiles::Matrix{T}; quantile_levels::Vector{T}
    dist::Symbol; leverage::Bool; n_samples::Int
end
# Convenience 1-arg constructor: c is ignored, just for backward compat
SVModel(c::Vector{T}) where T = SVModel(zeros(T, 100), zeros(T, 10, 100),
    zeros(T, 10), zeros(T, 10), zeros(T, 10), ones(T, 100),
    ones(T, 100, 3), T[0.16, 0.5, 0.84], :normal, false, 10)

struct VolatilityForecast{T<:Real}
    forecast::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}; se::Vector{T}
    horizon::Int; conf_level::T; model_type::Symbol
end
# Convenience 2-arg constructor for backward compat
VolatilityForecast(f::Vector{T}, h::Int) where T =
    VolatilityForecast(f, f, f, abs.(f) .* T(0.1), h, T(0.95), :garch)

# ─── Multivariate GARCH Types (C064b; MEMs 0.7.0 src/mgarch) ──
# Fields mirror the real MGARCHModel exactly (TS-3 field-subset conformance).
struct MGARCHModel{T<:Real} <: AbstractMGARCHModel
    Y::Matrix{T}
    mu::Vector{T}
    margins::Vector{GARCHModel{T}}
    H::Array{T,3}
    R::Union{Matrix{T},Array{T,3}}
    Rbar::Matrix{T}
    params::Vector{T}
    param_names::Vector{String}
    param_vcov::Matrix{T}
    loglik::T
    aic::T
    bic::T
    kind::Symbol
    correction::Symbol
    bekk_kind::Symbol
    converged::Bool
    n::Int
end

# ─── VECM Types ──────────────────────────────────────────

struct VECMModel{T<:Real}
    Y::Matrix{T}; p::Int; rank::Int
    alpha::Matrix{T}; beta::Matrix{T}; Pi::Matrix{T}
    Gamma::Vector{Matrix{T}}; mu::Vector{T}
    U::Matrix{T}; Sigma::Matrix{T}
    aic::T; bic::T; hqic::T; loglik::T
    deterministic::Symbol; method::Symbol
    johansen_result::Union{Nothing,JohansenResult{T}}  # align with real MEMs (C020)
    varnames::Vector{String}
end

# Real MEMs VECMModel carries johansen_result + varnames
VECMModel(Y::Matrix{T}, p::Int, rank::Int, alpha::Matrix{T}, beta::Matrix{T},
          Pi::Matrix{T}, Gamma::Vector{Matrix{T}}, mu::Vector{T}, U::Matrix{T},
          Sigma::Matrix{T}, aic::T, bic::T, hqic::T, loglik::T,
          deterministic::Symbol, method::Symbol) where {T<:Real} =
    VECMModel(Y, p, rank, alpha, beta, Pi, Gamma, mu, U, Sigma,
              aic, bic, hqic, loglik, deterministic, method, nothing,
              ["y$i" for i in 1:size(Y, 2)])

# After VECMModel: mocks.jl is one flat top-to-bottom module, so a signature
# type must already exist at include time.
struct SVECResult{T}
    B0::Matrix{T}; Q::Matrix{T}; Xi::Matrix{T}; n_permanent::Int
    vecm::VECMModel{T}; identification::IdentificationStatus
end

VECMModel(Y::Matrix{T}, p::Int, rank::Int, alpha::Matrix{T}, beta::Matrix{T},
          Pi::Matrix{T}, Gamma::Vector{Matrix{T}}, mu::Vector{T}, U::Matrix{T},
          Sigma::Matrix{T}, aic::T, bic::T, hqic::T, loglik::T,
          deterministic::Symbol, method::Symbol, varnames::Vector{String}) where {T<:Real} =
    VECMModel(Y, p, rank, alpha, beta, Pi, Gamma, mu, U, Sigma,
              aic, bic, hqic, loglik, deterministic, method, nothing, varnames)

# VECM restriction test result (C071) — fields mirror real MEMs VECMRestrictionTest.
struct VECMRestrictionTest{T<:Real}
    kind::Symbol
    lr_stat::T
    df::Int
    pvalue::T
    rank::Int
    description::String
    beta_restricted::Matrix{T}
    beta_unrestricted::Matrix{T}
    eigenvalues_restricted::Vector{T}
    eigenvalues_unrestricted::Vector{T}
    converged::Bool
    restricted_model::VECMModel{T}
end

struct VECMForecast{T<:Real}
    levels::Matrix{T}; differences::Matrix{T}
    ci_lower::Union{Matrix{T},Nothing}; ci_upper::Union{Matrix{T},Nothing}
    horizon::Int; ci_method::Symbol
end

struct VECMGrangerResult{T<:Real}
    short_run_stat::T; short_run_pvalue::T; short_run_df::Int
    long_run_stat::T; long_run_pvalue::T; long_run_df::Int
    strong_stat::T; strong_pvalue::T; strong_df::Int
    cause_var::Int; effect_var::Int
end

# ─── Mock Helper ──────────────────────────────────────────

function _mock_var(Y::Matrix{Float64}, p::Int; varnames=nothing)
    T_obs, n = size(Y)
    k = n * p + 1
    B = zeros(k, n)
    for i in 1:min(n, k)
        B[i, i] = 0.5
    end
    U = zeros(T_obs - p, n) .+ 0.01
    Sigma = Matrix{Float64}(I(n)) * 0.01
    VARModel(Y, p, B, U, Sigma, -100.0, -95.0, -97.0,
             varnames === nothing ? ["y$i" for i in 1:n] : varnames)
end

# ─── Mock Functions ───────────────────────────────────────

select_lag_order(Y, max_p; criterion=:aic) = min(2, max(1, max_p))
# varnames kwarg mirrors real estimate_var/estimate_bvar (#119) — real threads it
# into the model / posterior and every derived rendering.
estimate_var(Y, p; check_stability=true, varnames=nothing) = _mock_var(Y, p; varnames=varnames)

estimate_bvar(Y, p; sampler=:direct, n_draws=1000, prior=:normal, hyper=nothing,
              hyperopt::Symbol=:glp, varnames=nothing, seed=nothing) =
    BVARPosterior(zeros(10, size(Y,2)*p+1, size(Y,2)), zeros(10, size(Y,2), size(Y,2)),
                  10, p, size(Y,2), Y,
                  varnames === nothing ? ["y$i" for i in 1:size(Y,2)] : varnames,
                  seed === nothing ? nothing : capture_manifest(; seed=seed))

# reproduce() for the manifest-carrying mock posterior (W3/#167 — mirrors real
# reproduce(::BVARPosterior): decline without a recorded seed, else re-run from
# the manifest seed and compare draws; the mock estimator is deterministic so a
# re-run matches by construction, same simplification class as its ones()*0.1).
# Lives HERE (not beside save_model): the typed signature resolves at include
# time, so BVARPosterior must already be defined (forward-reference lesson).
function reproduce(post::BVARPosterior)
    m = post.manifest
    if m === nothing || m.seed === nothing
        return ReproReport(missing, ReproFieldDiff[], nothing, 1, 1,
                           "no recorded seed: estimate with seed=N")
    end
    fresh = estimate_bvar(post.data, post.p; n_draws=post.n_draws,
                          varnames=post.varnames, seed=m.seed)
    diffs = [ReproFieldDiff("B_draws", isequal(post.B_draws, fresh.B_draws), 0.0),
             ReproFieldDiff("Sigma_draws", isequal(post.Sigma_draws, fresh.Sigma_draws), 0.0)]
    matched = all(d.matched for d in diffs)
    return ReproReport(matched, diffs, m.seed, 1, Threads.nthreads(),
                       matched ? "" : "re-run differs")
end

# Universal fallback (W3/#167 — mirrors real `reproduce(x)` at MEMs 0.9.3):
# upstream answers EVERY other type with a missing-verdict ReproReport instead
# of throwing MethodError, so `model reproduce` reports "unverifiable", never
# model/unsupported, on real MEMs. Placed AFTER the typed methods above so
# they win dispatch; the handler's MethodError catch stays as defense-in-depth
# against a future upstream fallback removal.
function reproduce(x)
    return ReproReport(missing, ReproFieldDiff[], nothing, 0, Threads.nthreads(),
        "reproduce() is not implemented for $(typeof(x)); supported: BVARPosterior, " *
        "BayesianDSGE, KrusellSmithSolution, bootstrap ImpulseResponse (reproduce(ir, model)), " *
        "and randomized estimators that record a ReproManifest.")
end
posterior_mean_model(post::BVARPosterior; data=nothing) =
    _mock_var(post.data, post.p; varnames=post.varnames)
posterior_median_model(post::BVARPosterior; data=nothing) =
    _mock_var(post.data, post.p; varnames=post.varnames)
# Keep old (chain, p, n) signatures for backward compat
posterior_mean_model(chain::MockChains, p, n; data=nothing) =
    _mock_var(isnothing(data) ? ones(100, n) : data, p)
posterior_median_model(chain::MockChains, p, n; data=nothing) =
    _mock_var(isnothing(data) ? ones(100, n) : data, p)
optimize_hyperparameters(Y, p) = MinnesotaHyperparameters(tau=0.2, decay=1.0, lambda=0.5, omega=2.0)

# GLP joint optimization (real returns the hyperparameters PLUS diagnostics that
# `estimate_bvar` discards, which is why the CLI selects here and passes `hyper=`).
struct GLPHyperparameters
    hyper::MinnesotaHyperparameters
    log_ml::Float64
    log_posterior::Float64
    converged::Bool
    at_bound::Bool
    iterations::Int
    log_ml_default::Float64
end
optimize_hyperparameters_glp(Y, p; decay::Real=0.5, omega::Real=2.0, starts::Int=4,
                             max_iter::Int=500, f_reltol::Real=1e-8,
                             verbose::Bool=true) = GLPHyperparameters(
    MinnesotaHyperparameters(tau=0.35, decay=0.6, lambda=4.2, mu=1.8, omega=2.0),
    -123.45, -130.2, true, false, 17, -140.7)

# StatsAPI-like functions
coef(m::VARModel) = m.B
function coef(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel})
    m isa ARModel && return getfield(m, :phi)
    m isa MAModel && return getfield(m, :theta)
    return vcat(getfield(m, :phi), getfield(m, :theta))
end
loglikelihood(m::VARModel) = -500.0
loglikelihood(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.loglik
stderror(m::GMMModel) = fill(0.1, length(m.theta))
stderror(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = fill(0.01, length(coef(m)))
stderror(m::ARCHModel) = fill(0.01, 2 + m.q)  # mu, omega, alpha...
stderror(m::GARCHModel) = fill(0.01, 2 + m.q + m.p)  # mu, omega, alpha..., beta...
stderror(m::EGARCHModel) = fill(0.01, 2 + m.q + m.q + m.p)  # mu, omega, alpha..., gamma..., beta...
stderror(m::GJRGARCHModel) = fill(0.01, 2 + m.q + m.q + m.p)  # mu, omega, alpha..., gamma..., beta...

# predict: return fitted values
predict(m::VARModel) = m.Y[m.p+1:end, :]
predict(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = zeros(Float64, 50)

# residuals: return residuals
residuals(m::VARModel) = m.U
residuals(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = fill(0.01, 50)

# Factor model predict/residuals
predict(m::FactorModel) = m.factors * m.loadings'  # T × n common component
predict(m::DynamicFactorModel) = m.factors * m.loadings'  # T × n common component
predict(m::GeneralizedDynamicFactorModel) = m.common_component  # T × n
residuals(m::FactorModel) = m.X .- m.factors * m.loadings'  # T × n idiosyncratic
residuals(m::DynamicFactorModel) = ones(size(m.factors, 1), size(m.loadings, 1)) * 0.01
residuals(m::GeneralizedDynamicFactorModel) = ones(size(m.common_component)) * 0.01

# Volatility model predict/residuals
predict(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel}) = m.fitted
predict(m::SVModel) = m.volatility_mean
residuals(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel}) = m.residuals
residuals(m::SVModel) = m.y .- m.volatility_mean

report(::VARModel) = nothing
report(::ImpulseResponse) = nothing
report(::BayesianImpulseResponse) = nothing
report(::FEVD) = nothing
report(::BayesianFEVD) = nothing
report(::HistoricalDecomposition) = nothing
report(::BayesianHistoricalDecomposition) = nothing
report(::UhligSVARResult) = nothing
report(::RobustBayesResult) = nothing
report(::SVARModel) = nothing
report(::SVECResult) = nothing

# Global flag to control mock behavior for testing edge cases
const _MOCK_FLAGS = Dict{Symbol,Any}(
    :var_stationary => true,
    :pvar_stable => true,
    :verify_decomposition => true,
    :normality_all_pass => false,
    :lp_iv_weak => false,
    # W10/#112: drives `anderson_rubin_ci`'s degenerate set shapes so T1/T2 can exercise
    # the empty / unbounded / disjoint / whole-line rendering branches, which is where a
    # `[lo, hi]` assumption in a renderer would break.
    :ar_set_shape => :bounded,
)

function is_stationary(m::VARModel)
    if _MOCK_FLAGS[:var_stationary]
        (is_stationary=true, eigenvalues=[0.5+0.1im, 0.5-0.1im, 0.3+0.0im])
    else
        (is_stationary=false, eigenvalues=[1.2+0.0im, 0.5-0.1im, 0.3+0.0im])
    end
end
is_stationary(m::DynamicFactorModel) = (is_stationary=true,)

function companion_matrix(B::AbstractMatrix, n::Int, p::Int)
    np = n * p
    np == 0 && return zeros(1, 1)
    C = zeros(np, np)
    for j in 1:n, i in 1:min(size(B, 1), np)
        C[j, i] = B[i, j] * 0.3
    end
    np > n && (C[n+1:np, 1:np-n] = Matrix{Float64}(I(np - n)))
    C
end
companion_matrix_factors(m::DynamicFactorModel) = length(m.A) > 0 ? m.A[1] : zeros(m.r, m.r)
nvars(m::VARModel) = size(m.Y, 2)
nvars(m::VECMModel) = size(m.Y, 2)

# IRF
function irf(model::VARModel, horizon::Int; method=:cholesky, check_func=nothing,
             narrative_check=nothing, ci_type=:none, reps=200, conf_level=0.95,
             stationary_only=false, seed=nothing,
             bootstrap::Symbol=:iid, block_length::Int=0, wild_dist::Symbol=:rademacher,
             bias_correct::Bool=false, bias_reps::Int=0,
             instruments=nothing, target=nothing, restrictions=nothing,
             pattern=nothing, weighting=nothing, hetero=nothing,
             maxiter=nothing, gibbs_burn=nothing, gibbs_draws=nothing,
             init=nothing)
    bootstrap in (:iid, :wild, :block) || throw(ArgumentError(
        "bootstrap must be :iid, :wild, or :block; got :$bootstrap"))
    wild_dist in (:rademacher, :mammen) || throw(ArgumentError(
        "wild_dist must be :rademacher or :mammen; got :$wild_dist"))
    n = size(model.Y, 2)
    vals = ones(horizon + 1, n, n) * 0.1
    ci_lo = ci_type == :none ? nothing : vals .- 0.5
    ci_hi = ci_type == :none ? nothing : vals .+ 0.5
    # Real stores bootstrap draws in package layout reps × horizon × var × shock;
    # the CF adapters (W4/#126) slice them, so the mock must populate them too.
    drw = ci_type == :bootstrap ?
        reshape(vals, 1, horizon + 1, n, n) .+ 0.05 .* randn(reps, horizon + 1, n, n) :
        nothing
    ImpulseResponse{Float64}(vals, ci_lo, ci_hi, horizon, copy(model.varnames),
                             copy(model.varnames), :cholesky, drw)
end
function irf(chain::MockChains, p::Int, n::Int, horizon::Int;
             method=:cholesky, data=nothing, quantiles=[0.16, 0.5, 0.84],
             check_func=nothing, narrative_check=nothing)
    vals = ones(horizon + 1, n, n) * 0.1
    q_vals = ones(horizon + 1, n, n, length(quantiles)) * 0.1
    BayesianImpulseResponse(vals, q_vals, Float64.(quantiles))
end
function irf(post::BVARPosterior, horizon::Int;
             method=:cholesky, quantiles=[0.16, 0.5, 0.84],
             check_func=nothing, narrative_check=nothing,
             weighting=nothing, hetero=nothing, maxiter=nothing,
             gibbs_burn=nothing, gibbs_draws=nothing, init=nothing)
    n = post.n
    vals = ones(horizon + 1, n, n) * 0.1
    q_vals = ones(horizon + 1, n, n, length(quantiles)) * 0.1
    BayesianImpulseResponse(vals, q_vals, Float64.(quantiles))
end

# Cumulative IRF
cumulative_irf(r::ImpulseResponse) = r
cumulative_irf(r::BayesianImpulseResponse) = r

# Sign-Identified Set
struct SignIdentifiedSet{T<:AbstractFloat}
    Q_draws::Vector{Matrix{T}}
    irf_draws::Array{T,4}
    n_accepted::Int
    n_total::Int
    acceptance_rate::T
    variables::Vector{String}
    shocks::Vector{String}
end

irf_bounds(s::SignIdentifiedSet; quantiles=[0.16, 0.84]) = (zeros(size(s.irf_draws)[2:4]...), ones(size(s.irf_draws)[2:4]...))
irf_median(s::SignIdentifiedSet) = fill(0.5, size(s.irf_draws)[2:4]...)

function identify_sign(model::VARModel, horizon::Int, check_func; max_draws=1000, store_all=false,
                       seed=nothing)
    n = size(model.Y, 2)
    if store_all
        n_d = 10
        # Real compute_irf returns (horizon, n, n), impact-first (row 1 = horizon 0).
        irf_draws = ones(n_d, horizon, n, n) * 0.1
        Q_draws = [Matrix{Float64}(I(n)) for _ in 1:n_d]
        return SignIdentifiedSet(Q_draws, irf_draws, n_d, max_draws, Float64(n_d/max_draws),
            ["var$i" for i in 1:n], ["shock$i" for i in 1:n])
    end
    Q = Matrix{Float64}(I(n))
    irf_vals = ones(horizon, n, n) * 0.1
    return (Q, irf_vals)
end

# FEVD
function fevd(model::VARModel, horizon::Int; method=:cholesky, check_func=nothing, narrative_check=nothing,
              instruments=nothing, target=nothing, restrictions=nothing, pattern=nothing,
              weighting=nothing, hetero=nothing, maxiter=nothing, gibbs_burn=nothing,
              gibbs_draws=nothing, init=nothing)
    n = size(model.Y, 2)
    props = ones(n, n, horizon) / n
    # Real carries model.varnames into the result (same gap as irf above).
    FEVD(props, props, copy(model.varnames), copy(model.varnames))
end
function fevd(chain::MockChains, p::Int, n::Int, horizon::Int;
              data=nothing, quantiles=[0.16, 0.5, 0.84])
    # Real BayesianFEVD.point_estimate is (variable, shock, horizon) since MEMs 0.7.3
    # (#527 unified the axes) — match it, or the mock hides a transposed render (the
    # layout half of #84's fevd bvar bug, whose permutedims fix is now REMOVED).
    props = ones(n, n, horizon) / n
    q = ones(n, n, horizon, length(quantiles)) / n
    BayesianFEVD(props, q, Float64.(quantiles))
end
function fevd(post::BVARPosterior, horizon::Int;
              method=:cholesky, quantiles=[0.16, 0.5, 0.84],
              weighting=nothing, hetero=nothing, maxiter=nothing,
              gibbs_burn=nothing, gibbs_draws=nothing, init=nothing)
    n = post.n
    props = ones(n, n, horizon) / n
    q = ones(n, n, horizon, length(quantiles)) / n
    BayesianFEVD(props, q, Float64.(quantiles))
end

# Historical Decomposition
function historical_decomposition(model::VARModel, horizon::Int; method=:cholesky,
                                   check_func=nothing, narrative_check=nothing,
                                   instruments=nothing, target=nothing,
                                   restrictions=nothing, pattern=nothing,
                                   weighting=nothing, hetero=nothing,
                                   maxiter=nothing, gibbs_burn=nothing,
                                   gibbs_draws=nothing, init=nothing)
    n = size(model.Y, 2)
    T_eff = min(horizon, size(model.Y, 1) - model.p)
    contribs = ones(T_eff, n, n) * 0.1
    actual = ones(T_eff, n)
    initial = ones(T_eff, n) * 0.01
    shocks_mat = ones(T_eff, n)
    HistoricalDecomposition(contribs, initial, actual, shocks_mat, T_eff)
end
# Real (vecm/analysis.jl): method=:svec/:long_run go through identify_svec on the
# VECM itself (KPSW default); everything else converts to the levels VAR first.
# The mock mirrors the routing; the canned shapes come from the VAR path.
function irf(vecm::VECMModel, horizon::Int; method=:cholesky, kwargs...)
    (method === :svec || method === :long_run) ?
        irf(to_var(vecm), horizon; method=:cholesky) :
        irf(to_var(vecm), horizon; method=method, kwargs...)
end
function fevd(vecm::VECMModel, horizon::Int; method=:cholesky, kwargs...)
    (method === :svec || method === :long_run) ?
        fevd(to_var(vecm), horizon; method=:cholesky) :
        fevd(to_var(vecm), horizon; method=method, kwargs...)
end
function historical_decomposition(vecm::VECMModel, horizon::Int; method=:cholesky, kwargs...)
    (method === :svec || method === :long_run) ?
        historical_decomposition(to_var(vecm), horizon; method=:cholesky) :
        historical_decomposition(to_var(vecm), horizon; method=method, kwargs...)
end
function historical_decomposition(chain::MockChains, p::Int, n::Int, horizon::Int;
                                   data=nothing, method=:cholesky, quantiles=[0.16, 0.5, 0.84])
    T_eff = isnothing(data) ? horizon : size(data, 1) - p
    mean_c = ones(T_eff, n, n) * 0.1
    initial_m = ones(T_eff, n) * 0.01
    q = ones(T_eff, n, n, length(quantiles)) * 0.1
    BayesianHistoricalDecomposition(mean_c, initial_m, q, Float64.(quantiles))
end
function historical_decomposition(post::BVARPosterior, horizon::Int;
                                   method=:cholesky, quantiles=[0.16, 0.5, 0.84],
                                   seed=nothing, weighting=nothing, hetero=nothing,
                                   maxiter=nothing, gibbs_burn=nothing,
                                   gibbs_draws=nothing, init=nothing)
    n = post.n; p = post.p; data = post.data
    T_eff = size(data, 1) - p
    mean_c = ones(T_eff, n, n) * 0.1
    initial_m = ones(T_eff, n) * 0.01
    q = ones(T_eff, n, n, length(quantiles)) * 0.1
    BayesianHistoricalDecomposition(mean_c, initial_m, q, Float64.(quantiles))
end
function historical_decomposition(slp::StructuralLP, T_hd::Int)
    n = size(slp.var_model.Y, 2)
    T_eff = min(T_hd, size(slp.var_model.Y, 1) - slp.var_model.p)
    contribs = ones(T_eff, n, n) * 0.1
    actual = ones(T_eff, n)
    initial = ones(T_eff, n) * 0.01
    shocks_mat = ones(T_eff, n)
    HistoricalDecomposition(contribs, initial, actual, shocks_mat, T_eff)
end
verify_decomposition(hd::HistoricalDecomposition; tol=1e-6) = _MOCK_FLAGS[:verify_decomposition]
contribution(hd::HistoricalDecomposition, var::Int, shock::Int) = hd.contributions[:, var, shock]

# SVAR restrictions (builders with validation live in the 0.9.2 block below —
# zero/sign_restriction there cover :long_run and horizon ranges too)
function identify_arias(model::VARModel, restrictions::SVARRestrictions, horizon::Int;
                        n_draws=1000, n_rotations=1000, seed=nothing)
    n = size(model.Y, 2)
    n_d = 10
    irf_draws = ones(n_d, horizon + 1, n, n) * 0.1
    AriasSVARResult([Matrix{Float64}(I(n)) for _ in 1:n_d], irf_draws, ones(n_d), 0.5, restrictions)
end
using Statistics: mean as _mean
function irf_mean(result::AriasSVARResult)
    dropdims(_mean(result.irf_draws; dims=1); dims=1)
end

function identify_uhlig(model::VARModel, restrictions::SVARRestrictions, horizon::Int;
                        n_starts=50, n_refine=10, max_iter_coarse=500, max_iter_fine=2000,
                        tol_coarse=1e-4, tol_fine=1e-8, seed=nothing)
    n = size(model.Y, 2)
    Q = Matrix{Float64}(I(n))
    irf_vals = ones(horizon + 1, n, n) * 0.1
    UhligSVARResult(Q, irf_vals, 1e-6, fill(1e-7, n), restrictions, true)
end
function identify_narrative(model::VARModel, restrictions::SVARRestrictions, horizon::Int;
                            kwargs...)
    # Real (core/arias.jl) is a thin wrapper around identify_arias for ADRR
    # narrative restrictions — mirror it exactly.
    identify_arias(model, restrictions, horizon; kwargs...)
end

# ─── SVAR restriction builders (0.9.2) ────────────────────
# Mirror real validation (ArgumentError) so degenerate TOML fails the same way
# on mocks and real MEMs; sign convention stays Symbol (CLI-facing).
_mock_parse_sign(s::Symbol) =
    s === :positive ? s : s === :negative ? s :
        throw(ArgumentError("sign must be :positive or :negative"))
_mock_check_range(hs, what) =
    (all(h -> h isa Integer && h >= 1, hs) ||
        throw(ArgumentError("$what horizons must be ≥ 1"))) &&
    UnitRange{Int}(minimum(hs), maximum(hs))

function zero_restriction(variable, shock; horizon=0)
    horizon === :long_run && return LongRunZeroRestriction(variable, shock)
    horizon isa Integer && horizon >= 0 ||
        throw(ArgumentError("restriction horizon must be ≥ 0 or :long_run"))
    ZeroRestriction(variable, shock, Int(horizon))
end
function sign_restriction(variable, shock, sign::Symbol; horizon=0, horizons=nothing)
    s = _mock_parse_sign(sign)
    if horizons !== nothing
        r = _mock_check_range(horizons, "sign restriction")
        return [SignRestriction(variable, shock, s, h) for h in r]
    end
    horizon isa Integer && horizon >= 0 ||
        throw(ArgumentError("restriction horizon must be ≥ 0"))
    SignRestriction(variable, shock, s, Int(horizon))
end
a0_zero_restriction(equation, shock) = A0ZeroRestriction(equation, shock)
a0_sign_restriction(equation, shock, sign::Symbol) =
    A0SignRestriction(equation, shock, _mock_parse_sign(sign))
function elasticity_bound(numerator_var, denominator_var, shock;
                          horizon=0, lower=-Inf, upper=Inf)
    lower <= upper || throw(ArgumentError("elasticity lower bound exceeds upper"))
    ElasticityBound(numerator_var, denominator_var, shock, Int(horizon),
                    Float64(lower), Float64(upper))
end
function magnitude_bound(variable, shock; horizon=0, lower, upper)
    lower <= upper || throw(ArgumentError("magnitude lower bound exceeds upper"))
    MagnitudeBound(variable, shock, Int(horizon), Float64(lower), Float64(upper))
end
function cumulative_restriction(variable, shock, sign::Symbol; horizons)
    r = _mock_check_range(horizons, "cumulative restriction")
    CumulativeRestriction(variable, shock, r, _mock_parse_sign(sign))
end
function narrative_shock_restriction(shock, dates, sign::Symbol)
    ds = collect(Int, dates)
    (all(d -> d >= 1, ds) && !isempty(ds)) ||
        throw(ArgumentError("narrative dates must be nonempty and ≥ 1"))
    NarrativeShockRestriction(shock, ds, _mock_parse_sign(sign))
end
function narrative_contribution_restriction(variable, shock, window; kind=:most_important)
    kind in (:most_important, :overwhelming, :least_important) || throw(ArgumentError(
        "kind must be :most_important (Type A), :overwhelming (Type B), or :least_important"))
    w = _mock_check_range(window, "narrative contribution")
    NarrativeContributionRestriction(variable, shock, w, kind)
end

# ─── SVAR identification estimators (0.9.2, canned) ───────
function identify_proxy(model::VARModel, z::AbstractVector; normalize=1, normalize_value=1.0)
    (length(z) == size(model.Y, 1) || length(z) == size(model.Y, 1) - model.p) ||
        throw(ArgumentError("instrument length must match T or T - p"))
    n = size(model.Y, 2)
    ProxySVARResult(Matrix{Float64}(I(n)), Matrix{Float64}(I(n)), 1, 25.0, 0.9,
                    ["z"], model.varnames, ["shock$i" for i in 1:n], false)
end
identify_proxy(model::VARModel, Z::AbstractMatrix; normalize=1, normalize_value=1.0) =
    identify_proxy(model, vec(Z); normalize=normalize, normalize_value=normalize_value)
function identify_max_share(model::VARModel; target=nothing, horizons=nothing,
                            band=:auto, kwargs...)
    target === nothing && throw(ArgumentError("identify_max_share requires `target`"))
    n = size(model.Y, 2)
    (target isa Integer ? 1 <= target <= n : target in model.varnames) ||
        throw(ArgumentError("target must be a variable index or name"))
    hs = horizons === nothing ? [4, 8] : collect(Int, horizons)
    MaxShareResult(zeros(n), zeros(n), target isa Integer ? target : 1, hs, band,
                   0.75, ones(n), model.varnames, ["shock$i" for i in 1:n], false)
end
function identify_gmm_moments(model::VARModel; moments=:both, weighting=:two_step,
                              kwargs...)
    moments in (:independence, :cumulant, :both) ||
        throw(ArgumentError("moments must be :independence, :cumulant, or :both"))
    weighting in (:identity, :two_step, :cue) ||
        throw(ArgumentError("weighting must be :identity, :two_step, or :cue"))
    n = size(model.Y, 2)
    T = size(model.Y, 1)
    NonGaussianGMMResult(Matrix{Float64}(I(n)), Matrix{Float64}(I(n)), zeros(n),
                         Matrix{Float64}(I(n)), zeros(n), 1.5, 0.45, moments, weighting,
                         zeros(T, n), model.varnames, ["shock$i" for i in 1:n])
end
function identify_svec(vecm::VECMModel; long_run_zeros=nothing, short_run_zeros=nothing,
                       pattern=nothing, n_starts=5, max_iter=400, kwargs...)
    n = size(vecm.Y, 2)
    SVECResult(Matrix{Float64}(I(n)), Matrix{Float64}(I(n)), zeros(n, n), vecm.rank,
               vecm, IdentificationStatus(:exact, fill(n, 1), fill(n, 1), 0))
end
function estimate_svar(model::VARModel, pattern::SVARPattern; n_starts=5, max_iter=400,
                       long_run_matrix=nothing, rng=nothing, kwargs...)
    n = size(model.Y, 2)
    n_starts >= 1 || throw(ArgumentError("n_starts must be ≥ 1, got $n_starts"))
    (size(pattern.A) == (n, n) && size(pattern.B) == (n, n)) ||
        throw(ArgumentError("Pattern dimension ($(size(pattern.A, 1))) must match model ($n)"))
    SVARModel(Matrix{Float64}(I(n)), Matrix{Float64}(I(n)), Matrix{Float64}(I(n)),
              nothing, nothing, -100.0, 0.0, 0, 1.0, pattern,
              IdentificationStatus(:exact, fill(n, 1), fill(n, 1), 0), model.varnames)
end
function recursive_pattern(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    A = Matrix{Float64}(I, n, n)
    for i in 2:n, j in 1:i-1
        A[i, j] = NaN
    end
    B = zeros(n, n)
    for i in 1:n
        B[i, i] = NaN
    end
    SVARPattern(A, B)
end
function a_model_pattern(A::AbstractMatrix)
    n = size(A, 1)
    T = float(eltype(A))
    SVARPattern(A, Matrix{T}(I, n, n))
end
function b_model_pattern(B::AbstractMatrix)
    n = size(B, 1)
    T = float(eltype(B))
    SVARPattern(Matrix{T}(I, n, n), B)
end
ab_model_pattern(A::AbstractMatrix, B::AbstractMatrix; long_run=nothing) =
    SVARPattern(A, B; long_run=long_run)
function blanchard_quah_pattern(n::Integer)
    n >= 1 || throw(ArgumentError("n must be positive"))
    B = fill(NaN, n, n)
    lr = fill(NaN, n, n)
    for i in 1:n, j in i+1:n
        lr[i, j] = 0.0
    end
    SVARPattern(Matrix{Float64}(I, n, n), B; long_run=lr)
end
function identify_robust_bayes(post::BVARPosterior, restrictions::SVARRestrictions,
                               horizon::Int; level=0.68, solver=:optimize,
                               n_draws=200, n_rotations=100, rng=nothing, data=nothing)
    (0 < level < 1) || throw(ArgumentError("level must lie in (0, 1)"))
    horizon >= 1 || throw(ArgumentError("horizon must be ≥ 1"))
    solver === :draws || solver === :optimize ||
        throw(ArgumentError("solver must be :draws or :optimize, got :$solver"))
    n = post.n
    # Real returns (horizon, n, n) arrays (horizon × variable × shock).
    lo = fill(0.05, horizon, n, n)
    hi = fill(0.15, horizon, n, n)
    RobustBayesResult(lo, hi, lo .- 0.05, hi .+ 0.05, lo .+ 0.01, hi .- 0.01,
                      0.2, 0.0, 0.68)
end
label_shocks(result; by=:restrictions, restrictions=nothing, variables=nothing,
             convention=:unit_effect) = (collect(1:size(result.B0, 2)), ones(Int, size(result.B0, 2)))
median_target(s::SignIdentifiedSet) =
    (Q=s.Q_draws[1], irf=s.irf_draws[1, :, :, :], index=1)
modal_model(s::SignIdentifiedSet; bandwidth=nothing, kwargs...) =
    (Q=s.Q_draws[1], irf=s.irf_draws[1, :, :, :], index=1)
joint_band(s::SignIdentifiedSet; level=0.68, kwargs...) =
    (s.irf_draws[1, :, :, :] .- 0.1, s.irf_draws[1, :, :, :] .+ 0.1)
sup_t_band(s::SignIdentifiedSet; level=0.68, kwargs...) =
    (s.irf_draws[1, :, :, :] .- 0.1, s.irf_draws[1, :, :, :] .+ 0.1)
test_lambda_distinct(result; pairs=:all) =
    (statistic=[8.0], pvalue=[0.02], pvalue_bonferroni=[0.02], pairs=[(1, 2)])
test_gaussian_shock_count(result; alpha=0.05) =
    (statistic=2.0, pvalue=0.35, details=Dict(:n_gaussian => 1))
test_label_stability(model::VARModel; method=:fastica, n_bootstrap=999, rng=nothing,
                     transition_var=nothing, regime_indicator=nothing) =
    (statistic=0.95, pvalue=NaN)
function check_identification(restrictions::SVARRestrictions, model::VARModel; n_points=10, rng=nothing)
    IdentificationStatus(:set, fill(1, 1), fill(1, 1), 0)
end
function check_identification(pattern::SVARPattern, n::Int)
    n >= 1 || throw(ArgumentError("n must be positive"))
    size(pattern.A, 1) == n || throw(ArgumentError(
        "Pattern dimension ($(size(pattern.A, 1))) must match n=$n"))
    IdentificationStatus(:exact, fill(1, 1), fill(1, 1), 0)
end

# Chain parameter extraction (BVAR forecast)
function extract_chain_parameters(chain::MockChains)
    n_draws = 10
    b_vecs = ones(n_draws, 9) * 0.1
    sigmas = ones(n_draws, 6) * 0.01
    (b_vecs, sigmas)
end
function extract_chain_parameters(post::BVARPosterior)
    nd = post.n_draws
    k = post.n * post.p + 1
    b_vecs = ones(nd, k * post.n) * 0.1
    sigmas = ones(nd, post.n * (post.n + 1) ÷ 2) * 0.01
    (b_vecs, sigmas)
end
parameters_to_model(b_vec, sigma_vec, p, n, data) = parameters_to_model(b_vec, sigma_vec, p, n; data=data)
function parameters_to_model(b_vec, sigma_vec, p, n; data=nothing)
    Y = isnothing(data) ? ones(100, n) : data
    k = n * p + 1
    B = zeros(k, n)
    nb = min(length(b_vec), k * n)
    for i in 1:nb
        row = ((i - 1) % k) + 1
        col = ((i - 1) ÷ k) + 1
        col <= n && (B[row, col] = b_vec[i])
    end
    U = zeros(size(Y, 1) - p, n) .+ 0.01
    Sigma = Matrix{Float64}(I(n)) * 0.01
    VARModel(Y, p, B, U, Sigma, -100.0, -95.0, -97.0)
end

# LP functions
function estimate_lp(Y, shock_var, horizon; lags=4, cov_type=:newey_west)
    T_obs, n = size(Y)
    LPModel(Y, shock_var, horizon, lags, ones(lags+1, n)*0.1, ones(T_obs-lags, n)*0.01, Matrix{Float64}(I(n)) * 0.01, [T_obs - lags - h for h in 0:horizon])
end
function lp_irf(model::LPModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, abs.(ones(h, n)) * 0.1)
end
function estimate_lp_iv(Y, shock_var, Z, horizon; lags=4, cov_type=:newey_west,
                       response_vars=collect(1:size(Y, 2)), bandwidth=0,
                       varnames=["y$i" for i in 1:size(Y, 2)])
    T_obs = size(Y, 1)
    f_val = _MOCK_FLAGS[:lp_iv_weak] ? 5.0 : 15.0
    # F and T_eff both decay with the horizon, as they do upstream.
    fs = [f_val - 0.1 * h for h in 0:horizon]
    te = [T_obs - lags - h for h in 0:horizon]
    LPIVModel(Y, Z, fs, horizon, te, shock_var, collect(response_vars), collect(varnames))
end
function lp_iv_irf(model::LPIVModel; conf_level=0.95)
    n = length(model.response_vars); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, ones(h, n) * 0.1)
end
# Real returns (F_stats, weak_horizons, min_F, passes_threshold, threshold) — there is NO
# `F_stat`/`is_weak`. The old mock invented both, so `estimate lp --method iv` passed T1/T2
# while exiting 1 on every real invocation (W10/#112). Mirror real's tuple exactly.
function weak_instrument_test(model::LPIVModel; threshold=10.0)
    F_stats = model.first_stage_F
    weak_horizons = findall(F_stats .< threshold)
    (F_stats=F_stats, weak_horizons=weak_horizons, min_F=minimum(F_stats),
     passes_threshold=isempty(weak_horizons), threshold=threshold)
end
weak_instrument_test(F_stats::Vector; threshold=10.0) =
    (F_stats=F_stats, weak_horizons=findall(F_stats .< threshold),
     min_F=minimum(F_stats), passes_threshold=all(F_stats .>= threshold), threshold=threshold)
function estimate_smooth_lp(Y, shock_var, horizon; n_knots=3, lambda=0.0, degree=3)
    SmoothLPModel(Y, lambda, horizon)
end
function smooth_lp_irf(model::SmoothLPModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, ones(h, n) * 0.1)
end
cross_validate_lambda(Y, shock, horizon; k_folds=5) = 0.5
function estimate_state_lp(Y, shock_var, state_var, horizon; gamma=1.5, lags=4)
    n = size(Y, 2)
    StateLPModel(Y, ones(5, n)*0.1, ones(5, n)*0.1, horizon)
end
function state_irf(model::StateLPModel; regime=:both, conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    exp_vals = ones(h, n) * 0.1; rec_vals = ones(h, n) * 0.2
    (expansion=LPImpulseResponse(exp_vals, exp_vals .- 0.5, exp_vals .+ 0.5, ones(h, n)*0.1),
     recession=LPImpulseResponse(rec_vals, rec_vals .- 0.5, rec_vals .+ 0.5, ones(h, n)*0.1))
end
test_regime_difference(model::StateLPModel; h=nothing) =
    (joint_test=(avg_t_stat=2.5, p_value=0.012),)
function estimate_propensity_lp(Y, treatment, covariates, horizon; ps_method=:logit, trimming=(0.01,0.99))
    PropensityLPModel(Y, fill(0.5, horizon + 1, size(Y, 2)), fill(0.1, horizon + 1, size(Y, 2)), horizon)
end
function propensity_irf(model::PropensityLPModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, ones(h, n) * 0.1)
end
propensity_diagnostics(model::PropensityLPModel) =
    (propensity_summary=(treated=(mean=0.7,), control=(mean=0.3,)), balance=(max_weighted=0.05,))
doubly_robust_lp(Y, treatment, covariates, horizon; ps_method=:logit) =
    PropensityLPModel(Y, fill(0.6, horizon + 1, size(Y, 2)), fill(0.12, horizon + 1, size(Y, 2)), horizon)

function structural_lp(Y, horizon; method=:cholesky, lags=4, var_lags=4,
                       cov_type=:newey_west, ci_type=:none, reps=200, conf_level=0.95,
                       check_func=nothing, narrative_check=nothing, max_draws=1000,
                       seed=nothing)
    T_obs, n = size(Y); p = var_lags
    model = _mock_var(Y, p)
    irf_vals = ones(horizon + 1, n, n) * 0.1
    ci_lo = ci_type == :none ? nothing : irf_vals .- 0.5
    ci_hi = ci_type == :none ? nothing : irf_vals .+ 0.5
    irf_res = ImpulseResponse(irf_vals, ci_lo, ci_hi)
    Q = Matrix{Float64}(I(n))
    lp_models = [LPModel(Y, i, horizon, lags, ones(5, n)*0.1, ones(T_obs-lags, n)*0.01, Matrix{Float64}(I(n))*0.01, [T_obs - lags - h for h in 0:horizon]) for i in 1:n]
    StructuralLP(irf_res, model, Q, method, ones(horizon+1, n, n)*0.1, lp_models)
end
function lp_fevd(slp::StructuralLP, horizons::Int; estimator=:R2, n_boot=200, conf_level=0.95)
    n = size(slp.var_model.Y, 2)
    props = ones(n, n, horizons) / n
    LPFEVD(props, props, props, props, ones(n, n, horizons)*0.01, horizons, n, n)
end
function forecast(model::LPModel, shock_path; ci_method=:analytical, conf_level=0.95, n_boot=500)
    n = size(model.Y, 2); h = length(shock_path)
    fc = ones(h, n) * 0.1
    LPForecast(fc, fc .- 0.5, fc .+ 0.5, ones(h, n) * 0.1, h)
end

# Factor functions
function estimate_factors(X, r; standardize=true,
                          varnames::Union{Nothing,Vector{String}}=nothing)
    T_obs, n = size(X)
    # Real (factor/static.jl, MEMs#538) validates the length before storing.
    vn = varnames === nothing ? ["Var $i" for i in 1:n] : varnames
    length(vn) == n || throw(ArgumentError(
        "varnames has $(length(vn)) entries but X has $n columns"))
    FactorModel(X, ones(T_obs, r)*0.1, ones(n, r)*0.3, Float64[r-i+1 for i in 1:r])
end
function ic_criteria(X, max_factors; standardize=true)
    r = min(2, max_factors)
    (ic1=ones(max_factors), ic2=ones(max_factors), ic3=ones(max_factors),
     r_IC1=r, r_IC2=r, r_IC3=r)
end
function scree_plot_data(model::FactorModel)
    r = size(model.factors, 2)
    ev = Float64[r - i + 1 for i in 1:r]
    cv = cumsum(ev) ./ sum(ev)
    (factors=1:r, explained_variance=ev, cumulative_variance=cv)
end
function estimate_dynamic_factors(X, r, p; method=:twostep, max_iter=100, tol=1e-6)
    T_obs, n = size(X)
    DynamicFactorModel(ones(T_obs, r)*0.1, ones(n, r)*0.3, diagm(ones(r))*0.5)
end
function ic_criteria_gdfm(X, max_q; standardize=true)
    (q_ratio=min(2, max_q), q_opt=min(2, max_q))
end
function estimate_gdfm(X, q; r=2, standardize=true, bandwidth=0, kernel=:bartlett,
                       spectral=:lag_window)
    spectral in (:lag_window, :smoothed_periodogram) ||
        throw(ArgumentError("spectral must be :lag_window or :smoothed_periodogram, got :$spectral"))
    T_obs, n = size(X)
    bw = bandwidth == 0 ? 5 : bandwidth
    fac = ones(T_obs, r) * 0.1
    cc = ones(T_obs, n) * 0.5
    idio = ones(T_obs, n) * 0.1
    load_s = ones(n, r, 10) * 0.3
    sx = ones(n, n, 10) * 0.1
    schi = ones(n, n, 10) * 0.05
    eigs = ones(r, 10)
    freqs = collect(range(0, stop=π, length=10))
    GeneralizedDynamicFactorModel(Float64.(X), fac, cc, idio, load_s, sx, schi, eigs, freqs,
        q, r, bw, kernel, standardize, fill(0.5, r))
end
function common_variance_share(model::GeneralizedDynamicFactorModel)
    n = size(model.common_component, 2)
    fill(0.5, n)
end
function forecast(model::FactorModel, h::Int; ci_method=:none, conf_level=0.95)
    n = size(model.X, 2)
    r = size(model.factors, 2)
    obs = ones(h, n) * 0.1
    fac = ones(h, r) * 0.1
    FactorForecast(fac, obs, fac, fac, obs .- 0.5, obs .+ 0.5,
        abs.(fac) .* 0.1, ones(h, n)*0.1, h, conf_level, :analytical)
end
function forecast(model::DynamicFactorModel, h::Int; ci=false, ci_method=:none, conf_level=0.95, n_boot=500, ci_level=0.95)
    r = size(model.factors, 2)
    n = size(model.loadings, 1)
    factors = ones(h, r) * 0.1
    obs = factors * model.loadings'
    FactorForecast(factors, obs, factors, factors, obs .- 0.5, obs .+ 0.5,
        abs.(factors) .* 0.1, ones(h, n)*0.1, h, conf_level, :analytical)
end

function forecast(model::GeneralizedDynamicFactorModel, h::Int; kwargs...)
    n = size(model.common_component, 2)
    r = 2
    factors = ones(h, r) * 0.1
    obs = ones(h, n) * 0.1
    FactorForecast(factors, obs, factors, factors, obs .- 0.5, obs .+ 0.5,
        abs.(factors) .* 0.1, ones(h, n)*0.1, h, 0.95, :analytical)
end

# Unit root / cointegration tests
adf_test(y; lags=:aic, regression=:constant) = ADFResult(-3.5, 0.01, 2)
kpss_test(y; regression=:constant) = KPSSResult(0.3, 0.01)
pp_test(y; regression=:constant) = PPResult(-3.2, 0.02)
za_test(y; regression=:both, trim=0.15) = ZAResult(-4.5, 50)
ngperron_test(y; regression=:constant) = NgPerronResult(-20.0, -3.1, 0.15, 4.0)
function johansen_test(Y, p; deterministic=:constant)
    n = size(Y, 2)
    JohansenResult([30.0,10.0,2.0][1:n], [0.01,0.1,0.5][1:n],
                   [25.0,8.0,1.5][1:n], [0.02,0.15,0.6][1:n])
end

# GMM functions
function estimate_lp_gmm(Y, shock_var, horizon; lags=4, weighting=:two_step)
    theta = ones(3) * 0.1
    vcov = Matrix{Float64}(I(3)) * 0.01
    [GMMModel(theta, vcov, 4, 3, Matrix{Float64}(I(4)), ones(4)*0.01, 2.5, 0.65)]
end
gmm_summary(model::GMMModel) = (n_moments=model.n_moments, n_params=model.n_params, theta=model.theta)
j_test(model::GMMModel) = (J_stat=model.J_stat, p_value=model.J_pvalue, df=model.n_moments - model.n_params)

# ARIMA functions
function _ar_like_y(y)
    v = Float64.(vec(y))
    T = length(v)
    return v, ones(T) * 0.01, ones(T) * 0.1
end
function estimate_ar(y, p; method=:ols)
    v, res, fit = _ar_like_y(y)
    ARModel(v, p, 0.0, ones(p)*0.3, 0.5, res, fit, -50.0, -100.0, -95.0, method, true, 10)
end
function estimate_ma(y, q; method=:css_mle)
    v, res, fit = _ar_like_y(y)
    MAModel(v, q, 0.0, ones(q)*0.3, 0.5, res, fit, -50.0, -100.0, -95.0, method, true, 10)
end
function estimate_arma(y, p, q; method=:css_mle)
    v, res, fit = _ar_like_y(y)
    ARMAModel(v, p, q, 0.0, ones(p)*0.3, ones(q)*0.3, 0.5, res, fit, -50.0, -100.0, -95.0, method, true, 10)
end
function estimate_arima(y, p, d, q; method=:css_mle)
    v, res, fit = _ar_like_y(y)
    ARIMAModel(v, v, p, d, q, 0.0, ones(max(p,1))*0.3, ones(max(q,1))*0.3, 0.5, res, fit, -50.0, -100.0, -95.0, method, true, 10)
end
function auto_arima(y; max_p=5, max_q=5, max_d=2, criterion=:bic, method=:mle)
    estimate_arima(y, 1, 1, 1; method=method)
end

# W6/#108 — guards mirror real estimate_sarima/auto_sarima so the CLI's exit classes are
# exercised at T1/T2 with the same classes real produces.
function estimate_sarima(y, p::Int, d::Int, q::Int, P::Int, D::Int, Q::Int, s::Int;
                         method::Symbol=:css_mle, include_intercept::Bool=true,
                         max_iter::Int=500)
    all(>=(0), (p, d, q, P, D, Q)) ||
        throw(ArgumentError("orders p,d,q,P,D,Q must be non-negative"))
    if P > 0 || D > 0 || Q > 0
        s >= 2 || throw(ArgumentError(
            "seasonal period s must be ≥ 2 when any seasonal order is positive, got $s"))
    end
    s >= 0 || throw(ArgumentError("seasonal period s must be non-negative, got $s"))
    v, res, fit = _ar_like_y(y)
    phi = p > 0 ? fill(0.3, p) : Float64[]
    theta = q > 0 ? fill(0.2, q) : Float64[]
    Phi = P > 0 ? fill(0.4, P) : Float64[]
    Theta = Q > 0 ? fill(0.1, Q) : Float64[]
    SARIMAModel(v, v, p, d, q, P, D, Q, s, include_intercept ? 0.05 : 0.0,
                phi, theta, Phi, Theta, vcat(phi, Phi), vcat(theta, Theta),
                0.5, res, fit, -50.0, -100.0, -95.0, method, true, 12)
end

function auto_sarima(y, s::Int; d=nothing, D=nothing, max_p::Int=2, max_q::Int=2,
                     max_P::Int=1, max_Q::Int=1, criterion::Symbol=:aic,
                     method::Symbol=:css_mle, include_intercept::Bool=true)
    criterion in (:aic, :bic) ||
        throw(ArgumentError("criterion must be :aic or :bic, got :$criterion"))
    s >= 1 || throw(ArgumentError("seasonal period s must be ≥ 1, got $s"))
    estimate_sarima(y, min(1, max_p), d === nothing ? 1 : d, min(1, max_q),
                    min(1, max_P), D === nothing ? 1 : D, min(1, max_Q), s;
                    method=method, include_intercept=include_intercept)
end

function forecast(m::SARIMAModel, h::Int; conf_level::Real=0.95)
    h >= 1 || throw(ArgumentError("horizon must be ≥ 1"))
    base = isempty(m.y) ? 0.0 : Float64(m.y[end])
    f = fill(base, h)
    se = [sqrt(Float64(m.sigma2)) * sqrt(Float64(k)) for k in 1:h]
    ARIMAForecast{Float64}(f, f .- 1.96 .* se, f .+ 1.96 .* se, se, h, Float64(conf_level))
end

residuals(m::SARIMAModel) = m.residuals
predict(m::SARIMAModel) = m.fitted
fitted(m::SARIMAModel) = m.fitted
aic(m::SARIMAModel) = m.aic
bic(m::SARIMAModel) = m.bic
loglikelihood(m::SARIMAModel) = m.loglik
nobs(m::SARIMAModel) = length(m.y)

export SARIMAModel, estimate_sarima, auto_sarima
function estimate_arfima(y, p, q; method=:css, d0=nothing, trunc=200, max_iter=500)
    v, res, fit = _ar_like_y(y)
    d = isnothing(d0) ? 0.3 : Float64(d0)
    phi = p > 0 ? ones(p) * 0.2 : Float64[]
    theta = q > 0 ? ones(q) * 0.1 : Float64[]
    ARFIMAModel(v, p, d, q, 0.0, phi, theta, 0.5, 0.05, res, fit,
                -50.0, -100.0, -95.0, method, true, 10)
end

# #73: forecast(::ARFIMAModel, h; conf_level, trunc_lag) -> ARIMAForecast, mirroring real
# (arima/forecast.jl:321). Real's interval fields are ci_lower/ci_upper, NOT lower/upper.
function forecast(m::ARFIMAModel, h::Int; conf_level::Real=0.95, trunc_lag::Int=200)
    h >= 1 || throw(ArgumentError("horizon must be ≥ 1"))
    trunc_lag >= 1 || throw(ArgumentError("trunc_lag must be ≥ 1"))
    base = isempty(m.y) ? 0.0 : Float64(m.y[end])
    f = fill(base, h)
    se = [sqrt(Float64(m.sigma2)) * sqrt(Float64(k)) for k in 1:h]   # widens with h
    ARIMAForecast{Float64}(f, f .- 1.96 .* se, f .+ 1.96 .* se, se, h, Float64(conf_level))
end
function gph_test(y; m=:default, trim=0)
    n = length(vec(y))
    n < 8 && throw(ArgumentError("Series too short for GPH (n=$n)."))
    mm = m === :default ? floor(Int, sqrt(n)) : Int(m)
    GPHResult(0.3, 0.08, 3.75, 0.0002, mm, n, trim)
end
function local_whittle(y; m=:default)
    n = length(vec(y))
    n < 8 && throw(ArgumentError("Series too short for local Whittle (n=$n)."))
    mm = m === :default ? floor(Int, sqrt(n)) : Int(m)
    LocalWhittleResult(0.3, 0.06, 5.0, 1.0e-6, mm, n, -0.5)
end

ar_order(m::ARModel) = m.p;       ar_order(m::MAModel) = 0
ar_order(m::ARMAModel) = m.p;     ar_order(m::ARIMAModel) = m.p
ma_order(m::ARModel) = 0;         ma_order(m::MAModel) = m.q
ma_order(m::ARMAModel) = m.q;     ma_order(m::ARIMAModel) = m.q
diff_order(m::ARModel) = 0;       diff_order(m::MAModel) = 0
diff_order(m::ARMAModel) = 0;     diff_order(m::ARIMAModel) = m.d
aic(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.aic
bic(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.bic
# ARFIMA accessors (coef ordering [c, d, phi.., theta..], matching real MEMs)
ar_order(m::ARFIMAModel) = m.p
ma_order(m::ARFIMAModel) = m.q
diff_order(m::ARFIMAModel) = m.d
aic(m::ARFIMAModel) = m.aic
bic(m::ARFIMAModel) = m.bic
loglikelihood(m::ARFIMAModel) = m.loglik
coef(m::ARFIMAModel) = vcat(m.c, m.d, m.phi, m.theta)
stderror(m::ARFIMAModel) = fill(0.05, 2 + length(m.phi) + length(m.theta))
function forecast(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}, h::Int; conf_level=0.95)
    fc = ones(h) * 0.1
    ARIMAForecast(fc, fc .- 0.5, fc .+ 0.5, ones(h) * 0.1, h)
end

# Volatility model functions
# W11/#113: garch/egarch/gjr-garch take a conditional distribution upstream; arch does NOT
# (no `dist` kwarg at all), so it deliberately keeps the 3-positional signature here — a mock
# that accepted `dist` on arch would hide a guaranteed real MethodError. The guard mirrors
# upstream `_vol_dist_check` so the exit class matches.
_mock_dist_check(dist::Symbol) = dist in (:normal, :student, :ged) || throw(ArgumentError(
    "dist must be :normal, :student, or :ged; got :$dist"))

estimate_arch(y, q) = ARCHModel(ones(q+2) * 0.1)
estimate_garch(y, p, q; dist::Symbol=:normal) =
    (_mock_dist_check(dist); GARCHModel(ones(p+q+2) * 0.1))
estimate_egarch(y, p, q; dist::Symbol=:normal) =
    (_mock_dist_check(dist); EGARCHModel(ones(2*q+p+2) * 0.1))
estimate_gjr_garch(y, p, q; dist::Symbol=:normal) =
    (_mock_dist_check(dist); GJRGARCHModel(ones(2*q+p+2) * 0.1))
estimate_sv(y; n_samples=5000, seed=nothing) = SVModel(ones(3) * 0.1)
coef(m::ARCHModel) = [m.mu, m.omega, m.alpha...]
coef(m::GARCHModel) = [m.mu, m.omega, m.alpha..., m.beta...]
coef(m::EGARCHModel) = [m.mu, m.omega, m.alpha..., m.gamma..., m.beta...]
coef(m::GJRGARCHModel) = [m.mu, m.omega, m.alpha..., m.gamma..., m.beta...]
coef(m::SVModel) = [mean(m.mu_post), mean(m.phi_post), mean(m.sigma_eta_post)]
persistence(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}) = 0.85
halflife(m::Union{GARCHModel,GJRGARCHModel}) = 4.3
unconditional_variance(m::Union{ARCHModel,GARCHModel}) = 0.02
function forecast(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}, h::Int)
    VolatilityForecast(ones(h) * 0.01, h)
end

# VAR forecast with bootstrap CI
function forecast(model::VARModel, h::Int; ci_method=:none, reps=500, conf_level=0.95)
    n = size(model.Y, 2)
    fc = ones(h, n) * 0.1
    # Real's frequentist path stores draws (n_draws × h × n) — verified live at
    # 0.8.0; the mock stores a small stack so the OPP draw branch is reachable.
    drw = reshape(fc, 1, h, n) .+ 0.05 .* randn(50, h, n)
    VARForecast{Float64}(fc, fc .- 0.5, fc .+ 0.5, h, ci_method, Float64(conf_level),
                         ["var$i" for i in 1:n], drw)
end

# Volatility test functions
arch_lm_test(y, lags) = (statistic=15.0, pvalue=0.01)
ljung_box_squared(y, lags) = (statistic=20.0, pvalue=0.005)

# ─── Multivariate GARCH functions (C064b) ────────────────
# Mirror the real _mgarch_validate, estimators, accessors, StatsAPI, and diagnostics so
# T1/T2 catch shape bugs in the wide correlation rendering + typed error mapping.
function _mock_mgarch_validate(Y)
    Ymat = Matrix{Float64}(Y)
    Tn, n = size(Ymat)
    n >= 2 || throw(ArgumentError("multivariate GARCH requires at least 2 series (got n=$n)"))
    Tn >= 2 || throw(ArgumentError("need at least 2 observations (got T=$Tn)"))
    all(isfinite, Ymat) || throw(ArgumentError("Y contains non-finite values"))
    return Ymat, Tn, n
end

# Build a plausible n×n correlation matrix (unit diagonal, 0.3 off-diagonal) and a
# unit-variance covariance path Hₜ = R.
function _mock_mgarch_RH(n::Int, Tn::Int)
    R = fill(0.3, n, n)
    for i in 1:n; R[i, i] = 1.0; end
    H = Array{Float64,3}(undef, n, n, Tn)
    for t in 1:Tn; H[:, :, t] .= R; end
    return R, H
end

function estimate_ccc(Y; p::Int=1, q::Int=1)
    Ymat, Tn, n = _mock_mgarch_validate(Y)
    R, H = _mock_mgarch_RH(n, Tn)
    margins = GARCHModel{Float64}[GARCHModel(ones(p + q + 2) * 0.1) for _ in 1:n]
    MGARCHModel{Float64}(Ymat, zeros(n), margins, H, R, R, Float64[], String[],
        fill(NaN, 0, 0), -300.0, 620.0, 640.0, :ccc, :none, :none, true, n)
end

function estimate_dcc(Y; p::Int=1, q::Int=1, correction::Symbol=:none)
    correction in (:none, :aielli) ||
        throw(ArgumentError("correction must be :none or :aielli, got :$correction"))
    Ymat, Tn, n = _mock_mgarch_validate(Y)
    R, H = _mock_mgarch_RH(n, Tn)
    margins = GARCHModel{Float64}[GARCHModel(ones(p + q + 2) * 0.1) for _ in 1:n]
    ab = [0.03, 0.95]
    MGARCHModel{Float64}(Ymat, zeros(n), margins, H, R, R, ab, ["a", "b"],
        Matrix{Float64}(0.0001I, 2, 2), -300.0, 620.0, 640.0, :dcc, correction, :none, true, n)
end

function estimate_bekk(Y; kind::Symbol=:scalar)
    kind in (:scalar, :diagonal) ||
        throw(ArgumentError("kind must be :scalar or :diagonal, got :$kind"))
    Ymat, Tn, n = _mock_mgarch_validate(Y)
    R, H = _mock_mgarch_RH(n, Tn)
    params, pnames = kind === :scalar ?
        ([0.05, 0.9], ["a", "b"]) :
        (vcat(fill(0.05, n), fill(0.9, n)), vcat(["a$i" for i in 1:n], ["b$i" for i in 1:n]))
    k = length(params)
    MGARCHModel{Float64}(Ymat, zeros(n), GARCHModel{Float64}[], H, R, R, params, pnames,
        Matrix{Float64}(0.0001I, k, k), -300.0, 620.0, 640.0, :bekk, :none, kind, true, n)
end

covariances(m::MGARCHModel) = m.H
function correlations(m::MGARCHModel)
    m.R isa Array{Float64,3} && return m.R
    Rc = m.R
    Tn = size(m.H, 3)
    out = Array{Float64,3}(undef, m.n, m.n, Tn)
    for t in 1:Tn; out[:, :, t] .= Rc; end
    return out
end
function variances(m::MGARCHModel)
    Tn = size(m.H, 3)
    out = Matrix{Float64}(undef, Tn, m.n)
    for t in 1:Tn, i in 1:m.n; out[t, i] = m.H[i, i, t]; end
    return out
end

coef(m::MGARCHModel) = m.params
loglikelihood(m::MGARCHModel) = m.loglik
nobs(m::MGARCHModel) = size(m.Y, 1)
function stderror(m::MGARCHModel)
    isempty(m.params) && return Float64[]
    V = m.param_vcov
    (size(V, 1) == length(m.params) && all(isfinite, V)) || return fill(NaN, length(m.params))
    return Float64[sqrt(max(V[i, i], 0.0)) for i in 1:length(m.params)]
end

# ─── Volatility residual diagnostics (C064b) ─────────────
function sign_bias_test(z::AbstractVector)
    length(z) < 10 && throw(ArgumentError("Need at least 10 observations for sign-bias test"))
    return (sign_bias=0.05, sign_bias_t=1.2, sign_bias_p=0.23,
            neg_size_t=-0.8, neg_size_p=0.42, pos_size_t=0.5, pos_size_p=0.62,
            joint_statistic=4.5, joint_pvalue=0.21, dof=3)
end
sign_bias_test(m::Union{GARCHModel,EGARCHModel,GJRGARCHModel}) =
    sign_bias_test(m.standardized_residuals)

nyblom_test(m::Union{GARCHModel,EGARCHModel,GJRGARCHModel}) =
    (individual=fill(0.3, 4), joint=0.8, k=4, cv_individual=0.470, cv_joint=1.24,
     param_names=["μ", "ω", "α1", "β1"])

# Non-Gaussian identification
function _mock_ica(model::VARModel, method_sym::Symbol)
    n = size(model.Y, 2); T_u = size(model.U, 1)
    ICASVARResult(ones(n,n)*0.3, ones(n,n)*0.3, Matrix{Float64}(I(n)),
                  ones(T_u, n)*0.1, method_sym, true, 50, 0.001)
end
identify_fastica(model::VARModel; contrast=:logcosh, max_iter=200, tol=1e-6, seed=nothing) = _mock_ica(model, :fastica)
identify_jade(model::VARModel) = _mock_ica(model, :jade)
identify_sobi(model::VARModel) = _mock_ica(model, :sobi)
identify_dcov(model::VARModel) = _mock_ica(model, :dcov)
identify_hsic(model::VARModel; seed=nothing) = _mock_ica(model, :hsic)

function _mock_ngml(model::VARModel, dist::Symbol)
    n = size(model.Y, 2); T_u = size(model.U, 1)
    NonGaussianMLResult(ones(n,n)*0.3, Matrix{Float64}(I(n)), ones(T_u, n)*0.1,
        dist, -200.0, -210.0, Dict{Symbol,Any}(:df => 5.0), ones(n,n)*0.01,
        ones(n*n)*0.05, -180.0, -175.0)
end
identify_nongaussian_ml(model::VARModel; distribution=:student_t, max_iter=500, tol=1e-6) = _mock_ngml(model, distribution)
identify_mixture_normal(model::VARModel) = _mock_ngml(model, :mixture_normal)
identify_pml(model::VARModel) = _mock_ngml(model, :pml)
identify_skew_normal(model::VARModel) = _mock_ngml(model, :skew_normal)

identify_markov_switching(model::VARModel; n_regimes=2, max_iter=200, tol=1e-6) =
    MarkovSwitchingSVARResult(ones(size(model.Y,2), size(model.Y,2))*0.3)
identify_garch(model::VARModel; max_iter=200, tol=1e-6) =
    GARCHSVARResult(ones(size(model.Y,2), size(model.Y,2))*0.3)
identify_smooth_transition(model::VARModel, transition_var; gamma=1.0, c=0.0) =
    SmoothTransitionSVARResult(ones(size(model.Y,2), size(model.Y,2))*0.3)
identify_external_volatility(model::VARModel, regime_indicator; regimes=2) =
    ExternalVolatilitySVARResult(ones(size(model.Y,2), size(model.Y,2))*0.3)

function normality_test_suite(model::VARModel)
    if _MOCK_FLAGS[:normality_all_pass]
        NormalityTestSuite([
            NormalityTestResult(:jarque_bera, 1.5, 0.47, 2),
            NormalityTestResult(:skewness, 0.8, 0.37, 1),
            NormalityTestResult(:kurtosis, 0.3, 0.58, 1),
        ])
    else
        NormalityTestSuite([
            NormalityTestResult(:jarque_bera, 15.0, 0.001, 2),
            NormalityTestResult(:skewness, 8.0, 0.02, 1),
            NormalityTestResult(:kurtosis, 3.0, 0.08, 1),
        ])
    end
end
test_identification_strength(model::VARModel; seed=nothing) = (statistic=25.0, pvalue=0.001)
test_shock_gaussianity(result::ICASVARResult) = (statistic=12.0, pvalue=0.005)
test_shock_independence(result::ICASVARResult; seed=nothing) = (statistic=3.0, pvalue=0.08)
test_overidentification(model::VARModel, result::ICASVARResult; seed=nothing) = (statistic=1.5, pvalue=0.45)
test_overidentification(result::ICASVARResult) = (statistic=1.5, pvalue=0.45)
test_gaussian_vs_nongaussian(model::VARModel) = (statistic=18.0, pvalue=0.001)

# ─── VECM Functions ──────────────────────────────────────

function estimate_vecm(Y::AbstractMatrix, p::Int; rank=nothing, deterministic=:constant,
                       method=:johansen, significance=0.05,
                       varnames=["y$i" for i in 1:size(Y, 2)])
    T_obs, n = size(Y)
    r = isnothing(rank) ? min(1, n - 1) : rank
    alpha = ones(n, r) * 0.1
    beta = ones(n, r) * 0.2
    Pi = alpha * beta'
    Gamma = [ones(n, n) * 0.05 for _ in 1:max(1, p - 1)]
    mu = zeros(n)
    U = zeros(T_obs - p, n) .+ 0.01
    Sigma = Matrix{Float64}(I(n)) * 0.01
    VECMModel(Y, p, r, alpha, beta, Pi, Gamma, mu, U, Sigma,
              -100.0, -95.0, -97.0, -500.0, deterministic, method,
              Vector{String}(varnames))
end

select_vecm_rank(Y::AbstractMatrix, p::Int; criterion=:trace, significance=0.05) =
    min(1, size(Y, 2) - 1)

function to_var(vecm::VECMModel)
    # Real threads vecm.varnames into the VAR representation (#119).
    _mock_var(vecm.Y, vecm.p; varnames=vecm.varnames)
end

cointegrating_rank(m::VECMModel) = m.rank
coef(m::VECMModel) = m.Pi
loglikelihood(m::VECMModel) = m.loglik
report(::VECMModel) = nothing

function forecast(vecm::VECMModel, h::Int; ci_method=:none, reps=500, conf_level=0.95)
    n = size(vecm.Y, 2)
    levels = ones(h, n) * 0.1
    diffs = ones(h, n) * 0.01
    has_ci = ci_method != :none
    VECMForecast(levels, diffs,
        has_ci ? levels .- 0.5 : nothing,
        has_ci ? levels .+ 0.5 : nothing,
        h, ci_method)
end

function granger_causality_vecm(vecm::VECMModel, cause::Int, effect::Int)
    VECMGrangerResult(
        8.5, 0.014, 2,   # short-run
        5.2, 0.023, 1,   # long-run
        12.3, 0.006, 3,  # strong (joint)
        cause, effect)
end

# ── VECM restriction tests (C071) ─────────────────────────
# Validate the same way real MEMs does (r≥1, matrix rows==nvars, s/a≥r, var range)
# so T1/T2 catch the bad-input→typed-error mapping; df formulas are faithful, other
# constants are placeholders (like arch_lm_test).
function _vecm_restriction_result(kind::Symbol, m::VECMModel, df::Int, desc::String;
                                  converged::Bool=true)
    r = m.rank
    VECMRestrictionTest(kind, 3.2, df, 0.36, r, desc,
        m.beta, m.beta, fill(0.3, r), fill(0.3, r), converged, m)
end

function test_beta_restriction(m::VECMModel, H::AbstractMatrix)
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("β restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    size(H, 1) == n || throw(DimensionMismatch("H must have $n rows (nvars), got $(size(H,1))"))
    s = size(H, 2)
    s >= r || throw(ArgumentError("H must have at least r=$r columns, got s=$s"))
    _vecm_restriction_result(:beta, m, r * (n - s), "β = Hφ (restricted to span(H), s=$s)")
end

function test_alpha_restriction(m::VECMModel, A::AbstractMatrix)
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("α restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    size(A, 1) == n || throw(DimensionMismatch("A must have $n rows (nvars), got $(size(A,1))"))
    a = size(A, 2)
    a >= r || throw(ArgumentError("A must have at least r=$r columns, got a=$a"))
    _vecm_restriction_result(:alpha, m, r * (n - a), "α = Aψ (restricted to span(A), a=$a)")
end

function test_weak_exogeneity(m::VECMModel, vars)
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("weak-exogeneity test requires cointegrating rank ≥ 1 (got r=$r)"))
    vv = vars isa Union{AbstractVector,Tuple} ? collect(vars) : [vars]
    ex_idx = Int[]
    for v in vv
        if v isa Integer
            push!(ex_idx, Int(v))
        else
            name = String(v)
            j = findfirst(==(name), m.varnames)
            j === nothing && throw(ArgumentError("Variable '$name' not found. Available: $(m.varnames)"))
            push!(ex_idx, j)
        end
    end
    all(1 .<= ex_idx .<= n) || throw(ArgumentError("variable index out of range 1:$n"))
    isempty(setdiff(1:n, ex_idx)) && throw(ArgumentError("cannot make all variables weakly exogenous"))
    labels = join([m.varnames[i] for i in ex_idx], ", ")
    _vecm_restriction_result(:weak_exogeneity, m, r * length(ex_idx),
        "Weak exogeneity of {$labels} (α rows = 0, df = r·m)")
end

function test_known_beta(m::VECMModel, b::AbstractMatrix)
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("known-β test requires cointegrating rank ≥ 1 (got r=$r)"))
    size(b, 1) == n || throw(DimensionMismatch("b must have $n rows (nvars), got $(size(b,1))"))
    size(b, 2) == r || throw(DimensionMismatch("b must have exactly r=$r columns, got $(size(b,2))"))
    _vecm_restriction_result(:known_beta, m, r * (n - r), "β = b (fully known cointegrating space)")
end

function test_joint_restriction(m::VECMModel, H::AbstractMatrix, A::AbstractMatrix;
                                maxiter::Int=1000, tol::Real=1e-8)
    n = nvars(m); r = m.rank
    r >= 1 || throw(ArgumentError("joint restriction test requires cointegrating rank ≥ 1 (got r=$r)"))
    size(H, 1) == n || throw(DimensionMismatch("H must have $n rows, got $(size(H,1))"))
    size(A, 1) == n || throw(DimensionMismatch("A must have $n rows, got $(size(A,1))"))
    s = size(H, 2); a = size(A, 2)
    (s >= r && a >= r) || throw(ArgumentError("need s ≥ r and a ≥ r (got s=$s, a=$a, r=$r)"))
    _vecm_restriction_result(:joint, m, r * (n - s) + r * (n - a),
        "Joint β=Hφ (s=$s), α=Aψ (a=$a) via switching")
end

# ─── Panel VAR Types ────────────────────────────────────────

struct PanelData{T<:Real}
    data::Matrix{T}; varnames::Vector{String}; group_id::Vector{Int}; time_id::Vector{Int}
    n_groups::Int; n_vars::Int; T_obs::Int; balanced::Bool
end

struct PVARModel{T<:Real}
    Phi::Matrix{T}; Sigma::Matrix{T}; se::Matrix{T}; pvalues::Matrix{T}
    m::Int; p::Int; method::Symbol; transformation::Symbol; steps::Symbol
    n_groups::Int; n_periods::Int; n_obs::Int; n_instruments::Int
end

struct PVARStability{T<:Real}
    eigenvalues::Vector{Complex{T}}; moduli::Vector{T}; is_stable::Bool
end

struct PVARTestResult{T<:Real}
    test_name::String; statistic::T; pvalue::T; df::Int; n_instruments::Int; n_params::Int
end

struct GrangerCausalityResult{T<:Real}
    # Real layout (teststat/granger.jl:47): cause is a Vector of variable INDICES,
    # effect an index; the old mock invented String names here (#118 shape gate).
    statistic::T; pvalue::T; df::Int; cause::Vector{Int}; effect::Int
    n::Int; p::Int; nobs::Int; test_type::Symbol
end

struct LRTestResult{T<:Real}
    statistic::T; pvalue::T; df::Int; loglik_restricted::T; loglik_unrestricted::T
end

struct LMTestResult{T<:Real}
    statistic::T; pvalue::T; df::Int; nobs::Int; score_norm::T
end

# ─── Panel VAR Functions ────────────────────────────────────

# MEMs 0.7.0 signature: xtset(df, group_col::Symbol, time_col::Symbol; ...).
# `df` is left untyped so the mock module needs no DataFrames import — column
# access dispatches on the passed DataFrame via Base (propertynames/size/getindex).
function xtset(df, group_col::Symbol, time_col::Symbol;
               varnames=nothing, frequency=nothing, tcode=nothing,
               desc="", vardesc=nothing, cohort=nothing)
    exclude = Set{Symbol}([group_col, time_col])
    cohort === nothing || push!(exclude, cohort)
    num_cols = [c for c in propertynames(df)
                if !(c in exclude) && eltype(df[!, c]) <: Union{Missing,Number}]
    n = length(num_cols)
    vn = varnames === nothing ? String[string(c) for c in num_cols] : Vector{String}(varnames)
    T_obs = size(df, 1)
    data = Matrix{Float64}(undef, T_obs, n)
    for (j, c) in enumerate(num_cols)
        col = df[!, c]
        for i in 1:T_obs
            v = col[i]
            data[i, j] = ismissing(v) ? NaN : Float64(v)
        end
    end
    raw_g = df[!, group_col]
    ug = unique(raw_g)
    gmap = Dict(g => i for (i, g) in enumerate(ug))
    gid = Int[gmap[g] for g in raw_g]
    raw_t = df[!, time_col]
    tid = eltype(raw_t) <: Integer ? Int.(raw_t) : begin
        ut = sort(unique(raw_t)); tmap = Dict(t => i for (i, t) in enumerate(ut))
        Int[tmap[t] for t in raw_t]
    end
    # Mirror real xtset: a duplicate (group, time) pair is an invalid panel → ArgumentError
    # (load_panel_data maps this to a typed data/invalid, so the panel family's hardening is
    # exercised at T1/T2, not only against real MEMs at T3).
    length(unique(zip(gid, tid))) == T_obs ||
        throw(ArgumentError("duplicate (group, time) pairs are not allowed in a panel"))
    PanelData(data, vn, gid, tid, length(ug), n, T_obs, true)
end

isbalanced(pd::PanelData) = pd.balanced
ngroups(pd::PanelData) = pd.n_groups

# MEMs 0.7.0 kwargs (C054): system→system_instruments, dependent→dependent_vars,
# predetermined→predet_vars, exogenous→exog_vars. Kwargs enumerated (no catch-all)
# to stay within the check_mock_surface absorber budget.
function estimate_pvar(panel::PanelData, p::Int;
                       transformation=:fd, steps=:twostep, system_instruments=false,
                       collapse=false, dependent_vars=nothing,
                       predet_vars=String[], exog_vars=String[],
                       min_lag_endo=2, max_lag_endo=99)
    n = panel.n_vars
    k = n * p + 1
    Phi = ones(k, n) * 0.3
    Sigma = Matrix{Float64}(I(n)) * 0.01
    se = ones(k, n) * 0.05
    pvals = ones(k, n) * 0.02
    n_inst = system_instruments ? 2 * k : k + p
    PVARModel(Phi, Sigma, se, pvals, n, p, :gmm, transformation, steps,
              panel.n_groups, panel.T_obs ÷ panel.n_groups, panel.T_obs, n_inst)
end

function estimate_pvar_feols(panel::PanelData, p::Int;
                              dependent_vars=nothing, exog_vars=String[])
    n = panel.n_vars
    k = n * p + 1
    Phi = ones(k, n) * 0.25
    Sigma = Matrix{Float64}(I(n)) * 0.01
    se = ones(k, n) * 0.04
    pvals = ones(k, n) * 0.01
    PVARModel(Phi, Sigma, se, pvals, n, p, :feols, :fd, :onestep,
              panel.n_groups, panel.T_obs ÷ panel.n_groups, panel.T_obs, 0)
end

coef(m::PVARModel) = m.Phi
report(::PVARModel) = nothing

function pvar_oirf(model::PVARModel, horizon::Int)
    n = model.m
    vals = ones(horizon + 1, n, n) * 0.1
    ImpulseResponse(vals, nothing, nothing)
end

function pvar_girf(model::PVARModel, horizon::Int)
    n = model.m
    vals = ones(horizon + 1, n, n) * 0.12
    ImpulseResponse(vals, nothing, nothing)
end

# MEMs 0.7.0 (C054): kwargs n_boot→n_draws, conf_level→ci; returns a NamedTuple
# (irf, lower, upper, draws) of raw (H+1)×n×n arrays, not an ImpulseResponse.
function pvar_bootstrap_irf(model::PVARModel, horizon::Int;
                             n_draws=500, ci=0.95, irf_type=:oirf, seed=nothing)
    n = model.m
    vals = ones(horizon + 1, n, n) * 0.1
    (irf=vals, lower=vals .- 0.5, upper=vals .+ 0.5,
     draws=ones(horizon + 1, n, n, n_draws) * 0.1)
end

# MEMs 0.7.0 (C054): returns a raw (H+1)×n×n array [horizon, variable, shock].
function pvar_fevd(model::PVARModel, horizon::Int)
    n = model.m
    ones(horizon + 1, n, n) / n
end

function pvar_stability(model::PVARModel)
    if _MOCK_FLAGS[:pvar_stable]
        eigs = [0.5 + 0.1im, 0.5 - 0.1im, 0.3 + 0.0im]
        moduli = abs.(eigs)
        PVARStability(eigs, moduli, true)
    else
        eigs = [1.2 + 0.0im, 0.5 - 0.1im, 0.3 + 0.0im]
        moduli = abs.(eigs)
        PVARStability(eigs, moduli, false)
    end
end

function pvar_hansen_j(model::PVARModel)
    PVARTestResult("Hansen J", 8.5, 0.38, model.n_instruments - model.m * model.p - model.m,
                   model.n_instruments, model.m * model.p + model.m)
end

# MEMs 0.7.0 (C054): pvar_mmsc is a single-model criterion (was a selection loop).
function pvar_mmsc(model::PVARModel; hq_criterion=2.1)
    (mbic=-100.0, maic=-110.0, mqic=-105.0)
end

# MEMs 0.7.0 (C054): returns (table, best_bic, best_aic, best_hqic, models);
# `.table` is a Matrix{Any} with columns [p, BIC, AIC, HQIC].
function pvar_lag_selection(panel::PanelData, max_p::Int; dependent_vars=nothing)
    tbl = Matrix{Any}(undef, max_p, 4)
    for p in 1:max_p
        tbl[p, 1] = p
        tbl[p, 2] = string(-100.0 + p)
        tbl[p, 3] = string(-110.0 + p)
        tbl[p, 4] = string(-105.0 + p)
    end
    (table=tbl, best_bic=1, best_aic=1, best_hqic=1, models=PVARModel[])
end

# Enhanced Granger causality for VAR. Real granger_test takes NO kwargs, and
# granger_test_all returns an n×n Matrix{Union{GrangerCausalityResult,Nothing}}
# with nothing on the diagonal, indexed [effect, cause] — NOT a flat vector.
function granger_test(model::VARModel, cause::Int, effect::Int)
    n = size(model.Y, 2)
    GrangerCausalityResult(12.5, 0.003, model.p, [cause], effect,
                           n, model.p, size(model.Y, 1) - model.p, :pairwise)
end

function granger_test_all(model::VARModel)
    n = size(model.Y, 2)
    results = Matrix{Union{GrangerCausalityResult{Float64},Nothing}}(nothing, n, n)
    for effect in 1:n, cause in 1:n
        cause == effect && continue
        results[effect, cause] = GrangerCausalityResult(10.0 + cause, 0.01, model.p, [cause],
                                                        effect, n, model.p,
                                                        size(model.Y, 1) - model.p, :pairwise)
    end
    results
end

# LR and LM tests
function lr_test(m_restricted::VARModel, m_unrestricted::VARModel)
    ll_r = -510.0
    ll_u = -500.0
    stat = 2 * (ll_u - ll_r)
    LRTestResult(stat, 0.02, 3, ll_r, ll_u)
end

function lm_test(m_restricted::VARModel, m_unrestricted::VARModel)
    LMTestResult(15.0, 0.005, 3, size(m_restricted.Y, 1), 3.87)
end

# ─── Filter Types & Functions ─────────────────────────────

struct HPFilterResult{T}
    trend::Vector{T}; cycle::Vector{T}; lambda::T; T_obs::Int
end

struct HamiltonFilterResult{T}
    trend::Vector{T}; cycle::Vector{T}; beta::Vector{T}; h::Int; p::Int; T_obs::Int; valid_range::UnitRange{Int}
end

struct BeveridgeNelsonResult{T}
    permanent::Vector{T}; transitory::Vector{T}; drift::T; long_run_multiplier::T; arima_order::Tuple{Int,Int,Int}; T_obs::Int
end

struct BaxterKingResult{T}
    cycle::Vector{T}; trend::Vector{T}; weights::Vector{T}; pl::Int; pu::Int; K::Int; T_obs::Int; valid_range::UnitRange{Int}
end

struct BoostedHPResult{T}
    trend::Vector{T}; cycle::Vector{T}; lambda::T; iterations::Int; stopping::Symbol; bic_path::Vector{T}; adf_pvalues::Vector{T}; T_obs::Int
end

trend(r::HPFilterResult) = r.trend
cycle(r::HPFilterResult) = r.cycle
trend(r::HamiltonFilterResult) = r.trend
cycle(r::HamiltonFilterResult) = r.cycle
trend(r::BeveridgeNelsonResult) = r.permanent
cycle(r::BeveridgeNelsonResult) = r.transitory
trend(r::BaxterKingResult) = r.trend
cycle(r::BaxterKingResult) = r.cycle
trend(r::BoostedHPResult) = r.trend
cycle(r::BoostedHPResult) = r.cycle

function hp_filter(y::AbstractVector; lambda=1600.0)
    T = length(y)
    t = cumsum(ones(T)) .* mean(y) / T
    c = y .- t
    HPFilterResult(t, c, Float64(lambda), T)
end

function hamilton_filter(y::AbstractVector; h=8, p=4)
    T = length(y)
    start = h + p
    valid = (start+1):T
    t = cumsum(ones(T)) .* mean(y) / T
    c = y .- t
    beta = ones(p + 1) * 0.1
    HamiltonFilterResult(t, c, beta, h, p, T, valid)
end

function beveridge_nelson(y::AbstractVector; p=:auto, q=:auto, max_terms=500, method=:arima)
    T = length(y)
    t = cumsum(ones(T)) .* mean(y) / T
    c = y .- t
    p_val = p == :auto ? 1 : p
    q_val = q == :auto ? 0 : q
    BeveridgeNelsonResult(t, c, 0.01, 1.5, (p_val, 0, q_val), T)
end

function baxter_king(y::AbstractVector; pl=6, pu=32, K=12)
    T = length(y)
    valid = (K+1):(T-K)
    t = cumsum(ones(T)) .* mean(y) / T
    c = y .- t
    weights = ones(2K + 1) / (2K + 1)
    BaxterKingResult(c, t, weights, pl, pu, K, T, valid)
end

function boosted_hp(y::AbstractVector; lambda=1600.0, stopping=:BIC, max_iter=100, sig_p=0.05)
    T = length(y)
    t = cumsum(ones(T)) .* mean(y) / T
    c = y .- t
    iters = 3
    bic_path = [10.0, 8.0, 9.0]
    adf_pvals = [0.5, 0.1, 0.01]
    BoostedHPResult(t, c, Float64(lambda), iters, stopping, bic_path, adf_pvals, T)
end

# ─── Exports ──────────────────────────────────────────────

export _MOCK_FLAGS
export VARModel, MockChains, BVARPosterior, MinnesotaHyperparameters
export ImpulseResponse, BayesianImpulseResponse, FEVD, BayesianFEVD
export HistoricalDecomposition, BayesianHistoricalDecomposition
export ZeroRestriction, SignRestriction, SVARRestrictions, AriasSVARResult, UhligSVARResult
export LongRunZeroRestriction, A0ZeroRestriction, A0SignRestriction
export ElasticityBound, MagnitudeBound, CumulativeRestriction
export NarrativeShockRestriction, NarrativeContributionRestriction
export IdentificationStatus, SVARPattern
export ProxySVARResult, MaxShareResult, NonGaussianGMMResult, SVECResult, SVARModel
export RobustBayesResult
export LPModel, LPIVModel, SmoothLPModel, StateLPModel, PropensityLPModel
export LPImpulseResponse, StructuralLP, LPFEVD, LPForecast
export FactorModel, DynamicFactorModel, GeneralizedDynamicFactorModel, FactorForecast
export ARModel, MAModel, ARMAModel, ARIMAModel, ARIMAForecast, ARFIMAModel
export ICASVARResult, NonGaussianMLResult
export MarkovSwitchingSVARResult, GARCHSVARResult, SmoothTransitionSVARResult, ExternalVolatilitySVARResult
export NormalityTestResult, NormalityTestSuite
export ADFResult, KPSSResult, PPResult, ZAResult, NgPerronResult, JohansenResult
export GPHResult, LocalWhittleResult
export GMMModel
export ARCHModel, GARCHModel, EGARCHModel, GJRGARCHModel, SVModel, VolatilityForecast
export VECMModel, VECMForecast, VECMGrangerResult, VECMRestrictionTest
export test_beta_restriction, test_alpha_restriction, test_weak_exogeneity
export test_known_beta, test_joint_restriction
export PanelData, PVARModel, PVARStability, PVARTestResult
export GrangerCausalityResult, LRTestResult, LMTestResult
export HPFilterResult, HamiltonFilterResult, BeveridgeNelsonResult, BaxterKingResult, BoostedHPResult

export select_lag_order, estimate_var, estimate_bvar, posterior_mean_model, posterior_median_model
export optimize_hyperparameters, optimize_hyperparameters_glp, GLPHyperparameters
export coef, loglikelihood, stderror, predict, residuals, report
export is_stationary, companion_matrix, companion_matrix_factors, nvars
export irf, fevd, historical_decomposition, verify_decomposition, contribution
export cumulative_irf
export SignIdentifiedSet, identify_sign, irf_bounds, irf_median
export VARForecast
export zero_restriction, sign_restriction, identify_arias, irf_mean, identify_uhlig, identify_narrative
export a0_zero_restriction, a0_sign_restriction, elasticity_bound, magnitude_bound
export cumulative_restriction, narrative_shock_restriction, narrative_contribution_restriction
export identify_proxy, identify_max_share, identify_gmm_moments, identify_svec
export estimate_svar, recursive_pattern, a_model_pattern, b_model_pattern, ab_model_pattern
export blanchard_quah_pattern, identify_robust_bayes, label_shocks
export median_target, modal_model, joint_band, sup_t_band, check_identification
export test_lambda_distinct, test_gaussian_shock_count, test_label_stability
export estimate_lp, lp_irf, estimate_lp_iv, lp_iv_irf, weak_instrument_test
export estimate_smooth_lp, smooth_lp_irf, cross_validate_lambda
export estimate_state_lp, state_irf, test_regime_difference
export estimate_propensity_lp, propensity_irf, propensity_diagnostics, doubly_robust_lp
export structural_lp, lp_fevd, forecast
export estimate_factors, ic_criteria, scree_plot_data
export estimate_dynamic_factors, ic_criteria_gdfm, estimate_gdfm, common_variance_share
export adf_test, kpss_test, pp_test, za_test, ngperron_test, johansen_test
export gph_test, local_whittle
export estimate_lp_gmm, gmm_summary, j_test
export estimate_ar, estimate_ma, estimate_arma, estimate_arima, auto_arima, estimate_arfima
export ar_order, ma_order, diff_order, aic, bic
export estimate_arch, estimate_garch, estimate_egarch, estimate_gjr_garch, estimate_sv
export persistence, halflife, unconditional_variance
export arch_lm_test, ljung_box_squared
export MGARCHModel, estimate_ccc, estimate_dcc, estimate_bekk
export covariances, correlations, variances, sign_bias_test, nyblom_test
export identify_fastica, identify_jade, identify_sobi, identify_dcov, identify_hsic
export identify_nongaussian_ml, identify_mixture_normal, identify_pml, identify_skew_normal
export identify_markov_switching, identify_garch, identify_smooth_transition, identify_external_volatility
export normality_test_suite, test_identification_strength, test_shock_gaussianity
export test_shock_independence, test_overidentification, test_gaussian_vs_nongaussian
export estimate_vecm, select_vecm_rank, to_var, cointegrating_rank, granger_causality_vecm
export xtset, isbalanced, ngroups, estimate_pvar, estimate_pvar_feols
export pvar_oirf, pvar_girf, pvar_bootstrap_irf, pvar_fevd, pvar_stability
export pvar_hansen_j, pvar_mmsc, pvar_lag_selection
export granger_test, granger_test_all, lr_test, lm_test
export hp_filter, hamilton_filter, beveridge_nelson, baxter_king, boosted_hp, trend, cycle

# ─── Data Module Types & Functions ────────────────────────

struct TimeSeriesData{T<:Real}
    data::Matrix{T}; varnames::Vector{String}; frequency::Symbol
    # Real desc is a ONE-ELEMENT Vector{String} (accessor is d.desc[1]) — #118 shape gate.
    tcode::Vector{Int}; time_index::Vector{Int}; desc::Vector{String}; vardesc::Vector{String}
end
# Keyword constructor matching MacroEconometricModels v0.2.2 interface
function TimeSeriesData(data::AbstractMatrix{T}; varnames=String[], frequency=:unknown,
                        tcode=fill(1, size(data, 2)), time_index=collect(1:size(data, 1)),
                        desc="", vardesc=fill("", size(data, 2)), source_refs=Symbol[]) where T<:Real
    TimeSeriesData{T}(Matrix{T}(data), varnames, frequency, tcode, time_index,
                      desc isa AbstractString ? [String(desc)] : desc, vardesc)
end

# Field order is a prefix of the real CrossSectionData (source_refs omitted).
struct CrossSectionData{T<:Real}
    data::Matrix{T}; varnames::Vector{String}; obs_id::Vector{Int}
    N_obs::Int; n_vars::Int; desc::Vector{String}; vardesc::Vector{String}
end
# Keyword constructor matching real MEMs `CrossSectionData(data; varnames, obs_id, …)`.
# Mock `vardesc` is Vector{String} (real is Dict); convert a Dict when given.
function CrossSectionData(data::AbstractMatrix{T};
                          varnames=nothing,
                          obs_id=nothing,
                          desc::AbstractString="",
                          vardesc=nothing,
                          source_refs=Symbol[]) where {T<:Real}
    N_obs, n_vars = size(data)
    vn = varnames === nothing ? String["x$i" for i in 1:n_vars] : Vector{String}(varnames)
    oid = obs_id === nothing ? collect(1:N_obs) : Vector{Int}(obs_id)
    vd = if vardesc === nothing
        fill("", n_vars)
    elseif vardesc isa AbstractDict
        String[String(get(vardesc, v, "")) for v in vn]
    else
        Vector{String}(vardesc)
    end
    CrossSectionData{T}(Matrix{T}(data), vn, oid, N_obs, n_vars, [String(desc)], vd)
end

struct DataDiagnostic
    n_nan::Vector{Int}; n_inf::Vector{Int}; is_constant::Vector{Bool}
    is_short::Bool; is_clean::Bool
end

struct DataSummary{T<:Real}
    # Real n is Vector{Int} — the NON-NaN count per variable (a scalar here made
    # the mock blind to NaN-padded datasets like :mp_shocks).
    n::Vector{Int}; mean::Vector{T}; std::Vector{T}; min::Vector{T}
    p25::Vector{T}; median::Vector{T}; p75::Vector{T}; max::Vector{T}
    skewness::Vector{T}; kurtosis::Vector{T}
end

function load_example(name::Symbol)
    if name == :fred_md
        n_vars = 126; T_obs = 804
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["INDPRO", "CPIAUCSL", "FEDFUNDS", ["var$i" for i in 4:n_vars]...]
        tc = vcat([5, 5, 1], [1 for _ in 4:n_vars])
        vd = vcat(["Industrial Production", "CPI All Urban", "Fed Funds Rate"],
                  ["Variable $i" for i in 4:n_vars])
        TimeSeriesData(data, vn, :monthly, tc, collect(1:T_obs),
            ["FRED-MD Monthly Database (2024 vintage)"], vd)
    elseif name == :fred_qd
        n_vars = 245; T_obs = 268
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["GDP", "PCECC96", ["var$i" for i in 3:n_vars]...]
        tc = vcat([5, 5], [1 for _ in 3:n_vars])
        vd = vcat(["Real GDP", "Real PCE"], ["Variable $i" for i in 3:n_vars])
        TimeSeriesData(data, vn, :quarterly, tc, collect(1:T_obs),
            ["FRED-QD Quarterly Database"], vd)
    elseif name == :pwt
        n_vars = 42; n_countries = 38; T_per = 74
        T_obs = n_countries * T_per
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["rgdpna", "pop", ["var$i" for i in 3:n_vars]...]
        group_ids = repeat(1:n_countries, inner=T_per)
        time_ids = repeat(1:T_per, outer=n_countries)
        PanelData(data, vn, group_ids, time_ids, n_countries, n_vars, T_obs, true)
    elseif name == :mpdta
        # Callaway-Sant'Anna (2021) minimum wage panel: 500 counties × 5 years × 3 vars
        n_groups = 500; n_years = 5; n_vars = 3
        T_obs = n_groups * n_years
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["lemp", "lpop", "first_treat"]
        group_ids = repeat(1:n_groups, inner=n_years)
        time_ids = repeat(2003:2007, outer=n_groups)
        PanelData(data, vn, group_ids, time_ids, n_groups, n_vars, T_obs, true)
    elseif name == :ddcg
        # Acemoglu et al. democracy-GDP panel: 184 countries × 51 years
        n_groups = 184; n_years = 51; n_vars = 5
        T_obs = n_groups * n_years
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["y", "dem", "tradewb", "lgdp", "lpop"]
        group_ids = repeat(1:n_groups, inner=n_years)
        time_ids = repeat(1960:2010, outer=n_groups)
        PanelData(data, vn, group_ids, time_ids, n_groups, n_vars, T_obs, true)
    elseif name == :denmark
        # Johansen-Juselius Danish money demand: 55 quarters × 5 vars
        T_obs = 55
        vn = ["LRM", "LRY", "LPY", "IBO", "IDE"]
        TimeSeriesData(randn(T_obs, 5) .+ 1.0, vn, :quarterly, fill(1, 5),
            collect(1:T_obs), ["Danish money demand"], ["Variable $v" for v in vn])
    elseif name == :gnp_hamilton
        # Hamilton (1989) US GNP growth: 135 quarters × 1 var
        T_obs = 135
        TimeSeriesData(randn(T_obs, 1) .+ 1.0, ["gnp_growth"], :quarterly, [1],
            collect(1:T_obs), ["US GNP growth (Hamilton 1989)"], ["GNP growth"])
    elseif name == :mp_shocks
        # McKay-Wolf (2023) US monetary panel (CF-20/MEMs#400): 240 quarters × 8
        # vars, NaN outside each series' published sample. The NaN mask below is
        # the REAL per-column valid range (row 1 = 1960Q1), so T1/T2 exercise the
        # same valid-count/valid-window arithmetic as T3 — NaN is not zero.
        T_obs = 240
        vn = ["ygap", "infl", "ffr", "lpcom", "rr", "mp1", "ad", "bzk_ist"]
        valid = [(37, 240), (1, 223), (1, 223), (1, 213),
                 (37, 192), (116, 210), (92, 195), (1, 209)]
        data = fill(NaN, T_obs, 8)
        for (j, (f, l)) in enumerate(valid)
            data[f:l, j] = randn(l - f + 1) .+ 2.0
        end
        TimeSeriesData(data, vn, :quarterly, fill(1, 8), collect(1:T_obs),
            ["US monetary panel with published policy-shock series (McKay-Wolf 2023)"],
            ["Variable $v" for v in vn])
    elseif name == :nile
        # Nile annual flow: 100 years × 1 var
        T_obs = 100
        TimeSeriesData(randn(T_obs, 1) .+ 900.0, ["flow"], :annual, [1],
            collect(1:T_obs), ["Nile river annual flow"], ["Flow"])
    elseif name == :grunfeld
        # Grunfeld investment panel: 10 firms × 20 years × 3 vars
        n_groups = 10; n_years = 20; n_vars = 3
        T_obs = n_groups * n_years
        vn = ["invest", "value", "capital"]
        group_ids = repeat(1:n_groups, inner=n_years)
        time_ids = repeat(1935:1954, outer=n_groups)
        PanelData(randn(T_obs, n_vars) .+ 1.0, vn, group_ids, time_ids,
                  n_groups, n_vars, T_obs, true)
    elseif name == :mroz
        # Mroz (1987) female labour supply: 753 × 22 cross section
        vn = vcat(["inlf", "hours", "kidslt6", "kidsge6"], ["var$i" for i in 5:22])
        CrossSectionData(randn(753, 22) .+ 1.0, vn, collect(1:753), 753, 22,
                         ["Mroz (1987) female labour supply"], ["Variable $v" for v in vn])
    elseif name == :stackloss
        # Brownlee stack-loss plant data: 21 × 4 cross section
        vn = ["stackloss", "airflow", "watertemp", "acidconc"]
        CrossSectionData(randn(21, 4) .+ 1.0, vn, collect(1:21), 21, 4,
                         ["Brownlee stack loss"], ["Variable $v" for v in vn])
    elseif name == :wiot
        _mock_wiot()   # Miller & Blair (2009) IO fixture (defined below)
    else
        # Real load_example throws ArgumentError — match it so error-mapping is testable.
        throw(ArgumentError("Unknown dataset :$name. Available: fred_md, fred_qd, pwt, mpdta, ddcg, " *
                            "denmark, gnp_hamilton, grunfeld, mp_shocks, mroz, nile, stackloss, wiot"))
    end
end

to_matrix(d::TimeSeriesData) = d.data
to_matrix(d::PanelData) = d.data
to_matrix(d::CrossSectionData) = d.data
varnames(d::TimeSeriesData) = d.varnames
varnames(d::PanelData) = d.varnames
varnames(d::CrossSectionData) = d.varnames
frequency(d::TimeSeriesData) = d.frequency
desc(d::TimeSeriesData) = d.desc[1]
vardesc(d::TimeSeriesData) = d.vardesc
nobs(d::TimeSeriesData) = size(d.data, 1)
nvars(d::TimeSeriesData) = size(d.data, 2)

function describe_data(d::TimeSeriesData)
    # Real describe_data excludes NaN/Inf from every statistic and counts finite
    # observations per variable — averaging NaN straight in hid the whole
    # NaN-padded-dataset class (:mp_shocks) from T1/T2.
    nv = size(d.data, 2)
    cols = [filter(isfinite, d.data[:, j]) for j in 1:nv]
    n = [length(c) for c in cols]
    m = [isempty(c) ? NaN : mean(c) for c in cols]
    s = [length(c) < 2 ? NaN :
         sqrt(sum(abs2, c .- mean(c)) / (length(c) - 1)) for c in cols]
    mn = [isempty(c) ? NaN : minimum(c) for c in cols]
    mx = [isempty(c) ? NaN : maximum(c) for c in cols]
    p25 = m .- 0.67 .* s
    med = copy(m)
    p75 = m .+ 0.67 .* s
    sk = fill(0.1, nv)
    ku = fill(3.0, nv)
    DataSummary(n, m, s, mn, p25, med, p75, mx, sk, ku)
end
# Real MEMs has describe_data(::PanelData) / (::CrossSectionData) returning the
# same DataSummary. PanelData also prints panel_summary to stdout (real
# summary_stats.jl); the CLI captures that dump onto stderr.
function panel_summary(io::IO, d::PanelData)
    println(io, "Panel Structure: $(d.n_groups) groups, $(d.T_obs) total observations")
    println(io, "  Balance: ", d.balanced ? "balanced" : "unbalanced")
    println(io, "  Variables: ", join(d.varnames, ", "))
end
panel_summary(d::PanelData) = panel_summary(stdout, d)

function describe_data(d::PanelData)
    s = describe_data(TimeSeriesData(d.data; varnames=d.varnames))
    try
        panel_summary(stdout, d)
    catch e
        e isa Base.IOError || rethrow()
    end
    s
end
describe_data(d::CrossSectionData) = describe_data(TimeSeriesData(d.data; varnames=d.varnames))

# Simple std without Distributions dependency
function std_mock(X::AbstractMatrix)
    T_obs = size(X, 1)
    m = mean(X; dims=1)
    sqrt.(sum((X .- m).^2; dims=1) ./ max(1, T_obs - 1))
end

function diagnose(d::TimeSeriesData)
    T_obs, n = size(d.data)
    n_nan = [count(isnan, d.data[:, i]) for i in 1:n]
    n_inf = [count(isinf, d.data[:, i]) for i in 1:n]
    is_const = [all(d.data[:, i] .== d.data[1, i]) for i in 1:n]
    is_short = T_obs < 30
    is_clean = all(n_nan .== 0) && all(n_inf .== 0) && !any(is_const) && !is_short
    DataDiagnostic(n_nan, n_inf, is_const, is_short, is_clean)
end
# Real diagnose(::AbstractMacroData) covers PanelData / CrossSectionData.
diagnose(d::PanelData) = diagnose(TimeSeriesData(d.data; varnames=d.varnames))
diagnose(d::CrossSectionData) = diagnose(TimeSeriesData(d.data; varnames=d.varnames))

function fix(d::TimeSeriesData; method=:listwise)
    # Mock: return same data (pretend it was cleaned)
    TimeSeriesData(copy(d.data), d.varnames, d.frequency, d.tcode, d.time_index, d.desc, d.vardesc)
end

function apply_tcode(d::TimeSeriesData, codes::Vector{Int})
    # Mock: return same data (pretend transformations applied)
    TimeSeriesData(copy(d.data), d.varnames, d.frequency, codes, d.time_index, d.desc, d.vardesc)
end

function validate_for_model(d::TimeSeriesData, model_type::Symbol)
    n = nvars(d)
    T_obs = nobs(d)
    if model_type in (:arima, :arch, :garch, :egarch, :gjr_garch, :sv) && n > 1
        error("$model_type requires univariate data, got $n variables")
    end
    if T_obs < 10
        error("insufficient observations ($T_obs) for $model_type estimation")
    end
    nothing
end
# Real validate_for_model(::AbstractMacroData, ::Symbol) covers the other containers.
validate_for_model(d::PanelData, model_type::Symbol) =
    validate_for_model(TimeSeriesData(d.data; varnames=d.varnames), model_type)
validate_for_model(d::CrossSectionData, model_type::Symbol) =
    validate_for_model(TimeSeriesData(d.data; varnames=d.varnames), model_type)

function apply_filter(y::AbstractVector, method::Symbol; kwargs...)
    if method == :hp
        hp_filter(y; lambda=get(kwargs, :lambda, 1600.0))
    elseif method == :hamilton
        hamilton_filter(y; h=get(kwargs, :horizon, 8), p=get(kwargs, :lags, 4))
    elseif method == :bn
        beveridge_nelson(y)
    elseif method == :bk
        baxter_king(y; pl=get(kwargs, :pl, 6), pu=get(kwargs, :pu, 32), K=get(kwargs, :K, 12))
    elseif method == :bhp
        boosted_hp(y; lambda=get(kwargs, :lambda, 1600.0))
    else
        error("unknown filter method: $method")
    end
end

# ─── Plot Support ───────────────────────────────────────────

struct PlotOutput
    html::String
end

plot_result(x; kwargs...) = PlotOutput("<html><body>mock plot for $(typeof(x))</body></html>")
save_plot(p::PlotOutput, path::String) = (write(path, p.html); path)
display_plot(p::PlotOutput) = nothing  # no-op in tests

export PlotOutput, plot_result, save_plot, display_plot

# ─── Data Balance & Dates ────────────────────────────────────

function balance_panel(ts::TimeSeriesData; method::Symbol=:dfm, r::Int=3, p::Int=2)
    return ts  # mock returns unchanged
end

set_dates!(ts::TimeSeriesData, dt::AbstractVector{<:AbstractString}) = ts
dates(ts::TimeSeriesData) = String[]

export balance_panel, set_dates!, dates

# ─── Nowcast Types & Functions ──────────────────────────────

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
    # MEMs 0.7.2/#602 additions (real field names/semantics: theta_cross is NaN under
    # :conjugate, where it is not a free parameter)
    converged::Bool; theta_cross::T; prior::Symbol
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

function nowcast_dfm(Y::AbstractMatrix, nM::Int, nQ::Int; r=2, p=1, idio=:ar1, blocks=nothing, max_iter=100, thresh=1e-4)
    T_obs, N = size(Y)
    sd = r * p
    NowcastDFM{Float64}(copy(Y), randn(T_obs, sd), randn(N, sd), randn(sd, sd),
        Matrix{Float64}(I(sd)), Matrix{Float64}(I(N)), zeros(N), ones(N),
        zeros(sd), Matrix{Float64}(I(sd)), r, p, ones(Int, N, 1), -100.0, 50, nM, nQ, idio, copy(Y))
end

# Mirror the real kwarg surface (bvar_nowcast.jl) — no `kwargs...` catch-all: a
# swallowed kwarg is exactly how a real-MEMs MethodError hides behind a green suite.
function nowcast_bvar(Y::AbstractMatrix, nM::Int, nQ::Int; lags=5, thresh=1e-6,
                      max_iter=nothing, lambda0=0.2, theta0=1.0, miu0=1.0, alpha0=2.0,
                      prior::Symbol=:conjugate,
                      theta_cross0::Union{Real,Nothing}=nothing)
    T_obs, N = size(Y)
    N == nM + nQ || throw(ArgumentError("nM ($nM) + nQ ($nQ) must equal number of columns ($N)"))
    lags >= 1 || throw(ArgumentError("lags must be >= 1, got $lags"))
    prior in (:conjugate, :litterman) ||
        throw(ArgumentError("prior must be :conjugate or :litterman, got :$prior"))
    prior == :conjugate && theta_cross0 !== nothing &&
        throw(ArgumentError("theta_cross is not a parameter of the conjugate prior"))
    tc = prior == :litterman ? Float64(something(theta_cross0, 1.0)) : NaN
    NowcastBVAR{Float64}(copy(Y), randn(N*lags+1, N), Matrix{Float64}(I(N)),
        lambda0, theta0, miu0, alpha0, lags, -100.0, nM, nQ, copy(Y),
        true, tc, prior)
end

function nowcast_bridge(Y::AbstractMatrix, nM::Int, nQ::Int; lagM=1, lagQ=1, lagY=1)
    T_obs, N = size(Y)
    nQ_act = max(nQ, 1)
    NowcastBridge{Float64}(copy(Y), randn(nQ_act), randn(nQ_act, max(nM, 1)),
        max(nM, 1), [randn(3) for _ in 1:max(nM, 1)], nM, nQ, lagM, lagQ, lagY, copy(Y))
end

function nowcast(model::AbstractNowcastModel; target_var=nothing)
    idx = isnothing(target_var) ? size(model.data, 2) : target_var
    NowcastResult{Float64}(model, model.X_sm, idx, 1.5, 1.2, :dfm)
end

function nowcast_news(X_new, X_old, model::AbstractNowcastModel, target_period; target_var=size(X_new, 2), groups=nothing)
    # Validate like real (nowcast/news.jl): a vintage differs by which cells are
    # filled in, not by shape. Without this the CLI's data/shape mapping is untested.
    T_obs, N = size(X_new)
    size(X_old) == (T_obs, N) || throw(ArgumentError("X_new and X_old must have same size"))
    1 <= target_period <= T_obs || throw(ArgumentError("target_period out of range"))
    1 <= target_var <= N || throw(ArgumentError("target_var out of range"))
    NowcastNews{Float64}(1.0, 1.5, randn(N), 0.1, 0.05,
        isnothing(groups) ? randn(1) : randn(length(unique(groups))),
        ["var$i" for i in 1:N])
end

function forecast(model::AbstractNowcastModel, h::Int; target_var=nothing)
    N = size(model.data, 2)
    randn(h, N)
end

export AbstractNowcastModel, NowcastDFM, NowcastBVAR, NowcastBridge, NowcastResult, NowcastNews
export nowcast_dfm, nowcast_bvar, nowcast_bridge, nowcast, nowcast_news

export TimeSeriesData, CrossSectionData, DataDiagnostic, DataSummary
export load_example, to_matrix, varnames, frequency, desc, vardesc, nobs, nvars
export describe_data, panel_summary, diagnose, fix, apply_tcode, validate_for_model, apply_filter

# ─── DSGE Types (MEMs 0.9.0 ModelSpec; ModelSpec/HAModelSpec are gone) ──

abstract type AbstractDSGEModel end
abstract type AbstractAgentSystem{T<:AbstractFloat} end
const NoAgents = NamedTuple{(), Tuple{}}

struct HouseholdSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    individual::Any
    income::Any
    grid::Any
    aggregation::Any
    het_params::Dict{Symbol,T}
    n_assets::Int
    n_income::Int
    model::Symbol
    distribution::Symbol
end

function HouseholdSystem{T}(; model::Symbol=:aiyagari, distribution::Symbol=:young,
                             n_assets::Int=50, n_income::Int=2,
                             het_params=Dict{Symbol,T}(:alpha => T(0.36), :delta => T(0.025))) where {T<:AbstractFloat}
    HouseholdSystem{T}(nothing, nothing, nothing, Pair{Symbol,Function}[],
                       het_params, n_assets, n_income, model, distribution)
end
HouseholdSystem(; kwargs...) = HouseholdSystem{Float64}(; kwargs...)

struct ModelSpec{T<:AbstractFloat, A<:NamedTuple}
    endog::Vector{Symbol}; exog::Vector{Symbol}; params::Vector{Symbol}
    param_values::Dict{Symbol,T}; n_endog::Int; n_exog::Int; n_params::Int
    varnames::Vector{String}; steady_state::Vector{T}
    linear::Bool
    # W12/#114: real's augmentation bookkeeping. `dsge moments` needs both to filter the
    # moment labels down to the ORIGINAL variables the way upstream filters the matrices —
    # a mislabelled moment table otherwise. Added because a handler consumes them, per the
    # mock-surface rule (fields are added on demand, never eagerly).
    augmented::Bool
    original_endog::Vector{Symbol}
    bellman_utility::Any
    bellman_beta::Any
    bellman_consumption::Union{Nothing,Symbol}
    bellman_controls::Vector{Symbol}
    agents::A
end
function ModelSpec(; n_endog=3, n_exog=1, linear::Bool=false,
                   endog_names=nothing, exog_names=nothing,
                   params=nothing, param_values=nothing,
                   augmented::Bool=false, original_endog=nothing,
                   bellman_utility=nothing, bellman_beta=nothing,
                   bellman_consumption=nothing,
                   bellman_controls::Vector{Symbol}=Symbol[],
                   agents::NamedTuple=NamedTuple(), kwargs...)
    endog = endog_names === nothing ? [Symbol("y$i") for i in 1:n_endog] :
            Symbol[Symbol(v) for v in endog_names]
    exog = exog_names === nothing ? [Symbol("e$i") for i in 1:n_exog] :
           Symbol[Symbol(v) for v in exog_names]
    n_endog = length(endog)
    n_exog = length(exog)
    ps = params === nothing ? [:alpha, :beta, :delta] : Symbol[Symbol(p) for p in params]
    pv = param_values === nothing ?
        Dict{Symbol,Float64}(p => 0.5 for p in ps) :
        Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in param_values)
    # Keep the historical defaults when the block declared no parameters, so existing
    # fixtures that rely on alpha/beta/delta are unaffected.
    if params === nothing
        pv = Dict{Symbol,Float64}(:alpha => 0.33, :beta => 0.99, :delta => 0.025)
    end
    varnames = String[String(v) for v in endog]
    ss = zeros(Float64, n_endog)
    orig = original_endog === nothing ? endog : Symbol[Symbol(v) for v in original_endog]
    A = typeof(agents)
    ModelSpec{Float64, A}(endog, exog, ps, pv, n_endog, n_exog, length(ps),
                          varnames, ss, linear, augmented, orig,
                          bellman_utility, bellman_beta, bellman_consumption,
                          bellman_controls, agents)
end

has_kind(spec::ModelSpec, ::Type{S}) where {S} =
    any(v -> v isa S, values(spec.agents))
agents_of(spec::ModelSpec, ::Type{S}) where {S} =
    (v for v in values(spec.agents) if v isa S)

# ── Mock @dsge macro (mirrors real MEMs' @dsge; C051/RA-DSGE loader) ────────
# The CLI loads RA DSGE models by evaluating an `@dsge begin … end` block (from a .jl
# file, or synthesized from TOML). Real MEMs parses the block into residual functions;
# the mock only needs the shape, so it counts endogenous/exogenous names and delegates
# to the keyword `ModelSpec` constructor. `linear: true` inside the block is honoured.
# Block line ASTs (see the real declaration syntax):
#   `endogenous: Y, C` → Expr(:tuple, Expr(:call, :(:), :endogenous, :Y), :C)
#   `exogenous: e`      → Expr(:call, :(:), :exogenous, :e)
#   `linear: true`      → Expr(:call, :(:), :linear, true)
# Real never accepted `variables:`/`shocks:` (error at 0.9.0); `E[t](...)` errors.
function _mock_dsge_extract(block, kw::Symbol)
    (block isa Expr && block.head === :block) || return Any[]
    for arg in block.args
        arg isa Expr || continue
        if arg.head === :call && length(arg.args) == 3 && arg.args[1] === :(:) && arg.args[2] === kw
            return Any[arg.args[3]]
        end
        if arg.head === :tuple && !isempty(arg.args)
            f = arg.args[1]
            if f isa Expr && f.head === :call && length(f.args) == 3 &&
               f.args[1] === :(:) && f.args[2] === kw
                return vcat(Any[f.args[3]], arg.args[2:end])
            end
        end
    end
    return Any[]
end

"""Parse a `parameters: a = 1.0, b = 2.0` declaration into (names, values).

Real MEMs records the declared names on `spec.params`, and handlers validate user input
against them (`dsge determinacy-map` rejects an unknown swept parameter). A mock that
always reported `alpha, beta, delta` would make that guard untestable and would diverge
from real on the one field the guard reads.

`parameters: a = 1.0, b = 2.0` does NOT parse like the other declarations — `=` binds
looser than `,` there, so Julia produces a nest like

  Expr(:(=), :(parameters:a), quote (1.0, b) = 2.0 end)

rather than a flat tuple. Rather than reverse-engineer that shape (which changes with the
number of parameters), collect every Symbol in the subtree in source order, dropping the
`parameters` keyword and the `:` operator. Values are not recovered — the mock only needs
the NAMES, which is what handlers validate against.
"""
function _mock_dsge_collect_syms!(out::Vector{Symbol}, x)
    if x isa Symbol
        (x === :parameters || x === :(:)) || push!(out, x)
    elseif x isa Expr
        for a in x.args
            _mock_dsge_collect_syms!(out, a)
        end
    end
    return out
end

function _mock_dsge_params(block)
    (block isa Expr && block.head === :block) || return Symbol[], Dict{Symbol,Float64}()
    for arg in block.args
        arg isa Expr || continue
        syms = _mock_dsge_collect_syms!(Symbol[], arg)
        # The `parameters` line is the one whose subtree mentions the keyword.
        occursin("parameters", string(arg)) || continue
        startswith(strip(string(arg)), "parameters") || continue
        names = unique(syms)
        isempty(names) && continue
        return names, Dict{Symbol,Float64}(n => 0.5 for n in names)
    end
    return Symbol[], Dict{Symbol,Float64}()
end

macro dsge(block)
    occursin("E[t]", string(block)) && error(
        "@dsge: E[t](...) was removed; write the lead directly (x[t+1] is E_t x_{t+1})")
    !isempty(_mock_dsge_extract(block, :variables)) &&
        error("@dsge: unrecognized declaration :variables")
    !isempty(_mock_dsge_extract(block, :shocks)) &&
        error("@dsge: unrecognized declaration :shocks")
    en = Symbol[v for v in _mock_dsge_extract(block, :endogenous) if v isa Symbol]
    xn = Symbol[v for v in _mock_dsge_extract(block, :exogenous) if v isa Symbol]
    het = _mock_dsge_extract(block, :heterogeneous)
    idio = _mock_dsge_extract(block, :idiosyncratic)
    agg = _mock_dsge_extract(block, :aggregation)
    is_ha = !isempty(het) || !isempty(idio) || !isempty(agg)
    ne = max(length(en), 1)
    nx = max(length(xn), is_ha ? 0 : 1)
    pnames, pvals = _mock_dsge_params(block)
    lin_names = _mock_dsge_extract(block, :linear)
    is_linear = !isempty(lin_names) && lin_names[1] === true
    util_decl = _mock_dsge_extract(block, :utility)
    beta_decl = _mock_dsge_extract(block, :beta)
    ctrl_decl = Symbol[v for v in _mock_dsge_extract(block, :controls) if v isa Symbol]
    # Quote: real @dsge stores the utility form unevaluated, and splicing it
    # bare evaluates `C` in the caller's scope (UndefVarError). Latent until
    # the first T1/T2 VFI-success test (only the config-error path existed).
    bu = isempty(util_decl) ? nothing :
        (util_decl[1] === :log ? log : QuoteNode(util_decl[1]))
    bb = isempty(beta_decl) ? nothing : beta_decl[1]
    bc = isempty(ctrl_decl) ? nothing : ctrl_decl[1]
    agents = if is_ha
        (household = HouseholdSystem(; model=:huggett),)
    else
        NamedTuple()
    end
    # Splice the constructor object so the expansion needs nothing in the caller's scope.
    return :($(ModelSpec)(; n_endog=$ne, n_exog=$(max(nx, is_ha ? 0 : 1)), linear=$is_linear,
                          endog_names=$(isempty(en) ? nothing : en),
                          exog_names=$(isempty(xn) ? nothing : xn),
                          params=$(isempty(pnames) ? nothing : pnames),
                          param_values=$(isempty(pvals) ? nothing : pvals),
                          bellman_utility=$bu, bellman_beta=$(bb === nothing ? nothing : QuoteNode(bb)),
                          bellman_consumption=$(bc === nothing ? nothing : QuoteNode(bc)),
                          bellman_controls=$(ctrl_decl),
                          agents=$agents))
end

struct LinearDSGE{T<:Real}
    Gamma0::Matrix{T}; Gamma1::Matrix{T}; C::Vector{T}; Psi::Matrix{T}; Pi::Matrix{T}
    spec::ModelSpec{T}
end

struct DSGESolution{T<:Real}
    G1::Matrix{T}; impact::Matrix{T}; C_sol::Vector{T}; eu::Vector{Int}
    method::Symbol; eigenvalues::Vector{Complex{T}}; spec::ModelSpec{T}; linear::LinearDSGE{T}
end

struct PerturbationSolution{T<:Real}
    order::Int; gx::Matrix{T}; hx::Matrix{T}
    gxx::Union{Nothing,Array{T,3}}; hxx::Union{Nothing,Array{T,3}}
    gσσ::Union{Nothing,Vector{T}}; hσσ::Union{Nothing,Vector{T}}
    eta::Matrix{T}; steady_state::Vector{T}
    state_indices::Vector{Int}; control_indices::Vector{Int}
    eu::Vector{Int}; method::Symbol; spec::ModelSpec{T}; linear::LinearDSGE{T}
end

struct ProjectionSolution{T<:Real}
    coefficients::Matrix{T}; state_bounds::Matrix{T}; grid_type::Symbol; degree::Int
    residual_norm::T; converged::Bool; iterations::Int; method::Symbol
    spec::ModelSpec{T}; linear::LinearDSGE{T}; steady_state::Vector{T}
    state_indices::Vector{Int}; control_indices::Vector{Int}
    value_fn::Matrix{T}
    collocation_nodes::Matrix{T}
    value_coefficients::Vector{T}
    # Real field (MEMs ≥ 0.9.0): n_blocks × nx level set, 0×0 off Smolyak.
    smolyak_levels::Matrix{Int}
end

struct PerfectForesightPath{T<:Real}
    path::Matrix{T}; deviations::Matrix{T}; converged::Bool; iterations::Int
    spec::ModelSpec{T}
end

struct DSGEEstimation{T<:Real} <: AbstractDSGEModel
    theta::Vector{T}; vcov::Matrix{T}; param_names::Vector{String}; method::Symbol
    J_stat::T; J_pvalue::T; converged::Bool; spec::ModelSpec{T}
end

struct OccBinConstraint{T<:AbstractFloat}
    expr::Expr
    variable::Symbol
    bound::T
    direction::Symbol
    bind_expr::Expr
end

struct VariableBound{T<:AbstractFloat}
    var_name::Symbol
    lower::Union{T, Nothing}
    upper::Union{T, Nothing}
end

struct NonlinearConstraint{T<:Real}
    fn::Function
    label::String
end

function nonlinear_constraint(fn::Function; label::String="")
    NonlinearConstraint{Float64}(fn, label)
end

struct OccBinSolution{T<:Real}
    linear_path::Matrix{T}; piecewise_path::Matrix{T}; steady_state::Vector{T}
    regime_history::Vector{Int}; converged::Bool; iterations::Int
    spec::ModelSpec{T}; varnames::Vector{String}
    constraints::Vector{OccBinConstraint{T}}
end

struct OccBinIRF{T<:Real}
    linear::Array{T,3}; piecewise::Array{T,3}; regime_history::Vector{Int}
    varnames::Vector{String}; shock_name::String
end

# ─── DSGE Mock Helpers & Functions ───────────────────────────

function _mock_linear(spec::ModelSpec{T}) where T
    n = spec.n_endog
    ne = spec.n_exog
    Gamma0 = Matrix{T}(I(n))
    Gamma1 = Matrix{T}(I(n)) * T(0.5)
    C_vec = zeros(T, n)
    Psi = zeros(T, n, ne)
    for i in 1:min(n, ne); Psi[i, i] = T(1.0); end
    Pi_mat = zeros(T, n, n)
    LinearDSGE{T}(Gamma0, Gamma1, C_vec, Psi, Pi_mat, spec)
end

function _mock_solution(spec::ModelSpec{T}; method=:gensys) where T
    n = spec.n_endog
    ne = spec.n_exog
    ld = _mock_linear(spec)
    G1 = Matrix{T}(I(n)) * T(0.5)
    impact = zeros(T, n, ne)
    for i in 1:min(n, ne); impact[i, i] = T(1.0); end
    C_sol = zeros(T, n)
    eu = [1, 1]
    eigs = [complex(T(0.5), T(0.1)), complex(T(0.5), T(-0.1)), complex(T(0.3), T(0.0))]
    DSGESolution{T}(G1, impact, C_sol, eu, method, eigs[1:min(n, length(eigs))], spec, ld)
end

function compute_steady_state(spec::ModelSpec; solver=nothing, constraints=[], kwargs...)
    has_kind(spec, HouseholdSystem) && return _mock_ha_ss(spec; kwargs...)
    spec
end

function linearize(spec::ModelSpec)
    _mock_linear(spec)
end

function solve(spec::ModelSpec{T}; method=:gensys, order=1, degree=5, grid=:auto, solver=nothing, constraints=[], ss=nothing, n_reduced::Int=10, T_horizon::Int=300, kwargs...) where T
    has_kind(spec, HouseholdSystem) && return _mock_ha_solve(spec; method=method, ss=ss, n_reduced=n_reduced, T_horizon=T_horizon)
    n = spec.n_endog
    ne = spec.n_exog
    ld = _mock_linear(spec)
    if method == :perturbation
        n_states = max(1, n ÷ 2)
        n_controls = n - n_states
        # Real gx/hx are ny×nv and nx×nv where v = [states; shocks] (the Stage-14 #368
        # augmented layout, MEMs ≥0.7.2). A state-only mock gx hid a broken
        # `dsge solve --method perturbation` render branch for a whole release line.
        nv = n_states + ne
        gx = ones(T, n_controls, nv) * T(0.1)
        hx = hcat(Matrix{T}(I(n_states)) * T(0.5), zeros(T, n_states, ne))
        eta = zeros(T, nv, ne)
        for i in 1:ne; eta[n_states + i, i] = T(1.0); end
        gxx = order >= 2 ? zeros(T, n_controls, nv, nv) : nothing
        hxx = order >= 2 ? zeros(T, n_states, nv, nv) : nothing
        gσσ = order >= 2 ? zeros(T, n_controls) : nothing
        hσσ = order >= 2 ? zeros(T, n_states) : nothing
        ss = zeros(T, n)
        state_idx = collect(1:n_states)
        control_idx = collect(n_states+1:n)
        return PerturbationSolution{T}(order, gx, hx, gxx, hxx, gσσ, hσσ, eta, ss,
            state_idx, control_idx, [1, 1], :perturbation, spec, ld)
    elseif method in (:projection, :pfi, :vfi)
        n_states = max(1, n ÷ 2)
        n_controls = n - n_states
        coeffs = ones(T, n_controls, degree + 1) * T(0.1)
        bounds = hcat(fill(T(-2.0), n_states), fill(T(2.0), n_states))
        ss = zeros(T, n)
        state_idx = collect(1:n_states)
        control_idx = collect(n_states+1:n)
        # VFI-only mirrors of the real 0.9.5 validation (exit-class parity).
        # n_ctrl follows the real rule: explicit bellman_controls, else the
        # non-state endogenous (the mock's own control block). Unknown
        # optimizer symbols are CLI-unreachable (parser choices + map guard).
        opt = get(kwargs, :optimizer, :auto)
        mu = get(kwargs, :smolyak_mu, nothing)
        if method === :vfi
            nctrl = isempty(spec.bellman_controls) ? n_controls :
                length(spec.bellman_controls)
            opt === :grid1d && nctrl != 1 && throw(ArgumentError(
                "optimizer=:grid1d supports one continuous control (got $nctrl)"))
            gres = grid === :auto ? (n_states <= 3 ? :tensor : :smolyak) : grid
            if gres === :smolyak && mu !== nothing
                muv = mu isa Integer ? fill(Int(mu), n_states) : collect(Int, mu)
                ((mu isa Integer && mu >= 0) || all(>=(0), muv)) ||
                    throw(ArgumentError("smolyak_mu must be ≥ 0"))
                length(muv) == n_states || throw(ArgumentError(
                    "smolyak_mu must be a scalar or a vector of length nx=$n_states"))
            end
        else
            gres = grid == :auto ? :chebyshev : grid
        end
        vf = method === :vfi ? reshape(T[T(i) for i in 1:5], 5, 1) : zeros(T, 0, 0)
        nodes = method === :vfi ? hcat(range(T(-1), T(1); length=5)) : zeros(T, 0, 0)
        vc = method === :vfi ? T[0.1, 0.2, 0.3] : T[]
        lv = (method === :vfi && gres === :smolyak) ? fill(2, 3, n_states) :
            zeros(Int, 0, 0)
        return ProjectionSolution{T}(coeffs, bounds, gres, degree,
            T(1e-8), true, 50, method, spec, ld, ss, state_idx, control_idx,
            vf, nodes, vc, lv)
    else
        return _mock_solution(spec; method=method)
    end
end

function gensys(Γ0, Γ1, C, Ψ, Π)
    _mock_solution(ModelSpec())
end

function blanchard_kahn(ld::LinearDSGE, spec::ModelSpec)
    _mock_solution(spec; method=:blanchard_kahn)
end

function klein(Γ0, Γ1, C, Ψ, n_pre)
    _mock_solution(ModelSpec(); method=:klein)
end

function perturbation_solver(spec::ModelSpec; order=1)
    solve(spec; method=:perturbation, order=order)
end

function collocation_solver(spec::ModelSpec; degree=5, kwargs...)
    solve(spec; method=:projection, degree=degree)
end

function pfi_solver(spec::ModelSpec; kwargs...)
    solve(spec; method=:pfi)
end

function perfect_foresight(spec::ModelSpec{T}; shock_path=nothing, T_periods=100, solver=nothing, constraints=[], kwargs...) where T
    if shock_path !== nothing
        size(shock_path, 1) == T_periods || throw(AssertionError(
            "shock_path must have T_periods=$T_periods rows, got $(size(shock_path, 1))"))
    end
    n = spec.n_endog
    path = zeros(T, T_periods, n)
    devs = zeros(T, T_periods, n)
    PerfectForesightPath{T}(path, devs, true, 25, spec)
end

function _mock_occbin_sol(spec::ModelSpec{T}, cons; shock_path=nothing, nperiods::Int=40) where T
    n = spec.n_endog
    np = shock_path === nothing ? nperiods : size(shock_path, 1)
    lp = zeros(T, np, n)
    pp = zeros(T, np, n)
    ss = zeros(T, n)
    regimes = ones(Int, np)
    OccBinSolution{T}(lp, pp, ss, regimes, true, 15, spec, spec.varnames, cons)
end

function occbin_solve(spec::ModelSpec{T}, constraint::OccBinConstraint;
                      shock_path=zeros(T, 40, spec.n_exog), nperiods::Int=size(shock_path, 1),
                      maxiter::Int=100, kwargs...) where T
    _mock_occbin_sol(spec, [constraint]; shock_path=shock_path, nperiods=nperiods)
end
function occbin_solve(spec::ModelSpec{T}, c1::OccBinConstraint, c2::OccBinConstraint;
                      shock_path=zeros(T, 40, spec.n_exog), nperiods::Int=size(shock_path, 1),
                      kwargs...) where T
    _mock_occbin_sol(spec, [c1, c2]; shock_path=shock_path, nperiods=nperiods)
end
function occbin_solve(spec::ModelSpec{T}, expr::Expr; kwargs...) where T
    occbin_solve(spec, parse_constraint(expr, spec); kwargs...)
end

function occbin_irf(spec::ModelSpec{T}, constraint::OccBinConstraint, shock_idx::Int, horizon::Int;
                    magnitude::Real=one(T), maxiter::Int=100, kwargs...) where T
    n = spec.n_endog
    ne = spec.n_exog
    lin = zeros(T, horizon + 1, n, ne)
    pw = zeros(T, horizon + 1, n, ne)
    for h in 0:horizon
        for v in 1:n
            lin[h+1, v, min(shock_idx, ne)] = T(magnitude) * T(0.9)^h
            pw[h+1, v, min(shock_idx, ne)] = T(magnitude) * T(0.85)^h
        end
    end
    OccBinIRF{T}(lin, pw, ones(Int, horizon + 1), spec.varnames, "shock$shock_idx")
end
function occbin_irf(spec::ModelSpec{T}, c1::OccBinConstraint, c2::OccBinConstraint,
                    shock_idx::Int, horizon::Int; magnitude::Real=one(T), kwargs...) where T
    occbin_irf(spec, c1, shock_idx, horizon; magnitude=magnitude, kwargs...)
end

function parse_constraint(expr::Expr, spec::ModelSpec)
    OccBinConstraint{Float64}(expr, :x, 0.0, :geq, :(x[t] = 0.0))
end

function variable_bound(var::Symbol; lower::Union{Real,Nothing}=nothing,
                         upper::Union{Real,Nothing}=nothing)
    lower === nothing && upper === nothing &&
        throw(ArgumentError("At least one of lower or upper must be specified"))
    lo = lower === nothing ? nothing : Float64(lower)
    hi = upper === nothing ? nothing : Float64(upper)
    VariableBound{Float64}(var, lo, hi)
end

function estimate_dsge(spec::ModelSpec{T}, data, param_names; method=:irf_matching, kwargs...) where T
    np = length(param_names)
    theta = ones(T, np) * T(0.5)
    vcov_mat = Matrix{T}(I(np)) * T(0.01)
    DSGEEstimation{T}(theta, vcov_mat, String.(param_names), method,
                      T(2.5), T(0.65), true, spec)
end

function simulate(sol::DSGESolution{T}, T_periods::Int; kwargs...) where T
    randn(T, T_periods, sol.spec.n_endog)
end
function simulate(sol::PerturbationSolution{T}, T_periods::Int; kwargs...) where T
    randn(T, T_periods, sol.spec.n_endog)
end
function simulate(sol::ProjectionSolution{T}, T_periods::Int; kwargs...) where T
    # Real simulate (simulation.jl:206) takes shock_draws/seed/rng only —
    # reject antithetic like real (MethodError → internal/error both tiers).
    # Regression net for the W1 dsge-simulate fix: the CLI must never pass
    # antithetic here, so this fires only if the fix regresses.
    haskey(kwargs, :antithetic) && throw(MethodError(simulate, (sol, T_periods)))
    randn(T, T_periods, sol.spec.n_endog)
end

function irf(sol::DSGESolution{T}, horizon::Int; kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    vals = zeros(T, horizon + 1, n, ne)
    for h in 0:horizon, v in 1:n, s in 1:ne
        vals[h+1, v, s] = T(0.1) * T(0.9)^h
    end
    ImpulseResponse(vals, nothing, nothing, horizon,
        sol.spec.varnames, ["shock$i" for i in 1:ne], :dsge)
end
function irf(sol::PerturbationSolution{T}, horizon::Int; kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    vals = zeros(T, horizon + 1, n, ne)
    for h in 0:horizon, v in 1:n, s in 1:ne
        vals[h+1, v, s] = T(0.1) * T(0.9)^h
    end
    ImpulseResponse(vals, nothing, nothing, horizon,
        sol.spec.varnames, ["shock$i" for i in 1:ne], :perturbation)
end
function irf(sol::ProjectionSolution{T}, horizon::Int; kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    vals = zeros(T, horizon + 1, n, ne)
    for h in 0:horizon, v in 1:n, s in 1:ne
        vals[h+1, v, s] = T(0.1) * T(0.9)^h
    end
    ImpulseResponse(vals, nothing, nothing, horizon,
        sol.spec.varnames, ["shock$i" for i in 1:ne], :projection)
end

function fevd(sol::DSGESolution{T}, horizon::Int; kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    props = ones(T, n, ne, horizon) / T(ne)
    FEVD(props, props)
end
function fevd(sol::PerturbationSolution{T}, horizon::Int; unconditional::Bool=false, kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    # Unconditional (order≥2) returns asymptotic H=1 proportions (MEMs Andreasen path)
    n_h = (unconditional && sol.order >= 2) ? 1 : horizon
    props = ones(T, n, ne, n_h) / T(ne)
    FEVD(props, props)
end

function is_determined(sol::Union{DSGESolution,PerturbationSolution,ProjectionSolution})
    true
end

function is_stable(sol::Union{DSGESolution,PerturbationSolution,ProjectionSolution})
    true
end

function nshocks(sol::Union{DSGESolution,PerturbationSolution,ProjectionSolution})
    sol.spec.n_exog
end

export AbstractDSGEModel, AbstractAgentSystem, ModelSpec, HouseholdSystem, NoAgents
export has_kind, agents_of
export LinearDSGE, DSGESolution, PerturbationSolution
export ProjectionSolution, PerfectForesightPath, DSGEEstimation
export OccBinConstraint, VariableBound, NonlinearConstraint, nonlinear_constraint, OccBinSolution, OccBinIRF
export compute_steady_state, linearize, solve, gensys, blanchard_kahn, klein
export perturbation_solver, collocation_solver, pfi_solver
export perfect_foresight, occbin_solve, occbin_irf, parse_constraint, variable_bound
export estimate_dsge, simulate, is_determined, is_stable, nshocks
export @dsge

# ─── SMM Types & Functions ───────────────────────────────────

struct SMMModel{T<:Real}
    theta::Vector{T}; vcov::Matrix{T}; n_moments::Int; n_params::Int; n_obs::Int
    J_stat::T; J_pvalue::T; converged::Bool; sim_ratio::Int
end

struct ParameterTransform{T<:Real}
    lower::Vector{T}; upper::Vector{T}
end

# Real MEMs 0.7.0 signature: estimate_smm(simulator_fn, moments_fn, theta0, data; ...).
# (Was 3-arg here, which hid the real command being broken — the exact panel/DiD-class
# blind spot. Keep this in lock-step with real so the T1/T2 mock can't mask arity drift.)
function estimate_smm(simulator_fn, moments_fn, theta0, data;
                      weighting=:two_step, sim_ratio=5, burn=100,
                      contributions_fn=nothing, bounds=nothing,
                      rng=nothing, kwargs...)
    np = length(theta0)
    m_data = moments_fn(data)
    nm = max(length(m_data), np)
    theta = Float64.(collect(theta0))
    vcov_mat = Matrix{Float64}(I(np)) * 0.01
    n_obs = size(data, 1)
    SMMModel{Float64}(theta, vcov_mat, nm, np, n_obs, 2.0, 0.7, true, sim_ratio)
end

function autocovariance_moments(data; lags=1)
    zeros(Float64, size(data, 2) * (lags + 1))
end

# Per-observation moment contributions (n × q); column-mean equals the moment vector.
function autocovariance_moment_contributions(data; lags=1)
    zeros(Float64, size(data, 1), size(data, 2) * (lags + 1))
end

# Real order is to_unconstrained(pt::ParameterTransform, theta); mock is order-agnostic.
to_unconstrained(t::ParameterTransform, x) = x
to_constrained(t::ParameterTransform, x) = x
transform_jacobian(t::ParameterTransform, x) = Matrix{Float64}(I(length(x)))

export SMMModel, ParameterTransform
export estimate_smm, autocovariance_moments, autocovariance_moment_contributions
export to_unconstrained, to_constrained, transform_jacobian

# ─── Systems: SUR & 3SLS (C063) ──────────────────────────────
# Fields match the real MEMs SURModel/ThreeSLSModel (system/types.jl). estimate_*
# compute genuine per-equation OLS so T1/T2 exercises the handler's table-shaping.
struct SURModel{T<:AbstractFloat}
    eqnames::Vector{String}
    varnames::Vector{Vector{String}}
    betas::Vector{Vector{T}}
    ses::Vector{Vector{T}}
    vcov_mat::Matrix{T}
    Sigma::Matrix{T}
    residuals::Vector{Vector{T}}
    fitted::Vector{Vector{T}}
    nobs::Int
    det_sigma::T
    mcelroy_r2::T
    loglik::T
    iterations::Int
    iterated::Bool
    restricted::Bool
end

struct ThreeSLSModel{T<:AbstractFloat}
    eqnames::Vector{String}
    varnames::Vector{Vector{String}}
    betas::Vector{Vector{T}}
    ses::Vector{Vector{T}}
    vcov_mat::Matrix{T}
    Sigma::Matrix{T}
    residuals::Vector{Vector{T}}
    fitted::Vector{Vector{T}}
    nobs::Int
    det_sigma::T
    mcelroy_r2::T
    n_instruments::Vector{Int}
end

# Per-equation OLS point estimates + textbook SEs (a stand-in for FGLS/3SLS; enough
# for T1/T2 to check the handler renders a well-formed per-(equation,term) table).
function _mock_system_fit(eqs::AbstractVector)
    M = length(eqs)
    betas = Vector{Vector{Float64}}(undef, M); ses = Vector{Vector{Float64}}(undef, M)
    vns = Vector{Vector{String}}(undef, M)
    resids = Vector{Vector{Float64}}(undef, M); fitted = Vector{Vector{Float64}}(undef, M)
    Tn = 0
    for (j, e) in enumerate(eqs)
        y = Float64.(collect(e[1])); X = Matrix{Float64}(e[2])
        Tn = length(y); k = size(X, 2)
        vns[j] = length(e) >= 3 ? String.(collect(e[3])) : ["eq$(j)_x$i" for i in 1:k]
        b = X \ y; r = y .- X * b
        s2 = sum(abs2, r) / max(Tn - k, 1)
        XtXinv = inv(X' * X)
        betas[j] = b
        ses[j] = [sqrt(abs(XtXinv[i, i]) * s2) for i in 1:k]
        resids[j] = r; fitted[j] = X * b
    end
    (betas, ses, vns, resids, fitted, Tn)
end

function estimate_sur(eqs::AbstractVector; iterate::Bool=false, tol::Real=1e-8,
                      maxiter::Int=100, restrict=nothing, eqnames=nothing)
    betas, ses, vns, resids, fitted, Tn = _mock_system_fit(eqs)
    M = length(eqs)
    enames = eqnames === nothing ? ["eq$(j)" for j in 1:M] : String.(collect(eqnames))
    K = sum(length(b) for b in betas)
    SURModel{Float64}(enames, vns, betas, ses, Matrix{Float64}(I(K)) .* 0.01,
                      Matrix{Float64}(I(M)), resids, fitted, Tn, 1.0, 0.8, -100.0,
                      iterate ? 3 : 1, iterate, restrict !== nothing)
end

function estimate_3sls(eqs::AbstractVector, Z; instruments::Symbol=:common, eqnames=nothing)
    instruments in (:common, :perequation) ||
        throw(ArgumentError("instruments must be :common or :perequation; got :$instruments"))
    betas, ses, vns, resids, fitted, Tn = _mock_system_fit(eqs)
    M = length(eqs)
    enames = eqnames === nothing ? ["eq$(j)" for j in 1:M] : String.(collect(eqnames))
    K = sum(length(b) for b in betas)
    ninstr = Z isa AbstractVector ? [size(Matrix{Float64}(z), 2) for z in Z] :
                                    fill(size(Matrix{Float64}(Z), 2), M)
    ThreeSLSModel{Float64}(enames, vns, betas, ses, Matrix{Float64}(I(K)) .* 0.01,
                           Matrix{Float64}(I(M)), resids, fitted, Tn, 1.0, 0.8, ninstr)
end

export SURModel, ThreeSLSModel, estimate_sur, estimate_3sls

# ─── Forecast evaluation & combination: fceval (C072) ────────
# Fields match the real MEMs fceval/types.jl. The estimate functions compute
# genuine simple metrics (RMSE=√mean(e²), OLS a/b, real combination weights) so
# T1/T2 exercises the handler's table-shaping; canned finite p-values suffice.
struct ForecastEvaluation{T<:AbstractFloat}
    models::Vector{String}
    metrics::Vector{String}
    values::Matrix{T}
    decomp::Matrix{T}
    n::Int
end

struct DMTestResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    dbar::T
    lrvar::T
    h::Int
    loss::Symbol
    hln::Bool
    alternative::Symbol
    T_obs::Int
end

struct ClarkWestResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    fbar::T
    lrvar::T
    h::Int
    alternative::Symbol
    T_obs::Int
end

struct MincerZarnowitzResult{T<:AbstractFloat}
    a::T
    b::T
    se::Vector{T}
    wald::T
    pvalue_wald::T
    fstat::T
    pvalue_f::T
    lags::Int
    kernel::Symbol
    T_obs::Int
end

struct ForecastEncompassingResult{T<:AbstractFloat}
    b1::T
    b2::T
    se_b2::T
    tstat::T
    pvalue::T
    lags::Int
    kernel::Symbol
    T_obs::Int
end

struct ForecastCombination{T<:AbstractFloat}
    weights::Vector{T}
    combined::Vector{T}
    method::Symbol
    mse::Vector{T}
    models::Vector{String}
end

const _MOCK_FCEVAL_METRICS = ["ME", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "U1", "U2"]

# Logistic approx to the standard-normal survival function P(Z > z) — for finite,
# monotone, in-(0,1) mock p-values (var/std/dot are not imported into this module).
_mock_pnorm_sf(z) = clamp(1.0 / (1.0 + exp(1.702 * z)), 0.0, 1.0)
_mock_var0(x) = mean(abs2, x .- mean(x))

function _mock_point_metrics(a::Vector{Float64}, f::Vector{Float64})
    tol = 1e-8
    e = a .- f
    me = mean(e); mae = mean(abs, e); mse = mean(abs2, e); rmse = sqrt(mse)
    mape_terms = Float64[abs(e[t] / a[t]) for t in eachindex(a) if abs(a[t]) > tol]
    mape = isempty(mape_terms) ? 0.0 : 100 * mean(mape_terms)
    smape_terms = Float64[2 * abs(e[t]) / (abs(a[t]) + abs(f[t])) for t in eachindex(a) if abs(a[t]) + abs(f[t]) > tol]
    smape = isempty(smape_terms) ? 0.0 : 100 * mean(smape_terms)
    da = diff(a)
    mase = mae / max(mean(abs, da), tol)
    u1 = rmse / max(sqrt(mean(abs2, a)) + sqrt(mean(abs2, f)), tol)
    u2 = rmse / max(sqrt(mean(abs2, da)), tol)
    vals = Float64[me, mae, rmse, mape, smape, mase, u1, u2]
    mf = mean(f); ma = mean(a)
    sf = sqrt(_mock_var0(f)); sa = sqrt(_mock_var0(a))
    bias = (mf - ma)^2 / max(mse, tol)
    varp = (sf - sa)^2 / max(mse, tol)
    covp = max(1.0 - bias - varp, 0.0)
    return vals, Float64[bias, varp, covp]
end

function forecast_evaluate(actual::AbstractVector, fc::AbstractVector;
                           seasonal_period::Int=1, insample=nothing, model_names=nothing)
    a = Float64.(collect(actual)); f = Float64.(collect(fc))
    v, d = _mock_point_metrics(a, f)
    names = model_names === nothing ? ["Model 1"] : String.(collect(model_names))
    ForecastEvaluation{Float64}(names, copy(_MOCK_FCEVAL_METRICS),
                                reshape(v, 1, :), reshape(d, 1, :), length(a))
end

function forecast_evaluate(actual::AbstractVector, fc::AbstractMatrix;
                           seasonal_period::Int=1, insample=nothing, model_names=nothing)
    a = Float64.(collect(actual)); M = size(fc, 2)
    K = length(_MOCK_FCEVAL_METRICS)
    vals = Matrix{Float64}(undef, M, K); decomp = Matrix{Float64}(undef, M, 3)
    for j in 1:M
        v, d = _mock_point_metrics(a, Float64.(collect(fc[:, j])))
        vals[j, :] = v; decomp[j, :] = d
    end
    names = model_names === nothing ? ["Model $j" for j in 1:M] : String.(collect(model_names))
    length(names) == M || throw(ArgumentError("model_names must have $M entries"))
    ForecastEvaluation{Float64}(names, copy(_MOCK_FCEVAL_METRICS), vals, decomp, length(a))
end

function diebold_mariano(e1::AbstractVector, e2::AbstractVector; h::Int=1, loss=:se,
                         hln::Bool=true, kernel::Symbol=:rectangular, alternative::Symbol=:two_sided)
    length(e1) == length(e2) || throw(DimensionMismatch("e1 and e2 must have equal length"))
    g = loss === :ad ? abs : (x -> x^2)
    d = Float64[g(Float64(e1[t])) - g(Float64(e2[t])) for t in eachindex(e1)]
    n = length(d); dbar = mean(d); V = max(_mock_var0(d), 1e-12)
    stat = dbar / sqrt(V / n)
    pval = alternative === :two_sided ? 2 * _mock_pnorm_sf(abs(stat)) :
           alternative === :greater   ? _mock_pnorm_sf(stat) :
                                         1.0 - _mock_pnorm_sf(stat)
    DMTestResult{Float64}(stat, clamp(pval, 0.0, 1.0), dbar, V, h,
                          loss isa Symbol ? loss : :custom, hln, alternative, n)
end

function clark_west(e_small::AbstractVector, e_big::AbstractVector, f_adj::AbstractVector;
                    h::Int=1, alternative::Symbol=:greater)
    n = length(e_small)
    (length(e_big) == n && length(f_adj) == n) ||
        throw(DimensionMismatch("e_small, e_big, f_adj must have equal length"))
    fhat = Float64[Float64(e_small[t])^2 - (Float64(e_big[t])^2 - Float64(f_adj[t])^2) for t in 1:n]
    fbar = mean(fhat); V = max(_mock_var0(fhat), 1e-12)
    stat = fbar / sqrt(V / n)
    pval = alternative === :two_sided ? 2 * _mock_pnorm_sf(abs(stat)) :
           alternative === :greater   ? _mock_pnorm_sf(stat) :
                                         1.0 - _mock_pnorm_sf(stat)
    ClarkWestResult{Float64}(stat, clamp(pval, 0.0, 1.0), fbar, V, h, alternative, n)
end

function mincer_zarnowitz(actual::AbstractVector, fc::AbstractVector; lags::Int=0, kernel::Symbol=:bartlett)
    y = Float64.(collect(actual)); n = length(y)
    X = hcat(ones(n), Float64.(collect(fc)))
    beta = (X' * X) \ (X' * y)
    u = y .- X * beta
    s2 = sum(abs2, u) / max(n - 2, 1)
    XtXinv = inv(X' * X)
    se = Float64[sqrt(abs(XtXinv[i, i]) * s2) for i in 1:2]
    a, b = beta[1], beta[2]
    dvec = Float64[a - 0.0, b - 1.0]
    wald = abs(dvec' * inv(XtXinv .* s2) * dvec)
    fstat = wald / 2
    pw = _mock_pnorm_sf(sqrt(max(wald, 0.0)))
    MincerZarnowitzResult{Float64}(a, b, se, wald, pw, fstat, pw, lags, kernel, n)
end

function forecast_encompassing(actual::AbstractVector, fc1::AbstractVector, fc2::AbstractVector;
                               lags::Int=0, kernel::Symbol=:bartlett)
    y = Float64.(collect(actual)); n = length(y)
    X = hcat(ones(n), Float64.(collect(fc1)), Float64.(collect(fc2)))
    beta = (X' * X) \ (X' * y)
    u = y .- X * beta
    s2 = sum(abs2, u) / max(n - 3, 1)
    XtXinv = inv(X' * X)
    se_b2 = sqrt(abs(XtXinv[3, 3]) * s2)
    b1, b2 = beta[2], beta[3]
    tstat = b2 / max(se_b2, 1e-12)
    pval = 2 * _mock_pnorm_sf(abs(tstat))
    ForecastEncompassingResult{Float64}(b1, b2, se_b2, tstat, clamp(pval, 0.0, 1.0), lags, kernel, n)
end

function combine_forecasts(F::AbstractMatrix, actual::AbstractVector; method::Symbol=:equal, model_names=nothing)
    method in (:equal, :bates_granger, :granger_ramanathan) ||
        throw(ArgumentError("method must be :equal, :bates_granger, or :granger_ramanathan; got :$method"))
    a = Float64.(collect(actual)); Fm = Matrix{Float64}(F); n, M = size(Fm)
    n == length(a) || throw(DimensionMismatch("F rows must match length(actual)"))
    mse = Float64[mean(abs2, a .- Fm[:, j]) for j in 1:M]
    w = if method === :equal
        fill(1.0 / M, M)
    elseif method === :bates_granger
        any(mse .<= 0) && throw(ArgumentError(":bates_granger requires strictly positive MSEs"))
        inv_mse = 1.0 ./ mse; inv_mse ./ sum(inv_mse)
    else
        Sigma = Fm' * Fm; c = Fm' * a; Sinv = inv(Sigma)
        Sc = Sinv * c; S1 = Sinv * ones(M)
        Sc .+ S1 .* ((1.0 - sum(Sc)) / sum(S1))
    end
    combined = Fm * w
    names = model_names === nothing ? ["Model $j" for j in 1:M] : String.(collect(model_names))
    length(names) == M || throw(ArgumentError("model_names must have $M entries"))
    ForecastCombination{Float64}(w, combined, method, mse, names)
end

export ForecastEvaluation, DMTestResult, ClarkWestResult, MincerZarnowitzResult,
       ForecastEncompassingResult, ForecastCombination
export forecast_evaluate, diebold_mariano, clark_west, mincer_zarnowitz,
       forecast_encompassing, combine_forecasts

# ─── C064a: univariate GARCH variants (igarch/cgarch/aparch/figarch/fiegarch/garch-midas) ───
# Minimal stand-ins for the MEMs 0.7.0 volatility variants (src/garch): enough fields
# for the estimate handlers' hand-built coef table + diagnostics kv. Field names/coef
# order mirror real; estimate_* compute plausible finite values from the series.
struct IGARCHModel{T<:Real}
    p::Int; q::Int; mu::T; omega::T; alpha::Vector{T}; beta::Vector{T}
    conditional_variance::Vector{T}; residuals::Vector{T}
    loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end
struct CGARCHModel{T<:Real}
    mu::T; omega::T; rho::T; phi::T; alpha::T; beta::T
    conditional_variance::Vector{T}; residuals::Vector{T}
    loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end
struct APARCHModel{T<:Real}
    p::Int; q::Int; mu::T; omega::T; alpha::Vector{T}; gamma::Vector{T}; beta::Vector{T}; delta::T
    conditional_variance::Vector{T}; residuals::Vector{T}
    n_params::Int; loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end
struct FIGARCHModel{T<:Real}
    p::Int; q::Int; mu::T; omega::T; phi::Vector{T}; beta::Vector{T}; d::T
    conditional_variance::Vector{T}; residuals::Vector{T}
    truncation::Int; n_neg_lambda::Int; loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end
struct FIEGARCHModel{T<:Real}
    p::Int; q::Int; mu::T; omega::T; theta::T; gamma::T; phi::Vector{T}; beta::Vector{T}; d::T
    conditional_variance::Vector{T}; residuals::Vector{T}
    truncation::Int; loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end
struct GarchMidasModel{T<:Real}
    mu::T; alpha::T; beta::T; m_const::T; theta::T; w::T
    conditional_variance::Vector{T}; residuals::Vector{T}
    variance_ratio::T; K::Int; m_freq::Int; n_blocks::Int; rv::Symbol; span::Symbol
    loglik::T; aic::T; bic::T; converged::Bool; iterations::Int
end

"""Plausible conditional-variance / residual paths for the C064a variant mocks — real
carries both, and the #69 forecast/predict/residuals verbs consume them."""
_mock_vol_paths(y, v) = (fill(Float64(v), length(y)),
                         Float64.(collect(y)) .- mean(Float64.(collect(y))))

function estimate_igarch(y, p::Int=1, q::Int=1; method::Symbol=:mle)
    v = length(y) > 1 ? _mock_var0(Float64.(y)) : 1.0
    a = fill(0.1 / q, q); b = fill(0.9 / p, p)     # Σα+Σβ = 1 by construction
    cv, rs = _mock_vol_paths(y, v)
    IGARCHModel{Float64}(p, q, mean(y), 0.02 * v, a, b, cv, rs, -150.0, 306.0, 320.0, true, 45)
end
function estimate_cgarch(y; method::Symbol=:mle)
    v = length(y) > 1 ? _mock_var0(Float64.(y)) : 1.0
    cv, rs = _mock_vol_paths(y, v)
    CGARCHModel{Float64}(mean(y), v, 0.99, 0.05, 0.05, 0.85, cv, rs, -148.0, 308.0, 324.0, true, 60)
end
function estimate_aparch(y, p::Int=1, q::Int=1; fix_delta=nothing, fix_gamma=nothing, method::Symbol=:mle)
    v = length(y) > 1 ? _mock_var0(Float64.(y)) : 1.0
    a = fill(0.05, q); g = fill(0.1, q); b = fill(0.85 / p, p); delta = 1.5
    nfree = (3 + 2q + p) - (fix_delta === nothing ? 0 : 1) - (fix_gamma === nothing ? 0 : q)
    cv, rs = _mock_vol_paths(y, v)
    APARCHModel{Float64}(p, q, mean(y), (v^(delta / 2)) * 0.05, a, g, b, delta,
                         cv, rs, nfree, -147.0, 310.0, 330.0, true, 70)
end
function estimate_figarch(r; p::Int=1, q::Int=1, d0::Real=0.4, truncation::Int=1000, dist::Symbol=:normal)
    v = length(r) > 1 ? _mock_var0(Float64.(r)) : 1.0
    K = min(truncation, length(r) - 1)
    cv, rs = _mock_vol_paths(r, v)
    FIGARCHModel{Float64}(p, q, mean(r), 0.02 * v, fill(0.3 / q, q), fill(0.5 / p, p),
                          Float64(d0), cv, rs, K, 0, -149.0, 306.0, 322.0, true, 55)
end
function estimate_fiegarch(r; p::Int=1, q::Int=1, d0::Real=0.4, truncation::Int=1000, dist::Symbol=:normal)
    v = length(r) > 1 ? _mock_var0(Float64.(r)) : 1.0
    K = min(truncation, length(r) - 1)
    cv, rs = _mock_vol_paths(r, v)
    FIEGARCHModel{Float64}(p, q, mean(r), log(max(v, 1e-8)), -0.05, 0.1,
                           fill(0.3 / q, q), fill(0.5 / p, p), Float64(d0), cv, rs, K,
                           -146.0, 308.0, 328.0, true, 65)
end
function estimate_garch_midas(r, x_lf=Float64[]; K::Int=12, m_freq::Int, rv::Symbol=:realized, span::Symbol=:fixed)
    v = length(r) > 1 ? _mock_var0(Float64.(r)) : 1.0
    nblk = fld(length(r), max(m_freq, 1))
    cv, rs = _mock_vol_paths(r, v)
    GarchMidasModel{Float64}(mean(r), 0.05, 0.85, log(max(v, 1e-8)), 0.1, 3.0,
                             cv, rs, 0.4, K, m_freq, nblk, rv, span, -145.0, 302.0, 320.0, true, 80)
end


# #69 verbs. forecast returns a VolatilityForecast for five variants but a NamedTuple
# (total, long_run, short_run, horizon) for GarchMidasModel — exactly as real does.
const _MOCK_VOL_VARIANTS = Union{IGARCHModel,CGARCHModel,APARCHModel,FIGARCHModel,FIEGARCHModel}

function forecast(m::_MOCK_VOL_VARIANTS, h::Int; conf_level::Real=0.95, n_sim::Int=10000)
    h >= 1 || throw(ArgumentError("Forecast horizon must be ≥ 1"))
    base = m.conditional_variance[end]
    f = fill(Float64(base), h)
    VolatilityForecast{Float64}(f, f .* 0.8, f .* 1.2, f .* 0.1, h, Float64(conf_level), :variant)
end

function forecast(m::GarchMidasModel, h::Int)
    h < 1 && throw(ArgumentError("Forecast horizon must be ≥ 1"))
    tot = fill(Float64(m.conditional_variance[end]), h)
    (total=tot, long_run=fill(1.05, h), short_run=tot ./ 1.05, horizon=h)
end

# predict(m) is the in-sample conditional variance — real defines it for igarch/cgarch/
# aparch/garch-midas but NOT figarch/fiegarch, so the mock omits those two as well and
# the handler falls back to the .conditional_variance field (mock ⊆ real).
predict(m::IGARCHModel) = m.conditional_variance
predict(m::CGARCHModel) = m.conditional_variance
predict(m::APARCHModel) = m.conditional_variance
predict(m::GarchMidasModel) = m.conditional_variance

residuals(m::_MOCK_VOL_VARIANTS) = m.residuals
residuals(m::GarchMidasModel) = m.residuals

coef(m::IGARCHModel) = vcat(m.mu, m.omega, m.alpha, m.beta)
coef(m::CGARCHModel) = [m.mu, m.omega, m.rho, m.phi, m.alpha, m.beta]
coef(m::APARCHModel) = vcat(m.mu, m.omega, m.alpha, m.gamma, m.beta, m.delta)
coef(m::FIGARCHModel) = vcat(m.mu, m.omega, m.phi, m.beta, m.d)
coef(m::FIEGARCHModel) = vcat(m.mu, m.omega, m.theta, m.gamma, m.phi, m.beta, m.d)
coef(m::GarchMidasModel) = [m.mu, m.alpha, m.beta, m.m_const, m.theta, m.w]

stderror(m::IGARCHModel) = fill(0.02, length(coef(m)))
stderror(m::CGARCHModel) = fill(0.02, length(coef(m)))
stderror(m::APARCHModel) = fill(0.02, length(coef(m)))
stderror(m::FIGARCHModel) = fill(0.02, length(coef(m)))
stderror(m::FIEGARCHModel) = fill(0.02, length(coef(m)))
stderror(m::GarchMidasModel) = fill(0.02, length(coef(m)))

loglikelihood(m::Union{IGARCHModel,CGARCHModel,APARCHModel,FIGARCHModel,FIEGARCHModel,GarchMidasModel}) = m.loglik
persistence(m::IGARCHModel) = 1.0
persistence(m::CGARCHModel) = m.rho
persistence(m::APARCHModel) = sum(m.beta) + sum(m.alpha)
persistence(m::FIGARCHModel) = m.d
persistence(m::FIEGARCHModel) = m.d
persistence(m::GarchMidasModel) = m.alpha + m.beta
unconditional_variance(m::CGARCHModel) = m.omega
component_variances(m::CGARCHModel) =
    (permanent=fill(m.omega, 3), transitory=fill(0.0, 3), total=fill(m.omega, 3))

export IGARCHModel, CGARCHModel, APARCHModel, FIGARCHModel, FIEGARCHModel, GarchMidasModel
export estimate_igarch, estimate_cgarch, estimate_aparch, estimate_figarch,
       estimate_fiegarch, estimate_garch_midas
export component_variances

# ─── C067a: penalized & limited-dependent cross-section regression ──────────
# PenalizedRegModel (lasso/ridge/elastic-net), RobustRegModel, TobitModel — faithful
# to real MEMs 0.7.0 src/reg/{penalized,robust,tobit}. Genuine OLS/ridge fits so T1/T2
# catch shape bugs. `coef(::PenalizedRegModel)=beta` but stderror is DELIBERATELY not
# defined (mirrors the real MethodError → the CLI uses `_penalized_coef_table`).

struct PenalizedRegModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    beta0::T
    alpha::T
    lambda::T
    active_set::Vector{Int}
    df_star::T
    r2::T
    aic::T
    bic::T
    ebic::T
    lambda_min::T
    lambda_1se::T
    select::Symbol
    varnames::Vector{String}
end

struct RobustRegModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    scale::T
    weights::Vector{T}
    residuals::Vector{T}
    fitted::Vector{T}
    psi::Symbol
    method::Symbol
    tuning::T
    robust_r2::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
end

struct TobitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    sigma::T
    vcov_mat::Matrix{T}
    sigma_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    lower::T
    upper::T
    n_censored_left::Int
    n_censored_right::Int
    dist::Symbol
    varnames::Vector{String}
    method::Symbol
    converged::Bool
end

# C067b: truncated-normal regression (mirror of the real TruncRegModel fields).
struct TruncRegModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    sigma::T
    vcov_mat::Matrix{T}
    sigma_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    aic::T
    bic::T
    lower::T
    upper::T
    n_truncated::Int
    dist::Symbol
    varnames::Vector{String}
    method::Symbol
    converged::Bool
end

# C067b: Heckman sample-selection model (mirror of the real HeckmanModel fields).
struct HeckmanModel{T<:Real}
    beta::Vector{T}
    vcov_beta::Matrix{T}
    outcome_names::Vector{String}
    gamma::Vector{T}
    vcov_gamma::Matrix{T}
    select_names::Vector{String}
    rho::T
    sigma::T
    lambda::T
    rho_se::T
    sigma_se::T
    lambda_se::T
    mills::Vector{T}
    method::Symbol
    loglik::T
    aic::T
    bic::T
    n_selected::Int
    n_total::Int
    y::Vector{T}
    X::Matrix{T}
    converged::Bool
end

# Faithful validation + genuine fit so shape/argument bugs surface in T1/T2.
function estimate_elastic_net(y::AbstractVector, X::AbstractMatrix;
                              alpha::Real=1.0, lambda=:cv, select::Symbol=:cv,
                              cv::Symbol=:kfold, nfolds::Int=10, standardize::Bool=true,
                              varnames=nothing, seed::Int=1234, kwargs...)
    n, p = size(X)
    length(y) == n || throw(ArgumentError("y has length $(length(y)); X has $n rows"))
    n > 1 || throw(ArgumentError("need n > 1"))
    (0 <= alpha <= 1) || throw(ArgumentError("alpha must be in [0,1]; got $alpha"))
    select in (:cv, :aic, :bic, :ebic) ||
        throw(ArgumentError("select must be :cv, :aic, :bic, or :ebic; got :$select"))
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    λ = lambda isa Real ? Float64(lambda) : 0.1
    beta = (Xm'Xm + λ * Matrix{Float64}(I(p))) \ (Xm'yv)
    beta0 = mean(yv) - sum(vec(mean(Xm; dims=1)) .* beta)
    fitted = beta0 .+ Xm * beta
    resid = yv .- fitted
    ssr = sum(abs2, resid); tss = max(sum(abs2, yv .- mean(yv)), eps())
    r2 = 1.0 - ssr / tss
    vn = varnames === nothing ? ["x$j" for j in 1:p] : Vector{String}(varnames)
    PenalizedRegModel{Float64}(yv, Xm, beta, beta0, Float64(alpha), λ,
                               findall(!=(0.0), beta), Float64(count(!=(0.0), beta)),
                               r2, -120.0, -100.0, -95.0, λ, 2λ, select, vn)
end
# Explicit kwargs (not a `; kwargs...` absorber) — keeps the mock-surface budget flat.
estimate_lasso(y, X; lambda=:cv, select::Symbol=:cv, varnames=nothing) =
    estimate_elastic_net(y, X; alpha=1.0, lambda=lambda, select=select, varnames=varnames)
estimate_ridge(y, X; lambda=:cv, select::Symbol=:cv, varnames=nothing) =
    estimate_elastic_net(y, X; alpha=0.0, lambda=lambda, select=select, varnames=varnames)

function estimate_robust(y::AbstractVector, X::AbstractMatrix;
                         psi::Symbol=:huber, method::Symbol=:m, maxiter::Int=50,
                         tol::Real=1e-6, varnames=nothing, kwargs...)
    n = length(y); p = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > p || throw(ArgumentError("Need n > p (n=$n, p=$p)"))
    method in (:m, :mm) || throw(ArgumentError("method must be :m or :mm; got :$method"))
    psi in (:huber, :bisquare) ||
        throw(ArgumentError("psi must be :huber or :bisquare; got :$psi"))
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    beta = Xm \ yv
    fitted = Xm * beta; resid = yv .- fitted
    s2 = sum(abs2, resid) / max(n - p, 1)
    vcov_mat = s2 .* ((Xm'Xm) \ Matrix{Float64}(I(p)))
    tss = max(sum(abs2, yv .- mean(yv)), eps())
    vn = varnames === nothing ? ["x$j" for j in 1:p] : Vector{String}(varnames)
    RobustRegModel{Float64}(yv, Xm, beta, vcov_mat, sqrt(max(s2, 0.0)),
                            ones(Float64, n), resid, fitted, psi, method,
                            psi === :huber ? 1.345 : 4.685,
                            1.0 - sum(abs2, resid) / tss, vn, true, 8)
end

function estimate_tobit(y::AbstractVector, X::AbstractMatrix;
                        lower::Real=0.0, upper::Real=Inf, dist::Symbol=:normal,
                        varnames=nothing, kwargs...)
    n = length(y); k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    lower < upper || throw(ArgumentError("lower ($lower) must be < upper ($upper)"))
    dist in (:normal, :logistic, :extreme_value) ||
        throw(ArgumentError("dist must be :normal, :logistic, or :extreme_value; got :$dist"))
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    beta = Xm \ yv
    fitted = Xm * beta; resid = yv .- fitted
    s2 = sum(abs2, resid) / max(n - k, 1); sigma = sqrt(max(s2, eps()))
    vcov_mat = s2 .* ((Xm'Xm) \ Matrix{Float64}(I(k)))
    nL = isfinite(lower) ? count(<=(Float64(lower)), yv) : 0
    nR = isfinite(upper) ? count(>=(Float64(upper)), yv) : 0
    vn = varnames === nothing ? ["x$j" for j in 1:k] : Vector{String}(varnames)
    TobitModel{Float64}(yv, Xm, beta, sigma, vcov_mat, sigma / sqrt(2n), resid, fitted,
                        -110.0, 230.0, 245.0, Float64(lower), Float64(upper),
                        nL, nR, dist, vn, :normal, true)
end

# C067b: truncated regression — validates like real (every y strictly inside (lower,upper)).
function estimate_truncreg(y::AbstractVector, X::AbstractMatrix;
                           lower::Real=0.0, upper::Real=Inf, varnames=nothing,
                           maxiter::Int=1000, tol::Real=1e-10)
    n = length(y); k = size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    L = Float64(lower); U = Float64(upper)
    L < U || throw(ArgumentError("lower ($L) must be < upper ($U)"))
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    all(yi -> L < yi < U, yv) ||
        throw(ArgumentError("truncated regression requires every y strictly inside (lower, upper)"))
    beta = Xm \ yv
    fitted = Xm * beta; resid = yv .- fitted
    s2 = sum(abs2, resid) / max(n - k, 1); sigma = sqrt(max(s2, eps()))
    vcov_mat = s2 .* ((Xm'Xm) \ Matrix{Float64}(I(k)))
    vn = varnames === nothing ? ["x$j" for j in 1:k] : Vector{String}(varnames)
    TruncRegModel{Float64}(yv, Xm, beta, sigma, vcov_mat, sigma / sqrt(2n), resid, fitted,
                           -108.0, 226.0, 241.0, L, U, n, :normal, vn, :truncreg, true)
end

# C067b: Heckman selection — probit-ish selection γ + OLS-on-selected outcome β; validates
# sizes / binary d / enough selected obs like the real two-step estimator.
function estimate_heckman(y::AbstractVector, X::AbstractMatrix, d::AbstractVector,
                          Z::AbstractMatrix; method::Symbol=:twostep,
                          outcome_names=nothing, select_names=nothing,
                          maxiter::Int=1000, tol::Real=1e-10)
    method in (:twostep, :mle) ||
        throw(ArgumentError("method must be :twostep or :mle; got :$method"))
    n = length(y)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    size(Z, 1) == n || throw(ArgumentError("Z must have $n rows (got $(size(Z, 1)))"))
    length(d) == n || throw(ArgumentError("d must have length $n (got $(length(d)))"))
    k = size(X, 2); p = size(Z, 2)
    dv = Vector{Float64}(float.(d))
    all(v -> v == 0.0 || v == 1.0, dv) ||
        throw(ArgumentError("selection indicator d must be binary (0/1)"))
    sel = findall(==(1.0), dv); n1 = length(sel)
    n1 > k + 1 || throw(ArgumentError("need more selected obs than outcome params (n_sel=$n1, k=$k)"))
    Xm = Matrix{Float64}(X); Zm = Matrix{Float64}(Z); yv = Vector{Float64}(y)
    all(isfinite, yv[sel]) || throw(ArgumentError("y has non-finite values among selected observations"))
    γ = Zm \ dv
    Vγ = ((Zm'Zm) \ Matrix{Float64}(I(p)))
    Xs = Xm[sel, :]; ys = yv[sel]
    β = Xs \ ys
    resid = ys .- Xs * β
    s2 = sum(abs2, resid) / max(n1 - k, 1); σ = sqrt(max(s2, eps()))
    Vβ = s2 .* ((Xs'Xs) \ Matrix{Float64}(I(k)))
    on = outcome_names === nothing ? ["x$j" for j in 1:k] : Vector{String}(outcome_names)
    sn = select_names === nothing ? ["z$j" for j in 1:p] : Vector{String}(select_names)
    ρ = 0.3; λ = ρ * σ
    HeckmanModel{Float64}(β, Vβ, on, γ, Vγ, sn, ρ, σ, λ, 0.1, σ / sqrt(2n1), 0.15,
                          zeros(Float64, n1), method, -150.0, 310.0, 330.0, n1, n, ys, Xs, true)
end

# coef/stderror: PenalizedRegModel exposes only coef (stderror undefined → MethodError,
# mirroring real MEMs). Robust/Tobit/TruncReg expose both (sqrt of the vcov diagonal);
# Heckman's coef/stderror are the OUTCOME equation (mirrors real StatsAPI dispatch).
coef(m::PenalizedRegModel) = m.beta
coef(m::RobustRegModel) = m.beta
coef(m::TobitModel) = m.beta
coef(m::TruncRegModel) = m.beta
coef(m::HeckmanModel) = m.beta
stderror(m::RobustRegModel) = [sqrt(max(m.vcov_mat[i, i], 0.0)) for i in 1:length(m.beta)]
stderror(m::TobitModel) = [sqrt(max(m.vcov_mat[i, i], 0.0)) for i in 1:length(m.beta)]
stderror(m::TruncRegModel) = [sqrt(max(m.vcov_mat[i, i], 0.0)) for i in 1:length(m.beta)]
stderror(m::HeckmanModel) = [sqrt(max(m.vcov_beta[i, i], 0.0)) for i in 1:length(m.beta)]

export PenalizedRegModel, RobustRegModel, TobitModel, TruncRegModel, HeckmanModel
export estimate_lasso, estimate_ridge, estimate_elastic_net, estimate_robust, estimate_tobit
export estimate_truncreg, estimate_heckman

# ─── C062a: cointegrating regression (FMOLS / CCR / DOLS) ────────────────────
# CointRegModel / PanelCointRegModel mirror the real MEMs 0.7.0 field NAMES (a subset is
# fine — check_mock_surface is mock ⊆ real, and these are non-core). The estimators are
# genuine-ish OLS-on-levels fits that VALIDATE like the real ones (n>5, y/X shape, method /
# trend / pooling enums throw ArgumentError/DimensionMismatch) so T1/T2 exercise the error
# mapping and the hand-built coef-table shapes. The panel group-mean path deliberately lets
# a per-coefficient SE go to `Inf` (t==0 degenerate) to exercise the non-finite render path.
struct CointRegModel{T<:AbstractFloat}
    method::Symbol
    trend::Symbol
    kernel::Symbol
    bandwidth::T
    coef::Vector{T}
    vcov::Matrix{T}
    varnames::Vector{String}
    nobs::Int
    leads::Int
    lags::Int
    omega_uv::T
    d::Int
    k::Int
end

struct PanelCointRegModel{T<:AbstractFloat}
    method::Symbol
    pooling::Symbol
    trend::Symbol
    kernel::Symbol
    coef::Vector{T}
    se::Vector{T}
    tstats::Vector{T}
    pvalues::Vector{T}
    varnames::Vector{String}
    N::Int
    T_i::Vector{Int}
    nobs::Int
    k::Int
    d::Int
    balanced::Bool
end

coef(m::CointRegModel) = m.coef
vcov(m::CointRegModel) = m.vcov
stderror(m::CointRegModel) = [sqrt(max(m.vcov[i, i], 0.0)) for i in 1:length(m.coef)]
nobs(m::CointRegModel) = m.nobs

# Logistic approximation to 1 - Φ(|z|) (Φ ≈ 1/(1+exp(-1.702 z))); finite, monotone, in (0,1) —
# enough for a mock p-value (tests assert finiteness/shape, not magnitude). Avoids importing a
# real normal CDF into the mock module.
_mock_norm_sf(z::Real) = 1.0 / (1.0 + exp(1.702 * abs(z)))

function _mock_cointreg_det(n::Int, trend::Symbol, ::Type{T}) where {T}
    trend === :none && return zeros(T, n, 0), String[]
    trend === :const && return ones(T, n, 1), String["const"]
    trend === :linear && return hcat(ones(T, n), T.(1:n)), String["const", "trend"]
    throw(ArgumentError("trend must be :none, :const, or :linear; got :$trend"))
end

function estimate_cointreg(y::AbstractVector, X::AbstractVecOrMat;
                           method::Symbol=:fmols, trend::Symbol=:const,
                           kernel::Symbol=:bartlett, bandwidth=:andrews,
                           leads=:auto, lags=:auto, ic::Symbol=:aic, dols_se::Symbol=:lrv)
    method ∈ (:fmols, :ccr, :dols) ||
        throw(ArgumentError("method must be :fmols, :ccr, or :dols; got :$method"))
    T = Float64
    yv = collect(T, y)
    Xm = X isa AbstractVector ? reshape(collect(T, X), :, 1) : Matrix{T}(X)
    size(Xm, 1) == length(yv) ||
        throw(DimensionMismatch("length(y)=$(length(yv)) must equal size(X,1)=$(size(Xm, 1))"))
    n = length(yv)
    n > 5 || throw(ArgumentError("need at least 6 observations; got $n"))
    D, dnames = _mock_cointreg_det(n, trend, T)
    d = size(D, 2); k = size(Xm, 2)
    Z = hcat(D, Xm)
    beta = (Z'Z) \ (Z'yv)
    resid = yv .- Z * beta
    s2 = sum(abs2, resid) / max(n - d - k, 1)
    vcovm = s2 .* ((Z'Z) \ Matrix{T}(I(d + k)))
    xn = k == 1 ? String["x"] : String["x$i" for i in 1:k]
    varnames = vcat(dnames, xn)
    ld = method === :dols ? (leads === :auto ? 1 : Int(leads)) : 0
    lg = method === :dols ? (lags === :auto ? 1 : Int(lags)) : 0
    bw = bandwidth isa Symbol ? 4.0 : Float64(bandwidth)
    CointRegModel{T}(method, trend, kernel, bw, beta, Matrix{T}(vcovm), varnames,
                     n, ld, lg, T(max(s2, eps())), d, k)
end

function estimate_xtcointreg(pd::PanelData, y::Union{Symbol,String},
                             xs::Union{Symbol,String}...;
                             method::Symbol=:fmols, pooling::Symbol=:group,
                             trend::Symbol=:const, kernel::Symbol=:bartlett,
                             bandwidth=:andrews, leads=:auto, lags=:auto,
                             ic::Symbol=:aic, dols_se::Symbol=:lrv)
    method ∈ (:fmols, :dols) ||
        throw(ArgumentError("method must be :fmols or :dols; got :$method"))
    pooling ∈ (:group, :pooled) ||
        throw(ArgumentError("pooling must be :group or :pooled; got :$pooling"))
    trend ∈ (:none, :const, :linear) ||
        throw(ArgumentError("trend must be :none, :const, or :linear; got :$trend"))
    isempty(xs) && throw(ArgumentError("At least one regressor is required"))
    _vc(v) = (j = findfirst(==(string(v)), pd.varnames);
              j === nothing ? throw(ArgumentError("variable $v not found in panel")) : j)
    yc = _vc(y)
    xcs = Int[_vc(x) for x in xs]
    N = pd.n_groups
    N ≥ 1 || throw(ArgumentError("need at least one unit"))
    T = Float64
    k = length(xcs)
    d = trend === :none ? 0 : trend === :const ? 1 : 2
    unit_coefs = Vector{Vector{T}}()
    T_i = Int[]
    for g in unique(pd.group_id)
        rows = findall(==(g), pd.group_id)
        rr = rows[sortperm(pd.time_id[rows])]
        yg = T.(pd.data[rr, yc])
        Xg = T.(pd.data[rr, xcs])
        m = estimate_cointreg(yg, Xg; method=method, trend=trend, kernel=kernel,
                              bandwidth=bandwidth, leads=leads, lags=lags, ic=ic, dols_se=dols_se)
        push!(unit_coefs, m.coef)
        push!(T_i, length(yg))
    end
    balanced = all(==(first(T_i)), T_i)
    xn = k == 1 ? String["x"] : String["x$i" for i in 1:k]
    dnames = d == 0 ? String[] : d == 1 ? String["const"] : String["const", "trend"]
    if pooling === :group
        C = reduce(hcat, unit_coefs)               # p×N per-unit coefficient vectors
        coefv = vec(sum(C; dims=2)) ./ T(N)
        tstats = similar(coefv)
        for j in eachindex(coefv)
            tvec = [C[j, i] / (abs(C[j, i]) < 1e-8 ? one(T) : T(0.1) * abs(C[j, i])) for i in 1:N]
            tstats[j] = sum(tvec) / sqrt(T(N))
        end
        se = [tstats[j] == 0 ? T(Inf) : abs(coefv[j] / tstats[j]) for j in eachindex(coefv)]
        pv = [2.0 * _mock_norm_sf(t) for t in tstats]
        varnames = vcat(dnames, xn)
    else
        slopes = [uc[(d + 1):(d + k)] for uc in unit_coefs]
        coefv = reduce(+, slopes) ./ T(N)
        se = fill(T(0.1), k)
        tstats = coefv ./ se
        pv = [2.0 * _mock_norm_sf(t) for t in tstats]
        varnames = xn
    end
    PanelCointRegModel{T}(method, pooling, trend, kernel, coefv, se, tstats, pv,
                          varnames, N, T_i, sum(T_i), k, d, balanced)
end

export CointRegModel, PanelCointRegModel, estimate_cointreg, estimate_xtcointreg

# ─── C066: state-space + nonparametric estimation ───────────────────────────
# StateSpaceModel / KernelDensity / KernelRegression / LowessFit mirror the real MEMs 0.7.0
# field NAMES (a subset is fine — check_mock_surface is mock ⊆ real, and these are non-core).
# Estimators are genuine-ish fits that VALIDATE like the real ones (n bounds, kernel/method/bw
# enums, length matches) so T1/T2 exercise the error mapping and table shapes.

# Subset of real StateSpaceModel fields (check_mock_surface is mock ⊆ real by NAME, not order).
# We keep what the CLI renders: the system matrices (#71 --config general path), the state
# paths (#71 predict/residuals), and the fit summary.
struct StateSpaceModel{T<:AbstractFloat}
    # System matrices — the general fixed-matrix form y = Z α + d + ε, α' = T α + c + R η.
    # NOTE the field is `Tt`, not `T`: `T` is the struct's type parameter. Real has the same
    # constraint, which is why the TOML key (`T`), the constructor keyword (`T_mat`) and the
    # field (`Tt`) are three different spellings of one matrix.
    Z::Matrix{T}
    H::Matrix{T}
    Tt::Matrix{T}
    Q::Matrix{T}
    d::Vector{T}
    c::Vector{T}
    R::Matrix{T}
    init_mode::Symbol
    theta::Vector{T}
    param_names::Vector{String}
    # #71 consumes these; all exist in real (statespace/types.jl):
    #   filtered_state a_t|t, smoothed_state a_t|T, innovations v_t, std_residuals v_t/√F_t
    filtered_state::Matrix{T}
    smoothed_state::Matrix{T}
    innovations::Matrix{T}
    std_residuals::Matrix{T}
    loglik::T
    converged::Bool
    method::Symbol
    n_state::Int
    n_obs::Int
    T_obs::Int
end

"""Fixed-matrix outer constructor, mirroring real's signature and — critically — its
VALIDATION and DEFAULTS: `R` defaults to `I(n_state, size(Q,1))`, `d`/`c` to zeros, and
supplying `a1` AND `P1` switches init_mode to `:explicit`. Dimension violations raise the same
`ArgumentError` real does, so the CLI's mapping is exercised identically at both tiers."""
function StateSpaceModel(Z::AbstractMatrix, H::AbstractMatrix,
                         T_mat::AbstractMatrix, Q::AbstractMatrix;
                         d=nothing, c=nothing, R=nothing, a1=nothing, P1=nothing,
                         init_mode::Symbol=:kappa, kappa::Real=1e6)
    Zm = Matrix{Float64}(Z); Hm = Matrix{Float64}(H)
    Tm = Matrix{Float64}(T_mat); Qm = Matrix{Float64}(Q)
    n_obs, n_state = size(Zm)
    size(Hm) == (n_obs, n_obs) ||
        throw(ArgumentError("H must be n_obs×n_obs = $((n_obs, n_obs)), got $(size(Hm))"))
    size(Tm) == (n_state, n_state) ||
        throw(ArgumentError("T must be n_state×n_state = $((n_state, n_state)), got $(size(Tm))"))
    Rm = R === nothing ? Matrix{Float64}(I, n_state, size(Qm, 1)) : Matrix{Float64}(R)
    size(Rm, 1) == n_state ||
        throw(ArgumentError("R must have n_state=$n_state rows, got $(size(Rm, 1))"))
    size(Qm) == (size(Rm, 2), size(Rm, 2)) ||
        throw(ArgumentError("Q must be r×r with r=size(R,2)=$(size(Rm, 2)), got $(size(Qm))"))
    dv = d === nothing ? zeros(Float64, n_obs) : Vector{Float64}(d)
    cv = c === nothing ? zeros(Float64, n_state) : Vector{Float64}(c)
    length(dv) == n_obs || throw(ArgumentError("d must have length n_obs=$n_obs"))
    length(cv) == n_state || throw(ArgumentError("c must have length n_state=$n_state"))
    mode = (a1 !== nothing && P1 !== nothing) ? :explicit : init_mode
    mode in (:kappa, :diffuse, :stationary, :explicit) ||
        throw(ArgumentError("Unknown Kalman init mode :$mode"))
    StateSpaceModel{Float64}(Zm, Hm, Tm, Qm, dv, cv, Rm, mode, Float64[], String[],
        Matrix{Float64}(undef, 0, n_state), Matrix{Float64}(undef, 0, n_state),
        Matrix{Float64}(undef, 0, n_obs), Matrix{Float64}(undef, 0, n_obs),
        NaN, false, :spec, n_state, n_obs, 0)
end

"""Filter a FIXED-matrix system: no optimisation, so `theta`/`param_names` stay EMPTY and
`method` is `:filter`, exactly as real. A mock that returned a populated `theta` here would
hide the empty-parameter-table case the CLI has to render around."""
function estimate_statespace(ss::StateSpaceModel, y)
    Y = y isa AbstractMatrix ? Matrix{Float64}(y) : reshape(Vector{Float64}(y), :, 1)
    n = size(Y, 1)
    size(Y, 2) == ss.n_obs || throw(DimensionMismatch(
        "data has $(size(Y, 2)) series but Z implies n_obs=$(ss.n_obs)"))
    n >= 1 || throw(ArgumentError("need at least one observation"))
    # A crude but genuine Gaussian prediction-error decomposition against a static state at 0,
    # enough to give a finite, data-dependent loglik and correctly-shaped paths.
    state = zeros(n, ss.n_state)
    innov = Y .- (state * ss.Tt' * ss.Z')[:, 1:ss.n_obs]
    F = [max(ss.H[j, j] + ss.Z[j, :]' * ss.Q * ss.Z[j, :], 1e-8) for j in 1:ss.n_obs]
    stdres = innov ./ sqrt.(reshape(F, 1, :))
    ll = -0.5 * sum(log(2π * F[j]) * n + sum(abs2, innov[:, j]) / F[j] for j in 1:ss.n_obs)
    StateSpaceModel{Float64}(ss.Z, ss.H, ss.Tt, ss.Q, ss.d, ss.c, ss.R, ss.init_mode,
        Float64[], String[], state, state, innov, stdres,
        ll, true, :filter, ss.n_state, ss.n_obs, n)
end

struct KernelDensity{T<:AbstractFloat}
    x::Vector{T}
    density::Vector{T}
    bandwidth::T
    kernel::Symbol
    bw_method::Symbol
    data::Vector{T}
    nobs::Int
end

struct KernelRegression{T<:AbstractFloat}
    x::Vector{T}
    fitted::Vector{T}
    se::Vector{T}
    xdata::Vector{T}
    ydata::Vector{T}
    bandwidth::T
    method::Symbol
    degree::Int
    kernel::Symbol
    bw_method::Symbol
    sigma2::T
    nobs::Int
end

struct LowessFit{T<:AbstractFloat}
    x::Vector{T}
    fitted::Vector{T}
    ydata::Vector{T}
    span::T
    iter::Int
    nobs::Int
end

# Silverman rule-of-thumb bandwidth (matches the real `_bw_silverman` shape; used by the
# mock KDE/kernel-reg to produce a sensible positive h).
function _mock_bw_silverman(v::AbstractVector{<:Real})
    n = length(v)
    hi = std(v)
    lo = min(hi, (quantile(v, 0.75) - quantile(v, 0.25)) / 1.349)
    lo = lo > 0 ? lo : (hi > 0 ? hi : 1.0)
    return 0.9 * lo * n^(-1 / 5)
end

function local_level(y; init_mode::Symbol=:kappa, kappa::Real=1e6)
    yv = Vector{Float64}(collect(Float64, vec(y)))
    n = length(yv)
    n >= 2 || throw(ArgumentError("local_level requires at least 2 observations"))
    v0 = max(var(yv), 1.0)
    smoothed = reshape(yv, :, 1)                     # level ≈ observations (n_state=1)
    filtered = reshape(vcat(yv[1], yv[1:end-1]), :, 1)   # a_t|t lags a_t|T by construction
    innov = reshape(yv .- vec(filtered), :, 1)           # v_t = y_t - a_t|t-1
    stdres = innov ./ sqrt.(v0 .* (1 .+ 1 ./ (1:n)))   # F_t varies over t, as in real
    ll = -0.5 * n * (log(2π) + log(v0) + 1)
    # System matrices for the canned local level, so the struct is fully populated whichever
    # path built it: y = μ + ε, μ' = μ + η.
    StateSpaceModel{Float64}(reshape([1.0], 1, 1), reshape([v0 / 2], 1, 1),
                             reshape([1.0], 1, 1), reshape([v0 / 2], 1, 1),
                             [0.0], [0.0], reshape([1.0], 1, 1), init_mode,
                             [v0 / 2, v0 / 2], ["σ²_ε", "σ²_η"], filtered, smoothed,
                             innov, stdres, ll, true, :mle, 1, 1, n)
end

function local_linear_trend(y; init_mode::Symbol=:kappa, kappa::Real=1e6)
    yv = Vector{Float64}(collect(Float64, vec(y)))
    n = length(yv)
    n >= 2 || throw(ArgumentError("local_linear_trend requires at least 2 observations"))
    v0 = max(var(yv), 1.0)
    slope = vcat(diff(yv), yv[end] - yv[end-1])
    smoothed = hcat(yv, slope)                       # state = [μ, β] (n_state=2)
    filtered = hcat(vcat(yv[1], yv[1:end-1]), slope)
    innov = reshape(yv .- filtered[:, 1], :, 1)
    stdres = innov ./ sqrt.(v0 .* (1 .+ 1 ./ (1:n)))   # F_t varies over t, as in real
    ll = -0.5 * n * (log(2π) + log(v0) + 1)
    # y = μ + ε; [μ, β]' = [1 1; 0 1] [μ, β] + η  (the canonical local linear trend)
    StateSpaceModel{Float64}([1.0 0.0], reshape([v0 / 2], 1, 1),
                             [1.0 1.0; 0.0 1.0], [v0/10 0.0; 0.0 v0/100],
                             [0.0], [0.0, 0.0], Matrix{Float64}(I, 2, 2), init_mode,
                             [v0 / 2, v0 / 10, v0 / 100], ["σ²_ε", "σ²_ξ", "σ²_ζ"],
                             filtered, smoothed, innov, stdres, ll, true, :mle, 2, 1, n)
end

function estimate_tvp_reg(y, X::AbstractMatrix; intercept::Bool=true,
                          init_mode::Symbol=:kappa, kappa::Real=1e6,
                          iterations::Int=1000, g_tol::Real=1e-8)
    yv = Vector{Float64}(collect(Float64, vec(y)))
    n = length(yv)
    size(X, 1) == n || throw(ArgumentError("X must have the same number of rows as y ($n)"))
    n >= 2 || throw(ArgumentError("estimate_tvp_reg requires at least 2 observations"))
    Xm = Matrix{Float64}(X)
    Xf = intercept ? hcat(ones(Float64, n), Xm) : Xm
    k = size(Xf, 2)
    beta = Xf \ yv                                   # OLS as the constant "average" path
    smoothed = repeat(permutedims(beta), n)          # T×k (mock: constant path)
    v0 = max(var(yv), 1.0)
    theta = vcat(v0, fill(v0 / (100 * k), k))
    names = vcat("σ²_ε", ["σ²_η[$j]" for j in 1:k])
    ll = -0.5 * n * (log(2π) + log(v0) + 1)
    fitted = Xf * beta
    innov = reshape(yv .- fitted, :, 1)
    # TVP regression in state-space form: the state IS the coefficient vector β_t, so Z is the
    # regressor row (time-varying in truth; the mock holds the sample mean as a stand-in).
    StateSpaceModel{Float64}(reshape(vec(mean(Xf, dims=1)), 1, k), reshape([v0], 1, 1),
                             Matrix{Float64}(I, k, k),
                             Matrix{Float64}(v0 / (100 * k) * I, k, k),
                             [0.0], zeros(k), Matrix{Float64}(I, k, k), :kappa,
                             theta, names, smoothed, smoothed, innov,
                             innov ./ sqrt.(v0 .* (1 .+ 1 ./ (1:n))), ll, true, :mle, k, 1, n)
end

function kernel_density(y::AbstractVector; kernel::Symbol=:gaussian,
                        bw::Union{Symbol,Real}=:silverman, npoints::Int=512, cut::Real=3.0)
    data = Vector{Float64}(collect(Float64, y))
    n = length(data)
    n >= 2 || throw(ArgumentError("kernel_density requires at least 2 observations"))
    npoints >= 2 || throw(ArgumentError("npoints must be ≥ 2"))
    kernel in (:gaussian, :epanechnikov, :triangular, :uniform) ||
        throw(ArgumentError("unknown kernel :$kernel"))
    bw_method = :user; h = 0.0
    if bw isa Symbol
        bw in (:silverman, :sj) || throw(ArgumentError("unknown bandwidth rule :$bw"))
        bw_method = bw; h = _mock_bw_silverman(data)
    else
        h = Float64(bw); h > 0 || throw(ArgumentError("numeric bandwidth must be positive"))
    end
    lo = minimum(data) - cut * h; hi = maximum(data) + cut * h
    grid = collect(range(lo, hi; length=npoints))
    dens = [sum(exp.(-((g .- data) ./ h) .^ 2 ./ 2) ./ sqrt(2π)) / (n * h) for g in grid]
    KernelDensity{Float64}(grid, dens, h, kernel, bw_method, data, n)
end

function kernel_reg(y::AbstractVector, x::AbstractVector; method::Symbol=:ll,
                    degree::Int=1, bw::Union{Symbol,Real}=:cv, kernel::Symbol=:gaussian)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n = length(x)
    n >= 3 || throw(ArgumentError("kernel_reg requires at least 3 observations"))
    method in (:nw, :ll, :lp) || throw(ArgumentError("unknown method :$method"))
    kernel in (:gaussian, :epanechnikov, :triangular, :uniform) ||
        throw(ArgumentError("unknown kernel :$kernel"))
    deg = method === :nw ? 0 : (method === :ll ? 1 : degree)
    xs = Vector{Float64}(collect(Float64, x)); ys = Vector{Float64}(collect(Float64, y))
    perm = sortperm(xs); xs = xs[perm]; ys = ys[perm]
    bw_method = :user; h = 0.0
    if bw isa Symbol
        bw in (:cv, :rot) || throw(ArgumentError("unknown bandwidth rule :$bw"))
        bw_method = bw; h = _mock_bw_silverman(xs)
    else
        h = Float64(bw); h > 0 || throw(ArgumentError("numeric bandwidth must be positive"))
    end
    fitted = copy(ys)                                # mock fit interpolates the data
    se = fill(std(ys) / sqrt(n), n)
    KernelRegression{Float64}(xs, fitted, se, xs, ys, h, method, deg, kernel, bw_method,
                              var(ys), n)
end

function lowess(y::AbstractVector, x::AbstractVector; f::Real=2//3, iter::Int=3,
                delta::Union{Real,Nothing}=nothing)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n = length(x)
    n >= 2 || throw(ArgumentError("lowess requires at least 2 observations"))
    iter >= 0 || throw(ArgumentError("iter must be ≥ 0"))
    xs = Vector{Float64}(collect(Float64, x)); ys = Vector{Float64}(collect(Float64, y))
    perm = sortperm(xs); xs = xs[perm]; ys = ys[perm]
    LowessFit{Float64}(xs, ys, ys, Float64(f), iter, n)
end

export StateSpaceModel, KernelDensity, KernelRegression, LowessFit
export local_level, local_linear_trend, estimate_tvp_reg, estimate_statespace
export kernel_density, kernel_reg, lowess

# ─── BVARForecast Type & Forecast Accessors ──────────────────

struct BVARForecast{T<:AbstractFloat}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    conf_level::T
    point_estimate::Symbol
    varnames::Vector{String}
    # Real trailing field (0.8.0): posterior forecast draws (store_draws=true),
    # layout n_draws × h × n. Without store_draws upstream leaves it nothing and
    # estimate_opp silently falls back to IRF-only bands.
    _draws::Union{Nothing,Array{T,3}}
end
BVARForecast(f::Matrix{T}, cl, cu, h::Int, conf::T, pe::Symbol, vn) where T =
    BVARForecast{T}(f, cl, cu, h, conf, pe, vn, nothing)

point_forecast(f::Union{VARForecast,BVARForecast}) = f.forecast
lower_bound(f::Union{VARForecast,BVARForecast}) = f.ci_lower
upper_bound(f::Union{VARForecast,BVARForecast}) = f.ci_upper
forecast_horizon(f::Union{VARForecast,BVARForecast}) = f.horizon

# long_table — tidy/long view mirroring real MEMs (#346): array-valued results render
# as `horizon | variable | value | lower | upper` (lower/upper missing when ci_method==:none).
function long_table(f::VARForecast)
    H, nv = size(f.forecast)
    vn = length(f.varnames) == nv ? f.varnames : ["y$i" for i in 1:nv]
    has_ci = f.ci_method != :none
    horizon = Int[]; variable = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv
        push!(horizon, h); push!(variable, vn[v]); push!(value, f.forecast[h, v])
        push!(lower, has_ci ? f.ci_lower[h, v] : missing)
        push!(upper, has_ci ? f.ci_upper[h, v] : missing)
    end
    return DataFrame(; horizon, variable, value, lower, upper)
end

function long_table(irf::ImpulseResponse)
    H = size(irf.values, 1); nv = length(irf.variables); ns = length(irf.shocks)
    has_ci = irf.ci_type != :none && irf.ci_lower !== nothing
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, irf.variables[v]); push!(shock, irf.shocks[s])
        push!(value, irf.values[h, v, s])
        push!(lower, has_ci ? irf.ci_lower[h, v, s] : missing)
        push!(upper, has_ci ? irf.ci_upper[h, v, s] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

function long_table(f::FEVD)
    nv, ns, H = size(f.proportions)     # (variable, shock, horizon)
    horizon = Int[]; variable = String[]; shock = String[]; value = Float64[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, f.variables[v]); push!(shock, f.shocks[s])
        push!(value, f.proportions[v, s, h])
    end
    return DataFrame(; horizon, variable, shock, value)
end

function long_table(irf::BayesianImpulseResponse)
    H = size(irf.point_estimate, 1); nv = length(irf.variables); ns = length(irf.shocks)
    nq = size(irf.quantiles, 4)
    horizon = Int[]; variable = String[]; shock = String[]
    value = Float64[]; lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv, s in 1:ns
        push!(horizon, h); push!(variable, irf.variables[v]); push!(shock, irf.shocks[s])
        push!(value, irf.point_estimate[h, v, s])
        push!(lower, nq > 0 ? irf.quantiles[h, v, s, 1] : missing)
        push!(upper, nq > 0 ? irf.quantiles[h, v, s, nq] : missing)
    end
    return DataFrame(; horizon, variable, shock, value, lower, upper)
end

# Shared tidy builder for AbstractForecastResult-style types → horizon|variable|value|
# lower|upper (mirrors real MEMs long_table(::AbstractForecastResult); univariate vectors
# reshape to (h,1)). lo/hi === nothing ⇒ missing bands.
function _mock_fc_lt(pf, lo, hi, varnames::Vector{String})
    pfm = pf isa AbstractVector ? reshape(pf, :, 1) : pf
    H, nv = size(pfm)
    vn = length(varnames) == nv ? varnames : ["y$i" for i in 1:nv]
    _m(x) = x === nothing ? nothing : (x isa AbstractVector ? reshape(x, :, 1) : x)
    lom = _m(lo); him = _m(hi)
    horizon = Int[]; variable = String[]; value = Float64[]
    lower = Union{Missing,Float64}[]; upper = Union{Missing,Float64}[]
    for h in 1:H, v in 1:nv
        push!(horizon, h); push!(variable, vn[v]); push!(value, Float64(pfm[h, v]))
        push!(lower, lom === nothing ? missing : Float64(lom[h, v]))
        push!(upper, him === nothing ? missing : Float64(him[h, v]))
    end
    return DataFrame(; horizon, variable, value, lower, upper)
end
long_table(f::BVARForecast)       = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, f.varnames)
long_table(f::LPForecast)         = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])

# Coefficient-bearing models expose a tidy coef table via Tables.jl in real MEMs
# (`DataFrame(model)` → equation|term|estimate|std_error|stat|p_value|ci_lower|ci_upper,
# C051 #346). Tables isn't a test dep, so the mock extends DataFrames.DataFrame directly.
function _mock_coef_df_base(term, est::Vector{Float64})
    ne = length(est); se = fill(0.1, ne)
    DataFrames.DataFrame(term=term, estimate=est, std_error=se, stat=est ./ se,
        p_value=fill(0.5, ne), ci_lower=est .- 0.2, ci_upper=est .+ 0.2)
end
function DataFrames.DataFrame(m::VARModel)
    ncoef, neq = size(m.B)
    terms = vcat(["const"], ["$(m.varnames[v]).L$l" for l in 1:m.p for v in 1:length(m.varnames)])
    length(terms) == ncoef || (terms = ["term$i" for i in 1:ncoef])
    equation = String[]; term = String[]; est = Float64[]
    for j in 1:neq, i in 1:ncoef
        push!(equation, m.varnames[j]); push!(term, terms[i]); push!(est, Float64(m.B[i, j]))
    end
    df = _mock_coef_df_base(term, est)
    DataFrames.insertcols!(df, 1, :equation => equation)
    return df
end
# NB: the Union-typed single-equation/panel/ordered coef table and the multinomial coef
# table (DataFrame(::Union{RegModel,...}) / DataFrame(::MultinomialLogitModel)) live
# further down, after RegModel/PanelRegModel/OrderedLogitModel/MultinomialLogitModel etc.
# are actually defined — see "single-equation coefficient models" below estimate_mlogit.
long_table(f::VolatilityForecast) = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])
long_table(f::ARIMAForecast)      = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])
long_table(f::VECMForecast)       = _mock_fc_lt(f.levels, f.ci_lower, f.ci_upper, String[])
long_table(f::FactorForecast)     = _mock_fc_lt(f.observables, f.observables_lower, f.observables_upper, String[])
export long_table

# ─── C065a: SETAR / threshold nonlinear-TS mocks ─────────────
# Mirror the real MEMs 0.7.0 nonlinear types (src/nonlinear/{types,threshold}.jl), fields a
# subset in real order so check_mock_surface (mock ⊆ real) passes. HansenLinearityTest is
# defined BEFORE ThresholdModel (a Union field references it); all estimator/forecast/test
# fns are defined AFTER the structs. The estimators VALIDATE like real (throwing the same
# ArgumentError/DimensionMismatch classes) so the CLI's `_nonlinear_error` mapping is honestly
# exercised at T1/T2, then return a genuine-ish two-regime OLS fit. NO `; kwargs...` absorbers.
struct HansenLinearityTest{T<:AbstractFloat}
    sup_lm::T
    sup_wald::T
    pvalue_lm::T
    pvalue_wald::T
    gamma_sup::T
    reps::Int
    trim::T
    n_grid::Int
end

struct ThresholdModel{T<:AbstractFloat} <: AbstractNonlinearTSModel
    y::Vector{T}
    X::Matrix{T}
    q::Vector{T}
    gamma::T
    gamma_ci::Tuple{T,T}
    gamma_ci_level::T
    beta1::Vector{T}
    beta2::Vector{T}
    se1::Vector{T}
    se2::Vector{T}
    regime::Vector{Bool}
    ssr1::T
    ssr2::T
    ssr::T
    sigma2::T
    residuals::Vector{T}
    n::Int
    k::Int
    n1::Int
    n2::Int
    p::Int
    d::Int
    is_setar::Bool
    aic::T
    bic::T
    xnames::Vector{String}
    qname::String
    trim::T
    linearity::Union{Nothing,HansenLinearityTest{T}}
end

struct ThresholdForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
    reps::Int
end

# Ordinary least squares on (Xr, yr): returns (beta, se, ssr); falls back to a degenerate
# fit if the regime has too few observations (mirrors the pooled-fallback edge in real).
function _mock_ols_regime(Xr::AbstractMatrix, yr::AbstractVector)
    k = size(Xr, 2)
    if size(Xr, 1) > k
        b = Xr \ yr
        resid = yr .- Xr * b
        ssr = sum(abs2, resid)
        s2 = ssr / max(size(Xr, 1) - k, 1)
        # A rank-deficient regime design (collinear columns) makes `inv(Xr'Xr)` throw an untyped
        # SingularException. Real MEMs never reaches SE computation on such a split (its threshold
        # grid rejects it first — see the constant-q guard in estimate_setar), but guard here so a
        # degenerate regime yields Inf SEs (rendered non-finite-safe) rather than a raw throw.
        XtXinv = try
            inv(Xr' * Xr)
        catch
            return b, fill(Inf, k), ssr
        end
        se = Float64[sqrt(max(s2 * XtXinv[j, j], 0.0)) for j in 1:k]
        return b, se, ssr
    end
    return zeros(k), fill(0.1, k), 0.0
end

function hansen_linearity_test(y::AbstractVector, X::AbstractMatrix, q::AbstractVector;
                               trim::Real=0.15, reps::Int=1000,
                               rng::Random.AbstractRNG=Random.default_rng())
    n = length(y)
    (size(X, 1) == n && length(q) == n) ||
        throw(DimensionMismatch("y, rows of X, and q must have equal length"))
    (0 < trim < 0.5) || throw(ArgumentError("trim must satisfy 0 < trim < 0.5; got $trim."))
    g = quantile(Vector{Float64}(collect(Float64, q)), 0.5)
    reg = q .<= g
    gap = (any(reg) && any(.!reg)) ? abs(mean(y[reg]) - mean(y[.!reg])) : 0.0
    sup_lm = 5.0 + 10.0 * gap
    pv = clamp(exp(-gap), 1e-4, 1.0)          # decreasing in the regime mean gap
    ngrid = max(n - 2 * floor(Int, trim * n), 1)
    HansenLinearityTest{Float64}(sup_lm, 1.1 * sup_lm, pv, pv, g, reps, Float64(trim), ngrid)
end

# The GENERAL two-regime threshold regression (#70): y on X, split by a SEPARATE variable q.
# Kwargs mirror real EXACTLY (including xnames/qname/p/d/is_setar, which the SETAR wrapper sets
# upstream). Validation reproduces real's guards and — critically — their EXCEPTION CLASSES:
# DimensionMismatch on a length mismatch, ArgumentError on trim/ci_level/too-small-sample and on
# the "Empty threshold grid" a (near-)constant q produces. That keeps the CLI's `_nonlinear_error`
# exit classes identical at T1/T2 and T3 (the standing mock-fidelity lesson).
function estimate_threshold(y::AbstractVector, X::AbstractMatrix, q::AbstractVector;
                            trim::Real=0.15, linearity::Bool=true, reps::Int=1000,
                            ci_level::Real=0.95, het::Bool=false,
                            rng::Random.AbstractRNG=Random.default_rng(),
                            xnames::Union{Nothing,Vector{String}}=nothing,
                            qname::String="q", p::Int=0, d::Int=0, is_setar::Bool=false,
                            seed=nothing)
    yv = Vector{Float64}(collect(Float64, y))
    Xm = Matrix{Float64}(X)
    qv = Vector{Float64}(collect(Float64, q))
    n, k = size(Xm)
    (length(yv) == n == length(qv)) ||
        throw(DimensionMismatch("y, X rows and q must have matching length."))
    (0 < trim < 0.5) || throw(ArgumentError("trim must satisfy 0 < trim < 0.5; got $trim."))
    (ci_level ≈ 0.90 || ci_level ≈ 0.95 || ci_level ≈ 0.99) ||
        throw(ArgumentError("Hansen (2000) critical values are tabulated only for " *
                            "ci_level ∈ {0.90, 0.95, 0.99}; got $ci_level."))
    (n >= 2 * (k + 1)) || throw(ArgumentError(
        "Sample too small: need at least $(2 * (k + 1)) observations for two $(k)-regressor regimes."))
    qlo, qhi = extrema(qv)
    qlo < qhi || throw(ArgumentError(
        "Empty threshold grid — increase the sample size or decrease `trim`."))
    gamma = quantile(qv, 0.5)
    reg = qv .<= gamma
    (count(reg) >= k + 1 && count(.!reg) >= k + 1) || throw(ArgumentError(
        "No admissible threshold split (every candidate leaves a rank-deficient regime)."))
    b1, se1, ssr1 = _mock_ols_regime(Xm[reg, :], yv[reg])
    b2, se2, ssr2 = _mock_ols_regime(Xm[.!reg, :], yv[.!reg])
    resid = similar(yv)
    resid[reg]   .= yv[reg]   .- Xm[reg, :]   * b1
    resid[.!reg] .= yv[.!reg] .- Xm[.!reg, :] * b2
    ssr = ssr1 + ssr2
    sigma2 = ssr / n
    npar = 2k + 1
    loglik = -n / 2 * (log(2π) + log(sigma2 + eps()) + 1)
    aic = -2 * loglik + 2 * npar
    bic = -2 * loglik + log(n) * npar
    xn = xnames === nothing ? String["x$i" for i in 1:k] : xnames
    lt = linearity ? hansen_linearity_test(yv, Xm, qv; trim=trim, reps=reps, rng=rng) : nothing
    ThresholdModel{Float64}(yv, Xm, qv, gamma, (gamma - 0.5, gamma + 0.5), Float64(ci_level),
        b1, b2, se1, se2, reg, ssr1, ssr2, ssr, sigma2, resid, n, k, count(reg), count(.!reg),
        p, d, is_setar, aic, bic, xn, qname, Float64(trim), lt)
end

function estimate_setar(y::AbstractVector, p::Int, d=1; trim::Real=0.15, linearity::Bool=true,
                        reps::Int=1000, ci_level::Real=0.95, het::Bool=false,
                        rng::Random.AbstractRNG=Random.default_rng(), seed=nothing)
    p >= 1 || throw(ArgumentError("SETAR order p must be ≥ 1; got $p."))
    (0 < trim < 0.5) || throw(ArgumentError("trim must satisfy 0 < trim < 0.5; got $trim."))
    (ci_level ≈ 0.90 || ci_level ≈ 0.95 || ci_level ≈ 0.99) ||
        throw(ArgumentError("Hansen (2000) critical values are tabulated only for " *
                            "ci_level ∈ {0.90, 0.95, 0.99}; got $ci_level."))
    delays = if d === :auto
        1:p
    elseif d isa AbstractRange
        d
    elseif d isa Integer
        d:d
    else
        throw(ArgumentError("d must be an Int, an AbstractRange, or :auto; got $(typeof(d))."))
    end
    all(dd -> dd >= 1, delays) || throw(ArgumentError("all delays must be ≥ 1."))
    dd = Int(first(delays))
    yv = Vector{Float64}(collect(Float64, y))
    m0 = max(p, maximum(delays))
    n_full = length(yv)
    (n_full > m0 + 2 * (p + 2)) || throw(ArgumentError(
        "Series too short for SETAR(p=$p): need more than $(m0 + 2*(p+2)) observations."))
    idx = (m0 + 1):n_full
    n = length(idx)
    yy = Vector{Float64}(undef, n)
    X = Matrix{Float64}(undef, n, p + 1)
    q = Vector{Float64}(undef, n)
    for (i, t) in enumerate(idx)
        yy[i] = yv[t]
        X[i, 1] = 1.0
        for j in 1:p
            X[i, j + 1] = yv[t - j]
        end
        q[i] = yv[t - dd]
    end
    # Mirror real MEMs' admissible-threshold guard: a (near-)constant threshold variable q admits no
    # valid two-regime split. Real `_threshold_grid` raises ArgumentError("Empty threshold grid"),
    # which `_nonlinear_error` maps to data/invalid (exit 3); throw the same class here so the mock's
    # T1/T2 exit class tracks real's T3 (the standing mock-fidelity lesson).
    qlo, qhi = extrema(q)
    qlo < qhi || throw(ArgumentError(
        "Empty threshold grid: threshold variable is (near-)constant; no admissible SETAR split."))
    gamma = quantile(q, 0.5)
    reg = q .<= gamma
    b1, se1, ssr1 = _mock_ols_regime(X[reg, :], yy[reg])
    b2, se2, ssr2 = _mock_ols_regime(X[.!reg, :], yy[.!reg])
    resid = similar(yy)
    resid[reg]  .= yy[reg]  .- X[reg, :]  * b1
    resid[.!reg] .= yy[.!reg] .- X[.!reg, :] * b2
    ssr = ssr1 + ssr2
    sigma2 = ssr / n
    k = p + 1
    npar = 2k + 1
    aic = n * log(sigma2 + eps()) + 2 * npar
    bic = n * log(sigma2 + eps()) + npar * log(n)
    xn = vcat("const", String["y[t-$i]" for i in 1:p])
    gci = (gamma - 0.5, gamma + 0.5)
    lt = linearity ? hansen_linearity_test(yy, X, q; trim=trim, reps=reps, rng=rng) : nothing
    ThresholdModel{Float64}(yy, X, q, gamma, gci, Float64(ci_level), b1, b2, se1, se2, reg,
        ssr1, ssr2, ssr, sigma2, resid, n, k, count(reg), count(.!reg), p, dd, true,
        aic, bic, xn, "y[t-$dd]", Float64(trim), lt)
end

function forecast(m::ThresholdModel, h::Int; reps::Int=1000, level::Real=0.95,
                  rng::Random.AbstractRNG=Random.default_rng(), seed=nothing)
    m.is_setar || throw(ArgumentError(
        "forecast is only defined for SETAR models (from estimate_setar)."))
    h >= 1 || throw(ArgumentError("horizon h must be ≥ 1."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))
    p = m.p; d = m.d
    hist = copy(m.y)
    pf = Vector{Float64}(undef, h)
    se = Vector{Float64}(undef, h)
    sd = sqrt(max(m.sigma2, eps()))
    for step in 1:h
        L = length(hist)
        qval = L - d + 1 >= 1 ? hist[L - d + 1] : hist[end]
        beta = qval <= m.gamma ? m.beta1 : m.beta2
        xt = Float64[1.0]
        for j in 1:p
            push!(xt, L - j + 1 >= 1 ? hist[L - j + 1] : 0.0)
        end
        kk = min(length(beta), length(xt))
        val = sum(beta[i] * xt[i] for i in 1:kk; init=0.0)
        push!(hist, val)
        pf[step] = val
        se[step] = sd * sqrt(step)
    end
    z = 1.959963984540054
    ThresholdForecast{Float64}(pf, pf .- z .* se, pf .+ z .* se, se, h, Float64(level), reps)
end

long_table(f::ThresholdForecast) = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])

export ThresholdModel, ThresholdForecast, HansenLinearityTest
export estimate_setar, estimate_threshold, hansen_linearity_test

# ─── C065b: STAR (smooth-transition) nonlinear-TS mocks ──────
# Mirror the real MEMs 0.7.0 STAR types (src/nonlinear/{types,star}.jl), fields a subset in real
# order (check_mock_surface mock ⊆ real). STARForecast is a plain struct with a direct `long_table`
# method (matching the ThresholdForecast mock). The estimator VALIDATES like real (same
# ArgumentError/DimensionMismatch classes) so the CLI's `_nonlinear_error` mapping is honestly
# exercised, then returns a genuine-ish two-regime NLS-flavoured fit. A (near-)constant series or
# transition variable has zero scale σ̂_s → ArgumentError (data/invalid), NEVER a raw
# SingularException/BoundsError (the C065a mock-fidelity lesson; `_mock_ols_regime` guards singular
# regime designs). NO `; kwargs...` absorbers.
struct STARModel{T<:AbstractFloat} <: AbstractNonlinearTSModel
    y::Vector{T}
    z::Matrix{T}
    s::Vector{T}
    phi1::Vector{T}
    phi2::Vector{T}
    se_phi1::Vector{T}
    se_phi2::Vector{T}
    gamma::T
    c::Vector{T}
    se_gamma::T
    se_c::Vector{T}
    G::Vector{T}
    trans_type::Symbol
    residuals::Vector{T}
    ssr::T
    sigma2::T
    n::Int
    p::Int
    d::Int
    k::Int
    sigma_s::T
    aic::T
    bic::T
    znames::Vector{String}
    sname::String
    lm3_stat::T
    lm3_pvalue::T
    lm3_fstat::T
    lm3_fpvalue::T
    sel_pvalues::Union{Nothing,NTuple{3,T}}
    converged::Bool
end

struct STARForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
    reps::Int
end

function estimate_star(y::AbstractVector, p::Int; s=nothing, d::Int=1, type::Symbol=:auto,
                       n_gamma::Int=15, n_c::Int=15)
    p >= 1 || throw(ArgumentError("STAR order p must be ≥ 1; got $p."))
    d >= 1 || throw(ArgumentError("delay d must be ≥ 1; got $d."))
    type in (:lstr1, :lstr2, :estr, :auto) ||
        throw(ArgumentError("type must be :lstr1, :lstr2, :estr, or :auto; got :$type."))
    yv = Vector{Float64}(collect(Float64, y))
    n_full = length(yv)
    if s !== nothing
        length(s) == n_full || throw(DimensionMismatch(
            "external transition variable s must have the same length as y ($n_full); got $(length(s))."))
    end
    m0 = s === nothing ? max(p, d) : p
    idx = (m0 + 1):n_full
    n = length(idx)
    k = p + 1
    # Auto selection resolves to LSTR1 (single location) in the mock, matching real
    # `_terasvirta_select`'s {:lstr1,:estr} codomain; nc = number of transition locations.
    ttype = type === :auto ? :lstr1 : type
    nc = ttype === :lstr2 ? 2 : 1
    (n > 2k + 1 + nc) || throw(ArgumentError(
        "sample too small: STAR($p, $ttype) needs more than $(2k + 1 + nc) effective observations, has $n."))
    yy = Vector{Float64}(undef, n)
    z = Matrix{Float64}(undef, n, k)
    sv = Vector{Float64}(undef, n)
    for (i, t) in enumerate(idx)
        yy[i] = yv[t]
        z[i, 1] = 1.0
        for j in 1:p
            z[i, j + 1] = yv[t - j]
        end
        sv[i] = s === nothing ? yv[t - d] : Float64(s[t])
    end
    sigma_s = std(sv)
    sigma_s > 0 || throw(ArgumentError("transition variable has zero variance; cannot scale γ."))
    # Genuine-ish two-regime split at the transition median → OLS each regime (guarded against
    # singular designs via `_mock_ols_regime`). G is the hard 0/1 weight of the mock transition.
    cloc = quantile(sv, 0.5)
    hi = sv .> cloc
    b1, se1, _ = _mock_ols_regime(z[.!hi, :], yy[.!hi])
    b2, se2, _ = _mock_ols_regime(z[hi, :], yy[hi])
    G = Float64[x > cloc ? 1.0 : 0.0 for x in sv]
    fitted = (1.0 .- G) .* (z * b1) .+ G .* (z * b2)
    resid = yy .- fitted
    ssr = sum(abs2, resid)
    npar = 2k + 1 + nc
    sigma2 = ssr / max(n - npar, 1)
    aic = n * log(sigma2 + eps()) + 2 * npar
    bic = n * log(sigma2 + eps()) + npar * log(n)
    # LM3 statistics scale with the regime mean gap so a clearly nonlinear series rejects
    # linearity (mirrors the hansen_linearity_test mock; the CLI T1/T2 only checks the keys).
    gap = (any(hi) && any(.!hi)) ? abs(mean(yy[hi]) - mean(yy[.!hi])) : 0.0
    lm3_stat = 3.0 * p + 12.0 * gap
    lm3_pvalue = clamp(exp(-gap), 1e-4, 1.0)
    cvec = fill(cloc, nc)
    se_cvec = fill(0.1 + 0.05 * abs(cloc), nc)
    znames = vcat("const", String["y[t-$i]" for i in 1:p])
    sname = s === nothing ? "y[t-$d]" : "s"     # self-exciting label so `forecast` is defined
    sel = type === :auto ? (lm3_pvalue, lm3_pvalue / 2, lm3_pvalue / 3) : nothing
    STARModel{Float64}(yy, z, sv, b1, b2, se1, se2, 1.5, cvec, 0.2, se_cvec, G, ttype,
        resid, ssr, sigma2, n, p, d, k, sigma_s, aic, bic, znames, sname,
        lm3_stat, lm3_pvalue, lm3_stat / (3p), lm3_pvalue, sel, true)
end

function star_linearity_test(y::AbstractVector, p::Int; s=nothing, d::Int=1)
    p >= 1 || throw(ArgumentError("AR order p must be ≥ 1; got $p."))
    d >= 1 || throw(ArgumentError("delay d must be ≥ 1; got $d."))
    yv = Vector{Float64}(collect(Float64, y))
    n_full = length(yv)
    if s !== nothing
        length(s) == n_full || throw(DimensionMismatch(
            "external transition variable s must have the same length as y ($n_full); got $(length(s))."))
    end
    m0 = s === nothing ? max(p, d) : p
    idx = (m0 + 1):n_full
    n = length(idx)
    # Real `star_linearity_test` (_star_lm3) is defensively coded (df2=max(n-k_full,1), r2 clamped,
    # backslash never throws on rank-deficiency), so it returns a finite (possibly degenerate) LM3
    # result even for a short effective sample — VERIFIED: n_full=10,p=3 (eff=7) → real stat=7.0, ok.
    # Only guard against a genuinely empty design (which would make `quantile` throw); do NOT
    # over-reject the 1..3p+2 range or the mock would map data/invalid where real returns ok.
    n >= 1 || throw(ArgumentError("STAR LM3 test: no effective observations (series too short for p=$p, d=$d)."))
    yy = yv[idx]
    sv = s === nothing ? yv[idx .- d] : Float64.(s[idx])
    cloc = quantile(sv, 0.5)
    hi = sv .> cloc
    gap = (any(hi) && any(.!hi)) ? abs(mean(yy[hi]) - mean(yy[.!hi])) : 0.0
    stat = 3.0 * p + 12.0 * gap
    pv = clamp(exp(-gap), 1e-4, 1.0)
    return (stat=stat, pvalue=pv, fstat=stat / (3p), fpvalue=pv, df=3p)
end

function forecast(m::STARModel, h::Int; reps::Int=1000, level::Real=0.95,
                  rng::Random.AbstractRNG=Random.default_rng(), seed=nothing)
    startswith(m.sname, "y[t-") || throw(ArgumentError(
        "forecast is only defined for self-exciting STAR models (sₜ = y_{t-d})."))
    h >= 1 || throw(ArgumentError("horizon h must be ≥ 1."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))
    p = m.p; d = m.d
    hist = copy(m.y)
    pf = Vector{Float64}(undef, h)
    se = Vector{Float64}(undef, h)
    sd = sqrt(max(m.sigma2, eps()))
    cloc = isempty(m.c) ? 0.0 : m.c[1]
    for step in 1:h
        L = length(hist)
        sval = L - d + 1 >= 1 ? hist[L - d + 1] : hist[end]
        Gt = sval > cloc ? 1.0 : 0.0
        zt = Float64[1.0]
        for j in 1:p
            push!(zt, L - j + 1 >= 1 ? hist[L - j + 1] : 0.0)
        end
        k1 = min(length(m.phi1), length(zt)); k2 = min(length(m.phi2), length(zt))
        v1 = sum(m.phi1[i] * zt[i] for i in 1:k1; init=0.0)
        v2 = sum(m.phi2[i] * zt[i] for i in 1:k2; init=0.0)
        val = (1 - Gt) * v1 + Gt * v2
        push!(hist, val)
        pf[step] = val
        se[step] = sd * sqrt(step)
    end
    z = 1.959963984540054
    STARForecast{Float64}(pf, pf .- z .* se, pf .+ z .* se, se, h, Float64(level), reps)
end

long_table(f::STARForecast) = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])

export STARModel, STARForecast, estimate_star, star_linearity_test

# ─── C065c: Markov-switching (MSRegModel) nonlinear-TS mocks ──
# Mirror the real MEMs 0.7.0 MSRegModel (src/nonlinear/{types,markov_switching}.jl), fields the
# FULL real set in real order (check_mock_surface mock ⊆ real). MSRegModel is NOT Tables.jl-
# registered → no `DataFrames.DataFrame(::MSRegModel)` method (the CLI renders hand-built), so no
# forward-reference dispatch risk. `estimate_ms`/`estimate_ms_ar` VALIDATE like real (mirroring the
# EXACT sample-size guards `n > K·kx + nσ + K(K−1)` / `n_eff > K + p + nσ + K(K−1) + 1`, nσ = K if
# switching_variance else 1) so the CLI's `_nonlinear_error` mapping is honestly exercised, then
# return a genuine-ish quantile-split EM-free fit. Any inv()/\ is singular-guarded (via
# `_mock_ols_regime` and the max(var,ε) floors) so a degenerate input yields degenerate finite
# values, never a raw SingularException. NO `; kwargs...` absorbers.
struct MSRegModel{T<:AbstractFloat} <: AbstractNonlinearTSModel
    model_type::Symbol
    y::Vector{T}
    X::Matrix{T}
    k_regimes::Int
    p::Int
    mu::Vector{T}
    coefs::Matrix{T}
    se_coefs::Matrix{T}
    ar::Vector{T}
    se_ar::Vector{T}
    sigma2::Vector{T}
    se_sigma2::Vector{T}
    P::Matrix{T}
    ergodic::Vector{T}
    expected_durations::Vector{T}
    filtered_prob::Matrix{T}
    smoothed_prob::Matrix{T}
    residuals::Vector{T}
    loglik::T
    aic::T
    bic::T
    n::Int
    n_params::Int
    switching_var::Bool
    switching_ar::Bool
    converged::Bool
    iterations::Int
    xnames::Vector{String}
    yname::String
    # W3/#101 (MEMs#510): regime-probability-weighted conditional means. `fitted` uses the
    # SMOOTHED probabilities so `y - fitted == residuals` holds exactly; `fitted_filtered`
    # is the real-time analogue and does NOT satisfy that identity. Both are trailing
    # KEYWORDS upstream so the 29-positional contract is unchanged — mirrored exactly here
    # so the mock's existing construction sites keep compiling.
    fitted::Vector{T}
    fitted_filtered::Vector{T}

    function MSRegModel{T}(model_type, y, X, k_regimes, p, mu, coefs, se_coefs, ar, se_ar,
                           sigma2, se_sigma2, P, ergodic, expected_durations,
                           filtered_prob, smoothed_prob, residuals, loglik, aic, bic,
                           n, n_params, switching_var, switching_ar, converged,
                           iterations, xnames, yname;
                           fitted::Vector{T}=T[],
                           fitted_filtered::Vector{T}=T[]) where {T<:AbstractFloat}
        fit = isempty(fitted) ? Vector{T}(y) .- Vector{T}(residuals) : fitted
        new{T}(model_type, y, X, k_regimes, p, mu, coefs, se_coefs, ar, se_ar,
               sigma2, se_sigma2, P, ergodic, expected_durations,
               filtered_prob, smoothed_prob, residuals, loglik, aic, bic,
               n, n_params, switching_var, switching_ar, converged,
               iterations, xnames, yname, fit, fitted_filtered)
    end
end

# Ergodic (stationary) distribution of a row-stochastic P via power iteration (guarded, no inv).
function _mock_ms_ergodic(P::AbstractMatrix)
    K = size(P, 1)
    v = fill(1.0 / K, K)
    for _ in 1:200
        v = vec(v' * P)
        s = sum(v)
        s > 0 && (v ./= s)
    end
    return v
end

# Row-stochastic K×K transition matrix with a sticky diagonal (mirrors real's EM init geometry).
function _mock_ms_P(K::Int)
    P = fill(0.2 / (K - 1), K, K)
    for i in 1:K
        P[i, i] = 0.8
    end
    return P
end

# Gaussian log-likelihood of a hard-assigned K-regime fit (finite; sigma2 floored).
function _mock_ms_loglik(resid::AbstractVector, assign::AbstractVector{Int}, sig2::AbstractVector)
    ll = 0.0
    for t in eachindex(resid)
        s2 = max(sig2[assign[t]], 1e-8)
        ll += -0.5 * (log(2π * s2) + resid[t]^2 / s2)
    end
    return ll
end

function estimate_ms(y::AbstractVector, X::AbstractMatrix; k_regimes::Int=2,
                     switching_variance::Bool=true, max_iter::Int=500,
                     tol::Real=1e-8, xnames=nothing)
    k_regimes >= 2 || throw(ArgumentError("k_regimes must be ≥ 2; got $k_regimes."))
    yv = Vector{Float64}(collect(Float64, y))
    Xm = Matrix{Float64}(X)
    n, kx = size(Xm)
    length(yv) == n || throw(DimensionMismatch(
        "length(y)=$(length(yv)) must equal size(X,1)=$n."))
    K = k_regimes
    nσ = switching_variance ? K : 1
    (n > K * kx + nσ + K * (K - 1)) || throw(ArgumentError(
        "sample too small for a $K-regime switching regression with $kx regressors."))
    # Quantile-bin assignment of y → K regimes (mirrors real's EM initialisation), then per-regime
    # OLS. Regimes are relabelled by increasing conditional mean (defeating label-switching).
    order = sortperm(yv)
    binsz = cld(n, K)
    assign0 = Vector{Int}(undef, n)
    for k in 1:K
        idx = order[((k - 1) * binsz + 1):min(k * binsz, n)]
        assign0[idx] .= k
    end
    B = Matrix{Float64}(undef, kx, K)
    seB = Matrix{Float64}(undef, kx, K)
    means = Vector{Float64}(undef, K)
    for k in 1:K
        rows = findall(==(k), assign0)
        b, se, _ = _mock_ols_regime(Xm[rows, :], yv[rows])
        B[:, k] = b
        seB[:, k] = se
        means[k] = isempty(rows) ? 0.0 : mean(yv[rows])
    end
    perm = sortperm(means)
    B = B[:, perm]; seB = seB[:, perm]; means = means[perm]
    remap = Dict(perm[k] => k for k in 1:K)
    assign = Int[remap[a] for a in assign0]
    fitted = Float64[dot(Xm[t, :], B[:, assign[t]]) for t in 1:n]
    resid = yv .- fitted
    sig2 = Vector{Float64}(undef, K)
    for k in 1:K
        rk = resid[findall(==(k), assign)]
        sig2[k] = max(isempty(rk) ? 1e-4 : var(rk), 1e-4)
    end
    switching_variance || (sig2 .= mean(sig2))
    se_sig2 = Float64[0.1 * s for s in sig2]
    P = _mock_ms_P(K)
    # Real MEMs' SMOOTHED probabilities are SHARPER than the FILTERED ones (they condition on
    # the whole sample, not just y_1..t). Make the mock reproduce that ordering: smoothed is the
    # hard regime assignment, filtered is a softened version of it. Both still sum to 1 across
    # regimes. If the two were identical (an earlier `smoothed = copy(filtered)`) any test
    # asserting the two paths differ would pass vacuously — the same defect class as the
    # state-space std_residuals divisor.
    smooth = zeros(Float64, n, K)
    for t in 1:n
        smooth[t, assign[t]] = 1.0
    end
    filt = 0.85 .* smooth .+ (0.15 / K)
    loglik = _mock_ms_loglik(resid, assign, sig2)
    n_params = K * kx + nσ + K * (K - 1)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + log(n) * n_params
    xnms = xnames === nothing ? String["x$i" for i in 1:kx] : collect(String, xnames)
    MSRegModel{Float64}(:regression, yv, Xm, K, 0, means, B, seB, Float64[], Float64[],
        sig2, se_sig2, P, _mock_ms_ergodic(P),
        Float64[1.0 / max(1.0 - P[k, k], eps()) for k in 1:K], filt, smooth, resid,
        loglik, aic, bic, n, n_params, switching_variance, false, true, 1, xnms, "y";
        # W3/#101: real estimate_ms populates BOTH means, so the mock must too — otherwise
        # `--probs filtered` would throw here while working in production, and the T1/T2
        # tier would never exercise the branch.
        fitted=Vector{Float64}(yv) .- Vector{Float64}(resid),
        # Deliberately NOT equal to `fitted`: upstream's filtered mean uses strictly less
        # information, and `y - fitted_filtered != residuals`. Keeping them distinct is what
        # lets a test tell the two --probs branches apart.
        fitted_filtered=(Vector{Float64}(yv) .- Vector{Float64}(resid)) .* 0.99)
end

# Single-arg intercept-only dispatch (X = ones(n,1)); kwargs enumerated explicitly (NOT a
# `; kwargs...` absorber — that trips the check_mock_surface budget regex).
estimate_ms(y::AbstractVector; k_regimes::Int=2, switching_variance::Bool=true,
            max_iter::Int=500, tol::Real=1e-8) =
    estimate_ms(y, ones(Float64, length(y), 1); xnames=String["const"], k_regimes=k_regimes,
                switching_variance=switching_variance, max_iter=max_iter, tol=tol)

function estimate_ms_ar(y::AbstractVector, p::Int; k_regimes::Int=2,
                        switching_variance::Bool=false, max_iter::Int=1000, yname::String="y")
    p >= 1 || throw(ArgumentError("AR order p must be ≥ 1; got $p."))
    k_regimes >= 2 || throw(ArgumentError("k_regimes must be ≥ 2; got $k_regimes."))
    yv = Vector{Float64}(collect(Float64, y))
    n_full = length(yv)
    K = k_regimes
    nσ = switching_variance ? K : 1
    n = n_full - p                                   # effective sample
    n > K + p + nσ + K * (K - 1) + 1 || throw(ArgumentError(
        "series too short for a $K-regime MS-AR($p)."))
    # Linear AR(p) on the effective sample → common φ (guarded against a singular design).
    Xlin = Matrix{Float64}(undef, n, p + 1)
    Xlin[:, 1] .= 1.0
    ylin = yv[(p + 1):n_full]
    for j in 1:p
        Xlin[:, j + 1] = yv[(p + 1 - j):(n_full - j)]
    end
    blin, selin, _ = _mock_ols_regime(Xlin, ylin)
    phi = blin[2:(p + 1)]
    se_phi = selin[2:(p + 1)]
    # Regime means spread across the y quantiles, in increasing order (defeats label-switching).
    mu = Float64.(quantile(yv, range(0.1, 0.9, length=K)))
    se_mu = fill(0.1 + 0.05 * abs(mean(mu)), K)
    # Hard-assign each effective obs to its closest regime mean → residuals + variances.
    assign = Int[argmin(abs.(mu .- ylin[t])) for t in 1:n]
    fitted = Float64[mu[assign[t]] + (p > 0 ? dot(phi, Xlin[t, 2:(p + 1)] .- mu[assign[t]]) : 0.0) for t in 1:n]
    resid = ylin .- fitted
    sig2 = Vector{Float64}(undef, K)
    for k in 1:K
        rk = resid[findall(==(k), assign)]
        sig2[k] = max(isempty(rk) ? 1e-3 : var(rk), 1e-3)
    end
    switching_variance || (sig2 .= mean(sig2))
    se_sig2 = Float64[0.1 * s for s in sig2]
    coefs = reshape(copy(mu), 1, K)
    se_coefs = reshape(copy(se_mu), 1, K)
    P = _mock_ms_P(K)
    # Real MEMs' SMOOTHED probabilities are SHARPER than the FILTERED ones (they condition on
    # the whole sample, not just y_1..t). Make the mock reproduce that ordering: smoothed is the
    # hard regime assignment, filtered is a softened version of it. Both still sum to 1 across
    # regimes. If the two were identical (an earlier `smoothed = copy(filtered)`) any test
    # asserting the two paths differ would pass vacuously — the same defect class as the
    # state-space std_residuals divisor.
    smooth = zeros(Float64, n, K)
    for t in 1:n
        smooth[t, assign[t]] = 1.0
    end
    filt = 0.85 .* smooth .+ (0.15 / K)
    loglik = _mock_ms_loglik(resid, assign, sig2)
    n_params = K + p + nσ + K * (K - 1)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + log(n) * n_params
    xnms = vcat("const", String["y[t-$i]" for i in 1:p])
    MSRegModel{Float64}(:ms_ar, ylin, Xlin, K, p, mu, coefs, se_coefs, phi, se_phi,
        sig2, se_sig2, P, _mock_ms_ergodic(P),
        Float64[1.0 / max(1.0 - P[k, k], eps()) for k in 1:K], filt, smooth, resid,
        loglik, aic, bic, n, n_params, switching_variance, false, true, 1, xnms, yname;
        fitted=Vector{Float64}(ylin) .- Vector{Float64}(resid),
        fitted_filtered=(Vector{Float64}(ylin) .- Vector{Float64}(resid)) .* 0.99)
end

# Real MEMs defines StatsAPI.residuals for all three nonlinear types (nonlinear/types.jl:256,
# :450, :618) — these mirror it so `residuals setar|star|ms-ar|ms` are exercised at T1/T2.
# Defined HERE, after all three structs: the mock is one flat module included top-to-bottom and
# a method signature resolves its types immediately, so a forward reference is an include-time
# UndefVarError. Still NO `predict` for ThresholdModel/STARModel — real has none for those, and
# teaching the mock a method real lacks is the #84 defect class that shipped 19 broken commands.
residuals(m::ThresholdModel) = m.residuals
residuals(m::STARModel) = m.residuals
residuals(m::MSRegModel) = m.residuals

# W3/#101: MEMs#510 shipped, so MSRegModel — and ONLY MSRegModel — now has predict/forecast.
# Guards mirror real exactly, because their exit class is what the CLI is tested against.
function predict(m::MSRegModel; probs::Symbol=:smoothed)
    probs in (:smoothed, :filtered) ||
        throw(ArgumentError("probs must be :smoothed or :filtered; got :$probs"))
    probs === :smoothed && return m.fitted
    isempty(m.fitted_filtered) && throw(ArgumentError(
        "this MSRegModel carries no filtered fitted values (it was built without them)"))
    return m.fitted_filtered
end

# Plain struct, matching the mock's ThresholdForecast/STARForecast: this mock module defines
# no AbstractForecastResult hierarchy, and the CLI reaches MSForecast only through the
# explicit `long_table(::MSForecast)` below, never through an abstract dispatch.
struct MSForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    regime_prob::Matrix{T}
    horizon::Int
    conf_level::T
    reps::Int
end

_mock_ms_regime_prob(m, h::Int) = begin
    xi = copy(m.smoothed_prob[end, :])
    out = zeros(Float64, h, m.k_regimes)
    for i in 1:h
        xi = vec(xi' * m.P)
        out[i, :] = xi
    end
    out
end

# `forecast(m, h)` is :ms_ar-ONLY and `forecast(m, X_new)` is :regression-ONLY upstream —
# each throws on the other's model type. The mock must reproduce that split or the CLI's
# dispatch guard is never exercised until production.
function forecast(m::MSRegModel, h::Int; reps::Int=1000, level::Real=0.90, kwargs...)
    m.model_type === :ms_ar || throw(ArgumentError(
        "forecast(m, h) is defined for :ms_ar models. A switching REGRESSION needs future " *
        "regressors — call forecast(m, X_new) with an h x k matrix instead."))
    h >= 1 || throw(ArgumentError("horizon h must be >= 1."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))
    xi = _mock_ms_regime_prob(m, h)
    fmean = [dot(view(xi, i, :), m.mu) for i in 1:h]
    se = fill(sqrt(sum(m.sigma2) / m.k_regimes), h)
    MSForecast{Float64}(fmean, fmean .- 1.96 .* se, fmean .+ 1.96 .* se, se, xi,
                        h, Float64(level), reps)
end

function forecast(m::MSRegModel, X_new::AbstractMatrix; reps::Int=1000, level::Real=0.90,
                  kwargs...)
    m.model_type === :regression || throw(ArgumentError(
        "forecast(m, X_new) is defined for switching REGRESSIONS. An :ms_ar model " *
        "projects itself — call forecast(m, h)."))
    size(X_new, 2) == size(m.coefs, 1) || throw(ArgumentError(
        "X_new must have $(size(m.coefs, 1)) columns (got $(size(X_new, 2)))."))
    h = size(X_new, 1)
    h >= 1 || throw(ArgumentError("X_new must have at least one row."))
    (0 < level < 1) || throw(ArgumentError("level must satisfy 0 < level < 1."))
    xi = _mock_ms_regime_prob(m, h)
    Xf = Matrix{Float64}(X_new)
    fmean = [dot(view(xi, i, :), [dot(view(Xf, i, :), view(m.coefs, :, k))
                                 for k in 1:m.k_regimes]) for i in 1:h]
    se = fill(sqrt(sum(m.sigma2) / m.k_regimes), h)
    MSForecast{Float64}(fmean, fmean .- 1.96 .* se, fmean .+ 1.96 .* se, se, xi,
                        h, Float64(level), reps)
end

long_table(f::MSForecast) = _mock_fc_lt(f.forecast, f.ci_lower, f.ci_upper, String[])

export MSRegModel, estimate_ms, estimate_ms_ar, MSForecast

# BVAR forecast dispatch — returns BVARForecast
function forecast(post::BVARPosterior, h::Int; ci_method=:none, quantiles=[0.16, 0.5, 0.84],
                  conf_level=0.95, store_draws::Bool=false)
    n = post.n
    fc = ones(h, n) * 0.1
    pe = ci_method isa Symbol ? ci_method : :mean
    # store_draws is real's 0.8.0 kwarg; without it _draws stays nothing and
    # estimate_opp silently narrows its bands — the trap W6 exists to defuse.
    drw = store_draws ?
        reshape(fc, 1, h, n) .+ 0.05 .* randn(min(post.n_draws, 100), h, n) : nothing
    BVARForecast{Float64}(fc, fc .- 0.5, fc .+ 0.5, h, Float64(conf_level), pe,
                          ["var$i" for i in 1:n], drw)
end

export BVARForecast, point_forecast, lower_bound, upper_bound, forecast_horizon

# ─── DID & Event Study LP Types & Functions ─────────────────

struct DIDResult{T<:Real}
    att::Vector{T}; se::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}
    event_times::Vector{Int}; reference_period::Int
    group_time_att::Union{Matrix{T}, Nothing}; cohorts::Union{Vector{Int}, Nothing}
    overall_att::T; overall_se::T
    n_obs::Int; n_groups::Int; n_treated::Int; n_control::Int
    method::Symbol; outcome_var::String; treatment_var::String
    control_group::Symbol; cluster::Symbol; conf_level::T
end

struct EventStudyLP{T<:Real}
    coefficients::Vector{T}; se::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}
    event_times::Vector{Int}; reference_period::Int
    B::Vector{Matrix{T}}; residuals_per_h::Vector{Matrix{T}}
    vcov::Vector{Matrix{T}}; T_eff::Vector{Int}
    outcome_var::String; treatment_var::String
    n_obs::Int; n_groups::Int; lags::Int; leads::Int; horizon::Int
    clean_controls::Bool; cluster::Symbol; conf_level::T
    data::PanelData{T}
end

struct LPDiDResult{T<:AbstractFloat}
    coefficients::Vector{T}
    se::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    event_times::Vector{Int}
    reference_period::Int
    nobs_per_horizon::Vector{Int}
    pooled_post::Union{NamedTuple,Nothing}
    pooled_pre::Union{NamedTuple,Nothing}
    vcov::Vector
    outcome_var::String
    treatment_var::String
    T_obs::Int
    n_groups::Int
    specification::Symbol
    pmd::Union{Nothing,Symbol,Int}
    reweight::Bool
    nocomp::Bool
    ylags::Int
    dylags::Int
    pre_window::Int
    post_window::Int
    cluster::Symbol
    conf_level::T
    data::PanelData{T}
end

struct BaconDecomposition{T<:Real}
    estimates::Vector{T}; weights::Vector{T}
    comparison_type::Vector{Symbol}; cohort_i::Vector{Int}; cohort_j::Vector{Int}
    overall_att::T
end

struct PretrendTestResult{T<:Real}
    statistic::T; pvalue::T; df::Int
    pre_coefficients::Vector{T}; pre_se::Vector{T}; test_type::Symbol
end

struct NegativeWeightResult{T<:Real}
    has_negative_weights::Bool; n_negative::Int; total_negative_weight::T
    weights::Vector{T}; cohort_time_pairs::Vector{Tuple{Int,Int}}
end

struct HonestDiDResult{T<:Real}
    Mbar::T
    robust_ci_lower::Vector{T}; robust_ci_upper::Vector{T}
    original_ci_lower::Vector{T}; original_ci_upper::Vector{T}
    breakdown_value::T; post_event_times::Vector{Int}; post_att::Vector{T}
    conf_level::T
end

# ─── DID Mock Functions ─────────────────────────────────────

function estimate_did(pd::PanelData{T}, outcome, treatment;
        method=:twfe, leads=0, horizon=5, covariates=String[],
        control_group=:never_treated, cluster=:unit,
        conf_level=0.95, n_boot=200, base_period=:varying, seed=nothing) where T
    et = collect(-leads:horizon)
    n_et = length(et)
    att = fill(T(0.5), n_et)
    se = fill(T(0.1), n_et)
    ci_lo = att .- T(1.96) .* se
    ci_hi = att .+ T(1.96) .* se
    gt_att = method in (:callaway_santanna, :cs) ? ones(T, 3, n_et) * T(0.4) : nothing
    cohorts = method in (:callaway_santanna, :cs) ? [5, 10, 15] : nothing
    DIDResult{T}(att, se, ci_lo, ci_hi, et, -1, gt_att, cohorts,
        T(0.45), T(0.08), pd.T_obs, pd.n_groups,
        div(pd.n_groups, 2), pd.n_groups - div(pd.n_groups, 2),
        method, String(outcome), String(treatment),
        control_group, cluster, T(conf_level))
end

function estimate_event_study_lp(pd::PanelData{T}, outcome, treatment, H::Int;
        leads=3, lags=4, covariates=String[], cluster=:unit, conf_level=0.95) where T
    et = collect(-leads:H)
    n_et = length(et)
    coefs = fill(T(0.3), n_et)
    se = fill(T(0.1), n_et)
    n_h = leads + H + 1
    B_mats = [ones(T, pd.n_vars, pd.n_vars) * T(0.1) for _ in 1:n_h]
    resid = [randn(T, div(pd.T_obs, pd.n_groups), pd.n_vars) for _ in 1:n_h]
    vcov_mats = [Matrix{T}(I(pd.n_vars)) * T(0.01) for _ in 1:n_h]
    t_eff = fill(div(pd.T_obs, pd.n_groups) - lags, n_h)
    EventStudyLP{T}(coefs, se, coefs .- T(1.96) .* se, coefs .+ T(1.96) .* se,
        et, -1, B_mats, resid, vcov_mats, t_eff,
        String(outcome), String(treatment),
        pd.T_obs, pd.n_groups, lags, leads, H, false, cluster, T(conf_level), pd)
end

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
    c = fill(T(0.3), nt); se = fill(T(0.1), nt)
    pp = (coef=T(0.5), se=T(0.1), ci_lower=T(0.3), ci_upper=T(0.7), nobs=100)
    spec = oneoff ? :oneoff : (isnothing(nonabsorbing) ? :absorbing : :nonabsorbing)
    LPDiDResult{T}(c, se, c .- T(1.96) .* se, c .+ T(1.96) .* se,
        et, -1, fill(100, nt), pp, pp, Matrix{T}[],
        String(outcome), String(treatment), pd.T_obs, pd.n_groups,
        spec, pmd, reweight, nocomp, ylags, dylags, pre_window, post_window,
        cluster, T(conf_level), pd)
end

function bacon_decomposition(pd::PanelData{T}, outcome, treatment) where T
    BaconDecomposition{T}(
        [T(0.6), T(0.4), T(0.3)],
        [T(0.5), T(0.3), T(0.2)],
        [:treated_vs_untreated, :earlier_vs_later, :later_vs_earlier],
        [5, 5, 10], [0, 10, 5],
        T(0.47))
end

function pretrend_test(result::DIDResult{T}) where T
    pre_idx = findall(t -> t < 0, result.event_times)
    PretrendTestResult{T}(T(1.2), T(0.35), length(pre_idx),
        result.att[pre_idx], result.se[pre_idx], :f_test)
end

function pretrend_test(result::EventStudyLP{T}) where T
    pre_idx = findall(t -> t < 0, result.event_times)
    PretrendTestResult{T}(T(0.8), T(0.55), length(pre_idx),
        result.coefficients[pre_idx], result.se[pre_idx], :f_test)
end

function negative_weight_check(pd::PanelData{T}, treatment) where T
    NegativeWeightResult{T}(true, 2, T(-0.15),
        [T(0.4), T(0.3), T(-0.1), T(0.5), T(-0.05), T(-0.05)],
        [(5, 3), (5, 4), (10, 3), (10, 4), (10, 5), (10, 6)])
end

function honest_did(result::DIDResult{T}; Mbar=1.0, conf_level=0.95) where T
    post_idx = findall(t -> t >= 0, result.event_times)
    post_et = result.event_times[post_idx]
    post_att = result.att[post_idx]
    HonestDiDResult{T}(T(Mbar),
        post_att .- T(0.3), post_att .+ T(0.3),
        result.ci_lower[post_idx], result.ci_upper[post_idx],
        T(2.5), post_et, post_att, T(conf_level))
end

function honest_did(result::EventStudyLP{T}; Mbar=1.0, conf_level=0.95) where T
    post_idx = findall(t -> t >= 0, result.event_times)
    post_et = result.event_times[post_idx]
    post_att = result.coefficients[post_idx]
    HonestDiDResult{T}(T(Mbar),
        post_att .- T(0.3), post_att .+ T(0.3),
        result.ci_lower[post_idx], result.ci_upper[post_idx],
        T(2.5), post_et, post_att, T(conf_level))
end

export DIDResult, EventStudyLP, LPDiDResult, BaconDecomposition
export PretrendTestResult, NegativeWeightResult, HonestDiDResult
export estimate_did, estimate_event_study_lp, estimate_lp_did
export bacon_decomposition, pretrend_test, negative_weight_check, honest_did

# ─── FAVAR Types & Functions ─────────────────────────────────

struct FAVARModel{T<:Real}
    Y::Matrix{T}; p::Int; B::Matrix{T}; U::Matrix{T}; Sigma::Matrix{T}
    factors::Matrix{T}; loadings::Matrix{T}; n_factors::Int; n_key::Int
    aic::T; bic::T; loglik::T
    varnames::Vector{String}; panel_varnames::Vector{String}
end

struct BayesianFAVAR{T<:Real}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    factor_draws::Array{T,3}
    loadings_draws::Array{T,3}
    X_panel::Matrix{T}
    panel_varnames::Vector{String}
    Y_key_indices::Vector{Int}
    n_factors::Int
    n_key::Int
    n::Int
    p::Int
    data::Matrix{T}
    varnames::Vector{String}
end

function estimate_favar(X::Matrix{T}, key_indices::Vector{Int}, r::Int, p::Int;
                        method=:two_step, n_draws=5000, panel_varnames=nothing,
                        seed=nothing) where T
    n_obs, n_vars = size(X)
    n_key = length(key_indices)
    n_aug = r + n_key
    Y = X[p+1:end, 1:min(n_aug, n_vars)]
    B = ones(T, n_aug * p + 1, n_aug) * T(0.1)
    U = randn(T, n_obs - p, n_aug)
    Sigma = Matrix{T}(I(n_aug)) * T(0.5)
    factors = randn(T, n_obs, r)
    loadings = randn(T, n_vars, r)
    # Real aug names (favar/estimation.jl): F1..Fr then the KEY variables' panel
    # names — pvn[idx] whether given or the "X$i" default (MEMs#538 adoption).
    pvnames = panel_varnames === nothing ? ["X$i" for i in 1:n_vars] : panel_varnames
    length(pvnames) == n_vars || throw(ArgumentError(
        "panel_varnames has $(length(pvnames)) entries but X has $n_vars columns"))
    vnames = vcat(["F$i" for i in 1:r],
                  [1 <= idx <= n_vars ? pvnames[idx] : "Y$idx" for idx in key_indices])
    if method == :bayesian
        B_draws = ones(T, size(B, 1), size(B, 2), n_draws) * T(0.1)
        Sigma_draws = ones(T, n_aug, n_aug, n_draws) * T(0.5)
        factor_draws = ones(T, n_obs, r, n_draws) * T(0.1)
        loadings_draws = ones(T, n_vars, r, n_draws) * T(0.3)
        return BayesianFAVAR{T}(B_draws, Sigma_draws, factor_draws, loadings_draws,
            X, pvnames, key_indices, r, n_key, n_aug, p, Y, vnames)
    end
    FAVARModel{T}(Y, p, B, U, Sigma, factors, loadings, r, n_key,
                   T(-100.0), T(-95.0), T(-90.0), vnames, pvnames)
end

function to_var(favar::FAVARModel{T}) where T
    # Carry the augmented names into the VAR so irf/fevd favar label the key
    # variables with their panel names, exactly as real does.
    VARModel{T}(favar.Y, favar.p, favar.B, favar.U, favar.Sigma,
                favar.aic, favar.bic, T(-92.0), copy(favar.varnames))
end

function favar_panel_irf(favar::FAVARModel{T}, irf_result::ImpulseResponse{T}) where T
    N = size(favar.loadings, 1)
    H = irf_result.horizon
    n_shocks = length(irf_result.shocks)
    vals = ones(T, H + 1, N, n_shocks) * T(0.05)
    ImpulseResponse(vals, nothing, nothing, H,
        favar.panel_varnames, irf_result.shocks, :favar_panel)
end

function favar_panel_forecast(favar::FAVARModel{T}, fc::VARForecast{T}) where T
    N = size(favar.loadings, 1)
    h = fc.horizon
    panel_fc = ones(T, h, N) * T(0.1)
    VARForecast{T}(panel_fc, panel_fc .- T(0.5), panel_fc .+ T(0.5),
                    h, :none, T(0.95), favar.panel_varnames, nothing)
end

# FAVAR dispatches for irf/fevd/hd — delegate to VAR internals
function irf(favar::FAVARModel{T}, horizon::Int; kwargs...) where T
    var_model = to_var(favar)
    irf(var_model, horizon; kwargs...)
end
function fevd(favar::FAVARModel{T}, horizon::Int; kwargs...) where T
    var_model = to_var(favar)
    fevd(var_model, horizon; kwargs...)
end
function historical_decomposition(favar::FAVARModel{T}, horizon::Int; kwargs...) where T
    var_model = to_var(favar)
    historical_decomposition(var_model, horizon; kwargs...)
end
function forecast(favar::FAVARModel{T}, h::Int; kwargs...) where T
    var_model = to_var(favar)
    forecast(var_model, h; kwargs...)
end

export FAVARModel, BayesianFAVAR, estimate_favar, favar_panel_irf, favar_panel_forecast

# ─── Structural DFM Types & Functions ────────────────────────

struct StructuralDFM{T<:Real}
    gdfm::GeneralizedDynamicFactorModel{T}
    factor_var::VARModel{T}
    B0::Matrix{T}; Q::Matrix{T}
    identification::Symbol
    structural_irf::Array{T,3}
    loadings_td::Matrix{T}
    p_var::Int; shock_names::Vector{String}
    varnames::Vector{String}   # panel names (MEMs#538); real default "Var $i"
end

function estimate_structural_dfm(X::Matrix{T}, q::Int;
        identification=:cholesky, p=1, H=40, sign_check=nothing,
        max_draws=1000, standardize=true, bandwidth=0, kernel=:bartlett,
        spectral=:lag_window, method=:fglr, instrument=nothing, seed=nothing,
        r=0, varnames::Union{Nothing,Vector{String}}=nothing) where T
    method in (:fglr, :gdfm_var) ||
        throw(ArgumentError("method must be :fglr or :gdfm_var, got :$method"))
    spectral in (:lag_window, :smoothed_periodogram) ||
        throw(ArgumentError("spectral must be :lag_window or :smoothed_periodogram, got :$spectral"))
    # Real (factor/structural.jl) requires an instrument for proxy identification.
    identification === :proxy && instrument === nothing && throw(ArgumentError(
        "identification=:proxy requires `instrument`"))
    n_obs, n_vars = size(X)
    # Real (factor/structural.jl, MEMs#538) defaults and validates the length.
    vn = varnames === nothing ? ["Var $i" for i in 1:n_vars] : varnames
    length(vn) == n_vars || throw(ArgumentError(
        "varnames has $(length(vn)) entries but panel has $n_vars columns"))
    gdfm = estimate_gdfm(X, q; standardize=standardize, bandwidth=bandwidth, kernel=kernel,
                         spectral=spectral)
    factor_Y = randn(T, n_obs - p, q)
    B_fvar = ones(T, q * p + 1, q) * T(0.1)
    U_fvar = randn(T, n_obs - p, q)
    Sigma_fvar = Matrix{T}(I(q)) * T(0.5)
    # Real names the factor VAR "Factor $i" — fevd sdfm labels come from here.
    fvar = VARModel{T}(factor_Y, p, B_fvar, U_fvar, Sigma_fvar, T(-50.0), T(-48.0),
                       T(-45.0), ["Factor $i" for i in 1:q])
    B0 = Matrix{T}(I(q))
    Q_mat = Matrix{T}(I(q))
    loadings_td = randn(T, n_vars, q)
    s_irf = ones(T, H + 1, n_vars, q) * T(0.05)
    snames = ["structural_shock_$i" for i in 1:q]
    StructuralDFM{T}(gdfm, fvar, B0, Q_mat, identification, s_irf, loadings_td, p, snames, vn)
end

# Real (factor/structural.jl): q=:auto selects via q_method (hallin_liska default).
# Deterministic canned selection matching the old ic_criteria_gdfm auto (2 factors
# when the panel allows it).
function estimate_structural_dfm(X::Matrix{T}, q::Symbol;
        q_method=:hallin_liska, q_max=8, r=0, identification=:cholesky, p=1, H=40,
        method=:fglr, spectral=:lag_window, instrument=nothing, seed=nothing,
        sign_check=nothing,
        standardize=true, bandwidth=0, kernel=:bartlett, varnames=nothing) where T
    q === :auto || throw(ArgumentError("q must be a positive integer or :auto, got :$q"))
    q_method in (:hallin_liska, :bai_ng, :amengual_watson) || throw(ArgumentError(
        "q_method must be :hallin_liska, :bai_ng, or :amengual_watson, got :$q_method"))
    n_obs, n_vars = size(X)
    q_cap = min(q_max, max(1, n_vars - 1), max(1, n_obs - 1))
    q_hat = max(min(q_cap, 2), 1)
    # No `; kwargs...` absorber here (the mock budget in check_mock_surface.jl is
    # frozen): the forwarded set is enumerated explicitly, matching the CLI surface.
    estimate_structural_dfm(X, q_hat; r=r, identification=identification, p=p, H=H,
        method=method, spectral=spectral, instrument=instrument, seed=seed,
        sign_check=sign_check,
        standardize=standardize, bandwidth=bandwidth, kernel=kernel, varnames=varnames)
end

function irf(sdfm::StructuralDFM{T}, horizon::Int; kwargs...) where T
    h = min(horizon, size(sdfm.structural_irf, 1) - 1)
    vals = sdfm.structural_irf[1:h+1, :, :]
    # Real labels panel responses with sdfm.varnames (favar/analysis.jl).
    ImpulseResponse(vals, nothing, nothing, h, copy(sdfm.varnames), sdfm.shock_names, :structural_dfm)
end

# Real delegates to the factor VAR (favar/analysis.jl) — labels are "Factor $i".
# No kwargs absorber: the CLI never forwards kwargs here, and mock ⊆ real means
# stricter is the safe direction (keeps the check_mock_surface absorber budget flat).
fevd(sdfm::StructuralDFM{T}, horizon::Int) where T = fevd(sdfm.factor_var, horizon)

# Real (favar/analysis.jl): panel forecast from a Structural DFM → FactorForecast.
# Defined here (after the struct) because mocks.jl is one flat top-to-bottom
# module — a signature type must already exist at include time.
function forecast(sdfm::StructuralDFM{T}, h::Int;
        ci_method=:none, reps=200, conf_level=0.95, rng=nothing) where T
    h >= 1 || throw(ArgumentError("h must be ≥ 1"))
    ci_method in (:none, :bootstrap) ||
        throw(ArgumentError("ci_method must be :none or :bootstrap, got :$ci_method"))
    n = length(sdfm.varnames)
    r = size(sdfm.loadings_td, 2)
    factors = ones(h, r) * 0.1
    obs = ones(h, n) * 0.1
    FactorForecast(factors, obs, factors, factors, obs .- 0.5, obs .+ 0.5,
        abs.(factors) .* 0.1, ones(h, n)*0.1, h, conf_level, :analytical)
end

export StructuralDFM, estimate_structural_dfm

# ─── Bayesian DSGE Types & Functions ─────────────────────────

struct BayesianDSGE{T<:Real}
    theta_draws::Matrix{T}
    log_posterior::Vector{T}
    param_names::Vector{String}
    log_marginal_likelihood::T
    method::Symbol
    acceptance_rate::T
    ess_history::Vector{T}
    spec::ModelSpec{T}
    solution::DSGESolution{T}
end

# theta0 accepts a positional Vector OR a name→value Dict/NamedTuple (MEMs #136).
function estimate_dsge_bayes(spec::ModelSpec{T},
        data::Matrix, theta0::Union{AbstractVector,AbstractDict,NamedTuple};
        priors=Dict(), method=:smc, observables=Symbol[],
        n_smc=5000, n_particles=500, n_mh_steps=1,
        n_draws=10000, burnin=5000, ess_target=0.5,
        measurement_error=nothing, solver=:gensys,
        solver_kwargs=NamedTuple(), delayed_acceptance=false,
        n_screen=200, rng=nothing,
        prefilter::Symbol=:none, hp_lambda::Real=1600,
        observation_trends=nothing, warn_trends::Bool=true,
        ha_method::Symbol=:ssj, ha_kwargs=NamedTuple(),
        proposal_scale=0.01, adapt_interval::Int=100, seed=nothing) where T
    if has_kind(spec, HouseholdSystem)
        np = length(theta0 isa AbstractDict ? collect(values(theta0)) :
                    theta0 isa NamedTuple ? collect(theta0) : collect(theta0))
        n_kept = max(n_draws - burnin, 1)
        tv = T.(collect(theta0 isa AbstractDict ? values(theta0) :
                        theta0 isa NamedTuple ? collect(theta0) : theta0))
        draws = randn(T, n_kept, np) .* T(0.01) .+ reshape(tv, 1, np)
        log_post = fill(T(-100.0), n_kept)
        pnames = isempty(priors) ? ["param_$i" for i in 1:np] :
                 sort!([String(k) for k in keys(priors)])
        ess_hist = fill(T(0.8), 10)
        dspec = ModelSpec()
        sol = solve(dspec; method=:gensys)
        return BayesianDSGE{T}(draws, log_post, pnames, T(-450.0 + np), :rwmh, T(0.30),
                               ess_hist, dspec, sol)
    end
    # W12/#114: mirror real's enum so a bad --prefilter throws the same class here.
    prefilter in (:none, :demean, :first_difference, :linear_detrend, :hp) ||
        throw(ArgumentError(
            "prefilter must be one of (:none, :demean, :first_difference, :linear_detrend, :hp), got :$prefilter"))
    theta0v = theta0 isa AbstractDict ? collect(values(theta0)) :
              theta0 isa NamedTuple ? collect(theta0) : collect(theta0)
    np = length(theta0v)
    draws = randn(T, n_draws, np) .* T(0.01) .+ Float64.(theta0v)'
    log_post = fill(T(-100.0), n_draws)
    pnames = ["param_$i" for i in 1:np]
    ess_hist = fill(T(n_smc * 0.8), 20)
    sol = solve(spec; method=:gensys)
    BayesianDSGE{T}(draws, log_post, pnames, T(-500.0 + np), method, T(0.25), ess_hist, spec, sol)
end

export BayesianDSGE, estimate_dsge_bayes

# ─── Bayesian DSGE diagnostics (C073) ────────────────────────
# Struct field names/order mirror the real MEMs types (mock ⊆ real; check_mock_surface).
# Structs are defined BEFORE the functions that construct them (flat top-to-bottom module).

struct MCMCDiagnostics{T<:AbstractFloat}
    param_names::Vector{Symbol}
    rhat::Vector{T}
    ess_bulk::Vector{T}
    ess_tail::Vector{T}
    geweke_z::Vector{T}
    geweke_p::Vector{T}
    mean::Vector{T}
    sd::Vector{T}
    n_draws::Int
    method::Symbol
end

struct IdentificationDiagnostics{T<:AbstractFloat}
    param_names::Vector{Symbol}
    theta::Vector{T}
    rank::Int
    n_params::Int
    n_moments::Int
    n_lags::Int
    singular_values::Vector{T}
    tol::T
    null_space::Matrix{T}
    identified::Bool
end

struct LearningRateCheck{T<:AbstractFloat}
    param_names::Vector{Symbol}
    sample_sizes::Vector{Int}
    post_vars::Matrix{T}
    learning_rate::Vector{T}
    flagged::Vector{Bool}
    threshold::T
end

struct PriorPosteriorOverlap{T<:AbstractFloat}
    param_names::Vector{Symbol}
    overlap::Vector{T}
    flagged::Vector{Bool}
    threshold::T
end

function mcmc_diagnostics(result::BayesianDSGE{T}) where {T}
    draws = result.theta_draws
    np = size(draws, 2)
    nd = size(draws, 1)
    mu = T[mean(draws[:, i]) for i in 1:np]
    sd = T[nd > 1 ? sqrt(var(draws[:, i])) : zero(T) for i in 1:np]
    return MCMCDiagnostics{T}(
        Symbol.(result.param_names),
        fill(T(1.0), np),      # rhat ≈ 1 (converged)
        fill(T(nd), np),       # ess_bulk
        fill(T(nd), np),       # ess_tail
        zeros(T, np),          # geweke_z ≈ 0
        ones(T, np),           # geweke_p ≈ 1
        mu, sd, nd, result.method,
    )
end

function identification_diagnostics(spec::ModelSpec{T}, param_names::Vector{Symbol};
        theta=nothing, observables::Vector{Symbol}=Symbol[], n_lags::Int=2,
        tol_rel=1e-8, solver::Symbol=:gensys, solver_kwargs=NamedTuple()) where {T}
    d = length(param_names)
    theta_v = theta === nothing ? ones(T, d) : T.(theta)
    return IdentificationDiagnostics{T}(
        copy(param_names), theta_v, d, d, 2 * d, n_lags,
        collect(T, range(T(1.0), T(0.1); length=max(d, 1))),
        T(1e-8), zeros(T, d, 0), true,
    )
end

function learning_rate_check(result::BayesianDSGE{T}; fractions=[0.5, 1.0],
        n_smc::Int=300, threshold=0.2, rng=nothing) where {T}
    d = length(result.param_names)
    sizes = Int[round(Int, f * 100) for f in fractions]
    return LearningRateCheck{T}(
        Symbol.(result.param_names), sizes,
        fill(T(0.01), d, length(fractions)),  # post_vars
        fill(T(1.0), d),                      # learning_rate α ≈ 1 (identified)
        fill(false, d), T(threshold),
    )
end

function prior_posterior_overlap(result::BayesianDSGE{T}; n_grid::Int=0,
        threshold=0.8) where {T}
    d = length(result.param_names)
    ovl = fill(T(0.5), d)
    return PriorPosteriorOverlap{T}(Symbol.(result.param_names), ovl,
                                    ovl .>= T(threshold), T(threshold))
end

bridge_sampling_ml(result::BayesianDSGE{T}; proposal::Symbol=:normal, df=5,
        n_proposal::Int=0, max_iter::Int=1000, tol=1e-10, rng=nothing) where {T} =
    result.log_marginal_likelihood + T(0.1)

# ─── C073 remainder (#78): posterior mode + prior predictive.
# Field-order subsets of real (dsge/bayes_types.jl, bayes_estimation.jl).

struct PosteriorMode{T<:AbstractFloat}
    mode::Vector{T}
    inv_hessian::Matrix{T}
    hessian::Matrix{T}
    log_posterior::T
    log_likelihood::T
    laplace_log_ml::T
    param_names::Vector{Symbol}
    converged::Bool
    n_iterations::Int
end

struct PriorPredictiveResult{T<:AbstractFloat}
    stat_names::Vector{String}
    stats::Matrix{T}
    n_draws::Int
    n_effective::Int
    T_periods::Int
end

function posterior_mode(spec, data::AbstractMatrix, theta0;
                        priors, observables=Symbol[], measurement_error=nothing,
                        solver::Symbol=:gensys, solver_kwargs=NamedTuple(),
                        transform::Bool=true, optimizer=nothing,
                        f_reltol::Real=1e-8, max_iter::Int=500)
    names = sort(collect(keys(priors)))
    d = length(names)
    d >= 1 || throw(ArgumentError("priors must be non-empty"))
    PosteriorMode{Float64}(fill(0.5, d), Matrix{Float64}(I(d)) .* 0.01,
        Matrix{Float64}(I(d)) .* 100.0, -120.5, -118.0, -125.0, names, true, 12)
end

function prior_predictive(spec, priors; n_draws::Int=500, T_periods::Int=200,
                          observables=Symbol[], stats=nothing, solver::Symbol=:gensys,
                          solver_kwargs=NamedTuple(), rng=Random.default_rng(),
                          seed=nothing)
    n_draws >= 1 || throw(ArgumentError("n_draws must be ≥ 1"))
    T_periods >= 1 || throw(ArgumentError("T_periods must be ≥ 1"))
    obs = isempty(observables) ? [:Y] : observables
    names = vcat(["mean_$(o)" for o in obs], ["var_$(o)" for o in obs])
    neff = max(1, n_draws - 1)          # one draw fails to solve, like real can
    PriorPredictiveResult{Float64}(names, randn(neff, length(names)) .* 0.1 .+ 0.05,
        n_draws, neff, T_periods)
end

export PosteriorMode, PriorPredictiveResult, posterior_mode, prior_predictive

export MCMCDiagnostics, IdentificationDiagnostics, LearningRateCheck, PriorPosteriorOverlap
export mcmc_diagnostics, identification_diagnostics, learning_rate_check
export prior_posterior_overlap, bridge_sampling_ml

# ─── Structural Break Test Types & Functions ─────────────────

struct AndrewsResult{T<:AbstractFloat}
    statistic::T; pvalue::T; break_index::Int; break_fraction::T
    test_type::Symbol; critical_values::Dict{Int,T}
    stat_sequence::Vector{T}; trimming::T; nobs::Int; n_params::Int
end

struct BaiPerronResult{T<:AbstractFloat}
    n_breaks::Int; break_dates::Vector{Int}; break_cis::Vector{Tuple{Int,Int}}
    regime_coefs::Vector{Vector{T}}; regime_ses::Vector{Vector{T}}
    supf_stats::Vector{T}; supf_pvalues::Vector{T}
    sequential_stats::Vector{T}; sequential_pvalues::Vector{T}
    bic_values::Vector{T}; lwz_values::Vector{T}
    trimming::T; nobs::Int
end

function andrews_test(y::AbstractVector{T}, X::AbstractMatrix;
        test=:supwald, trimming=0.15) where T
    n = length(y)
    n_params = size(X, 2)
    bp = div(n, 2)
    seq = fill(T(5.0), n - 2 * round(Int, n * trimming))
    seq[div(length(seq), 2)] = T(12.0)
    cvs = Dict(1 => T(8.85), 5 => T(7.04), 10 => T(6.28))
    AndrewsResult{T}(T(12.0), T(0.02), bp, T(bp / n),
        test, cvs, seq, T(trimming), n, n_params)
end

function bai_perron_test(y::AbstractVector{T}, X::AbstractMatrix;
        max_breaks=5, trimming=0.15, criterion=:bic) where T
    n = length(y)
    k = size(X, 2)
    BaiPerronResult{T}(
        1, [div(n, 2)], [(div(n, 2) - 5, div(n, 2) + 5)],
        [ones(T, k) * T(2.0), ones(T, k) * T(5.0)],
        [ones(T, k) * T(0.3), ones(T, k) * T(0.4)],
        [T(15.0)], [T(0.01)], [T(12.0)], [T(0.03)],
        fill(T(-100.0), max_breaks + 1), fill(T(-98.0), max_breaks + 1),
        T(trimming), n)
end

export AndrewsResult, BaiPerronResult, andrews_test, bai_perron_test

# ─── Panel Unit Root Test Types & Functions ──────────────────

struct PANICResult{T<:AbstractFloat}
    factor_adf_stats::Vector{T}; factor_adf_pvalues::Vector{T}
    pooled_statistic::T; pooled_pvalue::T
    individual_stats::Vector{T}; individual_pvalues::Vector{T}
    n_factors::Int; method::Symbol; nobs::Int; n_units::Int
end

struct PesaranCIPSResult{T<:AbstractFloat}
    cips_statistic::T
    pvalue::T
    individual_cadf_stats::Vector{T}
    critical_values::Dict{Int,T}
    lags::Int
    deterministic::Symbol
    nobs::Int
    n_units::Int
end

struct MoonPerronResult{T<:AbstractFloat}
    t_a_statistic::T; t_b_statistic::T; pvalue_a::T; pvalue_b::T
    n_factors::Int; nobs::Int; n_units::Int
end

struct FactorBreakResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    break_date::Int
    method::Symbol
    n_factors::Int
    nobs::Int
    n_vars::Int
    # MEMs 0.7.3/#606: per-series sup statistics + maximizing dates — populated by the
    # two POOLED methods (breitung_eickmeier, han_inoue), nothing for chen_dolado_gonzalo.
    series_statistics::Union{Vector{T}, Nothing}
    series_break_dates::Union{Vector{Int}, Nothing}
end
# 7-arg back-compat constructor, mirroring real (#606)
FactorBreakResult{T}(s, p, bd, m, nf, no, nv) where {T<:AbstractFloat} =
    FactorBreakResult{T}(s, p, bd, m, nf, no, nv, nothing, nothing)

function panic_test(X::AbstractMatrix{T}; r=:auto, method=:pooled) where T
    n_obs, n_units = size(X)
    n_r = r == :auto ? 2 : r
    PANICResult{T}(
        fill(T(-3.0), n_r), fill(T(0.01), n_r),
        T(-5.0), T(0.001),
        fill(T(-2.5), n_units), fill(T(0.05), n_units),
        n_r, method, n_obs, n_units)
end
function panic_test(pd::PanelData{T}; r=:auto, method=:pooled) where T
    X = hcat([pd.data[:, i] for i in 1:pd.n_vars]...)
    panic_test(X; r=r, method=method)
end

function pesaran_cips_test(X::AbstractMatrix{T}; lags=:auto, deterministic=:constant) where T
    n_obs, n_units = size(X)
    p = lags == :auto ? max(1, round(Int, n_obs^(1/3))) : lags
    cvs = Dict(1 => T(-2.16), 5 => T(-2.04), 10 => T(-1.97))
    PesaranCIPSResult{T}(T(-2.5), T(0.01), fill(T(-2.3), n_units),
        cvs, p, deterministic, n_obs, n_units)
end
function pesaran_cips_test(pd::PanelData{T}; lags=:auto, deterministic=:constant) where T
    X = hcat([pd.data[:, i] for i in 1:pd.n_vars]...)
    pesaran_cips_test(X; lags=lags, deterministic=deterministic)
end

function moon_perron_test(X::AbstractMatrix{T}; r=:auto) where T
    n_obs, n_units = size(X)
    n_r = r == :auto ? 2 : r
    MoonPerronResult{T}(T(-3.5), T(-4.0), T(0.001), T(0.0005), n_r, n_obs, n_units)
end
function moon_perron_test(pd::PanelData{T}; r=:auto) where T
    X = hcat([pd.data[:, i] for i in 1:pd.n_vars]...)
    moon_perron_test(X; r=r)
end

function factor_break_test(X::AbstractMatrix{T}, r::Int; method=:breitung_eickmeier) where T
    n_obs, n_units = size(X)
    if method in (:breitung_eickmeier, :han_inoue)
        # Deliberately UNSORTED per-series stats so a handler that forgets to sort shows.
        stats = T[T(1 + (i * 7) % n_units) for i in 1:n_units]
        dates = Int[div(n_obs, 2) + (i % 3) for i in 1:n_units]
        FactorBreakResult{T}(T(8.5), T(0.03), div(n_obs, 2), method, r, n_obs, n_units,
                             stats, dates)
    else
        FactorBreakResult{T}(T(8.5), T(0.03), div(n_obs, 2), method, r, n_obs, n_units)
    end
end
function factor_break_test(pd::PanelData{T}, r::Int; method=:breitung_eickmeier) where T
    X = hcat([pd.data[:, i] for i in 1:pd.n_vars]...)
    factor_break_test(X, r; method=method)
end

function panel_unit_root_summary(X; tests=[:panic, :cips, :moon_perron])
    println("Panel unit root summary ($(length(tests)) tests)")
end

export PANICResult, PesaranCIPSResult, MoonPerronResult, FactorBreakResult
export panic_test, pesaran_cips_test, moon_perron_test, factor_break_test
export panel_unit_root_summary

# ─── C069/C070: randomness/nonlinearity + panel cointegration tests ────
# NamedTuple returns (no result structs) with the exact field names the real
# MEMs result types expose. Kwargs are enumerated (no `; kwargs...` catch-all) to
# stay within the check_mock_surface absorber budget. Minimal real-like validation
# so T1/T2 catch shape/arity bugs.

# Lo–MacKinlay / Chow–Denning variance-ratio test (real fields: q, vr, z, z_star,
# z_pvalue, z_star_pvalue, cd_stat, cd_pvalue, cd_star_stat, cd_star_pvalue, ...).
function variance_ratio_test(y; q=[2, 4, 8, 16], method=:lomackinlay, robust=true,
                             bootstrap=0, boot_weights=:rademacher, seed=1234)
    method in (:lomackinlay, :wright) ||
        throw(ArgumentError("method must be :lomackinlay or :wright, got :$method"))
    yv = float.(collect(y))
    nlev = length(yv)
    nlev >= 4 || throw(ArgumentError("need at least 4 level observations, got $nlev"))
    qvec = sort(unique(Int.(collect(q))))
    all(qi -> qi >= 2, qvec) || throw(ArgumentError("every q must be ≥ 2"))
    maximum(qvec) < nlev - 1 ||
        throw(ArgumentError("every q must be < number of returns ($(nlev - 1))"))
    nq = length(qvec)
    return (q=qvec, vr=fill(0.92, nq), z=fill(-0.9, nq), z_star=fill(-0.8, nq),
            z_pvalue=fill(0.37, nq), z_star_pvalue=fill(0.42, nq),
            cd_stat=1.55, cd_pvalue=0.34, cd_star_stat=1.42, cd_star_pvalue=0.41,
            method=method, robust=robust, nobs=nlev)
end

# BDS iid/nonlinearity test — statistic/pvalue are (n_dims × n_eps) matrices.
function bds_test(y; m=2:6, eps_frac=0.7, bootstrap=0, seed=1234)
    yv = float.(collect(y))
    n = length(yv)
    ms = sort(unique(filter(mm -> mm >= 1, collect(Int, m))))
    isempty(ms) && throw(ArgumentError("no valid embedding dimension in m=$m"))
    nm = length(ms)
    stat = reshape(Float64[1.4 + 0.1 * i for i in 1:nm], nm, 1)
    pval = reshape(fill(0.22, nm), nm, 1)
    return (m=ms, statistic=stat, pvalue=pval, nobs=n)
end

# Hadri panel stationarity test (H0: all units stationary).
function hadri_test(X; deterministic=:constant, hetero=true, cs_demean=false)
    deterministic in (:constant, :trend) ||
        throw(ArgumentError("Hadri deterministic must be :constant or :trend, got :$deterministic"))
    T_obs, N = size(X)
    T_obs >= 10 || throw(ArgumentError("Time dimension T=$T_obs too small for Hadri"))
    N >= 2 || throw(ArgumentError("Hadri needs at least N=2 panel units, got N=$N"))
    return (statistic=1.35, pvalue=0.11, n_units=N, nobs=T_obs,
            deterministic=deterministic, hetero=hetero)
end

# Panel cointegration trio (H0: no cointegration). All three expose the uniform
# names/statistics/pvalues/n_units/n_regressors/nobs output triple.
_panel_coint_meta(pd, xs) = (n_units=pd.n_groups, n_regressors=length(xs),
                             nobs=pd.n_groups == 0 ? pd.T_obs : pd.T_obs ÷ pd.n_groups)

function pedroni_test(pd::PanelData, y::Symbol, xs::Symbol...; trend=:constant,
                      lags=:auto, adf_lags=2)
    isempty(xs) && throw(ArgumentError("pedroni_test needs at least one regressor"))
    m = _panel_coint_meta(pd, xs)
    names = ["panel-v", "panel-rho", "panel-pp", "panel-adf", "group-rho", "group-pp", "group-adf"]
    return (names=names, statistics=[2.1, -1.9, -2.0, -2.2, -1.6, -1.9, -2.1],
            pvalues=[0.02, 0.04, 0.03, 0.02, 0.06, 0.04, 0.03],
            n_units=m.n_units, n_regressors=m.n_regressors, nobs=m.nobs)
end

function kao_test(pd::PanelData, y::Symbol, xs::Symbol...; lags=:auto, kernel_lags=:auto)
    isempty(xs) && throw(ArgumentError("kao_test needs at least one regressor"))
    m = _panel_coint_meta(pd, xs)
    names = ["DFrho", "DFt", "DFrho_star", "DFt_star", "ADF"]
    return (names=names, statistics=[-2.3, -2.1, -2.4, -2.2, -2.0],
            pvalues=[0.02, 0.03, 0.02, 0.03, 0.04],
            n_units=m.n_units, n_regressors=m.n_regressors, nobs=m.nobs)
end

function westerlund_test(pd::PanelData, y::Symbol, xs::Symbol...; trend=:constant,
                         lags=1, leads=0, lrwindow=2, bootstrap=0, seed=20240716)
    isempty(xs) && throw(ArgumentError("westerlund_test needs at least one regressor"))
    length(xs) <= 6 || throw(ArgumentError("Westerlund test supports at most 6 regressors"))
    m = _panel_coint_meta(pd, xs)
    names = ["Gt", "Ga", "Pt", "Pa"]
    return (names=names, statistics=[-2.5, -8.0, -3.0, -9.5],
            pvalues=[0.02, 0.03, 0.02, 0.03],
            n_units=m.n_units, n_regressors=m.n_regressors, nobs=m.nobs)
end

# ─── C070 remainder (#75): first-gen panel unit-root + Fisher-Johansen + DH causality.
# Structs are field-order subsets of real; validation mirrors upstream.

struct LLCResult{T<:AbstractFloat}
    statistic::T; pvalue::T; t_unadjusted::T; delta::T; S_N::T
    mu_star::T; sigma_star::T; T_tilde::T
    lags::Vector{Int}; deterministic::Symbol; nobs::Int; n_units::Int
end

struct IPSResult{T<:AbstractFloat}
    statistic::T; pvalue::T; tbar::T; individual_t::Vector{T}
    E_mean::T; V_mean::T
    lags::Vector{Int}; deterministic::Symbol; nobs::Int; n_units::Int
end

struct BreitungPanelResult{T<:AbstractFloat}
    statistic::T; pvalue::T; lags::Int; deterministic::Symbol; nobs::Int; n_units::Int
end

struct FisherJohansenResult{T<:AbstractFloat}
    ranks::Vector{Int}
    trace_statistics::Vector{T}; trace_pvalues::Vector{T}
    max_statistics::Vector{T}; max_pvalues::Vector{T}
    individual_trace_pvalues::Matrix{T}; individual_max_pvalues::Matrix{T}
    combine::Symbol; deterministic::Symbol; lags::Int; rank::Int; n_units::Int
end

struct DumitrescuHurlinResult{T<:AbstractFloat}
    Wbar::T; Zbar::T; Zbar_pvalue::T; Ztilde::T; Ztilde_pvalue::T
    W_i::Vector{T}; p::Int; N::Int; nobs::Int; n_skipped::Int
    bootstrap::Int; seed::Int; bootstrap_pvalue::T
    cause::Symbol; effect::Symbol
end

_mock_panel_det(d) = d in (:none, :constant, :trend) ? d :
    throw(ArgumentError("deterministic must be :none, :constant, or :trend, got :$d"))

function llc_test(X::AbstractMatrix; deterministic::Symbol=:constant, lags=:auto,
                  max_lags=nothing, criterion::Symbol=:aic, cs_demean::Bool=false)
    _mock_panel_det(deterministic)
    n, N = size(X)
    n > 5 || throw(ArgumentError("need more observations, got $n"))
    LLCResult{Float64}(-1.9, 0.03, -3.2, -0.04, 1.0, -0.5, 0.8, Float64(n),
        fill(lags isa Symbol ? 1 : Int(lags), N), deterministic, n, N)
end

function ips_test(X::AbstractMatrix; deterministic::Symbol=:constant, lags=:auto,
                  max_lags=nothing, criterion::Symbol=:aic, cs_demean::Bool=false)
    _mock_panel_det(deterministic)
    n, N = size(X)
    n > 5 || throw(ArgumentError("need more observations, got $n"))
    IPSResult{Float64}(-2.1, 0.02, -1.8, fill(-1.8, N), -1.5, 0.9,
        fill(lags isa Symbol ? 1 : Int(lags), N), deterministic, n, N)
end

function breitung_panel_test(X::AbstractMatrix; deterministic::Symbol=:constant,
                             lags::Int=0, cs_demean::Bool=false)
    _mock_panel_det(deterministic)
    lags >= 0 || throw(ArgumentError("lags must be ≥ 0, got $lags"))
    n, N = size(X)
    BreitungPanelResult{Float64}(-1.7, 0.045, lags, deterministic, n, N)
end

function fisher_johansen_test(pd, ys::Symbol...; deterministic::Symbol=:constant,
                              lags::Int=2, combine::Symbol=:mw)
    length(ys) >= 2 || throw(ArgumentError("fisher_johansen_test needs at least 2 series, got $(length(ys))"))
    combine in (:mw, :choi) || throw(ArgumentError("combine must be :mw or :choi; got :$combine"))
    lags >= 1 || throw(ArgumentError("lags must be ≥ 1, got $lags"))
    k = length(ys)
    nun = pd.n_groups
    FisherJohansenResult{Float64}(collect(0:(k - 1)),
        fill(25.0, k), fill(0.03, k), fill(18.0, k), fill(0.04, k),
        fill(0.03, nun, k), fill(0.04, nun, k),
        combine, deterministic, lags, 1, nun)
end

function dh_causality_test(pd, x::Symbol, y::Symbol; p::Int=1, bootstrap::Int=0, seed::Int=1234)
    p >= 1 || throw(ArgumentError("lag order p must be ≥ 1, got $p"))
    bootstrap >= 0 || throw(ArgumentError("bootstrap must be ≥ 0, got $bootstrap"))
    nun = pd.n_groups
    DumitrescuHurlinResult{Float64}(2.1, 3.4, 0.0007, 2.9, 0.0037, fill(2.1, nun),
        p, nun, size(pd.data, 1), 0, bootstrap, seed,
        bootstrap > 0 ? 0.01 : NaN, x, y)
end

export LLCResult, IPSResult, BreitungPanelResult, FisherJohansenResult, DumitrescuHurlinResult
export llc_test, ips_test, breitung_panel_test, fisher_johansen_test, dh_causality_test

export variance_ratio_test, bds_test, hadri_test
export pedroni_test, kao_test, westerlund_test

# ─── C069 (remainder): seasonal / point-optimal / bubble / EDF + residual
# cointegration. Every struct is a field-order subset of real, and every mock
# estimator reproduces the REAL argument validation — a mock looser than real
# turns a guaranteed MEMs failure into a green suite (#84).

struct HEGYResult{T<:AbstractFloat}
    frequency::Int
    deterministic::Symbol
    lags::Int
    pi_coefs::Vector{T}
    t_zero::T
    t_nyquist::T
    t_zero_cv::Dict{Int,T}
    t_nyquist_cv::Dict{Int,T}
    pair_freqs::Vector{T}
    pair_F::Vector{T}
    pair_F_cv::Dict{Int,T}
    F_seasonal::T
    F_all::T
    nobs::Int
end

struct ERSResult{T<:AbstractFloat}
    P_T::T
    pvalue::T
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

struct BubbleResult{T<:AbstractFloat}
    kind::Symbol
    statistic::T
    pvalue::T
    critical_values::Dict{Int,T}
    bsadf::Vector{T}
    cv_seq::Vector{T}
    r2_index::Vector{Int}
    episodes::Vector{Tuple{Int,Int}}
    r0::T
    adflag::Int
    cv_method::Symbol
    mc_reps::Int
    nobs::Int
end

struct EDFTestResult{T<:AbstractFloat}
    test::Symbol
    dist::Symbol
    params::Symbol
    statistic::T
    raw_statistic::T
    pvalue::T
    nobs::Int
    theta::Vector{T}
    critical_values::Dict{Int,T}
    case::String
end

struct EngleGrangerResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    lags::Int
    regression::Symbol
    k::Int
    N::Int
    nobs::Int
end

struct PhillipsOuliarisResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    z_alpha::T
    z_alpha_pvalue::T
    regression::Symbol
    kernel::Symbol
    bandwidth::T
    k::Int
    N::Int
    nobs::Int
end

struct HansenInstabilityResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    regression::Symbol
    trend::Symbol
    nparam::Int
    k::Int
    nobs::Int
end

struct ParkAddedResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    q_add::Int
    base_order::Int
    regression::Symbol
    trend::Symbol
    k::Int
    nobs::Int
end

function hegy_test(y::AbstractVector; frequency::Int=4,
                   deterministic::Symbol=:const_trend_seas, lags=:auto)
    frequency ∈ (4, 12) ||
        throw(ArgumentError("frequency must be 4 (quarterly) or 12 (monthly), got $frequency"))
    deterministic ∈ (:none, :const, :const_seas, :const_trend, :const_trend_seas) ||
        throw(ArgumentError("invalid deterministic :$deterministic"))
    n = length(y)
    n > 2 * frequency + 8 || throw(ArgumentError("Need more observations, got $n"))
    npair = frequency == 4 ? 1 : 5
    lg = lags === :auto ? 1 : Int(lags)
    return HEGYResult{Float64}(frequency, deterministic, lg, fill(-0.3, 1 + npair),
        -2.4, -2.9, Dict(1 => -3.7, 5 => -3.1, 10 => -2.8), Dict(1 => -3.6, 5 => -3.0, 10 => -2.7),
        [Float64(i) * pi / 2 for i in 1:npair], fill(6.2, npair),
        Dict(1 => 8.5, 5 => 6.5, 10 => 5.5), 7.1, 6.8, n - lg - 1)
end

function ers_test(y::AbstractVector; trend::Bool=false)
    n = length(y)
    n < 30 && throw(ArgumentError("Need at least 30 observations, got $n"))
    return ERSResult{Float64}(3.2, 0.04, trend ? :trend : :constant,
        Dict(1 => 1.9, 5 => 3.1, 10 => 4.5), n)
end

function _mock_bubble(y, kind::Symbol; r0=:auto, adflag::Int=0, mc_reps::Int=999,
                      cv::Symbol=:asymptotic, seed::Int=20240716)
    cv ∈ (:asymptotic, :wildboot) ||
        throw(ArgumentError("cv must be :asymptotic or :wildboot; got :$cv"))
    n = length(y)
    r0f = r0 === :auto ? 0.01 + 1.8 / sqrt(n) : Float64(r0)
    (0.0 < r0f < 1.0) || throw(ArgumentError("r0 must lie in (0,1), got $r0f"))
    floor(Int, r0f * n) >= adflag + 3 || throw(ArgumentError("window too small for adflag=$adflag"))
    nseq = max(1, n - floor(Int, r0f * n))
    return BubbleResult{Float64}(kind, 1.85, 0.03, Dict(1 => 2.1, 5 => 1.5, 10 => 1.2),
        fill(0.8, nseq), fill(1.5, nseq), collect(1:nseq),
        [(max(1, n - 20), max(2, n - 10))], r0f, adflag, cv, mc_reps, n)
end

# Explicit kwargs, NOT `; kwargs...` — the bare absorber form is budgeted by
# check_mock_surface (it hides signature drift), and each one-liner would spend two.
sadf_test(y::AbstractVector; r0=:auto, adflag::Int=0, mc_reps::Int=999,
          cv::Symbol=:asymptotic, seed::Int=20240716) =
    _mock_bubble(y, :sadf; r0=r0, adflag=adflag, mc_reps=mc_reps, cv=cv, seed=seed)

gsadf_test(y::AbstractVector; r0=:auto, adflag::Int=0, mc_reps::Int=999,
           cv::Symbol=:asymptotic, seed::Int=20240716) =
    _mock_bubble(y, :gsadf; r0=r0, adflag=adflag, mc_reps=mc_reps, cv=cv, seed=seed)

const _MOCK_EDF_DISTS = (:normal, :exponential, :logistic, :gumbel, :gamma, :weibull, :chisq)
const _MOCK_EDF_TESTS = (:ks, :lilliefors, :cvm, :ad, :watson)

function edf_test(y::AbstractVector; dist::Symbol=:normal, test::Symbol=:ad,
                  params::Symbol=:estimate, theta=nothing)
    dist ∈ _MOCK_EDF_DISTS || throw(ArgumentError("dist must be one of $(_MOCK_EDF_DISTS); got :$dist"))
    test ∈ _MOCK_EDF_TESTS || throw(ArgumentError("test must be one of $(_MOCK_EDF_TESTS); got :$test"))
    params ∈ (:estimate, :specified) ||
        throw(ArgumentError("params must be :estimate or :specified; got :$params"))
    params === :specified && theta === nothing &&
        throw(ArgumentError("params=:specified requires theta"))
    n = length(y)
    n >= 5 || throw(ArgumentError("Need at least 5 observations, got $n"))
    th = theta === nothing ? Float64[mean(y), std(y)] : Float64.(collect(theta))
    return EDFTestResult{Float64}(test, dist, params, 0.62, 0.58, 0.11, n, th,
        Dict(1 => 1.03, 5 => 0.75, 10 => 0.63), "case 3")
end

function engle_granger_test(y::AbstractVector, X::AbstractMatrix;
                            trend::Symbol=:constant, lags=:aic, max_lags=nothing)
    trend ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("trend must be :none, :constant, or :trend; got :$trend"))
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("length(y)=$n must equal size(X,1)=$(size(X,1))"))
    k = size(X, 2)
    k >= 1 || throw(ArgumentError("need at least one regressor column"))
    n > 3 * k + 12 || throw(ArgumentError("too few observations ($n) for $k regressor(s)"))
    lg = lags isa Symbol ? 1 : Int(lags)
    return EngleGrangerResult{Float64}(-3.85, 0.03, lg, trend, k, k + 1, n - lg - 1)
end

function phillips_ouliaris_test(y::AbstractVector, X::AbstractMatrix;
                                trend::Symbol=:constant, kernel::Symbol=:bartlett,
                                bandwidth=:nw)
    trend ∈ (:none, :constant, :trend) ||
        throw(ArgumentError("trend must be :none, :constant, or :trend; got :$trend"))
    kernel ∈ (:bartlett, :parzen, :qs, :quadratic_spectral, :tukey_hanning) ||
        throw(ArgumentError("invalid kernel :$kernel"))
    n = length(y)
    size(X, 1) == n ||
        throw(DimensionMismatch("length(y)=$n must equal size(X,1)=$(size(X,1))"))
    k = size(X, 2)
    k >= 1 || throw(ArgumentError("need at least one regressor column"))
    n > 3 * k + 12 || throw(ArgumentError("too few observations ($n) for $k regressor(s)"))
    bw = bandwidth isa Symbol ? floor(Int, 4 * ((n - 1) / 100)^0.25) : Float64(bandwidth)
    return PhillipsOuliarisResult{Float64}(-3.6, 0.04, -21.5, 0.05, trend, kernel,
        Float64(bw), k, k + 1, n - 1)
end

function hansen_instability_test(m)
    return HansenInstabilityResult{Float64}(0.42, 0.12, :constant, m.trend,
        m.k + 1, m.k, m.nobs)
end

function park_added_test(m; q_add::Int=2, kernel::Symbol=:bartlett, bandwidth=:nw)
    q_add >= 1 || throw(ArgumentError("q_add must be ≥ 1; got $q_add"))
    kernel ∈ (:bartlett, :parzen, :qs, :quadratic_spectral, :tukey_hanning) ||
        throw(ArgumentError("invalid kernel :$kernel"))
    return ParkAddedResult{Float64}(3.1, 0.21, q_add, m.trend === :linear ? 1 : 0,
        :constant, m.trend, m.k, m.nobs)
end

export HEGYResult, ERSResult, BubbleResult, EDFTestResult
export EngleGrangerResult, PhillipsOuliarisResult, HansenInstabilityResult, ParkAddedResult
export hegy_test, ers_test, sadf_test, gsadf_test, edf_test
export engle_granger_test, phillips_ouliaris_test, hansen_instability_test, park_added_test

# ─── Cross-Sectional Regression Types & Functions ──────────────────

struct RegModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    ssr::T
    tss::T
    r2::T
    adj_r2::T
    f_stat::T
    f_pval::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    weights::Union{Vector{T},Nothing}
    Z::Union{Matrix{T},Nothing}
    endogenous::Union{Vector{Int},Nothing}
    first_stage_f::Union{T,Nothing}
    sargan_stat::Union{T,Nothing}
    sargan_pval::Union{T,Nothing}
    cragg_donald_f::Union{T,Nothing}
    kleibergen_paap_f::Union{T,Nothing}
    stock_yogo_10pct::Union{T,Nothing}
    kclass_k::Union{T,Nothing}          # k-class scalar actually used (IV k-class only)
    kappa_hat::Union{T,Nothing}         # LIML minimum eigenvalue (liml/fuller only)
end

struct LogitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct ProbitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct MarginalEffects{T<:Real}
    effects::Vector{T}; se::Vector{T}; z_stat::Vector{T}; p_values::Vector{T}
    ci_lower::Vector{T}; ci_upper::Vector{T}; varnames::Vector{String}
    type::Symbol; conf_level::T
end

# StatsAPI dispatches for RegModel
coef(m::RegModel) = m.beta
vcov(m::RegModel) = m.vcov_mat
residuals(m::RegModel) = m.residuals
predict(m::RegModel) = m.fitted
stderror(m::RegModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat, 1)]
nobs(m::RegModel) = m.n_obs
loglikelihood(m::RegModel) = m.loglik
aic(m::RegModel) = m.aic
bic(m::RegModel) = m.bic
r2(m::RegModel) = m.r2
confint(m::RegModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# StatsAPI dispatches for LogitModel
coef(m::LogitModel) = m.beta
vcov(m::LogitModel) = m.vcov_mat
residuals(m::LogitModel) = m.residuals
predict(m::LogitModel) = m.fitted
stderror(m::LogitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat, 1)]
nobs(m::LogitModel) = length(m.y)
loglikelihood(m::LogitModel) = m.loglik
aic(m::LogitModel) = m.aic
bic(m::LogitModel) = m.bic
r2(m::LogitModel) = m.pseudo_r2
confint(m::LogitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# StatsAPI dispatches for ProbitModel
coef(m::ProbitModel) = m.beta
vcov(m::ProbitModel) = m.vcov_mat
residuals(m::ProbitModel) = m.residuals
predict(m::ProbitModel) = m.fitted
stderror(m::ProbitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat, 1)]
nobs(m::ProbitModel) = length(m.y)
loglikelihood(m::ProbitModel) = m.loglik
aic(m::ProbitModel) = m.aic
bic(m::ProbitModel) = m.bic
r2(m::ProbitModel) = m.pseudo_r2
confint(m::ProbitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# Mock functions

function estimate_reg(y::AbstractVector{T}, X::AbstractMatrix{T};
                      cov_type=:hc1, weights=nothing, varnames=nothing,
                      clusters=nothing, coords=nothing, cutoff::Real=0.0,
                      conley_kernel::Symbol=:bartlett, conley_metric::Symbol=:euclidean,
                      time=nothing, time_cutoff::Int=0, conley_psd::Bool=true) where T
    n, k = size(X)
    # W10/#112 — mirror real's validation EXACTLY so the mock throws the same exit class on
    # degenerate input (standing rule). `:conley` joins the enum; coords are required for it
    # and must have n rows; the kernel/metric enums are closed (the `--wild-dist` lesson).
    cov_type in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster, :conley) || throw(ArgumentError(
        "cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, :cluster, or :conley; got :$cov_type"))
    if cov_type == :conley
        coords === nothing && throw(ArgumentError("coords required for :conley cov_type"))
        size(coords, 1) == n || throw(ArgumentError("coords must have $n rows"))
        conley_kernel in (:bartlett, :uniform) || throw(ArgumentError(
            "kernel must be :bartlett or :uniform; got :$conley_kernel"))
        conley_metric in (:euclidean, :haversine) || throw(ArgumentError(
            "metric must be :euclidean or :haversine; got :$conley_metric"))
        conley_metric === :haversine && size(coords, 2) < 2 && throw(ArgumentError(
            "metric = :haversine needs coords with 2 columns (latitude, longitude)"))
        time_cutoff >= 0 || throw(ArgumentError("time_cutoff must be >= 0, got $time_cutoff"))
        time === nothing || length(time) == n || throw(ArgumentError("time must have length $n"))
    end
    if cov_type == :cluster
        clusters === nothing && throw(ArgumentError("clusters required for :cluster cov_type"))
        length(clusters) == n || throw(ArgumentError("clusters must have length $n"))
    end
    if weights !== nothing
        length(weights) == n || throw(ArgumentError("weights must have length $n"))
        all(w -> w > zero(T), weights) || throw(ArgumentError("All weights must be positive"))
    end
    beta = ones(T, k) * T(0.5)
    vcov_mat = Matrix{T}(I(k)) * T(0.01)
    fitted_vals = X * beta
    resids = y .- fitted_vals
    ssr = sum(resids .^ 2)
    tss = sum((y .- mean(y)) .^ 2)
    r2_val = one(T) - ssr / tss
    adj_r2_val = one(T) - (one(T) - r2_val) * (n - 1) / (n - k)
    f_val = T(25.0)
    f_p = T(0.001)
    ll = T(-100.0)
    aic_val = T(210.0)
    bic_val = T(220.0)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    RegModel{T}(y, X, beta, vcov_mat, resids, fitted_vals, ssr, tss,
                r2_val, adj_r2_val, f_val, f_p, ll, aic_val, bic_val,
                vnames, :ols, cov_type, weights, nothing, nothing,
                nothing, nothing, nothing, nothing, nothing, nothing,
                nothing, nothing)
end

function estimate_iv(y::AbstractVector{T}, X::AbstractMatrix{T}, Z::AbstractMatrix{T};
                     endogenous=Int[], cov_type=:hc1, varnames=nothing,
                     method::Symbol=:tsls, k=nothing, fuller_a::Real=1.0) where T
    # Mirror real's validation (reg/iv.jl): the k-class family and the k requirement.
    method in (:tsls, Symbol("2sls"), :liml, :fuller, :kclass) ||
        throw(ArgumentError("method must be :tsls, :liml, :fuller, or :kclass; got :$method"))
    method === :kclass && k === nothing &&
        throw(ArgumentError("k is required for method=:kclass"))
    isempty(endogenous) && throw(ArgumentError("endogenous must be non-empty for IV estimation"))
    # `k` is the k-class SCALAR kwarg here, so the regressor count must not reuse that
    # name — real calls it k_reg for the same reason.
    n, k_reg = size(X)
    beta = ones(T, k_reg) * T(0.5)
    vcov_mat = Matrix{T}(I(k_reg)) * T(0.01)
    fitted_vals = X * beta
    resids = y .- fitted_vals
    ssr = sum(resids .^ 2)
    tss = sum((y .- mean(y)) .^ 2)
    r2_val = one(T) - ssr / tss
    adj_r2_val = one(T) - (one(T) - r2_val) * (n - 1) / (n - k_reg)
    f_val = T(20.0)
    f_p = T(0.002)
    ll = T(-105.0)
    aic_val = T(220.0)
    bic_val = T(230.0)
    vnames = varnames === nothing ? ["x$i" for i in 1:k_reg] : varnames
    first_f = T(15.0)
    sargan_s = T(2.5)
    sargan_p = T(0.30)
    # Only the k-class methods populate these, exactly as real does.
    kk = method === :kclass ? T(k) :
         method === :liml   ? T(1.05) :
         method === :fuller ? T(1.05) - T(fuller_a) / T(n - size(Z, 2)) : nothing
    kap = method in (:liml, :fuller) ? T(1.05) : nothing
    RegModel{T}(y, X, beta, vcov_mat, resids, fitted_vals, ssr, tss,
                r2_val, adj_r2_val, f_val, f_p, ll, aic_val, bic_val,
                vnames, :iv, cov_type, nothing, Z, endogenous,
                first_f, sargan_s, sargan_p, T(15.0), T(14.0), T(7.0), kk, kap)
end

function _build_logit_probit(::Type{M}, y::AbstractVector{T}, X::AbstractMatrix{T};
                             cov_type=:ols, varnames=nothing, clusters=nothing,
                             maxiter=100, tol=1e-8) where {T, M}
    n, k = size(X)
    beta = ones(T, k) * T(0.3)
    vcov_mat = Matrix{T}(I(k)) * T(0.02)
    fitted_vals = ones(T, n) * T(0.5)
    resids = y .- fitted_vals
    ll = T(-80.0)
    ll_null = T(-100.0)
    pseudo = one(T) - ll / ll_null
    aic_val = T(170.0)
    bic_val = T(180.0)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    M{T}(y, X, beta, vcov_mat, resids, fitted_vals, ll, ll_null, pseudo,
          aic_val, bic_val, vnames, true, 5, cov_type)
end

function estimate_logit(y::AbstractVector{T}, X::AbstractMatrix{T};
                        cov_type=:ols, varnames=nothing, clusters=nothing,
                        maxiter=100, tol=1e-8) where T
    _build_logit_probit(LogitModel, y, X; cov_type=cov_type, varnames=varnames,
                        clusters=clusters, maxiter=maxiter, tol=tol)
end

function estimate_probit(y::AbstractVector{T}, X::AbstractMatrix{T};
                         cov_type=:ols, varnames=nothing, clusters=nothing,
                         maxiter=100, tol=1e-8) where T
    _build_logit_probit(ProbitModel, y, X; cov_type=cov_type, varnames=varnames,
                        clusters=clusters, maxiter=maxiter, tol=tol)
end

function marginal_effects(m::Union{LogitModel{T},ProbitModel{T}};
                          type=:ame, at=nothing, conf_level=0.95) where T
    k = length(m.beta)
    effects = ones(T, k) * T(0.1)
    se = ones(T, k) * T(0.02)
    z = effects ./ se
    pvals = ones(T, k) * T(0.001)
    z_crit = T(1.96)
    ci_lo = effects .- z_crit .* se
    ci_hi = effects .+ z_crit .* se
    MarginalEffects{T}(effects, se, z, pvals, ci_lo, ci_hi, m.varnames, type, conf_level)
end

function odds_ratio(m::LogitModel{T}; conf_level=0.95) where T
    or = exp.(m.beta)
    se = stderror(m)
    z_crit = T(1.96)
    ci_lo = exp.(m.beta .- z_crit .* se)
    ci_hi = exp.(m.beta .+ z_crit .* se)
    # Field is `or` upstream (OddsRatio struct) — name it the same here.
    (or=or, se=se, ci_lower=ci_lo, ci_upper=ci_hi, varnames=m.varnames, conf_level=conf_level)
end

# ─── W2/#107: count-data regression (MEMs#427) ───────────────────────────────
# Fields are the REAL names in real order (check_mock_surface: mock ⊆ real). Note
# NegBinModel carries `alpha`/`alpha_se` that PoissonModel does not, and that its
# `vcov_mat` is the joint (beta, log alpha) block — stderror slices beta out, exactly
# as upstream does.
struct PoissonModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    offset::Union{Nothing,Vector{T}}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    deviance::T
    null_deviance::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct NegBinModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    alpha::T
    vcov_mat::Matrix{T}
    alpha_se::T
    residuals::Vector{T}
    fitted::Vector{T}
    offset::Union{Nothing,Vector{T}}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    deviance::T
    null_deviance::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
    iterations::Int
end

struct DispersionTest{T<:AbstractFloat}
    nb2::NamedTuple{(:alpha, :se, :t_stat, :p_value),NTuple{4,T}}
    nb1::NamedTuple{(:alpha, :se, :t_stat, :p_value),NTuple{4,T}}
    n::Int
end

# Mirror upstream's own validation so the CLI's exit classes are exercised at T1/T2 with
# the SAME classes real produces (a looser mock turns a production failure green).
function _count_check(y, X, offset, exposure)
    length(y) == size(X, 1) ||
        throw(ArgumentError("y and X must have the same number of rows"))
    any(v -> v < 0, y) && throw(ArgumentError("count response must be non-negative"))
    all(isinteger, y) || throw(ArgumentError("count response must be integer-valued"))
    offset === nothing || exposure === nothing ||
        throw(ArgumentError("supply at most one of `offset` and `exposure`"))
    off = exposure !== nothing ? log.(Vector{Float64}(exposure)) :
          offset !== nothing ? Vector{Float64}(offset) : zeros(Float64, length(y))
    length(off) == length(y) ||
        throw(ArgumentError("offset/exposure must have length $(length(y))"))
    return off
end

function estimate_poisson(y::AbstractVector, X::AbstractMatrix;
                          offset=nothing, exposure=nothing, cov_type::Symbol=:robust,
                          varnames=nothing, clusters=nothing, maxiter::Int=100, tol=1e-10)
    cov_type in (:robust, :mle, :hc0, :hc1, :hc2, :hc3, :cluster) || throw(ArgumentError(
        "cov_type must be :robust, :mle, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$cov_type"))
    off = _count_check(y, X, offset, exposure)
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    n, k = size(Xm)
    beta = fill(0.1, k)
    mu = exp.(Xm * beta .+ off)
    vn = varnames === nothing ? ["x$i" for i in 1:k] : collect(String, varnames)
    PoissonModel{Float64}(yv, Xm, beta, Matrix{Float64}(0.01I, k, k), yv .- mu, mu,
                          offset === nothing && exposure === nothing ? nothing : off,
                          -120.0, -150.0, 0.2, 90.0, 120.0, 250.0, 260.0, vn, true, 5,
                          cov_type)
end

function estimate_nbreg(y::AbstractVector, X::AbstractMatrix;
                        offset=nothing, exposure=nothing, varnames=nothing,
                        maxiter::Int=1000, tol=1e-10)
    off = _count_check(y, X, offset, exposure)
    yv = Vector{Float64}(y); Xm = Matrix{Float64}(X)
    n, k = size(Xm)
    beta = fill(0.1, k)
    mu = exp.(Xm * beta .+ off)
    vn = varnames === nothing ? ["x$i" for i in 1:k] : collect(String, varnames)
    # Joint (beta, log alpha) block: k+1 square, so stderror must slice.
    NegBinModel{Float64}(yv, Xm, beta, 0.5, Matrix{Float64}(0.01I, k + 1, k + 1), 0.05,
                         yv .- mu, mu,
                         offset === nothing && exposure === nothing ? nothing : off,
                         -110.0, -150.0, 0.27, 80.0, 120.0, 240.0, 252.0, vn, true, 8)
end

dispersion_test(m::PoissonModel) =
    DispersionTest{Float64}((alpha=0.42, se=0.10, t_stat=4.2, p_value=0.00003),
                            (alpha=0.31, se=0.09, t_stat=3.4, p_value=0.0007),
                            length(m.y))

function incidence_rate_ratio(m::Union{PoissonModel{T},NegBinModel{T}};
                              conf_level=0.95) where T
    irr = exp.(m.beta)
    se = stderror(m)
    (or=irr, se=irr .* se, ci_lower=exp.(m.beta .- 1.96 .* se),
     ci_upper=exp.(m.beta .+ 1.96 .* se), varnames=m.varnames, conf_level=conf_level)
end

for M in (:PoissonModel, :NegBinModel)
    @eval begin
        coef(m::$M) = m.beta
        residuals(m::$M) = m.residuals
        predict(m::$M) = m.fitted
        fitted(m::$M) = m.fitted
        nobs(m::$M) = length(m.y)
        loglikelihood(m::$M) = m.loglik
        aic(m::$M) = m.aic
        bic(m::$M) = m.bic
        r2(m::$M) = m.pseudo_r2
    end
end
vcov(m::PoissonModel) = m.vcov_mat
stderror(m::PoissonModel) = sqrt.(max.(diag(m.vcov_mat), 0.0))
dof_residual(m::PoissonModel) = length(m.y) - length(m.beta)
# Slice beta out of the joint block, exactly as upstream does.
vcov(m::NegBinModel) = m.vcov_mat[1:length(m.beta), 1:length(m.beta)]
stderror(m::NegBinModel) = sqrt.(max.(diag(m.vcov_mat)[1:length(m.beta)], 0.0))
dof_residual(m::NegBinModel) = length(m.y) - length(m.beta) - 1

export PoissonModel, NegBinModel, DispersionTest
export estimate_poisson, estimate_nbreg, dispersion_test, incidence_rate_ratio

function vif(m::RegModel{T}) where T
    k = length(m.beta)
    # Return escalating VIF values so tests can trigger different warning branches
    # k=2: [2.5, 2.5], k=3: [2.5, 2.5, 7.0], k>=4: [2.5, 2.5, 7.0, 12.0, ...]
    vals = fill(T(2.5), k)
    if k >= 3
        vals[3] = T(7.0)  # moderate multicollinearity
    end
    if k >= 4
        vals[4] = T(12.0)  # severe multicollinearity
    end
    vals
end

# Real returns Dict{String,Any} mixing scalars with a "confusion" MATRIX (rows =
# actual 0/1, cols = predicted 0/1) — that mix is what made `sort(collect(ct))`
# compare a Matrix against a Float (#85). The old mock returned scalars only,
# under invented keys (recall/f1/true_positive/…) that real does not have, so the
# matrix branch of the renderer was never exercised. Mirror the real key set.
function classification_table(m::Union{LogitModel,ProbitModel}; threshold=0.5)
    tn, fp, fn, tp = 55.0, 8.0, 10.0, 30.0
    Dict{String,Any}(
        "confusion"   => [tn fp; fn tp],
        "accuracy"    => (tp + tn) / (tp + tn + fp + fn),
        "sensitivity" => tp / (tp + fn),
        "specificity" => tn / (tn + fp),
        "precision"   => tp / (tp + fp),
        "f1_score"    => 2 * (tp / (tp + fp)) * (tp / (tp + fn)) /
                             ((tp / (tp + fp)) + (tp / (tp + fn))),
        "n"           => Int(tp + tn + fp + fn),
        "threshold"   => threshold,
    )
end

# ─── C067 remainder (#72): cross-section OLS diagnostics. Structs are field-order
# subsets of real; every estimator reproduces the REAL argument validation, because a
# mock looser than real turns a guaranteed MEMs failure into a green suite (#84).
# PLACEMENT: this must come AFTER `struct RegModel` — mocks.jl is one flat module
# included top-to-bottom, so a method dispatching on ::RegModel defined earlier is an
# UndefVarError at include time, not a MethodError at call time.

struct RegDiagnosticResult{T<:AbstractFloat}
    test_name::String
    h0::String
    statistic::T
    pvalue::T
    df::Union{Int,Tuple{Int,Int}}
    f_stat::Union{Nothing,T}
    f_pvalue::Union{Nothing,T}
    f_df::Union{Nothing,Tuple{Int,Int}}
    aux_r2::T
    n::Int
end

struct StabilityResult{T<:AbstractFloat}
    kind::Symbol
    tindex::Vector{Int}
    stat_path::Vector{T}
    upper::Vector{T}
    lower::Vector{T}
    crossed::Bool
    first_crossing::Union{Nothing,Int}
    level::T
    recursive_resid::Vector{T}
    n::Int
    k::Int
end

struct InfluenceStats{T<:AbstractFloat}
    hat::Vector{T}
    student_internal::Vector{T}
    student_external::Vector{T}
    dffits::Vector{T}
    cooksd::Vector{T}
    dfbetas::Matrix{T}
    sigma::T
    high_leverage::Vector{Int}
    influential::Vector{Int}
    varnames::Vector{String}
    n::Int
    k::Int
end

# White/Glejser/Harvey all take (resid, X) with a RegModel convenience method, and all
# three return RegDiagnosticResult — exactly like real.
function white_test(resid::AbstractVector, X::AbstractMatrix; cross_terms::Bool=true)
    n = length(resid)
    size(X, 1) == n || throw(DimensionMismatch("length(resid)=$n must equal size(X,1)=$(size(X,1))"))
    df = cross_terms ? max(1, size(X, 2)) : max(1, size(X, 2) - 1)
    RegDiagnosticResult{Float64}("White test" * (cross_terms ? "" : " (no cross-terms)"),
        "Homoskedasticity (error variance unrelated to regressors)",
        7.4, 0.06, df, nothing, nothing, nothing, 0.12, n)
end
white_test(m::RegModel; cross_terms::Bool=true) =
    white_test(m.residuals, m.X; cross_terms=cross_terms)

function glejser_test(resid::AbstractVector, X::AbstractMatrix)
    n = length(resid)
    size(X, 1) == n || throw(DimensionMismatch("length(resid)=$n must equal size(X,1)=$(size(X,1))"))
    RegDiagnosticResult{Float64}("Glejser test",
        "Homoskedasticity (error variance unrelated to regressors)",
        5.1, 0.08, max(1, size(X, 2) - 1), 2.4, 0.09, (2, n - 3), 0.07, n)
end
glejser_test(m::RegModel) = glejser_test(m.residuals, m.X)

function harvey_test(resid::AbstractVector, X::AbstractMatrix)
    n = length(resid)
    size(X, 1) == n || throw(DimensionMismatch("length(resid)=$n must equal size(X,1)=$(size(X,1))"))
    RegDiagnosticResult{Float64}("Harvey test",
        "Homoskedasticity (multiplicative form)",
        4.2, 0.12, max(1, size(X, 2) - 1), nothing, nothing, nothing, 0.05, n)
end
harvey_test(m::RegModel) = harvey_test(m.residuals, m.X)

function chow_test(m::RegModel, break_index::Union{Integer,AbstractVector{<:Integer}};
                   type::Symbol=:breakpoint, level::Real=0.05)
    type ∈ (:breakpoint, :forecast) ||
        throw(ArgumentError("type must be :breakpoint or :forecast; got :$type"))
    n, k = size(m.X)
    breaks = sort(collect(Int, break_index isa Integer ? [break_index] : break_index))
    all(b -> 1 <= b < n, breaks) ||
        throw(ArgumentError("break index/indices must lie in 1:$(n-1) (got $breaks)"))
    if type === :breakpoint
        edges = vcat(0, breaks, n)
        for s in 1:(length(edges) - 1)
            (edges[s+1] - edges[s]) >= k ||
                throw(ArgumentError("segment $s has $(edges[s+1]-edges[s]) < k=$k observations; use type=:forecast"))
        end
    end
    df1 = k * length(breaks)
    RegDiagnosticResult{Float64}("Chow test ($(type))",
        "No structural break (coefficients constant across segments)",
        3.3, 0.04, (df1, n - k - df1), 3.3, 0.04, (df1, n - k - df1), 0.0, n)
end

function _mock_stability(m::RegModel, kind::Symbol, level::Real)
    (0 < level < 1) || throw(ArgumentError("level must lie in (0,1); got $level"))
    n, k = size(m.X)
    n > k + 2 || throw(ArgumentError("need more than k+2=$(k+2) observations, got $n"))
    idx = collect((k + 1):n)
    npath = length(idx)
    path = kind === :cusumsq ? collect(range(0.0, 1.0; length=npath)) : fill(0.3, npath)
    up = kind === :cusumsq ? [ (t - k) / (n - k) + 0.3 for t in idx ] : fill(1.2, npath)
    lo = kind === :cusumsq ? [ (t - k) / (n - k) - 0.3 for t in idx ] : fill(-1.2, npath)
    StabilityResult{Float64}(kind, idx, path, up, lo, false, nothing, Float64(level),
        fill(0.1, npath), n, k)
end

cusum_test(m::RegModel; level::Real=0.05) = _mock_stability(m, :cusum, level)
cusumsq_test(m::RegModel; level::Real=0.05) = _mock_stability(m, :cusumsq, level)

function recursive_residuals(m::RegModel)
    n, k = size(m.X)
    n > k || throw(ArgumentError("need n > k, got n=$n, k=$k"))
    return fill(0.1, n - k)
end

function influence_stats(m::RegModel)
    n, k = size(m.X)
    n > k || throw(ArgumentError("need n > k, got n=$n, k=$k"))
    InfluenceStats{Float64}(fill(Float64(k) / n, n), fill(0.2, n), fill(0.21, n),
        fill(0.05, n), fill(0.01, n), fill(0.02, n, k), 1.05,
        Int[], Int[], copy(m.varnames), n, k)
end

export RegDiagnosticResult, StabilityResult, InfluenceStats
export white_test, glejser_test, harvey_test, chow_test
export cusum_test, cusumsq_test, recursive_residuals, influence_stats

# ─── C067 (#72): variable selection. Field-order subset of real; validation mirrors
# reg/selection.jl so a bad --method/--criterion/p-threshold fails the same way.

struct SelectionResult{T<:AbstractFloat}
    method::Symbol
    criterion::Symbol
    selected::Vector{Int}
    keep::Vector{Int}
    varnames::Vector{String}
    path::Vector{Tuple{Symbol,Int,T}}
    terminal_models::Vector{Vector{Int}}
    encompassing_f::Union{Nothing,T}
    encompassing_pval::Union{Nothing,T}
    encompassing_df::Union{Nothing,Tuple{Int,Int}}
    final::RegModel{T}
    n_gum::Int
end

function select_variables(y::AbstractVector{T}, X::AbstractMatrix{T};
                          method::Symbol=:bidirectional, criterion::Symbol=:pvalue,
                          p_enter::Real=0.05, p_remove::Real=0.10, p_gets::Real=0.05,
                          diag_level::Real=0.05, bg_lags::Int=1,
                          keep=nothing, varnames=nothing) where T
    n, k = size(X)
    length(y) == n || throw(ArgumentError("X must have $(length(y)) rows (got $n)"))
    method ∈ (:forward, :backward, :bidirectional, :best_subset, :gets) ||
        throw(ArgumentError("method must be :forward, :backward, :bidirectional, :best_subset, or :gets; got :$method"))
    criterion ∈ (:pvalue, :aic, :bic) ||
        throw(ArgumentError("criterion must be :pvalue, :aic, or :bic; got :$criterion"))
    vn = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    length(vn) == k || throw(ArgumentError("varnames must have length $k"))
    kp = keep === nothing ? Int[] : collect(Int, keep)
    all(c -> 1 <= c <= k, kp) || throw(ArgumentError("keep indices must be in 1:$k"))
    method === :bidirectional && criterion === :pvalue && p_remove < p_enter &&
        throw(ArgumentError("bidirectional :pvalue search requires p_remove ≥ p_enter"))
    # Keep the first regressor plus anything forced; enough structure to exercise the
    # renderer without pretending to reproduce a real search.
    sel = sort(unique(vcat(kp, [1])))
    final = estimate_reg(y, X[:, sel]; varnames=vn[sel])
    path = Tuple{Symbol,Int,T}[(:enter, i, T(0.01)) for i in sel]
    SelectionResult{T}(method, criterion, sel, kp, vn, path, [sel],
                       T(1.5), T(0.22), (1, n - length(sel)), final, k)
end

export SelectionResult, select_variables

export RegModel, LogitModel, ProbitModel, MarginalEffects
export estimate_reg, estimate_iv, estimate_logit, estimate_probit
export marginal_effects, odds_ratio, vif, classification_table
export vcov, confint, r2

# ─── Advanced Unit Root Test Types & Functions ─────────────────

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
    statistic::T
    pvalue::T
    pt_statistic::T
    pt_pvalue::T
    MZa::T
    MZt::T
    MSB::T
    MPT::T
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    pt_critical_values::Dict{Int,T}
    mgls_critical_values::Dict{Symbol,Dict{Int,T}}
    nobs::Int
end

struct LMUnitRootResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    breaks::Int
    break_dates::Union{Nothing,Vector{Int}}
    break_fractions::Union{Nothing,Vector{T}}
    lags::Int
    regression::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

struct ADF2BreakResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    break1::Int
    break2::Int
    break1_fraction::T
    break2_fraction::T
    lags::Int
    model::Symbol
    critical_values::Dict{Int,T}
    nobs::Int
end

struct GregoryHansenResult{T<:AbstractFloat}
    adf_statistic::T
    adf_pvalue::T
    zt_statistic::T
    zt_pvalue::T
    za_statistic::T
    za_pvalue::T
    adf_break::Int
    zt_break::Int
    za_break::Int
    model::Symbol
    n_regressors::Int
    adf_critical_values::Dict{Int,T}
    za_critical_values::Dict{Int,T}
    nobs::Int
end

function fourier_adf_test(y::AbstractVector{T};
        regression=:constant, fmax=3, lags=:aic,
        max_lags=nothing, trim=0.15) where T
    n = length(y)
    p = lags == :aic ? max(1, round(Int, n^(1/3))) : lags
    freq = min(fmax, 3)
    cvs = Dict(1 => T(-4.82), 5 => T(-4.25), 10 => T(-3.96))
    f_cvs = Dict(1 => T(6.93), 5 => T(4.68), 10 => T(3.85))
    FourierADFResult{T}(T(-4.5), T(0.02), freq, T(8.5), T(0.005),
        p, regression, cvs, f_cvs, n)
end

function fourier_kpss_test(y::AbstractVector{T};
        regression=:constant, fmax=3, bandwidth=nothing) where T
    n = length(y)
    bw = isnothing(bandwidth) ? max(1, round(Int, n^(1/4))) : bandwidth
    freq = min(fmax, 3)
    cvs = Dict(1 => T(0.739), 5 => T(0.463), 10 => T(0.347))
    f_cvs = Dict(1 => T(6.93), 5 => T(4.68), 10 => T(3.85))
    FourierKPSSResult{T}(T(0.35), T(0.10), freq, T(5.2), T(0.01),
        regression, cvs, f_cvs, bw, n)
end

function dfgls_test(y::AbstractVector{T};
        regression=:constant, lags=:aic, max_lags=nothing) where T
    n = length(y)
    p = lags == :aic ? max(1, round(Int, n^(1/3))) : lags
    cvs = Dict(1 => T(-3.48), 5 => T(-2.89), 10 => T(-2.57))
    pt_cvs = Dict(1 => T(4.5), 5 => T(3.2), 10 => T(2.5))
    mgls_cvs = Dict(:MZa => cvs, :MZt => cvs, :MSB => cvs, :MPT => cvs)
    DFGLSResult{T}(T(-3.2), T(0.02), T(4.5), T(0.05),
        T(-15.0), T(-2.7), T(0.18), T(3.5),
        p, regression, cvs, pt_cvs, mgls_cvs, n)
end

function lm_unitroot_test(y::AbstractVector{T};
        breaks=0, regression=:level, lags=:aic,
        max_lags=nothing, trim=0.15) where T
    n = length(y)
    p = lags == :aic ? max(1, round(Int, n^(1/3))) : lags
    cvs = Dict(1 => T(-4.24), 5 => T(-3.57), 10 => T(-3.21))
    bi = breaks > 0 ? [div(n, i + 1) for i in 1:breaks] : nothing
    bf = breaks > 0 ? [T(1.0 / (i + 1)) for i in 1:breaks] : nothing
    LMUnitRootResult{T}(T(-3.8), T(0.03), breaks, bi, bf, p, regression, cvs, n)
end

function adf_2break_test(y::AbstractVector{T};
        model=:level, lags=:aic, max_lags=nothing, trim=0.10) where T
    n = length(y)
    p = lags == :aic ? max(1, round(Int, n^(1/3))) : lags
    b1 = div(n, 3)
    b2 = div(2n, 3)
    cvs = Dict(1 => T(-5.65), 5 => T(-5.13), 10 => T(-4.82))
    ADF2BreakResult{T}(T(-5.3), T(0.03), b1, b2, T(b1 / n), T(b2 / n),
        p, model, cvs, n)
end

function gregory_hansen_test(Y::AbstractMatrix{T};
        model=:C, lags=:aic, max_lags=nothing, trim=0.15) where T
    # Real rejects a single-column matrix (teststat/gregory_hansen.jl) — a
    # cointegrating regression needs a dependent plus at least one regressor.
    size(Y, 2) >= 2 || throw(ArgumentError("Need at least 2 columns (dependent + regressor)"))
    n = size(Y, 1)
    bp = div(n, 2)
    cvs = Dict(1 => T(-5.13), 5 => T(-4.61), 10 => T(-4.34))
    GregoryHansenResult{T}(T(-4.8), T(0.03), T(-4.5), T(0.04), T(-35.0), T(0.02),
        bp, bp + 2, bp - 1, model, 1, cvs, cvs, n)
end

export FourierADFResult, FourierKPSSResult, DFGLSResult
export LMUnitRootResult, ADF2BreakResult, GregoryHansenResult
export fourier_adf_test, fourier_kpss_test, dfgls_test
export lm_unitroot_test, adf_2break_test, gregory_hansen_test

# ─── Bayesian DSGE Enhancements ────────────────────────────

struct BayesianDSGESimulation{T<:AbstractFloat}
    quantiles::Array{T,3}
    point_estimate::Matrix{T}
    all_paths::Array{T,3}
    variables::Vector{String}
    quantile_levels::Vector{T}
end

# irf dispatch on BayesianDSGE
function irf(result::BayesianDSGE{T}, horizon::Int;
        n_draws=200, quantiles=[0.05, 0.16, 0.84, 0.95],
        solver=:gensys, solver_kwargs=NamedTuple(), rng=nothing) where T
    nv = length(result.param_names)
    ns = max(1, nv)
    q = Array{T,4}(undef, horizon + 1, nv, ns, length(quantiles))
    fill!(q, T(0.1))
    m = zeros(T, horizon + 1, nv, ns)
    BayesianImpulseResponse(m, q, T.(quantiles))
end

# fevd dispatch on BayesianDSGE
function fevd(result::BayesianDSGE{T}, horizon::Int;
        n_draws=200, quantiles=[0.05, 0.16, 0.84, 0.95],
        solver=:gensys, solver_kwargs=NamedTuple(), rng=nothing) where T
    nv = length(result.param_names)
    ns = max(1, nv)
    q = Array{T,4}(undef, horizon, nv, ns, length(quantiles))
    fill!(q, T(1.0 / ns))
    m = fill(T(1.0 / ns), horizon, nv, ns)
    BayesianFEVD(m, q, T.(quantiles))
end

# simulate dispatch on BayesianDSGE
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
    # Match real MEMs: return the LOG Bayes factor (logML₁ − logML₂), positive favors M1.
    r1.log_marginal_likelihood - r2.log_marginal_likelihood
end

function prior_posterior_table(result::BayesianDSGE{T}) where T
    [(param=p, prior_mean=T(0.5), prior_std=T(0.2),
      post_mean=T(0.5), post_std=T(0.1), post_q05=T(0.3), post_q95=T(0.7))
     for p in result.param_names]
end

function posterior_predictive(result::BayesianDSGE{T}, n_sim::Int;
        T_periods=100, rng=nothing, seed=nothing) where T
    nv = length(result.param_names)
    randn(T, n_sim, T_periods, nv)
end

export BayesianDSGESimulation
export posterior_summary, bayes_factor, prior_posterior_table, posterior_predictive

# ─── GPL Notice Functions ────────────────────────────────────

function warranty()
    println("THERE IS NO WARRANTY FOR THE PROGRAM (mock)")
    nothing
end

function conditions()
    println("You may convey verbatim copies of the Program (mock)")
    nothing
end

export warranty, conditions

# ─── DSGE Historical Decomposition (v0.4.0) ──────────────────

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
    varnames_hd = states == :all ? sol.spec.varnames : [string(s) for s in observables]
    shock_names = string.(sol.spec.exog)
    HistoricalDecomposition{T}(
        randn(T_obs, n_vars, n_shocks), randn(T_obs, n_vars), randn(T_obs, n_vars),
        randn(T_obs, n_shocks), T_obs, varnames_hd, shock_names, :dsge_linear)
end

struct BayesianDSGEHistoricalDecomposition{T<:Real}
    quantiles::Array{T,4}
    point_estimate::Array{T,3}
    initial_quantiles::Array{T,3}
    initial_point_estimate::Matrix{T}
    shocks_point_estimate::Matrix{T}
    actual::Matrix{T}
    T_eff::Int
    variables::Vector{String}
    shock_names::Vector{String}
    quantile_levels::Vector{T}
    method::Symbol
end

function historical_decomposition(bd::BayesianDSGE{T}, data::AbstractMatrix,
        observables::Vector{Symbol}; mode_only::Bool=false, n_draws::Int=200,
        quantiles::Vector{<:Real}=T[0.16, 0.5, 0.84],
        measurement_error=nothing, states::Symbol=:observables) where {T}
    T_obs = size(data, 1)
    n_obs = length(observables)
    n_shocks = bd.spec.n_exog
    n_q = length(quantiles)
    varnames_bd = [string(s) for s in observables]
    shock_names = string.(bd.spec.exog)
    BayesianDSGEHistoricalDecomposition{T}(
        randn(T_obs, n_obs, n_shocks, n_q), randn(T_obs, n_obs, n_shocks),
        randn(T_obs, n_obs, n_q), randn(T_obs, n_obs), randn(T_obs, n_shocks),
        randn(T_obs, n_obs), T_obs, varnames_bd, shock_names, T.(quantiles), :dsge_bayes)
end

contribution(hd::BayesianDSGEHistoricalDecomposition, var::Int, shock::Int) = hd.point_estimate[:, var, shock]
total_shock_contribution(hd::HistoricalDecomposition, var::Int) = dropdims(sum(hd.contributions[:, var, :]; dims=2); dims=2)
verify_decomposition(hd::BayesianDSGEHistoricalDecomposition) = _MOCK_FLAGS[:verify_decomposition]

function dsge_particle_smoother(args...; kwargs...)
    nothing
end

export KalmanSmootherResult, BayesianDSGEHistoricalDecomposition
export dsge_smoother, dsge_particle_smoother
export total_shock_contribution

# ─── Spectral Analysis Types & Functions (v0.4.0) ────────────

struct ACFResult{T<:AbstractFloat}
    lags::Vector{Int}
    acf::Vector{T}
    pacf::Vector{T}
    ci::T
    ccf::Union{Nothing,Vector{T}}
    q_stats::Vector{T}
    q_pvalues::Vector{T}
    nobs::Int
end

struct SpectralDensityResult{T<:AbstractFloat}
    freq::Vector{T}
    density::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    method::Symbol
    bandwidth::T
    nobs::Int
end

struct CrossSpectrumResult{T<:AbstractFloat}
    freq::Vector{T}
    co_spectrum::Vector{T}
    quad_spectrum::Vector{T}
    coherence::Vector{T}
    phase::Vector{T}
    gain::Vector{T}
    nobs::Int
end

struct TransferFunctionResult{T<:AbstractFloat}
    freq::Vector{T}
    gain::Vector{T}
    phase::Vector{T}
    filter::Symbol
end

struct FisherTestResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    peak_freq::T
    nobs::Int
end

struct BartlettWhiteNoiseResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    nobs::Int
end

struct BoxPierceResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    df::Int
    lags::Int
    nobs::Int
end

struct DurbinWatsonResult{T<:AbstractFloat}
    statistic::T
    pvalue::T
    nobs::Int
end

function acf(y::AbstractVector{T}; lags::Int=20, maxlag::Union{Int,Nothing}=nothing,
             conf_level::Real=0.95, varname::String="y") where T
    n = length(y)
    nlags = isnothing(maxlag) ? lags : maxlag
    acf_vals = [T(0.9)^k for k in 0:nlags]
    pacf_vals = [k == 0 ? T(1.0) : T(0.9) * T(0.5)^(k-1) for k in 0:nlags]
    ci = T(1.96) / sqrt(n)
    q_stats = [T(k+1) * T(0.1) for k in 0:nlags]
    q_pvals = [T(0.05) for _ in 0:nlags]
    lags_vec = collect(0:nlags)
    ACFResult{T}(lags_vec, acf_vals, pacf_vals, ci, nothing, q_stats, q_pvals, n)
end

function pacf(y::AbstractVector{T}; lags::Int=20, maxlag::Union{Int,Nothing}=nothing,
              conf_level::Real=0.95, varname::String="y") where T
    acf(y; lags=lags, maxlag=maxlag, conf_level=conf_level, varname=varname)
end

struct CCFResult{T<:AbstractFloat}
    ccf::Vector{T}
    lags::Vector{Int}
    conf_level::T
    ci_band::T
    varnames::Tuple{String,String}
    nobs::Int
end

function ccf(y1::AbstractVector{T}, y2::AbstractVector{T}; lags::Int=20,
             maxlag::Union{Int,Nothing}=nothing, conf_level::Real=0.95,
             var1::String="y1", var2::String="y2") where T
    n = length(y1)
    nlags = isnothing(maxlag) ? lags : maxlag
    ccf_vals = [T(0.5) * T(0.8)^abs(k) for k in -nlags:nlags]
    ci = T(1.96) / sqrt(n)
    lags_vec = collect(-nlags:nlags)
    CCFResult{T}(ccf_vals, lags_vec, T(conf_level), ci, (var1, var2), n)
end

function periodogram(y::AbstractVector{T}; varname::String="y") where T
    n = length(y)
    freqs = [T(k) / n for k in 1:div(n, 2)]
    spec = abs2.(randn(T, length(freqs))) .+ T(0.01)
    log_spec = log.(spec)
    SpectralDensityResult{T}(freqs, spec, fill(T(0.0), length(freqs)), fill(T(0.0), length(freqs)), :periodogram, T(0.0), n)
end

function spectral_density(y::AbstractVector{T}; method::Symbol=:welch, bandwidth=nothing,
                          kernel::Symbol=:bartlett, varname::String="y") where T
    n = length(y)
    freqs = [T(k) / n for k in 1:div(n, 2)]
    bw = isnothing(bandwidth) ? T(sqrt(n)) : T(bandwidth)
    dens = abs2.(randn(T, length(freqs))) .+ T(0.01)
    SpectralDensityResult{T}(freqs, dens, dens .* T(0.5), dens .* T(1.5), method, bw, n)
end

function cross_spectrum(y1::AbstractVector{T}, y2::AbstractVector{T};
                        bandwidth=nothing, kernel::Symbol=:bartlett,
                        var1::String="y1", var2::String="y2") where T
    n = length(y1)
    freqs = [T(k) / n for k in 1:div(n, 2)]
    nf = length(freqs)
    cs = randn(Complex{T}, nf)
    coh = abs.(cs) ./ (abs.(cs) .+ T(0.1))
    ph = angle.(cs)
    gain = abs.(cs)
    CrossSpectrumResult{T}(freqs, real.(cs), imag.(cs), coh, ph, gain, n)
end

function transfer_function(input::AbstractVector{T}, output::AbstractVector{T};
                           bandwidth=nothing, kernel::Symbol=:bartlett,
                           var_input::String="input", var_output::String="output") where T
    n = length(input)
    freqs = [T(k) / n for k in 1:div(n, 2)]
    nf = length(freqs)
    gain = abs.(randn(T, nf)) .+ T(0.5)
    phase = randn(T, nf)
    coherence = rand(T, nf) .* T(0.8) .+ T(0.1)
    TransferFunctionResult{T}(freqs, gain, phase, :empirical)
end

function transfer_function(filter_name::Symbol; lambda::Real=1600.0, nobs::Int=200,
                           kwargs...)
    n = nobs
    freqs = [Float64(k) / n for k in 1:div(n, 2)]
    nf = length(freqs)
    gain = abs.(randn(nf)) .+ 0.5
    phase = randn(nf)
    TransferFunctionResult{Float64}(freqs, gain, phase, filter_name)
end

function fisher_test(y::AbstractVector{T}) where T
    n = length(y)
    FisherTestResult{T}(T(8.5), T(0.05), T(0.1), n)
end

function bartlett_white_noise_test(y::AbstractVector{T}; lags::Int=20) where T
    n = length(y)
    BartlettWhiteNoiseResult{T}(T(12.0), T(0.15), n)
end

function box_pierce_test(y::AbstractVector{T}; lags::Int=20, ljung_box::Bool=true) where T
    n = length(y)
    BoxPierceResult{T}(T(25.0), T(0.10), lags, lags, n)
end

function durbin_watson_test(residuals::AbstractVector{T}) where T
    n = length(residuals)
    DurbinWatsonResult{T}(T(2.0), T(0.5), n)
end

# Field aliases for spectral handler compat (legacy mock names → real fields)

export ACFResult, SpectralDensityResult, CrossSpectrumResult, TransferFunctionResult
export FisherTestResult, BartlettWhiteNoiseResult, BoxPierceResult, DurbinWatsonResult
export ACFResult, CCFResult
export acf, pacf, ccf, periodogram, spectral_density, cross_spectrum, transfer_function
export fisher_test, bartlett_white_noise_test, box_pierce_test, durbin_watson_test

# ─── Panel Regression Types & Functions (v0.4.0) ─────────────

struct PanelRegModel{T<:Real}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T
    sigma_e::T
    rho::T
    theta::T
    f_stat::T
    f_pval::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    method::Symbol
    twoway::Bool
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    n_periods_avg::T
    group_effects::Union{Nothing,Vector{T}}
    data::PanelData{T}
    dynamic_diagnostics::Union{Nothing,NamedTuple}
    # Real's tail (preg/types.jl): Prais-Winsten rho and the HDFE absorption diagnostics.
    # `hdfe` is `nothing` unless `absorb=` was used (W10/#112).
    ar1_rho::Union{Nothing,T,Vector{T}}
    hdfe::Union{Nothing,NamedTuple}
end

struct PanelIVModel{T<:Real}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    y::Vector{T}
    X::Matrix{T}
    Z::Matrix{T}
    r2_within::T
    r2_between::T
    r2_overall::T
    sigma_u::T
    sigma_e::T
    rho::T
    # Real (preg/types.jl): first_stage_f is NOT nullable (Inf when no endog,
    # NaN when degenerate); the other five are Union{Nothing,T} — nothing means
    # the computation failed OR the model is just-/under-identified.
    first_stage_f::T
    sargan_stat::Union{Nothing,T}
    sargan_pval::Union{Nothing,T}
    cragg_donald_f::Union{Nothing,T}
    kleibergen_paap_f::Union{Nothing,T}
    stock_yogo_10pct::Union{Nothing,T}
    varnames::Vector{String}
    endog_names::Vector{String}
    instrument_names::Vector{String}
    method::Symbol
    cov_type::Symbol
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

struct PanelLogitModel{T<:Real}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    y::Vector{T}
    X::Matrix{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    sigma_u::T
    rho::T
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    converged::Bool
    iterations::Int
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

struct PanelProbitModel{T<:Real}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    y::Vector{T}
    X::Matrix{T}
    fitted::Vector{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    sigma_u::T
    rho::T
    varnames::Vector{String}
    method::Symbol
    cov_type::Symbol
    converged::Bool
    iterations::Int
    n_obs::Int
    n_groups::Int
    data::PanelData{T}
end

struct PanelTestResult{T<:Real}
    test_name::String
    statistic::T
    pvalue::T
    df::Union{Int,Tuple{Int,Int},Nothing}
    description::String
end

# StatsAPI dispatches for panel regression types
coef(m::PanelRegModel) = m.beta
coef(m::PanelIVModel) = m.beta
coef(m::PanelLogitModel) = m.beta
coef(m::PanelProbitModel) = m.beta
vcov(m::PanelRegModel) = m.vcov_mat
vcov(m::PanelIVModel) = m.vcov_mat
vcov(m::PanelLogitModel) = m.vcov_mat
vcov(m::PanelProbitModel) = m.vcov_mat
residuals(m::PanelRegModel) = m.residuals
residuals(m::PanelIVModel) = m.residuals
residuals(m::PanelLogitModel) = m.y .- m.fitted
residuals(m::PanelProbitModel) = m.y .- m.fitted
predict(m::PanelRegModel) = m.fitted
predict(m::PanelIVModel) = m.fitted
predict(m::PanelLogitModel) = m.fitted
predict(m::PanelProbitModel) = m.fitted
stderror(m::PanelRegModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
stderror(m::PanelIVModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
stderror(m::PanelLogitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
stderror(m::PanelProbitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
nobs(m::PanelRegModel) = m.n_obs
nobs(m::PanelIVModel) = m.n_obs
nobs(m::PanelLogitModel) = m.n_obs
nobs(m::PanelProbitModel) = m.n_obs
loglikelihood(m::PanelRegModel) = m.loglik
loglikelihood(m::PanelLogitModel) = m.loglik
loglikelihood(m::PanelProbitModel) = m.loglik
aic(m::PanelRegModel) = m.aic
aic(m::PanelLogitModel) = m.aic
aic(m::PanelProbitModel) = m.aic
bic(m::PanelRegModel) = m.bic
bic(m::PanelLogitModel) = m.bic
bic(m::PanelProbitModel) = m.bic
r2(m::PanelRegModel) = m.within_r2
r2(m::PanelLogitModel) = m.pseudo_r2
r2(m::PanelProbitModel) = m.pseudo_r2
confint(m::PanelRegModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))
confint(m::PanelIVModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))
confint(m::PanelLogitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))
confint(m::PanelProbitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

function estimate_xtreg(pd::PanelData{T}, outcome, covariates;
        model=:fe, twoway=false, fe=:twoway, cov_type=:cluster, clusters=nothing,
        varnames=nothing, ar1::Symbol=:none, pcse_unbalanced::Symbol=:casewise,
        absorb::Vector{Symbol}=Symbol[], hdfe_tol::Real=1e-8, hdfe_maxiter::Int=1000,
        hdfe_accel::Bool=true,
        collapse::Bool=false, min_lag_endo::Int=2, max_lag_endo::Int=99) where T
    # Real validates the model symbol up front (preg/estimation.jl).
    model in (:fe, :re, :fd, :between, :cre, :ab, :bb) ||
        throw(ArgumentError("model must be :fe, :re, :fd, :between, :cre, :ab, or :bb; got :$model"))
    # Mirror real's validation (preg/estimation.jl) for the #75 additions.
    # W10/#112 absorb guards, verbatim from real — including the twoway exclusivity whose
    # error message points at `absorb=[:entity, :time]` (the correct route on an UNBALANCED
    # panel, where the additive two-way identity does not hold).
    if !isempty(absorb)
        model === :fe || throw(ArgumentError(
            "absorb= is supported only for model=:fe (the within estimator); got :$model"))
        twoway && throw(ArgumentError(
            "absorb= and twoway=true are mutually exclusive — pass absorb=[:entity, :time] " *
            "to absorb entity and time fixed effects"))
        length(unique(absorb)) == length(absorb) || throw(ArgumentError(
            "absorb contains duplicate dimensions: $absorb"))
        hdfe_maxiter >= 1 || throw(ArgumentError("maxiter must be >= 1, got $hdfe_maxiter"))
        hdfe_tol > 0 || throw(ArgumentError("tol must be positive, got $hdfe_tol"))
        # Real resolves each dimension against the panel: a matching variable column wins,
        # else the reserved index aliases. An unknown name is an ArgumentError there too.
        for d in absorb
            (String(d) in pd.varnames ||
             d in (:entity, :id, :unit, :group, :time, :period, :cohort)) || throw(ArgumentError(
                "absorb dimension :$d not found in panel data. Available: $(pd.varnames) " *
                "or the reserved indices :entity, :time, :cohort"))
        end
    end
    cov_type in (:ols, :cluster, :twoway, :driscoll_kraay, :pcse) ||
        throw(ArgumentError("cov_type must be :ols, :cluster, :twoway, :driscoll_kraay, or :pcse; got :$cov_type"))
    pcse_unbalanced in (:casewise, :pairwise) ||
        throw(ArgumentError("pcse_unbalanced must be :casewise or :pairwise; got :$pcse_unbalanced"))
    ar1 in (:none, :common, :panel_specific) ||
        throw(ArgumentError("ar1 must be :none, :common, or :panel_specific; got :$ar1"))
    n, k = pd.T_obs, length(covariates) + 1
    beta = ones(T, k) * T(0.5)
    vcov_mat = Matrix{T}(I(k)) * T(0.01)
    y = pd.data[:, 1]
    X = ones(T, n, k)
    fitted_vals = X * beta
    resids = y .- fitted_vals
    vnames = varnames === nothing ? ["const"; string.(covariates)] : varnames
    meth = model isa Symbol ? model : :fe
    hdfe_info = isempty(absorb) ? nothing :
        (absorb = copy(absorb),
         n_absorbed = sum(3 for _ in absorb) - length(absorb) + 1,
         n_levels = [3 for _ in absorb],
         n_components = 1,
         marginal = [true for _ in absorb],
         n_absorbed_cluster = 0,
         converged = true,
         iterations = 4,
         sweeps = 8,
         change = 1e-10,
         tol = Float64(hdfe_tol),
         accel = hdfe_accel)
    # Real's :ab/:bb path routes through PVAR GMM and returns a PanelRegModel
    # whose dynamic_diagnostics NamedTuple carries the AB tests + Hansen J +
    # n_instruments (W10/#131). The count responds to the proliferation controls
    # the same direction as real: collapsing shrinks it, widening the lag window
    # grows it.
    dyn = if model in (:ab, :bb)
        maxl = min(max_lag_endo, 8)
        n_inst = collapse ? (maxl - min_lag_endo + 1) : (maxl - min_lag_endo + 1) * 5
        (ar1 = T(-2.1), ar1_p = T(0.036), ar2 = T(0.4), ar2_p = T(0.69),
         hansen = T(11.2), hansen_df = max(n_inst - length(covariates) - 1, 1),
         hansen_p = T(0.34), n_instruments = n_inst)
    else
        nothing
    end
    model in (:ab, :bb) && (vnames = ["L.$(outcome)"; string.(covariates)])
    PanelRegModel{T}(beta, vcov_mat, resids, fitted_vals, y, X,
        T(0.35), T(0.25), T(0.30), T(0.5), T(1.0), T(0.2), T(0.5),
        T(20.0), T(0.001), T(-200.0), T(410.0), T(420.0),
        vnames, meth, Bool(twoway), cov_type, n, pd.n_groups, T(n / max(pd.n_groups, 1)),
        nothing, pd, dyn, nothing, hdfe_info)
end

function estimate_xtiv(pd::PanelData{T}, outcome, covariates, endog=Symbol[];
        instruments=Symbol[], model=:fe, fe=:twoway, cov_type=:cluster, clusters=nothing, varnames=nothing) where T
    n, k = pd.T_obs, length(covariates) + 1
    kz = max(1, length(instruments) + 1)
    beta = ones(T, k) * T(0.5)
    vcov_mat = Matrix{T}(I(k)) * T(0.01)
    y = pd.data[:, 1]
    X = ones(T, n, k)
    Z = ones(T, n, kz)
    fitted_vals = X * beta
    resids = y .- fitted_vals
    vnames = varnames === nothing ? ["const"; string.(covariates)] : varnames
    endog_names = string.(endog)
    inst_names = string.(instruments)
    # Real: Sargan is nothing when just-identified (dof = n_inst - n_endog <= 0).
    overid = length(instruments) > length(endog)
    sargan_s = overid ? T(2.5) : nothing
    sargan_p = overid ? T(0.30) : nothing
    PanelIVModel{T}(beta, vcov_mat, resids, fitted_vals, y, X, Z,
        T(0.30), T(0.20), T(0.25), T(0.5), T(1.0), T(0.2),
        T(12.0), sargan_s, sargan_p, T(10.0), T(9.0), T(7.0),
        vnames, endog_names, inst_names, model isa Symbol ? model : :fe, cov_type,
        n, pd.n_groups, pd)
end

function estimate_xtlogit(pd::PanelData{T}, outcome, covariates;
        model=:pooled, fe=:fe, cov_type=:cluster, clusters=nothing, varnames=nothing,
        maxiter=100, tol=1e-8) where T
    n, k = pd.T_obs, length(covariates) + 1
    beta = ones(T, k) * T(0.3)
    vcov_mat = Matrix{T}(I(k)) * T(0.02)
    y = pd.data[:, 1]
    X = ones(T, n, k)
    fitted_vals = ones(T, n) * T(0.5)
    vnames = varnames === nothing ? ["const"; string.(covariates)] : varnames
    PanelLogitModel{T}(beta, vcov_mat, y, X, fitted_vals,
        T(-80.0), T(-100.0), T(0.20), T(170.0), T(180.0),
        T(0.5), T(0.2), vnames, model isa Symbol ? model : :pooled, cov_type,
        true, 10, n, pd.n_groups, pd)
end

function estimate_xtprobit(pd::PanelData{T}, outcome, covariates;
        model=:pooled, fe=:re, cov_type=:cluster, clusters=nothing, varnames=nothing,
        maxiter=100, tol=1e-8) where T
    n, k = pd.T_obs, length(covariates) + 1
    beta = ones(T, k) * T(0.3)
    vcov_mat = Matrix{T}(I(k)) * T(0.02)
    y = pd.data[:, 1]
    X = ones(T, n, k)
    fitted_vals = ones(T, n) * T(0.5)
    vnames = varnames === nothing ? ["const"; string.(covariates)] : varnames
    PanelProbitModel{T}(beta, vcov_mat, y, X, fitted_vals,
        T(-80.0), T(-100.0), T(0.20), T(170.0), T(180.0),
        T(0.5), T(0.2), vnames, model isa Symbol ? model : :pooled, cov_type,
        true, 10, n, pd.n_groups, pd)
end

function hausman_test(fe_model::PanelRegModel{T}, re_model::PanelRegModel{T}) where T
    k = length(fe_model.beta)
    PanelTestResult{T}("Hausman", T(10.0), T(0.01), k, "FE vs RE")
end

function breusch_pagan_test(model::PanelRegModel{T}) where T
    PanelTestResult{T}("Breusch-Pagan LM", T(45.0), T(0.001), 1, "LM test for random effects")
end

function f_test_fe(model::PanelRegModel{T}) where T
    df2 = model.n_obs - model.n_groups - length(model.beta)
    PanelTestResult{T}("F-test for FE", T(12.0), T(0.001), (model.n_groups - 1, df2), "joint FE significance")
end

function pesaran_cd_test(model::Union{PanelRegModel{T},PanelLogitModel{T},PanelProbitModel{T}}) where T
    PanelTestResult{T}("Pesaran CD", T(3.5), T(0.001), 0, "cross-sectional dependence")
end

function wooldridge_ar_test(model::PanelRegModel{T}) where T
    PanelTestResult{T}("Wooldridge AR(1)", T(5.2), T(0.02), 1, "serial correlation in panel")
end

function modified_wald_test(model::PanelRegModel{T}) where T
    PanelTestResult{T}("Modified Wald", T(28.0), T(0.005), model.n_groups, "groupwise heteroskedasticity")
end

export PanelRegModel, PanelIVModel, PanelLogitModel, PanelProbitModel, PanelTestResult
export estimate_xtreg, estimate_xtiv, estimate_xtlogit, estimate_xtprobit
export hausman_test, breusch_pagan_test, f_test_fe, pesaran_cd_test
export wooldridge_ar_test, modified_wald_test

# ─── Ordered/Multinomial & Data Utilities (v0.4.0) ───────────

struct OrderedLogitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    fitted::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    categories::Vector{Int}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct OrderedProbitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Vector{T}
    cutpoints::Vector{T}
    vcov_mat::Matrix{T}
    fitted::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    categories::Vector{Int}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

struct MultinomialLogitModel{T<:Real}
    y::Vector{T}
    X::Matrix{T}
    beta::Matrix{T}
    vcov_mat::Array{T,3}
    fitted::Matrix{T}
    loglik::T
    loglik_null::T
    pseudo_r2::T
    aic::T
    bic::T
    varnames::Vector{String}
    categories::Vector{Int}
    converged::Bool
    iterations::Int
    cov_type::Symbol
end

# StatsAPI dispatches for ordered/multinomial models
coef(m::OrderedLogitModel) = vcat(m.beta, m.cutpoints)
coef(m::OrderedProbitModel) = vcat(m.beta, m.cutpoints)
coef(m::MultinomialLogitModel) = vec(m.beta)
vcov(m::OrderedLogitModel) = m.vcov_mat
vcov(m::OrderedProbitModel) = m.vcov_mat
vcov(m::MultinomialLogitModel) = m.vcov_mat[:, :, 1]
# W4/#87 — mirror real MEMs 0.7.2 (MEMs#507). These three used to return
# `m.y .- m.fitted[:, 1]`, a length-n vector that real MEMs NEVER defined: the exact #84
# trap, a mock method more permissive than real. Real returns an n x J matrix (one column
# per category) and takes `kind`; the CLI renders it with _choice_resid_table, so the mock
# MUST produce the same SHAPE or the renderer is only ever exercised in production.
# Real MEMs recodes the response to integer category indices 1..J before storing it; the
# mock keeps whatever the caller passed, and the shared T1/T2 fixture hands these
# estimators CONTINUOUS columns, so `Int(y[i])` would throw InexactError. Rank the observed
# values instead — that IS the category index for an ordered response — and clamp to the
# fitted width so a fixture with more distinct values than columns still renders.
function _cat_index(y, J::Int)
    pos = Dict(c => i for (i, c) in enumerate(sort(unique(y))))
    return [clamp(pos[v], 1, J) for v in y]
end

function _category_residuals(y, fitted::AbstractMatrix, kind::Symbol)
    n, J = size(fitted)
    D = zeros(Float64, n, J)
    idx = _cat_index(y, J)
    for i in 1:n
        D[i, idx[i]] = 1.0
    end
    R = D .- fitted
    kind === :response && return R
    kind === :pearson && return R ./ sqrt.(max.(fitted .* (1 .- fitted), eps()))
    kind === :deviance &&
        return sign.(R) .* sqrt.(max.(-2 .* D .* log.(max.(fitted, eps())), 0.0))
    throw(ArgumentError("residual kind must be :response, :pearson or :deviance, got :$kind"))
end

residuals(m::OrderedLogitModel; kind::Symbol=:response) =
    _category_residuals(m.y, m.fitted, kind)
residuals(m::OrderedProbitModel; kind::Symbol=:response) =
    _category_residuals(m.y, m.fitted, kind)
residuals(m::MultinomialLogitModel; kind::Symbol=:response) =
    _category_residuals(m.y, m.fitted, kind)

# Length-n score residual — ORDERED MODELS ONLY, matching real. Upstream deliberately
# defines no generalized_residuals for MultinomialLogitModel (an unordered response has no
# meaningful length-n scalar residual), so the mock must not define one either.
_mock_gen_resid(m) = begin
    n, J = size(m.fitted)
    idx = _cat_index(m.y, J)
    [1.0 - m.fitted[i, idx[i]] for i in 1:n]
end
generalized_residuals(m::OrderedLogitModel) = _mock_gen_resid(m)
generalized_residuals(m::OrderedProbitModel) = _mock_gen_resid(m)
predict(m::OrderedLogitModel) = m.fitted[:, 1]
predict(m::OrderedProbitModel) = m.fitted[:, 1]
predict(m::MultinomialLogitModel) = m.fitted[:, 1]

# Compat aliases for handlers still using legacy field names
stderror(m::OrderedLogitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
stderror(m::OrderedProbitModel) = [sqrt(m.vcov_mat[i,i]) for i in 1:size(m.vcov_mat,1)]
stderror(m::MultinomialLogitModel) = vcat([[sqrt(m.vcov_mat[i,i,c]) for i in 1:size(m.vcov_mat,1)] for c in 1:size(m.vcov_mat,3)]...)
nobs(m::OrderedLogitModel) = length(m.y)
nobs(m::OrderedProbitModel) = length(m.y)
nobs(m::MultinomialLogitModel) = length(m.y)
loglikelihood(m::OrderedLogitModel) = m.loglik
loglikelihood(m::OrderedProbitModel) = m.loglik
loglikelihood(m::MultinomialLogitModel) = m.loglik
aic(m::OrderedLogitModel) = m.aic
aic(m::OrderedProbitModel) = m.aic
aic(m::MultinomialLogitModel) = m.aic
bic(m::OrderedLogitModel) = m.bic
bic(m::OrderedProbitModel) = m.bic
bic(m::MultinomialLogitModel) = m.bic
r2(m::OrderedLogitModel) = m.pseudo_r2
r2(m::OrderedProbitModel) = m.pseudo_r2
r2(m::MultinomialLogitModel) = m.pseudo_r2

function _build_ordered(::Type{M}, y::AbstractVector{T}, X::AbstractMatrix{T};
        n_categories::Int=3, cov_type::Symbol=:hc1, varnames=nothing,
        maxiter::Int=100, tol::Real=1e-8) where {T, M}
    n, k = size(X)
    beta = ones(T, k) * T(0.3)
    cutpoints = [T(c) * T(0.5) for c in 1:(n_categories - 1)]
    nb = k + length(cutpoints)
    vcov_mat = Matrix{T}(I(nb)) * T(0.02)
    fitted_mat = ones(T, n, n_categories) / n_categories
    ll = T(-80.0); ll_null = T(-100.0)
    pseudo = one(T) - ll / ll_null
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    cats = collect(1:n_categories)
    M{T}(y, X, beta, cutpoints, vcov_mat, fitted_mat, ll, ll_null, pseudo,
         T(170.0), T(180.0), vnames, cats, true, 15, cov_type)
end

function estimate_ologit(y::AbstractVector{T}, X::AbstractMatrix{T};
        n_categories::Int=3, cov_type::Symbol=:hc1, varnames=nothing, clusters=nothing,
        maxiter::Int=100, tol::Real=1e-8) where T
    _build_ordered(OrderedLogitModel, y, X; n_categories=n_categories,
                   cov_type=cov_type, varnames=varnames, maxiter=maxiter, tol=tol)
end

function estimate_oprobit(y::AbstractVector{T}, X::AbstractMatrix{T};
        n_categories::Int=3, cov_type::Symbol=:hc1, varnames=nothing, clusters=nothing,
        maxiter::Int=100, tol::Real=1e-8) where T
    _build_ordered(OrderedProbitModel, y, X; n_categories=n_categories,
                   cov_type=cov_type, varnames=varnames, maxiter=maxiter, tol=tol)
end

function estimate_mlogit(y::AbstractVector{T}, X::AbstractMatrix{T};
        n_categories::Int=3, base_category::Int=1, cov_type::Symbol=:hc1,
        varnames=nothing, clusters=nothing, maxiter::Int=100, tol::Real=1e-8) where T
    n, k = size(X)
    nc = n_categories
    beta = ones(T, k, nc) * T(0.3)
    beta[:, base_category] .= T(0.0)
    vcov_mat = zeros(T, k, k, nc)
    for c in 1:nc
        vcov_mat[:, :, c] = Matrix{T}(I(k)) * T(0.02)
    end
    fitted_mat = ones(T, n, nc) / nc
    ll = T(-90.0); ll_null = T(-110.0)
    pseudo = one(T) - ll / ll_null
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    cats = collect(1:nc)
    MultinomialLogitModel{T}(y, X, beta, vcov_mat, fitted_mat, ll, ll_null, pseudo,
        T(190.0), T(200.0), vnames, cats, true, 20, cov_type)
end

# Single-equation coefficient models → 7-col base (term first, no equation). Placed here
# (not with the other DataFrame(model) methods near VARModel) because RegModel/PanelReg*/
# OrderedLogit*/MultinomialLogitModel aren't defined until this point in the file.
function DataFrames.DataFrame(m::Union{RegModel,LogitModel,ProbitModel,
                                       PanelRegModel,PanelIVModel,PanelLogitModel,PanelProbitModel,
                                       OrderedLogitModel,OrderedProbitModel})
    b = Float64.(m.beta)
    terms = length(b) <= 1 ? ["x1"] : vcat(["_cons"], ["x$i" for i in 1:length(b)-1])
    _mock_coef_df_base(terms, b)
end
# Multinomial logit → tidy coef table keyed by alternative (real MEMs merges `alternative`).
function DataFrames.DataFrame(m::MultinomialLogitModel)
    B = m.beta                      # n_terms × (n_alt - 1)
    nterms, nalt = size(B)
    terms0 = nterms <= 1 ? ["x1"] : vcat(["_cons"], ["x$i" for i in 1:nterms-1])
    alt = String[]; term = String[]; est = Float64[]
    for j in 1:nalt, i in 1:nterms
        push!(alt, "alt$(j + 1)"); push!(term, terms0[i]); push!(est, Float64(B[i, j]))
    end
    df = _mock_coef_df_base(term, est)
    DataFrames.insertcols!(df, 1, :alternative => alt)
    return df
end

# Real (0.8.0, MEMs#550): NO kwargs on either family, and TWO different shapes —
# the ordered models return a plain NamedTuple whose effects/se are K×J MATRICES
# (categories UNTYPED, copy(m.categories)); mlogit returns the exported
# MultinomialMarginalEffects struct (se NULLABLE, base-category column all zeros).
# Neither carries z/p/CI, so they must never route through the shared
# MarginalEffects renderer. The old mock did exactly that, with three invented
# kwargs real does not have (#85 class).
function marginal_effects(m::Union{OrderedLogitModel{T},OrderedProbitModel{T}}) where T
    k = length(m.beta)
    nc = length(m.categories)
    # AMEs across categories sum to zero per variable (∂Σp/∂x = 0) — keep the
    # real property so tests can pin it.
    effects = Matrix{T}(repeat(collect(range(-0.1, 0.1; length=nc))', k))
    (effects = effects, se = fill(T(0.02), k, nc),
     varnames = copy(m.varnames), categories = copy(m.categories))
end

# Real struct (reg/multinomial.jl, exported): se is nothing when the model vcov
# is rank-deficient; categories are String (unlike ordered's untyped vector).
# Live-verified on 0.8.0: every category INCLUDING the base gets a real
# probability-scale AME and each variable's effects sum to ~0 across categories.
struct MultinomialMarginalEffects{T<:AbstractFloat}
    effects::Matrix{T}
    se::Union{Matrix{T},Nothing}
    varnames::Vector{String}
    categories::Vector{String}
end

function marginal_effects(m::MultinomialLogitModel{T}) where T
    k = size(m.beta, 1)
    nc = length(m.categories)
    effects = Matrix{T}(repeat(collect(range(-0.1, 0.1; length=nc))', k))
    MultinomialMarginalEffects{T}(effects, fill(T(0.02), k, nc),
        copy(m.varnames), string.(m.categories))
end

function brant_test(m::Union{OrderedLogitModel{T},OrderedProbitModel{T}}) where T
    k = length(m.beta)
    nc = length(m.categories)
    PanelTestResult{T}("Brant", T(5.0), T(0.25), k * (nc - 2),
        "proportional odds test")
end

function hausman_iia(m::MultinomialLogitModel{T}; omit_category::Int=2) where T
    k = size(m.beta, 1)
    PanelTestResult{T}("Hausman IIA", T(3.5), T(0.48), k, "IIA test")
end

# Real dropna/keeprows semantics replicated exactly. The old no-op versions
# accepted anything (an invented `cols` kwarg; a Vector{SubString} for `vars`,
# which real's `::Union{Vector{String},Nothing}` assertion rejects with a
# TypeError) and returned the data UNCHANGED — hiding both the SubString crash
# and the whole NaN-row-dropping behavior from T1/T2.
function dropna(d::TimeSeriesData; vars::Union{Vector{String},Nothing}=nothing)
    mat = d.data
    if vars === nothing
        good = [all(isfinite, mat[i, :]) for i in 1:size(mat, 1)]
    else
        col_idx = [findfirst(==(v), d.varnames) for v in vars]
        any(isnothing, col_idx) && throw(ArgumentError(
            "Variable(s) not found: $(vars[findall(isnothing, col_idx)])"))
        cidx = Int[c for c in col_idx]
        good = [all(isfinite, mat[i, cidx]) for i in 1:size(mat, 1)]
    end
    idx = findall(good)
    isempty(idx) && throw(ArgumentError("All rows contain NaN or Inf — no data remaining"))
    TimeSeriesData(mat[idx, :], copy(d.varnames), d.frequency, copy(d.tcode),
                   d.time_index[idx], d.desc, copy(d.vardesc))
end

function keeprows(d::TimeSeriesData, idx::Vector{Int})
    isempty(idx) && throw(ArgumentError("No rows selected — empty result"))
    all(i -> 1 <= i <= size(d.data, 1), idx) || throw(BoundsError(d, idx))
    TimeSeriesData(d.data[idx, :], copy(d.varnames), d.frequency, copy(d.tcode),
                   d.time_index[idx], d.desc, copy(d.vardesc))
end

# ─── Policy-counterfactual module (W4/#126, MEMs 0.8.0 CF-01..24) ────────────
# Field names, layouts and validation mirror real (counterfactual/types.jl).
# The engine is real's :ls math (assembly + pinv projection) — it is pure linear
# algebra, so the mock implements it faithfully rather than inventing shapes.

struct PolicyCausalEffects{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    Theta_x::Vector{Matrix{T}}
    Theta_z::Vector{Matrix{T}}
    Theta_x_draws::Union{Nothing,Vector{Array{T,3}}}
    Theta_z_draws::Union{Nothing,Vector{Array{T,3}}}
    H::Int
    shock_labels::Vector{String}
    source::Symbol
end

struct PolicyRule{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    A_x::Vector{Matrix{T}}
    A_z::Vector{Matrix{T}}
    wedge::Vector{T}
    name::String
end

struct PolicyLoss{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    W_x::Vector{Matrix{T}}
    W_z::Union{Nothing,Vector{Matrix{T}}}
    lambda::Vector{T}
    beta::T
    name::String
end

struct BaselinePath{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    x::Vector{Vector{T}}
    z::Vector{Vector{T}}
    x_draws::Union{Nothing,Vector{Matrix{T}}}
    z_draws::Union{Nothing,Vector{Matrix{T}}}
    H::Int
    label::String
end

struct PolicyCounterfactual{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    x_base::Vector{Vector{T}}
    z_base::Vector{Vector{T}}
    x_cf::Vector{Vector{T}}
    z_cf::Vector{Vector{T}}
    x_bands::Union{Nothing,Vector{Matrix{T}}}
    z_bands::Union{Nothing,Vector{Matrix{T}}}
    nu::Vector{T}
    shock_labels::Vector{String}
    error_path::Vector{T}
    rel_residual::T
    rel_residual_bands::Union{Nothing,Vector{T}}
    spanned::Bool
    rule_name::String
    H::Int
    quantile_levels::Vector{T}
    n_draws_used::Int
    n_draws_failed::Int
    loss_base::T
    loss_cf::T
    foc_norm::T
end

is_square(ce::PolicyCausalEffects) = size(ce.Theta_x[1], 2) == ce.H

# Variable/shock resolution: Int index or String name, real's error style.
function _cf_resolve(v, names::Vector{String}, what::String)
    if v isa Integer
        (1 <= v <= length(names)) || throw(ArgumentError(
            "$what index $v out of range 1:$(length(names))"))
        return Int(v)
    end
    i = findfirst(==(String(v)), names)
    i === nothing && throw(ArgumentError(
        "$what '$v' not found in $(names)"))
    return i
end

function _cf_check_horizon(H::Int, stored::Int)
    1 <= H <= stored || throw(ArgumentError(
        "H: expected 1 <= H <= $stored (the stored IRF horizon), got $H"))
end

function _pce_build_mock(values::Array{T,3}, draws4, variables, shocknames,
                         shocks, outcomes, instruments;
                         H::Int, normalize::Symbol, source::Symbol) where T
    normalize in (:none, :instrument_impact) || throw(ArgumentError(
        "normalize: expected :none or :instrument_impact, got :$normalize"))
    _cf_check_horizon(H, size(values, 1))
    isempty(shocks) && throw(ArgumentError("shocks: expected at least one policy shock"))
    shock_idx = Int[_cf_resolve(s, shocknames, "shock") for s in shocks]
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), variables, "variable") for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    ins_idx = Int[_cf_resolve(last(p), variables, "variable") for p in instruments]
    slice(vi) = Matrix{T}(values[1:H, vi, shock_idx])
    dslice(vi) = Array{T,3}(permutedims(draws4[:, 1:H, vi, shock_idx], (2, 3, 1)))
    Theta_x = [slice(vi) for vi in out_idx]
    Theta_z = [slice(vi) for vi in ins_idx]
    Dx = draws4 === nothing ? nothing : Array{T,3}[dslice(vi) for vi in out_idx]
    Dz = draws4 === nothing ? nothing : Array{T,3}[dslice(vi) for vi in ins_idx]
    PolicyCausalEffects{T}(out_syms, ins_syms, Theta_x, Theta_z, Dx, Dz,
                           H, shocknames[shock_idx], source)
end

policy_causal_effects(ir::ImpulseResponse{T}, shocks::AbstractVector,
                      outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                      H::Int=ir.horizon, normalize::Symbol=:none,
                      source::Symbol=:var) where T =
    _pce_build_mock(ir.values, ir._draws, ir.variables, ir.shocks,
                    shocks, outcomes, instruments; H=H, normalize=normalize, source=source)

function policy_causal_effects(bir::BayesianImpulseResponse{T}, shocks::AbstractVector,
                               outcomes::AbstractVector{<:Pair},
                               instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                               H::Int=bir.horizon, normalize::Symbol=:none,
                               source::Symbol=:bvar) where T
    _pce_build_mock(bir.point_estimate, bir._draws, bir.variables, bir.shocks,
                    shocks, outcomes, instruments; H=H, normalize=normalize, source=source)
end

policy_causal_effects(s::SignIdentifiedSet{T}, shocks::AbstractVector,
                      outcomes::AbstractVector{<:Pair},
                      instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                      H::Int=size(s.irf_draws, 2), normalize::Symbol=:none) where T =
    _pce_build_mock(irf_median(s), s.irf_draws, s.variables, s.shocks,
                    shocks, outcomes, instruments; H=H, normalize=normalize,
                    source=:sign_set)

function policy_causal_effects(slp::StructuralLP{T}, shocks::AbstractVector,
                               outcomes::AbstractVector{<:Pair},
                               instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
                               H::Int=slp.irf.horizon, normalize::Symbol=:none,
                               n_draws::Int=500, rng=nothing, seed=nothing) where T
    n_draws >= 1 || throw(ArgumentError("n_draws: expected n_draws >= 1, got $n_draws"))
    Hh, nv, ns = size(slp.irf.values)
    draws4 = reshape(slp.irf.values, 1, Hh, nv, ns) .+
             T(0.02) .* randn(T, n_draws, Hh, nv, ns)
    _pce_build_mock(slp.irf.values, draws4, slp.irf.variables, slp.irf.shocks,
                    shocks, outcomes, instruments; H=H, normalize=normalize, source=:lp)
end

function _bp_build_mock(values::Array{T,3}, draws4, variables, shocknames,
                        nonpolicy_shock, outcomes, instruments;
                        H::Int, negate::Bool) where T
    _cf_check_horizon(H, size(values, 1))
    si = _cf_resolve(nonpolicy_shock, shocknames, "shock")
    sgn = negate ? -one(T) : one(T)
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), variables, "variable") for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    ins_idx = Int[_cf_resolve(last(p), variables, "variable") for p in instruments]
    x = [sgn .* Vector{T}(values[1:H, vi, si]) for vi in out_idx]
    z = [sgn .* Vector{T}(values[1:H, vi, si]) for vi in ins_idx]
    dm(vi) = Matrix{T}(sgn .* permutedims(draws4[:, 1:H, vi, si]))
    xd = draws4 === nothing ? nothing : Matrix{T}[dm(vi) for vi in out_idx]
    zd = draws4 === nothing ? nothing : Matrix{T}[dm(vi) for vi in ins_idx]
    BaselinePath{T}(out_syms, ins_syms, x, z, xd, zd, H,
                    shocknames[si] * (negate ? " (negated)" : ""))
end

baseline_path(ir::ImpulseResponse{T}, nonpolicy_shock,
              outcomes::AbstractVector{<:Pair},
              instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
              H::Int=ir.horizon, negate::Bool=false) where T =
    _bp_build_mock(ir.values, ir._draws, ir.variables, ir.shocks, nonpolicy_shock,
                   outcomes, instruments; H=H, negate=negate)

baseline_path(bir::BayesianImpulseResponse{T}, nonpolicy_shock,
              outcomes::AbstractVector{<:Pair},
              instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[];
              H::Int=bir.horizon, negate::Bool=false) where T =
    _bp_build_mock(bir.point_estimate, bir._draws, bir.variables, bir.shocks,
                   nonpolicy_shock, outcomes, instruments; H=H, negate=negate)

# Rule builders (real validation: single instrument for peg/target/taylor,
# named variables required among outcomes, pi ≠ y for ngdp/taylor).
function _cf_single_instrument(instruments, who)
    length(instruments) == 1 || throw(ArgumentError(
        "$who: expected exactly 1 instrument, got $(length(instruments))"))
end
function _cf_require_var(v::Symbol, outcomes, argname, who)
    i = findfirst(==(v), outcomes)
    i === nothing && throw(ArgumentError(
        "$who: $argname :$v not found in outcomes $(outcomes)"))
    return i
end
_lag_shift_mock(H) = [Float64(i == j + 1) for i in 1:H, j in 1:H]

function rate_peg_rule(H::Int; outcomes=[:infl, :ygap], instruments=[:rate])
    _cf_single_instrument(instruments, "rate_peg_rule")
    PolicyRule{Float64}(collect(outcomes), collect(instruments),
                        [zeros(H, H) for _ in outcomes], [Matrix{Float64}(I, H, H)],
                        zeros(H), "rate peg")
end
function rate_target_rule(H::Int, path::AbstractVector{<:Real};
                          outcomes=[:infl, :ygap], instruments=[:rate])
    _cf_single_instrument(instruments, "rate_target_rule")
    length(path) == H || throw(ArgumentError(
        "rate_target_rule: path: expected length H = $H, got $(length(path))"))
    PolicyRule{Float64}(collect(outcomes), collect(instruments),
                        [zeros(H, H) for _ in outcomes], [Matrix{Float64}(I, H, H)],
                        Vector{Float64}(path), "rate target path")
end
function inflation_target_rule(H::Int; pi_var::Symbol=:infl,
                               outcomes=[:infl, :ygap], instruments=[:rate])
    i = _cf_require_var(pi_var, outcomes, "pi_var", "inflation_target_rule")
    A_x = [zeros(H, H) for _ in outcomes]; A_x[i] = Matrix{Float64}(I, H, H)
    PolicyRule{Float64}(collect(outcomes), collect(instruments), A_x,
                        [zeros(H, H) for _ in instruments], zeros(H), "inflation target")
end
function output_gap_rule(H::Int; y_var::Symbol=:ygap,
                         outcomes=[:infl, :ygap], instruments=[:rate])
    i = _cf_require_var(y_var, outcomes, "y_var", "output_gap_rule")
    A_x = [zeros(H, H) for _ in outcomes]; A_x[i] = Matrix{Float64}(I, H, H)
    PolicyRule{Float64}(collect(outcomes), collect(instruments), A_x,
                        [zeros(H, H) for _ in instruments], zeros(H), "output gap target")
end
function ngdp_rule(H::Int; pi_var::Symbol=:infl, y_var::Symbol=:ygap,
                   outcomes=[:infl, :ygap], instruments=[:rate])
    i_pi = _cf_require_var(pi_var, outcomes, "pi_var", "ngdp_rule")
    i_y = _cf_require_var(y_var, outcomes, "y_var", "ngdp_rule")
    i_pi == i_y && throw(ArgumentError(
        "ngdp_rule: pi_var and y_var must differ, both are :$pi_var"))
    A_x = [zeros(H, H) for _ in outcomes]
    A_x[i_pi] = Matrix{Float64}(I, H, H)
    A_x[i_y] = Matrix{Float64}(I, H, H) - _lag_shift_mock(H)
    PolicyRule{Float64}(collect(outcomes), collect(instruments), A_x,
                        [zeros(H, H) for _ in instruments], zeros(H), "ngdp target")
end
function taylor_rule(H::Int; rho::Real=0.5, phi_pi::Real=1.5, phi_y::Real=1.0,
                     z_lag::Real=0.0, pi_var::Symbol=:infl, y_var::Symbol=:ygap,
                     outcomes=[:infl, :ygap], instruments=[:rate])
    _cf_single_instrument(instruments, "taylor_rule")
    i_pi = _cf_require_var(pi_var, outcomes, "pi_var", "taylor_rule")
    i_y = _cf_require_var(y_var, outcomes, "y_var", "taylor_rule")
    i_pi == i_y && throw(ArgumentError(
        "taylor_rule: pi_var and y_var must differ, both are :$pi_var"))
    A_x = [zeros(H, H) for _ in outcomes]
    A_x[i_pi] = -(1 - rho) * phi_pi * Matrix{Float64}(I, H, H)
    A_x[i_y] = -(1 - rho) * phi_y * Matrix{Float64}(I, H, H)
    A_z = Matrix{Float64}(I, H, H) - rho * _lag_shift_mock(H)
    wedge = zeros(H); wedge[1] = rho * z_lag
    PolicyRule{Float64}(collect(outcomes), collect(instruments), A_x, [A_z], wedge,
                        "taylor(ρ=$(rho), φπ=$(phi_pi), φy=$(phi_y))")
end

function policy_loss(outcomes::AbstractVector{Symbol}, H::Int;
                     lambda::AbstractVector{<:Real},   # REQUIRED, like real
                     beta::Real=1.0, instruments::AbstractVector{Symbol}=Symbol[],
                     W_z=nothing, name::AbstractString="discounted diagonal")
    length(lambda) == length(outcomes) || throw(ArgumentError(
        "policy_loss: lambda: expected $(length(outcomes)) weights (one per outcome), got $(length(lambda))"))
    (0 < beta <= 1) || throw(ArgumentError("beta: expected 0 < beta <= 1, got $beta"))
    W_x = [Matrix{Float64}(Float64(lam) .* Diagonal([Float64(beta)^(h - 1) for h in 1:H]))
           for lam in lambda]
    PolicyLoss{Float64}(collect(outcomes), collect(instruments), W_x,
                        W_z === nothing ? nothing : collect(Matrix{Float64}, W_z),
                        Vector{Float64}(lambda), Float64(beta), String(name))
end

function ait_loss(H::Int; beta::Real=1/1.01, lambda_avg::Real=0.6, lambda_t::Real=0.4,
                  lambda_y::Real=1.0, delta::Real=0.1, K::Int=19)
    K >= 1 || throw(ArgumentError("K: expected K >= 1, got $K"))
    policy_loss([:infl, :ygap], H; lambda=[lambda_avg + lambda_t, lambda_y],
                beta=beta, name="average inflation targeting")
end

function smoothing_penalty(H::Int; lambda::Real=1.0, beta::Real=1.0, z_lag::Real=0.0)
    D = Matrix{Float64}(I, H, H) - _lag_shift_mock(H)
    W = Float64(lambda) .* (D' * Diagonal([Float64(beta)^(h - 1) for h in 1:H]) * D)
    wedge = zeros(H); wedge[1] = Float64(lambda) * Float64(z_lag)
    (W_z=W, wedge_term=wedge)   # NamedTuple, NOT a PolicyLoss — real's contract
end

_rule_horizon_mock(rule::PolicyRule) =
    isempty(rule.A_x) ? size(rule.A_z[1], 1) : size(rule.A_x[1], 1)

# Real's :ls engine — assembly + pinv projection + honesty numbers.
function policy_counterfactual(base::BaselinePath{T}, ce::PolicyCausalEffects{T},
                               rule::PolicyRule;
                               method::Symbol=:auto, draws::Symbol=:auto,
                               baseline_draws::Symbol=:fixed,
                               quantiles=(0.16, 0.5, 0.84),
                               spanned_tol::Real=0.05) where T
    H = ce.H
    base.H == H || throw(ArgumentError(
        "baseline H = $(base.H) does not match the container H = $H"))
    _rule_horizon_mock(rule) == H || throw(ArgumentError(
        "rule H = $(_rule_horizon_mock(rule)) does not match the container H = $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    baseline_draws in (:fixed, :match) || throw(ArgumentError(
        "baseline_draws: expected :fixed or :match, got :$baseline_draws"))
    method in (:auto, :ls, :exact) || throw(ArgumentError(
        "method: expected :auto, :ls or :exact, got :$method"))
    for sym in rule.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "rule outcome :$sym not found in the container outcomes $(ce.outcomes)"))
    end
    for sym in rule.instruments
        sym in ce.instruments || throw(ArgumentError(
            "rule instrument :$sym not found in the container instruments $(ce.instruments)"))
    end

    xb = [Vector{T}(base.x[findfirst(==(s), base.outcomes)]) for s in ce.outcomes]
    zb = [Vector{T}(base.z[findfirst(==(s), base.instruments)]) for s in ce.instruments]
    n_s = size(ce.Theta_x[1], 2)
    M = zeros(T, H, n_s)
    b = -Vector{T}(rule.wedge)
    for (i, sym) in enumerate(rule.outcomes)
        ci = findfirst(==(sym), ce.outcomes)
        M .+= rule.A_x[i] * ce.Theta_x[ci]
        b .+= rule.A_x[i] * xb[ci]
    end
    for (k, sym) in enumerate(rule.instruments)
        ck = findfirst(==(sym), ce.instruments)
        M .+= rule.A_z[k] * ce.Theta_z[ck]
        b .+= rule.A_z[k] * zb[ck]
    end
    nu = -pinv(M) * b
    err = M * nu + b
    rel = norm(err) / max(norm(b), eps(T))
    x_cf = [xb[i] + ce.Theta_x[i] * nu for i in eachindex(ce.outcomes)]
    z_cf = [zb[k] + ce.Theta_z[k] * nu for k in eachindex(ce.instruments)]

    qlev = collect(T, quantiles)
    nd = ce.Theta_x_draws === nothing ? 0 : size(ce.Theta_x_draws[1], 3)
    use = draws == :on || (draws == :auto && nd > 0)
    (draws == :on && nd == 0) && throw(ArgumentError(
        "draws = :on requires a draws-bearing container"))
    x_bands = nothing; z_bands = nothing; rr_bands = nothing
    n_used = 0
    if use
        n_used = nd
        nq = length(qlev)
        band(v) = Matrix{T}(hcat([v .+ (q - T(0.5)) .* T(0.1) for q in qlev]...))
        x_bands = [band(x_cf[i]) for i in eachindex(ce.outcomes)]
        z_bands = [band(z_cf[k]) for k in eachindex(ce.instruments)]
        rr_bands = [rel + (q - T(0.5)) * T(0.01) for q in qlev]
    end
    PolicyCounterfactual{T}(copy(ce.outcomes), copy(ce.instruments), xb, zb,
                            x_cf, z_cf, x_bands, z_bands, nu,
                            copy(ce.shock_labels), err, rel, rr_bands,
                            rel < T(spanned_tol), rule.name, H, qlev,
                            n_used, 0, T(NaN), T(NaN), T(NaN))
end

# ── W5/#127: optimal policy + second-moment counterfactuals ──────────────────

struct WoldRepresentation{T<:AbstractFloat}
    Theta::Array{T,3}
    Sigma_u::Matrix{T}
    varnames::Vector{String}
    draws::Union{Nothing,Array{T,4}}
end

struct CounterfactualMoments{T<:AbstractFloat}
    varnames::Vector{Symbol}
    Sigma_base::Matrix{T}
    Sigma_cf::Matrix{T}
    sd_base::Vector{T}
    sd_cf::Vector{T}
    corr_base::Matrix{T}
    corr_cf::Matrix{T}
    sd_cf_bands::Union{Nothing,Matrix{T}}
    Theta_cf::Array{T,3}
    policy_name::String
    H::Int
    tail_share::T
    freq_band::Union{Nothing,Tuple{T,T}}
end

function wold_representation(m::VARModel{T}; H::Int,
                             orthogonalize::Symbol=:cholesky) where T
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    n = size(m.Y, 2)
    Theta = zeros(T, H, n, n)
    for h in 1:H, i in 1:n, j in 1:n
        Theta[h, i, j] = (i == j ? T(0.5)^(h - 1) : T(0.1) * T(0.5)^h)
    end
    WoldRepresentation{T}(Theta, Matrix{T}(I(n)) * T(0.5), copy(m.varnames), nothing)
end

function wold_representation(post::BVARPosterior{T}; H::Int,
                             orthogonalize::Symbol=:cholesky,
                             max_draws::Int=post.n_draws) where T
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    n = post.n
    Theta = zeros(T, H, n, n)
    for h in 1:H, i in 1:n, j in 1:n
        Theta[h, i, j] = (i == j ? T(0.5)^(h - 1) : T(0.1) * T(0.5)^h)
    end
    nd = min(max_draws, post.n_draws)
    draws = reshape(Theta, H, n, n, 1) .+ T(0.02) .* randn(T, H, n, n, nd)
    WoldRepresentation{T}(Theta, Matrix{T}(I(n)) * T(0.5),
                          ["y$i" for i in 1:n], draws)
end

_loss_horizon_mock(loss::PolicyLoss) = size(loss.W_x[1], 1)

function optimal_rule(ce::PolicyCausalEffects{T}, loss::PolicyLoss;
                      z_wedge=nothing) where T
    is_square(ce) || throw(ArgumentError(
        "optimal_rule requires a square (model-implied) container: the targeting rule needs the full news menu"))
    H = ce.H
    _loss_horizon_mock(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon_mock(loss)) does not match the container H = $H"))
    A_x = Matrix{Float64}[]
    for (j, sym) in enumerate(loss.outcomes)
        i = findfirst(==(sym), ce.outcomes)
        i === nothing && throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
        push!(A_x, Matrix{Float64}(ce.Theta_x[i]' * loss.W_x[j]))
    end
    A_z = Matrix{Float64}[]
    wedge = zeros(H)
    for (j, sym) in enumerate(loss.instruments)
        k = findfirst(==(sym), ce.instruments)
        push!(A_z, Matrix{Float64}(ce.Theta_z[k]' *
            (loss.W_z === nothing ? zeros(H, H) : loss.W_z[j])))
        z_wedge !== nothing && (wedge .+= ce.Theta_z[k]' * Vector{Float64}(z_wedge[j]))
    end
    # Rule symbols cover the FULL container (every path gets a counterfactual).
    all_A_x = [any(==(s), loss.outcomes) ?
                   A_x[findfirst(==(s), loss.outcomes)] : zeros(H, H)
               for s in ce.outcomes]
    all_A_z = [any(==(s), loss.instruments) ?
                   A_z[findfirst(==(s), loss.instruments)] : zeros(H, H)
               for s in ce.instruments]
    PolicyRule{Float64}(copy(ce.outcomes), copy(ce.instruments), all_A_x, all_A_z,
                        wedge, "optimal($(loss.name))")
end

function optimal_policy(base::BaselinePath{T}, ce::PolicyCausalEffects{T},
                        loss::PolicyLoss;
                        z_wedge=nothing, draws::Symbol=:auto,
                        baseline_draws::Symbol=:fixed,
                        quantiles=(0.16, 0.5, 0.84)) where T
    H = ce.H
    base.H == H || throw(ArgumentError(
        "baseline H = $(base.H) does not match the container H = $H"))
    _loss_horizon_mock(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon_mock(loss)) does not match the container H = $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    for sym in loss.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
    end
    xb = [Vector{T}(base.x[findfirst(==(s), base.outcomes)]) for s in ce.outcomes]
    zb = [Vector{T}(base.z[findfirst(==(s), base.instruments)]) for s in ce.instruments]
    n_s = size(ce.Theta_x[1], 2)
    # Normal equations: (Σ Θ'WΘ) ν = −Σ Θ'W x_base (+ instrument/wedge terms)
    A = zeros(T, n_s, n_s)
    rhs = zeros(T, n_s)
    for (j, sym) in enumerate(loss.outcomes)
        i = findfirst(==(sym), ce.outcomes)
        A .+= ce.Theta_x[i]' * loss.W_x[j] * ce.Theta_x[i]
        rhs .-= ce.Theta_x[i]' * loss.W_x[j] * xb[i]
    end
    for (j, sym) in enumerate(loss.instruments)
        k = findfirst(==(sym), ce.instruments)
        Wz = loss.W_z === nothing ? nothing : loss.W_z[j]
        Wz === nothing || (A .+= ce.Theta_z[k]' * Wz * ce.Theta_z[k];
                           rhs .-= ce.Theta_z[k]' * Wz * zb[k])
        z_wedge === nothing ||
            (rhs .+= ce.Theta_z[k]' * Vector{T}(z_wedge[j]))
    end
    nu = pinv(A) * rhs
    x_cf = [xb[i] + ce.Theta_x[i] * nu for i in eachindex(ce.outcomes)]
    z_cf = [zb[k] + ce.Theta_z[k] * nu for k in eachindex(ce.instruments)]
    lossof(xs, zs) = begin
        L = zero(T)
        for (j, sym) in enumerate(loss.outcomes)
            i = findfirst(==(sym), ce.outcomes)
            L += xs[i]' * loss.W_x[j] * xs[i]
        end
        L
    end
    loss_base = lossof(xb, zb)
    loss_cf = lossof(x_cf, z_cf)
    foc = A * nu - rhs
    err = vcat([loss.W_x[j] * x_cf[findfirst(==(sym), ce.outcomes)]
                for (j, sym) in enumerate(loss.outcomes)]...)
    rel = norm(err) / max(norm(vcat(xb...)), eps(T))
    qlev = collect(T, quantiles)
    PolicyCounterfactual{T}(copy(ce.outcomes), copy(ce.instruments), xb, zb,
                            x_cf, z_cf, nothing, nothing, nu,
                            copy(ce.shock_labels), err, rel, nothing,
                            rel < T(0.05),                      # hardcoded like real
                            "optimal($(loss.name))", H, qlev, 0, 0,
                            loss_base, loss_cf, norm(foc))
end

function _cf_resolve_band_mock(frequencies)
    frequencies === :none && return nothing
    frequencies === :business_cycle && return (2π / 32, 2π / 6)
    frequencies isa Tuple || throw(ArgumentError(
        "frequencies: expected :none, :business_cycle or an (ω_lo, ω_hi) tuple, got $frequencies"))
    lo, hi = frequencies
    (0 <= lo < hi <= π + 1e-12) || throw(ArgumentError(
        "frequencies: expected 0 <= lo < hi <= pi, got ($lo, $hi)"))
    (Float64(lo), Float64(hi))
end

function counterfactual_moments(wold::WoldRepresentation{T}, ce::PolicyCausalEffects{T},
                                policy::Union{PolicyRule,PolicyLoss};
                                outcomes::AbstractVector{<:Pair},
                                instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[],
                                draws::Symbol=:auto, draw_source::Symbol=:ce,
                                quantiles=(0.16, 0.5, 0.84),
                                frequencies=:none,
                                warn_invertibility::Bool=true) where T
    H = ce.H
    Hw = size(wold.Theta, 1)
    Hw >= H || throw(ArgumentError(
        "Wold horizon $Hw is shorter than the container H = $H; re-run wold_representation with H >= $H"))
    draws in (:auto, :on, :off) || throw(ArgumentError(
        "draws: expected :auto, :on or :off, got :$draws"))
    draw_source in (:ce, :wold, :both) || throw(ArgumentError(
        "draw_source: expected :ce, :wold or :both, got :$draw_source"))
    band = _cf_resolve_band_mock(frequencies)
    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    sort(out_syms) == sort(ce.outcomes) || throw(ArgumentError(
        "outcomes must map exactly the container outcomes $(ce.outcomes), got $(out_syms)"))
    sort(ins_syms) == sort(ce.instruments) || throw(ArgumentError(
        "instruments must map exactly the container instruments $(ce.instruments), got $(ins_syms)"))
    syms = vcat(out_syms, ins_syms)
    rows = Int[_cf_resolve(last(p), wold.varnames, "Wold variable")
               for p in vcat(collect(outcomes), collect(instruments))]
    nv = length(syms)
    n = size(wold.Theta, 2)
    Sigma_base = zeros(T, nv, nv)
    for h in 1:H
        Th = wold.Theta[h, rows, :]
        Sigma_base .+= Th * Th'
    end
    # A stabilizing counterfactual shrinks variance; a band restricts it further.
    scale = band === nothing ? T(0.8) : T(0.5)
    Sigma_cf = scale .* Sigma_base
    sd_base = sqrt.(diag(Sigma_base))
    sd_cf = sqrt.(diag(Sigma_cf))
    corr(M, s) = [M[i, j] / max(s[i] * s[j], eps(T)) for i in 1:nv, j in 1:nv]
    nd = ce.Theta_x_draws === nothing ? 0 : size(ce.Theta_x_draws[1], 3)
    use = draws == :on || (draws == :auto && nd > 0)
    bands = use ? Matrix{T}(hcat([sd_cf .+ (q - T(0.5)) .* T(0.05)
                                  for q in collect(T, quantiles)]...)) : nothing
    pname = policy isa PolicyRule ? policy.name : policy.name
    CounterfactualMoments{T}(syms, Sigma_base, Sigma_cf, sd_base, sd_cf,
                             corr(Sigma_base, sd_base), corr(Sigma_cf, sd_cf),
                             bands, T(0.05) .* ones(T, H, nv, n), pname, H,
                             T(0.005),
                             band === nothing ? nothing : (T(band[1]), T(band[2])))
end

# ── W6/#128: the OPP family (Barnichon–Mesters) ──────────────────────────────

struct PolicyForecast{T<:AbstractFloat}
    outcomes::Vector{Symbol}
    values::Vector{Vector{T}}
    draws::Union{Nothing,Vector{Matrix{T}}}
    H::Int
    origin::String
end

abstract type OPPConstraint end
struct PathFloorConstraint{T<:AbstractFloat} <: OPPConstraint
    instrument::Symbol
    floor::T
    horizons::UnitRange{Int}
end
zlb_constraint(; floor::Real=0.0, instrument::Symbol=:rate,
               horizons::UnitRange{Int}=1:typemax(Int)) =
    PathFloorConstraint{Float64}(instrument, Float64(floor), horizons)

struct OPPResult{T<:AbstractFloat}
    delta::Vector{T}
    delta_plugin::Vector{T}
    shock_labels::Vector{String}
    gradient::Vector{T}
    loss_base::T
    loss_opp::T
    Y_base::Vector{Vector{T}}
    Y_opp::Vector{Vector{T}}
    P_base::Union{Nothing,Vector{Vector{T}}}
    P_opp::Union{Nothing,Vector{Vector{T}}}
    outcomes::Vector{Symbol}
    instruments::Vector{Symbol}
    H::Int
    origin::String
    delta_draws::Union{Nothing,Matrix{T}}
    bands::Union{Nothing,Dict{T,Matrix{T}}}
    reject::Union{Nothing,Dict{T,Vector{Bool}}}
    n_failed::Int
end

struct OPPSequence{T<:AbstractFloat}
    dates::Vector{String}
    delta::Matrix{T}        # n_s × n_dates — shocks are ROWS (real layout)
    delta_tc::Matrix{T}
    news_part::Matrix{T}
    pref_part::Matrix{T}
    aging_part::Matrix{T}
    bands::Union{Nothing,Dict{T,Array{T,3}}}
    reject::Union{Nothing,Dict{T,Matrix{Bool}}}
    shock_labels::Vector{String}
    loss_name::String
end

function policy_forecast(fc::Union{VARForecast{T},BVARForecast{T}},
                         outcomes::AbstractVector{<:Pair};
                         targets::AbstractVector{<:Pair}=Pair{Symbol,Float64}[],
                         H::Int=fc.horizon,
                         origin::AbstractString="") where T
    1 <= H <= fc.horizon || throw(ArgumentError(
        "H: expected 1 <= H <= $(fc.horizon) (the forecast horizon), got $H"))
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    out_syms = Symbol[first(p) for p in outcomes]
    out_idx = Int[_cf_resolve(last(p), fc.varnames, "variable") for p in outcomes]
    tmap = Dict{Symbol,Float64}(Symbol(first(p)) => Float64(last(p)) for p in targets)
    for k in keys(tmap)
        k in out_syms || throw(ArgumentError(
            "targets: :$k is not among the requested outcomes $(out_syms)"))
    end
    vals = [Vector{T}(fc.forecast[1:H, vi]) .- T(get(tmap, out_syms[i], 0.0))
            for (i, vi) in enumerate(out_idx)]
    drws = fc._draws === nothing ? nothing :
           Matrix{T}[Matrix{T}(permutedims(fc._draws[:, 1:H, vi])) .-
                     T(get(tmap, out_syms[i], 0.0))
                     for (i, vi) in enumerate(out_idx)]
    PolicyForecast{T}(out_syms, vals, drws, H, String(origin))
end

function policy_forecast(outcomes::AbstractVector{Symbol},
                         values::AbstractVector{<:AbstractVector{<:Real}};
                         sd=nothing, rho::Real=0.9, n_draws::Int=1000, rng=nothing,
                         H::Int=isempty(values) ? 0 : length(first(values)),
                         cross_corr=:independent, min_sd::Real=0.0,
                         origin::AbstractString="", seed=nothing)
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    length(values) == length(outcomes) || throw(ArgumentError(
        "values: expected $(length(outcomes)) paths (one per outcome), got $(length(values))"))
    n_draws >= 1 || throw(ArgumentError("n_draws: expected n_draws >= 1, got $n_draws"))
    (sd === nothing && cross_corr === :independent) && throw(ArgumentError(
        "sd: per-horizon standard deviations are required unless a full cross_corr covariance is given"))
    vals = [Vector{Float64}(v) for v in values]
    for v in vals
        length(v) == H || throw(ArgumentError(
            "values: expected length H = $H paths, got $(length(v))"))
    end
    drws = [hcat([v .+ 0.1 .* randn(H) for _ in 1:n_draws]...) for v in vals]
    PolicyForecast{Float64}(collect(outcomes), vals, drws, H, String(origin))
end

interp_to_quarterly(annual::AbstractVector{<:Real}, H::Int) =
    [Float64(annual[min(cld(h, 4), length(annual))]) for h in 1:H]

function _opp_core(pf::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                   loss::PolicyLoss; instrument_path=nothing, z_wedge=nothing) where T
    H = ce.H
    pf.H == H || throw(ArgumentError(
        "forecast H = $(pf.H) does not match the IRF container H = $H; re-build the two on a common horizon"))
    _loss_horizon_mock(loss) == H || throw(ArgumentError(
        "loss H = $(_loss_horizon_mock(loss)) does not match the container H = $H"))
    for sym in loss.outcomes
        sym in ce.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the container outcomes $(ce.outcomes)"))
        sym in pf.outcomes || throw(ArgumentError(
            "loss outcome :$sym not found in the forecast outcomes $(pf.outcomes)"))
    end
    is_square(ce) && @warn "full-menu OPP on a square container is ill-posed (BM design around thin subsets)"
    n_s = size(ce.Theta_x[1], 2)
    A = zeros(T, n_s, n_s)
    g = zeros(T, n_s)
    Yb = [Vector{T}(pf.values[findfirst(==(s), pf.outcomes)]) for s in ce.outcomes]
    for (j, sym) in enumerate(loss.outcomes)
        i = findfirst(==(sym), ce.outcomes)
        A .+= ce.Theta_x[i]' * loss.W_x[j] * ce.Theta_x[i]
        g .+= ce.Theta_x[i]' * loss.W_x[j] * Yb[i]
    end
    delta = -pinv(A) * g
    Yo = [Yb[i] + ce.Theta_x[i] * delta for i in eachindex(ce.outcomes)]
    lossof(Y) = sum(Y[findfirst(==(sym), ce.outcomes)]' * loss.W_x[j] *
                    Y[findfirst(==(sym), ce.outcomes)]
                    for (j, sym) in enumerate(loss.outcomes))
    Pb = nothing; Po = nothing
    if instrument_path !== nothing
        Pb = [Vector{T}(last(p)) for p in instrument_path]
        Po = [Pb[k] + ce.Theta_z[k] * delta for k in eachindex(Pb)]
    end
    (delta=delta, gradient=g, Yb=Yb, Yo=Yo, Pb=Pb, Po=Po,
     loss_base=lossof(Yb), loss_opp=lossof(Yo))
end

function opp(pf::PolicyForecast{T}, ce::PolicyCausalEffects{T}, loss::PolicyLoss;
             instrument_path=nothing, z_wedge=nothing) where T
    c = _opp_core(pf, ce, loss; instrument_path=instrument_path, z_wedge=z_wedge)
    OPPResult{T}(c.delta, copy(c.delta), copy(ce.shock_labels), c.gradient,
                 c.loss_base, c.loss_opp, c.Yb, c.Yo, c.Pb, c.Po,
                 copy(ce.outcomes), copy(ce.instruments), ce.H, pf.origin,
                 nothing, nothing, nothing, 0)
end

function estimate_opp(pf::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                      loss::PolicyLoss; instrument_path=nothing, z_wedge=nothing,
                      independent::Bool=true, levels=(0.60, 0.75, 0.90),
                      n_sim::Int=2000, rng=nothing, seed=nothing) where T
    all(l -> 0 < l < 1, levels) || throw(ArgumentError(
        "levels: expected levels in (0, 1), got $levels"))
    (pf.draws === nothing && ce.Theta_x_draws === nothing) && throw(ArgumentError(
        "estimate_opp requires draws on at least one source (forecast or menu)"))
    c = _opp_core(pf, ce, loss; instrument_path=instrument_path, z_wedge=z_wedge)
    n_s = length(c.delta)
    dd = c.delta .+ T(0.1) .* randn(T, n_s, max(n_sim, 10))
    med = [T(quantile(dd[k, :], 0.5)) for k in 1:n_s]
    bands = Dict{T,Matrix{T}}()
    rej = Dict{T,Vector{Bool}}()
    for l in levels
        lo = [T(quantile(dd[k, :], (1 - l) / 2)) for k in 1:n_s]
        hi = [T(quantile(dd[k, :], 1 - (1 - l) / 2)) for k in 1:n_s]
        bands[T(l)] = hcat(lo, hi)
        rej[T(l)] = [!(lo[k] <= 0 <= hi[k]) for k in 1:n_s]
    end
    OPPResult{T}(med, c.delta, copy(ce.shock_labels), c.gradient,
                 c.loss_base, c.loss_opp, c.Yb, c.Yo, c.Pb, c.Po,
                 copy(ce.outcomes), copy(ce.instruments), ce.H, pf.origin,
                 Matrix{T}(dd'), bands, rej, 0)
end

function constrained_opp(pf::PolicyForecast{T}, ce::PolicyCausalEffects{T},
                         loss::PolicyLoss, constraints::AbstractVector{<:OPPConstraint};
                         instrument_path=nothing, z_wedge=nothing,
                         method::Symbol=:auto, delta0=nothing, multistart::Int=1,
                         rng=nothing, n_sim::Int=0, levels=(0.6, 0.75, 0.9),
                         independent::Bool=true, seed=nothing) where T
    method in (:auto, :slsqp, :projection) || throw(ArgumentError(
        "method: expected :auto, :slsqp or :projection, got :$method"))
    instrument_path === nothing && throw(ArgumentError(
        "constrained_opp requires instrument_path (the announced path the constraints act on)"))
    r = opp(pf, ce, loss; instrument_path=instrument_path, z_wedge=z_wedge)
    # Real returns a NamedTuple, NOT a result type (the #118 blind-spot class) —
    # field list copied verbatim.
    (; result=r, method_used=(method === :projection ? :projection : :slsqp),
     binding=fill(false, length(constraints)), kkt_residual=0.0,
     warm_start_feasible=true)
end

function opp_sequence(forecasts::AbstractVector, ce::PolicyCausalEffects{T},
                      loss::PolicyLoss; dates=nothing, ce_by_date=nothing,
                      instrument_paths=nothing, constraints=OPPConstraint[],
                      z_wedge=nothing, n_sim::Int=0, levels=(0.6, 0.75, 0.9),
                      independent::Bool=true, rng=nothing, seed=nothing) where T
    nd = length(forecasts)
    nd >= 2 || throw(ArgumentError("opp_sequence: expected >= 2 dates, got $nd"))
    ds = dates === nothing ? [string("t", i) for i in 1:nd] : collect(String, dates)
    n_s = size(ce.Theta_x[1], 2)
    delta = zeros(T, n_s, nd)
    for (d, f) in enumerate(forecasts)
        f === missing && continue
        delta[:, d] = opp(f, ce, loss).delta
    end
    news = zeros(T, n_s, nd); pref = zeros(T, n_s, nd); aging = zeros(T, n_s, nd)
    for d in 2:nd
        rev = delta[:, d] - delta[:, d-1]
        news[:, d] = T(0.9) .* rev      # exact three-part: parts SUM to the revision
        aging[:, d] = T(0.1) .* rev
    end
    OPPSequence{T}(ds, delta, copy(delta), news, pref, aging,
                   nothing, nothing, copy(ce.shock_labels), loss.name)
end

# ── W7/#129: structural routes ───────────────────────────────────────────────

struct CounterfactualHistory{T<:AbstractFloat}
    dates::Vector{String}
    varnames::Vector{Symbol}
    realized::Matrix{T}
    cf::Matrix{T}
    cf_bands::Union{Nothing,Array{T,3}}
    nu::Matrix{T}
    rel_residual::Vector{T}
    policy_name::String
    H::Int
    quantile_levels::Vector{T}
    n_draws_used::Int
    n_draws_failed::Int
end

struct SpanningDiagnostic{T<:AbstractFloat}
    gap::Vector{T}
    gap_rel::Vector{T}
    loading_inside::T
    rel_residual_emp::T
    spanned::Bool
    outcomes::Vector{Symbol}
    x_cf_emp::Vector{Vector{T}}
    x_cf_full::Vector{Vector{T}}
    bands_gap::Union{Nothing,Vector{Matrix{T}}}
end

struct ForecastSufficiency{T<:AbstractFloat}
    observables::Vector{Symbol}
    fev_ratio::Matrix{T}
    one_step_ratio::Vector{T}
    invertible::Bool
    H::Int
end

function policy_news_matrix(spec::ModelSpec, policy_shock::Symbol,
                            outcomes::AbstractVector{<:Pair{Symbol,Symbol}},
                            instruments::AbstractVector{<:Pair{Symbol,Symbol}}=Pair{Symbol,Symbol}[];
                            H::Int=100, solver::Symbol=:gensys, chunk::Int=0)
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    solver in (:gensys, :klein, :blanchard_kahn) || throw(ArgumentError(
        "policy_news_matrix supports the linear solvers :gensys/:klein/:blanchard_kahn only (nonlinear news menus are out of scope), got :$solver"))
    isempty(outcomes) && throw(ArgumentError("outcomes: expected at least one outcome"))
    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    # SQUARE by construction: n_s = H news columns, decaying loadings.
    mk() = [Float64(0.5)^(abs(h - k)) * 0.1 for h in 1:H, k in 1:H]
    PolicyCausalEffects{Float64}(out_syms, ins_syms,
        [mk() for _ in out_syms], [mk() for _ in ins_syms], nothing, nothing,
        H, ["news $k" for k in 1:H], :dsge)
end

function behavioral(ce::PolicyCausalEffects{T}; m::Real=1.0, theta::Real=0.0) where T
    (0 <= m <= 1 && 0 <= theta <= 1) || throw(ArgumentError(
        "behavioral: expected m and theta in [0, 1], got m=$m, theta=$theta"))
    is_square(ce) || throw(ArgumentError(
        "behavioral operators need a square (model news) container; empirical thin menus are already behavior-inclusive"))
    scale = T(m) * (1 - T(theta) / 2)
    PolicyCausalEffects{T}(copy(ce.outcomes), copy(ce.instruments),
        [scale .* M for M in ce.Theta_x], [scale .* M for M in ce.Theta_z],
        nothing, nothing, ce.H, copy(ce.shock_labels), ce.source)
end

function counterfactual_history(mdl::Union{VARModel{T},BVARPosterior{T}},
                                data::AbstractMatrix{<:Real},
                                t_range::AbstractUnitRange{Int},
                                ce::PolicyCausalEffects,
                                policy::Union{PolicyRule,PolicyLoss};
                                outcomes::AbstractVector{<:Pair},
                                instruments::AbstractVector{<:Pair}=Pair{Symbol,Int}[],
                                H::Int=ce.H, dates=nothing, wedge_builder=nothing,
                                draws::Symbol=:auto,
                                quantiles=(0.16, 0.5, 0.84)) where T
    H == ce.H || throw(ArgumentError(
        "H = $H must equal the container H = $(ce.H)"))
    length(t_range) <= H - 1 || throw(ArgumentError(
        "window length $(length(t_range)) must be <= H - 1 = $(H - 1)"))
    syms = vcat(Symbol[first(p) for p in outcomes], Symbol[first(p) for p in instruments])
    nd = length(t_range); nv = length(syms)
    ds = dates === nothing ? [string("t", t) for t in t_range] : collect(String, dates)
    realized = Matrix{Float64}(data[collect(t_range), 1:nv])
    pname = policy isa PolicyRule ? policy.name : policy.name
    CounterfactualHistory{Float64}(ds, syms, realized, 0.9 .* realized, nothing,
        zeros(nd, size(ce.Theta_x[1], 2)), fill(0.02, nd), pname, H,
        collect(Float64, quantiles), 0, 0)
end

function spanning_diagnostic(base::BaselinePath, ce_emp::PolicyCausalEffects,
                             ce_full::PolicyCausalEffects,
                             policy::Union{PolicyRule,PolicyLoss};
                             draws::Symbol=:auto, tol::Real=0.1, n_sim::Int=200,
                             quantiles=(0.16, 0.5, 0.84), rng=nothing, seed=nothing)
    is_square(ce_full) || throw(ArgumentError(
        "ce_full must be a square (model-implied) container"))
    ce_emp.H == ce_full.H || throw(ArgumentError(
        "container horizons differ: emp H = $(ce_emp.H) vs full H = $(ce_full.H)"))
    ce_emp.outcomes == ce_full.outcomes || throw(ArgumentError(
        "container outcomes differ: $(ce_emp.outcomes) vs $(ce_full.outcomes)"))
    H = ce_emp.H
    xe = [0.1 .* ones(H) for _ in ce_emp.outcomes]
    xf = [0.12 .* ones(H) for _ in ce_emp.outcomes]
    gap = fill(0.02, H)
    SpanningDiagnostic{Float64}(gap, fill(0.05, H), 0.95, 0.04,
        maximum(fill(0.05, H)) < Float64(tol), copy(ce_emp.outcomes), xe, xf, nothing)
end

function forecast_sufficiency(sol::Union{DSGESolution,PerturbationSolution},
                              observables::AbstractVector{Symbol}; H::Int=40)
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    isempty(observables) && throw(ArgumentError(
        "observables: expected at least one observable"))
    n = length(observables)
    ForecastSufficiency{Float64}(collect(observables),
        ones(H, n) .+ 0.01, fill(1.01, n), true, H)
end

export PolicyCausalEffects, PolicyRule, PolicyLoss, BaselinePath, PolicyCounterfactual,
       is_square, policy_causal_effects, baseline_path, policy_counterfactual,
       rate_peg_rule, rate_target_rule, inflation_target_rule, output_gap_rule,
       ngdp_rule, taylor_rule, policy_loss, ait_loss, smoothing_penalty,
       WoldRepresentation, CounterfactualMoments, wold_representation,
       optimal_policy, optimal_rule, counterfactual_moments,
       PolicyForecast, OPPResult, OPPSequence, OPPConstraint, PathFloorConstraint,
       policy_forecast, interp_to_quarterly, opp, estimate_opp, constrained_opp,
       zlb_constraint, opp_sequence,
       CounterfactualHistory, SpanningDiagnostic, ForecastSufficiency,
       policy_news_matrix, sequence_jacobian, behavioral, counterfactual_history,
       spanning_diagnostic, forecast_sufficiency

# NOTE: this mock deliberately defines NO `Base.getproperty` compat aliases.
#
# Aliases here (`result.cips` → `cips_statistic`, `bfevd.mean` → `point_estimate`,
# `model.nobs`, `arima.coefficients`, …) invent a property surface real MEMs does
# not have. Because they are methods rather than fields, `check_mock_surface`'s
# field-subset test could not see them, so eleven commands shipped reading fields
# that do not exist and crashed with `FieldError` (exit 1) on real MEMs while
# T1/T2 stayed green — see #84. `check_mock_surface.jl` now fails the gate on any
# mock `getproperty` alias that is not a real field. Keep it that way: make the
# handler use the real accessor instead of teaching the mock a new name.

# --- C039 Phase-4 surface mocks (MEMs 0.6.7 fields ⊆ real) ---
# HAModelSpec/HADSGESpec deleted at 0.9.0 — HA is ModelSpec + HouseholdSystem.

struct HASteadyState{T<:AbstractFloat}
    policies::Any
    distribution::Any
    value_fn::Any
    prices::Dict{Symbol,T}
    aggregates::Dict{Symbol,T}
    grid::Any
    income::Any
    converged::Bool
    iterations::Int
    euler_error::T
    excess_demand::T
    # MEMs#508: real 0.7.2 carries both Euler conventions alongside the selected scalar.
    # Field names/order mirror real exactly — the mock must stay a strict subset.
    parametric::Any
    euler::Any
end

struct HADSGESolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    linear_solution::Any
    method::Symbol
    spec::ModelSpec{T}
    reduction_basis::Any
    n_full_states::Int
    n_reduced::Int
    explained_variance::T
    jacobians::Any
    C_obs::Matrix{T}
    D_obs::Matrix{T}
end

struct BlanchardOLG{T<:AbstractFloat}
    alpha::T
    beta::T
    delta::T
    gamma::T
    Z::T
    b::T
end

struct BlanchardOLGSteadyState{T<:AbstractFloat}
    k::T
    C::T
    r::T
    w::T
    H::T
    mpc::T
    b::T
    converged::Bool
end

struct BlanchardOLGSolution{T<:AbstractFloat}
    ss::BlanchardOLGSteadyState{T}
    M::Matrix{T}
    eigenvalues::Vector{ComplexF64}
    stable_eig::T
    policy_slope::T
    determinate::Bool
end

# Continuous-time HA (C041)
struct CTPoissonIncome{T<:AbstractFloat}
    z::Vector{T}
    lambda::Vector{T}
end

struct CTAiyagari{T<:AbstractFloat}
    alpha::T
    rho::T
    sigma::T
    delta::T
    Z::T
    income::CTPoissonIncome{T}
    a_min::T
    a_max::T
    I::Int
end

function CTAiyagari(; alpha::Real=0.36, rho::Real=0.05, sigma::Real=2.0, delta::Real=0.05,
                      Z::Real=1.0, z::AbstractVector=[0.1, 0.2],
                      lambda::AbstractVector=[0.5, 0.5],
                      a_min::Real=0.0, a_max::Real=30.0, I::Int=500)
    T = Float64
    inc = CTPoissonIncome{T}(collect(T, z), collect(T, lambda))
    CTAiyagari{T}(T(alpha), T(rho), T(sigma), T(delta), T(Z), inc, T(a_min), T(a_max), I)
end

struct CTSteadyState{T<:AbstractFloat}
    r::T
    w::T
    K::T
    L::T
    a::Vector{T}
    g::Matrix{T}
    v::Matrix{T}
    c::Matrix{T}
    s::Matrix{T}
    A::Any
    converged::Bool
end

struct CTTransition{T<:AbstractFloat}
    t::Vector{T}
    Z::Vector{T}
    K::Vector{T}
    r::Vector{T}
    w::Vector{T}
    C::Vector{T}
    converged::Bool
    iterations::Int
end

struct CTTwoAsset{T<:AbstractFloat}
    sigma::T
    rho::T
    r_a::T
    r_b::T
    chi::T
    w::T
    income::Any
    b_max::T
    a_max::T
    Ib::Int
    Ia::Int
end

function CTTwoAsset(; sigma::Real=2.0, rho::Real=0.06, r_a::Real=0.05, r_b::Real=0.02,
                      chi::Real=0.03, w::Real=1.0, b_max::Real=40.0, a_max::Real=70.0,
                      Ib::Int=40, Ia::Int=25)
    T = Float64
    CTTwoAsset{T}(T(sigma), T(rho), T(r_a), T(r_b), T(chi), T(w), nothing,
                  T(b_max), T(a_max), Ib, Ia)
end

struct CTTwoAssetSolution{T<:AbstractFloat}
    b::Vector{T}
    a::Vector{T}
    V::Array{T,3}
    c::Array{T,3}
    d::Array{T,3}
    sb::Array{T,3}
    sa::Array{T,3}
    g::Array{T,3}
    B::T
    A::T
    gen::Any
    hjb_converged::Bool
end

struct X13FilterResult{T<:AbstractFloat}
    trend::Vector{T}
    seasonal::Vector{T}
    irregular::Vector{T}
    adjusted::Vector{T}
    original::Vector{T}
    method::Symbol
    arima_order::NTuple{N,Int} where N
    frequency::Int
    transform::Symbol
    sigma2::T
    aic::T
    n_outliers::Int
    T_obs::Int
end

struct IOData{T}
    Z::Matrix{T}
    Y::Matrix{T}
    va::Matrix{T}
    x::Vector{T}
    sectors::Vector{String}
    regions::Vector{String}
    fd_cats::Vector{String}
    va_cats::Vector{String}
    extensions::Dict{String,Any}
    unit::String
    year::Int
    source::String
    meta::Dict{String,Any}
end

const _HA_EXAMPLE_NAMES = (:krusell_smith, :one_asset_hank, :two_asset_hank, :huggett, :endogenous_labor)

function load_ha_example(name::Symbol; distribution::Symbol=:young)
    name in _HA_EXAMPLE_NAMES || error(
        "Unknown HA-DSGE example: :$name. Available: :krusell_smith, :one_asset_hank, :two_asset_hank, :huggett, :endogenous_labor")
    distribution in (:young, :winberry) || error("distribution must be :young or :winberry")
    hh = HouseholdSystem{Float64}(; model=name)
    return ModelSpec(; n_endog=0, n_exog=1, endog_names=Symbol[], exog_names=[:eps_Z],
                     agents=(household=hh,))
end
load_ha_example(name::String; distribution::Symbol=:young) =
    load_ha_example(Symbol(replace(name, "-" => "_")); distribution=distribution)

_mock_ha_model(spec::ModelSpec) = only(agents_of(spec, HouseholdSystem)).model

struct KrusellSmithSolution{T<:AbstractFloat}
    steady_state::HASteadyState{T}
    plm_coefficients::Dict{Symbol,Vector{T}}
    r_squared::Dict{Symbol,T}
    spec::ModelSpec{T}
    converged::Bool
    iterations::Int
end

# W13/#115 — Den Haan (2010) accuracy. Field names/order mirror real DenHaanAccuracy.
struct DenHaanAccuracy{T<:AbstractFloat}
    aggregate::Symbol
    dh_max::T
    dh_mean::T
    sigma_ref::T
    sigma_plm::T
    ref_path::Vector{T}
    plm_path::Vector{T}
    T_sim::Int
    T_burn::Int
    source::Symbol
end

# Guards mirror real: @assert on the horizon, and :huggett is unsupported because it has no
# aggregate capital. The CLI pre-guards both, but the mock must still refuse so a regression
# in that pre-guard cannot pass at T1/T2.
function den_haan_test(ks::KrusellSmithSolution{T}; T_sim::Int=10000, T_burn::Int=1000,
                       rho_z::Real=0.95, sigma_z::Real=0.007, seed::Int=98765) where T
    T_sim > T_burn + 10 || throw(AssertionError("T_sim must exceed T_burn by at least 10"))
    _mock_ha_model(ks.spec) === :huggett &&
        error("den_haan_test is implemented for the capital models (:aiyagari/:ks)")
    n = T_sim - T_burn
    ref = T[T(1.0) + T(0.01) * sin(i / 10) for i in 1:n]
    plm = ref .+ T(0.0005)
    DenHaanAccuracy{T}(:K, T(0.08), T(0.03), T(0.005), T(0.005), ref, plm,
                       T_sim, T_burn, :plm)
end

# Real 0.7.2 has a SECOND method for the linearized solutions, which recover the implied
# law of motion by regression over T_fit periods. It rejects :krusell_smith (that case
# takes the KrusellSmithSolution above) and, like the other method, refuses :huggett.
function den_haan_test(sol::HADSGESolution{T}; T_sim::Int=2000, T_burn::Int=200,
                       T_fit::Int=4000, rho_z::Real=0.95, sigma_z::Real=0.007,
                       seed::Int=98765) where T
    T_sim > T_burn + 10 || throw(AssertionError("T_sim must exceed T_burn by at least 10"))
    T_fit > 100 || throw(AssertionError("T_fit must be > 100 to fit the implied law of motion"))
    _mock_ha_model(sol.spec) === :huggett &&
        error("den_haan_test is implemented for the capital models (:aiyagari/:ks)")
    sol.method in (:ssj, :reiter) || error(
        "den_haan_test(::HADSGESolution) supports the linearized methods :ssj and :reiter; " *
        "got :$(sol.method). For a Krusell-Smith solution pass the KrusellSmithSolution.")
    n = T_sim - T_burn
    ref = T[T(1.0) + T(0.02) * sin(i / 8) for i in 1:n]
    plm = ref .+ T(0.004)
    # Deliberately worse than the fitted-PLM numbers above: upstream measures 12.2% (ssj)
    # and 5.5% (reiter) against 0.07% for Krusell-Smith on the same model.
    DenHaanAccuracy{T}(:K, T(0.122), T(0.061), T(0.011), T(0.005), ref, plm,
                       T_sim, T_burn, :implied)
end

function _mock_ha_ss(spec::ModelSpec{T}; euler_points::Symbol=:midpoints, kwargs...) where T
    euler_points in (:nodes, :midpoints) || throw(ArgumentError(
        "_ha_steady_state: euler_points must be :nodes or :midpoints, got :$euler_points."))
    ss = HASteadyState{T}(
        Dict{Symbol,Any}(:savings => ones(T, 10, 2) * T(0.5)),
        ones(T, 10, 2) ./ 20,
        ones(T, 10, 2),
        Dict{Symbol,T}(:r => T(0.01), :w => T(1.0)),
        Dict{Symbol,T}(:K => T(10.0), :Y => T(1.0), :excess_demand => T(0.0),
                       :B => T(1.5), :B_supply => T(1.5)),
        nothing, nothing, true, 10, T(1e-6), T(0.0),
        nothing,
        (midpoints=(points=:midpoints, max=T(-2.1), mean=T(-3.0),
                    n_evaluated=18, n_constrained=2, n_offgrid=0),
         nodes=(points=:nodes, max=T(-5.4), mean=T(-6.2),
                n_evaluated=20, n_constrained=0, n_offgrid=0)))
    sel = euler_points === :nodes ? ss.euler.nodes.max : ss.euler.midpoints.max
    HASteadyState{T}(ss.policies, ss.distribution, ss.value_fn, ss.prices, ss.aggregates,
                     ss.grid, ss.income, ss.converged, ss.iterations, T(sel),
                     ss.excess_demand, ss.parametric, ss.euler)
end

function _mock_ha_solve(spec::ModelSpec{T}; method::Symbol=:ssj, ss=nothing,
                        n_reduced::Int=10, T_horizon::Int=300) where T
    ss0 = ss === nothing ? _mock_ha_ss(spec) : ss
    if method === :krusell_smith
        return KrusellSmithSolution{T}(ss0, Dict(:K => T[0.1, 0.9, 0.05]),
                                       Dict(:K => T(0.99)), spec, true, 5)
    end
    method in (:ssj, :reiter) || error(
        "Unknown HA-DSGE method: :$method. Use :ssj, :reiter, or :krusell_smith.")
    n_red = n_reduced
    n_sys = max(n_red + 1, 2)
    endog = [Symbol("x_$i") for i in 1:n_sys]
    dummy_dsge = ModelSpec(; n_endog=n_sys, n_exog=1,
                           endog_names=endog, exog_names=[:epsilon])
    G1 = Matrix{T}(I, n_sys, n_sys) * T(0.5)
    impact = ones(T, n_sys, 1) * T(0.1)
    lin = LinearDSGE{T}(Matrix{T}(I, n_sys, n_sys), G1, zeros(T, n_sys), impact,
                         zeros(T, n_sys, 0), dummy_dsge)
    dsol = DSGESolution{T}(G1, impact, zeros(T, n_sys), [1, 1], method,
                            Complex{T}[T(0.5) + 0im], dummy_dsge, lin)
    n_full = 20
    U = ones(T, n_full, min(n_red, n_full)) ./ T(n_full)
    C_obs = method === :ssj ? ones(T, 1, n_sys) : Matrix{T}(I, n_sys, n_sys)
    D_obs = method === :ssj ? ones(T, 1, 1) * T(0.1) : zeros(T, n_sys, 1)
    HADSGESolution{T}(ss0, dsol, method, spec, U, n_full, n_red, T(0.95),
                      nothing, C_obs, D_obs)
end

function irf(sol::HADSGESolution{T}, horizon::Int; ci_type::Symbol=:none) where T
    n_out = size(sol.C_obs, 1)
    vals = ones(T, horizon, n_out, 1) * T(0.05)
    vars = ["y$i" for i in 1:n_out]
    ImpulseResponse(vals, nothing, nothing, horizon, vars, ["epsilon"], ci_type)
end

function fevd(sol::HADSGESolution{T}, horizon::Int) where T
    n_out = size(sol.C_obs, 1)
    props = ones(T, n_out, 1, horizon)
    FEVD(copy(props), props, ["y$i" for i in 1:n_out], ["epsilon"])
end

function simulate(sol::HADSGESolution{T}, T_periods::Int;
                  shock_draws=nothing, rng=Random.default_rng()) where T
    n_out = size(sol.C_obs, 1)
    ones(T, T_periods, n_out) * T(0.01)
end

function historical_decomposition(sol::HADSGESolution{T}, data::AbstractMatrix,
        observables::Vector{Symbol}; measurement_error=nothing) where {T}
    T_obs = size(data, 1)
    n_obs = length(observables)
    n_shocks = max(sol.spec.n_exog, 1)
    varnames_hd = [string(s) for s in observables]
    shock_names = ["epsilon"]
    HistoricalDecomposition{T}(
        ones(T, T_obs, n_obs, n_shocks) * T(0.1),
        ones(T, T_obs, n_obs) * T(0.01),
        ones(T, T_obs, n_obs),
        ones(T, T_obs, n_shocks),
        T_obs, varnames_hd, shock_names, :ha)
end

function distribution_irf(sol::HADSGESolution{T}, horizon::Int;
                          shock_index::Int=1, shock_size::Real=1.0) where T
    sol.method === :ssj && error(
        "distribution IRFs are unavailable for method=:ssj; use method=:reiter.")
    zeros(T, 10, 2, horizon)
end

function inequality_irf(sol::HADSGESolution{T}, horizon::Int;
                        shock_index::Int=1, shock_size::Real=1.0) where T
    Dict{Symbol,Vector{T}}(
        :gini => fill(T(0.4), horizon),
        :p10 => fill(T(0.1), horizon),
        :p25 => fill(T(0.2), horizon),
        :p50 => fill(T(0.5), horizon),
        :p75 => fill(T(0.8), horizon),
        :p90 => fill(T(1.5), horizon),
    )
end

function inequality_irf(ss::HASteadyState{T}; T_periods::Int=50) where T
    Dict{Symbol,Vector{T}}(
        :gini => fill(T(0.4), T_periods),
        :p10 => fill(T(0.1), T_periods),
        :p25 => fill(T(0.2), T_periods),
        :p50 => fill(T(0.5), T_periods),
        :p75 => fill(T(0.8), T_periods),
        :p90 => fill(T(1.5), T_periods),
    )
end

function simulate_panel(ss::HASteadyState{T};
                        N_agents::Int=1000, T_periods::Int=100,
                        rng=Random.default_rng()) where T
    ones(T, N_agents, T_periods) .* T(1.0) .+ randn(rng, T, N_agents, T_periods) .* T(0.1)
end

function x13_filter(y::AbstractVector{T};
                    frequency::Int=12,
                    method::Symbol=:seats,
                    start::Tuple{Int,Int}=(1,1),
                    transform::Symbol=:auto,
                    model::Symbol=:auto,
                    trading_day::Bool=false,
                    easter::Bool=false,
                    easter_window::Int=8,
                    outliers::Bool=true,
                    critical_value::Float64=0.0) where {T<:AbstractFloat}
    n = length(y)
    n < 3 * frequency && throw(ArgumentError(
        "x13_filter requires at least 3 × frequency = $(3 * frequency) observations, got $n"))
    frequency ∉ (4, 12) && throw(ArgumentError(
        "x13_filter supports frequency 4 (quarterly) or 12 (monthly), got $frequency"))
    method ∉ (:seats, :x11) && throw(ArgumentError(
        "method must be :seats or :x11, got :$method"))
    z = Float64.(y)
    seas = [0.5 * sin(2π * t / frequency) for t in 1:n]
    trend = [z[t] - seas[t] for t in 1:n]
    irr = zeros(n)
    adj = z .- seas
    X13FilterResult{Float64}(trend, seas, irr, adj, z, method, (0,1,1,0,1,1),
                             frequency, transform === :auto ? :none : transform,
                             1.0, 0.0, outliers ? 1 : 0, n)
end
x13_filter(y::AbstractVector; frequency::Int=12, method::Symbol=:seats,
           transform::Symbol=:auto, trading_day::Bool=false, easter::Bool=false,
           outliers::Bool=true, critical_value::Float64=0.0) =
    x13_filter(Float64.(y); frequency=frequency, method=method, transform=transform,
               trading_day=trading_day, easter=easter, outliers=outliers,
               critical_value=critical_value)

# ── Input-Output analysis mock (C049) ─────────────────────────
# Faithful re-implementation of the MEMs io module (real formulas) so handler
# unit tests catch table-shaping bugs. `IOData` is defined above (~L4423).

struct IOExtension{T}
    F::Matrix{T}
    F_Y::Matrix{T}
    S::Matrix{T}
    stressors::Vector{String}
    unit::Vector{String}
end

struct IOMetaData
    source::String
    version::String
    history::Vector{String}
    files::Vector{Pair{String,String}}
end

struct LeontiefModel{T}
    A::Matrix{T}
    L::Matrix{T}
    x::Vector{T}
    io::IOData{T}
end

struct GhoshModel{T}
    B::Matrix{T}
    G::Matrix{T}
    x::Vector{T}
    io::IOData{T}
end

struct IOMultipliers
    values::Vector{Float64}
    kind::Symbol
    type::Symbol
    sectors::Vector{String}
end

struct LinkageResult
    backward::Vector{Float64}
    forward::Vector{Float64}
    Ui::Vector{Float64}
    Uj::Vector{Float64}
    classification::Vector{Symbol}
    sectors::Vector{String}
end

struct SDAResult
    effects::Dict{Symbol,Vector{Float64}}
    total::Vector{Float64}
    residual::Vector{Float64}
    method::Symbol
end

struct ExtractionResult
    total_loss::Float64
    sector_loss::Vector{Float64}
    extracted::Vector{Int}
end

struct FootprintResult
    total::Matrix{Float64}
    by_sector::Matrix{Float64}
    stressors::Vector{String}
    name::String
end

struct RegionalFootprintResult
    production::Matrix{Float64}
    consumption::Matrix{Float64}
    stressors::Vector{String}
    regions::Vector{String}
    name::String
end

struct BaqaeeFarhiResult
    domar::Vector{Float64}
    first_order::Vector{Float64}
    second_order::Matrix{Float64}
    influence::Vector{Float64}
    upstreamness::Vector{Float64}
    downstreamness::Vector{Float64}
    sectors::Vector{String}
end

# W3/#154 — Baqaee–Farhi standard-form types (subset of real fields).
struct ProductionNetwork{T<:AbstractFloat}
    theta::Vector{T}
    lambda::Vector{T}
    lambda_rev::Vector{T}
    mu::Vector{T}
    factor_supplies::Vector{T}
    node_names::Vector{String}
    parent::Vector{Int}
    n::Int
    M::Int
    F::Int
    nests::Symbol
    outer_nodes::Vector{Int}
    io::IOData{T}
end

struct BFEquilibrium{T<:AbstractFloat}
    dlogY::T
    dlog_x::Vector{T}
    dlog_p::Vector{T}
    hulten::T
    technology::T
    allocative::T
    profit_share::T
    dlogA::Vector{T}
    dlogL::Vector{T}
    dlogmu::Vector{T}
    converged::Bool
    iterations::Int
    residual::T
    sectors::Vector{String}
end

struct BFElasticities{T<:AbstractFloat}
    dlogw_dlogA::Matrix{T}
    dlogp_dlogA::Matrix{T}
    dlambda_dlogA::Matrix{T}
    factor_names::Vector{String}
    sectors::Vector{String}
end

struct BFLocal{T<:AbstractFloat}
    first_order::Vector{T}
    second_order::Matrix{T}
    lambda::Vector{T}
    Lambda::Vector{T}
    elasticities::Union{BFElasticities{T},Nothing}
    sectors::Vector{String}
    nests::Symbol
end

struct BFShockCurve{T<:AbstractFloat}
    shocks::Vector{T}
    exact::Vector{T}
    hulten::Vector{T}
    second_order::Vector{T}
    sector::String
    sector_index::Int
end

struct BFWedgeDecomp{T<:AbstractFloat}
    dlogY::T
    technology::T
    allocative::T
    allocative_mu::T
    allocative_factor::T
    factor_supply::T
    lambda_cost::Vector{T}
    lambda_rev::Vector{T}
    Lambda_cost::Vector{T}
    Lambda_rev::Vector{T}
    mu::Vector{T}
    dlogA::Vector{T}
    dlogmu::Vector{T}
    dlogL::Vector{T}
    dlog_Lambda::Vector{T}
    sectors::Vector{String}
end

struct BFMisallocation{T<:AbstractFloat}
    distance::T
    first_order::T
    second_order::T
    H_mu::Matrix{T}
    delta_logmu::Vector{T}
    point::Symbol
    lambda::Vector{T}
    mu::Vector{T}
    sectors::Vector{String}
end

# W4/#155 — classical + MRIO result types (subset of real fields).
struct PriceModelResult
    dp::Vector{Float64}
    p::Vector{Float64}
    dv::Vector{Float64}
    mode::Symbol
    sectors::Vector{String}
end

struct ImpactResult
    total::Float64
    by_sector::Vector{Float64}
    dy::Vector{Float64}
    kind::Symbol
    type::Symbol
    sectors::Vector{String}
    fixed::Vector{Int}
end

struct NetworkStatsResult
    domar::Vector{Float64}
    herfindahl::Float64
    multipliers::Vector{Float64}
    multiplier_dispersion::Float64
    apl::Matrix{Float64}
    in_degree::Vector{Float64}
    out_degree::Vector{Float64}
    upstreamness::Vector{Float64}
    downstreamness::Vector{Float64}
    sectors::Vector{String}
end

struct VerticalSpecialization
    vs::Float64
    vs_share::Float64
    vs1::Float64
    domestic_content::Float64
    dc_share::Float64
    gross_exports::Float64
    region::String
    by_sector::Vector{Float64}
end

struct ExportDecomposition
    dva::Float64
    rdv::Float64
    fva::Float64
    pdc::Float64
    gross_exports::Float64
    vax_ratio::Float64
    region::String
    terms::Vector{Float64}
    by_sector::Matrix{Float64}
    sectors::Vector{String}
end

struct IOSourceTable
    rows::Vector{Tuple{Symbol,NamedTuple}}
end

_io_invdiag(x::AbstractVector{T}) where {T} =
    T[xi == zero(T) ? zero(T) : one(T) / xi for xi in x]

technical_coefficients(io::IOData) = io.Z * Diagonal(_io_invdiag(io.x))
function leontief_inverse(io::IOData{T}) where {T}
    A = technical_coefficients(io); Matrix{T}(inv(I - A))
end
allocation_coefficients(io::IOData) = Diagonal(_io_invdiag(io.x)) * io.Z
function ghosh_inverse(io::IOData{T}) where {T}
    B = allocation_coefficients(io); Matrix{T}(inv(I - B))
end
function leontief(io::IOData{T}) where {T}
    A = technical_coefficients(io); LeontiefModel{T}(A, Matrix{T}(inv(I - A)), copy(io.x), io)
end
function ghosh(io::IOData{T}) where {T}
    B = allocation_coefficients(io); GhoshModel{T}(B, Matrix{T}(inv(I - B)), copy(io.x), io)
end

function _io_household(io::IOData, kind::Symbol)
    invx = _io_invdiag(io.x)
    if kind == :output
        return ones(length(io.x))
    elseif kind == :income
        return vec(sum(io.va, dims=1)) .* invx
    elseif kind == :employment
        haskey(io.extensions, "employment") ||
            throw(ArgumentError("no 'employment' extension; add one with add_extension!"))
        return vec(sum(io.extensions["employment"].F, dims=1)) .* invx
    else
        throw(ArgumentError("kind must be :output, :income, or :employment"))
    end
end

function _io_closed_leontief(io::IOData)
    A = technical_coefficients(io); n = size(A, 1); invx = _io_invdiag(io.x)
    hinc = vec(io.va[1, :]) .* invx
    y = vec(sum(io.Y, dims=2)); hc = y ./ max(sum(y), eps())
    Abar = [A hc; reshape(collect(float.(hinc)), 1, n) 0.0]
    Matrix{Float64}(inv(I - Abar))
end

function multipliers(io::IOData; kind::Symbol=:output, type::Symbol=:I)
    L = leontief_inverse(io); h = _io_household(io, kind)
    if type == :I
        vals = kind == :output ? vec(sum(L, dims=1)) : vec(L' * h)
    elseif type == :II
        n = length(io.x); L2 = _io_closed_leontief(io)
        vals = kind == :output ? vec(sum(view(L2, 1:n, 1:n), dims=1)) :
               kind == :income ? collect(view(L2, n + 1, 1:n)) :
               vec(transpose(view(L2, 1:n, 1:n)) * h)
    else
        throw(ArgumentError("type must be :I or :II"))
    end
    IOMultipliers(vals, kind, type, copy(io.sectors))
end

_io_classify(ui, uj) = ui > 1 && uj > 1 ? :key : ui > 1 && uj <= 1 ? :backward :
                       ui <= 1 && uj > 1 ? :forward : :weak

function linkages(io::IOData; forward::Symbol=:ghosh)
    L = leontief_inverse(io); n = size(L, 1)
    backward = vec(sum(L, dims=1))
    fwd = forward == :ghosh ? vec(sum(ghosh_inverse(io), dims=2)) :
          forward == :leontief ? vec(sum(L, dims=2)) :
          throw(ArgumentError("forward must be :ghosh or :leontief"))
    Ui = backward ./ (sum(backward) / n); Uj = fwd ./ (sum(fwd) / n)
    LinkageResult(backward, fwd, Ui, Uj,
                  [_io_classify(Ui[i], Uj[i]) for i in 1:n], copy(io.sectors))
end
rasmussen(io::IOData) = linkages(io)
key_sectors(io::IOData) = linkages(io).classification

function sda(io0::IOData, io1::IOData; method::Symbol=:additive,
            factors=nothing, on=:output)
    is_ext = !(on === :output || on === "output")
    on_key = is_ext ? String(on) : "output"
    if is_ext
        haskey(io0.extensions, on_key) || throw(ArgumentError("no extension '$on_key'"))
        haskey(io1.extensions, on_key) || throw(ArgumentError("no extension '$on_key'"))
    end
    default_facs = is_ext ? [:intensity, :technology, :final_demand] : [:technology, :final_demand]
    use_legacy = factors === nothing && !is_ext && method in (:additive, :multiplicative)
    facs = factors === nothing ? default_facs : collect(Symbol.(factors))
    isempty(facs) && throw(ArgumentError("factors must be non-empty"))
    has_int = any(f -> f in (:intensity, :emission_intensity), facs)
    has_int && !is_ext && throw(ArgumentError(":intensity requires on=<extension name>"))
    is_ext && !has_int && throw(ArgumentError("emission SDA requires :intensity among factors"))
    if method == :multiplicative
        (is_ext || facs != [:technology, :final_demand]) && throw(ArgumentError(
            "method=:multiplicative is only implemented for the two-factor " *
            "output path factors=[:technology, :final_demand]"))
    elseif method != :additive
        throw(ArgumentError("method must be :additive or :multiplicative"))
    end
    L0 = leontief_inverse(io0); L1 = leontief_inverse(io1)
    y0 = vec(sum(io0.Y, dims=2)); y1 = vec(sum(io1.Y, dims=2))
    ΔL = L1 - L0; Δy = y1 - y0
    if method == :additive
        L_eff = 0.5 .* (ΔL * y0 .+ ΔL * y1); Y_eff = 0.5 .* (L1 * Δy .+ L0 * Δy)
        total = L1 * y1 .- L0 * y0
        residual = total .- (L_eff .+ Y_eff)
        if use_legacy
            return SDAResult(Dict(:L => L_eff, :Y => Y_eff), total, residual, :additive)
        end
        effects = Dict{Symbol,Vector{Float64}}()
        z = zeros(length(total))
        for f in facs
            if f === :technology
                effects[f] = L_eff
            elseif f in (:final_demand, :fd)
                effects[f] = Y_eff
            elseif f in (:intensity, :emission_intensity)
                effects[f] = 0.5 .* (L_eff .+ Y_eff)
            else
                effects[f] = z
            end
        end
        return SDAResult(effects, total, residual, :additive)
    else
        x0 = L0 * y0; x1 = L1 * y1; ratio = x1 ./ max.(x0, eps())
        L_eff = (L1 * y0) ./ max.(x0, eps()); Y_eff = ratio ./ max.(L_eff, eps())
        return SDAResult(Dict(:L => L_eff, :Y => Y_eff), ratio,
                         ratio .- (L_eff .* Y_eff), :multiplicative)
    end
end

_io_sector_idx(io::IOData, s::Integer) = [Int(s)]
_io_sector_idx(io::IOData, s::AbstractVector{<:Integer}) = collect(Int, s)
function _io_sector_idx(io::IOData, s::AbstractString)
    idx = findfirst(==(s), io.sectors)
    idx === nothing && throw(ArgumentError("sector '$s' not found")); [idx]
end
_io_sector_idx(io::IOData, s::AbstractVector{<:AbstractString}) =
    reduce(vcat, _io_sector_idx.(Ref(io), s))

function hypothetical_extraction(io::IOData, sectors; mode::Symbol=:complete,
                                 share::Real=1.0, region=nothing)
    (0.0 < share <= 1.0 + 1e-15) || throw(ArgumentError("share must be in (0, 1]; got $share"))
    mode in (:complete, :backward, :forward, :partial) ||
        throw(ArgumentError("mode must be :complete, :backward, :forward or :partial"))
    if region !== nothing
        region in io.regions || throw(ArgumentError("region '$region' not found; available: $(io.regions)"))
        ns = length(io.sectors) ÷ max(1, length(io.regions))
        ridx = findfirst(==(region), io.regions)
        idx = collect((ridx - 1) * ns + 1 : ridx * ns)
    else
        idx = _io_sector_idx(io, sectors)
    end
    A = technical_coefficients(io)
    y = vec(sum(io.Y, dims=2)); x_base = (I - A) \ y
    Ae = copy(A); Ae[idx, :] .= 0.0; Ae[:, idx] .= 0.0
    ye = copy(y); ye[idx] .= 0.0; x_red = (I - Ae) \ ye
    loss = x_base .- x_red
    ExtractionResult(sum(loss), loss, idx)
end

function add_extension!(io::IOData{T}, name::AbstractString, F::AbstractMatrix;
                        stressors, unit, F_Y=nothing) where {T}
    Fm = Matrix{T}(F); S = Fm * Diagonal(_io_invdiag(io.x))
    FYm = F_Y === nothing ? zeros(T, size(Fm, 1), size(io.Y, 2)) : Matrix{T}(F_Y)
    io.extensions[String(name)] =
        IOExtension{T}(Fm, FYm, S, collect(String.(stressors)), collect(String.(unit)))
    io
end
_io_ext(io, name) = haskey(io.extensions, name) ? io.extensions[name] :
    throw(ArgumentError("no extension '$name'"))
intensities(io::IOData, name::AbstractString) = _io_ext(io, name).S
emission_multipliers(io::IOData, name::AbstractString) =
    _io_ext(io, name).S * leontief_inverse(io)
function footprint(io::IOData, name::AbstractString; by::Symbol=:sector)
    by in (:sector, :region) || throw(ArgumentError("by must be :sector or :region; got :$by"))
    ext = _io_ext(io, name); L = leontief_inverse(io); M = ext.S * L
    total = M * io.Y .+ ext.F_Y; y = vec(sum(io.Y, dims=2))
    by === :sector && return FootprintResult(total, M .* reshape(y, 1, :), ext.stressors, String(name))
    G = max(length(io.regions), 1)
    ns = length(io.sectors) ÷ G
    prod = zeros(size(M, 1), G); consu = zeros(size(M, 1), G)
    for g in 1:G
        sl = (g - 1) * ns + 1 : g * ns
        prod[:, g] = vec(sum(M[:, sl]; dims=2))
        consu[:, g] = vec(sum(total; dims=2)) ./ G
    end
    return RegionalFootprintResult(prod, consu, ext.stressors, io.regions, String(name))
end

function parse_icio(path::AbstractString; year=nothing, member::AbstractString="",
                    aggregate_cn_mx::Bool=true, check::Bool=false, unit::AbstractString="Million USD")
    ext = lowercase(splitext(path)[2])
    ext == ".zip" && error("Reading zip members requires the ZipFile package. " *
                           "Run `]add ZipFile` and `using ZipFile` to enable it.")
    ext == ".xlsx" && error("Reading Excel sheets requires the XLSX package. " *
                            "Run `]add XLSX` and `using XLSX` to enable it.")
    lines = String[strip(l) for l in eachline(path) if !isempty(strip(l))]
    isempty(lines) && throw(ArgumentError("parse_icio: need a header row and at least one data row"))
    header = String[strip(p) for p in split(lines[1], ',')]
    col_labels = header[2:end]
    row_labels = String[]
    data_rows = Vector{Vector{Float64}}()
    fd_re = r"HFCE|NPISH|NPS|GGFC|GFCF|INVNT|INV|DIRP|DPABR|FD|P33|DISC"
    va_exact = Set(["TLS", "VA", "VALU", "TAXES", "TAXSUB"])
    tot_col = Set(["OUT", "TOTAL"])
    tot_row = Set(["OUT", "OUTPUT"])
    for line in lines[2:end]
        parts = String[strip(p) for p in split(line, ',')]
        lab = parts[1]
        (isempty(lab) || lab in tot_row) && continue
        push!(row_labels, lab)
        vals = Float64[]
        for j in 1:length(col_labels)
            raw = j + 1 <= length(parts) ? parts[j + 1] : "0"
            v = tryparse(Float64, raw)
            push!(vals, v === nothing ? 0.0 : v)
        end
        push!(data_rows, vals)
    end
    keep_c = [j for (j, c) in enumerate(col_labels) if !(c in tot_col)]
    col_labels = col_labels[keep_c]
    M = reduce(vcat, (reshape(r[keep_c], 1, :) for r in data_rows))
    is_va = [r in va_exact || occursin(r"VALU|TAX", r) for r in row_labels]
    is_fd = [occursin(fd_re, c) for c in col_labels]
    ind_rows = findall(!, is_va); va_rows = findall(identity, is_va)
    ind_cols = findall(!, is_fd); fd_cols = findall(identity, is_fd)
    Z = Matrix{Float64}(M[ind_rows, ind_cols])
    Y = isempty(fd_cols) ? zeros(Float64, length(ind_rows), 0) : Matrix{Float64}(M[ind_rows, fd_cols])
    va = if isempty(va_rows)
        x_tmp = vec(sum(Z; dims=2)) .+ vec(sum(Y; dims=2))
        reshape(x_tmp .- vec(sum(Z; dims=1)), 1, length(x_tmp))
    else
        Matrix{Float64}(M[va_rows, ind_cols])
    end
    ind_labs = row_labels[ind_rows]
    regions = unique(String[let i = findfirst('_', lab)
                                i === nothing ? lab : lab[1:i-1]
                            end for lab in ind_labs])
    fd_cats = isempty(fd_cols) ? String[] : col_labels[fd_cols]
    va_cats = isempty(va_rows) ? ["VA"] : row_labels[va_rows]
    x = vec(sum(Z; dims=2)) .+ vec(sum(Y; dims=2))
    yr = year === nothing ? 0 : Int(year)
    IOData{Float64}(Z, Y, va, x, ind_labs, regions, fd_cats, va_cats,
                    Dict{String,Any}(), String(unit), yr, "OECD ICIO", Dict{String,Any}())
end
function region_indices(io::IOData, region::AbstractString)
    i = findfirst(==(region), io.regions)
    i === nothing && throw(ArgumentError("region '$region' not found"))
    ns = length(io.sectors) ÷ max(1, length(io.regions))
    collect((i - 1) * ns + 1 : i * ns)
end
region_indices(io::IOData, region::Integer) = region_indices(io, io.regions[region])

domar_weights(io::IOData) = io.x ./ sum(io.va)
function baqaee_farhi(io::IOData; theta=nothing, sigma=nothing)
    λ = domar_weights(io); L = leontief_inverse(io)
    y = vec(sum(io.Y, dims=2)); β = y ./ sum(y)
    n = length(λ)
    BaqaeeFarhiResult(λ, copy(λ), zeros(n, n), vec(L' * β),
                      vec(sum(L, dims=2)), vec(sum(L, dims=1)), copy(io.sectors))
end

function _io_bf_as_vector(::Type{T}, x, n::Int, name::String) where {T}
    if x isa AbstractVector
        length(x) == n || throw(ArgumentError(
            "$name must be a scalar or length-$n vector; got length $(length(x))"))
        return T[T(v) for v in x]
    else
        return fill(T(x), n)
    end
end

function production_network(io::IOData{T};
                            theta=1.0, sigma=1.0, epsilon=1.0, eta=1.0,
                            nests::Symbol=:single, factors=:single, mu=1.0,
                            check::Bool=true) where {T<:AbstractFloat}
    nests in (:single, :two) || throw(ArgumentError(
        "nests must be :single or :two (got $nests); :custom is not yet implemented"))
    n = length(io.x)
    θ_sec = _io_bf_as_vector(T, theta, n, "theta")
    _io_bf_as_vector(T, epsilon, n, "epsilon")
    _io_bf_as_vector(T, eta, n, "eta")
    μ_sec = _io_bf_as_vector(T, mu, n, "mu")
    all(μ_sec .>= one(T) - T(1e-14)) || throw(ArgumentError(
        "mu must satisfy μ ≥ 1 for all sectors (got min=$(minimum(μ_sec)))"))
    if factors === :single
        F = 1; fnames = ["factor"]
    elseif factors === :va_cats
        F = size(io.va, 1); fnames = String.(io.va_cats)
    elseif factors isa AbstractMatrix
        F = size(factors, 1); fnames = ["factor$f" for f in 1:F]
    else
        throw(ArgumentError(
            "factors must be :single, :va_cats, or an F×n matrix; got $factors"))
    end
    F >= 1 || throw(ArgumentError("production_network requires at least one factor"))
    M = nests === :single ? n : 3n
    N = 1 + M + F
    outer = collect(2:n+1)
    λ = zeros(T, N)
    λr = zeros(T, N)
    dw = T.(io.x ./ max(sum(io.va), eps(T)))
    for (k, g) in enumerate(outer)
        λ[g] = dw[k]; λr[g] = dw[k]
    end
    λ[1] = one(T)
    parent = zeros(Int, N)
    for i in 1:n
        parent[i + 1] = i
    end
    names = String["household"]
    append!(names, String.(io.sectors))
    append!(names, fnames)
    while length(names) < N
        push!(names, "node$(length(names)+1)")
    end
    θ = vcat(T(sigma), θ_sec)
    pad = T(theta isa AbstractVector ? theta[1] : theta)
    length(θ) < 1 + M && (θ = vcat(θ, fill(pad, 1 + M - length(θ))))
    length(θ) > 1 + M && (θ = θ[1:1+M])
    μ = ones(T, M)
    for (k, g) in enumerate(outer)
        μ[g - 1] = μ_sec[k]
    end
    Lfac = T.(dw[1:min(end, F)])
    length(Lfac) < F && (Lfac = vcat(Lfac, fill(T(1 / F), F - length(Lfac))))
    ProductionNetwork{T}(θ, λ, λr, μ, Lfac, names[1:N], parent, n, M, F,
                         nests, outer, io)
end

function bf_equilibrium(net::ProductionNetwork{T};
                        dlogA=nothing, dlogL=nothing, dlogmu=nothing,
                        method::Symbol=:newton, tol::Real=1e-10,
                        maxiter::Int=500, damping::Real=0.5) where {T<:AbstractFloat}
    method in (:newton, :fixedpoint) || throw(ArgumentError(
        "method must be :newton or :fixedpoint; got $method"))
    n, F = net.n, net.F
    dA = dlogA === nothing ? zeros(T, n) : _io_bf_as_vector(T, dlogA, n, "dlogA")
    dL = dlogL === nothing ? zeros(T, F) : _io_bf_as_vector(T, dlogL, F, "dlogL")
    dμ = dlogmu === nothing ? zeros(T, n) : _io_bf_as_vector(T, dlogmu, n, "dlogmu")
    λ_out = T[net.lambda[g] for g in net.outer_nodes]
    hult = dot(λ_out, dA) + (isempty(dL) ? zero(T) : dot(net.factor_supplies, dL))
    tech = dot(λ_out, dA)
    alloc = -dot(λ_out, dμ)
    dlogY = hult + T(0.5) * alloc
    conv = maxiter >= 2
    BFEquilibrium{T}(dlogY, dA, zeros(T, n), hult, tech, alloc, zero(T),
                     dA, dL, dμ, conv, conv ? 1 : maxiter, conv ? T(0) : T(1),
                     String.(net.io.sectors))
end

function bf_elasticities(net::ProductionNetwork{T}) where {T<:AbstractFloat}
    n, F = net.n, net.F
    I_n = Matrix{T}(I, n, n)
    BFElasticities{T}(zeros(T, F, n), -I_n, zeros(T, n, n),
                      String.(net.node_names[net.M+2:net.M+1+F]),
                      String.(net.io.sectors))
end

function baqaee_farhi(net::ProductionNetwork{T};
                      hessian::Symbol=:auto, elasticities::Bool=true) where {T<:AbstractFloat}
    hessian in (:full, :none, :auto) || throw(ArgumentError(
        "hessian must be :full, :none, or :auto; got $hessian"))
    n = net.n
    do_H = hessian === :none ? false : true
    λ_out = T[net.lambda[g] for g in net.outer_nodes]
    Λ = net.lambda[net.M+2:net.M+1+net.F]
    H = do_H ? zeros(T, n, n) : zeros(T, 0, 0)
    elast = elasticities ? bf_elasticities(net) : nothing
    BFLocal{T}(λ_out, H, copy(net.lambda), Vector{T}(Λ), elast,
               String.(net.io.sectors), net.nests)
end

function _io_bf_sector_index(net::ProductionNetwork, sector::Integer)
    i = Int(sector)
    1 <= i <= net.n || throw(ArgumentError("sector index $i out of range 1:$(net.n)"))
    return i
end
function _io_bf_sector_index(net::ProductionNetwork, sector::AbstractString)
    i = findfirst(==(String(sector)), String.(net.io.sectors))
    i === nothing && throw(ArgumentError("sector $(repr(sector)) not found in network sectors"))
    return i
end
_io_bf_sector_index(net::ProductionNetwork, sector::Symbol) =
    _io_bf_sector_index(net, String(sector))

function bf_shock_curve(net::ProductionNetwork{T}, sector;
                        range::Tuple{<:Real,<:Real}=(-0.5, 0.5),
                        points::Int=41) where {T<:AbstractFloat}
    points >= 2 || throw(ArgumentError("points must be ≥ 2"))
    lo, hi = T(range[1]), T(range[2])
    lo < hi || throw(ArgumentError("range must be increasing"))
    idx = _io_bf_sector_index(net, sector)
    label = String(net.io.sectors[idx])
    λ_i = net.lambda[net.outer_nodes[idx]]
    shocks = collect(Base.range(lo, hi; length=points))
    exact = T[λ_i * s + T(0.1) * s * s for s in shocks]
    hult = T[λ_i * s for s in shocks]
    so = T[λ_i * s + T(0.05) * s * s for s in shocks]
    BFShockCurve{T}(shocks, exact, hult, so, label, idx)
end

function bf_wedge_decomp(net::ProductionNetwork{T};
                         dlogA=nothing, dlogmu=nothing, dlogL=nothing) where {T<:AbstractFloat}
    eq = bf_equilibrium(net; dlogA=dlogA, dlogL=dlogL, dlogmu=dlogmu)
    n, M, F = net.n, net.M, net.F
    λc = T[net.lambda[g] for g in net.outer_nodes]
    λr = T[net.lambda_rev[g] for g in net.outer_nodes]
    μ_out = T[net.mu[g - 1] for g in net.outer_nodes]
    Λc = net.lambda[M+2:M+1+F]
    Λr = net.lambda_rev[M+2:M+1+F]
    BFWedgeDecomp{T}(eq.dlogY, eq.technology, eq.allocative,
                     -dot(λc, eq.dlogmu), zero(T), dot(Vector{T}(Λc), eq.dlogL),
                     λc, λr, Vector{T}(Λc), Vector{T}(Λr), μ_out,
                     eq.dlogA, eq.dlogmu, eq.dlogL, zeros(T, F),
                     String.(net.io.sectors))
end

function bf_misallocation(net::ProductionNetwork{T}; point::Symbol=:efficient,
                          hessian::Symbol=:auto) where {T<:AbstractFloat}
    point in (:efficient, :observed) || throw(ArgumentError(
        "point must be :efficient or :observed; got $point"))
    hessian in (:full, :none, :auto) || throw(ArgumentError(
        "hessian must be :full, :none, or :auto; got $hessian"))
    n = net.n
    μ_sec = T[net.mu[g - 1] for g in net.outer_nodes]
    v = log.(μ_sec)
    λ_out = T[net.lambda[g] for g in net.outer_nodes]
    do_H = hessian !== :none
    H = do_H ? zeros(T, n, n) : zeros(T, 0, 0)
    BFMisallocation{T}(zero(T), zero(T), zero(T), H, v, point, λ_out, μ_sec,
                       String.(net.io.sectors))
end

# ── W4/#155 classical + MRIO ─────────────────────────────────

function _io_price_shock_vector(io::IOData, shock, n::Int)
    shock === nothing && return zeros(Float64, n)
    if shock isa AbstractDict
        v = zeros(Float64, n)
        for (k, val) in shock
            idx = _io_sector_idx(io, k)
            length(idx) == 1 || throw(ArgumentError(
                "price shock key must identify a single sector; got $k → $idx"))
            v[idx[1]] = Float64(val)
        end
        return v
    end
    v = Float64.(vec(collect(shock)))
    length(v) == n || throw(ArgumentError("shock vector length $(length(v)) must equal n=$n"))
    return v
end

function price_model(io::IOData; dva=nothing, dtax=nothing, mode::Symbol=:leontief)
    n = length(io.x)
    dv = _io_price_shock_vector(io, dva, n) .+ _io_price_shock_vector(io, dtax, n)
    if mode === :leontief
        A = technical_coefficients(io)
        dp = Matrix{Float64}(inv(Matrix(I - A'))) * dv
    elseif mode === :ghosh
        B = allocation_coefficients(io)
        dp = Matrix{Float64}(inv(Matrix(I - B))) * dv
    else
        throw(ArgumentError("mode must be :leontief or :ghosh; got :$mode"))
    end
    PriceModelResult(dp, ones(Float64, n) .+ dp, dv, mode, copy(io.sectors))
end

function _io_impact_dy(io::IOData, dy, n::Int)
    if dy isa AbstractDict
        v = zeros(Float64, n)
        for (k, val) in dy
            idx = _io_sector_idx(io, k)
            length(idx) == 1 || throw(ArgumentError(
                "dy key must identify a single sector; got $k → $idx"))
            v[idx[1]] = Float64(val)
        end
        return v
    end
    v = Float64.(vec(collect(dy)))
    length(v) == n || throw(ArgumentError("dy length $(length(v)) must equal n=$n"))
    return v
end

function impact(io::IOData, dy; kind=:output, type::Symbol=:I, fix=Dict())
    n = length(io.x)
    dyv = _io_impact_dy(io, dy, n)
    kind_sym = kind isa Symbol ? kind : Symbol(kind)
    if type === :I
        dx = leontief_inverse(io) * dyv
    elseif type === :II
        L2 = _io_closed_leontief(io)
        dx = (L2 * vcat(dyv, 0.0))[1:n]
    else
        throw(ArgumentError("type must be :I or :II; got :$type"))
    end
    ImpactResult(sum(dx), Float64.(dx), dyv, kind_sym, type, copy(io.sectors), Int[])
end

function network_stats(io::IOData)
    λ = Float64.(domar_weights(io))
    L = leontief_inverse(io)
    A = technical_coefficients(io)
    mult = vec(sum(L, dims=1))
    n = length(mult)
    NetworkStatsResult(λ, sum(abs2, λ), Float64.(mult),
                       n > 1 ? std(Float64.(mult); corrected=true) : 0.0,
                       zeros(n, n),
                       Float64.(vec(sum(A, dims=2))), Float64.(vec(sum(A, dims=1))),
                       Float64.(vec(sum(L, dims=2))), Float64.(vec(sum(L, dims=1))),
                       copy(io.sectors))
end

function _io_ns_per_region(io::IOData)
    G = max(length(io.regions), 1)
    n = length(io.x)
    ns = n ÷ G
    G * ns == n || throw(ArgumentError(
        "unbalanced MRIO layout: length(x)=$n, nregions=$G, nsectors=$ns " *
        "(expected n = G·nsectors)"))
    ns
end

function _io_resolve_region(io::IOData, region::AbstractString)
    ridx = findfirst(==(region), io.regions)
    ridx === nothing && throw(ArgumentError(
        "region '$region' not found; available: $(io.regions)"))
    ridx
end
function _io_resolve_region(io::IOData, region::Integer)
    G = length(io.regions)
    (1 <= region <= G) || throw(ArgumentError("region index $region out of 1:$G"))
    Int(region)
end

function bilateral_trade(io::IOData{T}, exporter, importer;
                         kind::Symbol=:total) where {T}
    kind in (:total, :intermediate, :final) || throw(ArgumentError(
        "kind must be :total, :intermediate, or :final; got :$kind"))
    ri = _io_resolve_region(io, exporter)
    si = _io_resolve_region(io, importer)
    ns = _io_ns_per_region(io)
    Ir = (ri - 1) * ns + 1 : ri * ns
    Is = (si - 1) * ns + 1 : si * ns
    Zrs = io.Z[Ir, Is]
    inter_by = vec(sum(Zrs; dims=2))
    G = length(io.regions)
    n_fd = size(io.Y, 2)
    blocked = G > 1 && n_fd > 0 && n_fd % G == 0
    n_fd_r = blocked ? n_fd ÷ G : n_fd
    if blocked
        Jr = (si - 1) * n_fd_r + 1 : si * n_fd_r
        final_by = vec(sum(io.Y[Ir, Jr]; dims=2))
    else
        final_by = ri == si ? vec(sum(io.Y[Ir, :]; dims=2)) : zeros(T, ns)
    end
    if kind === :intermediate
        by_sec = inter_by; fin = zero(T); inter = sum(inter_by); tot = inter
    elseif kind === :final
        by_sec = final_by; inter = zero(T); fin = sum(final_by); tot = fin
    else
        by_sec = inter_by .+ final_by
        inter = sum(inter_by); fin = sum(final_by); tot = inter + fin
    end
    (intermediate=Float64(inter), final=Float64(fin), total=Float64(tot),
     by_sector=Float64.(by_sec))
end

function gross_exports(io::IOData{T}, region) where {T}
    ri = _io_resolve_region(io, region)
    G = length(io.regions)
    ns = _io_ns_per_region(io)
    E = zeros(T, ns)
    for s in 1:G
        s == ri && continue
        bt = bilateral_trade(io, ri, s; kind=:total)
        E .+= T.(bt.by_sector)
    end
    E
end

function aggregate(io::IOData{T}; region_map=nothing, sector_map=nothing) where {T}
    G = length(io.regions)
    ns = _io_ns_per_region(io)
    n = length(io.x)
    rmap = Dict{String,String}()
    if region_map !== nothing
        for (k, v) in region_map
            rmap[String(k)] = String(v)
        end
    end
    old_to_new_r = [get(rmap, io.regions[r], io.regions[r]) for r in 1:G]
    new_regions = unique(old_to_new_r)
    G2 = length(new_regions)
    r_of = [findfirst(==(old_to_new_r[r]), new_regions) for r in 1:G]
    sec_types = String[io.sectors[i] for i in 1:ns]
    smap = Dict{String,String}()
    if sector_map !== nothing
        for (k, v) in sector_map
            smap[String(k)] = String(v)
        end
    end
    old_to_new_s = [get(smap, sec_types[j], sec_types[j]) for j in 1:ns]
    new_sec = unique(old_to_new_s)
    ns2 = length(new_sec)
    s_of = [findfirst(==(old_to_new_s[j]), new_sec) for j in 1:ns]
    if G2 == G && ns2 == ns
        return io
    end
    n2 = G2 * ns2
    function new_ind(old_i::Int)
        r_old = (old_i - 1) ÷ ns + 1
        s_old = (old_i - 1) % ns + 1
        (r_of[r_old] - 1) * ns2 + s_of[s_old]
    end
    Z2 = zeros(T, n2, n2); x2 = zeros(T, n2)
    for i in 1:n, j in 1:n
        Z2[new_ind(i), new_ind(j)] += io.Z[i, j]
    end
    for i in 1:n
        x2[new_ind(i)] += io.x[i]
    end
    n_fd = size(io.Y, 2)
    Y2 = zeros(T, n2, n_fd)
    for i in 1:n, j in 1:n_fd
        Y2[new_ind(i), j] += io.Y[i, j]
    end
    n_va = size(io.va, 1)
    va2 = zeros(T, n_va, n2)
    for f in 1:n_va, j in 1:n
        va2[f, new_ind(j)] += io.va[f, j]
    end
    secs2 = String[]
    for r in 1:G2, s in 1:ns2
        push!(secs2, G2 == 1 ? new_sec[s] : string(new_regions[r], "_", new_sec[s]))
    end
    IOData{T}(Z2, Y2, va2, x2, secs2, new_regions, copy(io.fd_cats), copy(io.va_cats),
              Dict{String,Any}(), io.unit, io.year,
              isempty(io.source) ? "aggregate" : io.source * " [aggregated]",
              io.meta)
end

function balance(io::IOData{T}; method::Symbol=:ras, tol::Real=1e-10,
                 maxiter::Integer=1000) where {T<:AbstractFloat}
    method in (:ras, :gras) ||
        throw(ArgumentError("method must be :ras or :gras; got :$method"))
    io  # already-balanced tables are a fixed point
end

function vertical_specialization(io::IOData{T}, region=nothing) where {T}
    G = length(io.regions)
    if region === nothing
        G == 1 || throw(ArgumentError(
            "region is required when nregions=$(G) > 1"))
        region = io.regions[1]
    end
    ri = _io_resolve_region(io, region)
    rname = io.regions[ri]
    ns = _io_ns_per_region(io)
    E = gross_exports(io, ri)
    ge = sum(E)
    if ge == zero(T) || G == 1
        return VerticalSpecialization(0.0, 0.0, 0.0, Float64(ge),
                                      ge == 0 ? 0.0 : 1.0, Float64(ge),
                                      rname, zeros(Float64, ns))
    end
    L = leontief_inverse(io)
    va_tot = vec(sum(io.va; dims=1))
    v = T[io.x[j] == 0 ? zero(T) : va_tot[j] / io.x[j] for j in 1:length(io.x)]
    Is = (ri - 1) * ns + 1 : ri * ns
    vs_by = zeros(T, ns); vs_tot = zero(T)
    for t in 1:G
        t == ri && continue
        It = (t - 1) * ns + 1 : t * ns
        fva_per = vec(v[It]' * L[It, Is])
        vs_by .+= fva_per .* E
        vs_tot += dot(fva_per, E)
    end
    dc = ge - vs_tot
    VerticalSpecialization(Float64(vs_tot),
                           ge == 0 ? 0.0 : Float64(vs_tot / ge),
                           0.0, Float64(dc),
                           ge == 0 ? 0.0 : Float64(dc / ge),
                           Float64(ge), rname, Float64.(vs_by))
end

function export_decomposition(io::IOData{T}, region=nothing) where {T}
    G = length(io.regions)
    if region === nothing
        G == 1 || throw(ArgumentError(
            "region is required when nregions=$(G) > 1"))
        region = io.regions[1]
    end
    s = _io_resolve_region(io, region)
    rname = io.regions[s]
    ns = _io_ns_per_region(io)
    sec_types = String[io.sectors[i] for i in 1:ns]
    if G == 1
        return ExportDecomposition(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, rname,
                                   zeros(9), zeros(ns, 4), sec_types)
    end
    Es = gross_exports(io, s)
    ge = Float64(sum(Es))
    vs = vertical_specialization(io, s)
    fva = vs.vs
    # Split the residual domestic content into DVA (absorbed abroad) vs RDV
    # using the share of final vs intermediate exports, so the four aggregates
    # add up to gross exports. Exact KWW terms are T3's job (real MEMs).
    bt_final = 0.0; bt_inter = 0.0
    for r in 1:G
        r == s && continue
        bt = bilateral_trade(io, s, r; kind=:total)
        bt_final += bt.final; bt_inter += bt.intermediate
    end
    dc = ge - fva
    dva = dc * (bt_final / max(bt_final + bt_inter, eps()))
    rdv = dc - dva
    pdc = 0.0
    vax = ge == 0 ? 0.0 : dva / ge
    by = zeros(ns, 4)
    if ge > 0
        w = Float64.(Es) ./ ge
        by[:, 1] .= dva .* w; by[:, 2] .= rdv .* w
        by[:, 3] .= fva .* w; by[:, 4] .= pdc .* w
    end
    ExportDecomposition(dva, rdv, fva, pdc, ge, vax, rname, zeros(9), by, sec_types)
end

const _IO_SOURCES = Dict{Symbol,NamedTuple}(
    :oecd => (name="OECD ICIO", needs_credentials=false, versions=["v2016","v2018","v2021","v2023"], note="ICIO tables."),
    :wiod => (name="WIOD 2013", needs_credentials=false, versions=["2013"], note="World IO Database."),
    :exiobase3 => (name="EXIOBASE 3", needs_credentials=false, versions=["3.8.2"], note="Zenodo-hosted."),
    :eora26 => (name="EORA26", needs_credentials=true, versions=["26"], note="Requires worldmrio.com account."),
    :gloria => (name="GLORIA", needs_credentials=false, versions=["053"], note="Fixed URL set."),
)
list_io_sources() = IOSourceTable([(k, _IO_SOURCES[k]) for k in sort(collect(keys(_IO_SOURCES)))])

# Per-source downloaders mirror the REAL restricted signatures (sources.jl): only
# exiobase3 accepts `system`; eora26 accepts neither `system` nor `verify`. This
# lets T1/T2 catch a handler that over-forwards kwargs (real download_io relays
# extras verbatim → MethodError), instead of the mock silently swallowing them.
_io_dl_meta(name, ver, key) = IOMetaData(name, ver, ["mock download of :$key"],
    Pair{String,String}["https://example.org/$(key)_1.zip" => "$(key)_1.zip",
                        "https://example.org/$(key)_2.zip" => "$(key)_2.zip"])
download_oecd(folder; version="v2023", years=nothing, overwrite_existing::Bool=false, verify::Bool=true) =
    _io_dl_meta("OECD ICIO", version, :oecd)
download_wiod(folder; years=nothing, overwrite_existing::Bool=false, verify::Bool=true) =
    _io_dl_meta("WIOD 2013", "2013", :wiod)
download_exiobase3(folder; years=nothing, system::AbstractString="pxp",
                   overwrite_existing::Bool=false, verify::Bool=true) =
    _io_dl_meta("EXIOBASE3", "3.8.2", :exiobase3)
download_eora26(folder; email, password, years=nothing, overwrite_existing::Bool=false) =
    _io_dl_meta("EORA26", "26", :eora26)
download_gloria(folder; years=nothing, overwrite_existing::Bool=false, verify::Bool=true) =
    _io_dl_meta("GLORIA", "053", :gloria)

function download_io(source::Symbol; storage_folder, years=nothing,
                     overwrite_existing::Bool=false, version=nothing,
                     email=nothing, password=nothing, kwargs...)
    if source == :oecd
        download_oecd(storage_folder; version=something(version, "v2023"), years=years,
                      overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :wiod
        download_wiod(storage_folder; years=years, overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :exiobase3
        download_exiobase3(storage_folder; years=years, overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :eora26
        download_eora26(storage_folder; email=something(email, ""), password=something(password, ""),
                        years=years, overwrite_existing=overwrite_existing, kwargs...)
    elseif source == :gloria
        download_gloria(storage_folder; years=years, overwrite_existing=overwrite_existing, kwargs...)
    else
        throw(ArgumentError("unknown source :$source; see list_io_sources()"))
    end
end

# Miller & Blair (2009) 2-sector fixture — mirrors data/wiot.toml.
function _mock_wiot()
    Z = [150.0 500.0; 200.0 100.0]; Y = reshape([350.0, 1700.0], 2, 1)
    va = [300.0 1000.0; 350.0 400.0]
    x = vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2))
    exts = Dict{String,Any}()
    io = IOData{Float64}(Z, Y, va, x, ["Agriculture","Manufacturing"], ["total"],
                         ["final_demand"], ["compensation","other_va"], exts,
                         "millions", 2009,
                         "Miller, R.E. & Blair, P.D. (2009) Input-Output Analysis, 2nd ed., Table 2.3",
                         Dict{String,Any}())
    add_extension!(io, "employment", reshape([30.0, 40.0], 1, 2);
                   stressors=["jobs"], unit=["thousand persons"])
    add_extension!(io, "CO2", reshape([100.0, 300.0], 1, 2);
                   stressors=["CO2"], unit=["kt"])
    io
end

# Mirrors the real parse_io CSV path (parse.jl): reads the file and slices
# raw[1:n_sectors, ...], so an oversized n_sectors raises BoundsError just like
# real _parse_csv_io — keeps the handler's error-mapping honest under T1/T2.
function parse_io(path::AbstractString; source::Symbol=:csv, year=nothing,
                  n_sectors::Int=2, n_fd::Int=1, sectors=String[], delim::AbstractChar=',')
    ext = lowercase(splitext(path)[2])
    ext in (".csv", ".tsv", ".txt") ||
        throw(ArgumentError("unsupported file type '$ext' for parse_io"))
    rows = Vector{Float64}[]
    for l in eachline(path)
        s = strip(l)
        isempty(s) && continue
        push!(rows, [parse(Float64, t) for t in split(s, delim)])
    end
    raw = permutedims(reduce(hcat, rows))            # nrow × ncol
    Z = raw[1:n_sectors, 1:n_sectors]                # BoundsError if n_sectors too large
    Y = raw[1:n_sectors, n_sectors+1:n_sectors+n_fd]
    x = vec(sum(Z, dims=2)) .+ vec(sum(Y, dims=2))
    va = reshape(x .- vec(sum(Z, dims=1)), 1, n_sectors)
    secs = isempty(sectors) ? ["sector$i" for i in 1:n_sectors] : collect(String.(sectors))
    IOData{Float64}(Matrix{Float64}(Z), Matrix{Float64}(Y), Matrix{Float64}(va), x,
                    secs, ["total"], ["fd$j" for j in 1:n_fd], ["va1"],
                    Dict{String,Any}(), "", year === nothing ? 0 : Int(year),
                    string(source), Dict{String,Any}())
end

function BlanchardOLG(; alpha::Real=0.36, beta::Real=0.96, delta::Real=0.08,
                        gamma::Real=0.98, Z::Real=1.0, b::Real=0.0)
    T = Float64
    BlanchardOLG{T}(T(alpha), T(beta), T(delta), T(gamma), T(Z), T(b))
end

function blanchard_steady_state(m::BlanchardOLG{T}; tol::Real=1e-10, max_iter::Int=200) where T
    BlanchardOLGSteadyState{T}(T(5.0), T(1.2), T(0.04), T(1.0), T(10.0),
                               one(T) - m.beta * m.gamma, m.b, true)
end

function blanchard_solve(m::BlanchardOLG{T},
                          ss::BlanchardOLGSteadyState{T}=blanchard_steady_state(m)) where T
    BlanchardOLGSolution{T}(ss, Matrix{T}(I, 2, 2), ComplexF64[0.5, 1.2],
                            T(0.85), T(0.16), true)
end

function blanchard_transition(m::BlanchardOLG{T}, sol::BlanchardOLGSolution{T}, k0::Real;
                               H::Int=50) where T
    kpath = [T(k0) + (sol.ss.k - T(k0)) * (one(T) - sol.stable_eig^t) for t in 0:H]
    Cpath = [sol.ss.C + sol.policy_slope * (k - sol.ss.k) for k in kpath]
    rpath = fill(sol.ss.r, H + 1)
    wpath = fill(sol.ss.w, H + 1)
    return (k=kpath, C=Cpath, r=rpath, w=wpath)
end

function ct_steady_state(m::CTAiyagari{T}; r_bounds=nothing, max_iter::Int=100,
                          tol::Real=1e-6, hjb_max_iter::Int=100, hjb_tol::Real=1e-6,
                          Delta::Real=1000.0) where T
    a = collect(range(m.a_min, m.a_max; length=m.I))
    g = ones(T, m.I, 2) ./ T(2 * m.I)
    v = ones(T, m.I, 2)
    c = ones(T, m.I, 2) * T(0.5)
    s = zeros(T, m.I, 2)
    CTSteadyState{T}(T(0.04), T(1.0), T(3.0), T(1.0), a, g, v, c, s, nothing, true)
end

function ct_mit_shock(m::CTAiyagari{T}, ss0::CTSteadyState{T}, Z_path::AbstractVector;
                       dt::Real=0.25, max_iter::Int=300, tol::Real=1e-6,
                       relax::Real=0.3) where T
    N = length(Z_path)
    t = collect(T, 0:N-1) .* T(dt)
    Z = collect(T, Z_path)
    K = fill(ss0.K, N); K[1] = ss0.K * T(0.98)
    for i in 2:N
        K[i] = ss0.K + (K[1] - ss0.K) * T(0.9)^(i - 1)
    end
    r = fill(ss0.r, N); w = fill(ss0.w, N); C = fill(ss0.K * ss0.r + ss0.w, N)
    CTTransition{T}(t, Z, K, r, w, C, true, 5)
end

function ct_two_asset_solve(m::CTTwoAsset{T}; max_iter::Int=200, tol::Real=1e-6,
                             Delta::Real=1000.0) where T
    b = collect(range(zero(T), m.b_max; length=m.Ib))
    a = collect(range(zero(T), m.a_max; length=m.Ia))
    V = ones(T, m.Ib, m.Ia, 2)
    c = ones(T, m.Ib, m.Ia, 2) * T(0.5)
    d = zeros(T, m.Ib, m.Ia, 2)
    sb = zeros(T, m.Ib, m.Ia, 2)
    sa = zeros(T, m.Ib, m.Ia, 2)
    g = ones(T, m.Ib, m.Ia, 2) ./ T(2 * m.Ib * m.Ia)
    CTTwoAssetSolution{T}(b, a, V, c, d, sb, sa, g, T(1.0), T(5.0), nothing, true)
end

export OrderedLogitModel, OrderedProbitModel, MultinomialLogitModel,
       MultinomialMarginalEffects
export generalized_residuals   # MEMs#507: ordered-model score residual (W4/#87)
export estimate_ologit, estimate_oprobit, estimate_mlogit
export brant_test, hausman_iia, dropna, keeprows

# ── W5–W8 family mocks (MEMs 0.9.0 DCEGM / lifecycle / Khan–Thomas / Bewley banks) ──
struct HAGrid{T<:AbstractFloat}
    grids::Vector{Vector{T}}
    n_points::Vector{Int}
    n_dims::Int
    n_income::Int
    bounds::Vector{Tuple{T,T}}
    labels::Vector{Symbol}
    total_individual_states::Int
end
function HAGrid(grids::Vector{<:AbstractVector{T}}; n_income::Int=1,
                labels::Vector{Symbol}=[:a]) where {T<:AbstractFloat}
    gs = Vector{T}[collect(T, g) for g in grids]
    n_dims = length(gs)
    n_points = Int[length(g) for g in gs]
    bounds = Tuple{T,T}[(minimum(g), maximum(g)) for g in gs]
    labs = length(labels) == n_dims ? labels : Symbol[Symbol("x$i") for i in 1:n_dims]
    HAGrid{T}(gs, n_points, n_dims, n_income, bounds, labs, prod(n_points) * n_income)
end

struct IncomeProcess{T<:AbstractFloat}
    transition::Matrix{T}
    states::Vector{T}
    stationary_dist::Vector{T}
    labels::Symbol
end
function IncomeProcess(n::Int=1; labels::Symbol=:e)
    T = Float64
    P = ones(T, n, n) ./ T(n)
    s = ones(T, n)
    IncomeProcess{T}(P, s, ones(T, n) ./ T(n), labels)
end

struct DCEGMProblem{T<:AbstractFloat}
    beta::T; R::T; utility::Any; utility_prime::Any; utility_prime_inv::Any; income::Any
    options::Vector{Symbol}; absorbing::Vector{Bool}; asset_grid::Vector{T}
    income_process::IncomeProcess{T}; n_periods::Int; taste_shock_scale::T; credit_limit::T
end
struct DCEGMSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    problem::DCEGMProblem{T}
end
struct DCEGMSolution{T<:AbstractFloat}
    M::Array{Vector{T},3}; c::Array{Vector{T},3}; v::Array{Vector{T},3}
    ev_constrained::Array{T,3}; n_kinks::Array{Int,3}; prob::DCEGMProblem{T}
    n_periods::Int; converged::Bool; iterations::Int; sup_diff::T
end
struct DCEGMDistribution{T<:AbstractFloat}
    grid::Vector{T}; dist::Array{T,4}; shares::Matrix{T}
    consumption::Vector{T}; assets::Vector{T}; n_periods::Int
end
struct DCEGMFirm{T<:AbstractFloat}
    alpha::T; delta::T; Z::T; L::T
end
DCEGMFirm(; alpha::Real=0.36, delta::Real=0.08, Z::Real=1.0, L::Real=1.0) =
    DCEGMFirm{Float64}(Float64(alpha), Float64(delta), Float64(Z), Float64(L))
struct DCEGMEquilibrium{T<:AbstractFloat}
    r::T; w::T; K::T; L::T; Y::T; K_demand::T; excess_demand::T
    solution::DCEGMSolution{T}; distribution::DCEGMDistribution{T}; firm::DCEGMFirm{T}
    converged::Bool; iterations::Int
end
struct DCEGMTransition{T<:AbstractFloat}
    Z::Vector{T}; K::Vector{T}; r::Vector{T}; w::Vector{T}; A::Vector{T}; Y::Vector{T}
    equilibrium::DCEGMEquilibrium{T}; method::Symbol; converged::Bool
end

function _mock_dcegm_prob(; n_periods=20, beta=0.98, R=1.0, n_a=8)
    T = Float64
    ag = collect(range(zero(T), T(10); length=max(n_a, 2)))
    DCEGMProblem{T}(T(beta), T(R), log, inv, inv, identity,
                    [:work, :retire], [false, true], ag, IncomeProcess(1),
                    n_periods, zero(T), zero(T))
end
dcegm_retirement_model(; n_periods::Int=20, beta::Real=0.98, R::Real=1.0, n_a::Int=8,
                       wage=20.0, disutility=1.0, sigma=0.0, n_shocks=1,
                       taste_shock_scale=0.0, a_max=50.0, pension=0.0,
                       credit_limit=0.0, curvature=2.0) =
    _mock_dcegm_prob(; n_periods=n_periods, beta=beta, R=R, n_a=n_a)

function dcegm_solve(prob::DCEGMProblem{T}; max_iter::Int=500, tol::Real=1e-8) where T
    n_t = max(prob.n_periods, 1); n_d = length(prob.options); n_e = 1
    knots = T[T(0), T(1), T(2)]
    M = Array{Vector{T},3}(undef, n_t, n_d, n_e)
    c = similar(M); v = similar(M)
    for t in 1:n_t, d in 1:n_d, j in 1:n_e
        M[t, d, j] = knots; c[t, d, j] = knots ./ T(2); v[t, d, j] = log.(knots .+ one(T))
    end
    DCEGMSolution{T}(M, c, v, zeros(T, n_t, n_d, n_e), zeros(Int, n_t, n_d, n_e),
                     prob, n_t, true, 2, T(1e-10))
end
function dcegm_steady_state(source, firm=DCEGMFirm(); r_bounds=(0.001, 0.20),
                             labor::Symbol=:exogenous, reprice_wage::Bool=false,
                             work_option::Symbol=:work, n_sim::Int=40,
                             tol::Real=1e-4, max_iter::Int=40)
    prob = source isa DCEGMProblem ? source : _mock_dcegm_prob()
    sol = dcegm_solve(prob)
    T = typeof(sol.sup_diff)
    dist = DCEGMDistribution{T}(prob.asset_grid, zeros(T, 2, 2, 2, 2),
                                ones(T, 2, 2) ./ T(4), T[1, 1], T[1, 1], sol.n_periods)
    DCEGMEquilibrium{T}(T(0.04), T(1.0), T(3.0), T(1.0), T(1.2), T(3.0), T(1e-8),
                        sol, dist, firm isa DCEGMFirm ? firm : DCEGMFirm(), true, 4)
end
function dcegm_mit(eq::DCEGMEquilibrium{T}, Z_path::AbstractVector) where T
    n = length(Z_path)
    DCEGMTransition{T}(collect(T, Z_path), fill(eq.K, n), fill(eq.r, n), fill(eq.w, n),
                       fill(eq.K, n), fill(eq.Y, n), eq, :mit, true)
end
function irf(eq::DCEGMEquilibrium{T}, horizon::Int; shock_size::Real=0.01, persist::Real=0.0) where T
    vars = ["K", "r", "w", "Y", "Z"]
    vals = zeros(T, horizon, length(vars), 1)
    for h in 1:horizon; vals[h, :, 1] .= T(0.01) * T(0.9)^(h - 1); end
    ImpulseResponse(vals, nothing, nothing, horizon, vars, ["tfp"], :dcegm)
end
function fevd(eq::DCEGMEquilibrium{T}, horizon::Int; shock_size::Real=0.01, persist::Real=0.0) where T
    n = 5; props = ones(T, n, 1, horizon)
    FEVD(copy(props), props, ["K", "r", "w", "Y", "Z"], ["tfp"])
end
function simulate(eq::DCEGMEquilibrium{T}, periods::Int; shock_size::Real=0.0, persist::Real=0.0) where T
    hcat(fill(eq.K, periods), fill(eq.r, periods), fill(eq.w, periods),
         fill(eq.Y, periods), fill(eq.firm.Z, periods))
end

struct LifeCycleOLG{T<:AbstractFloat}
    J::Int; J_retire::Int; survival::Vector{T}; earnings::Vector{T}
    income::IncomeProcess{T}; grid::HAGrid{T}; beta::T; sigma::T
    alpha::T; delta::T; Z::T; n_pop::T; replacement::T; credit_limit::T; annuities::Bool
end
struct LifeCycleSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    model::LifeCycleOLG{T}
end
function LifeCycleOLG(; J::Int=12, J_retire::Int=9, survival=0.99, earnings=nothing,
                       income=IncomeProcess(3), a_max=40.0, n_a::Int=8,
                       beta=0.97, sigma=2.0, alpha=0.36, delta=0.06, Z=1.0,
                       n_pop=0.0, replacement=0.4, credit_limit=0.0, annuities::Bool=true)
    T = Float64
    surv = survival isa AbstractVector ? T.(survival) : fill(T(survival), J)
    earn = earnings === nothing ? ones(T, J) : T.(earnings)
    grid = HAGrid([collect(range(T(credit_limit), T(a_max); length=n_a))]; n_income=length(income.states))
    LifeCycleOLG{T}(J, J_retire, surv, earn, income, grid, T(beta), T(sigma),
                    T(alpha), T(delta), T(Z), T(n_pop), T(replacement), T(credit_limit), annuities)
end
lifecycle_income(rho, sigma, n) = IncomeProcess(n; labels=:e)
struct LifeCycleSteadyState{T<:AbstractFloat}
    r::T; w::T; K::T; L::T; Y::T; tau::T; pension::T; transfer::T
    c_policy::Array{T,3}; a_policy::Array{T,3}; dist::Array{T,3}
    cohort_mass::Vector{T}; asset_profile::Vector{T}; consumption_profile::Vector{T}
    income_profile::Vector{T}; converged::Bool; iterations::Int; excess_demand::T
    spec::LifeCycleOLG{T}
end
struct LifeCycleTransition{T<:AbstractFloat}
    K::Vector{T}; r::Vector{T}; w::Vector{T}; Y::Vector{T}; C::Vector{T}; Z::Vector{T}
    pension::Vector{T}; transfer::Vector{T}; tau::T; converged::Bool; iterations::Int
    ss::LifeCycleSteadyState{T}
end
function lifecycle_steady_state(m::LifeCycleOLG{T}; r_bounds=(-0.02, 0.10),
                                  tol::Real=1e-6, max_iter::Int=60, bequest_iter::Int=50) where T
    J = m.J
    mu = ones(T, J) ./ T(J)
    LifeCycleSteadyState{T}(T(0.04), T(1.0), T(3.0), T(1.0), T(1.2), T(0.1), T(0.2), T(0.0),
                            zeros(T, 3, 2, J), zeros(T, 3, 2, J), zeros(T, 3, 2, J),
                            mu, ones(T, J), ones(T, J), ones(T, J), true, 6, T(1e-8), m)
end
function lifecycle_transition(m::LifeCycleOLG{T}, arg; H::Int=20,
                                tol::Real=1e-5, max_iter::Int=80, relax::Real=0.5) where T
    ss = lifecycle_steady_state(m)
    n = arg isa AbstractVector ? length(arg) : H + 1
    Z = arg isa AbstractVector ? collect(T, arg) : fill(m.Z, n)
    LifeCycleTransition{T}(fill(ss.K, n), fill(ss.r, n), fill(ss.w, n), fill(ss.Y, n),
                           fill(ss.K, n), Z, fill(ss.pension, n), fill(ss.transfer, n),
                           ss.tau, true, 3, ss)
end
function irf(ss::LifeCycleSteadyState{T}, horizon::Int; shock_size::Real=0.01, persist::Real=0.0) where T
    vars = ["K", "r", "w", "Y", "Z"]
    vals = zeros(T, horizon, length(vars), 1)
    for h in 1:horizon; vals[h, :, 1] .= T(0.01) * T(0.9)^(h - 1); end
    ImpulseResponse(vals, nothing, nothing, horizon, vars, ["tfp"], :lifecycle)
end
function fevd(ss::LifeCycleSteadyState{T}, horizon::Int; shock_size::Real=0.01, persist::Real=0.0) where T
    n = 5; props = ones(T, n, 1, horizon)
    FEVD(copy(props), props, ["K", "r", "w", "Y", "Z"], ["tfp"])
end
function simulate(ss::LifeCycleSteadyState{T}, periods::Int; shock_size::Real=0.0, persist::Real=0.0) where T
    hcat(fill(ss.K, periods), fill(ss.r, periods), fill(ss.w, periods),
         fill(ss.Y, periods), fill(ss.spec.Z, periods))
end

struct FirmSystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    k_grid::Vector{T}; productivity::IncomeProcess{T}
    alpha::T; nu::T; delta::T; beta::T; gamma::T; xi_bar::T; b::T; phi::T
    rho_z::T; sigma_z::T; Z::T
end
function khan_thomas_example(; n_k::Int=8, n_eps::Int=2, alpha=0.256, nu=0.640,
                              delta=0.069, beta=0.977, gamma=1.016, xi_bar=0.0083,
                              b=0.011, phi=2.4, rho_z=0.859, sigma_z=0.014,
                              rho_e=0.859, sigma_e=0.022, Z=1.0)
    T = Float64
    kg = collect(range(T(0.1), T(2.0); length=max(n_k, 3)))
    FirmSystem{T}(kg, IncomeProcess(n_eps), T(alpha), T(nu), T(delta), T(beta), T(gamma),
                  T(xi_bar), T(b), T(phi), T(rho_z), T(sigma_z), T(Z))
end
struct KhanThomasSteadyState{T<:AbstractFloat}
    firm::FirmSystem{T}; w::T; p::T; K::T; N::T; Y::T; I::T; C::T; inaction::T
    distribution::Matrix{T}; value::Matrix{T}; k_star::Vector{T}
    k_constrained::Matrix{T}; adj_prob::Matrix{T}; labor::Matrix{T}
    converged::Bool; iterations::Int; method::Symbol
end
struct KhanThomasTransition{T<:AbstractFloat}
    Z::Vector{T}; Y::Vector{T}; I::Vector{T}; K::Vector{T}; N::Vector{T}; C::Vector{T}
    w::Vector{T}; ss::KhanThomasSteadyState{T}; method::Symbol; converged::Bool
end
function khan_thomas_steady_state(fs::FirmSystem{T}; tol::Real=1e-5, max_iter::Int=16) where T
    n_k = length(fs.k_grid); n_e = length(fs.productivity.states)
    KhanThomasSteadyState{T}(fs, T(1.0), T(1.0), T(3.0), T(0.3), T(1.0), T(0.2), T(0.8),
                             T(0.5), ones(T, n_k, n_e) ./ T(n_k * n_e), ones(T, n_k, n_e),
                             fill(T(1.0), n_e), ones(T, n_k, n_e), fill(T(0.4), n_k, n_e),
                             ones(T, n_k, n_e), true, 4, :ge)
end
function khan_thomas_mit(ss::KhanThomasSteadyState{T}, Z_path::AbstractVector; prices::Symbol=:ss) where T
    n = length(Z_path)
    KhanThomasTransition{T}(collect(T, Z_path), fill(ss.Y, n), fill(ss.I, n), fill(ss.K, n),
                            fill(ss.N, n), fill(ss.C, n), fill(ss.w, n), ss, :mit, true)
end
function irf(ss::KhanThomasSteadyState{T}, horizon::Int; shock_size::Real=0.01,
              persist::Real=0.5, prices::Symbol=:ss) where T
    vars = ["Y", "I", "K", "N", "C", "Z"]
    vals = zeros(T, horizon, length(vars), 1)
    for h in 1:horizon; vals[h, :, 1] .= T(0.01) * T(0.9)^(h - 1); end
    ImpulseResponse(vals, nothing, nothing, horizon, vars, ["tfp"], :khan_thomas)
end

struct IntermediarySystem{T<:AbstractFloat} <: AbstractAgentSystem{T}
    grid::HAGrid{T}; xi::IncomeProcess{T}; kappa::T; beta::T; sigma::T; lambda::T
    zeta1::T; zeta2::T; R::T; rk::T; Z::T; alpha::T; n_enter::T
    het_params::Dict{Symbol,T}; aggregation::Vector{Pair{Symbol,Function}}
    model::Symbol; distribution::Symbol
end
function IntermediarySystem(; n_n::Int=8, n_xi::Int=2, n_min=0.05, n_max=8.0,
                             beta=0.99, sigma=0.95, lambda=0.20, zeta1=0.02, zeta2=2.0,
                             R=1.01, rk=0.05, Z=0.25, alpha=0.33)
    T = Float64
    g = HAGrid([collect(range(T(n_min), T(n_max); length=max(n_n, 3)))]; n_income=n_xi, labels=[:n])
    IntermediarySystem{T}(g, IncomeProcess(n_xi; labels=:xi), T(0.0), T(beta), T(sigma), T(lambda),
                          T(zeta1), T(zeta2), T(R), T(rk), T(Z), T(alpha), T(0.0),
                          Dict{Symbol,T}(), Pair{Symbol,Function}[], :bewley, :young)
end
struct IntermediaryPE{T<:AbstractFloat}
    V::Matrix{T}; l_policy::Matrix{T}; b_policy::Matrix{T}
    prices::Dict{Symbol,T}; converged::Bool; iterations::Int
end
struct IntermediarySteadyState{T<:AbstractFloat}
    system::IntermediarySystem{T}; V::Matrix{T}; l_policy::Matrix{T}; b_policy::Matrix{T}
    distribution::Matrix{T}; prices::Dict{Symbol,T}; aggregates::Dict{Symbol,T}
    grid::HAGrid{T}; xi::IncomeProcess{T}; converged::Bool; iterations::Int; excess_demand::T
end
struct IntermediaryTransition{T<:AbstractFloat}
    Z::Vector{T}; L::Vector{T}; Y::Vector{T}; K::Vector{T}; rk::Vector{T}
    ss::IntermediarySteadyState{T}; method::Symbol; converged::Bool
end
function intermediary_pe(sys::IntermediarySystem{T}; R=sys.R, rk=sys.rk, max_iter=250, tol=1e-6) where T
    n_n = length(sys.grid.grids[1]); n_e = length(sys.xi.states)
    IntermediaryPE{T}(ones(T, n_n, n_e), ones(T, n_n, n_e), ones(T, n_n, n_e),
                      Dict(:R => T(R), :rk => T(rk)), true, 3)
end
function intermediary_steady_state(sys::IntermediarySystem{T}; r_bounds=nothing, max_iter=24, tol=1e-4) where T
    lo, hi = r_bounds === nothing ? (T(0.01), T(0.5)) : (T(r_bounds[1]), T(r_bounds[2]))
    hi > lo || throw(ArgumentError("intermediary_steady_state: r_bounds must satisfy lo < hi"))
    # A bracket that cannot contain a typical credit-market root reports honest
    # non-convergence (mirrors real's no-sign-change path).
    conv = lo < T(0.3) && hi > T(0.01)
    n_n = length(sys.grid.grids[1]); n_e = length(sys.xi.states)
    IntermediarySteadyState{T}(sys, ones(T, n_n, n_e), ones(T, n_n, n_e), ones(T, n_n, n_e),
                               ones(T, n_n, n_e) ./ T(n_n * n_e),
                               Dict(:R => sys.R, :rk => sys.rk),
                               Dict(:L => T(1.0), :N => T(1.0), :B => T(0.5),
                                    :leverage => T(2.0), :Y => T(0.4)),
                               sys.grid, sys.xi, conv, conv ? 4 : max_iter,
                               conv ? T(1e-8) : T(1.0))
end
function intermediary_mit(ss::IntermediarySteadyState{T}, Z_path::AbstractVector) where T
    n = length(Z_path)
    IntermediaryTransition{T}(collect(T, Z_path), fill(ss.aggregates[:L], n),
                              fill(ss.aggregates[:Y], n), fill(ss.aggregates[:L], n),
                              fill(ss.prices[:rk], n), ss, :mit, true)
end
function irf(ss::IntermediarySteadyState{T}, horizon::Int; shock_size::Real=0.01, persist::Real=0.5) where T
    vars = ["L", "Y", "K", "rk", "Z"]
    vals = zeros(T, horizon, length(vars), 1)
    for h in 1:horizon; vals[h, :, 1] .= T(0.01) * T(0.9)^(h - 1); end
    ImpulseResponse(vals, nothing, nothing, horizon, vars, ["tfp"], :bank)
end

export HASteadyState, HADSGESolution, KrusellSmithSolution
export DenHaanAccuracy, den_haan_test
export CTAiyagari, CTSteadyState, CTTransition, CTTwoAsset, CTTwoAssetSolution, CTPoissonIncome
export BlanchardOLG, BlanchardOLGSteadyState, BlanchardOLGSolution
export X13FilterResult, IOData
export load_ha_example, compute_steady_state, distribution_irf, inequality_irf, simulate_panel
export ct_steady_state, ct_mit_shock, ct_two_asset_solve
export x13_filter, parse_io, parse_icio, region_indices, blanchard_steady_state, blanchard_solve, blanchard_transition
export HAGrid, IncomeProcess
export DCEGMProblem, DCEGMSystem, DCEGMSolution, DCEGMFirm, DCEGMEquilibrium, DCEGMTransition, DCEGMDistribution
export dcegm_retirement_model, dcegm_solve, dcegm_steady_state, dcegm_mit
export LifeCycleOLG, LifeCycleSystem, LifeCycleSteadyState, LifeCycleTransition
export lifecycle_income, lifecycle_steady_state, lifecycle_transition
export FirmSystem, KhanThomasSteadyState, KhanThomasTransition
export khan_thomas_example, khan_thomas_steady_state, khan_thomas_mit
export IntermediarySystem, IntermediaryPE, IntermediarySteadyState, IntermediaryTransition
export intermediary_pe, intermediary_steady_state, intermediary_mit
export RegionalFootprintResult
# Input-Output analysis (C049)
export IOExtension, IOMetaData, LeontiefModel, GhoshModel, IOMultipliers, LinkageResult
export SDAResult, ExtractionResult, FootprintResult, BaqaeeFarhiResult, IOSourceTable
export technical_coefficients, leontief_inverse, allocation_coefficients, ghosh_inverse
export leontief, ghosh, multipliers, linkages, rasmussen, key_sectors
export sda, hypothetical_extraction, add_extension!, intensities, emission_multipliers, footprint
export domar_weights, baqaee_farhi, list_io_sources, download_io
export download_oecd, download_wiod, download_exiobase3, download_eora26, download_gloria
export ProductionNetwork, production_network
export BFEquilibrium, bf_equilibrium
export BFLocal, BFElasticities, BFShockCurve, BFWedgeDecomp, BFMisallocation
export bf_elasticities, bf_shock_curve, bf_wedge_decomp, bf_misallocation
export PriceModelResult, price_model, ImpactResult, impact
export NetworkStatsResult, network_stats
export aggregate, balance, bilateral_trade, gross_exports
export VerticalSpecialization, vertical_specialization
export ExportDecomposition, export_decomposition

# ─── C062b: single-equation ARDL / NARDL (EV-08/09) ─────────────────────────
# Mirror the real MEMs 0.7.0 ARDL field NAMES (a faithful subset is fine — check_mock_surface
# is mock ⊆ real). Estimators are GENUINE ARDL OLS-on-lagged-levels fits that validate like the
# real ones (case∈1:5, ic∈{aic,bic}, q-length==k, empty-asym ArgumentError, y/X shape) so T1/T2
# exercise the error mapping AND the hand-built coef/long-run/bounds/multiplier renderers.
# Struct order matters (flat include): ARDLLongRun before ARDLModel (its `longrun` field);
# ARDLModel + ARDLBoundsTest before NARDLModel (which embeds both).

struct ARDLLongRun{T<:AbstractFloat}
    theta::Vector{T}
    se::Vector{T}
    denom::T
    varnames::Vector{String}
end

struct ARDLModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    coef::Vector{T}
    vcov::Matrix{T}
    residuals::Vector{T}
    fitted::Vector{T}
    p::Int
    q::Vector{Int}
    case::Int
    trend::Symbol
    ssr::T
    sigma2::T
    loglik::T
    aic::T
    bic::T
    n::Int
    K::Int
    ar_idx::Vector{Int}
    x_idx::Vector{Vector{Int}}
    coefnames::Vector{String}
    xnames::Vector{String}
    yname::String
    selected::Bool
    ic::Symbol
    longrun::ARDLLongRun{T}
end

struct ARDLBoundsTest{T<:AbstractFloat}
    fstat::T
    tstat::T
    k::Int
    case::Int
    cv_source::Symbol
    levels::Vector{T}
    f_lower::Vector{T}
    f_upper::Vector{T}
    t_lower::Vector{T}
    t_upper::Vector{T}
    level::T
    f_decision::Symbol
    t_decision::Symbol
    n::Int
end

struct NARDLModel{T<:AbstractFloat}
    ardl::ARDLModel{T}
    bounds::ARDLBoundsTest{T}
    y::Vector{T}
    X::Matrix{T}
    Xsplit::Matrix{T}
    asym::Vector{Int}
    meta::Vector{Tuple{Int,Symbol}}
    k_orig::Int
    k::Int
    xnames::Vector{String}
    enames::Vector{String}
    yname::String
end

struct NARDLSymmetryTest{T<:AbstractFloat}
    reg_index::Vector{Int}
    reg_names::Vector{String}
    lr_stat::Vector{T}
    lr_p_chi2::Vector{T}
    lr_p_f::Vector{T}
    sr_stat::Vector{T}
    sr_p_chi2::Vector{T}
    sr_p_f::Vector{T}
    theta_pos::Vector{T}
    theta_neg::Vector{T}
    df::Int
    dof_resid::Int
end

struct NARDLMultipliers{T<:AbstractFloat}
    horizons::Vector{Int}
    reg_index::Vector{Int}
    reg_names::Vector{String}
    m_pos::Matrix{T}
    m_neg::Matrix{T}
    m_diff::Matrix{T}
    m_pos_lo::Matrix{T}
    m_pos_hi::Matrix{T}
    m_neg_lo::Matrix{T}
    m_neg_hi::Matrix{T}
    m_diff_lo::Matrix{T}
    m_diff_hi::Matrix{T}
    theta_pos::Vector{T}
    theta_neg::Vector{T}
    nreps::Int
    level::T
end

coef(m::ARDLModel) = m.coef
vcov(m::ARDLModel) = m.vcov
stderror(m::ARDLModel) = [sqrt(max(m.vcov[i, i], 0.0)) for i in 1:length(m.coef)]
nobs(m::ARDLModel) = m.n
coef(m::NARDLModel) = m.ardl.coef
stderror(m::NARDLModel) = stderror(m.ardl)

# χ²(1) survival function proxy p = P(χ²₁ > s) = 2·(1−Φ(√s)); finite, monotone, in (0,1) —
# enough for a mock p-value (tests assert finiteness/shape, not magnitude).
_mock_chisq1_sf(s::Real) = 2.0 * _mock_norm_sf(sqrt(max(Float64(s), 0.0)))

# Genuine levels-ARDL design [det; y_{t-1..p}; x_j lags], mirroring real `_ardl_design`.
function _mock_ardl_design(y::Vector{T}, X0::Matrix{T}, p::Int, q::Vector{Int}, case::Int,
                           xnames::Vector{String}, yname::String) where {T}
    N = size(X0, 1); k = size(X0, 2)
    L = max(p, maximum(q)); rows = (L + 1):N; n = length(rows)
    cols = Vector{Vector{T}}(); names = String[]
    trend = case == 1 ? :none : (case in (2, 3) ? :const : :trend)
    if trend == :const || trend == :trend
        push!(cols, ones(T, n)); push!(names, "(Intercept)")
    end
    if trend == :trend
        push!(cols, T.(collect(rows))); push!(names, "trend")
    end
    ar_idx = Int[]
    for i in 1:p
        push!(cols, T[y[t-i] for t in rows]); push!(names, "L$i.$yname"); push!(ar_idx, length(cols))
    end
    x_idx = Vector{Vector{Int}}(undef, k)
    for j in 1:k
        idxj = Int[]
        for l in 0:q[j]
            push!(cols, T[X0[t-l, j] for t in rows])
            push!(names, l == 0 ? xnames[j] : "L$l.$(xnames[j])")
            push!(idxj, length(cols))
        end
        x_idx[j] = idxj
    end
    (reduce(hcat, cols), T[y[t] for t in rows], ar_idx, x_idx, names, trend)
end

function estimate_ardl(y::AbstractVector, X::AbstractVecOrMat;
                       p::Union{Symbol,Integer}=:auto,
                       q::Union{Symbol,Integer,AbstractVector}=:auto,
                       max_p::Int=4, max_q::Int=4, ic::Symbol=:aic, case::Int=3,
                       trend::Symbol=:none, xnames=nothing, yname::AbstractString="y")
    T = Float64
    yv = collect(T, y)
    X0 = X isa AbstractVector ? reshape(collect(T, X), :, 1) : Matrix{T}(X)
    N, k = size(X0)
    length(yv) == N || throw(DimensionMismatch("y has length $(length(yv)); X has $N rows"))
    (1 <= case <= 5) || throw(ArgumentError("case must be in 1:5; got $case"))
    ic in (:aic, :bic) || throw(ArgumentError("ic must be :aic or :bic; got :$ic"))
    vnames = xnames === nothing ? ["x$j" for j in 1:k] : collect(String, xnames)
    length(vnames) == k || throw(ArgumentError("xnames must have length $k"))
    yn = String(yname)
    selected = (p === :auto) || (q === :auto)
    pp = p === :auto ? 1 : Int(p)          # mock: no grid search — fix :auto at 1
    pp >= 1 || throw(ArgumentError("p must be ≥ 1; got $pp"))
    qq = q === :auto ? fill(1, k) : (q isa Integer ? fill(Int(q), k) : collect(Int, q))
    length(qq) == k || throw(ArgumentError("q must have length $k; got $(length(qq))"))
    all(>=(0), qq) || throw(ArgumentError("every q must be ≥ 0"))
    L = max(pp, maximum(qq))
    N > L || throw(ArgumentError("effective sample empty: need N > $L"))
    Xd, yeff, ar_idx, x_idx, names, tr = _mock_ardl_design(yv, X0, pp, qq, case, vnames, yn)
    n, K = size(Xd)
    n > K || throw(ArgumentError("effective sample ($n) ≤ #coefficients ($K); reduce lags"))
    XtX = Xd'Xd
    beta = XtX \ (Xd'yeff)
    fitted = Xd * beta
    resid = yeff .- fitted
    ssr = sum(abs2, resid)
    sigma2 = ssr / max(n - K, 1)
    vcovm = sigma2 .* (XtX \ Matrix{T}(I(K)))
    ll = -T(n) / 2 * (log(2π) + log(max(ssr / n, eps())) + 1)
    aic = -2ll + 2 * (K + 1); bic = -2ll + log(T(n)) * (K + 1)
    denom = one(T) - sum(beta[ar_idx])
    theta = zeros(T, k); se = zeros(T, k)
    for j in 1:k
        num = sum(beta[x_idx[j]])
        theta[j] = num / denom
        g = zeros(T, K)
        for c in x_idx[j]; g[c] = one(T) / denom; end
        for c in ar_idx;   g[c] = num / denom^2; end
        se[j] = sqrt(max(sum(g .* (vcovm * g)), zero(T)))
    end
    lr = ARDLLongRun{T}(theta, se, denom, copy(vnames))
    ARDLModel{T}(yeff, Xd, beta, vcovm, resid, fitted, pp, qq, case, tr, ssr, sigma2,
                 ll, aic, bic, n, K, ar_idx, x_idx, names, vnames, yn, selected, ic, lr)
end

long_run(m::ARDLModel) = m.longrun

function ecm_form(m::ARDLModel{T}) where {T}
    r = zeros(T, m.K)
    for c in m.ar_idx; r[c] = one(T); end
    alpha = sum(r .* m.coef) - one(T)
    alpha_se = sqrt(max(sum(r .* (m.vcov * r)), zero(T)))
    (alpha=alpha, alpha_se=alpha_se, alpha_t=alpha / alpha_se, longrun=m.longrun)
end

const _MOCK_PSS_LEVELS = [0.10, 0.05, 0.025, 0.01]

function bounds_test(m::ARDLModel{T}; case::Int=m.case, level::Real=0.05,
                     cv_source::Symbol=:pss) where {T}
    cv_source == :narayan &&
        throw(ArgumentError("cv_source=:narayan finite-sample bounds are not bundled; use :pss"))
    cv_source == :pss || throw(ArgumentError("cv_source must be :pss; got :$cv_source"))
    (1 <= case <= 5) || throw(ArgumentError("case must be in 1:5; got $case"))
    li = findfirst(x -> isapprox(x, level), _MOCK_PSS_LEVELS)
    li === nothing && throw(ArgumentError("level must be one of $_MOCK_PSS_LEVELS; got $level"))
    k = length(m.q)
    b = m.coef; V = m.vcov
    lvl_cols = vcat(m.ar_idx, reduce(vcat, m.x_idx))
    zs = T[V[c, c] > 0 ? b[c] / sqrt(V[c, c]) : zero(T) for c in lvl_cols]
    fstat = sum(abs2, zs) / length(zs)
    ecm = ecm_form(m)
    tstat = ecm.alpha_t
    f_lower = T[4.04, 4.94, 5.77, 6.84]        # PSS case III k=1 canonical (mock: fixed table)
    f_upper = T[4.78, 5.73, 6.68, 7.84]
    if case in (2, 4)
        t_lower = fill(T(NaN), 4); t_upper = fill(T(NaN), 4)
    else
        t_lower = T[-2.57, -2.86, -3.13, -3.43]
        t_upper = T[-2.91, -3.22, -3.50, -3.82]
    end
    f_dec = fstat > f_upper[li] ? :cointegrated : fstat < f_lower[li] ? :not_cointegrated : :inconclusive
    t_dec = isnan(t_lower[li]) ? :undefined :
            (tstat < t_upper[li] ? :cointegrated : tstat > t_lower[li] ? :not_cointegrated : :inconclusive)
    ARDLBoundsTest{T}(T(fstat), T(tstat), k, case, cv_source, T.(_MOCK_PSS_LEVELS),
                     f_lower, f_upper, t_lower, t_upper, T(level), f_dec, t_dec, m.n)
end

function _mock_partial_sums(x::Vector{T}) where {T}
    N = length(x); xp = zeros(T, N); xn = zeros(T, N)
    for t in 2:N
        dx = x[t] - x[t-1]
        xp[t] = xp[t-1] + max(dx, zero(T))
        xn[t] = xn[t-1] + min(dx, zero(T))
    end
    (xp, xn)
end

function estimate_nardl(y::AbstractVector, X::AbstractVecOrMat;
                        asymmetric::Union{Symbol,AbstractVector{<:Integer}}=:all,
                        p::Union{Symbol,Integer}=:auto,
                        q::Union{Symbol,Integer,AbstractVector}=:auto,
                        max_p::Int=4, max_q::Int=4, ic::Symbol=:aic, case::Int=3,
                        xnames=nothing, yname::AbstractString="y")
    T = Float64
    yv = collect(T, y)
    X0 = X isa AbstractVector ? reshape(collect(T, X), :, 1) : Matrix{T}(X)
    N, k0 = size(X0)
    length(yv) == N || throw(DimensionMismatch("y has length $(length(yv)); X has $N rows"))
    vnames = xnames === nothing ? ["x$j" for j in 1:k0] : collect(String, xnames)
    length(vnames) == k0 || throw(ArgumentError("xnames must have length $k0"))
    if asymmetric === :all
        asym = collect(1:k0)
    else
        asym = sort(unique(collect(Int, asymmetric)))
        all(j -> 1 <= j <= k0, asym) || throw(ArgumentError("asymmetric indices must be in 1:$k0; got $asym"))
    end
    isempty(asym) &&
        throw(ArgumentError("NARDL needs at least one asymmetric regressor; use estimate_ardl"))
    cols = Vector{Vector{T}}(); meta = Tuple{Int,Symbol}[]; enames = String[]
    for j in 1:k0
        if j in asym
            xp, xn = _mock_partial_sums(X0[:, j])
            push!(cols, xp); push!(meta, (j, :pos)); push!(enames, vnames[j] * "_POS")
            push!(cols, xn); push!(meta, (j, :neg)); push!(enames, vnames[j] * "_NEG")
        else
            push!(cols, X0[:, j]); push!(meta, (j, :sym)); push!(enames, vnames[j])
        end
    end
    Xsplit = reduce(hcat, cols)
    kk = size(Xsplit, 2)
    ardl = estimate_ardl(yv, Xsplit; p=p, q=q, max_p=max_p, max_q=max_q, ic=ic, case=case,
                         xnames=enames, yname=String(yname))
    bt = bounds_test(ardl; case=case)
    NARDLModel{T}(ardl, bt, yv, X0, Xsplit, asym, meta, k0, kk, vnames, enames, String(yname))
end

long_run(m::NARDLModel) = m.ardl.longrun
bounds_test(m::NARDLModel) = m.bounds

function _mock_enlarged_index(m::NARDLModel, orig::Int, kind::Symbol)
    for (e, (o, kd)) in enumerate(m.meta)
        (o == orig && kd == kind) && return e
    end
    0
end

function symmetry_test(m::NARDLModel{T}) where {T}
    a = m.ardl; asym = m.asym; na = length(asym)
    denom = one(T) - sum(a.coef[a.ar_idx])
    lr_stat = zeros(T, na); lr_pc = zeros(T, na); lr_pf = zeros(T, na)
    sr_stat = zeros(T, na); sr_pc = zeros(T, na); sr_pf = zeros(T, na)
    tp = zeros(T, na); tn = zeros(T, na); names = String[]
    dof_r = max(a.n - a.K, 1)
    for (i, orig) in enumerate(asym)
        push!(names, m.xnames[orig])
        ep = _mock_enlarged_index(m, orig, :pos); en = _mock_enlarged_index(m, orig, :neg)
        Sp = sum(a.coef[a.x_idx[ep]]); Sn = sum(a.coef[a.x_idx[en]])
        thp = Sp / denom; thn = Sn / denom
        tp[i] = thp; tn[i] = thn
        diff = thp - thn
        var = 0.05 + abs(diff) * 0.01
        s = diff^2 / var
        lr_stat[i] = s; lr_pc[i] = _mock_chisq1_sf(s); lr_pf[i] = _mock_chisq1_sf(s)
        ss = 0.5 * s
        sr_stat[i] = ss; sr_pc[i] = _mock_chisq1_sf(ss); sr_pf[i] = _mock_chisq1_sf(ss)
    end
    NARDLSymmetryTest{T}(copy(asym), names, lr_stat, lr_pc, lr_pf, sr_stat, sr_pc, sr_pf,
                        tp, tn, 1, dof_r)
end

function _mock_iterate_multiplier(phi::Vector{T}, beta::Vector{T}, H::Int) where {T}
    p = length(phi); q = length(beta) - 1; g = zeros(T, H + 1)
    for h in 0:H
        val = zero(T)
        for i in 1:p; (h - i) >= 0 && (val += phi[i] * g[h-i+1]); end
        for l in 0:q; h >= l && (val += beta[l+1]); end
        g[h+1] = val
    end
    g
end

function dynamic_multipliers(m::NARDLModel{T}, H::Int; bootstrap::Bool=true, nreps::Int=500,
                             level::Real=0.95, rng::AbstractRNG=Random.default_rng(),
                             seed=nothing) where {T}
    H >= 0 || throw(ArgumentError("H must be ≥ 0; got $H"))
    a = m.ardl; na = length(m.asym); Hp1 = H + 1
    m_pos = zeros(T, na, Hp1); m_neg = zeros(T, na, Hp1); m_dif = zeros(T, na, Hp1)
    tp = zeros(T, na); tn = zeros(T, na)
    lr = a.longrun
    phi = T[a.coef[c] for c in a.ar_idx]
    for (i, orig) in enumerate(m.asym)
        ep = _mock_enlarged_index(m, orig, :pos); en = _mock_enlarged_index(m, orig, :neg)
        bp = T[a.coef[c] for c in a.x_idx[ep]]; bn = T[a.coef[c] for c in a.x_idx[en]]
        gp = _mock_iterate_multiplier(phi, bp, H); gn = _mock_iterate_multiplier(phi, bn, H)
        m_pos[i, :] .= gp; m_neg[i, :] .= gn; m_dif[i, :] .= gp .- gn
        tp[i] = lr.theta[ep]; tn[i] = lr.theta[en]
    end
    if bootstrap && nreps > 0
        w = T(0.1)                       # mock deterministic bands (rng accepted, unused)
        pl = m_pos .- w; ph = m_pos .+ w
        nl = m_neg .- w; nh = m_neg .+ w
        dl = m_dif .- w; dh = m_dif .+ w
    else
        z = Matrix{T}(undef, 0, 0)
        pl = ph = nl = nh = dl = dh = z
        nreps = 0
    end
    names = [m.xnames[o] for o in m.asym]
    NARDLMultipliers{T}(collect(0:H), copy(m.asym), names, m_pos, m_neg, m_dif,
                       pl, ph, nl, nh, dl, dh, tp, tn, nreps, T(level))
end

export ARDLLongRun, ARDLModel, ARDLBoundsTest, NARDLModel, NARDLSymmetryTest, NARDLMultipliers
export estimate_ardl, estimate_nardl, long_run, ecm_form, bounds_test, symmetry_test, dynamic_multipliers

# ─── C062c: dynamic heterogeneous-panel ARDL (PMG / MG / DFE) ────────────────
# PMGModel mirrors the real MEMs 0.7.0 field NAMES/ORDER (a subset is fine — check_mock_surface
# is mock ⊆ real). `estimate_pmg` is a genuine-ish per-unit error-correction fit that VALIDATES
# like the real one (N≥2, ≥1 regressor, method/trend enums, p≥1/q≥0, per-unit sample length →
# ArgumentError) so T1/T2 exercise the error mapping AND the hand-built long-run/short-run
# renderers. `hausman_test(::PMGModel, ::PMGModel)` is the PMG-typed dispatch — DISTINCT from the
# generic FE-vs-RE `hausman_test(::PanelRegModel, ::PanelRegModel)` above (no name clash). NOTE
# the trend vocabulary is PMG-specific: none|constant|trend (`:constant` spelled out).
struct PMGModel{T<:AbstractFloat}
    method::Symbol
    yname::String
    xnames::Vector{String}
    srnames::Vector{String}
    theta::Vector{T}
    theta_se::Vector{T}
    theta_vcov::Matrix{T}
    theta_i::Matrix{T}
    phi_i::Vector{T}
    phi::T
    phi_se::T
    sr::Vector{T}
    sr_se::Vector{T}
    sr_i::Matrix{T}
    sigma2_i::Vector{T}
    loglik::T
    N::Int
    T_i::Vector{Int}
    p::Int
    q::Int
    n_nonconv::Int
    converged::Bool
    iters::Int
end

coef(m::PMGModel) = m.theta
vcov(m::PMGModel) = m.theta_vcov
stderror(m::PMGModel) = m.theta_se
nobs(m::PMGModel) = sum(m.T_i)

# One unit's EC design [ylag Xlag W], W = [det  Δy-lags  Δx-lags], mirroring real
# `_pmg_unit_design`. Returns (dy, Z, m_sr, srnames).
function _mock_pmg_unit(y::Vector{T}, X::Matrix{T}, p::Int, q::Int, trend::Symbol,
                        yname::String, xnames::Vector{String}) where {T}
    Ti = length(y); k = size(X, 2); L = max(p, q); rows = (L + 1):Ti; n = length(rows)
    dy = T[y[t] - y[t-1] for t in rows]
    ylag = T[y[t-1] for t in rows]
    Xlag = Matrix{T}(undef, n, k)
    for j in 1:k, (r, t) in enumerate(rows)
        Xlag[r, j] = X[t-1, j]
    end
    cols = Vector{Vector{T}}(); srnames = String[]
    if trend === :constant || trend === :trend
        push!(cols, ones(T, n)); push!(srnames, "(Intercept)")
    end
    if trend === :trend
        push!(cols, T.(collect(rows))); push!(srnames, "trend")
    end
    for j in 1:(p-1)
        push!(cols, T[y[t-j] - y[t-j-1] for t in rows]); push!(srnames, "L$j.D.$yname")
    end
    for jx in 1:k, l in 0:(q-1)
        push!(cols, T[X[t-l, jx] - X[t-l-1, jx] for t in rows])
        push!(srnames, l == 0 ? "D.$(xnames[jx])" : "L$l.D.$(xnames[jx])")
    end
    W = isempty(cols) ? Matrix{T}(undef, n, 0) : reduce(hcat, cols)
    (dy, hcat(ylag, Xlag, W), size(W, 2), srnames)
end

function estimate_pmg(pd::PanelData, y::Symbol, xs::Symbol...;
                      p::Int=1, q::Int=1, method::Symbol=:pmg, trend::Symbol=:constant,
                      maxiter::Int=100, tol::Real=1e-8)
    isempty(xs) && throw(ArgumentError("at least one long-run regressor is required"))
    method in (:pmg, :mg, :dfe) ||
        throw(ArgumentError("method must be :pmg, :mg, or :dfe; got :$method"))
    trend in (:none, :constant, :trend) ||
        throw(ArgumentError("trend must be :none, :constant, or :trend; got :$trend"))
    p >= 1 || throw(ArgumentError("p must be ≥ 1; got $p"))
    q >= 0 || throw(ArgumentError("q must be ≥ 0; got $q"))
    T = Float64
    _vc(v) = (j = findfirst(==(string(v)), pd.varnames);
              j === nothing ? throw(ArgumentError("Variable :$v not found. Available: $(pd.varnames)")) : j)
    yc = _vc(y); xcs = Int[_vc(x) for x in xs]
    k = length(xcs); yname = string(y); xnames = String[string(x) for x in xs]
    ug = sort(unique(pd.group_id)); N = length(ug)
    N >= 2 || throw(ArgumentError("need at least 2 units; got $N"))
    L = max(p, q)
    theta_i = Matrix{T}(undef, N, k); phi_i = zeros(T, N)
    Ti = zeros(Int, N); sigma2_i = zeros(T, N)
    sr_list = Vector{Vector{T}}(); srnames = String[]; m_sr = 0
    for (i, g) in enumerate(ug)
        rows = findall(==(g), pd.group_id)
        rr = rows[sortperm(pd.time_id[rows])]
        yg = T.(pd.data[rr, yc]); Xg = T.(pd.data[rr, xcs])
        length(yg) > L + k + (trend === :none ? 0 : 1) + (p - 1) + k * q ||
            throw(ArgumentError("unit $i: sample too short for ARDL($p,$q)"))
        dy, Z, msr, srn = _mock_pmg_unit(yg, Xg, p, q, trend, yname, xnames)
        i == 1 && (srnames = srn; m_sr = msr)
        b = Z \ dy
        resid = dy .- Z * b
        phi = b[1]; beta = b[2:(k+1)]
        phi_i[i] = phi
        denom = abs(phi) < 1e-8 ? (phi < 0 ? -T(1e-8) : T(1e-8)) : phi
        theta_i[i, :] .= -beta ./ denom
        push!(sr_list, b[(k+2):end])
        Ti[i] = length(dy)
        sigma2_i[i] = sum(abs2, resid) / max(length(dy), 1)
    end
    theta = vec(sum(theta_i; dims=1)) ./ T(N)
    # Swamy between-unit covariance: (N(N-1))⁻¹ Σ (θ_i−θ̄)(θ_i−θ̄)'.
    V = zeros(T, k, k)
    for i in 1:N
        d = theta_i[i, :] .- theta
        V .+= d * d'
    end
    V ./= T(N * (N - 1))
    # Pooled (PMG) / DFE are nominally more efficient than MG → smaller vcov, so the Hausman
    # difference dV = V_mg − V_eff is PSD (mock convention; keeps the quadratic form well-posed).
    theta_vcov = method === :mg ? V : T(0.5) .* V
    theta_se = T[sqrt(max(theta_vcov[j, j], zero(T))) for j in 1:k]
    phi = sum(phi_i) / T(N)
    phi_se = std(phi_i; corrected=true) / sqrt(T(N))
    sr_mat = reduce(vcat, [reshape(s, 1, m_sr) for s in sr_list])
    sr = vec(sum(sr_mat; dims=1)) ./ T(N)
    sr_se = T[std(@view(sr_mat[:, j]); corrected=true) / sqrt(T(N)) for j in 1:m_sr]
    loglik = -sum(T(Ti[i]) / 2 * (log(2π) + log(max(sigma2_i[i], eps())) + 1) for i in 1:N)
    n_nonconv = count(>=(zero(T)), phi_i)
    theta_i_store = method === :mg ? theta_i : Matrix{T}(undef, 0, k)
    iters = method === :pmg ? 3 : 0
    PMGModel{T}(method, yname, xnames, srnames, theta, theta_se, theta_vcov, theta_i_store,
                phi_i, phi, phi_se, sr, sr_se, sr_mat, sigma2_i, loglik, N, Ti, p, q,
                n_nonconv, true, iters)
end

# PMG-typed generalized Hausman (efficient vs consistent=:mg). A diagonalised quadratic form
# on the common long-run θ (mock: avoids importing pinv/Chisq) — finite, monotone survival in
# (0,1]; enough for T1/T2 shape/error assertions.
function hausman_test(efficient::PMGModel{T}, consistent::PMGModel{T}) where {T}
    consistent.method === :mg ||
        throw(ArgumentError("second argument must be the Mean Group model (:mg); got :$(consistent.method)"))
    length(efficient.theta) == length(consistent.theta) ||
        throw(ArgumentError("the two models must share the same long-run dimension"))
    db = efficient.theta .- consistent.theta
    dv = T[max(consistent.theta_vcov[i, i] - efficient.theta_vcov[i, i], T(1e-12)) for i in eachindex(db)]
    chi2 = sum((db .^ 2) ./ dv)
    df = length(db)
    pval = exp(-max(chi2, zero(T)) / 2)
    name = "Hausman test ($(uppercase(string(efficient.method))) vs MG)"
    h0 = efficient.method === :pmg ? "long-run homogeneity" : "$(uppercase(string(efficient.method))) consistent"
    desc = pval < T(0.05) ? "Reject H0 ($h0): use MG" :
           "Fail to reject H0 ($h0): $(uppercase(string(efficient.method))) preferred"
    PanelTestResult{T}(name, T(chi2), T(pval), df, desc)
end

export PMGModel, estimate_pmg

# ─── C062d: MIDAS mixed-frequency regression (EV-01) ─────────────────────────
# MidasModel mirrors the real MEMs 0.7.0 field NAMES/ORDER (a subset is fine — check_mock_surface
# is mock ⊆ real). `estimate_midas` is a genuine-ish concentrated-OLS fit that VALIDATES like the
# real one (m≥1, K≥1 / K≥2 for Beta, weights enum, p_ar≥0, complete-HF-block availability →
# ArgumentError) so T1/T2 exercise the error mapping AND the hand-built weight-curve/coef renderers.
# `midas_weights(m::MidasModel)` is the accessor form (== m.w); the pure-math `midas_weights(θ,K)`
# evaluator is out of scope. StatsAPI accessors mirror real (coef = [β; θ], stderror uses abs.diag).
struct MidasModel{T<:AbstractFloat}
    y::Vector{T}
    Xlags::Matrix{T}
    Wlin::Matrix{T}
    theta::Vector{T}
    beta::Vector{T}
    vcov_mat::Matrix{T}
    weights_kind::Symbol
    m::Int
    K::Int
    p_ar::Int
    poly_degree::Int
    h::Int
    w::Vector{T}
    fitted::Vector{T}
    residuals::Vector{T}
    ssr::T
    sigma2::T
    r2::T
    adj_r2::T
    loglik::T
    aic::T
    bic::T
    varnames::Vector{String}
    converged::Bool
end

coef(m::MidasModel) = vcat(m.beta, m.theta)
vcov(m::MidasModel) = m.vcov_mat
stderror(m::MidasModel) = sqrt.(abs.([m.vcov_mat[i, i] for i in 1:size(m.vcov_mat, 1)]))
nobs(m::MidasModel) = length(m.y)
residuals(m::MidasModel) = m.residuals
predict(m::MidasModel) = m.fitted

"""Accessor form: realized weight curve w(θ̂) (== m.w, length K, most-recent-first)."""
midas_weights(m::MidasModel) = m.w

# Normalized MIDAS weight curve (mirrors the real `_midas_weights`; sums to 1 for restricted kinds).
function _mock_midas_w(theta::AbstractVector{T}, K::Int, kind::Symbol) where {T}
    if kind === :expalmon
        k = T.(1:K); z = theta[1] .* k .+ theta[2] .* (k .^ 2); z .-= maximum(z)
        u = exp.(z); return u ./ sum(u)
    elseif kind === :beta2 || kind === :beta3
        K >= 2 || throw(ArgumentError("Beta weights require K ≥ 2 (got K=$K)"))
        x = T[clamp((T(kk) - one(T)) / (T(K) - one(T)), T(1e-8), one(T) - T(1e-8)) for kk in 1:K]
        u = (x .^ (theta[1] - one(T))) .* ((one(T) .- x) .^ (theta[2] - one(T)))
        kind === :beta3 && (u = u .+ theta[3])
        return u ./ sum(u)
    elseif kind === :almon
        k = T.(1:K); u = zeros(T, K)
        for (j, tj) in enumerate(theta); u .+= tj .* (k .^ (j - 1)); end
        return u ./ sum(u)
    elseif kind === :umidas
        return fill(one(T) / T(K), K)
    else
        throw(ArgumentError("unknown MIDAS weight kind: $kind"))
    end
end

function estimate_midas(y_lf::AbstractVector, X_hf::AbstractVector;
                        m::Int, K::Int, weights::Symbol=:expalmon,
                        p_ar::Int=0, poly_degree::Int=2, h::Int=1, max_iter::Int=500)
    weights ∈ (:expalmon, :beta2, :beta3, :almon, :umidas) ||
        throw(ArgumentError("unknown MIDAS weight kind: $weights"))
    m >= 1 || throw(ArgumentError("m must be ≥ 1 (got m=$m)"))
    K >= 1 || throw(ArgumentError("K must be ≥ 1"))
    p_ar >= 0 || throw(ArgumentError("p_ar must be ≥ 0"))
    h >= 1 || throw(ArgumentError("h must be ≥ 1 (got h=$h)"))
    (weights ∈ (:beta2, :beta3) && K < 2) && throw(ArgumentError("Beta weights require K ≥ 2 (got K=$K)"))
    T = Float64
    yv = collect(T, y_lf); xv = collect(T, X_hf)
    Tlf = length(yv); lenhf = length(xv)
    # Frequency alignment (mirror `_align_hf`): most-recent-first K-block per LF period; drop the
    # leading ragged edge (incomplete early blocks).
    retained = Int[]; blocks = Vector{Vector{T}}()
    for t in 1:Tlf
        hi = lenhf - (Tlf - t) * m; lo = hi - K + 1
        (lo >= 1 && hi <= lenhf) || continue
        push!(retained, t); push!(blocks, xv[hi:-1:lo])
    end
    isempty(retained) && throw(ArgumentError(
        "no complete high-frequency blocks: need ≥ $K HF obs before a low-frequency period"))
    # AR block: keep periods with p_ar available own-lags AND the direct-h
    # target in sample. h was INERT upstream at ≤0.7.2 and implemented at 0.7.3
    # (MEMs#574: regress y_{t+h-1} on information dated t) — the mock must not
    # stay on the old inert behavior or T1/T2 can never see h change anything.
    keep = Int[i for (i, t) in enumerate(retained) if t - p_ar >= 1 && t + h - 1 <= Tlf]
    isempty(keep) && throw(ArgumentError(
        "no periods with $p_ar autoregressive lags and direct horizon h=$h available"))
    n = length(keep)
    Xlags = reduce(vcat, [reshape(blocks[i], 1, K) for i in keep])
    Wlin = Matrix{T}(undef, n, 1 + p_ar); yv_used = Vector{T}(undef, n)
    for (r, i) in enumerate(keep)
        t = retained[i]; Wlin[r, 1] = one(T)
        for j in 1:p_ar; Wlin[r, 1 + j] = yv[t - j]; end
        yv_used[r] = yv[t + h - 1]   # direct-h target (h=1 ≡ nowcast of y_t)
    end
    # Concentrated OLS at a documented default θ (no NLS — genuine-ish fit for shape/error tests).
    if weights === :umidas
        M = hcat(Wlin[:, 1], Xlags, Wlin[:, 2:end])
        beta = M \ yv_used
        theta = T[]; w = beta[2:(1 + K)]
        varnames = vcat("const", String["HF lag $kk" for kk in 1:K], String["AR($j)" for j in 1:p_ar])
        Jfull = M
    else
        theta = weights === :expalmon ? T[0.0, 0.0] :
                weights === :beta2 ? T[1.0, 3.0] :
                weights === :beta3 ? T[1.0, 3.0, 0.0] :
                vcat(one(T), zeros(T, poly_degree))            # :almon (poly_degree + 1 params)
        w = _mock_midas_w(theta, K, weights)
        s = Xlags * w
        M = hcat(Wlin[:, 1], s, Wlin[:, 2:end])
        beta = M \ yv_used
        varnames = vcat("const", "β₁ (HF loading)", String["AR($j)" for j in 1:p_ar],
                        String["θ$l" for l in 1:length(theta)])
        # Stand-in Gauss-Newton θ-gradient block (Vandermonde in lag index) — right-sized, generally
        # full-rank → finite SEs; the mock exercises shape/finiteness, T3 covers real numerics.
        Jtheta = beta[2] .* (Xlags * T[T(kk)^l for kk in 1:K, l in 1:length(theta)])
        Jfull = hcat(M, Jtheta)
    end
    fitted = M * beta
    resid = yv_used .- fitted
    ssr = sum(abs2, resid)
    p = length(beta) + length(theta)
    dofres = max(n - p, 1); sigma2 = ssr / T(dofres)
    G = Jfull' * Jfull
    vcov = sigma2 .* ((G + T(1e-8) .* Matrix{T}(I, p, p)) \ Matrix{T}(I, p, p))
    ybar = mean(yv_used); tss = sum(abs2, yv_used .- ybar)
    r2 = tss > 0 ? one(T) - ssr / tss : zero(T)
    adj_r2 = n > p ? one(T) - (one(T) - r2) * T(n - 1) / T(n - p) : r2
    loglik = -T(0.5) * n * (log(T(2π)) + log(max(ssr / n, eps())) + one(T))
    aic = T(2) * p - T(2) * loglik; bic = T(log(n)) * p - T(2) * loglik
    MidasModel{T}(yv_used, Xlags, Wlin, theta, beta, Matrix{T}(vcov), weights, m, K, p_ar,
                  poly_degree, h, w, fitted, resid, ssr, sigma2, r2, adj_r2, loglik, aic, bic,
                  varnames, true)
end

# #67: MidasForecast + forecast(::MidasModel, X_new). Fields mirror real
# (midas/types.jl:144): forecast/ci_lower/ci_upper/se/horizon/conf_level.
#
# The mock encodes `X_new[1]` into the point forecast on purpose, so a test can pin the
# MOST-RECENT-FIRST contract: real applies the decaying weight curve to X_new in that
# order, and passing the block chronologically does not error — it silently returns a
# wrong number.
struct MidasForecast{T<:AbstractFloat}
    forecast::Vector{T}
    ci_lower::Vector{T}
    ci_upper::Vector{T}
    se::Vector{T}
    horizon::Int
    conf_level::T
end

function forecast(m::MidasModel, X_new::AbstractVector; y_lags=nothing, level::Real=0.95)
    xn = Float64.(collect(X_new))
    length(xn) >= m.K || throw(ArgumentError(
        "X_new needs ≥ K=$(m.K) high-frequency observations (got $(length(xn)))"))
    if m.p_ar > 0 && y_lags !== nothing
        length(collect(y_lags)) >= m.p_ar ||
            throw(ArgumentError("y_lags needs ≥ p_ar=$(m.p_ar) values"))
    end
    point = xn[1]                      # ← first element = most recent, by contract
    se = 0.25
    MidasForecast{Float64}([point], [point - 1.96se], [point + 1.96se], [se],
                           m.h, Float64(level))
end

export MidasForecast

export MidasModel, estimate_midas, midas_weights

# ─── W7/#109: TVP-VAR-SV and MF-VAR ──────────────────────────────────────────
# Field names and ARRAY LAYOUTS mirror real exactly. H_draws holds log-VARIANCES (the
# Kim-Shephard-Chib state), so volatility_path must return exp(h/2) -- a mock that stored
# standard deviations directly would hide a unit bug in the handler.

struct TVPVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}       # n_draws x T_eff x k,  k = n(1+np)
    A_draws::Array{T,3}       # n_draws x T_eff x n_a
    H_draws::Array{T,3}       # n_draws x T_eff x n   (log variances)
    Q_draws::Array{T,3}
    S_draws::Array{T,3}
    W_draws::Matrix{T}
    Y::Matrix{T}
    p::Int
    n::Int
    T_eff::Int
    n_train::Int
    tvp::Bool
    sv::Bool
    varnames::Vector{String}
end

function estimate_tvpvar(Y, p::Int; tvp::Bool=true, sv::Bool=true,
                         n_draws::Int=2000, n_burn::Int=1000, thin::Int=1,
                         n_train::Int=0, k_Q::Real=0.01, k_S::Real=0.1, k_W::Real=0.01,
                         varnames::Vector{String}=String[], rng=nothing, seed=nothing)
    T_obs, n = size(Y)
    n >= 2 || throw(ArgumentError("TVP-VAR requires at least 2 variables, got $n"))
    p >= 1 || throw(ArgumentError("p must be at least 1, got $p"))
    n_draws >= 1 || throw(ArgumentError("n_draws must be positive"))
    vn = isempty(varnames) ? ["y$i" for i in 1:n] : copy(varnames)
    T_eff = max(T_obs - p - n_train, 1)
    k = n * (1 + n * p); n_a = n * (n - 1) ÷ 2
    N = max(cld(n_draws, thin), 1)
    B = zeros(Float64, N, T_eff, k)
    for d in 1:N, t in 1:T_eff, j in 1:k
        B[d, t, j] = 0.1 + 0.001 * d + 0.0001 * t
    end
    A = zeros(Float64, N, T_eff, max(n_a, 1))
    H = fill(-0.5, N, T_eff, n)          # log variance -> sd = exp(-0.25) ≈ 0.7788
    TVPVARPosterior{Float64}(B, A, H, zeros(Float64, N, k, k),
                             zeros(Float64, N, max(n_a,1), max(n_a,1)),
                             zeros(Float64, N, n), Matrix{Float64}(Y), p, n, T_eff,
                             n_train, tvp, sv, vn)
end

function volatility_path(post::TVPVARPosterior; quantile_levels::Vector{<:Real}=[0.16,0.5,0.84])
    vol = exp.(post.H_draws ./ 2)
    mu = dropdims(sum(vol; dims=1) ./ size(vol,1); dims=1)
    qs = Array{Float64,3}(undef, post.T_eff, post.n, length(quantile_levels))
    for q in eachindex(quantile_levels), j in 1:post.n, t in 1:post.T_eff
        qs[t, j, q] = mu[t, j]
    end
    return mu, qs
end

function irf(post::TVPVARPosterior, horizon::Int; t::Int=post.T_eff, n_draws::Int=500,
             quantile_levels::Vector{<:Real}=[0.05,0.16,0.84,0.95],
             stationary_only::Bool=true)
    horizon >= 1 || throw(ArgumentError("horizon must be positive"))
    1 <= t <= post.T_eff || throw(ArgumentError("t must be in 1:$(post.T_eff), got $t"))
    n = post.n
    point = zeros(Float64, horizon, n, n)
    for h in 1:horizon, i in 1:n, j in 1:n
        point[h, i, j] = (i == j ? 1.0 : 0.3) * 0.8^(h-1)
    end
    ql = Float64.(quantile_levels)
    quant = Array{Float64,4}(undef, horizon, n, n, length(ql))
    for q in eachindex(ql), j in 1:n, i in 1:n, h in 1:horizon
        quant[h, i, j, q] = point[h, i, j] * (0.9 + 0.05 * q)
    end
    shocks = ["$(nm) shock" for nm in post.varnames]
    BayesianImpulseResponse{Float64}(quant, point, horizon, copy(post.varnames), shocks,
                                     ql, zeros(Float64, 1, horizon, n, n), n_draws, n_draws, 0)
end

struct MFVARPosterior{T<:AbstractFloat}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    Z_draws::Array{T,3}       # n_draws x T_hf x n
    data::Matrix{T}
    p::Int
    n::Int
    T_hf::Int
    low_freq::Vector{Int}
    freq_ratio::Int
    aggregation::Vector{Symbol}
    varnames::Vector{String}
end

function estimate_mfvar(data, p::Int; low_freq::Vector{Int}=Int[], freq_ratio::Int=3,
                        aggregation=:growth, n_draws::Int=1000, n_burn::Int=500,
                        prior::Symbol=:minnesota, hyper=nothing,
                        varnames::Vector{String}=String[], rng=nothing, seed=nothing)
    p >= 1 || throw(ArgumentError("p must be at least 1, got $p"))
    prior in (:minnesota, :diffuse) ||
        throw(ArgumentError("prior must be :minnesota or :diffuse, got :$prior"))
    freq_ratio >= 1 || throw(ArgumentError("freq_ratio must be ≥ 1, got $freq_ratio"))
    T_hf, n = size(data)
    all(1 .<= low_freq .<= n) ||
        throw(ArgumentError("low_freq indices must be in 1:$n, got $low_freq"))
    vn = isempty(varnames) ? ["y$i" for i in 1:n] : copy(varnames)
    aggs = aggregation isa Symbol ? fill(aggregation, length(low_freq)) : copy(aggregation)
    length(aggs) == length(low_freq) || throw(ArgumentError(
        "aggregation must be a Symbol or one Symbol per low_freq series"))
    for a in aggs
        a in (:stock, :flow, :average, :growth) || throw(ArgumentError(
            "aggregation must be :stock, :flow, :average or :growth, got :$a"))
    end
    # High-frequency columns must be complete; low-frequency ones need some data.
    is_low = fill(false, n); for j in low_freq; is_low[j] = true; end
    for i in 1:n
        if is_low[i]
            any(!isnan, @view data[:, i]) ||
                throw(ArgumentError("low-frequency series $(vn[i]) has no observations"))
        else
            any(isnan, @view data[:, i]) && throw(ArgumentError(
                "high-frequency series $(vn[i]) contains NaN; either list it in low_freq " *
                "or supply a complete series"))
        end
    end
    N = max(n_draws, 1)
    Z = zeros(Float64, N, T_hf, n)
    for d in 1:N, t in 1:T_hf, j in 1:n
        v = data[t, j]
        Z[d, t, j] = isnan(v) ? 0.5 + 0.001 * d : v
    end
    k = n * p + 1
    MFVARPosterior{Float64}(zeros(Float64, N, k, n), zeros(Float64, N, n, n), Z,
                            Matrix{Float64}(data), p, n, T_hf, copy(low_freq),
                            freq_ratio, aggs, vn)
end

function latent_path(post::MFVARPosterior; quantile_levels::Vector{<:Real}=[0.16,0.5,0.84])
    mu = dropdims(sum(post.Z_draws; dims=1) ./ size(post.Z_draws,1); dims=1)
    qs = Array{Float64,3}(undef, post.T_hf, post.n, length(quantile_levels))
    for q in eachindex(quantile_levels), j in 1:post.n, t in 1:post.T_hf
        qs[t, j, q] = mu[t, j]
    end
    return mu, qs
end

report(p::TVPVARPosterior) = "TVP-VAR($(p.p)) mock report"
report(p::MFVARPosterior) = "MF-VAR($(p.p)) mock report"

export TVPVARPosterior, estimate_tvpvar, volatility_path
export MFVARPosterior, estimate_mfvar, latent_path

# ─── W8/#110: generalized FEVD + Waggoner-Zha conditional forecasts ──────────

# Pesaran-Shin generalized FEVD. Deliberately does NOT sum to 1 across shocks unless
# normalize=true -- a mock that normalized unconditionally would hide exactly the property
# the renderer and the T3 assertion have to get right.
function generalized_fevd(model::VARModel, horizon::Int; normalize::Bool=false,
                          shock_names::Union{Nothing,Vector{String}}=nothing)
    n = size(model.Y, 2)
    props = Array{Float64,3}(undef, n, n, horizon)
    for h in 1:horizon, i in 1:n, j in 1:n
        props[i, j, h] = i == j ? 0.8 : 0.3          # rows sum to > 1 for n >= 2
    end
    if normalize
        for h in 1:horizon, i in 1:n
            props[i, :, h] ./= sum(@view props[i, :, h])
        end
    end
    FEVD(props, props)
end
generalized_fevd(post::BVARPosterior, horizon::Int; normalize::Bool=false,
                 shock_names::Union{Nothing,Vector{String}}=nothing) =
    generalized_fevd(_mock_var(post.data, post.p), horizon;
                     normalize=normalize, shock_names=shock_names)

struct ForecastCondition{T<:AbstractFloat}
    variable::Union{Int,String,Symbol}
    horizon::Int
    value::T
    sd::T
    function ForecastCondition{T}(variable, horizon::Integer, value::Real,
                                  sd::Real=zero(T)) where {T<:AbstractFloat}
        horizon >= 1 || throw(ArgumentError("condition horizon must be ≥ 1, got $horizon"))
        sd >= 0 || throw(ArgumentError("condition sd must be non-negative, got $sd"))
        new{T}(variable, Int(horizon), T(value), T(sd))
    end
end
forecast_condition(variable::Union{Int,String,Symbol}, horizon::Integer, value::Real;
                   sd::Real=0.0) = ForecastCondition{Float64}(variable, horizon, value, sd)

struct ConditionalForecast{T<:AbstractFloat}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    conf_level::T
    varnames::Vector{String}
    conditions::Vector{ForecastCondition{T}}
    unconditional::Matrix{T}
    shocks::Matrix{T}
    identification::Symbol
    n_draws::Int
end

function _mock_conditional_forecast(varnames::Vector{String}, conds, h::Int,
                                    reps::Int, conf_level::Real)
    h >= 1 || throw(ArgumentError("Forecast horizon must be positive"))
    reps >= 1 || throw(ArgumentError("reps must be positive"))
    (0 < conf_level < 1) || throw(ArgumentError("conf_level must be in (0, 1)"))
    n = length(varnames)
    cl = Vector{ForecastCondition{Float64}}()
    for c in conds
        idx = c.variable isa Integer ? Int(c.variable) :
              something(findfirst(==(String(c.variable)), varnames), 0)
        idx >= 1 && idx <= n || throw(ArgumentError(
            "condition variable $(repr(c.variable)) not found. Available: $varnames"))
        c.horizon <= h || throw(ArgumentError(
            "condition horizon $(c.horizon) exceeds forecast horizon $h"))
        push!(cl, c)
    end
    uncond = fill(0.25, h, n)
    fcast = copy(uncond)
    lo = fcast .- 1.0
    hi = fcast .+ 1.0
    # A HARD condition pins the path exactly and collapses the band, which is what the
    # renderer and T3 check.
    for c in cl
        idx = c.variable isa Integer ? Int(c.variable) :
              findfirst(==(String(c.variable)), varnames)
        fcast[c.horizon, idx] = c.value
        if c.sd == 0
            lo[c.horizon, idx] = c.value
            hi[c.horizon, idx] = c.value
        end
    end
    ConditionalForecast{Float64}(fcast, lo, hi, h, Float64(conf_level), copy(varnames),
                                 cl, uncond, fill(0.1, h, n), :cholesky, reps)
end

conditional_forecast(model::VARModel, conditions, h::Int; Q=nothing, reps::Int=1000,
                     conf_level::Real=0.95, rng=nothing, seed=nothing) =
    _mock_conditional_forecast(model.varnames, conditions, h, reps, conf_level)
conditional_forecast(post::BVARPosterior, conditions, h::Int; Q=nothing, reps::Int=1000,
                     conf_level::Real=0.95, rng=nothing, seed=nothing) =
    _mock_conditional_forecast(post.varnames, conditions, h, reps, conf_level)

report(fc::ConditionalForecast) = "ConditionalForecast mock report"

export generalized_fevd, ForecastCondition, forecast_condition, ConditionalForecast
export conditional_forecast

# ─── W9/#111: quantile regression + RDD ──────────────────────────────────────
# Field names and shapes mirror real. NOTE beta/stderr/residuals/fitted are k x n_tau (or
# n x n_tau) MATRICES even for a single tau -- a mock that collapsed them to vectors would
# hide the renderer's indexing.

struct QuantileRegModel{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    taus::Vector{T}
    beta::Matrix{T}
    vcov_mats::Vector{Matrix{T}}
    stderr::Matrix{T}
    residuals::Matrix{T}
    fitted::Matrix{T}
    objective::Vector{T}
    pseudo_r2::Vector{T}
    varnames::Vector{String}
    se_type::Symbol
    n_obs::Int
    converged::Vector{Bool}
end

function estimate_qreg(y::AbstractVector, X::AbstractMatrix, tau=0.5;
                       se::Symbol=:iid, varnames=nothing, n_boot::Int=500,
                       rng=nothing, alpha::Real=0.05, seed=nothing)
    n, k = length(y), size(X, 2)
    size(X, 1) == n || throw(ArgumentError("X must have $n rows (got $(size(X, 1)))"))
    n > k || throw(ArgumentError("Need n > k (n=$n, k=$k)"))
    se in (:iid, :robust, :boot) ||
        throw(ArgumentError("se must be :iid, :robust, or :boot; got :$se"))
    taus = tau isa Real ? Float64[tau] : Float64.(collect(tau))
    all(t -> 0 < t < 1, taus) || throw(ArgumentError("tau must lie in (0, 1)"))
    nt = length(taus)
    vn = varnames === nothing ? ["x$i" for i in 1:k] : copy(varnames)
    beta = Matrix{Float64}(undef, k, nt)
    for j in 1:nt, i in 1:k
        beta[i, j] = 0.5 + 0.1 * i + taus[j]        # varies with tau, as a real fit would
    end
    QuantileRegModel{Float64}(Float64.(y), Float64.(X), taus, beta,
        [Matrix{Float64}(I, k, k) for _ in 1:nt], fill(0.2, k, nt),
        fill(0.1, n, nt), fill(0.5, n, nt), fill(12.5, nt), fill(0.3, nt),
        vn, se, n, fill(true, nt))
end

struct RDDResult{T<:AbstractFloat}
    tau_conventional::T
    tau_bias_corrected::T
    se_conventional::T
    se_robust::T
    ci_conventional::Tuple{T,T}
    ci_robust::Tuple{T,T}
    pvalue_robust::T
    z_robust::T
    h::T
    b::T
    n_left::Int
    n_right::Int
    cutoff::T
    p::Int
    kernel::Symbol
    level::T
    design::Symbol
    first_stage::Union{Nothing,T}
end

function estimate_rdd(y::AbstractVector, running::AbstractVector; cutoff::Real=0.0,
                      fuzzy=nothing, kernel::Symbol=:triangular, p::Int=1,
                      h=nothing, b=nothing, level::Real=0.95)
    length(running) == length(y) ||
        throw(ArgumentError("running must have length $(length(y))"))
    kernel in (:triangular, :epanechnikov, :uniform) || throw(ArgumentError(
        "kernel must be :triangular, :epanechnikov, or :uniform; got :$kernel"))
    p >= 1 || throw(ArgumentError("p must be >= 1, got $p"))
    0 < level < 1 || throw(ArgumentError("level must lie in (0,1), got $level"))
    fuzzy === nothing || length(fuzzy) == length(y) ||
        throw(ArgumentError("fuzzy must have length $(length(y))"))
    nl = count(<(cutoff), running); nr = count(>=(cutoff), running)
    hh = h === nothing ? 1.5 : Float64(h)
    bb = b === nothing ? 2.5 : Float64(b)
    RDDResult{Float64}(2.0, 2.1, 0.4, 0.45, (1.216, 2.784), (1.218, 2.982),
                       0.0001, 4.67, hh, bb, nl, nr, Float64(cutoff), p, kernel,
                       Float64(level), fuzzy === nothing ? :sharp : :fuzzy,
                       fuzzy === nothing ? nothing : 0.8)
end

report(m::QuantileRegModel) = "QuantileRegModel mock report"
report(r::RDDResult) = "RDDResult mock report"

export QuantileRegModel, estimate_qreg, RDDResult, estimate_rdd

# ─── W10/#112: micro inference riders ────────────────────────────────────────────────
#
# Field names, types and ARRAY SHAPES mirror real MEMs 0.7.2 exactly. Nothing here invents
# a name real lacks (#84) — the CIPS/`plot_result` lessons — and the AR/wild-bootstrap
# results deliberately reproduce real's degenerate shapes (empty set, unbounded side,
# enumerated sign space) so a handler that assumes `[lo, hi]` fails at T1/T2 rather than in
# production.

struct AndersonRubinTest{T<:AbstractFloat}
    beta0::Vector{T}
    statistic::T
    p_value::T
    df1::Int
    df2::Int
    distribution::Symbol
    cov_type::Symbol
    endog_names::Vector{String}
end

struct AndersonRubinCI{T<:AbstractFloat}
    intervals::Vector{Tuple{T,T}}
    is_empty::Bool
    is_whole_line::Bool
    bounded::Bool
    level::T
    critical_value::T
    grid_lo::T
    grid_hi::T
    wald_lower::T
    wald_upper::T
    estimate::T
    df1::Int
    distribution::Symbol
    endog_name::String
end

# Mirrors real's `_ar_unpack`: the model must carry Z and the endogenous indices, i.e. it
# must have come from `estimate_iv`. Same ArgumentError, hence the same exit class.
function _mock_ar_unpack(m::RegModel)
    m.Z === nothing && throw(ArgumentError(
        "the model was not estimated by IV — anderson_rubin_test requires a model from estimate_iv"))
    m.endogenous === nothing && throw(ArgumentError("the model carries no endogenous indices"))
    return m.endogenous
end

function anderson_rubin_test(model, beta0; cov_type=nothing, clusters=nothing)
    endog = _mock_ar_unpack(model)
    ct = cov_type === nothing ? model.cov_type : cov_type
    ct in (:ols, :hc0, :hc1, :hc2, :hc3, :cluster) || throw(ArgumentError(
        "cov_type must be :ols, :hc0, :hc1, :hc2, :hc3, or :cluster; got :$ct"))
    ct === :cluster && clusters === nothing && throw(ArgumentError(
        "clusters is required for cov_type=:cluster"))
    b0 = beta0 isa Number ? Float64[Float64(beta0)] : Vector{Float64}(beta0)
    length(b0) == length(endog) || throw(ArgumentError(
        "beta0 has length $(length(b0)) but the model has $(length(endog)) endogenous regressors"))
    q = size(model.Z, 2) - (size(model.X, 2) - length(endog))
    q = max(q, 1)
    dist = ct === :ols ? :F : :chisq
    AndersonRubinTest{Float64}(b0, 2.5, 0.08, q, length(model.y) - size(model.Z, 2),
                               dist, ct, model.varnames[endog])
end

# `_MOCK_FLAGS[:ar_set_shape]` drives the degenerate shapes so T1/T2 can exercise the
# renderer's empty / unbounded / disjoint branches, which is where a `[lo, hi]` assumption
# would break.
function anderson_rubin_ci(model; level=0.95, n_grid=1001, span=20, grid=nothing,
                           cov_type=nothing, clusters=nothing)
    endog = _mock_ar_unpack(model)
    length(endog) == 1 || throw(ArgumentError(
        "anderson_rubin_ci inverts over a single endogenous coefficient; this model has " *
        "$(length(endog)). Use anderson_rubin_test at specific vectors instead."))
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1)"))
    n_grid >= 5 || throw(ArgumentError("n_grid must be at least 5"))
    ct = cov_type === nothing ? model.cov_type : cov_type
    ct === :cluster && clusters === nothing && throw(ArgumentError(
        "clusters is required for cov_type=:cluster"))
    j = endog[1]
    est = Float64(model.beta[j])
    shape = get(_MOCK_FLAGS, :ar_set_shape, :bounded)
    ivals, empt, whole, bnd = if shape === :empty
        (Tuple{Float64,Float64}[], true, false, true)
    elseif shape === :whole
        ([(-Inf, Inf)], false, true, false)
    elseif shape === :unbounded
        ([(est - 0.4, Inf)], false, false, false)
    elseif shape === :disjoint
        ([(est - 1.2, est - 0.6), (est + 0.6, est + 1.2)], false, false, true)
    else
        ([(est - 0.5, est + 0.5)], false, false, true)
    end
    AndersonRubinCI{Float64}(ivals, empt, whole, bnd, Float64(level), 3.84,
                             est - Float64(span) * 0.1, est + Float64(span) * 0.1,
                             est - 0.3, est + 0.3, est, 1,
                             ct === :ols ? :F : :chisq, model.varnames[j])
end

struct WildClusterBootstrap{T<:AbstractFloat}
    coefname::String
    coefindex::Int
    estimate::T
    null_value::T
    t_stat::T
    p_value::T
    p_value_equaltail::T
    p_value_asymptotic::T
    ci_lower::T
    ci_upper::T
    level::T
    t_boot::Vector{T}
    n_boot::Int
    n_clusters::Int
    weighttype::Symbol
    imposenull::Bool
    enumerated::Bool
end

function wild_cluster_bootstrap(model::RegModel, coefficient, null_value::Real=0.0;
                                clusters=nothing, n_boot::Int=999,
                                weights::Symbol=:rademacher, imposenull::Bool=true,
                                ci::Bool=true, level::Real=0.95, ci_gridpoints::Int=25,
                                enumerate=nothing, rng=nothing, seed=nothing)
    # Mirror real's guards: clusters are REQUIRED for a RegModel, and the weight scheme is
    # a closed two-member enum (the `--wild-dist` lesson — never infer enum members).
    clusters === nothing && throw(ArgumentError(
        "clusters is required for a RegModel — pass the same cluster vector used for the " *
        "cluster-robust covariance"))
    weights in (:rademacher, :webb) || throw(ArgumentError(
        "weights must be :rademacher or :webb; got :$weights"))
    # NOT `something(findfirst(...), throw(...))`: `something` is an ordinary function, so
    # BOTH arguments are evaluated and the throw fires even on a successful lookup.
    idx = if coefficient isa Integer
        Int(coefficient)
    else
        hit = findfirst(==(string(coefficient)), model.varnames)
        hit === nothing && throw(ArgumentError(
            "coefficient $(coefficient) not found in $(model.varnames)"))
        hit
    end
    (1 <= idx <= length(model.varnames)) || throw(ArgumentError(
        "coefficient index $idx out of range 1:$(length(model.varnames))"))
    G = length(unique(clusters))
    G >= 2 || throw(ArgumentError("Need at least 2 clusters for the wild cluster bootstrap"))
    # Real enumerates whenever 2^G <= n_boot and the weights are Rademacher; `enumerate`
    # forces or forbids it — and FORCING it when it is impossible is an error upstream,
    # so mirror that rather than silently obliging (a mock looser than real hides a real
    # failure).
    can_enum = weights === :rademacher && G <= 20 && 2^G <= n_boot
    if enumerate === true && !can_enum
        throw(ArgumentError(
            "enumerate=true requires Rademacher weights, G ≤ 20 and 2^G ≤ n_boot " *
            "(G=$G, n_boot=$n_boot, weights=:$weights)"))
    end
    enumerated = enumerate === nothing ? can_enum : (enumerate === true)
    nb = enumerated ? 2^min(G, 20) : n_boot
    est = Float64(model.beta[idx])
    WildClusterBootstrap{Float64}(model.varnames[idx], idx, est, Float64(null_value),
                                  2.1, 0.07, 0.065, 0.03,
                                  ci ? est - 0.4 : NaN, ci ? est + 0.4 : NaN,
                                  Float64(level), fill(0.5, min(nb, 16)), nb, G,
                                  weights, imposenull, enumerated)
end

struct MontielOleaPfluegerF{T<:AbstractFloat}
    f_effective::T
    critical_value::T
    tau::T
    weak::Bool
    n_instruments::Int
    bandwidth::Int
    f_naive::T
end

const _MOP_SIMPLIFIED_CV = Dict(0.05 => 37.42, 0.10 => 23.11, 0.20 => 15.06, 0.30 => 12.04)

function montiel_olea_pflueger_f(model::LPIVModel; tau::Real=0.10, bandwidth::Int=0)
    haskey(_MOP_SIMPLIFIED_CV, Float64(tau)) || throw(ArgumentError(
        "tau must be one of $(sort(collect(keys(_MOP_SIMPLIFIED_CV)))), got $tau"))
    crit = _MOP_SIMPLIFIED_CV[Float64(tau)]
    f_naive = Float64(first(model.first_stage_F))
    f_eff = f_naive * 0.9
    MontielOleaPfluegerF{Float64}(f_eff, crit, Float64(tau), f_eff < crit,
                                  size(model.instruments, 2), bandwidth, f_naive)
end

struct LPIVARBand{T<:AbstractFloat}
    lower::Matrix{T}
    upper::Matrix{T}
    sets::Matrix{Vector{Tuple{T,T}}}
    bounded::Matrix{Bool}
    is_empty::Matrix{Bool}
    wald_lower::Matrix{T}
    wald_upper::Matrix{T}
    point::Matrix{T}
    bandwidths::Matrix{Int}
    horizon::Int
    level::T
    critical_value::T
    df1::Int
    response_names::Vector{String}
    shock_name::String
end

function lp_iv_ar_band(model::LPIVModel; level::Real=0.95, n_grid::Int=401,
                       span::Real=20, bandwidth::Int=0, responses=nothing)
    (0 < level < 1) || throw(ArgumentError("level must be in (0, 1)"))
    n_grid >= 5 || throw(ArgumentError("n_grid must be at least 5"))
    H = model.horizon
    names = model.varnames[model.response_vars]
    nr = length(names)
    responses === nothing || (all(1 .<= responses .<= nr) || throw(ArgumentError(
        "responses must index into 1:$nr")))
    idx = responses === nothing ? collect(1:nr) : responses
    nk = length(idx)
    pt = fill(0.1, H + 1, nk)
    lo = fill(-0.4, H + 1, nk)
    hi = fill(0.6, H + 1, nk)
    bnd = trues(H + 1, nk)
    emp = falses(H + 1, nk)
    # Real routinely returns unbounded cells at long horizons; make the LAST horizon
    # unbounded so a renderer that assumes finite bounds fails at T1/T2, not in production.
    lo[H + 1, :] .= -Inf
    bnd[H + 1, :] .= false
    sets = Matrix{Vector{Tuple{Float64,Float64}}}(undef, H + 1, nk)
    for i in 1:(H + 1), j in 1:nk
        sets[i, j] = [(lo[i, j], hi[i, j])]
    end
    LPIVARBand{Float64}(lo, hi, sets, bnd, emp, pt .- 0.3, pt .+ 0.3, pt,
                        fill(max(bandwidth, 1), H + 1, nk), H, Float64(level), 3.84,
                        size(model.instruments, 2), names[idx], "$(model.varnames[model.shock_var]) (IV)")
end

report(t::AndersonRubinTest) = "AndersonRubinTest mock report"
report(c::AndersonRubinCI) = "AndersonRubinCI mock report"
report(b::WildClusterBootstrap) = "WildClusterBootstrap mock report"
report(m::MontielOleaPfluegerF) = "MontielOleaPfluegerF mock report"
report(b::LPIVARBand) = "LPIVARBand mock report"

export AndersonRubinTest, AndersonRubinCI, anderson_rubin_test, anderson_rubin_ci,
       WildClusterBootstrap, wild_cluster_bootstrap,
       MontielOleaPfluegerF, montiel_olea_pflueger_f, LPIVARBand, lp_iv_ar_band

# ─── W12/#114: determinacy mapping + closed-form moments ──────────────────────────────
#
# Field names and shapes mirror real MEMs 0.7.2 exactly. The verdict codes are real's
# `DETERMINACY_CODES` values, and `determinacy_boundary` reproduces real's rule that a pair
# involving a FAILED point is not a boundary crossing — a solve failure is missing
# information, not a region, and inventing a frontier out of it would be a real bug the
# mock must be able to expose.

const DETERMINACY_CODES = (determinate=1, indeterminate=0, no_solution=-1, failed=-2)

function determinacy_label(code::Integer)
    code == DETERMINACY_CODES.determinate   && return "determinate"
    code == DETERMINACY_CODES.indeterminate && return "indeterminate"
    code == DETERMINACY_CODES.no_solution   && return "no solution"
    code == DETERMINACY_CODES.failed        && return "failed"
    return "unknown"
end

struct DeterminacyMap{T<:AbstractFloat}
    params::Vector{Symbol}
    axes::Vector{Vector{T}}
    verdict::Matrix{Int}
    eu::Array{Int,3}
    failures::Dict{Tuple{Int,Int},String}
    base_values::Dict{Symbol,T}
    div::Float64
    method::Symbol
end

function determinacy_region(spec::ModelSpec{T},
                            theta_base::AbstractDict=spec.param_values;
                            params, grids, div::Real=1.0 + 1e-8, rank_rtol::Real=1e-8,
                            method::Symbol=:gensys, threaded::Bool=false,
                            quiet::Bool=true) where {T}
    pnames = params isa Symbol ? [params] : collect(Symbol.(params))
    (1 <= length(pnames) <= 2) || throw(ArgumentError(
        "determinacy_region sweeps 1 or 2 parameters, got $(length(pnames))"))
    for p in pnames
        p in spec.params || throw(ArgumentError(
            "parameter :$p is not a parameter of this model (have $(spec.params))"))
    end
    length(unique(pnames)) == length(pnames) || throw(ArgumentError(
        "the swept parameters must be distinct, got $pnames"))
    method in (:gensys, :klein, :blanchard_kahn) || throw(ArgumentError(
        "method must be :gensys, :klein, or :blanchard_kahn; got :$method"))
    gaxes = if length(pnames) == 1 && !(grids isa Tuple) &&
               !(grids isa AbstractVector{<:AbstractVector})
        [collect(Float64, grids)]
    else
        [collect(Float64, g) for g in grids]
    end
    length(gaxes) == length(pnames) || throw(ArgumentError(
        "got $(length(pnames)) parameter(s) but $(length(gaxes)) grid(s)"))
    all(!isempty, gaxes) || throw(ArgumentError("grids must be non-empty"))

    n1 = length(gaxes[1])
    n2 = length(pnames) == 2 ? length(gaxes[2]) : 1
    verdict = Matrix{Int}(undef, n1, n2)
    eu = Array{Int,3}(undef, n1, n2, 2)
    # A Taylor-principle-shaped boundary: determinate above 1.0 on the first axis, so a T3
    # or T1/T2 case can assert region labels on BOTH sides of a known frontier.
    for j in 1:n2, i in 1:n1
        det = gaxes[1][i] > 1.0
        verdict[i, j] = det ? DETERMINACY_CODES.determinate : DETERMINACY_CODES.indeterminate
        eu[i, j, 1] = 1
        eu[i, j, 2] = det ? 1 : 0
    end
    base = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in theta_base)
    return DeterminacyMap{Float64}(pnames, gaxes, verdict, eu,
                                   Dict{Tuple{Int,Int},String}(), base,
                                   Float64(div), method)
end

function determinacy_boundary(m::DeterminacyMap{T}) where {T}
    length(m.params) == 1 || throw(ArgumentError(
        "determinacy_boundary is defined for a one-parameter sweep; this map sweeps " *
        "$(length(m.params)) parameters — read `verdict` directly."))
    g = m.axes[1]
    out = T[]
    fail = DETERMINACY_CODES.failed
    for i in 2:length(g)
        a, b = m.verdict[i-1, 1], m.verdict[i, 1]
        (a == fail || b == fail) && continue      # a hole is not a crossing
        a == b && continue
        push!(out, (g[i-1] + g[i]) / 2)
    end
    return out
end

"""
Mock `analytical_moments`, packed EXACTLY as real packs it — the CLI unpacks by position,
so a divergence here would hide a mislabelled-moment bug in production.

`:covariance` → upper-triangle of Var_y, then diagonal autocov per lag.
`:gmm`        → means, then upper-triangle PRODUCT moments (Var + E·E), then diagonal
                autocov + E² per lag.
"""
function analytical_moments(sol::PerturbationSolution{T}; lags::Int=1,
                            format::Symbol=:covariance) where {T}
    format in (:covariance, :gmm) ||
        throw(ArgumentError("format must be :covariance or :gmm; got $format"))
    spec = sol.spec
    idx = spec.augmented ? Int[findfirst(==(v), spec.endog) for v in spec.original_endog] :
                           collect(1:spec.n_endog)
    k = length(idx)
    # Deterministic, order-dependent fixtures: the risk correction is zero at order 1 and
    # non-zero above, which is the property the handler's `mean_minus_ss` column reports.
    E = T[sol.order == 1 ? zero(T) : T(0.01 * i * (sol.order - 1)) for i in 1:k]
    Var = zeros(T, k, k)
    for i in 1:k, j in 1:k
        Var[i, j] = i == j ? T(1.0 + 0.1 * i) : T(0.2 / (1 + abs(i - j)))
    end
    out = T[]
    if format === :gmm
        append!(out, E)
        for i in 1:k, j in i:k
            push!(out, Var[i, j] + E[i] * E[j])
        end
        for lag in 1:lags, i in 1:k
            push!(out, T(0.7^lag) * Var[i, i] + E[i]^2)
        end
    else
        for i in 1:k, j in i:k
            push!(out, Var[i, j])
        end
        for lag in 1:lags, i in 1:k
            push!(out, T(0.7^lag) * Var[i, i])
        end
    end
    return out
end

report(m::DeterminacyMap) = "DeterminacyMap mock report"

export DeterminacyMap, determinacy_region, determinacy_boundary, determinacy_label,
       DETERMINACY_CODES, analytical_moments

# ─── Native-serialization registry (mirrors real `_SERIALIZABLE_TYPES`, MEMs#506) ─────
#
# Deliberately LAST in the file: it maps names → Types, and mocks.jl is one flat module
# included top-to-bottom, so every referenced type must already be defined (the same
# forward-reference trap that broke the C051 `DataFrame(::Union{...})` dispatches).
#
# The names are upstream's 350 at MEMs 0.9.3 (RSER-02–14), filtered to those this
# mock actually defines. Filtering keeps the mock a SUBSET of real, which is the safe
# direction (#84): a mock that accepted a type real rejects would turn a guaranteed
# production `SerializationError` into a green suite. At 0.9.3 the DSGE/HA solution
# types ARE registered upstream (their equations recompile at load under an AST
# allowlist), so any the mock defines round-trip through the mock `save_model` too —
# per-type fidelity is proven on real MEMs by the T3 serialization round-trips.
const _SERIALIZABLE_TYPE_NAMES = (
    "ACFResult", "ADF2BreakResult", "ADFResult", "AmengualWatsonResult", "AndersonRubinCI",
    "AndersonRubinTest", "AndrewsResult", "APARCHModel", "ARCHModel", "ARDLBoundsTest",
    "ARDLLongRun", "ARDLModel", "ARFIMAModel", "AriasSVARResult", "ARIMAForecast", "ARIMAModel",
    "ARIMAOrderSelection", "ARMAModel", "ARModel", "BaconDecomposition", "BaiNgQResult",
    "BaiPerronResult", "BaqaeeFarhiResult", "BartlettWhiteNoiseResult", "BaselinePath",
    "BaxterKingResult", "BayesianDSGE", "BayesianDSGESimulation", "BayesianFAVAR", "BayesianFEVD",
    "BayesianHistoricalDecomposition", "BayesianImpulseResponse", "BayesianSetIdentifiedSVAR",
    "BDSResult", "BeveridgeNelsonResult", "BFElasticities", "BFEquilibrium", "BFLocal",
    "BFMisallocation", "BFShockCurve", "BFWedgeDecomp", "BlanchardOLG", "BlanchardOLGSolution",
    "BlanchardOLGSteadyState", "BoostedHPResult", "BoxPierceResult", "BreitungPanelResult",
    "BSplineBasis", "BubbleResult", "BVARForecast", "BVARPosterior", "CGARCHModel",
    "ClarkWestResult", "CointRegModel", "ConditionalForecast", "ContinuousHouseholdSystem",
    "CorTestResult", "CounterfactualHistory", "CounterfactualMoments", "CrossSectionData",
    "CrossSpectrumResult", "CTAiyagari", "CTPoissonIncome", "CTSteadyState", "CTTransition",
    "CTTwoAsset", "CTTwoAssetGE", "CTTwoAssetSolution", "CTTwoAssetTransition", "DataDiagnostic",
    "DataSummary", "DCEGMDistribution", "DCEGMEquilibrium", "DCEGMFirm", "DCEGMProblem",
    "DCEGMSolution", "DCEGMSystem", "DCEGMTransition", "DenHaanAccuracy", "DeterminacyMap",
    "DFGLSResult", "DIDResult", "DispersionTest", "DMTestResult", "DSGEEstimation", "DSGEPrior",
    "DSGESolution", "DSGEStateSpace", "DumitrescuHurlinResult", "DurbinWatsonResult",
    "DynamicFactorModel", "EDFTestResult", "EGARCHModel", "EngleGrangerResult",
    "EqualityTestResult", "ERSResult", "EventStudyLP", "ExportDecomposition",
    "ExternalVolatilitySVARResult", "ExtractionResult", "FactorBreakResult", "FactorForecast",
    "FactorModel", "FAVARModel", "FEVD", "FIEGARCHModel", "FIGARCHModel", "FirmSystem",
    "FisherJohansenResult", "FisherPanelResult", "FisherTestResult", "FootprintResult",
    "ForecastCombination", "ForecastCondition", "ForecastEncompassingResult", "ForecastEvaluation",
    "ForecastSufficiency", "FourierADFResult", "FourierKPSSResult", "FunctionConstraint",
    "GarchMidasModel", "GARCHModel", "GARCHSVARResult", "GeneralizedDynamicFactorModel",
    "GJRGARCHModel", "GLPHyperparameters", "GMMModel", "GMMWeighting", "GPHResult",
    "GrangerCausalityResult", "GregoryHansenResult", "HadriResult", "HADSGESolution", "HAGrid",
    "HAGridDiagnostics", "HallinLiskaResult", "HamiltonFilterResult", "HansenInstabilityResult",
    "HansenLinearityTest", "HASteadyState", "HeckmanModel", "HEGYResult", "HetBlock",
    "HistoricalDecomposition", "HonestDiDResult", "HouseholdSystem", "HPFilterResult",
    "ICASVARResult", "IdentifiabilityTestResult", "IdentificationDiagnostics", "IGARCHModel",
    "ImpactResult", "ImpulseResponse", "IncomeProcess", "IndividualProblem", "InfluenceStats",
    "IntermediaryPE", "IntermediarySteadyState", "IntermediarySystem", "IntermediaryTransition",
    "IOData", "IOExtension", "IOMetaData", "IOMultipliers", "IPSResult", "IRDecl", "IREquation",
    "JohansenResult", "KalmanSmootherResult", "KaoResult", "KernelDensity", "KernelRegression",
    "KhanThomasSteadyState", "KhanThomasTransition", "KPSSResult", "KrusellSmithSolution",
    "LearningRateCheck", "LifeCycleOLG", "LifeCycleSteadyState", "LifeCycleSystem",
    "LifeCycleTransition", "LinearDSGE", "LinkageResult", "LjungBoxResult", "LLCResult",
    "LMTestResult", "LMUnitRootResult", "LocalWhittleResult", "LogitModel", "LowessFit",
    "LPDiDResult", "LPFEVD", "LPForecast", "LPImpulseResponse", "LPIVARBand", "LPIVModel",
    "LPModel", "LRTestResult", "MAModel", "MarginalEffects", "MarkovSwitchingSVARResult",
    "MaxShareResult", "MCMCDiagnostics", "MFVARPosterior", "MGARCHModel", "MidasForecast",
    "MidasModel", "MincerZarnowitzResult", "MinnesotaHyperparameters", "MitBlock",
    "ModelBankMember", "ModelIR", "ModelSpec", "MontielOleaPfluegerF", "MoonPerronResult",
    "MSForecast", "MSRegModel", "MultinomialLogitModel", "MultinomialMarginalEffects",
    "NamedEquation", "NARDLModel", "NARDLMultipliers", "NARDLSymmetryTest", "NegativeWeightResult",
    "NegBinModel", "NetworkStatsResult", "NgPerronResult", "NonGaussianGMMResult",
    "NonGaussianMLResult", "NonlinearStateSpace", "NormalityTestResult", "NormalityTestSuite",
    "NowcastBridge", "NowcastBVAR", "NowcastDFM", "NowcastForecast", "NowcastNews",
    "NowcastResult", "ObservationTrends", "OccBinConstraint", "OccBinIRF", "OccBinRegime",
    "OccBinSolution", "OddsRatio", "OPPResult", "OPPSequence", "OrderedLogitModel",
    "OrderedProbitModel", "PanelCointRegModel", "PanelData", "PanelIVModel", "PanelLogitModel",
    "PanelProbitModel", "PanelRegModel", "PanelTestResult", "PanelUnitRootSummary", "PANICResult",
    "ParameterTransform", "ParkAddedResult", "PathFloorConstraint", "PedroniResult",
    "PenalizedRegModel", "PerfectForesightPath", "PerturbationSolution", "PesaranCIPSResult",
    "PhillipsOuliarisResult", "PMGModel", "PoissonModel", "PolicyCausalEffects",
    "PolicyCounterfactual", "PolicyForecast", "PolicyLoss", "PolicyRule", "PosteriorMode",
    "PosteriorPredictiveCheck", "PPResult", "PrefilterSpec", "PretrendTestResult",
    "PriceModelResult", "PriorPosteriorOverlap", "PriorPredictiveResult", "ProbitModel",
    "ProductionNetwork", "ProjectionSolution", "ProjectionStateSpace", "PropensityLPModel",
    "PropensityScoreConfig", "ProxySVARResult", "PrunedStateSpace", "PVARModel", "PVARStability",
    "PVARTestResult", "QuantileRegModel", "RASResult", "RDDResult", "RegDiagnosticResult",
    "RegionalFootprintResult", "RegModel", "RobustBayesResult", "RobustRegModel", "SARIMAModel",
    "SDAResult", "SelectionResult", "SignIdentifiedSet", "SimpleBlock", "SMMModel",
    "SmoothLPModel", "SmoothTransitionSVARResult", "SpanningDiagnostic", "SpectralDensityResult",
    "SSJGEJacobian", "SSJImpulseResponse", "SSJModel", "StabilityResult", "STARForecast",
    "STARModel", "StateLPModel", "StateSpaceModel", "StructuralDFM", "StructuralLP", "SURModel",
    "SVARModel", "SVECResult", "SVModel", "ThreeSLSModel", "ThresholdForecast", "ThresholdModel",
    "TimeSeriesData", "TimingInfo", "TobitModel", "TransferFunctionResult", "TruncRegModel",
    "TVPVARPosterior", "UhligSVARResult", "VARForecast", "VarianceRatioResult", "VARModel",
    "VARStationarityResult", "VECMForecast", "VECMGrangerResult", "VECMModel",
    "VECMRestrictionTest", "VerticalSpecialization", "VolatilityForecast", "WesterlundResult",
    "WildClusterBootstrap", "WinberryFamily", "WoldRepresentation", "X13FilterResult", "ZAResult"
)
const _SERIALIZABLE_TYPES = Dict{String,Type}(
    n => getfield(@__MODULE__, Symbol(n))
    for n in _SERIALIZABLE_TYPE_NAMES if isdefined(@__MODULE__, Symbol(n))
)


# W7 HA-typed mocks live BELOW the HA type definitions — mocks.jl is one
# flat top-to-bottom module and a typed signature is resolved at include time
# (the standing forward-reference lesson).
function policy_causal_effects(spec::ModelSpec, ss::HASteadyState;
                               outcomes::AbstractVector{<:Pair{Symbol,Symbol}},
                               instruments::AbstractVector{<:Pair{Symbol,Symbol}}=[:rate => :r],
                               H::Int=100, T_horizon::Int=300,
                               rule_closure::Symbol=:administered, dx::Real=1e-4)
    rule_closure in (:administered, :market) || throw(ArgumentError(
        "rule_closure: expected :administered or :market, got :$rule_closure"))
    H >= 1 || throw(ArgumentError("H: expected H >= 1, got $H"))
    T_horizon >= H || throw(ArgumentError(
        "T_horizon = $T_horizon must be >= H = $H"))
    out_syms = Symbol[first(p) for p in outcomes]
    ins_syms = Symbol[first(p) for p in instruments]
    mk() = [Float64(0.6)^(abs(h - k)) * 0.05 for h in 1:H, k in 1:H]
    PolicyCausalEffects{Float64}(out_syms, ins_syms,
        [mk() for _ in out_syms], [mk() for _ in ins_syms], nothing, nothing,
        H, ["news $k" for k in 1:H], :ha)
end

function sequence_jacobian(spec::ModelSpec, ss::HASteadyState,
                           input::Symbol, output::Symbol;
                           T_horizon::Int=300, dx::Real=1e-4)
    input in (:r, :w) || throw(ArgumentError(
        "input: expected :r or :w, got :$input"))
    T_horizon >= 1 || throw(ArgumentError("T_horizon: expected >= 1, got $T_horizon"))
    [Float64(0.7)^(abs(i - j)) * 0.1 for i in 1:T_horizon, j in 1:T_horizon]
end


end # module
