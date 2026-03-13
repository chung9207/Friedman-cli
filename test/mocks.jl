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

using LinearAlgebra: I, diagm
using Statistics: mean

# ─── Core Types ───────────────────────────────────────────

struct VARModel{T<:Real}
    Y::Matrix{T}; p::Int; B::Matrix{T}; U::Matrix{T}; Sigma::Matrix{T}
    aic::T; bic::T; hqic::T
end

struct MockChains end

struct BVARPosterior{T}
    B_draws::Array{T,3}
    Sigma_draws::Array{T,3}
    n_draws::Int
    p::Int
    n::Int
    data::Matrix{T}
end

struct MinnesotaHyperparameters
    tau::Float64; decay::Float64; lambda::Float64; mu::Float64; omega::Vector{Float64}
end
MinnesotaHyperparameters(; tau=0.2, decay=1.0, lambda=0.5, mu=1.0, omega=[1.0]) =
    MinnesotaHyperparameters(tau, decay, lambda, mu, omega)

struct ImpulseResponse{T}
    values::Array{T,3}; ci_lower::Union{Array{T,3},Nothing}; ci_upper::Union{Array{T,3},Nothing}
    horizon::Int; variables::Vector{String}; shocks::Vector{String}; ci_type::Symbol
end
# Convenience 3-arg constructor for backward compat with existing handler tests
ImpulseResponse(v::Array{T,3}, cl, cu) where T = ImpulseResponse(v, cl, cu, size(v,1),
    ["var$i" for i in 1:size(v,2)], ["shock$i" for i in 1:size(v,3)], :cholesky)

struct BayesianImpulseResponse{T}
    quantiles::Array{T,4}; mean::Array{T,3}
    horizon::Int; variables::Vector{String}; shocks::Vector{String}; quantile_levels::Vector{T}
end
# Convenience 3-arg constructor for backward compat
BayesianImpulseResponse(m::Array{T,3}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianImpulseResponse(q, m, size(m,1),
    ["var$i" for i in 1:size(m,2)], ["shock$i" for i in 1:size(m,3)], ql)

struct FEVD{T}
    decomposition::Array{T,3}; proportions::Array{T,3}
end
struct BayesianFEVD{T}
    quantiles::Array{T,4}; mean::Array{T,3}
    horizon::Int; variables::Vector{String}; shocks::Vector{String}; quantile_levels::Vector{T}
end
# Convenience 3-arg constructor for backward compat
BayesianFEVD(m::Array{T,3}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianFEVD(q, m, size(m,1),
    ["var$i" for i in 1:size(m,2)], ["shock$i" for i in 1:size(m,3)], ql)

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
    quantiles::Array{T,4}; mean::Array{T,3}; initial_quantiles::Array{T,3}
    initial_mean::Matrix{T}; shocks_mean::Matrix{T}; actual::Matrix{T}
    T_eff::Int; variables::Vector{String}; shock_names::Vector{String}
    quantile_levels::Vector{T}; method::Symbol
end
# Convenience 4-arg constructor for backward compat
BayesianHistoricalDecomposition(m::Array{T,3}, im::Matrix{T}, q::Array{T,4}, ql::Vector{T}) where T =
    BayesianHistoricalDecomposition(q, m, zeros(T, 0, 0, 0), im, zeros(T, 0, 0), zeros(T, 0, 0),
    size(m,1), ["var$i" for i in 1:size(m,2)], ["shock$i" for i in 1:size(m,3)], ql, :cholesky)

struct ZeroRestriction
    variable::Int; shock::Int; horizon::Int
end
struct SignRestriction
    variable::Int; shock::Int; sign::Symbol; horizon::Int
end
struct SVARRestrictions
    n_vars::Int; zeros::Vector{ZeroRestriction}; signs::Vector{SignRestriction}
end
SVARRestrictions(n::Int; zeros=ZeroRestriction[], signs=SignRestriction[]) =
    SVARRestrictions(n, zeros, signs)
struct AriasSVARResult{T}
    Q_draws::Vector{Matrix{T}}; irf_draws::Array{T,4}; weights::Vector{T}; acceptance_rate::T
    restrictions::SVARRestrictions
end

struct UhligSVARResult{T}
    Q::Matrix{T}; irf::Array{T,3}; penalty::T; shock_penalties::Vector{T}
    restrictions::SVARRestrictions; converged::Bool
end

# ─── LP Types ─────────────────────────────────────────────

struct LPModel{T}
    Y::Matrix{T}; shock_var::Int; horizon::Int; lags::Int
    B::Matrix{T}; residuals::Matrix{T}; vcov::Matrix{T}; T_eff::Int
end
struct LPIVModel{T}
    Y::Matrix{T}; instruments::Matrix{T}; first_stage_F::T; horizon::Int; T_eff::Int
end
struct SmoothLPModel{T}
    Y::Matrix{T}; lambda::T; horizon::Int
end
struct StateLPModel{T}
    Y::Matrix{T}; B_expansion::Matrix{T}; B_recession::Matrix{T}; horizon::Int
end
struct PropensityLPModel{T}
    Y::Matrix{T}; ate::T; ate_se::T; horizon::Int
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
    common_component::Matrix{T}; loadings::Matrix{T}
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
    coefficients::Vector{T}; p::Int; sigma::T; aic_val::T; bic_val::T; ll::T
end
struct MAModel{T}
    coefficients::Vector{T}; q::Int; sigma::T; aic_val::T; bic_val::T; ll::T
end
struct ARMAModel{T}
    coefficients::Vector{T}; p::Int; q::Int; sigma::T; aic_val::T; bic_val::T; ll::T
end
struct ARIMAModel{T}
    coefficients::Vector{T}; p::Int; d::Int; q::Int; sigma::T; aic_val::T; bic_val::T; ll::T
end
struct ARIMAForecast{T}
    forecast::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}; se::Vector{T}
    horizon::Int; conf_level::T
end
# Convenience 5-arg constructor for backward compat
ARIMAForecast(f::Vector{T}, cl, cu, se, h::Int) where T =
    ARIMAForecast(f, cl, cu, se, h, T(0.95))

# ─── VAR Forecast Type ──────────────────────────────────

struct VARForecast{T<:AbstractFloat}
    forecast::Matrix{T}
    ci_lower::Matrix{T}
    ci_upper::Matrix{T}
    horizon::Int
    ci_method::Symbol
    conf_level::T
    varnames::Vector{String}
end

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

# ─── VECM Types ──────────────────────────────────────────

struct VECMModel{T<:Real}
    Y::Matrix{T}; p::Int; rank::Int
    alpha::Matrix{T}; beta::Matrix{T}; Pi::Matrix{T}
    Gamma::Vector{Matrix{T}}; mu::Vector{T}
    U::Matrix{T}; Sigma::Matrix{T}
    aic::T; bic::T; hqic::T; loglik::T
    deterministic::Symbol; method::Symbol
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

function _mock_var(Y::Matrix{Float64}, p::Int)
    T_obs, n = size(Y)
    k = n * p + 1
    B = zeros(k, n)
    for i in 1:min(n, k)
        B[i, i] = 0.5
    end
    U = zeros(T_obs - p, n) .+ 0.01
    Sigma = Matrix{Float64}(I(n)) * 0.01
    VARModel(Y, p, B, U, Sigma, -100.0, -95.0, -97.0)
end

# ─── Mock Functions ───────────────────────────────────────

select_lag_order(Y, max_p; criterion=:aic) = min(2, max(1, max_p))
estimate_var(Y, p; check_stability=true) = _mock_var(Y, p)

estimate_bvar(Y, p; sampler=:direct, n_draws=1000, prior=:normal, hyper=nothing) =
    BVARPosterior(zeros(10, size(Y,2)*p+1, size(Y,2)), zeros(10, size(Y,2), size(Y,2)),
                  10, p, size(Y,2), Y)
posterior_mean_model(post::BVARPosterior; data=nothing) = _mock_var(post.data, post.p)
posterior_median_model(post::BVARPosterior; data=nothing) = _mock_var(post.data, post.p)
# Keep old (chain, p, n) signatures for backward compat
posterior_mean_model(chain::MockChains, p, n; data=nothing) =
    _mock_var(isnothing(data) ? ones(100, n) : data, p)
posterior_median_model(chain::MockChains, p, n; data=nothing) =
    _mock_var(isnothing(data) ? ones(100, n) : data, p)
optimize_hyperparameters(Y, p) = MinnesotaHyperparameters(tau=0.2, decay=1.0, lambda=0.5, omega=ones(size(Y,2)))

# StatsAPI-like functions
coef(m::VARModel) = m.B
coef(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.coefficients
loglikelihood(m::VARModel) = -500.0
loglikelihood(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.ll
stderror(m::GMMModel) = fill(0.1, length(m.theta))
stderror(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = fill(0.01, length(m.coefficients))
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

is_stationary(m::VARModel) = (is_stationary=true, eigenvalues=[0.5+0.1im, 0.5-0.1im, 0.3+0.0im])
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

# IRF
function irf(model::VARModel, horizon::Int; method=:cholesky, check_func=nothing,
             narrative_check=nothing, ci_type=:none, reps=200, conf_level=0.95,
             stationary_only=false)
    n = size(model.Y, 2)
    vals = ones(horizon + 1, n, n) * 0.1
    ci_lo = ci_type == :none ? nothing : vals .- 0.5
    ci_hi = ci_type == :none ? nothing : vals .+ 0.5
    ImpulseResponse(vals, ci_lo, ci_hi)
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
             check_func=nothing, narrative_check=nothing)
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

function identify_sign(model::VARModel, horizon::Int, check_func; max_draws=1000, store_all=false)
    n = size(model.Y, 2)
    if store_all
        n_d = 10
        irf_draws = ones(n_d, horizon + 1, n, n) * 0.1
        Q_draws = [Matrix{Float64}(I(n)) for _ in 1:n_d]
        return SignIdentifiedSet(Q_draws, irf_draws, n_d, max_draws, Float64(n_d/max_draws),
            ["var$i" for i in 1:n], ["shock$i" for i in 1:n])
    end
    Q = Matrix{Float64}(I(n))
    irf_vals = ones(horizon + 1, n, n) * 0.1
    return (Q, irf_vals)
end

# FEVD
function fevd(model::VARModel, horizon::Int; method=:cholesky, check_func=nothing, narrative_check=nothing)
    n = size(model.Y, 2)
    props = ones(n, n, horizon) / n
    FEVD(props, props)
end
function fevd(chain::MockChains, p::Int, n::Int, horizon::Int;
              data=nothing, quantiles=[0.16, 0.5, 0.84])
    props = ones(n, n, horizon) / n
    q = ones(n, n, horizon, length(quantiles)) / n
    BayesianFEVD(props, q, Float64.(quantiles))
end
function fevd(post::BVARPosterior, horizon::Int;
              quantiles=[0.16, 0.5, 0.84])
    n = post.n
    props = ones(n, n, horizon) / n
    q = ones(n, n, horizon, length(quantiles)) / n
    BayesianFEVD(props, q, Float64.(quantiles))
end

# Historical Decomposition
function historical_decomposition(model::VARModel, horizon::Int; method=:cholesky,
                                   check_func=nothing, narrative_check=nothing)
    n = size(model.Y, 2)
    T_eff = min(horizon, size(model.Y, 1) - model.p)
    contribs = ones(T_eff, n, n) * 0.1
    actual = ones(T_eff, n)
    initial = ones(T_eff, n) * 0.01
    shocks_mat = ones(T_eff, n)
    HistoricalDecomposition(contribs, initial, actual, shocks_mat, T_eff)
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
                                   method=:cholesky, quantiles=[0.16, 0.5, 0.84])
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
verify_decomposition(hd::HistoricalDecomposition; tol=1e-6) = true
contribution(hd::HistoricalDecomposition, var::Int, shock::Int) = hd.contributions[:, var, shock]

# SVAR restrictions
zero_restriction(variable, shock; horizon=0) = ZeroRestriction(variable, shock, horizon)
sign_restriction(variable, shock, sign::Symbol; horizon=0) = SignRestriction(variable, shock, sign, horizon)
function identify_arias(model::VARModel, restrictions::SVARRestrictions, horizon::Int;
                        n_draws=1000, n_rotations=1000)
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
                        tol_coarse=1e-4, tol_fine=1e-8)
    n = size(model.Y, 2)
    Q = Matrix{Float64}(I(n))
    irf_vals = ones(horizon + 1, n, n) * 0.1
    UhligSVARResult(Q, irf_vals, 1e-6, fill(1e-7, n), restrictions, true)
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
    LPModel(Y, shock_var, horizon, lags, ones(lags+1, n)*0.1, ones(T_obs-lags, n)*0.01, Matrix{Float64}(I(n)) * 0.01, T_obs-lags)
end
function lp_irf(model::LPModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, abs.(ones(h, n)) * 0.1)
end
function estimate_lp_iv(Y, shock_var, Z, horizon; lags=4, cov_type=:newey_west)
    T_obs = size(Y, 1)
    LPIVModel(Y, Z, 15.0, horizon, T_obs-4)
end
function lp_iv_irf(model::LPIVModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, ones(h, n) * 0.1)
end
weak_instrument_test(model::LPIVModel; threshold=10.0) = (F_stat=model.first_stage_F, is_weak=model.first_stage_F < threshold)
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
    PropensityLPModel(Y, 0.5, 0.1, horizon)
end
function propensity_irf(model::PropensityLPModel; conf_level=0.95)
    n = size(model.Y, 2); h = model.horizon + 1
    vals = ones(h, n) * 0.1
    LPImpulseResponse(vals, vals .- 0.5, vals .+ 0.5, ones(h, n) * 0.1)
end
propensity_diagnostics(model::PropensityLPModel) =
    (propensity_summary=(treated=(mean=0.7,), control=(mean=0.3,)), balance=(max_weighted=0.05,))
doubly_robust_lp(Y, treatment, covariates, horizon; ps_method=:logit) =
    PropensityLPModel(Y, 0.6, 0.12, horizon)

function structural_lp(Y, horizon; method=:cholesky, lags=4, var_lags=4,
                       cov_type=:newey_west, ci_type=:none, reps=200, conf_level=0.95,
                       check_func=nothing, narrative_check=nothing, max_draws=1000)
    T_obs, n = size(Y); p = var_lags
    model = _mock_var(Y, p)
    irf_vals = ones(horizon + 1, n, n) * 0.1
    ci_lo = ci_type == :none ? nothing : irf_vals .- 0.5
    ci_hi = ci_type == :none ? nothing : irf_vals .+ 0.5
    irf_res = ImpulseResponse(irf_vals, ci_lo, ci_hi)
    Q = Matrix{Float64}(I(n))
    lp_models = [LPModel(Y, i, horizon, lags, ones(5, n)*0.1, ones(T_obs-lags, n)*0.01, Matrix{Float64}(I(n))*0.01, T_obs-lags) for i in 1:n]
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
function estimate_factors(X, r; standardize=true)
    T_obs, n = size(X)
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
function estimate_gdfm(X, q; r=2, standardize=true, bandwidth=0, kernel=:bartlett)
    T_obs, n = size(X)
    GeneralizedDynamicFactorModel(ones(T_obs, n)*0.5, ones(n, q)*0.3)
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

# Unit root / cointegration tests
adf_test(y; lags=:aic, regression=:constant) = ADFResult(-3.5, 0.01, 2)
kpss_test(y; regression=:constant) = KPSSResult(0.3)
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
estimate_ar(y, p; method=:ols) = ARModel(ones(p)*0.3, p, 0.5, -100.0, -95.0, -50.0)
estimate_ma(y, q; method=:css_mle) = MAModel(ones(q)*0.3, q, 0.5, -100.0, -95.0, -50.0)
estimate_arma(y, p, q; method=:css_mle) = ARMAModel(ones(p+q)*0.3, p, q, 0.5, -100.0, -95.0, -50.0)
estimate_arima(y, p, d, q; method=:css_mle) = ARIMAModel(ones(p+q)*0.3, p, d, q, 0.5, -100.0, -95.0, -50.0)
auto_arima(y; max_p=5, max_q=5, max_d=2, criterion=:bic, method=:mle) =
    ARIMAModel(ones(2)*0.3, 1, 1, 1, 0.5, -100.0, -95.0, -50.0)

ar_order(m::ARModel) = m.p;       ar_order(m::MAModel) = 0
ar_order(m::ARMAModel) = m.p;     ar_order(m::ARIMAModel) = m.p
ma_order(m::ARModel) = 0;         ma_order(m::MAModel) = m.q
ma_order(m::ARMAModel) = m.q;     ma_order(m::ARIMAModel) = m.q
diff_order(m::ARModel) = 0;       diff_order(m::MAModel) = 0
diff_order(m::ARMAModel) = 0;     diff_order(m::ARIMAModel) = m.d
aic(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.aic_val
bic(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}) = m.bic_val
function forecast(m::Union{ARModel,MAModel,ARMAModel,ARIMAModel}, h::Int; conf_level=0.95)
    fc = ones(h) * 0.1
    ARIMAForecast(fc, fc .- 0.5, fc .+ 0.5, ones(h) * 0.1, h)
end

# Volatility model functions
estimate_arch(y, q) = ARCHModel(ones(q+2) * 0.1)
estimate_garch(y, p, q) = GARCHModel(ones(p+q+2) * 0.1)
estimate_egarch(y, p, q) = EGARCHModel(ones(2*q+p+2) * 0.1)
estimate_gjr_garch(y, p, q) = GJRGARCHModel(ones(2*q+p+2) * 0.1)
estimate_sv(y; n_samples=5000) = SVModel(ones(3) * 0.1)
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
    VARForecast(fc, fc .- 0.5, fc .+ 0.5, h, ci_method, conf_level,
                ["var$i" for i in 1:n])
end

# Volatility test functions
arch_lm_test(y, lags) = (statistic=15.0, pvalue=0.01)
ljung_box_squared(y, lags) = (statistic=20.0, pvalue=0.005)

# Non-Gaussian identification
function _mock_ica(model::VARModel, method_sym::Symbol)
    n = size(model.Y, 2); T_u = size(model.U, 1)
    ICASVARResult(ones(n,n)*0.3, ones(n,n)*0.3, Matrix{Float64}(I(n)),
                  ones(T_u, n)*0.1, method_sym, true, 50, 0.001)
end
identify_fastica(model::VARModel; contrast=:logcosh, max_iter=200, tol=1e-6) = _mock_ica(model, :fastica)
identify_jade(model::VARModel) = _mock_ica(model, :jade)
identify_sobi(model::VARModel) = _mock_ica(model, :sobi)
identify_dcov(model::VARModel) = _mock_ica(model, :dcov)
identify_hsic(model::VARModel) = _mock_ica(model, :hsic)

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
    NormalityTestSuite([
        NormalityTestResult(:jarque_bera, 15.0, 0.001, 2),
        NormalityTestResult(:skewness, 8.0, 0.02, 1),
        NormalityTestResult(:kurtosis, 3.0, 0.08, 1),
    ])
end
test_identification_strength(model::VARModel) = (statistic=25.0, pvalue=0.001)
test_shock_gaussianity(result::ICASVARResult) = (statistic=12.0, pvalue=0.005)
test_shock_independence(result::ICASVARResult) = (statistic=3.0, pvalue=0.08)
test_overidentification(model::VARModel, result::ICASVARResult) = (statistic=1.5, pvalue=0.45)
test_overidentification(result::ICASVARResult) = (statistic=1.5, pvalue=0.45)
test_gaussian_vs_nongaussian(model::VARModel) = (statistic=18.0, pvalue=0.001)

# ─── VECM Functions ──────────────────────────────────────

function estimate_vecm(Y::AbstractMatrix, p::Int; rank=nothing, deterministic=:constant,
                       method=:johansen, significance=0.05)
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
              -100.0, -95.0, -97.0, -500.0, deterministic, method)
end

select_vecm_rank(Y::AbstractMatrix, p::Int; criterion=:trace, significance=0.05) =
    min(1, size(Y, 2) - 1)

function to_var(vecm::VECMModel)
    _mock_var(vecm.Y, vecm.p)
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
    statistic::T; pvalue::T; df::Int; test_type::Symbol; cause::String; effect::String
end

struct LRTestResult{T<:Real}
    statistic::T; pvalue::T; df::Int; loglik_restricted::T; loglik_unrestricted::T
end

struct LMTestResult{T<:Real}
    statistic::T; pvalue::T; df::Int; nobs::Int; score_norm::T
end

# ─── Panel VAR Functions ────────────────────────────────────

function xtset(data::AbstractMatrix, group_col::AbstractVector, time_col::AbstractVector;
               varnames=nothing)
    T_obs, n = size(data)
    groups = sort(unique(group_col))
    n_groups = length(groups)
    vn = isnothing(varnames) ? ["var$i" for i in 1:n] : varnames
    PanelData(Float64.(data), vn, Int.(group_col), Int.(time_col),
              n_groups, n, T_obs, true)
end

isbalanced(pd::PanelData) = pd.balanced
ngroups(pd::PanelData) = pd.n_groups

function estimate_pvar(panel::PanelData, p::Int;
                       transformation=:fd, steps=:twostep, system=false, collapse=false,
                       dependent=nothing, predetermined=nothing, exogenous=nothing,
                       min_lag_endo=2, max_lag_endo=99)
    n = panel.n_vars
    k = n * p + 1
    Phi = ones(k, n) * 0.3
    Sigma = Matrix{Float64}(I(n)) * 0.01
    se = ones(k, n) * 0.05
    pvals = ones(k, n) * 0.02
    n_inst = system ? 2 * k : k + p
    PVARModel(Phi, Sigma, se, pvals, n, p, :gmm, transformation, steps,
              panel.n_groups, panel.T_obs ÷ panel.n_groups, panel.T_obs, n_inst)
end

function estimate_pvar_feols(panel::PanelData, p::Int;
                              dependent=nothing, exogenous=nothing)
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

function pvar_bootstrap_irf(model::PVARModel, horizon::Int;
                             n_boot=500, conf_level=0.95, irf_type=:oirf)
    n = model.m
    vals = ones(horizon + 1, n, n) * 0.1
    ci_lo = vals .- 0.5
    ci_hi = vals .+ 0.5
    ImpulseResponse(vals, ci_lo, ci_hi)
end

function pvar_fevd(model::PVARModel, horizon::Int)
    n = model.m
    props = ones(n, n, horizon) / n
    FEVD(props, props)
end

function pvar_stability(model::PVARModel)
    n = model.m * model.p
    eigs = [0.5 + 0.1im, 0.5 - 0.1im, 0.3 + 0.0im]
    moduli = abs.(eigs)
    PVARStability(eigs, moduli, all(moduli .< 1.0))
end

function pvar_hansen_j(model::PVARModel)
    PVARTestResult("Hansen J", 8.5, 0.38, model.n_instruments - model.m * model.p - model.m,
                   model.n_instruments, model.m * model.p + model.m)
end

function pvar_mmsc(panel::PanelData, max_p::Int; criterion=:bic)
    results = [(p=p, bic=-100.0+p, aic=-110.0+p, hqic=-105.0+p) for p in 1:max_p]
    (results=results, optimal_lag=1, criterion=criterion)
end

function pvar_lag_selection(panel::PanelData, max_p::Int; criterion=:bic)
    results = [(p=p, bic=-100.0+p, aic=-110.0+p, hqic=-105.0+p) for p in 1:max_p]
    (results=results, optimal_lag=1, criterion=criterion)
end

# Enhanced Granger causality for VAR
function granger_test(model::VARModel, cause::Int, effect::Int; lags=nothing)
    GrangerCausalityResult(12.5, 0.003, 2, :pairwise, "var$cause", "var$effect")
end

function granger_test_all(model::VARModel; lags=nothing)
    n = size(model.Y, 2)
    results = GrangerCausalityResult[]
    for i in 1:n, j in 1:n
        i == j && continue
        push!(results, GrangerCausalityResult(10.0 + i, 0.01, 2, :pairwise, "var$i", "var$j"))
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

export VARModel, MockChains, BVARPosterior, MinnesotaHyperparameters
export ImpulseResponse, BayesianImpulseResponse, FEVD, BayesianFEVD
export HistoricalDecomposition, BayesianHistoricalDecomposition
export ZeroRestriction, SignRestriction, SVARRestrictions, AriasSVARResult, UhligSVARResult
export LPModel, LPIVModel, SmoothLPModel, StateLPModel, PropensityLPModel
export LPImpulseResponse, StructuralLP, LPFEVD, LPForecast
export FactorModel, DynamicFactorModel, GeneralizedDynamicFactorModel, FactorForecast
export ARModel, MAModel, ARMAModel, ARIMAModel, ARIMAForecast
export ICASVARResult, NonGaussianMLResult
export MarkovSwitchingSVARResult, GARCHSVARResult, SmoothTransitionSVARResult, ExternalVolatilitySVARResult
export NormalityTestResult, NormalityTestSuite
export ADFResult, KPSSResult, PPResult, ZAResult, NgPerronResult, JohansenResult
export GMMModel
export ARCHModel, GARCHModel, EGARCHModel, GJRGARCHModel, SVModel, VolatilityForecast
export VECMModel, VECMForecast, VECMGrangerResult
export PanelData, PVARModel, PVARStability, PVARTestResult
export GrangerCausalityResult, LRTestResult, LMTestResult
export HPFilterResult, HamiltonFilterResult, BeveridgeNelsonResult, BaxterKingResult, BoostedHPResult

export select_lag_order, estimate_var, estimate_bvar, posterior_mean_model, posterior_median_model
export optimize_hyperparameters, coef, loglikelihood, stderror, predict, residuals, report
export is_stationary, companion_matrix, companion_matrix_factors, nvars
export irf, fevd, historical_decomposition, verify_decomposition, contribution
export cumulative_irf
export SignIdentifiedSet, identify_sign, irf_bounds, irf_median
export VARForecast
export zero_restriction, sign_restriction, identify_arias, irf_mean, identify_uhlig
export estimate_lp, lp_irf, estimate_lp_iv, lp_iv_irf, weak_instrument_test
export estimate_smooth_lp, smooth_lp_irf, cross_validate_lambda
export estimate_state_lp, state_irf, test_regime_difference
export estimate_propensity_lp, propensity_irf, propensity_diagnostics, doubly_robust_lp
export structural_lp, lp_fevd, forecast
export estimate_factors, ic_criteria, scree_plot_data
export estimate_dynamic_factors, ic_criteria_gdfm, estimate_gdfm, common_variance_share
export adf_test, kpss_test, pp_test, za_test, ngperron_test, johansen_test
export estimate_lp_gmm, gmm_summary, j_test
export estimate_ar, estimate_ma, estimate_arma, estimate_arima, auto_arima
export ar_order, ma_order, diff_order, aic, bic
export estimate_arch, estimate_garch, estimate_egarch, estimate_gjr_garch, estimate_sv
export persistence, halflife, unconditional_variance
export arch_lm_test, ljung_box_squared
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
    tcode::Vector{Int}; time_index::Vector{Int}; desc::String; vardesc::Vector{String}
end
# Keyword constructor matching MacroEconometricModels v0.2.2 interface
function TimeSeriesData(data::AbstractMatrix{T}; varnames=String[], frequency=:unknown,
                        tcode=fill(1, size(data, 2)), time_index=collect(1:size(data, 1)),
                        desc="", vardesc=fill("", size(data, 2)), source_refs=Symbol[]) where T<:Real
    TimeSeriesData{T}(Matrix{T}(data), varnames, frequency, tcode, time_index, desc, vardesc)
end

struct DataDiagnostic
    n_nan::Vector{Int}; n_inf::Vector{Int}; is_constant::Vector{Bool}
    is_short::Bool; is_clean::Bool
end

struct DataSummary{T<:Real}
    n::Int; mean::Vector{T}; std::Vector{T}; min::Vector{T}
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
            "FRED-MD Monthly Database (2024 vintage)", vd)
    elseif name == :fred_qd
        n_vars = 245; T_obs = 268
        data = randn(T_obs, n_vars) .+ 1.0
        vn = ["GDP", "PCECC96", ["var$i" for i in 3:n_vars]...]
        tc = vcat([5, 5], [1 for _ in 3:n_vars])
        vd = vcat(["Real GDP", "Real PCE"], ["Variable $i" for i in 3:n_vars])
        TimeSeriesData(data, vn, :quarterly, tc, collect(1:T_obs),
            "FRED-QD Quarterly Database", vd)
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
    else
        error("unknown dataset: $name (available: fred_md, fred_qd, pwt, mpdta, ddcg)")
    end
end

to_matrix(d::TimeSeriesData) = d.data
to_matrix(d::PanelData) = d.data
varnames(d::TimeSeriesData) = d.varnames
varnames(d::PanelData) = d.varnames
frequency(d::TimeSeriesData) = d.frequency
desc(d::TimeSeriesData) = d.desc
vardesc(d::TimeSeriesData) = d.vardesc
nobs(d::TimeSeriesData) = size(d.data, 1)
nvars(d::TimeSeriesData) = size(d.data, 2)

function describe_data(d::TimeSeriesData)
    T_obs, n = size(d.data)
    m = vec(mean(d.data; dims=1))
    s = vec(std_mock(d.data))
    mn = vec(minimum(d.data; dims=1))
    mx = vec(maximum(d.data; dims=1))
    p25 = m .- 0.67 .* s
    med = copy(m)
    p75 = m .+ 0.67 .* s
    sk = fill(0.1, n)
    ku = fill(3.0, n)
    DataSummary(T_obs, m, s, mn, p25, med, p75, mx, sk, ku)
end

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

function nowcast_bvar(Y::AbstractMatrix, nM::Int, nQ::Int; lags=5, kwargs...)
    T_obs, N = size(Y)
    NowcastBVAR{Float64}(copy(Y), randn(N*lags+1, N), Matrix{Float64}(I(N)),
        0.2, 1.0, 1.0, 2.0, lags, -100.0, nM, nQ, copy(Y))
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
    N = size(X_new, 2)
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

export TimeSeriesData, DataDiagnostic, DataSummary
export load_example, to_matrix, varnames, frequency, desc, vardesc, nobs, nvars
export describe_data, diagnose, fix, apply_tcode, validate_for_model, apply_filter

# ─── DSGE Types ──────────────────────────────────────────────

abstract type AbstractDSGEModel end

struct DSGESpec{T<:Real}
    endog::Vector{Symbol}; exog::Vector{Symbol}; params::Vector{Symbol}
    param_values::Dict{Symbol,T}; n_endog::Int; n_exog::Int; n_params::Int
    varnames::Vector{String}; steady_state::Vector{T}
end
function DSGESpec(; n_endog=3, n_exog=1)
    endog = [Symbol("y$i") for i in 1:n_endog]
    exog = [Symbol("e$i") for i in 1:n_exog]
    params = [:alpha, :beta, :delta]
    param_values = Dict{Symbol,Float64}(:alpha => 0.33, :beta => 0.99, :delta => 0.025)
    varnames = ["y$i" for i in 1:n_endog]
    ss = zeros(Float64, n_endog)
    DSGESpec{Float64}(endog, exog, params, param_values, n_endog, n_exog, length(params),
                      varnames, ss)
end

struct LinearDSGE{T<:Real}
    Gamma0::Matrix{T}; Gamma1::Matrix{T}; C::Vector{T}; Psi::Matrix{T}; Pi::Matrix{T}
    spec::DSGESpec{T}
end

struct DSGESolution{T<:Real}
    G1::Matrix{T}; impact::Matrix{T}; C_sol::Vector{T}; eu::Vector{Int}
    method::Symbol; eigenvalues::Vector{Complex{T}}; spec::DSGESpec{T}; linear::LinearDSGE{T}
end

struct PerturbationSolution{T<:Real}
    order::Int; gx::Matrix{T}; hx::Matrix{T}
    gxx::Union{Nothing,Array{T,3}}; hxx::Union{Nothing,Array{T,3}}
    gσσ::Union{Nothing,Vector{T}}; hσσ::Union{Nothing,Vector{T}}
    eta::Matrix{T}; steady_state::Vector{T}
    state_indices::Vector{Int}; control_indices::Vector{Int}
    eu::Vector{Int}; method::Symbol; spec::DSGESpec{T}; linear::LinearDSGE{T}
end

struct ProjectionSolution{T<:Real}
    coefficients::Matrix{T}; state_bounds::Matrix{T}; grid_type::Symbol; degree::Int
    residual_norm::T; converged::Bool; iterations::Int; method::Symbol
    spec::DSGESpec{T}; linear::LinearDSGE{T}; steady_state::Vector{T}
    state_indices::Vector{Int}; control_indices::Vector{Int}
end

struct PerfectForesightPath{T<:Real}
    path::Matrix{T}; deviations::Matrix{T}; converged::Bool; iterations::Int
    spec::DSGESpec{T}
end

struct DSGEEstimation{T<:Real} <: AbstractDSGEModel
    theta::Vector{T}; vcov::Matrix{T}; param_names::Vector{String}; method::Symbol
    J_stat::T; J_pvalue::T; converged::Bool; spec::DSGESpec{T}
end

struct OccBinConstraint{T<:Real}
    variable::Symbol; bound::T; direction::Symbol
end

struct OccBinSolution{T<:Real}
    linear_path::Matrix{T}; piecewise_path::Matrix{T}; steady_state::Vector{T}
    regime_history::Vector{Int}; converged::Bool; iterations::Int
    spec::DSGESpec{T}; varnames::Vector{String}
    constraints::Vector{OccBinConstraint{T}}
end

struct OccBinIRF{T<:Real}
    linear::Array{T,3}; piecewise::Array{T,3}; regime_history::Vector{Int}
    varnames::Vector{String}; shock_name::String
end

# ─── DSGE Mock Helpers & Functions ───────────────────────────

function _mock_linear(spec::DSGESpec{T}) where T
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

function _mock_solution(spec::DSGESpec{T}; method=:gensys) where T
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

function compute_steady_state(spec::DSGESpec; kwargs...)
    spec
end

function linearize(spec::DSGESpec)
    _mock_linear(spec)
end

function solve(spec::DSGESpec{T}; method=:gensys, order=1, degree=5, grid=:auto, kwargs...) where T
    n = spec.n_endog
    ne = spec.n_exog
    ld = _mock_linear(spec)
    if method == :perturbation
        n_states = max(1, n ÷ 2)
        n_controls = n - n_states
        gx = ones(T, n_controls, n_states) * T(0.1)
        hx = Matrix{T}(I(n_states)) * T(0.5)
        eta = zeros(T, n_states, ne)
        for i in 1:min(n_states, ne); eta[i, i] = T(1.0); end
        gxx = order >= 2 ? zeros(T, n_controls, n_states, n_states) : nothing
        hxx = order >= 2 ? zeros(T, n_states, n_states, n_states) : nothing
        gσσ = order >= 2 ? zeros(T, n_controls) : nothing
        hσσ = order >= 2 ? zeros(T, n_states) : nothing
        ss = zeros(T, n)
        state_idx = collect(1:n_states)
        control_idx = collect(n_states+1:n)
        return PerturbationSolution{T}(order, gx, hx, gxx, hxx, gσσ, hσσ, eta, ss,
            state_idx, control_idx, [1, 1], :perturbation, spec, ld)
    elseif method in (:projection, :pfi)
        n_states = max(1, n ÷ 2)
        n_controls = n - n_states
        coeffs = ones(T, n_controls, degree + 1) * T(0.1)
        bounds = hcat(fill(T(-2.0), n_states), fill(T(2.0), n_states))
        ss = zeros(T, n)
        state_idx = collect(1:n_states)
        control_idx = collect(n_states+1:n)
        return ProjectionSolution{T}(coeffs, bounds, grid == :auto ? :chebyshev : grid, degree,
            T(1e-8), true, 50, method, spec, ld, ss, state_idx, control_idx)
    else
        return _mock_solution(spec; method=method)
    end
end

function gensys(Γ0, Γ1, C, Ψ, Π)
    _mock_solution(DSGESpec())
end

function blanchard_kahn(ld::LinearDSGE, spec::DSGESpec)
    _mock_solution(spec; method=:blanchard_kahn)
end

function klein(Γ0, Γ1, C, Ψ, n_pre)
    _mock_solution(DSGESpec(); method=:klein)
end

function perturbation_solver(spec::DSGESpec; order=1)
    solve(spec; method=:perturbation, order=order)
end

function collocation_solver(spec::DSGESpec; degree=5, kwargs...)
    solve(spec; method=:projection, degree=degree)
end

function pfi_solver(spec::DSGESpec; kwargs...)
    solve(spec; method=:pfi)
end

function perfect_foresight(spec::DSGESpec{T}; shocks=nothing, T_periods=100, kwargs...) where T
    n = spec.n_endog
    path = zeros(T, T_periods, n)
    devs = zeros(T, T_periods, n)
    PerfectForesightPath{T}(path, devs, true, 25, spec)
end

function occbin_solve(spec::DSGESpec{T}, shocks, constraints; T_periods=40, kwargs...) where T
    n = spec.n_endog
    lp = zeros(T, T_periods, n)
    pp = zeros(T, T_periods, n)
    ss = zeros(T, n)
    regimes = ones(Int, T_periods)
    cons = constraints isa Vector ? constraints : [constraints]
    OccBinSolution{T}(lp, pp, ss, regimes, true, 15, spec, spec.varnames, cons)
end

function occbin_irf(spec::DSGESpec{T}, constraints, shock_idx; shock_size=1.0, horizon=40, kwargs...) where T
    n = spec.n_endog
    ne = spec.n_exog
    lin = zeros(T, horizon + 1, n, ne)
    pw = zeros(T, horizon + 1, n, ne)
    # Set decaying response for shock_idx
    for h in 0:horizon
        for v in 1:n
            lin[h+1, v, min(shock_idx, ne)] = T(shock_size) * T(0.9)^h
            pw[h+1, v, min(shock_idx, ne)] = T(shock_size) * T(0.85)^h
        end
    end
    regimes = ones(Int, horizon + 1)
    OccBinIRF{T}(lin, pw, regimes, spec.varnames, "shock$shock_idx")
end

function parse_constraint(expr, spec::DSGESpec)
    OccBinConstraint{Float64}(:i, 0.0, :geq)
end

function variable_bound(var::Symbol; lower=-Inf, upper=Inf)
    if upper < Inf
        OccBinConstraint{Float64}(var, upper, :leq)
    else
        OccBinConstraint{Float64}(var, lower, :geq)
    end
end

function estimate_dsge(spec::DSGESpec{T}, data, param_names; method=:irf_matching, kwargs...) where T
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
function fevd(sol::PerturbationSolution{T}, horizon::Int; kwargs...) where T
    n = sol.spec.n_endog; ne = sol.spec.n_exog
    props = ones(T, n, ne, horizon) / T(ne)
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

export AbstractDSGEModel, DSGESpec, LinearDSGE, DSGESolution, PerturbationSolution
export ProjectionSolution, PerfectForesightPath, DSGEEstimation
export OccBinConstraint, OccBinSolution, OccBinIRF
export compute_steady_state, linearize, solve, gensys, blanchard_kahn, klein
export perturbation_solver, collocation_solver, pfi_solver
export perfect_foresight, occbin_solve, occbin_irf, parse_constraint, variable_bound
export estimate_dsge, simulate, is_determined, is_stable, nshocks

# ─── SMM Types & Functions ───────────────────────────────────

struct SMMModel{T<:Real}
    theta::Vector{T}; vcov::Matrix{T}; n_moments::Int; n_params::Int; n_obs::Int
    J_stat::T; J_pvalue::T; converged::Bool; sim_ratio::Int
end

struct ParameterTransform{T<:Real}
    lower::Vector{T}; upper::Vector{T}
end

function estimate_smm(moment_fn, theta0, data; weighting=:two_step, sim_ratio=5, burn=100, kwargs...)
    np = length(theta0)
    nm = np + 2
    theta = ones(Float64, np) * 0.5
    vcov_mat = Matrix{Float64}(I(np)) * 0.01
    n_obs = size(data, 1)
    SMMModel{Float64}(theta, vcov_mat, nm, np, n_obs, 2.0, 0.7, true, sim_ratio)
end

function autocovariance_moments(data; lags=1)
    zeros(Float64, size(data, 2) * (lags + 1))
end

to_unconstrained(x, t::ParameterTransform) = x
to_constrained(x, t::ParameterTransform) = x
transform_jacobian(x, t::ParameterTransform) = Matrix{Float64}(I(length(x)))

export SMMModel, ParameterTransform
export estimate_smm, autocovariance_moments, to_unconstrained, to_constrained, transform_jacobian

# ─── BVARForecast Type & Forecast Accessors ──────────────────

struct BVARForecast{T<:AbstractFloat}
    forecast::Matrix{T}; ci_lower::Matrix{T}; ci_upper::Matrix{T}
    horizon::Int; ci_method::Symbol; conf_level::T; varnames::Vector{String}
end

point_forecast(f::Union{VARForecast,BVARForecast}) = f.forecast
lower_bound(f::Union{VARForecast,BVARForecast}) = f.ci_lower
upper_bound(f::Union{VARForecast,BVARForecast}) = f.ci_upper
forecast_horizon(f::Union{VARForecast,BVARForecast}) = f.horizon

# BVAR forecast dispatch — returns BVARForecast
function forecast(post::BVARPosterior, h::Int; ci_method=:none, quantiles=[0.16, 0.5, 0.84], conf_level=0.95)
    n = post.n
    fc = ones(h, n) * 0.1
    BVARForecast{Float64}(fc, fc .- 0.5, fc .+ 0.5, h, ci_method, conf_level,
                           ["var$i" for i in 1:n])
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
    coefficients::Vector{T}; se_vec::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}
    event_times::Vector{Int}; reference_period::Int; nobs_h::Vector{Int}
    pooled_post_result::Union{NamedTuple,Nothing}; pooled_pre_result::Union{NamedTuple,Nothing}
    vcov_all::Vector; outcome_name::String; treatment_name::String
    T_obs::Int; n_groups::Int; spec_type::Symbol
    pmd::Union{Nothing,Symbol,Int}; reweight::Bool; nocomp::Bool
    ylags::Int; dylags::Int; pre_window::Int; post_window::Int
    cluster::Symbol; conf_level::T; pd::PanelData{T}
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
        conf_level=0.95, n_boot=200, base_period=:varying) where T
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
    Y::Matrix{T}; p::Int; n_factors::Int; n_key::Int
    factors::Matrix{T}; loadings::Matrix{T}
    varnames::Vector{String}; panel_varnames::Vector{String}
    n_draws::Int
end

function estimate_favar(X::Matrix{T}, key_indices::Vector{Int}, r::Int, p::Int;
                        method=:two_step, n_draws=5000, panel_varnames=nothing) where T
    n_obs, n_vars = size(X)
    n_key = length(key_indices)
    n_aug = r + n_key
    Y = X[p+1:end, 1:min(n_aug, n_vars)]
    B = ones(T, n_aug * p + 1, n_aug) * T(0.1)
    U = randn(T, n_obs - p, n_aug)
    Sigma = Matrix{T}(I(n_aug)) * T(0.5)
    factors = randn(T, n_obs, r)
    loadings = randn(T, n_vars, r)
    vnames = ["aug$i" for i in 1:n_aug]
    pvnames = panel_varnames === nothing ? ["var$i" for i in 1:n_vars] : panel_varnames
    if method == :bayesian
        return BayesianFAVAR{T}(Y, p, r, n_key, factors, loadings, vnames, pvnames, n_draws)
    end
    FAVARModel{T}(Y, p, B, U, Sigma, factors, loadings, r, n_key,
                   T(-100.0), T(-95.0), T(-90.0), vnames, pvnames)
end

function to_var(favar::FAVARModel{T}) where T
    n = size(favar.Y, 2)
    VARModel{T}(favar.Y, favar.p, favar.B, favar.U, favar.Sigma,
                favar.aic, favar.bic, T(-92.0))
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
                    h, :none, T(0.95), favar.panel_varnames)
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
end

function estimate_structural_dfm(X::Matrix{T}, q::Int;
        identification=:cholesky, p=1, H=40, sign_check=nothing,
        max_draws=1000, standardize=true, bandwidth=0, kernel=:bartlett) where T
    n_obs, n_vars = size(X)
    gdfm = estimate_gdfm(X, q; standardize=standardize, bandwidth=bandwidth, kernel=kernel)
    factor_Y = randn(T, n_obs - p, q)
    B_fvar = ones(T, q * p + 1, q) * T(0.1)
    U_fvar = randn(T, n_obs - p, q)
    Sigma_fvar = Matrix{T}(I(q)) * T(0.5)
    fvar = VARModel{T}(factor_Y, p, B_fvar, U_fvar, Sigma_fvar, T(-50.0), T(-48.0), T(-45.0))
    B0 = Matrix{T}(I(q))
    Q_mat = Matrix{T}(I(q))
    loadings_td = randn(T, n_vars, q)
    s_irf = ones(T, H + 1, n_vars, q) * T(0.05)
    snames = ["structural_shock_$i" for i in 1:q]
    StructuralDFM{T}(gdfm, fvar, B0, Q_mat, identification, s_irf, loadings_td, p, snames)
end

function irf(sdfm::StructuralDFM{T}, horizon::Int; kwargs...) where T
    n_vars = size(sdfm.loadings_td, 1)
    q = size(sdfm.B0, 1)
    h = min(horizon, size(sdfm.structural_irf, 1) - 1)
    vals = sdfm.structural_irf[1:h+1, :, :]
    vnames = ["var$i" for i in 1:n_vars]
    ImpulseResponse(vals, nothing, nothing, h, vnames, sdfm.shock_names, :structural_dfm)
end

function fevd(sdfm::StructuralDFM{T}, horizon::Int; kwargs...) where T
    q = size(sdfm.B0, 1)
    props = ones(T, q, q, horizon) / T(q)
    FEVD(props, props)
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
    spec::DSGESpec{T}
    solution::DSGESolution{T}
end

function estimate_dsge_bayes(spec::DSGESpec{T}, data::Matrix, theta0::Vector;
        priors=Dict(), method=:smc, observables=Symbol[],
        n_smc=5000, n_particles=500, n_mh_steps=1,
        n_draws=10000, burnin=5000, ess_target=0.5,
        measurement_error=nothing, solver=:gensys,
        solver_kwargs=NamedTuple(), delayed_acceptance=false,
        n_screen=200, rng=nothing) where T
    np = length(theta0)
    draws = randn(T, n_draws, np) .* T(0.01) .+ theta0'
    log_post = fill(T(-100.0), n_draws)
    pnames = ["param_$i" for i in 1:np]
    ess_hist = fill(T(n_smc * 0.8), 20)
    sol = solve(spec; method=:gensys)
    BayesianDSGE{T}(draws, log_post, pnames, T(-500.0), method, T(0.25), ess_hist, spec, sol)
end

export BayesianDSGE, estimate_dsge_bayes

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
    cips::T; pvalue::T; individual_cadf::Vector{T}
    critical_values::Dict{Int,T}; lags::Int; deterministic::Symbol
    nobs::Int; n_units::Int
end

struct MoonPerronResult{T<:AbstractFloat}
    t_a_statistic::T; t_b_statistic::T; pvalue_a::T; pvalue_b::T
    n_factors::Int; nobs::Int; n_units::Int
end

struct FactorBreakResult{T<:AbstractFloat}
    statistic::T; pvalue::T; break_date::Int; method::Symbol
    r::Int; nobs::Int; n_units::Int
end

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
    FactorBreakResult{T}(T(8.5), T(0.03), div(n_obs, 2), method, r, n_obs, n_units)
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

# ─── Cross-Sectional Regression Types & Functions ──────────────────

struct RegModel{T<:Real}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}; ssr::T; tss::T; r2::T; adj_r2::T
    f_stat::T; f_pvalue::T; loglik::T; aic::T; bic::T
    nobs::Int; rank::Int; dof_resid::Int; cov_type::Symbol
    weights::Union{Vector{T},Nothing}; varnames::Vector{String}
    clusters::Union{Vector{Int},Nothing}; method::Symbol
    Z::Union{Matrix{T},Nothing}; endogenous::Union{Vector{Int},Nothing}
    first_stage_f::Union{T,Nothing}; sargan_stat::Union{T,Nothing}; sargan_pval::Union{T,Nothing}
end

struct LogitModel{T<:Real}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}; loglik::T; loglik_null::T; pseudo_r2::T
    aic::T; bic::T; nobs::Int; varnames::Vector{String}
    converged::Bool; iterations::Int; cov_type::Symbol
end

struct ProbitModel{T<:Real}
    y::Vector{T}; X::Matrix{T}; beta::Vector{T}; var_beta::Matrix{T}
    residuals::Vector{T}; fitted::Vector{T}; loglik::T; loglik_null::T; pseudo_r2::T
    aic::T; bic::T; nobs::Int; varnames::Vector{String}
    converged::Bool; iterations::Int; cov_type::Symbol
end

struct MarginalEffects{T<:Real}
    effects::Vector{T}; se::Vector{T}; z_stat::Vector{T}; p_values::Vector{T}
    ci_lower::Vector{T}; ci_upper::Vector{T}; varnames::Vector{String}
    type::Symbol; conf_level::T
end

# StatsAPI dispatches for RegModel
coef(m::RegModel) = m.beta
vcov(m::RegModel) = m.var_beta
residuals(m::RegModel) = m.residuals
predict(m::RegModel) = m.fitted
stderror(m::RegModel) = [sqrt(m.var_beta[i,i]) for i in 1:size(m.var_beta, 1)]
nobs(m::RegModel) = m.nobs
loglikelihood(m::RegModel) = m.loglik
aic(m::RegModel) = m.aic
bic(m::RegModel) = m.bic
r2(m::RegModel) = m.r2
confint(m::RegModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# StatsAPI dispatches for LogitModel
coef(m::LogitModel) = m.beta
vcov(m::LogitModel) = m.var_beta
residuals(m::LogitModel) = m.residuals
predict(m::LogitModel) = m.fitted
stderror(m::LogitModel) = [sqrt(m.var_beta[i,i]) for i in 1:size(m.var_beta, 1)]
nobs(m::LogitModel) = m.nobs
loglikelihood(m::LogitModel) = m.loglik
aic(m::LogitModel) = m.aic
bic(m::LogitModel) = m.bic
r2(m::LogitModel) = m.pseudo_r2
confint(m::LogitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# StatsAPI dispatches for ProbitModel
coef(m::ProbitModel) = m.beta
vcov(m::ProbitModel) = m.var_beta
residuals(m::ProbitModel) = m.residuals
predict(m::ProbitModel) = m.fitted
stderror(m::ProbitModel) = [sqrt(m.var_beta[i,i]) for i in 1:size(m.var_beta, 1)]
nobs(m::ProbitModel) = m.nobs
loglikelihood(m::ProbitModel) = m.loglik
aic(m::ProbitModel) = m.aic
bic(m::ProbitModel) = m.bic
r2(m::ProbitModel) = m.pseudo_r2
confint(m::ProbitModel; level=0.95) = hcat(m.beta .- 1.96 .* stderror(m), m.beta .+ 1.96 .* stderror(m))

# Mock functions

function estimate_reg(y::AbstractVector{T}, X::AbstractMatrix{T};
                      cov_type=:hc1, weights=nothing, varnames=nothing,
                      clusters=nothing) where T
    n, k = size(X)
    beta = ones(T, k) * T(0.5)
    var_beta = Matrix{T}(I(k)) * T(0.01)
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
    RegModel{T}(y, X, beta, var_beta, resids, fitted_vals, ssr, tss,
                r2_val, adj_r2_val, f_val, f_p, ll, aic_val, bic_val,
                n, k, n - k, cov_type, weights, vnames, clusters, :ols,
                nothing, nothing, nothing, nothing, nothing)
end

function estimate_iv(y::AbstractVector{T}, X::AbstractMatrix{T}, Z::AbstractMatrix{T};
                     endogenous=Int[], cov_type=:hc1, varnames=nothing) where T
    n, k = size(X)
    beta = ones(T, k) * T(0.5)
    var_beta = Matrix{T}(I(k)) * T(0.01)
    fitted_vals = X * beta
    resids = y .- fitted_vals
    ssr = sum(resids .^ 2)
    tss = sum((y .- mean(y)) .^ 2)
    r2_val = one(T) - ssr / tss
    adj_r2_val = one(T) - (one(T) - r2_val) * (n - 1) / (n - k)
    f_val = T(20.0)
    f_p = T(0.002)
    ll = T(-105.0)
    aic_val = T(220.0)
    bic_val = T(230.0)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    first_f = T(15.0)
    sargan_s = T(2.5)
    sargan_p = T(0.30)
    RegModel{T}(y, X, beta, var_beta, resids, fitted_vals, ssr, tss,
                r2_val, adj_r2_val, f_val, f_p, ll, aic_val, bic_val,
                n, k, n - k, cov_type, nothing, vnames, nothing, :iv,
                Z, endogenous, first_f, sargan_s, sargan_p)
end

function _build_logit_probit(::Type{M}, y::AbstractVector{T}, X::AbstractMatrix{T};
                             cov_type=:ols, varnames=nothing, clusters=nothing,
                             maxiter=100, tol=1e-8) where {T, M}
    n, k = size(X)
    beta = ones(T, k) * T(0.3)
    var_beta = Matrix{T}(I(k)) * T(0.02)
    fitted_vals = ones(T, n) * T(0.5)
    resids = y .- fitted_vals
    ll = T(-80.0)
    ll_null = T(-100.0)
    pseudo = one(T) - ll / ll_null
    aic_val = T(170.0)
    bic_val = T(180.0)
    vnames = varnames === nothing ? ["x$i" for i in 1:k] : varnames
    M{T}(y, X, beta, var_beta, resids, fitted_vals, ll, ll_null, pseudo,
          aic_val, bic_val, n, vnames, true, 5, cov_type)
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
    (odds_ratio=or, ci_lower=ci_lo, ci_upper=ci_hi, varnames=m.varnames)
end

function vif(m::RegModel{T}) where T
    k = length(m.beta)
    fill(T(2.5), k)
end

function classification_table(m::Union{LogitModel,ProbitModel}; threshold=0.5)
    Dict("accuracy" => 0.85, "precision" => 0.80, "recall" => 0.75,
         "f1" => 0.77, "true_positive" => 30, "true_negative" => 55,
         "false_positive" => 8, "false_negative" => 10, "threshold" => threshold)
end

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
    mgls = Dict(:MZa => T(-15.0), :MZt => T(-2.7), :MSB => T(0.18), :MPT => T(3.5))
    cvs = Dict(1 => T(-3.48), 5 => T(-2.89), 10 => T(-2.57))
    DFGLSResult{T}(T(-3.2), T(4.5), mgls, T(0.02), p, regression, cvs, n)
end

function lm_unitroot_test(y::AbstractVector{T};
        breaks=0, regression=:level, lags=:aic,
        max_lags=nothing, trim=0.15) where T
    n = length(y)
    p = lags == :aic ? max(1, round(Int, n^(1/3))) : lags
    cvs = Dict(1 => T(-4.24), 5 => T(-3.57), 10 => T(-3.21))
    bi = breaks > 0 ? [div(n, i + 1) for i in 1:breaks] : nothing
    bf = breaks > 0 ? [T(1.0 / (i + 1)) for i in 1:breaks] : nothing
    LMUnitRootResult{T}(T(-3.8), T(0.03), bi, bf, breaks, regression, cvs, p, n)
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
    n = size(Y, 1)
    bp = div(n, 2)
    cvs = Dict(1 => T(-5.13), 5 => T(-4.61), 10 => T(-4.34))
    GregoryHansenResult{T}(T(-4.8), T(0.03), bp,
        T(-4.5), T(0.04), bp + 2,
        T(-35.0), T(0.02), bp - 1,
        model, cvs, n)
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
    BayesianImpulseResponse{T}(q, m, horizon,
        String.(result.param_names), ["shock$i" for i in 1:ns], T.(quantiles))
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
    BayesianFEVD{T}(q, m, horizon,
        String.(result.param_names), ["shock$i" for i in 1:ns], T.(quantiles))
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
verify_decomposition(hd::BayesianDSGEHistoricalDecomposition) = true

function dsge_particle_smoother(args...; kwargs...)
    nothing
end

export KalmanSmootherResult, BayesianDSGEHistoricalDecomposition
export dsge_smoother, dsge_particle_smoother
export total_shock_contribution

end # module
