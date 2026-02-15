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
end
struct BayesianImpulseResponse{T}
    mean::Array{T,3}; quantiles::Array{T,4}; quantile_levels::Vector{T}
end

struct FEVD{T}
    decomposition::Array{T,3}; proportions::Array{T,3}
end
struct BayesianFEVD{T}
    mean::Array{T,3}; quantiles::Array{T,4}; quantile_levels::Vector{T}
end

struct HistoricalDecomposition{T}
    contributions::Array{T,3}; initial_conditions::Matrix{T}; actual::Matrix{T}
    shocks::Matrix{T}; T_eff::Int
end
struct BayesianHistoricalDecomposition{T}
    mean::Array{T,3}; initial_mean::Matrix{T}; quantiles::Array{T,4}; quantile_levels::Vector{T}
end

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
    R2::Array{T,3}; lp_a::Array{T,3}; lp_b::Array{T,3}
    bias_corrected::Array{T,3}; bootstrap_se::Array{T,3}
    horizons::Int; variables::Int; shocks::Int
end
struct LPForecast{T}
    forecasts::Matrix{T}; ci_lower::Matrix{T}; ci_upper::Matrix{T}
    se::Matrix{T}; horizon::Int
end

# ─── Factor Types ─────────────────────────────────────────

struct FactorModel{T}
    data::Matrix{T}; factors::Matrix{T}; loadings::Matrix{T}; eigenvalues::Vector{T}
end
struct DynamicFactorModel{T}
    factors::Matrix{T}; loadings::Matrix{T}; coefficients::Matrix{T}
end
struct GeneralizedDynamicFactorModel{T}
    common_component::Matrix{T}; loadings::Matrix{T}
end
struct FactorForecast{T}
    factors::Matrix{T}; observables::Matrix{T}
    observables_lower::Union{Matrix{T},Nothing}; observables_upper::Union{Matrix{T},Nothing}
    observables_se::Union{Matrix{T},Nothing}; horizon::Int; conf_level::T
end

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
    forecast::Vector{T}; ci_lower::Vector{T}; ci_upper::Vector{T}; se::Vector{T}; horizon::Int
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
    coefficients::Vector{T}
end
struct GARCHModel{T<:Real}
    coefficients::Vector{T}
end
struct EGARCHModel{T<:Real}
    coefficients::Vector{T}
end
struct GJRGARCHModel{T<:Real}
    coefficients::Vector{T}
end
struct SVModel{T<:Real}
    coefficients::Vector{T}
end
struct VolatilityForecast{T<:Real}
    forecast::Vector{T}; horizon::Int
end

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

estimate_bvar(Y, p; sampler=:nuts, n_samples=1000, n_draws=1000, prior=:normal, hyper=nothing) =
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
residuals(m::FactorModel) = m.data .- m.factors * m.loadings'  # T × n idiosyncratic
residuals(m::DynamicFactorModel) = ones(size(m.factors, 1), size(m.loadings, 1)) * 0.01
residuals(m::GeneralizedDynamicFactorModel) = ones(size(m.common_component)) * 0.01

# Volatility model predict/residuals
predict(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}) = fill(0.01, 50)
residuals(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}) = fill(0.01, 50)

report(::VARModel) = nothing
report(::ImpulseResponse) = nothing
report(::BayesianImpulseResponse) = nothing
report(::FEVD) = nothing
report(::BayesianFEVD) = nothing
report(::HistoricalDecomposition) = nothing
report(::BayesianHistoricalDecomposition) = nothing

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
companion_matrix_factors(m::DynamicFactorModel) = m.coefficients
nvars(m::VARModel) = size(m.Y, 2)

# IRF
function irf(model::VARModel, horizon::Int; method=:cholesky, check_func=nothing,
             narrative_check=nothing, ci_type=:none, reps=200, conf_level=0.95)
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
    n = size(model.data, 2)
    obs = ones(h, n) * 0.1
    has_ci = ci_method != :none
    FactorForecast(ones(h, size(model.factors,2))*0.1, obs,
        has_ci ? obs .- 0.5 : nothing, has_ci ? obs .+ 0.5 : nothing,
        has_ci ? ones(h, n)*0.1 : nothing, h, conf_level)
end
function forecast(model::DynamicFactorModel, h::Int; ci=false)
    r = size(model.factors, 2)
    (factors=ones(h, r) * 0.1,)
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
coef(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}) = m.coefficients
persistence(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}) = 0.85
halflife(m::Union{GARCHModel,GJRGARCHModel}) = 4.3
unconditional_variance(m::Union{ARCHModel,GARCHModel}) = 0.02
function forecast(m::Union{ARCHModel,GARCHModel,EGARCHModel,GJRGARCHModel,SVModel}, h::Int)
    VolatilityForecast(ones(h) * 0.01, h)
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

function beveridge_nelson(y::AbstractVector; p=:auto, q=:auto, max_terms=500)
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
export ZeroRestriction, SignRestriction, SVARRestrictions, AriasSVARResult
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
export zero_restriction, sign_restriction, identify_arias, irf_mean
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
    else
        error("unknown dataset: $name (available: fred_md, fred_qd, pwt)")
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

export TimeSeriesData, DataDiagnostic, DataSummary
export load_example, to_matrix, varnames, frequency, desc, vardesc, nobs, nvars
export describe_data, diagnose, fix, apply_tcode, validate_for_model, apply_filter

end # module
