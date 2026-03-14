# MacroEconometricModels.jl API Reference

Upstream library (v0.4.0): 350+ exports. Full docs: https://friedmanjp.github.io/MacroEconometricModels.jl/dev/
Deps: DataFrames, Distributions, LinearAlgebra, Optim, MCMCChains, PrettyTables, Random, SparseArrays, SpecialFunctions, Statistics, StatsAPI, Turing. Weak: FFTW, JuMP/Ipopt/PATHSolver.

## Key Types

**VAR/BVAR:** `VARModel{T}` (Y, p, B, U, Sigma, aic, bic, hqic), `MinnesotaHyperparameters` (tau, decay, lambda, mu, omega), `VARForecast{T}`, `BVARForecast{T}` — use `point_forecast()`, `lower_bound()`, `upper_bound()`, `forecast_horizon()` accessors
**IRF/FEVD/HD:** `ImpulseResponse{T}` (values::Array{T,3}, ci_lower, ci_upper), `BayesianImpulseResponse{T}` (quantiles::Array{T,4}, mean), `FEVD{T}` (proportions::Array{T,3}), `BayesianFEVD{T}`, `HistoricalDecomposition{T}` (contributions::Array{T,3}, initial_conditions, T_eff), `BayesianHistoricalDecomposition{T}`
**SVAR:** `SVARRestrictions`, `AriasSVARResult{T}` (Q_draws, irf_draws, weights, acceptance_rate), `UhligSVARResult{T}` (Q, irf, penalty, converged), `ICASVARResult{T}`, `NonGaussianMLResult{T}`, `MarkovSwitchingSVARResult{T}`, `GARCHSVARResult{T}`, `SmoothTransitionSVARResult{T}`, `ExternalVolatilitySVARResult{T}`
**LP:** `LPModel{T}`, `LPIVModel{T}`, `SmoothLPModel{T}`, `StateLPModel{T}`, `PropensityLPModel{T}`, `StructuralLP{T}`, `LPFEVD{T}`, `LPForecast{T}` (.forecast field), `LPImpulseResponse{T}`
**Factor:** `FactorModel{T}`, `DynamicFactorModel{T}`, `GeneralizedDynamicFactorModel{T}`, `FactorForecast{T}`
**GMM/SMM:** `GMMModel{T}` (theta, vcov, J_stat, J_pvalue, converged), `SMMModel{T}` (+sim_ratio)
**ARIMA:** `ARModel{T}`, `MAModel{T}`, `ARMAModel{T}`, `ARIMAModel{T}`, `ARIMAForecast{T}`
**Volatility:** `ARCHModel{T}`, `GARCHModel{T}`, `EGARCHModel{T}`, `GJRGARCHModel{T}`, `SVModel{T}`, `VolatilityForecast{T}`
**Tests:** `NormalityTestSuite{T}`, `JohansenResult{T}`, `ADFResult`, `KPSSResult`, `PPResult`, `ZAResult`, `NgPerronResult`
**VECM:** `VECMModel{T}`, `VECMForecast{T}`
**Panel:** `PanelData{T}`, `PVARModel{T}`, `PVARStability{T}`, `PVARTestResult{T}`
**Data:** `TimeSeriesData{T}` (keyword ctor), `DataDiagnostic`, `DataSummary{T}`
**Nowcast:** `NowcastDFM`, `NowcastBVAR`, `NowcastBridge`, `NowcastResult`, `NowcastNews`
**Filter:** `HPFilterResult`, `HamiltonFilterResult`, `BeveridgeNelsonResult`, `BaxterKingResult`, `BoostedHPResult`
**DSGE:** `DSGESpec{T}`, `LinearDSGE{T}`, `DSGESolution{T}` (G1, impact, eigenvalues), `PerturbationSolution{T}` (gx, hx), `ProjectionSolution{T}`, `PerfectForesightPath{T}`, `DSGEEstimation{T}`, `OccBinConstraint{T}`, `OccBinSolution{T}`, `OccBinIRF{T}` (linear, piecewise, shock_name)
**DID:** `DIDResult{T}` (att, se, ci_lower, ci_upper, event_times, overall_att, group_time_att, cohorts), `EventStudyLP{T}`, `LPDiDResult{T}` (se_vec, nobs_h, outcome_name, spec_type), `BaconDecomposition{T}`, `PretrendTestResult{T}`, `NegativeWeightResult{T}`, `HonestDiDResult{T}`
**FAVAR:** `FAVARModel{T}`, `BayesianFAVAR{T}` — FAVAR estimation (two-step/Bayesian)
**Structural DFM:** `StructuralDFM{T}` — structural dynamic factor model
**Bayesian DSGE:** `BayesianDSGE{T}` — Bayesian DSGE posterior (SMC/SMC²/MH)
**Structural breaks:** `AndrewsResult{T}` (sup/exp/mean statistics), `BaiPerronResult{T}` (break dates, segments)
**Regression:** `RegModel{T}` (OLS/WLS), `IVModel{T}` (2SLS), `LogitModel{T}`, `ProbitModel{T}`, `MarginalEffects{T}` — cross-sectional regression types
**Advanced unit root:** `FourierADFResult{T}`, `FourierKPSSResult{T}`, `DFGLSResult{T}`, `LMUnitRootResult{T}`, `ADF2BreakResult{T}`, `GregoryHansenResult{T}`
**Panel unit root:** `PANICResult{T}`, `PesaranCIPSResult{T}`, `MoonPerronResult{T}`, `FactorBreakResult{T}`
**Panel regression (v0.4.0):** `PanelRegModel{T}` (beta, var_beta, within_r2, effect_type), `PanelIVModel{T}` (beta, var_beta, first_stage_f, sargan_stat), `PanelLogitModel{T}`, `PanelProbitModel{T}` — panel FE/RE/pooled regression types
**Ordered/multinomial choice (v0.4.0):** `OrderedLogitModel{T}` (beta, thresholds/cutpoints, categories), `OrderedProbitModel{T}`, `MultinomialLogitModel{T}` (beta::Matrix, base_category) — discrete choice models
**Spectral analysis (v0.4.0):** `ACFResult{T}` (acf, pacf, lags, ci, q_stats, q_pvals), `SpectralDensityResult{T}` (frequencies, density, log_density), `CrossSpectrumResult{T}` (coherence, phase, gain), `TransferFunctionResult{T}` (gain, phase, coherence)
**Panel diagnostics (v0.4.0):** `PanelTestResult{T}` — used by hausman, breusch-pagan, pesaran-cd, wooldridge, modified-wald, f-fe, fisher, brant, hausman-iia tests

## Functions by Domain

**VAR estimation:**
`estimate_var(Y, p)` → VARModel | `estimate_bvar(Y, p; n_samples, n_adapts, prior, hyper, sampler)` → Chains (sampler: :nuts/:hmc/:smc)
`select_lag_order(Y, max_p; criterion)` → Int | `companion_matrix(model)` | `is_stationary(model)`
`posterior_mean_model(chain, p, n; data)` | `posterior_median_model(chain, p, n; data)` → VARModel
`MinnesotaHyperparameters(; tau, decay, lambda, mu, omega)` | `optimize_hyperparameters(Y, p)` | `optimize_hyperparameters_full(Y, p)` | `gen_dummy_obs(Y, p, hyper)` | `log_marginal_likelihood(Y, p, hyper)`

**Structural identification:**
`identify_cholesky(model)`, `identify_sign(model, horizon, check_func)`, `identify_narrative(model, horizon, sign_check, narrative_check)`, `identify_long_run(model)`
Arias: `zero_restriction(var, shock)`, `sign_restriction(var, shock, sign)`, `SVARRestrictions(n; zeros, signs)`, `identify_arias(model, restrictions, horizon; n_draws, n_rotations)` → AriasSVARResult, `identify_arias_bayesian(chain, p, n, restrictions, horizon; data, n_rotations, quantiles)`, `irf_mean(result)`, `irf_percentiles(result; probs)`
Uhlig: `identify_uhlig(model, restrictions, horizon; n_starts, n_refine, max_iter_coarse, max_iter_fine, tol_coarse, tol_fine)` → UhligSVARResult

**Non-Gaussian SVAR:**
ICA: `identify_fastica(model; contrast)`, `identify_jade`, `identify_sobi`, `identify_dcov`, `identify_hsic`
ML: `identify_nongaussian_ml(model; distribution)`, `identify_student_t`, `identify_mixture_normal`, `identify_pml`, `identify_skew_normal`
Heterosked: `identify_markov_switching(model; n_regimes)`, `identify_garch(model)`, `identify_smooth_transition(model, transition_var)`, `identify_external_volatility(model, regime_indicator)`
Tests: `normality_test_suite`, `test_identification_strength`, `test_shock_gaussianity`, `test_gaussian_vs_nongaussian`, `test_shock_independence`, `test_overidentification`

**IRF/FEVD/HD:**
`irf(model::VARModel, horizon; method, check_func, narrative_check, ci_type, reps, conf_level, stationary_only)` — method: :cholesky/:sign/:narrative/:long_run, ci_type: :none/:bootstrap/:theoretical
`irf(chain::Chains, p, n, horizon; method, data, quantiles)` → BayesianImpulseResponse
`irf(sol::DSGESolution, horizon; shock_size, n_sim)` | `irf(sol::PerturbationSolution, horizon)`
`irf_median(result)`, `irf_bounds(result)`, `cumulative_irf(irf)`
`fevd(model, horizon; method)` | `fevd(chain, p, n, horizon; quantiles)` | `fevd(sol, horizon)`
`historical_decomposition(model, horizon; method)` | `historical_decomposition(chain, p, n, horizon; data, quantiles)` | `historical_decomposition(model, restrictions, horizon)`
`historical_decomposition(sol::DSGESolution, data; horizon)` | `historical_decomposition(bd::BayesianDSGE, data; horizon, quantiles)` — DSGE HD (v0.4.0)
`contribution(hd, var, shock)`, `verify_decomposition(hd)`

**LP:**
`estimate_lp(Y, shock_var, horizon; lags, cov_type)`, `structural_lp(Y, horizon; method, lags, var_lags, cov_type, ci_type, reps, conf_level, check_func, narrative_check, max_draws)` → StructuralLP
`lp_irf(model)`, `lp_fevd(slp, horizons; estimator, n_boot)` → LPFEVD, `historical_decomposition(slp, T_hd)`
`forecast(model::LPModel, shock_path)` | `forecast(slp, shock_idx, shock_path)` → LPForecast
`estimate_lp_iv(Y, shock_var, instruments, horizon)`, `weak_instrument_test(model)`, `sargan_test(model, h)`
`estimate_smooth_lp(Y, shock_var, horizon; degree, n_knots, lambda)`, `cross_validate_lambda(Y, shock_var, horizon)`
`estimate_state_lp(Y, shock_var, state_var, horizon; gamma)`, `state_irf(model)`, `test_regime_difference(model)`
`estimate_propensity_lp(Y, treatment, covariates, horizon)`, `doubly_robust_lp(Y, treatment, covariates, horizon)`

**GMM/SMM:**
`estimate_gmm(moment_fn, theta0, data; weighting, hac)` | `estimate_lp_gmm(Y, shock_var, horizon)` | `j_test(model)`
`estimate_smm(moment_fn, theta0, Y; weighting, sim_ratio, burn)` | `autocovariance_moments(Y; lags)`

**ARIMA:**
`estimate_ar(y, p)`, `estimate_ma(y, q)`, `estimate_arma(y, p, q)`, `estimate_arima(y, p, d, q)`, `auto_arima(y; max_p, max_q, max_d, criterion)` → ARIMAModel
`forecast(model, h; conf_level)` → ARIMAForecast

**Volatility:** `estimate_arch`, `estimate_garch`, `estimate_egarch`, `estimate_gjr_garch`, `estimate_sv` — all `(y; ...)` → respective model types

**VAR/BVAR Forecast:**
`forecast(model::VARModel, horizon; ci_method, reps)` → VARForecast | `forecast(chain, p, n, horizon; data, quantiles)` → BVARForecast
Accessors: `point_forecast(fc)`, `lower_bound(fc)`, `upper_bound(fc)`, `forecast_horizon(fc)`

**VECM:**
`estimate_vecm(Y, p; rank, deterministic, method, significance)` | `select_vecm_rank(Y, p)` | `to_var(vecm)` | `granger_causality_vecm(vecm, cause, effect)`

**Panel VAR:**
`xtset(data, group_col, time_col)` → PanelData | `estimate_pvar(panel, p; transformation, steps, system, collapse, dependent, predetermined, exogenous)` | `estimate_pvar_feols(panel, p)`
`pvar_oirf(model, h)`, `pvar_girf(model, h)`, `pvar_bootstrap_irf(model, h; n_boot, conf_level, irf_type)`, `pvar_fevd(model, h)`, `pvar_stability(model)`, `pvar_hansen_j(model)`, `pvar_mmsc(panel, max_p)`, `pvar_lag_selection(panel, max_p)`
`granger_test(model, cause, effect)`, `granger_test_all(model)`, `lr_test(m1, m2)`, `lm_test(m1, m2)`

**Unit root & cointegration:**
`adf_test(y; lags, regression)`, `kpss_test(y)`, `pp_test(y)`, `za_test(y; trim)`, `ngperron_test(y)`, `johansen_test(Y, p; deterministic)`
`arch_lm_test(residuals; lags)`, `ljung_box_squared(residuals; lags)`

**Structural breaks:**
`andrews_test(y, X; test, trimming)` → AndrewsResult — test: :wald/:lr/:lm, reports sup/exp/mean statistics
`bai_perron_test(y, X; max_breaks, trimming, criterion)` → BaiPerronResult — criterion: :bic/:lwz, reports break dates + segment estimates

**Panel unit root:**
`panic_test(X; r, method)` → PANICResult | `pesaran_cips_test(X; lags, deterministic)` → PesaranCIPSResult
`moon_perron_test(X; r)` → MoonPerronResult | `factor_break_test(X, r; method)` → FactorBreakResult
`panel_unit_root_summary(X; tests)` — convenience summary across multiple panel tests

**Filters:**
`hp_filter(y; lambda)`, `hamilton_filter(y; h, p)`, `beveridge_nelson(y; p, q, method)` (method: :arima/:statespace), `baxter_king(y; pl, pu, K)`, `boosted_hp(y; lambda, stopping, max_iter)`
`trend(result)`, `cycle(result)` — accessors (may return valid-range-only for Hamilton/BK)
`apply_filter(ts::TimeSeriesData, method; kwargs...)` — unified interface

**Factor models:**
`estimate_factors(X, r)` | `ic_criteria(X, max_factors)` → .r_IC1/.r_IC2/.r_IC3
`estimate_dynamic_factors(X, r, p)` | `ic_criteria_dynamic(X, max_r, max_p)` → .r_opt
`estimate_gdfm(X, q; bandwidth, kernel, r)` | `ic_criteria_gdfm(X, max_q)` → .q_opt/.r_opt
`forecast(model::FactorModel, h; ci_method)` → FactorForecast | `forecast(model::DynamicFactorModel, h)`

**FAVAR:**
`estimate_favar(X, key_indices, r, p; method, n_draws)` → FAVARModel | BayesianFAVAR
`favar_panel_irf(favar, irf_result)` → ImpulseResponse — panel-wide IRF via loadings
`favar_panel_forecast(favar, fc)` → VARForecast — panel-wide forecast via loadings

**Structural DFM:**
`estimate_structural_dfm(X, q; identification, p, H, sign_check)` → StructuralDFM

**Data:**
`TimeSeriesData(data; varnames, frequency, tcode, time_index)` | `load_example(name)` — :fred_md/:fred_qd/:pwt/:mpdta/:ddcg
`describe_data(d)`, `diagnose(d)`, `fix(d; method)`, `apply_tcode(d, codes)`, `validate_for_model(d, model_type)`
`to_matrix(d)`, `varnames(d)`, `balance_panel(d)`, `set_dates!(d, dates)`, `dates(d)`
`dropna(ts; vars, cols)`, `keeprows(ts, indices)` — row-level data filtering (v0.4.0)

**Nowcast:**
`nowcast_dfm(Y; factors, lags)`, `nowcast_bvar(Y; lags)`, `nowcast_bridge(Y, target)`
`nowcast(model)` → NowcastResult | `nowcast_news(model, Y_old, Y_new)` | `forecast(model::AbstractNowcastModel, h)`

**DSGE:**
`DSGESpec(; n_endog, n_exog, varnames, exog, parameters, equations, steady_state)`
`compute_steady_state(spec; constraints)` | `linearize(spec)` → LinearDSGE
`solve(spec; method, order, degree, grid)` — method: :gensys/:blanchard_kahn/:klein/:perturbation/:projection/:pfi
`gensys(spec)`, `blanchard_kahn(spec)`, `klein(spec)` → DSGESolution | `perturbation_solver(spec; order)` (order: 1/2/3, 3rd-order NEW in v0.3.2) | `collocation_solver(spec; degree)` | `pfi_solver(spec; degree)`
`is_determined(sol)`, `is_stable(sol)`, `nshocks(sol)`
`simulate(sol, T; antithetic, rng)` | `simulate(spec, T)` → Matrix
`estimate_dsge(spec, Y, param_names; method, solve_method, weighting, irf_horizon, var_lags, sim_ratio)` → DSGEEstimation
`perfect_foresight(spec; shocks, T_periods)` → PerfectForesightPath
OccBin: `OccBinConstraint(var, lower, upper)`, `variable_bound(var; lower, upper)`, `parse_constraint(expr)`, `occbin_solve(spec, shocks, constraints; T_periods)`, `occbin_irf(spec, constraints, shock_idx; shock_size, horizon)`
`estimate_dsge_bayes(spec, data, theta0; priors, method, n_smc, n_mh, n_blocks, conf_level)` → BayesianDSGE — method: :smc/:smc2/:mh
`posterior_summary(bd::BayesianDSGE)`, `bayes_factor(bd1, bd2)`, `prior_posterior_table(bd)`, `posterior_predictive(bd; n_sim, periods)`
`historical_decomposition(sol::DSGESolution, data; horizon)` → HistoricalDecomposition — DSGE frequentist HD (v0.4.0)
`historical_decomposition(bd::BayesianDSGE, data; horizon, quantiles)` → BayesianHistoricalDecomposition — DSGE Bayesian HD (v0.4.0)

**Cross-sectional regression:**
`estimate_reg(y, X; cov_type, weights, varnames, clusters)` → RegModel | `estimate_iv(y, X, Z; endogenous, cov_type, varnames)` → IVModel
`estimate_logit(y, X; cov_type, varnames, clusters, maxiter, tol)` → LogitModel | `estimate_probit(y, X; cov_type, varnames, clusters, maxiter, tol)` → ProbitModel
`vif(model::RegModel)` → Vector{Float64} | `marginal_effects(model)` → MarginalEffects | `odds_ratio(model::LogitModel)`, `classification_table(model; threshold)`

**Panel regression (v0.4.0):**
`estimate_panel_reg(panel, dep; effect, cov_type, weights, clusters, varnames)` → PanelRegModel — effect: :fe/:re/:pooled
`estimate_panel_iv(panel, dep, endog, instruments; effect, cov_type, varnames)` → PanelIVModel
`estimate_panel_logit(panel, dep; effect, cov_type, varnames, clusters, maxiter, tol)` → PanelLogitModel
`estimate_panel_probit(panel, dep; effect, cov_type, varnames, clusters, maxiter, tol)` → PanelProbitModel

**Panel specification tests (v0.4.0):**
`hausman_test(fe_model, re_model)` → PanelTestResult — FE vs RE specification
`breusch_pagan_test(model)` → PanelTestResult — LM test for random effects
`f_test_fe(model)` → PanelTestResult — F-test for fixed effects significance
`pesaran_cd_test(model)` → PanelTestResult — cross-sectional dependence
`wooldridge_ar_test(model)` → PanelTestResult — serial correlation in panel data
`modified_wald_test(model)` → PanelTestResult — groupwise heteroskedasticity

**Ordered/multinomial choice models (v0.4.0):**
`estimate_ologit(y, X; n_categories, cov_type, varnames, clusters, maxiter, tol)` → OrderedLogitModel
`estimate_oprobit(y, X; n_categories, cov_type, varnames, clusters, maxiter, tol)` → OrderedProbitModel
`estimate_mlogit(y, X; n_categories, base_category, cov_type, varnames, maxiter, tol)` → MultinomialLogitModel
`marginal_effects(model::OrderedLogitModel)`, `marginal_effects(model::MultinomialLogitModel)` → MarginalEffects
`brant_test(model)` → PanelTestResult — proportional odds assumption (ordered logit/probit)
`hausman_iia(model; omit_category)` → PanelTestResult — IIA test (multinomial logit)

**Spectral analysis (v0.4.0):**
`acf(y; lags, conf_level)` → ACFResult | `pacf(y; lags)` | `ccf(y1, y2; lags, conf_level)` → ACFResult
`periodogram(y; detrend)` → ACFResult — raw periodogram (DFT squared magnitudes)
`spectral_density(y; method, bandwidth, kernel, varname)` → SpectralDensityResult — kernel: :bartlett/:parzen/:tukey_hanning
`cross_spectrum(y1, y2; bandwidth, var1, var2)` → CrossSpectrumResult — coherence, phase, gain
`transfer_function(input, output; bandwidth)` → TransferFunctionResult

**Additional time series tests (v0.4.0):**
`fisher_test(y)` — Fisher-type panel unit root test
`bartlett_white_noise_test(y; lags)` — Bartlett white noise test
`box_pierce_test(y; lags, ljung_box)` — Box-Pierce/Ljung-Box portmanteau test
`durbin_watson_test(residuals)` — Durbin-Watson first-order serial correlation test

**Advanced unit root tests:**
`fourier_adf_test(y; regression, fmax, lags, trim)` → FourierADFResult | `fourier_kpss_test(y; regression, fmax, bandwidth)` → FourierKPSSResult
`dfgls_test(y; regression, lags)` → DFGLSResult | `lm_unitroot_test(y; breaks, regression, lags, trim)` → LMUnitRootResult
`adf_2break_test(y; model, lags, trim)` → ADF2BreakResult | `gregory_hansen_test(Y; model, lags, trim)` → GregoryHansenResult

**DID:**
`estimate_did(panel, outcome, treatment; method, leads, horizon, covariates, control_group, cluster, conf_level, n_boot, base_period)` — method: :twfe/:cs/:sa/:bjs/:dcdh, base_period: :varying/:universal (CS only)
`estimate_event_study_lp(panel, outcome, treatment, horizon; leads, lags, covariates, cluster, conf_level)`
`estimate_lp_did(panel, outcome, treatment, horizon; pre_window, post_window, ylags, dylags, covariates, cluster, conf_level, pmd, reweight, nocomp, nonabsorbing, notyet, nevertreated, firsttreat, oneoff)` → LPDiDResult
`bacon_decomposition(panel, outcome, treatment)`, `pretrend_test(result)` (DIDResult or EventStudyLP dispatch), `negative_weight_check(panel, treatment)`, `honest_did(result; Mbar, conf_level)` (DIDResult or EventStudyLP dispatch)

**Covariance estimators:** `newey_west(X, u; bandwidth, kernel)`, `white_vcov(X, u; variant)`, `driscoll_kraay(X, u; bandwidth, kernel)` — kernels: :bartlett/:parzen/:quadratic_spectral/:tukey_hanning

**StatsAPI:** VARModel/ARIMA/RegModel/LogitModel/ProbitModel/PanelRegModel/OrderedLogitModel/MultinomialLogitModel implement: `coef`, `vcov`, `residuals`, `predict`, `r2`, `aic`, `bic`, `dof`, `dof_residual`, `nobs`, `loglikelihood`, `confint`, `stderror`, `islinear`, `fit`

**Plotting:** `plot_result(result; kwargs...)` → PlotOutput, `save_plot(p, path)`, `display_plot(p)` — interactive D3.js HTML plots

**GPL Notice:** `warranty()`, `conditions()` — display GPL-3.0 warranty/conditions text
