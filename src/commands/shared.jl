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

# Shared utilities for command handlers

# ── Data Loading Helpers ───────────────────────────────────

"""
    load_multivariate_data(data) → (Y::Matrix{Float64}, varnames::Vector{String})

Load CSV, convert to numeric matrix and extract variable names.
"""
function load_multivariate_data(data::String)
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    return Y, varnames
end

"""
    load_univariate_series(data, column) → (y::Vector{Float64}, vname::String)

Load CSV and extract a single numeric column by index.
"""
function load_univariate_series(data::String, column::Int)
    df = load_data(data)
    varnames = variable_names(df)
    column > length(varnames) && error("column $column out of range (data has $(length(varnames)) numeric columns)")
    y = Vector{Float64}(df[!, varnames[column]])
    return y, varnames[column]
end

# ── Naming Helpers ─────────────────────────────────────────

"""Safe shock name: uses variable name if in range, else "shock_N"."""
_shock_name(varnames::Vector{String}, idx::Int) =
    idx <= length(varnames) ? varnames[idx] : "shock_$idx"

"""Safe variable name: uses variable name if in range, else "var_N"."""
_var_name(varnames::Vector{String}, idx::Int) =
    idx <= length(varnames) ? varnames[idx] : "var_$idx"

"""Generate per-variable output path by inserting suffix before extension."""
function _per_var_output_path(output::String, suffix::String)
    isempty(output) && return ""
    return replace(output, "." => "_$(suffix).")
end

# ── Output Helpers ─────────────────────────────────────────

"""Build a coefficient table DataFrame for VAR/BVAR models."""
function _build_var_coef_table(coef_mat::AbstractMatrix, varnames::Vector{String}, p::Int)
    n = length(varnames)
    n_rows = size(coef_mat, 1)
    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end
    coef_df = DataFrame(permutedims(coef_mat), row_names)
    insertcols!(coef_df, 1, :equation => varnames)
    return coef_df
end

"""Output AIC/BIC/HQC/loglik for a VAR-like model."""
function output_model_criteria(model; format::String="table", output::String="", title::String="Information Criteria")
    pairs = Pair{String,Any}[
        "AIC" => model.aic,
        "BIC" => model.bic,
        "HQC" => model.hqic,
    ]
    if hasproperty(model, :loglik) || hasmethod(loglikelihood, (typeof(model),))
        try
            push!(pairs, "Log-likelihood" => loglikelihood(model))
        catch; end
    end
    output_kv(pairs; format=format, title=title)
end

"""Output per-variable FEVD tables."""
function _output_fevd_tables(proportions::AbstractArray, varnames::Vector{String},
                              horizons::Int; id::String="cholesky",
                              title_prefix::String="FEVD",
                              format::String="table", output::String="")
    n = length(varnames)
    for vi in 1:n
        fevd_df = DataFrame()
        fevd_df.horizon = 1:horizons
        for si in 1:n
            fevd_df[!, _shock_name(varnames, si)] = proportions[vi, si, :]
        end
        vname = _var_name(varnames, vi)
        output_result(fevd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="$title_prefix for $vname ($id identification)")
        println()
    end
end

"""Output per-variable HD tables."""
function _output_hd_tables(get_contrib::Function, varnames::Vector{String},
                            T_eff::Int; id::String="cholesky",
                            title_prefix::String="Historical Decomposition",
                            format::String="table", output::String="",
                            actual=nothing, initial=nothing)
    n = length(varnames)
    for vi in 1:n
        hd_df = DataFrame()
        hd_df.period = 1:T_eff
        if !isnothing(actual)
            hd_df.actual = actual[:, vi]
        end
        if !isnothing(initial)
            hd_df.initial = initial[:, vi]
        end
        for si in 1:n
            hd_df[!, "contrib_$(_shock_name(varnames, si))"] = get_contrib(vi, si)
        end
        vname = _var_name(varnames, vi)
        output_result(hd_df; format=Symbol(format),
                      output=_per_var_output_path(output, vname),
                      title="$title_prefix: $vname ($id identification)")
        println()
    end
end

# ── Validation Helpers ─────────────────────────────────────

"""Validate that a method string is in the allowed set."""
function validate_method(method::String, allowed::Vector{String}, context::String)
    method in allowed || error("unknown $context: $method (expected $(join(allowed, "|")))")
    return method
end

# ── Test Helpers ───────────────────────────────────────────

"""Print colored p-value interpretation for hypothesis tests."""
function interpret_test_result(pvalue::Real, reject_msg::String, accept_msg::String; level::Float64=0.05)
    println()
    if pvalue < level
        printstyled("-> $reject_msg\n"; color=:yellow)
    else
        printstyled("-> $accept_msg\n"; color=:green)
    end
end

"""Convert trend string to Symbol for test regression kwarg."""
function to_regression_symbol(trend::String)
    trend == "none" && return :none
    trend == "both" && return :both
    return Symbol(trend)
end

# ── Volatility Output Helpers ──────────────────────────────

"""Standard normal CDF approximation (Abramowitz & Stegun)."""
function _normal_cdf(x::Real)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327  # 1/sqrt(2*pi)
    p = d * exp(-x * x / 2.0) * t *
        (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    x >= 0.0 ? 1.0 - p : p
end

"""Shared volatility model estimation output: coefficients + persistence."""
function _vol_estimate_output(model, vname::String, param_names::Vector{String},
                               model_label::String; format::String="table", output::String="")
    c = coef(model)
    names = param_names[1:length(c)]
    coef_df = try
        # SVModel has no vcov/stderror — StatsAPI defaults cause infinite recursion
        model isa SVModel && throw(ErrorException("no SE for SV"))
        se = stderror(model)
        z = c ./ se
        pv = [2.0 * (1.0 - _normal_cdf(abs(zi))) for zi in z]
        DataFrame(parameter=names, estimate=round.(c; digits=6),
                  std_error=round.(se; digits=6),
                  z_stat=round.(z; digits=3), p_value=round.(pv; digits=4))
    catch
        DataFrame(parameter=names, estimate=round.(c; digits=6))
    end
    output_result(coef_df; format=Symbol(format), output=output,
                  title="$model_label Coefficients ($vname)")
    println()
    p_val = persistence(model)
    println("Persistence: $(round(p_val; digits=4))")
end

"""Shared volatility forecast output: horizon/variance/volatility table."""
function _vol_forecast_output(fc, vname::String, model_label::String,
                               horizons::Int; format::String="table", output::String="")
    fc_df = DataFrame(
        horizon=1:horizons,
        variance=round.(fc.forecast; digits=6),
        volatility=round.(sqrt.(fc.forecast); digits=6)
    )
    output_result(fc_df; format=Symbol(format), output=output,
                  title="$model_label Volatility Forecast ($vname, h=$horizons)")
end

# ── Constants ──────────────────────────────────────────────

"""
    ID_METHOD_MAP

Maps CLI identification method strings to MacroEconometricModels symbols.
"""
const ID_METHOD_MAP = Dict(
    "cholesky"          => :cholesky,
    "sign"              => :sign,
    "narrative"         => :narrative,
    "longrun"           => :long_run,
    "fastica"           => :fastica,
    "jade"              => :jade,
    "sobi"              => :sobi,
    "dcov"              => :dcov,
    "hsic"              => :hsic,
    "student_t"         => :student_t,
    "mixture_normal"    => :mixture_normal,
    "pml"               => :pml,
    "skew_normal"       => :skew_normal,
    "markov_switching"  => :markov_switching,
    "garch_id"          => :garch,
    "uhlig"             => :uhlig,
)

"""
    _load_and_estimate_var(data, lags) -> (model, Y, varnames, p)

Load data from CSV, optionally auto-select lag order, and estimate a frequentist VAR.
"""
function _load_and_estimate_var(data::String, lags)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)

    p = if isnothing(lags)
        select_lag_order(Y, min(12, size(Y,1) ÷ (3*n)); criterion=:aic)
    else
        lags
    end

    model = estimate_var(Y, p)
    return model, Y, varnames, p
end

"""
    _load_and_estimate_bvar(data, lags, config, draws, sampler) -> (post, Y, varnames, p, n)

Load data from CSV, build prior, and estimate a Bayesian VAR.
Returns a BVARPosterior (which carries p, n, data internally).
"""
function _load_and_estimate_bvar(data::String, lags::Int, config::String,
                                  draws::Int, sampler::String)
    Y, varnames = load_multivariate_data(data)
    n = size(Y, 2)
    p = lags

    prior_obj = _build_prior(config, Y, p)
    prior_sym = isnothing(prior_obj) ? :normal : :minnesota

    post = estimate_bvar(Y, p;
        sampler=Symbol(sampler), n_draws=draws,
        prior=prior_sym, hyper=prior_obj)

    return post, Y, varnames, p, n
end

"""
    _build_prior(config_path, Y, p) -> MinnesotaHyperparameters or nothing

Build a Minnesota prior from TOML config, or return nothing for default prior.
"""
function _build_prior(config_path::String, Y::AbstractMatrix, p::Int)
    if isempty(config_path)
        return nothing
    end
    cfg = load_config(config_path)
    prior_cfg = get_prior(cfg)

    if prior_cfg["type"] == "minnesota"
        if prior_cfg["optimize"]
            println("Optimizing Minnesota prior hyperparameters...")
            return optimize_hyperparameters(Y, p)
        else
            sigma_ar = ones(size(Y, 2))
            for i in 1:size(Y, 2)
                y = Y[:, i]
                if length(y) > 2
                    X = y[1:end-1]
                    y_dep = y[2:end]
                    b = X \ y_dep
                    resid = y_dep .- X .* b
                    sigma_ar[i] = sqrt(sum(resid .^ 2) / (length(resid) - 1))
                end
            end
            return MinnesotaHyperparameters(;
                tau=prior_cfg["lambda1"],
                decay=prior_cfg["lambda3"],
                lambda=prior_cfg["lambda2"],
                omega=sigma_ar
            )
        end
    end
    return nothing
end

"""
    _build_check_func(config_path) -> (check_func, narrative_check)

Build sign restriction and narrative restriction check functions from TOML config.
Returns `(nothing, nothing)` if no config or no restrictions.
"""
function _build_check_func(config_path::String)
    if isempty(config_path)
        return nothing, nothing
    end
    cfg = load_config(config_path)
    id_cfg = get_identification(cfg)

    check_func = nothing
    narrative_check = nothing

    if haskey(id_cfg, "sign_matrix")
        sign_mat = id_cfg["sign_matrix"]
        horizons = get(id_cfg, "horizons", [0])
        check_func = function(irf_values)
            for h_idx in 1:length(horizons)
                h = horizons[h_idx] + 1  # 1-based indexing
                if h > size(irf_values, 1)
                    continue
                end
                for i in 1:size(sign_mat, 1)
                    for j in 1:size(sign_mat, 2)
                        s = sign_mat[i, j]
                        if s != 0
                            if s > 0 && irf_values[h, j, i] < 0
                                return false
                            elseif s < 0 && irf_values[h, j, i] > 0
                                return false
                            end
                        end
                    end
                end
            end
            return true
        end
    end

    if haskey(id_cfg, "narrative")
        narr = id_cfg["narrative"]
        shock_idx = narr["shock_index"]
        periods = narr["periods"]
        signs = narr["signs"]
        narrative_check = function(structural_shocks)
            for (t, s) in zip(periods, signs)
                if t > size(structural_shocks, 1)
                    continue
                end
                if s > 0 && structural_shocks[t, shock_idx] < 0
                    return false
                elseif s < 0 && structural_shocks[t, shock_idx] > 0
                    return false
                end
            end
            return true
        end
    end

    return check_func, narrative_check
end

"""
    _build_identification_kwargs(id, config) -> Dict{Symbol,Any}

Build the kwargs dict for irf/fevd/historical_decomposition calls
based on identification method and config file.
"""
function _build_identification_kwargs(id::String, config::String)
    method = get(ID_METHOD_MAP, id, :cholesky)
    kwargs = Dict{Symbol,Any}(:method => method)

    check_func, narrative_check = _build_check_func(config)
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    return kwargs
end

"""
    _load_and_structural_lp(data, horizons, lags, var_lags, id, vcov, config;
                            ci_type=:none, reps=200, conf_level=0.95)

Load data, build identification kwargs, and compute structural LP.
Returns `(slp, Y, varnames)`.
"""
function _load_and_structural_lp(data::String, horizons::Int, lags::Int,
                                  var_lags, id::String, vcov::String,
                                  config::String;
                                  ci_type::Symbol=:none, reps::Int=200,
                                  conf_level::Float64=0.95)
    Y, varnames = load_multivariate_data(data)

    method = get(ID_METHOD_MAP, id, :cholesky)
    check_func, narrative_check = _build_check_func(config)

    vp = isnothing(var_lags) ? lags : var_lags

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :lags => lags,
        :var_lags => vp,
        :cov_type => Symbol(vcov),
        :ci_type => ci_type,
        :reps => reps,
        :conf_level => conf_level,
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    slp = structural_lp(Y, horizons; kwargs...)
    return slp, Y, varnames
end

"""
    _load_and_estimate_vecm(data, lags, rank, deterministic, method, significance)
        -> (vecm, Y, varnames, p)

Load data from CSV and estimate a VECM. When rank=="auto", uses select_vecm_rank().
"""
function _load_and_estimate_vecm(data::String, lags::Int, rank::String,
                                  deterministic::String, method::String,
                                  significance::Float64)
    Y, varnames = load_multivariate_data(data)

    r = if rank == "auto"
        select_vecm_rank(Y, lags; significance=significance)
    else
        parse(Int, rank)
    end

    vecm = estimate_vecm(Y, lags; rank=r, deterministic=Symbol(deterministic),
                         method=Symbol(method), significance=significance)
    return vecm, Y, varnames, lags
end

"""
    _var_forecast_point(B, Y, p, horizons) -> Matrix{Float64}

Iterate the VAR(p) equation h steps ahead to produce point forecasts.
B is the coefficient matrix (k × n), Y is the data matrix (T × n).
Returns a (horizons × n) matrix of forecast values.
"""
function _var_forecast_point(B::AbstractMatrix, Y::AbstractMatrix, p::Int, horizons::Int)
    T, n = size(Y)
    has_const = size(B, 1) > n * p

    forecasts = zeros(horizons, n)

    # lag_buf: [Y_T, Y_{T-1}, ..., Y_{T-p+1}] flattened to np vector
    lag_buf = zeros(n * p)
    for lag in 1:p
        lag_buf[(lag-1)*n+1:lag*n] = Y[T-lag+1, :]
    end

    for h in 1:horizons
        x = has_const ? vcat(lag_buf, 1.0) : lag_buf
        y_hat = B' * x
        forecasts[h, :] = y_hat

        # Shift lag buffer forward
        if p > 1
            lag_buf[n+1:end] = lag_buf[1:end-n]
        end
        lag_buf[1:n] = y_hat
    end

    return forecasts
end

# ── Panel VAR Helpers ─────────────────────────────────────

"""
    _parse_varlist(str) -> Vector{String}

Parse a comma-separated variable list string. Returns empty vector for empty input.
"""
function _parse_varlist(str::String)
    isempty(str) && return String[]
    return [strip(s) for s in split(str, ",") if !isempty(strip(s))]
end

"""
    load_panel_data(data, id_col, time_col; varnames=nothing) -> PanelData

Load CSV data and set panel structure using xtset().
"""
function load_panel_data(data::String, id_col::String, time_col::String)
    df = load_data(data)
    id_col in names(df) || error("id column '$id_col' not found in data (columns: $(join(names(df), ", ")))")
    time_col in names(df) || error("time column '$time_col' not found in data (columns: $(join(names(df), ", ")))")
    group_ids = Int.(df[!, id_col])
    time_ids = Int.(df[!, time_col])
    # Get numeric columns excluding id and time
    varnames = [n for n in variable_names(df) if n != id_col && n != time_col]
    isempty(varnames) && error("no numeric variables found after excluding id/time columns")
    Y = Matrix{Float64}(df[!, varnames])
    return xtset(Y, group_ids, time_ids; varnames=varnames)
end

"""
    _load_and_estimate_pvar(data, id_col, time_col, lags; kwargs...) -> (model, panel, varnames)

Combined load + estimate for Panel VAR.
"""
function _load_and_estimate_pvar(data::String, id_col::String, time_col::String,
                                  lags::Int; method::String="gmm",
                                  transformation::String="fd", steps::String="twostep",
                                  system::Bool=false, collapse::Bool=false,
                                  dependent::String="", predet::String="", exog::String="",
                                  min_lag_endo::Int=2, max_lag_endo::Int=99)
    panel = load_panel_data(data, id_col, time_col)

    dep = _parse_varlist(dependent)
    pre = _parse_varlist(predet)
    exo = _parse_varlist(exog)

    model = if method == "feols"
        estimate_pvar_feols(panel, lags;
            dependent=isempty(dep) ? nothing : dep,
            exogenous=isempty(exo) ? nothing : exo)
    else
        estimate_pvar(panel, lags;
            transformation=Symbol(transformation), steps=Symbol(steps),
            system=system, collapse=collapse,
            dependent=isempty(dep) ? nothing : dep,
            predetermined=isempty(pre) ? nothing : pre,
            exogenous=isempty(exo) ? nothing : exo,
            min_lag_endo=min_lag_endo, max_lag_endo=max_lag_endo)
    end
    return model, panel, panel.varnames
end

"""
    _build_pvar_coef_table(model, varnames, p) -> DataFrame

Build a coefficient table for Panel VAR model with SE and p-values.
"""
function _build_pvar_coef_table(model, varnames::Vector{String}, p::Int)
    n = length(varnames)
    n_rows = size(model.Phi, 1)
    row_names = String[]
    for lag in 1:p
        for v in varnames
            push!(row_names, "$(v)_L$(lag)")
        end
    end
    if n_rows > n * p
        push!(row_names, "const")
    end

    coef_df = DataFrame()
    for (vi, vname) in enumerate(varnames)
        coef_df[!, Symbol("$(vname)_coef")] = round.(model.Phi[1:length(row_names), vi]; digits=6)
        coef_df[!, Symbol("$(vname)_se")] = round.(model.se[1:length(row_names), vi]; digits=6)
        coef_df[!, Symbol("$(vname)_pval")] = round.(model.pvalues[1:length(row_names), vi]; digits=4)
    end
    insertcols!(coef_df, 1, :parameter => row_names)
    return coef_df
end

# ── FAVAR Helpers ─────────────────────────────────────────

"""
    _load_and_estimate_favar(data, factors, lags, key_vars, method, draws) → (favar, Y, varnames)
"""
function _load_and_estimate_favar(data::String, factors, lags::Int,
                                   key_vars::String, method::String, draws::Int)
    Y, varnames = load_multivariate_data(data)
    T_obs, n = size(Y)

    # Parse key variables (comma-separated names or indices)
    key_indices = Int[]
    if !isempty(key_vars)
        for kv in split(key_vars, ",")
            kv = strip(kv)
            idx = tryparse(Int, kv)
            if idx !== nothing
                push!(key_indices, idx)
            else
                found = findfirst(==(kv), varnames)
                found === nothing && error("key variable '$kv' not found in data columns: $varnames")
                push!(key_indices, found)
            end
        end
    end
    isempty(key_indices) && error("--key-vars is required for FAVAR (comma-separated column names or indices)")

    # Auto-select factors if not specified
    r = if factors === nothing
        auto_r = ic_criteria(Y, min(10, n - 1))
        printstyled("  Auto-selected factors: $(auto_r.r_IC1) (IC1)\n"; color=:cyan)
        auto_r.r_IC1
    else
        factors
    end

    println("Estimating FAVAR: $r factors, $lags lags, method=$method, $(length(key_indices)) key variables")

    favar = estimate_favar(Y, key_indices, r, lags;
                           method=Symbol(method),
                           n_draws=draws)
    return favar, Y, varnames
end

# ── Panel/Matrix Loading Helper ──────────────────────────

"""
    _load_panel_or_matrix(data; id_col, time_col) → (result, is_panel)

Load data as PanelData if id_col/time_col are provided, else as Matrix.
"""
function _load_panel_or_matrix(data::String; id_col::String="", time_col::String="")
    if !isempty(id_col) && !isempty(time_col)
        pd = load_panel_data(data, id_col, time_col)
        printstyled("  Panel: $(pd.n_groups) units, $(div(pd.T_obs, pd.n_groups)) periods\n"; color=:cyan)
        return pd, true
    else
        Y, varnames = load_multivariate_data(data)
        println("  Matrix: $(size(Y, 1)) obs × $(size(Y, 2)) units")
        return Y, false
    end
end

# ── Plot Helpers ──────────────────────────────────────────

"""
    _maybe_plot(result; plot, plot_save, kwargs...)

Optionally plot a result using MacroEconometricModels' interactive D3.js plotting.
If `plot` is true, opens in browser. If `plot_save` is non-empty, saves to HTML file.
"""
function _maybe_plot(result; plot::Bool=false, plot_save::String="", kwargs...)
    !plot && isempty(plot_save) && return
    p = plot_result(result; kwargs...)
    if !isempty(plot_save)
        save_plot(p, plot_save)
        printstyled("  Plot saved: $plot_save\n"; color=:green)
    end
    if plot
        display_plot(p)
        printstyled("  Plot opened in browser\n"; color=:cyan)
    end
end

# ── DSGE Helpers ───────────────────────────────────────────

"""
    _load_dsge_model(path) → DSGESpec

Load a DSGE model from a .toml or .jl file.
- .toml: parse [model] section, construct DSGESpec via TOML config
- .jl: include() the file, expect a `model` variable of type DSGESpec
"""
function _load_dsge_model(path::String)
    isfile(path) || error("model file not found: $path")
    ext = lowercase(splitext(path)[2])

    if ext == ".toml"
        config = load_config(path)
        dsge_cfg = get_dsge(config)

        isempty(dsge_cfg["endogenous"]) && error("TOML model must have [model] with endogenous variables")
        isempty(dsge_cfg["equations"]) && error("TOML model must have [[model.equations]]")

        param_dict = dsge_cfg["parameters"]
        endog = Symbol.(dsge_cfg["endogenous"])
        exog = Symbol.(dsge_cfg["exogenous"])

        spec = MacroEconometricModels.DSGESpec(; n_endog=length(endog), n_exog=length(exog))

        println("Loaded DSGE model from TOML: $(length(endog)) endogenous, $(length(exog)) exogenous, $(length(dsge_cfg["equations"])) equations")
        return spec

    elseif ext == ".jl"
        mod = Module()
        Base.eval(mod, :(const MacroEconometricModels = $(MacroEconometricModels)))
        result = Base.include(mod, path)
        result isa MacroEconometricModels.DSGESpec || error(
            ".jl model file must evaluate to a DSGESpec (last expression), got $(typeof(result))")
        spec = result
        println("Loaded DSGE model from Julia file: $(spec.n_endog) endogenous, $(spec.n_exog) exogenous")
        return spec

    else
        error("unsupported model file extension '$ext' — use .toml or .jl")
    end
end

"""
    _solve_dsge(spec; method="gensys", order=1, degree=5, grid="auto", constraint_solver="") → solution

Solve a DSGE model: compute steady state → linearize → solve.
Returns DSGESolution, PerturbationSolution, or ProjectionSolution.
"""
function _solve_dsge(spec::MacroEconometricModels.DSGESpec;
                     method::String="gensys", order::Int=1,
                     degree::Int=5, grid::String="auto",
                     constraint_solver::String="")
    println("Computing steady state...")
    ss_kw = isempty(constraint_solver) ? (;) : (; solver=Symbol(constraint_solver))
    spec = compute_steady_state(spec; ss_kw...)

    println("Linearizing model...")
    linearize(spec)

    println("Solving with method=$method" *
            (method == "perturbation" ? ", order=$order" : "") *
            (method in ("projection", "pfi") ? ", degree=$degree, grid=$grid" : "") *
            "...")

    solve_kw = isempty(constraint_solver) ? (;) : (; solver=Symbol(constraint_solver))
    sol = solve(spec; method=Symbol(method), order=order,
                degree=degree, grid=Symbol(grid), solve_kw...)

    # Report diagnostics
    if sol isa MacroEconometricModels.DSGESolution ||
       sol isa MacroEconometricModels.PerturbationSolution
        det_status = is_determined(sol) ? "unique" : "indeterminate"
        stab_status = is_stable(sol) ? "stable" : "unstable"
        printstyled("  Determinacy: $det_status\n"; color = is_determined(sol) ? :green : :red)
        printstyled("  Stability: $stab_status\n"; color = is_stable(sol) ? :green : :red)
    end

    return sol
end

"""
    _load_dsge_constraints(path; spec=nothing) → Vector{constraint}

Load OccBin and/or nonlinear constraints from a TOML file.
Nonlinear constraints require a loaded DSGE spec.
"""
function _load_dsge_constraints(path::String; spec=nothing)
    config = load_config(path)
    con_cfg = get_dsge_constraints(config)

    has_bounds = !isempty(get(con_cfg, "bounds", []))
    has_nonlinear = !isempty(get(con_cfg, "nonlinear", []))

    if has_nonlinear && spec === nothing
        error("nonlinear constraints require a loaded DSGE spec (pass spec keyword)")
    end

    constraints = Any[]

    if has_bounds
        for b in con_cfg["bounds"]
            lower = get(b, "lower", -Inf)
            c = variable_bound(Symbol(b["variable"]); lower=lower,
                               upper=get(b, "upper", Inf))
            push!(constraints, c)
        end
    end

    if has_nonlinear
        for nl in con_cfg["nonlinear"]
            c = parse_constraint(nl["expr"], spec)
            push!(constraints, c)
        end
    end

    return constraints
end

"""
    _load_panel_for_did(data, id_col, time_col) -> PanelData

Load panel CSV and print summary for DID/event study commands.
"""
function _load_panel_for_did(data::String, id_col::String, time_col::String)
    pd = load_panel_data(data, id_col, time_col)
    printstyled("  Panel: $(pd.n_groups) groups, $(div(pd.T_obs, pd.n_groups)) periods, " *
                "$(pd.n_vars) variables"; color=:cyan)
    pd.balanced && printstyled(" (balanced)"; color=:cyan)
    println()
    return pd
end

# ── Regression Helpers ────────────────────────────────────

const _REG_COMMON_OPTIONS = [
    Option("dep"; type=String, default="", description="Dependent variable column name (default: first numeric column)"),
    Option("cov-type"; type=String, default="hc1", description="ols|hc0|hc1|hc2|hc3|cluster"),
    Option("clusters"; type=String, default="", description="Cluster variable column name"),
    Option("output"; short="o", type=String, default="", description="Export results to file"),
    Option("format"; short="f", type=String, default="table", description="table|csv|json"),
]

"""
    _load_reg_data(data, dep; weights_col="", clusters_col="") → (y, X, varnames)

Load CSV, split into dependent variable y and regressor matrix X.
If dep is empty, uses first numeric column as y.
"""
function _load_reg_data(data::String, dep::String; weights_col::String="", clusters_col::String="")
    df = load_data(data)
    numcols = variable_names(df)

    dep_col = isempty(dep) ? numcols[1] : dep
    !isempty(dep) && !(dep_col in numcols) && error("dependent variable '$dep_col' not found in numeric columns: $numcols")

    exclude = Set([dep_col])
    !isempty(weights_col) && push!(exclude, weights_col)
    !isempty(clusters_col) && push!(exclude, clusters_col)
    xcols = filter(c -> !(c in exclude), numcols)
    isempty(xcols) && error("no regressor columns remaining after excluding dep='$dep_col'")

    y = Vector{Float64}(df[!, dep_col])
    X = Matrix{Float64}(df[!, xcols])
    return y, X, xcols
end

"""Load cluster assignments from a CSV column, or return nothing."""
function _load_clusters(data::String, clusters_col::String)
    isempty(clusters_col) && return nothing
    df = load_data(data)
    clusters_col in names(df) || error("cluster column '$clusters_col' not found")
    return Vector{Int}(df[!, clusters_col])
end

"""Load observation weights from a CSV column, or return nothing."""
function _load_weights(data::String, weights_col::String)
    isempty(weights_col) && return nothing
    df = load_data(data)
    weights_col in names(df) || error("weights column '$weights_col' not found")
    return Vector{Float64}(df[!, weights_col])
end

"""Build coefficient table DataFrame from a regression model."""
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
