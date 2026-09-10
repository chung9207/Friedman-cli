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

# Nowcast commands: dfm, bvar, bridge, news, forecast

function nowcast_specs()::Vector{CommandSpec}
    data_arg = [ArgSpec(name="data", description="Path to CSV data file")]
    out_fmt = [
        OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
        OptionSpec(name="format", short="f", type=String, default="table",
                   choices=["table", "csv", "json"], description="table|csv|json"),
    ]
    return [
        CommandSpec(path=["nowcast", "dfm"], summary="Nowcast via Dynamic Factor Model (EM algorithm)",
            args=data_arg,
            options=[
                OptionSpec(name="monthly-vars", type=Int, default=0, description="Number of monthly variables (first N columns)"),
                OptionSpec(name="quarterly-vars", type=Int, default=0, description="Number of quarterly variables (remaining columns)"),
                OptionSpec(name="factors", short="r", type=Int, default=2, description="Number of factors"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Factor VAR lags"),
                OptionSpec(name="idio", type=String, default="ar1", description="Idiosyncratic component: ar1|iid"),
                OptionSpec(name="max-iter", type=Int, default=100, description="Maximum EM iterations"),
                OptionSpec(name="target-var", type=Int, default=0, description="Target variable index (0=last)"),
                out_fmt..., PLOT_OPTIONS...,
            ],
            tables=[TableSpec(name=:nowcast_dfm,
                              description="Nowcast, forecast, log-likelihood and EM iteration count")],
            flags=copy(PLOT_FLAGS), category="nowcast", handler=wrap_legacy(_nowcast_dfm)),
        CommandSpec(path=["nowcast", "bvar"], summary="Nowcast via Bayesian VAR",
            args=data_arg,
            options=[
                OptionSpec(name="monthly-vars", type=Int, default=0, description="Number of monthly variables"),
                OptionSpec(name="quarterly-vars", type=Int, default=0, description="Number of quarterly variables"),
                OptionSpec(name="lags", short="p", type=Int, default=5, description="VAR lags"),
                OptionSpec(name="target-var", type=Int, default=0, description="Target variable index (0=last)"),
                OptionSpec(name="prior", type=String, default="conjugate", choices=["conjugate", "litterman"],
                           description="conjugate (GLP dummy-observation NIW) | litterman (fixed-Σ; enables --theta-cross)"),
                OptionSpec(name="theta-cross", type=String, default="",
                           description="Initial cross-variable relative tightness > 0 (litterman only; rejected with conjugate)"),
                OptionSpec(name="lambda0", type=Float64, default=0.2, description="Initial overall shrinkage λ"),
                OptionSpec(name="theta0", type=Float64, default=1.0, description="Initial lag-decay exponent (Litterman d / GLP α)"),
                OptionSpec(name="miu0", type=Float64, default=1.0, description="Initial sum-of-coefficients weight μ"),
                OptionSpec(name="alpha0", type=Float64, default=2.0, description="Initial co-persistence weight α"),
                out_fmt...,
            ],
            tables=[
                TableSpec(name=:nowcast_bvar,
                          description="Nowcast, forecast and log-likelihood"),
                TableSpec(name=:nowcast_bvar_hyperparameters,
                          description="Optimized prior hyperparameters (lambda, theta, miu, alpha, theta_cross)"),
            ],
            category="nowcast", handler=wrap_legacy(_nowcast_bvar)),
        CommandSpec(path=["nowcast", "bridge"], summary="Nowcast via bridge equations",
            args=data_arg,
            options=[
                OptionSpec(name="monthly-vars", type=Int, default=0, description="Number of monthly variables"),
                OptionSpec(name="quarterly-vars", type=Int, default=0, description="Number of quarterly variables"),
                OptionSpec(name="lag-m", type=Int, default=1, description="Monthly indicator lags"),
                OptionSpec(name="lag-q", type=Int, default=1, description="Quarterly indicator lags"),
                OptionSpec(name="lag-y", type=Int, default=1, description="Dependent variable lags"),
                OptionSpec(name="target-var", type=Int, default=0, description="Target variable index (0=last)"),
                out_fmt...,
            ],
            tables=[TableSpec(name=:nowcast_bridge,
                              description="Nowcast, forecast and number of bridge equations")],
            category="nowcast", handler=wrap_legacy(_nowcast_bridge)),
        CommandSpec(path=["nowcast", "news"], summary="Nowcast news decomposition (Banbura & Modugno 2014)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="data-new", type=String, default="", description="Path to new vintage CSV"),
                OptionSpec(name="data-old", type=String, default="", description="Path to old vintage CSV"),
                OptionSpec(name="monthly-vars", type=Int, default=0, description="Number of monthly variables"),
                OptionSpec(name="quarterly-vars", type=Int, default=0, description="Number of quarterly variables"),
                OptionSpec(name="method", type=String, default="dfm", choices=["dfm", "bvar"], description="dfm|bvar"),
                OptionSpec(name="factors", short="r", type=Int, default=2, description="Number of factors (DFM)"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Factor VAR lags"),
                OptionSpec(name="target-period", type=Int, default=0, description="Target period (0=last)"),
                OptionSpec(name="target-var", type=Int, default=0, description="Target variable index (0=last)"),
                out_fmt..., PLOT_OPTIONS...,
            ],
            tables=[TableSpec(name=:nowcast_news_decomposition,
                              description="Per-variable news impact on the nowcast revision")],
            flags=copy(PLOT_FLAGS), category="nowcast", handler=wrap_legacy(_nowcast_news)),
        CommandSpec(path=["nowcast", "forecast"], summary="Forecast from a nowcasting model",
            args=data_arg,
            options=[
                OptionSpec(name="monthly-vars", type=Int, default=0, description="Number of monthly variables"),
                OptionSpec(name="quarterly-vars", type=Int, default=0, description="Number of quarterly variables"),
                OptionSpec(name="method", type=String, default="dfm", choices=["dfm", "bvar", "bridge"], description="dfm|bvar|bridge"),
                OptionSpec(name="factors", short="r", type=Int, default=2, description="Number of factors (DFM)"),
                OptionSpec(name="lags", short="p", type=Int, default=1, description="Factor VAR lags"),
                OptionSpec(name="horizons", type=Int, default=4, description="Forecast horizon"),
                OptionSpec(name="target-var", type=Int, default=0, description="Target variable index (0=last)"),
                out_fmt..., PLOT_OPTIONS...,
            ],
            tables=[TableSpec(name=:nowcast_forecast,
                              description="Forecast path by horizon, one column per variable")],
            flags=copy(PLOT_FLAGS), category="nowcast", handler=wrap_legacy(_nowcast_forecast)),
    ]
end

function register_nowcast_commands!()
    specs = with_default_csv_kinds(with_data_kinds(nowcast_specs(), [:timeseries, :csv]))
    register!(specs)
    return build_node("nowcast", specs;
        description="Nowcasting: DFM, BVAR, bridge equations, news decomposition")
end

# ── Helpers ──────────────────────────────────────────────

"""Validate that monthly_vars + quarterly_vars == total columns, or infer split."""
function _validate_nowcast_vars(Y::AbstractMatrix, monthly_vars::Int, quarterly_vars::Int)
    N = size(Y, 2)
    nM = monthly_vars
    nQ = quarterly_vars
    if nM == 0 && nQ == 0
        # Default: treat all as monthly with last as quarterly target
        nM = N - 1
        nQ = 1
    end
    if nM + nQ != N
        error("monthly-vars ($nM) + quarterly-vars ($nQ) must equal number of variables ($N)")
    end
    return nM, nQ
end

# ── Handlers ─────────────────────────────────────────────

function _nowcast_dfm(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                       factors::Int=2, lags::Int=1, idio::String="ar1",
                       max_iter::Int=100, target_var::Int=0,
                       output::String="", format::String="table",
                       plot::Bool=false, plot_save::String="")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    _status("Nowcast DFM: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    _status("  Factors: $factors, VAR lags: $lags, idiosyncratic: $idio")
    _status()

    model = nowcast_dfm(Y, nM, nQ; r=factors, p=lags, idio=Symbol(idio), max_iter=max_iter)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    _maybe_plot(result; plot=plot, plot_save=plot_save)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    _status("  Target: $target_name (index $idx)")
    _status_styled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    _status_styled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    _status()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "log-likelihood", "EM iterations"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               round(model.loglik; digits=2), model.n_iter]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast DFM (r=$factors, p=$lags, target=$target_name)", key="nowcast_dfm")
end

function _nowcast_bvar(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                        lags::Int=5, target_var::Int=0,
                        prior::String="conjugate", theta_cross::String="",
                        lambda0::Float64=0.2, theta0::Float64=1.0,
                        miu0::Float64=1.0, alpha0::Float64=2.0,
                        output::String="", format::String="table")
    # --theta-cross is a Litterman-only knob: under the conjugate NIW prior the
    # cross/own tightness ratio is pinned by Σ whatever the dummy rows are (MEMs#602),
    # so upstream rejects the combination — guard it as a typed usage error up front.
    tc = nothing
    if !isempty(theta_cross)
        prior == "conjugate" && throw(CliError("usage/invalid",
            "nowcast bvar: --theta-cross is not a parameter of the conjugate prior " *
            "(the dummy-observation NIW pins the cross/own tightness ratio via Σ)";
            hint="use --prior litterman, which fixes Σ and frees the cross-lag knob"))
        tc = tryparse(Float64, theta_cross)
        (tc === nothing || tc <= 0) && throw(CliError("usage/invalid",
            "nowcast bvar: --theta-cross must be a positive number (got '$theta_cross')"))
    end
    lambda0 > 0 || throw(CliError("usage/invalid", "nowcast bvar: --lambda0 must be > 0 (got $lambda0)"))
    theta0 > 0 || throw(CliError("usage/invalid", "nowcast bvar: --theta0 must be > 0 (got $theta0)"))
    miu0 > 0 || throw(CliError("usage/invalid", "nowcast bvar: --miu0 must be > 0 (got $miu0)"))
    alpha0 > 0 || throw(CliError("usage/invalid", "nowcast bvar: --alpha0 must be > 0 (got $alpha0)"))

    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    _status("Nowcast BVAR: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    _status("  Lags: $lags, Prior: $prior" * (tc === nothing ? "" : ", theta_cross0=$tc"))
    _status()

    model = tc === nothing ?
        nowcast_bvar(Y, nM, nQ; lags=lags, lambda0=lambda0, theta0=theta0,
                     miu0=miu0, alpha0=alpha0, prior=Symbol(prior)) :
        nowcast_bvar(Y, nM, nQ; lags=lags, lambda0=lambda0, theta0=theta0,
                     miu0=miu0, alpha0=alpha0, prior=Symbol(prior), theta_cross0=tc)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    _status("  Target: $target_name (index $idx)")
    _status_styled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    _status_styled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    _status()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "log-likelihood"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               round(model.loglik; digits=2)]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast BVAR (lags=$lags, prior=$prior, target=$target_name)", key="nowcast_bvar")

    # Optimized hyperparameters (Nelder-Mead), not the initial values. loglik is NOT
    # comparable across priors (conjugate integrates Σ out; Litterman holds it fixed) —
    # the settings table records the prior so a reader can't compare across it blindly.
    settings = Pair{String,Any}[
        "prior" => string(model.prior),
        "lambda (overall shrinkage)" => round(model.lambda; digits=6),
        "theta (lag decay)" => round(model.theta; digits=6),
        "miu (sum-of-coefficients)" => round(model.miu; digits=6),
        "alpha (co-persistence)" => round(model.alpha; digits=6),
    ]
    isnan(model.theta_cross) ||
        push!(settings, "theta_cross (cross-variable tightness)" => round(model.theta_cross; digits=6))
    output_kv(settings; format=format, output=_per_var_output_path(output, "settings"),
              title="Nowcast BVAR Hyperparameters")
    prior == "litterman" &&
        _status("Note: loglik under litterman is a fixed-Σ objective — not comparable to the conjugate prior's value.")
end

function _nowcast_bridge(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                          lag_m::Int=1, lag_q::Int=1, lag_y::Int=1, target_var::Int=0,
                          output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    _status("Nowcast Bridge: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    _status("  Lags: lagM=$lag_m, lagQ=$lag_q, lagY=$lag_y")
    _status()

    model = nowcast_bridge(Y, nM, nQ; lagM=lag_m, lagQ=lag_q, lagY=lag_y)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    _status("  Target: $target_name (index $idx)")
    _status_styled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    _status_styled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    _status()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "n_equations"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               model.n_equations]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast Bridge (lagM=$lag_m, lagQ=$lag_q, lagY=$lag_y, target=$target_name)",
                  key="nowcast_bridge")
end

function _nowcast_news(; data_new::String="", data_old::String="",
                        monthly_vars::Int=0, quarterly_vars::Int=0,
                        method::String="dfm", factors::Int=2, lags::Int=1,
                        target_period::Int=0, target_var::Int=0,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    isempty(data_new) && throw(CliError("usage/missing", "--data-new is required";
        hint="path to the newer data vintage"))
    isempty(data_old) && throw(CliError("usage/missing", "--data-old is required";
        hint="path to the older data vintage"))

    Y_new, varnames_new = load_multivariate_data(data_new)
    Y_old, _ = load_multivariate_data(data_old)
    nM, nQ = _validate_nowcast_vars(Y_new, monthly_vars, quarterly_vars)
    T_new, N = size(Y_new)
    T_old = size(Y_old, 1)

    _status("Nowcast News: $N variables ($nM monthly, $nQ quarterly)")
    _status("  Old vintage: T=$T_old, New vintage: T=$T_new")
    _status("  Method: $method")
    _status()

    # Estimate model on old data
    model = if method == "dfm"
        nowcast_dfm(Y_old, nM, nQ; r=factors, p=lags)
    elseif method == "bvar"
        nowcast_bvar(Y_old, nM, nQ; lags=lags)
    else
        throw(CliError("usage/invalid",
            "unknown nowcast method for news: $method"; hint="expected dfm|bvar"))
    end

    tp = target_period > 0 ? target_period : T_new
    tv = target_var > 0 ? target_var : size(Y_new, 2)
    # The two vintages must be the SAME shape — a vintage differs by which cells are
    # filled in, not by row count. Upstream signals a mismatch with a bare
    # ArgumentError, which would exit 1 rather than flag the user's input.
    news = try
        nowcast_news(Y_new, Y_old, model, tp; target_var=tv)
    catch e
        e isa ArgumentError && throw(CliError("data/shape",
            "the two vintages must have the same dimensions";
            hint="new is $(size(Y_new)), old is $(size(Y_old)); a newer vintage fills in " *
                 "missing cells rather than adding rows"))
        rethrow()
    end

    _maybe_plot(news; plot=plot, plot_save=plot_save)

    _status_styled("  Old nowcast: $(round(news.old_nowcast; digits=4))\n"; color=:yellow)
    _status_styled("  New nowcast: $(round(news.new_nowcast; digits=4))\n"; color=:green)
    revision = news.new_nowcast - news.old_nowcast
    _status_styled("  Revision: $(round(revision; digits=4))\n"; color=:cyan)
    _status()

    # News impact table
    result_df = DataFrame(
        variable=news.variable_names,
        news_impact=round.(news.impact_news; digits=6)
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast News Decomposition (method=$method)", key="nowcast_news_decomposition")
end

function _nowcast_forecast(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                             method::String="dfm", factors::Int=2, lags::Int=1,
                             horizons::Int=4, target_var::Int=0,
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    _status("Nowcast Forecast: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    _status("  Method: $method, horizons: $horizons")
    _status()

    model = if method == "dfm"
        nowcast_dfm(Y, nM, nQ; r=factors, p=lags)
    elseif method == "bvar"
        nowcast_bvar(Y, nM, nQ; lags=lags)
    elseif method == "bridge"
        nowcast_bridge(Y, nM, nQ)
    else
        error("unknown nowcast method: $method (expected dfm|bvar|bridge)")
    end

    tv = target_var > 0 ? target_var : nothing
    fc_mat = forecast(model, horizons; target_var=tv)

    _maybe_plot(fc_mat; plot=plot, plot_save=plot_save)

    # Build forecast table
    fc_df = DataFrame()
    fc_df[!, :horizon] = 1:horizons
    for (i, vname) in enumerate(varnames)
        if i <= size(fc_mat, 2)
            fc_df[!, Symbol(vname)] = round.(fc_mat[:, i]; digits=6)
        end
    end

    output_result(fc_df; format=Symbol(format), output=output,
                  title="Nowcast Forecast ($method, h=$horizons)", key="nowcast_forecast")
end
