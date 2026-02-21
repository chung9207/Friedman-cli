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

function register_nowcast_commands!()
    nc_dfm = LeafCommand("dfm", _nowcast_dfm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; type=Int, default=0, description="Number of monthly variables (first N columns)"),
            Option("quarterly-vars"; type=Int, default=0, description="Number of quarterly variables (remaining columns)"),
            Option("factors"; short="r", type=Int, default=2, description="Number of factors"),
            Option("lags"; short="p", type=Int, default=1, description="Factor VAR lags"),
            Option("idio"; type=String, default="ar1", description="Idiosyncratic component: ar1|iid"),
            Option("max-iter"; type=Int, default=100, description="Maximum EM iterations"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Nowcast via Dynamic Factor Model (EM algorithm)")

    nc_bvar = LeafCommand("bvar", _nowcast_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; type=Int, default=0, description="Number of quarterly variables"),
            Option("lags"; short="p", type=Int, default=5, description="VAR lags"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Nowcast via Bayesian VAR")

    nc_bridge = LeafCommand("bridge", _nowcast_bridge;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; type=Int, default=0, description="Number of quarterly variables"),
            Option("lag-m"; type=Int, default=1, description="Monthly indicator lags"),
            Option("lag-q"; type=Int, default=1, description="Quarterly indicator lags"),
            Option("lag-y"; type=Int, default=1, description="Dependent variable lags"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Nowcast via bridge equations")

    nc_news = LeafCommand("news", _nowcast_news;
        args=Argument[],
        options=[
            Option("data-new"; type=String, default="", description="Path to new vintage CSV"),
            Option("data-old"; type=String, default="", description="Path to old vintage CSV"),
            Option("monthly-vars"; type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; type=Int, default=0, description="Number of quarterly variables"),
            Option("method"; type=String, default="dfm", description="dfm|bvar"),
            Option("factors"; short="r", type=Int, default=2, description="Number of factors (DFM)"),
            Option("lags"; short="p", type=Int, default=1, description="Factor VAR lags"),
            Option("target-period"; type=Int, default=0, description="Target period (0=last)"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Nowcast news decomposition (Banbura & Modugno 2014)")

    nc_forecast = LeafCommand("forecast", _nowcast_forecast;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("monthly-vars"; type=Int, default=0, description="Number of monthly variables"),
            Option("quarterly-vars"; type=Int, default=0, description="Number of quarterly variables"),
            Option("method"; type=String, default="dfm", description="dfm|bvar|bridge"),
            Option("factors"; short="r", type=Int, default=2, description="Number of factors (DFM)"),
            Option("lags"; short="p", type=Int, default=1, description="Factor VAR lags"),
            Option("horizons"; short="h", type=Int, default=4, description="Forecast horizon"),
            Option("target-var"; type=Int, default=0, description="Target variable index (0=last)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Forecast from a nowcasting model")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "dfm"      => nc_dfm,
        "bvar"     => nc_bvar,
        "bridge"   => nc_bridge,
        "news"     => nc_news,
        "forecast" => nc_forecast,
    )
    return NodeCommand("nowcast", subcmds, "Nowcasting: DFM, BVAR, bridge equations, news decomposition")
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

    println("Nowcast DFM: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    println("  Factors: $factors, VAR lags: $lags, idiosyncratic: $idio")
    println()

    model = nowcast_dfm(Y, nM, nQ; r=factors, p=lags, idio=Symbol(idio), max_iter=max_iter)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    _maybe_plot(result; plot=plot, plot_save=plot_save)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    println("  Target: $target_name (index $idx)")
    printstyled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    printstyled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    println()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "log-likelihood", "EM iterations"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               round(model.loglik; digits=2), model.n_iter]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast DFM (r=$factors, p=$lags, target=$target_name)")
end

function _nowcast_bvar(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                        lags::Int=5, target_var::Int=0,
                        output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    println("Nowcast BVAR: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    println("  Lags: $lags")
    println()

    model = nowcast_bvar(Y, nM, nQ; lags=lags)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    println("  Target: $target_name (index $idx)")
    printstyled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    printstyled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    println()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "log-likelihood"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               round(model.loglik; digits=2)]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast BVAR (lags=$lags, target=$target_name)")
end

function _nowcast_bridge(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                          lag_m::Int=1, lag_q::Int=1, lag_y::Int=1, target_var::Int=0,
                          output::String="", format::String="table")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    println("Nowcast Bridge: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    println("  Lags: lagM=$lag_m, lagQ=$lag_q, lagY=$lag_y")
    println()

    model = nowcast_bridge(Y, nM, nQ; lagM=lag_m, lagQ=lag_q, lagY=lag_y)
    tv = target_var > 0 ? target_var : nothing
    result = nowcast(model; target_var=tv)

    idx = result.target_index
    target_name = idx <= length(varnames) ? varnames[idx] : "var_$idx"
    println("  Target: $target_name (index $idx)")
    printstyled("  Nowcast: $(round(result.nowcast; digits=4))\n"; color=:green)
    printstyled("  Forecast: $(round(result.forecast; digits=4))\n"; color=:cyan)
    println()

    result_df = DataFrame(
        metric=["nowcast", "forecast", "n_equations"],
        value=[round(result.nowcast; digits=6), round(result.forecast; digits=6),
               model.n_equations]
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast Bridge (lagM=$lag_m, lagQ=$lag_q, lagY=$lag_y, target=$target_name)")
end

function _nowcast_news(; data_new::String="", data_old::String="",
                        monthly_vars::Int=0, quarterly_vars::Int=0,
                        method::String="dfm", factors::Int=2, lags::Int=1,
                        target_period::Int=0, target_var::Int=0,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    isempty(data_new) && error("--data-new is required")
    isempty(data_old) && error("--data-old is required")

    Y_new, varnames_new = load_multivariate_data(data_new)
    Y_old, _ = load_multivariate_data(data_old)
    nM, nQ = _validate_nowcast_vars(Y_new, monthly_vars, quarterly_vars)
    T_new, N = size(Y_new)
    T_old = size(Y_old, 1)

    println("Nowcast News: $N variables ($nM monthly, $nQ quarterly)")
    println("  Old vintage: T=$T_old, New vintage: T=$T_new")
    println("  Method: $method")
    println()

    # Estimate model on old data
    model = if method == "dfm"
        nowcast_dfm(Y_old, nM, nQ; r=factors, p=lags)
    elseif method == "bvar"
        nowcast_bvar(Y_old, nM, nQ; lags=lags)
    else
        error("unknown nowcast method for news: $method (expected dfm|bvar)")
    end

    tp = target_period > 0 ? target_period : T_new
    tv = target_var > 0 ? target_var : size(Y_new, 2)
    news = nowcast_news(Y_new, Y_old, model, tp; target_var=tv)

    _maybe_plot(news; plot=plot, plot_save=plot_save)

    printstyled("  Old nowcast: $(round(news.old_nowcast; digits=4))\n"; color=:yellow)
    printstyled("  New nowcast: $(round(news.new_nowcast; digits=4))\n"; color=:green)
    revision = news.new_nowcast - news.old_nowcast
    printstyled("  Revision: $(round(revision; digits=4))\n"; color=:cyan)
    println()

    # News impact table
    result_df = DataFrame(
        variable=news.variable_names,
        news_impact=round.(news.impact_news; digits=6)
    )
    output_result(result_df; format=Symbol(format), output=output,
                  title="Nowcast News Decomposition (method=$method)")
end

function _nowcast_forecast(; data::String, monthly_vars::Int=0, quarterly_vars::Int=0,
                             method::String="dfm", factors::Int=2, lags::Int=1,
                             horizons::Int=4, target_var::Int=0,
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    Y, varnames = load_multivariate_data(data)
    nM, nQ = _validate_nowcast_vars(Y, monthly_vars, quarterly_vars)
    T_obs, N = size(Y)

    println("Nowcast Forecast: $N variables ($nM monthly, $nQ quarterly), T=$T_obs")
    println("  Method: $method, horizons: $horizons")
    println()

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
                  title="Nowcast Forecast ($method, h=$horizons)")
end
