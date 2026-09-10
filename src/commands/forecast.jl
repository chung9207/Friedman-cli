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

# Forecast commands: var, bvar, lp, arima, static, dynamic, gdfm,
#                    arch, garch, egarch, gjr_garch, sv

# Evaluate --result is a comma-separated stem *string* (handle=false, empty
# result_types). RESULT_OPTION would make wrap_legacy load "a,b" as one path.
const FCEVAL_RESULT_OPTION = OptionSpec(
    name="result", type=String, default="",
    description="Comma-separated forecast-result handle stems (alternative to --forecasts columns)",
    handle=false,
)

function forecast_specs()::Vector{CommandSpec}
    return [
        # #73: ARFIMA gains the downstream verbs. forecast(::ARFIMAModel, h) returns an
        # ARIMAForecast (a real plot recipe exists for it, so plot flags are legitimate);
        # --trunc-lag is ARFIMA-specific and has no arima equivalent.
        CommandSpec(
            path=["forecast", "arfima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=0, description="AR order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="method", short="m", type=String, default="css", description="css|mle (fractional-integration estimator)", choices=["css","mle"]),
                OptionSpec(name="d0", type=Float64, default=nothing, description="Starting value for d (default: GPH pre-estimate)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations"),
                OptionSpec(name="horizons", short="H", type=Int, default=12, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Interval level in (0,1)"),
                OptionSpec(name="trunc-lag", type=Int, default=200, description="AR(inf) truncation lag for the fractional filter (≥ 1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:arfima_forecast, description="Point forecasts with interval bounds: horizon | forecast | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_arfima),
        ),
        # #67: MIDAS. NOTE there is deliberately NO --horizons: forecast(::MidasModel,
        # X_new) takes a high-frequency block, and the horizon is fixed at estimation
        # time via --horizon (stored as m.h). Advertising --horizons would be a lie.
        # Also no plot flags: MEMs has a plot_result(::MidasModel) recipe (the weight
        # curve) but none for MidasForecast.
        CommandSpec(
            path=["forecast", "midas"],
            summary="Path to low-frequency target CSV",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to low-frequency target CSV")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Target column index (1-based)"),
                OptionSpec(name="hf-data", type=String, default="", description="Path to the high-frequency indicator CSV (required)"),
                OptionSpec(name="hf-column", type=Int, default=1, description="High-frequency column index (1-based)"),
                OptionSpec(name="m", type=Int, default=0, description="High-frequency observations per low-frequency period (required, ≥ 1)"),
                OptionSpec(name="k", type=Int, default=0, description="Number of high-frequency lags K (required, ≥ 1)"),
                OptionSpec(name="weights", type=String, default="expalmon", description="MIDAS weight family", choices=["expalmon","beta2","beta3","almon","umidas"]),
                OptionSpec(name="p-ar", type=Int, default=0, description="Autoregressive lags of the target (ADL-MIDAS)"),
                OptionSpec(name="poly-degree", type=Int, default=2, description="Almon polynomial degree (≥ 0)"),
                OptionSpec(name="horizon", type=Int, default=1, description="Direct forecast horizon, fixed at estimation"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimizer iterations"),
                OptionSpec(name="level", type=Float64, default=0.95, description="Prediction-interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:midas_forecast, description="Direct h-step forecast of the low-frequency target: horizon | forecast | lower | upper | se"),
                    TableSpec(name=:midas_forecast_summary, description="Horizon, HF lags K, HF-per-LF ratio m, AR lags, weight family and interval level")],
            category="forecast",
            handler=wrap_legacy(_forecast_midas),
        ),
        # C064 remainder (#69): the six C064a GARCH variants gain this verb.
        # garch-midas has NO --conf-level: forecast(::GarchMidasModel, h) takes none.
        CommandSpec(
            path=["forecast", "igarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Forecast interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                # plot-save is an OPTION (a String path), NOT a flag — a FlagSpec here
                # binds Bool to the handler's plot_save::String and TypeErrors on every
                # invocation (the #85 declared-vs-handler mismatch).
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:igarch_volatility_forecast, description="Conditional variance path: horizon | variance | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_igarch),
        ),
        CommandSpec(
            path=["forecast", "cgarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Forecast interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                # plot-save is an OPTION (a String path), NOT a flag — a FlagSpec here
                # binds Bool to the handler's plot_save::String and TypeErrors on every
                # invocation (the #85 declared-vs-handler mismatch).
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:cgarch_volatility_forecast, description="Conditional variance path: horizon | variance | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_cgarch),
        ),
        CommandSpec(
            path=["forecast", "aparch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="fix-delta", type=Float64, default=nothing, description="Fix the power parameter delta"),
                OptionSpec(name="fix-gamma", type=Float64, default=nothing, description="Fix the asymmetry parameter gamma"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Forecast interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                # plot-save is an OPTION (a String path), NOT a flag — a FlagSpec here
                # binds Bool to the handler's plot_save::String and TypeErrors on every
                # invocation (the #85 declared-vs-handler mismatch).
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:aparch_volatility_forecast, description="Conditional variance path: horizon | variance | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_aparch),
        ),
        CommandSpec(
            path=["forecast", "figarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Forecast interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                # plot-save is an OPTION (a String path), NOT a flag — a FlagSpec here
                # binds Bool to the handler's plot_save::String and TypeErrors on every
                # invocation (the #85 declared-vs-handler mismatch).
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:figarch_volatility_forecast, description="Conditional variance path: horizon | variance | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_figarch),
        ),
        CommandSpec(
            path=["forecast", "fiegarch"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="GARCH order p"),
                OptionSpec(name="q", type=Int, default=1, description="ARCH order q"),
                OptionSpec(name="d0", type=Float64, default=0.4, description="Initial fractional differencing parameter"),
                OptionSpec(name="truncation", type=Int, default=1000, description="Truncation lag for the ARCH(inf) expansion"),
                OptionSpec(name="dist", type=String, default="normal", description="Innovation distribution"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Forecast interval level in (0,1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                # plot-save is an OPTION (a String path), NOT a flag — a FlagSpec here
                # binds Bool to the handler's plot_save::String and TypeErrors on every
                # invocation (the #85 declared-vs-handler mismatch).
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:fiegarch_volatility_forecast, description="Conditional variance path: horizon | variance | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_fiegarch),
        ),
        CommandSpec(
            path=["forecast", "garch-midas"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="m-freq", type=Int, default=0, description="High-frequency observations per low-frequency block (required, ≥ 1)"),
                OptionSpec(name="k", type=Int, default=12, description="Number of MIDAS lags"),
                OptionSpec(name="rv", type=String, default="realized", description="Long-run driver", choices=["realized","macro"]),
                OptionSpec(name="span", type=String, default="fixed", description="Span", choices=["fixed","rolling"]),
                OptionSpec(name="config", type=String, default="", description="TOML with [garch_midas] x_lf (required for --rv macro)"),
                OptionSpec(name="horizons", short="H", type=Int, default=10, description="Forecast horizons (≥ 1)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            # NO plot flags: MEMs ships no plot_result for the NamedTuple that
            # forecast(::GarchMidasModel, h) returns, and advertising --plot would be an
            # uncaught MethodError (the C065a rule).
            flags=FlagSpec[],
            tables=[TableSpec(name=:garch_midas_volatility_forecast, description="Variance path split into its components: horizon | total_variance | long_run | short_run | volatility")],
            category="forecast",
            handler=wrap_legacy(_forecast_garch_midas),
        ),
        CommandSpec(
            path=["forecast", "var"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Confidence level for intervals"),
                OptionSpec(name="ci-method", type=String, default="analytical", description="analytical|bootstrap"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:var_forecast, description="Point forecasts with interval bounds, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_var),
        ),
        CommandSpec(
            path=["forecast", "scenario"],
            summary="Waggoner-Zha conditional (scenario) forecast",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="conditions-file", type=String, default="", description="REQUIRED long-format CSV: variable,period,value[,sd]"),
                OptionSpec(name="method", type=String, default="var", description="Model to condition: var|bvar", choices=["var","bvar"]),
                OptionSpec(name="lags", short="p", type=Int, default=nothing, description="Lag order (default: auto for var, 4 for bvar)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="replications", type=Int, default=1000, description="Draws used for the conditional bands"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Confidence level in (0, 1)"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws (--method bvar)"),
                OptionSpec(name="sampler", type=String, default="direct", description="direct|gibbs (--method bvar)"),
                OptionSpec(name="config", type=String, default="", description="TOML config for the BVAR prior"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:conditional_forecast, description="Conditioned path beside the unconditional baseline: horizon | variable | value | lower | upper | unconditional"),
                    TableSpec(name=:implied_structural_shocks, description="Shocks that deliver the scenario: horizon | shock | value"),
                    TableSpec(name=:scenario_settings, description="Model, horizon, condition count, confidence level, identification and draws used")],
            category="forecast",
            handler=wrap_legacy(_forecast_scenario),
        ),
        CommandSpec(
            path=["forecast", "bvar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Lag order"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="draws", short="n", type=Int, default=2000, description="MCMC draws"),
                OptionSpec(name="sampler", type=String, default="direct", description="direct|gibbs"),
                OptionSpec(name="config", type=String, default="", description="TOML config for prior hyperparameters"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file")
            ],
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:bvar_forecast, description="Posterior-mean forecasts with 68% credible bands, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_bvar),
        ),
        CommandSpec(
            path=["forecast", "lp"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="shock", type=Int, default=1, description="Shock variable index (1-based)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="shock-size", type=Float64, default=1.0, description="Impulse shock size"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="LP control lags"),
                OptionSpec(name="vcov", type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
                OptionSpec(name="ci-method", type=String, default="analytical", description="analytical|bootstrap|none"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="n-boot", type=Int, default=500, description="Bootstrap replications"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:lp_forecast, description="Local-projection forecast along the shock path, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_lp),
        ),
        CommandSpec(
            path=["forecast", "arima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=nothing, description="AR order (default: auto selection)"),
                OptionSpec(name="d", type=Int, default=0, description="Differencing order"),
                OptionSpec(name="q", type=Int, default=0, description="MA order"),
                OptionSpec(name="max-p", type=Int, default=5, description="Max AR order for auto selection"),
                OptionSpec(name="max-d", type=Int, default=2, description="Max differencing order for auto selection"),
                OptionSpec(name="max-q", type=Int, default=5, description="Max MA order for auto selection"),
                OptionSpec(name="criterion", type=String, default="bic", description="aic|bic"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="method", short="m", type=String, default="css_mle", description="ols|css|mle|css_mle"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:arima_forecast, description="Point forecasts with interval bounds, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_arima),
        ),
        # C065a: SETAR bootstrap-simulation forecast. Re-estimates the SETAR then simulates
        # `forecast(::ThresholdModel, h)` → ThresholdForecast (AbstractForecastResult → tidy
        # long_table). `forecast` is SETAR-only upstream, but `estimate_setar` always sets
        # is_setar=true, so no extra guard; `--transition-col` is NOT offered (that path exists
        # only for STAR external-s models, which are not forecastable). `--ci-level` MUST be
        # exactly 0.90/0.95/0.99 (Hansen 2000 tabulation used by the re-estimated threshold CI).
        # NO `--plot`/`--plot-save`: MEMs 0.7.0 ships NO `plot_result(::ThresholdForecast)` recipe
        # (only `ThresholdModel`/`STARModel` + the 8 registered forecast types are plottable), so
        # advertising the flag would drive `_maybe_plot` into an uncaught MethodError → exit 1. Per
        # the C051 convention only plot-capable leaves add the flags; revisit if MEMs adds one.
        CommandSpec(
            path=["forecast", "setar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=String, default="1", description="Delay lag: an integer ≥ 1, or 'auto' (=1:p grid)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon (≥ 1)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Bootstrap simulation paths (≥ 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.95, description="Band coverage: 0.90|0.95|0.99", choices=["0.90", "0.95", "0.99"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:setar_forecast, description="Bootstrap-simulated forecasts with bands, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_setar),
        ),
        # C065b: STAR bootstrap-simulation forecast. Re-estimates a SELF-EXCITING STAR then
        # simulates `forecast(::STARModel, h)` → STARForecast (AbstractForecastResult → tidy
        # long_table). `--transition-col` is NOT offered: `forecast(::STARModel)` throws unless
        # the model is self-exciting (sₜ = y[t-d]), so external-s STARs are not forecastable.
        # NO `--plot`/`--plot-save`: MEMs 0.7.0 ships NO `plot_result(::STARForecast)` recipe
        # (only `ThresholdModel`/`STARModel` + the 8 registered forecast types are plottable), so
        # advertising the flag would drive `_maybe_plot` into an uncaught MethodError → exit 1
        # (the identical gap fixed for `forecast setar`). Per the C051 convention only plot-capable
        # leaves add the flags; revisit if MEMs adds a STARForecast recipe.
        CommandSpec(
            path=["forecast", "star"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="d", type=Int, default=1, description="Delay lag for the self-exciting transition var (≥ 1)"),
                OptionSpec(name="type", type=String, default="auto", description="Transition shape: lstr1|lstr2|estr|auto", choices=["lstr1", "lstr2", "estr", "auto"]),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon (≥ 1)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Bootstrap simulation paths (≥ 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.95, description="Band coverage: 0.90|0.95|0.99", choices=["0.90", "0.95", "0.99"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table", "csv", "json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:star_forecast, description="Bootstrap-simulated forecasts with bands, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_star),
        ),
        # W6/#108: SARIMA forecast. `forecast(::SARIMAModel, h)` returns an ARIMAForecast,
        # which DOES have a plot_result recipe — unlike the threshold/STAR/MS forecast types
        # — so this leaf legitimately carries the plot flags.
        CommandSpec(
            path=["forecast", "sarima"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[SARIMA_OPTIONS...,
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon (>= 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.95, description="Band coverage, 0 < level < 1"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                PLOT_OPTIONS...],
            flags=[SARIMA_FLAGS..., PLOT_FLAGS...],
            tables=[TableSpec(name=:sarima_forecast, description="Point forecasts with interval bounds, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_sarima),
        ),
        # W3/#101: Markov-switching forecasts, un-gated by MEMs#510. The two dispatches are
        # mutually exclusive upstream and each throws on the other's model type:
        # `forecast(m, h)` requires :ms_ar, `forecast(m, X_new)` requires :regression with an
        # h x k matrix of FUTURE regressors — hence `--x-future` on `forecast ms` only.
        # NO `--plot`/`--plot-save` on either: MEMs 0.7.2 ships no `plot_result(::MSForecast)`
        # recipe (verified on the tag, same gap as ThresholdForecast/STARForecast), so
        # advertising them would drive `_maybe_plot` into an uncaught MethodError → exit 1.
        CommandSpec(
            path=["forecast", "ms-ar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
                OptionSpec(name="p", type=Int, default=1, description="AR order (≥ 1)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=1000, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon (≥ 1)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Simulated regime paths for the bands (≥ 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.90, description="Band coverage, 0 < level < 1"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="switching-variance", description="Let σ² switch across regimes (default: off, Hamilton form)")],
            tables=[TableSpec(name=:ms_ar_forecast, description="Regime-averaged forecasts with simulated bands, tidy long form: horizon | variable | value | lower | upper"),
                    TableSpec(name=:ms_ar_predicted_regime_probabilities, description="Predicted regime probabilities: horizon | one column per regime")],
            category="forecast",
            handler=wrap_legacy(_forecast_ms_ar),
        ),
        CommandSpec(
            path=["forecast", "ms"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="dep", type=String, default="", description="Dependent variable column (default: first numeric)"),
                OptionSpec(name="k-regimes", type=Int, default=2, description="Number of regimes (≥ 2)"),
                OptionSpec(name="max-iter", type=Int, default=500, description="Max EM iterations (≥ 1)"),
                OptionSpec(name="tol", type=Float64, default=1e-8, description="EM convergence tolerance (> 0)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon (intercept-only models; else use --x-future)"),
                OptionSpec(name="x-future", type=String, default="", description="CSV of future regressors: h rows x k columns (required unless intercept-only)"),
                OptionSpec(name="reps", type=Int, default=1000, description="Simulated regime paths for the bands (≥ 1)"),
                OptionSpec(name="ci-level", type=Float64, default=0.90, description="Band coverage, 0 < level < 1"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=[FlagSpec(name="no-switching-variance", description="Force common σ² across regimes (default: σ² switches)")],
            tables=[TableSpec(name=:ms_regression_forecast, description="Regime-averaged forecasts with simulated bands, tidy long form: horizon | variable | value | lower | upper"),
                    TableSpec(name=:ms_regression_predicted_regime_probabilities, description="Predicted regime probabilities: horizon | one column per regime")],
            category="forecast",
            handler=wrap_legacy(_forecast_ms),
        ),
        CommandSpec(
            path=["forecast", "static"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of factors (default: auto via IC)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="ci-method", type=String, default="none", description="none|bootstrap|parametric"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level for intervals"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:static_factor_forecast, description="Observable forecasts reconstructed from the factors, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_static),
        ),
        CommandSpec(
            path=["forecast", "dynamic"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="factor-lags", short="p", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="method", type=String, default="twostep", description="twostep|em"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:dynamic_factor_forecast, description="Observable forecasts reconstructed from the factors, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_dynamic),
        ),
        CommandSpec(
            path=["forecast", "gdfm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="nfactors", short="r", type=Int, default=nothing, description="Number of static factors (default: auto)"),
                OptionSpec(name="dynamic-rank", short="q", type=Int, default=nothing, description="Dynamic rank (default: auto)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="method", type=String, default="ar", description="Factor projection: ar (two-sided)|one-sided|spectral (FHLR 2005)", choices=["ar","one-sided","spectral"]),
                OptionSpec(name="spectral", type=String, default="lag-window", description="GDFM spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:gdfm_forecast, description="Observable forecasts from the generalized dynamic factor model, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_gdfm),
        ),
        CommandSpec(
            path=["forecast", "sdfm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="q", type=Int, default=nothing, description="Number of dynamic factors (default: auto via --q-method)"),
                OptionSpec(name="id", type=String, default="cholesky", description="cholesky|sign|proxy (--id proxy requires --instrument)"),
                OptionSpec(name="q-method", type=String, default="hallin-liska", description="Auto factor selection: hallin-liska|bai-ng|amengual-watson", choices=["hallin-liska","bai-ng","amengual-watson"]),
                OptionSpec(name="method", type=String, default="fglr", description="Structural estimator: fglr|gdfm-var (gdfm-var is the legacy path)", choices=["fglr","gdfm-var"]),
                OptionSpec(name="spectral", type=String, default="lag-window", description="GDFM spectrum: lag-window (FHLR)|smoothed-periodogram", choices=["lag-window","smoothed-periodogram"]),
                OptionSpec(name="instrument", type=String, default="", description="Proxy-instrument CSV column (only with --id proxy)"),
                OptionSpec(name="var-lags", type=Int, default=1, description="Factor VAR lag order"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="config", type=String, default="", description="TOML config for sign restrictions"),
                OptionSpec(name="ci", type=String, default="none", description="Interval method: none|bootstrap", choices=["none","bootstrap"]),
                OptionSpec(name="reps", type=Int, default=200, description="Bootstrap replications (with --ci bootstrap)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:sdfm_forecast, description="Panel forecasts from the structural dynamic factor model, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_sdfm),
        ),
        # Volatility 20-plex (forecast side): generated from VOL_MODELS
        _vol_specs(:forecast)...,
        CommandSpec(
            path=["forecast", "vecm"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Lag order (in levels)"),
                OptionSpec(name="rank", short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
                OptionSpec(name="deterministic", type=String, default="constant", description="none|constant|trend"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="ci-method", type=String, default="none", description="none|bootstrap|parametric"),
                OptionSpec(name="replications", type=Int, default=500, description="Bootstrap replications"),
                OptionSpec(name="confidence", type=Float64, default=0.95, description="Confidence level for intervals"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:vecm_forecast, description="Level forecasts with optional interval bounds, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_vecm),
        ),
        CommandSpec(
            path=["forecast", "favar"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="factors", short="r", type=Int, default=nothing, description="Number of factors (default: auto)"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="VAR lag order"),
                OptionSpec(name="key-vars", type=String, default="", description="Key variable names or indices (comma-separated)"),
                OptionSpec(name="horizons", type=Int, default=12, description="Forecast horizon"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="panel-forecast", description="Output panel-wide forecast instead of factor-level"),
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:favar_forecast, description="Factor-level (or panel-wide under --panel-forecast) forecasts, tidy long form: horizon | variable | value | lower | upper")],
            category="forecast",
            handler=wrap_legacy(_forecast_favar),
        ),
        # ── forecast evaluate: forecast evaluation & combination (C072, M5c) ──
        # Nested depth-3 sub-node. Uniform input: a CSV + --actual <col> +
        # --forecasts <c1,c2,...>; the handler forms errors / f_adj / the matrix.
        CommandSpec(
            path=["forecast", "evaluate", "metrics"],
            summary="Point forecast-accuracy metrics (ME/MAE/RMSE/MAPE/sMAPE/MASE/U1/U2) + Theil decomposition",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="Forecast column names, comma-separated (required, >=1)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="seasonal-period", type=Int, default=1, description="Seasonal lag for MASE naive-forecast scaling"),
                # #95: ForecastEvaluation has a real plot_result recipe (a bar chart of
                # the chosen metric); the other five `evaluate` leaves return types that
                # have none, so only this one gains plot flags.
                OptionSpec(name="plot-save", type=String, default="", description="Save interactive plot to HTML file"),
            ], OUTPUT_OPTIONS),
            flags=[FlagSpec(name="plot", description="Display an interactive plot")],
            tables=[TableSpec(name=:forecast_accuracy_metrics, description="Point accuracy metrics, one row per forecast: model | ME | MAE | RMSE | MAPE | sMAPE | MASE | U1 | U2"),
                    TableSpec(name=:theil_mse_decomposition, description="Theil MSE proportions summing to 1: model | bias | variance | covariance")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_metrics),
        ),
        CommandSpec(
            path=["forecast", "evaluate", "dm"],
            summary="Diebold-Mariano (1995) equal-predictive-accuracy test (exactly 2 forecasts)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="Two forecast column names, comma-separated (required)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="loss", type=String, default="se", choices=["se","ad"], description="Loss: se (squared) | ad (absolute)"),
                OptionSpec(name="horizon", type=Int, default=1, description="Forecast horizon (sets truncation lag h-1)"),
                OptionSpec(name="alternative", type=String, default="two-sided", choices=["two-sided","less","greater"], description="Alternative hypothesis"),
            ], OUTPUT_OPTIONS),
            flags=[FlagSpec(name="no-hln", description="Disable the Harvey-Leybourne-Newbold small-sample correction (use N(0,1))")],
            tables=[TableSpec(name=:diebold_mariano_test, description="DM statistic, p-value, mean loss differential, long-run variance and HLN setting")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_dm),
        ),
        CommandSpec(
            path=["forecast", "evaluate", "clark-west"],
            summary="Clark-West (2007) adjusted-MSPE test for nested models (exactly 2 forecasts: small then big)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="Two forecast columns: small (restricted), big (unrestricted)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="horizon", type=Int, default=1, description="Forecast horizon (sets truncation lag h-1)"),
                OptionSpec(name="alternative", type=String, default="greater", choices=["two-sided","less","greater"], description="Alternative hypothesis"),
            ], OUTPUT_OPTIONS),
            flags=FlagSpec[],
            tables=[TableSpec(name=:clark_west_test, description="CW statistic, p-value, mean adjusted difference and long-run variance")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_clark_west),
        ),
        CommandSpec(
            path=["forecast", "evaluate", "mincer-zarnowitz"],
            summary="Mincer-Zarnowitz (1969) forecast-efficiency regression (exactly 1 forecast)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="One forecast column name (required)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="lags", type=Int, default=0, description="Newey-West HAC truncation lag (0 = White)"),
                OptionSpec(name="kernel", type=String, default="bartlett", choices=["bartlett","parzen","quadratic_spectral","tukey_hanning"], description="HAC kernel"),
            ], OUTPUT_OPTIONS),
            flags=FlagSpec[],
            tables=[TableSpec(name=:mincer_zarnowitz_efficiency_test, description="Intercept a, slope b, HAC standard errors and the joint Wald/F test of (a,b)=(0,1)")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_mincer_zarnowitz),
        ),
        CommandSpec(
            path=["forecast", "evaluate", "encompassing"],
            summary="Harvey-Leybourne-Newbold (1998) forecast-encompassing test (exactly 2 forecasts)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="Two forecast column names, comma-separated (required)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="lags", type=Int, default=0, description="Newey-West HAC truncation lag (0 = White)"),
                OptionSpec(name="kernel", type=String, default="bartlett", choices=["bartlett","parzen","quadratic_spectral","tukey_hanning"], description="HAC kernel"),
            ], OUTPUT_OPTIONS),
            flags=FlagSpec[],
            tables=[TableSpec(name=:forecast_encompassing_test, description="Combination weights b1/b2 with the HAC t-test of b2=0")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_encompassing),
        ),
        CommandSpec(
            path=["forecast", "evaluate", "combine"],
            summary="Forecast combination (equal / Bates-Granger / Granger-Ramanathan weights; >=2 forecasts)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=vcat([
                OptionSpec(name="actual", type=String, default="", description="Realized-values column name (required)"),
                OptionSpec(name="forecasts", type=String, default="", description="Forecast column names, comma-separated (required, >=2)"),
                FCEVAL_RESULT_OPTION,
                OptionSpec(name="method", type=String, default="equal", choices=["equal","bates-granger","granger-ramanathan"], description="Combination method"),
            ], OUTPUT_OPTIONS),
            flags=[FlagSpec(name="emit-series", description="Also emit the combined forecast series (index|combined)")],
            tables=[TableSpec(name=:forecast_combination_weights, description="Combination weights summing to 1: model | weight | mse"),
                    TableSpec(name=:combined_forecast_series, description="The combined forecast itself (--emit-series): index | combined")],
            category="forecast",
            handler=wrap_legacy(_forecast_eval_combine),
        )
    ]
end

const _FORECAST_SLOT_TYPES = Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}}(
    ["forecast", "var"]         => ([:VARModel], [:VARForecast]),
    ["forecast", "bvar"]        => ([:BVARPosterior], [:BVARForecast]),
    ["forecast", "lp"]          => ([:LPModel], [:LPForecast]),
    ["forecast", "arima"]       => ([:ARIMAModel, :ARMAModel, :ARModel, :MAModel], [:ARIMAForecast]),
    ["forecast", "arfima"]      => ([:ARFIMAModel], [:ARIMAForecast]),
    ["forecast", "sarima"]      => ([:SARIMAModel], [:ARIMAForecast]),
    ["forecast", "setar"]       => ([:ThresholdModel], [:ThresholdForecast]),
    ["forecast", "star"]        => ([:STARModel], [:STARForecast]),
    ["forecast", "ms-ar"]       => ([:MSRegModel], [:MSForecast]),
    ["forecast", "ms"]          => ([:MSRegModel], [:MSForecast]),
    ["forecast", "static"]      => ([:FactorModel], [:FactorForecast]),
    ["forecast", "dynamic"]     => ([:DynamicFactorModel], [:FactorForecast]),
    ["forecast", "gdfm"]        => ([:GeneralizedDynamicFactorModel], [:FactorForecast]),
    ["forecast", "sdfm"]        => ([:StructuralDFM], [:FactorForecast]),
    ["forecast", "vecm"]        => ([:VECMModel], [:VECMForecast]),
    ["forecast", "favar"]       => ([:FAVARModel], [:VARForecast]),
    ["forecast", "scenario"]    => ([:VARModel, :BVARPosterior], [:ConditionalForecast]),
    ["forecast", "midas"]       => ([:MidasModel], [:MidasForecast]),
    ["forecast", "arch"]        => ([:ARCHModel], [:VolatilityForecast]),
    ["forecast", "garch"]       => ([:GARCHModel], [:VolatilityForecast]),
    ["forecast", "egarch"]      => ([:EGARCHModel], [:VolatilityForecast]),
    ["forecast", "gjr-garch"]   => ([:GJRGARCHModel], [:VolatilityForecast]),
    ["forecast", "sv"]          => ([:SVModel], [:VolatilityForecast]),
    ["forecast", "igarch"]      => ([:IGARCHModel], [:VolatilityForecast]),
    ["forecast", "cgarch"]      => ([:CGARCHModel], [:VolatilityForecast]),
    ["forecast", "aparch"]      => ([:APARCHModel], [:VolatilityForecast]),
    ["forecast", "figarch"]     => ([:FIGARCHModel], [:VolatilityForecast]),
    ["forecast", "fiegarch"]    => ([:FIEGARCHModel], [:VolatilityForecast]),
    ["forecast", "garch-midas"] => ([:GarchMidasModel], Symbol[]),
)

function register_forecast_commands!()
    all_specs = forecast_specs()
    # Evaluate leaves stay off with_model_option / with_result_handles: they declare
    # a string --result (FCEVAL_RESULT_OPTION, handle=false) parsed in _fceval_load.
    is_eval(s) = length(s.path) >= 2 && s.path[2] == "evaluate"
    producing = _tag_slot_types(filter(!is_eval, all_specs), _FORECAST_SLOT_TYPES)
    specs = with_config_ergonomics(vcat(
        with_result_handles(with_model_option(producing)),
        filter(is_eval, all_specs)))
    out = CommandSpec[]
    for s in specs
        kinds = is_eval(s) ? [:csv, :timeseries, :panel, :cross_section] : [:timeseries, :csv]
        push!(out, _copy_spec(s; data_kinds=kinds))
    end
    out = with_default_csv_kinds(out)
    register!(out)
    return build_node("forecast", out; description="Forecasting")
end


# ── VAR Forecast ─────────────────────────────────────────

function _forecast_var(; data::String="", result=nothing, model=nothing, lags=nothing, horizons::Int=12,
                        confidence::Float64=0.95, ci_method::String="analytical",
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    loaded = _loaded_result(result; data, model, lags, check_lags=true, leaf="forecast var")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="VAR Forecast", key="var_forecast", plot, plot_save)
    if isnothing(model)
        model, _, _, p = _load_and_estimate_var(data, lags)
    else
        p = model.p
    end

    # C051: render the forecast (point + CI) through MEMs' uniform tidy long_table
    # (horizon | variable | value | lower | upper), replacing the hand-built wide table
    # and the hand-rolled companion-matrix MSE. MEMs' symbol is :analytic (not :analytical).
    ci_sym = ci_method == "bootstrap" ? :bootstrap :
             ci_method == "none"      ? :none : :analytic

    _status("Computing VAR($p) forecast: horizons=$horizons, confidence=$confidence, ci=$ci_method")
    _status()

    fc_kw = ci_sym == :bootstrap ?
        (; ci_method=:bootstrap, reps=500, conf_level=confidence) :
        (; ci_method=ci_sym, conf_level=confidence)
    fc_result = forecast(model, horizons; fc_kw...)
    fc_df = long_table(fc_result)

    ci_label = ci_sym == :bootstrap ? "bootstrap $(Int(round(confidence*100)))% CI" :
               ci_sym == :none      ? "point forecast" :
               "$(Int(round(confidence*100)))% CI"
    output_result(fc_df; format=Symbol(format), output=output,
                  title="VAR($p) Forecast (h=$horizons, $ci_label)", key="var_forecast")
    _maybe_plot(fc_result; plot=plot, plot_save=plot_save)
    return (; model, result=fc_result)
end

# Normal quantile without importing Distributions (Abramowitz & Stegun 26.2.23)
function quantile_normal(p::Float64)
    if p < 0.5
        return -quantile_normal(1.0 - p)
    end
    t = sqrt(-2.0 * log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1*t + c2*t^2) / (1.0 + d1*t + d2*t^2 + d3*t^3)
end

# ── BVAR Forecast ────────────────────────────────────────

function _forecast_bvar(; data::String="", result=nothing, lags::Int=4, horizons::Int=12,
                         draws::Int=2000, sampler::String="direct",
                         config::String="",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="",
                         model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast bvar")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="Bayesian VAR Forecast", key="bvar_forecast", plot, plot_save)
    if isnothing(model)
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    else
        post = model
        varnames = post.varnames
        p = post.p
        n = length(varnames)
        Y = post.Y
    end

    _status("Computing Bayesian forecast: BVAR($p), horizons=$horizons")
    _status("  Sampler: $sampler, Draws: $draws")
    _status()

    # C051: route the posterior forecast through MEMs (→ BVARForecast with the posterior
    # mean + credible bands) and render its tidy long_table (horizon|variable|value|lower|
    # upper), replacing the hand-rolled per-draw simulation and quantile computation.
    fc = forecast(post, horizons; conf_level=0.68)
    # #95: BVARForecast has a real plot_result recipe upstream — this leaf simply never
    # advertised it.
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="Bayesian VAR($p) Forecast (h=$horizons, 68% credible interval)",
                  key="bvar_forecast")
    return (; model=post, result=fc)
end

# ── LP Forecast ──────────────────────────────────────────

function _forecast_lp(; data::String="", result=nothing, shock::Int=1, horizons::Int=12,
                       shock_size::Float64=1.0, lags::Int=4,
                       vcov::String="newey_west",
                       ci_method::String="analytical", conf_level::Float64=0.95,
                       n_boot::Int=500,
                       output::String="", format::String="table",
                       plot::Bool=false, plot_save::String="",
                       model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast lp")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="LP Forecast", key="lp_forecast", plot, plot_save)
    if isnothing(model)
        Y, varnames = load_multivariate_data(data)
        model = estimate_lp(Y, shock, horizons;
            lags=lags, cov_type=Symbol(vcov))
    else
        varnames = model.varnames
    end

    _status("Computing LP forecast: shock=$shock, horizons=$horizons, shock_size=$shock_size, ci=$ci_method")
    _status()

    shock_path = fill(shock_size, horizons)

    fc = forecast(model, shock_path;
        ci_method=Symbol(ci_method), conf_level=conf_level, n_boot=n_boot)

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    shock_name = _shock_name(varnames, shock)
    # C051: MEMs tidy long_table (horizon|variable|value|lower|upper).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="LP Forecast (shock=$shock_name, h=$horizons, $(Int(round(conf_level*100)))% CI)",
                  key="lp_forecast")
    return (; model, result=fc)
end

# ── ARIMA Forecast ───────────────────────────────────────

function _forecast_arima(; data::String="", result=nothing, column::Int=1, p=nothing, d::Int=0, q::Int=0,
                           max_p::Int=5, max_d::Int=2, max_q::Int=5,
                           criterion::String="bic", horizons::Int=12,
                           confidence::Float64=0.95, method::String="css_mle",
                           format::String="table", output::String="",
                           plot::Bool=false, plot_save::String="",
                           model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast arima")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="ARIMA Forecast", key="arima_forecast", plot, plot_save)
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        method_sym = Symbol(method)
        safe_method = method_sym == :css_mle ? :mle : method_sym

        model = if isnothing(p)
            crit_sym = Symbol(lowercase(criterion))
            _status("Auto ARIMA forecast: variable=$vname, observations=$(length(y))")
            _status("  Search: p=0:$max_p, d=0:$max_d, q=0:$max_q, criterion=$criterion")
            _status()
            m = auto_arima(y; max_p=max_p, max_q=max_q, max_d=max_d, criterion=crit_sym, method=safe_method)
            label = _model_label(ar_order(m), diff_order(m), ma_order(m))
            _status_styled("Selected model: $label\n"; bold=true)
            _status()
            m
        else
            label = _model_label(p, d, q)
            _status("$label forecast: variable=$vname, horizons=$horizons")
            _status()
            _estimate_arima_model(y, p, d, q; method=method_sym)
        end
    end

    fc = forecast(model, horizons; conf_level=confidence)

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    p_sel = ar_order(model)
    d_sel = diff_order(model)
    q_sel = ma_order(model)
    label = _model_label(p_sel, d_sel, q_sel)

    # C051: MEMs tidy long_table (horizon|variable|value|lower|upper).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="$label Forecast for $vname (h=$horizons, $(Int(round(confidence*100)))% CI)",
                  key="arima_forecast")
    return (; model, result=fc)
end

# ── C065a: SETAR bootstrap forecast ─────────────────────────
# Re-estimate the SETAR (no attached linearity test — unused here), then simulate
# `forecast(::ThresholdModel, h)`. Both MEMs calls are try-wrapped → typed CliError via
# the shared `_nonlinear_error`; every option is guarded up-front → usage/invalid. The
# ThresholdForecast is an AbstractForecastResult, so it renders via the generic long_table.
function _forecast_setar(; data::String="", result=nothing, column::Int=1, p::Int=1, d::String="1",
                          horizons::Int=12, reps::Int=1000, ci_level::Float64=0.95,
                          format::String="table", output::String="", model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast setar")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="SETAR Forecast", key="setar_forecast")
    p >= 1 || throw(CliError("usage/invalid", "forecast setar: --p must be ≥ 1 (got $p)"))
    (ci_level == 0.90 || ci_level == 0.95 || ci_level == 0.99) || throw(CliError("usage/invalid",
        "forecast setar: --ci-level must be exactly 0.90, 0.95, or 0.99 (got $ci_level)"))
    horizons >= 1 || throw(CliError("usage/invalid", "forecast setar: --horizons must be ≥ 1 (got $horizons)"))
    reps >= 1 || throw(CliError("usage/invalid", "forecast setar: --reps must be ≥ 1 (got $reps)"))
    d_arg = _parse_setar_delay(d)
    vname = "y"
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        _status("SETAR forecast (h=$horizons): variable=$vname, obs=$(length(y)), d=$d, ci=$ci_level"); _status()
        model = try
            estimate_setar(y, p, d_arg; reps=reps, ci_level=ci_level, linearity=false, _fwd_seed()...)
        catch e
            throw(_nonlinear_error(e, "SETAR forecast"))
        end
    else
        _status("SETAR forecast (h=$horizons): loaded model, ci=$ci_level"); _status()
    end
    fc = try
        forecast(model, horizons; reps=reps, level=ci_level, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "SETAR forecast"))
    end
    # ThresholdForecast <: AbstractForecastResult → MEMs tidy long_table (horizon|variable|value|lower|upper).
    # No _maybe_plot: MEMs ships no plot_result(::ThresholdForecast) recipe (see the CommandSpec note).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="SETAR Forecast for $vname (h=$horizons, $(Int(round(ci_level*100)))% CI)",
                  key="setar_forecast")
    return (; model, result=fc)
end

# ── C065b: STAR bootstrap forecast ──────────────────────────
# Re-estimate a SELF-EXCITING STAR (s=nothing; `--transition-col` is not offered because
# external-s STARs are not forecastable), then simulate `forecast(::STARModel, h)`. Both MEMs
# calls are try-wrapped → typed CliError via the shared `_nonlinear_error`; every option is
# guarded up-front → usage/invalid. STARForecast <: AbstractForecastResult → generic long_table.
function _forecast_star(; data::String="", result=nothing, column::Int=1, p::Int=1, d::Int=1,
                         type::String="auto", horizons::Int=12, reps::Int=1000,
                         ci_level::Float64=0.95, format::String="table", output::String="",
                         model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast star")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="STAR Forecast", key="star_forecast")
    p >= 1 || throw(CliError("usage/invalid", "forecast star: --p must be ≥ 1 (got $p)"))
    d >= 1 || throw(CliError("usage/invalid", "forecast star: --d must be ≥ 1 (got $d)"))
    horizons >= 1 || throw(CliError("usage/invalid", "forecast star: --horizons must be ≥ 1 (got $horizons)"))
    reps >= 1 || throw(CliError("usage/invalid", "forecast star: --reps must be ≥ 1 (got $reps)"))
    (ci_level == 0.90 || ci_level == 0.95 || ci_level == 0.99) || throw(CliError("usage/invalid",
        "forecast star: --ci-level must be exactly 0.90, 0.95, or 0.99 (got $ci_level)"))
    ttype = Symbol(type)
    ttype in (:lstr1, :lstr2, :estr, :auto) || throw(CliError("usage/invalid",
        "forecast star: --type must be one of lstr1|lstr2|estr|auto (got '$type')"))
    vname = "y"
    if isnothing(model)
        y, vname = load_univariate_series(data, column)
        _status("STAR forecast (h=$horizons) [$type]: variable=$vname, obs=$(length(y)), d=$d, ci=$ci_level"); _status()
        model = try
            estimate_star(y, p; d=d, type=ttype)
        catch e
            throw(_nonlinear_error(e, "STAR forecast"))
        end
    else
        _status("STAR forecast (h=$horizons) [$type]: loaded model, ci=$ci_level"); _status()
    end
    fc = try
        forecast(model, horizons; reps=reps, level=ci_level, _fwd_seed()...)
    catch e
        throw(_nonlinear_error(e, "STAR forecast"))
    end
    # STARForecast <: AbstractForecastResult → MEMs tidy long_table (horizon|variable|value|lower|upper).
    # No _maybe_plot: MEMs ships no plot_result(::STARForecast) recipe (see the CommandSpec note).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="STAR Forecast for $vname (h=$horizons, $(Int(round(ci_level*100)))% CI)",
                  key="star_forecast")
    return (; model, result=fc)
end

# ── Factor Model Forecasts ───────────────────────────────

function _forecast_static(; data::String="", result=nothing, nfactors=nothing, horizons::Int=12,
                            ci_method::String="none", conf_level::Float64=0.95,
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="",
                            model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast static")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="Static Factor Forecast", key="static_factor_forecast", plot, plot_save)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            _status("Selecting number of factors via Bai-Ng information criteria...")
            ic = ic_criteria(X, min(20, size(X, 2)))
            optimal_r = ic.r_IC1
            _status("  IC1 suggests $optimal_r factors")
            optimal_r
        else
            nfactors
        end

        _status("Forecasting with static factor model: $r factors, horizon=$horizons, CI=$ci_method")
        _status()

        fm = estimate_factors(X, r)
    else
        fm = model
        varnames = fm.varnames
    end
    fc = forecast(fm, horizons; ci_method=Symbol(ci_method), conf_level=conf_level)

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    # C051: MEMs tidy long_table (horizon|variable|value|lower|upper).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="Static Factor Forecast (h=$horizons, $(length(varnames)) variables)",
                  key="static_factor_forecast")

    if !isnothing(fc.observables_se)
        _status()
        avg_se = round.(mean(fc.observables_se; dims=1)[1, :]; digits=4)
        _status("Average forecast standard errors:")
        for (vi, vname) in enumerate(varnames)
            _status("  $vname: $(avg_se[vi])")
        end
    end
    return (; model=fm, result=fc)
end

function _forecast_dynamic(; data::String="", result=nothing, nfactors=nothing, horizons::Int=12,
                             factor_lags::Int=1, method::String="twostep",
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="",
                             model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast dynamic")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="Dynamic Factor Forecast", key="dynamic_factor_forecast", plot, plot_save)
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        r = if isnothing(nfactors)
            _status("Selecting number of factors...")
            ic = ic_criteria(X, min(10, size(X, 2)))
            optimal_r = ic.r_IC1
            _status("  Auto-selected $optimal_r factors")
            optimal_r
        else
            nfactors
        end

        _status("Forecasting with dynamic factor model: $r factors, $factor_lags lags, method=$method, horizon=$horizons")
        _status()

        fm = estimate_dynamic_factors(X, r, factor_lags; method=Symbol(method))
    else
        fm = model
        varnames = fm.varnames
    end
    fc = forecast(fm, horizons)

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    # C051: render the FactorForecast's observable forecasts through MEMs' tidy
    # long_table (horizon|variable|value|lower|upper), replacing the hand-rolled
    # loadings reconstruction.
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="Dynamic Factor Forecast (h=$horizons, $(length(varnames)) variables)",
                  key="dynamic_factor_forecast")
    return (; model=fm, result=fc)
end

const _GDFM_FORECAST_METHODS = Dict(
    "ar" => :ar,
    "one-sided" => :one_sided,
    "spectral" => :spectral,
)

function _forecast_gdfm(; data::String="", result=nothing, nfactors=nothing, dynamic_rank=nothing,
                          horizons::Int=12, method::String="ar",
                          spectral::String="lag-window",
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="",
                          model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast gdfm")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="GDFM Forecast", key="gdfm_forecast", plot, plot_save)
    haskey(_GDFM_FORECAST_METHODS, method) || throw(CliError("usage/invalid",
        "forecast gdfm: --method must be ar|one-sided|spectral (got '$method')"))
    haskey(_GDFM_SPECTRAL, spectral) || throw(CliError("usage/invalid",
        "forecast gdfm: --spectral must be lag-window|smoothed-periodogram (got '$spectral')"))
    if isnothing(model)
        X, varnames = load_multivariate_data(data)

        q = if isnothing(dynamic_rank)
            _status("Selecting dynamic rank...")
            ic = ic_criteria_gdfm(X, min(5, size(X, 2)))
            q_opt = ic.q_ratio
            _status("  Auto-selected $q_opt dynamic factors")
            q_opt
        else
            dynamic_rank
        end

        r = if isnothing(nfactors)
            _status("Selecting static rank...")
            ic_static = ic_criteria(X, min(20, size(X, 2)))
            r_opt = ic_static.r_IC1
            _status("  Auto-selected $r_opt static factors")
            r_opt
        else
            nfactors
        end

        _status("Forecasting with GDFM: static rank=$r, dynamic rank=$q, horizon=$horizons, method=$method")
        _status()

        fm = estimate_gdfm(X, q; r=r, spectral=_GDFM_SPECTRAL[spectral])
    else
        fm = model
        varnames = fm.varnames
    end

    # C051: route through MEMs' GDFM forecast (→ FactorForecast) and render its tidy
    # long_table (horizon|variable|value|lower|upper), replacing the hand-rolled AR(1)
    # extrapolation on the common-component factors. W1/#165: --method selects the
    # factor projection (:ar fits AR(1) on two-sided factors; :one_sided/:spectral
    # are the FHLR 2005 one-sided projection).
    fc = forecast(fm, horizons; method=_GDFM_FORECAST_METHODS[method])
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="GDFM Forecast (h=$horizons, $(length(varnames)) variables)",
                  key="gdfm_forecast")

    _status()
    var_shares = common_variance_share(fm)
    _status("Average common variance share: $(round(mean(var_shares); digits=4))")
    return (; model=fm, result=fc)
end

function _forecast_sdfm(; data::String="", result=nothing, factors=nothing, id::String="cholesky",
                         var_lags::Int=1, horizons::Int=12,
                         config::String="", method::String="fglr",
                         spectral::String="lag-window", instrument::String="",
                         q_method::String="hallin-liska",
                         ci::String="none", reps::Int=200,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="",
                         model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast sdfm")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="SDFM Forecast", key="sdfm_forecast", plot, plot_save)
    ci in ("none", "bootstrap") || throw(CliError("usage/invalid",
        "forecast sdfm: --ci must be none|bootstrap (got '$ci')"))
    reps >= 1 || throw(CliError("usage/invalid",
        "forecast sdfm: --reps must be ≥ 1 (got $reps)"))
    if isnothing(model)
        # W1/#165: same estimation surface as `estimate sdfm` (--method here is
        # the structural estimator, fglr|gdfm-var; the forecast itself takes
        # --ci/--reps). H=horizons: forecast(sdfm) projects forward itself.
        sdfm, _, varnames, q = _load_and_estimate_sdfm(data, factors, id, var_lags,
            horizons, config, method, spectral, instrument, q_method)
        _status("Forecasting with SDFM: $q factors, id=$id, method=$method, horizon=$horizons")
        _status()
    else
        sdfm = model
        varnames = sdfm.varnames
    end

    fc = ci == "bootstrap" ?
        forecast(sdfm, horizons; ci_method=:bootstrap, reps=reps) :
        forecast(sdfm, horizons)
    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="SDFM Forecast (h=$horizons, $(length(varnames)) variables)",
                  key="sdfm_forecast")
    return (; model=sdfm, result=fc)
end

# Volatility forecast handlers live in shared.jl (VOL_MODELS / _VOL_FORECAST_HANDLERS).

# ── VECM Forecast ───────────────────────────────────────

function _forecast_vecm(; data::String="", result=nothing, lags::Int=2, rank::String="auto",
                          deterministic::String="constant", horizons::Int=12,
                          ci_method::String="none", replications::Int=500,
                          confidence::Float64=0.95,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="",
                          model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast vecm")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="VECM Forecast", key="vecm_forecast", plot, plot_save)
    if isnothing(model)
        vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    else
        vecm = model
        varnames = vecm.varnames
        p = vecm.p
    end
    r = cointegrating_rank(vecm)

    _status("Computing VECM forecast: rank=$r, horizons=$horizons, CI=$ci_method")
    _status()

    fc = forecast(vecm, horizons; ci_method=Symbol(ci_method), reps=replications, conf_level=confidence)

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    ci_label = ci_method == "none" ? "" : ", $(Int(round(confidence*100)))% CI"
    # C051: MEMs tidy long_table (horizon|variable|value|lower|upper).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="VECM Forecast (rank=$r, h=$horizons$ci_label)", key="vecm_forecast")
    return (; model=vecm, result=fc)
end

# ── FAVAR Forecast ────────────────────────────────────────

function _forecast_favar(; data::String="", result=nothing, factors=nothing, lags::Int=2,
                          key_vars::String="", horizons::Int=12,
                          panel_forecast::Bool=false,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="",
                          model=nothing)
    loaded = _loaded_result(result; data, model, leaf="forecast favar")
    loaded === nothing || return _rerender_long_table(loaded; format, output,
        title="FAVAR Forecast", key="favar_forecast", plot, plot_save)
    if isnothing(model)
        favar, Y, varnames = _load_and_estimate_favar(data, factors, lags, key_vars, "two_step", 5000)
    else
        favar = model
        varnames = favar.varnames
    end

    _status("FAVAR Forecast: horizon=$horizons" * (panel_forecast ? ", panel-wide" : ""))
    _status()

    fc = forecast(favar, horizons)

    if panel_forecast
        fc = favar_panel_forecast(favar, fc)
    end

    _maybe_plot(fc; plot=plot, plot_save=plot_save)

    # C051: MEMs tidy long_table (horizon|variable|value|lower|upper).
    output_result(long_table(fc); format=Symbol(format), output=output,
                  title="FAVAR Forecast (h=$horizons)", key="favar_forecast")
    return (; model=favar, result=fc)
end

# ── forecast evaluate: forecast evaluation & combination (C072, M5c) ──
# Wraps the MEMs `fceval/` module (model-agnostic, plain vectors). The result
# types (ForecastEvaluation/DMTestResult/…) are NOT Tables.jl-registered upstream,
# so tables are hand-built (a documented C051 exception, like the io and SUR/3SLS
# families). Uniform input: data + --actual, then either --forecasts columns or
# --result stems; the handler forms the errors / f_adj / forecast matrix.
#
# Convention notes (mirrored in docs): DM consumes forecast ERRORS e=actual-fc;
# Clark-West needs f_adj = f_small - f_big (the forecast difference; the library
# squares it internally).

function _forecast_wrong_result(obj)
    throw(CliError("data/wrong-result",
        "$(typeof(obj)) is not a forecast result";
        hint="pass VARForecast/BVARForecast/ARIMAForecast/… from forecast * --save-result"))
end

function _forecast_varnames(obj)::Vector{String}
    for f in (:varnames, :variables, :names)
        hasproperty(obj, f) || continue
        vn = getproperty(obj, f)
        vn isa AbstractVector || continue
        return String[string(x) for x in vn]
    end
    return String[]
end

"""H×n point forecasts → one series. Match `--actual` against varnames, else column 1."""
function _forecast_series(raw, obj, actual::String)::Vector{Float64}
    A = try
        Float64.(raw)
    catch
        _forecast_wrong_result(obj)
    end
    if A isa Number
        return Float64[A]
    elseif A isa AbstractVector || (A isa AbstractMatrix && size(A, 2) == 1)
        return vec(Float64.(A))
    elseif A isa AbstractMatrix
        j = 1
        names = _forecast_varnames(obj)
        if !isempty(actual) && !isempty(names)
            idx = findfirst(==(actual), names)
            idx !== nothing && (j = Int(idx))
        end
        (1 <= j <= size(A, 2)) || _forecast_wrong_result(obj)
        return Float64.(A[:, j])
    end
    _forecast_wrong_result(obj)
end

function _forecast_raw(obj)
    if hasproperty(obj, :forecast)
        try
            return obj.forecast
        catch
            _forecast_wrong_result(obj)
        end
    elseif hasproperty(obj, :levels)
        return obj.levels
    elseif hasproperty(obj, :differences)
        return obj.differences
    elseif hasproperty(obj, :observables)
        return obj.observables
    else
        try
            return point_forecast(obj)
        catch
            _forecast_wrong_result(obj)
        end
    end
end

function _forecast_points(obj; actual::String="")::Vector{Float64}
    raw = try
        _forecast_raw(obj)
    catch e
        e isa CliError && rethrow()
        _forecast_wrong_result(obj)
    end
    return _forecast_series(raw, obj, actual)
end

function _fceval_result_name(stem::String)::String
    b = basename(stem)
    for suf in (".jld2", ".fmod")
        endswith(lowercase(b), suf) && return b[1:end-length(suf)]
    end
    return b
end

"""Resolve actual + forecast columns or --result stems; return (y, fnames, fcols)."""
function _fceval_load(data::String, actual::String, forecasts::String; leaf::String,
                      result::String="")
    isempty(actual) && throw(CliError("usage/missing-actual",
        "forecast evaluate $leaf requires --actual <column> (the realized-values column)"))
    result_stems = String[String(strip(s)) for s in split(result, ",") if !isempty(strip(s))]
    fnames = String[String(strip(s)) for s in split(forecasts, ",") if !isempty(strip(s))]
    if !isempty(result_stems) && !isempty(fnames)
        throw(CliError("usage/invalid",
            "forecast evaluate $leaf: --result cannot be combined with --forecasts";
            hint="pass --result stems or --forecasts columns, not both"))
    end
    if isempty(result_stems)
        isempty(strip(forecasts)) && throw(CliError("usage/missing-forecasts",
            "forecast evaluate $leaf requires --forecasts <col1,col2,...>"))
        isempty(fnames) && throw(CliError("usage/missing-forecasts",
            "forecast evaluate $leaf requires at least one --forecasts column"))
    else
        isempty(data) && throw(CliError("usage/missing-arg",
            "forecast evaluate $leaf requires <data> (realized values) even with --result";
            hint="pass a CSV or data handle plus --actual <column>"))
    end
    df = load_data(data)
    numcols = variable_names(df)
    actual in numcols || throw(CliError("data/bad-column",
        "actual column '$actual' not found in numeric columns: $(join(numcols, ", "))"))
    # `variable_names` admits Union{Number,Missing} columns, so guard for missing
    # values → typed data error (a blank cell would otherwise MethodError → exit 1).
    _col(c) = any(ismissing, df[!, c]) ?
        throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`) before forecast evaluation")) :
        Vector{Float64}(df[!, c])
    y = _col(actual)
    if !isempty(result_stems)
        fcols = Vector{Float64}[]
        names = String[]
        for stem in result_stems
            resolved = resolve_stem(stem; slot=:result)
            if !startswith(resolved, ":") && !startswith(resolved, "model://")
                _validate_input_path(resolved)
            end
            pts = _forecast_points(load_model_dispatch(resolved); actual)
            nm = _fceval_result_name(stem)
            length(pts) == length(y) || throw(CliError("data/shape",
                "forecast evaluate $leaf: result '$nm' has $(length(pts)) points, actual has $(length(y))";
                hint="the forecast result length must match the realized-values column"))
            push!(names, nm)
            push!(fcols, pts)
        end
        return (y, names, fcols)
    end
    for c in fnames
        c in numcols || throw(CliError("data/bad-column",
            "forecast column '$c' not found in numeric columns: $(join(numcols, ", "))"))
    end
    fcols = Vector{Float64}[_col(c) for c in fnames]
    return (y, fnames, fcols)
end

# Enforce the per-leaf forecast-count arity → usage error (never a downstream crash).
function _fceval_arity(fnames::Vector{String}, leaf::String, want::String, ok::Bool)
    ok || throw(CliError("usage/arity",
        "forecast evaluate $leaf needs $want --forecasts column(s); got $(length(fnames))"))
    return nothing
end

# Map an fceval failure to a typed CliError (never an uncaught exit-1 — io-family lesson).
function _fceval_error(e, what::String)
    e isa CliError && return e
    (e isa ArgumentError || e isa DimensionMismatch) && return CliError("data/fceval",
        sprint(showerror, e);
        hint="check --actual/--forecasts refer to equal-length numeric columns with >=2 observations")
    return CliError("model/error", "$what failed: $(sprint(showerror, e))")
end

function _forecast_eval_metrics(; data::String, actual::String="", forecasts::String="",
                                 result::String="", seasonal_period::Int=1, output::String="", format::String="table",
                                 plot::Bool=false, plot_save::String="")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="metrics", result)
    _status("Forecast evaluation: $(length(fnames)) forecast(s), n=$(length(y)), seasonal_period=$seasonal_period")
    _status()
    Fmat = reduce(hcat, fcols)
    ev = try
        forecast_evaluate(y, Fmat; seasonal_period=seasonal_period, model_names=fnames)
    catch e
        throw(_fceval_error(e, "forecast evaluation"))
    end
    # C051 exception: hand-built WIDE accuracy table model | ME | MAE | ... | U2.
    acc = DataFrame(model = ev.models)
    for (k, mname) in enumerate(ev.metrics)
        acc[!, mname] = round.(ev.values[:, k]; digits=6)
    end
    output_result(acc; format=Symbol(format), output=output, title="Forecast Accuracy Metrics")
    # Theil MSE decomposition (proportions sum to 1): model | bias | variance | covariance.
    dec = DataFrame(model = ev.models,
                    bias = round.(ev.decomp[:, 1]; digits=6),
                    variance = round.(ev.decomp[:, 2]; digits=6),
                    covariance = round.(ev.decomp[:, 3]; digits=6))
    output_result(dec; format=Symbol(format), output=output, title="Theil MSE Decomposition")
    # #95: ForecastEvaluation has a real recipe (bar chart of the chosen metric). The
    # other five `evaluate` leaves return DMTestResult / MincerZarnowitzResult /
    # ForecastEncompassingResult / ForecastCombination, none of which have one — so they
    # correctly stay flagless.
    _maybe_plot(ev; plot=plot, plot_save=plot_save)
    return ev
end

function _forecast_eval_dm(; data::String, actual::String="", forecasts::String="",
                            result::String="", loss::String="se", horizon::Int=1, alternative::String="two-sided",
                            no_hln::Bool=false, output::String="", format::String="table")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="dm", result)
    _fceval_arity(fnames, "dm", "exactly 2", length(fnames) == 2)
    e1 = y .- fcols[1]; e2 = y .- fcols[2]
    alt = Symbol(replace(alternative, "-" => "_"))
    _status("Diebold-Mariano: $(fnames[1]) vs $(fnames[2]), loss=$loss, h=$horizon, HLN=$(!no_hln), alt=$alternative")
    _status()
    r = try
        diebold_mariano(e1, e2; h=horizon, loss=Symbol(loss), hln=!no_hln, alternative=alt)
    catch e
        throw(_fceval_error(e, "Diebold-Mariano test"))
    end
    output_kv(Pair{String,Any}[
        "test"           => "Diebold-Mariano",
        "model_1"        => fnames[1],
        "model_2"        => fnames[2],
        "loss"           => string(r.loss),
        "statistic"      => round(r.statistic; digits=4),
        "p_value"        => round(r.pvalue; digits=4),
        "mean_loss_diff" => round(r.dbar; digits=6),
        "lrvar"          => round(r.lrvar; digits=6),
        "horizon"        => r.h,
        "hln"            => r.hln,
        "alternative"    => string(r.alternative),
        "n"              => r.T_obs,
    ]; format=format, output=output, title="Diebold-Mariano Test")
    return r
end

function _forecast_eval_clark_west(; data::String, actual::String="", forecasts::String="",
                                    result::String="", horizon::Int=1, alternative::String="greater",
                                    output::String="", format::String="table")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="clark-west", result)
    _fceval_arity(fnames, "clark-west", "exactly 2 (small then big)", length(fnames) == 2)
    f_small = fcols[1]; f_big = fcols[2]
    e_small = y .- f_small; e_big = y .- f_big; f_adj = f_small .- f_big
    alt = Symbol(replace(alternative, "-" => "_"))
    _status("Clark-West (nested): small=$(fnames[1]) big=$(fnames[2]), h=$horizon, alt=$alternative")
    _status()
    r = try
        clark_west(e_small, e_big, f_adj; h=horizon, alternative=alt)
    catch e
        throw(_fceval_error(e, "Clark-West test"))
    end
    output_kv(Pair{String,Any}[
        "test"          => "Clark-West",
        "model_small"   => fnames[1],
        "model_big"     => fnames[2],
        "statistic"     => round(r.statistic; digits=4),
        "p_value"       => round(r.pvalue; digits=4),
        "mean_adj_diff" => round(r.fbar; digits=6),
        "lrvar"         => round(r.lrvar; digits=6),
        "horizon"       => r.h,
        "alternative"   => string(r.alternative),
        "n"             => r.T_obs,
    ]; format=format, output=output, title="Clark-West Test")
    return r
end

function _forecast_eval_mincer_zarnowitz(; data::String, actual::String="", forecasts::String="",
                                          result::String="", lags::Int=0, kernel::String="bartlett",
                                          output::String="", format::String="table")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="mincer-zarnowitz", result)
    _fceval_arity(fnames, "mincer-zarnowitz", "exactly 1", length(fnames) == 1)
    _status("Mincer-Zarnowitz efficiency: forecast=$(fnames[1]), lags=$lags, kernel=$kernel")
    _status()
    r = try
        mincer_zarnowitz(y, fcols[1]; lags=lags, kernel=Symbol(kernel))
    catch e
        throw(_fceval_error(e, "Mincer-Zarnowitz test"))
    end
    output_kv(Pair{String,Any}[
        "test"         => "Mincer-Zarnowitz",
        "forecast"     => fnames[1],
        "a"            => round(r.a; digits=6),
        "b"            => round(r.b; digits=6),
        "se_a"         => round(r.se[1]; digits=6),
        "se_b"         => round(r.se[2]; digits=6),
        "wald_chi2"    => round(r.wald; digits=4),
        "p_value_wald" => round(r.pvalue_wald; digits=4),
        "fstat"        => round(r.fstat; digits=4),
        "p_value_f"    => round(r.pvalue_f; digits=4),
        "hac_lags"     => r.lags,
        "kernel"       => string(r.kernel),
        "n"            => r.T_obs,
    ]; format=format, output=output, title="Mincer-Zarnowitz Efficiency Test")
    return r
end

function _forecast_eval_encompassing(; data::String, actual::String="", forecasts::String="",
                                      result::String="", lags::Int=0, kernel::String="bartlett",
                                      output::String="", format::String="table")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="encompassing", result)
    _fceval_arity(fnames, "encompassing", "exactly 2", length(fnames) == 2)
    _status("Forecast encompassing: fc1=$(fnames[1]) fc2=$(fnames[2]), lags=$lags, kernel=$kernel")
    _status()
    r = try
        forecast_encompassing(y, fcols[1], fcols[2]; lags=lags, kernel=Symbol(kernel))
    catch e
        throw(_fceval_error(e, "forecast encompassing test"))
    end
    output_kv(Pair{String,Any}[
        "test"     => "Forecast-Encompassing",
        "model_1"  => fnames[1],
        "model_2"  => fnames[2],
        "b1"       => round(r.b1; digits=6),
        "b2"       => round(r.b2; digits=6),
        "se_b2"    => round(r.se_b2; digits=6),
        "t_stat"   => round(r.tstat; digits=4),
        "p_value"  => round(r.pvalue; digits=4),
        "hac_lags" => r.lags,
        "kernel"   => string(r.kernel),
        "n"        => r.T_obs,
    ]; format=format, output=output, title="Forecast Encompassing Test")
    return r
end

function _forecast_eval_combine(; data::String, actual::String="", forecasts::String="",
                                 result::String="", method::String="equal", emit_series::Bool=false,
                                 output::String="", format::String="table")
    y, fnames, fcols = _fceval_load(data, actual, forecasts; leaf="combine", result)
    _fceval_arity(fnames, "combine", "at least 2", length(fnames) >= 2)
    F = reduce(hcat, fcols)
    meth = Symbol(replace(method, "-" => "_"))
    _status("Forecast combination: $(length(fnames)) forecasts, method=$method")
    _status()
    r = try
        combine_forecasts(F, y; method=meth, model_names=fnames)
    catch e
        throw(_fceval_error(e, "forecast combination"))
    end
    # C051 exception: hand-built weights table model | weight | mse (weights sum to 1).
    wtab = DataFrame(model = r.models,
                     weight = round.(r.weights; digits=6),
                     mse = round.(r.mse; digits=6))
    output_result(wtab; format=Symbol(format), output=output, title="Forecast Combination Weights")
    if emit_series
        stab = DataFrame(index = collect(1:length(r.combined)),
                         combined = round.(r.combined; digits=6))
        output_result(stab; format=Symbol(format), output=output, title="Combined Forecast Series")
    end
    return r
end

# ── W8/#110: Waggoner-Zha conditional (scenario) forecasts ───────────────────
#
# `conditional_forecast` dispatches on VARModel and BVARPosterior, so --method picks
# which to fit. `plot_result(::ConditionalForecast)` EXISTS (plotting/forecast.jl:262),
# so --plot is genuinely backed here.

"""
    _load_forecast_conditions(path, varnames, horizon) -> Vector{ForecastCondition}

Read a long-format conditions CSV: `variable,period,value[,sd]`.

`variable` may be a name or a 1-based index; `period` is the forecast horizon the condition
applies to (1 = first forecast period). `sd` is optional and defaults to 0, i.e. a HARD
condition — the path is pinned exactly. A positive `sd` makes it soft.

Missing cells are rejected BEFORE any numeric conversion: a blank `value` would otherwise
reach `Float64(missing)` as an untyped MethodError (exit 1). Same for the whole file being
empty, a duplicate (variable, period), or a period beyond the forecast horizon — all of
which are user errors and none of which upstream reports in CLI terms.
"""
function _load_forecast_conditions(path::String, varnames::Vector{String}, horizon::Int)
    _validate_input_path(path)
    isfile(path) || throw(CliError("data/file-not-found",
        "conditions file not found: $path"))
    df = load_data(path)
    cols = lowercase.(String.(names(df)))
    for req in ("variable", "period", "value")
        req in cols || throw(CliError("data/missing-column",
            "conditions CSV must have columns variable, period, value (optional sd); " *
            "got $(join(names(df), ", "))"))
    end
    ci = Dict(c => i for (i, c) in enumerate(cols))
    has_sd = "sd" in cols
    nrow(df) == 0 && throw(CliError("data/empty",
        "conditions file has no rows: $path"))

    conds = ForecastCondition{Float64}[]
    seen = Set{Tuple{Int,Int}}()
    for r in 1:nrow(df)
        vraw = df[r, ci["variable"]]
        praw = df[r, ci["period"]]
        xraw = df[r, ci["value"]]
        (ismissing(vraw) || ismissing(praw) || ismissing(xraw)) &&
            throw(CliError("data/missing",
                "conditions row $r has a blank variable, period or value"))

        # A name or a 1-based index; resolve to an index so duplicates can be detected
        # whichever form the file used.
        vidx = if vraw isa Real
            Int(vraw)
        else
            nm = strip(String(vraw))
            k = findfirst(==(nm), varnames)
            if k === nothing
                pi_ = tryparse(Int, nm)
                pi_ === nothing && throw(CliError("data/invalid",
                    "conditions row $r: unknown variable '$nm'";
                    hint="model variables: $(join(varnames, ", "))"))
                pi_
            else
                k
            end
        end
        1 <= vidx <= length(varnames) || throw(CliError("data/invalid",
            "conditions row $r: variable index $vidx is outside 1:$(length(varnames))"))

        per = praw isa Real ? Int(praw) : something(tryparse(Int, strip(String(praw))), 0)
        1 <= per <= horizon || throw(CliError("data/invalid",
            "conditions row $r: period $per is outside the forecast horizon 1:$horizon"))
        (vidx, per) in seen && throw(CliError("data/invalid",
            "conditions row $r: duplicate condition for $(varnames[vidx]) at period $per"))
        push!(seen, (vidx, per))

        val = xraw isa Real ? Float64(xraw) :
              something(tryparse(Float64, strip(String(xraw))), NaN)
        isfinite(val) || throw(CliError("data/invalid",
            "conditions row $r: value is not a finite number"))
        sd = 0.0
        if has_sd
            sraw = df[r, ci["sd"]]
            if !ismissing(sraw)
                sd = sraw isa Real ? Float64(sraw) :
                     something(tryparse(Float64, strip(String(sraw))), NaN)
                (isfinite(sd) && sd >= 0) || throw(CliError("data/invalid",
                    "conditions row $r: sd must be a non-negative number"))
            end
        end
        # Build with the INDEX, not the name. `_load_and_estimate_var` does not forward
        # varnames to `estimate_var`, so the fitted model carries y1..yn while the CSV
        # header says something else -- resolving the user's name here and passing the
        # index makes the leaf work with the names they actually typed. (The wider
        # cosmetic issue, that the VAR family renders y1/yn everywhere, is its own fix.)
        push!(conds, ForecastCondition{Float64}(vidx, per, val, sd))
    end
    return conds
end

# NOTE: the option is `--conditions-file`, not `--conditions`. `dispatch.jl` matches
# `"--conditions" in args` across the WHOLE argv to print the GPL notice, so a leaf option
# of that name is swallowed before dispatch ever runs. Recorded as an engine flaw for the
# C055 freeze; renaming here is the zero-risk fix.
function _forecast_scenario(; data::String="", result=nothing, conditions_file::String="", lags=nothing,
                             horizons::Int=12, method::String="var",
                             draws::Int=2000, sampler::String="direct",
                             replications::Int=1000, confidence::Float64=0.95,
                             config::String="",
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="",
                             model=nothing)
    loaded = _loaded_result(result; data, model, lags, check_lags=true, leaf="forecast scenario")
    if loaded !== nothing
        H = loaded.horizon
        n = length(loaded.varnames)
        df = DataFrame(
            horizon       = repeat(1:H, outer=n),
            variable      = repeat(loaded.varnames; inner=H),
            value         = vec(Float64.(loaded.forecast)),
            lower         = vec(Float64.(loaded.ci_lower)),
            upper         = vec(Float64.(loaded.ci_upper)),
            unconditional = vec(Float64.(loaded.unconditional)),
        )
        output_result(df; format=Symbol(format), output=output,
                      title="Conditional Forecast", key="conditional_forecast")
        _maybe_plot(loaded; plot=plot, plot_save=plot_save)
        return loaded
    end
    horizons >= 1 || throw(CliError("usage/invalid", "--horizons must be ≥ 1 (got $horizons)"))
    replications >= 1 || throw(CliError("usage/invalid",
        "--replications must be ≥ 1 (got $replications)"))
    0 < confidence < 1 || throw(CliError("usage/invalid",
        "--confidence must be in (0, 1) (got $confidence)"))
    meth = lowercase(strip(method))
    meth in ("var", "bvar") || throw(CliError("usage/invalid-option",
        "invalid --method '$method'; must be var or bvar"))
    isempty(strip(conditions_file)) && throw(CliError("usage/missing",
        "forecast scenario: --conditions-file <csv> is required";
        hint="a long-format CSV with columns variable,period,value (optional sd)"))

    obj, varnames = if !isnothing(model)
        model, (hasproperty(model, :varnames) ? model.varnames : String[])
    elseif meth == "bvar"
        post, _, vn, _, _ = _load_and_estimate_bvar(data, lags === nothing ? 4 : lags,
                                                    config, draws, sampler)
        post, vn
    else
        m, _, vn, _ = _load_and_estimate_var(data, lags)
        m, vn
    end

    conds = _load_forecast_conditions(conditions_file, varnames, horizons)
    _status("Conditional forecast ($meth): $(length(conds)) condition(s) over $horizons periods")
    for c in conds
        nm = c.variable isa Integer && 1 <= c.variable <= length(varnames) ?
             varnames[c.variable] : string(c.variable)
        _status("  $nm @ h=$(c.horizon) → $(c.value)" *
                (c.sd > 0 ? " (sd $(c.sd))" : " (hard)"))
    end

    fc = try
        conditional_forecast(obj, conds, horizons; reps=replications, conf_level=confidence,
                             _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "forecast scenario"))
    end

    _maybe_plot(fc; plot=plot, plot_save=plot_save)
    _status_report(() -> report(fc))

    # Tidy long form, with the UNCONDITIONAL path alongside: the scenario is only
    # interpretable against the baseline it departs from, and having to run a second
    # command to get it invites comparing paths from different draws.
    H = fc.horizon
    n = length(fc.varnames)
    df = DataFrame(
        horizon       = repeat(1:H, outer=n),
        variable      = repeat(fc.varnames; inner=H),
        value         = vec(Float64.(fc.forecast)),
        lower         = vec(Float64.(fc.ci_lower)),
        upper         = vec(Float64.(fc.ci_upper)),
        unconditional = vec(Float64.(fc.unconditional)),
    )
    output_result(df; format=Symbol(format), output=output,
                  title="Conditional Forecast ($(round(Int, 100*fc.conf_level))% interval, " *
                        "$(fc.identification) identification)", key="conditional_forecast")

    # The implied structural shocks are what actually delivers the scenario; a scenario
    # requiring implausibly large shocks is not a credible one.
    sh = fc.shocks
    output_result(DataFrame(
        horizon = repeat(1:size(sh, 1), outer=size(sh, 2)),
        shock   = repeat(fc.varnames; inner=size(sh, 1)),
        value   = vec(Float64.(sh)),
    ); format=Symbol(format), output=_per_var_output_path(output, "shocks"),
       title="Implied Structural Shocks")

    output_kv(Pair{String,Any}[
        "method"         => meth,
        "horizon"        => H,
        "conditions"     => length(fc.conditions),
        "conf_level"     => fc.conf_level,
        "identification" => String(fc.identification),
        "n_draws"        => fc.n_draws,
    ]; format=format, title="Scenario Settings")
    return (; model=obj, result=fc)
end
