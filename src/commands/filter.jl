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

# Filter commands: hp, hamilton, bn, bk, bhp — registry migrated (C023-1)

function filter_specs()::Vector{CommandSpec}
    data_arg = [ArgSpec(name="data", description="Path to CSV data file")]
    plot_opts = [OUTPUT_OPTIONS; PLOT_OPTIONS]
    plot_flags = copy(PLOT_FLAGS)
    col = OptionSpec(name="columns", short="c", type=String, default="",
                     description="Column indices, comma-separated (default: all numeric)")

    return [
        CommandSpec(
            path=["filter", "hp"],
            summary="Hodrick-Prescott filter",
            args=data_arg,
            options=[
                OptionSpec(name="lambda", short="l", type=Float64, default=1600.0,
                           description="Smoothing parameter (6.25 annual, 1600 quarterly, 129600 monthly)"),
                col, plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:hp_filter,
                              description="Per-variable HP trend and cycle by time index")],
            category="filter",
            handler=_filter_hp,
        ),
        CommandSpec(
            path=["filter", "hamilton"],
            summary="Hamilton (2018) regression filter",
            args=data_arg,
            options=[
                OptionSpec(name="horizon", type=Int, default=8, description="Forecast horizon"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Number of lags in regression"),
                col, plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:hamilton_filter,
                              description="Per-variable Hamilton trend and cycle over the valid range")],
            category="filter",
            handler=_filter_hamilton,
        ),
        CommandSpec(
            path=["filter", "bn"],
            summary="Beveridge-Nelson decomposition",
            args=data_arg,
            options=[
                OptionSpec(name="method", type=String, default="arima",
                           choices=["arima", "statespace"], description="arima|statespace"),
                OptionSpec(name="p", type=Int, default=nothing, description="AR order (default: auto)"),
                OptionSpec(name="q", type=Int, default=nothing, description="MA order (default: auto)"),
                col, plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:beveridge_nelson_decomposition,
                              description="Per-variable permanent (trend) and transitory (cycle) components")],
            category="filter",
            handler=_filter_bn,
        ),
        CommandSpec(
            path=["filter", "bk"],
            summary="Baxter-King band-pass filter",
            args=data_arg,
            options=[
                OptionSpec(name="pl", type=Int, default=6, description="Minimum period of oscillation"),
                OptionSpec(name="pu", type=Int, default=32, description="Maximum period of oscillation"),
                OptionSpec(name="K", type=Int, default=12, description="Truncation length (leads/lags)"),
                col, plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:baxter_king_filter,
                              description="Per-variable band-pass trend and cycle over the untrimmed range")],
            category="filter",
            handler=_filter_bk,
        ),
        CommandSpec(
            path=["filter", "bhp"],
            summary="Boosted HP filter (Phillips & Shi 2021)",
            args=data_arg,
            options=[
                OptionSpec(name="lambda", short="l", type=Float64, default=1600.0, description="Smoothing parameter"),
                OptionSpec(name="stopping", type=String, default="BIC",
                           choices=["BIC", "ADF", "fixed"], description="BIC|ADF|fixed"),
                OptionSpec(name="max-iter", type=Int, default=100, description="Maximum boosting iterations"),
                OptionSpec(name="sig-p", type=Float64, default=0.05, description="ADF significance level"),
                col, plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:boosted_hp_filter,
                              description="Per-variable boosted-HP trend and cycle by time index")],
            category="filter",
            handler=_filter_bhp,
        ),
        # C042 — X-13ARIMA-SEATS (pure-Julia MEMs port; no external binary required)
        CommandSpec(
            path=["filter", "x13"],
            summary="X-13ARIMA-SEATS seasonal adjustment (X-11 / SEATS)",
            args=data_arg,
            options=[
                OptionSpec(name="frequency", type=Int, default=12,
                           description="Seasonal period: 4 (quarterly) or 12 (monthly)"),
                OptionSpec(name="method", type=String, default="seats",
                           choices=["seats", "x11"],
                           description="Preferred decomposition: seats|x11"),
                OptionSpec(name="transform", type=String, default="auto",
                           choices=["auto", "log", "none"],
                           description="Pre-transformation"),
                OptionSpec(name="critical-value", type=Float64, default=0.0,
                           description="Outlier critical value (0 = automatic)"),
                OptionSpec(name="outliers", type=String, default="true",
                           choices=["true", "false"],
                           description="Detect AO/LS/TC outliers (default true)"),
                col, plot_opts...,
            ],
            flags=[
                FlagSpec(name="trading-day", description="Include trading-day regressors"),
                FlagSpec(name="easter", description="Include Easter effect regressor"),
                plot_flags...,
            ],
            tables=[
                TableSpec(name=:x_13_seasonally_adjusted,
                          description="Seasonally adjusted series, one column per variable"),
                TableSpec(name=:x_13_trend,
                          description="Trend-cycle component, one column per variable"),
                TableSpec(name=:x_13_seasonal_factors,
                          description="Seasonal component, one column per variable"),
                TableSpec(name=:x_13_irregular,
                          description="Irregular component, one column per variable"),
                TableSpec(name=:x_13_diagnostics,
                          description="Per-variable ARIMA order, AIC, sigma2, outlier count and T"),
            ],
            category="filter",
            handler=_filter_x13,
        ),
    ]
end

const _FILTER_SLOT_TYPES = Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}}(
    ["filter", "hp"]       => (Symbol[], [:HPFilterResult]),
    ["filter", "hamilton"] => (Symbol[], [:HamiltonFilterResult]),
    ["filter", "bn"]       => (Symbol[], [:BeveridgeNelsonResult]),
    ["filter", "bk"]       => (Symbol[], [:BaxterKingResult]),
    ["filter", "bhp"]      => (Symbol[], [:BoostedHPResult]),
    ["filter", "x13"]      => (Symbol[], [:X13FilterResult]),
)

function register_filter_commands!()
    specs = CommandSpec[]
    for s in _tag_slot_types(filter_specs(), _FILTER_SLOT_TYPES)
        leaf = join(s.path, " ")
        key = isempty(s.tables) ? "" : string(s.tables[1].name)
        h = wrap_legacy(_with_result(s.handler, leaf; key=key))
        push!(specs, _copy_spec(s; handler=h))
    end
    specs = with_result_handles(with_default_csv_kinds(with_data_kinds(specs, [:timeseries, :csv])))
    register!(specs)
    return build_node("filter", specs;
        description="Time series filtering and trend-cycle decomposition")
end

# ── Column Parsing ───────────────────────────────────────

"""
    _parse_columns(columns_str, n) → Vector{Int}

Parse comma-separated column indices or return all (1:n).
"""
function _parse_columns(columns_str::String, n::Int)
    isempty(columns_str) && return collect(1:n)
    indices = Int[]
    for s in split(columns_str, ",")
        idx = parse(Int, strip(s))
        (idx < 1 || idx > n) && error("column index $idx out of range (data has $n numeric columns)")
        push!(indices, idx)
    end
    return indices
end

# ── Variance Ratio Summary ──────────────────────────────

function _print_variance_ratios(varnames::Vector{String}, cycles::Vector{Vector{Float64}},
                                 originals::Vector{Vector{Float64}})
    _status()
    _status_styled("Cycle Variance Ratios:\n"; bold=true)
    for (i, vname) in enumerate(varnames)
        total_var = var(originals[i])
        cycle_var = var(cycles[i])
        ratio = total_var > 0 ? cycle_var / total_var : 0.0
        _status("  $vname: $(round(ratio; digits=4))")
    end
end

# ── HP Filter ────────────────────────────────────────────

function _filter_hp(; data::String, lambda::Float64=1600.0, columns::String="",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    _status("HP Filter (λ=$(lambda)): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

    result_df = DataFrame()
    result_df.t = 1:T_obs
    cycles = Vector{Float64}[]
    originals = Vector{Float64}[]
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = hp_filter(y; lambda=lambda)
        last_result = res
        _maybe_plot(res; plot=plot, plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname))
        t = trend(res)
        c = cycle(res)
        result_df[!, "$(vname)_trend"] = round.(t; digits=6)
        result_df[!, "$(vname)_cycle"] = round.(c; digits=6)
        push!(cycles, c)
        push!(originals, y)
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="HP Filter (λ=$(lambda))", key="hp_filter")
    _print_variance_ratios(sel_names, cycles, originals)
    return last_result
end

# ── Hamilton Filter ──────────────────────────────────────

function _filter_hamilton(; data::String, horizon::Int=8, lags::Int=4, columns::String="",
                           output::String="", format::String="table",
                           plot::Bool=false, plot_save::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    _status("Hamilton Filter (h=$horizon, p=$lags): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

    result_df = DataFrame()
    cycles = Vector{Float64}[]
    originals = Vector{Float64}[]
    first_valid = nothing
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = hamilton_filter(y; h=horizon, p=lags)
        last_result = res
        _maybe_plot(res; plot=plot, plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname), original=y)
        t = trend(res)
        c = cycle(res)
        vr = res.valid_range

        if isnothing(first_valid)
            first_valid = vr
            result_df.t = collect(vr)
        end

        # trend()/cycle() may be full-length or valid-range-only
        tv = length(t) == T_obs ? t[vr] : t
        cv = length(c) == T_obs ? c[vr] : c
        result_df[!, "$(vname)_trend"] = round.(tv; digits=6)
        result_df[!, "$(vname)_cycle"] = round.(cv; digits=6)
        push!(cycles, cv)
        push!(originals, y[vr])
    end

    lost = isnothing(first_valid) ? 0 : first(first_valid) - 1
    if lost > 0
        _status_styled("Note: $lost initial observations lost due to filter requirements\n"; color=:yellow)
        _status()
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Hamilton Filter (h=$horizon, p=$lags)", key="hamilton_filter")
    _print_variance_ratios(sel_names, cycles, originals)
    return last_result
end

# ── Beveridge-Nelson Decomposition ───────────────────────

function _filter_bn(; data::String, method::String="arima", p=nothing, q=nothing,
                     columns::String="",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    p_label = isnothing(p) ? "auto" : string(p)
    q_label = isnothing(q) ? "auto" : string(q)
    _status("Beveridge-Nelson Decomposition (method=$method, p=$p_label, q=$q_label): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

    result_df = DataFrame()
    result_df.t = 1:T_obs
    cycles = Vector{Float64}[]
    originals = Vector{Float64}[]
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = if method == "statespace"
            beveridge_nelson(y; method=:statespace)
        else
            kwargs = Dict{Symbol,Any}(:method => :arima)
            isnothing(p) || (kwargs[:p] = p)
            isnothing(q) || (kwargs[:q] = q)
            beveridge_nelson(y; kwargs...)
        end
        last_result = res
        _maybe_plot(res; plot=plot, plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname))
        t = trend(res)
        c = cycle(res)
        result_df[!, "$(vname)_trend"] = round.(t; digits=6)
        result_df[!, "$(vname)_cycle"] = round.(c; digits=6)
        push!(cycles, c)
        push!(originals, y)
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Beveridge-Nelson Decomposition")
    _print_variance_ratios(sel_names, cycles, originals)
    return last_result
end

# ── Baxter-King Band-Pass Filter ─────────────────────────

function _filter_bk(; data::String, pl::Int=6, pu::Int=32, K::Int=12, columns::String="",
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    _status("Baxter-King Filter (pl=$pl, pu=$pu, K=$K): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

    result_df = DataFrame()
    cycles = Vector{Float64}[]
    originals = Vector{Float64}[]
    first_valid = nothing
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = baxter_king(y; pl=pl, pu=pu, K=K)
        last_result = res
        _maybe_plot(res; plot=plot, plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname), original=y)
        t = trend(res)
        c = cycle(res)
        vr = res.valid_range

        if isnothing(first_valid)
            first_valid = vr
            result_df.t = collect(vr)
        end

        # trend()/cycle() may be full-length or valid-range-only
        tv = length(t) == T_obs ? t[vr] : t
        cv = length(c) == T_obs ? c[vr] : c
        result_df[!, "$(vname)_trend"] = round.(tv; digits=6)
        result_df[!, "$(vname)_cycle"] = round.(cv; digits=6)
        push!(cycles, cv)
        push!(originals, y[vr])
    end

    lost = isnothing(first_valid) ? 0 : first(first_valid) - 1
    total_lost = isnothing(first_valid) ? 0 : T_obs - length(first_valid)
    if total_lost > 0
        _status_styled("Note: $total_lost observations lost ($K leads/lags trimmed from each end)\n"; color=:yellow)
        _status()
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Baxter-King Filter (pl=$pl, pu=$pu, K=$K)", key="baxter_king_filter")
    _print_variance_ratios(sel_names, cycles, originals)
    return last_result
end

# ── Boosted HP Filter ────────────────────────────────────

function _filter_bhp(; data::String, lambda::Float64=1600.0, stopping::String="BIC",
                      max_iter::Int=100, sig_p::Float64=0.05, columns::String="",
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    stop_sym = Symbol(stopping)
    _status("Boosted HP Filter (λ=$(lambda), stopping=$stopping): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

    result_df = DataFrame()
    result_df.t = 1:T_obs
    cycles = Vector{Float64}[]
    originals = Vector{Float64}[]
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = boosted_hp(y; lambda=lambda, stopping=stop_sym, max_iter=max_iter, sig_p=sig_p)
        last_result = res
        _maybe_plot(res; plot=plot, plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname))
        t = trend(res)
        c = cycle(res)
        result_df[!, "$(vname)_trend"] = round.(t; digits=6)
        result_df[!, "$(vname)_cycle"] = round.(c; digits=6)
        push!(cycles, c)
        push!(originals, y)

        _status("  $vname: $(res.iterations) iteration(s)")
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Boosted HP Filter (λ=$(lambda), stopping=$stopping)", key="boosted_hp_filter")
    _print_variance_ratios(sel_names, cycles, originals)
    return last_result
end

# ── X-13ARIMA-SEATS (C042 / MEMs 0.6.7 pure-Julia port) ───

function _filter_x13(; data::String,
                      frequency::Int=12,
                      method::String="seats",
                      transform::String="auto",
                      trading_day::Bool=false,
                      easter::Bool=false,
                      outliers::String="true",
                      critical_value::Float64=0.0,
                      columns::String="",
                      output::String="", format::String="table",
                      plot::Bool=false, plot_save::String="")
    frequency in (4, 12) || throw(CliError("usage/invalid-option",
        "--frequency must be 4 (quarterly) or 12 (monthly), got $frequency"))
    method_sym = Symbol(lowercase(method))
    method_sym in (:seats, :x11) || throw(CliError("usage/invalid-option",
        "--method must be seats or x11, got '$method'"))
    transform_sym = Symbol(lowercase(transform))
    transform_sym in (:auto, :log, :none) || throw(CliError("usage/invalid-option",
        "--transform must be auto|log|none, got '$transform'"))
    outliers_on = lowercase(outliers) == "true"

    df = load_data(data)
    Y = df_to_matrix(df)
    varnames = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    min_T = 3 * frequency
    T_obs < min_T && throw(CliError("data/too-short",
        "X-13 requires at least 3×frequency = $min_T observations, got T=$T_obs",
        hint="use a longer series or lower --frequency"))

    _status("X-13ARIMA-SEATS (method=$method_sym, frequency=$frequency): " *
            "$(length(col_idx)) variable(s), T=$T_obs")
    # Note: pure-Julia MEMs port (no external Census X-13 binary). Upstream #205
    # documents exact-ML likelihood determinant questions for SEATS paths.
    _status()

    adj_df = DataFrame(t = 1:T_obs)
    trend_df = DataFrame(t = 1:T_obs)
    seas_df = DataFrame(t = 1:T_obs)
    irr_df = DataFrame(t = 1:T_obs)
    diag_rows = NamedTuple[]
    last_result = nothing

    for ci in col_idx
        vname = varnames[ci]
        y = Y[:, ci]
        res = try
            MacroEconometricModels.x13_filter(y;
                frequency=frequency,
                method=method_sym,
                transform=transform_sym,
                trading_day=trading_day,
                easter=easter,
                outliers=outliers_on,
                critical_value=critical_value)
        catch e
            msg = sprint(showerror, e)
            # Reserved for platforms/builds that still require an external binary
            if occursin(r"(?i)(x-?13|binary|not found|could not find|executable)", msg)
                throw(CliError("env/x13-missing",
                    "X-13 seasonal adjustment is unavailable: $msg",
                    hint="MEMs ships a pure-Julia X-13 port; if you see this, check MacroEconometricModels install"))
            end
            if e isa ArgumentError
                throw(CliError("data/invalid", string(e)))
            end
            rethrow()
        end

        _maybe_plot(res; plot=plot,
                    plot_save=isempty(plot_save) ? "" : _per_var_output_path(plot_save, vname))

        adj_df[!, vname] = round.(res.adjusted; digits=6)
        trend_df[!, vname] = round.(res.trend; digits=6)
        seas_df[!, vname] = round.(res.seasonal; digits=6)
        irr_df[!, vname] = round.(res.irregular; digits=6)

        order = res.arima_order
        order_str = order isa Tuple ? join(string.(order), ",") : string(order)
        push!(diag_rows, (
            variable = vname,
            method = string(res.method),
            frequency = res.frequency,
            transform = string(res.transform),
            arima_order = order_str,
            aic = round(Float64(res.aic); digits=4),
            sigma2 = round(Float64(res.sigma2); digits=6),
            n_outliers = Int(res.n_outliers),
            T_obs = Int(res.T_obs),
        ))
        _status("  $vname: method=$(res.method), ARIMA=($order_str), outliers=$(res.n_outliers)")
        last_result = res
    end

    diag_df = DataFrame(diag_rows)
    output_result(adj_df; format=Symbol(format), output=output,
                  title="X-13 Seasonally Adjusted")
    output_result(trend_df; format=Symbol(format), output=output,
                  title="X-13 Trend")
    output_result(seas_df; format=Symbol(format), output=output,
                  title="X-13 Seasonal Factors")
    output_result(irr_df; format=Symbol(format), output=output,
                  title="X-13 Irregular")
    output_result(diag_df; format=Symbol(format), output=output,
                  title="X-13 Diagnostics")
    return last_result
end
