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

# Filter commands: hp, hamilton, bn, bk, bhp

function register_filter_commands!()
    filt_hp = LeafCommand("hp", _filter_hp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lambda"; short="l", type=Float64, default=1600.0, description="Smoothing parameter (6.25 annual, 1600 quarterly, 129600 monthly)"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all numeric)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Hodrick-Prescott filter")

    filt_hamilton = LeafCommand("hamilton", _filter_hamilton;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("horizon"; short="h", type=Int, default=8, description="Forecast horizon"),
            Option("lags"; short="p", type=Int, default=4, description="Number of lags in regression"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all numeric)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Hamilton (2018) regression filter")

    filt_bn = LeafCommand("bn", _filter_bn;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("method"; type=String, default="arima", description="arima|statespace"),
            Option("p"; type=Int, default=nothing, description="AR order (default: auto)"),
            Option("q"; type=Int, default=nothing, description="MA order (default: auto)"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all numeric)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Beveridge-Nelson decomposition")

    filt_bk = LeafCommand("bk", _filter_bk;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("pl"; type=Int, default=6, description="Minimum period of oscillation"),
            Option("pu"; type=Int, default=32, description="Maximum period of oscillation"),
            Option("K"; type=Int, default=12, description="Truncation length (leads/lags)"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all numeric)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Baxter-King band-pass filter")

    filt_bhp = LeafCommand("bhp", _filter_bhp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lambda"; short="l", type=Float64, default=1600.0, description="Smoothing parameter"),
            Option("stopping"; type=String, default="BIC", description="BIC|ADF|fixed"),
            Option("max-iter"; type=Int, default=100, description="Maximum boosting iterations"),
            Option("sig-p"; type=Float64, default=0.05, description="ADF significance level"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all numeric)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Boosted HP filter (Phillips & Shi 2021)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "hp"       => filt_hp,
        "hamilton" => filt_hamilton,
        "bn"       => filt_bn,
        "bk"       => filt_bk,
        "bhp"      => filt_bhp,
    )
    return NodeCommand("filter", subcmds, "Time series filtering and trend-cycle decomposition")
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
    println()
    printstyled("Cycle Variance Ratios:\n"; bold=true)
    for (i, vname) in enumerate(varnames)
        total_var = var(originals[i])
        cycle_var = var(cycles[i])
        ratio = total_var > 0 ? cycle_var / total_var : 0.0
        println("  $vname: $(round(ratio; digits=4))")
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

    println("HP Filter (λ=$(lambda)): $(length(col_idx)) variable(s), T=$T_obs")
    println()

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
                  title="HP Filter (λ=$(lambda))")
    _print_variance_ratios(sel_names, cycles, originals)
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

    println("Hamilton Filter (h=$horizon, p=$lags): $(length(col_idx)) variable(s), T=$T_obs")
    println()

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
        printstyled("Note: $lost initial observations lost due to filter requirements\n"; color=:yellow)
        println()
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Hamilton Filter (h=$horizon, p=$lags)")
    _print_variance_ratios(sel_names, cycles, originals)
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
    println("Beveridge-Nelson Decomposition (method=$method, p=$p_label, q=$q_label): $(length(col_idx)) variable(s), T=$T_obs")
    println()

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

    println("Baxter-King Filter (pl=$pl, pu=$pu, K=$K): $(length(col_idx)) variable(s), T=$T_obs")
    println()

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
        printstyled("Note: $total_lost observations lost ($K leads/lags trimmed from each end)\n"; color=:yellow)
        println()
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Baxter-King Filter (pl=$pl, pu=$pu, K=$K)")
    _print_variance_ratios(sel_names, cycles, originals)
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
    println("Boosted HP Filter (λ=$(lambda), stopping=$stopping): $(length(col_idx)) variable(s), T=$T_obs")
    println()

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

        println("  $vname: $(res.iterations) iteration(s)")
    end

    sel_names = [varnames[ci] for ci in col_idx]
    output_result(result_df; format=Symbol(format), output=output,
                  title="Boosted HP Filter (λ=$(lambda), stopping=$stopping)")
    _print_variance_ratios(sel_names, cycles, originals)
end
