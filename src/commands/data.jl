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

# Data commands: list, load, describe, diagnose, fix, transform, filter, validate

function register_data_commands!()
    data_list = LeafCommand("list", _data_list;
        args=Argument[],
        options=[
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="List available example datasets")

    data_load = LeafCommand("load", _data_load;
        args=[Argument("name"; description="Dataset name (fred_md|fred_qd|pwt)")],
        options=[
            Option("output"; short="o", type=String, default="", description="Output CSV file path"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("vars"; type=String, default="", description="Comma-separated variable subset"),
            Option("country"; type=String, default="", description="Country filter (for PWT panel data)"),
        ],
        flags=[Flag("transform"; short="t", description="Apply FRED transformation codes")],
        description="Load example dataset and export as CSV")

    data_describe = LeafCommand("describe", _data_describe;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Summary statistics for a dataset")

    data_diagnose = LeafCommand("diagnose", _data_diagnose;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Data quality diagnostics (NaN, Inf, constant columns)")

    data_fix = LeafCommand("fix", _data_fix;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("method"; short="m", type=String, default="listwise", description="listwise|interpolate|mean"),
            Option("output"; short="o", type=String, default="", description="Output CSV file path"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Clean data (handle NaN/Inf/constant columns)")

    data_transform = LeafCommand("transform", _data_transform;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("tcodes"; type=String, default="", description="Comma-separated FRED transformation codes"),
            Option("output"; short="o", type=String, default="", description="Output CSV file path"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Apply FRED transformation codes")

    data_filter = LeafCommand("filter", _data_filter;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("method"; short="m", type=String, default="hp", description="hp|hamilton|bn|bk|bhp"),
            Option("component"; type=String, default="cycle", description="cycle|trend"),
            Option("lambda"; short="l", type=Float64, default=1600.0, description="Smoothing parameter (HP/BHP)"),
            Option("horizon"; type=Int, default=8, description="Forecast horizon (Hamilton)"),
            Option("lags"; short="p", type=Int, default=4, description="Number of lags (Hamilton/BN)"),
            Option("columns"; short="c", type=String, default="", description="Column indices, comma-separated (default: all)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
        ],
        description="Apply time series filter (unified interface)")

    data_validate = LeafCommand("validate", _data_validate;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("model"; type=String, default="", description="Model type (var|bvar|vecm|arima|garch|sv|lp|gmm|factor)"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
        ],
        description="Validate data suitability for a model type")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "list"      => data_list,
        "load"      => data_load,
        "describe"  => data_describe,
        "diagnose"  => data_diagnose,
        "fix"       => data_fix,
        "transform" => data_transform,
        "filter"    => data_filter,
        "validate"  => data_validate,
    )
    return NodeCommand("data", subcmds, "Data management: load example datasets, inspect, clean, transform")
end

# ── Handlers ─────────────────────────────────────────────

function _data_list(; format::String="table", output::String="")
    datasets = [
        ("fred_md",  "Time Series", "804 × 126",    "FRED-MD Monthly Database (126 macroeconomic indicators)"),
        ("fred_qd",  "Time Series", "268 × 245",    "FRED-QD Quarterly Database (245 macroeconomic indicators)"),
        ("pwt",      "Panel",       "38 × 74 × 42", "Penn World Table (38 OECD countries, 74 years, 42 variables)"),
    ]

    df = DataFrame(
        name=String[d[1] for d in datasets],
        type=String[d[2] for d in datasets],
        dimensions=String[d[3] for d in datasets],
        description=String[d[4] for d in datasets]
    )

    output_result(df; format=Symbol(format), output=output, title="Available Datasets")
end

function _data_load(; name::String, output::String="", format::String="table",
                     vars::String="", country::String="", transform::Bool=false)
    name_sym = Symbol(name)
    dataset = load_example(name_sym)

    if dataset isa PanelData
        data_mat = to_matrix(dataset)
        vn = varnames(dataset)

        if !isempty(country)
            println("Loading $name: filtering for country=$country")
        end

        out_path = isempty(output) ? "$name.csv" : output
        n_obs, n_vars = size(data_mat)
        df = DataFrame(data_mat, vn)
        # Add group/time columns for panel data
        insertcols!(df, 1, :group => dataset.group_id, :time => dataset.time_id)

        if !isempty(vars)
            var_list = [strip(s) for s in split(vars, ",") if !isempty(strip(s))]
            keep_cols = ["group", "time"]
            for v in var_list
                v in vn || error("variable '$v' not found in $name (available: $(join(vn[1:min(5, length(vn))], ", "))...)")
                push!(keep_cols, v)
            end
            df = df[!, keep_cols]
        end

        CSV.write(out_path, df)
        println("Loaded $name: $n_obs observations × $n_vars variables (Panel, $(dataset.n_groups) groups)")
        println("Written to $out_path")
    else
        # TimeSeriesData
        data_mat = to_matrix(dataset)
        vn = varnames(dataset)

        if transform
            dataset = apply_tcode(dataset, dataset.tcode)
            data_mat = to_matrix(dataset)
            println("Applied FRED transformation codes")
        end

        if !isempty(vars)
            var_list = [strip(s) for s in split(vars, ",") if !isempty(strip(s))]
            col_idx = Int[]
            for v in var_list
                idx = findfirst(==(v), vn)
                isnothing(idx) && error("variable '$v' not found in $name (available: $(join(vn[1:min(5, length(vn))], ", "))...)")
                push!(col_idx, idx)
            end
            data_mat = data_mat[:, col_idx]
            vn = vn[col_idx]
        end

        out_path = isempty(output) ? "$name.csv" : output
        n_obs, n_vars = size(data_mat)
        df = DataFrame(data_mat, vn)
        CSV.write(out_path, df)
        freq_str = string(frequency(dataset))
        println("Loaded $name: $n_obs × $n_vars ($freq_str)")
        println("Written to $out_path")
    end
end

function _data_describe(; data::String, format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    n_obs, n_vars = size(Y)

    tsd = TimeSeriesData(Y; varnames=vn, tcode=fill(1, n_vars), time_index=collect(1:n_obs))
    summary = describe_data(tsd)

    result_df = DataFrame(
        variable=vn,
        n=fill(summary.n, n_vars),
        mean=round.(summary.mean; digits=4),
        std=round.(summary.std; digits=4),
        min=round.(summary.min; digits=4),
        p25=round.(summary.p25; digits=4),
        median=round.(summary.median; digits=4),
        p75=round.(summary.p75; digits=4),
        max=round.(summary.max; digits=4),
        skewness=round.(summary.skewness; digits=4),
        kurtosis=round.(summary.kurtosis; digits=4),
    )

    println("Data Summary: $n_obs observations × $n_vars variables")
    println()
    output_result(result_df; format=Symbol(format), output=output, title="Descriptive Statistics")
end

function _data_diagnose(; data::String, format::String="table", output::String="")
    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    n_obs, n_vars = size(Y)

    tsd = TimeSeriesData(Y; varnames=vn, tcode=fill(1, n_vars), time_index=collect(1:n_obs))
    diag = diagnose(tsd)

    result_df = DataFrame(
        variable=vn,
        n_nan=diag.n_nan,
        n_inf=diag.n_inf,
        is_constant=diag.is_constant,
    )

    println("Data Diagnostics: $n_obs observations × $n_vars variables")
    println()
    output_result(result_df; format=Symbol(format), output=output, title="Data Diagnostics")

    println()
    if diag.is_clean
        printstyled("Data is clean: no issues found\n"; color=:green)
    else
        n_issues = count(diag.n_nan .> 0) + count(diag.n_inf .> 0) + count(diag.is_constant)
        printstyled("Found issues in $n_issues variable(s)\n"; color=:yellow)
        if diag.is_short
            printstyled("Warning: series is short ($n_obs observations)\n"; color=:yellow)
        end
    end
end

function _data_fix(; data::String, method::String="listwise", output::String="", format::String="table")
    validate_method(method, ["listwise", "interpolate", "mean"], "fix method")

    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    n_obs, n_vars = size(Y)

    tsd = TimeSeriesData(Y; varnames=vn, tcode=fill(1, n_vars), time_index=collect(1:n_obs))
    fixed = fix(tsd; method=Symbol(method))
    fixed_mat = to_matrix(fixed)

    out_path = if !isempty(output)
        output
    else
        base = replace(basename(data), r"\.[^.]+$" => "")
        "$(base)_clean.csv"
    end

    fixed_df = DataFrame(fixed_mat, vn)
    CSV.write(out_path, fixed_df)
    println("Fixed data ($method): $n_obs observations × $n_vars variables")
    println("Written to $out_path")
end

function _data_transform(; data::String, tcodes::String="", output::String="", format::String="table")
    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    n_obs, n_vars = size(Y)

    isempty(tcodes) && error("--tcodes is required (comma-separated FRED transformation codes, e.g., 5,5,1,6)")

    codes = [parse(Int, strip(s)) for s in split(tcodes, ",") if !isempty(strip(s))]
    length(codes) == n_vars || error("number of tcodes ($(length(codes))) must match number of variables ($n_vars)")

    tsd = TimeSeriesData(Y; varnames=vn, tcode=codes, time_index=collect(1:n_obs))
    transformed = apply_tcode(tsd, codes)
    trans_mat = to_matrix(transformed)

    tcode_names = Dict(1=>"level", 2=>"Δ", 3=>"Δ²", 4=>"log", 5=>"Δlog", 6=>"Δ²log", 7=>"Δ%")

    out_path = if !isempty(output)
        output
    else
        base = replace(basename(data), r"\.[^.]+$" => "")
        "$(base)_transformed.csv"
    end

    trans_df = DataFrame(trans_mat, vn)
    CSV.write(out_path, trans_df)

    println("Transformed $n_vars variable(s):")
    for (i, vname) in enumerate(vn)
        code = codes[i]
        label = get(tcode_names, code, "code=$code")
        println("  $vname: tcode=$code ($label)")
    end
    println("Written to $out_path")
end

function _data_filter(; data::String, method::String="hp", component::String="cycle",
                       lambda::Float64=1600.0, horizon::Int=8, lags::Int=4,
                       columns::String="", output::String="", format::String="table")
    validate_method(method, ["hp", "hamilton", "bn", "bk", "bhp"], "filter method")
    component in ("cycle", "trend") || error("unknown component: $component (expected cycle|trend)")

    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    T_obs, n = size(Y)
    col_idx = _parse_columns(columns, n)

    method_sym = Symbol(method)
    println("Data Filter ($method, component=$component): $(length(col_idx)) variable(s), T=$T_obs")
    println()

    result_df = DataFrame()
    first_done = false

    for ci in col_idx
        vname = vn[ci]
        y = Y[:, ci]

        # Call specific filter functions directly (apply_filter no longer accepts raw vectors in v0.2.2)
        res = if method_sym == :hp
            hp_filter(y; lambda=lambda)
        elseif method_sym == :hamilton
            hamilton_filter(y; h=horizon, p=lags)
        elseif method_sym == :bn
            beveridge_nelson(y)
        elseif method_sym == :bk
            baxter_king(y)
        elseif method_sym == :bhp
            boosted_hp(y; lambda=lambda)
        end

        t = trend(res)
        c = cycle(res)

        # Handle valid_range for Hamilton and BK
        # trend()/cycle() may be full-length or valid-range-only
        if hasproperty(res, :valid_range)
            vr = res.valid_range
            if !first_done
                result_df.t = collect(vr)
                first_done = true
            end
            val = component == "cycle" ? c : t
            selected = length(val) == T_obs ? val[vr] : val
        else
            if !first_done
                result_df.t = 1:T_obs
                first_done = true
            end
            selected = component == "cycle" ? c : t
        end

        result_df[!, vname] = round.(selected; digits=6)
    end

    title = "Data Filter: $method ($component component)"
    output_result(result_df; format=Symbol(format), output=output, title=title)
end

function _data_validate(; data::String, model::String="", format::String="table", output::String="")
    isempty(model) && error("--model is required (var|bvar|vecm|arima|garch|sv|lp|gmm|factor)")

    allowed_models = ["var", "bvar", "vecm", "arima", "garch", "sv", "lp", "gmm", "factor",
                       "arch", "egarch", "gjr_garch", "static", "dynamic", "gdfm"]
    validate_method(model, allowed_models, "model type")

    df = load_data(data)
    Y = df_to_matrix(df)
    vn = variable_names(df)
    n_obs, n_vars = size(Y)

    tsd = TimeSeriesData(Y; varnames=vn, tcode=fill(1, n_vars), time_index=collect(1:n_obs))

    try
        validate_for_model(tsd, Symbol(model))
        printstyled("Data is valid for $model estimation ($n_vars variable(s), $n_obs observations)\n"; color=:green)
    catch e
        printstyled("Data validation failed for $model:\n"; color=:red)
        println("  ", e.msg)
    end
end
