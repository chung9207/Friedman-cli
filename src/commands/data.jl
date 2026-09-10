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

# Data commands: list, load, import, export, describe, diagnose, fix, transform, filter, validate

function data_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["data", "list"],
            summary="table|csv|json",
            args=ArgSpec[],
            options=[
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:available_datasets,
                              description="Bundled example datasets with type, dimensions and description")],
            category="data",
            handler=wrap_legacy(_data_list),
        ),
        CommandSpec(
            path=["data", "load"],
            summary="Example dataset name (see 'data list'), or omit and pass --path for a CSV",
            args=[ArgSpec(name="name", type=String, required=false, default="", description="Example dataset name")],
            options=[
                OptionSpec(name="output", short="o", type=String, default="", description="Output CSV file path"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="vars", type=String, default="", description="Comma-separated variable subset"),
                OptionSpec(name="country", type=String, default="", description="Country filter (for PWT panel data)"),
                OptionSpec(name="dates", type=String, default="", description="Column name for date labels"),
                OptionSpec(name="path", type=String, default="", description="Path to CSV file (alternative to named dataset)")
            ],
            flags=[FlagSpec(name="transform", short="t", description="Apply FRED transformation codes")],
            # Emits no envelope table — writes CSV directly; gate-exempt, see check_table_keys.jl
            tables=TableSpec[],
            category="data",
            handler=wrap_legacy(_data_load),
        ),
        CommandSpec(
            path=["data", "describe"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:descriptive_statistics,
                              description="Per-variable count, first/last valid row, moments and quantiles")],
            category="data",
            handler=wrap_legacy(_data_describe),
        ),
        CommandSpec(
            path=["data", "diagnose"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:data_diagnostics,
                              description="Per-variable NaN and Inf counts and constant-series flag")],
            category="data",
            handler=wrap_legacy(_data_diagnose),
        ),
        CommandSpec(
            path=["data", "fix"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="method", short="m", type=String, default="listwise", description="listwise|interpolate|mean"),
                OptionSpec(name="output", short="o", type=String, default="", description="Output stem or CSV path"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            # Emits no envelope table — writes CSV directly; gate-exempt, see check_table_keys.jl
            tables=TableSpec[],
            category="data",
            handler=wrap_legacy(_data_fix),
        ),
        CommandSpec(
            path=["data", "transform"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="tcodes", type=String, default="", description="Comma-separated FRED transformation codes"),
                OptionSpec(name="output", short="o", type=String, default="", description="Output stem or CSV path"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            # Emits no envelope table — writes CSV directly; gate-exempt, see check_table_keys.jl
            tables=TableSpec[],
            category="data",
            handler=wrap_legacy(_data_transform),
        ),
        CommandSpec(
            path=["data", "filter"],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to CSV data file")],
            options=[
                OptionSpec(name="method", short="m", type=String, default="hp", description="hp|hamilton|bn|bk|bhp"),
                OptionSpec(name="component", type=String, default="cycle", description="cycle|trend"),
                OptionSpec(name="lambda", short="l", type=Float64, default=1600.0, description="Smoothing parameter (HP/BHP)"),
                OptionSpec(name="horizon", type=Int, default=8, description="Forecast horizon (Hamilton)"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Number of lags (Hamilton/BN)"),
                OptionSpec(name="columns", short="c", type=String, default="", description="Column indices, comma-separated (default: all)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:data_filter,
                              description="Selected filter component by time index, one column per variable")],
            category="data",
            handler=wrap_legacy(_data_filter),
        ),
        CommandSpec(
            path=["data", "validate"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="model", type=String, default="", description="Model type (var|bvar|vecm|arima|garch|sv|lp|gmm|factor)"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file")
            ],
            flags=FlagSpec[],
            # Emits no envelope table — stderr report only; gate-exempt, see check_table_keys.jl
            tables=TableSpec[],
            category="data",
            handler=wrap_legacy(_data_validate),
        ),
        CommandSpec(
            path=["data", "balance"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="method", type=String, default="dfm", description="dfm"),
                OptionSpec(name="factors", short="r", type=Int, default=3, description="Number of factors"),
                OptionSpec(name="lags", short="p", type=Int, default=2, description="Factor VAR lags"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:balanced_panel,
                              description="Balanced data matrix, one column per variable")],
            category="data",
            handler=wrap_legacy(_data_balance),
        ),
        CommandSpec(
            path=["data", "dropna"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="vars", type=String, default="", description="Column names to check (comma-separated; default: all)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:cleaned_data,
                              description="Rows surviving the NaN/Inf drop, one column per variable")],
            category="data",
            handler=wrap_legacy(_data_dropna),
        ),
        CommandSpec(
            path=["data", "keeprows"],
            summary="Handle stem or CSV path",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Handle stem or CSV path")],
            options=[
                OptionSpec(name="rows", type=String, default="", description="Row indices (e.g. 1:100, 1,5,10)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:filtered_data,
                              description="Rows selected by --rows, one column per variable")],
            category="data",
            handler=wrap_legacy(_data_keeprows),
        ),
        CommandSpec(
            path=["data", "import"],
            summary="Import CSV or :example to a typed .jld2 handle",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="CSV path, stem, or :example dataset")],
            options=[
                OptionSpec(name="kind", type=String, default="",
                           description="timeseries|panel|cross-section (required for CSV)",
                           choices=["timeseries", "panel", "cross-section"]),
                OptionSpec(name="frequency", type=String, default="other",
                           description="daily|monthly|quarterly|annual|mixed|other",
                           choices=["daily", "monthly", "quarterly", "annual", "mixed", "other"]),
                OptionSpec(name="dates", type=String, default="",
                           description="CSV column of date labels (timeseries)"),
                OptionSpec(name="id-col", type=String, default="",
                           description="Panel group column (required for --kind panel)"),
                OptionSpec(name="time-col", type=String, default="",
                           description="Panel time column (required for --kind panel)"),
                OptionSpec(name="vars", type=String, default="",
                           description="Comma-separated variable subset"),
                OptionSpec(name="tcodes", type=String, default="",
                           description="Comma-separated FRED tcode per variable"),
                OptionSpec(name="note", type=String, default="",
                           description="Free-form note stored in the handle header"),
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Output stem or path (default: input basename)"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table", "csv", "json"]),
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:imported_data,
                              description="Imported handle kind, dimensions, frequency and path")],
            category="data",
            data_kinds=[:csv, :timeseries, :panel, :cross_section],
            handler=wrap_legacy(_data_import),
        ),
        CommandSpec(
            path=["data", "export"],
            summary="Export a typed handle to CSV (frequency/tcode/dates dropped)",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Handle stem or path (TimeSeriesData/PanelData/CrossSectionData)")],
            options=[
                OptionSpec(name="output", short="o", type=String, default="",
                           description="Output CSV path (default: <stem>.csv)"),
                OptionSpec(name="format", short="f", type=String, default="table",
                           description="table|csv|json", choices=["table", "csv", "json"]),
            ],
            flags=FlagSpec[],
            # Emits no envelope table — writes CSV directly; gate-exempt, see check_table_keys.jl
            tables=TableSpec[],
            category="data",
            data_kinds=[:timeseries, :panel, :cross_section],
            handler=wrap_legacy(_data_export),
        )
    ]
end

const _DATA_CONTAINER_LEAVES = Set(["describe", "diagnose", "validate", "fix", "transform",
    "filter", "balance", "dropna", "keeprows"])

function register_data_commands!()
    specs = CommandSpec[]
    for s in data_specs()
        leaf = s.path[end]
        if leaf == "filter"
            # HP/Hamilton on a PanelData handle would treat group/time as series.
            push!(specs, _copy_spec(s; data_kinds=[:timeseries, :csv]))
        elseif leaf in _DATA_CONTAINER_LEAVES
            # data filter returns a DataFrame — do not tag MEMs filter result types.
            push!(specs, _copy_spec(s; data_kinds=[:timeseries, :panel, :cross_section, :csv]))
        elseif leaf == "load"
            push!(specs, _copy_spec(s; data_kinds=[:csv]))
        else
            # import already has kinds; export already has three-container kinds without :csv;
            # list has no data slot.
            push!(specs, s)
        end
    end
    specs = with_result_handles(with_default_csv_kinds(specs))
    register!(specs)
    return build_node("data", specs; description="Data management: import/export handles, load example datasets, inspect, clean, transform")
end


# ── Handlers ─────────────────────────────────────────────

# Descriptions for the bundled example datasets, keyed by their MEMs symbol.
# The table is built by walking EXAMPLE_DATASETS, so this map cannot silently
# omit a dataset that `data load` accepts.
const _DATASET_INFO = Dict(
    :fred_md      => ("Time Series",  "804 × 126",    "FRED-MD Monthly Database (126 macroeconomic indicators)"),
    :fred_qd      => ("Time Series",  "268 × 245",    "FRED-QD Quarterly Database (245 macroeconomic indicators)"),
    :pwt          => ("Panel",        "38 × 74 × 42", "Penn World Table (38 OECD countries, 74 years, 42 variables)"),
    :mpdta        => ("Panel",        "500 × 5 × 3",  "Callaway-Sant'Anna (2021) minimum wage panel"),
    :ddcg         => ("Panel",        "184 × 51 × 2", "Acemoglu et al. democracy-GDP panel"),
    :denmark      => ("Time Series",  "55 × 5",       "Danish money-demand data (Johansen-Juselius cointegration)"),
    :gnp_hamilton => ("Time Series",  "135 × 1",      "US GNP growth (Hamilton 1989 Markov-switching example)"),
    :grunfeld     => ("Panel",        "10 × 20 × 3",  "Grunfeld investment panel (10 firms, 20 years)"),
    :mp_shocks    => ("Time Series",  "240 × 8",      "US monetary panel + policy-shock series (McKay-Wolf 2023; NaN outside published samples)"),
    :mroz         => ("Cross Section", "753 × 22",    "Mroz (1987) female labour supply"),
    :nile         => ("Time Series",  "100 × 1",      "Nile river annual flow (local-level state space)"),
    :stackloss    => ("Cross Section", "21 × 4",      "Brownlee stack-loss plant data (robust regression)"),
)

function _data_list(; format::String="table", output::String="")
    info(d) = get(_DATASET_INFO, d, ("", "", ""))

    df = DataFrame(
        name=String[String(d) for d in EXAMPLE_DATASETS],
        type=String[info(d)[1] for d in EXAMPLE_DATASETS],
        dimensions=String[info(d)[2] for d in EXAMPLE_DATASETS],
        description=String[info(d)[3] for d in EXAMPLE_DATASETS]
    )

    output_result(df; format=Symbol(format), output=output, title="Available Datasets")
end

function _data_load(; name::String="", output::String="", format::String="table",
                     vars::String="", country::String="", transform::Bool=false,
                     dates::String="", path::String="")
    if isempty(name) && isempty(path)
        throw(CliError("usage/missing",
                       "give an example dataset name or --path <file.csv>";
                       hint="run 'friedman data list' to see the available datasets"))
    end

    # If --path is given, load from CSV instead of example datasets
    if !isempty(path)
        isempty(name) || _status("Both <name> and --path given; loading --path $path")
        path = _expanduser(path)
        df = load_data(path)
        Y = df_to_matrix(df)
        vn = variable_names(df)
        n_obs, n_vars = size(Y)

        ts = TimeSeriesData(Y; varnames=vn, tcode=fill(1, n_vars), time_index=collect(1:n_obs))

        if !isempty(dates)
            if dates in names(df)
                date_values = string.(df[!, dates])
                set_dates!(ts, date_values)
                _status_styled("  Date labels set from column: $dates\n"; color=:cyan)
            else
                _status_styled("  Warning: dates column '$dates' not found in data\n"; color=:yellow)
            end
        end

        out_path = isempty(output) ? "$(dataset_stem(path))_loaded.csv" : output
        _validate_output_path(out_path)
        out_df = DataFrame(Y, vn)
        CSV.write(out_path, out_df)
        _status("Loaded $(basename(path)): $n_obs × $n_vars")
        _status("Written to $out_path")
        return
    end

    name_sym = parse_dataset_name(name)
    name = String(name_sym)   # canonical stem for messages and default output paths
    dataset = load_example(name_sym)

    if dataset isa PanelData
        data_mat = to_matrix(dataset)
        vn = varnames(dataset)

        if !isempty(country)
            _status("Loading $name: filtering for country=$country")
        end

        out_path = isempty(output) ? "$name.csv" : output
        _validate_output_path(out_path)
        n_obs, n_vars = size(data_mat)
        df = DataFrame(data_mat, vn)
        # Add group/time columns for panel data
        insertcols!(df, 1, :group => dataset.group_id, :time => dataset.time_id)

        if !isempty(vars)
            var_list = [strip(s) for s in split(vars, ",") if !isempty(strip(s))]
            keep_cols = ["group", "time"]
            for v in var_list
                v in vn || throw(CliError("data/column-range",
                    "variable '$v' not found in $name";
                    hint="available: $(join(vn[1:min(5, length(vn))], ", "))..."))
                push!(keep_cols, v)
            end
            df = df[!, keep_cols]
        end

        CSV.write(out_path, df)
        _status("Loaded $name: $n_obs observations × $n_vars variables (Panel, $(dataset.n_groups) groups)")
        _status("Written to $out_path")
    else
        # TimeSeriesData, or a CrossSectionData set with no time dimension
        is_ts = dataset isa TimeSeriesData
        data_mat = to_matrix(dataset)
        vn = varnames(dataset)

        if transform
            is_ts || throw(CliError("usage/invalid",
                "--transform applies FRED transformation codes, which $name (cross section) does not have"))
            dataset = apply_tcode(dataset, dataset.tcode)
            data_mat = to_matrix(dataset)
            _status("Applied FRED transformation codes")
        end

        if !isempty(vars)
            var_list = [strip(s) for s in split(vars, ",") if !isempty(strip(s))]
            col_idx = Int[]
            for v in var_list
                idx = findfirst(==(v), vn)
                isnothing(idx) && throw(CliError("data/column-range",
                    "variable '$v' not found in $name";
                    hint="available: $(join(vn[1:min(5, length(vn))], ", "))..."))
                push!(col_idx, idx)
            end
            data_mat = data_mat[:, col_idx]
            vn = vn[col_idx]
        end

        if !isempty(dates) && is_ts
            # For named datasets, dates column would need to be in the variable names
            if dates in vn
                date_values = string.(data_mat[:, findfirst(==(dates), vn)])
                set_dates!(dataset, date_values)
                _status_styled("  Date labels set from column: $dates\n"; color=:cyan)
            else
                _status_styled("  Warning: dates column '$dates' not found in data\n"; color=:yellow)
            end
        end

        out_path = isempty(output) ? "$name.csv" : output
        _validate_output_path(out_path)
        n_obs, n_vars = size(data_mat)
        df = DataFrame(data_mat, vn)
        CSV.write(out_path, df)
        # `frequency` is only defined for time-series/panel sets, not cross sections.
        kind = is_ts ? string(frequency(dataset)) : "Cross Section"
        _status("Loaded $name: $n_obs × $n_vars ($kind)")
        _status("Written to $out_path")
    end
end

function _data_describe(; data::String, format::String="table", output::String="")
    tsd = _as_macro_data(data)
    summary = _status_stdout() do
        describe_data(tsd)
    end
    Y = to_matrix(tsd)
    vn = varnames(tsd)
    n_obs, n_vars = size(Y)

    # NaN-padded series (:mp_shocks ships 6 of 8 columns NaN outside their
    # published samples) are valid only inside a window that `n` alone cannot
    # locate — report the first/last finite row per column (0 = none finite).
    fv = [something(findfirst(isfinite, @view Y[:, j]), 0) for j in 1:n_vars]
    lv = [something(findlast(isfinite, @view Y[:, j]), 0) for j in 1:n_vars]

    result_df = DataFrame(
        variable=vn,
        n=summary.n,   # already per-variable; fill() nested a vector into every cell
        first_valid=fv,
        last_valid=lv,
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

    _status("Data Summary: $n_obs observations × $n_vars variables")
    _status()
    output_result(result_df; format=Symbol(format), output=output, title="Descriptive Statistics")
    return tsd
end

function _data_diagnose(; data::String, format::String="table", output::String="")
    tsd = _as_macro_data(data)
    diag = diagnose(tsd)
    Y = to_matrix(tsd)
    vn = varnames(tsd)
    n_obs, n_vars = size(Y)

    result_df = DataFrame(
        variable=vn,
        n_nan=diag.n_nan,
        n_inf=diag.n_inf,
        is_constant=diag.is_constant,
    )

    _status("Data Diagnostics: $n_obs observations × $n_vars variables")
    _status()
    output_result(result_df; format=Symbol(format), output=output, title="Data Diagnostics")

    _status()
    if diag.is_clean
        _status_styled("Data is clean: no issues found\n"; color=:green)
    else
        n_issues = count(diag.n_nan .> 0) + count(diag.n_inf .> 0) + count(diag.is_constant)
        _status_styled("Found issues in $n_issues variable(s)\n"; color=:yellow)
        if diag.is_short
            _status_styled("Warning: series is short ($n_obs observations)\n"; color=:yellow)
        end
    end
    return tsd
end

function _as_macro_data(obj)
    obj isa DataFrame || return obj
    Y = df_to_matrix(obj)
    vn = variable_names(obj)
    return TimeSeriesData(Y; varnames=vn, tcode=fill(1, length(vn)),
                          time_index=collect(1:size(Y, 1)))
end
_as_macro_data(data::String) = _as_macro_data(resolve_data(data))

function _persist_edit(obj, data::String, output::String, suffix::String)
    out_default = _default_edit_output(data, suffix)
    return _write_macro_data(obj, isempty(output) ? out_default : output;
                             input_path=data, default_stem=dataset_stem(data) * suffix)
end

function _data_fix(; data::String, method::String="listwise", output::String="", format::String="table")
    validate_method(method, ["listwise", "interpolate", "mean"], "fix method")
    tsd = _as_macro_data(resolve_data(data))
    fixed = fix(tsd; method=Symbol(method))
    n_obs, n_vars = size(to_matrix(fixed))
    _status("Fixed data ($method): $n_obs observations × $n_vars variables")
    _persist_edit(fixed, data, output, "_clean")
    return fixed
end

function _data_transform(; data::String, tcodes::String="", output::String="", format::String="table")
    tsd = _as_macro_data(resolve_data(data))
    _data_kind_of(tsd) === :cross_section && throw(CliError("data/wrong-kind",
        "transform does not apply to CrossSectionData"))
    vn = varnames(tsd)
    n_vars = length(vn)

    isempty(tcodes) && throw(CliError("usage/missing",
        "--tcodes is required";
        hint="comma-separated FRED transformation codes, e.g. --tcodes 5,5,1,6"))

    codes = [parse(Int, strip(s)) for s in split(tcodes, ",") if !isempty(strip(s))]
    length(codes) == n_vars || throw(CliError("usage/invalid",
        "number of tcodes ($(length(codes))) must match number of variables ($n_vars)"))

    transformed = apply_tcode(tsd, codes)
    tcode_names = Dict(1=>"level", 2=>"Δ", 3=>"Δ²", 4=>"log", 5=>"Δlog", 6=>"Δ²log", 7=>"Δ%")

    _status("Transformed $n_vars variable(s):")
    for (i, vname) in enumerate(vn)
        code = codes[i]
        label = get(tcode_names, code, "code=$code")
        _status("  $vname: tcode=$code ($label)")
    end
    _persist_edit(transformed, data, output, "_transformed")
    return transformed
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
    _status("Data Filter ($method, component=$component): $(length(col_idx)) variable(s), T=$T_obs")
    _status()

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
    output_result(result_df; format=Symbol(format), output=output, title=title, key="data_filter")
    return result_df
end

function _data_validate(; data::String, model::String="", format::String="table", output::String="")
    isempty(model) && error("--model is required (var|bvar|vecm|arima|garch|sv|lp|gmm|factor)")

    allowed_models = ["var", "bvar", "vecm", "arima", "garch", "sv", "lp", "gmm", "factor",
                       "arch", "egarch", "gjr_garch", "static", "dynamic", "gdfm"]
    validate_method(model, allowed_models, "model type")

    tsd = _as_macro_data(data)
    n_obs, n_vars = size(to_matrix(tsd))

    try
        validate_for_model(tsd, Symbol(model))
        _status_styled("Data is valid for $model estimation ($n_vars variable(s), $n_obs observations)\n"; color=:green)
    catch e
        _status_styled("Data validation failed for $model:\n"; color=:red)
        _status("  ", e.msg)
    end
    return tsd
end

function _data_balance(; data::String, method::String="dfm", factors::Int=3,
                        lags::Int=2, output::String="", format::String="table")
    ts = _as_macro_data(resolve_data(data))
    vn = varnames(ts)
    Y = to_matrix(ts)

    _status("Balancing panel via $(method): $(length(vn)) variables, T=$(size(Y, 1))")
    _status()

    balanced = balance_panel(ts; method=Symbol(method), r=factors, p=lags)
    _persist_edit(balanced, data, output, "_balanced")
    bal_Y = hasproperty(balanced, :data) ? balanced.data : to_matrix(balanced)
    result_df = DataFrame(bal_Y, vn)
    output_result(result_df; format=Symbol(format), output="",
                  title="Balanced Panel (method=$method, r=$factors, p=$lags)", key="balanced_panel")
end

function _data_dropna(; data::String, vars::String="",
                       output::String="", format::String="table")
    ts = _as_macro_data(resolve_data(data))
    vnames = varnames(ts)
    n_before = size(to_matrix(ts), 1)
    # Real dropna's kwarg is asserted ::Union{Vector{String},Nothing} — a bare
    # strip() comprehension is Vector{SubString} and TypeErrors on every --vars
    # invocation, so materialize Strings and pre-check the names ourselves.
    var_list = isempty(vars) ? nothing :
               String[String(strip(s)) for s in split(vars, ",") if !isempty(strip(s))]
    if var_list !== nothing
        for v in var_list
            v in vnames || throw(CliError("data/column-range",
                "variable '$v' not found";
                hint="available: $(join(vnames[1:min(5, length(vnames))], ", "))..."))
        end
    end
    cleaned = try
        dropna(ts; vars=var_list)
    catch e
        # "All rows contain NaN or Inf" is an untyped ArgumentError upstream.
        e isa ArgumentError ? throw(CliError("data/invalid", e.msg)) : rethrow()
    end
    n_after = size(to_matrix(cleaned), 1)

    _status("Drop NA: $data")
    _status("  Rows before: $n_before, after: $n_after, dropped: $(n_before - n_after)")
    _status()

    _persist_edit(cleaned, data, output, "_dropna")
    result_df = DataFrame(to_matrix(cleaned), varnames(cleaned))
    output_result(result_df; format=Symbol(format), output="", title="Cleaned Data")
    return cleaned
end

function _data_keeprows(; data::String, rows::String="",
                         output::String="", format::String="table")
    isempty(rows) && throw(CliError("usage/missing",
        "--rows is required (e.g. 1:100, 1,5,10)"))

    ts = _as_macro_data(resolve_data(data))
    n_total = size(to_matrix(ts), 1)

    _rows_err() = throw(CliError("usage/invalid",
        "--rows must be a range like 1:100 (or 1:end) or a comma list like 1,5,10 (got '$rows')"))
    indices = if occursin(":", rows)
        parts = split(rows, ":")
        length(parts) == 2 || _rows_err()
        lo = tryparse(Int, strip(parts[1]))
        hi_str = strip(parts[2])
        hi = hi_str == "end" ? n_total : tryparse(Int, hi_str)
        (lo === nothing || hi === nothing) && _rows_err()
        collect(lo:hi)
    else
        ix = [tryparse(Int, strip(s)) for s in split(rows, ",")]
        any(isnothing, ix) && _rows_err()
        Int[i for i in ix]
    end
    # Real keeprows raises an untyped ArgumentError on an empty selection and a
    # BoundsError past T_obs — both would exit 1; guard here instead.
    isempty(indices) && throw(CliError("usage/invalid",
        "--rows selects no rows (got '$rows')"))
    (minimum(indices) < 1 || maximum(indices) > n_total) &&
        throw(CliError("usage/invalid",
            "--rows out of range: data has rows 1:$n_total (got '$rows')"))

    filtered = keeprows(ts, indices)

    _status("Keep Rows: $data")
    _status("  Selected $(length(indices)) of $n_total rows")
    _status()

    _persist_edit(filtered, data, output, "_rows")
    result_df = DataFrame(to_matrix(filtered), varnames(filtered))
    output_result(result_df; format=Symbol(format), output="", title="Filtered Data")
    return filtered
end

function _parse_kind(kind::String)
    k = lowercase(strip(kind))
    k == "timeseries" && return :timeseries
    k == "cross-section" && return :cross_section
    k == "panel" && return :panel
    throw(CliError("usage/invalid",
        "--kind must be timeseries|panel|cross-section (got '$kind')"))
end

function _frequency_value(s::String)
    key = lowercase(strip(s))
    key == "annual" && (key = "yearly")
    # Bare Daily/Monthly/... would be UndefVarError under the mock (no Frequency
    # enum). Resolve via getfield so this works on mock Symbols and real enums.
    if isdefined(MacroEconometricModels, :Monthly)
        M = MacroEconometricModels
        key == "daily" && return getfield(M, :Daily)
        key == "monthly" && return getfield(M, :Monthly)
        key == "quarterly" && return getfield(M, :Quarterly)
        key == "yearly" && return getfield(M, :Yearly)
        key == "mixed" && return getfield(M, :Mixed)
        return getfield(M, :Other)
    else
        return Symbol(key)
    end
end

function _parse_tcodes(tcodes::String, n_vars::Int)
    codes = if isempty(tcodes)
        fill(1, n_vars)
    else
        out = Int[]
        for raw in split(tcodes, ",")
            t = strip(raw)
            isempty(t) && continue
            v = tryparse(Int, t)
            v === nothing && throw(CliError("usage/invalid",
                "--tcodes must be comma-separated integers (got '$t')"))
            (1 <= v <= 7) || throw(CliError("usage/invalid",
                "tcode must be 1:7 (got $v)"))
            push!(out, v)
        end
        out
    end
    length(codes) == n_vars || throw(CliError("usage/invalid",
        "number of tcodes ($(length(codes))) must match number of variables ($n_vars)"))
    return codes
end

function _require_columns(df, names_::Vector{String}; label::String="variable")
    colnames = names(df)
    for v in names_
        v in colnames || throw(CliError("data/column-range",
            "$label '$v' not found";
            hint="available: $(join(colnames[1:min(5, length(colnames))], ", "))..."))
    end
    return nothing
end

function _reject_missing_numeric(df, vn::Vector{String})
    for c in vn
        any(ismissing, df[!, c]) && throw(CliError("data/missing-values",
            "column '$c' contains missing values; drop or impute them (e.g. via `data dropna`/`data fix`) first"))
    end
    return nothing
end

function _import_mems_error(e)
    e isa CliError && return e
    mapped = _domain_error_class(e)
    mapped !== nothing && return mapped
    if e isa ArgumentError
        msg = _err_message(e)
        lmsg = lowercase(msg)
        occursin("not found", lmsg) && return CliError("data/column-range", msg)
        occursin("tcode", lmsg) && return CliError("usage/invalid", msg)
        return CliError("data/invalid", msg)
    end
    return e
end

function _data_import(; data::String, kind::String="", frequency::String="other",
                       dates::String="", id_col::String="", time_col::String="",
                       vars::String="", tcodes::String="", note::String="",
                       output::String="", format::String="table")
    src = data
    if !startswith(src, ":")
        src = resolve_stem(src; slot=:data)
        # Import is CSV|:example → handle, not handle → handle.
        _is_handle_path(src) && throw(CliError("usage/invalid",
            "data import reads CSV or :example, not an existing handle";
            hint="use `data export` or pass a CSV path"))
    end
    out_stem = isempty(output) ? joinpath(dirname(src), dataset_stem(src)) : output
    out_path = resolve_save_path(out_stem)
    _validate_output_path(out_path)

    obj = if startswith(src, ":")
        ds = load_example(parse_dataset_name(src))
        inferred = _data_kind_of(ds)
        if !isempty(kind)
            want = _parse_kind(kind)
            want === inferred || throw(CliError("data/wrong-kind",
                ":example is $inferred; --kind $kind does not match"))
        end
        ds
    else
        isempty(kind) && throw(CliError("usage/invalid",
            "--kind is required when importing a CSV";
            hint="--kind timeseries|panel|cross-section"))
        want = _parse_kind(kind)
        # Panel identifiers are required even when --vars would otherwise subset
        # them away (DataFrames ArgumentError on the empty name).
        if want === :panel
            (isempty(id_col) || isempty(time_col)) && throw(CliError("usage/missing",
                "panel import requires --id-col and --time-col"))
        elseif want === :cross_section
            frequency == "other" || throw(CliError("usage/invalid",
                "--frequency does not apply to --kind cross-section"))
        end
        df = load_data(src)
        if want === :panel
            _require_columns(df, String[id_col]; label="id column")
            _require_columns(df, String[time_col]; label="time column")
        end
        if !isempty(dates)
            _require_columns(df, String[dates]; label="dates column")
        end
        extra = String[]
        want === :panel && append!(extra, (id_col, time_col))
        !isempty(dates) && push!(extra, dates)
        if !isempty(vars)
            keep = String[String(strip(s)) for s in split(vars, ",") if !isempty(strip(s))]
            _require_columns(df, keep)
            df = df[!, unique(vcat(extra, keep))]
        end
        # Numeric --dates / panel id-col/time-col are metadata, not variables
        # (same class as lat/lon in `_load_reg_data`). Keep dates on `df` for set_dates!.
        exclude = Set{String}(extra)
        vn = [n for n in variable_names(df) if n ∉ exclude]
        isempty(vn) && throw(CliError("data/no-numeric-columns", "no numeric columns found in data"))
        _reject_missing_numeric(df, vn)
        try
            if want === :timeseries
                Y = Matrix{Float64}(df[!, vn])
                codes = _parse_tcodes(tcodes, length(vn))
                ts = TimeSeriesData(Y; varnames=vn, frequency=_frequency_value(frequency),
                                    tcode=codes, time_index=collect(1:size(Y, 1)))
                if !isempty(dates)
                    set_dates!(ts, string.(df[!, dates]))
                end
                ts
            elseif want === :panel
                xtset(df[!, unique(vcat(String[id_col, time_col], vn))],
                      Symbol(id_col), Symbol(time_col); frequency=_frequency_value(frequency))
            else
                Y = Matrix{Float64}(df[!, vn])
                CrossSectionData(Y; varnames=vn)
            end
        catch e
            throw(_import_mems_error(e))
        end
    end

    # Always persist first; `note=` is a best-effort native-header rewrite.
    save_model_dispatch(out_path, obj)
    if !isempty(note)
        try
            MacroEconometricModels.save_model(obj, out_path; note=note)
        catch e
            e isa MethodError || rethrow()
            _status("note= ignored: save_model does not accept note in this build")
        end
    end
    k = _data_kind_of(obj)
    nobs = hasproperty(obj, :data) ? size(obj.data, 1) : 0
    nvars = hasproperty(obj, :varnames) ? length(obj.varnames) : 0
    output_kv(Pair{String,Any}[
        "kind" => string(k),
        "n_obs" => nobs,
        "n_vars" => nvars,
        "path" => out_path,
        "frequency" => string(frequency),
    ]; format=format, output="", title="Imported Data", key="imported_data")
    _status("Imported $k → $out_path")
    return obj
end

function _data_export(; data::String, output::String="", format::String="table")
    obj = resolve_data(data)
    k = _data_kind_of(obj)
    k in (:timeseries, :panel, :cross_section) || throw(CliError("data/wrong-kind",
        "$data is not a data container (got $(typeof(obj)))";
        hint="pass a TimeSeriesData/PanelData/CrossSectionData handle"))
    out = if isempty(output)
        resolved = try
            resolve_stem(data; slot=:data)
        catch
            data
        end
        dir = startswith(resolved, ":") ? "" : dirname(resolved)
        base = dataset_stem(data) * ".csv"
        (isempty(dir) || dir == ".") ? base : joinpath(dir, base)
    else
        output
    end
    endswith(lowercase(out), ".csv") || (out = out * ".csv")
    _write_macro_data(obj, out; input_path=data, default_stem=dataset_stem(data),
                      warn_metadata=true)
    return obj
end
