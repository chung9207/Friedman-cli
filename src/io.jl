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

# IO utilities: CSV reading, table/CSV/JSON output

"""
    load_data(path) → DataFrame

Read a CSV file and return a DataFrame. Validates that the file exists and is non-empty.
"""
function load_data(path::String)
    isfile(path) || error("file not found: $path")
    df = CSV.read(path, DataFrame)
    nrow(df) == 0 && error("empty dataset: $path")
    return df
end

"""
    _numeric_column_names(df) → Vector{String}

Return the names of numeric columns in a DataFrame.
"""
_numeric_column_names(df::DataFrame) =
    [n for n in names(df) if eltype(df[!, n]) <: Union{Number, Missing}]

"""
    df_to_matrix(df) → Matrix{Float64}

Convert a DataFrame to a numeric matrix, selecting only numeric columns.
"""
function df_to_matrix(df::DataFrame)
    numeric_cols = _numeric_column_names(df)
    isempty(numeric_cols) && error("no numeric columns found in data")
    mat = Matrix{Float64}(df[!, numeric_cols])
    return mat
end

"""
    variable_names(df) → Vector{String}

Extract numeric column names from a DataFrame.
"""
variable_names(df::DataFrame) = _numeric_column_names(df)

"""
    output_result(result, varnames; format, output, title)

Route output to table (terminal), CSV, or JSON based on `format`.
- `result`: a Matrix or DataFrame
- `varnames`: column names
- `format`: :table, :csv, or :json
- `output`: file path (empty string = stdout)
- `title`: table title for terminal display
"""
function output_result(result::AbstractMatrix, varnames::Vector{String};
                       format::String="table", output::String="", title::String="Results")
    fmt = Symbol(lowercase(format))
    df = DataFrame(result, varnames)
    output_result(df; format=fmt, output=output, title=title)
end

function output_result(df::DataFrame; format::Symbol=:table, output::String="", title::String="Results")
    if format == :csv
        _write_csv(df, output)
    elseif format == :json
        _write_json(df, output)
    else
        _write_table(df, output, title)
    end
end

"""
    output_kv(pairs; format, output, title)

Output key-value results (e.g., test statistics).
"""
function output_kv(pairs::Vector{<:Pair{String}}; format::String="table", output::String="", title::String="Results")
    fmt = Symbol(lowercase(format))
    if fmt == :json
        d = Dict(pairs)
        _write_json_raw(d, output)
    elseif fmt == :csv
        df = DataFrame(; metric=first.(pairs), value=last.(pairs))
        _write_csv(df, output)
    else
        df = DataFrame(; metric=first.(pairs), value=last.(pairs))
        _write_table(df, output, title)
    end
end

# Internal helpers

function _write_table(df::DataFrame, output::String, title::String)
    io = isempty(output) ? stdout : open(output, "w")
    try
        pretty_table(io, df;
            title=title,
            alignment=:c)
    finally
        isempty(output) || close(io)
    end
    isempty(output) || println("Results written to $output")
end

function _write_csv(df::DataFrame, output::String)
    if isempty(output)
        CSV.write(stdout, df)
    else
        CSV.write(output, df)
        println("Results written to $output")
    end
end

function _write_json(df::DataFrame, output::String)
    rows = [Dict(string(k) => v for (k, v) in zip(names(df), r)) for r in eachrow(df)]
    _write_json_raw(rows, output)
end

function _write_json_raw(data, output::String)
    json_str = JSON3.write(data)
    if isempty(output)
        println(json_str)
    else
        open(output, "w") do io
            write(io, json_str)
        end
        println("Results written to $output")
    end
end
