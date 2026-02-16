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

# BSON-based model/result storage with auto-tagging
# Stores to .friedmanlog.bson in the current working directory.

const STORAGE_FILE = ".friedmanlog.bson"

"""
    _storage_path() -> String

Return the path to the storage file in the current directory.
"""
_storage_path() = joinpath(pwd(), STORAGE_FILE)

"""
    _load_storage() -> Dict{String,Any}

Load the storage file, or return an empty dict if it doesn't exist.
"""
function _load_storage()
    path = _storage_path()
    isfile(path) || return Dict{String,Any}("entries" => Dict{String,Any}(), "counters" => Dict{String,Int}())
    try
        return BSON.load(path)
    catch
        return Dict{String,Any}("entries" => Dict{String,Any}(), "counters" => Dict{String,Int}())
    end
end

"""
    _save_storage!(store::Dict)

Atomically save the storage file (write to temp, then rename).
"""
function _save_storage!(store::Dict)
    path = _storage_path()
    tmp = path * ".tmp." * string(rand(UInt32), base=16)
    try
        BSON.bson(tmp, store)
        mv(tmp, path; force=true)
    catch e
        isfile(tmp) && rm(tmp; force=true)
        rethrow(e)
    end
end

"""
    auto_tag(type_prefix::String) -> String

Generate the next auto-incrementing tag for a given type prefix.
E.g., "var" -> "var001", "var002", etc.
"""
function auto_tag(type_prefix::String)
    store = _load_storage()
    counters = get(store, "counters", Dict{String,Int}())
    count = get(counters, type_prefix, 0) + 1
    return type_prefix * lpad(count, 3, '0')
end

"""
    serialize_model(model) -> Dict{String,Any}

Convert a model to a serializable dict of primitives.
"""
function serialize_model(model)
    d = Dict{String,Any}("_type" => string(typeof(model)))
    for fname in fieldnames(typeof(model))
        val = getfield(model, fname)
        d[string(fname)] = _serialize_value(val)
    end
    return d
end

function _serialize_value(v::AbstractArray)
    return collect(v)
end
function _serialize_value(v::Number)
    return v
end
function _serialize_value(v::String)
    return v
end
function _serialize_value(v::Symbol)
    return string(v)
end
function _serialize_value(v::Nothing)
    return nothing
end
function _serialize_value(v::Dict)
    return Dict(string(k) => _serialize_value(val) for (k, val) in v)
end
function _serialize_value(v::NamedTuple)
    return Dict(string(k) => _serialize_value(v[k]) for k in keys(v))
end
function _serialize_value(v)
    # Fallback: lossy — store as string
    @warn "Lossy serialization: $(typeof(v)) stored as string" maxlog=1
    return string(v)
end

"""
    storage_save!(tag::String, type_prefix::String, data::Dict{String,Any}, meta::Dict{String,Any}=Dict{String,Any}())

Save a model or result to storage under the given tag.
"""
function storage_save!(tag::String, type_prefix::String, data::Dict{String,Any},
                       meta::Dict{String,Any}=Dict{String,Any}())
    store = _load_storage()
    entries = get!(store, "entries", Dict{String,Any}())
    counters = get!(store, "counters", Dict{String,Int}())

    # Update counter
    count = get(counters, type_prefix, 0)
    tag_num = tryparse(Int, tag[length(type_prefix)+1:end])
    if !isnothing(tag_num) && tag_num > count
        counters[type_prefix] = tag_num
    end

    entry = Dict{String,Any}(
        "tag" => tag,
        "type" => type_prefix,
        "timestamp" => string(Dates.now()),
        "data" => data,
        "meta" => meta,
    )
    entries[tag] = entry
    store["entries"] = entries
    store["counters"] = counters

    _save_storage!(store)

    # Auto-register project on first save
    if length(entries) == 1
        register_project!(basename(pwd()), pwd())
    end

    return tag
end

"""
    storage_save_auto!(type_prefix::String, data::Dict{String,Any}, meta::Dict{String,Any}=Dict{String,Any}()) -> String

Save with auto-generated tag. Returns the tag.
"""
function storage_save_auto!(type_prefix::String, data::Dict{String,Any},
                            meta::Dict{String,Any}=Dict{String,Any}())
    tag = auto_tag(type_prefix)
    storage_save!(tag, type_prefix, data, meta)
    printstyled("  Saved as: $tag\n"; color=:cyan)
    return tag
end

"""
    storage_load(tag::String) -> Dict{String,Any} or nothing

Load a stored entry by tag.
"""
function storage_load(tag::String)
    store = _load_storage()
    entries = get(store, "entries", Dict{String,Any}())
    return get(entries, tag, nothing)
end

"""
    storage_list(; type_filter::String="") -> Vector{Dict{String,Any}}

List all stored entries, optionally filtered by type prefix.
"""
function storage_list(; type_filter::String="")
    store = _load_storage()
    entries = get(store, "entries", Dict{String,Any}())
    result = Dict{String,Any}[]
    for (tag, entry) in entries
        if isempty(type_filter) || get(entry, "type", "") == type_filter
            push!(result, entry)
        end
    end
    sort!(result; by=e -> get(e, "timestamp", ""))
    return result
end

"""
    storage_rename!(old_tag::String, new_tag::String) -> Bool

Rename a stored entry. Returns true if successful.
"""
function storage_rename!(old_tag::String, new_tag::String)
    store = _load_storage()
    entries = get(store, "entries", Dict{String,Any}())
    !haskey(entries, old_tag) && return false
    haskey(entries, new_tag) && error("tag '$new_tag' already exists")
    entry = entries[old_tag]
    entry["tag"] = new_tag
    entries[new_tag] = entry
    delete!(entries, old_tag)
    store["entries"] = entries
    _save_storage!(store)
    return true
end

"""
    resolve_stored_tags(args::Vector{String}) -> Vector{String}

Pre-dispatch hook: if args[2] looks like a stored tag (e.g., "var001"),
rewrite args to pass --from-tag option to the appropriate handler.
Only applies to irf/fevd/hd/forecast/predict/residuals commands.
"""
function resolve_stored_tags(args::Vector{String})
    length(args) < 2 && return args
    cmd = args[1]
    cmd in ("irf", "fevd", "hd", "forecast", "predict", "residuals") || return args

    potential_tag = args[2]
    # Check if it looks like a tag pattern (letters + digits)
    m = match(r"^([a-z]+)(\d{3,})$", potential_tag)
    isnothing(m) && return args

    # Check if it exists in storage
    entry = storage_load(potential_tag)
    isnothing(entry) && return args

    # Determine model type from the tag prefix
    model_type = get(entry, "type", m.captures[1])

    # Rewrite: ["irf", "var001", ...] → ["irf", model_type, "--from-tag=var001", ...]
    new_args = [cmd, model_type, "--from-tag=$potential_tag"]
    if length(args) > 2
        append!(new_args, args[3:end])
    end
    return new_args
end
