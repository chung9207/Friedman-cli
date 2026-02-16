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

# Global settings: ~/.friedman/ directory management

"""
    friedman_home() -> String

Return the Friedman home directory, defaulting to ~/.friedman/.
Respects FRIEDMAN_HOME environment variable.
"""
function friedman_home()
    get(ENV, "FRIEDMAN_HOME", joinpath(homedir(), ".friedman"))
end

"""
    ensure_friedman_home!()

Create the Friedman home directory if it doesn't exist.
"""
function ensure_friedman_home!()
    dir = friedman_home()
    isdir(dir) || mkpath(dir)
    return dir
end

"""
    get_username() -> String

Generate a username from system user + 4 hex chars.
"""
function get_username()
    user = get(ENV, "USER", get(ENV, "USERNAME", "user"))
    suffix = string(hash(string(Dates.now())), base=16)[end-3:end]
    return "$(user)$(suffix)"
end

"""
    load_settings() -> Dict{String,Any}

Load global settings from ~/.friedman/settings.json.
Returns empty dict if file doesn't exist.
"""
function load_settings()
    dir = friedman_home()
    path = joinpath(dir, "settings.json")
    isfile(path) || return Dict{String,Any}()
    try
        json_str = read(path, String)
        return Dict{String,Any}(JSON3.read(json_str, Dict{String,Any}))
    catch
        return Dict{String,Any}()
    end
end

"""
    save_settings!(settings::Dict{String,Any})

Save global settings to ~/.friedman/settings.json.
"""
function save_settings!(settings::Dict{String,Any})
    dir = ensure_friedman_home!()
    path = joinpath(dir, "settings.json")
    open(path, "w") do io
        write(io, JSON3.write(settings))
    end
end

"""
    init_settings!()

Initialize global settings on first run if they don't exist.
"""
function init_settings!()
    dir = friedman_home()
    path = joinpath(dir, "settings.json")
    if !isfile(path)
        ensure_friedman_home!()
        settings = Dict{String,Any}(
            "username" => get_username(),
            "created" => string(Dates.now()),
        )
        save_settings!(settings)
        return settings
    end
    return load_settings()
end

"""
    load_projects() -> Vector{Dict{String,Any}}

Load project registry from ~/.friedman/projects.json.
"""
function load_projects()
    dir = friedman_home()
    path = joinpath(dir, "projects.json")
    isfile(path) || return Dict{String,Any}[]
    try
        json_str = read(path, String)
        return Vector{Dict{String,Any}}(JSON3.read(json_str, Vector{Dict{String,Any}}))
    catch
        return Dict{String,Any}[]
    end
end

"""
    save_projects!(projects::Vector{Dict{String,Any}})

Save project registry to ~/.friedman/projects.json.
"""
function save_projects!(projects::Vector{Dict{String,Any}})
    dir = ensure_friedman_home!()
    path = joinpath(dir, "projects.json")
    open(path, "w") do io
        write(io, JSON3.write(projects))
    end
end

"""
    register_project!(name::String, path::String)

Register a project in the global registry (deduplicates by path).
"""
function register_project!(name::String, dir_path::String)
    projects = load_projects()
    # Deduplicate by path
    idx = findfirst(p -> p["path"] == dir_path, projects)
    if isnothing(idx)
        push!(projects, Dict{String,Any}("name" => name, "path" => dir_path))
        save_projects!(projects)
    end
end
