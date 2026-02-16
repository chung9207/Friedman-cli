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

# TOML config file parsing for complex model specifications

"""
    load_config(path) → Dict

Load and validate a TOML configuration file.
"""
function load_config(path::String)
    isfile(path) || error("config file not found: $path")
    try
        return TOML.parsefile(path)
    catch e
        error("failed to parse config file '$path': $(sprint(showerror, e))")
    end
end

"""
    get_identification(config) → Dict

Extract identification settings from a config dict.
Returns method, sign_matrix, narrative constraints, etc.
"""
function get_identification(config::Dict)
    id_cfg = get(config, "identification", Dict())
    method = get(id_cfg, "method", "cholesky")
    result = Dict{String,Any}("method" => method)

    if method == "sign"
        sm = get(id_cfg, "sign_matrix", Dict())
        if haskey(sm, "matrix")
            result["sign_matrix"] = _parse_matrix(sm["matrix"])
        end
        if haskey(sm, "horizons")
            result["horizons"] = Int.(sm["horizons"])
        end
    end

    if method == "narrative" || haskey(id_cfg, "narrative")
        narr = get(id_cfg, "narrative", Dict())
        result["narrative"] = Dict{String,Any}(
            "shock_index" => get(narr, "shock_index", 1),
            "periods" => get(narr, "periods", Int[]),
            "signs" => get(narr, "signs", Int[])
        )
    end

    return result
end

"""
    get_prior(config) → Dict

Extract Bayesian prior settings from a config dict.
"""
function get_prior(config::Dict)
    pr = get(config, "prior", Dict())
    prior_type = get(pr, "type", "minnesota")
    result = Dict{String,Any}("type" => prior_type)

    hyper = get(pr, "hyperparameters", Dict())
    if !isempty(hyper)
        result["lambda1"] = get(hyper, "lambda1", 0.2)
        result["lambda2"] = get(hyper, "lambda2", 0.5)
        result["lambda3"] = get(hyper, "lambda3", 1.0)
        result["lambda4"] = get(hyper, "lambda4", 1e5)
    end

    opt = get(pr, "optimization", Dict())
    result["optimize"] = get(opt, "enabled", false)

    return result
end

"""
    get_gmm(config) → Dict

Extract GMM specification from a config dict.
"""
function get_gmm(config::Dict)
    gmm = get(config, "gmm", Dict())
    result = Dict{String,Any}()

    result["moment_conditions"] = get(gmm, "moment_conditions", String[])
    result["instruments"] = get(gmm, "instruments", String[])
    result["weighting"] = get(gmm, "weighting", "twostep")

    return result
end

"""
    get_nongaussian(config) → Dict

Extract non-Gaussian SVAR settings from a config dict.
"""
function get_nongaussian(config::Dict)
    ng = get(config, "nongaussian", Dict())
    result = Dict{String,Any}()

    result["method"] = get(ng, "method", "fastica")
    result["contrast"] = get(ng, "contrast", "logcosh")
    result["distribution"] = get(ng, "distribution", "student_t")
    result["n_regimes"] = get(ng, "n_regimes", 2)
    result["transition_variable"] = get(ng, "transition_variable", "")
    result["regime_variable"] = get(ng, "regime_variable", "")

    return result
end

"""
    get_uhlig_params(config) → Dict

Extract Uhlig SVAR identification tuning parameters from a config dict.
"""
function get_uhlig_params(config::Dict)
    id_cfg = get(config, "identification", Dict())
    uhlig = get(id_cfg, "uhlig", Dict())
    Dict{String,Any}(
        "n_starts"        => get(uhlig, "n_starts", 50),
        "n_refine"        => get(uhlig, "n_refine", 10),
        "max_iter_coarse" => get(uhlig, "max_iter_coarse", 500),
        "max_iter_fine"   => get(uhlig, "max_iter_fine", 2000),
        "tol_coarse"      => get(uhlig, "tol_coarse", 1e-4),
        "tol_fine"        => get(uhlig, "tol_fine", 1e-8),
    )
end

# Internal helpers

function _parse_matrix(rows::Vector)
    n = length(rows)
    n == 0 && return Matrix{Float64}(undef, 0, 0)
    m = length(rows[1])
    for i in 2:n
        length(rows[i]) == m || error("matrix row $i has $(length(rows[i])) elements, expected $m")
    end
    mat = Matrix{Float64}(undef, n, m)
    for i in 1:n
        for j in 1:m
            mat[i, j] = Float64(rows[i][j])
        end
    end
    return mat
end
