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
    _validate_input_path(path)
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

"""
    get_dsge(config) → Dict

Extract DSGE model specification from a config dict.
Returns parameters, endogenous/exogenous variables, equations.
"""
function get_dsge(config::Dict)
    model = get(config, "model", Dict())
    result = Dict{String,Any}()

    result["parameters"] = get(model, "parameters", Dict{String,Any}())
    result["endogenous"] = get(model, "endogenous", String[])
    result["exogenous"] = get(model, "exogenous", String[])

    eqs_raw = get(model, "equations", Dict[])
    result["equations"] = String[eq["expr"] for eq in eqs_raw if haskey(eq, "expr")]

    # Optional solver section
    solver = get(config, "solver", Dict())
    result["solver_method"] = get(solver, "method", "gensys")
    result["solver_order"] = get(solver, "order", 1)
    result["solver_degree"] = get(solver, "degree", 5)
    result["solver_grid"] = get(solver, "grid", "auto")

    return result
end

"""
    get_dsge_constraints(config) → Dict

Extract DSGE constraint specifications (OccBin bounds, nonlinear).
"""
function get_dsge_constraints(config::Dict)
    con = get(config, "constraints", Dict())
    result = Dict{String,Any}()

    bounds_raw = get(con, "bounds", Dict[])
    bounds = Dict{String,Any}[]
    for b in bounds_raw
        bound = Dict{String,Any}("variable" => get(b, "variable", ""))
        if haskey(b, "lower")
            bound["lower"] = Float64(b["lower"])
        end
        if haskey(b, "upper")
            bound["upper"] = Float64(b["upper"])
        end
        push!(bounds, bound)
    end
    result["bounds"] = bounds

    return result
end

"""
    get_smm(config) → Dict

Extract SMM specification from a config dict.
"""
function get_smm(config::Dict)
    smm = get(config, "smm", Dict())
    Dict{String,Any}(
        "weighting" => get(smm, "weighting", "two_step"),
        "sim_ratio" => get(smm, "sim_ratio", 5),
        "burn"      => get(smm, "burn", 100),
    )
end

"""
    get_dsge_priors(config) → Dict{String,Any}

Parse Bayesian DSGE prior specification from [priors] TOML section.
Each parameter maps to {dist, a, b} (distribution name + 2 shape params).
"""
function get_dsge_priors(config::Dict)
    priors_raw = get(config, "priors", Dict())
    isempty(priors_raw) && error("TOML must have [priors] section with parameter distributions")
    result = Dict{String,Any}()
    for (param, spec) in priors_raw
        spec isa Dict || error("prior for '$param' must be a table with dist, a, b keys")
        haskey(spec, "dist") || error("prior for '$param' missing 'dist' key")
        result[param] = Dict{String,Any}(
            "dist" => spec["dist"],
            "a"    => get(spec, "a", 0.0),
            "b"    => get(spec, "b", 1.0),
        )
    end
    return result
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
