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

# Plot command: friedman plot <tag> — re-plot a stored model or result

function register_plot_commands!()
    return LeafCommand("plot", _plot;
        args=[Argument("tag"; description="Stored tag to plot (e.g. irf001, arch001)")],
        options=[
            Option("title"; short="t", type=String, default="", description="Custom plot title"),
            Option("save"; type=String, default="", description="Save plot to HTML file path"),
        ],
        flags=[Flag("no-open"; description="Don't open browser (use with --save)")],
        description="Plot a stored model or result")
end

function _plot(; tag::String, title::String="", save::String="", no_open::Bool=false)
    entry = storage_load(tag)
    isnothing(entry) && error("Tag '$tag' not found in storage. Run 'friedman list models' to see stored tags.")

    data = get(entry, "data", Dict{String,Any}())
    meta = get(entry, "meta", Dict{String,Any}())
    type_str = get(data, "_type", "")

    isempty(type_str) && error("Stored entry '$tag' has no type information — cannot plot. Re-run the command with a newer Friedman version.")

    obj = _deserialize_for_plot(data, type_str)

    kwargs = Dict{Symbol,Any}()
    if !isempty(title)
        kwargs[:title] = title
    end

    p = plot_result(obj; kwargs...)

    if !isempty(save)
        save_plot(p, save)
        printstyled("  Plot saved: $save\n"; color=:green)
    end
    if !no_open
        display_plot(p)
        printstyled("  Plot opened in browser\n"; color=:cyan)
    end

    if isempty(save) && no_open
        printstyled("  Warning: --no-open with no --save means nothing happened\n"; color=:yellow)
    end
end

# ── Deserialization Helpers ─────────────────────────────────

"""Convert stored array value to typed array, handling BSON type preservation."""
_to_vec(v) = Float64.(collect(v))
_to_mat(v) = Float64.(hcat(v...))  # fallback for column-stored
_to_mat(v::AbstractMatrix) = Float64.(v)
_to_mat(v::AbstractVector{<:AbstractVector}) = Float64.(hcat(v...))
_to_arr3(v::AbstractArray{<:Any,3}) = Float64.(v)
_to_arr3(v) = Float64.(Array(v))
_to_arr4(v::AbstractArray{<:Any,4}) = Float64.(v)
_to_arr4(v) = Float64.(Array(v))

"""Parse a UnitRange from stored representation (UnitRange, Vector, or string "13:100")."""
function _parse_range(v)
    v isa UnitRange && return v
    v isa AbstractVector && return Int(first(v)):Int(last(v))
    s = string(v)
    parts = split(s, ":")
    return parse(Int, parts[1]):parse(Int, parts[2])
end

"""Parse a Tuple{Int,Int,Int} from string representation."""
function _parse_int_tuple3(v)
    v isa Tuple && return v
    s = strip(string(v), ['(', ')', ' '])
    parts = split(s, ",")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])), parse(Int, strip(parts[3])))
end

# ── Main Deserializer Dispatch ───────────────────────────────

function _deserialize_for_plot(d::Dict, type_str::String)
    # Strip module prefix and type parameters: "MacroEconometricModels.ImpulseResponse{Float64}" → "ImpulseResponse"
    base = replace(type_str, r"^.*\." => "")
    base = replace(base, r"\{.*\}" => "")

    # IRF types
    base == "ImpulseResponse" && return _deser_irf(d)
    base == "BayesianImpulseResponse" && return _deser_birf(d)

    # FEVD types
    base == "FEVD" && return _deser_fevd(d)
    base == "BayesianFEVD" && return _deser_bfevd(d)
    base == "LPFEVD" && return _deser_lpfevd(d)

    # HD types
    base == "HistoricalDecomposition" && return _deser_hd(d)
    base == "BayesianHistoricalDecomposition" && return _deser_bhd(d)

    # Forecast types
    base == "ARIMAForecast" && return _deser_arima_fc(d)
    base == "VolatilityForecast" && return _deser_vol_fc(d)
    base == "VECMForecast" && return _deser_vecm_fc(d)
    base == "FactorForecast" && return _deser_factor_fc(d)
    base == "LPForecast" && return _deser_lp_fc(d)

    # Filter types
    base == "HPFilterResult" && return _deser_hp(d)
    base == "HamiltonFilterResult" && return _deser_hamilton(d)
    base == "BeveridgeNelsonResult" && return _deser_bn(d)
    base == "BaxterKingResult" && return _deser_bk(d)
    base == "BoostedHPResult" && return _deser_bhp(d)

    # Volatility model types
    base == "ARCHModel" && return _deser_arch(d)
    base == "GARCHModel" && return _deser_garch(d)
    base == "EGARCHModel" && return _deser_egarch(d)
    base == "GJRGARCHModel" && return _deser_gjr_garch(d)
    base == "SVModel" && return _deser_sv(d)

    # Factor model types
    base == "FactorModel" && return _deser_factor(d)
    base == "DynamicFactorModel" && return _deser_dynamic_factor(d)

    error("Type '$type_str' is not plottable via 'friedman plot'. Supported types: ImpulseResponse, BayesianImpulseResponse, FEVD, BayesianFEVD, LPFEVD, HistoricalDecomposition, BayesianHistoricalDecomposition, ARIMAForecast, VolatilityForecast, VECMForecast, FactorForecast, LPForecast, HPFilterResult, HamiltonFilterResult, BeveridgeNelsonResult, BaxterKingResult, BoostedHPResult, ARCHModel, GARCHModel, EGARCHModel, GJRGARCHModel, SVModel, FactorModel, DynamicFactorModel")
end

# ── Type-Specific Deserializers ──────────────────────────────

function _deser_irf(d)
    ImpulseResponse(
        _to_arr3(d["values"]),
        get(d, "ci_lower", nothing) === nothing ? zeros(0,0,0) : _to_arr3(d["ci_lower"]),
        get(d, "ci_upper", nothing) === nothing ? zeros(0,0,0) : _to_arr3(d["ci_upper"]),
        Int(d["horizon"]),
        String.(d["variables"]),
        String.(d["shocks"]),
        Symbol(d["ci_type"]))
end

function _deser_birf(d)
    BayesianImpulseResponse(
        _to_arr4(d["quantiles"]), _to_arr3(d["mean"]),
        Int(d["horizon"]), String.(d["variables"]), String.(d["shocks"]),
        _to_vec(d["quantile_levels"]))
end

function _deser_fevd(d)
    FEVD(_to_arr3(d["decomposition"]), _to_arr3(d["proportions"]))
end

function _deser_bfevd(d)
    BayesianFEVD(
        _to_arr4(d["quantiles"]), _to_arr3(d["mean"]),
        Int(d["horizon"]), String.(d["variables"]), String.(d["shocks"]),
        _to_vec(d["quantile_levels"]))
end

function _deser_lpfevd(d)
    LPFEVD(
        _to_arr3(d["proportions"]), _to_arr3(d["bias_corrected"]),
        _to_arr3(d["se"]), _to_arr3(d["ci_lower"]), _to_arr3(d["ci_upper"]),
        Symbol(d["method"]), Int(d["horizon"]), Int(d["n_boot"]),
        Float64(d["conf_level"]), Bool(d["bias_correction"]))
end

function _deser_hd(d)
    HistoricalDecomposition(
        _to_arr3(d["contributions"]), _to_mat(d["initial_conditions"]),
        _to_mat(d["actual"]), _to_mat(d["shocks"]),
        Int(d["T_eff"]), String.(d["variables"]), String.(d["shock_names"]),
        Symbol(d["method"]))
end

function _deser_bhd(d)
    BayesianHistoricalDecomposition(
        _to_arr4(d["quantiles"]), _to_arr3(d["mean"]),
        _to_arr3(d["initial_quantiles"]), _to_mat(d["initial_mean"]),
        _to_mat(d["shocks_mean"]), _to_mat(d["actual"]),
        Int(d["T_eff"]), String.(d["variables"]), String.(d["shock_names"]),
        _to_vec(d["quantile_levels"]), Symbol(d["method"]))
end

function _deser_arima_fc(d)
    ARIMAForecast(
        _to_vec(d["forecast"]), _to_vec(d["ci_lower"]),
        _to_vec(d["ci_upper"]), _to_vec(d["se"]),
        Int(d["horizon"]), Float64(d["conf_level"]))
end

function _deser_vol_fc(d)
    VolatilityForecast(
        _to_vec(d["forecast"]), _to_vec(d["ci_lower"]),
        _to_vec(d["ci_upper"]), _to_vec(d["se"]),
        Int(d["horizon"]), Float64(d["conf_level"]), Symbol(d["model_type"]))
end

function _deser_vecm_fc(d)
    VECMForecast(
        _to_mat(d["levels"]), _to_mat(d["differences"]),
        _to_mat(d["ci_lower"]), _to_mat(d["ci_upper"]),
        Int(d["horizon"]), Symbol(d["ci_method"]))
end

function _deser_factor_fc(d)
    FactorForecast(
        _to_mat(d["factors"]), _to_mat(d["observables"]),
        _to_mat(d["factors_lower"]), _to_mat(d["factors_upper"]),
        _to_mat(d["observables_lower"]), _to_mat(d["observables_upper"]),
        _to_mat(d["factors_se"]), _to_mat(d["observables_se"]),
        Int(d["horizon"]), Float64(d["conf_level"]), Symbol(d["ci_method"]))
end

function _deser_lp_fc(d)
    LPForecast(
        _to_mat(d["forecasts"]), _to_mat(d["ci_lower"]),
        _to_mat(d["ci_upper"]), _to_mat(d["se"]),
        Int(d["horizon"]), Int.(collect(d["response_vars"])),
        Int(d["shock_var"]), _to_vec(d["shock_path"]),
        Float64(d["conf_level"]), Symbol(d["ci_method"]))
end

function _deser_hp(d)
    HPFilterResult(_to_vec(d["trend"]), _to_vec(d["cycle"]),
        Float64(d["lambda"]), Int(d["T_obs"]))
end

function _deser_hamilton(d)
    HamiltonFilterResult(
        _to_vec(d["trend"]), _to_vec(d["cycle"]), _to_vec(d["beta"]),
        Int(d["h"]), Int(d["p"]), Int(d["T_obs"]),
        _parse_range(d["valid_range"]))
end

function _deser_bn(d)
    BeveridgeNelsonResult(
        _to_vec(d["permanent"]), _to_vec(d["transitory"]),
        Float64(d["drift"]), Float64(d["long_run_multiplier"]),
        _parse_int_tuple3(d["arima_order"]), Int(d["T_obs"]))
end

function _deser_bk(d)
    BaxterKingResult(
        _to_vec(d["cycle"]), _to_vec(d["trend"]), _to_vec(d["weights"]),
        Int(d["pl"]), Int(d["pu"]), Int(d["K"]), Int(d["T_obs"]),
        _parse_range(d["valid_range"]))
end

function _deser_bhp(d)
    BoostedHPResult(
        _to_vec(d["trend"]), _to_vec(d["cycle"]),
        Float64(d["lambda"]), Int(d["iterations"]), Symbol(d["stopping"]),
        _to_vec(d["bic_path"]), _to_vec(d["adf_pvalues"]), Int(d["T_obs"]))
end

function _deser_arch(d)
    ARCHModel(
        _to_vec(d["y"]), Int(d["q"]),
        Float64(d["mu"]), Float64(d["omega"]), _to_vec(d["alpha"]),
        _to_vec(d["conditional_variance"]), _to_vec(d["standardized_residuals"]),
        _to_vec(d["residuals"]), _to_vec(d["fitted"]),
        Float64(d["loglik"]), Float64(d["aic"]), Float64(d["bic"]),
        Symbol(d["method"]), Bool(d["converged"]), Int(d["iterations"]))
end

function _deser_garch(d)
    GARCHModel(
        _to_vec(d["y"]), Int(d["p"]), Int(d["q"]),
        Float64(d["mu"]), Float64(d["omega"]),
        _to_vec(d["alpha"]), _to_vec(d["beta"]),
        _to_vec(d["conditional_variance"]), _to_vec(d["standardized_residuals"]),
        _to_vec(d["residuals"]), _to_vec(d["fitted"]),
        Float64(d["loglik"]), Float64(d["aic"]), Float64(d["bic"]),
        Symbol(d["method"]), Bool(d["converged"]), Int(d["iterations"]))
end

function _deser_egarch(d)
    EGARCHModel(
        _to_vec(d["y"]), Int(d["p"]), Int(d["q"]),
        Float64(d["mu"]), Float64(d["omega"]),
        _to_vec(d["alpha"]), _to_vec(d["gamma"]), _to_vec(d["beta"]),
        _to_vec(d["conditional_variance"]), _to_vec(d["standardized_residuals"]),
        _to_vec(d["residuals"]), _to_vec(d["fitted"]),
        Float64(d["loglik"]), Float64(d["aic"]), Float64(d["bic"]),
        Symbol(d["method"]), Bool(d["converged"]), Int(d["iterations"]))
end

function _deser_gjr_garch(d)
    GJRGARCHModel(
        _to_vec(d["y"]), Int(d["p"]), Int(d["q"]),
        Float64(d["mu"]), Float64(d["omega"]),
        _to_vec(d["alpha"]), _to_vec(d["gamma"]), _to_vec(d["beta"]),
        _to_vec(d["conditional_variance"]), _to_vec(d["standardized_residuals"]),
        _to_vec(d["residuals"]), _to_vec(d["fitted"]),
        Float64(d["loglik"]), Float64(d["aic"]), Float64(d["bic"]),
        Symbol(d["method"]), Bool(d["converged"]), Int(d["iterations"]))
end

function _deser_sv(d)
    SVModel(
        _to_vec(d["y"]), _to_mat(d["h_draws"]),
        _to_vec(d["mu_post"]), _to_vec(d["phi_post"]), _to_vec(d["sigma_eta_post"]),
        _to_vec(d["volatility_mean"]), _to_mat(d["volatility_quantiles"]),
        _to_vec(d["quantile_levels"]),
        Symbol(d["dist"]), Bool(d["leverage"]), Int(d["n_samples"]))
end

function _deser_factor(d)
    FactorModel(
        _to_mat(d["X"]), _to_mat(d["factors"]), _to_mat(d["loadings"]),
        _to_vec(d["eigenvalues"]), _to_vec(d["explained_variance"]),
        _to_vec(d["cumulative_variance"]),
        Int(d["r"]), Bool(d["standardized"]))
end

function _deser_dynamic_factor(d)
    # A is a Vector{Matrix} — stored as array of arrays
    A_raw = d["A"]
    A = Matrix{Float64}[_to_mat(a) for a in A_raw]
    DynamicFactorModel(
        _to_mat(d["X"]), _to_mat(d["factors"]), _to_mat(d["loadings"]),
        A, _to_mat(d["factor_residuals"]),
        _to_mat(d["Sigma_eta"]), _to_mat(d["Sigma_e"]),
        _to_vec(d["eigenvalues"]), _to_vec(d["explained_variance"]),
        _to_vec(d["cumulative_variance"]),
        Int(d["r"]), Int(d["p"]), Symbol(d["method"]),
        Bool(d["standardized"]), Bool(d["converged"]),
        Int(d["iterations"]), Float64(d["loglik"]))
end
