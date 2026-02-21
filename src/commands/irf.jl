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

# IRF commands: var, bvar, lp (action-first: friedman irf var ...)

function register_irf_commands!()
    irf_var = LeafCommand("var", _irf_var;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=nothing, description="Lag order (default: auto)"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun|arias|uhlig|fastica|jade|sobi|dcov|hsic|student_t|mixture_normal|pml|skew_normal|markov_switching|garch_id"),
            Option("ci"; type=String, default="bootstrap", description="none|bootstrap|theoretical"),
            Option("replications"; type=Int, default=1000, description="Bootstrap replications"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[
            Flag("plot"; description="Open interactive plot in browser"),
            Flag("cumulative"; description="Compute cumulative IRFs (for differenced data)"),
            Flag("identified-set"; description="Return full identified set for sign restrictions"),
            Flag("stationary-only"; description="Filter non-stationary bootstrap draws"),
        ],
        description="Compute frequentist impulse response functions")

    irf_bvar = LeafCommand("bvar", _irf_bvar;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=4, description="Lag order"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("draws"; short="n", type=Int, default=2000, description="MCMC draws"),
            Option("sampler"; type=String, default="direct", description="direct|gibbs"),
            Option("config"; type=String, default="", description="TOML config for identification/prior"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[
            Flag("plot"; description="Open interactive plot in browser"),
            Flag("cumulative"; description="Compute cumulative IRFs (for differenced data)"),
        ],
        description="Compute Bayesian impulse response functions with credible intervals")

    irf_lp = LeafCommand("lp", _irf_lp;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("shock"; type=Int, default=1, description="Single shock index (1-based)"),
            Option("shocks"; type=String, default="", description="Comma-separated shock indices (e.g. 1,2,3)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("lags"; short="p", type=Int, default=4, description="LP control lags"),
            Option("var-lags"; type=Int, default=nothing, description="VAR lag order for identification (default: same as --lags)"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("ci"; type=String, default="none", description="none|bootstrap"),
            Option("replications"; type=Int, default=200, description="Bootstrap replications"),
            Option("conf-level"; type=Float64, default=0.95, description="Confidence level"),
            Option("vcov"; type=String, default="newey_west", description="newey_west|white|driscoll_kraay"),
            Option("config"; type=String, default="", description="TOML config for sign/narrative restrictions"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[
            Flag("plot"; description="Open interactive plot in browser"),
            Flag("cumulative"; description="Compute cumulative IRFs (for differenced data)"),
        ],
        description="Compute structural LP impulse response functions")

    irf_vecm = LeafCommand("vecm", _irf_vecm;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("lags"; short="p", type=Int, default=2, description="Lag order (in levels)"),
            Option("rank"; short="r", type=String, default="auto", description="Cointegration rank (auto|1|2|...)"),
            Option("deterministic"; type=String, default="constant", description="none|constant|trend"),
            Option("shock"; type=Int, default=1, description="Shock variable index (1-based)"),
            Option("horizons"; short="h", type=Int, default=20, description="IRF horizon"),
            Option("id"; type=String, default="cholesky", description="cholesky|sign|narrative|longrun"),
            Option("ci"; type=String, default="bootstrap", description="none|bootstrap|theoretical"),
            Option("replications"; type=Int, default=1000, description="Bootstrap replications"),
            Option("config"; type=String, default="", description="TOML config for identification"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Compute impulse response functions via VECM → VAR representation")

    irf_pvar = LeafCommand("pvar", _irf_pvar;
        args=[Argument("data"; description="Path to CSV panel data file")],
        options=[
            Option("id-col"; type=String, default="", description="Panel group identifier column"),
            Option("time-col"; type=String, default="", description="Time period column"),
            Option("lags"; short="p", type=Int, default=1, description="Lag order"),
            Option("horizons"; short="h", type=Int, default=10, description="IRF horizon"),
            Option("irf-type"; type=String, default="oirf", description="oirf|girf"),
            Option("boot-draws"; type=Int, default=500, description="Bootstrap draws for CIs"),
            Option("confidence"; type=Float64, default=0.95, description="Confidence level"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Compute Panel VAR impulse response functions (OIRF/GIRF)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "var"  => irf_var,
        "bvar" => irf_bvar,
        "lp"   => irf_lp,
        "vecm" => irf_vecm,
        "pvar" => irf_pvar,
    )
    return NodeCommand("irf", subcmds, "Impulse Response Functions")
end

# ── VAR IRF ──────────────────────────────────────────────

function _irf_var(; data::String, lags=nothing, shock::Int=1, horizons::Int=20,
                   id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                   config::String="",
                   output::String="", format::String="table",
                   plot::Bool=false, plot_save::String="",
                   cumulative::Bool=false, identified_set::Bool=false,
                   stationary_only::Bool=false)
    model, Y, varnames, p = _load_and_estimate_var(data, lags)
    n = size(Y, 2)

    println("Computing IRFs: VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    println()

    # Arias identification handled separately
    if id == "arias"
        _var_irf_arias(model, config, horizons, varnames, shock; format=format, output=output)
        return
    end

    # Uhlig identification handled separately
    if id == "uhlig"
        _var_irf_uhlig(model, config, horizons, varnames, shock; format=format, output=output)
        return
    end

    # Sign-identified set: return full draw set instead of point estimates
    if identified_set && id == "sign"
        check_func, _ = _build_check_func(config)
        isnothing(check_func) && error("--identified-set requires a --config file with sign restrictions")
        set = identify_sign(model, horizons, check_func; max_draws=replications, store_all=true)
        lower, upper = irf_bounds(set)
        med = irf_median(set)
        println("Sign-Identified Set: $(set.n_accepted)/$(set.n_total) accepted ($(round(set.acceptance_rate*100; digits=1))%)")
        irf_df = DataFrame()
        irf_df.horizon = 0:horizons
        for (vi, vname) in enumerate(varnames)
            irf_df[!, vname] = med[:, vi, shock]
            irf_df[!, "$(vname)_lower"] = lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = upper[:, vi, shock]
        end
        shock_name = _shock_name(varnames, shock)
        output_result(irf_df; format=Symbol(format), output=output,
                      title="IRF Identified Set (sign, $shock_name shock)")
        return
    end

    kwargs = _build_identification_kwargs(id, config)
    kwargs[:ci_type] = Symbol(ci)
    kwargs[:reps] = replications
    if stationary_only
        kwargs[:stationary_only] = true
    end

    irf_result = irf(model, horizons; kwargs...)

    if cumulative
        irf_result = cumulative_irf(irf_result)
        printstyled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    report(irf_result)

    irf_vals = irf_result.values  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    if ci != "none" && !isnothing(irf_result.ci_lower)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock]
        end
    end

    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock ($id identification)")
end

function _var_irf_arias(model, config::String, horizons::Int,
                        varnames::Vector{String}, shock::Int; format::String="table", output::String="")
    isempty(config) && error("Arias identification requires a --config file with restrictions")
    cfg = load_config(config)
    id_cfg = get(cfg, "identification", Dict())

    zeros_list = get(id_cfg, "zero_restrictions", [])
    signs_list = get(id_cfg, "sign_restrictions", [])

    n = nvars(model)
    zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
    sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]

    restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)
    result = identify_arias(model, restrictions, horizons)

    irf_vals = irf_mean(result)  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Arias et al. identification)")
end

function _var_irf_uhlig(model, config::String, horizons::Int,
                        varnames::Vector{String}, shock::Int; format::String="table", output::String="")
    isempty(config) && error("Uhlig identification requires a --config file with restrictions")
    cfg = load_config(config)
    id_cfg = get(cfg, "identification", Dict())

    zeros_list = get(id_cfg, "zero_restrictions", [])
    signs_list = get(id_cfg, "sign_restrictions", [])

    n = nvars(model)
    zero_restrs = [zero_restriction(r["var"], r["shock"]; horizon=r["horizon"]) for r in zeros_list]
    sign_restrs = [sign_restriction(r["var"], r["shock"], Symbol(r["sign"]); horizon=r["horizon"]) for r in signs_list]

    restrictions = SVARRestrictions(n; zeros=zero_restrs, signs=sign_restrs)

    uhlig_params = get_uhlig_params(cfg)
    result = identify_uhlig(model, restrictions, horizons;
        n_starts=uhlig_params["n_starts"], n_refine=uhlig_params["n_refine"],
        max_iter_coarse=uhlig_params["max_iter_coarse"], max_iter_fine=uhlig_params["max_iter_fine"],
        tol_coarse=uhlig_params["tol_coarse"], tol_fine=uhlig_params["tol_fine"])

    # Convergence info
    println("Uhlig identification: penalty=$(round(result.penalty; digits=6)), converged=$(result.converged)")
    for (si, sp) in enumerate(result.shock_penalties)
        println("  Shock $si penalty: $(round(sp; digits=6))")
    end
    println()

    irf_vals = result.irf  # H x n x n
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="IRF to $shock_name shock (Uhlig identification)")
end

# ── BVAR IRF ─────────────────────────────────────────────

function _irf_bvar(; data::String, lags::Int=4, shock::Int=1, horizons::Int=20,
                    id::String="cholesky", draws::Int=2000, sampler::String="direct",
                    config::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="",
                    cumulative::Bool=false)
    post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags, config, draws, sampler)
    method = get(ID_METHOD_MAP, id, :cholesky)

    println("Computing Bayesian IRFs: BVAR($p), shock=$shock, horizons=$horizons, id=$id")
    println("  Sampler: $sampler, Draws: $draws")
    println()

    check_func, narrative_check = _build_check_func(config)

    kwargs = Dict{Symbol,Any}(
        :method => method,
        :quantiles => [0.16, 0.5, 0.84],
    )
    if !isnothing(check_func)
        kwargs[:check_func] = check_func
    end
    if !isnothing(narrative_check)
        kwargs[:narrative_check] = narrative_check
    end

    birf = irf(post, horizons; kwargs...)

    if cumulative
        birf = cumulative_irf(birf)
        printstyled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(birf; plot=plot, plot_save=plot_save)

    report(birf)

    irf_mean_vals = birf.mean
    n_h = size(irf_mean_vals, 1)
    q_levels = birf.quantile_levels
    q_idx_lo = findfirst(==(0.16), q_levels)
    q_idx_med = findfirst(==(0.5), q_levels)
    q_idx_hi = findfirst(==(0.84), q_levels)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)

    for (vi, vname) in enumerate(varnames)
        if !isnothing(q_idx_med)
            irf_df[!, vname] = birf.quantiles[:, vi, shock, q_idx_med]
        else
            irf_df[!, vname] = irf_mean_vals[:, vi, shock]
        end
        if !isnothing(q_idx_lo)
            irf_df[!, "$(vname)_16pct"] = birf.quantiles[:, vi, shock, q_idx_lo]
        end
        if !isnothing(q_idx_hi)
            irf_df[!, "$(vname)_84pct"] = birf.quantiles[:, vi, shock, q_idx_hi]
        end
    end

    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="Bayesian IRF to $shock_name shock ($id, 68% credible interval)")
end

# ── LP IRF ───────────────────────────────────────────────

function _irf_lp(; data::String, shock::Int=1, shocks::String="",
                  horizons::Int=20, lags::Int=4, var_lags=nothing,
                  id::String="cholesky", ci::String="none",
                  replications::Int=200, conf_level::Float64=0.95,
                  vcov::String="newey_west", config::String="",
                  output::String="", format::String="table",
                  plot::Bool=false, plot_save::String="",
                  cumulative::Bool=false)
    # Multi-shock mode
    if !isempty(shocks)
        shock_indices = parse.(Int, split(shocks, ","))
    else
        shock_indices = [shock]
    end

    slp, Y, varnames = _load_and_structural_lp(data, horizons, lags, var_lags,
        id, vcov, config; ci_type=Symbol(ci), reps=replications, conf_level=conf_level)

    n = size(Y, 2)
    irf_result = slp.irf

    if cumulative
        irf_result = cumulative_irf(irf_result)
        printstyled("  Cumulative IRFs computed\n"; color=:cyan)
    end

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    println("Computing LP IRFs: horizons=$horizons, id=$id, ci=$ci")
    println()

    for shock_idx in shock_indices
        (shock_idx < 1 || shock_idx > n) && error("shock index $shock_idx out of range (data has $n variables)")
        shock_name = _shock_name(varnames, shock_idx)

        irf_vals = irf_result.values  # H x n x n
        n_h = size(irf_vals, 1)

        irf_df = DataFrame()
        irf_df.horizon = 0:(n_h-1)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, vname] = irf_vals[:, vi, shock_idx]
        end

        if ci != "none" && !isnothing(irf_result.ci_lower)
            for (vi, vname) in enumerate(varnames)
                irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock_idx]
                irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock_idx]
            end
        end

        output_result(irf_df; format=Symbol(format),
                      output=isempty(output) ? "" : (length(shock_indices) > 1 ? replace(output, "." => "_$(shock_name).") : output),
                      title="LP IRF to $shock_name shock ($id identification)")
        println()
    end
end

# ── VECM IRF ────────────────────────────────────────────

function _irf_vecm(; data::String, lags::Int=2, rank::String="auto",
                    deterministic::String="constant",
                    shock::Int=1, horizons::Int=20,
                    id::String="cholesky", ci::String="bootstrap", replications::Int=1000,
                    config::String="",
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="")
    vecm, Y, varnames, p = _load_and_estimate_vecm(data, lags, rank, deterministic, "johansen", 0.05)
    var_model = to_var(vecm)
    n = size(Y, 2)
    r = cointegrating_rank(vecm)

    println("Computing VECM IRFs: rank=$r, VAR($p), shock=$shock, horizons=$horizons, id=$id, ci=$ci")
    println()

    kwargs = _build_identification_kwargs(id, config)
    kwargs[:ci_type] = Symbol(ci)
    kwargs[:reps] = replications

    irf_result = irf(var_model, horizons; kwargs...)

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    report(irf_result)

    irf_vals = irf_result.values
    n_h = size(irf_vals, 1)

    irf_df = DataFrame()
    irf_df.horizon = 0:(n_h-1)
    for (vi, vname) in enumerate(varnames)
        irf_df[!, vname] = irf_vals[:, vi, shock]
    end

    if ci != "none" && !isnothing(irf_result.ci_lower)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock]
            irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock]
        end
    end

    shock_name = _shock_name(varnames, shock)
    output_result(irf_df; format=Symbol(format), output=output,
                  title="VECM IRF to $shock_name shock ($id identification)")
end

# ── Panel VAR IRF ──────────────────────────────────────────

function _irf_pvar(; data::String, id_col::String="", time_col::String="",
                    lags::Int=1, horizons::Int=10,
                    irf_type::String="oirf", boot_draws::Int=500,
                    confidence::Float64=0.95,
                    output::String="", format::String="table",
                    plot::Bool=false, plot_save::String="")
    isempty(id_col) && error("Panel VAR IRF requires --id-col")
    isempty(time_col) && error("Panel VAR IRF requires --time-col")
    validate_method(irf_type, ["oirf", "girf"], "IRF type")

    model, panel, varnames = _load_and_estimate_pvar(data, id_col, time_col, lags)
    n = length(varnames)

    println("Computing Panel VAR IRFs: type=$irf_type, horizons=$horizons, bootstrap=$boot_draws")
    println()

    # Compute IRFs with bootstrap CIs
    irf_result = pvar_bootstrap_irf(model, horizons;
        n_boot=boot_draws, conf_level=confidence, irf_type=Symbol(irf_type))

    _maybe_plot(irf_result; plot=plot, plot_save=plot_save)

    # Output per-shock IRF tables
    for shock in 1:n
        shock_name = _shock_name(varnames, shock)
        irf_vals = irf_result.values
        n_h = size(irf_vals, 1)

        irf_df = DataFrame()
        irf_df.horizon = 0:(n_h-1)
        for (vi, vname) in enumerate(varnames)
            irf_df[!, vname] = irf_vals[:, vi, shock]
        end
        if !isnothing(irf_result.ci_lower)
            for (vi, vname) in enumerate(varnames)
                irf_df[!, "$(vname)_lower"] = irf_result.ci_lower[:, vi, shock]
                irf_df[!, "$(vname)_upper"] = irf_result.ci_upper[:, vi, shock]
            end
        end

        output_result(irf_df; format=Symbol(format),
                      output=_per_var_output_path(output, shock_name),
                      title="Panel VAR $(uppercase(irf_type)) to $shock_name shock")
        println()
    end
end
