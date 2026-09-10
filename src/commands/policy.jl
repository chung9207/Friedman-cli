# ── policy: McKay–Wolf policy counterfactuals (W4/#126, MEMs 0.8.0 CF module) ──
#
# New top-level family (18 → 19). W4 lands the foundations: the causal-effects
# menu (`policy effects var|bvar|lp|sign`) and the rule-counterfactual engine
# (`policy counterfactual var|bvar|lp`). W5–W7 build on these.
#
# Design constraints inherited from the module (verified on the 0.8.0 tag):
# - None of the CF types is Tables.jl-registered or serializable → hand-built
#   tidy tables (the `io`-family precedent) and NO --save-model/--model handles.
#   Containers are re-derived per invocation; the global `Random.seed!` is the
#   reproducibility story (CF functions take `rng=`, never `seed=`).
# - The honesty numbers (rel_residual, spanned, error_path, n_draws_used/failed)
#   go in STDOUT DATA TABLES, not stderr — `report()` output is swallowed by
#   `_status_report`, and a thin menu that cannot enforce an H-period rule
#   exactly must be visible to a user reading only the envelope.
# - `outcomes`/`instruments` are `Pair{Symbol, Int|String}` maps (module symbol
#   → IRF variable). W7's `policy_news_matrix` uses `Pair{Symbol,Symbol}` — a
#   DIFFERENT shape; do not share this parser with it.
# - Square vs thin is the module's central axis (`is_square`): square = model
#   news menu (exact solve), thin = empirically identified shocks (least
#   squares). Surfaced in every effects/counterfactual summary.
# - plot_result exists for PolicyCounterfactual (recipe appends an
#   implementation-error panel past spanned_tol); PolicyCausalEffects has NO
#   recipe → `policy effects` carries no plot flags.

# ── Shared parsing ──────────────────────────────────────────

"""Parse `--outcomes infl=2,ygap=out` into the module's `Pair{Symbol,Int|String}`
maps (name → IRF variable index or column name)."""
function _parse_cf_pairs(spec::String, opt::String; required::Bool=true)
    out = Pair{Symbol,Union{Int,String}}[]
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        parts = split(tok, "=")
        (length(parts) == 2 && !isempty(strip(parts[1])) && !isempty(strip(parts[2]))) ||
            throw(CliError("usage/invalid",
                "policy: $opt entries must be name=column (got '$tok')";
                hint="e.g. $opt infl=2,ygap=1 (index) or $opt infl=cpi (column name)"))
        v = tryparse(Int, strip(parts[2]))
        push!(out, Symbol(strip(parts[1])) => (v === nothing ? String(strip(parts[2])) : v))
    end
    (required && isempty(out)) && throw(CliError("usage/missing",
        "policy: $opt is required";
        hint="map module names to IRF variables, e.g. $opt infl=2,ygap=1"))
    length(unique(first.(out))) == length(out) || throw(CliError("usage/invalid",
        "policy: $opt names must be distinct"))
    return out
end

"""Parse `--shocks 1,mp` into a vector of Int indices / String shock names."""
function _parse_cf_shocks(spec::String, opt::String="--shocks")
    isempty(strip(spec)) && throw(CliError("usage/missing",
        "policy: $opt is required (the identified POLICY shock columns of the IRF)";
        hint="indices or names, e.g. $opt 3 or $opt mp1,mp2"))
    out = Union{Int,String}[]
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        v = tryparse(Int, tok)
        push!(out, v === nothing ? String(tok) : v)
    end
    isempty(out) && throw(CliError("usage/missing", "policy: $opt is required"))
    return out
end

function _parse_cf_quantiles(spec::String)
    qs = Float64[]
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        v = tryparse(Float64, tok)
        (v === nothing || !(0.0 < v < 1.0)) && throw(CliError("usage/invalid",
            "policy: --quantiles must be numbers in (0, 1) (got '$tok')"))
        push!(qs, v)
    end
    isempty(qs) && throw(CliError("usage/invalid", "policy: --quantiles must not be empty"))
    issorted(qs) || throw(CliError("usage/invalid",
        "policy: --quantiles must be increasing (got $spec)"))
    return qs
end

# ── Rule construction ───────────────────────────────────────

"""Build the `PolicyRule` from `--rule <builtin>` or a `[rule]` TOML section.

Builtins use the CLI's `--outcomes`/`--instruments` symbols with upstream's
variable-name defaults (`:infl`/`:ygap`), so `--rule taylor` requires outcomes
actually NAMED infl/ygap — a friendly usage error points at the config route
otherwise. The TOML route controls everything, including the CMW-vs-textbook
Taylor coefficients (see `get_policy_rule`)."""
function _build_policy_rule(rule::String, config::String, H::Int,
                            cli_outcomes::Vector{Symbol},
                            cli_instruments::Vector{Symbol})
    if !isempty(config)
        isempty(rule) || throw(CliError("usage/invalid",
            "policy: give either --rule <builtin> or --config <toml>, not both"))
        spec = get_policy_rule(load_config(config))
        for s in spec.outcomes
            s in cli_outcomes || throw(CliError("config/invalid",
                "[rule] outcome '$s' is not among the --outcomes names ($(join(cli_outcomes, ", ")))";
                hint="the rule can only reference variables the causal-effects menu maps"))
        end
        for s in spec.instruments
            s in cli_instruments || throw(CliError("config/invalid",
                "[rule] instrument '$s' is not among the --instruments names ($(join(cli_instruments, ", ")))"))
        end
        p = spec.params
        return spec.type === :rate_peg ?
                   rate_peg_rule(H; outcomes=spec.outcomes, instruments=spec.instruments) :
               spec.type === :rate_target ?
                   (length(p.path) == H || throw(CliError("config/shape",
                        "[rule] path has $(length(p.path)) entries but --horizon is $H"));
                    rate_target_rule(H, p.path; outcomes=spec.outcomes,
                                     instruments=spec.instruments)) :
               spec.type === :inflation_target ?
                   inflation_target_rule(H; pi_var=p.pi_var, outcomes=spec.outcomes,
                                         instruments=spec.instruments) :
               spec.type === :output_gap ?
                   output_gap_rule(H; y_var=p.y_var, outcomes=spec.outcomes,
                                   instruments=spec.instruments) :
               spec.type === :ngdp ?
                   ngdp_rule(H; pi_var=p.pi_var, y_var=p.y_var, outcomes=spec.outcomes,
                             instruments=spec.instruments) :
               taylor_rule(H; rho=p.rho, phi_pi=p.phi_pi, phi_y=p.phi_y, z_lag=p.z_lag,
                           pi_var=p.pi_var, y_var=p.y_var, outcomes=spec.outcomes,
                           instruments=spec.instruments)
    end
    isempty(rule) && throw(CliError("usage/missing",
        "policy counterfactual: --rule <builtin> or --config <toml> is required";
        hint="builtins: rate-peg|inflation-target|output-gap|ngdp|taylor"))
    rule in ("rate-peg", "inflation-target", "output-gap", "ngdp", "taylor") ||
        throw(CliError("usage/invalid-option",
            "invalid --rule '$rule'; must be rate-peg, inflation-target, output-gap, ngdp or taylor";
            hint="rate-target (a fixed instrument path) needs --config (the path lives in the TOML)"))
    needs = rule == "inflation-target" ? [:infl] :
            rule == "output-gap" ? [:ygap] :
            rule in ("ngdp", "taylor") ? [:infl, :ygap] : Symbol[]
    for s in needs
        s in cli_outcomes || throw(CliError("usage/invalid",
            "--rule $rule expects an outcome named '$s' (builtins use upstream's variable-name defaults)";
            hint="name it in --outcomes (e.g. $s=2), or use --config to pick pi_var/y_var freely"))
    end
    length(cli_instruments) == 1 || rule in ("inflation-target", "output-gap", "ngdp") ||
        throw(CliError("usage/invalid",
            "--rule $rule needs exactly ONE instrument (got $(length(cli_instruments)))"))
    return rule == "rate-peg" ?
               rate_peg_rule(H; outcomes=cli_outcomes, instruments=cli_instruments) :
           rule == "inflation-target" ?
               inflation_target_rule(H; outcomes=cli_outcomes, instruments=cli_instruments) :
           rule == "output-gap" ?
               output_gap_rule(H; outcomes=cli_outcomes, instruments=cli_instruments) :
           rule == "ngdp" ?
               ngdp_rule(H; outcomes=cli_outcomes, instruments=cli_instruments) :
           taylor_rule(H; outcomes=cli_outcomes, instruments=cli_instruments)  # textbook defaults
end

# ── Loss construction (W5/#127) ─────────────────────────────

"""Build `(PolicyLoss, z_wedge)` from a `[loss]` TOML. The `smoothing_penalty`
split is owned HERE: its `.W_z` feeds `policy_loss(W_z=…)`, its `.wedge_term`
the engines' `z_wedge=` — handlers cannot get the split wrong."""
function _build_policy_loss(loss_config::String, H::Int,
                            cli_outcomes::Vector{Symbol},
                            cli_instruments::Vector{Symbol})
    isempty(loss_config) && throw(CliError("usage/missing",
        "policy: --loss-config <toml> is required (the quadratic loss; lambda has NO default upstream)";
        hint="e.g. [loss] outcomes = [\"infl\",\"ygap\"], lambda = [1.0, 0.5]"))
    spec = get_policy_loss(load_config(loss_config))
    sm = spec.smoothing
    W_z = nothing
    z_wedge = nothing
    loss_instruments = Symbol[]
    if sm !== nothing
        length(cli_instruments) == 1 || throw(CliError("usage/invalid",
            "policy: [loss.smoothing] penalizes ONE instrument; map exactly one in --instruments (got $(length(cli_instruments)))"))
        sp = smoothing_penalty(H; lambda=sm.lambda, beta=sm.beta, z_lag=sm.z_lag)
        W_z = [sp.W_z]
        z_wedge = [sp.wedge_term]
        loss_instruments = cli_instruments
    end
    if spec.type === :ait
        for s in (spec.pi_var, spec.y_var)
            s in cli_outcomes || throw(CliError("config/invalid",
                "[loss] ait variable '$s' is not among the --outcomes names ($(join(cli_outcomes, ", ")))"))
        end
        base = ait_loss(H; beta=spec.beta, lambda_avg=spec.lambda_avg,
                        lambda_t=spec.lambda_t, lambda_y=spec.lambda_y,
                        delta=spec.delta, K=spec.K,
                        pi_var=spec.pi_var, y_var=spec.y_var)
        # ait_loss carries no instrument penalty — rebuild with the smoothing block.
        loss = sm === nothing ? base :
               PolicyLoss(outcomes=base.outcomes, W_x=base.W_x,
                          instruments=loss_instruments, W_z=W_z,
                          lambda=base.lambda, beta=base.beta, name=base.name)
        return loss, z_wedge
    end
    for s in spec.outcomes
        s in cli_outcomes || throw(CliError("config/invalid",
            "[loss] outcome '$s' is not among the --outcomes names ($(join(cli_outcomes, ", ")))"))
    end
    loss = policy_loss(spec.outcomes, H; lambda=spec.lambda, beta=spec.beta,
                       instruments=loss_instruments, W_z=W_z)
    return loss, z_wedge
end

# ── Menu construction per source route ──────────────────────

"""Build `(ce, ir_or_bir)` for one source route. The IRF object is returned too
because `policy counterfactual` extracts the baseline from the SAME estimate."""
function _policy_menu(route::String; data::String, lags, horizon::Int, draws::Int,
                      replications::Int, n_draws::Int, config::String,
                      shocks, outcomes, instruments, normalize::Symbol)
    if route == "var"
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
        kw = Dict{Symbol,Any}()
        if replications > 0
            kw[:ci_type] = :bootstrap
            kw[:reps] = replications
        end
        ir = irf(model, horizon; kw...)
        ce = policy_causal_effects(ir, shocks, outcomes, instruments;
                                   H=horizon, normalize=normalize)
        return ce, ir, model
    elseif route == "bvar"
        post, Y, varnames, p, n = _load_and_estimate_bvar(data, lags === nothing ? 4 : lags,
                                                          config, draws, "direct")
        bir = irf(post, horizon)
        ce = policy_causal_effects(bir, shocks, outcomes, instruments;
                                   H=horizon, normalize=normalize)
        return ce, bir, post
    elseif route == "lp"
        slp, Y, varnames = _load_and_structural_lp(data, horizon,
            lags === nothing ? 4 : lags, nothing, "cholesky", "newey_west", config;
            ci_type=:none, reps=200, conf_level=0.95)
        # LP draws are an INDEPENDENT-NORMAL N(value, se) approximation — fine
        # for pointwise bands, not a joint posterior (documented + settings row).
        ce = policy_causal_effects(slp, shocks, outcomes, instruments;
                                   H=horizon, n_draws=n_draws, _fwd_seed()...)
        normalize === :none || throw(CliError("usage/invalid",
            "policy: --normalize applies to the var|bvar|sign routes; the lp route keeps the estimator's scale"))
        return ce, slp.irf, slp
    elseif route == "sign"
        model, Y, varnames, p = _load_and_estimate_var(data, lags)
        check_func, _ = _build_check_func(config)
        check_func === nothing && throw(CliError("usage/missing",
            "policy effects sign requires --config with [identification] sign restrictions"))
        set = identify_sign(model, horizon, check_func; max_draws=replications > 0 ? replications : 1000,
                            store_all=true, _fwd_seed()...)
        ce = policy_causal_effects(set, shocks, outcomes, instruments;
                                   H=horizon, normalize=normalize)
        return ce, set, set
    end
    throw(CliError("usage/invalid", "unknown policy source route '$route'"))
end

# ── Renderers ───────────────────────────────────────────────

"""Tidy causal-effects menu: variable|role|shock|horizon|value."""
function _policy_effects_table(ce)
    rows_var = String[]; rows_role = String[]; rows_shock = String[]
    rows_h = Int[]; rows_v = Float64[]
    for (i, sym) in enumerate(ce.outcomes), (k, lab) in enumerate(ce.shock_labels), h in 1:ce.H
        push!(rows_var, String(sym)); push!(rows_role, "outcome")
        push!(rows_shock, lab); push!(rows_h, h)
        push!(rows_v, round(Float64(ce.Theta_x[i][h, k]); digits=6))
    end
    for (i, sym) in enumerate(ce.instruments), (k, lab) in enumerate(ce.shock_labels), h in 1:ce.H
        push!(rows_var, String(sym)); push!(rows_role, "instrument")
        push!(rows_shock, lab); push!(rows_h, h)
        push!(rows_v, round(Float64(ce.Theta_z[i][h, k]); digits=6))
    end
    DataFrame(variable=rows_var, role=rows_role, shock=rows_shock,
              horizon=rows_h, value=rows_v)
end

function _policy_effects_summary(ce, normalize)
    # upstream's n_draws(ce) is NOT exported — read the container fields directly
    nd = ce.Theta_x_draws === nothing ? 0 : size(ce.Theta_x_draws[1], 3)
    Pair{String,Any}[
        "H (horizon)"    => ce.H,
        "n_shocks"       => length(ce.shock_labels),
        "shock_labels"   => join(ce.shock_labels, ", "),
        # Square = model news menu (exact solve); thin = empirically identified
        # shocks (least-squares counterfactual) — the module's central axis.
        "is_square"      => is_square(ce),
        "source"         => String(ce.source),
        "normalize"      => String(normalize),
        "n_draws"        => nd,
    ]
end

# ── policy effects handlers ─────────────────────────────────

function _policy_effects(route::String; data::String, lags=nothing, horizon::Int=20,
                         draws::Int=2000, replications::Int=0, n_draws::Int=500,
                         config::String="", shocks::String="", outcomes::String="",
                         instruments::String="", normalize::String="none",
                         output::String="", format::String="table")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    n_draws >= 1 || throw(CliError("usage/invalid",
        "policy: --n-draws must be ≥ 1 (got $n_draws)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments"; required=false)
    norm_sym === :instrument_impact && isempty(ins_pairs) && throw(CliError("usage/invalid",
        "policy: --normalize instrument-impact needs at least one --instruments entry (it rescales to the FIRST instrument's impact)"))

    _status("Policy causal effects ($route): H=$horizon, shocks=$shocks")
    _status()

    ce, _, _ = try
        _policy_menu(route; data=data, lags=lags, horizon=horizon, draws=draws,
                     replications=replications, n_draws=n_draws, config=config,
                     shocks=shock_list, outcomes=out_pairs, instruments=ins_pairs,
                     normalize=norm_sym)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "policy causal effects"))
    end

    output_result(_policy_effects_table(ce); format=Symbol(format), output=output,
                  title="Policy Causal Effects Menu")
    output_kv(_policy_effects_summary(ce, norm_sym); format=format,
              title="Policy Causal Effects Summary")
    return ce
end

# ── policy counterfactual handlers ──────────────────────────

function _policy_counterfactual(route::String; data::String, lags=nothing,
                                horizon::Int=20, draws::Int=2000, replications::Int=0,
                                n_draws::Int=500, config::String="",
                                shocks::String="", nonpolicy_shock::String="",
                                outcomes::String="", instruments::String="",
                                normalize::String="none", rule::String="",
                                rule_config::String="", method::String="auto",
                                use_draws::String="auto", baseline_draws::String="fixed",
                                quantiles::String="0.16,0.5,0.84",
                                spanned_tol::Float64=0.05, negate::Bool=false,
                                output::String="", format::String="table",
                                plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    n_draws >= 1 || throw(CliError("usage/invalid",
        "policy: --n-draws must be ≥ 1 (got $n_draws)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    spanned_tol > 0 || throw(CliError("usage/invalid",
        "policy: --spanned-tol must be > 0 (got $spanned_tol)"))
    method in ("auto", "ls", "exact") || throw(CliError("usage/invalid-option",
        "invalid --method '$method'; must be auto, ls or exact"))
    use_draws in ("auto", "on", "off") || throw(CliError("usage/invalid-option",
        "invalid --use-draws '$use_draws'; must be auto, on or off"))
    baseline_draws in ("fixed", "match") || throw(CliError("usage/invalid-option",
        "invalid --baseline-draws '$baseline_draws'; must be fixed or match"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    qs = _parse_cf_quantiles(quantiles)
    isempty(strip(nonpolicy_shock)) && throw(CliError("usage/missing",
        "policy counterfactual: --nonpolicy-shock is required (the disturbance the rule responds to)";
        hint="a shock index or name, distinct from the policy --shocks"))
    np_shock = (v = tryparse(Int, strip(nonpolicy_shock)); v === nothing ?
                String(strip(nonpolicy_shock)) : v)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments")
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]

    # Rule construction is pure config/argument work — before any estimation.
    rule_obj = _build_policy_rule(rule, rule_config, horizon, out_syms, ins_syms)

    _status("Policy counterfactual ($route): rule=$(rule_obj.name), H=$horizon")
    _status()

    result = try
        ce, ir, _ = _policy_menu(route; data=data, lags=lags, horizon=horizon, draws=draws,
                              replications=replications, n_draws=n_draws, config=config,
                              shocks=shock_list, outcomes=out_pairs, instruments=ins_pairs,
                              normalize=norm_sym)
        # The LP route has no draw-bearing IRF object for the baseline; its
        # baseline comes from the SAME LP point estimates (draws are the
        # container's independent-normal approximation) — baseline stays fixed.
        base = baseline_path(ir, np_shock, out_pairs, ins_pairs;
                             H=horizon, negate=negate)
        policy_counterfactual(base, ce, rule_obj;
                              method=Symbol(method), draws=Symbol(use_draws),
                              baseline_draws=Symbol(baseline_draws),
                              quantiles=Tuple(qs), spanned_tol=spanned_tol)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "policy counterfactual"))
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
    _render_policy_counterfactual(result; format=format, output=output)
    return result
end

# ── policy optimal handler (W5/#127) ────────────────────────

function _policy_optimal(route::String; data::String, lags=nothing,
                         horizon::Int=20, draws::Int=2000, replications::Int=0,
                         n_draws::Int=500, config::String="",
                         shocks::String="", nonpolicy_shock::String="",
                         outcomes::String="", instruments::String="",
                         normalize::String="none", loss_config::String="",
                         use_draws::String="auto", baseline_draws::String="fixed",
                         quantiles::String="0.16,0.5,0.84", negate::Bool=false,
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    n_draws >= 1 || throw(CliError("usage/invalid",
        "policy: --n-draws must be ≥ 1 (got $n_draws)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    use_draws in ("auto", "on", "off") || throw(CliError("usage/invalid-option",
        "invalid --use-draws '$use_draws'; must be auto, on or off"))
    baseline_draws in ("fixed", "match") || throw(CliError("usage/invalid-option",
        "invalid --baseline-draws '$baseline_draws'; must be fixed or match"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    qs = _parse_cf_quantiles(quantiles)
    isempty(strip(nonpolicy_shock)) && throw(CliError("usage/missing",
        "policy optimal: --nonpolicy-shock is required (the disturbance the optimal policy responds to)"))
    np_shock = (v = tryparse(Int, strip(nonpolicy_shock)); v === nothing ?
                String(strip(nonpolicy_shock)) : v)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments")
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]

    loss, z_wedge = _build_policy_loss(loss_config, horizon, out_syms, ins_syms)

    _status("Optimal policy ($route): loss=$(loss.name), H=$horizon")
    _status()

    result = try
        ce, ir, _ = _policy_menu(route; data=data, lags=lags, horizon=horizon, draws=draws,
                              replications=replications, n_draws=n_draws, config=config,
                              shocks=shock_list, outcomes=out_pairs, instruments=ins_pairs,
                              normalize=norm_sym)
        base = baseline_path(ir, np_shock, out_pairs, ins_pairs;
                             H=horizon, negate=negate)
        # NOTE: optimal_policy hardcodes spanned_tol = 0.05 upstream (no kwarg) —
        # this leaf deliberately declares NO --spanned-tol (#85 class).
        optimal_policy(base, ce, loss;
                       z_wedge=z_wedge, draws=Symbol(use_draws),
                       baseline_draws=Symbol(baseline_draws), quantiles=Tuple(qs))
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "optimal policy"))
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
    _render_policy_counterfactual(result; format=format, output=output)
    return result
end

# ── policy moments handler (W5/#127) ────────────────────────

function _parse_cf_frequencies(spec::String)
    s = strip(spec)
    s == "none" && return :none
    s == "business-cycle" && return :business_cycle
    parts = split(s, ",")
    length(parts) == 2 || throw(CliError("usage/invalid",
        "policy moments: --frequencies must be none, business-cycle, or lo,hi in radians (got '$spec')"))
    lo = tryparse(Float64, strip(parts[1]))
    hi = tryparse(Float64, strip(parts[2]))
    (lo === nothing || hi === nothing) && throw(CliError("usage/invalid",
        "policy moments: --frequencies band bounds must be numbers (got '$spec')"))
    (0.0 <= lo < hi <= Float64(π) + 1e-12) || throw(CliError("usage/invalid",
        "policy moments: --frequencies needs 0 ≤ lo < hi ≤ π (got $lo, $hi)"))
    return (lo, hi)
end

function _policy_moments(route::String; data::String, lags=nothing,
                         horizon::Int=20, draws::Int=2000, replications::Int=0,
                         config::String="", shocks::String="",
                         outcomes::String="", instruments::String="",
                         normalize::String="none", rule::String="",
                         rule_config::String="", loss_config::String="",
                         use_draws::String="auto", draw_source::String="ce",
                         quantiles::String="0.16,0.5,0.84",
                         frequencies::String="none",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="",
                         plot_view::String="sd")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    use_draws in ("auto", "on", "off") || throw(CliError("usage/invalid-option",
        "invalid --use-draws '$use_draws'; must be auto, on or off"))
    draw_source in ("ce", "wold", "both") || throw(CliError("usage/invalid-option",
        "invalid --draw-source '$draw_source'; must be ce, wold or both"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    plot_view in ("sd", "corr") || throw(CliError("usage/invalid-option",
        "invalid --plot-view '$plot_view'; must be sd or corr"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    qs = _parse_cf_quantiles(quantiles)
    freq = _parse_cf_frequencies(frequencies)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments"; required=false)
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]

    # Policy = a rule XOR a loss.
    has_rule = !isempty(rule) || !isempty(rule_config)
    has_loss = !isempty(loss_config)
    (has_rule && has_loss) && throw(CliError("usage/invalid",
        "policy moments: give a rule (--rule/--rule-config) OR a loss (--loss-config), not both"))
    (has_rule || has_loss) || throw(CliError("usage/missing",
        "policy moments: a counterfactual policy is required — --rule/--rule-config or --loss-config"))
    policy = has_rule ?
        _build_policy_rule(rule, rule_config, horizon, out_syms, ins_syms) :
        first(_build_policy_loss(loss_config, horizon, out_syms, ins_syms))

    _status("Counterfactual second moments ($route): H=$horizon" *
            (freq === :none ? "" : ", band=$frequencies"))
    _status()

    mom = try
        if route == "var"
            model, Y, varnames, p = _load_and_estimate_var(data, lags)
            kw = Dict{Symbol,Any}()
            if replications > 0
                kw[:ci_type] = :bootstrap
                kw[:reps] = replications
            end
            ir = irf(model, horizon; kw...)
            ce = policy_causal_effects(ir, shock_list, out_pairs, ins_pairs;
                                       H=horizon, normalize=norm_sym)
            # orthogonalize carries NO identification content here — moments are
            # rotation-invariant (CMW App. A.2); cholesky is just a factorization.
            wold = wold_representation(model; H=horizon)
        else
            post, Y, varnames, p, n = _load_and_estimate_bvar(data,
                lags === nothing ? 4 : lags, config, draws, "direct")
            bir = irf(post, horizon)
            ce = policy_causal_effects(bir, shock_list, out_pairs, ins_pairs;
                                       H=horizon, normalize=norm_sym)
            wold = wold_representation(post; H=horizon)
        end
        counterfactual_moments(wold, ce, policy;
                               outcomes=out_pairs, instruments=ins_pairs,
                               draws=Symbol(use_draws), draw_source=Symbol(draw_source),
                               quantiles=Tuple(qs), frequencies=freq)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "counterfactual moments"))
    end

    _maybe_plot(mom; plot=plot, plot_save=plot_save, view=Symbol(plot_view))
    _render_policy_moments(mom; format=format, output=output, draw_source=draw_source)
    return mom
end

function _render_policy_moments(m; format::String="table", output::String="",
                                draw_source::String="ce")
    nv = length(m.varnames)
    # 1. Standard deviations, baseline vs counterfactual (+ draw bands).
    sd = DataFrame(variable=String.(m.varnames),
                   sd_base=round.(Float64.(m.sd_base); digits=6),
                   sd_cf=round.(Float64.(m.sd_cf); digits=6))
    if m.sd_cf_bands !== nothing
        for q in 1:size(m.sd_cf_bands, 2)
            sd[!, Symbol("sd_cf_q$q")] = round.(Float64.(m.sd_cf_bands[:, q]); digits=6)
        end
    end
    output_result(sd; format=Symbol(format), output=output,
                  title="Counterfactual Standard Deviations")

    # 2. Correlations, tidy upper triangle.
    v1 = String[]; v2 = String[]; cb = Float64[]; cc = Float64[]
    for i in 1:nv, j in (i+1):nv
        push!(v1, String(m.varnames[i])); push!(v2, String(m.varnames[j]))
        push!(cb, round(Float64(m.corr_base[i, j]); digits=6))
        push!(cc, round(Float64(m.corr_cf[i, j]); digits=6))
    end
    isempty(v1) || output_result(
        DataFrame(var1=v1, var2=v2, corr_base=cb, corr_cf=cc);
        format=Symbol(format), output=_per_var_output_path(output, "corr"),
        title="Counterfactual Correlations")

    # 3. Summary — tail_share is the VMA-truncation honesty number.
    pairs = Pair{String,Any}[
        "policy"      => m.policy_name,
        "H"           => m.H,
        "tail_share (VMA truncation; > 0.01 means grow --horizon)" =>
            round(Float64(m.tail_share); digits=6),
        "draw_source" => draw_source,
        "freq_band"   => m.freq_band === nothing ? "full spectrum" :
                         "$(round(Float64(m.freq_band[1]); digits=4)):$(round(Float64(m.freq_band[2]); digits=4))",
    ]
    output_kv(pairs; format=format, title="Moments Summary")
end

function _render_policy_counterfactual(r; format::String="table", output::String="")
    # 1. Paths table (baseline vs counterfactual, bands when propagated).
    nq = length(r.quantile_levels)
    has_bands = r.x_bands !== nothing
    rows = DataFrame(variable=String[], role=String[], horizon=Int[],
                     baseline=Float64[], counterfactual=Float64[])
    if has_bands
        for q in 1:nq
            rows[!, Symbol("q", round(Int, 100 * r.quantile_levels[q]))] = Float64[]
        end
    end
    _push_paths!(rows, r.outcomes, "outcome", r.x_base, r.x_cf, r.x_bands, nq)
    _push_paths!(rows, r.instruments, "instrument", r.z_base, r.z_cf, r.z_bands, nq)
    # Title (→ envelope table key) is FIXED — the rule name lives in the summary
    # kv; a rule-dependent slug would make named_table selection impossible.
    output_result(rows; format=Symbol(format), output=output,
                  title="Policy Counterfactual Paths")

    # 2. The enforcing date-0 policy-shock vector ν*.
    nu_df = DataFrame(shock=r.shock_labels, nu=round.(Float64.(r.nu); digits=6))
    output_result(nu_df; format=Symbol(format),
                  output=_per_var_output_path(output, "nu"),
                  title="Enforcing Policy Shocks (nu)")

    # 3. Implementation-error path — the honesty signal for thin menus. NOT
    # indexed by horizon: optimal_policy stacks the outcome+instrument FOC
    # blocks, so the path can be a multiple of H — index it by component.
    err_df = DataFrame(index=1:length(r.error_path),
                       error=round.(Float64.(r.error_path); digits=6))
    output_result(err_df; format=Symbol(format),
                  output=_per_var_output_path(output, "error"),
                  title="Implementation Error Path")

    # 4. Honesty/summary kv — in the DATA, not stderr.
    pairs = Pair{String,Any}[
        "rule"            => r.rule_name,
        "H"               => r.H,
        "rel_residual"    => round(Float64(r.rel_residual); digits=6),
        "spanned"         => r.spanned,
        "n_draws_used"    => r.n_draws_used,
        "n_draws_failed"  => r.n_draws_failed,
        "quantile_levels" => join(r.quantile_levels, ", "),
    ]
    if r.rel_residual_bands !== nothing
        pairs = vcat(pairs, Pair{String,Any}[
            "rel_residual_bands" => join(round.(Float64.(r.rel_residual_bands); digits=6), ", ")])
    end
    # Loss accounting — populated by optimal_policy (W5), NaN on plain rule
    # counterfactuals. foc_norm ≈ 0 is the optimality check.
    if isfinite(r.loss_base)
        append!(pairs, Pair{String,Any}[
            "loss_base" => round(Float64(r.loss_base); digits=6),
            "loss_cf"   => round(Float64(r.loss_cf); digits=6),
            "foc_norm (≈0 at the optimum)" => round(Float64(r.foc_norm); digits=8),
        ])
        # Upstream warns "loss increased" on a kernel/sign bug — that is a
        # RESULT, so it must be data, not a swallowed stderr line.
        r.loss_cf > r.loss_base && push!(pairs, "warning" =>
            "loss INCREASED under the optimal policy — upstream flags this as a kernel/sign bug; do not use these paths")
    end
    r.spanned || push!(pairs, "note" =>
        "rule NOT enforceable within the span of the supplied policy shocks; the counterfactual is a least-squares approximation — read error_path")
    output_kv(pairs; format=format, title="Counterfactual Summary")
end

function _push_paths!(rows, syms, role, base, cf, bands, nq)
    for (i, sym) in enumerate(syms)
        for h in eachindex(base[i])
            row = Any[String(sym), role, h,
                      round(Float64(base[i][h]); digits=6),
                      round(Float64(cf[i][h]); digits=6)]
            if bands !== nothing
                append!(row, [round(Float64(bands[i][h, q]); digits=6) for q in 1:nq])
            end
            push!(rows, row)
        end
    end
end

# ── OPP family (W6/#128, Barnichon–Mesters) ─────────────────

"""Parse `--targets infl=2.0,ygap=0` into Pair{Symbol,Float64}. REQUIRED to
cover every outcome: the OPP consumes GAPS, and an omitted outcome silently
defaults to a ZERO target — feeding levels drives levels to zero with no
error, the module's loudest trap."""
function _parse_cf_targets(spec::String, out_syms::Vector{Symbol})
    isempty(strip(spec)) && throw(CliError("usage/missing",
        "policy opp: --targets is required — the OPP consumes GAPS (forecast − target), and an implicit zero target on a LEVEL forecast silently drives the level to zero";
        hint="one entry per outcome, e.g. --targets infl=2.0,ygap=0"))
    out = Pair{Symbol,Float64}[]
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        parts = split(tok, "=")
        length(parts) == 2 || throw(CliError("usage/invalid",
            "policy opp: --targets entries must be name=value (got '$tok')"))
        v = tryparse(Float64, strip(parts[2]))
        v === nothing && throw(CliError("usage/invalid",
            "policy opp: --targets values must be numbers (got '$tok')"))
        push!(out, Symbol(strip(parts[1])) => v)
    end
    tnames = first.(out)
    for s in out_syms
        s in tnames || throw(CliError("usage/missing",
            "policy opp: --targets must cover EVERY outcome — '$s' has no target (an implicit zero is the gaps-vs-levels trap)"))
    end
    for s in tnames
        s in out_syms || throw(CliError("usage/invalid",
            "policy opp: --targets names '$s', which is not among the --outcomes"))
    end
    return out
end

function _parse_cf_levels(spec::String)
    ls = _parse_cf_quantiles(spec)   # same numeric/order validation
    return Tuple(ls)
end

"""`--instrument-path v1,v2,…` (length H) for the single mapped instrument."""
function _parse_cf_instrument_path(spec::String, H::Int, ins_syms::Vector{Symbol})
    isempty(strip(spec)) && return nothing
    length(ins_syms) == 1 || throw(CliError("usage/invalid",
        "policy opp: --instrument-path applies to a SINGLE mapped instrument (got $(length(ins_syms)))"))
    vals = Float64[]
    for tok in split(spec, ",")
        v = tryparse(Float64, strip(tok))
        v === nothing && throw(CliError("usage/invalid",
            "policy opp: --instrument-path values must be numbers (got '$tok')"))
        push!(vals, v)
    end
    length(vals) == H || throw(CliError("usage/invalid",
        "policy opp: --instrument-path has $(length(vals)) values but --horizon is $H"))
    return [ins_syms[1] => vals]
end

function _policy_opp(route::String; data::String, lags=nothing, horizon::Int=20,
                     draws::Int=2000, replications::Int=0, n_draws::Int=500,
                     config::String="", shocks::String="", outcomes::String="",
                     instruments::String="", normalize::String="none",
                     loss_config::String="", targets::String="",
                     values_file::String="", sd::String="", rho::Float64=0.9,
                     cross_corr_file::String="", min_sd::Float64=0.0,
                     interp_quarterly::Bool=false, origin::String="",
                     instrument_path::String="", constraints_file::String="",
                     method::String="auto", n_sim::Int=2000,
                     levels::String="0.6,0.75,0.9", matched_draws::Bool=false,
                     output::String="", format::String="table",
                     plot::Bool=false, plot_save::String="", plot_view::String="delta")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    n_draws >= 1 || throw(CliError("usage/invalid",
        "policy: --n-draws must be ≥ 1 (got $n_draws)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    n_sim >= 0 || throw(CliError("usage/invalid",
        "policy opp: --n-sim must be ≥ 0 (got $n_sim)"))
    (0.0 <= min_sd) || throw(CliError("usage/invalid",
        "policy opp: --min-sd must be ≥ 0 (got $min_sd)"))
    method in ("auto", "slsqp", "projection") || throw(CliError("usage/invalid-option",
        "invalid --method '$method'; must be auto, slsqp or projection (constrained OPP only)"))
    plot_view in ("delta", "paths") || throw(CliError("usage/invalid-option",
        "invalid --plot-view '$plot_view'; must be delta or paths"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    lv = _parse_cf_levels(levels)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments")
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]
    loss, z_wedge = _build_policy_loss(loss_config, horizon, out_syms, ins_syms)
    ipath = _parse_cf_instrument_path(instrument_path, horizon, ins_syms)

    use_external = !isempty(values_file)
    (use_external && isempty(sd) && isempty(cross_corr_file)) && throw(CliError("usage/missing",
        "policy opp: external GAP paths need forecast uncertainty — give --sd (per-outcome, BM damped covariance) or --cross-corr-file"))
    tpairs = use_external ? Pair{Symbol,Float64}[] :
             _parse_cf_targets(targets, out_syms)
    (use_external && !isempty(targets)) && throw(CliError("usage/invalid",
        "policy opp: --targets applies to the model-forecast route; --values-file paths are GAPS already"))

    _status("Optimal policy perturbation ($route): H=$horizon" *
            (isempty(origin) ? "" : ", origin=$origin"))
    _status()

    result_nt = try
        ce, ir, est = _policy_menu(route; data=data, lags=lags, horizon=horizon, draws=draws,
                              replications=replications, n_draws=n_draws, config=config,
                              shocks=shock_list, outcomes=out_pairs, instruments=ins_pairs,
                              normalize=norm_sym)
        pf = if use_external
            df = load_data(values_file)
            cols = names(df)
            vals = Vector{Vector{Float64}}()
            for s in out_syms
                String(s) in cols || throw(CliError("data/missing-column",
                    "policy opp: --values-file has no column '$s' (columns: $(join(cols, ", ")))"))
                v = Float64.(collect(skipmissing(df[!, String(s)])))
                interp_quarterly && (v = interp_to_quarterly(v, horizon))
                length(v) == horizon || throw(CliError("data/shape",
                    "policy opp: --values-file paths must have length --horizon = $horizon (got $(length(v)); annual SEP paths need --interp-quarterly)"))
                push!(vals, v)
            end
            cc = :independent
            if !isempty(cross_corr_file)
                isempty(sd) || throw(CliError("usage/invalid",
                    "policy opp: --cross-corr-file IGNORES --sd/--rho upstream — give one or the other, not both"))
                cc = Matrix{Float64}(df_to_matrix(load_data(cross_corr_file)))
            end
            sdv = nothing
            if !isempty(sd)
                ss = [tryparse(Float64, strip(t)) for t in split(sd, ",")]
                any(isnothing, ss) && throw(CliError("usage/invalid",
                    "policy opp: --sd must be numbers, one per outcome"))
                length(ss) == length(out_syms) || throw(CliError("usage/invalid",
                    "policy opp: --sd has $(length(ss)) entries but $(length(out_syms)) outcomes"))
                sdv = [fill(Float64(x), horizon) for x in ss]
            end
            policy_forecast(out_syms, vals; sd=sdv, rho=rho, n_draws=n_draws,
                            H=horizon, cross_corr=cc, min_sd=min_sd, origin=origin,
                            _fwd_seed()...)
        elseif route == "bvar"
            # store_draws is LOAD-BEARING: without it estimate_opp silently
            # falls back to IRF-only bands (narrower, one @info line).
            fc = forecast(est, horizon; store_draws=true)
            policy_forecast(fc, out_pairs; targets=tpairs, H=horizon, origin=origin)
        else
            fc = forecast(est, horizon)
            policy_forecast(fc, out_pairs; targets=tpairs, H=horizon, origin=origin)
        end

        if !isempty(constraints_file)
            ipath === nothing && throw(CliError("usage/missing",
                "policy opp: --constraints-file requires --instrument-path (the ANNOUNCED path the constraints act on)"))
            specs = get_opp_constraints(load_config(constraints_file))
            cons = [zlb_constraint(; floor=c.floor, instrument=c.instrument,
                                   horizons=c.horizons isa UnitRange ? c.horizons :
                                            (1:typemax(Int)))
                    for c in specs]
            constrained_opp(pf, ce, loss, cons;
                            instrument_path=ipath, z_wedge=z_wedge,
                            method=Symbol(method), n_sim=n_sim, levels=lv,
                            independent=!matched_draws, _fwd_seed()...)
        else
            has_draws = pf.draws !== nothing || ce.Theta_x_draws !== nothing
            if has_draws && n_sim > 0
                (; result=estimate_opp(pf, ce, loss;
                                       instrument_path=ipath, z_wedge=z_wedge,
                                       independent=!matched_draws, levels=lv,
                                       n_sim=n_sim, _fwd_seed()...),
                 method_used=:unconstrained, binding=Bool[],
                 kkt_residual=NaN, warm_start_feasible=true)
            else
                (; result=opp(pf, ce, loss; instrument_path=ipath, z_wedge=z_wedge),
                 method_used=:unconstrained, binding=Bool[],
                 kkt_residual=NaN, warm_start_feasible=true)
            end
        end
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "optimal policy perturbation"))
    end

    r = result_nt.result
    _maybe_plot(r; plot=plot, plot_save=plot_save, view=Symbol(plot_view))
    _render_opp(r, result_nt; format=format, output=output,
                constrained=!isempty(constraints_file))
    return r
end

function _render_opp(r, nt; format::String="table", output::String="",
                     constrained::Bool=false)
    # 1. The recommendation per identified shock direction. delta is the draw
    # MEDIAN after estimate_opp (BM convention); delta_plugin keeps the point.
    dd = DataFrame(shock=r.shock_labels,
                   delta=round.(Float64.(r.delta); digits=6),
                   delta_plugin=round.(Float64.(r.delta_plugin); digits=6),
                   gradient=round.(Float64.(r.gradient); digits=6))
    if r.bands !== nothing
        for lev in sort(collect(keys(r.bands)))
            B = r.bands[lev]
            dd[!, Symbol("lo", round(Int, 100lev))] = round.(Float64.(B[:, 1]); digits=6)
            dd[!, Symbol("hi", round(Int, 100lev))] = round.(Float64.(B[:, 2]); digits=6)
        end
    end
    if r.reject !== nothing
        for lev in sort(collect(keys(r.reject)))
            dd[!, Symbol("reject", round(Int, 100lev))] = r.reject[lev]
        end
    end
    output_result(dd; format=Symbol(format), output=output,
                  title="OPP Recommendation (delta)")

    # 2. Objective gap paths before/after (+ instrument paths when present).
    gv = String[]; gh = Int[]; gb = Float64[]; go = Float64[]
    for (i, sym) in enumerate(r.outcomes), h in 1:r.H
        push!(gv, String(sym)); push!(gh, h)
        push!(gb, round(Float64(r.Y_base[i][h]); digits=6))
        push!(go, round(Float64(r.Y_opp[i][h]); digits=6))
    end
    output_result(DataFrame(variable=gv, horizon=gh, gap_base=gb, gap_opp=go);
                  format=Symbol(format), output=_per_var_output_path(output, "gaps"),
                  title="Objective Gap Paths")
    if r.P_base !== nothing && r.P_opp !== nothing
        pv = String[]; ph = Int[]; pb = Float64[]; po = Float64[]
        for (k, sym) in enumerate(r.instruments), h in 1:r.H
            push!(pv, String(sym)); push!(ph, h)
            push!(pb, round(Float64(r.P_base[k][h]); digits=6))
            push!(po, round(Float64(r.P_opp[k][h]); digits=6))
        end
        output_result(DataFrame(instrument=pv, horizon=ph,
                                announced=pb, recommended=po);
                      format=Symbol(format),
                      output=_per_var_output_path(output, "paths"),
                      title="Instrument Paths (announced vs recommended)")
    end

    # 3. Summary — reversed-polarity note is an envelope FIELD, not stderr.
    pairs = Pair{String,Any}[
        "loss_base" => round(Float64(r.loss_base); digits=6),
        "loss_opp"  => round(Float64(r.loss_opp); digits=6),
        "H"         => r.H,
        "origin"    => isempty(r.origin) ? "(unset)" : r.origin,
        "n_failed"  => r.n_failed,
    ]
    r.bands !== nothing && push!(pairs, "band_polarity" =>
        "BM reversed polarity: bands at 60/75/90% — rejection at the LOWER level rejects more readily (conservative for a policymaker averse to non-optimal policy)")
    if constrained
        append!(pairs, Pair{String,Any}[
            "method_used" => String(nt.method_used),
            "binding" => isempty(nt.binding) ? "none" :
                         join(string.(nt.binding), ","),
            "kkt_residual" => isfinite(nt.kkt_residual) ?
                              round(Float64(nt.kkt_residual); digits=8) : "n/a",
            "warm_start_feasible" => nt.warm_start_feasible,
        ])
        nt.method_used === :projection && push!(pairs, "warning" =>
            "projection fallback: feasible but NOT the constrained optimum (upstream's own caveat)")
    end
    output_kv(pairs; format=format, title="OPP Summary")
end

# ── policy opp-sequence (W6/#128) ───────────────────────────

function _policy_opp_sequence(route::String; data::String, lags=nothing,
                              horizon::Int=20, draws::Int=2000, replications::Int=0,
                              n_draws::Int=500, config::String="", shocks::String="",
                              outcomes::String="", instruments::String="",
                              normalize::String="none", loss_config::String="",
                              forecasts_dir::String="", sd::String="",
                              rho::Float64=0.9, n_sim::Int=0,
                              levels::String="0.6,0.75,0.9", matched_draws::Bool=false,
                              output::String="", format::String="table",
                              plot::Bool=false, plot_save::String="",
                              plot_view::String="fan")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    n_sim >= 0 || throw(CliError("usage/invalid",
        "policy opp-sequence: --n-sim must be ≥ 0 (got $n_sim)"))
    draws >= 1 || throw(CliError("usage/invalid",
        "policy: --draws must be ≥ 1 (got $draws)"))
    n_draws >= 1 || throw(CliError("usage/invalid",
        "policy: --n-draws must be ≥ 1 (got $n_draws)"))
    replications >= 0 || throw(CliError("usage/invalid",
        "policy: --replications must be ≥ 0 (got $replications)"))
    plot_view in ("fan", "decomposition") || throw(CliError("usage/invalid-option",
        "invalid --plot-view '$plot_view'; must be fan or decomposition"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    lv = _parse_cf_levels(levels)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments")
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]
    loss, z_wedge = _build_policy_loss(loss_config, horizon, out_syms, ins_syms)

    isempty(sd) && throw(CliError("usage/missing",
        "policy opp-sequence: --sd is required (per-outcome forecast sd — the external forecast containers need uncertainty)"))
    isempty(forecasts_dir) && throw(CliError("usage/missing",
        "policy opp-sequence: --forecasts-dir is required — one GAP-path CSV per date (sorted filenames = dates; one column per outcome)"))
    isdir(forecasts_dir) || throw(CliError("data/file-not-found",
        "policy opp-sequence: --forecasts-dir '$forecasts_dir' is not a directory"))
    files = sort(filter(f -> endswith(f, ".csv"), readdir(forecasts_dir)))
    isempty(files) && throw(CliError("data/empty",
        "policy opp-sequence: no .csv files in '$forecasts_dir'"))
    length(files) >= 2 || throw(CliError("usage/invalid",
        "policy opp-sequence needs ≥ 2 dates (got $(length(files))); a single date is plain policy opp"))

    sdv = nothing
    if !isempty(sd)
        ss = [tryparse(Float64, strip(t)) for t in split(sd, ",")]
        any(isnothing, ss) && throw(CliError("usage/invalid",
            "policy opp-sequence: --sd must be numbers, one per outcome"))
        length(ss) == length(out_syms) || throw(CliError("usage/invalid",
            "policy opp-sequence: --sd has $(length(ss)) entries but $(length(out_syms)) outcomes"))
        sdv = [fill(Float64(x), horizon) for x in ss]
    end

    _status("OPP sequence ($route): $(length(files)) dates, H=$horizon")
    _status()

    seq = try
        ce, _, _ = _policy_menu(route; data=data, lags=lags, horizon=horizon,
                                draws=draws, replications=replications,
                                n_draws=n_draws, config=config, shocks=shock_list,
                                outcomes=out_pairs, instruments=ins_pairs,
                                normalize=norm_sym)
        dates = String[splitext(f)[1] for f in files]
        fcs = map(files) do f
            df = load_data(joinpath(forecasts_dir, f))
            cols = names(df)
            vals = Vector{Vector{Float64}}()
            for s in out_syms
                String(s) in cols || throw(CliError("data/missing-column",
                    "policy opp-sequence: '$f' has no column '$s' (columns: $(join(cols, ", ")))"))
                v = Float64.(collect(skipmissing(df[!, String(s)])))
                length(v) == horizon || throw(CliError("data/shape",
                    "policy opp-sequence: '$f' paths must have length --horizon = $horizon (got $(length(v)))"))
                push!(vals, v)
            end
            policy_forecast(out_syms, vals; sd=sdv, rho=rho, n_draws=n_draws,
                            H=horizon, origin=splitext(f)[1], _fwd_seed()...)
        end
        opp_sequence(collect(Union{PolicyForecast,Missing}, fcs), ce, loss;
                     dates=dates, z_wedge=z_wedge, n_sim=n_sim, levels=lv,
                     independent=!matched_draws, _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "OPP sequence"))
    end

    _maybe_plot(seq; plot=plot, plot_save=plot_save, view=Symbol(plot_view))
    _render_opp_sequence(seq; format=format, output=output)
    return seq
end

function _render_opp_sequence(s; format::String="table", output::String="")
    # Upstream layout is n_s × n_dates (shocks are ROWS) — types.jl docstring.
    ns, nd = size(s.delta)
    dv = String[]; sv = String[]; dl = Float64[]; dtc = Float64[]
    nw = Float64[]; pf = Float64[]; ag = Float64[]
    for d in 1:nd, k in 1:ns
        push!(dv, s.dates[d]); push!(sv, s.shock_labels[k])
        push!(dl, round(Float64(s.delta[k, d]); digits=6))
        push!(dtc, round(Float64(s.delta_tc[k, d]); digits=6))
        push!(nw, round(Float64(s.news_part[k, d]); digits=6))
        push!(pf, round(Float64(s.pref_part[k, d]); digits=6))
        push!(ag, round(Float64(s.aging_part[k, d]); digits=6))
    end
    output_result(DataFrame(date=dv, shock=sv, delta=dl, delta_tc=dtc);
                  format=Symbol(format), output=output,
                  title="OPP Sequence (delta by date)")
    # Exact three-part revision decomposition — news + preference + aging
    # (deliberate finite-H deviation from BM eq. 32; the parts SUM to the
    # revision exactly).
    output_result(DataFrame(date=dv, shock=sv, news=nw, pref=pf, aging=ag);
                  format=Symbol(format),
                  output=_per_var_output_path(output, "decomposition"),
                  title="OPP Revision Decomposition")
    pairs = Pair{String,Any}[
        "loss" => s.loss_name,
        "n_dates" => nd,
        "shock_labels" => join(s.shock_labels, ", "),
    ]
    s.bands !== nothing && push!(pairs, "band_polarity" =>
        "BM reversed polarity: bands at 60/75/90% — rejection at the LOWER level rejects more readily")
    output_kv(pairs; format=format, title="OPP Sequence Summary")
end

# ── W7/#129: structural routes ──────────────────────────────

"""Second Pair shape (`Symbol => Symbol`): model-variable maps for the DSGE/HA
news menus. Deliberately NOT shared with `_parse_cf_pairs` — the empirical maps
resolve to IRF columns (Int|String), these to model symbols."""
function _parse_cf_sym_pairs(spec::String, opt::String; required::Bool=true)
    out = Pair{Symbol,Symbol}[]
    for tok in split(spec, ",")
        tok = strip(tok)
        isempty(tok) && continue
        parts = split(tok, "=")
        (length(parts) == 2 && !isempty(strip(parts[1])) && !isempty(strip(parts[2]))) ||
            throw(CliError("usage/invalid",
                "policy: $opt entries must be name=model_variable (got '$tok')";
                hint="e.g. $opt infl=pi,ygap=y (model SYMBOLS, not column indices)"))
        push!(out, Symbol(strip(parts[1])) => Symbol(strip(parts[2])))
    end
    (required && isempty(out)) && throw(CliError("usage/missing",
        "policy: $opt is required (module name = model variable symbol)"))
    return out
end

"""Optional behavioral discounting on a SQUARE menu. Sentinel NaN defaults let
the handler distinguish "not requested" from the identity values (m=1, θ=0 ARE
the identity — a bare invocation would be a silent no-op otherwise)."""
function _apply_behavioral(ce, m::Float64, theta::Float64)
    isnan(m) && isnan(theta) && return ce, false
    mm = isnan(m) ? 1.0 : m
    tt = isnan(theta) ? 0.0 : theta
    (0.0 <= mm <= 1.0) || throw(CliError("usage/invalid",
        "policy: --behavioral-m must be in [0, 1] (got $mm)"))
    (0.0 <= tt <= 1.0) || throw(CliError("usage/invalid",
        "policy: --behavioral-theta must be in [0, 1] (got $tt)"))
    return behavioral(ce; m=mm, theta=tt), true
end

function _policy_news(route::String; model::String, policy_shock::String="",
                      outcomes::String="", instruments::String="",
                      horizon::Int=100, solver::String="gensys", chunk::Int=0,
                      t_horizon::Int=300, rule_closure::String="administered",
                      dx::Float64=1e-4, behavioral_m::Float64=NaN,
                      behavioral_theta::Float64=NaN,
                      output::String="", format::String="table")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    chunk >= 0 || throw(CliError("usage/invalid",
        "policy news: --chunk must be ≥ 0 (got $chunk)"))
    ce, applied = try
        if route == "dsge"
            solver in ("gensys", "klein", "blanchard-kahn") ||
                throw(CliError("usage/invalid-option",
                    "invalid --solver '$solver'; must be gensys, klein or blanchard-kahn (linear only)"))
            isempty(strip(policy_shock)) && throw(CliError("usage/missing",
                "policy news dsge: --policy-shock is required (the exogenous the news menu perturbs)"))
            out_pairs = _parse_cf_sym_pairs(outcomes, "--outcomes")
            ins_pairs = _parse_cf_sym_pairs(instruments, "--instruments"; required=false)
            spec = _load_dsge_model(model)
            _status("DSGE news menu: shock=$policy_shock, H=$horizon (one QZ of dimension n+H−1)")
            _status()
            # The menu build EVALUATES the runtime-loaded spec's residual fns →
            # world-age barrier, exactly like the RA solve path.
            ce0 = _dsge_call(policy_news_matrix, spec, Symbol(policy_shock),
                             out_pairs, ins_pairs;
                             H=horizon, solver=Symbol(replace(solver, '-' => '_')),
                             chunk=chunk)
            _apply_behavioral(ce0, behavioral_m, behavioral_theta)
        else  # ha
            t_horizon >= horizon || throw(CliError("usage/invalid",
                "policy news ha: --t-horizon ($t_horizon) must be ≥ --horizon ($horizon)"))
            dx > 0 || throw(CliError("usage/invalid",
                "policy news ha: --dx must be > 0 (got $dx)"))
            rule_closure in ("administered", "market") ||
                throw(CliError("usage/invalid-option",
                    "invalid --rule-closure '$rule_closure'; must be administered or market (market is huggett-only upstream)"))
            out_pairs = _parse_cf_sym_pairs(outcomes, "--outcomes")
            ins_pairs = isempty(strip(instruments)) ? [:rate => :r] :
                        _parse_cf_sym_pairs(instruments, "--instruments")
            spec = _load_ha_model(model)
            _status("HA news menu: H=$horizon, T_horizon=$t_horizon, closure=$rule_closure")
            _status()
            ss = compute_steady_state(spec)
            # Verified on the tag: the CF HA path never invokes the aggregate
            # spec's residual closures — no invokelatest (the W13 lesson cuts
            # both ways; do not pattern-match the RA fix).
            ce0 = policy_causal_effects(spec, ss; outcomes=out_pairs,
                                        instruments=ins_pairs, H=horizon,
                                        T_horizon=t_horizon,
                                        rule_closure=Symbol(rule_closure), dx=dx)
            _apply_behavioral(ce0, behavioral_m, behavioral_theta)
        end
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "policy news menu"))
    end

    output_result(_policy_effects_table(ce); format=Symbol(format), output=output,
                  title="Policy Causal Effects Menu")
    s = _policy_effects_summary(ce, :none)
    applied && push!(s, "behavioral" =>
        "m=$(isnan(behavioral_m) ? 1.0 : behavioral_m), theta=$(isnan(behavioral_theta) ? 0.0 : behavioral_theta) (approximation on a GE-closed menu — CMW apply per block before closure)")
    route == "ha" && t_horizon < horizon + 50 && push!(s, "truncation_warning" =>
        "T_horizon = $t_horizon < H + 50 — the sequence-space truncation may bite; grow --t-horizon")
    output_kv(s; format=format, title="Policy Causal Effects Summary")
    return ce
end

function _policy_jacobian(; model::String, input::String="r", jac_output::String="",
                          t_horizon::Int=300, dx::Float64=1e-4,
                          output::String="", format::String="table")
    input in ("r", "w") || throw(CliError("usage/invalid-option",
        "invalid --input '$input'; must be r or w"))
    t_horizon >= 1 || throw(CliError("usage/invalid",
        "policy jacobian: --t-horizon must be ≥ 1 (got $t_horizon)"))
    dx > 0 || throw(CliError("usage/invalid",
        "policy jacobian: --dx must be > 0 (got $dx)"))
    isempty(strip(jac_output)) && throw(CliError("usage/missing",
        "policy jacobian: --jac-output is required (the household aggregate to differentiate, e.g. C or A)"))
    J = try
        spec = _load_ha_model(model)
        ss = compute_steady_state(spec)
        _status("Sequence-space jacobian: d$(jac_output)/d$(input), T=$t_horizon")
        _status()
        sequence_jacobian(spec, ss, Symbol(input), Symbol(jac_output);
                          T_horizon=t_horizon, dx=dx)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "sequence jacobian"))
    end
    # A bare T×T Matrix upstream (no report/plot recipes) — tidy long render;
    # at T=300 this is 90k rows, which is the data contract, not a bug.
    rows = Int[]; cols = Int[]; vals = Float64[]
    for j in axes(J, 2), i in axes(J, 1)
        push!(rows, i); push!(cols, j)
        push!(vals, round(Float64(J[i, j]); digits=8))
    end
    output_result(DataFrame(row=rows, col=cols, value=vals);
                  format=Symbol(format), output=output,
                  title="Sequence-Space Jacobian d$(jac_output)/d$(input)",
                  key="sequence_space_jacobian")
end

function _policy_history(route::String; data::String, lags=nothing, horizon::Int=20,
                         draws::Int=2000, replications::Int=0, n_draws::Int=500,
                         config::String="", shocks::String="", outcomes::String="",
                         instruments::String="", normalize::String="none",
                         rule::String="", rule_config::String="",
                         loss_config::String="", t_range::String="",
                         use_draws::String="auto",
                         quantiles::String="0.16,0.5,0.84",
                         output::String="", format::String="table",
                         plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    use_draws in ("auto", "on", "off") || throw(CliError("usage/invalid-option",
        "invalid --use-draws '$use_draws'; must be auto, on or off"))
    normalize in ("none", "instrument-impact") || throw(CliError("usage/invalid-option",
        "invalid --normalize '$normalize'; must be none or instrument-impact"))
    norm_sym = normalize == "none" ? :none : :instrument_impact
    qs = _parse_cf_quantiles(quantiles)
    m = match(r"^(\d+):(\d+)$", strip(t_range))
    m === nothing && throw(CliError("usage/missing",
        "policy history: --t-range lo:hi is required (the observation window to re-run under the counterfactual rule)"))
    lo, hi = parse(Int, m[1]), parse(Int, m[2])
    (1 <= lo <= hi) || throw(CliError("usage/invalid",
        "policy history: --t-range needs 1 ≤ lo ≤ hi (got $t_range)"))
    (hi - lo + 1) <= horizon - 1 || throw(CliError("usage/invalid",
        "policy history: window length $(hi - lo + 1) must be ≤ H − 1 = $(horizon - 1) (grow --horizon or shrink the window)"))
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments"; required=false)
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]
    has_rule = !isempty(rule) || !isempty(rule_config)
    has_loss = !isempty(loss_config)
    (has_rule && has_loss) && throw(CliError("usage/invalid",
        "policy history: give a rule (--rule/--rule-config) OR a loss (--loss-config), not both"))
    (has_rule || has_loss) || throw(CliError("usage/missing",
        "policy history: a counterfactual policy is required — --rule/--rule-config or --loss-config"))
    policy = has_rule ?
        _build_policy_rule(rule, rule_config, horizon, out_syms, ins_syms) :
        first(_build_policy_loss(loss_config, horizon, out_syms, ins_syms))

    _status("Counterfactual history ($route): t=$t_range, H=$horizon")
    # Built from FORECAST REVISIONS, never identified shocks (raw forecasts
    # double-count — CMW subtlety #9); rests on forecast sufficiency.
    _status()

    hist = try
        ce, _, est = _policy_menu(route; data=data, lags=lags, horizon=horizon,
                                  draws=draws, replications=replications,
                                  n_draws=n_draws, config=config, shocks=shock_list,
                                  outcomes=out_pairs, instruments=ins_pairs,
                                  normalize=norm_sym)
        Y, varnames = load_multivariate_data(data)
        counterfactual_history(est, Y, lo:hi, ce, policy;
                               outcomes=out_pairs, instruments=ins_pairs,
                               H=horizon, draws=Symbol(use_draws),
                               quantiles=Tuple(qs))
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "counterfactual history"))
    end

    _maybe_plot(hist; plot=plot, plot_save=plot_save)
    nd_, nv_ = size(hist.realized)
    dv = String[]; vv = String[]; rv = Float64[]; cv = Float64[]; rr = Float64[]
    for d in 1:nd_, v in 1:nv_
        push!(dv, hist.dates[d]); push!(vv, String(hist.varnames[v]))
        push!(rv, round(Float64(hist.realized[d, v]); digits=6))
        push!(cv, round(Float64(hist.cf[d, v]); digits=6))
        push!(rr, round(Float64(hist.rel_residual[d]); digits=6))
    end
    output_result(DataFrame(date=dv, variable=vv, realized=rv,
                            counterfactual=cv, rel_residual=rr);
                  format=Symbol(format), output=output,
                  title="Counterfactual History")
    output_kv(Pair{String,Any}[
        "policy" => hist.policy_name,
        "H" => hist.H,
        "n_dates" => length(hist.dates),
        "n_draws_used" => hist.n_draws_used,
        "n_draws_failed" => hist.n_draws_failed,
        "note" => "built from forecast revisions, never identified shocks (raw forecasts double-count); rests on forecast sufficiency — see policy sufficiency",
    ]; format=format, title="History Summary")
    return hist
end

function _policy_spanning(; data::String, model::String, lags=nothing,
                          horizon::Int=20, replications::Int=0,
                          shocks::String="", nonpolicy_shock::String="",
                          outcomes::String="", instruments::String="",
                          model_outcomes::String="", model_instruments::String="",
                          policy_shock::String="", solver::String="gensys",
                          rule::String="", rule_config::String="", tol::Float64=0.1,
                          n_sim::Int=200, quantiles::String="0.16,0.5,0.84",
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    tol > 0 || throw(CliError("usage/invalid",
        "policy spanning: --tol must be > 0 (got $tol)"))
    n_sim >= 0 || throw(CliError("usage/invalid",
        "policy spanning: --n-sim must be ≥ 0 (got $n_sim)"))
    qs = _parse_cf_quantiles(quantiles)
    isempty(strip(nonpolicy_shock)) && throw(CliError("usage/missing",
        "policy spanning: --nonpolicy-shock is required"))
    isempty(strip(policy_shock)) && throw(CliError("usage/missing",
        "policy spanning: --policy-shock is required (the model shock behind the FULL news menu)"))
    np_shock = (v = tryparse(Int, strip(nonpolicy_shock)); v === nothing ?
                String(strip(nonpolicy_shock)) : v)
    shock_list = _parse_cf_shocks(shocks)
    out_pairs = _parse_cf_pairs(outcomes, "--outcomes")
    ins_pairs = _parse_cf_pairs(instruments, "--instruments"; required=false)
    mo_pairs = _parse_cf_sym_pairs(model_outcomes, "--model-outcomes")
    mi_pairs = _parse_cf_sym_pairs(model_instruments, "--model-instruments";
                                   required=false)
    out_syms = Symbol[first(p) for p in out_pairs]
    ins_syms = Symbol[first(p) for p in ins_pairs]
    # The diagnostic compares by exact symbol equality — enforce name agreement
    # HERE with a typed message, before upstream's ArgumentError does it untyped.
    Symbol[first(p) for p in mo_pairs] == out_syms || throw(CliError("usage/invalid",
        "policy spanning: --model-outcomes must use the SAME names in the SAME order as --outcomes ($(join(out_syms, ", ")))"))
    Symbol[first(p) for p in mi_pairs] == ins_syms || throw(CliError("usage/invalid",
        "policy spanning: --model-instruments must use the SAME names in the SAME order as --instruments"))
    pol = _build_policy_rule(rule, rule_config, horizon, out_syms, ins_syms)

    _status("Spanning diagnostic: empirical menu vs full news menu, H=$horizon")
    _status()

    sp = try
        ce_emp, ir, _ = _policy_menu("var"; data=data, lags=lags, horizon=horizon,
                                     draws=2000, replications=replications,
                                     n_draws=500, config="", shocks=shock_list,
                                     outcomes=out_pairs, instruments=ins_pairs,
                                     normalize=:none)
        base = baseline_path(ir, np_shock, out_pairs, ins_pairs; H=horizon)
        spec = _load_dsge_model(model)
        ce_full = _dsge_call(policy_news_matrix, spec, Symbol(policy_shock),
                             mo_pairs, mi_pairs; H=horizon,
                             solver=Symbol(replace(solver, '-' => '_')))
        spanning_diagnostic(base, ce_emp, ce_full, pol;
                            tol=tol, n_sim=n_sim, quantiles=Tuple(qs),
                            _fwd_seed()...)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "spanning diagnostic"))
    end

    _maybe_plot(sp; plot=plot, plot_save=plot_save)
    gv = String[]; gh = Int[]; ge = Float64[]; gf = Float64[]
    gg = Float64[]; gr = Float64[]
    H = length(sp.gap) ÷ max(length(sp.outcomes), 1)
    idx = 0
    for (i, sym) in enumerate(sp.outcomes), h in 1:length(sp.x_cf_emp[i])
        idx += 1
        push!(gv, String(sym)); push!(gh, h)
        push!(ge, round(Float64(sp.x_cf_emp[i][h]); digits=6))
        push!(gf, round(Float64(sp.x_cf_full[i][h]); digits=6))
    end
    output_result(DataFrame(variable=gv, horizon=gh, cf_thin=ge, cf_full=gf);
                  format=Symbol(format), output=output,
                  title="Spanning: thin vs full counterfactual paths")
    output_kv(Pair{String,Any}[
        # The verdict is a convenience; the raw numbers are the result.
        "spanned" => sp.spanned,
        "gap_rel (max)" => round(maximum(Float64.(sp.gap_rel)); digits=6),
        "loading_inside" => round(Float64(sp.loading_inside); digits=6),
        "rel_residual_emp" => round(Float64(sp.rel_residual_emp); digits=6),
        "interpretation" => "does the model choice matter for THIS counterfactual? spanned=true → the thin empirical menu already carries it",
    ]; format=format, title="Spanning Verdict")
    return sp
end

function _policy_sufficiency(; model::String, observables::String="",
                             horizon::Int=40, method::String="gensys",
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    horizon >= 1 || throw(CliError("usage/invalid",
        "policy: --horizon must be ≥ 1 (got $horizon)"))
    isempty(strip(observables)) && throw(CliError("usage/missing",
        "policy sufficiency: --observables is required (comma-separated model variables the econometrician sees)"))
    obs = Symbol[Symbol(strip(t)) for t in split(observables, ",") if !isempty(strip(t))]
    fs = try
        spec = _load_dsge_model(model)
        sol = _solve_dsge(spec; method=method)
        # Pure population laboratory — consumes NO data.
        forecast_sufficiency(sol, obs; H=horizon)
    catch e
        e isa CliError && rethrow()
        throw(_domain_or_data_error(e, "forecast sufficiency"))
    end
    _maybe_plot(fs; plot=plot, plot_save=plot_save)
    fv = String[]; fh = Int[]; fr = Float64[]
    for (j, s) in enumerate(fs.observables), h in 1:fs.H
        push!(fv, String(s)); push!(fh, h)
        push!(fr, round(Float64(fs.fev_ratio[h, j]); digits=6))
    end
    output_result(DataFrame(observable=fv, horizon=fh, fev_ratio=fr);
                  format=Symbol(format), output=output,
                  title="Forecast Sufficiency FEV Ratios")
    output_kv(Pair{String,Any}[
        "invertible" => fs.invertible,
        "one_step_ratio" => join(round.(Float64.(fs.one_step_ratio); digits=6), ", "),
        "H" => fs.H,
        "note" => "fev_ratio ≥ 1 = Wold-info FEV over full-info FEV; invertibility is SUFFICIENT for forecast sufficiency, not necessary",
    ]; format=format, title="Sufficiency Summary")
    return fs
end

# ── Registry ────────────────────────────────────────────────

const _POLICY_COMMON = [
    OptionSpec(name="horizon", type=Int, default=20,
               description="Truncation horizon H (≥ 1; MW use H=100)"),
    OptionSpec(name="shocks", type=String, default="",
               description="Identified POLICY shock columns (indices or names, comma-separated; REQUIRED)"),
    OptionSpec(name="outcomes", type=String, default="",
               description="Outcome map name=index-or-column, comma-separated (REQUIRED), e.g. infl=2,ygap=1"),
    OptionSpec(name="instruments", type=String, default="",
               description="Instrument map name=index-or-column, comma-separated, e.g. rate=3"),
    OptionSpec(name="output", short="o", type=String, default="",
               description="Export results to file"),
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="table|csv|json"),
]

function _policy_route_options(route::String)
    opts = [OptionSpec(name="lags", short="p", type=Int, default=nothing,
                       description="Lag order (default: AIC for var/sign, 4 for bvar/lp)")]
    extra = route == "bvar" ? [
        OptionSpec(name="draws", short="n", type=Int, default=2000, description="Posterior draws"),
        OptionSpec(name="config", type=String, default="", description="TOML prior config")] :
      route == "lp" ? [
        OptionSpec(name="n-draws", type=Int, default=500,
                   description="Independent-normal N(value, se) draws (pointwise approximation, NOT a joint posterior)"),
        OptionSpec(name="config", type=String, default="", description="TOML identification config")] :
      route == "sign" ? [
        OptionSpec(name="replications", type=Int, default=1000, description="Candidate rotations"),
        OptionSpec(name="config", type=String, default="", description="TOML sign-restriction config (REQUIRED)")] :
      [OptionSpec(name="replications", type=Int, default=0,
                  description="Bootstrap draws for uncertainty bands (0 = point only)")]
    return vcat(opts, extra)
end

const _POLICY_CF_OPTIONS = [
    OptionSpec(name="nonpolicy-shock", type=String, default="",
               description="The ONE non-policy shock the rule responds to (index or name; REQUIRED)"),
    OptionSpec(name="rule", type=String, default="",
               description="Builtin rule: rate-peg|inflation-target|output-gap|ngdp|taylor (taylor = TEXTBOOK rho/phi, not CMW — use --rule-config for that)"),
    OptionSpec(name="rule-config", type=String, default="",
               description="TOML [rule] section (rate-target paths, CMW taylor, custom pi_var/y_var)"),
    OptionSpec(name="method", type=String, default="auto",
               choices=["auto", "ls", "exact"],
               description="Projection: auto (exact when square) | ls | exact"),
    OptionSpec(name="use-draws", type=String, default="auto",
               choices=["auto", "on", "off"],
               description="Propagate menu draws into bands: auto|on|off"),
    OptionSpec(name="baseline-draws", type=String, default="fixed",
               choices=["fixed", "match"],
               description="fixed (MW convention, separate estimations) | match (pair draw d with draw d; equal counts enforced)"),
    OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84",
               description="Band quantiles, comma-separated in (0,1)"),
    OptionSpec(name="spanned-tol", type=Float64, default=0.05,
               description="rel_residual threshold for the spanned flag"),
]

function register_policy_commands!()
    specs = CommandSpec[]
    for route in ("var", "bvar", "lp", "sign")
        push!(specs, CommandSpec(
            path=["policy", "effects", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     # instrument-impact renormalizes per draw; near-zero-impact
                     # draws are dropped (count surfaced in the summary table)
                     OptionSpec(name="normalize", type=String, default="none",
                                choices=["none", "instrument-impact"],
                                description="none | instrument-impact (rescale so the first instrument's impact is +1)")],
            flags=FlagSpec[],   # PolicyCausalEffects has NO plot_result recipe
            tables=[TableSpec(name=:policy_causal_effects_menu,
                              description="Causal-effect menu entries by outcome, instrument and horizon"),
                    TableSpec(name=:policy_causal_effects_summary,
                              description="Menu shape, normalization and dropped-draw honesty counts")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_effects(route; kw...)),
        ))
    end
    for route in ("var", "bvar", "lp")
        push!(specs, CommandSpec(
            path=["policy", "counterfactual", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     _POLICY_CF_OPTIONS...;
                     OptionSpec(name="normalize", type=String, default="none",
                                choices=["none", "instrument-impact"],
                                description="none | instrument-impact");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="negate",
                            description="Flip the non-policy shock's sign (e.g. the contractionary version)"),
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:policy_counterfactual_paths,
                              description="Baseline vs counterfactual path per variable and horizon, with draw bands when propagated"),
                    TableSpec(name=:enforcing_policy_shocks_nu,
                              description="The date-0 policy-shock vector nu* that enforces the rule"),
                    TableSpec(name=:implementation_error_path,
                              description="Per-component residual of the rule's least-squares implementation"),
                    TableSpec(name=:counterfactual_summary,
                              description="Rule, horizon, rel_residual, spanned flag and draw counts")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_counterfactual(route; kw...)),
        ))
    end
    # ── W5/#127: policy optimal + policy moments ────────────
    for route in ("var", "bvar", "lp")
        push!(specs, CommandSpec(
            path=["policy", "optimal", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     OptionSpec(name="nonpolicy-shock", type=String, default="",
                                description="The ONE non-policy shock the optimal policy responds to (REQUIRED)");
                     OptionSpec(name="loss-config", type=String, default="",
                                description="TOML [loss] section (REQUIRED; lambda has no default upstream)");
                     OptionSpec(name="use-draws", type=String, default="auto",
                                choices=["auto", "on", "off"],
                                description="Propagate menu draws into bands: auto|on|off");
                     OptionSpec(name="baseline-draws", type=String, default="fixed",
                                choices=["fixed", "match"],
                                description="fixed | match (equal draw counts enforced)");
                     OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84",
                                description="Band quantiles, comma-separated in (0,1)");
                     # NO --spanned-tol: optimal_policy hardcodes 0.05 upstream.
                     OptionSpec(name="normalize", type=String, default="none",
                                choices=["none", "instrument-impact"],
                                description="none | instrument-impact");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="negate",
                            description="Flip the non-policy shock's sign"),
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:policy_counterfactual_paths,
                              description="Baseline vs optimal-policy path per variable and horizon, with draw bands when propagated"),
                    TableSpec(name=:enforcing_policy_shocks_nu,
                              description="The date-0 policy-shock vector nu* implementing the optimum"),
                    TableSpec(name=:implementation_error_path,
                              description="Per-component residual of the stacked outcome/instrument FOC blocks"),
                    TableSpec(name=:counterfactual_summary,
                              description="Horizon, rel_residual, spanned flag, baseline/optimal loss and FOC norm")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_optimal(route; kw...)),
        ))
    end
    for route in ("var", "bvar")
        push!(specs, CommandSpec(
            path=["policy", "moments", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     OptionSpec(name="rule", type=String, default="",
                                description="Builtin counterfactual rule (see policy counterfactual)");
                     OptionSpec(name="rule-config", type=String, default="",
                                description="TOML [rule] section");
                     OptionSpec(name="loss-config", type=String, default="",
                                description="TOML [loss] section (rule XOR loss)");
                     OptionSpec(name="use-draws", type=String, default="auto",
                                choices=["auto", "on", "off"],
                                description="Propagate draws into sd bands: auto|on|off");
                     OptionSpec(name="draw-source", type=String, default="ce",
                                choices=["ce", "wold", "both"],
                                description="Uncertainty source: ce | wold | both (matching counts enforced)");
                     OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84",
                                description="Band quantiles, comma-separated in (0,1)");
                     OptionSpec(name="frequencies", type=String, default="none",
                                description="none | business-cycle (2π/32..2π/6) | lo,hi in radians (0 ≤ lo < hi ≤ π)");
                     OptionSpec(name="plot-view", type=String, default="sd",
                                choices=["sd", "corr"],
                                description="Plot panel: sd | corr");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:counterfactual_standard_deviations,
                              description="Baseline vs counterfactual standard deviation per variable, with draw bands"),
                    TableSpec(name=:counterfactual_correlations,
                              description="Baseline vs counterfactual correlation per variable pair (needs at least two variables)"),
                    TableSpec(name=:moments_summary,
                              description="Policy name, horizon, VMA tail_share, draw source and frequency band")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_moments(route; kw...)),
        ))
    end
    # ── W6/#128: OPP family (Barnichon–Mesters) ─────────────
    _OPP_SHARED = OptionSpec[
        OptionSpec(name="loss-config", type=String, default="",
                   description="TOML [loss] section (REQUIRED)");
        OptionSpec(name="n-sim", type=Int, default=2000,
                   description="Simulation draws for estimate_opp bands (0 = point only)");
        OptionSpec(name="levels", type=String, default="0.6,0.75,0.9",
                   description="Band levels — BM REVERSED polarity: rejection at the LOWER level is the conservative call");
        OptionSpec(name="normalize", type=String, default="none",
                   choices=["none", "instrument-impact"],
                   description="none | instrument-impact");
    ]
    for route in ("var", "bvar")
        push!(specs, CommandSpec(
            path=["policy", "opp", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     _OPP_SHARED...;
                     OptionSpec(name="targets", type=String, default="",
                                description="Explicit targets name=value per outcome (REQUIRED on the model-forecast route — gaps, NOT levels)");
                     OptionSpec(name="values-file", type=String, default="",
                                description="External GAP paths CSV (one column per outcome; replaces the model forecast)");
                     OptionSpec(name="sd", type=String, default="",
                                description="External route: per-outcome forecast sd (BM damped covariance)");
                     OptionSpec(name="rho", type=Float64, default=0.9,
                                description="External route: BM damping rho in Σ[j,k]=sd_j·sd_k·ρ^|j−k|");
                     OptionSpec(name="cross-corr-file", type=String, default="",
                                description="External route: full covariance CSV — IGNORES --sd/--rho upstream (mutually exclusive)");
                     OptionSpec(name="min-sd", type=Float64, default=0.0,
                                description="External route: sd floor (warns when it binds)");
                     OptionSpec(name="origin", type=String, default="",
                                description="Forecast origin label, e.g. 2008M4");
                     OptionSpec(name="instrument-path", type=String, default="",
                                description="Announced instrument path, comma H values (REQUIRED with --constraints-file)");
                     OptionSpec(name="constraints-file", type=String, default="",
                                description="TOML [[constraint]] tables → constrained OPP (named deliberately: --conditions is swallowed pre-dispatch)");
                     OptionSpec(name="method", type=String, default="auto",
                                choices=["auto", "slsqp", "projection"],
                                description="Constrained solver: auto | slsqp | projection (crude floor-only fallback)");
                     OptionSpec(name="plot-view", type=String, default="delta",
                                choices=["delta", "paths"],
                                description="Plot panel: delta | paths");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="matched-draws",
                            description="Pair draw d across sources (independent=false; equal counts enforced)"),
                   FlagSpec(name="interp-quarterly",
                            description="External route: interpolate annual SEP paths to quarterly"),
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:opp_recommendation_delta,
                              description="Recommended policy perturbation delta and gradient by horizon, with bands and rejections"),
                    TableSpec(name=:objective_gap_paths,
                              description="Objective gap per outcome and horizon, before vs after the perturbation"),
                    TableSpec(name=:instrument_paths_announced_vs_recommended,
                              description="Announced vs recommended instrument path by horizon (when instrument paths are available)"),
                    TableSpec(name=:opp_summary,
                              description="Baseline/OPP loss, horizon, forecast origin, failure count and constrained-solver diagnostics")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_opp(route; kw...)),
        ))
        push!(specs, CommandSpec(
            path=["policy", "opp-sequence", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     OptionSpec(name="loss-config", type=String, default="",
                                description="TOML [loss] section (REQUIRED)");
                     OptionSpec(name="forecasts-dir", type=String, default="",
                                description="Directory of per-date GAP-path CSVs (sorted filenames = dates; REQUIRED)");
                     OptionSpec(name="sd", type=String, default="",
                                description="Per-outcome forecast sd shared across dates");
                     OptionSpec(name="rho", type=Float64, default=0.9,
                                description="BM damping rho");
                     OptionSpec(name="n-sim", type=Int, default=0,
                                description="Simulation draws per date (compounds cost)");
                     OptionSpec(name="levels", type=String, default="0.6,0.75,0.9",
                                description="Band levels (BM reversed polarity)");
                     OptionSpec(name="normalize", type=String, default="none",
                                choices=["none", "instrument-impact"],
                                description="none | instrument-impact");
                     OptionSpec(name="plot-view", type=String, default="fan",
                                choices=["fan", "decomposition"],
                                description="Plot panel: fan | decomposition");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="matched-draws",
                            description="Pair draw d across sources (independent=false)"),
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:opp_sequence_delta_by_date,
                              description="Recommended perturbation delta per forecast date and shock"),
                    TableSpec(name=:opp_revision_decomposition,
                              description="Per-date split of the revision into news, preference and aging components"),
                    TableSpec(name=:opp_sequence_summary,
                              description="Sequence span, loss path and draw/failure counts across dates")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_opp_sequence(route; kw...)),
        ))
    end
    # ── W7/#129: structural routes ──────────────────────────
    _FMT = OptionSpec[
        OptionSpec(name="output", short="o", type=String, default="",
                   description="Export results to file");
        OptionSpec(name="format", short="f", type=String, default="table",
                   choices=["table", "csv", "json"], description="table|csv|json");
    ]
    _BEH = OptionSpec[
        OptionSpec(name="behavioral-m", type=Float64, default=NaN,
                   description="Cognitive discounting m in [0,1] (behavioral operator; square menus only)");
        OptionSpec(name="behavioral-theta", type=Float64, default=NaN,
                   description="Sticky-expectations theta in [0,1]");
    ]
    push!(specs, CommandSpec(
        path=["policy", "news", "dsge"],
        summary="DSGE model file (TOML or .jl ModelSpec)",
        args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                      description="DSGE model file (TOML or .jl ModelSpec)")],
        options=[OptionSpec(name="policy-shock", type=String, default="",
                            description="Exogenous shock the news menu perturbs (REQUIRED)");
                 OptionSpec(name="outcomes", type=String, default="",
                            description="name=model_variable map (SYMBOLS; REQUIRED)");
                 OptionSpec(name="instruments", type=String, default="",
                            description="name=model_variable map");
                 OptionSpec(name="horizon", type=Int, default=100,
                            description="News horizon H (one QZ of dimension n+H−1 — MW use 100)");
                 OptionSpec(name="solver", type=String, default="gensys",
                            choices=["gensys", "klein", "blanchard-kahn"],
                            description="Linear solver (nonlinear menus unsupported upstream)");
                 OptionSpec(name="chunk", type=Int, default=0,
                            description="Shock columns per solve (0 = all at once)");
                 _BEH...; _FMT...],
        flags=FlagSpec[],   # PolicyCausalEffects has no plot recipe
        tables=[TableSpec(name=:policy_causal_effects_menu,
                          description="Square DSGE news menu by outcome, instrument and horizon"),
                TableSpec(name=:policy_causal_effects_summary,
                          description="Menu shape, normalization and solver diagnostics")],
        category="policy",
        handler=wrap_legacy((; kw...) -> _policy_news("dsge"; kw...)),
    ))
    push!(specs, CommandSpec(
        path=["policy", "news", "ha"],
        summary="HA model (builtin name or .jl HA ModelSpec)",
        args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                      description="HA model (builtin name or .jl HA ModelSpec)")],
        options=[OptionSpec(name="outcomes", type=String, default="",
                            description="name=model_variable map (SYMBOLS; REQUIRED)");
                 OptionSpec(name="instruments", type=String, default="",
                            description="name=model_variable map (default rate=r)");
                 OptionSpec(name="horizon", type=Int, default=100,
                            description="News horizon H");
                 OptionSpec(name="t-horizon", type=Int, default=300,
                            description="Sequence-space truncation (≥ H; < H+50 warns)");
                 OptionSpec(name="rule-closure", type=String, default="administered",
                            choices=["administered", "market"],
                            description="administered | market (market: huggett-only; the wedge is exactly neutral there BY CONSTRUCTION)");
                 OptionSpec(name="dx", type=Float64, default=1e-4,
                            description="Finite-difference step");
                 _BEH...; _FMT...],
        flags=FlagSpec[],
        tables=[TableSpec(name=:policy_causal_effects_menu,
                          description="HA news menu by outcome, instrument and horizon, built from sequence-space jacobians"),
                TableSpec(name=:policy_causal_effects_summary,
                          description="Menu shape, normalization and rule-closure diagnostics")],
        category="policy",
        handler=wrap_legacy((; kw...) -> _policy_news("ha"; kw...)),
    ))
    push!(specs, CommandSpec(
        path=["policy", "jacobian", "ha"],
        summary="HA model (builtin name or .jl HA ModelSpec)",
        args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                      description="HA model (builtin name or .jl HA ModelSpec)")],
        options=[OptionSpec(name="input", type=String, default="r",
                            choices=["r", "w"], description="Price input: r | w");
                 OptionSpec(name="jac-output", type=String, default="",
                            description="Household aggregate to differentiate, e.g. C or A (REQUIRED)");
                 OptionSpec(name="t-horizon", type=Int, default=300,
                            description="Jacobian dimension T (the table is T² rows)");
                 OptionSpec(name="dx", type=Float64, default=1e-4,
                            description="Finite-difference step");
                 _FMT...],
        flags=FlagSpec[],   # bare Matrix upstream — no report/plot
        tables=[TableSpec(name=:sequence_space_jacobian,
                          description="Sequence-space household jacobian in tidy row/column/value form")],
        category="policy",
        handler=wrap_legacy((; kw...) -> _policy_jacobian(; kw...)),
    ))
    for route in ("var", "bvar")
        push!(specs, CommandSpec(
            path=["policy", "history", route],
            summary="Path to CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                          description="Path to CSV data file")],
            options=[_POLICY_COMMON...; _policy_route_options(route)...;
                     OptionSpec(name="t-range", type=String, default="",
                                description="Observation window lo:hi to re-run under the rule (REQUIRED; length ≤ H−1)");
                     OptionSpec(name="rule", type=String, default="",
                                description="Builtin counterfactual rule");
                     OptionSpec(name="rule-config", type=String, default="",
                                description="TOML [rule] section");
                     OptionSpec(name="loss-config", type=String, default="",
                                description="TOML [loss] section (rule XOR loss)");
                     OptionSpec(name="use-draws", type=String, default="auto",
                                choices=["auto", "on", "off"],
                                description="Propagate draws into bands");
                     OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84",
                                description="Band quantiles in (0,1)");
                     OptionSpec(name="normalize", type=String, default="none",
                                choices=["none", "instrument-impact"],
                                description="none | instrument-impact");
                     PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:counterfactual_history,
                              description="Realized vs counterfactual value per date and variable, with bands when propagated"),
                    TableSpec(name=:history_summary,
                              description="Rule, window, spanned flag and draw counts for the historical re-run")],
            category="policy",
            handler=wrap_legacy((; kw...) -> _policy_history(route; kw...)),
        ))
    end
    push!(specs, CommandSpec(
        path=["policy", "spanning", "var"],
        summary="Path to CSV data file",
        args=[ArgSpec(name="data", type=String, required=true, default=nothing,
                      description="Path to CSV data file"),
              ArgSpec(name="model", type=String, required=true, default=nothing,
                      description="DSGE model file for the FULL news menu")],
        options=[_POLICY_COMMON...;
                 OptionSpec(name="lags", short="p", type=Int, default=nothing,
                            description="VAR lag order (default AIC)");
                 OptionSpec(name="replications", type=Int, default=0,
                            description="Bootstrap draws on the empirical menu");
                 OptionSpec(name="nonpolicy-shock", type=String, default="",
                            description="Baseline non-policy shock (REQUIRED)");
                 OptionSpec(name="model-outcomes", type=String, default="",
                            description="name=model_variable map for the news menu (SAME names as --outcomes)");
                 OptionSpec(name="model-instruments", type=String, default="",
                            description="name=model_variable map");
                 OptionSpec(name="policy-shock", type=String, default="",
                            description="Model shock behind the full news menu (REQUIRED)");
                 OptionSpec(name="solver", type=String, default="gensys",
                            choices=["gensys", "klein", "blanchard-kahn"],
                            description="Linear DSGE solver");
                 OptionSpec(name="rule", type=String, default="",
                            description="Builtin counterfactual rule");
                 OptionSpec(name="rule-config", type=String, default="",
                            description="TOML [rule] section");
                 OptionSpec(name="tol", type=Float64, default=0.1,
                            description="Spanned-verdict tolerance on gap_rel");
                 OptionSpec(name="n-sim", type=Int, default=200,
                            description="Draw propagation for gap bands");
                 OptionSpec(name="quantiles", type=String, default="0.16,0.5,0.84",
                            description="Band quantiles in (0,1)");
                 PLOT_OPTIONS...],
        flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
        tables=[TableSpec(name=:spanning_thin_vs_full_counterfactual_paths,
                          description="Counterfactual path per variable and horizon under the thin empirical menu vs the full model menu"),
                TableSpec(name=:spanning_verdict,
                          description="Relative gap between the two menus against --tol, and the spanned verdict")],
        category="policy",
        handler=wrap_legacy((; kw...) -> _policy_spanning(; kw...)),
    ))
    push!(specs, CommandSpec(
        path=["policy", "sufficiency", "dsge"],
        summary="DSGE model file (TOML or .jl ModelSpec)",
        args=[ArgSpec(name="model", type=String, required=true, default=nothing,
                      description="DSGE model file (TOML or .jl ModelSpec)")],
        options=[OptionSpec(name="observables", type=String, default="",
                            description="Comma-separated model variables the econometrician sees (REQUIRED)");
                 OptionSpec(name="horizon", type=Int, default=40,
                            description="FEV comparison horizon");
                 OptionSpec(name="method", type=String, default="gensys",
                            description="DSGE solve method");
                 _FMT...; PLOT_OPTIONS...],
        flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
        tables=[TableSpec(name=:forecast_sufficiency_fev_ratios,
                          description="Forecast-error-variance ratio per observable and horizon"),
                TableSpec(name=:sufficiency_summary,
                          description="Observable set, horizon and the overall sufficiency verdict")],
        category="policy",
        handler=wrap_legacy((; kw...) -> _policy_sufficiency(; kw...)),
    ))
    specs = with_default_csv_kinds(specs)
    register!(specs)
    return build_node("policy", specs;
        description="Policy counterfactuals: menus (empirical + structural), rule counterfactuals, optimal policy, moments, OPP, histories and diagnostics")
end
