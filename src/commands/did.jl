# Friedman-cli — macroeconometric analysis from the terminal
# Copyright (C) 2026 Wookyung Chung <chung@friedman.jp>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# ─── DID & Event Study LP Commands ──────────────────────────────

# ─── Common option builders ─────────────────────────────────────

const _DID_PANEL_OPTIONS = [
    OptionSpec(name="id-col", type=String, default="", description="Panel unit ID column (default: first column)"),
    OptionSpec(name="time-col", type=String, default="", description="Time column (default: second column)"),
]

function _did_panel_cols(df, id_col::String, time_col::String)
    cols = names(df)
    id = isempty(id_col) ? cols[1] : id_col
    tc = isempty(time_col) ? cols[2] : time_col
    return id, tc
end

# ─── Handlers ────────────────────────────────────────────────────

function _did_estimate(; data::String, outcome::String, treatment::String,
        method::String="twfe", id_col::String="", time_col::String="",
        leads::Int=0, horizon::Int=5, covariates::String="",
        control_group::String="never_treated", cluster::String="unit",
        conf_level::Float64=0.95, n_boot::Int=200,
        base_period::String="varying",
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    covs = isempty(covariates) ? String[] : String.(split(covariates, ","))

    result = estimate_did(pd, outcome, treatment;
        method=Symbol(method), leads=leads, horizon=horizon,
        covariates=covs, control_group=Symbol(control_group),
        cluster=Symbol(cluster), conf_level=conf_level, n_boot=n_boot,
        base_period=Symbol(base_period), _fwd_seed()...)

    # C051: DIDResult is deliberately NOT rendered via DataFrame(model)/long_table — the
    # event-time ATT summary (plus the optional group-time ATT block below) is a
    # domain-specific report, not a coefficient table or an array-valued IRF/forecast, so
    # a principled exception (like the volatility forecast variance|volatility table).
    att_df = DataFrame(
        Event_Time = result.event_times,
        ATT = round.(result.att; digits=6),
        SE = round.(result.se; digits=6),
        CI_Lower = round.(result.ci_lower; digits=6),
        CI_Upper = round.(result.ci_upper; digits=6)
    )
    fmt = Symbol(lowercase(format))
    output_result(att_df; format=fmt, output=output,
        title="DID Estimation — $(uppercase(method))", key="did_estimation")

    _status()
    _status_styled("  Overall ATT: "; bold=true)
    _status("$(round(result.overall_att; digits=6))  (SE: $(round(result.overall_se; digits=6)))")
    _status_styled("  Method: "; bold=true)
    _status(method)
    _status_styled("  N: "; bold=true)
    _status("$(result.n_obs) obs, $(result.n_groups) groups ($(result.n_treated) treated, $(result.n_control) control)")

    if !isnothing(result.group_time_att) && !isnothing(result.cohorts)
        _status()
        gt_df = DataFrame(result.group_time_att, ["t=$(t)" for t in result.event_times])
        insertcols!(gt_df, 1, :Cohort => result.cohorts)
        output_result(gt_df; format=fmt, output="",
            title="Group-Time ATT (Callaway-Sant'Anna)")
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
end

function _did_event_study(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        leads::Int=3, horizon::Int=5, lags::Int=4,
        covariates::String="", cluster::String="unit",
        conf_level::Float64=0.95,
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    covs = isempty(covariates) ? String[] : String.(split(covariates, ","))

    result = estimate_event_study_lp(pd, outcome, treatment, horizon;
        leads=leads, lags=lags, covariates=covs,
        cluster=Symbol(cluster), conf_level=conf_level)

    coef_df = DataFrame(
        Event_Time = result.event_times,
        Coefficient = round.(result.coefficients; digits=6),
        SE = round.(result.se; digits=6),
        CI_Lower = round.(result.ci_lower; digits=6),
        CI_Upper = round.(result.ci_upper; digits=6)
    )
    fmt = Symbol(lowercase(format))
    output_result(coef_df; format=fmt, output=output,
        title="Event Study LP — $(result.outcome_var)", key="event_study_lp")

    _status()
    _status_styled("  N: "; bold=true)
    _status("$(result.n_obs) obs, $(result.n_groups) groups")
    _status_styled("  Lags: "; bold=true)
    _status_styled("$(result.lags)  ")
    _status_styled("Leads: "; bold=true)
    _status_styled("$(result.leads)  ")
    _status_styled("Horizon: "; bold=true)
    _status("$(result.horizon)")

    _maybe_plot(result; plot=plot, plot_save=plot_save)
end

function _did_lp_did(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        horizon::Int=5, pre_window::Int=3, post_window::Int=0,
        ylags::Int=0, dylags::Int=0,
        covariates::String="", cluster::String="unit",
        conf_level::Float64=0.95,
        pmd::String="", nonabsorbing::String="",
        reweight::Bool=false, nocomp::Bool=false,
        notyet::Bool=false, nevertreated::Bool=false,
        firsttreat::Bool=false, oneoff::Bool=false,
        only_pooled::Bool=false, only_event::Bool=false,
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    covs = isempty(covariates) ? String[] : String.(split(covariates, ","))

    # Parse pmd: empty→nothing, "ccs"→:ccs, "ipw"→:ipw, integer string→Int
    pmd_val::Union{Nothing,Symbol,Int} = if isempty(pmd)
        nothing
    elseif pmd in ("ccs", "ipw")
        Symbol(pmd)
    else
        parse(Int, pmd)
    end

    # Parse nonabsorbing: empty→nothing, else→parse Int
    nonabs_val = isempty(nonabsorbing) ? nothing : parse(Int, nonabsorbing)

    pw = post_window == 0 ? horizon : post_window

    result = estimate_lp_did(pd, outcome, treatment, horizon;
        pre_window=pre_window, post_window=pw,
        ylags=ylags, dylags=dylags,
        covariates=covs, nonabsorbing=nonabs_val,
        notyet=notyet, nevertreated=nevertreated,
        firsttreat=firsttreat, oneoff=oneoff,
        pmd=pmd_val, reweight=reweight, nocomp=nocomp,
        cluster=Symbol(cluster), conf_level=conf_level,
        only_pooled=only_pooled, only_event=only_event)

    coef_df = DataFrame(
        Event_Time = result.event_times,
        Coefficient = round.(result.coefficients; digits=6),
        SE = round.(result.se; digits=6),
        CI_Lower = round.(result.ci_lower; digits=6),
        CI_Upper = round.(result.ci_upper; digits=6),
        N_obs = result.nobs_per_horizon
    )
    fmt = Symbol(lowercase(format))
    output_result(coef_df; format=fmt, output=output,
        title="LP-DiD (Dube et al. 2023) — $(result.outcome_var)", key="lp_did_dube_et_al_2023")

    _status()
    _status_styled("  Specification: "; bold=true)
    _status(result.specification)
    _status_styled("  N: "; bold=true)
    _status("$(result.T_obs) obs, $(result.n_groups) groups")
    _status_styled("  Window: "; bold=true)
    _status("pre=$(result.pre_window), post=$(result.post_window)")

    if !isnothing(result.pooled_post)
        pp = result.pooled_post
        _status()
        _status_styled("  Pooled post-treatment: "; bold=true)
        _status("coef=$(round(pp.coef; digits=6))  SE=$(round(pp.se; digits=6))  CI=[$(round(pp.ci_lower; digits=6)), $(round(pp.ci_upper; digits=6))]")
    end
    if !isnothing(result.pooled_pre)
        pp = result.pooled_pre
        _status_styled("  Pooled pre-treatment:  "; bold=true)
        _status("coef=$(round(pp.coef; digits=6))  SE=$(round(pp.se; digits=6))  CI=[$(round(pp.ci_lower; digits=6)), $(round(pp.ci_upper; digits=6))]")
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
end

function _did_test_bacon(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    result = bacon_decomposition(pd, outcome, treatment)

    dec_df = DataFrame(
        Comparison = String.(result.comparison_type),
        Cohort_i = result.cohort_i,
        Cohort_j = result.cohort_j,
        Estimate = round.(result.estimates; digits=6),
        Weight = round.(result.weights; digits=6)
    )
    fmt = Symbol(lowercase(format))
    output_result(dec_df; format=fmt, output=output,
        title="Bacon Decomposition (Goodman-Bacon 2021)")

    _status()
    _status_styled("  Overall ATT (TWFE): "; bold=true)
    _status(round(result.overall_att; digits=6))

    _maybe_plot(result; plot=plot, plot_save=plot_save)
end

function _did_test_pretrend(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        leads::Int=3, horizon::Int=5, lags::Int=4,
        cluster::String="unit", conf_level::Float64=0.95,
        method::String="did", did_method::String="twfe",
        output::String="", format::String="table")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    if method == "event-study"
        est = estimate_event_study_lp(pd, outcome, treatment, horizon;
            leads=leads, lags=lags, cluster=Symbol(cluster), conf_level=conf_level)
        result = pretrend_test(est)
    else
        est = estimate_did(pd, outcome, treatment;
            method=Symbol(did_method), leads=leads, horizon=horizon,
            cluster=Symbol(cluster), conf_level=conf_level, _fwd_seed()...)
        result = pretrend_test(est)
    end

    output_kv([
        "Test type" => String(result.test_type),
        "F-statistic" => round(result.statistic; digits=4),
        "p-value" => round(result.pvalue; digits=4),
        "Degrees of freedom" => result.df,
        "Verdict" => result.pvalue > 0.05 ?
            "Cannot reject parallel trends (p > 0.05)" :
            "Reject parallel trends (p ≤ 0.05)"
    ]; format=format, output=output, title="Pre-Trend Test")
end

function _did_test_negweight(; data::String, treatment::String,
        id_col::String="", time_col::String="",
        output::String="", format::String="table")

    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    result = negative_weight_check(pd, treatment)

    fmt = Symbol(lowercase(format))
    output_kv([
        "Negative weights found" => result.has_negative_weights ? "yes" : "no",
        "Number of negative weights" => result.n_negative,
        "Total negative weight" => round(result.total_negative_weight; digits=6),
    ]; format=format, output=output, title="Negative Weight Check (de Chaisemartin-D'Haultfoeuille 2020)")

    if result.has_negative_weights && !isempty(result.cohort_time_pairs)
        wt_df = DataFrame(
            Cohort = [p[1] for p in result.cohort_time_pairs],
            Time = [p[2] for p in result.cohort_time_pairs],
            Weight = round.(result.weights; digits=6)
        )
        _status()
        output_result(wt_df; format=fmt, output="",
            title="Weight Details")
    end
end

function _did_test_honest(; data::String, outcome::String, treatment::String,
        id_col::String="", time_col::String="",
        mbar::Float64=1.0,
        leads::Int=3, horizon::Int=5, lags::Int=4,
        cluster::String="unit", conf_level::Float64=0.95,
        method::String="did", did_method::String="twfe",
        output::String="", format::String="table",
        plot::Bool=false, plot_save::String="")

    isempty(outcome) && error("--outcome is required")
    isempty(treatment) && error("--treatment is required")

    df = load_data(data)
    id, tc = _did_panel_cols(df, id_col, time_col)
    pd = _load_panel_for_did(data, id, tc)

    if method == "event-study"
        est = estimate_event_study_lp(pd, outcome, treatment, horizon;
            leads=leads, lags=lags, cluster=Symbol(cluster), conf_level=conf_level)
        result = honest_did(est; Mbar=mbar, conf_level=conf_level)
    else
        est = estimate_did(pd, outcome, treatment;
            method=Symbol(did_method), leads=leads, horizon=horizon,
            cluster=Symbol(cluster), conf_level=conf_level, _fwd_seed()...)
        result = honest_did(est; Mbar=mbar, conf_level=conf_level)
    end

    hon_df = DataFrame(
        Event_Time = result.post_event_times,
        ATT = round.(result.post_att; digits=6),
        Robust_CI_Lower = round.(result.robust_ci_lower; digits=6),
        Robust_CI_Upper = round.(result.robust_ci_upper; digits=6),
        Original_CI_Lower = round.(result.original_ci_lower; digits=6),
        Original_CI_Upper = round.(result.original_ci_upper; digits=6)
    )
    fmt = Symbol(lowercase(format))
    output_result(hon_df; format=fmt, output=output,
        title="HonestDiD Sensitivity (Rambachan-Roth 2023, M̄=$(mbar))",
        key="honestdid_sensitivity_rambachan_roth_2023")

    _status()
    _status_styled("  Breakdown value: "; bold=true)
    _status(round(result.breakdown_value; digits=4))

    _maybe_plot(result; plot=plot, plot_save=plot_save)
end

# ─── Command Registration ───────────────────────────────────────

function did_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["did", "estimate"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                OptionSpec(name="method", type=String, default="twfe", description="twfe|cs|sa|bjs|dcdh"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="leads", type=Int, default=0, description="Pre-treatment periods"),
                OptionSpec(name="horizon", type=Int, default=5, description="Post-treatment periods"),
                OptionSpec(name="covariates", type=String, default="", description="Comma-separated covariate column names"),
                OptionSpec(name="control-group", type=String, default="never_treated", description="never_treated|not_yet_treated"),
                OptionSpec(name="cluster", type=String, default="unit", description="unit|time|twoway"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="n-boot", type=Int, default=200, description="Bootstrap replications (dcdh only)"),
                OptionSpec(name="base-period", type=String, default="varying", description="varying|universal (Callaway-Sant'Anna only)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[
                TableSpec(name=:did_estimation,
                          description="ATT with standard error and confidence band by event time"),
                TableSpec(name=:group_time_att_callaway_sant_anna,
                          description="Cohort-by-event-time ATT matrix (Callaway-Sant'Anna estimators only)"),
            ],
            category="did",
            handler=wrap_legacy(_did_estimate),
        ),
        CommandSpec(
            path=["did", "event-study"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="leads", type=Int, default=3, description="Pre-treatment leads"),
                OptionSpec(name="horizon", type=Int, default=5, description="Post-treatment horizon"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Control lags"),
                OptionSpec(name="covariates", type=String, default="", description="Comma-separated covariate column names"),
                OptionSpec(name="cluster", type=String, default="unit", description="unit|time|twoway"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:event_study_lp,
                              description="Local-projection event-study coefficient, SE and confidence band by event time")],
            category="did",
            handler=wrap_legacy(_did_event_study),
        ),
        CommandSpec(
            path=["did", "lp-did"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="horizon", type=Int, default=5, description="Post-treatment horizon"),
                OptionSpec(name="pre-window", type=Int, default=3, description="Pre-treatment window"),
                OptionSpec(name="post-window", type=Int, default=0, description="Post-treatment window (0 = use horizon)"),
                OptionSpec(name="ylags", type=Int, default=0, description="Outcome lags"),
                OptionSpec(name="dylags", type=Int, default=0, description="Differenced outcome lags"),
                OptionSpec(name="covariates", type=String, default="", description="Comma-separated covariate column names"),
                OptionSpec(name="cluster", type=String, default="unit", description="unit|time|twoway"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="pmd", type=String, default="", description="Pre-treatment matching: ccs|ipw|<integer>"),
                OptionSpec(name="nonabsorbing", type=String, default="", description="Non-absorbing treatment (integer periods)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser"),
                FlagSpec(name="reweight", description="Reweight observations"),
                FlagSpec(name="nocomp", description="No composition adjustment"),
                FlagSpec(name="notyet", description="Use not-yet-treated as controls"),
                FlagSpec(name="nevertreated", description="Use never-treated as controls"),
                FlagSpec(name="firsttreat", description="Use first-treatment timing"),
                FlagSpec(name="oneoff", description="One-off treatment specification"),
                FlagSpec(name="only-pooled", description="Only report pooled estimates"),
                FlagSpec(name="only-event", description="Only report event-time estimates")
            ],
            tables=[TableSpec(name=:lp_did_dube_et_al_2023,
                              description="LP-DiD coefficient, SE, confidence band and observation count by event time")],
            category="did",
            handler=wrap_legacy(_did_lp_did),
        ),
        CommandSpec(
            path=["did", "test", "bacon"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:bacon_decomposition_goodman_bacon_2021,
                              description="Estimate and weight of every 2x2 DiD comparison by cohort pair and type")],
            category="did",
            handler=wrap_legacy(_did_test_bacon),
        ),
        CommandSpec(
            path=["did", "test", "pretrend"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="leads", type=Int, default=3, description="Pre-treatment leads"),
                OptionSpec(name="horizon", type=Int, default=5, description="Post-treatment horizon"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Control lags (event-study only)"),
                OptionSpec(name="cluster", type=String, default="unit", description="unit|time|twoway"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="method", type=String, default="did", description="did|event-study"),
                OptionSpec(name="did-method", type=String, default="twfe", description="twfe|cs|sa|bjs|dcdh (did method only)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[TableSpec(name=:pre_trend_test,
                              description="Joint pre-trend test type, statistic, p-value, degrees of freedom and verdict")],
            category="did",
            handler=wrap_legacy(_did_test_pretrend),
        ),
        CommandSpec(
            path=["did", "test", "negweight"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"])
            ],
            flags=FlagSpec[],
            tables=[
                TableSpec(name=:negative_weight_check_de_chaisemartin_d_haultfoeuille_2020,
                          description="Whether TWFE assigns negative weights, how many, and their total"),
                TableSpec(name=:weight_details,
                          description="The offending cohort-time cells and their weights (only when negative weights exist)"),
            ],
            category="did",
            handler=wrap_legacy(_did_test_negweight),
        ),
        CommandSpec(
            path=["did", "test", "honest"],
            summary="Path to panel CSV data file",
            args=[ArgSpec(name="data", type=String, required=true, default=nothing, description="Path to panel CSV data file")],
            options=[
                OptionSpec(name="outcome", type=String, default="", description="Outcome variable column name (required)"),
                OptionSpec(name="treatment", type=String, default="", description="Treatment indicator column name (required)"),
                _DID_PANEL_OPTIONS...,
                OptionSpec(name="mbar", type=Float64, default=1.0, description="Violation bound M̄"),
                OptionSpec(name="leads", type=Int, default=3, description="Pre-treatment leads"),
                OptionSpec(name="horizon", type=Int, default=5, description="Post-treatment horizon"),
                OptionSpec(name="lags", short="p", type=Int, default=4, description="Control lags (event-study only)"),
                OptionSpec(name="cluster", type=String, default="unit", description="unit|time|twoway"),
                OptionSpec(name="conf-level", type=Float64, default=0.95, description="Confidence level"),
                OptionSpec(name="method", type=String, default="did", description="did|event-study"),
                OptionSpec(name="did-method", type=String, default="twfe", description="twfe|cs|sa|bjs|dcdh (did method only)"),
                OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
                OptionSpec(name="format", short="f", type=String, default="table", description="table|csv|json", choices=["table","csv","json"]),
                OptionSpec(name="plot-save", type=String, default="", description="Save plot to HTML file")
            ],
            flags=[
                FlagSpec(name="plot", description="Open interactive plot in browser")
            ],
            tables=[TableSpec(name=:honestdid_sensitivity_rambachan_roth_2023,
                              description="Post-treatment ATT with robust and original confidence bands by event time")],
            category="did",
            handler=wrap_legacy(_did_test_honest),
        )
    ]
end

function register_did_commands!()
    specs = with_default_csv_kinds(with_data_kinds(did_specs(), [:panel, :csv]))
    register!(specs)
    return build_node("did", specs; description="Difference-in-differences: estimation, event study LP, diagnostics")
end

