# Declarative command registry (P2-1)

Base.@kwdef struct ArgSpec
    name::String
    type::Type = String
    required::Bool = true
    default::Any = nothing
    description::String = ""
end

Base.@kwdef struct OptionSpec
    name::String
    short::String = ""
    type::Type = String
    default::Any = nothing
    choices::Union{Nothing,Vector{String}} = nothing
    description::String = ""
    since::VersionNumber = v"0.5.0"
    handle::Bool = false
end

Base.@kwdef struct FlagSpec
    name::String
    short::String = ""
    description::String = ""
end

Base.@kwdef struct TableSpec
    name::Symbol
    description::String = ""
    # W3/#138: true when one invocation can emit MULTIPLE sibling tables of this
    # kind (per-shock IRFs, per-variable HDs). The envelope data key is then
    # `<name>_<variable-slug>` and `name` is validated as a PREFIX by
    # test/tools/check_table_keys.jl; singleton tables use `name` verbatim.
    family::Bool = false
end

Base.@kwdef struct CommandSpec
    path::Vector{String}
    summary::String
    args::Vector{ArgSpec} = ArgSpec[]
    options::Vector{OptionSpec} = OptionSpec[]
    flags::Vector{FlagSpec} = FlagSpec[]
    tables::Vector{TableSpec} = TableSpec[]
    category::String = ""
    aliases::Vector{String} = String[]
    handler::Function = (ctx) -> ctx  # (ctx::CmdContext) -> Any
    data_kinds::Vector{Symbol} = Symbol[]
    model_types::Vector{Symbol} = Symbol[]
    result_types::Vector{Symbol} = Symbol[]
end

"""
Validated invocation context for a CommandSpec handler.
"""
struct CmdContext
    args::Dict{Symbol,Any}
    opts::Dict{Symbol,Any}
    flags::Dict{Symbol,Bool}
    fmt::Symbol
    output::String
    env::Envelope
    status::Function
    spec::CommandSpec
end

# ── Shared option groups (compose by name — never index-slice) ──────────

const OUTPUT_OPTIONS = [
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="Output format"),
    OptionSpec(name="output", short="o", type=String, default="",
               description="Write to file instead of stdout"),
]

const PLOT_OPTIONS = [
    OptionSpec(name="plot-save", type=String, default="",
               description="Save interactive plot to HTML file"),
]

# Model handles (P2-7 / C029) — compose onto estimate vs downstream specs
const SAVE_MODEL_OPTION = OptionSpec(
    name="save-model", type=String, default="",
    description="Save estimated model to a handle file (.jld2 native, .fmod interim)",
)
const MODEL_OPTION = OptionSpec(
    name="model", type=String, default="",
    description="Load model from a handle file (.jld2 native, .fmod interim; skip re-estimation)",
    handle=true,
)

const RESULT_OPTION = OptionSpec(
    name="result", type=String, default="",
    description="Load a result handle (skip computation)",
    handle=true,
)
const SAVE_RESULT_OPTION = OptionSpec(
    name="save-result", type=String, default="",
    description="Save the result object to a handle (.jld2 native)",
)
function with_result_handles(specs::Vector{CommandSpec})
    out = CommandSpec[]
    for s in specs
        isempty(s.result_types) ? push!(out, s) :
            push!(out, _copy_spec(s; options=vcat(s.options, [RESULT_OPTION, SAVE_RESULT_OPTION])))
    end
    return out
end

# Config ergonomics (P2-8 / C030) — append to every leaf that has --config
const CONFIG_ERGONOMICS_OPTIONS = [
    OptionSpec(name="config-json", type=String, default="",
               description="JSON object merged over --config (file < json < --set)"),
    OptionSpec(name="set", type=String, default="",
               description="Override config key=value; repeatable; dotted keys OK"),
]
const STRICT_FLAG = FlagSpec(name="strict",
                             description="Treat config schema warnings as errors (exit 4)")

"""Copy a CommandSpec, overriding any provided fields."""
function _copy_spec(s::CommandSpec; kwargs...)
    CommandSpec(
        path      = get(kwargs, :path, s.path),
        summary   = get(kwargs, :summary, s.summary),
        args      = get(kwargs, :args, s.args),
        options   = get(kwargs, :options, s.options),
        flags     = get(kwargs, :flags, s.flags),
        tables    = get(kwargs, :tables, s.tables),
        category  = get(kwargs, :category, s.category),
        aliases   = get(kwargs, :aliases, s.aliases),
        handler   = get(kwargs, :handler, s.handler),
        data_kinds   = get(kwargs, :data_kinds, s.data_kinds),
        model_types  = get(kwargs, :model_types, s.model_types),
        result_types = get(kwargs, :result_types, s.result_types),
    )
end

"""Append --config-json/--set/--strict to specs that already declare --config."""
function with_config_ergonomics(specs::Vector{CommandSpec})
    out = CommandSpec[]
    for s in specs
        has_config = any(o -> o.name == "config", s.options)
        if !has_config
            push!(out, s)
            continue
        end
        push!(out, _copy_spec(s; options=vcat(s.options, CONFIG_ERGONOMICS_OPTIONS),
                              flags=vcat(s.flags, [STRICT_FLAG])))
    end
    return out
end

"""Append option specs to every CommandSpec (by copy)."""
function with_options(specs::Vector{CommandSpec}, extra::Vector{OptionSpec})
    return [_copy_spec(s; options=vcat(s.options, extra)) for s in specs]
end

function with_data_kinds(specs::Vector{CommandSpec}, kinds::Vector{Symbol})
    return [_copy_spec(s; data_kinds=kinds) for s in specs]
end

function _has_data_slot(s::CommandSpec)
    any(a -> a.name == "data", s.args) || any(o -> o.name == "data", s.options)
end

function with_default_csv_kinds(specs::Vector{CommandSpec})
    [_copy_spec(s; data_kinds = (!isempty(s.data_kinds) || !_has_data_slot(s)) ? s.data_kinds : [:csv])
     for s in specs]
end

function with_model_types(specs::Vector{CommandSpec}, types::Vector{Symbol})
    return [_copy_spec(s; model_types=types) for s in specs]
end

function with_result_types(specs::Vector{CommandSpec}, types::Vector{Symbol})
    return [_copy_spec(s; result_types=types) for s in specs]
end

"""Set `model_types` / `result_types` per leaf from a path-keyed catalog."""
function _tag_slot_types(specs::Vector{CommandSpec},
                         catalog::Dict{Vector{String},Tuple{Vector{Symbol},Vector{Symbol}}})
    out = CommandSpec[]
    for s in specs
        if haskey(catalog, s.path)
            mt, rt = catalog[s.path]
            s = _copy_spec(s; model_types=mt, result_types=rt)
        end
        push!(out, s)
    end
    return out
end

with_save_model(specs::Vector{CommandSpec}) = with_options(specs, [SAVE_MODEL_OPTION])
with_model_option(specs::Vector{CommandSpec}) = with_options(specs, [MODEL_OPTION])

const PLOT_FLAGS = [
    FlagSpec(name="plot", description="Open interactive plot in browser"),
]

# Cross-sectional regression shared options
const REG_OPTIONS = [
    OptionSpec(name="dep", type=String, default="",
               description="Dependent variable column name (default: first numeric column)"),
    OptionSpec(name="cov-type", type=String, default="hc1",
               choices=["ols", "hc0", "hc1", "hc2", "hc3", "cluster"],
               description="ols|hc0|hc1|hc2|hc3|cluster"),
    OptionSpec(name="clusters", type=String, default="",
               description="Cluster variable column name"),
    OptionSpec(name="output", short="o", type=String, default="",
               description="Export results to file"),
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="table|csv|json"),
]

# Panel regression shared options
# W6/#108 multiplicative seasonal ARIMA. `--p` has NO default so that leaving it out means
# "select automatically" (auto_sarima), matching how `estimate arima` distinguishes the two
# modes; `--auto` forces selection even when orders are given.
const SARIMA_OPTIONS = [
    OptionSpec(name="column", short="c", type=Int, default=1, description="Column index (1-based)"),
    OptionSpec(name="p", type=Int, default=nothing, description="Non-seasonal AR order (omit to auto-select)"),
    OptionSpec(name="d", type=Int, default=0, description="Non-seasonal differencing order"),
    OptionSpec(name="q", type=Int, default=0, description="Non-seasonal MA order"),
    OptionSpec(name="P", type=Int, default=0, description="Seasonal AR order"),
    OptionSpec(name="D", type=Int, default=0, description="Seasonal differencing order"),
    OptionSpec(name="Q", type=Int, default=0, description="Seasonal MA order"),
    OptionSpec(name="s", type=Int, default=12, description="Seasonal period (12 monthly, 4 quarterly)"),
    OptionSpec(name="max-p", type=Int, default=2, description="Auto search bound for p"),
    OptionSpec(name="max-q", type=Int, default=2, description="Auto search bound for q"),
    OptionSpec(name="max-P", type=Int, default=1, description="Auto search bound for seasonal P"),
    OptionSpec(name="max-Q", type=Int, default=1, description="Auto search bound for seasonal Q"),
    OptionSpec(name="criterion", type=String, default="aic", choices=["aic", "bic"],
               description="Auto-selection criterion"),
    OptionSpec(name="method", type=String, default="css_mle",
               choices=["css_mle", "mle", "css"], description="Estimation method"),
    OptionSpec(name="max-iter", type=Int, default=500, description="Maximum optimiser iterations"),
]
const SARIMA_FLAGS = [
    FlagSpec(name="auto", description="Force automatic order selection (auto_sarima)"),
    FlagSpec(name="no-intercept", description="Exclude the intercept term"),
]

# W2/#107 count-data regression. `--offset` and `--exposure` are mutually exclusive
# (exposure is log-transformed into the offset) and the handler rejects the pair with a
# typed usage/invalid. `--irr` is a FlagSpec because its handler kwarg is a Bool — a String
# OptionSpec bound to a Bool kwarg fails on EVERY invocation (#85).
const COUNT_COMMON_OPTIONS = [
    OptionSpec(name="dep", type=String, default="",
               description="Dependent count column (default: first numeric column)"),
    OptionSpec(name="offset", type=String, default="",
               description="Offset column, already on the log scale (exclusive with --exposure)"),
    OptionSpec(name="exposure", type=String, default="",
               description="Exposure column, strictly positive; enters as log(exposure)"),
]
const COUNT_IRR_OPTIONS = [
    OptionSpec(name="conf-level", type=Float64, default=0.95,
               description="Confidence level for the incidence-rate-ratio CI (0 < level < 1)"),
    OptionSpec(name="output", short="o", type=String, default="",
               description="Export results to file"),
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="table, csv or json"),
]
const COUNT_IRR_FLAG = FlagSpec(name="irr",
    description="Also report incidence-rate ratios exp(beta) with delta-method SEs")

const PREG_OPTIONS = [
    OptionSpec(name="dep", type=String, default="", description="Dependent variable column name"),
    OptionSpec(name="indep", type=String, default="", description="Independent variables (comma-separated)"),
    OptionSpec(name="id-col", type=String, default="", description="Panel group ID column (default: first column)"),
    OptionSpec(name="time-col", type=String, default="", description="Panel time column (default: second column)"),
    OptionSpec(name="cov-type", type=String, default="cluster",
               choices=["ols", "cluster", "twoway", "driscoll-kraay"],
               description="ols|cluster|twoway|driscoll-kraay"),
    OptionSpec(name="method", short="m", type=String, default="fe", description="Estimation method"),
    OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="table|csv|json"),
]

# Bayesian DSGE shared options (replaces _bayes_common_options)
const BAYES_OPTIONS = [
    OptionSpec(name="data", short="d", type=String, default="", description="Path to CSV data file"),
    OptionSpec(name="params", type=String, default="", description="Comma-separated parameter names"),
    OptionSpec(name="priors", type=String, default="", description="Path to priors TOML file"),
    # #148 rider: choices enforced — an unknown sampler used to reach upstream as an
    # untyped ArgumentError (exit 1); now it is a usage error at parse (exit 2).
    OptionSpec(name="sampler", type=String, default="smc", choices=["smc", "smc2", "mh"],
               description="smc|smc2|mh"),
    OptionSpec(name="n-smc", type=Int, default=5000, description="SMC particles"),
    OptionSpec(name="n-particles", type=Int, default=500, description="Particle filter particles (smc2)"),
    OptionSpec(name="n-draws", type=Int, default=10000, description="Total posterior draws"),
    OptionSpec(name="burnin", type=Int, default=5000, description="Burn-in draws"),
    OptionSpec(name="ess-target", type=Float64, default=0.5, description="ESS target for resampling"),
    OptionSpec(name="observables", type=String, default="", description="Observable variable names (comma-separated)"),
    OptionSpec(name="solver", type=String, default="gensys", description="gensys|klein|perturbation"),
    OptionSpec(name="order", type=Int, default=1, description="Perturbation order (1, 2, or 3)"),
    OptionSpec(name="constraint-solver", type=String, default="",
               description="Constraint solver: nonlinearsolve|optim|nlopt|ipopt|path"),
    # W12/#114 (MEMs#339): trends in observables. Shared across the whole `dsge bayes`
    # family because every leaf re-estimates from scratch — a prefilter available only on
    # `bayes estimate` could not be carried into `bayes irf`/`fevd`/`hd`. NOT offered on the
    # frequentist `dsge estimate`: `estimate_dsge` has no such kwarg upstream.
    OptionSpec(name="prefilter", type=String, default="none",
               choices=["none", "demean", "first-difference", "linear-detrend", "hp"],
               description="Observable transform applied before estimation (Dynare `prefilter`)"),
    OptionSpec(name="hp-lambda", type=Float64, default=1600.0,
               description="HP smoothing parameter for --prefilter hp (1600 quarterly, 129600 monthly, 6.25 annual)"),
    # #148: measurement error for the DSGE likelihood. Shared across the whole family
    # for the same reason as --prefilter (every leaf re-estimates from scratch). With
    # more observables than structural shocks the likelihood is stochastically
    # singular (model/stochastic-singularity, exit 5) unless this is set.
    OptionSpec(name="measurement-error", type=String, default="none",
               description="Measurement error std devs: none|auto|comma-separated values (one per observable)"),
    OptionSpec(name="output", short="o", type=String, default="", description="Export results to file"),
    OptionSpec(name="format", short="f", type=String, default="table",
               choices=["table", "csv", "json"], description="table|csv|json"),
]

"""Replace the default of a named option in a group (by name, not index)."""
function with_default(group::Vector{OptionSpec}, name::String, default)
    out = OptionSpec[]
    found = false
    for o in group
        if o.name == name
            push!(out, OptionSpec(name=o.name, short=o.short, type=o.type,
                                  default=default, choices=o.choices,
                                  description=o.description, since=o.since,
                                  handle=o.handle))
            found = true
        else
            push!(out, o)
        end
    end
    found || error("option '$name' not found in group")
    return out
end

"""Select named options from a group (order follows `names`, not the group)."""
function select_options(group::Vector{OptionSpec}, names::String...)
    out = OptionSpec[]
    for name in names
        idx = findfirst(o -> o.name == name, group)
        isnothing(idx) && error("option '$name' not found in group")
        push!(out, group[idx])
    end
    return out
end

# Global registry of migrated specs (build_app merges with legacy nodes)
const REGISTRY = CommandSpec[]

function register!(spec::CommandSpec)
    push!(REGISTRY, spec)
    return spec
end

function register!(specs::Vector{CommandSpec})
    append!(REGISTRY, specs)
    return specs
end

