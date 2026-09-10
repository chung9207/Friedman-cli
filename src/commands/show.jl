# Top-level `show` — render any loadable handle (data, model, or result)

function _show_handle(; path::String="", data::String="",
                       format::String="table", output::String="",
                       plot::Bool=false, plot_save::String="")
    p = !isempty(path) ? path : data
    isempty(p) && throw(CliError("usage/missing-arg", "show requires a handle stem"))
    obj = load_model_dispatch(resolve_stem(p; slot=:result))
    if obj isa AbstractDict
        keys_ = sort!(collect(keys(obj)))
        types = [string(nameof(typeof(obj[k]))) for k in keys_]
        output_result(DataFrame(entry=keys_, type=types); format=Symbol(format),
                      output=output, title="Bundle Entries", key="show_payload")
        _status("pass a single-object handle; bundles are not unpacked")
    else
        k = _data_kind_of(obj)
        # Only rectangular data containers have to_matrix / describe_data.
        # :io (IOData) and other kinds fall through — never to_matrix an IOData.
        if k in (:timeseries, :panel, :cross_section)
            Y = to_matrix(obj)
            vn = Vector{String}(varnames(obj))
            n_obs, n_vars = size(Y)
            summary = _status_stdout() do
                describe_data(obj)
            end
            fv = [something(findfirst(isfinite, @view Y[:, j]), 0) for j in 1:n_vars]
            lv = [something(findlast(isfinite, @view Y[:, j]), 0) for j in 1:n_vars]
            output_result(DataFrame(
                variable=vn, n=summary.n, first_valid=fv, last_valid=lv,
                mean=summary.mean, std=summary.std, min=summary.min,
                p25=summary.p25, median=summary.median, p75=summary.p75,
                max=summary.max, skewness=summary.skewness, kurtosis=summary.kurtosis,
            ); format=Symbol(format), output=output, title="Descriptive Statistics",
               key="show_payload")
        else
            tbl = try
                long_table(obj)
            catch
                try
                    DataFrame(obj)
                catch
                    nothing
                end
            end
            if tbl === nothing
                fields = String[string(n) for n in propertynames(obj)]
                values = Any[try string(getproperty(obj, Symbol(n))) catch; missing; end for n in fields]
                output_result(DataFrame(field=fields, value=values); format=Symbol(format),
                              output=output, title="Handle Fields", key="show_payload")
            else
                output_result(tbl; format=Symbol(format), output=output,
                              title="Handle", key="show_payload")
            end
        end
    end
    _maybe_plot(obj; plot=plot, plot_save=plot_save)
    return obj
end

function register_show_commands!()
    specs = [CommandSpec(
        path=["show"],
        summary="Render any loadable handle (data, model, or result)",
        args=[ArgSpec(name="path", type=String, required=true,
                      description="Handle stem or path")],
        options=[
            OptionSpec(name="output", short="o", type=String, default="",
                       description="Write to file instead of stdout"),
            OptionSpec(name="format", short="f", type=String, default="table",
                       choices=["table","csv","json"], description="table|csv|json"),
            OptionSpec(name="plot-save", type=String, default="",
                       description="Save interactive plot to HTML file"),
        ],
        flags=[FlagSpec(name="plot", description="Open interactive plot if a recipe exists")],
        tables=[TableSpec(name=:show_payload, description="Rendered payload of the handle"),
                TableSpec(name=:show_summary, description="Optional kv summary")],
        category="show",
        handler=wrap_legacy(_show_handle),
    )]
    register!(specs)
    return to_leaf(specs[1])
end
