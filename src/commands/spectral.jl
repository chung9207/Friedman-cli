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

# Spectral analysis commands: acf, periodogram, density, cross, transfer

function register_spectral_commands!()
    spec_acf = LeafCommand("acf", _spectral_acf;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("max-lag"; type=Int, default=nothing, description="Maximum lag (default: min(20, T-1))"),
            Option("ccf-with"; type=Int, default=nothing, description="Column index for cross-correlation"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Autocorrelation / partial autocorrelation / cross-correlation")

    spec_periodogram = LeafCommand("periodogram", _spectral_periodogram;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Raw periodogram")

    spec_density = LeafCommand("density", _spectral_density;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("column"; short="c", type=Int, default=1, description="Column index (1-based)"),
            Option("method"; short="m", type=String, default="welch", description="periodogram|welch|smoothed|ar"),
            Option("bandwidth"; type=Float64, default=nothing, description="Smoothing bandwidth"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Spectral density estimation")

    spec_cross = LeafCommand("cross", _spectral_cross;
        args=[Argument("data"; description="Path to CSV data file")],
        options=[
            Option("var1"; type=Int, default=1, description="First variable column index"),
            Option("var2"; type=Int, default=2, description="Second variable column index"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Cross-spectral analysis (coherence, phase, gain)")

    spec_transfer = LeafCommand("transfer", _spectral_transfer;
        args=Argument[],
        options=[
            Option("filter"; type=String, default="hp", description="hp|bk|hamilton|ideal"),
            Option("lambda"; type=Float64, default=1600.0, description="Filter parameter (e.g. HP lambda)"),
            Option("nobs"; type=Int, default=200, description="Number of observations (for frequency grid)"),
            Option("output"; short="o", type=String, default="", description="Export results to file"),
            Option("format"; short="f", type=String, default="table", description="table|csv|json"),
            Option("plot-save"; type=String, default="", description="Save plot to HTML file"),
        ],
        flags=[Flag("plot"; description="Open interactive plot in browser")],
        description="Filter transfer function (theoretical frequency response)")

    subcmds = Dict{String,Union{NodeCommand,LeafCommand}}(
        "acf"         => spec_acf,
        "periodogram" => spec_periodogram,
        "density"     => spec_density,
        "cross"       => spec_cross,
        "transfer"    => spec_transfer,
    )
    return NodeCommand("spectral", subcmds,
        "Spectral analysis: ACF/PACF, periodogram, spectral density, cross-spectrum, transfer function")
end

# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

function _spectral_acf(; data::String, column::Int=1,
                        max_lag::Union{Int,Nothing}=nothing,
                        ccf_with::Union{Int,Nothing}=nothing,
                        output::String="", format::String="table",
                        plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    Y = df_to_matrix(df)
    y = Y[:, column]

    kwargs = isnothing(max_lag) ? (;) : (; maxlag=max_lag)
    result = acf(y; kwargs...)

    println("ACF/PACF: $(vnames[column])  (T = $(length(y)))")
    println()

    acf_df = DataFrame(
        Lag     = result.lags,
        ACF     = round.(result.acf; digits=6),
        PACF    = round.(result.pacf; digits=6),
        Q_stat  = round.(result.q_stats; digits=4),
        p_value = round.(result.q_pvalues; digits=4),
    )
    output_result(acf_df; format=Symbol(format), output=output, title="ACF / PACF")

    if !isnothing(ccf_with)
        z = Y[:, ccf_with]
        ccf_result = ccf(y, z; kwargs...)
        ccf_df = DataFrame(
            Lag = ccf_result.lags,
            CCF = round.(ccf_result.ccf; digits=6),
        )
        println()
        output_result(ccf_df; format=Symbol(format), output="",
            title="CCF: $(vnames[column]) x $(vnames[ccf_with])")
    end

    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_periodogram(; data::String, column::Int=1,
                                output::String="", format::String="table",
                                plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    y = df_to_matrix(df)[:, column]

    result = periodogram(y)

    println("Periodogram: $(vnames[column])  (T = $(length(y)))")
    println()

    peri_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Power     = round.(result.density; digits=6),
    )
    output_result(peri_df; format=Symbol(format), output=output, title="Periodogram")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_density(; data::String, column::Int=1,
                            method::String="welch",
                            bandwidth::Union{Float64,Nothing}=nothing,
                            output::String="", format::String="table",
                            plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    y = df_to_matrix(df)[:, column]

    kwargs = Dict{Symbol,Any}(:method => Symbol(method))
    !isnothing(bandwidth) && (kwargs[:bandwidth] = bandwidth)
    result = spectral_density(y; kwargs...)

    println("Spectral Density ($(method)): $(vnames[column])  (T = $(length(y)))")
    println()

    sd_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Density   = round.(result.density; digits=6),
        CI_Lower  = round.(result.ci_lower; digits=6),
        CI_Upper  = round.(result.ci_upper; digits=6),
    )
    output_result(sd_df; format=Symbol(format), output=output, title="Spectral Density")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_cross(; data::String, var1::Int=1, var2::Int=2,
                          output::String="", format::String="table",
                          plot::Bool=false, plot_save::String="")
    df = load_data(data)
    vnames = variable_names(df)
    Y = df_to_matrix(df)
    y = Y[:, var1]; z = Y[:, var2]

    result = cross_spectrum(y, z)

    println("Cross-Spectrum: $(vnames[var1]) x $(vnames[var2])  (T = $(length(y)))")
    println()

    cs_df = DataFrame(
        Frequency     = round.(result.freq; digits=6),
        Co_spectrum   = round.(result.co_spectrum; digits=6),
        Quad_spectrum = round.(result.quad_spectrum; digits=6),
        Coherence     = round.(result.coherence; digits=6),
        Phase         = round.(result.phase; digits=6),
        Gain          = round.(result.gain; digits=6),
    )
    output_result(cs_df; format=Symbol(format), output=output, title="Cross-Spectral Analysis")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end

function _spectral_transfer(; filter::String="hp", lambda::Float64=1600.0,
                             nobs::Int=200,
                             output::String="", format::String="table",
                             plot::Bool=false, plot_save::String="")
    result = transfer_function(Symbol(filter); lambda=lambda, nobs=nobs)

    println("Transfer Function: $(filter) filter  (lambda = $lambda, T = $nobs)")
    println()

    tf_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Gain      = round.(result.gain; digits=6),
        Phase     = round.(result.phase; digits=6),
    )
    output_result(tf_df; format=Symbol(format), output=output, title="Filter Transfer Function")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end
