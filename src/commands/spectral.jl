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

# Spectral analysis — registry pilot (P2-1 / C021)

function spectral_specs()::Vector{CommandSpec}
    data_arg = [ArgSpec(name="data", description="Path to CSV data file")]
    plot_opts = [OUTPUT_OPTIONS; PLOT_OPTIONS]
    plot_flags = copy(PLOT_FLAGS)

    return [
        CommandSpec(
            path=["spectral", "acf"],
            summary="Autocorrelation / partial autocorrelation / cross-correlation",
            args=data_arg,
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1,
                           description="Column index (1-based)"),
                OptionSpec(name="max-lag", type=Int, default=nothing,
                           description="Maximum lag (default: min(20, T-1))"),
                OptionSpec(name="ccf-with", type=Int, default=nothing,
                           description="Column index for cross-correlation"),
                plot_opts...,
            ],
            flags=plot_flags,
            tables=[
                TableSpec(name=:acf_pacf,
                          description="Autocorrelation, partial autocorrelation, Ljung-Box Q and p-value by lag"),
                TableSpec(name=:cross_correlation,
                          description="Cross-correlation by lag against --ccf-with (only with that option)"),
            ],
            category="spectral",
            handler=wrap_legacy(_spectral_acf),
        ),
        CommandSpec(
            path=["spectral", "periodogram"],
            summary="Raw periodogram",
            args=data_arg,
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1,
                           description="Column index (1-based)"),
                plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:periodogram,
                              description="Raw periodogram power by Fourier frequency")],
            category="spectral",
            handler=wrap_legacy(_spectral_periodogram),
        ),
        CommandSpec(
            path=["spectral", "density"],
            summary="Spectral density estimation",
            args=data_arg,
            options=[
                OptionSpec(name="column", short="c", type=Int, default=1,
                           description="Column index (1-based)"),
                OptionSpec(name="method", short="m", type=String, default="welch",
                           choices=["periodogram", "welch", "smoothed", "ar"],
                           description="periodogram|welch|smoothed|ar"),
                OptionSpec(name="bandwidth", type=Float64, default=nothing,
                           description="Smoothing bandwidth"),
                plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:spectral_density,
                              description="Estimated spectral density with confidence band by frequency")],
            category="spectral",
            handler=wrap_legacy(_spectral_density),
        ),
        CommandSpec(
            path=["spectral", "cross"],
            summary="Cross-spectral analysis (coherence, phase, gain)",
            args=data_arg,
            options=[
                OptionSpec(name="var1", type=Int, default=1,
                           description="First variable column index"),
                OptionSpec(name="var2", type=Int, default=2,
                           description="Second variable column index"),
                plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:cross_spectral_analysis,
                              description="Co-/quadrature spectrum, coherence, phase and gain by frequency")],
            category="spectral",
            handler=wrap_legacy(_spectral_cross),
        ),
        CommandSpec(
            path=["spectral", "transfer"],
            summary="Filter transfer function (theoretical frequency response)",
            args=ArgSpec[],
            options=[
                OptionSpec(name="filter", type=String, default="hp",
                           choices=["hp", "bk", "hamilton", "ideal"],
                           description="hp|bk|hamilton|ideal"),
                OptionSpec(name="lambda", type=Float64, default=1600.0,
                           description="Filter parameter (e.g. HP lambda)"),
                OptionSpec(name="nobs", type=Int, default=200,
                           description="Number of observations (for frequency grid)"),
                plot_opts...,
            ],
            flags=plot_flags,
            tables=[TableSpec(name=:filter_transfer_function,
                              description="Theoretical filter gain and phase by frequency")],
            category="spectral",
            handler=wrap_legacy(_spectral_transfer),
        ),
    ]
end

function register_spectral_commands!()
    specs = with_default_csv_kinds(with_data_kinds(spectral_specs(), [:timeseries, :csv]))
    register!(specs)
    return build_node("spectral", specs;
        description="Spectral analysis: ACF/PACF, periodogram, spectral density, cross-spectrum, transfer function")
end

# --------------------------------------------------------------------------
# Handlers (legacy kwargs — wrapped via wrap_legacy for the registry pilot)
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

    _status("ACF/PACF: $(vnames[column])  (T = $(length(y)))")
    _status()

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
        _status()
        output_result(ccf_df; format=Symbol(format), output="",
            title="CCF: $(vnames[column]) x $(vnames[ccf_with])", key="cross_correlation")
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

    _status("Periodogram: $(vnames[column])  (T = $(length(y)))")
    _status()

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

    _status("Spectral Density ($(method)): $(vnames[column])  (T = $(length(y)))")
    _status()

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

    _status("Cross-Spectrum: $(vnames[var1]) x $(vnames[var2])  (T = $(length(y)))")
    _status()

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

    _status("Transfer Function: $(filter) filter  (lambda = $lambda, T = $nobs)")
    _status()

    tf_df = DataFrame(
        Frequency = round.(result.freq; digits=6),
        Gain      = round.(result.gain; digits=6),
        Phase     = round.(result.phase; digits=6),
    )
    output_result(tf_df; format=Symbol(format), output=output, title="Filter Transfer Function")
    _maybe_plot(result; plot=plot, plot_save=plot_save)
    return result
end
