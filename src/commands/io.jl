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

# Input-Output analysis (C049 / P5-4 / V0110 W3–W4). Wraps the MEMs `io` module:
#   sources | download | load | leontief | ghosh | multipliers | linkages |
#   key-sectors | sda | extract | footprint | baqaee-farhi |
#   bf {network,equilibrium,local,elasticities,shock-curve,wedges,misallocation} |
#   price | impact | network-stats | aggregate | balance |
#   vertical-specialization | export-decomposition | bilateral-trade
#
# IO result types are NOT Tables.jl-registered in MEMs (no `DataFrame(result)` /
# `long_table(result)`), so io leaves render via hand-built tables — a documented
# C051 fallback path (like the Arias/Uhlig/volatility leaves). Square matrices
# (Leontief/Ghosh inverses, technical/allocation coefficients) render WIDE
# (sector×sector); vector-valued results render as per-sector row tables.
#
# Every non-download leaf runs offline: with no `--data`, the bundled
# `load_example(:wiot)` Miller & Blair 2-sector fixture (with employment + CO2
# satellite accounts) is used. `io download` is the only network-touching leaf.

# ── Shared option groups ─────────────────────────────────────

"Input options shared by every analysis leaf (CSV path / bundled example / labels)."
const IO_INPUT_OPTIONS = [
    OptionSpec(name="data", type=String, default="",
               description="IO table: CSV path, :wiot (bundled example, the default), or another :example"),
    OptionSpec(name="n-sectors", type=Int, default=0,
               description="Number of sectors (CSV: Z is the first n_sectors columns)"),
    OptionSpec(name="n-fd", type=Int, default=1,
               description="Number of final-demand columns (CSV)"),
    OptionSpec(name="sectors", type=String, default="",
               description="Comma-separated sector labels (CSV; default: sector1..sectorN)"),
]

"ICIO/CSV parser selection for leaves that load an IO table from --data."
const IO_PARSER_OPTIONS = [
    OptionSpec(name="parser", type=String, default="csv",
               choices=["csv", "icio"],
               description="Input parser: csv (parse_io) | icio (OECD ICIO text)"),
]

"Baqaee–Farhi ProductionNetwork calibration (shared by every `io bf` leaf)."
const BF_NETWORK_OPTIONS = [
    OptionSpec(name="theta", type=String, default="1.0",
               description="Production substitution elasticity (scalar or per-sector comma list)"),
    OptionSpec(name="sigma", type=Float64, default=1.0,
               description="Household consumption substitution elasticity"),
    OptionSpec(name="epsilon", type=String, default="1.0",
               description="Outer VA-vs-intermediate elasticity (:two nests; scalar or comma list)"),
    OptionSpec(name="eta", type=String, default="1.0",
               description="Across-factor elasticity inside the VA bundle (:two nests; scalar or comma list)"),
    OptionSpec(name="nests", type=String, default="single",
               choices=["single", "two"],
               description="Nesting scheme: single | two"),
    OptionSpec(name="factors", type=String, default="single",
               choices=["single", "va-cats"],
               description="Factor mapping: single (sum VA rows) | va-cats (one factor per VA row)"),
    OptionSpec(name="mu", type=String, default="1.0",
               description="Sectoral markups μ≥1 (scalar or per-sector comma list; 1 = efficient)"),
]
const BF_NETWORK_FLAGS = [
    FlagSpec(name="no-check",
             description="Allow clipped negative cost shares above 1% of the row"),
]

function io_specs()::Vector{CommandSpec}
    return [
        CommandSpec(
            path=["io", "sources"],
            summary="List downloadable IO/MRIO sources (offline catalog)",
            args=ArgSpec[],
            options=copy(OUTPUT_OPTIONS),
            tables=[TableSpec(name=:io_mrio_sources,
                              description="Known IO/MRIO sources with available versions and credential requirements")],
            category="io", handler=wrap_legacy(_io_sources),
        ),
        CommandSpec(
            path=["io", "download"],
            summary="Download an IO/MRIO table (network); respects --offline",
            args=ArgSpec[],
            options=[
                OptionSpec(name="source", type=String, default="",
                           choices=["oecd", "wiod", "exiobase3", "eora26", "gloria"],
                           description="oecd|wiod|exiobase3|eora26|gloria"),
                OptionSpec(name="storage", type=String, default="",
                           description="Destination folder for downloaded archives (required)"),
                # Named --source-version, NOT --version: that is a reserved
                # pre-dispatch global (#117) and the old spelling was swallowed
                # by it on every invocation (printed the CLI version, exit 0).
                OptionSpec(name="source-version", type=String, default="",
                           description="Source version (e.g. OECD v2023)"),
                OptionSpec(name="years", type=String, default="",
                           description="Comma-separated year filter (default: all)"),
                OptionSpec(name="system", type=String, default="pxp",
                           choices=["pxp", "ixi"],
                           description="EXIOBASE product-by-product|industry-by-industry"),
                OptionSpec(name="email", type=String, default="",
                           description="Account email (EORA26 only)"),
                OptionSpec(name="password", type=String, default="",
                           description="Account password (EORA26 only)"),
                OUTPUT_OPTIONS...,
            ],
            flags=[
                FlagSpec(name="offline", description="Refuse network access (exit 6 env/network)"),
                FlagSpec(name="overwrite", description="Re-download files that already exist"),
                FlagSpec(name="no-verify", description="Skip SHA-256 checksum verification"),
            ],
            tables=[
                TableSpec(name=:download_summary,
                          description="Source, resolved version and number of files fetched"),
                TableSpec(name=:download_log,
                          description="One row per fetched file: source URL and local filename"),
            ],
            category="io", handler=wrap_legacy(_io_download),
        ),
        CommandSpec(
            path=["io", "load"],
            summary="Parse/inspect an IO table: dimensions, balance, per-sector totals",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="parser", type=String, default="csv",
                           choices=["csv", "icio"],
                           description="Input parser: csv (parse_io) | icio (OECD ICIO text; .zip needs ZipFile, deferred W9)"),
                OptionSpec(name="year", type=String, default="",
                           description="ICIO year filter"),
                OptionSpec(name="member", type=String, default="",
                           description="ICIO member filter"),
                OUTPUT_OPTIONS...],
            flags=[FlagSpec(name="no-aggregate-cn-mx",
                            description="ICIO: do not aggregate CN/MX processing units"),
                   FlagSpec(name="check",
                            description="ICIO: run parser balance checks")],
            tables=[TableSpec(name=:io_table_summary,
                              description="Sector/region/category counts, year, unit, source, extensions and total output"),
                    TableSpec(name=:sectors,
                              description="Per-sector gross output, final demand, value added and region")],
            category="io", handler=wrap_legacy(_io_load),
        ),
        CommandSpec(
            path=["io", "leontief"],
            summary="Leontief (demand-driven) representation: A and inverse L",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="matrix", type=String, default="L", choices=["L", "A", "both"],
                           description="L = Leontief inverse | A = technical coefficients | both"),
                OUTPUT_OPTIONS...],
            tables=[
                TableSpec(name=:technical_coefficients_a,
                          description="Technical coefficient matrix A, sector by sector (--matrix A|both)"),
                TableSpec(name=:leontief_inverse_l,
                          description="Leontief inverse L, sector by sector (--matrix L|both)"),
            ],
            category="io", handler=wrap_legacy(_io_leontief),
        ),
        CommandSpec(
            path=["io", "ghosh"],
            summary="Ghosh (supply-driven) representation: B and inverse G",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="matrix", type=String, default="G", choices=["G", "B", "both"],
                           description="G = Ghosh inverse | B = allocation coefficients | both"),
                OUTPUT_OPTIONS...],
            tables=[
                TableSpec(name=:allocation_coefficients_b,
                          description="Allocation coefficient matrix B, sector by sector (--matrix B|both)"),
                TableSpec(name=:ghosh_inverse_g,
                          description="Ghosh inverse G, sector by sector (--matrix G|both)"),
            ],
            category="io", handler=wrap_legacy(_io_ghosh),
        ),
        CommandSpec(
            path=["io", "multipliers"],
            summary="Sectoral output/income/employment multipliers (Type I & II)",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="kind", type=String, default="output",
                           choices=["output", "income", "employment"],
                           description="output|income|employment"),
                OptionSpec(name="type", type=String, default="I", choices=["I", "II"],
                           description="Type I (open) | Type II (household-closed)"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:io_multipliers,
                              description="Per-sector multiplier of the requested kind and type")],
            category="io", handler=wrap_legacy(_io_multipliers),
        ),
        CommandSpec(
            path=["io", "linkages"],
            summary="Backward/forward linkages + Rasmussen dispersion indices",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="forward", type=String, default="ghosh",
                           choices=["ghosh", "leontief"],
                           description="Forward linkage basis: ghosh (row sums of G) | leontief"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:linkages,
                              description="Per-sector backward and forward linkages, Rasmussen Ui/Uj and class")],
            category="io", handler=wrap_legacy(_io_linkages),
        ),
        CommandSpec(
            path=["io", "key-sectors"],
            summary="Key-sector classification (Rasmussen quadrants)",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="forward", type=String, default="ghosh",
                           choices=["ghosh", "leontief"],
                           description="Forward linkage basis: ghosh | leontief"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:key_sectors,
                              description="Per-sector key/forward/backward/weak quadrant classification")],
            category="io", handler=wrap_legacy(_io_key_sectors),
        ),
        CommandSpec(
            path=["io", "sda"],
            summary="Structural decomposition of Δoutput between two periods",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="data2", type=String, default="",
                           description="Second-period IO table (CSV path / :example; default: same as --data)"),
                OptionSpec(name="method", type=String, default="additive",
                           choices=["additive", "multiplicative"],
                           description="additive (exact, zero residual) | multiplicative"),
                OptionSpec(name="factors", type=String, default="",
                           description="Comma-separated SDA factors (kebab); omit for legacy L_effect/Y_effect"),
                OptionSpec(name="on", type=String, default="output",
                           description="output | satellite account name (emission SDA)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:structural_decomposition,
                              description="Per-sector (or per-stressor) SDA effects; legacy L_effect/Y_effect when --factors is omitted and --on is output; intensity/technology/final_demand when --on is a satellite")],
            category="io", handler=wrap_legacy(_io_sda),
        ),
        CommandSpec(
            path=["io", "extract"],
            summary="Hypothetical extraction: output loss from removing sector(s)",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="sectors-extract", type=String, default="",
                           description="Sector(s) to extract: names or 1-based indices, comma-separated (required)"),
                OptionSpec(name="mode", type=String, default="complete",
                           choices=["complete", "backward", "forward", "partial"],
                           description="Extraction variant: complete|backward|forward|partial"),
                OptionSpec(name="share", type=Float64, default=1.0,
                           description="Partial extraction share in (0, 1]"),
                OptionSpec(name="region", type=String, default="",
                           description="Extract a whole MRIO region block"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:hypothetical_extraction_loss,
                              description="Per-sector output loss caused by extracting the target sector(s)"),
                    TableSpec(name=:hypothetical_extraction_summary,
                              description="total_loss, loss_pct_go, loss_pct_gdp, mode, share")],
            category="io", handler=wrap_legacy(_io_extract),
        ),
        CommandSpec(
            path=["io", "footprint"],
            summary="Consumption-based footprint of a satellite (environmental) account",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="account", type=String, default="",
                           description="Satellite account name (default: first available, e.g. CO2)"),
                OptionSpec(name="by", type=String, default="sector",
                           choices=["sector", "region"],
                           description="sector (default) | region (MRIO production vs consumption)"),
                OUTPUT_OPTIONS...],
            flags=[FlagSpec(name="detail",
                            description="Also emit intensities (S) and emission multipliers (M=SL)")],
            tables=[TableSpec(name=:footprint,
                              description="Consumption-based footprint total per stressor"),
                    TableSpec(name=:footprint_by_sector,
                              description="Per-sector footprint contribution, one column per stressor"),
                    TableSpec(name=:regional_footprint,
                              description="Per-region production vs consumption footprint (--by region)"),
                    TableSpec(name=:intensities_s,
                              description="Direct stressor intensities S by sector (--detail only)"),
                    TableSpec(name=:emission_multipliers,
                              description="Total emission multipliers M=SL by sector (--detail only)")],
            category="io", handler=wrap_legacy(_io_footprint),
        ),
        CommandSpec(
            path=["io", "baqaee-farhi"],
            summary="Baqaee & Farhi (2019) nonlinear IO: Domar weights, centralities",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS...,
                OptionSpec(name="theta", type=Float64, default=1.0,
                           description="Production substitution elasticity (1 = Cobb-Douglas ⇒ Hulten exact)"),
                OptionSpec(name="sigma", type=Float64, default=1.0,
                           description="Consumption substitution elasticity (1 = Cobb-Douglas)"),
                OUTPUT_OPTIONS...],
            flags=[FlagSpec(name="second-order",
                            description="Also emit the second-order 'beyond Hulten' Hessian (sector×sector)")],
            tables=[TableSpec(name=:baqaee_farhi_2019_decomposition,
                              description="Per-sector Domar weight, influence, upstreamness and downstreamness"),
                    TableSpec(name=:second_order_hessian_beyond_hulten,
                              description="Second-order 'beyond Hulten' Hessian, sector by sector (--second-order only)")],
            category="io", handler=wrap_legacy(_io_baqaee_farhi),
        ),
        # ── W3/#154: Baqaee–Farhi standard-form node ──
        CommandSpec(
            path=["io", "bf", "network"],
            summary="Calibrate a Baqaee–Farhi ProductionNetwork from an IO table",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                     OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:production_network_summary,
                              description="n, M, F, nests of the calibrated network"),
                    TableSpec(name=:production_network_sectors,
                              description="Per-sector cost/revenue Domar weights and markups")],
            category="io", handler=wrap_legacy(_io_bf_network),
        ),
        CommandSpec(
            path=["io", "bf", "equilibrium"],
            summary="Exact nested-CES counterfactual equilibrium (Baqaee–Farhi 2019/2020)",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                OptionSpec(name="dlog-a", type=String, default="",
                           description="Hicks-neutral productivity shocks (scalar or comma list; default 0)"),
                OptionSpec(name="dlog-l", type=String, default="",
                           description="Factor-supply shocks (scalar or comma list; default 0)"),
                OptionSpec(name="dlog-mu", type=String, default="",
                           description="Markup shocks (scalar or comma list; default 0)"),
                OptionSpec(name="method", type=String, default="newton",
                           choices=["newton", "fixedpoint"],
                           description="Inner price solver: newton | fixedpoint"),
                OptionSpec(name="tol", type=Float64, default=1e-10,
                           description="Solver tolerance"),
                OptionSpec(name="maxiter", type=Int, default=500,
                           description="Maximum solver iterations"),
                OptionSpec(name="damping", type=Float64, default=0.5,
                           description="Quasi-Newton damping in (0, 1]"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:bf_equilibrium_summary,
                              description="dlogY, Hulten, technology/allocative split, convergence"),
                    TableSpec(name=:bf_equilibrium_sectors,
                              description="Per-sector log output and log price changes")],
            category="io", handler=wrap_legacy(_io_bf_equilibrium),
        ),
        CommandSpec(
            path=["io", "bf", "local"],
            summary="Local Hulten + second-order Hessian on a ProductionNetwork",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                OptionSpec(name="hessian", type=String, default="auto",
                           choices=["auto", "full", "none"],
                           description="Form n×n Hessian: auto (n≤500) | full | none"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="no-elasticities",
                            description="Skip the attached BFElasticities block"),
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:bf_local_hulten,
                              description="Per-sector first-order (Hulten) Domar weights"),
                    TableSpec(name=:bf_local_hessian,
                              description="Second-order Hessian of log Y in log A (--hessian full|auto)")],
            category="io", handler=wrap_legacy(_io_bf_local),
        ),
        CommandSpec(
            path=["io", "bf", "elasticities"],
            summary="Factor-price, goods-price and Domar-share incidence at the base point",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                     OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:price_incidence,
                              description="∂ log p_i / ∂ log A_j, shocked sector by column"),
                    TableSpec(name=:factor_price_incidence,
                              description="∂ log w_f / ∂ log A_j, shocked sector by column")],
            category="io", handler=wrap_legacy(_io_bf_elasticities),
        ),
        CommandSpec(
            path=["io", "bf", "shock-curve"],
            summary="One-sector productivity shock: exact vs Hulten vs second-order Taylor",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                OptionSpec(name="sector", type=String, default="",
                           description="Shocked sector: name or 1-based index (required)"),
                OptionSpec(name="range", type=String, default="-0.5,0.5",
                           description="(lo,hi) grid for Δ log A"),
                OptionSpec(name="points", type=Int, default=41,
                           description="Number of grid points (≥ 2)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:bf_shock_curve,
                              description="Δ log A grid with exact, Hulten and second-order Δ log Y")],
            category="io", handler=wrap_legacy(_io_bf_shock_curve),
        ),
        CommandSpec(
            path=["io", "bf", "wedges"],
            summary="Baqaee–Farhi (2020) Theorem 1 technology vs allocative-efficiency split",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                OptionSpec(name="dlog-a", type=String, default="",
                           description="Productivity shocks (scalar or comma list; default 0)"),
                OptionSpec(name="dlog-l", type=String, default="",
                           description="Factor-supply shocks (scalar or comma list; default 0)"),
                OptionSpec(name="dlog-mu", type=String, default="",
                           description="Markup shocks (scalar or comma list; default 0)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:bf_wedge_decomp,
                              description="dlogY, technology, allocative, factor-supply terms"),
                    TableSpec(name=:bf_wedge_domar,
                              description="Per-sector cost vs revenue Domar weights and markups")],
            category="io", handler=wrap_legacy(_io_bf_wedges),
        ),
        CommandSpec(
            path=["io", "bf", "misallocation"],
            summary="Baqaee–Farhi (2020) Prop. 5 Harberger misallocation distance",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS..., BF_NETWORK_OPTIONS...,
                OptionSpec(name="point", type=String, default="efficient",
                           choices=["efficient", "observed"],
                           description="Evaluation point: efficient | observed"),
                OptionSpec(name="hessian", type=String, default="auto",
                           choices=["auto", "full", "none"],
                           description="Form n×n H_μ: auto (n≤500) | full | none"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[BF_NETWORK_FLAGS...,
                   FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:bf_misallocation_summary,
                              description="Exact distance L, first- and second-order Harberger terms"),
                    TableSpec(name=:bf_misallocation_sectors,
                              description="Per-sector Δlog μ, Domar weight and markup")],
            category="io", handler=wrap_legacy(_io_bf_misallocation),
        ),
        # ── W4/#155: classical + MRIO ──
        CommandSpec(
            path=["io", "price"],
            summary="Leontief cost-push (or Ghosh dual) price model",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="dva", type=String, default="",
                           description="Value-added coefficient shocks: comma list or sector=value"),
                OptionSpec(name="dtax", type=String, default="",
                           description="Production-tax shocks: comma list or sector=value"),
                OptionSpec(name="mode", type=String, default="leontief",
                           choices=["leontief", "ghosh"],
                           description="leontief (cost-push dual) | ghosh (descriptive dual)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:price_model,
                              description="Per-sector price change, new price and primary-cost shock")],
            category="io", handler=wrap_legacy(_io_price),
        ),
        CommandSpec(
            path=["io", "impact"],
            summary="Final-demand impact / scenario through the Leontief inverse",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="dy", type=String, default="",
                           description="Final-demand change: comma list or sector=value (required)"),
                OptionSpec(name="kind", type=String, default="output",
                           description="output | va | income | employment | satellite account name"),
                OptionSpec(name="type", type=String, default="I",
                           choices=["I", "II"],
                           description="Type I (open) | Type II (household-closed)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:impact_summary,
                              description="Economy-wide impact total, kind and type"),
                    TableSpec(name=:impact_by_sector,
                              description="Per-sector impact of the demand scenario")],
            category="io", handler=wrap_legacy(_io_impact),
        ),
        CommandSpec(
            path=["io", "network-stats"],
            summary="Domar weights, Herfindahl, APL, degrees, upstreamness/downstreamness",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                     OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:network_stats_summary,
                              description="Herfindahl of Domar weights and multiplier dispersion"),
                    TableSpec(name=:network_stats_sectors,
                              description="Per-sector Domar, multiplier, degrees and up/down-streamness")],
            category="io", handler=wrap_legacy(_io_network_stats),
        ),
        CommandSpec(
            path=["io", "aggregate"],
            summary="Aggregate an IO/MRIO table over regions and/or sector types",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="region-map", type=String, default="",
                           description="old=new region pairs, comma-separated"),
                OptionSpec(name="sector-map", type=String, default="",
                           description="old=new sector-type pairs, comma-separated"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:io_table_summary,
                              description="Sector/region/category counts after aggregation"),
                    TableSpec(name=:sectors,
                              description="Per-sector gross output, final demand and value added")],
            category="io", handler=wrap_legacy(_io_aggregate),
        ),
        CommandSpec(
            path=["io", "balance"],
            summary="Repair intermediate flows so row and column accounts close (RAS/GRAS)",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="method", type=String, default="ras",
                           choices=["ras", "gras"],
                           description="ras (non-negative Z) | gras (sign-preserving)"),
                OptionSpec(name="tol", type=Float64, default=1e-10,
                           description="Biproportional convergence tolerance"),
                OptionSpec(name="maxiter", type=Int, default=1000,
                           description="Maximum RAS/GRAS iterations"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:io_table_summary,
                              description="Sector/region counts of the balanced table"),
                    TableSpec(name=:sectors,
                              description="Per-sector gross output, final demand and value added")],
            category="io", handler=wrap_legacy(_io_balance),
        ),
        CommandSpec(
            path=["io", "vertical-specialization"],
            summary="Hummels–Ishii–Yi / KWW import content of exports",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="region", type=String, default="",
                           description="Region name or 1-based index (required when nregions>1)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:vertical_specialization,
                              description="VS, VS share, VS1, domestic content and gross exports"),
                    TableSpec(name=:vertical_specialization_by_sector,
                              description="Per-sector foreign content in that sector's exports")],
            category="io", handler=wrap_legacy(_io_vertical_specialization),
        ),
        CommandSpec(
            path=["io", "export-decomposition"],
            summary="Koopman–Wang–Wei (2014) DVA/RDV/FVA/PDC decomposition of gross exports",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="region", type=String, default="",
                           description="Region name or 1-based index (required when nregions>1)"),
                OUTPUT_OPTIONS..., PLOT_OPTIONS...],
            flags=[FlagSpec(name="plot", description="Open interactive plot in browser")],
            tables=[TableSpec(name=:kww_export_aggregates,
                              description="One-row DVA/RDV/FVA/PDC plus gross exports and VAX ratio"),
                    TableSpec(name=:kww_export_by_sector,
                              description="Per-sector DVA/RDV/FVA/PDC of the region's exports")],
            category="io", handler=wrap_legacy(_io_export_decomposition),
        ),
        CommandSpec(
            path=["io", "bilateral-trade"],
            summary="Bilateral intermediate/final/total trade from exporter to importer",
            args=ArgSpec[],
            options=[IO_INPUT_OPTIONS..., IO_PARSER_OPTIONS...,
                OptionSpec(name="exporter", type=String, default="",
                           description="Exporting region name or 1-based index (required)"),
                OptionSpec(name="importer", type=String, default="",
                           description="Importing region name or 1-based index (required)"),
                OptionSpec(name="kind", type=String, default="total",
                           choices=["total", "intermediate", "final"],
                           description="total | intermediate | final"),
                OUTPUT_OPTIONS...],
            tables=[TableSpec(name=:bilateral_trade_summary,
                              description="Intermediate, final and total bilateral flows"),
                    TableSpec(name=:bilateral_trade_by_sector,
                              description="Per-sector gross exports from exporter to importer")],
            category="io", handler=wrap_legacy(_io_bilateral_trade),
        ),
    ]
end

function register_io_commands!()
    specs = map(io_specs()) do s
        if s.path in (["io", "load"], ["io", "bf", "network"],
                      ["io", "aggregate"], ["io", "balance"])
            with_save_model([s])[1]
        elseif length(s.path) == 3 && s.path[1] == "io" && s.path[2] == "bf"
            with_model_option([_copy_spec(s; model_types=[:ProductionNetwork])])[1]
        else
            s
        end
    end
    out = CommandSpec[]
    for s in specs
        if _has_data_slot(s)
            push!(out, _copy_spec(s; data_kinds=[:io, :csv]))
        else
            push!(out, s)
        end
    end
    out = with_default_csv_kinds(out)
    register!(out)
    return build_node("io", out;
        description="Input-Output analysis: Leontief/Ghosh, multipliers, linkages, SDA, footprints, Baqaee-Farhi, MRIO")
end

# ── Shared helpers ───────────────────────────────────────────

"""
    _load_io(data; n_sectors, n_fd, sectors, delim) → IOData

Load an [`IOData`] table. Empty / `:wiot` → the bundled Miller & Blair example
(offline). Another `:name` → that example (must be IO-typed). A file path is
parsed with `parse_io`, which needs `--n-sectors` (and reads `Z` from the first
`n_sectors` columns, the next `n_fd` as final demand).
"""
function _io_missing_extension(e)
    msg = _err_message(e)
    (occursin("ZipFile package", msg) || occursin("XLSX package", msg) ||
     occursin("requires the ZipFile", msg) || occursin("requires the XLSX", msg)) &&
        return CliError("env/missing-extension", msg;
            hint="ZipFile/XLSX are MEMs weak deps and are not bundled; use a CSV/ICIO text table or a native .jld2 IOData handle")
    return nothing
end

function _load_io(data::String; n_sectors::Int=0, n_fd::Int=1,
                  sectors::String="", delim::String=",",
                  format::String="csv", year::String="", member::String="",
                  aggregate_cn_mx::Bool=true, check::Bool=false)
    if isempty(data) || data == ":wiot" || data == "wiot"
        return load_example(:wiot)
    end
    if startswith(data, ":")
        name = Symbol(replace(data[2:end], "-" => "_"))
        io = try
            load_example(name)
        catch e
            e isa ArgumentError && throw(CliError("usage/unknown-example", _err_message(e);
                hint="use :wiot for the bundled IO example, or a CSV path with --n-sectors"))
            rethrow()
        end
        io isa IOData || throw(CliError("data/not-io",
            "example :$name is not an Input-Output table";
            hint="use :wiot or an IO CSV via --data path --n-sectors N"))
        return io
    end
    lc = lowercase(data)
    if endswith(lc, ".jld2") || startswith(data, "model://")
        m = load_model_dispatch(data)
        m isa IOData || throw(CliError("data/invalid",
            "$data is not an IOData handle (got $(typeof(m)))";
            hint="save an IO table with `io load --save-model table.jld2`"))
        return m
    end
    isfile(data) || throw(CliError("data/file-not-found", "file not found: $data";
                                   hint="check the path, or omit --data to use the bundled :wiot example"))
    fmt = lowercase(strip(format))
    if fmt == "icio"
        # MEMs `parse_icio` calls `something(detected_year, …)` and throws
        # "No value arguments present" when the filename has no year and
        # `year=` is omitted. Default to 0 (unknown) so a standalone CSV works.
        yr = isempty(strip(year)) ? 0 : something(tryparse(Int, year), 0)
        return try
            parse_icio(data; year=yr, member=member,
                       aggregate_cn_mx=aggregate_cn_mx, check=check)
        catch e
            miss = _io_missing_extension(e)
            miss !== nothing && throw(miss)
            (e isa ArgumentError || e isa ErrorException) && throw(CliError("data/parse",
                "failed to parse ICIO table '$data': $(_err_message(e))"))
            rethrow()
        end
    end
    n_sectors > 0 || throw(CliError("usage/missing-option",
        "--n-sectors is required to parse an IO CSV";
        hint="Z is read from the first n_sectors columns, the next n_fd as final demand"))
    secs = _split_csv(sectors)
    dchar = isempty(delim) ? ',' : first(delim)
    return try
        parse_io(data; source=:csv, n_sectors=n_sectors, n_fd=n_fd,
                 sectors=secs, delim=dchar)
    catch e
        miss = _io_missing_extension(e)
        miss !== nothing && throw(miss)
        (e isa BoundsError || e isa ArgumentError || e isa ErrorException) && throw(CliError("data/parse",
            "failed to parse IO CSV '$data': $(_err_message(e))";
            hint="check --n-sectors/--n-fd against the file: Z is the first n_sectors columns, then n_fd final-demand columns"))
        rethrow()
    end
end

"Split a comma-separated option into trimmed non-empty tokens."
_split_csv(s::AbstractString) = String[strip(t) for t in split(s, ",") if !isempty(strip(t))]

"""
    _io_matrix_df(M, rowlabels, collabels=rowlabels) → DataFrame

Render a matrix WIDE: first column `sector` holds the row labels, then one
numeric column per `collabels` entry (rounded). Columns are made unique if the
labels collide (real MRIO tables occasionally repeat sector names across regions).
"""
function _io_matrix_df(M::AbstractMatrix, rowlabels::Vector{String},
                       collabels::Vector{String}=rowlabels)
    df = DataFrame()
    df[!, :sector] = rowlabels
    used = Set{String}(["sector"])   # reserve the row-label column
    for (j, c) in enumerate(collabels)
        col = c; k = 1
        while col in used            # bump past collisions (dups, pre-suffixed, "sector")
            k += 1
            col = "$(c)_$(k)"
        end
        push!(used, col)
        df[!, Symbol(col)] = round.(M[:, j]; digits=6)
    end
    return df
end

# ── Handlers ─────────────────────────────────────────────────

function _io_sources(; format::String="table", output::String="")
    t = list_io_sources()
    src = String[]; nm = String[]; vers = String[]; cred = String[]; note = String[]
    for (k, v) in t.rows
        push!(src, string(k)); push!(nm, v.name)
        push!(vers, join(v.versions, ", "))
        push!(cred, v.needs_credentials ? "yes" : "no")
        push!(note, v.note)
    end
    df = DataFrame(source=src, name=nm, versions=vers, credentials=cred, note=note)
    output_result(df; format=Symbol(format), output=output, title="IO/MRIO Sources")
end

function _io_download(; source::String="", storage::String="", source_version::String="",
                      years::String="", system::String="pxp",
                      email::String="", password::String="",
                      offline::Bool=false, overwrite::Bool=false, no_verify::Bool=false,
                      format::String="table", output::String="")
    isempty(source) && throw(CliError("usage/missing-option",
        "--source is required"; hint="one of oecd|wiod|exiobase3|eora26|gloria (see `io sources`)"))
    isempty(storage) && throw(CliError("usage/missing-option",
        "--storage <folder> is required (download destination)"))

    if offline || _offline_env()
        throw(CliError("env/network",
            "network access disabled (--offline); refusing to download :$source";
            hint="drop --offline (and unset FRIEDMAN_OFFLINE) to allow the download"))
    end

    src = Symbol(source)
    yrs = _parse_years(years)
    ver = isempty(source_version) ? nothing : source_version

    _status("Downloading :$source → $storage" *
            (isempty(source_version) ? "" : " (version=$source_version)"))
    # Forward source-specific kwargs only where the per-source downloader accepts
    # them: `system` → exiobase3 only; `verify` → every source except eora26 (which
    # takes neither). download_io relays extras verbatim, so over-forwarding is a
    # MethodError, not a no-op.
    extra = src === :exiobase3 ? (; system=system, verify=!no_verify) :
            src === :eora26    ? NamedTuple() :
                                 (; verify=!no_verify)
    meta = try
        download_io(src; storage_folder=storage, years=yrs, version=ver,
                    email=email, password=password, overwrite_existing=overwrite, extra...)
    catch e
        throw(_map_download_error(e, source))
    end

    output_kv([
        "source" => meta.source,
        "version" => meta.version,
        "files" => string(length(meta.files)),
    ]; format=Symbol(format), output=output, title="Download Summary")

    if !isempty(meta.files)
        df = DataFrame(url=String[first(p) for p in meta.files],
                       filename=String[last(p) for p in meta.files])
        output_result(df; format=Symbol(format), output=output, title="Download Log")
    else
        for h in meta.history
            _status("  ", h)
        end
    end
    return meta
end

function _io_sector_regions(io)
    n = length(io.sectors)
    out = fill("", n)
    for r in io.regions
        idxs = try
            collect(region_indices(io, r))
        catch
            Int[]
        end
        for i in idxs
            1 <= i <= n && (out[i] = r)
        end
    end
    return out
end

function _io_load(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                  parser::String="csv", year::String="", member::String="",
                  no_aggregate_cn_mx::Bool=false, check::Bool=false,
                  format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                  format=parser, year=year, member=member,
                  aggregate_cn_mx=!no_aggregate_cn_mx, check=check)
    # nsectors/nregions are internal (unexported) MEMs helpers — compute inline.
    n_reg = length(io.regions)
    n_sec = length(io.sectors) ÷ max(1, n_reg)

    output_kv([
        "sectors" => string(n_sec),
        "regions" => string(n_reg),
        "fd_categories" => string(length(io.fd_cats)),
        "va_categories" => string(length(io.va_cats)),
        "year" => io.year === nothing ? "—" : string(io.year),
        "unit" => isempty(io.unit) ? "—" : io.unit,
        "source" => isempty(io.source) ? "—" : io.source,
        "extensions" => isempty(io.extensions) ? "—" : join(sort(collect(keys(io.extensions))), ", "),
        "total_output" => string(round(sum(io.x); digits=4)),
    ]; format=Symbol(format), output=output, title="IO Table Summary")

    fd_tot = vec(sum(io.Y, dims=2))
    va_tot = vec(sum(io.va, dims=1))
    df = DataFrame(sector=io.sectors,
                   region=_io_sector_regions(io),
                   gross_output=round.(io.x; digits=6),
                   final_demand=round.(fd_tot; digits=6),
                   value_added=round.(va_tot; digits=6))
    output_result(df; format=Symbol(format), output=output, title="Sectors")
    return io
end

function _io_leontief(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                      matrix::String="L", format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    m = leontief(io)
    if matrix in ("A", "both")
        output_result(_io_matrix_df(m.A, io.sectors);
                      format=Symbol(format), output=output, title="Technical Coefficients (A)")
    end
    if matrix in ("L", "both")
        output_result(_io_matrix_df(m.L, io.sectors);
                      format=Symbol(format), output=output, title="Leontief Inverse (L)")
    end
    return m
end

function _io_ghosh(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                   matrix::String="G", format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    m = ghosh(io)
    if matrix in ("B", "both")
        output_result(_io_matrix_df(m.B, io.sectors);
                      format=Symbol(format), output=output, title="Allocation Coefficients (B)")
    end
    if matrix in ("G", "both")
        output_result(_io_matrix_df(m.G, io.sectors);
                      format=Symbol(format), output=output, title="Ghosh Inverse (G)")
    end
    return m
end

function _io_multipliers(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                         kind::String="output", type::String="I",
                         format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    m = try
        multipliers(io; kind=Symbol(kind), type=Symbol(type))
    catch e
        e isa ArgumentError && throw(CliError("data/no-extension", _err_message(e);
            hint="employment multipliers need an 'employment' satellite account (present in :wiot; a plain CSV table has none)"))
        rethrow()
    end
    df = DataFrame(sector=m.sectors, multiplier=round.(m.values; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Type $(type) $(kind) multipliers", key="io_multipliers")
    return m
end

function _io_linkages(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                      forward::String="ghosh", format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    lk = linkages(io; forward=Symbol(forward))
    df = DataFrame(sector=lk.sectors,
                   backward=round.(lk.backward; digits=6),
                   forward=round.(lk.forward; digits=6),
                   Ui=round.(lk.Ui; digits=6),
                   Uj=round.(lk.Uj; digits=6),
                   class=string.(lk.classification))
    output_result(df; format=Symbol(format), output=output,
                  title="Linkages (forward=$forward)", key="linkages")
    return lk
end

function _io_key_sectors(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                         forward::String="ghosh", format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    lk = linkages(io; forward=Symbol(forward))
    df = DataFrame(sector=lk.sectors, class=string.(lk.classification))
    output_result(df; format=Symbol(format), output=output, title="Key Sectors")
    # Quadrant counts to stderr (diagnostic, not part of the data envelope).
    counts = Dict{Symbol,Int}()
    for c in lk.classification
        counts[c] = get(counts, c, 0) + 1
    end
    _status("Key-sector quadrants: " *
            join(["$(k)=$(get(counts, k, 0))" for k in (:key, :forward, :backward, :weak)], ", "))
    return lk
end

function _parse_sda_factors(s::String)
    isempty(strip(s)) && return nothing
    out = Symbol[]
    for t in _split_csv(s)
        tok = replace(lowercase(t), "-" => "_")
        tok in ("fd", "final_demand") && (tok = "final_demand")
        tok in ("intensity", "emission_intensity") && (tok = "intensity")
        push!(out, Symbol(tok))
    end
    return out
end

function _io_sda(; data::String="", data2::String="", n_sectors::Int=0, n_fd::Int=1,
                 sectors::String="", method::String="additive",
                 factors::String="", on::String="output",
                 format::String="table", output::String="",
                 plot::Bool=false, plot_save::String="")
    io0 = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    io1 = try
        _load_io(isempty(data2) ? data : data2;
                 n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    catch e
        e isa CliError && rethrow()
        throw(e)
    end
    facs = _parse_sda_factors(factors)
    on_arg = on == "output" ? :output : on
    r = try
        facs === nothing ?
            sda(io0, io1; method=Symbol(method), on=on_arg) :
            sda(io0, io1; method=Symbol(method), factors=facs, on=on_arg)
    catch e
        e isa ArgumentError && occursin("same number of sectors", _err_message(e)) &&
            throw(CliError("data/shape", _err_message(e)))
        e isa ArgumentError && throw(CliError("usage/invalid", _err_message(e)))
        rethrow()
    end
    labels = length(r.total) == length(io0.sectors) ? io0.sectors :
             (hasproperty(r, :stressors) ? collect(r.stressors) : ["row$i" for i in 1:length(r.total)])
    # Legacy L/Y column names only when --factors is omitted AND --on is output.
    # Explicit --factors (even technology,final-demand) and satellite --on use
    # the named-factor *_effect columns (intensity/technology/final_demand).
    if facs === nothing && on_arg === :output && haskey(r.effects, :L) && haskey(r.effects, :Y)
        df = DataFrame(sector=labels,
                       L_effect=round.(r.effects[:L]; digits=6),
                       Y_effect=round.(r.effects[:Y]; digits=6),
                       total=round.(r.total; digits=6),
                       residual=round.(r.residual; digits=6))
    else
        df = DataFrame(sector=labels)
        for (k, v) in r.effects
            df[!, Symbol(string(k) * "_effect")] = round.(v; digits=6)
        end
        df[!, :total] = round.(r.total; digits=6)
        df[!, :residual] = round.(r.residual; digits=6)
    end
    output_result(df; format=Symbol(format), output=output,
                  title="Structural Decomposition ($method)", key="structural_decomposition")
    _maybe_plot(r; plot=plot, plot_save=plot_save)
    return r
end

function _io_extract(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                     sectors_extract::String="", mode::String="complete",
                     share::Float64=1.0, region::String="",
                     format::String="table", output::String="",
                     plot::Bool=false, plot_save::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    (0.0 < share <= 1.0) || throw(CliError("usage/invalid",
        "io extract: --share must be in (0, 1] (got $share)"))
    isempty(sectors_extract) && isempty(region) && throw(CliError("usage/missing-option",
        "--sectors-extract is required (sector name(s) or 1-based index/indices, comma-separated)"))
    toks = _split_csv(sectors_extract)
    n = length(io.sectors)
    target = if isempty(toks)
        String[]
    elseif all(t -> tryparse(Int, t) !== nothing, toks)
        idx = [parse(Int, t) for t in toks]
        for i in idx
            (1 <= i <= n) || throw(CliError("data/bad-sector",
                "sector index $i out of range (1:$n)";
                hint="valid indices 1:$n, or use sector names: $(join(io.sectors, ", "))"))
        end
        idx
    else
        toks
    end
    reg = isempty(region) ? nothing : region
    r = try
        if isempty(target) && reg !== nothing
            hypothetical_extraction(io, String[]; mode=Symbol(mode), share=share, region=reg)
        else
            hypothetical_extraction(io, target; mode=Symbol(mode), share=share, region=reg)
        end
    catch e
        e isa ArgumentError && occursin("share must", _err_message(e)) &&
            throw(CliError("usage/invalid", _err_message(e)))
        e isa ArgumentError && occursin("mode must", _err_message(e)) &&
            throw(CliError("usage/invalid", _err_message(e)))
        (e isa ArgumentError || e isa BoundsError) && throw(CliError("data/bad-sector",
            _err_message(e); hint="valid sectors: $(join(io.sectors, ", "))"))
        rethrow()
    end
    extracted_labels = [io.sectors[i] for i in r.extracted]
    _status("Extracted sector(s): $(join(extracted_labels, ", "))")
    _status("Total output loss: $(round(r.total_loss; digits=6))")
    _status()
    df = DataFrame(sector=io.sectors, output_loss=round.(r.sector_loss; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Hypothetical Extraction (loss)")
    pct_go = hasproperty(r, :loss_pct_go) ? r.loss_pct_go : NaN
    pct_gdp = hasproperty(r, :loss_pct_gdp) ? r.loss_pct_gdp : NaN
    output_kv([
        "total_loss" => round(r.total_loss; digits=6),
        "loss_pct_go" => isfinite(pct_go) ? round(Float64(pct_go); digits=6) : string(pct_go),
        "loss_pct_gdp" => isfinite(pct_gdp) ? round(Float64(pct_gdp); digits=6) : string(pct_gdp),
        "mode" => String(hasproperty(r, :mode) ? r.mode : Symbol(mode)),
        "share" => hasproperty(r, :share) ? r.share : share,
    ]; format=format, output=_per_var_output_path(output, "summary"),
       title="Hypothetical Extraction Summary")
    _maybe_plot(r; plot=plot, plot_save=plot_save)
    return r
end

function _io_footprint(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                       account::String="", detail::Bool=false, by::String="sector",
                       format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    isempty(io.extensions) && throw(CliError("data/no-extension",
        "IO table has no satellite accounts; footprint needs an extension (e.g. CO2)";
        hint="the bundled :wiot example carries CO2 and employment accounts"))
    name = isempty(account) ? first(sort(collect(keys(io.extensions)))) : account
    haskey(io.extensions, name) || throw(CliError("data/no-extension",
        "no satellite account '$name'";
        hint="available: $(join(sort(collect(keys(io.extensions))), ", "))"))

    by_sym = Symbol(by)
    fp = try
        footprint(io, name; by=by_sym)
    catch e
        e isa ArgumentError && throw(CliError("usage/invalid", _err_message(e)))
        rethrow()
    end
    if by_sym === :region
        # RegionalFootprintResult: production/consumption stressor×G
        prod = hasproperty(fp, :production) ? fp.production : fp.total
        consu = hasproperty(fp, :consumption) ? fp.consumption : fp.total
        regs = hasproperty(fp, :regions) ? collect(fp.regions) : io.regions
        stressors = hasproperty(fp, :stressors) ? collect(fp.stressors) : [name]
        df = DataFrame(region=regs)
        for (i, s) in enumerate(stressors)
            i <= size(prod, 1) && (df[!, Symbol("production_" * s)] = round.(prod[i, :]; digits=6))
            i <= size(consu, 1) && (df[!, Symbol("consumption_" * s)] = round.(consu[i, :]; digits=6))
        end
        output_result(df; format=Symbol(format), output=output,
                      title="Regional Footprint ($name)", key="regional_footprint")
        return fp
    end
    # Total footprint per stressor (summed over final-demand categories).
    tot = vec(sum(fp.total; dims=2))
    output_result(DataFrame(stressor=fp.stressors, footprint=round.(tot; digits=6));
                  format=Symbol(format), output=output, title="Footprint ($name)", key="footprint")
    # Per-sector contribution: sector × stressor(s).
    output_result(_io_matrix_df(permutedims(fp.by_sector), io.sectors, fp.stressors);
                  format=Symbol(format), output=output, title="Footprint by Sector ($name)",
                  key="footprint_by_sector")

    if detail
        output_result(_io_matrix_df(permutedims(intensities(io, name)), io.sectors, fp.stressors);
                      format=Symbol(format), output=output, title="Intensities S ($name)",
                      key="intensities_s")
        output_result(_io_matrix_df(permutedims(emission_multipliers(io, name)), io.sectors, fp.stressors);
                      format=Symbol(format), output=output, title="Emission Multipliers M=SL ($name)",
                      key="emission_multipliers")
    end
    return fp
end

function _io_baqaee_farhi(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                          theta::Float64=1.0, sigma::Float64=1.0, second_order::Bool=false,
                          format::String="table", output::String="")
    io = _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors)
    bf = baqaee_farhi(io; theta=theta, sigma=sigma)
    df = DataFrame(sector=bf.sectors,
                   domar=round.(bf.domar; digits=6),
                   influence=round.(bf.influence; digits=6),
                   upstreamness=round.(bf.upstreamness; digits=6),
                   downstreamness=round.(bf.downstreamness; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Baqaee & Farhi (2019) decomposition")
    if second_order
        output_result(_io_matrix_df(bf.second_order, io.sectors);
                      format=Symbol(format), output=output, title="Second-order Hessian (beyond Hulten)")
    end
    return bf
end

# ── Download support ─────────────────────────────────────────

"True when the environment forces offline mode (FRIEDMAN_OFFLINE=1|true)."
_offline_env() = lowercase(get(ENV, "FRIEDMAN_OFFLINE", "")) in ("1", "true", "yes")

"Parse a comma-separated year filter into a Vector{Int} (or nothing for 'all')."
function _parse_years(years::String)
    isempty(years) && return nothing
    toks = _split_csv(years)
    isempty(toks) && return nothing
    out = Int[]
    for t in toks
        y = tryparse(Int, t)
        y === nothing && throw(CliError("usage/bad-years",
            "invalid --years value '$t' (expected comma-separated integers, e.g. 2018,2019)"))
        push!(out, y)
    end
    return out
end

"""Map a download failure to a typed CliError: checksum mismatch → env/checksum-mismatch
(#250); unknown source/version/credentials → usage; anything else → env/network."""
function _map_download_error(e, source::String)
    e isa CliError && return e
    msg = _err_message(e)
    if occursin("checksum mismatch", lowercase(msg)) || occursin("has been substituted", msg)
        return CliError("env/checksum-mismatch",
            "downloaded archive failed SHA-256 verification: $msg";
            hint="the file is corrupt or substituted; retry, or --no-verify to bypass (unsafe)")
    elseif e isa ArgumentError
        return CliError("usage/bad-argument", msg; hint="see `io sources` for valid source/version")
    end
    return CliError("env/network", "download of :$source failed: $msg";
                    hint="check connectivity, the source URL set, or run later")
end

# ── W3/#154 + W4/#155 helpers ────────────────────────────────

"""Parse a scalar-or-comma-list numeric option into a Float64 or Vector{Float64}."""
function _parse_bf_num(s::AbstractString, name::String)
    toks = _split_csv(s)
    isempty(toks) && throw(CliError("usage/invalid",
        "io bf: --$name is empty (expected a number or comma-separated list)"))
    vals = Float64[]
    for t in toks
        v = tryparse(Float64, t)
        v === nothing && throw(CliError("usage/invalid",
            "io bf: --$name value '$t' is not a number"))
        push!(vals, v)
    end
    return length(vals) == 1 ? vals[1] : vals
end

"Empty string → `nothing`; otherwise `_parse_bf_num`."
function _parse_bf_num_or_nothing(s::AbstractString, name::String)
    isempty(strip(s)) && return nothing
    return _parse_bf_num(s, name)
end

"""Parse `old=new,old2=new2` into Dict{String,String} (empty → nothing)."""
function _parse_name_map(s::AbstractString, name::String)
    isempty(strip(s)) && return nothing
    d = Dict{String,String}()
    for t in _split_csv(s)
        i = findfirst('=', t)
        i === nothing && throw(CliError("usage/invalid",
            "io: --$name entry '$t' must be old=new"))
        old, newv = strip(t[1:i-1]), strip(t[i+1:end])
        (isempty(old) || isempty(newv)) && throw(CliError("usage/invalid",
            "io: --$name entry '$t' must be old=new"))
        d[old] = newv
    end
    return d
end

"""Parse a shock spec: comma list of numbers, or `sector=value` pairs. Empty → nothing."""
function _parse_shock_spec(s::AbstractString, name::String)
    isempty(strip(s)) && return nothing
    toks = _split_csv(s)
    isempty(toks) && return nothing
    if all(t -> occursin('=', t), toks)
        d = Dict{String,Float64}()
        for t in toks
            i = findfirst('=', t)
            k, vs = strip(t[1:i-1]), strip(t[i+1:end])
            v = tryparse(Float64, vs)
            (isempty(k) || v === nothing) && throw(CliError("usage/invalid",
                "io: --$name entry '$t' must be sector=number"))
            d[k] = v
        end
        return d
    end
    vals = Float64[]
    for t in toks
        occursin('=', t) && throw(CliError("usage/invalid",
            "io: --$name mixes bare numbers and sector=value pairs"))
        v = tryparse(Float64, t)
        v === nothing && throw(CliError("usage/invalid",
            "io: --$name value '$t' is not a number"))
        push!(vals, v)
    end
    return vals
end

function _io_from_opts(; data::String="", n_sectors::Int=0, n_fd::Int=1,
                       sectors::String="", parser::String="csv")
    _load_io(data; n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, format=parser)
end

function _bf_calibrate(; data::String="", n_sectors::Int=0, n_fd::Int=1,
                       sectors::String="", parser::String="csv",
                       theta::String="1.0", sigma::Float64=1.0,
                       epsilon::String="1.0", eta::String="1.0",
                       nests::String="single", factors::String="single",
                       mu::String="1.0", no_check::Bool=false, model=nothing)
    if model isa ProductionNetwork
        return model
    elseif model !== nothing
        throw(CliError("data/invalid",
            "io bf: --model is not a ProductionNetwork (got $(typeof(model)))";
            hint="save one with `io bf network --save-model net.jld2`"))
    end
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd,
                       sectors=sectors, parser=parser)
    fac = Symbol(replace(factors, "-" => "_"))
    try
        return production_network(io;
            theta=_parse_bf_num(theta, "theta"),
            sigma=sigma,
            epsilon=_parse_bf_num(epsilon, "epsilon"),
            eta=_parse_bf_num(eta, "eta"),
            nests=Symbol(nests),
            factors=fac,
            mu=_parse_bf_num(mu, "mu"),
            check=!no_check)
    catch e
        throw(_domain_or_data_error(e, "production_network"))
    end
end

function _io_emit_iodata(io; format::String="table", output::String="")
    n_reg = length(io.regions)
    n_sec = length(io.sectors) ÷ max(1, n_reg)
    output_kv([
        "sectors" => string(n_sec),
        "regions" => string(n_reg),
        "fd_categories" => string(length(io.fd_cats)),
        "va_categories" => string(length(io.va_cats)),
        "year" => io.year === nothing ? "—" : string(io.year),
        "unit" => isempty(io.unit) ? "—" : io.unit,
        "source" => isempty(io.source) ? "—" : io.source,
        "extensions" => isempty(io.extensions) ? "—" : join(sort(collect(keys(io.extensions))), ", "),
        "total_output" => string(round(sum(io.x); digits=4)),
    ]; format=Symbol(format), output=output, title="IO Table Summary")
    fd_tot = vec(sum(io.Y, dims=2))
    va_tot = vec(sum(io.va, dims=1))
    df = DataFrame(sector=io.sectors,
                   region=_io_sector_regions(io),
                   gross_output=round.(io.x; digits=6),
                   final_demand=round.(fd_tot; digits=6),
                   value_added=round.(va_tot; digits=6))
    output_result(df; format=Symbol(format), output=output, title="Sectors")
    return io
end

# ── W3/#154 handlers ─────────────────────────────────────────

function _io_bf_network(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                        parser::String="csv",
                        theta::String="1.0", sigma::Float64=1.0,
                        epsilon::String="1.0", eta::String="1.0",
                        nests::String="single", factors::String="single",
                        mu::String="1.0", no_check::Bool=false,
                        format::String="table", output::String="",
                        plot::Bool=false, plot_save::String="")
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check)
    output_kv([
        "n" => string(net.n),
        "M" => string(net.M),
        "F" => string(net.F),
        "nests" => string(net.nests),
    ]; format=Symbol(format), output=output, title="Production Network Summary")
    λ = [net.lambda[g] for g in net.outer_nodes]
    λr = [net.lambda_rev[g] for g in net.outer_nodes]
    μ = [net.mu[g - 1] for g in net.outer_nodes]
    df = DataFrame(sector=net.io.sectors,
                   lambda=round.(λ; digits=6),
                   lambda_rev=round.(λr; digits=6),
                   mu=round.(μ; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Production Network Sectors", key="production_network_sectors")
    _maybe_plot(net; plot=plot, plot_save=plot_save)
    return net
end

function _io_bf_equilibrium(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                            parser::String="csv",
                            theta::String="1.0", sigma::Float64=1.0,
                            epsilon::String="1.0", eta::String="1.0",
                            nests::String="single", factors::String="single",
                            mu::String="1.0", no_check::Bool=false, model=nothing,
                            dlog_a::String="", dlog_l::String="", dlog_mu::String="",
                            method::String="newton", tol::Float64=1e-10,
                            maxiter::Int=500, damping::Float64=0.5,
                            format::String="table", output::String="",
                            plot::Bool=false, plot_save::String="")
    (0.0 < damping <= 1.0) || throw(CliError("usage/invalid",
        "io bf equilibrium: --damping must be in (0, 1] (got $damping)"))
    tol > 0 || throw(CliError("usage/invalid",
        "io bf equilibrium: --tol must be > 0 (got $tol)"))
    maxiter >= 1 || throw(CliError("usage/invalid",
        "io bf equilibrium: --maxiter must be ≥ 1 (got $maxiter)"))
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    eq = try
        bf_equilibrium(net;
            dlogA=_parse_bf_num_or_nothing(dlog_a, "dlog-a"),
            dlogL=_parse_bf_num_or_nothing(dlog_l, "dlog-l"),
            dlogmu=_parse_bf_num_or_nothing(dlog_mu, "dlog-mu"),
            method=Symbol(method), tol=tol, maxiter=maxiter, damping=damping)
    catch e
        throw(_domain_or_data_error(e, "bf_equilibrium"))
    end
    # Non-convergence is a result, not exit 5: surface `converged` in the kv table.
    output_kv([
        "dlogY" => round(eq.dlogY; digits=6),
        "hulten" => round(eq.hulten; digits=6),
        "technology" => round(eq.technology; digits=6),
        "allocative" => round(eq.allocative; digits=6),
        "profit_share" => round(eq.profit_share; digits=6),
        "converged" => string(eq.converged),
        "iterations" => eq.iterations,
        "residual" => round(eq.residual; digits=8),
    ]; format=Symbol(format), output=output, title="BF Equilibrium Summary")
    df = DataFrame(sector=eq.sectors,
                   dlog_x=round.(eq.dlog_x; digits=6),
                   dlog_p=round.(eq.dlog_p; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="BF Equilibrium Sectors", key="bf_equilibrium_sectors")
    _maybe_plot(eq; plot=plot, plot_save=plot_save)
    return eq
end

function _io_bf_local(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                      parser::String="csv",
                      theta::String="1.0", sigma::Float64=1.0,
                      epsilon::String="1.0", eta::String="1.0",
                      nests::String="single", factors::String="single",
                      mu::String="1.0", no_check::Bool=false, model=nothing,
                      hessian::String="auto", no_elasticities::Bool=false,
                      format::String="table", output::String="",
                      plot::Bool=false, plot_save::String="")
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    loc = try
        baqaee_farhi(net; hessian=Symbol(hessian), elasticities=!no_elasticities)
    catch e
        throw(_domain_or_data_error(e, "baqaee_farhi"))
    end
    df = DataFrame(sector=loc.sectors, first_order=round.(loc.first_order; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="BF Local Hulten", key="bf_local_hulten")
    if !isempty(loc.second_order)
        output_result(_io_matrix_df(loc.second_order, loc.sectors);
                      format=Symbol(format), output=output,
                      title="BF Local Hessian", key="bf_local_hessian")
    end
    _maybe_plot(loc; plot=plot, plot_save=plot_save)
    return loc
end

function _io_bf_elasticities(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                             parser::String="csv",
                             theta::String="1.0", sigma::Float64=1.0,
                             epsilon::String="1.0", eta::String="1.0",
                             nests::String="single", factors::String="single",
                             mu::String="1.0", no_check::Bool=false, model=nothing,
                             format::String="table", output::String="",
                             plot::Bool=false, plot_save::String="")
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    el = try
        bf_elasticities(net)
    catch e
        throw(_domain_or_data_error(e, "bf_elasticities"))
    end
    output_result(_io_matrix_df(el.dlogp_dlogA, el.sectors);
                  format=Symbol(format), output=output,
                  title="Price Incidence ∂log p / ∂log A", key="price_incidence")
    if !isempty(el.dlogw_dlogA)
        output_result(_io_matrix_df(el.dlogw_dlogA, el.factor_names, el.sectors);
                      format=Symbol(format), output=output,
                      title="Factor Price Incidence ∂log w / ∂log A",
                      key="factor_price_incidence")
    end
    _maybe_plot(el; plot=plot, plot_save=plot_save)
    return el
end

function _io_bf_shock_curve(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                            parser::String="csv",
                            theta::String="1.0", sigma::Float64=1.0,
                            epsilon::String="1.0", eta::String="1.0",
                            nests::String="single", factors::String="single",
                            mu::String="1.0", no_check::Bool=false, model=nothing,
                            sector::String="", range::String="-0.5,0.5", points::Int=41,
                            format::String="table", output::String="",
                            plot::Bool=false, plot_save::String="")
    isempty(strip(sector)) && throw(CliError("usage/missing-option",
        "io bf shock-curve: --sector is required (name or 1-based index)"))
    points >= 2 || throw(CliError("usage/invalid",
        "io bf shock-curve: --points must be ≥ 2 (got $points)"))
    rtoks = _split_csv(range)
    length(rtoks) == 2 || throw(CliError("usage/invalid",
        "io bf shock-curve: --range must be lo,hi (got '$range')"))
    lo = tryparse(Float64, rtoks[1]); hi = tryparse(Float64, rtoks[2])
    (lo === nothing || hi === nothing) && throw(CliError("usage/invalid",
        "io bf shock-curve: --range values must be numbers (got '$range')"))
    lo < hi || throw(CliError("usage/invalid",
        "io bf shock-curve: --range must be increasing (got $lo ≥ $hi)"))
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    sec = let i = tryparse(Int, strip(sector))
        i === nothing ? strip(sector) : i
    end
    sc = try
        bf_shock_curve(net, sec; range=(lo, hi), points=points)
    catch e
        throw(_domain_or_data_error(e, "bf_shock_curve"))
    end
    df = DataFrame(shock=round.(sc.shocks; digits=6),
                   exact=round.(sc.exact; digits=6),
                   hulten=round.(sc.hulten; digits=6),
                   second_order=round.(sc.second_order; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="BF Shock Curve ($(sc.sector))", key="bf_shock_curve")
    _maybe_plot(sc; plot=plot, plot_save=plot_save)
    return sc
end

function _io_bf_wedges(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                       parser::String="csv",
                       theta::String="1.0", sigma::Float64=1.0,
                       epsilon::String="1.0", eta::String="1.0",
                       nests::String="single", factors::String="single",
                       mu::String="1.0", no_check::Bool=false, model=nothing,
                       dlog_a::String="", dlog_l::String="", dlog_mu::String="",
                       format::String="table", output::String="",
                       plot::Bool=false, plot_save::String="")
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    w = try
        bf_wedge_decomp(net;
            dlogA=_parse_bf_num_or_nothing(dlog_a, "dlog-a"),
            dlogL=_parse_bf_num_or_nothing(dlog_l, "dlog-l"),
            dlogmu=_parse_bf_num_or_nothing(dlog_mu, "dlog-mu"))
    catch e
        throw(_domain_or_data_error(e, "bf_wedge_decomp"))
    end
    output_kv([
        "dlogY" => round(w.dlogY; digits=6),
        "technology" => round(w.technology; digits=6),
        "allocative" => round(w.allocative; digits=6),
        "allocative_mu" => round(w.allocative_mu; digits=6),
        "allocative_factor" => round(w.allocative_factor; digits=6),
        "factor_supply" => round(w.factor_supply; digits=6),
    ]; format=Symbol(format), output=output, title="BF Wedge Decomp")
    df = DataFrame(sector=w.sectors,
                   lambda_cost=round.(w.lambda_cost; digits=6),
                   lambda_rev=round.(w.lambda_rev; digits=6),
                   mu=round.(w.mu; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="BF Wedge Domar", key="bf_wedge_domar")
    _maybe_plot(w; plot=plot, plot_save=plot_save)
    return w
end

function _io_bf_misallocation(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                              parser::String="csv",
                              theta::String="1.0", sigma::Float64=1.0,
                              epsilon::String="1.0", eta::String="1.0",
                              nests::String="single", factors::String="single",
                              mu::String="1.0", no_check::Bool=false, model=nothing,
                              point::String="efficient", hessian::String="auto",
                              format::String="table", output::String="",
                              plot::Bool=false, plot_save::String="")
    net = _bf_calibrate(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors,
                        parser=parser, theta=theta, sigma=sigma, epsilon=epsilon, eta=eta,
                        nests=nests, factors=factors, mu=mu, no_check=no_check, model=model)
    m = try
        bf_misallocation(net; point=Symbol(point), hessian=Symbol(hessian))
    catch e
        throw(_domain_or_data_error(e, "bf_misallocation"))
    end
    output_kv([
        "distance" => round(m.distance; digits=6),
        "first_order" => round(m.first_order; digits=6),
        "second_order" => round(m.second_order; digits=6),
        "point" => string(m.point),
    ]; format=Symbol(format), output=output, title="BF Misallocation Summary")
    df = DataFrame(sector=m.sectors,
                   delta_logmu=round.(m.delta_logmu; digits=6),
                   lambda=round.(m.lambda; digits=6),
                   mu=round.(m.mu; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="BF Misallocation Sectors", key="bf_misallocation_sectors")
    _maybe_plot(m; plot=plot, plot_save=plot_save)
    return m
end

# ── W4/#155 handlers ─────────────────────────────────────────

function _io_price(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                   parser::String="csv", dva::String="", dtax::String="",
                   mode::String="leontief",
                   format::String="table", output::String="",
                   plot::Bool=false, plot_save::String="")
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    r = try
        price_model(io; dva=_parse_shock_spec(dva, "dva"),
                    dtax=_parse_shock_spec(dtax, "dtax"), mode=Symbol(mode))
    catch e
        throw(_domain_or_data_error(e, "price_model"))
    end
    df = DataFrame(sector=r.sectors,
                   dp=round.(r.dp; digits=6),
                   p=round.(r.p; digits=6),
                   dv=round.(r.dv; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Price Model ($mode)", key="price_model")
    _maybe_plot(r; plot=plot, plot_save=plot_save)
    return r
end

function _io_impact(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                    parser::String="csv", dy::String="", kind::String="output",
                    type::String="I",
                    format::String="table", output::String="",
                    plot::Bool=false, plot_save::String="")
    isempty(strip(dy)) && throw(CliError("usage/missing-option",
        "io impact: --dy is required (comma list or sector=value)"))
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    dyv = _parse_shock_spec(dy, "dy")
    r = try
        impact(io, dyv; kind=Symbol(kind), type=Symbol(type))
    catch e
        throw(_domain_or_data_error(e, "impact"))
    end
    output_kv([
        "total" => round(r.total; digits=6),
        "kind" => string(r.kind),
        "type" => string(r.type),
    ]; format=Symbol(format), output=output, title="Impact Summary")
    df = DataFrame(sector=r.sectors, impact=round.(r.by_sector; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Impact by Sector", key="impact_by_sector")
    _maybe_plot(r; plot=plot, plot_save=plot_save)
    return r
end

function _io_network_stats(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                           parser::String="csv",
                           format::String="table", output::String="",
                           plot::Bool=false, plot_save::String="")
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    r = try
        network_stats(io)
    catch e
        throw(_domain_or_data_error(e, "network_stats"))
    end
    output_kv([
        "herfindahl" => round(r.herfindahl; digits=6),
        "multiplier_dispersion" => round(r.multiplier_dispersion; digits=6),
    ]; format=Symbol(format), output=output, title="Network Stats Summary")
    df = DataFrame(sector=r.sectors,
                   domar=round.(r.domar; digits=6),
                   multiplier=round.(r.multipliers; digits=6),
                   in_degree=round.(r.in_degree; digits=6),
                   out_degree=round.(r.out_degree; digits=6),
                   upstreamness=round.(r.upstreamness; digits=6),
                   downstreamness=round.(r.downstreamness; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Network Stats Sectors", key="network_stats_sectors")
    _maybe_plot(r; plot=plot, plot_save=plot_save)
    return r
end

function _io_aggregate(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                       parser::String="csv", region_map::String="", sector_map::String="",
                       format::String="table", output::String="")
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    out = try
        aggregate(io; region_map=_parse_name_map(region_map, "region-map"),
                  sector_map=_parse_name_map(sector_map, "sector-map"))
    catch e
        throw(_domain_or_data_error(e, "aggregate"))
    end
    return _io_emit_iodata(out; format=format, output=output)
end

function _io_balance(; data::String="", n_sectors::Int=0, n_fd::Int=1, sectors::String="",
                     parser::String="csv", method::String="ras",
                     tol::Float64=1e-10, maxiter::Int=1000,
                     format::String="table", output::String="")
    tol > 0 || throw(CliError("usage/invalid",
        "io balance: --tol must be > 0 (got $tol)"))
    maxiter >= 1 || throw(CliError("usage/invalid",
        "io balance: --maxiter must be ≥ 1 (got $maxiter)"))
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    out = try
        balance(io; method=Symbol(method), tol=tol, maxiter=maxiter)
    catch e
        throw(_domain_or_data_error(e, "balance"))
    end
    return _io_emit_iodata(out; format=format, output=output)
end

function _io_resolve_region(s::String)
    isempty(strip(s)) && return nothing
    i = tryparse(Int, strip(s))
    return i === nothing ? strip(s) : i
end

function _io_vertical_specialization(; data::String="", n_sectors::Int=0, n_fd::Int=1,
                                     sectors::String="", parser::String="csv",
                                     region::String="",
                                     format::String="table", output::String="",
                                     plot::Bool=false, plot_save::String="")
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    vs = try
        vertical_specialization(io, _io_resolve_region(region))
    catch e
        msg = _err_message(e)
        e isa ArgumentError && occursin("region is required", msg) &&
            throw(CliError("usage/missing-option", msg;
                hint="pass --region <name|index> (table has $(length(io.regions)) regions)"))
        throw(_domain_or_data_error(e, "vertical_specialization"))
    end
    output_kv([
        "vs" => round(vs.vs; digits=6),
        "vs_share" => round(vs.vs_share; digits=6),
        "vs1" => round(vs.vs1; digits=6),
        "domestic_content" => round(vs.domestic_content; digits=6),
        "dc_share" => round(vs.dc_share; digits=6),
        "gross_exports" => round(vs.gross_exports; digits=6),
        "region" => vs.region,
    ]; format=Symbol(format), output=output, title="Vertical Specialization")
    # by_sector is length nsectors (one region's industries), not full n.
    ns = length(vs.by_sector)
    labs = if ns == length(io.sectors)
        io.sectors
    else
        # Prefer labels of the named region block.
        idxs = try
            collect(region_indices(io, vs.region))
        catch
            Int[]
        end
        length(idxs) == ns ? io.sectors[idxs] : ["sector$i" for i in 1:ns]
    end
    df = DataFrame(sector=labs, foreign_content=round.(vs.by_sector; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Vertical Specialization by Sector",
                  key="vertical_specialization_by_sector")
    _maybe_plot(vs; plot=plot, plot_save=plot_save)
    return vs
end

function _io_export_decomposition(; data::String="", n_sectors::Int=0, n_fd::Int=1,
                                  sectors::String="", parser::String="csv",
                                  region::String="",
                                  format::String="table", output::String="",
                                  plot::Bool=false, plot_save::String="")
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    ed = try
        export_decomposition(io, _io_resolve_region(region))
    catch e
        msg = _err_message(e)
        e isa ArgumentError && occursin("region is required", msg) &&
            throw(CliError("usage/missing-option", msg;
                hint="pass --region <name|index> (table has $(length(io.regions)) regions)"))
        throw(_domain_or_data_error(e, "export_decomposition"))
    end
    # Distinctive columns dva/rdv/fva/pdc live on this one-row aggregates table.
    df = DataFrame(region=[ed.region],
                   dva=round.([ed.dva]; digits=6),
                   rdv=round.([ed.rdv]; digits=6),
                   fva=round.([ed.fva]; digits=6),
                   pdc=round.([ed.pdc]; digits=6),
                   gross_exports=round.([ed.gross_exports]; digits=6),
                   vax_ratio=round.([ed.vax_ratio]; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="KWW Export Aggregates", key="kww_export_aggregates")
    ns = size(ed.by_sector, 1)
    labs = length(ed.sectors) == ns ? ed.sectors : ["sector$i" for i in 1:ns]
    sdf = DataFrame(sector=labs,
                    dva=round.(ed.by_sector[:, 1]; digits=6),
                    rdv=round.(ed.by_sector[:, 2]; digits=6),
                    fva=round.(ed.by_sector[:, 3]; digits=6),
                    pdc=round.(ed.by_sector[:, 4]; digits=6))
    output_result(sdf; format=Symbol(format), output=output,
                  title="KWW Export by Sector", key="kww_export_by_sector")
    _maybe_plot(ed; plot=plot, plot_save=plot_save)
    return ed
end

function _io_bilateral_trade(; data::String="", n_sectors::Int=0, n_fd::Int=1,
                             sectors::String="", parser::String="csv",
                             exporter::String="", importer::String="",
                             kind::String="total",
                             format::String="table", output::String="")
    isempty(strip(exporter)) && throw(CliError("usage/missing-option",
        "io bilateral-trade: --exporter is required"))
    isempty(strip(importer)) && throw(CliError("usage/missing-option",
        "io bilateral-trade: --importer is required"))
    io = _io_from_opts(; data=data, n_sectors=n_sectors, n_fd=n_fd, sectors=sectors, parser=parser)
    ex = _io_resolve_region(exporter)
    im = _io_resolve_region(importer)
    bt = try
        bilateral_trade(io, ex, im; kind=Symbol(kind))
    catch e
        throw(_domain_or_data_error(e, "bilateral_trade"))
    end
    output_kv([
        "intermediate" => round(bt.intermediate; digits=6),
        "final" => round(bt.final; digits=6),
        "total" => round(bt.total; digits=6),
        "exporter" => string(ex),
        "importer" => string(im),
        "kind" => kind,
    ]; format=Symbol(format), output=output, title="Bilateral Trade Summary")
    ns = length(bt.by_sector)
    labs = try
        idxs = collect(region_indices(io, ex))
        length(idxs) == ns ? io.sectors[idxs] : ["sector$i" for i in 1:ns]
    catch
        ["sector$i" for i in 1:ns]
    end
    df = DataFrame(sector=labs, by_sector=round.(bt.by_sector; digits=6))
    output_result(df; format=Symbol(format), output=output,
                  title="Bilateral Trade by Sector", key="bilateral_trade_by_sector")
    return bt
end
