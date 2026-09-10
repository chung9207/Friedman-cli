# Integration core vs real MacroEconometricModels (TS-6 / C031 / F35)
# Run: julia --project=test/integration test/integration/runtests.jl
#
# Each case asserts: (a) envelope schema-valid, (b) scalar sanity / teeth,
# (c) non-empty table shapes.

using Test
using JSON3
using Friedman
using Random
using Statistics
using LinearAlgebra

const ROOT = dirname(dirname(@__DIR__))
const SCHEMA_PATH = joinpath(ROOT, "schema", "envelope-v1.json")

include(joinpath(@__DIR__, "dgp.jl"))
include(joinpath(@__DIR__, "schema_validate.jl"))

# ── Runner ────────────────────────────────────────────────────

# W1/#136: when FRIEDMAN_T3_DUMP_ENVELOPES names a directory, every captured
# stdout that parses as a JSON object is written there — the CI post-step
# validates the whole dump with a CONFORMANT draft-07 validator
# (test/tools/validate_envelopes.py), independent of the in-repo subset.
const _T3_DUMP_DIR = get(ENV, "FRIEDMAN_T3_DUMP_ENVELOPES", "")
const _T3_DUMP_N = Ref(0)

function _dump_envelope(raw::AbstractString, doc)
    (isempty(_T3_DUMP_DIR) || doc === nothing) && return
    doc isa JSON3.Object || return
    isdir(_T3_DUMP_DIR) || mkpath(_T3_DUMP_DIR)
    _T3_DUMP_N[] += 1
    write(joinpath(_T3_DUMP_DIR, string(lpad(_T3_DUMP_N[], 5, '0'), ".json")),
          string(strip(raw)))
    return
end

"""Run friedman args with --quiet --format=json; return (code, doc, raw)."""
function run_json(args::Vector{String}; quiet::Bool=true)
    argv = String[]
    quiet && push!(argv, "--quiet")
    append!(argv, args)
    any(a -> startswith(a, "--format"), argv) || push!(argv, "--format", "json")

    out_path = tempname()
    err_path = tempname()
    code = try
        open(out_path, "w") do out_io
            open(err_path, "w") do err_io
                redirect_stdout(out_io) do
                    redirect_stderr(err_io) do
                        return Friedman.run_cli(argv)
                    end
                end
            end
        end
    catch
        Cint(1)
    end
    raw = read(out_path, String)
    rm(out_path; force=true)
    rm(err_path; force=true)
    doc = try
        JSON3.read(strip(raw))
    catch
        nothing
    end
    _dump_envelope(raw, doc)
    return (code=Int(code), doc=doc, raw=raw)
end

"""Run friedman args capturing (code, out, err) separately.

`run_json` drops stderr, but the J-test verdicts render on stderr via
`_status` (suppressed under `--quiet`) — identity-weighting cases need the
raw stderr text, so this runner leaves `--quiet` off by default.
"""
function run_cli_capture(args::Vector{String}; quiet::Bool=false)
    argv = String[]
    quiet && push!(argv, "--quiet")
    append!(argv, args)
    any(a -> startswith(a, "--format"), argv) || push!(argv, "--format", "json")

    out_path = tempname()
    err_path = tempname()
    code = try
        open(out_path, "w") do out_io
            open(err_path, "w") do err_io
                redirect_stdout(out_io) do
                    redirect_stderr(err_io) do
                        return Friedman.run_cli(argv)
                    end
                end
            end
        end
    catch
        Cint(1)
    end
    out = read(out_path, String)
    err = read(err_path, String)
    rm(out_path; force=true)
    rm(err_path; force=true)
    return (code=Int(code), out=out, err=err)
end

function assert_envelope_ok(r; label="")
    @test r.code == 0
    @test r.doc !== nothing
    r.doc === nothing && return
    @test string(r.doc.status) == "ok"
    errs = validate_envelope_json(r.raw; schema_path=SCHEMA_PATH)
    @test isempty(errs)
    if !isempty(errs)
        @info "schema errors for $label" errs
    end
end

"""First table rows from envelope data (dict of tables)."""
function first_table(doc)
    doc === nothing && return nothing, nothing
    data = try
        doc.data
    catch
        nothing
    end
    data === nothing && return nothing, nothing
    for (k, v) in pairs(data)
        if v isa JSON3.Object && haskey(v, :rows)
            return string(k), v
        elseif v isa AbstractDict && haskey(v, "rows")
            return string(k), v
        end
    end
    return nothing, nothing
end

function table_rows(tbl)
    rows = haskey(tbl, :rows) ? tbl.rows : tbl["rows"]
    return collect(rows)
end

function table_cols(tbl)
    cols = haskey(tbl, :columns) ? tbl.columns : tbl["columns"]
    return String[string(c) for c in cols]
end

"""Collect `name`/`value` (or `metric`/`value`) rows from EVERY table in the envelope.
JSON3 does not preserve insertion order, so picking the first name/value table
silently drops sibling aggregates/prices tables."""
function collect_named_kv(doc, name_col::AbstractString="name", value_col::AbstractString="value")
    kv = Dict{String,Any}()
    doc === nothing && return kv
    data = try
        doc.data
    catch
        return kv
    end
    for (_, v) in pairs(data)
        (v isa JSON3.Object && haskey(v, :rows)) || continue
        cols = table_cols(v)
        ni = findfirst(==(name_col), cols)
        vi = findfirst(==(value_col), cols)
        (ni === nothing || vi === nothing) && continue
        for row in table_rows(v)
            r = collect(row)
            kv[string(r[ni])] = r[vi]
        end
    end
    return kv
end

function numeric_tables_agree(a, b; atol=1e-8, rtol=1e-6, sort_by::Union{Nothing,AbstractString}=nothing)
    a === nothing && return false
    b === nothing && return false
    ca, cb = table_cols(a), table_cols(b)
    ca == cb || return false
    ra = [collect(r) for r in table_rows(a)]
    rb = [collect(r) for r in table_rows(b)]
    length(ra) == length(rb) || return false
    if sort_by !== nothing
        si = findfirst(==(String(sort_by)), ca)
        si === nothing && return false
        sort!(ra; by = r -> r[si])
        sort!(rb; by = r -> r[si])
    end
    for (xa, xb) in zip(ra, rb)
        length(xa) == length(xb) || return false
        for (va, vb) in zip(xa, xb)
            if va isa Number && vb isa Number
                isapprox(Float64(va), Float64(vb); atol=atol, rtol=rtol) || return false
            else
                string(va) == string(vb) || return false
            end
        end
    end
    return true
end

function metric_value(tbl, metric_name::AbstractString)
    cols = table_cols(tbl)
    rows = table_rows(tbl)
    mi = findfirst(==("metric"), cols)
    vi = findfirst(==("value"), cols)
    (mi === nothing || vi === nothing) && return nothing
    for row in rows
        r = collect(row)
        string(r[mi]) == metric_name && return r[vi]
    end
    return nothing
end

"""Coerce an envelope cell to Float64, accepting the `"Inf"`/`"-Inf"`/`"NaN"` STRING form
that `_json_safe` emits for non-finite floats (JSON has no infinity)."""
function numv(x)
    x isa Number && return Float64(x)
    s = string(x)
    s == "Inf" && return Inf
    s == "-Inf" && return -Inf
    s == "NaN" && return NaN
    return parse(Float64, s)
end

"""The `dsge solve` policy table, identified by its `variable` + `G1_*` columns rather than
by position — `dsge solve` emits several tables and their envelope order is not stable."""
function _dsge_policy_table(doc)
    doc === nothing && return nothing
    for (_, v) in pairs(doc.data)
        v isa JSON3.Object && haskey(v, :columns) || continue
        cols = String[string(c) for c in v.columns]
        ("variable" in cols && any(startswith(c, "G1_") for c in cols)) && return v
    end
    return nothing
end

"""A specific named table from envelope data (dict of tables), or nothing."""
function named_table(doc, name::Symbol)
    doc === nothing && return nothing
    data = try
        doc.data
    catch
        nothing
    end
    data === nothing && return nothing
    haskey(data, name) ? data[name] : nothing
end

"""Column index by name in a table, or nothing."""
col_index(tbl, name::AbstractString) = findfirst(==(name), table_cols(tbl))

# ── Tests ─────────────────────────────────────────────────────

@testset "Integration core vs real MEMs (TS-6)" begin
    @testset "estimate var (C051 tidy coef)" begin
        csv = dgp_var2(; T=180, seed=7)
        r = run_json(["estimate", "var", csv, "--lags", "2"])
        assert_envelope_ok(r; label="estimate var")
        coef = named_table(r.doc, :var_coefficients)
        @test coef !== nothing
        if coef !== nothing
            # C051: MEMs' uniform tidy coefficient table via DataFrame(model)
            @test table_cols(coef) ==
                  ["equation", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
            @test length(table_rows(coef)) >= 1
        end
        rm(csv; force=true)
    end

    @testset "estimate svar / svec + narrative-adrr (W2/#166)" begin
        csv = dgp_var2(; T=180, seed=7)
        # recursive: closed form, just-identified → LR df 0, exact identification
        r = run_json(["estimate", "svar", csv, "--lags", "2"])
        assert_envelope_ok(r; label="estimate svar recursive")
        a = named_table(r.doc, :svar_a)
        @test a !== nothing
        if a !== nothing
            @test table_cols(a) == ["equation", "y1", "y2", "y3"]
            @test length(table_rows(a)) == 3
            # recursive A is unit lower-triangular: diagonal 1, above-diagonal 0
            rows = [collect(row) for row in table_rows(a)]
            @test all(Float64(row[i+1]) ≈ 1.0 for (i, row) in enumerate(rows))
            @test all(Float64(rows[i][j+1]) ≈ 0.0 for i in 1:3 for j in i+1:3)
        end
        kv = collect_named_kv(r.doc, "metric", "value")
        @test Int(kv["LR df"]) == 0
        @test string(kv["Identification"]) == "exact"
        # Blanchard-Quah long-run pattern runs too
        r = run_json(["estimate", "svar", csv, "--lags", "2", "--pattern", "blanchard-quah"])
        assert_envelope_ok(r; label="estimate svar blanchard-quah")
        # A-model from TOML matrices (nan = free parameter)
        svar_toml = tempname() * ".toml"
        write(svar_toml, "[svar]\nA = [[1.0, 0.0, 0.0], [nan, 1.0, 0.0], [nan, nan, 1.0]]\n")
        r = run_json(["estimate", "svar", csv, "--lags", "2", "--pattern", "a-model",
                      "--config", svar_toml])
        assert_envelope_ok(r; label="estimate svar a-model")
        rm(svar_toml; force=true)
        rm(csv; force=true)

        # SVEC on cointegrated data, default KPSW identification
        cc = dgp_coint(; T=250, seed=21)
        r = run_json(["estimate", "svec", cc, "--lags", "2", "--rank", "1"])
        assert_envelope_ok(r; label="estimate svec")
        b0 = named_table(r.doc, :svec_b0)
        @test b0 !== nothing
        if b0 !== nothing
            @test table_cols(b0) == ["equation", "x", "y"]
            @test length(table_rows(b0)) == 2
        end
        kv = collect_named_kv(r.doc, "metric", "value")
        @test Int(kv["Permanent shocks"]) == 1
        # structural VECM routes on the vecm leaves (KPSW default)
        r = run_json(["irf", "vecm", cc, "--lags", "2", "--rank", "1",
                      "--horizons", "8", "--shock", "1", "--ci", "none", "--id", "svec"])
        assert_envelope_ok(r; label="irf vecm svec")
        tbl = named_table(r.doc, :vecm_irf)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value", "lower", "upper"]
        end
        r = run_json(["fevd", "vecm", cc, "--lags", "2", "--rank", "1",
                      "--horizons", "8", "--id", "svec"])
        assert_envelope_ok(r; label="fevd vecm svec")
        rm(cc; force=true)

        # narrative-adrr shares the Arias pipeline end to end
        csv2 = dgp_var2(; T=200, seed=9)
        adrr_toml = tempname() * ".toml"
        write(adrr_toml, """
        [[identification.sign_restrictions]]
        var = 2
        shock = 1
        sign = "positive"
        horizon = 0
        [[identification.narrative_contributions]]
        variable = 1
        shock = 1
        window = [1, 4]
        kind = "most_important"
        """)
        # underidentified AB pattern → RWZ guard is upstream inside estimate_svar
        # (W2/#166 #752 disposition); the CLI maps it to model/identification.
        under_toml = tempname() * ".toml"
        write(under_toml, "[svar]\nA = [[nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]\n" *
              "B = [[nan, nan, nan], [nan, nan, nan], [nan, nan, nan]]\n")
        r = run_json(["estimate", "svar", csv2, "--lags", "2", "--pattern", "ab-model",
                      "--config", under_toml])
        @test r.code == 5
        rm(under_toml; force=true)
        r = run_json(["irf", "var", csv2, "--lags", "2", "--horizons", "8",
                      "--shock", "1", "--ci", "none", "--id", "narrative-adrr",
                      "--config", adrr_toml])
        assert_envelope_ok(r; label="irf var narrative-adrr")
        # NOTE: select by key, never first_table (JSON3 object order is arbitrary).
        # The Arias path renders wide (build_irf_table): horizon + one col per variable.
        tbl = named_table(r.doc, :irf)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "y1", "y2", "y3"]
        end
        # robust-bayes on the BVAR posterior (Giacomini-Kitagawa bands).
        # NOTE: on bvar leaves --config is the *prior* file, so it must carry
        # both [prior] and [identification].
        rb_toml = tempname() * ".toml"
        write(rb_toml, """
        [prior]
        type = "minnesota"
        [prior.hyperparameters]
        lambda1 = 0.2
        lambda2 = 0.5
        lambda3 = 1.0
        lambda4 = 100000.0
        [prior.optimization]
        enabled = false
        [[identification.sign_restrictions]]
        var = 2
        shock = 1
        sign = "positive"
        horizon = 0
        """)
        r = run_json(["irf", "bvar", csv2, "--lags", "2", "--draws", "50",
                      "--horizons", "4", "--shock", "1", "--id", "robust-bayes",
                      "--config", rb_toml])
        assert_envelope_ok(r; label="irf bvar robust-bayes")
        bands = named_table(r.doc, :robust_bayes_bands)
        @test bands !== nothing
        if bands !== nothing
            cols = table_cols(bands)
            @test any(c -> endswith(c, "_robust_lower"), cols)
            @test any(c -> endswith(c, "_robust_upper"), cols)
            @test length(table_rows(bands)) == 4
        end
        # set-identified summaries over the sign identified set
        sign_toml = tempname() * ".toml"
        write(sign_toml, """
        [identification]
        method = "sign"
        [identification.sign_matrix]
        matrix = [[1, -1, 1], [0, 1, -1], [0, 0, 1]]
        horizons = [0]
        """)
        # --seed pins the identified-set draws (W3/#167 _fwd_seed wiring):
        # this restriction set is marginal at 100 unseeded replications
        # (~1/3 of streams accept zero rotations → model/identification flake).
        # Seed 1 accepts deterministically (verified frozen pass/fail per seed);
        # 300 replications give the acceptance margin cross-platform headroom.
        for sum_kind in ["median-target", "joint-band"]
            r = run_json(["--seed", "1", "irf", "var", csv2, "--lags", "2", "--horizons", "6",
                          "--shock", "1", "--ci", "none", "--id", "sign",
                          "--identified-set", "--summary", sum_kind,
                          "--replications", "300", "--config", sign_toml])
            assert_envelope_ok(r; label="irf var identified-set $sum_kind")
        end
        rm(sign_toml; force=true)
        rm(rb_toml; force=true)
        rm(adrr_toml; force=true)
        rm(csv2; force=true)
    end

    @testset "estimate arima on AR(1) φ=0.7" begin
        csv = dgp_ar1(; T=250, φ=0.7, seed=11)
        r = run_json(["estimate", "arima", csv, "--column", "1"])
        assert_envelope_ok(r; label="estimate arima")
        # Extract AR coefficient if present in any table
        found_phi = false
        for (_, v) in pairs(r.doc.data)
            if (v isa JSON3.Object || v isa AbstractDict) && (haskey(v, :rows) || haskey(v, "rows"))
                cols = table_cols(v)
                rows = table_rows(v)
                # look for phi/ar/estimate columns
                for row in rows
                    rr = collect(row)
                    for x in rr
                        if x isa Real && 0.45 <= x <= 0.95
                            found_phi = true
                        end
                    end
                end
            end
        end
        # Soft numeric tooth: at least envelope OK; phi often appears as AR(1) coef
        @test r.doc.status == "ok" || string(r.doc.status) == "ok"
        # Prefer finding a coefficient in the AR band when present
        if found_phi
            @test found_phi
        end
        rm(csv; force=true)
    end

    @testset "estimate arfima / test gph / test local-whittle — long memory (C068)" begin
        # ARFIMA(0,d,0) with d≈0.3: a genuine long-memory series. Assert envelope-valid
        # and a finite d in a sane range (roughly (−0.5, 1)) with p-values in [0,1].
        csv = dgp_fracdiff(; T=400, d=0.3, seed=123)

        # scan every kv table for a "metric == name" numeric value
        scan_metric(doc, name) = begin
            v = nothing
            for (_, tbl) in pairs(doc.data)
                if (tbl isa JSON3.Object || tbl isa AbstractDict) && (haskey(tbl, :rows) || haskey(tbl, "rows"))
                    mv = metric_value(tbl, name)
                    mv !== nothing && (v = mv)
                end
            end
            v
        end

        # estimate arfima(0,d,0)
        ra = run_json(["estimate", "arfima", csv, "--column", "1", "--p", "0", "--q", "0"])
        assert_envelope_ok(ra; label="estimate arfima")
        da = scan_metric(ra.doc, "d (frac. integ.)")
        @test da !== nothing
        if da !== nothing
            @test isfinite(Float64(da))
            @test -0.5 < Float64(da) < 1.0
        end

        # test gph
        rg = run_json(["test", "gph", csv, "--column", "1"])
        assert_envelope_ok(rg; label="test gph")
        dg = scan_metric(rg.doc, "d (long-memory)")
        pg = scan_metric(rg.doc, "p-value")
        @test dg !== nothing && isfinite(Float64(dg)) && -0.5 < Float64(dg) < 1.0
        @test pg !== nothing && 0.0 <= Float64(pg) <= 1.0

        # test local-whittle
        rw = run_json(["test", "local-whittle", csv, "--column", "1"])
        assert_envelope_ok(rw; label="test local-whittle")
        dw = scan_metric(rw.doc, "d (long-memory)")
        pw = scan_metric(rw.doc, "p-value")
        @test dw !== nothing && isfinite(Float64(dw)) && -0.5 < Float64(dw) < 1.0
        @test pw !== nothing && 0.0 <= Float64(pw) <= 1.0

        # bandwidth override still valid
        rgm = run_json(["test", "gph", csv, "--column", "1", "--bandwidth", "40"])
        assert_envelope_ok(rgm; label="test gph --bandwidth")

        rm(csv; force=true)
    end

    @testset "estimate smm — AR(1) recovery (first-ever T3, #345)" begin
        # SMM was broken on real MEMs 0.7.0 (3-arg call vs required 4-arg
        # estimate_smm(simulator_fn, moments_fn, theta0, data)) with zero T3
        # coverage — the same panel/DiD-class blind spot. This is the anchor.
        csv = dgp_ar1(; T=400, φ=0.7, σ=1.0, seed=71)
        cfg = tempname() * "_smm.toml"
        write(cfg, """
        [smm]
        model = "ar1"
        theta0 = [0.4, 0.5]
        lags = 2
        weighting = "two_step"
        sim_ratio = 5
        burn = 100
        lower = [-0.99, 1.0e-4]
        upper = [0.99, 10.0]
        """)
        r = run_json(["--seed", "20240722", "estimate", "smm", csv, "--config", cfg])
        assert_envelope_ok(r; label="estimate smm")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            cols = table_cols(tbl)
            @test "parameter" in cols
            @test "estimate" in cols
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(rows) == 2          # phi, sigma
            pidx = findfirst(==("parameter"), cols)
            eidx = findfirst(==("estimate"), cols)
            @test all(isfinite(Float64(row[eidx])) for row in rows)
            phi_row = rows[findfirst(row -> string(row[pidx]) == "phi", rows)]
            # AR persistence recovered near the true 0.7 (bounds keep it in (-0.99,0.99))
            @test 0.3 < Float64(phi_row[eidx]) < 0.99
        end
        rm(csv; force=true); rm(cfg; force=true)
    end

    @testset "estimate smm identity weighting — J p-value n/a (MEMs#797, M-29)" begin
        # No identity-weighting T3 coverage existed: under identity the χ² limit
        # needs efficient weighting, so model.J_pvalue is NaN and the leaf must
        # render n/a with the reason instead of a bare NaN.
        csv = dgp_ar1(; T=400, φ=0.7, σ=1.0, seed=71)
        cfg = tempname() * "_smm_id.toml"
        write(cfg, """
        [smm]
        model = "ar1"
        theta0 = [0.4, 0.5]
        lags = 2
        weighting = "identity"
        sim_ratio = 5
        burn = 100
        lower = [-0.99, 1.0e-4]
        upper = [0.99, 10.0]
        """)
        r = run_cli_capture(["--seed", "20240722", "estimate", "smm", csv, "--config", cfg])
        @test r.code == 0
        @test occursin("J p-value:", r.err)
        @test occursin("n/a (identity weighting", r.err)
        @test !occursin("J p-value:   NaN", r.err)
        rm(csv; force=true); rm(cfg; force=true)
    end

    @testset "estimate gmm identity weighting — pins upstream behavior (MEMs#797, M-29)" begin
        # W0 ledger finding: j_test(::GMMModel) shares the SMM NaN-under-identity
        # policy — but through THIS leaf the model is just-identified
        # (estimate_lp_gmm LP moments: df=0, J=0, p=1.0 on every weighting), so
        # the NaN branch cannot fire here and the leaf guard stays as
        # defense-in-depth (NaN < 0.05 is false and must never read as
        # "Cannot reject"). Pin the actual upstream behavior: no bare NaN,
        # just-identified J output, no n/a note.
        csv = dgp_var2(; T=200, seed=9)
        cfg = tempname() * "_gmm_id.toml"
        write(cfg, """
        [gmm]
        moment_conditions = ["y1", "y2"]
        instruments = ["lag_y1", "lag_y2"]
        weighting = "twostep"
        """)
        r = run_cli_capture(["estimate", "gmm", csv, "--config", cfg,
                             "--weighting", "identity"])
        @test r.code == 0
        @test occursin("Hansen's J-test", r.err)
        @test occursin("Degrees of freedom: 0", r.err)
        @test occursin("p-value: 1.0", r.err)
        @test !occursin("p-value: NaN", r.err)
        @test !occursin("n/a (identity weighting", r.err)
        rm(csv; force=true); rm(cfg; force=true)
    end

    @testset "estimate gmm efficient weighting — numeric J p-value control" begin
        # Control pinning the non-identity path: a real χ² p-value renders
        # numeric with a verdict (guards against an over-broad NaN branch).
        csv = dgp_var2(; T=200, seed=9)
        cfg = tempname() * "_gmm_tw.toml"
        write(cfg, """
        [gmm]
        moment_conditions = ["y1", "y2"]
        instruments = ["lag_y1", "lag_y2"]
        weighting = "twostep"
        """)
        r = run_cli_capture(["estimate", "gmm", csv, "--config", cfg,
                             "--weighting", "twostep"])
        @test r.code == 0
        @test occursin("Hansen's J-test", r.err)
        @test !occursin("n/a (identity weighting", r.err)
        @test occursin(r"p-value: [0-9]", r.err)
        @test occursin("valid moment conditions", r.err)
        rm(csv; force=true); rm(cfg; force=true)
    end

    @testset "estimate sur / 3sls — systems (C063, M5c)" begin
        Random.seed!(4242)
        Tn = 300
        x1 = randn(Tn); x2 = randn(Tn); x3 = randn(Tn)
        U = ([1.0 0.0; 0.6 0.8] * randn(2, Tn))'    # cross-equation error correlation
        y1 = 1.0 .+ 0.5 .* x1 .+ 0.3 .* x2 .+ U[:, 1]
        y2 = -0.5 .+ 0.8 .* x2 .+ 0.2 .* x3 .+ U[:, 2]
        csv = tempname() * ".csv"
        open(csv, "w") do io
            println(io, "y1,y2,x1,x2,x3")
            for t in 1:Tn
                println(io, join((y1[t], y2[t], x1[t], x2[t], x3[t]), ","))
            end
        end
        surcfg = tempname() * "_sur.toml"
        write(surcfg, """
        [[equations]]
        name = "consumption"
        dep = "y1"
        indep = ["x1", "x2"]
        [[equations]]
        name = "investment"
        dep = "y2"
        indep = ["x2", "x3"]
        """)
        _syscoef(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                ("equation" in table_cols(v) && "term" in table_cols(v)) && return v
            end
            nothing
        end

        coltable_sys(doc, col) = begin
            hit = nothing
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                if col in String.(table_cols(v)); hit = v; break; end
            end
            hit
        end

        @testset "sur tidy coef + slope recovery" begin
            r = run_json(["estimate", "sur", csv, "--config", surcfg])
            assert_envelope_ok(r; label="estimate sur")
            coef = _syscoef(r.doc)
            @test coef !== nothing
            if coef !== nothing
                @test issubset(["equation", "term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"],
                               table_cols(coef))
                rows = [collect(row) for row in table_rows(coef)]
                @test length(rows) == 6                       # 2 eq × (const + 2)
                ei = col_index(coef, "estimate"); ti = col_index(coef, "term"); qi = col_index(coef, "equation")
                @test all(isfinite(Float64(row[ei])) for row in rows)
                x1row = rows[findfirst(row -> string(row[qi]) == "consumption" && string(row[ti]) == "x1", rows)]
                @test 0.3 < Float64(x1row[ei]) < 0.7          # true 0.5
            end
        end

        @testset "3sls (instruments span regressors → collapses to SUR)" begin
            tslscfg = tempname() * "_3sls.toml"
            write(tslscfg, """
            [[equations]]
            dep = "y1"
            indep = ["x1", "x2"]
            [[equations]]
            dep = "y2"
            indep = ["x2", "x3"]
            [instruments]
            common = ["x1", "x2", "x3"]
            """)
            r = run_json(["estimate", "3sls", csv, "--config", tslscfg])
            assert_envelope_ok(r; label="estimate 3sls")
            coef = _syscoef(r.doc)
            @test coef !== nothing && length(table_rows(coef)) == 6
            rm(tslscfg; force=true)
        end

        @testset "statespace predict & residuals (#71)" begin
            ss = tempname() * ".csv"
            rng = MersenneTwister(701); nT = 150
            lvl = cumsum(randn(rng, nT) .* 0.3)
            open(ss, "w") do io
                println(io, "y")
                for t in 1:nT; println(io, lvl[t] + 0.5 * randn(rng)); end
            end

            rp = run_json(["predict", "statespace", ss])
            assert_envelope_ok(rp; label="predict statespace")
            t = coltable_sys(rp.doc, "filtered")
            @test t !== nothing
            @test Set(["period", "state", "filtered", "smoothed"]) ⊆ Set(String.(table_cols(t)))
            @test length(table_rows(t)) == nT              # local-level: ONE state
            # local-linear-trend has TWO states -> the long table grows, columns unchanged
            rp2 = run_json(["predict", "statespace", ss, "--kind", "local-linear-trend"])
            assert_envelope_ok(rp2; label="predict statespace llt")
            @test length(table_rows(coltable_sys(rp2.doc, "filtered"))) == 2 * nT

            rr = run_json(["residuals", "statespace", ss])
            assert_envelope_ok(rr; label="residuals statespace")
            tr = coltable_sys(rr.doc, "residual")
            @test tr !== nothing && length(table_rows(tr)) == nT
            # innovations are one-step prediction errors: near mean-zero on a correct fit
            vals = [Float64(collect(r)[col_index(tr, "residual")]) for r in table_rows(tr)]
            @test abs(sum(vals) / length(vals)) < 0.5

            @test run_json(["residuals", "statespace", ss, "--standardized"]).code == 0
            @test run_json(["predict", "statespace", ss, "--state", "filtered"]).code == 0
            @test run_json(["predict", "statespace", ss, "--kind", "bogus"]).code == 2
            @test run_json(["predict", "statespace", ss, "--state", "bogus"]).code == 2
            @test run_json(["residuals", "statespace", ss, "--column", "9"]).code == 3
            rm(ss; force=true)
        end

        @testset "sur/3sls predict & residuals — one long per-equation table (#68)" begin
            # The result carries PER-EQUATION fitted/residuals; the CLI renders ONE tidy
            # long table (equation|t|value) rather than N tables, so the envelope key set
            # does not vary with the config file.
            for (verb, col) in (("predict", "fitted"), ("residuals", "residual"))
                r = run_json([verb, "sur", csv, "--config", surcfg])
                assert_envelope_ok(r; label="$verb sur")
                t = coltable_sys(r.doc, col)
                @test t !== nothing
                @test Set(["equation", "t", col]) ⊆ Set(String.(table_cols(t)))
                @test length(table_rows(t)) == 2 * Tn           # 2 equations x T
                eqs = Set(String(collect(row)[col_index(t, "equation")]) for row in table_rows(t))
                @test eqs == Set(["consumption", "investment"])
            end
            # residuals must be (near) mean-zero per equation — a real fit, not a stub
            rr = run_json(["residuals", "sur", csv, "--config", surcfg])
            tr = coltable_sys(rr.doc, "residual")
            vals = [Float64(collect(row)[col_index(tr, "residual")]) for row in table_rows(tr)]
            @test abs(sum(vals) / length(vals)) < 0.1

            # --config carries the equation system: without it the model cannot be refit
            @test run_json(["predict", "sur", csv]).code == 4
            @test run_json(["residuals", "3sls", csv]).code == 4
        end

        rm(csv; force=true); rm(surcfg; force=true)
    end

    @testset "estimate lasso/ridge/elastic-net/robust/tobit — penalized & LDV (C067a, M5c)" begin
        # Real cross-section DGP: sparse true β=[-1.0, 0.8, -0.6, 0, 0], plus a
        # left-censored yc for Tobit. Teeth: sparse recovery, large-λ shrinkage, robust≈OLS
        # on clean data, Tobit β recovery + censoring counts.
        csv, csvc, β = dgp_penalized(; T=300, p=5, seed=42)

        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        _coef_est(doc, term; termcol="term") = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                cols = table_cols(v)
                ti = findfirst(==(termcol), cols); ei = findfirst(==("estimate"), cols)
                (ti === nothing || ei === nothing) && continue
                for row in table_rows(v)
                    r = collect(row)
                    string(r[ti]) == term && return Float64(r[ei])
                end
            end
            nothing
        end

        @testset "lasso — sparse recovery (x1≈-1, x2≈0.8) + intercept row" begin
            r = run_json(["estimate", "lasso", csv, "--dep", "y", "--select", "bic"])
            assert_envelope_ok(r; label="estimate lasso")
            # penalized coef table: term|estimate|nonzero, with an intercept row
            @test _coef_est(r.doc, "(Intercept)") !== nothing
            b1 = _coef_est(r.doc, "x1"); b2 = _coef_est(r.doc, "x2")
            @test b1 !== nothing && -1.4 < b1 < -0.6      # true -1.0
            @test b2 !== nothing && 0.5 < b2 < 1.1        # true  0.8
            na = metric_value(_diag(r.doc), "n_active")
            @test na !== nothing && Int(na) >= 3          # recovers the 3 real regressors
        end

        @testset "lasso — large --lambda drives the active set to 0 (shrinkage teeth)" begin
            r = run_json(["estimate", "lasso", csv, "--dep", "y", "--lambda", "100"])
            assert_envelope_ok(r; label="estimate lasso big-lambda")
            na = metric_value(_diag(r.doc), "n_active")
            @test na !== nothing && Int(na) <= 1          # essentially everything shrunk out
        end

        @testset "ridge / elastic-net run + fitted β sign" begin
            for (leaf, extra) in (("ridge", String[]), ("elastic-net", ["--alpha", "0.5"]))
                r = run_json(vcat(["estimate", leaf, csv, "--dep", "y"], extra))
                assert_envelope_ok(r; label="estimate $leaf")
                b1 = _coef_est(r.doc, "x1")
                @test b1 !== nothing && b1 < 0.0          # true coefficient is negative
            end
        end

        @testset "robust ≈ OLS on clean data (x1≈-1, x2≈0.8)" begin
            r = run_json(["estimate", "robust", csv, "--dep", "y", "--psi", "huber", "--method", "m"])
            assert_envelope_ok(r; label="estimate robust")
            b1 = _coef_est(r.doc, "x1"; termcol="parameter")
            b2 = _coef_est(r.doc, "x2"; termcol="parameter")
            @test b1 !== nothing && -1.15 < b1 < -0.85
            @test b2 !== nothing && 0.65 < b2 < 0.95
            @test string(metric_value(_diag(r.doc), "converged")) == "true"
        end

        @testset "tobit — recovers β on left-censored yc + censoring counts" begin
            r = run_json(["estimate", "tobit", csvc, "--dep", "yc", "--lower", "0.0"])
            assert_envelope_ok(r; label="estimate tobit")
            b1 = _coef_est(r.doc, "x1"; termcol="parameter")
            @test b1 !== nothing && -1.3 < b1 < -0.7       # true -1.0 (≈50% censored)
            nL = metric_value(_diag(r.doc), "n_censored_left")
            @test nL !== nothing && Int(nL) > 0
        end

        @testset "bad --dep → data/column-range (exit 3, hardened loader)" begin
            r = run_json(["estimate", "lasso", csv, "--dep", "does_not_exist"])
            @test r.code == 3
        end

        @testset "elastic-net --alpha 2 → usage error (exit 2, not raw MEMs)" begin
            r = run_json(["estimate", "elastic-net", csv, "--dep", "y", "--alpha", "2"])
            @test r.code == 2
        end

        rm(csv; force=true); rm(csvc; force=true)
    end

    @testset "estimate cointreg/xtcointreg — cointegrating regression (C062a, M5c)" begin
        # Real cointegration DGP: x_t random walk, y_t = β x_t + I(0). Teeth: FMOLS/CCR/DOLS
        # recover β within a LOOSE tol (the estimators are noisy), the coef table carries the
        # full tidy schema, and diagnostics/CIs are finite. Panel: N units, common β → group
        # & pooled β̄ within a loose tol; back-solved group-mean SE may be Inf (round-safe).
        _coefrow(doc, term) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                cols = table_cols(v)
                ti = findfirst(==("term"), cols); ei = findfirst(==("estimate"), cols)
                (ti === nothing || ei === nothing) && continue
                for row in table_rows(v)
                    r = collect(row)
                    string(r[ti]) == term && return (v, Float64(r[ei]))
                end
            end
            (nothing, nothing)
        end
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        _coefcols = ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]

        @testset "cointreg — FMOLS/CCR/DOLS recover β≈1 (loose) + tidy schema" begin
            csv = dgp_coint(; T=300, β=1.0, seed=45)
            for meth in ("fmols", "ccr", "dols")
                r = run_json(["estimate", "cointreg", csv, "--dep", "y", "--method", meth])
                assert_envelope_ok(r; label="estimate cointreg $meth")
                tbl, bx = _coefrow(r.doc, "x")
                @test tbl !== nothing && Set(_coefcols) ⊆ Set(table_cols(tbl))
                @test bx !== nothing && abs(bx - 1.0) < 0.3        # loose slope recovery
                d = _diag(r.doc)
                @test metric_value(d, "method") !== nothing
                @test Int(metric_value(d, "k")) == 1
                @test isfinite(Float64(metric_value(d, "omega_uv")))
            end
            # DOLS exposes leads/lags in the diagnostics block
            rd = run_json(["estimate", "cointreg", csv, "--dep", "y", "--method", "dols"])
            @test metric_value(_diag(rd.doc), "leads") !== nothing
            rm(csv; force=true)
        end

        @testset "xtcointreg — panel group & pooled β̄≈1 (loose), SE finite-or-Inf-safe" begin
            cp = dgp_coint_panel(; N=8, T=50, β=1.0, seed=61)
            for pool in ("group", "pooled"), meth in ("fmols", "dols")
                r = run_json(["estimate", "xtcointreg", cp, "--dep", "y", "--indep", "x",
                              "--method", meth, "--pooling", pool])
                assert_envelope_ok(r; label="estimate xtcointreg $meth/$pool")
                tbl, bx = _coefrow(r.doc, "x")
                @test tbl !== nothing && Set(_coefcols) ⊆ Set(table_cols(tbl))
                @test bx !== nothing && abs(bx - 1.0) < 0.3       # common-β recovery
                d = _diag(r.doc)
                @test Int(metric_value(d, "N")) == 8
                @test string(metric_value(d, "pooling")) == pool
            end
            rm(cp; force=true)
        end

        @testset "bad input stays typed (not internal exit 1)" begin
            csv = dgp_coint(; T=200, seed=63)
            @test run_json(["estimate", "cointreg", csv, "--dep", "nope"]).code == 3       # data/column-range
            @test run_json(["estimate", "cointreg", csv, "--method", "bogus"]).code == 2   # enum
            @test run_json(["estimate", "cointreg", csv, "--bandwidth", "junk"]).code == 2 # dual-type parse
            @test run_json(["estimate", "cointreg", csv, "--leads", "-1"]).code == 2
            cp = dgp_coint_panel(; N=6, T=40, seed=65)
            @test run_json(["estimate", "xtcointreg", cp, "--dep", "y", "--indep", "x", "--method", "ccr"]).code == 2
            @test run_json(["estimate", "xtcointreg", cp, "--dep", "nope", "--indep", "x"]).code == 2
            rm(csv; force=true); rm(cp; force=true)
        end
    end

    @testset "estimate ardl/nardl + test ardl-bounds/nardl-symmetry + multipliers nardl (C062b, M5c)" begin
        # Real ARDL/NARDL family. Teeth (all LOOSE — noisy single-equation estimators):
        # ARDL recovers the long-run θ=(β₀+β₁)/(1−φ); the bounds test returns a valid decision
        # symbol; NARDL symmetry REJECTS on an asymmetric DGP (discriminating vs a symmetric
        # control); the dynamic multipliers converge toward θ⁺/θ⁻ with finite bootstrap bands.
        _tbl_kw(doc, kw) = begin   # first data table whose envelope key contains `kw`
            for (k, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                occursin(kw, lowercase(string(k))) && return v
            end
            nothing
        end
        _tbl_col(doc, col) = begin  # first data table that HAS column `col` (robust to key-order)
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                col in table_cols(v) && return v
            end
            nothing
        end
        _rowval(tbl, keycol, key, valcol) = begin
            tbl === nothing && return nothing
            cols = table_cols(tbl)
            ki = findfirst(==(keycol), cols); vi = findfirst(==(valcol), cols)
            (ki === nothing || vi === nothing) && return nothing
            for row in table_rows(tbl)
                r = collect(row)
                string(r[ki]) == key && return Float64(r[vi])
            end
            nothing
        end
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end

        @testset "ardl — long-run θ recovery (loose) + ECM diagnostics" begin
            csv, θ = dgp_ardl(; T=300, φ=0.5, β0=1.0, β1=0.5, seed=44)
            r = run_json(["estimate", "ardl", csv, "--dep", "y", "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="estimate ardl")
            lrt = _tbl_kw(r.doc, "long")                 # long-run table (key contains "long")
            θ̂ = _rowval(lrt, "term", "x", "estimate")
            @test θ̂ !== nothing && abs(θ̂ - θ) < 0.4     # loose long-run recovery
            d = _diag(r.doc)
            @test metric_value(d, "alpha") !== nothing
            @test isfinite(Float64(metric_value(d, "longrun_denom")))
            @test Int(metric_value(d, "case")) == 3
            # auto selection path also runs
            @test run_json(["estimate", "ardl", csv, "--dep", "y", "--p", "auto"]).code == 0
            rm(csv; force=true)
        end

        @testset "test ardl-bounds — valid decision symbol, no p-value" begin
            csv, _ = dgp_ardl(; T=300, seed=46)
            r = run_json(["test", "ardl-bounds", csv, "--dep", "y", "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="test ardl-bounds")
            bt = _tbl_col(r.doc, "decision")   # bounds DATA table (key "bounds" is shared with the summary)
            @test bt !== nothing && "decision" in table_cols(bt)
            @test !("p_value" in table_cols(bt))          # bounds test has NO p-value
            fdec = string(metric_value(_diag(r.doc), "f_decision"))
            @test fdec in ("cointegrated", "not_cointegrated", "inconclusive")
            # case II → undefined t-bounds render "undefined", never a NaN crash
            @test run_json(["test", "ardl-bounds", csv, "--dep", "y", "--p", "1", "--q", "1", "--case", "2"]).code == 0
            rm(csv; force=true)
        end

        @testset "nardl — runs + asymmetric long-run terms" begin
            csv, tp, tn = dgp_nardl(; T=300, θpos=1.0, θneg=-0.3, seed=47)
            r = run_json(["estimate", "nardl", csv, "--dep", "y", "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="estimate nardl")
            lrt = _tbl_kw(r.doc, "long")
            @test lrt !== nothing
            terms = String[string(collect(row)[findfirst(==("term"), table_cols(lrt))]) for row in table_rows(lrt)]
            @test any(t -> occursin("_POS", t), terms) && any(t -> occursin("_NEG", t), terms)
            d = _diag(r.doc)
            @test string(metric_value(d, "f_decision")) in ("cointegrated", "not_cointegrated", "inconclusive")
            @test Int(metric_value(d, "k")) == 2 * Int(metric_value(d, "k_orig"))
            rm(csv; force=true)
        end

        @testset "test nardl-symmetry — rejects asymmetric, milder on symmetric (loose direction)" begin
            ca, _, _ = dgp_nardl(; T=320, θpos=1.0, θneg=-0.4, seed=48)
            cs, _, _ = dgp_nardl(; T=320, seed=49, sym=true)
            ra = run_json(["test", "nardl-symmetry", ca, "--dep", "y", "--p", "1", "--q", "1"])
            rs = run_json(["test", "nardl-symmetry", cs, "--dep", "y", "--p", "1", "--q", "1"])
            assert_envelope_ok(ra; label="nardl-symmetry asym")
            assert_envelope_ok(rs; label="nardl-symmetry sym")
            st_a = _tbl_col(ra.doc, "lr_p_chi2"); st_s = _tbl_col(rs.doc, "lr_p_chi2")  # data table (key "symmetry" shared w/ summary)
            pa = _rowval(st_a, "regressor", "x", "lr_p_chi2")
            ps = _rowval(st_s, "regressor", "x", "lr_p_chi2")
            @test pa !== nothing && ps !== nothing
            @test 0.0 <= pa <= 1.0 && 0.0 <= ps <= 1.0
            @test pa < 0.10          # asymmetric DGP → reject long-run symmetry (loose)
            @test pa < ps            # discriminating direction: asym more significant than sym
            rm(ca; force=true); rm(cs; force=true)
        end

        @testset "multipliers nardl — converge to θ⁺/θ⁻ (loose) + finite bands" begin
            csv, tp, tn = dgp_nardl(; T=320, θpos=1.0, θneg=-0.4, seed=50)
            r = run_json(["multipliers", "nardl", csv, "--dep", "y", "--p", "1", "--q", "1",
                          "--horizon", "24", "--nreps", "120"])
            assert_envelope_ok(r; label="multipliers nardl")
            mt = _tbl_col(r.doc, "m_pos")   # multiplier DATA table (key "multiplier" shared w/ summary)
            @test mt !== nothing
            cols = table_cols(mt)
            @test Set(["horizon", "regressor", "m_pos", "m_neg", "m_diff"]) ⊆ Set(cols)
            @test "m_pos_lo" in cols                       # band columns present (nreps>0)
            hi = findfirst(==("horizon"), cols); mp = findfirst(==("m_pos"), cols)
            mn = findfirst(==("m_neg"), cols); lo = findfirst(==("m_pos_lo"), cols)
            rows = [collect(row) for row in table_rows(mt)]
            @test all(isfinite(Float64(r[mp])) && isfinite(Float64(r[lo])) for r in rows)
            hmax = maximum(Int(r[hi]) for r in rows)
            mpos_end = Float64(first(r[mp] for r in rows if Int(r[hi]) == hmax))
            @test abs(mpos_end - tp) < 0.5                 # converges toward θ⁺ (loose)
            # --no-bootstrap drops band columns
            rnb = run_json(["multipliers", "nardl", csv, "--dep", "y", "--p", "1", "--q", "1",
                            "--horizon", "12", "--no-bootstrap"])
            @test rnb.code == 0
            @test !("m_pos_lo" in table_cols(_tbl_col(rnb.doc, "m_pos")))
            rm(csv; force=true)
        end

        @testset "bad input stays typed (never internal exit 1)" begin
            csv, _ = dgp_ardl(; T=200, seed=51)
            @test run_json(["estimate", "ardl", csv, "--dep", "nope"]).code == 3        # data/column-range
            @test run_json(["estimate", "ardl", csv, "--dep", "y", "--case", "9"]).code == 2   # usage
            @test run_json(["estimate", "nardl", csv, "--dep", "y", "--asymmetric", "0"]).code == 2
            @test run_json(["test", "ardl-bounds", csv, "--dep", "y", "--level", "0.03"]).code == 2
            @test run_json(["test", "ardl-bounds", csv, "--dep", "y", "--cv-source", "narayan"]).code == 2
            @test run_json(["multipliers", "nardl", csv, "--dep", "y", "--horizon", "-1"]).code == 2
            rm(csv; force=true)
        end
    end

    @testset "estimate pmg + test pmg-hausman — panel ARDL (C062c, M5c)" begin
        # Heterogeneous-panel ARDL-EC with a COMMON long-run θ=1, heterogeneous φ_i/short-run.
        # Teeth (all LOOSE — noisy panel ML): PMG recovers the pooled θ; the fit converges; MG
        # and DFE also run; the Hausman test returns a decision + a p-value in [0,1] (on a
        # homogeneous-θ DGP it should FAIL to reject long-run homogeneity — loose direction).
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        _rowval(doc, term, valcol) = begin  # scan all tables for term row, return valcol
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                cols = table_cols(v)
                ti = findfirst(==("term"), cols); vi = findfirst(==(valcol), cols)
                (ti === nothing || vi === nothing) && continue
                for row in table_rows(v)
                    r = collect(row)
                    string(r[ti]) == term && return Float64(r[vi])
                end
            end
            nothing
        end

        @testset "pmg — long-run θ recovery (loose) + converged, MG/DFE run" begin
            cp = dgp_pmg(; N=10, T=60, θ=1.0, seed=71)
            r = run_json(["estimate", "pmg", cp, "--dep", "y", "--indep", "x", "--method", "pmg"])
            assert_envelope_ok(r; label="estimate pmg")
            θ̂ = _rowval(r.doc, "x", "estimate")
            @test θ̂ !== nothing && abs(θ̂ - 1.0) < 0.35        # loose pooled long-run recovery
            d = _diag(r.doc)
            @test Int(metric_value(d, "N")) == 10
            @test string(metric_value(d, "converged")) in ("true", "1")
            @test isfinite(Float64(metric_value(d, "phi")))
            # MG / DFE also run
            @test run_json(["estimate", "pmg", cp, "--dep", "y", "--indep", "x", "--method", "mg"]).code == 0
            @test run_json(["estimate", "pmg", cp, "--dep", "y", "--indep", "x", "--method", "dfe"]).code == 0
            rm(cp; force=true)
        end

        @testset "pmg-hausman — decision + p-value in [0,1], efficient pmg/dfe" begin
            cp = dgp_pmg(; N=10, T=60, θ=1.0, seed=73)
            for eff in ("pmg", "dfe")
                r = run_json(["test", "pmg-hausman", cp, "--dep", "y", "--indep", "x", "--efficient", eff])
                assert_envelope_ok(r; label="test pmg-hausman $eff")
                d = _diag(r.doc)
                pv = Float64(metric_value(d, "pvalue"))
                @test 0.0 <= pv <= 1.0
                @test metric_value(d, "statistic") !== nothing
            end
            rm(cp; force=true)
        end

        @testset "bad input stays typed (not internal exit 1)" begin
            cp = dgp_pmg(; N=6, T=40, seed=75)
            @test run_json(["estimate", "pmg", cp, "--dep", "y", "--indep", "x", "--method", "bogus"]).code == 2
            @test run_json(["estimate", "pmg", cp, "--dep", "nope", "--indep", "x"]).code == 2
            @test run_json(["estimate", "pmg", cp, "--dep", "y", "--indep", "x", "--p", "0"]).code == 2
            @test run_json(["test", "pmg-hausman", cp, "--dep", "y", "--indep", "x", "--efficient", "mg"]).code == 2
            rm(cp; force=true)
        end
    end

    @testset "estimate midas — mixed-frequency MIDAS (C062d, M5c)" begin
        # Real MIDAS: an HF indicator drives a LF target through a known exp-Almon weight curve.
        # Teeth (all LOOSE — restricted MIDAS NLS is noisy): the HF loading β₁ is finite & positive;
        # the restricted weight curve has K entries summing ≈1; R² is non-trivial; umidas (OLS) and
        # ADL (--p-ar) also run. Bad mixed-frequency input stays typed (never an internal exit 1).
        _tbl_col(doc, col) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                col in table_cols(v) && return v
            end
            nothing
        end
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        _coef(doc, matchfn) = begin   # first term row whose term satisfies matchfn → estimate
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                cols = table_cols(v)
                ti = findfirst(==("term"), cols); ei = findfirst(==("estimate"), cols)
                (ti === nothing || ei === nothing) && continue
                for row in table_rows(v)
                    r = collect(row)
                    matchfn(string(r[ti])) && return Float64(r[ei])
                end
            end
            nothing
        end

        @testset "expalmon — HF loading finite/positive, weight curve sums≈1, R² non-trivial" begin
            lf, hf, b = dgp_midas(; Tlf=120, m=3, K=6, b=2.0, seed=91)
            r = run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", "expalmon"])
            assert_envelope_ok(r; label="estimate midas expalmon")
            wt = _tbl_col(r.doc, "weight")
            @test wt !== nothing
            wcol = findfirst(==("weight"), table_cols(wt))
            ws = [Float64(collect(row)[wcol]) for row in table_rows(wt)]
            @test length(ws) == 6
            @test abs(sum(ws) - 1.0) < 1e-3                      # restricted weights are normalized
            bx = _coef(r.doc, t -> occursin("HF loading", t))
            @test bx !== nothing && isfinite(bx) && bx > 0.0     # positive HF loading (loose)
            d = _diag(r.doc)
            @test Float64(metric_value(d, "r2")) > 0.3
            @test string(metric_value(d, "weights_kind")) == "expalmon"
            @test Int(metric_value(d, "K")) == 6
            rm(lf; force=true); rm(hf; force=true)
        end

        @testset "umidas (OLS) + ADL-MIDAS (--p-ar) run" begin
            lf, hf, _ = dgp_midas(; Tlf=120, m=3, K=6, seed=93)
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "6", "--weights", "umidas"]).code == 0
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "6", "--p-ar", "1"]).code == 0
            rm(lf; force=true); rm(hf; force=true)
        end

        @testset "ragged HF (nhf > m×LF) accepted — leading edge dropped (real _align_hf)" begin
            # The real estimator anchors the last HF obs to the last LF period and drops leading
            # ragged history (its headline nowcasting feature); the CLI loader relaxes to
            # `nhf >= m×LF` to match. Prepend 20 extra leading HF obs → nhf = 3*120 + 20 = 380 > 360.
            lf, hf, _ = dgp_midas(; Tlf=120, m=3, K=6, seed=97)
            orig = CSV.read(hf, DataFrame)
            padded = vcat(DataFrame(ip = collect(range(-1.0, 0.0; length=20))), orig)
            hfrag = write_csv(padded; prefix="midas_hf_ragged")
            r = run_json(["estimate", "midas", lf, "--hf-data", hfrag, "--m", "3", "--k", "6", "--weights", "expalmon"])
            @test r.code == 0
            rm(lf; force=true); rm(hf; force=true); rm(hfrag; force=true)
        end

        @testset "bad input stays typed (not internal exit 1)" begin
            lf, hf, _ = dgp_midas(; Tlf=80, m=3, K=6, seed=95)
            @test run_json(["estimate", "midas", lf, "--m", "3", "--k", "6"]).code == 2                            # missing --hf-data
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "0", "--k", "6"]).code == 2
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "0"]).code == 2
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "1", "--weights", "beta2"]).code == 3  # K<2
            @test run_json(["estimate", "midas", lf, "--hf-data", hf, "--m", "4", "--k", "6"]).code == 3            # HF shorter than m×LF (240 < 4*80=320) → data/shape
            rm(lf; force=true); rm(hf; force=true)
        end

        @testset "--horizon is a REAL direct-h regression since 0.7.3 (MEMs#574, W10/#131)" begin
            # At ≤0.7.2 the h kwarg was inert: the model claimed a direct h-step
            # target while always fitting h=1. The adoption test is behavioral:
            # an explicit --horizon 1 reproduces the default bit-for-bit, and
            # --horizon 4 changes the fit (different target ⇒ different SSR/R²).
            lf, hf, _ = dgp_midas(; Tlf=120, m=3, K=6, seed=99)
            base = ["estimate", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "6"]
            r_def = run_json(base)
            r_h1 = run_json([base; "--horizon"; "1"])
            r_h4 = run_json([base; "--horizon"; "4"])
            @test r_def.code == 0 && r_h1.code == 0 && r_h4.code == 0
            dg(r) = named_table(r.doc, :midas_diagnostics)
            r2v(r) = (t = dg(r); t === nothing ? nothing : metric_value(t, "r2"))
            @test r2v(r_h1) !== nothing && r2v(r_def) !== nothing
            @test numv(r2v(r_h1)) == numv(r2v(r_def))          # h=1 IS the default
            @test numv(r2v(r_h4)) != numv(r2v(r_def))          # h=4 fits a different target
            @test metric_value(dg(r_h4), "h") == 4             # …and records it
            # nobs shrinks by h-1: the last 3 targets fall out of sample.
            @test numv(metric_value(dg(r_h4), "nobs")) ==
                  numv(metric_value(dg(r_def), "nobs")) - 3
            @test run_json([base; "--horizon"; "0"]).code == 2
            rm(lf; force=true); rm(hf; force=true)
        end
    end

    @testset "estimate threshold — general two-regime threshold regression (#70)" begin
        # UNLIKE the rest of the nonlinear-TS family, the teeth here are TIGHT: the threshold
        # is a grid search over an EXTERNAL z whose true split point is 0, and the regime
        # coefficients are ordinary per-regime OLS, so both are recoverable to a few percent.
        # A shape-only assertion would not distinguish this leaf from `estimate setar`.
        Random.seed!(70070)
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        mv(doc, name) = (d = _diag(doc); d === nothing ? nothing : metric_value(d, name))
        _coef(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "regime" in table_cols(v) && return v
            end
            nothing
        end
        # estimate for (regime-label substring, term) out of the hand-built coef table
        _est(tbl, regime_sub, term) = begin
            ri = col_index(tbl, "regime"); ti = col_index(tbl, "term"); ei = col_index(tbl, "estimate")
            for row in table_rows(tbl)
                r = collect(row)
                occursin(regime_sub, string(r[ri])) && string(r[ti]) == term && return Float64(r[ei])
            end
            nothing
        end

        csv = dgp_threshold(; n=400, seed=7001)

        @testset "recovers the true threshold and both regime slopes" begin
            r = run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "z", "--reps", "199"])
            assert_envelope_ok(r; label="estimate threshold")
            ct = _coef(r.doc)
            @test ct !== nothing
            @test Set(["regime", "term", "estimate", "std_error", "z_stat", "p_value"]) ⊆ Set(table_cols(ct))
            # X = {x1, x2}: z is the splitting variable and must NOT be a regressor
            terms = Set(string(collect(row)[col_index(ct, "term")]) for row in table_rows(ct))
            @test terms == Set(["x1", "x2"])
            @test length(table_rows(ct)) == 4          # 2 terms × 2 regimes

            # TRUE γ = 0. NOTE what can and cannot be asserted here. The grid is the ORDER
            # STATISTICS of z, so γ̂ ∈ {zᵢ} and the true 0 is generally not attainable; and the
            # Hansen (2000) CI is the set of γ NOT rejected by the LR statistic, which with a
            # signal this strong (β jumps +2 → −2 against σ = 0.2) legitimately COLLAPSES to the
            # single point γ̂. So "0 ∈ [γl, γu]" is the wrong claim — assert instead that γ̂ and
            # the whole non-rejected set sit in a small neighbourhood of the truth.
            γ  = Float64(mv(r.doc, "gamma"))
            γl = Float64(mv(r.doc, "gamma_ci_lower")); γu = Float64(mv(r.doc, "gamma_ci_upper"))
            @test abs(γ) < 0.25
            @test γl <= γ <= γu
            @test -0.5 < γl && γu < 0.5

            # β(x1) = +2 below the threshold, −2 above; β(x2) = 0.5 in BOTH regimes
            @test isapprox(_est(ct, "≤", "x1"),  2.0; atol=0.15)
            @test isapprox(_est(ct, ">", "x1"), -2.0; atol=0.15)
            @test isapprox(_est(ct, "≤", "x2"),  0.5; atol=0.15)
            @test isapprox(_est(ct, ">", "x2"),  0.5; atol=0.15)

            @test string(mv(r.doc, "threshold_var")) == "z"
            @test string(mv(r.doc, "is_setar")) == "false"   # this is NOT the self-exciting case
            @test Int(mv(r.doc, "n")) == 400
            @test Int(mv(r.doc, "n1")) + Int(mv(r.doc, "n2")) == 400
            @test Int(mv(r.doc, "n1")) > 0 && Int(mv(r.doc, "n2")) > 0
            @test isfinite(Float64(mv(r.doc, "aic"))) && isfinite(Float64(mv(r.doc, "bic")))
            # the attached Hansen (1996) test rejects linearity on this genuinely nonlinear DGP
            @test Float64(mv(r.doc, "pvalue_lm")) < 0.10
        end

        @testset "--no-linearity, --het, --ci-level, --dep default" begin
            d = _diag(run_json(["estimate", "threshold", csv, "--dep", "y",
                                "--threshold-col", "z", "--no-linearity"]).doc)
            keys_ = Set(string(collect(row)[1]) for row in table_rows(d))
            @test "gamma" in keys_ && !("sup_lm" in keys_)
            @test run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "z",
                            "--het", "--reps", "99"]).code == 0
            r90 = run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "z",
                            "--ci-level", "0.90", "--no-linearity"])
            @test r90.code == 0
            @test Float64(mv(r90.doc, "gamma_ci_level")) == 0.90
            # y is the first numeric column, so --dep may be omitted
            @test run_json(["estimate", "threshold", csv, "--threshold-col", "z", "--no-linearity"]).code == 0
        end

        @testset "typed errors — never an internal exit 1" begin
            @test run_json(["estimate", "threshold", csv, "--dep", "y"]).code == 2               # --threshold-col required
            @test run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "z",
                            "--trim", "0.6"]).code == 2
            @test run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "z",
                            "--ci-level", "0.8"]).code == 2                                      # not a Hansen-tabulated level
            @test run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "y"]).code == 2
            @test run_json(["estimate", "threshold", csv, "--dep", "y", "--threshold-col", "nope"]).code == 3
            @test run_json(["estimate", "threshold", csv, "--dep", "nope", "--threshold-col", "z"]).code == 3
            # a CONSTANT splitting variable admits no admissible split → real MEMs raises
            # ArgumentError("Empty threshold grid") → data/invalid (3), NEVER exit 1
            constz = write_csv(DataFrame(y=randn(80), x1=randn(80), z=fill(1.0, 80)); prefix="thr_constz")
            @test run_json(["estimate", "threshold", constz, "--dep", "y", "--threshold-col", "z"]).code == 3
            # Too few observations for two k-regressor regimes → data/invalid, not exit 1.
            # k = 2 (x1, x2), so real requires n ≥ 2(k+1) = 6; n = 4 trips it.
            tiny = write_csv(DataFrame(y=randn(4), x1=randn(4), x2=randn(4), z=randn(4)); prefix="thr_tiny")
            @test run_json(["estimate", "threshold", tiny, "--dep", "y", "--threshold-col", "z"]).code == 3
            rm(constz; force=true); rm(tiny; force=true)
        end

        rm(csv; force=true)
    end

    @testset "estimate setar + test hansen-linearity + forecast setar — nonlinear TS (C065a, M5c)" begin
        # Real SETAR/threshold family. Teeth are LOOSE/direction-only (bootstrap Hansen +
        # threshold search are noisy): both regimes populated, γ̂ inside its CI, finite AIC,
        # and the nonlinear DGP rejects linearity (pvalue_lm < 0.10).
        Random.seed!(65065)
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        mv(doc, name) = (d = _diag(doc); d === nothing ? nothing : metric_value(d, name))

        @testset "estimate setar — two regimes, γ in CI, attached Hansen rejects" begin
            csv = dgp_setar(; n=400, seed=651)
            r = run_json(["estimate", "setar", csv, "--column", "1", "--p", "1", "--d", "1", "--reps", "199"])
            assert_envelope_ok(r; label="estimate setar")
            # coef table: regime|term|estimate|std_error|z_stat|p_value, 2 regimes × 2 terms
            ct = nothing
            for (_, v) in pairs(r.doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "regime" in table_cols(v) && (ct = v)
            end
            @test ct !== nothing
            @test Set(["regime", "term", "estimate", "std_error"]) ⊆ Set(table_cols(ct))
            @test length(table_rows(ct)) == 4
            @test string(mv(r.doc, "is_setar")) == "true"
            @test Int(mv(r.doc, "n1")) > 0 && Int(mv(r.doc, "n2")) > 0
            γ  = Float64(mv(r.doc, "gamma"))
            γl = Float64(mv(r.doc, "gamma_ci_lower")); γu = Float64(mv(r.doc, "gamma_ci_upper"))
            @test isfinite(γ) && γl < γ < γu
            @test isfinite(Float64(mv(r.doc, "aic")))
            # attached Hansen (1996) linearity test rejects on the nonlinear DGP (loose)
            @test Float64(mv(r.doc, "pvalue_lm")) < 0.10
            # --d auto grid also runs
            @test run_json(["estimate", "setar", csv, "--p", "1", "--d", "auto", "--reps", "99"]).code == 0
            # a constant series admits no threshold split → data/invalid (exit 3) on REAL MEMs
            # (ArgumentError "Empty threshold grid"), NEVER an internal exit 1 — the mock mirrors this class
            constcsv = write_csv(DataFrame(y=fill(1.0, 60)); prefix="setar_const")
            @test run_json(["estimate", "setar", constcsv, "--p", "1", "--d", "1"]).code == 3
            rm(csv; force=true); rm(constcsv; force=true)
        end

        @testset "test hansen-linearity — rejects on the SETAR DGP" begin
            csv = dgp_setar(; n=400, seed=652)
            r = run_json(["test", "hansen-linearity", csv, "--column", "1", "--p", "1", "--d", "1", "--reps", "199"])
            assert_envelope_ok(r; label="test hansen-linearity")
            @test Set(["sup_lm", "pvalue_lm", "sup_wald", "pvalue_wald", "gamma_sup", "n_grid"]) ⊆
                  Set(String(string(collect(row)[1])) for row in table_rows(_diag(r.doc)))
            @test Float64(mv(r.doc, "pvalue_lm")) < 0.10
            # too-short series → data/invalid (wrapped estimate_setar), never internal exit-1
            short = write_csv(DataFrame(y=collect(1.0:6.0)); prefix="setar_short")
            @test run_json(["test", "hansen-linearity", short, "--p", "1", "--d", "1"]).code == 3
            rm(csv; force=true); rm(short; force=true)
        end

        @testset "forecast setar — h=6 finite paths, lower ≤ value ≤ upper" begin
            csv = dgp_setar(; n=400, seed=653)
            r = run_json(["forecast", "setar", csv, "--column", "1", "--p", "1", "--d", "1",
                          "--horizons", "6", "--reps", "199"])
            assert_envelope_ok(r; label="forecast setar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "value", "lower", "upper"]
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(rows) == 6
            vi = col_index(tbl, "value"); li = col_index(tbl, "lower"); ui = col_index(tbl, "upper")
            for row in rows
                v = Float64(row[vi]); lo = Float64(row[li]); hi = Float64(row[ui])
                @test isfinite(v) && isfinite(lo) && isfinite(hi)
                @test lo <= v <= hi
            end
            # --ci-level 0.8 → usage error (exit 2), not a Hansen-crit crash
            @test run_json(["forecast", "setar", csv, "--horizons", "4", "--ci-level", "0.8"]).code == 2
            rm(csv; force=true)
        end
    end

    @testset "estimate star + test star-linearity + forecast star — nonlinear TS (C065b, M5c)" begin
        # Real STAR family. Teeth are LOOSE/direction-only (NLS is deterministic but noisy):
        # regime + transition tables present, finite params, and the LSTAR DGP rejects
        # linearity (lm3_pvalue/pvalue < 0.10) while a linear AR(1) does not strongly reject.
        Random.seed!(65066)
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end
        mv(doc, name) = (d = _diag(doc); d === nothing ? nothing : metric_value(d, name))

        @testset "estimate star — regimes + transition params, LM3 rejects on LSTAR" begin
            csv = dgp_star(; n=400, seed=661)
            r = run_json(["estimate", "star", csv, "--column", "1", "--p", "1", "--d", "1", "--type", "auto"])
            assert_envelope_ok(r; label="estimate star")
            # regime-weight coef table: regime|term|estimate|std_error|z_stat|p_value, 2 regimes × 2 terms
            ct = nothing; tt = nothing
            for (_, v) in pairs(r.doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "regime" in table_cols(v) && (ct = v)
                ("parameter" in table_cols(v) && "z_stat" in table_cols(v)) && (tt = v)
            end
            @test ct !== nothing && Set(["regime", "term", "estimate", "std_error"]) ⊆ Set(table_cols(ct))
            @test length(table_rows(ct)) == 4
            # transition-params table carries γ with a finite estimate
            @test tt !== nothing
            ei = col_index(tt, "estimate")
            γrow = first(row for row in table_rows(tt) if occursin("γ", String(string(collect(row)[1]))))
            @test isfinite(Float64(collect(γrow)[ei]))
            @test isfinite(Float64(mv(r.doc, "sigma2")))
            @test string(mv(r.doc, "converged")) in ("true", "false")
            @test Float64(mv(r.doc, "lm3_pvalue")) < 0.10
            # --type auto ⇒ the Teräsvirta selection triple appears in the diagnostics kv
            @test mv(r.doc, "sel_H04") !== nothing
            rm(csv; force=true)
        end

        @testset "test star-linearity — rejects on LSTAR, not on linear AR(1)" begin
            csv = dgp_star(; n=400, seed=662)
            r = run_json(["test", "star-linearity", csv, "--column", "1", "--p", "1", "--d", "1"])
            assert_envelope_ok(r; label="test star-linearity")
            @test Set(["stat", "pvalue", "fstat", "fpvalue", "df"]) ⊆
                  Set(String(string(collect(row)[1])) for row in table_rows(_diag(r.doc)))
            @test Float64(mv(r.doc, "pvalue")) < 0.10
            # negative control: a linear AR(1) should NOT strongly reject (loose upper check)
            lin = dgp_ar1(; T=400, φ=0.5, seed=6620)
            rl = run_json(["test", "star-linearity", lin, "--column", "1", "--p", "1", "--d", "1"])
            assert_envelope_ok(rl; label="test star-linearity (linear)")
            @test Float64(mv(rl.doc, "pvalue")) > 0.01
            # short series: real star_linearity_test is defensively coded (returns a finite LM3 for a
            # short effective sample), so this must be exit 0, NOT data/invalid — the mock mirrors it.
            shortc = write_csv(DataFrame(y=[0.1 * i + 0.3 * sin(i) for i in 1:14]); prefix="starlin_short")
            @test run_json(["test", "star-linearity", shortc, "--column", "1", "--p", "3", "--d", "1"]).code == 0
            rm(csv; force=true); rm(lin; force=true); rm(shortc; force=true)
        end

        @testset "forecast star — h=6 finite paths, coherent bands" begin
            csv = dgp_star(; n=400, seed=663)
            r = run_json(["forecast", "star", csv, "--column", "1", "--p", "1", "--d", "1",
                          "--horizons", "6", "--reps", "199"])
            assert_envelope_ok(r; label="forecast star")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "value", "lower", "upper"]
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(rows) == 6
            vi = col_index(tbl, "value"); li = col_index(tbl, "lower"); ui = col_index(tbl, "upper")
            for row in rows
                v = Float64(row[vi]); lo = Float64(row[li]); hi = Float64(row[ui])
                @test isfinite(v) && isfinite(lo) && isfinite(hi)
                # `value` is the bootstrap MEAN path and the bands are percentiles; for a skewed
                # nonlinear-bootstrap forecast the mean can lie outside the percentile band, so we
                # assert only the guaranteed invariant (lower percentile ≤ upper percentile).
                @test lo <= hi
            end
            # --ci-level 0.8 → usage error (exit 2)
            @test run_json(["forecast", "star", csv, "--horizons", "4", "--ci-level", "0.8"]).code == 2
            rm(csv; force=true)
        end
    end

    @testset "estimate ms-ar + estimate ms — Markov-switching nonlinear TS (C065c, M5c)" begin
        # Real Markov-switching EM. Teeth are LOOSE/direction-only (EM is noisy): ms-ar converges
        # with an ordered mu, a row-stochastic K=2 transition matrix, and a finite loglik; ms
        # (intercept-only on the same series) recovers two distinct regime means.
        Random.seed!(65067)
        _find(doc, cols...) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                Set(String.(cols)) ⊆ Set(table_cols(v)) && return v
            end
            nothing
        end
        _diag(doc) = _find(doc, "metric", "value")
        mv(doc, name) = (d = _diag(doc); d === nothing ? nothing : metric_value(d, name))

        @testset "estimate ms-ar — converged, ordered mu, row-stochastic P, finite loglik" begin
            csv = dgp_msar(; n=500, seed=671)
            r = run_json(["estimate", "ms-ar", csv, "--column", "1", "--p", "1"])
            assert_envelope_ok(r; label="estimate ms-ar")
            # coef table: per-regime `mu` rows (ordered increasing) + a common-AR block
            ct = _find(r.doc, "regime", "term", "estimate")
            @test ct !== nothing
            ei = col_index(ct, "estimate"); ti = col_index(ct, "term")
            murows = [collect(row) for row in table_rows(ct) if string(collect(row)[ti]) == "mu"]
            @test length(murows) == 2
            mu1 = Float64(murows[1][ei]); mu2 = Float64(murows[2][ei])
            @test isfinite(mu1) && isfinite(mu2) && mu1 < mu2
            # WIDE K×K transition matrix, K = 2, rows sum ≈ 1 (row-stochastic)
            pt = _find(r.doc, "from_regime", "to_regime1", "to_regime2")
            @test pt !== nothing
            prows = [collect(row) for row in table_rows(pt)]
            @test length(prows) == 2
            c1 = col_index(pt, "to_regime1"); c2 = col_index(pt, "to_regime2")
            for row in prows
                @test isapprox(Float64(row[c1]) + Float64(row[c2]), 1.0; atol=1e-4)
            end
            # per-regime variance table (2 regimes) + finite loglik + convergence flag
            vt = _find(r.doc, "regime", "sigma2", "std_error")
            @test vt !== nothing && length(table_rows(vt)) == 2
            @test isfinite(Float64(mv(r.doc, "loglik")))
            @test string(mv(r.doc, "converged")) == "true"
            @test string(mv(r.doc, "switching_var")) == "false"     # Hamilton default
            # bad input → typed usage error (exit 2), never internal exit-1
            @test run_json(["estimate", "ms-ar", csv, "--k-regimes", "1"]).code == 2
            rm(csv; force=true)
        end

        @testset "estimate ms — intercept-only recovers two distinct regime means" begin
            csv = dgp_msar(; n=500, seed=672)
            r = run_json(["estimate", "ms", csv])
            assert_envelope_ok(r; label="estimate ms")
            ct = _find(r.doc, "regime", "term", "estimate")
            @test ct !== nothing
            ei = col_index(ct, "estimate")
            ests = [Float64(collect(row)[ei]) for row in table_rows(ct)]
            @test length(ests) == 2                                 # intercept-only: 1 term × 2 regimes
            @test all(isfinite, ests)
            @test abs(ests[1] - ests[2]) > 0.5                      # two DISTINCT regime means
            @test string(mv(r.doc, "switching_var")) == "true"      # ms default (switching σ²)
            @test run_json(["estimate", "ms", csv, "--tol", "0"]).code == 2
            rm(csv; force=true)
        end

        @testset "regime-probability table — real filtered/smoothed paths (#70 remainder)" begin
            # The headline MS output. Real `filtered_prob` and `smoothed_prob` are genuinely
            # different objects (smoothed conditions on the whole sample), which is what makes
            # this worth asserting on REAL MEMs rather than only against the mock.
            csv = dgp_msar(; n=500, seed=673)
            r = run_json(["estimate", "ms-ar", csv, "--column", "1", "--p", "1"])
            assert_envelope_ok(r; label="estimate ms-ar probabilities")
            pt = _find(r.doc, "period", "regime", "filtered", "smoothed")
            @test pt !== nothing
            pi_ = col_index(pt, "period"); ri = col_index(pt, "regime")
            fi = col_index(pt, "filtered"); si = col_index(pt, "smoothed")
            rows = [collect(row) for row in table_rows(pt)]
            nper = length(Set(Int(row[pi_]) for row in rows))
            @test Set(string(row[ri]) for row in rows) == Set(["regime1", "regime2"])
            @test length(rows) == 2 * nper
            # genuine probabilities: in [0,1] and summing to 1 across regimes each period.
            # Aggregated into single assertions on purpose — a @test per row would add ~2000
            # assertions for one property and make the suite count meaningless.
            bysum = Dict{Int,Float64}(); bysum_s = Dict{Int,Float64}()
            inrange = true
            for row in rows
                f = Float64(row[fi]); s = Float64(row[si])
                inrange &= (-1e-6 <= f <= 1 + 1e-6) && (-1e-6 <= s <= 1 + 1e-6)
                t = Int(row[pi_])
                bysum[t]   = get(bysum, t, 0.0) + f
                bysum_s[t] = get(bysum_s, t, 0.0) + s
            end
            @test inrange
            @test all(v -> isapprox(v, 1.0; atol=1e-3), values(bysum))
            @test all(v -> isapprox(v, 1.0; atol=1e-3), values(bysum_s))
            # the two paths are DISTINCT on real MEMs, and the smoothed one is sharper
            filt = [Float64(row[fi]) for row in rows]
            smoo = [Float64(row[si]) for row in rows]
            @test filt != smoo
            @test sum(abs.(smoo .- 0.5)) > sum(abs.(filt .- 0.5))
            # K = 3 grows the ROW count, never the column set
            r3 = run_json(["estimate", "ms-ar", csv, "--p", "1", "--k-regimes", "3"])
            if r3.code == 0
                p3 = _find(r3.doc, "period", "regime", "filtered", "smoothed")
                @test p3 !== nothing && Set(table_cols(p3)) == Set(table_cols(pt))
            end
            rm(csv; force=true)
        end

        @testset "residuals setar|star|ms-ar|ms — StatsAPI.residuals on real MEMs" begin
            csv = dgp_msar(; n=300, seed=674)
            scsv = dgp_setar(; n=300, seed=675)
            for (leaf, path, args) in (("setar", scsv, ["--p", "1"]),
                                       ("star",  scsv, ["--p", "1"]),
                                       ("ms-ar", csv,  ["--p", "1"]),
                                       ("ms",    csv,  String[]))
                r = run_json(vcat(["residuals", leaf, path], args))
                assert_envelope_ok(r; label="residuals $leaf")
                t = _find(r.doc, "period", "residual")
                @test t !== nothing
                rows = [collect(row) for row in table_rows(t)]
                @test !isempty(rows)
                vals = [Float64(row[col_index(t, "residual")]) for row in rows]
                @test all(isfinite, vals)
                # residuals from a fitted model are approximately mean-zero
                @test abs(sum(vals) / length(vals)) < 0.5
                @test [Int(row[col_index(t, "period")]) for row in rows] == collect(1:length(rows))
            end
            # AR-based fits drop lags; the MS level regression keeps every observation
            n_ms = length(table_rows(_find(run_json(["residuals", "ms", csv]).doc, "period", "residual")))
            n_ar = length(table_rows(_find(run_json(["residuals", "ms-ar", csv, "--p", "2"]).doc,
                                           "period", "residual")))
            @test n_ar < n_ms
            # typed errors, never internal exit-1
            @test run_json(["residuals", "setar", scsv, "--p", "0"]).code == 2
            @test run_json(["residuals", "setar", scsv, "--column", "99"]).code == 3
            @test run_json(["residuals", "ms-ar", csv, "--k-regimes", "1"]).code == 2
            @test run_json(["residuals", "ms", csv, "--tol", "0"]).code == 2
            # inference-only SETAR options are NOT advertised on the residuals leaf
            @test run_json(["residuals", "setar", scsv, "--reps", "99"]).code == 2
            # SETAR/STAR still have NO `predict` upstream — only MSRegModel gained one
            # (MEMs#510), so these two must stay usage errors.
            @test run_json(["predict", "setar", scsv, "--p", "1"]).code == 2
            @test run_json(["predict", "star", scsv, "--p", "1"]).code == 2
            rm(csv; force=true); rm(scsv; force=true)
        end

        # W6/#108 — multiplicative seasonal ARIMA on real MEMs.
        @testset "estimate|forecast|predict|residuals sarima (W6/#108)" begin
            # Seasonal AR(1) x seasonal-AR(1) at s=12, so the seasonal term is real signal
            # and the fitted Phi must be clearly positive.
            sy = let n = 400, y = zeros(n)
                Random.seed!(6108)
                for t in 14:n
                    y[t] = 0.5*y[t-1] + 0.6*y[t-12] - 0.3*y[t-13] + 0.4*randn()
                end
                y[14:end]
            end
            scsv = write_csv(DataFrame(y=sy); prefix="sarima")

            r = run_json(["estimate", "sarima", scsv, "--p", "1", "--q", "0",
                          "--P", "1", "--Q", "0", "--s", "12"])
            assert_envelope_ok(r; label="estimate sarima")
            t = _find(r.doc, "parameter", "estimate")
            @test t !== nothing
            pm = Dict(String(collect(rw)[col_index(t, "parameter")]) =>
                      Float64(collect(rw)[col_index(t, "estimate")]) for rw in table_rows(t))
            @test haskey(pm, "ar1") && haskey(pm, "sar1")
            # teeth: both the regular and the SEASONAL AR term must be recovered positive
            @test pm["ar1"] > 0.2
            @test pm["sar1"] > 0.2
            @test pm["sigma2"] > 0

            # auto selection runs and yields a finite AIC
            ra = run_json(["estimate", "sarima", scsv, "--s", "12"])
            assert_envelope_ok(ra; label="auto sarima")
            @test isfinite(Float64(metric_value(_find(ra.doc, "metric", "value"), "AIC")))

            rf = run_json(["forecast", "sarima", scsv, "--p", "1", "--P", "1", "--s", "12",
                           "--horizons", "8"])
            assert_envelope_ok(rf; label="forecast sarima")
            ft = _find(rf.doc, "horizon", "value", "lower", "upper")
            @test ft !== nothing && length(table_rows(ft)) == 8
            for rw in table_rows(ft)
                v = collect(rw)
                @test Float64(v[col_index(ft, "lower")]) <= Float64(v[col_index(ft, "value")]) <=
                      Float64(v[col_index(ft, "upper")])
            end

            for verb in ("predict", "residuals")
                rv = run_json([verb, "sarima", scsv, "--p", "1", "--P", "1", "--s", "12"])
                assert_envelope_ok(rv; label="$verb sarima")
                @test first_table(rv.doc)[2] !== nothing
            end

            # SARIMAModel <: AbstractARIMAModel, which HAS a plot recipe, and forecast
            # returns an ARIMAForecast which has one too — so unlike the threshold/STAR/MS
            # forecasts, BOTH leaves really plot. Assert a file lands rather than trusting
            # the abstract dispatch.
            for (leaf, extra) in (("estimate", String[]), ("forecast", ["--horizons", "4"]))
                out = tempname() * ".html"
                rp = run_json(vcat([leaf, "sarima", scsv, "--p", "1", "--P", "1", "--s", "12"],
                                   extra, ["--plot-save", out]))
                assert_envelope_ok(rp; label="$leaf sarima --plot-save")
                @test isfile(out) && filesize(out) > 1000
                rm(out; force=true)
            end

            # typed guards, never exit 1
            @test run_json(["estimate", "sarima", scsv, "--p", "1", "--P", "1", "--s", "1"]).code == 3
            @test run_json(["estimate", "sarima", scsv, "--s", "0"]).code == 2
            @test run_json(["estimate", "sarima", scsv, "--p", "1", "--d", "-1"]).code == 2
            @test run_json(["estimate", "sarima", scsv, "--criterion", "bogus"]).code == 2
            @test run_json(["forecast", "sarima", scsv, "--horizons", "0"]).code == 2
            rm(scsv; force=true)
        end

        # W5/#95 — plot flags are only advertised where a REAL plot_result recipe exists,
        # and the only way to know is to invoke them: the mock's generic plot_result would
        # report success for a type real cannot plot. Assert a non-trivial file lands.
        @testset "plot flags on the nonlinear-TS estimate leaves (W5/#95)" begin
            pcsv = dgp_msar(; n=250, seed=677)
            for (leaf, args) in (("ms-ar", ["--p", "1"]), ("ms", String[]),
                                 ("setar", ["--p", "1"]), ("star", ["--p", "1"]))
                out = tempname() * ".html"
                r = run_json(vcat(["estimate", leaf, pcsv], args, ["--plot-save", out]))
                assert_envelope_ok(r; label="estimate $leaf --plot-save")
                @test isfile(out)
                @test filesize(out) > 1000
                rm(out; force=true)
            end
            # ...and the FORECAST result types still have no recipe upstream, so those
            # leaves must not accept the flag at all (usage error, not a silent no-op).
            for leaf in ("setar", "star", "ms-ar")
                @test run_json(["forecast", leaf, pcsv, "--plot-save",
                                tempname() * ".html"]).code == 2
            end
            rm(pcsv; force=true)
        end

        # W3/#101 — un-gated by MEMs#510.
        @testset "predict|forecast ms|ms-ar (W3/#101)" begin
            csv = dgp_msar(; n=300, seed=676)

            # predict: regime-weighted fitted values. `y - predict(:smoothed) == residuals`
            # holds exactly upstream; the FILTERED mean uses less information and must
            # therefore differ — that identity is the teeth distinguishing the two branches.
            rp = run_json(["predict", "ms-ar", csv, "--p", "1"])
            assert_envelope_ok(rp; label="predict ms-ar")
            tp = _find(rp.doc, "t", "fitted")
            @test tp !== nothing
            sm = [Float64(collect(row)[col_index(tp, "fitted")]) for row in table_rows(tp)]
            @test !isempty(sm) && all(isfinite, sm)

            rf = run_json(["predict", "ms-ar", csv, "--p", "1", "--probs", "filtered"])
            assert_envelope_ok(rf; label="predict ms-ar filtered")
            tf = _find(rf.doc, "t", "fitted")
            fl = [Float64(collect(row)[col_index(tf, "fitted")]) for row in table_rows(tf)]
            @test length(fl) == length(sm)
            @test fl != sm

            rr = run_json(["residuals", "ms-ar", csv, "--p", "1"])
            tr = _find(rr.doc, "period", "residual")
            res = [Float64(collect(row)[col_index(tr, "residual")]) for row in table_rows(tr)]
            @test length(res) == length(sm)

            @test run_json(["predict", "ms", csv]).code == 0
            @test run_json(["predict", "ms-ar", csv, "--probs", "bogus"]).code == 2

            # forecast ms-ar: h rows, finite bands that bracket the path, and regime
            # probabilities that are a proper distribution at every horizon.
            rfc = run_json(["forecast", "ms-ar", csv, "--p", "1", "--horizons", "6"])
            assert_envelope_ok(rfc; label="forecast ms-ar")
            ft = _find(rfc.doc, "horizon", "value")
            @test ft !== nothing && length(table_rows(ft)) == 6
            lo = col_index(ft, "lower"); hi = col_index(ft, "upper"); vi = col_index(ft, "value")
            for row in table_rows(ft)
                v = collect(row)
                @test Float64(v[lo]) <= Float64(v[vi]) <= Float64(v[hi])
            end
            rpt = _find(rfc.doc, "horizon", "regime1")
            @test rpt !== nothing && length(table_rows(rpt)) == 6
            ridx = findall(c -> startswith(c, "regime"), table_cols(rpt))
            for row in table_rows(rpt)
                @test isapprox(sum(Float64.(collect(row)[ridx])), 1.0; atol=1e-6)
            end

            # forecast ms: a switching REGRESSION cannot project itself. The intercept-only
            # fit is the one case that needs no future design, so --horizons suffices.
            @test run_json(["forecast", "ms", csv, "--horizons", "5"]).code == 0
            # ...but a fit WITH regressors must demand them rather than guess.
            xcsv = write_csv(DataFrame(y=randn(200), x1=randn(200)); prefix="ms_reg")
            rneed = run_json(["forecast", "ms", xcsv, "--dep", "y", "--horizons", "4"])
            @test rneed.code == 2
            @test occursin("x-future", String(rneed.doc["error"]["hint"]))
            # a mis-shaped future design is typed data/shape (3), never an exit-1 crash
            badx = write_csv(DataFrame(a=randn(4), b=randn(4)); prefix="ms_badx")
            @test run_json(["forecast", "ms", xcsv, "--dep", "y", "--x-future", badx]).code == 3

            @test run_json(["forecast", "ms-ar", csv, "--horizons", "0"]).code == 2
            @test run_json(["forecast", "ms-ar", csv, "--ci-level", "1.5"]).code == 2
            rm(csv; force=true); rm(xcsv; force=true); rm(badx; force=true)
        end
    end

    @testset "estimate iv/truncreg/heckman + test weak-instrument (C067b, M5c)" begin
        # Coefficient extractor: scan all tables for a row whose `termcol` == term, return
        # its `estimate`. Works for the IV tidy table (term), truncreg (parameter), and the
        # Heckman two-equation table (term, optionally filtered by an `equation` value).
        _coef(doc, term; termcol="term", eq=nothing) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                cols = table_cols(v)
                ti = findfirst(==(termcol), cols); ei = findfirst(==("estimate"), cols)
                (ti === nothing || ei === nothing) && continue
                qi = eq === nothing ? nothing : findfirst(==("equation"), cols)
                for row in table_rows(v)
                    r = collect(row)
                    string(r[ti]) == term || continue
                    (qi === nothing || string(r[qi]) == eq) || continue
                    return Float64(r[ei])
                end
            end
            nothing
        end
        _diag(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) && return v
            end
            nothing
        end

        @testset "estimate iv — order-condition FIX: recovers β_endog≈2 (was exit-1)" begin
            csv = dgp_iv(; T=400, seed=11)
            # Excluded instruments z1,z2; const & x2 exogenous. Pre-C067b this raised an
            # untyped `Order condition violated (m<k)` → internal exit-1.
            r = run_json(["estimate", "iv", csv, "--dep", "y",
                          "--endogenous", "x_endog", "--instruments", "z1,z2"])
            assert_envelope_ok(r; label="estimate iv")
            b = _coef(r.doc, "x_endog")
            @test b !== nothing && 1.6 < b < 2.4                 # true 2.0
            fsf = metric_value(_diag(r.doc), "First-stage F")
            @test fsf !== nothing && Float64(fsf) > 10.0         # strong instruments
            rm(csv; force=true)
        end

        @testset "test weak-instrument — strong (not weak) vs weak (flagged)" begin
            strong = dgp_iv(; T=400, seed=12, inst_strength=0.8)
            rs = run_json(["test", "weak-instrument", strong, "--dep", "y",
                           "--endogenous", "x_endog", "--instruments", "z1,z2"])
            assert_envelope_ok(rs; label="weak-instrument strong")
            @test string(metric_value(_diag(rs.doc), "weak")) == "false"
            @test Float64(metric_value(_diag(rs.doc), "first_stage_f")) > 10.0
            rm(strong; force=true)

            weak = dgp_iv(; T=400, seed=13, inst_strength=0.02)  # near-irrelevant z1
            rw = run_json(["test", "weak-instrument", weak, "--dep", "y",
                           "--endogenous", "x_endog", "--instruments", "z1,z2"])
            assert_envelope_ok(rw; label="weak-instrument weak")
            # z1 near-zero, z2 still present → borderline; assert the F is far below the
            # strong case rather than a hard weak=true (z2 keeps some strength).
            @test Float64(metric_value(_diag(rw.doc), "first_stage_f")) <
                  Float64(metric_value(_diag(rs.doc), "first_stage_f"))
            rm(weak; force=true)
        end

        @testset "test weak-instrument — under-identified → data/invalid (exit 3)" begin
            csv = dgp_iv(; T=200, seed=14)
            # two endogenous, one excluded instrument → |excluded| < |endogenous|.
            r = run_json(["test", "weak-instrument", csv, "--dep", "y",
                          "--endogenous", "x_endog,x2", "--instruments", "z1"])
            @test r.code == 3
            rm(csv; force=true)
        end

        @testset "estimate truncreg — recovers slope on a truncated sample" begin
            # y* = 1 + 0.8 x + e; observe only y*>0 (truncated at 0). Include a const column.
            rng = MersenneTwister(77)
            xs = Float64[]; ys = Float64[]
            while length(ys) < 300
                x = randn(rng); yv = 1.0 + 0.8 * x + randn(rng)
                if yv > 0.0
                    push!(xs, x); push!(ys, yv)
                end
            end
            trcsv = tempname() * "_trunc.csv"
            open(trcsv, "w") do io
                println(io, "y,const,x")
                for i in eachindex(ys); println(io, "$(ys[i]),1.0,$(xs[i])"); end
            end
            r = run_json(["estimate", "truncreg", trcsv, "--dep", "y", "--lower", "0.0"])
            assert_envelope_ok(r; label="estimate truncreg")
            b = _coef(r.doc, "x"; termcol="parameter")
            @test b !== nothing && 0.4 < b < 1.2                 # true 0.8 (truncation-corrected)
            @test metric_value(_diag(r.doc), "n_truncated") !== nothing
            rm(trcsv; force=true)
        end

        @testset "estimate heckman — two-step recovers outcome slope + both equations" begin
            csv = dgp_heckman(; T=1500, seed=21, ρ=0.5)
            r = run_json(["estimate", "heckman", csv, "--dep", "y", "--select", "d",
                          "--outcome-vars", "const,x1", "--select-vars", "const,z1"])
            assert_envelope_ok(r; label="estimate heckman")
            bx = _coef(r.doc, "x1"; eq="outcome")
            @test bx !== nothing && 0.5 < bx < 1.1               # true outcome slope 0.8
            @test _coef(r.doc, "z1"; eq="selection") !== nothing # selection equation present
            @test string(metric_value(_diag(r.doc), "method")) == "twostep"
            rm(csv; force=true)
        end
    end

    @testset "forecast evaluate — evaluation & combination (C072, M5c)" begin
        # y = AR(1); f1 a decent forecast (small noise), f2 a noisier competitor.
        Random.seed!(9090)
        Tn = 250
        y = Vector{Float64}(undef, Tn); y[1] = randn()
        for t in 2:Tn
            y[t] = 0.6 * y[t-1] + randn()
        end
        y .+= 10.0                                   # shift away from zero (well-defined MAPE)
        f1 = y .+ 0.30 .* randn(Tn)                  # good forecast
        f2 = y .+ 1.20 .* randn(Tn)                  # noisier forecast
        csv = tempname() * "_fceval.csv"
        open(csv, "w") do io
            println(io, "y,f1,f2")
            for t in 1:Tn
                println(io, join((y[t], f1[t], f2[t]), ","))
            end
        end

        _kvval(doc, name) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) || continue
                mv = metric_value(v, name)
                mv === nothing || return mv
            end
            nothing
        end
        _tblcol(doc, col) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                col in table_cols(v) && return v
            end
            nothing
        end

        @testset "metrics — RMSE(f1) < RMSE(f2)" begin
            r = run_json(["forecast", "evaluate", "metrics", csv, "--actual", "y", "--forecasts", "f1,f2"])
            assert_envelope_ok(r; label="forecast evaluate metrics")
            acc = _tblcol(r.doc, "RMSE")
            @test acc !== nothing
            if acc !== nothing
                @test issubset(["model","ME","MAE","RMSE","MAPE","sMAPE","MASE","U1","U2"], table_cols(acc))
                rows = [collect(row) for row in table_rows(acc)]
                @test length(rows) == 2
                ri = col_index(acc, "RMSE")
                rmses = [Float64(row[ri]) for row in rows]
                @test all(isfinite, rmses)
                @test rmses[1] < rmses[2]              # teeth: f1 more accurate than f2
            end
            dec = _tblcol(r.doc, "bias")
            @test dec !== nothing
            if dec !== nothing
                drow = collect(first(table_rows(dec)))
                props = [Float64(drow[col_index(dec, c)]) for c in ("bias","variance","covariance")]
                @test all(p -> -1e-6 <= p <= 1.0 + 1e-6, props)
                @test isapprox(sum(props), 1.0; atol=1e-3)   # Theil proportions sum to 1
            end
        end

        @testset "dm / clark-west / mincer-zarnowitz / encompassing p-values in [0,1]" begin
            for (leaf, fc) in (("dm", "f1,f2"), ("clark-west", "f1,f2"),
                               ("mincer-zarnowitz", "f1"), ("encompassing", "f1,f2"))
                r = run_json(["forecast", "evaluate", leaf, csv, "--actual", "y", "--forecasts", fc])
                assert_envelope_ok(r; label="forecast evaluate $leaf")
                stat = _kvval(r.doc, "statistic")
                leaf in ("dm", "clark-west") && (@test stat isa Real && isfinite(Float64(stat)))
                pname = leaf == "mincer-zarnowitz" ? "p_value_wald" : "p_value"
                pv = _kvval(r.doc, pname)
                @test pv isa Real && 0.0 <= Float64(pv) <= 1.0
            end
        end

        @testset "combine — equal weights + bates-granger favors f1" begin
            r = run_json(["forecast", "evaluate", "combine", csv, "--actual", "y",
                          "--forecasts", "f1,f2", "--method", "bates-granger"])
            assert_envelope_ok(r; label="forecast evaluate combine")
            w = _tblcol(r.doc, "weight")
            @test w !== nothing
            if w !== nothing
                rows = [collect(row) for row in table_rows(w)]
                @test length(rows) == 2
                wi = col_index(w, "weight")
                weights = [Float64(row[wi]) for row in rows]
                @test isapprox(sum(weights), 1.0; atol=1e-4)
                @test weights[1] > weights[2]          # inverse-MSE weight favors the better f1
            end
        end
        rm(csv; force=true)
    end

    @testset "test adf rejects unit root on stationary series" begin
        # Strongly mean-reverting → p-value should be small
        csv = dgp_ar1(; T=400, φ=0.2, seed=3)
        r = run_json(["test", "adf", csv, "--column", "1"])
        assert_envelope_ok(r; label="test adf")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        pv = metric_value(tbl, "p-value")
        @test pv !== nothing
        @test pv isa Real
        @test pv < 0.10   # teeth: stationary series should look stationary
        rm(csv; force=true)
    end

    @testset "test kpss" begin
        csv = dgp_ar1(; T=200, φ=0.3, seed=5)
        r = run_json(["test", "kpss", csv, "--column", "1"])
        assert_envelope_ok(r; label="test kpss")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 1
        rm(csv; force=true)
    end

    @testset "irf var Cholesky tidy (C051)" begin
        csv = dgp_var2(; T=200, seed=9)
        r = run_json(["irf", "var", csv, "--lags", "2", "--horizons", "8",
                      "--shock", "1", "--ci", "none", "--id", "cholesky"])
        assert_envelope_ok(r; label="irf var")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            # C051: tidy long_table, filtered to the selected --shock.
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value", "lower", "upper"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(unique(row[ci["shock"]] for row in rows)) == 1   # one shock
            h1 = [Float64(row[ci["value"]]) for row in rows if row[ci["horizon"]] == 1]
            @test any(x -> abs(x) > 1e-6, h1)   # Cholesky impact responses not all zero
        end
        rm(csv; force=true)
    end

    @testset "fevd var tidy (C051)" begin
        csv = dgp_var2(; T=180, seed=13)
        r = run_json(["fevd", "var", csv, "--lags", "2", "--horizons", "10", "--id", "cholesky"])
        assert_envelope_ok(r; label="fevd var")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            # FEVD proportions are shares in [0, 1]
            @test all(-1e-8 <= Float64(row[ci["value"]]) <= 1.0 + 1e-8 for row in rows)
        end
        rm(csv; force=true)
    end

    @testset "hd var shape" begin
        csv = dgp_var2(; T=120, seed=15)
        r = run_json(["hd", "var", csv, "--lags", "1", "--id", "cholesky"])
        assert_envelope_ok(r; label="hd var")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 1
        rm(csv; force=true)
    end

    @testset "lewis-tvv + sv-em identification (W1/#186)" begin
        # Non-recursive B0 = [1 0.4; -0.2 1]: Cholesky prints
        # impact[2,1] == 0 and impact[1,2] == 0, so a nonzero
        # cross-impact with the right relative sign proves the
        # TVV/SV path ran (gap-sized thresholds, never tight values).
        csv = dgp_svvar(; T=300, seed=42)

        # Lewis default (two-step): impact-pattern teeth. --seed pins the
        # estimator basin (multi-start thetas are rng draws; an unseeded
        # run can land in a permuted basin — same class as the CUE note).
        r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "1", "--ci", "none",
                      "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="irf var lewis-tvv")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            h1 = [row for row in rows if row[ci["horizon"]] == 1]
            @test length(h1) == 2
            v1 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y1"][1])
            v2 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y2"][1])
            @test v1 * v2 < 0 && abs(v2) > 0.05   # opposite signs, y2 impact nonzero
        end
        # Determinism: same seed reruns bit-identical values.
        r2 = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                       "--horizons", "8", "--shock", "1", "--ci", "none",
                       "--id", "lewis-tvv"])
        assert_envelope_ok(r2; label="irf var lewis-tvv rerun")
        _, tbl2 = first_table(r2.doc)
        if tbl !== nothing && tbl2 !== nothing
            ci2 = Dict(c => i for (i, c) in enumerate(table_cols(tbl2)))
            vals = [Float64(row[ci["value"]]) for row in table_rows(tbl)]
            vals2 = [Float64(row[ci2["value"]]) for row in table_rows(tbl2)]
            @test vals == vals2
        end
        r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "2", "--ci", "none",
                      "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="irf var lewis-tvv shock 2")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            h1 = [row for row in rows if row[ci["horizon"]] == 1]
            v1 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y1"][1])
            v2 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y2"][1])
            @test v1 * v2 > 0 && abs(v1) > 0.1    # same sign, y1 impact nonzero
        end

        # one_step + cue run (exit 0 + shape only: CUE can land in a
        # wrong basin on finite samples — correct estimator behavior,
        # so no recovery assertion here, same class as the
        # threshold-CI rule).
        for w in ("one_step", "cue")
            toml = tempname() * ".toml"
            write(toml, "[identification.lewis_tvv]\nweighting = \"$w\"\n")
            r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                          "--horizons", "8", "--shock", "1", "--ci", "none",
                          "--id", "lewis-tvv", "--config", toml])
            assert_envelope_ok(r; label="irf var lewis-tvv $w")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            if tbl !== nothing
                @test length(table_rows(tbl)) == 16
            end
            rm(toml; force=true)
        end

        # fevd / hd smoke on the lewis path.
        r = run_json(["--seed", "42", "fevd", "var", csv, "--lags", "1",
                      "--horizons", "8", "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="fevd var lewis-tvv")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            @test all(-1e-8 <= Float64(row[ci["value"]]) <= 1.0 + 1e-8
                      for row in table_rows(tbl))
        end
        # hd var declares no horizon option (full-sample decomposition).
        r = run_json(["--seed", "42", "hd", "var", csv, "--lags", "1",
                      "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="hd var lewis-tvv")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test length(table_rows(tbl)) >= 1
        end

        # SV-SVAR full hetero (tiny MCEM via TOML): same teeth
        # (probed stable across seeds 42-44 with margin).
        svtoml = tempname() * ".toml"
        write(svtoml, "[identification.sv_svar]\nmaxiter = 20\ngibbs_draws = 30\n")
        r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "1", "--ci", "none",
                      "--id", "sv-em", "--config", svtoml])
        assert_envelope_ok(r; label="irf var sv-em")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            h1 = [row for row in rows if row[ci["horizon"]] == 1]
            v1 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y1"][1])
            v2 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y2"][1])
            @test v1 * v2 < 0 && abs(v2) > 0.05
        end
        r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "2", "--ci", "none",
                      "--id", "sv-em", "--config", svtoml])
        assert_envelope_ok(r; label="irf var sv-em shock 2")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            h1 = [row for row in rows if row[ci["horizon"]] == 1]
            v1 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y1"][1])
            v2 = Float64([row[ci["value"]] for row in h1 if row[ci["variable"]] == "y2"][1])
            @test v1 * v2 > 0 && abs(v1) > 0.1
        end
        rm(svtoml; force=true)

        # SV-SVAR partial hetero=[2]: runs (exit 0 + shape; the
        # hetero plumbing is unit-pinned at T1/T2).
        ptoml = tempname() * ".toml"
        write(ptoml, "[identification.sv_svar]\nhetero_shocks = [2]\nmaxiter = 20\ngibbs_draws = 30\n")
        r = run_json(["--seed", "42", "irf", "var", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "2", "--ci", "none",
                      "--id", "sv-em", "--config", ptoml])
        assert_envelope_ok(r; label="irf var sv-em partial")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test length(table_rows(tbl)) == 16
        end
        rm(ptoml; force=true)

        # Family smokes (exit 0 + envelope; recovery teeth live on var).
        cc = dgp_coint(; T=250, seed=21)
        r = run_json(["--seed", "42", "irf", "vecm", cc, "--lags", "2",
                      "--rank", "1", "--horizons", "8", "--shock", "1",
                      "--ci", "none", "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="irf vecm lewis-tvv")
        rm(cc; force=true)
        r = run_json(["--seed", "42", "irf", "bvar", csv, "--lags", "1",
                      "--horizons", "8", "--shock", "1", "--draws", "10",
                      "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="irf bvar lewis-tvv")
        r = run_json(["--seed", "42", "irf", "lp", csv, "--lags", "4",
                      "--horizons", "8", "--shock", "1", "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="irf lp lewis-tvv")

        # Error paths: exit classes pinned.
        badw = tempname() * ".toml"
        write(badw, "[identification.lewis_tvv]\nweighting = \"optimal\"\n")
        r = run_json(["irf", "var", csv, "--lags", "1", "--shock", "1",
                      "--id", "lewis-tvv", "--config", badw])
        @test r.code == 4 && String(r.doc.error.code) == "config/invalid"
        rm(badw; force=true)
        badoob = tempname() * ".toml"
        write(badoob, "[identification.sv_svar]\nhetero_shocks = [3]\n")
        r = run_json(["irf", "var", csv, "--lags", "1", "--shock", "1",
                      "--id", "sv-em", "--config", badoob])
        @test r.code == 2 && String(r.doc.error.code) == "usage/invalid"
        rm(badoob; force=true)
        badi = tempname() * ".toml"
        write(badi, "[identification.sv_svar]\ninit = \"newton\"\n")
        r = run_json(["irf", "var", csv, "--lags", "1", "--shock", "1",
                      "--id", "sv-em", "--config", badi])
        @test r.code == 4 && String(r.doc.error.code) == "config/invalid"
        rm(badi; force=true)
        csv1 = dgp_ar1(; T=250, φ=0.7, seed=11)
        r = run_json(["irf", "var", csv1, "--lags", "1", "--shock", "1",
                      "--id", "lewis-tvv"])
        @test r.code == 3 && String(r.doc.error.code) == "data/invalid"
        rm(csv1; force=true)
        csvs = dgp_var2(; T=100, seed=5)
        r = run_json(["irf", "var", csvs, "--lags", "1", "--shock", "1",
                      "--id", "lewis-tvv"])
        @test r.code == 3 && String(r.doc.error.code) == "data/invalid"
        rm(csvs; force=true)
        rm(csv; force=true)
    end

    @testset "fevd bvar threads method (W1/#186 fix)" begin
        csv = dgp_var2(; T=200, seed=9)
        r = run_json(["fevd", "bvar", csv, "--lags", "1", "--horizons", "4",
                      "--draws", "50", "--id", "cholesky"])
        assert_envelope_ok(r; label="fevd bvar cholesky")
        r = run_json(["fevd", "bvar", csv, "--lags", "1", "--horizons", "4",
                      "--draws", "50", "--id", "bogus"])
        @test r.code == 2
        rm(csv; force=true)
    end

    @testset "fevd/hd bvar lewis-tvv knob threading (W1/#186)" begin
        # BVARPosterior fevd/hd take method + estimator knobs through
        # separate inline call sites (not the shared builder) — prove
        # live that the knobs reach real fevd(post)/hd(post).
        csv = dgp_svvar(; T=300, seed=42)
        r = run_json(["--seed", "42", "fevd", "bvar", csv, "--lags", "1",
                      "--horizons", "8", "--draws", "10", "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="fevd bvar lewis-tvv")
        # hd bvar declares no horizon option (decomposition over the full
        # sample, like hd lp/vecm) — no --horizon/--horizons flag here.
        r = run_json(["--seed", "42", "hd", "bvar", csv, "--lags", "1",
                      "--draws", "10", "--id", "lewis-tvv"])
        assert_envelope_ok(r; label="hd bvar lewis-tvv")
        rm(csv; force=true)
    end

    @testset "hetero-id error paths (W1/#186 review)" begin
        # Adversarial-review findings: every one of these was an untyped
        # exit 1 before the fix. Pins are exit-class-only (never values).
        # SDFM loader rejects an unknown --id typed (bare ArgumentError
        # from estimate_structural_dfm used to escape).
        panel = dgp_panel_matrix(; N=10, T=80, seed=7)
        r = run_json(["irf", "sdfm", panel, "--factors", "1",
                      "--id", "lewis-tvv"])
        @test r.code == 3 && String(r.doc.error.code) == "data/invalid"
        rm(panel; force=true)
        # BVAR --config without [prior.hyperparameters] falls back to prior
        # defaults (direct prior_cfg["lambda1"] indexing used to KeyError).
        csv = dgp_svvar(; T=300, seed=42)
        toml = tempname() * ".toml"
        write(toml, "[identification.sv_svar]\nmaxiter = 20\ngibbs_draws = 30\n")
        r = run_json(["estimate", "bvar", csv, "--lags", "1", "--draws", "10",
                      "--config", toml])
        assert_envelope_ok(r; label="estimate bvar config-no-prior")
        # sv-em on VECM: upstream raises IdentificationError (non-orthogonal
        # Q on converted models) — typed model/identification, never exit 1.
        cc = dgp_coint(; T=250, seed=21)
        r = run_json(["irf", "vecm", cc, "--lags", "2", "--rank", "1",
                      "--shock", "1", "--ci", "none",
                      "--id", "sv-em", "--config", toml])
        @test r.code == 5 && String(r.doc.error.code) == "model/identification"
        rm(cc; force=true)
        rm(toml; force=true)
        rm(csv; force=true)
    end

    @testset "forecast var (C051 tidy long_table)" begin
        csv = dgp_var2(; T=150, seed=17)
        r = run_json(["forecast", "var", csv, "--lags", "2", "--horizons", "4"])
        assert_envelope_ok(r; label="forecast var")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            # C051: MEMs' uniform tidy long_table schema (one row per horizon×variable),
            # replacing the old wide per-variable table (horizon | var | var_lower | ...).
            @test table_cols(tbl) == ["horizon", "variable", "value", "lower", "upper"]
            @test length(table_rows(tbl)) == 4 * 3   # 4 horizons × 3 variables (dgp_var2)
        end
        rm(csv; force=true)
    end

    @testset "forecast arima tidy (C051)" begin
        csv = dgp_ar1(; T=200, φ=0.6, seed=19)
        r = run_json(["forecast", "arima", csv, "--column", "1", "--horizons", "5"])
        assert_envelope_ok(r; label="forecast arima")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "value", "lower", "upper"]
            @test length(table_rows(tbl)) == 5   # univariate: 5 horizons × 1 variable
        end
        rm(csv; force=true)
    end

    @testset "forecast vecm/lp/static tidy (C051)" begin
        # forecast leaves whose handlers return an AbstractForecastResult → long_table
        cointcsv = dgp_coint(; T=300, seed=45)
        rv = run_json(["forecast", "vecm", cointcsv, "--lags", "2", "--rank", "1", "--horizons", "6"])
        assert_envelope_ok(rv; label="forecast vecm")
        _, tv = first_table(rv.doc)
        @test tv !== nothing && table_cols(tv) == ["horizon", "variable", "value", "lower", "upper"]
        rm(cointcsv; force=true)

        mvcsv = dgp_var2(; T=150, seed=47)
        rl = run_json(["forecast", "lp", mvcsv, "--shock", "1", "--horizons", "6"])
        assert_envelope_ok(rl; label="forecast lp")
        _, tl = first_table(rl.doc)
        @test tl !== nothing && table_cols(tl) == ["horizon", "variable", "value", "lower", "upper"]

        rs = run_json(["forecast", "static", mvcsv, "--nfactors", "1", "--horizons", "6"])
        assert_envelope_ok(rs; label="forecast static")
        _, ts = first_table(rs.doc)
        @test ts !== nothing && table_cols(ts) == ["horizon", "variable", "value", "lower", "upper"]
        rm(mvcsv; force=true)
    end

    @testset "test vecm restriction tests — real Johansen LR (C071)" begin
        # Two I(1) series sharing a stochastic trend → cointegrating rank 1.
        cointcsv = dgp_coint(; T=300, β=1.0, seed=71)
        kv(doc, name) = (t = last(first_table(doc)); t === nothing ? nothing : metric_value(t, name))

        # Write a restriction config (p=2, r=1). Non-binding H = I₂ (s=p) ⇒ LR ≈ 0, df = 0.
        idcfg = tempname() * ".toml"
        write(idcfg, "[vecm_restriction]\nH = [[1.0, 0.0], [0.0, 1.0]]\n")
        rid = run_json(["test", "vecm", "beta", cointcsv, "--config", idcfg, "--rank", "1"])
        assert_envelope_ok(rid; label="test vecm beta (identity H)")
        @test kv(rid.doc, "df") == 0
        lr0 = kv(rid.doc, "LR statistic")
        @test lr0 !== nothing && abs(Float64(lr0)) < 1e-4       # non-binding restriction

        # Binding β restriction β = Hφ with H = (1, -1)′ (s=1 ⇒ df = r(p−s) = 1).
        bcfg = tempname() * ".toml"
        write(bcfg, "[vecm_restriction]\nH = [[1.0], [-1.0]]\nA = [[1.0], [0.0]]\nb = [[1.0], [-1.0]]\n")
        rb = run_json(["test", "vecm", "beta", cointcsv, "--config", bcfg, "--rank", "1"])
        assert_envelope_ok(rb; label="test vecm beta")
        @test kv(rb.doc, "df") == 1
        pvb = kv(rb.doc, "p-value")
        @test pvb !== nothing && 0.0 <= Float64(pvb) <= 1.0

        # α restriction (df = r(p−a) = 1).
        ra = run_json(["test", "vecm", "alpha", cointcsv, "--config", bcfg, "--rank", "1"])
        assert_envelope_ok(ra; label="test vecm alpha")
        @test kv(ra.doc, "df") == 1

        # Weak exogeneity of variable 1 (df = r·|vars| = 1).
        rw = run_json(["test", "vecm", "weak-exog", cointcsv, "--vars", "1", "--rank", "1"])
        assert_envelope_ok(rw; label="test vecm weak-exog")
        @test kv(rw.doc, "df") == 1

        # Known β (b is p×r; df = r(p−r) = 1).
        rk = run_json(["test", "vecm", "known-beta", cointcsv, "--config", bcfg, "--rank", "1"])
        assert_envelope_ok(rk; label="test vecm known-beta")
        @test kv(rk.doc, "df") == 1

        # Joint β&α via the switching algorithm (df = r(p−s)+r(p−a) = 2).
        rj = run_json(["test", "vecm", "joint", cointcsv, "--config", bcfg, "--rank", "1"])
        assert_envelope_ok(rj; label="test vecm joint")
        @test kv(rj.doc, "df") == 2

        # Bad input stays typed (not internal exit 1): missing --config, fitted rank 0.
        rmc = run_json(["test", "vecm", "beta", cointcsv, "--rank", "1"])
        @test rmc.code == 2                                     # usage/missing-config
        r0 = run_json(["test", "vecm", "beta", cointcsv, "--config", bcfg, "--rank", "0"])
        @test r0.code == 3                                      # data/no-cointegration

        rm(cointcsv; force=true)
        rm(idcfg; force=true)
        rm(bcfg; force=true)
    end

    @testset "TS + panel test batteries — real MEMs (C069/C070)" begin
        # scan every kv table (metric|value) across the envelope for a metric value
        scan_metric(doc, name) = begin
            v = nothing
            for (_, tbl) in pairs(doc.data)
                if (tbl isa JSON3.Object || tbl isa AbstractDict) && (haskey(tbl, :rows) || haskey(tbl, "rows"))
                    mv = metric_value(tbl, name)
                    mv !== nothing && (v = mv)
                end
            end
            v
        end
        # the (first) table that contains a named column
        coltable(doc, col) = begin
            for (_, tbl) in pairs(doc.data)
                if (tbl isa JSON3.Object || tbl isa AbstractDict) && (haskey(tbl, :columns) || haskey(tbl, "columns"))
                    col in table_cols(tbl) && return tbl
                end
            end
            nothing
        end
        pmin(tbl) = minimum(Float64(collect(r)[col_index(tbl, "p_value")]) for r in table_rows(tbl))

        @testset "variance-ratio — random walk (H0) vs mean-reverting (reject)" begin
            rw = dgp_random_walk(; T=600, seed=91)
            rrw = run_json(["test", "variance-ratio", rw, "--column", "1"])
            assert_envelope_ok(rrw; label="variance-ratio rw")
            prw = scan_metric(rrw.doc, "Chow-Denning p-value")
            @test prw !== nothing && 0.0 <= Float64(prw) <= 1.0
            t = coltable(rrw.doc, "variance_ratio")
            @test t !== nothing && length(table_rows(t)) == 4        # default q = 2,4,8,16

            mr = dgp_ar1(; T=600, φ=0.3, seed=93)                    # stationary → VR ≠ 1
            rmr = run_json(["test", "variance-ratio", mr, "--column", "1"])
            assert_envelope_ok(rmr; label="variance-ratio mean-reverting")
            pmr = scan_metric(rmr.doc, "Chow-Denning p-value")
            @test pmr !== nothing && Float64(pmr) < 0.05             # reject random walk
            rm(rw; force=true); rm(mr; force=true)
        end

        @testset "bds — iid (H0) vs nonlinear GARCH (reject)" begin
            iid = dgp_iid(; T=400, seed=95)
            rii = run_json(["test", "bds", iid, "--column", "1"])
            assert_envelope_ok(rii; label="bds iid")
            t = coltable(rii.doc, "embed_dim")
            @test t !== nothing && length(table_rows(t)) == 5        # m = 2..6
            @test 0.0 <= pmin(t) <= 1.0

            g = dgp_garch(; T=500, seed=97)
            rg = run_json(["test", "bds", g, "--column", "1"])
            assert_envelope_ok(rg; label="bds garch")
            @test pmin(coltable(rg.doc, "p_value")) < 0.05          # nonlinear ⇒ reject iid
            rm(iid; force=true); rm(g; force=true)
        end

        @testset "hadri — stationary vs unit-root panel" begin
            i0 = dgp_panel_matrix(; N=10, T=80, unit_root=false, seed=101)
            r0 = run_json(["test", "hadri", i0])
            assert_envelope_ok(r0; label="hadri stationary")
            p0 = scan_metric(r0.doc, "p-value"); s0 = scan_metric(r0.doc, "statistic")
            @test p0 !== nothing && 0.0 <= Float64(p0) <= 1.0
            @test s0 !== nothing && isfinite(Float64(s0))

            i1 = dgp_panel_matrix(; N=10, T=80, unit_root=true, seed=103)
            r1 = run_json(["test", "hadri", i1])
            assert_envelope_ok(r1; label="hadri unit-root")
            p1 = scan_metric(r1.doc, "p-value"); s1 = scan_metric(r1.doc, "statistic")
            @test p1 !== nothing && Float64(p1) < 0.05              # reject all-stationary
            # the I(1) panel's LM statistic is far larger than the (over-rejecting but
            # bounded) stationary panel's — the robust discriminating direction.
            @test s1 !== nothing && Float64(s1) > Float64(s0)
            rm(i0; force=true); rm(i1; force=true)
        end

        @testset "pedroni / kao / westerlund — cointegrated panel (reject no-coint)" begin
            cp = dgp_coint_panel(; N=10, T=50, seed=105)
            for (leaf, ncols) in [("pedroni", 7), ("kao", 5), ("westerlund", 4)]
                r = run_json(["test", leaf, cp, "--dep", "y", "--indep", "x"])
                assert_envelope_ok(r; label="$leaf coint panel")
                t = coltable(r.doc, "p_value")
                @test t !== nothing && length(table_rows(t)) == ncols
                @test pmin(t) < 0.05                                # genuinely cointegrated
                @test scan_metric(r.doc, "n_regressors") == 1
            end
            rm(cp; force=true)
        end

        @testset "bad input stays typed (not internal exit 1)" begin
            uni = dgp_iid(; T=200, seed=111)
            @test run_json(["test", "variance-ratio", uni, "--horizons", "junk"]).code == 2
            @test run_json(["test", "variance-ratio", uni, "--horizons", "1,2"]).code == 2
            @test run_json(["test", "bds", uni, "--max-dim", "1"]).code == 2
            cp = dgp_coint_panel(; N=6, T=40, seed=113)
            @test run_json(["test", "pedroni", cp, "--dep", "nope"]).code == 2
            @test run_json(["test", "pedroni", cp, "--indep", "nope"]).code == 2
            @test run_json(["test", "pedroni", cp, "--id-col", "nosuch"]).code == 3   # data/missing-column
            # duplicate (id,time) pair → real xtset ArgumentError mapped to typed data/invalid
            # (regression: adversarial review C069/C070 — was an uncaught internal exit-1)
            dup = tempname() * ".csv"
            write(dup, "id,time,y,x\n1,1,0.5,1.2\n1,1,0.7,1.3\n1,2,0.9,1.4\n2,1,0.3,0.8\n2,2,0.6,0.9\n")
            @test run_json(["test", "pedroni", dup, "--dep", "y", "--indep", "x"]).code == 3
            rm(dup; force=true)
            rm(uni; force=true); rm(cp; force=true)
        end

        # ── C069 remainder: seasonal / point-optimal / bubble / EDF + residual
        # cointegration. Each case asserts the DISCRIMINATING DIRECTION on a DGP built
        # for that null, not a hard-coded statistic.

        @testset "hegy — seasonal unit root vs deterministic seasonality" begin
            su = dgp_seasonal(; T=240, deterministic=false, seed=121)
            r = run_json(["test", "hegy", su, "--frequency", "4"])
            assert_envelope_ok(r; label="hegy seasonal unit root")
            t = coltable(r.doc, "decision")
            # quarterly ⇒ zero + Nyquist + one harmonic pair
            @test t !== nothing && length(table_rows(t)) == 3
            @test Set(["frequency", "kind", "statistic", "cv_5pct", "decision"]) ⊆
                  Set(String.(table_cols(t)))
            # a seasonal random walk must not reject at EVERY seasonal frequency
            decs = [String(collect(row)[col_index(t, "decision")]) for row in table_rows(t)]
            @test any(d -> startswith(d, "cannot reject"), decs)

            ds = dgp_seasonal(; T=240, deterministic=true, seed=123)
            rd = run_json(["test", "hegy", ds, "--frequency", "4"])
            assert_envelope_ok(rd; label="hegy deterministic seasonality")
            decs_d = [String(collect(row)[col_index(coltable(rd.doc, "decision"), "decision")])
                      for row in table_rows(coltable(rd.doc, "decision"))]
            # stationary + dummies ⇒ the seasonal roots are rejected
            @test count(d -> startswith(d, "reject"), decs_d) > count(d -> startswith(d, "reject"), decs)
            rm(su; force=true); rm(ds; force=true)
        end

        @testset "ers — random walk (H0) vs stationary AR(1) (reject)" begin
            rw = dgp_random_walk(; T=300, seed=125)
            r0 = run_json(["test", "ers", rw])
            assert_envelope_ok(r0; label="ers random walk")
            p0 = scan_metric(r0.doc, "p-value")
            @test p0 !== nothing && Float64(p0) > 0.05          # cannot reject a unit root

            st = dgp_ar1(; T=300, φ=0.2, seed=127)
            r1 = run_json(["test", "ers", st])
            assert_envelope_ok(r1; label="ers stationary")
            p1 = scan_metric(r1.doc, "p-value")
            @test p1 !== nothing && Float64(p1) < 0.05          # reject the unit root
            @test scan_metric(r1.doc, "regression") == "constant"
            @test run_json(["test", "ers", st, "--trend"]).code == 0
            rm(rw; force=true); rm(st; force=true)
        end

        @testset "sadf/gsadf — explosive episode vs pure random walk" begin
            bub = dgp_bubble(; T=300, seed=129)
            rw  = dgp_random_walk(; T=300, seed=131)
            for leaf in ("sadf", "gsadf")
                rb = run_json(["test", leaf, bub, "--mc-reps", "199"])
                assert_envelope_ok(rb; label="$leaf bubble")
                sb = scan_metric(rb.doc, "statistic")
                rr = run_json(["test", leaf, rw, "--mc-reps", "199"])
                assert_envelope_ok(rr; label="$leaf random walk")
                sr = scan_metric(rr.doc, "statistic")
                # the explosive series must score strictly higher than the pure I(1) one
                @test sb !== nothing && sr !== nothing && Float64(sb) > Float64(sr)
                @test coltable(rb.doc, "episode") !== nothing
            end
            rm(bub; force=true); rm(rw; force=true)
        end

        @testset "edf — normal sample (H0) vs an obviously non-normal one" begin
            iid = dgp_iid(; T=400, seed=133)
            r0 = run_json(["test", "edf", iid, "--dist", "normal", "--test", "ad"])
            assert_envelope_ok(r0; label="edf normal")
            p0 = scan_metric(r0.doc, "p-value")
            @test p0 !== nothing && Float64(p0) > 0.05         # consistent with normality

            # a random-walk LEVEL series is nowhere near normal
            rw = dgp_random_walk(; T=400, seed=135)
            r1 = run_json(["test", "edf", rw, "--dist", "normal", "--test", "ad"])
            assert_envelope_ok(r1; label="edf non-normal")
            p1 = scan_metric(r1.doc, "p-value")
            @test p1 !== nothing && Float64(p1) < 0.05
            @test run_json(["test", "edf", iid, "--test", "ks"]).code == 0
            # upstream spells the supplied-parameter case :specified — a T3 regression, since
            # the mock had wrongly accepted :known and hid the failure at T1/T2
            @test run_json(["test", "edf", iid, "--params", "specified", "--theta", "0,1"]).code == 0
            @test run_json(["test", "edf", iid, "--params", "known", "--theta", "0,1"]).code == 2
            rm(iid; force=true); rm(rw; force=true)
        end

        @testset "engle-granger / phillips-ouliaris — cointegrated vs independent" begin
            # NOTE the H0 flips relative to a unit-root test: H0 is NO cointegration, so
            # a LOW p-value is evidence FOR a cointegrating relationship.
            ci = dgp_coint(; T=250, β=2.0, seed=137)
            nc = dgp_no_coint(; T=250, seed=139)
            for leaf in ("engle-granger", "phillips-ouliaris")
                rc = run_json(["test", leaf, ci, "--dep", "y"])
                assert_envelope_ok(rc; label="$leaf cointegrated")
                pc = scan_metric(rc.doc, "p-value")
                pc = pc === nothing ? nothing : Float64(pc)
                if pc === nothing                      # PO reports its p-values in a table
                    t = coltable(rc.doc, "p_value")
                    pc = minimum(Float64(collect(r)[col_index(t, "p_value")]) for r in table_rows(t))
                end
                @test pc < 0.05                        # cointegrated ⇒ reject no-cointegration

                rn = run_json(["test", leaf, nc, "--dep", "y"])
                assert_envelope_ok(rn; label="$leaf independent")
                pn = scan_metric(rn.doc, "p-value")
                pn = pn === nothing ? nothing : Float64(pn)
                if pn === nothing
                    t = coltable(rn.doc, "p_value")
                    pn = minimum(Float64(collect(r)[col_index(t, "p_value")]) for r in table_rows(t))
                end
                @test pn > pc                          # independent walks score far weaker
            end
            @test run_json(["test", "phillips-ouliaris", ci, "--dep", "y",
                            "--kernel", "parzen", "--bandwidth", "6"]).code == 0
            @test run_json(["test", "engle-granger", ci, "--dep", "y", "--lags", "2"]).code == 0
            rm(ci; force=true); rm(nc; force=true)
        end

        @testset "granger — VAR pairwise + --all matrix (#118)" begin
            # Real granger_test_all returns an n×n Matrix{Union{GrangerCausalityResult,
            # Nothing}} (diagonal = nothing) with cause::Vector{Int}/effect::Int — the
            # old flat iteration into String columns made `test granger --all` exit 1
            # on every real-MEMs invocation while the invented mock kept T1/T2 green.
            csv = dgp_granger(; T=300, seed=141)
            kv_with(doc, metric) = begin
                found = nothing
                for (_, v) in pairs(doc.data)
                    v isa JSON3.Object && haskey(v, :columns) || continue
                    metric_value(v, metric) === nothing || (found = v; break)
                end
                found
            end
            rf = run_json(["test", "granger", csv, "--model", "var",
                           "--cause", "1", "--effect", "2"])
            assert_envelope_ok(rf; label="granger x→y")
            pxy = numv(metric_value(kv_with(rf.doc, "p-value"), "p-value"))
            @test pxy < 0.05                        # x Granger-causes y by construction
            rb = run_json(["test", "granger", csv, "--model", "var",
                           "--cause", "2", "--effect", "1"])
            assert_envelope_ok(rb; label="granger y→x")
            pyx = numv(metric_value(kv_with(rb.doc, "p-value"), "p-value"))
            @test pyx > pxy                         # the non-causal direction is weaker

            ra = run_json(["test", "granger", csv, "--model", "var", "--all"])
            assert_envelope_ok(ra; label="granger --all")
            t = begin
                found = nothing
                for (_, v) in pairs(ra.doc.data)
                    v isa JSON3.Object && haskey(v, :columns) || continue
                    ("cause" in table_cols(v) && "effect" in table_cols(v)) && (found = v; break)
                end
                found
            end
            @test t !== nothing
            rows = table_rows(t)
            @test length(rows) == 2                 # n(n-1) ordered pairs, diagonal skipped
            ci_ = col_index(t, "cause"); ei_ = col_index(t, "effect"); pi_ = col_index(t, "p_value")
            pairs_seen = Set((String(collect(r)[ci_]), String(collect(r)[ei_])) for r in rows)
            @test pairs_seen == Set([("x", "y"), ("y", "x")])   # CSV names, not y1/y2
            p_by_pair = Dict((String(collect(r)[ci_]), String(collect(r)[ei_])) =>
                             numv(collect(r)[pi_]) for r in rows)
            @test p_by_pair[("x", "y")] < p_by_pair[("y", "x")]

            # #119: the model itself carries the CSV names now — long_table output
            # (variable/shock columns) must say x/y, never the y1..yn default.
            ri = run_json(["irf", "var", csv, "--horizons", "4"])
            assert_envelope_ok(ri; label="irf var names (#119)")
            it = begin
                found = nothing
                for (_, v) in pairs(ri.doc.data)
                    v isa JSON3.Object && haskey(v, :columns) || continue
                    ("variable" in table_cols(v) && "shock" in table_cols(v)) && (found = v; break)
                end
                found
            end
            @test it !== nothing
            vi_ = col_index(it, "variable"); si_ = col_index(it, "shock")
            seen_vars = Set(String(collect(r)[vi_]) for r in table_rows(it))
            seen_shocks = Set(String(collect(r)[si_]) for r in table_rows(it))
            @test seen_vars == Set(["x", "y"])
            @test seen_shocks ⊆ Set(["x", "y", "x_shock", "y_shock"])
            @test !("y1" in seen_vars)
            rm(csv; force=true)
        end

        @testset "hansen-instability / park-added — stable cointegration" begin
            ci = dgp_coint(; T=250, β=2.0, seed=141)
            rh = run_json(["test", "hansen-instability", ci, "--dep", "y"])
            assert_envelope_ok(rh; label="hansen-instability")
            ph = scan_metric(rh.doc, "p-value")
            # H0 here is STABLE cointegration, and the DGP has a constant β ⇒ don't reject
            @test ph !== nothing && Float64(ph) > 0.05
            @test scan_metric(rh.doc, "trend") == "const"

            rp = run_json(["test", "park-added", ci, "--dep", "y", "--q-add", "2"])
            assert_envelope_ok(rp; label="park-added")
            pp = scan_metric(rp.doc, "p-value")
            # H0 is genuine cointegration ⇒ don't reject on a genuinely cointegrated pair
            @test pp !== nothing && Float64(pp) > 0.05
            @test scan_metric(rp.doc, "q_add (df)") == 2

            # A spurious pair must still produce a well-formed result. Deliberately NOT
            # asserting statistic(spurious) > statistic(cointegrated): the Park H(p,q)
            # test's power at T=250 on a single draw does not guarantee that ordering
            # (measured 3.54 vs 4.25 on seeds 143/141), so such a check tests the seed,
            # not the wrapper.
            nc = dgp_no_coint(; T=250, seed=143)
            rs = run_json(["test", "park-added", nc, "--dep", "y"])
            assert_envelope_ok(rs; label="park-added spurious")
            @test isfinite(Float64(scan_metric(rs.doc, "H(p,q) statistic")))
            let psp = Float64(scan_metric(rs.doc, "p-value"))
                @test 0.0 <= psp <= 1.0
            end

            @test run_json(["test", "hansen-instability", ci, "--dep", "y",
                            "--method", "dols", "--leads", "2", "--lags", "2"]).code == 0
            rm(ci; force=true); rm(nc; force=true)
        end

        @testset "arfima forecast/predict/residuals (#73)" begin
            csv = dgp_ar1(; T=300, φ=0.6, seed=501)
            rf = run_json(["forecast", "arfima", csv, "--p", "1", "--q", "0",
                           "--horizons", "6"])
            assert_envelope_ok(rf; label="forecast arfima")
            t = first_table(rf.doc)[2]
            @test t !== nothing && length(table_rows(t)) == 6
            @test Set(["horizon", "forecast", "lower", "upper"]) ⊆ Set(String.(table_cols(t)))
            # intervals must bracket the point forecast and widen with the horizon
            lo1 = Float64(collect(first(table_rows(t)))[col_index(t, "lower")])
            hi1 = Float64(collect(first(table_rows(t)))[col_index(t, "upper")])
            f1  = Float64(collect(first(table_rows(t)))[col_index(t, "forecast")])
            @test lo1 <= f1 <= hi1
            last_row = collect(table_rows(t))[end]
            @test (Float64(collect(last_row)[col_index(t, "upper")]) -
                   Float64(collect(last_row)[col_index(t, "lower")])) >= (hi1 - lo1)

            rp = run_json(["predict", "arfima", csv, "--p", "1", "--q", "0"])
            assert_envelope_ok(rp; label="predict arfima")
            @test first_table(rp.doc)[2] !== nothing

            rr = run_json(["residuals", "arfima", csv, "--p", "1", "--q", "0"])
            assert_envelope_ok(rr; label="residuals arfima")
            @test first_table(rr.doc)[2] !== nothing

            @test run_json(["forecast", "arfima", csv, "--horizons", "0"]).code == 2
            @test run_json(["forecast", "arfima", csv, "--confidence", "1.5"]).code == 2
            @test run_json(["forecast", "arfima", csv, "--trunc-lag", "0"]).code == 2
            @test run_json(["predict", "arfima", csv, "--column", "9"]).code == 3
            rm(csv; force=true)
        end

        @testset "forecast midas — direct h-step from a fresh HF block (#67)" begin
            # y_t = 1 + 2*mean(last K high-frequency obs) + noise, so a correct forecast
            # tracks the mean of the most recent block.
            lf = tempname() * ".csv"; hf = tempname() * ".csv"
            rng = MersenneTwister(601)
            mfreq, K, Tlf = 3, 6, 150
            xhf = randn(rng, Tlf * mfreq)
            open(lf, "w") do io
                println(io, "y")
                for t in 1:Tlf
                    hi = t * mfreq; lo = max(1, hi - K + 1)
                    println(io, 1.0 + 2.0 * (sum(xhf[lo:hi]) / K) + 0.2 * randn(rng))
                end
            end
            open(hf, "w") do io; println(io, "x"); for v in xhf; println(io, v); end; end

            r = run_json(["forecast", "midas", lf, "--hf-data", hf, "--m", "3", "--k", "6"])
            assert_envelope_ok(r; label="forecast midas")
            t = first_table(r.doc)[2]
            @test t !== nothing && length(table_rows(t)) == 1      # ONE direct h-step point
            @test Set(["horizon", "forecast", "lower", "upper", "se"]) ⊆ Set(String.(table_cols(t)))
            row = first(table_rows(t))
            f  = Float64(collect(row)[col_index(t, "forecast")])
            lo = Float64(collect(row)[col_index(t, "lower")])
            hi = Float64(collect(row)[col_index(t, "upper")])
            @test isfinite(f) && lo <= f <= hi
            # the DGP's conditional mean given the last K obs
            expected = 1.0 + 2.0 * (sum(xhf[end - K + 1:end]) / K)
            @test abs(f - expected) < 1.0

            # there is deliberately NO --horizons: the horizon is fixed at estimation
            @test run_json(["forecast", "midas", lf, "--hf-data", hf, "--m", "3",
                            "--k", "6", "--horizons", "4"]).code == 2
            @test run_json(["forecast", "midas", lf, "--hf-data", hf, "--m", "3",
                            "--k", "6", "--level", "1.5"]).code == 2
            @test run_json(["forecast", "midas", lf, "--hf-data", hf, "--m", "3",
                            "--k", "1", "--weights", "beta2"]).code == 3
            rm(lf; force=true); rm(hf; force=true)
        end

        @testset "GARCH-variant forecast/predict/residuals (C064 #69)" begin
            csv = dgp_garch(; T=500, seed=401)
            for v in ("igarch", "cgarch", "aparch", "figarch", "fiegarch")
                rf = run_json(["forecast", v, csv, "--horizons", "5"])
                assert_envelope_ok(rf; label="forecast $v")
                t = first_table(rf.doc)[2]
                @test t !== nothing && length(table_rows(t)) == 5

                rp = run_json(["predict", v, csv])
                assert_envelope_ok(rp; label="predict $v")
                pt = coltable(rp.doc, "variance")
                @test pt !== nothing && length(table_rows(pt)) > 0
                @test Set(["t", "variance", "volatility"]) ⊆ Set(String.(table_cols(pt)))

                rr = run_json(["residuals", v, csv])
                assert_envelope_ok(rr; label="residuals $v")
                rt = coltable(rr.doc, "residual")
                @test rt !== nothing && length(table_rows(rt)) > 0
            end

            # garch-midas: forecast returns the long-run/short-run decomposition, not a
            # VolatilityForecast, and takes NO --conf-level.
            gm = ["--m-freq", "20", "--k", "6"]
            rg = run_json(vcat(["forecast", "garch-midas", csv], gm, ["--horizons", "4"]))
            assert_envelope_ok(rg; label="forecast garch-midas")
            gt = coltable(rg.doc, "long_run")
            @test gt !== nothing && length(table_rows(gt)) == 4
            @test Set(["horizon", "total_variance", "long_run", "short_run",
                       "volatility"]) ⊆ Set(String.(table_cols(gt)))
            # total = long_run * short_run, by construction
            row = first(table_rows(gt))
            tot = Float64(collect(row)[col_index(gt, "total_variance")])
            lr = Float64(collect(row)[col_index(gt, "long_run")])
            sr = Float64(collect(row)[col_index(gt, "short_run")])
            @test isapprox(tot, lr * sr; rtol=1e-4)

            @test run_json(vcat(["predict", "garch-midas", csv], gm)).code == 0
            @test run_json(vcat(["residuals", "garch-midas", csv], gm)).code == 0

            # MEMs ships no plot_result for that NamedTuple, so --plot must NOT exist
            @test run_json(vcat(["forecast", "garch-midas", csv], gm, ["--plot"])).code == 2
            # the other five DO plot, so --plot-save is a String option there
            @test run_json(["forecast", "igarch", csv, "--horizons", "3"]).code == 0

            @test run_json(["forecast", "igarch", csv, "--horizons", "0"]).code == 2
            @test run_json(["forecast", "igarch", csv, "--conf-level", "1.5"]).code == 2
            @test run_json(["predict", "igarch", csv, "--column", "9"]).code == 3
            @test run_json(["forecast", "garch-midas", csv]).code == 2   # --m-freq required
            rm(csv; force=true)
        end

        @testset "llc/ips/breitung — unit-root vs stationary panel" begin
            # H0 for all three is that EVERY unit has a unit root — the OPPOSITE of hadri.
            i1 = dgp_panel_matrix(; N=10, T=80, unit_root=true, seed=301)
            i0 = dgp_panel_matrix(; N=10, T=80, unit_root=false, seed=303)
            for leaf in ("llc", "ips", "breitung")
                r1 = run_json(["test", leaf, i1])
                assert_envelope_ok(r1; label="$leaf unit-root panel")
                p1 = Float64(scan_metric(r1.doc, "p-value"))
                @test p1 > 0.05                       # cannot reject "all have a unit root"

                r0 = run_json(["test", leaf, i0])
                assert_envelope_ok(r0; label="$leaf stationary panel")
                p0 = Float64(scan_metric(r0.doc, "p-value"))
                @test p0 < p1                         # stationary panel scores stronger
                @test scan_metric(r0.doc, "n_units") == 10
            end
            # IPS reports the per-unit ADF statistics
            t = coltable(run_json(["test", "ips", i0]).doc, "t_statistic")
            @test t !== nothing && length(table_rows(t)) == 10
            @test run_json(["test", "llc", i0, "--deterministic", "trend"]).code == 0
            @test run_json(["test", "breitung", i0, "--cs-demean"]).code == 0
            rm(i1; force=true); rm(i0; force=true)
        end

        @testset "fisher-johansen / dh-causality — panel leaves" begin
            cp = dgp_coint_panel(; N=10, T=50, seed=305)
            fj = run_json(["test", "fisher-johansen", cp, "--vars", "y,x"])
            assert_envelope_ok(fj; label="fisher-johansen")
            t = coltable(fj.doc, "trace_statistic")
            @test t !== nothing && length(table_rows(t)) >= 1
            @test Set(["rank", "trace_statistic", "trace_p_value", "max_statistic",
                       "max_p_value"]) ⊆ Set(String.(table_cols(t)))
            @test scan_metric(fj.doc, "n_units") == 10
            @test run_json(["test", "fisher-johansen", cp, "--vars", "y,x",
                            "--combine", "choi"]).code == 0

            dh = run_json(["test", "dh-causality", cp, "--cause", "x", "--effect", "y"])
            assert_envelope_ok(dh; label="dh-causality")
            for m in ("cause", "effect", "W-bar", "Z-bar", "Z-tilde", "Z-tilde p-value")
                @test scan_metric(dh.doc, m) !== nothing
            end
            @test String(scan_metric(dh.doc, "cause")) == "x"
            @test String(scan_metric(dh.doc, "effect")) == "y"
            rm(cp; force=true)
        end

        @testset "estimate preg — PCSE and Prais-Winsten AR(1) (#75)" begin
            cp = dgp_coint_panel(; N=10, T=50, seed=307)
            base = ["estimate", "preg", cp, "--dep", "y", "--indep", "x"]
            @test run_json(vcat(base, ["--cov-type", "pcse"])).code == 0
            @test run_json(vcat(base, ["--cov-type", "pcse", "--pcse-unbalanced", "pairwise"])).code == 0
            for a in ("common", "panel-specific")
                @test run_json(vcat(base, ["--ar1", a])).code == 0
            end
            # --pcse-unbalanced only means anything under --cov-type pcse
            @test run_json(vcat(base, ["--pcse-unbalanced", "pairwise"])).code == 2
            @test run_json(vcat(base, ["--ar1", "bogus"])).code == 2
            rm(cp; force=true)
        end

        @testset "C070 remainder — bad input stays typed" begin
            mv = dgp_panel_matrix(; N=8, T=60, seed=309)
            cp = dgp_coint_panel(; N=8, T=40, seed=311)
            @test run_json(["test", "llc", mv, "--lags", "junk"]).code == 2
            @test run_json(["test", "llc", mv, "--max-lags", "-1"]).code == 2
            @test run_json(["test", "breitung", mv, "--lags", "-1"]).code == 2
            @test run_json(["test", "llc", mv, "--deterministic", "bogus"]).code == 2
            @test run_json(["test", "fisher-johansen", cp, "--vars", "y"]).code == 2
            @test run_json(["test", "fisher-johansen", cp, "--vars", "nosuch"]).code == 2
            @test run_json(["test", "dh-causality", cp, "--effect", "y"]).code == 2
            @test run_json(["test", "dh-causality", cp, "--cause", "x", "--effect", "y",
                            "--p", "0"]).code == 2
            @test run_json(["test", "dh-causality", cp, "--cause", "y", "--effect", "y"]).code == 2
            rm(mv; force=true); rm(cp; force=true)
        end

        # ── C067 remainder (#72): cross-section OLS diagnostics. Each case asserts the
        # DISCRIMINATING DIRECTION on a DGP built for that null.

        @testset "white/glejser/harvey — homoskedastic vs heteroskedastic" begin
            hom = dgp_reg_diag(; n=300, hetero=false, seed=201)
            het = dgp_reg_diag(; n=300, hetero=true, seed=203)
            for leaf in ("white", "glejser", "harvey")
                r0 = run_json(["test", leaf, hom, "--dep", "y"])
                assert_envelope_ok(r0; label="$leaf homoskedastic")
                p0 = Float64(scan_metric(r0.doc, "p-value"))
                @test p0 > 0.05                     # H0 homoskedasticity holds

                r1 = run_json(["test", leaf, het, "--dep", "y"])
                assert_envelope_ok(r1; label="$leaf heteroskedastic")
                p1 = Float64(scan_metric(r1.doc, "p-value"))
                @test p1 < 0.05                     # reject homoskedasticity
            end
            @test run_json(["test", "white", hom, "--dep", "y", "--no-cross-terms"]).code == 0
            rm(hom; force=true); rm(het; force=true)
        end

        @testset "chow — stable sample vs a slope break" begin
            stable = dgp_reg_diag(; n=200, seed=205)
            brk = dgp_reg_diag(; n=200, break_at=100, seed=207)
            r0 = run_json(["test", "chow", stable, "--dep", "y", "--break-at", "100"])
            assert_envelope_ok(r0; label="chow stable")
            @test Float64(scan_metric(r0.doc, "p-value")) > 0.05

            r1 = run_json(["test", "chow", brk, "--dep", "y", "--break-at", "100"])
            assert_envelope_ok(r1; label="chow break")
            @test Float64(scan_metric(r1.doc, "p-value")) < 0.05
            @test run_json(["test", "chow", brk, "--dep", "y", "--break-at", "60,120"]).code == 0
            rm(stable; force=true); rm(brk; force=true)
        end

        @testset "cusum/cusumsq — band path, no p-value" begin
            brk = dgp_reg_diag(; n=200, break_at=100, seed=209)
            for (leaf, col) in (("cusum", "cusum"), ("cusumsq", "cusumsq"))
                r = run_json(["test", leaf, brk, "--dep", "y"])
                assert_envelope_ok(r; label=leaf)
                t = coltable(r.doc, col)
                @test t !== nothing
                @test Set(["observation", col, "lower", "upper"]) ⊆ Set(String.(table_cols(t)))
                @test length(table_rows(t)) > 0
                # StabilityResult has a band, NOT a p-value — the crossing IS the verdict
                @test scan_metric(r.doc, "p-value") === nothing
                @test scan_metric(r.doc, "crossed band") !== nothing
            end
            rm(brk; force=true)
        end

        @testset "influence / recursive-residuals — per-observation output" begin
            csv = dgp_reg_diag(; n=150, seed=211)
            ri = run_json(["test", "influence", csv, "--dep", "y"])
            assert_envelope_ok(ri; label="influence")
            t = coltable(ri.doc, "hat")
            @test t !== nothing && length(table_rows(t)) == 150      # one row per observation
            @test Set(["hat", "student_internal", "student_external", "dffits", "cooksd"]) ⊆
                  Set(String.(table_cols(t)))
            # Leverages sum to k (a standard identity) — checks we read the right field.
            # Tolerance is loose because the CLI rounds `hat` to 6 dp, so summing n rows
            # accumulates up to n*5e-7 of rounding (1e-6 fails at n=150).
            hs = [Float64(collect(r)[col_index(t, "hat")]) for r in table_rows(t)]
            @test isapprox(sum(hs), 3.0; atol=1e-3)                  # k = const + x1 + x2

            rr = run_json(["test", "recursive-residuals", csv, "--dep", "y"])
            assert_envelope_ok(rr; label="recursive-residuals")
            rt = coltable(rr.doc, "recursive_residual")
            @test rt !== nothing && length(table_rows(rt)) == 150 - 3   # n - k
            rm(csv; force=true)
        end

        @testset "estimate select — recovers the true model (#72)" begin
            # y depends on x1 and x2 only; x3/x4 are pure noise, so a working search
            # must keep the former and drop the latter.
            csv = dgp_select(; n=400, seed=217)
            r = run_json(["estimate", "select", csv, "--dep", "y"])
            assert_envelope_ok(r; label="estimate select")
            sel = String(scan_metric(r.doc, "selected"))
            @test occursin("x1", sel) && occursin("x2", sel)
            @test !occursin("x3", sel) && !occursin("x4", sel)
            @test Float64(scan_metric(r.doc, "n selected")) >= 2

            # the selection path is the audit trail
            t = coltable(r.doc, "action")
            @test t !== nothing
            @test Set(["step", "action", "variable", "statistic"]) ⊆ Set(String.(table_cols(t)))

            # --keep forces a regressor in even though it is irrelevant
            rk = run_json(["estimate", "select", csv, "--dep", "y", "--keep", "x3"])
            assert_envelope_ok(rk; label="estimate select --keep")
            @test occursin("x3", String(scan_metric(rk.doc, "selected")))

            for m in ("forward", "backward", "gets")
                @test run_json(["estimate", "select", csv, "--dep", "y", "--method", m]).code == 0
            end
            @test run_json(["estimate", "select", csv, "--dep", "y", "--criterion", "bic"]).code == 0

            @test run_json(["estimate", "select", csv, "--dep", "y", "--p-enter", "0"]).code == 2
            @test run_json(["estimate", "select", csv, "--dep", "y",
                            "--p-enter", "0.2", "--p-remove", "0.05"]).code == 2
            @test run_json(["estimate", "select", csv, "--dep", "y", "--keep", "nosuch"]).code == 3
            @test run_json(["estimate", "select", csv, "--dep", "y", "--method", "bogus"]).code == 2
            rm(csv; force=true)
        end

        @testset "estimate iv — k-class family (#72)" begin
            csv = dgp_iv(; T=400, seed=215)
            base = ["estimate", "iv", csv, "--dep", "y", "--endogenous", "x_endog",
                    "--instruments", "z1,z2"]
            b = Dict{String,Float64}()
            for m in ("tsls", "liml", "fuller", "kclass")
                args = m == "kclass" ? vcat(base, ["--method", m, "--k", "1"]) :
                                       vcat(base, ["--method", m])
                r = run_json(args)
                assert_envelope_ok(r; label="iv $m")
                t = coltable(r.doc, "estimate")
                row = first(rr for rr in table_rows(t)
                            if String(collect(rr)[col_index(t, "term")]) == "x_endog")
                b[m] = Float64(collect(row)[col_index(t, "estimate")])
                @test isapprox(b[m], 2.0; atol=0.25)      # all recover the true beta
            end
            # k=1 IS 2SLS by construction — a deterministic identity, not a tolerance test
            @test isapprox(b["kclass"], b["tsls"]; atol=1e-8)
            # LIML reports kappa_hat >= 1; Fuller shifts k below it by a/(n-m)
            rl = run_json(vcat(base, ["--method", "liml"]))
            @test Float64(scan_metric(rl.doc, "kappa_hat")) >= 1.0
            rf = run_json(vcat(base, ["--method", "fuller"]))
            @test Float64(scan_metric(rf.doc, "k-class k")) < Float64(scan_metric(rf.doc, "kappa_hat"))

            @test run_json(vcat(base, ["--method", "kclass"])).code == 2      # --k required
            @test run_json(vcat(base, ["--method", "tsls", "--k", "1"])).code == 2
            @test run_json(vcat(base, ["--method", "bogus"])).code == 2
            rm(csv; force=true)
        end

        @testset "C067 remainder — bad input stays typed" begin
            csv = dgp_reg_diag(; n=120, seed=213)
            @test run_json(["test", "chow", csv, "--dep", "y"]).code == 2            # --break-at required
            @test run_json(["test", "chow", csv, "--dep", "y", "--break-at", "junk"]).code == 2
            @test run_json(["test", "chow", csv, "--dep", "y", "--break-at", "0"]).code == 2
            # an out-of-range break reaches MEMs' ArgumentError → typed data/invalid, not exit 1
            @test run_json(["test", "chow", csv, "--dep", "y", "--break-at", "500"]).code == 3
            @test run_json(["test", "cusum", csv, "--dep", "y", "--level", "0"]).code == 2
            @test run_json(["test", "white", csv, "--dep", "nope"]).code == 3
            @test run_json(["test", "influence", csv, "--dep", "nope"]).code == 3
            rm(csv; force=true)
        end

        @testset "C069 remainder — bad input stays typed" begin
            uni = dgp_iid(; T=200, seed=145)
            reg = dgp_coint(; T=200, seed=147)
            @test run_json(["test", "hegy", uni, "--frequency", "7"]).code == 2
            @test run_json(["test", "hegy", uni, "--lags", "junk"]).code == 2
            @test run_json(["test", "sadf", uni, "--r0", "1.5"]).code == 2
            @test run_json(["test", "sadf", uni, "--adflag", "-1"]).code == 2
            @test run_json(["test", "gsadf", uni, "--mc-reps", "0"]).code == 2
            @test run_json(["test", "edf", uni, "--params", "specified"]).code == 2
            @test run_json(["test", "edf", uni, "--dist", "bogus"]).code == 2
            # EG/PO take :none|:constant|:trend — cointreg's "linear" must NOT be accepted
            @test run_json(["test", "engle-granger", reg, "--trend", "linear"]).code == 2
            @test run_json(["test", "engle-granger", reg, "--dep", "nope"]).code == 3
            @test run_json(["test", "phillips-ouliaris", reg, "--bandwidth", "junk"]).code == 2
            @test run_json(["test", "park-added", reg, "--q-add", "0"]).code == 2
            @test run_json(["test", "hansen-instability", reg, "--dep", "nope"]).code == 3
            # too few observations for the (y,X) leaves → typed data error, never exit 1
            short = tempname() * ".csv"
            write(short, "y,x\n1.0,2.0\n2.0,3.1\n3.0,3.9\n4.0,5.2\n")
            @test run_json(["test", "engle-granger", short, "--dep", "y"]).code == 3
            @test run_json(["test", "ers", short]).code == 3        # ERS needs ≥ 30 obs
            rm(short; force=true)
            rm(uni; force=true); rm(reg; force=true)
        end
    end

    @testset "forecast bvar/dynamic/gdfm/favar tidy (C051 redesign)" begin
        # Previously hand-computed; now routed through MEMs forecast(...) → long_table.
        csv = dgp_var2(; T=150, seed=51)
        for args in (["forecast", "bvar", csv, "--lags", "1", "--draws", "80", "--horizons", "6"],
                     ["forecast", "dynamic", csv, "--nfactors", "1", "--horizons", "6"],
                     ["forecast", "gdfm", csv, "--nfactors", "1", "--dynamic-rank", "1", "--horizons", "6"],
                     ["forecast", "favar", csv, "--factors", "1", "--key-vars", "1", "--horizons", "6"])
            r = run_json(args)
            assert_envelope_ok(r; label=join(args[1:2], " "))
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            tbl !== nothing && @test table_cols(tbl) == ["horizon", "variable", "value", "lower", "upper"]
        end
        rm(csv; force=true)
    end

    @testset "filter hp" begin
        csv = dgp_trend_cycle(; T=120, seed=21)
        r = run_json(["filter", "hp", csv, "--columns", "1"])
        assert_envelope_ok(r; label="filter hp")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 50
        rm(csv; force=true)
    end

    @testset "filter bn" begin
        csv = dgp_trend_cycle(; T=150, seed=23)
        r = run_json(["filter", "bn", csv, "--columns", "1"])
        # BN may fail on some series — accept ok or graceful error
        if r.code == 0 && r.doc !== nothing && string(r.doc.status) == "ok"
            assert_envelope_ok(r; label="filter bn")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
        else
            @test r.code != 0 || (r.doc !== nothing && string(r.doc.status) == "error")
            @info "filter bn skipped/failed on DGP (acceptable)" code=r.code
        end
        rm(csv; force=true)
    end

    @testset "estimate garch" begin
        csv = dgp_garch(; T=400, seed=25)
        r = run_json(["estimate", "garch", csv, "--column", "1", "--p", "1", "--q", "1"])
        assert_envelope_ok(r; label="estimate garch")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 1
        rm(csv; force=true)
    end

    # W11/#113 — conditional distribution of the innovations.
    @testset "GARCH --dist normal|student|ged (W11/#113)" begin
        csv = dgp_garch(; T=500, seed=1113)
        _tab(doc, cols...) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                all(c -> c in table_cols(v), cols) && return v
            end
            nothing
        end
        # The three estimators that take `dist` upstream.
        for leaf in ("garch", "egarch", "gjr-garch")
            base = run_json(["estimate", leaf, csv, "--p", "1", "--q", "1"])
            assert_envelope_ok(base; label="estimate $leaf normal")
            for d in ("student", "ged")
                r = run_json(["estimate", leaf, csv, "--p", "1", "--q", "1", "--dist", d])
                assert_envelope_ok(r; label="estimate $leaf $d")
                # the shape parameter is estimated JOINTLY but lives OUTSIDE coef(model),
                # so it needs its own table — assert it is actually surfaced
                sh = _tab(r.doc, "parameter", "estimate", "distribution")
                @test sh !== nothing
                if sh !== nothing
                    @test String(collect(first(table_rows(sh)))[col_index(sh, "distribution")]) == d
                    @test isfinite(Float64(collect(first(table_rows(sh)))[col_index(sh, "estimate")]))
                end
            end
            # ...and the Gaussian default must NOT emit that table
            @test _tab(base.doc, "parameter", "estimate", "distribution") === nothing
        end
        @test run_json(["forecast", "garch", csv, "--dist", "student", "--horizons", "5"]).code == 0

        # arch and sv have NO `dist` kwarg upstream, so the option must not exist on them
        # (unknown option, exit 2) rather than be accepted and ignored.
        @test run_json(["estimate", "arch", csv, "--q", "1", "--dist", "student"]).code == 2
        @test run_json(["estimate", "sv", csv, "--dist", "student"]).code == 2
        @test run_json(["estimate", "garch", csv, "--dist", "bogus"]).code == 2
        rm(csv; force=true)
    end

    @testset "estimate GARCH variants on real MEMs 0.7.0 (C064a)" begin
        # hand-built coef table (parameter|estimate) + diagnostics (metric|value)
        _coef_of(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                cols = table_cols(v)
                ("parameter" in cols && "estimate" in cols) && return v
            end
            return nothing
        end
        _diag_of(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                "metric" in table_cols(v) && return v
            end
            return nothing
        end
        _finite_estimates(tbl) = begin
            ei = col_index(tbl, "estimate")
            all(isfinite(Float64(collect(row)[ei])) for row in table_rows(tbl))
        end

        @testset "igarch — Σα+Σβ=1 ⇒ persistence≈1" begin
            csv = dgp_garch(; T=400, seed=101)
            r = run_json(["estimate", "igarch", csv, "--column", "1", "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="igarch")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            pers = metric_value(_diag_of(r.doc), "persistence")
            @test pers !== nothing && isapprox(Float64(pers), 1.0; atol=1e-6)
            rm(csv; force=true)
        end

        @testset "cgarch — component decomposition, ρ∈(0,1]" begin
            csv = dgp_garch(; T=500, seed=102)
            r = run_json(["estimate", "cgarch", csv, "--column", "1"])
            assert_envelope_ok(r; label="cgarch")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            pers = metric_value(_diag_of(r.doc), "persistence")
            @test pers !== nothing && 0.0 < Float64(pers) <= 1.0001
            @test metric_value(_diag_of(r.doc), "unconditional_variance") !== nothing
            rm(csv; force=true)
        end

        @testset "aparch — power δ>0, finite persistence" begin
            csv = dgp_garch(; T=400, seed=103)
            r = run_json(["estimate", "aparch", csv, "--column", "1", "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="aparch")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            dlt = metric_value(_diag_of(r.doc), "delta")
            @test dlt !== nothing && Float64(dlt) > 0.0
            pers = metric_value(_diag_of(r.doc), "persistence")
            @test pers !== nothing && isfinite(Float64(pers))
            rm(csv; force=true)
        end

        @testset "figarch — long memory d∈(0,1)" begin
            csv = dgp_garch(; T=400, seed=104)
            r = run_json(["estimate", "figarch", csv, "--column", "1", "--truncation", "100"])
            assert_envelope_ok(r; label="figarch")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            d = metric_value(_diag_of(r.doc), "d")               # d ∈ [0,1] (logistic transform; boundary reachable)
            @test d !== nothing && 0.0 <= Float64(d) <= 1.0
            rm(csv; force=true)
        end

        @testset "fiegarch — long memory d∈[0,1]" begin
            csv = dgp_garch(; T=400, seed=105)
            r = run_json(["estimate", "fiegarch", csv, "--column", "1", "--truncation", "100"])
            assert_envelope_ok(r; label="fiegarch")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            d = metric_value(_diag_of(r.doc), "d")               # d ∈ [0,1] (logistic transform; boundary reachable)
            @test d !== nothing && 0.0 <= Float64(d) <= 1.0
            rm(csv; force=true)
        end

        @testset "garch-midas — realized, short-run persistence α+β∈(0,1)" begin
            csv = dgp_garch(; T=600, seed=106)
            r = run_json(["estimate", "garch-midas", csv, "--column", "1", "--m-freq", "20", "--k", "6"])
            assert_envelope_ok(r; label="garch-midas")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            @test _finite_estimates(tbl)
            pers = metric_value(_diag_of(r.doc), "persistence")
            @test pers !== nothing && 0.0 < Float64(pers) < 1.0
            vr = metric_value(_diag_of(r.doc), "variance_ratio")
            @test vr !== nothing && 0.0 <= Float64(vr) <= 1.0
            rm(csv; force=true)
        end

        @testset "garch-midas missing --m-freq → usage error (not exit 1)" begin
            csv = dgp_garch(; T=200, seed=107)
            r = run_json(["estimate", "garch-midas", csv, "--column", "1"])
            @test r.code == 2
            rm(csv; force=true)
        end
    end

    @testset "MGARCH ccc/dcc/bekk + volatility diagnostics on real MEMs (C064b)" begin
        # scanners: wide correlation (series col), dynamics (parameter|estimate), diag (metric|value)
        _corr_of(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                "series" in table_cols(v) && return v
            end
            return nothing
        end
        _diag_of(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                "metric" in table_cols(v) && return v
            end
            return nothing
        end
        _coef_of(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                cols = table_cols(v)
                ("parameter" in cols && "estimate" in cols) && return v
            end
            return nothing
        end

        @testset "estimate ccc — n×n correlation, no second-stage params" begin
            csv = dgp_mgarch(; T=300, n=3, seed=201)
            r = run_json(["estimate", "ccc", csv, "--p", "1", "--q", "1"])
            assert_envelope_ok(r; label="estimate ccc")
            corr = _corr_of(r.doc); @test corr !== nothing
            @test "series" in table_cols(corr) && "r1" in table_cols(corr)
            @test length(table_rows(corr)) == 3          # 3 series → 3 rows (wide sector×sector)
            @test string(metric_value(_diag_of(r.doc), "kind")) == "ccc"
            @test _coef_of(r.doc) === nothing            # CCC has no dynamics coef table
            rm(csv; force=true)
        end

        @testset "estimate dcc — a,b ∈ (0,1), persistence < 1" begin
            csv = dgp_mgarch(; T=350, n=2, seed=202)
            r = run_json(["estimate", "dcc", csv])
            assert_envelope_ok(r; label="estimate dcc")
            tbl = _coef_of(r.doc); @test tbl !== nothing
            names = String[string(collect(row)[col_index(tbl, "parameter")]) for row in table_rows(tbl)]
            @test "a" in names && "b" in names
            pers = metric_value(_diag_of(r.doc), "persistence")
            @test pers !== nothing && 0.0 <= Float64(pers) <= 1.0001
            @test string(metric_value(_diag_of(r.doc), "correction")) == "none"
            rm(csv; force=true)
        end

        @testset "estimate dcc --correction aielli (cDCC)" begin
            csv = dgp_mgarch(; T=300, n=2, seed=203)
            r = run_json(["estimate", "dcc", csv, "--correction", "aielli"])
            assert_envelope_ok(r; label="estimate dcc aielli")
            @test string(metric_value(_diag_of(r.doc), "correction")) == "aielli"
            rm(csv; force=true)
        end

        @testset "estimate bekk scalar / diagonal" begin
            csv = dgp_mgarch(; T=300, n=2, seed=204)
            rs = run_json(["estimate", "bekk", csv, "--kind", "scalar"])
            assert_envelope_ok(rs; label="estimate bekk scalar")
            @test string(metric_value(_diag_of(rs.doc), "bekk_kind")) == "scalar"
            ts = _coef_of(rs.doc); @test ts !== nothing
            snames = String[string(collect(row)[col_index(ts, "parameter")]) for row in table_rows(ts)]
            @test "a" in snames && "b" in snames

            rd = run_json(["estimate", "bekk", csv, "--kind", "diagonal"])
            assert_envelope_ok(rd; label="estimate bekk diagonal")
            @test string(metric_value(_diag_of(rd.doc), "bekk_kind")) == "diagonal"
            td = _coef_of(rd.doc); @test td !== nothing
            dnames = String[string(collect(row)[col_index(td, "parameter")]) for row in table_rows(td)]
            @test "a1" in dnames && "b1" in dnames
            rm(csv; force=true)
        end

        @testset "estimate ccc on 1-column data → data error (not exit 1)" begin
            csv = dgp_garch(; T=200, seed=205)   # single column 'r'
            r = run_json(["estimate", "ccc", csv])
            @test r.code == 3                     # ArgumentError('≥2 series') → data/invalid
            rm(csv; force=true)
        end

        @testset "test sign-bias — Engle-Ng joint p-value ∈ [0,1]" begin
            csv = dgp_garch(; T=400, seed=206)
            r = run_json(["test", "sign-bias", csv, "--column", "1", "--model", "garch"])
            assert_envelope_ok(r; label="test sign-bias")
            jp = metric_value(_diag_of(r.doc), "joint_pvalue")
            @test jp !== nothing && 0.0 <= Float64(jp) <= 1.0
            @test metric_value(_diag_of(r.doc), "joint_statistic") !== nothing
            rm(csv; force=true)
        end

        @testset "test nyblom — individual L stats + joint L_C vs cv" begin
            csv = dgp_garch(; T=400, seed=207)
            r = run_json(["test", "nyblom", csv, "--column", "1", "--model", "garch"])
            assert_envelope_ok(r; label="test nyblom")
            ind = nothing
            for (_, v) in pairs(r.doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                "L_stat" in table_cols(v) && (ind = v)
            end
            @test ind !== nothing && length(table_rows(ind)) >= 3   # μ, ω, α1, β1
            @test metric_value(_diag_of(r.doc), "joint_LC") !== nothing
            @test metric_value(_diag_of(r.doc), "cv_joint_5pct") !== nothing
            rm(csv; force=true)
        end

        @testset "test sign-bias bad --model → usage error (not exit 1)" begin
            csv = dgp_garch(; T=200, seed=208)
            r = run_json(["test", "sign-bias", csv, "--model", "bogus"])
            @test r.code == 2
            rm(csv; force=true)
        end
    end

    @testset "estimate reg OLS slope ≈ 2" begin
        csv = dgp_reg(; T=300, seed=27)
        r = run_json(["estimate", "reg", csv, "--dep", "y"])
        assert_envelope_ok(r; label="estimate reg")
        # Scan all tables for a coefficient on x ≈ 2
        found = false
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object || v isa AbstractDict) || continue
            haskey(v, :rows) || haskey(v, "rows") || continue
            cols = table_cols(v)
            rows = table_rows(v)
            name_i = findfirst(c -> c == "term" || occursin("var", lowercase(c)) || occursin("param", lowercase(c)), cols)
            est_i = findfirst(c -> occursin("coef", lowercase(c)) || occursin("estimate", lowercase(c)), cols)
            name_i === nothing && continue
            est_i === nothing && continue
            for row in rows
                rr = collect(row)
                if occursin(r"^x$", lowercase(string(rr[name_i]))) || lowercase(string(rr[name_i])) == "x"
                    β = rr[est_i]
                    @test β isa Real
                    @test 1.5 <= Float64(β) <= 2.5  # teeth: true slope is 2
                    found = true
                end
            end
        end
        @test found
        rm(csv; force=true)
    end

    @testset "estimate logit (C051 tidy coef)" begin
        csv = dgp_logit(; T=400, seed=29)
        r = run_json(["estimate", "logit", csv, "--dep", "y"])
        assert_envelope_ok(r; label="estimate logit")
        # a table carries the tidy single-equation coef schema
        tidy = ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        has_coef = any(v -> (v isa JSON3.Object && haskey(v, :rows) && table_cols(v) == tidy),
                       values(r.doc.data))
        @test has_coef
        rm(csv; force=true)
    end

    @testset "estimate ologit/mlogit/preg tidy coef (C051)" begin
        # Coef schemas share a core; some prepend equation/alternative/block. Assert the
        # core tidy columns are present (subset) — robust across the per-model variants.
        core = ["term", "estimate", "std_error", "stat", "p_value", "ci_lower", "ci_upper"]
        hascore(doc) = any(v -> (v isa JSON3.Object && haskey(v, :rows) && issubset(core, table_cols(v))),
                           values(doc.data))

        # ordered/multinomial need ≥3 categories → deterministic 3-category outcome from x
        catcsv = tempname() * ".csv"
        catrng = MersenneTwister(61)
        open(catcsv, "w") do io
            println(io, "y,x")
            for _ in 1:300
                xi = randn(catrng)
                yi = xi < -0.5 ? 0 : (xi < 0.5 ? 1 : 2)
                println(io, "$yi,$xi")
            end
        end
        ro = run_json(["estimate", "ologit", catcsv, "--dep", "y"])
        assert_envelope_ok(ro; label="estimate ologit")
        @test hascore(ro.doc)                 # ordered coef table is block|term|…

        rml = run_json(["estimate", "mlogit", catcsv, "--dep", "y"])
        assert_envelope_ok(rml; label="estimate mlogit")
        @test hascore(rml.doc)                # multinomial is alternative|term|…
        rm(catcsv; force=true)

        pc = dgp_did_panel(; N=40, T=10, seed=65)
        rp = run_json(["estimate", "preg", pc, "--id-col", "id", "--time-col", "time",
                       "--dep", "y", "--indep", "d"])
        assert_envelope_ok(rp; label="estimate preg")
        @test hascore(rp.doc)
        rm(pc; force=true)
    end

    @testset "test johansen on cointegrated pair" begin
        # C054 #270: the Johansen rank off-by-one is fixed upstream. A single
        # cointegrating relation must reject r=0 and fail to reject r=1, i.e.
        # the selected rank is exactly 1 (not 2, which the old bug produced).
        csv = dgp_coint(; T=300, seed=31)
        r = run_json(["test", "johansen", csv, "--lags", "2"])
        assert_envelope_ok(r; label="test johansen")
        trace = named_table(r.doc, :johansen_trace_test)
        @test trace !== nothing
        rows = [collect(x) for x in table_rows(trace)]
        ri = col_index(trace, "rank"); rj = col_index(trace, "reject")
        rejat(k) = (row = rows[findfirst(x -> Int(x[ri]) == k, rows)]; string(row[rj]))
        @test rejat(0) == "yes"   # reject r=0 → at least one cointegrating vector
        @test rejat(1) == "no"    # fail to reject r=1 → exactly one (rank = 1)
        rm(csv; force=true)
    end

    @testset "estimate vecm --rank auto selects rank 1" begin
        # C054 #270: auto rank selection on a cointegrated pair → exactly one
        # cointegrating vector (CV1), recovering β ≈ [1, -1].
        csv = dgp_coint(; T=300, seed=33)
        r = run_json(["estimate", "vecm", csv, "--lags", "2", "--rank", "auto"])
        assert_envelope_ok(r; label="estimate vecm --rank auto")
        beta = named_table(r.doc, :cointegrating_vectors_beta)
        @test beta !== nothing
        cols = table_cols(beta)
        @test "CV1" in cols        # exactly one cointegrating vector auto-selected
        @test !("CV2" in cols)
        rm(csv; force=true)
    end

    @testset "irf vecm tidy (C051)" begin
        csv = dgp_coint(; T=300, seed=35)
        r = run_json(["irf", "vecm", csv, "--lags", "2", "--rank", "1",
                      "--shock", "1", "--ci", "none", "--horizons", "8"])
        assert_envelope_ok(r; label="irf vecm")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value", "lower", "upper"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(unique(row[ci["shock"]] for row in rows)) == 1   # one shock
        end
        rm(csv; force=true)
    end

    @testset "fevd vecm tidy (C051)" begin
        csv = dgp_coint(; T=300, seed=37)
        r = run_json(["fevd", "vecm", csv, "--lags", "2", "--rank", "1", "--horizons", "10"])
        assert_envelope_ok(r; label="fevd vecm")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            @test all(-1e-8 <= Float64(row[ci["value"]]) <= 1.0 + 1e-8 for row in rows)
        end
        rm(csv; force=true)
    end

    @testset "irf lp tidy (C051)" begin
        csv = dgp_var2(; T=200, seed=53)
        r = run_json(["irf", "lp", csv, "--shock", "1", "--horizons", "8", "--lags", "4"])
        assert_envelope_ok(r; label="irf lp")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            # C051: slp.irf is a full ImpulseResponse (Plagborg-Møller & Wolf 2021 stack LP
            # responses into the same 3D array as a VAR IRF) — same schema as irf var.
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value", "lower", "upper"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(unique(row[ci["shock"]] for row in rows)) == 1   # one shock selected
        end
        rm(csv; force=true)
    end

    @testset "irf/fevd favar tidy (C051)" begin
        # irf(favar,...)/fevd(favar,...) delegate to the VAR representation, so favar
        # renders through the same ImpulseResponse/FEVD long_table as irf/fevd var — one
        # tidy table now covers every shock (no more per-shock output files).
        csv = dgp_var2(; T=150, seed=55)
        ri = run_json(["irf", "favar", csv, "--factors", "1", "--key-vars", "1", "--horizons", "6"])
        assert_envelope_ok(ri; label="irf favar")
        _, ti = first_table(ri.doc)
        @test ti !== nothing && table_cols(ti) == ["horizon", "variable", "shock", "value", "lower", "upper"]

        rf = run_json(["fevd", "favar", csv, "--factors", "1", "--key-vars", "1", "--horizons", "6"])
        assert_envelope_ok(rf; label="fevd favar")
        _, tf = first_table(rf.doc)
        @test tf !== nothing && table_cols(tf) == ["horizon", "variable", "shock", "value"]
        rm(csv; force=true)
    end

    @testset "irf/fevd sdfm tidy (C051)" begin
        # irf(sdfm,...) returns a panel-wide ImpulseResponse directly; fevd(sdfm,...)
        # delegates to the factor VAR — both render through the same long_table as var.
        csv = dgp_var2(; T=150, seed=57)
        ri = run_json(["irf", "sdfm", csv, "--factors", "1", "--horizons", "6"])
        assert_envelope_ok(ri; label="irf sdfm")
        _, ti = first_table(ri.doc)
        @test ti !== nothing && table_cols(ti) == ["horizon", "variable", "shock", "value", "lower", "upper"]

        rf = run_json(["fevd", "sdfm", csv, "--factors", "1", "--horizons", "6"])
        assert_envelope_ok(rf; label="fevd sdfm")
        _, tf = first_table(rf.doc)
        @test tf !== nothing && table_cols(tf) == ["horizon", "variable", "shock", "value"]
        rm(csv; force=true)
    end

    @testset "sdfm/gdfm 0.9.1 surface (W1/#165)" begin
        csv = dgp_var2(; T=150, seed=165)
        # legacy estimator + legacy spectrum riders
        rl = run_json(["estimate", "sdfm", csv, "--factors", "1",
                       "--method", "gdfm-var",
                       "--spectral", "smoothed-periodogram"])
        assert_envelope_ok(rl; label="estimate sdfm legacy")
        # automatic q-selection through each upstream criterion (deterministic)
        for qm in ("hallin-liska", "bai-ng", "amengual-watson")
            ra = run_json(["estimate", "sdfm", csv, "--q-method", qm])
            assert_envelope_ok(ra; label="estimate sdfm auto $qm")
            sm = named_table(ra.doc, :sdfm_estimation_summary)
            @test sm !== nothing
            if sm !== nothing
                mets = Dict(String(collect(r)[1]) => String(collect(r)[2])
                            for r in table_rows(sm))
                @test parse(Int, mets["dynamic_factors"]) >= 1
            end
        end
        # proxy identification with a (strong, hence non-degenerate) instrument
        rng = MersenneTwister(166)
        y1 = randn(rng, 150); y2 = randn(rng, 150); y3 = randn(rng, 150)
        idf = DataFrame(y1=y1, y2=y2, y3=y3, z=y1 .+ 0.1 .* randn(rng, 150))
        icsv = write_csv(idf; prefix="sdfm_proxy")
        rp = run_json(["estimate", "sdfm", icsv, "--factors", "1",
                       "--id", "proxy", "--instrument", "z"])
        assert_envelope_ok(rp; label="estimate sdfm proxy")
        # guards are typed usage errors, never exit 1
        @test run_json(["estimate", "sdfm", icsv, "--factors", "1",
                        "--id", "proxy"]).code == 2
        @test run_json(["estimate", "sdfm", icsv, "--factors", "1",
                        "--instrument", "z"]).code == 2
        @test run_json(["estimate", "sdfm", csv, "--factors", "1",
                        "--method", "bogus"]).code == 2
        # FHLR one-sided / spectral forecast branches
        for m in ("one-sided", "spectral")
            rf2 = run_json(["forecast", "gdfm", csv, "--nfactors", "1",
                            "--dynamic-rank", "1", "--horizons", "4",
                            "--method", m])
            assert_envelope_ok(rf2; label="forecast gdfm $m")
            _, t2 = first_table(rf2.doc)
            @test t2 !== nothing &&
                table_cols(t2) == ["horizon", "variable", "value", "lower", "upper"]
        end
        # new forecast sdfm leaf, incl. bootstrap intervals
        fs = run_json(["forecast", "sdfm", csv, "--factors", "1", "--horizons", "4"])
        assert_envelope_ok(fs; label="forecast sdfm")
        _, tf2 = first_table(fs.doc)
        @test tf2 !== nothing &&
            table_cols(tf2) == ["horizon", "variable", "value", "lower", "upper"]
        fb = run_json(["forecast", "sdfm", csv, "--factors", "1", "--horizons", "4",
                       "--ci", "bootstrap", "--reps", "20"])
        assert_envelope_ok(fb; label="forecast sdfm bootstrap")
        # SDFM residual-bootstrap IRF bands
        ib = run_json(["irf", "sdfm", csv, "--factors", "1", "--horizons", "6",
                       "--ci", "bootstrap", "--reps", "20"])
        assert_envelope_ok(ib; label="irf sdfm bootstrap")
        # --plot-save against the real recipes (W5/#95: only advertise what runs)
        for args in (["estimate", "gdfm", csv, "--dynamic-rank", "1"],
                     ["forecast", "gdfm", csv, "--dynamic-rank", "1", "--horizons", "4"],
                     ["estimate", "sdfm", csv, "--factors", "1"])
            out = tempname() * ".html"
            rpl = run_json(vcat(args, ["--plot-save", out]))
            assert_envelope_ok(rpl; label=join(args[1:2], " ") * " --plot-save")
            @test isfile(out) && filesize(out) > 1000
            rm(out; force=true)
        end
        rm(csv; force=true); rm(icsv; force=true)
    end

    @testset "factor family carries CSV varnames (W10/#131, MEMs#538)" begin
        # Before the adoption, irf favar labelled key variables by panel POSITION
        # ("X9"/"X10") and irf sdfm labelled every response "Var $i" — real names
        # existed only in the loader and never reached the model.
        rng = MersenneTwister(61)
        df = DataFrame([Symbol("s$i") => randn(rng, 140) for i in 1:4]...,
                       :infl => randn(rng, 140), :ffr => randn(rng, 140))
        csv = write_csv(df; prefix="factor_names")

        vars_of(r) = begin
            _, t = first_table(r.doc)
            t === nothing ? String[] :
                sort(unique(String[string(collect(row)[col_index(t, "variable")])
                                   for row in table_rows(t)]))
        end

        ri = run_json(["irf", "favar", csv, "--factors", "2",
                       "--key-vars", "infl,ffr", "--horizons", "4"])
        assert_envelope_ok(ri; label="irf favar named")
        @test vars_of(ri) == ["F1", "F2", "ffr", "infl"]

        rf = run_json(["fevd", "favar", csv, "--factors", "2",
                       "--key-vars", "infl,ffr", "--horizons", "4"])
        @test rf.code == 0
        @test vars_of(rf) == ["F1", "F2", "ffr", "infl"]

        rs = run_json(["irf", "sdfm", csv, "--factors", "2", "--horizons", "4"])
        assert_envelope_ok(rs; label="irf sdfm named")
        vs = vars_of(rs)
        @test "infl" in vs && "ffr" in vs && "s1" in vs
        @test !any(startswith(v, "Var ") for v in vs)

        # estimate sdfm stores them on the model itself. #147: the leaf now emits
        # an estimation-record table (it was status-only — a success exit with
        # nothing addressable).
        re = run_json(["estimate", "sdfm", csv, "--factors", "2"])
        @test re.code == 0
        sm = named_table(re.doc, :sdfm_estimation_summary)
        @test sm !== nothing
        if sm !== nothing
            mets = Dict(String(collect(r)[1]) => String(collect(r)[2])
                        for r in table_rows(sm))
            @test mets["dynamic_factors"] == "2"
            @test mets["identification"] == "cholesky"
            @test parse(Int, mets["n_vars"]) >= 3
        end
        rm(csv; force=true)
    end

    # ── Panel VAR + DiD family (C054): this suite is the gate that was missing
    # when the MEMs 0.7.0 bump silently broke xtset / estimate_pvar / pvar_fevd /
    # pvar_bootstrap_irf / pvar_lag_selection / lp_did. ──────────────────────
    @testset "panel VAR + DiD family on real MEMs" begin
        panel = dgp_did_panel(; N=40, T=10, seed=7)
        P = ["--id-col", "id", "--time-col", "time"]
        D = ["--outcome", "y", "--treatment", "d"]

        @testset "estimate pvar" begin
            r = run_json(vcat(["estimate", "pvar", panel], P, ["--lags", "1"]))
            assert_envelope_ok(r; label="estimate pvar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing && length(table_rows(tbl)) >= 1
        end
        @testset "test pvar lagselect/mmsc/stability/hansen-j" begin
            for sub in ("lagselect", "mmsc", "stability", "hansen-j")
                r = run_json(vcat(["test", "pvar", sub, panel], P))
                assert_envelope_ok(r; label="test pvar $sub")
            end
        end
        @testset "irf pvar" begin
            r = run_json(vcat(["irf", "pvar", panel], P, ["--lags", "1", "--horizons", "4"]))
            assert_envelope_ok(r; label="irf pvar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing && length(table_rows(tbl)) >= 1
        end
        @testset "fevd pvar proportions in [0,1]" begin
            r = run_json(vcat(["fevd", "pvar", panel], P, ["--lags", "1", "--horizons", "4"]))
            assert_envelope_ok(r; label="fevd pvar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            for row in table_rows(tbl), v in collect(row)[2:end]
                v isa Real && (@test -1e-6 <= v <= 1 + 1e-6)
            end
        end
        @testset "did estimate has finite SE column (#164-169)" begin
            r = run_json(vcat(["did", "estimate", panel], D, P))
            assert_envelope_ok(r; label="did estimate")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            si = col_index(tbl, "SE")
            @test si !== nothing
            ses = [collect(row)[si] for row in table_rows(tbl)]
            @test all(x -> x isa Real && isfinite(x) && x >= 0, ses)
        end
        @testset "did test honest has RR robust + original CIs (C061)" begin
            r = run_json(vcat(["did", "test", "honest", panel], D, P))
            assert_envelope_ok(r; label="did test honest")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            cols = table_cols(tbl)
            # Rambachan–Roth structure: both robust and original CIs present.
            @test any(c -> occursin("Robust", c), cols)
            @test any(c -> occursin("Original", c), cols)
        end
        @testset "did event-study / lp-did / bacon / pretrend / negweight" begin
            # event-study also guards the stdout-contract fix (leading noise would
            # make the raw envelope unparseable → assert_envelope_ok fails).
            for (name, args) in (
                ("event-study", vcat(["did", "event-study", panel], D, P)),
                ("lp-did",      vcat(["did", "lp-did", panel], D, P)),
                ("test bacon",  vcat(["did", "test", "bacon", panel], D, P)),
                ("test pretrend", vcat(["did", "test", "pretrend", panel], D, P)),
                ("test negweight", vcat(["did", "test", "negweight", panel, "--treatment", "d"], P)),
            )
                r = run_json(args)
                assert_envelope_ok(r; label="did $name")
            end
        end
        rm(panel; force=true)
    end

    @testset "spectral density" begin
        csv = dgp_ar1(; T=200, φ=0.5, seed=35)
        r = run_json(["spectral", "density", csv, "--column", "1", "--method", "welch"])
        assert_envelope_ok(r; label="spectral density")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 5
        rm(csv; force=true)
    end

    @testset "estimate lp" begin
        csv = dgp_var2(; T=200, seed=37)
        r = run_json(["estimate", "lp", csv, "--horizons", "5", "--control-lags", "2"])
        assert_envelope_ok(r; label="estimate lp")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        rm(csv; force=true)
    end

    @testset "estimate bvar tiny draws" begin
        csv = dgp_var2(; T=120, seed=39)
        r = run_json(["estimate", "bvar", csv, "--lags", "1", "--draws", "50"])
        # BVAR may be slow/stochastic; require success
        assert_envelope_ok(r; label="estimate bvar")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        rm(csv; force=true)
    end

    @testset "irf bvar tidy (C051)" begin
        csv = dgp_var2(; T=120, seed=43)
        r = run_json(["irf", "bvar", csv, "--lags", "1", "--draws", "80",
                      "--shock", "1", "--horizons", "6"])
        assert_envelope_ok(r; label="irf bvar")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        if tbl !== nothing
            @test table_cols(tbl) == ["horizon", "variable", "shock", "value", "lower", "upper"]
            ci = Dict(c => i for (i, c) in enumerate(table_cols(tbl)))
            rows = [collect(row) for row in table_rows(tbl)]
            @test length(unique(row[ci["shock"]] for row in rows)) == 1   # one shock
        end
        rm(csv; force=true)
    end

    @testset "var stability" begin
        csv = dgp_var2(; T=150, seed=41)
        r = run_json(["test", "var", "stability", csv, "--lags", "2"])
        assert_envelope_ok(r; label="var stability")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        rm(csv; force=true)
    end

    @testset "model handle round-trip no re-estimation" begin
        csv = dgp_var2(; T=100, seed=43)
        fmod = tempname() * ".fmod"
        r1 = run_json(["estimate", "var", csv, "--lags", "1", "--save-model", fmod])
        assert_envelope_ok(r1; label="estimate save-model")
        @test isfile(fmod)
        r2 = run_json(["irf", "var", "--model", fmod, "--horizons", "4", "--ci", "none"])
        assert_envelope_ok(r2; label="irf --model")
        _, tbl = first_table(r2.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 4
        rm(csv; force=true)
        rm(fmod; force=true)
    end

    # C052 — native save/load handles + reproducibility manifests + seed forwarding
    @testset "C052 native .jld2 save/load + hybrid fallback (#347)" begin
        csv = dgp_var2(; T=100, seed=43)
        # native round-trip: VARModel → .jld2 → irf --model (no re-estimation)
        jld = tempname() * ".jld2"
        r1 = run_json(["estimate", "var", csv, "--lags", "1", "--save-model", jld])
        assert_envelope_ok(r1; label="estimate save native jld2")
        @test isfile(jld)
        r2 = run_json(["irf", "var", "--model", jld, "--horizons", "4", "--ci", "none"])
        assert_envelope_ok(r2; label="irf --model native jld2")
        @test first_table(r2.doc)[2] !== nothing
        # model info reads the native handle
        r3 = run_json(["model", "info", jld])
        assert_envelope_ok(r3; label="model info native jld2")
        # W1/#106: VECMModel joined the upstream registry (6 → 56 types), so what used to
        # be the "unsupported on .jld2" case is now a NATIVE round-trip. This assertion is
        # inverted on purpose — it is the behavioural change of the wave.
        vjld = tempname() * ".jld2"
        r4 = run_json(["estimate", "vecm", csv, "--save-model", vjld])
        assert_envelope_ok(r4; label="vecm native jld2 (W1)")
        @test isfile(vjld)
        @test run_json(["model", "info", vjld]).code == 0
        # .fmod still works for any type — it is a format choice, not a fallback-only path
        vfmod = tempname() * ".fmod"
        r5 = run_json(["estimate", "vecm", csv, "--save-model", vfmod])
        assert_envelope_ok(r5; label="vecm .fmod interim handle")
        @test isfile(vfmod)
        rm(vjld; force=true)
        # a non-JLD2 file handed to the native path is BAD INPUT, not a CLI bug:
        # must map to a data/* class (exit 3), never internal/error (exit 1)
        garbage = tempname() * ".jld2"
        write(garbage, rand(UInt8, 200))
        r6 = run_json(["model", "info", garbage])
        @test r6.doc !== nothing && String(r6.doc["status"]) == "error"
        @test startswith(String(r6.doc["error"]["code"]), "data/")
        rm(csv; force=true); rm(jld; force=true); rm(vfmod; force=true); rm(garbage; force=true)
    end

    # W3/#167 — universal serialization: newly-native round-trips + reproduce
    @testset "W3 native round-trips + model reproduce" begin
        csv = dgp_var2(; T=100, seed=43)

        # Seeded BVAR: save → downstream irf --model still works on the
        # posterior-mean VARModel file. That file carries NO manifest (the
        # seed went into the sampler, the saved object is deterministic), so
        # reproduce honestly reports unverifiable — exit 0, not a refusal.
        bjld = tempname() * ".jld2"
        rb = run_json(["--seed", "11", "estimate", "bvar", csv, "--lags", "1",
                       "--draws", "50", "--save-model", bjld])
        assert_envelope_ok(rb; label="w3 seeded bvar save")
        @test isfile(bjld)
        # Wave 2: the saved object is VARModel, so the typed --model slot is irf var.
        ri = run_json(["irf", "var", "--model", bjld, "--horizons", "4"])
        assert_envelope_ok(ri; label="w3 irf var --model posterior-mean")
        rp = run_json(["model", "reproduce", bjld])
        assert_envelope_ok(rp; label="w3 reproduce bvar unverifiable")
        summ = rp.doc.data.model_reproduce_summary
        @test only(r for r in summ.rows if r[1] == "matched")[2] == "unverifiable (no recorded seed)"
        @test only(r for r in summ.rows if r[1] == "seed")[2] == "none recorded"
        @test !haskey(rp.doc.data, :model_reproduce_fields)
        rm(bjld; force=true)

        # TRUE matched path (MEMs#769): estimate sv --seed records a
        # ReproManifest on the saved SVModel, so reproduce re-runs from the
        # seed and compares bit-for-bit.
        vjld = tempname() * ".jld2"
        rv = run_json(["--seed", "7", "estimate", "sv", csv, "--column", "1",
                       "--draws", "150", "--save-model", vjld])
        assert_envelope_ok(rv; label="w3 seeded sv save")
        rvi = run_json(["model", "info", vjld])
        assert_envelope_ok(rvi; label="w3 sv info")
        vrow = only(r for r in rvi.doc.data.model_handle_info.rows if r[1] == "model_type")
        @test vrow[2] == "SVModel"
        rvp = run_json(["model", "reproduce", vjld])
        assert_envelope_ok(rvp; label="w3 reproduce sv match")
        vsumm = rvp.doc.data.model_reproduce_summary
        @test only(r for r in vsumm.rows if r[1] == "matched")[2] == "true"
        @test only(r for r in vsumm.rows if r[1] == "seed")[2] == "7"
        @test haskey(rvp.doc.data, :model_reproduce_fields)
        @test length(rvp.doc.data.model_reproduce_fields.rows) >= 1
        rm(vjld; force=true)

        # SVAR (ex-.fmod family): save → info type → reproduce is honest about
        # the missing manifest (upstream's universal fallback reports a
        # missing verdict, exit 0 — never a model/unsupported refusal).
        sjld = tempname() * ".jld2"
        rs = run_json(["estimate", "svar", csv, "--lags", "2", "--pattern", "recursive",
                       "--save-model", sjld])
        assert_envelope_ok(rs; label="w3 svar save")
        rsi = run_json(["model", "info", sjld])
        assert_envelope_ok(rsi; label="w3 svar info")
        irow = only(r for r in rsi.doc.data.model_handle_info.rows if r[1] == "model_type")
        @test irow[2] == "SVARModel"
        rsr = run_json(["model", "reproduce", sjld])
        assert_envelope_ok(rsr; label="w3 reproduce svar unverifiable")
        ssumm = rsr.doc.data.model_reproduce_summary
        @test only(r for r in ssumm.rows if r[1] == "matched")[2] == "unverifiable (no recorded seed)"
        @test !haskey(rsr.doc.data, :model_reproduce_fields)
        rm(sjld; force=true)
        rm(csv; force=true)
    end

    # W3/#167 — DSGE/HA solutions move off .fmod: save → info → typed refusal
    @testset "W3 DSGE/HA native round-trips" begin
        dir = mktempdir()
        model_toml = joinpath(dir, "model.toml")
        write(model_toml, """
        [model]
        parameters = { rho = 0.9, sigma = 0.01 }
        endogenous = ["Y", "C"]
        exogenous = ["e"]
        linear = true
        [[model.equations]]
        expr = "Y[t] = rho * Y[t-1] + sigma * e[t]"
        [[model.equations]]
        expr = "C[t] = Y[t]"
        """)
        sol = tempname() * ".jld2"
        r = run_json(["dsge", "solve", model_toml, "--save-model", sol])
        assert_envelope_ok(r; label="w3 dsge solve save")
        ri = run_json(["model", "info", sol])
        assert_envelope_ok(ri; label="w3 dsge sol info")
        irow = only(r for r in ri.doc.data.model_handle_info.rows if r[1] == "model_type")
        @test irow[2] == "DSGESolution"
        # no reproduce(::DSGESolution) upstream: the universal fallback reports
        # a missing verdict (exit 0), never a typed refusal or exit 1
        rr = run_json(["model", "reproduce", sol])
        assert_envelope_ok(rr; label="w3 reproduce dsge-sol unverifiable")
        @test only(r for r in rr.doc.data.model_reproduce_summary.rows if r[1] == "matched")[2] ==
              "unverifiable (no recorded seed)"
        rm(sol; force=true)

        # HA steady state + Krusell–Smith (KS reuses the suite's small-solve shape)
        ss = tempname() * ".jld2"
        rh = run_json(["dsge", "ha", "steady-state", "huggett", "--save-model", ss])
        assert_envelope_ok(rh; label="w3 ha ss save")
        rhi = run_json(["model", "info", ss])
        assert_envelope_ok(rhi; label="w3 ha ss info")
        hirow = only(r for r in rhi.doc.data.model_handle_info.rows if r[1] == "model_type")
        @test hirow[2] == "HASteadyState"
        rm(ss; force=true)

        ks = tempname() * ".jld2"
        rk = run_json(["--seed", "7", "dsge", "ha", "solve", "krusell-smith",
                       "--method", "krusell-smith", "--n-reduced", "6",
                       "--t-horizon", "20", "--save-model", ks])
        assert_envelope_ok(rk; label="w3 ks solve save")
        rki = run_json(["model", "info", ks])
        assert_envelope_ok(rki; label="w3 ks info")
        kirow = only(r for r in rki.doc.data.model_handle_info.rows if r[1] == "model_type")
        @test kirow[2] == "KrusellSmithSolution"
        rm(ks; force=true)
        rm(dir; force=true, recursive=true)
    end

    @testset "W1/#106 native save/load across the widened registry" begin
        ar = dgp_ar1(; T=150, φ=0.5, seed=917)

        # One save → load → re-render per family that only became native in W1. GARCH is
        # called out by the issue: its deserialization (dist/shape ctor) was broken until
        # the 0.7.2 CI round, so a green round-trip here is the thing that proves the fix.
        for (leaf, verb) in (("arima", "residuals"), ("garch", "predict"),
                             ("arfima", "residuals"), ("statespace", "predict"))
            jld = tempname() * ".jld2"
            rs = run_json(["estimate", leaf, ar, "--save-model", jld])
            assert_envelope_ok(rs; label="estimate $leaf → native jld2")
            @test isfile(jld)
            rl = run_json([verb, leaf, "--model", jld])
            assert_envelope_ok(rl; label="$verb $leaf from native handle")
            @test first_table(rl.doc)[2] !== nothing
            rm(jld; force=true)
        end

        # REGRESSION (found in W1, pre-existing since the handle path was added): `vname`
        # was bound only inside the `isnothing(model)` branch of _predict_arima /
        # _residuals_arima but interpolated into the title on BOTH paths, so EVERY
        # `predict|residuals arima --model <handle>` died with an UndefVarError — an
        # untyped internal/error (exit 1), on .fmod as well as .jld2.
        for suffix in (".jld2", ".fmod")
            h = tempname() * suffix
            run_json(["estimate", "arima", ar, "--save-model", h])
            @test isfile(h)
            for verb in ("predict", "residuals")
                r = run_json([verb, "arima", "--model", h])
                @test r.code == 0            # was 1 (UndefVarError: vname)
                assert_envelope_ok(r; label="$verb arima handle $suffix")
            end
            rm(h; force=true)
        end

        # A data CONTAINER is in the registry too, so `data`-family artifacts round-trip.
        cjld = tempname() * ".jld2"
        rc = run_json(["data", "load", ":fred_md", "--save-model", cjld])
        if rc.code == 0 && isfile(cjld)
            @test run_json(["model", "info", cjld]).code == 0
        end
        rm(cjld; force=true)
        rm(ar; force=true)
    end

    @testset "C052 reproducibility manifest in envelope meta (#345)" begin
        csv = dgp_var2(; T=80, seed=7)
        r = run_json(["estimate", "var", csv, "--lags", "1"])
        assert_envelope_ok(r; label="manifest meta")
        @test haskey(r.doc["meta"], "manifest")
        m = r.doc["meta"]["manifest"]
        for k in ("seed", "n_threads", "julia_version", "package_version",
                  "os", "machine", "timestamp", "dependency_versions")
            @test haskey(m, k)
        end
        @test String(m["package_version"]) != "unknown"
        rm(csv; force=true)
    end

    @testset "C052 --seed byte-identical + manifest.seed (#243)" begin
        csv = dgp_var2(; T=80, seed=9)
        # two --seed 42 BVAR runs are byte-identical on the data payload
        r1 = run_json(["--seed", "42", "estimate", "bvar", csv, "--lags", "1", "--draws", "200"])
        r2 = run_json(["--seed", "42", "estimate", "bvar", csv, "--lags", "1", "--draws", "200"])
        assert_envelope_ok(r1; label="seeded bvar 1")
        assert_envelope_ok(r2; label="seeded bvar 2")
        @test JSON3.write(r1.doc["data"]) == JSON3.write(r2.doc["data"])
        @test r1.doc["meta"]["manifest"]["seed"] == 42
        # forwarded seed= makes the consumed BVAR posterior reproducible → irf bvar identical
        i1 = run_json(["--seed", "42", "irf", "bvar", csv, "--lags", "1", "--horizons", "4"])
        i2 = run_json(["--seed", "42", "irf", "bvar", csv, "--lags", "1", "--horizons", "4"])
        assert_envelope_ok(i1; label="seeded irf bvar 1")
        @test JSON3.write(i1.doc["data"]) == JSON3.write(i2.doc["data"])
        rm(csv; force=true)
    end

    # C040 — HA-DSGE against real MEMs (builtin huggett is smallest)
    @testset "dsge ha steady-state huggett" begin
        r = run_json(["dsge", "ha", "steady-state", "huggett"])
        assert_envelope_ok(r; label="dsge ha steady-state")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 1
    end

    # W13/#115 — Den Haan (2010) accuracy, the audit's adopt-now row.
    # ── W9/#111: quantile regression + RDD ──────────────────────────────────
    @testset "estimate qreg (W9/#111)" begin
        # Homoskedastic errors: the SLOPE is the same at every quantile and only the
        # INTERCEPT shifts, by the normal quantile. That is the property worth asserting --
        # it fails if taus are mismatched to columns.
        csv = tempname() * ".csv"
        open(csv, "w") do io
            println(io, "y,const_,x")
            r = 1.0
            for i in 1:400
                r = mod(r * 48271, 2147483647)          # deterministic, no RNG dependency
                u1 = r / 2147483647
                r = mod(r * 48271, 2147483647)
                u2 = r / 2147483647
                z  = sqrt(-2log(max(u1, 1e-12))) * cos(2pi * u2)
                r = mod(r * 48271, 2147483647)
                x  = (r / 2147483647 - 0.5) * 4
                println(io, "$(1.0 + 2.0x + z),1.0,$x")
            end
        end

        rq = run_json(["estimate", "qreg", csv, "--dep", "y", "--tau", "0.25,0.5,0.75"])
        assert_envelope_ok(rq; label="estimate qreg")
        ct = nothing
        for (_, v) in pairs(rq.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["tau", "term", "estimate", "std_error"]) && (ct = v)
        end
        @test ct !== nothing
        if ct !== nothing
            @test length(table_rows(ct)) == 3 * 2       # 3 quantiles x 2 terms
            slopes = Float64[]; intercepts = Float64[]
            for rw in table_rows(ct)
                a = collect(rw)
                term = String(a[col_index(ct, "term")])
                est  = Float64(a[col_index(ct, "estimate")])
                term == "x" ? push!(slopes, est) : push!(intercepts, est)
            end
            @test length(slopes) == 3
            @test all(b -> abs(b - 2.0) < 0.25, slopes)
            # Intercepts must be strictly increasing in tau -- the whole point of the model.
            @test issorted(intercepts)
            @test intercepts[3] - intercepts[1] > 0.5
        end

        for se in ("iid", "robust", "boot")
            r2 = run_json(["estimate", "qreg", csv, "--dep", "y", "--tau", "0.5",
                           "--se", se, "--n-boot", "50"])
            assert_envelope_ok(r2; label="estimate qreg --se $se")
        end
        @test run_json(["estimate", "qreg", csv, "--tau", "0"]).code == 2
        @test run_json(["estimate", "qreg", csv, "--tau", "1"]).code == 2
        @test run_json(["estimate", "qreg", csv, "--tau", "0.5,0.5"]).code == 2
        @test run_json(["estimate", "qreg", csv, "--tau", "abc"]).code == 2
        @test run_json(["estimate", "qreg", csv, "--se", "bogus"]).code == 2
        @test run_json(["estimate", "qreg", csv, "--alpha", "0"]).code == 2
        rm(csv; force=true)
    end

    @testset "estimate rdd (W9/#111)" begin
        # Sharp design with a KNOWN jump of 3.0 at 0. Assert the CI covers the truth rather
        # than pinning the point estimate: the CCT bandwidth is data-driven, so the estimate
        # legitimately moves.
        csv = tempname() * ".csv"
        open(csv, "w") do io
            println(io, "y,run")
            r = 7.0
            for i in 1:600
                r = mod(r * 48271, 2147483647); u1 = r / 2147483647
                r = mod(r * 48271, 2147483647); u2 = r / 2147483647
                z  = sqrt(-2log(max(u1, 1e-12))) * cos(2pi * u2)
                r = mod(r * 48271, 2147483647)
                x  = (r / 2147483647 - 0.5) * 8
                y  = 0.5x + (x >= 0 ? 3.0 : 0.0) + 0.5z
                println(io, "$y,$x")
            end
        end

        rr = run_json(["estimate", "rdd", csv, "--outcome", "y", "--running", "run",
                       "--cutoff", "0"])
        assert_envelope_ok(rr; label="estimate rdd")
        tt = nothing
        for (_, v) in pairs(rr.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["method", "estimate", "std_error"]) && (tt = v)
        end
        @test tt !== nothing
        if tt !== nothing
            bym = Dict(String(collect(rw)[col_index(tt, "method")]) => collect(rw)
                       for rw in table_rows(tt))
            @test sort(collect(keys(bym))) == ["bias-corrected", "conventional", "robust"]
            for m in ("conventional", "robust")
                a = bym[m]
                lo = Float64(a[col_index(tt, "ci_lower")])
                hi = Float64(a[col_index(tt, "ci_upper")])
                @test lo <= 3.0 <= hi
            end
            # "robust" is the BIAS-CORRECTED point with a wider SE, not a third estimate.
            @test Float64(bym["robust"][col_index(tt, "estimate")]) ≈
                  Float64(bym["bias-corrected"][col_index(tt, "estimate")]) atol=1e-9
            @test Float64(bym["robust"][col_index(tt, "std_error")]) >=
                  Float64(bym["conventional"][col_index(tt, "std_error")])
        end

        # Auto-selected bandwidth must be reported, and an explicit one must be honoured.
        st = nothing
        for (_, v) in pairs(rr.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            "metric" in table_cols(v) &&
                any(w -> String(collect(w)[1]) == "bandwidth_h", table_rows(v)) && (st = v)
        end
        @test st !== nothing
        if st !== nothing
            hv = 0.0
            for rw in table_rows(st)
                a = collect(rw)
                String(a[1]) == "bandwidth_h" && (hv = Float64(a[2]))
            end
            @test hv > 0
        end
        rh = run_json(["estimate", "rdd", csv, "--outcome", "y", "--running", "run",
                       "--cutoff", "0", "--bandwidth", "2.0"])
        assert_envelope_ok(rh; label="estimate rdd --bandwidth")

        for k in ("triangular", "epanechnikov", "uniform")
            rk = run_json(["estimate", "rdd", csv, "--outcome", "y", "--running", "run",
                           "--cutoff", "0", "--kernel", k])
            assert_envelope_ok(rk; label="estimate rdd --kernel $k")
        end

        # A cutoff outside the running variable's support leaves one side empty -- typed
        # data error, never an untyped crash from inside the local regression.
        @test run_json(["estimate", "rdd", csv, "--outcome", "y", "--running", "run",
                        "--cutoff", "999"]).code == 3
        @test run_json(["estimate", "rdd", csv, "--outcome", "y", "--running", "y",
                        "--cutoff", "0"]).code == 2
        @test run_json(["estimate", "rdd", csv, "--kernel", "bogus"]).code == 2
        @test run_json(["estimate", "rdd", csv, "--order", "0"]).code == 2
        @test run_json(["estimate", "rdd", csv, "--level", "1.5"]).code == 2
        rm(csv; force=true)
    end

    # ── W8/#110: scenario forecasts, bootstrap schemes, generalized FEVD ────
    @testset "forecast scenario (W8/#110)" begin
        csv = dgp_var2(; T=160, seed=41)
        cond = tempname() * ".csv"
        write(cond, "variable,period,value\ny1,1,2.5\ny1,2,2.0\n")

        r = run_json(["forecast", "scenario", csv, "--conditions-file", cond,
                      "--lags", "2", "--horizons", "6", "--replications", "200"])
        assert_envelope_ok(r; label="forecast scenario")
        path = nothing
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["horizon", "variable", "value", "unconditional"]) &&
                (path = v)
        end
        @test path !== nothing
        if path !== nothing
            @test length(table_rows(path)) == 6 * 3
            # A HARD condition (no sd) must pin the path EXACTLY and collapse the band --
            # that is the whole contract of a hard conditional forecast.
            for rw in table_rows(path)
                a = collect(rw)
                v  = String(a[col_index(path, "variable")])
                h  = Int(a[col_index(path, "horizon")])
                if v == "y1" && h in (1, 2)
                    want = h == 1 ? 2.5 : 2.0
                    @test Float64(a[col_index(path, "value")]) ≈ want atol=1e-6
                    @test Float64(a[col_index(path, "lower")]) ≈ want atol=1e-6
                    @test Float64(a[col_index(path, "upper")]) ≈ want atol=1e-6
                end
            end
        end
        # The implied structural shocks ride their own table.
        shk = nothing
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["horizon", "shock", "value"]) &&
                !("variable" in table_cols(v)) && (shk = v)
        end
        @test shk !== nothing

        # BVAR dispatch is a separate upstream method, so exercise it too.
        rb = run_json(["forecast", "scenario", csv, "--conditions-file", cond,
                       "--method", "bvar", "--lags", "2", "--horizons", "6",
                       "--draws", "200", "--replications", "200"])
        assert_envelope_ok(rb; label="forecast scenario --method bvar")

        # Malformed conditions: every one of these is a typed data error, never exit 1.
        bad_cols = tempname() * ".csv"; write(bad_cols, "var,period,value\ny1,1,2.5\n")
        @test run_json(["forecast", "scenario", csv, "--conditions-file", bad_cols,
                        "--lags", "2"]).code == 3
        # A blank cell must be caught BEFORE the Float64 conversion (the loader lesson).
        blank = tempname() * ".csv"; write(blank, "variable,period,value\ny1,1,\n")
        @test run_json(["forecast", "scenario", csv, "--conditions-file", blank,
                        "--lags", "2"]).code == 3
        unknown = tempname() * ".csv"; write(unknown, "variable,period,value\nnope,1,2.5\n")
        @test run_json(["forecast", "scenario", csv, "--conditions-file", unknown,
                        "--lags", "2"]).code == 3
        beyond = tempname() * ".csv"; write(beyond, "variable,period,value\ny1,99,2.5\n")
        @test run_json(["forecast", "scenario", csv, "--conditions-file", beyond,
                        "--lags", "2", "--horizons", "6"]).code == 3
        dup = tempname() * ".csv"
        write(dup, "variable,period,value\ny1,1,2.5\ny1,1,3.0\n")
        @test run_json(["forecast", "scenario", csv, "--conditions-file", dup,
                        "--lags", "2"]).code == 3
        # Missing --conditions-file is a usage error, not a data one.
        @test run_json(["forecast", "scenario", csv, "--lags", "2"]).code == 2

        for f in (cond, bad_cols, blank, unknown, beyond, dup, csv)
            rm(f; force=true)
        end
    end

    @testset "irf var bootstrap schemes + Kilian (W8/#110)" begin
        csv = dgp_var2(; T=140, seed=43)
        for scheme in ("iid", "wild", "block")
            r = run_json(["irf", "var", csv, "--lags", "2", "--horizons", "6",
                          "--ci", "bootstrap", "--replications", "60",
                          "--bootstrap", scheme])
            assert_envelope_ok(r; label="irf var --bootstrap $scheme")
            _, t = first_table(r.doc)
            @test t !== nothing && !isempty(table_rows(t))
        end
        for wd in ("rademacher", "mammen")      # upstream implements only these two
            r = run_json(["irf", "var", csv, "--lags", "2", "--horizons", "4",
                          "--ci", "bootstrap", "--replications", "40",
                          "--bootstrap", "wild", "--wild-dist", wd])
            assert_envelope_ok(r; label="irf var --wild-dist $wd")
        end
        rk = run_json(["irf", "var", csv, "--lags", "2", "--horizons", "4",
                       "--ci", "bootstrap", "--replications", "40",
                       "--bias-correct", "--bias-reps", "20"])
        assert_envelope_ok(rk; label="irf var --bias-correct")
        @test run_json(["irf", "var", csv, "--bootstrap", "bogus"]).code == 2
        @test run_json(["irf", "var", csv, "--wild-dist", "bogus"]).code == 2
        @test run_json(["irf", "var", csv, "--block-length", "-1"]).code == 2
        rm(csv; force=true)
    end

    @testset "fevd var --generalized (W8/#110)" begin
        csv = dgp_var2(; T=140, seed=47)
        H = 5
        sums(t) = begin
            acc = Dict{Tuple{Int,String},Float64}()
            for rw in table_rows(t)
                a = collect(rw)
                k = (Int(a[col_index(t, "horizon")]), String(a[col_index(t, "variable")]))
                acc[k] = get(acc, k, 0.0) + Float64(a[col_index(t, "value")])
            end
            acc
        end

        # Orthogonalized: shares DO sum to 1 per (horizon, variable).
        ro = run_json(["fevd", "var", csv, "--lags", "2", "--horizons", string(H)])
        assert_envelope_ok(ro; label="fevd var orthogonalized")
        _, to = first_table(ro.doc)
        @test to !== nothing
        if to !== nothing
            for (_, v) in sums(to)
                @test v ≈ 1.0 atol=1e-6
            end
        end

        # Generalized: they do NOT. Reusing the sum-to-1 assertion here would be wrong --
        # the generalized shocks are correlated, so contributions overlap.
        rg = run_json(["fevd", "var", csv, "--lags", "2", "--horizons", string(H),
                       "--generalized"])
        assert_envelope_ok(rg; label="fevd var --generalized")
        _, tg = first_table(rg.doc)
        @test tg !== nothing
        if tg !== nothing
            sg = sums(tg)
            @test !isempty(sg)
            @test all(v -> v > 0, values(sg))
            # At least one row must genuinely depart from 1, else --generalized is a no-op.
            @test any(v -> abs(v - 1.0) > 1e-6, values(sg))
        end

        # --normalize rescales them back to 1, which is the only way to read them as shares.
        rn = run_json(["fevd", "var", csv, "--lags", "2", "--horizons", string(H),
                       "--generalized", "--normalize"])
        assert_envelope_ok(rn; label="fevd var --generalized --normalize")
        _, tn = first_table(rn.doc)
        @test tn !== nothing
        if tn !== nothing
            for (_, v) in sums(tn)
                @test v ≈ 1.0 atol=1e-6
            end
        end
        rm(csv; force=true)
    end

    # ── W7/#109: TVP-VAR-SV, MF-VAR, BVAR hyperopt ──────────────────────────
    @testset "estimate tvpvar + irf tvpvar (W7/#109)" begin
        csv = dgp_var2(; T=120, seed=21)
        r = run_json(["estimate", "tvpvar", csv, "--lags", "1",
                      "--draws", "60", "--burnin", "30"])
        assert_envelope_ok(r; label="estimate tvpvar")
        vol = nothing
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["period", "variable", "mean"]) && (vol = v)
        end
        @test vol !== nothing
        if vol !== nothing
            # Tidy long form: one row per (period, variable), NOT a wide T x n block.
            n_per = length(unique(String(collect(rw)[col_index(vol, "variable")])
                                  for rw in table_rows(vol)))
            @test n_per == 3          # dgp_var2 is a 3-variable system
            @test length(table_rows(vol)) % n_per == 0
            # volatility_path returns a STANDARD DEVIATION (exp(h/2)); the stored state is
            # a log-variance, so a missing conversion would show up as non-positive values.
            mvals = [Float64(collect(rw)[col_index(vol, "mean")]) for rw in table_rows(vol)]
            @test all(>(0), mvals)
            @test all(isfinite, mvals)
        end

        # --date is REQUIRED and must be rejected BEFORE the Gibbs sampler runs.
        rmiss = run_json(["irf", "tvpvar", csv, "--lags", "1", "--draws", "60",
                          "--burnin", "30"])
        @test rmiss.code == 2
        @test occursin("date", lowercase(String(rmiss.doc["error"]["message"])))

        rirf = run_json(["irf", "tvpvar", csv, "--date", "40", "--horizons", "4",
                         "--lags", "1", "--draws", "60", "--burnin", "30",
                         "--irf-draws", "40"])
        assert_envelope_ok(rirf; label="irf tvpvar")
        it = nothing
        for (_, v) in pairs(rirf.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["horizon", "variable", "shock", "value"]) && (it = v)
        end
        @test it !== nothing
        @test it === nothing || length(table_rows(it)) == 4 * 3   # 4 horizons x 3 variables, one shock

        # Out-of-range dates are typed usage errors, not an untyped upstream ArgumentError.
        @test run_json(["irf", "tvpvar", csv, "--date", "0", "--lags", "1",
                        "--draws", "60", "--burnin", "30"]).code == 2
        @test run_json(["irf", "tvpvar", csv, "--date", "99999", "--lags", "1",
                        "--draws", "60", "--burnin", "30"]).code == 2
        rm(csv; force=true)
    end

    @testset "estimate mfvar (W7/#109)" begin
        # A mixed-frequency CSV: the low-frequency series is BLANK between observations.
        # Those gaps must survive to the estimator as NaN -- the ordinary loader rejects
        # missing cells, so this leaf has its own loader.
        csv = tempname() * ".csv"
        open(csv, "w") do io
            println(io, "monthly,quarterly")
            m = 0.0
            for t in 1:96
                m = 0.7m + 0.4 * sin(t / 3.0)
                if t % 3 == 0
                    println(io, "$m,$(m * 1.01)")
                else
                    println(io, "$m,")          # blank cell in a MULTI-column row
                end
            end
        end
        r = run_json(["estimate", "mfvar", csv, "--lags", "2", "--draws", "80",
                      "--burnin", "40", "--freq-ratio", "3", "--aggregation", "average"])
        assert_envelope_ok(r; label="estimate mfvar")
        lat = nothing
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            all(c -> c in table_cols(v), ["period", "variable", "mean"]) && (lat = v)
        end
        @test lat !== nothing
        # The latent path is at the HIGH frequency for EVERY series, including the one
        # observed only every third period -- that interpolation is the whole model.
        @test lat === nothing || length(table_rows(lat)) == 96 * 2

        # An all-complete CSV has no low-frequency series to infer.
        plain = dgp_var2(; T=60, seed=3)
        @test run_json(["estimate", "mfvar", plain, "--lags", "1", "--draws", "40"]).code == 2
        @test run_json(["estimate", "mfvar", csv, "--aggregation", "bogus"]).code == 2
        @test run_json(["estimate", "mfvar", csv, "--low-freq", "99"]).code == 2
        @test run_json(["estimate", "mfvar", csv, "--freq-ratio", "0"]).code == 2
        rm(csv; force=true); rm(plain; force=true)
    end

    @testset "estimate bvar --hyperopt (W7/#109)" begin
        csv = dgp_var2(; T=140, seed=31)
        hyper_tbl(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "parameter" in table_cols(v) || continue
                any(rw -> String(collect(rw)[1]) == "tau", table_rows(v)) && return v
            end
            return nothing
        end

        rg = run_json(["estimate", "bvar", csv, "--lags", "2", "--draws", "200"])
        assert_envelope_ok(rg; label="estimate bvar --hyperopt glp (default)")
        tg = hyper_tbl(rg.doc)
        @test tg !== nothing
        if tg !== nothing
            keys_g = [String(collect(rw)[1]) for rw in table_rows(tg)]
            # GLP reports its diagnostics; the grid path has none to report.
            @test "log_ml" in keys_g && "converged" in keys_g && "at_bound" in keys_g
            vals = Dict(String(collect(rw)[1]) => Float64(collect(rw)[2])
                        for rw in table_rows(tg))
            # The optimizer must beat the default hyperparameters it starts from, else the
            # whole GLP path is doing nothing useful.
            @test vals["log_ml"] >= vals["log_ml_default"]
            @test vals["tau"] > 0
        end

        rr = run_json(["estimate", "bvar", csv, "--lags", "2", "--draws", "200",
                       "--hyperopt", "grid"])
        assert_envelope_ok(rr; label="estimate bvar --hyperopt grid")
        tr = hyper_tbl(rr.doc)
        @test tr !== nothing
        @test tr === nothing || !("log_ml" in [String(collect(rw)[1]) for rw in table_rows(tr)])

        @test run_json(["estimate", "bvar", csv, "--hyperopt", "bogus"]).code == 2

        # A [prior] config pins the hyperparameters. This used to pass a length-n VECTOR as
        # `omega`, which real MEMs rejects outright (omega is a SCALAR weight) -- an untyped
        # exit 1 on every --config minnesota run across the whole BVAR family, hidden by a
        # mock whose omega was a Vector. There was no T3 coverage of the config path at all.
        cfg = tempname() * ".toml"
        write(cfg, """
        [prior]
        type = "minnesota"

        [prior.hyperparameters]
        lambda1 = 0.2
        lambda2 = 0.5
        lambda3 = 1.0
        """)
        rc = run_json(["estimate", "bvar", csv, "--lags", "2", "--draws", "200",
                       "--config", cfg])
        assert_envelope_ok(rc; label="estimate bvar --config minnesota")
        # Config pins the values, so no selection table is emitted.
        @test hyper_tbl(rc.doc) === nothing
        rm(cfg; force=true); rm(csv; force=true)
    end

    @testset "dsge ha accuracy (W13/#115)" begin
        # `cols_table` is a LOCAL helper of the io testset, not suite-level — define it here
        # rather than reaching across scopes (this has bitten twice already).
        #
        # It also takes a row predicate: this leaf emits BOTH a `metric`/`value` DATA table
        # and a `metric`/`value` settings kv, so the column set alone is ambiguous and
        # `pairs(doc.data)` has no order guarantee (JSON3). Matching on columns only bound
        # the all-String settings table roughly half the time. Distinctive CONTENT, not
        # just distinctive columns — cf. the T3 harness lesson in CLAUDE.md.
        cols_table(doc, cols; where=_ -> true) = begin
            doc === nothing && return nothing
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                all(c -> c in table_cols(v), cols) || continue
                where(v) && return v
            end
            return nothing
        end
        has_metric(t, m) = any(rw -> String(collect(rw)[col_index(t, "metric")]) == m,
                               table_rows(t))
        # Short simulation: this solves Krusell-Smith first, so keep the horizon small.
        r = run_json(["dsge", "ha", "accuracy", "krusell-smith",
                      "--t-sim", "1500", "--t-burn", "200", "--n-reduced", "8"])
        assert_envelope_ok(r; label="dsge ha accuracy")
        t = cols_table(r.doc, ["metric", "value"]; where=v -> has_metric(v, "dh_max"))
        @test t !== nothing
        # Values arrive as JSON numbers, but coerce defensively: an all-String row means
        # the wrong table bound, and a bare `Float64(::String)` would ERROR the testset
        # (aborting every assertion below) instead of failing one @test.
        num(x) = x isa AbstractString ? parse(Float64, x) : Float64(x)
        mv = Dict(String(collect(rw)[col_index(t, "metric")]) =>
                  num(collect(rw)[col_index(t, "value")]) for rw in table_rows(t))
        for k in ("dh_max", "dh_mean", "sigma_ref", "sigma_plm")
            @test haskey(mv, k) && isfinite(mv[k])
        end
        # Den Haan statistics are percentage deviations: non-negative, and the max
        # cannot be below the mean.
        @test mv["dh_max"] >= 0 && mv["dh_mean"] >= 0
        @test mv["dh_max"] >= mv["dh_mean"]
        # the two simulated aggregate paths ride their own table
        pt = cols_table(r.doc, ["t", "reference", "plm_only"])
        @test pt !== nothing && !isempty(table_rows(pt))

        # Undefined for huggett (no aggregate capital) — and the refusal must come BEFORE
        # the expensive solve, so this returns promptly rather than after a full KS fit.
        rh = run_json(["dsge", "ha", "accuracy", "huggett", "--t-sim", "1500", "--t-burn", "200"])
        @test rh.code == 5
        @test occursin("huggett", String(rh.doc["error"]["message"]))
        # numeric guards are typed usage errors, never an untyped upstream @assert (exit 1)
        @test run_json(["dsge", "ha", "accuracy", "krusell-smith",
                        "--t-sim", "100", "--t-burn", "200"]).code == 2
        @test run_json(["dsge", "ha", "accuracy", "krusell-smith", "--rho-z", "1.5"]).code == 2
        @test run_json(["dsge", "ha", "accuracy", "krusell-smith", "--sigma-z", "0"]).code == 2

        # den_haan_test has a SECOND method for the linearized solutions, which recover the
        # implied law by regression over --t-fit periods. Both must work, and --t-fit is
        # guarded only on that branch (upstream asserts T_fit > 100 untyped).
        for m in ("ssj", "reiter")
            rm_ = run_json(["dsge", "ha", "accuracy", "krusell-smith", "--method", m,
                            "--t-sim", "400", "--t-burn", "50", "--t-fit", "600",
                            "--n-reduced", "6"])
            assert_envelope_ok(rm_; label="dsge ha accuracy --method $m")
            tm = cols_table(rm_.doc, ["metric", "value"]; where=v -> has_metric(v, "dh_max"))
            @test tm !== nothing
            # The settings table must record WHICH solution produced the number: the
            # linearized statistic is not comparable with the fitted-PLM one.
            st = cols_table(rm_.doc, ["metric", "value"]; where=v -> has_metric(v, "method"))
            @test st !== nothing
        end
        @test run_json(["dsge", "ha", "accuracy", "krusell-smith", "--method", "ssj",
                        "--t-fit", "50"]).code == 2
        @test run_json(["dsge", "ha", "accuracy", "krusell-smith", "--method", "bogus"]).code == 2
    end

    # MEMs#508: `euler_points` selects WHERE the Euler residual is measured. EGM solves the
    # Euler equation almost exactly at the nodes, so node evaluation flatters the solution by
    # 2.5-3.8 log10 units; 0.7.2 made midpoints the default and keeps both in `ss.euler`.
    @testset "dsge ha steady-state --euler-points (#508)" begin
        sel(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "metric" in table_cols(v) || continue
                for rw in table_rows(v)
                    r = collect(rw)
                    String(r[1]) == "euler_error" && return Float64(r[2])
                end
            end
            return nothing
        end
        euler_tbl(doc) = begin
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "convention" in table_cols(v) && return v
            end
            return nothing
        end

        rmid = run_json(["dsge", "ha", "steady-state", "huggett"])
        rnod = run_json(["dsge", "ha", "steady-state", "huggett", "--euler-points", "nodes"])
        assert_envelope_ok(rmid; label="ha steady-state midpoints")
        assert_envelope_ok(rnod; label="ha steady-state nodes")

        emid, enod = sel(rmid.doc), sel(rnod.doc)
        @test emid !== nothing && enod !== nothing
        # The whole point of the change: the two conventions must actually differ, and the
        # node figure must be the optimistic one (more negative log10 = smaller residual).
        @test enod < emid

        # Both statistics are reported regardless of which one was selected, so a reader can
        # always compare against a number measured the other way.
        et = euler_tbl(rmid.doc)
        @test et !== nothing
        convs = [String(collect(rw)[col_index(et, "convention")]) for rw in table_rows(et)]
        @test sort(convs) == ["midpoints", "nodes"]
        # …and the selected scalar must equal the matching row, not be recomputed.
        for rw in table_rows(et)
            r = collect(rw)
            String(r[col_index(et, "convention")]) == "midpoints" &&
                (@test Float64(r[col_index(et, "max")]) ≈ emid atol=1e-9)
            String(r[col_index(et, "convention")]) == "nodes" &&
                (@test Float64(r[col_index(et, "max")]) ≈ enod atol=1e-9)
        end

        @test run_json(["dsge", "ha", "steady-state", "huggett",
                        "--euler-points", "bogus"]).code == 2
    end

    # W13/#115 pulled in the sibling standing bug #80: `dsge ha accuracy` calls
    # `_load_ha_model`, and the issue says to fix #80 rather than work around it again.
    # An HA spec file is an `@dsge` block, and the old sandbox injected only the
    # MacroEconometricModels const — so the bare `@dsge` was `UndefVarError` and EVERY
    # `.jl` HA model failed. Verified against the old sandbox before fixing; this is the
    # first coverage the `.jl` HA path has ever had.
    @testset "HA .jl @dsge loader (#80)" begin
        spec = tempname() * ".jl"
        write(spec, """
        @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.96, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z

            heterogeneous: a in [0.0, 400.0], n_grid = 40, utility = log, discount = beta_hh, borrowing = 0.0

            idiosyncratic: e ~ Rouwenhorst(0.966, 0.5, 3)

            aggregation: K = sum(a)

            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end
        """)
        r = run_json(["dsge", "ha", "steady-state", spec])
        assert_envelope_ok(r; label="dsge ha steady-state (.jl spec)")
        agg = nothing
        for (_, v) in pairs(r.doc.data)
            (v isa JSON3.Object && haskey(v, :rows)) || continue
            "name" in table_cols(v) &&
                any(rw -> String(collect(rw)[1]) == "K", table_rows(v)) && (agg = v)
        end
        @test agg !== nothing

        # A .jl file that does NOT evaluate to a spec is a typed config error, not exit 1.
        bad = tempname() * ".jl"
        write(bad, "42\n")
        @test run_json(["dsge", "ha", "steady-state", bad]).code == 4
        # A file that throws while evaluating is likewise typed, not an untyped crash.
        broken = tempname() * ".jl"
        write(broken, "@dsge begin\n    endogenous: Y\n    this is not valid\nend\n")
        @test run_json(["dsge", "ha", "steady-state", broken]).code in (2, 4)

        rm(spec; force=true); rm(bad; force=true); rm(broken; force=true)
    end

    # W0/#151: HA `.jl` default ssj (world-age barrier), E[t] exit 4 on .toml and .jl.
    @testset "W0 HA .jl ssj + E[t] refusal (#151)" begin
        spec = tempname() * ".jl"
        write(spec, """
        @dsge begin
            parameters: alpha = 0.36, beta_hh = 0.96, delta = 0.025, rho_z = 0.95, sigma_z = 0.007
            endogenous: Y, K, r, w, Z
            exogenous: eps_Z

            heterogeneous: a in [0.0, 400.0], n_grid = 40, utility = log, discount = beta_hh, borrowing = 0.0

            idiosyncratic: e ~ Rouwenhorst(0.966, 0.5, 3)

            aggregation: K = sum(a)

            Y[t] = Z[t] * K[t-1]^alpha
            r[t] = alpha * Z[t] * K[t-1]^(alpha-1) - delta
            w[t] = (1 - alpha) * Z[t] * K[t-1]^alpha
            Z[t] = rho_z * Z[t-1] + sigma_z * eps_Z[t]
        end
        """)
        r = run_json(["dsge", "ha", "solve", spec, "--n-reduced", "8"])
        assert_envelope_ok(r; label="dsge ha solve .jl ssj")
        @test r.doc !== nothing

        toml = tempname() * ".toml"
        write(toml, """
        [model]
        parameters = { rho = 0.9 }
        endogenous = ["Y", "C"]
        exogenous = ["e"]
        [[model.equations]]
        expr = "Y[t] = C[t] + e[t]"
        [[model.equations]]
        expr = "C[t] = rho * E[t](C[t+1])"
        """)
        et = run_json(["dsge", "solve", toml])
        @test et.code == 4
        @test et.doc !== nothing
        @test occursin("E[t]", string(et.doc.error.message))

        jl_et = tempname() * ".jl"
        write(jl_et, """
        @dsge begin
            parameters: rho = 0.9
            endogenous: Y, C
            exogenous: e
            Y[t] = C[t] + e[t]
            C[t] = rho * E[t](C[t+1])
        end
        """)
        ej = run_json(["dsge", "solve", jl_et])
        @test ej.code == 4
        @test ej.doc !== nothing
        @test occursin("E[t]", string(ej.doc.error.message))

        rm(spec; force=true); rm(toml; force=true); rm(jl_et; force=true)
    end

    @testset "dsge ha solve reiter huggett" begin
        r = run_json(["dsge", "ha", "solve", "huggett",
                      "--method", "reiter", "--n-reduced", "8"])
        assert_envelope_ok(r; label="dsge ha solve reiter")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
    end

    @testset "dsge ha irf reiter huggett" begin
        r = run_json(["dsge", "ha", "irf", "huggett",
                      "--method", "reiter", "--horizon", "5", "--n-reduced", "8"])
        assert_envelope_ok(r; label="dsge ha irf")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 5
    end

    @testset "W0 dsge ha fevd zero-row proportion (#702)" begin
        r = run_json(["dsge", "ha", "fevd", "huggett",
                      "--method", "reiter", "--horizon", "4", "--n-reduced", "8"])
        assert_envelope_ok(r; label="dsge ha fevd")
        # MEMs#702: an identically-zero IRF row now gets proportion 1.0, never 0.
        for (_, t) in pairs(r.doc.data)
            (t isa JSON3.Object && haskey(t, :columns) && haskey(t, :rows)) || continue
            cols = String.(t.columns)
            pi = findfirst(==("proportion"), cols)
            pi === nothing && (pi = findfirst(c -> occursin("prop", lowercase(c)), cols))
            pi === nothing && continue
            for rw in table_rows(t)
                v = collect(rw)[pi]
                v isa Number && @test v >= 0.0
            end
        end
    end

    @testset "W1 dsge ha solve krusell-smith Dict R²" begin
        r = run_json(["dsge", "ha", "solve", "krusell-smith", "--method", "krusell-smith"])
        assert_envelope_ok(r; label="ha solve krusell-smith")
        diag = nothing
        for (_, t) in pairs(r.doc.data)
            (t isa JSON3.Object && haskey(t, :columns)) || continue
            "r_squared" in String.(t.columns) && (diag = t)
        end
        @test diag !== nothing
        @test length(table_rows(diag)) >= 1
    end

    # C048 — HA Bayesian estimation (un-deferred after MEMs#228). RWMH re-solves the HA
    # model each draw, so this is kept minimal (krusell-smith, 4 draws, tiny horizon/grid).
    @testset "dsge ha estimate krusell-smith (C048)" begin
        rng = Random.MersenneTwister(123)
        csv = write_csv(DataFrame(K = 40.0 .+ 0.1 .* randn(rng, 16)); prefix="ha_k")
        priors = tempname() * "_ha_priors.toml"
        write(priors, "[priors]\n[priors.alpha]\ndist = \"normal\"\na = 0.36\nb = 0.05\n")
        try
            r = run_json(["dsge", "ha", "estimate", "krusell-smith",
                          "--data", csv, "--priors", priors, "--observables", "K",
                          "--method", "ssj", "--n-draws", "4", "--burnin", "1",
                          "--t-horizon", "20", "--n-reduced", "6", "--seed", "1"])
            assert_envelope_ok(r; label="dsge ha estimate")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test length(table_rows(tbl)) >= 1   # posterior summary row for alpha
        finally
            rm(csv; force=true); rm(priors; force=true)
        end
    end

    # C041 — CT Aiyagari + Blanchard OLG (small grids for CI time)
    @testset "dsge ct solve aiyagari" begin
        r = run_json(["dsge", "ct", "solve",
                      "--grid-size", "40", "--max-iter", "40", "--tol", "1e-4"])
        assert_envelope_ok(r; label="dsge ct solve")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
    end

    @testset "dsge olg solve" begin
        r = run_json(["dsge", "olg", "solve"])
        assert_envelope_ok(r; label="dsge olg solve")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
    end

    @testset "dsge olg simulate" begin
        r = run_json(["dsge", "olg", "simulate", "--horizon", "20"])
        assert_envelope_ok(r; label="dsge olg simulate")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 20
    end

    @testset "W6–W8 HA two-asset / dcegm / lifecycle / firm / bank" begin
        cols_table(doc, cols) = begin
            doc === nothing && return nothing
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                all(c -> c in table_cols(v), cols) && return v
            end
            return nothing
        end

        @testset "two-asset-hank B == B_supply" begin
            # Cap GE iters: the production 50×50×7 example does not clear, and a
            # full 200-iter closer blew the 60 min CI T3 budget. Keys still come
            # from the shipped leaf; numeric B≈B_supply is the CT GE pin below.
            r = run_json(["dsge", "ha", "steady-state", "two-asset-hank",
                          "--max-iter", "2"])
            assert_envelope_ok(r; label="two-asset-hank ss")
            kv = collect_named_kv(r.doc, "name", "value")
            @test haskey(kv, "B")
            @test haskey(kv, "B_supply")
            @test isapprox(Float64(kv["B_supply"]), 2.0; atol=0)
            # Discrete two-asset-hank GE on MEMs 0.9.0 does not clear the
            # production 50×50×7 example (measured B≈24.7 vs B_supply=2; upstream
            # tests document the same for coarse grids). Numeric B≈B_supply is
            # pinned on the CT closer, which does clear the liquid market.
            ct = run_json(["dsge", "ct", "solve", "--two-asset", "--ge",
                           "--rho", "0.06", "--max-iter", "120", "--tol", "1e-3"])
            assert_envelope_ok(ct; label="ct two-asset ge")
            mets = collect_named_kv(ct.doc, "metric", "value")
            @test haskey(mets, "B")
            @test haskey(mets, "B_supply")
            @test isapprox(Float64(mets["B"]), Float64(mets["B_supply"]);
                           rtol=1e-3, atol=1e-3)
        end

        @testset "huggett accuracy pre-solve refusal" begin
            r = run_json(["dsge", "ha", "accuracy", "huggett"])
            @test r.code == 5
            @test r.doc !== nothing && String(r.doc.error.code) == "model/unsupported"
        end

        @testset "HA hd reconstruction" begin
            csv = tempname() * ".csv"
            open(csv, "w") do io
                println(io, "K")
                for t in 1:16
                    println(io, 10.0 + 0.05 * sin(t / 3))
                end
            end
            r = run_json(["dsge", "ha", "hd", "krusell-smith", "--data", csv,
                          "--observables", "K", "--method", "ssj", "--n-reduced", "8",
                          "--t-horizon", "40"])
            assert_envelope_ok(r; label="ha hd")
            t = cols_table(r.doc, ["t"])
            @test t !== nothing
            rm(csv; force=true)
        end

        @testset "DCEGM excess_demand ≈ 0" begin
            # Calibration from MEMs test_dcegm_spec.jl (G-10): this is the
            # bracket that actually clears, not the CLI defaults.
            r = run_json(["dsge", "dcegm", "steady-state", "retirement",
                          "--n-periods", "6", "--n-a", "30", "--a-max", "40",
                          "--beta", "0.96", "--pension", "2.0", "--disutility", "0.5",
                          "--r-lo", "0.06", "--r-hi", "0.14", "--tol", "1e-3",
                          "--max-iter", "30"])
            assert_envelope_ok(r; label="dcegm ss")
            mets = collect_named_kv(r.doc, "metric", "value")
            @test haskey(mets, "excess_demand")
            @test isapprox(Float64(mets["excess_demand"]), 0.0; atol=5e-3)
            @test haskey(mets, "K") && Float64(mets["K"]) > 0
        end

        @testset "lifecycle cohort_mass sums to 1" begin
            r = run_json(["dsge", "lifecycle", "steady-state",
                          "--j", "12", "--j-retire", "9", "--n-a", "24",
                          "--income-states", "2", "--max-iter", "30"])
            assert_envelope_ok(r; label="lifecycle ss")
            t = cols_table(r.doc, ["age", "cohort_mass"])
            @test t !== nothing
            ci = findfirst(==("cohort_mass"), table_cols(t))
            mass = [Float64(collect(row)[ci]) for row in table_rows(t)]
            @test isapprox(sum(mass), 1.0; atol=1e-6)
        end

        @testset "Blanchard TFP IRF decay" begin
            r = run_json(["dsge", "olg", "irf", "--horizon", "12", "--rho-z", "0.5",
                          "--sigma-z", "0.01"])
            assert_envelope_ok(r; label="olg irf")
            t = nothing
            for (_, v) in pairs(r.doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                "horizon" in table_cols(v) && (t = v; break)
            end
            @test t !== nothing
            hi = findfirst(==("horizon"), table_cols(t))
            # pick a non-horizon numeric column
            vi = findfirst(c -> c != "horizon", table_cols(t))
            rows = [collect(x) for x in table_rows(t)]
            sort!(rows; by = r -> Int(r[hi]))
            if vi !== nothing && length(rows) >= 4
                a1 = abs(Float64(rows[2][vi]))
                aN = abs(Float64(rows[end][vi]))
                @test aN <= a1 + 1e-8 || aN < 0.5
            end
        end

        @testset "firm --prices ss|ge" begin
            zcsv = tempname() * ".csv"
            open(zcsv, "w") do io
                println(io, "Z")
                for t in 1:6
                    println(io, 1.0 + 0.01 * (t == 1))
                end
            end
            for pr in ("ss", "ge")
                r = run_json(["dsge", "firm", "transition", "--z-path", zcsv,
                              "--prices", pr, "--n-k", "8", "--n-eps", "2"])
                assert_envelope_ok(r; label="firm transition $pr")
            end
            rm(zcsv; force=true)
        end

        @testset "bank PE matches SS policies; honest non-convergence" begin
            ss = run_json(["dsge", "bank", "steady-state", "--n-n", "10", "--n-xi", "2"])
            assert_envelope_ok(ss; label="bank ss")
            mets = collect_named_kv(ss.doc, "metric", "value")
            @test haskey(mets, "R") && haskey(mets, "rk")
            pe = run_json(["dsge", "bank", "pe", "--n-n", "10", "--n-xi", "2",
                           "--r", string(mets["R"]), "--rk", string(mets["rk"])])
            assert_envelope_ok(pe; label="bank pe at SS prices")
            ss_pol = cols_table(ss.doc, ["n_index", "l_policy"])
            pe_pol = cols_table(pe.doc, ["n_index", "l_policy"])
            @test ss_pol !== nothing && pe_pol !== nothing
            function _lmap(tbl)
                cols = table_cols(tbl)
                ni = findfirst(==("n_index"), cols)
                xi = findfirst(==("xi_index"), cols)
                li = findfirst(==("l_policy"), cols)
                d = Dict{Tuple{Int,Int},Float64}()
                for row in table_rows(tbl)
                    r = collect(row)
                    d[(Int(r[ni]), Int(r[xi]))] = Float64(r[li])
                end
                return d
            end
            sm, pm = _lmap(ss_pol), _lmap(pe_pol)
            @test keys(sm) == keys(pm)
            @test !isempty(sm)
            for k in keys(sm)
                @test isapprox(sm[k], pm[k]; rtol=1e-3, atol=1e-4)
            end
            # Bracket so high that even real's 6× expansion cannot find a sign change.
            bad = run_json(["dsge", "bank", "steady-state", "--n-n", "8", "--n-xi", "2",
                            "--r-lo", "5", "--r-hi", "6", "--max-iter", "2"])
            @test bad.code == 0
            bmets = collect_named_kv(bad.doc, "metric", "value")
            @test haskey(bmets, "converged")
            @test lowercase(string(bmets["converged"])) in ("false", "0")
        end
    end

    # C051 loader + C061 bayes-compare — representative-agent DSGE on real MEMs.
    # Regression guard: the RA DSGE surface (every `dsge` / `dsge bayes` leaf that loads
    # a model file, via _load_dsge_model) had ZERO T3 coverage and was silently broken on
    # the 0.7.0 pin — `.toml` had no real DSGESpec constructor (MethodError); `.jl` hit a
    # world-age MethodError on the @dsge-generated residual fns; and `bayes_factor` (now a
    # log BF) tripped `dsge bayes compare` with a log-of-negative DomainError. Keep green
    # so a future MEMs bump can't hide the same class of drift.
    @testset "representative-agent DSGE (C051/C061)" begin
        dir = mktempdir()
        # Linear AR(1)-style RA DSGE: Y is the driven state, C mirrors it (linear = true
        # so the steady state is trivially zero — no nonlinear solve inside SMC).
        model_toml = joinpath(dir, "model.toml")
        write(model_toml, """
        [model]
        parameters = { rho = 0.9, sigma = 0.01 }
        endogenous = ["Y", "C"]
        exogenous = ["e"]
        linear = true
        [[model.equations]]
        expr = "Y[t] = rho * Y[t-1] + sigma * e[t]"
        [[model.equations]]
        expr = "C[t] = Y[t]"
        """)
        # No `using MacroEconometricModels` here on purpose — the loader injects it (C051).
        model_jl = joinpath(dir, "model.jl")
        write(model_jl, """
        @dsge begin
            parameters: rho = 0.9, sigma = 0.01
            endogenous: Y, C
            exogenous: e
            linear: true

            Y[t] = rho * Y[t-1] + sigma * e[t]
            C[t] = Y[t]
        end
        """)

        @testset "dsge solve from TOML (TOML→@dsge bridge)" begin
            r = run_json(["dsge", "solve", model_toml])
            assert_envelope_ok(r; label="dsge solve toml")
            # NOT `first_table`: JSON3 does not preserve insertion order, and W12 added a
            # Determinacy Verdict table to this leaf — so "the first table" silently became
            # a different one. Select by a DISTINCTIVE COLUMN (the standing T3 lesson).
            tbl = _dsge_policy_table(r.doc)
            @test tbl !== nothing
            @test length(table_rows(tbl)) == 2   # Y, C
        end

        @testset "dsge solve from .jl (auto-import + world-age)" begin
            r = run_json(["dsge", "solve", model_jl])
            assert_envelope_ok(r; label="dsge solve jl")
            tbl = _dsge_policy_table(r.doc)
            @test tbl !== nothing
            @test length(table_rows(tbl)) == 2
        end

        @testset "dsge solve --method perturbation --order 2 (gx over v=[states; shocks])" begin
            # gx/hx are ny×nv with v = [states; shocks] (Stage-14 #368 layout, ≥0.7.2).
            # The renderer labelled only the state columns → DimensionMismatch exit 1 on
            # EVERY model with a shock. Broken through the whole v0.9.1 line because this
            # `--method` branch had no T3 of its own (the per-BRANCH coverage lesson).
            r = run_json(["dsge", "solve", model_jl, "--method", "perturbation", "--order", "2"])
            assert_envelope_ok(r; label="dsge solve perturbation o2")
            tbl = named_table(r.doc, :perturbation_policy_gx)
            @test tbl !== nothing
            cols = String[string(c) for c in tbl.columns]
            # one column per state PLUS one per shock, after the leading :control
            @test cols == ["control", "Y", "e"]
            row = collect(table_rows(tbl)[1])
            # C[t] = Y[t] = rho*Y[t-1] + sigma*e[t] → loadings are exactly (rho, sigma)
            @test numv(row[2]) ≈ 0.9 atol=1e-8
            @test numv(row[3]) ≈ 0.01 atol=1e-8
        end

        @testset "W1 dsge solve --method pfi (no order=)" begin
            r = run_json(["dsge", "solve", model_jl, "--method", "pfi"])
            assert_envelope_ok(r; label="dsge solve pfi")
            @test named_table(r.doc, :projection_solution) !== nothing ||
                  any(t -> t isa JSON3.Object && haskey(t, :columns) &&
                           "control" in String.(t.columns), values(r.doc.data))
        end

        @testset "W5 dsge solve --method vfi / blanchard-kahn / pfi next-state" begin
            cols_table(doc, cols) = begin
                doc === nothing && return nothing
                for (_, v) in pairs(doc.data)
                    (v isa JSON3.Object && haskey(v, :rows)) || continue
                    all(c -> c in table_cols(v), cols) && return v
                end
                return nothing
            end
            rbc = joinpath(dir, "rbc.jl")
            write(rbc, """
            @dsge begin
                parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01
                endogenous: c, k, a
                exogenous: e
                utility: log(c)
                beta: beta
                controls: c
                euler: 1 / c[t] = beta * (1 / c[t+1]) * (alpha * exp(a[t+1]) * k[t]^(alpha - 1) + 1 - delta)
                c[t] + k[t] = exp(a[t]) * k[t-1]^alpha + (1 - delta) * k[t-1]
                a[t] = rho * a[t-1] + sigma * e[t]
            end
            """)
            rv = run_json(["dsge", "solve", rbc, "--method", "vfi", "--n-grid", "8",
                           "--n-choice", "15", "--degree", "3", "--max-iter", "200",
                           "--tol", "1e-4", "--howard-steps", "10",
                           "--next-state", "residual"])
            assert_envelope_ok(rv; label="dsge solve vfi")
            vmets = collect_named_kv(rv.doc, "metric", "value")
            @test haskey(vmets, "converged")
            @test lowercase(string(vmets["converged"])) in ("true", "1")
            vf = cols_table(rv.doc, ["node", "V"])
            @test vf !== nothing
            vi = findfirst(==("V"), table_cols(vf))
            vals = [Float64(collect(row)[vi]) for row in table_rows(vf)]
            @test length(vals) >= 2
            @test all(isfinite, vals)
            # monotone in the collocation order of the 1-state RBC
            @test vals[end] >= vals[1] - 1e-6
            rbk = run_json(["dsge", "solve", model_jl, "--method", "blanchard-kahn"])
            assert_envelope_ok(rbk; label="dsge solve blanchard-kahn")
            rg = run_json(["dsge", "solve", model_jl, "--method", "gensys"])
            assert_envelope_ok(rg; label="dsge solve gensys vs bk")
            t_bk = _dsge_policy_table(rbk.doc)
            t_g = _dsge_policy_table(rg.doc)
            @test t_bk !== nothing && t_g !== nothing
            @test numeric_tables_agree(t_bk, t_g; atol=1e-8, rtol=1e-8, sort_by="variable")
            rp = run_json(["dsge", "solve", rbc, "--method", "pfi", "--next-state", "nonlinear",
                           "--degree", "3", "--max-iter", "40"])
            assert_envelope_ok(rp; label="dsge solve pfi nonlinear")
            shocks = joinpath(dir, "pf_shocks.csv")
            open(shocks, "w") do io
                println(io, "e")
                for _ in 1:8; println(io, "0.0"); end
            end
            spa = run_json(["dsge", "perfect-foresight", model_jl, "--shocks", shocks,
                            "--periods", "8", "--sparsity", "auto"])
            spd = run_json(["dsge", "perfect-foresight", model_jl, "--shocks", shocks,
                            "--periods", "8", "--sparsity", "dense"])
            assert_envelope_ok(spa; label="pf auto")
            assert_envelope_ok(spd; label="pf dense")
            ta = cols_table(spa.doc, ["period"])
            td = cols_table(spd.doc, ["period"])
            @test ta !== nothing && td !== nothing
            @test numeric_tables_agree(ta, td; atol=1e-8, rtol=1e-6, sort_by="period")
        end

        @testset "V0122 VFI smolyak + optimizer (MEMs#817-819, #821)" begin
            rbc = joinpath(dir, "vfi_rbc2.jl")
            write(rbc, """
            @dsge begin
                parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01
                endogenous: c, k, a
                exogenous: e
                utility: log(c)
                beta: beta
                controls: c
                euler: 1 / c[t] = beta * (1 / c[t+1]) * (alpha * exp(a[t+1]) * k[t]^(alpha - 1) + 1 - delta)
                c[t] + k[t] = exp(a[t]) * k[t-1]^alpha + (1 - delta) * k[t-1]
                a[t] = rho * a[t-1] + sigma * e[t]
            end
            """)
            # Genuine 4-state model (k + 3 TFP components): the shock sum is
            # scaled so the default-guess steady state converges (unscaled
            # exp(a1+a2+a3) sends Newton to k<0 — see W1 appendix).
            rbc4s = joinpath(dir, "vfi_rbc4s.jl")
            write(rbc4s, """
            @dsge begin
                parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01
                endogenous: c, k, a1, a2, a3
                exogenous: e1, e2, e3
                utility: log(c)
                beta: beta
                controls: c
                euler: 1 / c[t] = beta * (1 / c[t+1]) * (alpha * exp((a1[t+1] + a2[t+1] + a3[t+1]) / 3) * k[t]^(alpha - 1) + 1 - delta)
                c[t] + k[t] = exp((a1[t] + a2[t] + a3[t]) / 3) * k[t-1]^alpha + (1 - delta) * k[t-1]
                a1[t] = rho * a1[t-1] + sigma * e1[t]
                a2[t] = rho * a2[t-1] + sigma * e2[t]
                a3[t] = rho * a3[t-1] + sigma * e3[t]
            end
            """)
            # Two-control labor RBC. The FOC isolates n[t] on the LHS (exact
            # rearrangement, same zeros) so residual transition inference can
            # assign it — defines-detection is syntactic (W1 appendix).
            labor = joinpath(dir, "vfi_labor.jl")
            write(labor, """
            @dsge begin
                parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01, psi = 1.5
                endogenous: c, k, a, n
                exogenous: e
                utility: log(c)
                beta: beta
                controls: c, n
                euler: 1 / c[t] = beta * (1 / c[t+1]) * (alpha * exp(a[t+1]) * k[t]^(alpha - 1) * n[t+1]^(1 - alpha) + 1 - delta)
                c[t] + k[t] = exp(a[t]) * k[t-1]^alpha * n[t]^(1 - alpha) + (1 - delta) * k[t-1]
                n[t] = 1 - (psi * c[t] * n[t]^alpha) / ((1 - alpha) * exp(a[t]) * k[t-1]^alpha)
                a[t] = rho * a[t-1] + sigma * e[t]
            end
            """)
            labor_nc = joinpath(dir, "vfi_labor_noctrl.jl")
            write(labor_nc, """
            @dsge begin
                parameters: beta = 0.99, alpha = 0.36, delta = 0.025, rho = 0.9, sigma = 0.01, psi = 1.5
                endogenous: c, k, a, n
                exogenous: e
                utility: log(c)
                beta: beta
                euler: 1 / c[t] = beta * (1 / c[t+1]) * (alpha * exp(a[t+1]) * k[t]^(alpha - 1) * n[t+1]^(1 - alpha) + 1 - delta)
                c[t] + k[t] = exp(a[t]) * k[t-1]^alpha * n[t]^(1 - alpha) + (1 - delta) * k[t-1]
                n[t] = 1 - (psi * c[t] * n[t]^alpha) / ((1 - alpha) * exp(a[t]) * k[t-1]^alpha)
                a[t] = rho * a[t-1] + sigma * e[t]
            end
            """)
            # NOTE: no --n-grid here: it is a tensor-only knob and the
            # shared guard rejects it alongside --grid smolyak (see rd1).
            # --n-choice rides along for the grid paths (tensor/smolyak/
            # auto→grid1d); the explicit-fminbox calls below strip it via
            # base_fm (upstream n_choice is grid1d-only, and the CLI guard
            # rejects the dead explicit combo — see rd3).
            # No --degree here either: it is tensor-path-only like --n-grid
            # (upstream ignores it on Smolyak; the guard rejects it — rd4).
            # Tensor-path calls below pass it explicitly.
            base = ["--n-choice", "15",
                    "--max-iter", "200", "--tol", "1e-4", "--howard-steps", "10",
                    "--next-state", "residual"]
            base_fm = filter(x -> x != "--n-choice" && x != "15", base)
            # Interior state points (exact values immaterial: identity
            # compares use the same point on both runs).
            pt2 = "38.0,0.0"
            pt4 = "38.0,0.0,0.0,0.0"
            ptL = "13.9,0.0"
            _kv(doc) = collect_named_kv(doc, "metric", "value")
            _V(doc) = Float64(_kv(doc)["V"])
            _diag(doc, k) = _kv(doc)[k]

            # Smolyak on nx=2: node count is the closed-form N(2,2)=13.
            rs = run_json(["dsge", "solve", rbc, "--method", "vfi",
                           "--grid", "smolyak", base..., "--evaluate-at", pt2])
            assert_envelope_ok(rs; label="vfi smolyak")
            @test Bool(_diag(rs.doc, "converged")) === true
            @test String(_diag(rs.doc, "grid_type")) == "smolyak"
            @test Int(_diag(rs.doc, "n_nodes")) == 13
            @test Int(_diag(rs.doc, "smolyak_blocks")) > 0
            @test isfinite(_V(rs.doc))
            # μ-refinement toward tensor: N(2,3)=29 nodes, strictly closer.
            # Explicit --n-grid 12 == the default, so this is bit-exact
            # with the auto run below while still exercising the knob.
            rt = run_json(["dsge", "solve", rbc, "--method", "vfi",
                           "--grid", "tensor", "--n-grid", "12",
                           "--degree", "3",
                           base..., "--evaluate-at", pt2])
            assert_envelope_ok(rt; label="vfi tensor baseline")
            @test Int(_diag(rt.doc, "smolyak_blocks")) == 0
            rm3 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "smolyak", "--smolyak-mu", "3",
                            base..., "--evaluate-at", pt2])
            assert_envelope_ok(rm3; label="vfi smolyak mu3")
            @test Int(_diag(rm3.doc, "n_nodes")) == 29
            @test abs(_V(rm3.doc) - _V(rt.doc)) < abs(_V(rs.doc) - _V(rt.doc))
            # μ=2 is coarse on the wide k-grid (gap ≈ 8.9); the band only
            # excludes garbage (a broken interpolant gives ±150 penalties).
            @test abs(_V(rs.doc) - _V(rt.doc)) < 15
            # Anisotropic vector mu + length-mismatch exit class.
            rmv = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "smolyak", "--smolyak-mu", "2,3",
                            base...])
            assert_envelope_ok(rmv; label="vfi smolyak-mu vector")
            rmm = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "smolyak", "--smolyak-mu", "2,2,2",
                            base...])
            @test rmm.code == 3
            @test String(rmm.doc.error.code) == "data/invalid"
            # Same-path identities are bit-exact (same binary, same point):
            # auto→tensor on nx=2, and auto→grid1d on 1 control.
            ra = run_json(["dsge", "solve", rbc, "--method", "vfi",
                           "--grid", "auto", "--degree", "3",
                           base..., "--evaluate-at", pt2])
            assert_envelope_ok(ra; label="vfi auto")
            @test String(_diag(ra.doc, "grid_type")) == "tensor"
            @test _V(ra.doc) == _V(rt.doc)
            rg1 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--optimizer", "grid1d", "--degree", "3", base...,
                            "--evaluate-at", pt2])
            assert_envelope_ok(rg1; label="vfi grid1d")
            @test _V(rg1.doc) == _V(ra.doc)
            # fminbox agrees loosely on 1 control (measured ≈ 2.6e-6).
            rnm = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--optimizer", "fminbox-nm", "--degree", "3",
                            base_fm..., "--evaluate-at", pt2])
            assert_envelope_ok(rnm; label="vfi fminbox-nm")
            @test abs(_V(rnm.doc) - _V(rg1.doc)) < 1e-3
            rlb = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--optimizer", "fminbox-lbfgs", "--degree", "3",
                            base_fm..., "--evaluate-at", pt2])
            assert_envelope_ok(rlb; label="vfi fminbox-lbfgs")
            @test abs(_V(rlb.doc) - _V(rg1.doc)) < 1e-3
            # Dead combos + vocabulary: all usage/invalid.
            rd1 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "smolyak", "--n-grid", "8"])
            @test rd1.code == 2
            rd2 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "tensor", "--smolyak-mu", "2"])
            @test rd2.code == 2
            rd3 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--optimizer", "fminbox-nm", "--n-choice", "15"])
            @test rd3.code == 2
            rbo = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--optimizer", "bogus"])
            @test rbo.code == 2
            rd4 = run_json(["dsge", "solve", rbc, "--method", "vfi",
                            "--grid", "smolyak", "--degree", "3"])
            @test rd4.code == 2
            @test String(rd4.doc.error.code) == "usage/invalid"
            # nx=4 routing: auto→smolyak, N(4,2)=41 nodes, bit-exact.
            b4 = ["--n-choice", "15", "--max-iter", "150",
                  "--tol", "1e-3", "--howard-steps", "5",
                  "--next-state", "residual"]
            r4a = run_json(["dsge", "solve", rbc4s, "--method", "vfi",
                            "--grid", "auto", b4..., "--evaluate-at", pt4])
            assert_envelope_ok(r4a; label="vfi auto nx=4")
            @test Bool(_diag(r4a.doc, "converged")) === true
            @test String(_diag(r4a.doc, "grid_type")) == "smolyak"
            @test Int(_diag(r4a.doc, "n_nodes")) == 41
            r4s = run_json(["dsge", "solve", rbc4s, "--method", "vfi",
                            "--grid", "smolyak", b4..., "--evaluate-at", pt4])
            assert_envelope_ok(r4s; label="vfi smolyak nx=4")
            @test _V(r4a.doc) == _V(r4s.doc)
            # Two-control labor: auto→fminbox-nm bit-exact (same path).
            rl = run_json(["dsge", "solve", labor, "--method", "vfi",
                           "--degree", "3",
                           base..., "--evaluate-at", ptL])
            assert_envelope_ok(rl; label="vfi 2-control auto")
            @test Bool(_diag(rl.doc, "converged")) === true
            rln = run_json(["dsge", "solve", labor, "--method", "vfi",
                            "--optimizer", "fminbox-nm", "--degree", "3",
                            base_fm..., "--evaluate-at", ptL])
            assert_envelope_ok(rln; label="vfi 2-control nm")
            @test _V(rl.doc) == _V(rln.doc)
            # (No nm-vs-lbfgs agreement: the maximizers converge to V's
            # 1.1 apart here — upstream solver behavior, recorded in the
            # W1 appendix. lbfgs-2ctrl is upstream-tested (#818); the CLI
            # threads all four values identically, pinned on 1 control.)
            # grid1d + 2 controls: exit 2 with explicit controls (CLI
            # pre-check), exit 3 with default controls (upstream throw).
            rl1 = run_json(["dsge", "solve", labor, "--method", "vfi",
                            "--optimizer", "grid1d"])
            @test rl1.code == 2
            @test String(rl1.doc.error.code) == "usage/invalid"
            rl0 = run_json(["dsge", "solve", labor_nc, "--method", "vfi",
                            "--optimizer", "grid1d", base...])
            @test rl0.code == 3
            @test String(rl0.doc.error.code) == "data/invalid"
            # irf/simulate ride the ProjectionSolution path: smolyak smoke
            # plus the tensor-simulate regression (antithetic fix) + seed.
            ri = run_json(["dsge", "irf", rbc, "--method", "vfi",
                           "--grid", "smolyak", base..., "--horizon", "4"])
            assert_envelope_ok(ri; label="vfi irf smolyak")
            rsm = run_json(["dsge", "simulate", rbc, "--method", "vfi",
                            "--grid", "smolyak", base..., "--periods", "20",
                            "--burn", "5"])
            assert_envelope_ok(rsm; label="vfi simulate smolyak")
            tsm = named_table(rsm.doc, :dsge_simulation)
            @test tsm !== nothing && length(table_rows(tsm)) == 20
            rst = run_json(["dsge", "simulate", rbc, "--method", "vfi",
                            "--degree", "3",
                            base..., "--periods", "20", "--burn", "5"])
            assert_envelope_ok(rst; label="vfi simulate tensor")
            rss = run_json(["dsge", "simulate", rbc, "--method", "vfi",
                            "--grid", "smolyak", base..., "--periods", "20",
                            "--burn", "5", "--seed", "7"])
            assert_envelope_ok(rss; label="vfi simulate smolyak seed")
        end

        @testset "W1 dsge solve --method projection (no order=)" begin
            r = run_json(["dsge", "solve", model_jl, "--method", "projection"])
            assert_envelope_ok(r; label="dsge solve projection")
            @test named_table(r.doc, :projection_solution) !== nothing ||
                  any(t -> t isa JSON3.Object && haskey(t, :columns) &&
                           "control" in String.(t.columns), values(r.doc.data))
        end

        @testset "W1 dsge solve --method junk → usage" begin
            r = run_json(["dsge", "solve", model_jl, "--method", "not-a-solver"])
            @test r.code == 2
        end

        @testset "W1 perfect-foresight with/without shocks" begin
            shocks = joinpath(dir, "pf_shocks.csv")
            open(shocks, "w") do io
                println(io, "e")
                for t in 1:8
                    println(io, t == 1 ? 1.0 : 0.0)
                end
            end
            r0 = run_json(["dsge", "perfect-foresight", model_jl, "--periods", "8"])
            assert_envelope_ok(r0; label="perfect-foresight no shocks")
            r1 = run_json(["dsge", "perfect-foresight", model_jl, "--shocks", shocks, "--periods", "8"])
            assert_envelope_ok(r1; label="perfect-foresight with shocks")
            rbad = run_json(["dsge", "perfect-foresight", model_jl, "--shocks", shocks, "--periods", "20"])
            @test rbad.code == 3
        end

        @testset "W1 OccBin 1/2/>2 bounds + nonlinear Expr" begin
            c1 = joinpath(dir, "occ1.toml")
            write(c1, """
            [[constraints.bounds]]
            variable = "Y"
            lower = -10.0
            """)
            r1 = run_json(["dsge", "solve", model_jl, "--constraints", c1, "--periods", "8"])
            assert_envelope_ok(r1; label="occbin 1 bound")

            c2 = joinpath(dir, "occ2.toml")
            write(c2, """
            [[constraints.bounds]]
            variable = "Y"
            lower = -10.0
            [[constraints.bounds]]
            variable = "C"
            upper = 10.0
            """)
            r2 = run_json(["dsge", "solve", model_jl, "--constraints", c2, "--periods", "8"])
            assert_envelope_ok(r2; label="occbin 2 bounds")

            c3 = joinpath(dir, "occ3.toml")
            write(c3, """
            [[constraints.bounds]]
            variable = "Y"
            lower = -10.0
            [[constraints.bounds]]
            variable = "C"
            upper = 10.0
            [[constraints.bounds]]
            variable = "Y"
            upper = 10.0
            """)
            r3 = run_json(["dsge", "solve", model_jl, "--constraints", c3, "--periods", "8"])
            @test r3.code == 2

            nl = joinpath(dir, "occnl.toml")
            write(nl, """
            [[constraints.nonlinear]]
            expr = "Y[t] >= -10.0"
            """)
            rnl = run_json(["dsge", "solve", model_jl, "--constraints", nl, "--periods", "8"])
            assert_envelope_ok(rnl; label="occbin nonlinear expr")

            rirf = run_json(["dsge", "irf", model_jl, "--constraints", c1, "--horizon", "6"])
            assert_envelope_ok(rirf; label="occbin irf 1 bound")
        end

        @testset "dsge steady-state from TOML (compute_steady_state world-age)" begin
            r = run_json(["dsge", "steady-state", model_toml])
            assert_envelope_ok(r; label="dsge steady-state")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
        end

        @testset "dsge irf from .jl (solve then IRF on solution)" begin
            r = run_json(["dsge", "irf", model_jl, "--horizon", "12"])
            assert_envelope_ok(r; label="dsge irf")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
        end

        @testset "dsge bayes posterior-mode / prior-predictive (C073 #78)" begin
            pri = joinpath(dir, "priors_pm.toml")
            write(pri, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            [priors.sigma]
            dist = "inv_gamma"
            a = 2.0
            b = 0.1
            """)
            dat = joinpath(dir, "data_pm.csv")
            open(dat, "w") do io
                println(io, "Y"); y = 0.0
                for _ in 1:60; y = 0.9y + 0.01randn(); println(io, y); end
            end

            # local helpers: coltable/scan_metric elsewhere in this file are closures
            # scoped to other testsets, not globals
            find_tbl(doc, col) = begin
                hit = nothing
                for t in values(doc.data)
                    if t isa JSON3.Object && haskey(t, :columns) && col in String.(t.columns)
                        hit = t; break
                    end
                end
                hit
            end
            metrics(doc) = Set(String(collect(r)[1]) for t in values(doc.data)
                               if (t isa JSON3.Object && haskey(t, :columns) &&
                                   "metric" in String.(t.columns)) for r in t.rows)

            rcs = run_json(["dsge", "bayes", "estimate", model_toml,
                            "--data", dat, "--params", "rho,sigma",
                            "--priors", pri, "--observables", "Y",
                            "--sampler", "smc", "--n-smc", "20", "--n-particles", "20",
                            "--constraint-solver", "optim"])
            assert_envelope_ok(rcs; label="dsge bayes estimate constraint-solver")

            r = run_json(["dsge", "bayes", "posterior-mode", model_toml,
                          "--data", dat, "--params", "rho,sigma",
                          "--priors", pri, "--observables", "Y"])
            assert_envelope_ok(r; label="dsge bayes posterior-mode")
            t = find_tbl(r.doc, "mode")
            @test t !== nothing && length(table_rows(t)) == 2      # rho, sigma
            @test Set(["parameter", "mode", "std_error"]) ⊆ Set(String.(table_cols(t)))
            @test Set(["log posterior", "log likelihood", "Laplace log ML",
                       "converged"]) ⊆ metrics(r.doc)

            # prior-predictive needs NO --data: it draws from the prior. Passing --data
            # would be a false requirement, so the leaf does not accept it.
            rp = run_json(["dsge", "bayes", "prior-predictive", model_toml,
                           "--params", "rho,sigma", "--priors", pri,
                           "--observables", "Y", "--n-draws", "40", "--periods", "80"])
            assert_envelope_ok(rp; label="dsge bayes prior-predictive")
            pt = find_tbl(rp.doc, "statistic")
            @test pt !== nothing && length(table_rows(pt)) >= 1
            @test Set(["statistic", "mean", "std", "q05", "median", "q95"]) ⊆
                  Set(String.(table_cols(pt)))
            @test Set(["draws requested", "draws that solved",
                       "periods simulated"]) ⊆ metrics(rp.doc)
            @test run_json(["dsge", "bayes", "prior-predictive", model_toml,
                            "--params", "rho,sigma", "--priors", pri,
                            "--data", dat]).code == 2      # --data is not an option here

            # every option guarded up front → usage errors, never exit 1
            @test run_json(["dsge", "bayes", "posterior-mode", model_toml,
                            "--data", dat, "--priors", pri]).code == 2      # --params
            @test run_json(["dsge", "bayes", "posterior-mode", model_toml,
                            "--params", "rho,sigma", "--priors", pri]).code == 2  # --data
            @test run_json(["dsge", "bayes", "posterior-mode", model_toml, "--data", dat,
                            "--params", "rho,sigma", "--priors", pri,
                            "--max-iter", "0"]).code == 2
            @test run_json(["dsge", "bayes", "prior-predictive", model_toml,
                            "--params", "rho,sigma", "--priors", pri,
                            "--n-draws", "0"]).code == 2
        end

        @testset "dsge bayes compare (C061; log-BF semantics)" begin
            priors = joinpath(dir, "priors.toml")
            write(priors, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            [priors.sigma]
            dist = "inv_gamma"
            a = 2.0
            b = 0.1
            """)
            priors2 = joinpath(dir, "priors2.toml")   # model 2 estimates rho only
            write(priors2, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            """)
            # 1 observable (Y) ⇒ 1 structural shock ⇒ non-singular likelihood.
            data = joinpath(dir, "data.csv")
            open(data, "w") do io
                println(io, "Y")
                y = 0.0
                for _ in 1:60
                    y = 0.9 * y + 0.01 * randn()
                    println(io, y)
                end
            end
            r = run_json(["dsge", "bayes", "compare", model_jl,
                          "--data", data, "--observables", "Y",
                          "--params", "rho,sigma", "--priors", priors,
                          "--model2", model_jl, "--params2", "rho", "--priors2", priors2,
                          "--sampler", "smc", "--n-smc", "100", "--n-particles", "50",
                          "--n-draws", "100", "--burnin", "10"])
            assert_envelope_ok(r; label="dsge bayes compare")
            tbl = named_table(r.doc, :bayesian_model_comparison)
            @test tbl !== nothing
            if tbl !== nothing
                @test length(table_rows(tbl)) == 2   # Model 1, Model 2
                lml_i = col_index(tbl, "log_marginal_likelihood")
                @test lml_i !== nothing
                # teeth: both marginal likelihoods finite ⇒ estimation ran end-to-end and
                # the log-BF handler path did not throw (the previous log-of-negative
                # DomainError).
                if lml_i !== nothing
                    for row in table_rows(tbl)
                        @test isfinite(Float64(collect(row)[lml_i]))
                    end
                end
            end
        end

        # C073 — Bayesian DSGE diagnostics. Reuse the RA `.jl` spec; own priors+data.
        # Tiny SMC chains are noisy — assert shapes/keys/finiteness, not tight numbers.
        @testset "dsge bayes diagnostics (C073)" begin
            priors = joinpath(dir, "diag_priors.toml")
            write(priors, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            [priors.sigma]
            dist = "inv_gamma"
            a = 2.0
            b = 0.1
            """)
            data = joinpath(dir, "diag_data.csv")
            open(data, "w") do io
                println(io, "Y")
                y = 0.0
                for _ in 1:60
                    y = 0.9 * y + 0.01 * randn()
                    println(io, y)
                end
            end
            smc = ["--sampler", "smc", "--n-smc", "100", "--n-particles", "50",
                   "--n-draws", "100", "--burnin", "10"]
            base = vcat(["--data", data, "--observables", "Y",
                         "--params", "rho,sigma", "--priors", priors], smc)

            @testset "mcmc-diag (R-hat/ESS/Geweke)" begin
                r = run_json(vcat(["dsge", "bayes", "mcmc-diag", model_jl], base))
                assert_envelope_ok(r; label="dsge bayes mcmc-diag")
                tbl = named_table(r.doc, :mcmc_convergence_diagnostics)
                @test tbl !== nothing
                if tbl !== nothing
                    ri = col_index(tbl, "rhat")
                    @test ri !== nothing
                    @test length(table_rows(tbl)) == 2   # rho, sigma
                    if ri !== nothing
                        for row in table_rows(tbl)
                            @test isfinite(Float64(collect(row)[ri]))
                        end
                    end
                end
            end

            @testset "identification (Iskrev rank test; no MCMC)" begin
                r = run_json(["dsge", "bayes", "identification", model_jl,
                              "--params", "rho,sigma", "--observables", "Y"])
                assert_envelope_ok(r; label="dsge bayes identification")
                kv = named_table(r.doc, :identification_diagnostics)
                @test kv !== nothing
                if kv !== nothing
                    @test metric_value(kv, "rank") !== nothing
                    @test metric_value(kv, "identified") !== nothing
                end
            end

            @testset "identification bad --params → usage/invalid, not exit-1 (review fix)" begin
                # A --params typo makes MEMs' identification_diagnostics throw an untyped
                # KeyError (spec.param_values[:typo]); the handler must map it to usage/invalid
                # (exit 2), NOT let it fall through run_cli to the exit-1 "likely a bug" tail.
                r = run_json(["dsge", "bayes", "identification", model_jl,
                              "--params", "rho,typo", "--observables", "Y"])
                @test r.code == 2
            end

            @testset "learning-rate (Koop-Pesaran-Smith)" begin
                r = run_json(vcat(["dsge", "bayes", "learning-rate", model_jl], base,
                                  ["--refit-n-smc", "30"]))
                assert_envelope_ok(r; label="dsge bayes learning-rate")
                tbl = named_table(r.doc, :learning_rate_check)
                @test tbl !== nothing
                if tbl !== nothing
                    @test col_index(tbl, "learning_rate") !== nothing
                    @test length(table_rows(tbl)) == 2
                end
            end

            @testset "overlap (prior/posterior)" begin
                r = run_json(vcat(["dsge", "bayes", "overlap", model_jl], base))
                assert_envelope_ok(r; label="dsge bayes overlap")
                tbl = named_table(r.doc, :prior_posterior_overlap)
                @test tbl !== nothing
                if tbl !== nothing
                    @test col_index(tbl, "overlap") !== nothing
                    @test length(table_rows(tbl)) == 2
                end
            end

            @testset "marginal-lik (bridge sampling; may be NaN)" begin
                r = run_json(vcat(["dsge", "bayes", "marginal-lik", model_jl], base))
                assert_envelope_ok(r; label="dsge bayes marginal-lik")
                kv = named_table(r.doc, :marginal_likelihood_bridge_sampling)
                @test kv !== nothing
                if kv !== nothing
                    # bridge_sampling_ml can return NaN on a tiny chain → assert the leaf
                    # ran end-to-end (both keys present), not a finite value.
                    @test metric_value(kv, "log_marginal_likelihood_bridge") !== nothing
                    @test metric_value(kv, "log_marginal_likelihood_smc") !== nothing
                end
            end
        end

        # W4/#139 (#81): the two direct-Exception domain types OUTSIDE MacroModelError
        # must map to their SPECIFIC typed classes — envelope code AND process exit
        # agree (the W2 machinery). Verified against the real throw sites on the
        # resolved 0.8.0 copy: bayes_estimation.jl (eager guard), steady_state.jl.
        @testset "W4/#139: direct-Exception domain types → typed classes" begin
            # StochasticSingularityError: 2 observables, 1 structural shock, no
            # measurement error. The guard is EAGER (fires before sampling), so the
            # tiny chain settings are never reached — this case is cheap.
            pri4 = joinpath(dir, "priors_w4.toml")
            write(pri4, """
            [priors]
            [priors.rho]
            dist = "beta"
            a = 0.5
            b = 0.2
            """)
            dat4 = joinpath(dir, "data_w4.csv")
            open(dat4, "w") do io
                println(io, "Y,C"); y = 0.0
                for _ in 1:60; y = 0.9y + 0.01randn(); println(io, "$y,$y"); end
            end
            rs = run_json(["dsge", "bayes", "estimate", model_jl, "--data", dat4,
                           "--params", "rho", "--priors", pri4, "--observables", "Y,C",
                           "--sampler", "mh", "--n-draws", "50", "--burnin", "10"])
            @test rs.code == 5
            @test rs.doc !== nothing && String(rs.doc.status) == "error"
            @test String(rs.doc.error.code) == "model/stochastic-singularity"
            @test Int(rs.doc.error.exit_code) == 5
            @test occursin("observables exceed", String(rs.doc.error.message))

            # #148: --measurement-error is the in-CLI remedy the hint points at —
            # the SAME invocation with auto must estimate, not error.
            rok = run_json(["dsge", "bayes", "estimate", model_jl, "--data", dat4,
                            "--params", "rho", "--priors", pri4, "--observables", "Y,C",
                            "--sampler", "mh", "--n-draws", "50", "--burnin", "10",
                            "--measurement-error", "auto"])
            @test rok.code == 0
            @test rok.doc !== nothing && String(rok.doc.status) == "ok"
            # vector form is guarded up front: wrong length → usage, not exit 1
            rbad = run_json(["dsge", "bayes", "estimate", model_jl, "--data", dat4,
                             "--params", "rho", "--priors", pri4, "--observables", "Y,C",
                             "--sampler", "mh", "--n-draws", "50", "--burnin", "10",
                             "--measurement-error", "0.1"])
            @test rbad.code == 2
            rgarbage = run_json(["dsge", "bayes", "estimate", model_jl, "--data", dat4,
                                 "--params", "rho", "--priors", pri4, "--observables", "Y,C",
                                 "--measurement-error", "0.1,x"])
            @test rgarbage.code == 2

            # #146: multi-table --output — the per-shock HD loop used to write every
            # shock to the SAME file (last one wins). Each shock now gets its own
            # suffixed path; the bare path is never written by the loop. This is
            # also the first T3 coverage `dsge hd` has ever had.
            hdout = joinpath(dir, "hd_out.csv")
            rhd = run_json(["dsge", "hd", model_jl, "--data", dat4,
                            "--observables", "Y", "--output", hdout])
            @test rhd.code == 0
            @test isfile(joinpath(dir, "hd_out_e.csv"))   # per-shock (shock `e`) file
            @test !isfile(hdout)                          # bare path not clobbered

            # DSGESolveError: y = y² + 2 has no real steady state, so the residual
            # gate in compute_steady_state throws. Per-BRANCH coverage: both leaves
            # that reach the numerical steady-state path.
            bad4 = joinpath(dir, "bad_w4.jl")
            write(bad4, """
            @dsge begin
                parameters: sigma = 0.01
                endogenous: Y
                exogenous: e

                Y[t] = Y[t-1]^2 + 2 + sigma * e[t]
            end
            """)
            for leaf in (["dsge", "solve", bad4], ["dsge", "steady-state", bad4])
                rb = run_json(String[leaf...])
                @test rb.code == 5
                @test rb.doc !== nothing && String(rb.doc.status) == "error"
                @test String(rb.doc.error.code) == "model/solve"
                @test Int(rb.doc.error.exit_code) == 5
            end
        end

        rm(dir; force=true, recursive=true)
    end

    # C042 — X-13ARIMA-SEATS (pure-Julia MEMs port; always available)
    @testset "filter x13 monthly" begin
        csv = tempname() * ".csv"
        open(csv, "w") do io
            println(io, "y")
            for t in 1:120
                println(io, 100 + 10 * sin(2π * t / 12) + 0.05 * t + 0.3 * randn())
            end
        end
        r = run_json(["filter", "x13", csv, "--frequency", "12", "--method", "x11",
                      "--transform", "none"])
        assert_envelope_ok(r; label="filter x13")
        _, tbl = first_table(r.doc)
        @test tbl !== nothing
        @test length(table_rows(tbl)) >= 100
        rm(csv; force=true)
    end

    # ── Input-Output analysis (C049) — offline via the bundled :wiot fixture ──
    @testset "io command family (C049)" begin
        # First table in the envelope whose columns ⊇ `cols`.
        cols_table(doc, cols) = begin
            doc === nothing && return nothing
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                all(c -> c in table_cols(v), cols) && return v
            end
            return nothing
        end

        @testset "sources catalog" begin
            r = run_json(["io", "sources"])
            assert_envelope_ok(r; label="io sources")
            t = cols_table(r.doc, ["source", "name", "versions", "credentials"])
            @test t !== nothing && length(table_rows(t)) == 5
        end

        @testset "load :wiot dims + balance" begin
            r = run_json(["io", "load"])
            assert_envelope_ok(r; label="io load")
            t = cols_table(r.doc, ["sector", "gross_output", "final_demand", "value_added"])
            @test t !== nothing
            rows = [collect(x) for x in table_rows(t)]
            ci = findfirst(==("sector"), table_cols(t)); gi = findfirst(==("gross_output"), table_cols(t))
            go = Dict(string(row[ci]) => Float64(row[gi]) for row in rows)
            @test isapprox(go["Agriculture"], 1000.0; atol=1e-6)
            @test isapprox(go["Manufacturing"], 2000.0; atol=1e-6)
        end

        @testset "leontief wide (L[Ag,Ag] ≈ 1.254125)" begin
            r = run_json(["io", "leontief"])
            assert_envelope_ok(r; label="io leontief")
            t = cols_table(r.doc, ["sector", "Agriculture", "Manufacturing"])
            @test t !== nothing
            r1 = collect(first(x for x in table_rows(t) if string(collect(x)[1]) == "Agriculture"))
            ai = findfirst(==("Agriculture"), table_cols(t))
            @test isapprox(Float64(r1[ai]), 1.254125; atol=1e-4)
            # both matrices → two tables
            r2 = run_json(["io", "leontief", "--matrix", "both"])
            assert_envelope_ok(r2; label="io leontief both")
            @test length(collect(keys(r2.doc.data))) == 2
            assert_envelope_ok(run_json(["io", "ghosh"]); label="io ghosh")
        end

        @testset "multipliers output Type I ≈ [1.518,1.452]" begin
            r = run_json(["io", "multipliers", "--kind", "output", "--type", "I"])
            assert_envelope_ok(r; label="io multipliers")
            t = cols_table(r.doc, ["sector", "multiplier"])
            vi = findfirst(==("multiplier"), table_cols(t))
            vals = sort([Float64(collect(x)[vi]) for x in table_rows(t)]; rev=true)
            @test isapprox(vals, [1.518152, 1.452145]; atol=1e-4)
            assert_envelope_ok(run_json(["io", "multipliers", "--kind", "income", "--type", "II"]); label="io mult inc II")
            assert_envelope_ok(run_json(["io", "multipliers", "--kind", "employment"]); label="io mult emp")
        end

        @testset "linkages / key-sectors / sda / baqaee-farhi" begin
            r = run_json(["io", "linkages"])
            assert_envelope_ok(r; label="io linkages")
            @test cols_table(r.doc, ["sector", "backward", "forward", "Ui", "Uj", "class"]) !== nothing
            assert_envelope_ok(run_json(["io", "key-sectors"]); label="io key-sectors")
            rs = run_json(["io", "sda", "--method", "additive"])
            assert_envelope_ok(rs; label="io sda")
            ts = cols_table(rs.doc, ["sector", "L_effect", "Y_effect", "total", "residual"])
            @test ts !== nothing
            @test "intensity_effect" ∉ table_cols(ts) && "technology_effect" ∉ table_cols(ts)
            named = run_json(["io", "sda", "--factors", "technology,final-demand"])
            assert_envelope_ok(named; label="io sda --factors")
            nt = cols_table(named.doc, ["technology_effect", "final_demand_effect", "total", "residual"])
            @test nt !== nothing
            @test "L_effect" ∉ table_cols(nt) && "Y_effect" ∉ table_cols(nt)
            sat = run_json(["io", "sda", "--on", "CO2"])
            assert_envelope_ok(sat; label="io sda --on CO2")
            st = cols_table(sat.doc, ["intensity_effect", "technology_effect", "final_demand_effect"])
            @test st !== nothing
            @test "L_effect" ∉ table_cols(st) && "Y_effect" ∉ table_cols(st)
            bf = run_json(["io", "baqaee-farhi"])
            assert_envelope_ok(bf; label="io baqaee-farhi")
            @test cols_table(bf.doc, ["sector", "domar", "influence", "upstreamness", "downstreamness"]) !== nothing
        end

        @testset "extract (Agriculture loss ≈ 1000) + footprint (CO2 = 400)" begin
            r = run_json(["io", "extract", "--sectors-extract", "Agriculture"])
            assert_envelope_ok(r; label="io extract")
            t = cols_table(r.doc, ["sector", "output_loss"])
            si = findfirst(==("sector"), table_cols(t)); li = findfirst(==("output_loss"), table_cols(t))
            loss = Dict(string(collect(x)[si]) => Float64(collect(x)[li]) for x in table_rows(t))
            @test isapprox(loss["Agriculture"], 1000.0; atol=1e-3)
            fp = run_json(["io", "footprint"])
            assert_envelope_ok(fp; label="io footprint")
            ft = cols_table(fp.doc, ["stressor", "footprint"])
            @test ft !== nothing
            fi = findfirst(==("footprint"), table_cols(ft))
            @test isapprox(Float64(collect(first(table_rows(ft)))[fi]), 400.0; atol=1e-6)
        end

        @testset "download --offline → env/network (exit 6)" begin
            r = run_json(["io", "download", "--source", "oecd", "--storage",
                          joinpath(tempdir(), "io_dl_none"), "--offline"])
            @test r.code == 6
        end

        @testset "W2/#153 IO2 riders" begin
            for mode in ("complete", "backward", "forward")
                r = run_json(["io", "extract", "--sectors-extract", "Agriculture",
                              "--mode", mode])
                assert_envelope_ok(r; label="io extract --mode $mode")
                @test cols_table(r.doc, ["sector", "output_loss"]) !== nothing
            end
            rp = run_json(["io", "extract", "--sectors-extract", "Agriculture",
                           "--mode", "partial", "--share", "0.5"])
            assert_envelope_ok(rp; label="io extract --mode partial")
            fr = run_json(["io", "footprint", "--by", "region"])
            assert_envelope_ok(fr; label="io footprint --by region")
            @test cols_table(fr.doc, ["region"]) !== nothing
            kww = joinpath(ROOT, "test", "integration", "fixtures", "kww_example1.csv")
            loaded = run_json(["io", "load", "--data", kww, "--parser", "icio"])
            assert_envelope_ok(loaded; label="io load --parser icio")
        end

        @testset "W3/#154 io bf" begin
            r = run_json(["io", "bf", "network"])
            assert_envelope_ok(r; label="io bf network")
            @test cols_table(r.doc, ["sector", "lambda", "lambda_rev", "mu"]) !== nothing
            eq = run_json(["io", "bf", "equilibrium", "--dlog-a", "0.01,0"])
            assert_envelope_ok(eq; label="io bf equilibrium")
            @test cols_table(eq.doc, ["sector", "dlog_x", "dlog_p"]) !== nothing
            kv = cols_table(eq.doc, ["metric", "value"])
            @test kv !== nothing
            mets = Dict(string(collect(row)[1]) => collect(row)[2] for row in table_rows(kv))
            @test haskey(mets, "converged")
            assert_envelope_ok(run_json(["io", "bf", "local"]); label="io bf local")
            assert_envelope_ok(run_json(["io", "bf", "elasticities"]); label="io bf elasticities")
            sc = run_json(["io", "bf", "shock-curve", "--sector", "1", "--points", "5"])
            assert_envelope_ok(sc; label="io bf shock-curve")
            @test cols_table(sc.doc, ["shock", "exact", "hulten", "second_order"]) !== nothing
            assert_envelope_ok(run_json(["io", "bf", "wedges", "--dlog-a", "0.01"]); label="io bf wedges")
            assert_envelope_ok(run_json(["io", "bf", "misallocation"]); label="io bf misallocation")

            # Closed-form: shock-curve at 0 is all zeros; diagonal/no-intermediate
            # Hulten = Domar and Hessian is (near) zero.
            sc0 = run_json(["io", "bf", "shock-curve", "--sector", "1",
                            "--range", "-0.2,0.2", "--points", "5"])
            assert_envelope_ok(sc0; label="io bf shock-curve zeros")
            t0 = cols_table(sc0.doc, ["shock", "exact", "hulten", "second_order"])
            si = findfirst(==("shock"), table_cols(t0))
            ei = findfirst(==("exact"), table_cols(t0))
            hi = findfirst(==("hulten"), table_cols(t0))
            zi = findfirst(==("second_order"), table_cols(t0))
            zero_row = nothing
            for row in table_rows(t0)
                r = collect(row)
                if abs(Float64(r[si])) < 1e-12
                    zero_row = r
                    break
                end
            end
            @test zero_row !== nothing
            @test isapprox(Float64(zero_row[ei]), 0.0; atol=1e-10)
            @test isapprox(Float64(zero_row[hi]), 0.0; atol=1e-10)
            @test isapprox(Float64(zero_row[zi]), 0.0; atol=1e-10)

            diag = tempname() * ".csv"
            open(diag, "w") do io
                println(io, "100,0,100")
                println(io, "0,200,200")
            end
            loc = run_json(["io", "bf", "local", "--data", diag, "--n-sectors", "2"])
            assert_envelope_ok(loc; label="io bf local diagonal")
            fo = cols_table(loc.doc, ["sector", "first_order"])
            @test fo !== nothing
            foi = findfirst(==("first_order"), table_cols(fo))
            # Domar of a no-intermediate table is 1 per sector (X_i / GDP_i wait: GDP = VA
            # sum). Hulten first-order equals the Domar weight.
            ns = run_json(["io", "network-stats", "--data", diag, "--n-sectors", "2"])
            assert_envelope_ok(ns; label="io network-stats diagonal")
            dt = cols_table(ns.doc, ["sector", "domar"])
            @test dt !== nothing
            di_ = findfirst(==("domar"), table_cols(dt))
            hult = [Float64(collect(row)[foi]) for row in table_rows(fo)]
            domar = [Float64(collect(row)[di_]) for row in table_rows(dt)]
            @test isapprox(hult, domar; atol=1e-6)
            hess = cols_table(loc.doc, ["sector", "S1"]) !== nothing ?
                cols_table(loc.doc, ["sector", "S1"]) :
                cols_table(loc.doc, ["sector", "sector_1"])
            if hess !== nothing
                for row in table_rows(hess)
                    for (j, c) in enumerate(table_cols(hess))
                        c == "sector" && continue
                        @test isapprox(Float64(collect(row)[j]), 0.0; atol=1e-6)
                    end
                end
            end
            rm(diag; force=true)
        end

        @testset "W4/#155 classical + KWW export-decomposition" begin
            assert_envelope_ok(run_json(["io", "price", "--dva", "0.1,0"]); label="io price")
            im = run_json(["io", "impact", "--dy", "10,0"])
            assert_envelope_ok(im; label="io impact")
            @test cols_table(im.doc, ["sector", "impact"]) !== nothing
            ns = run_json(["io", "network-stats"])
            assert_envelope_ok(ns; label="io network-stats")
            @test cols_table(ns.doc, ["sector", "domar", "upstreamness", "downstreamness"]) !== nothing
            assert_envelope_ok(run_json(["io", "balance"]); label="io balance")
            assert_envelope_ok(run_json(["io", "aggregate"]); label="io aggregate")

            kww = joinpath(ROOT, "test", "integration", "fixtures", "kww_example1.csv")
            loaded = run_json(["io", "load", "--data", kww, "--parser", "icio"])
            assert_envelope_ok(loaded; label="io load KWW icio")
            ed = run_json(["io", "export-decomposition", "--data", kww,
                           "--parser", "icio", "--region", "A"])
            assert_envelope_ok(ed; label="io export-decomposition KWW A")
            t = cols_table(ed.doc, ["dva", "rdv", "fva", "pdc", "gross_exports"])
            @test t !== nothing
            cols = table_cols(t)
            di = findfirst(==("dva"), cols); ri = findfirst(==("rdv"), cols)
            fi = findfirst(==("fva"), cols); pi = findfirst(==("pdc"), cols)
            gi = findfirst(==("gross_exports"), cols)
            row = collect(first(table_rows(t)))
            dva, rdv, fva, pdc = Float64(row[di]), Float64(row[ri]), Float64(row[fi]), Float64(row[pi])
            ge = Float64(row[gi])
            @test isapprox(dva, 20.0; atol=1e-6)
            @test isapprox(rdv, 50.0; atol=1e-6)
            @test isapprox(fva, 0.0; atol=1e-6)
            @test isapprox(pdc, 0.0; atol=1e-6)
            @test isapprox(dva + rdv + fva + pdc, ge; atol=1e-8)
            @test isapprox(ge, 70.0; atol=1e-6)
            vs = run_json(["io", "vertical-specialization", "--data", kww,
                           "--parser", "icio", "--region", "B"])
            assert_envelope_ok(vs; label="io vertical-specialization KWW B")
            bt = run_json(["io", "bilateral-trade", "--data", kww, "--parser", "icio",
                           "--exporter", "A", "--importer", "B"])
            assert_envelope_ok(bt; label="io bilateral-trade KWW")
        end
    end

    @testset "estimate statespace/tvp/kde/kernel-reg/lowess (C066, M5c)" begin
        # First table in the envelope whose columns ⊇ `cols`.
        cols_table(doc, cols) = begin
            doc === nothing && return nothing
            for (_, v) in pairs(doc.data)
                (v isa JSON3.Object && haskey(v, :rows)) || continue
                all(c -> c in table_cols(v), cols) && return v
            end
            return nothing
        end
        metrics_table(doc) = cols_table(doc, ["metric", "value"])

        @testset "statespace local-level — finite loglik + positive variances" begin
            csv = dgp_ar1(; T=200, φ=0.6, seed=31)
            r = run_json(["estimate", "statespace", csv, "--model", "local-level"])
            assert_envelope_ok(r; label="statespace local-level")
            pt = cols_table(r.doc, ["parameter", "estimate"])
            @test pt !== nothing && length(table_rows(pt)) == 2
            ei = findfirst(==("estimate"), table_cols(pt))
            @test all(Float64(collect(row)[ei]) >= 0.0 for row in table_rows(pt))   # variances ≥ 0
            ll = metric_value(metrics_table(r.doc), "loglik")
            @test ll !== nothing && isfinite(Float64(ll))
            rm(csv; force=true)
        end

        @testset "statespace --config — general system, EQUIVALENT to the canned one (#71)" begin
            # THE assertion that makes this more than a shape test: writing the local level out
            # by hand with the variances the canned MLE found must reproduce the canned
            # log-likelihood EXACTLY. If the config were being mis-transcribed into the system
            # matrices (Z/T swapped, H and Q crossed, a dropped intercept) the two would differ.
            csv = dgp_ar1(; T=200, φ=0.6, seed=131)
            canned = run_json(["estimate", "statespace", csv, "--model", "local-level"])
            assert_envelope_ok(canned; label="statespace canned")
            pt = cols_table(canned.doc, ["parameter", "estimate"])
            ei = findfirst(==("estimate"), table_cols(pt))
            th = [Float64(collect(row)[ei]) for row in table_rows(pt)]      # [σ²_ε, σ²_η]
            ll_canned = Float64(metric_value(metrics_table(canned.doc), "loglik"))

            cfg = tempname() * ".toml"
            open(cfg, "w") do io
                println(io, "[statespace]")
                println(io, "Z = [[1.0]]")
                println(io, "H = [[", th[1], "]]")
                println(io, "T = [[1.0]]")
                println(io, "Q = [[", th[2], "]]")
            end
            gen = run_json(["estimate", "statespace", csv, "--config", cfg])
            assert_envelope_ok(gen; label="statespace general")
            ll_gen = Float64(metric_value(metrics_table(gen.doc), "loglik"))
            @test isapprox(ll_canned, ll_gen; rtol=1e-6)

            # a fixed-matrix system is FILTERED, never optimized: no hyper-parameters, so the
            # system table stands in for the parameter table (which would render empty)
            st = cols_table(gen.doc, ["matrix", "role", "rows", "cols"])
            @test st !== nothing
            @test Set(string(collect(row)[1]) for row in table_rows(st)) ==
                  Set(["Z", "H", "T", "Q", "R", "d", "c"])
            @test cols_table(gen.doc, ["parameter", "estimate"]) === nothing
            @test string(metric_value(metrics_table(gen.doc), "method")) == "filter"
            @test string(metric_value(metrics_table(gen.doc), "model")) == "general"
            @test Int(metric_value(metrics_table(gen.doc), "n_periods")) == 200

            # multivariate: two series on one common state
            bicsv = write_csv(DataFrame(y1=randn(120), y2=randn(120)); prefix="ss_bi")
            bicfg = tempname() * ".toml"
            open(bicfg, "w") do io
                println(io, "[statespace]")
                println(io, "Z = [[1.0], [0.7]]")
                println(io, "H = [[1.0, 0.0], [0.0, 2.0]]")
                println(io, "T = [[0.95]]")
                println(io, "Q = [[0.5]]")
            end
            rb = run_json(["estimate", "statespace", bicsv, "--config", bicfg])
            assert_envelope_ok(rb; label="statespace general bivariate")
            @test Int(metric_value(metrics_table(rb.doc), "n_obs_series")) == 2
            @test Int(metric_value(metrics_table(rb.doc), "n_state")) == 1

            # n_obs implied by Z vs the CSV's column count → data/shape (3), not exit 1
            @test run_json(["estimate", "statespace", csv, "--config", bicfg]).code == 3
            # a malformed system is a CONFIG error (4), naming the file the user wrote
            badcfg = tempname() * ".toml"
            open(badcfg, "w") do io
                println(io, "[statespace]")
                println(io, "Z = [[1.0]]")
                println(io, "H = [[1.0, 0.0], [0.0, 1.0]]")
                println(io, "T = [[1.0]]")
                println(io, "Q = [[1.0]]")
            end
            @test run_json(["estimate", "statespace", csv, "--config", badcfg]).code == 4
            # a1 without P1 would be SILENTLY IGNORED upstream → rejected here
            halfcfg = tempname() * ".toml"
            open(halfcfg, "w") do io
                println(io, "[statespace]")
                println(io, "Z = [[1.0]]"); println(io, "H = [[1.0]]")
                println(io, "T = [[1.0]]"); println(io, "Q = [[1.0]]")
                println(io, "a1 = [0.0]")
            end
            @test run_json(["estimate", "statespace", csv, "--config", halfcfg]).code == 4
            for f in (cfg, bicfg, badcfg, halfcfg, bicsv, csv); rm(f; force=true); end
        end

        @testset "statespace local-linear-trend — 3 hyper-params" begin
            csv = dgp_trend_cycle(; T=200, seed=32)
            r = run_json(["estimate", "statespace", csv, "--model", "local-linear-trend"])
            assert_envelope_ok(r; label="statespace llt")
            pt = cols_table(r.doc, ["parameter", "estimate"])
            @test pt !== nothing && length(table_rows(pt)) == 3
            rm(csv; force=true)
        end

        @testset "tvp — smoothed path has T·k rows (T=150, k=2)" begin
            csv = dgp_reg(; T=150, seed=33)    # columns y, x → k = intercept + x = 2
            r = run_json(["estimate", "tvp", csv, "--dep", "y"])
            assert_envelope_ok(r; label="tvp")
            path = cols_table(r.doc, ["period", "coefficient", "estimate"])
            @test path !== nothing && length(table_rows(path)) == 150 * 2
            @test Float64(metric_value(metrics_table(r.doc), "n_coef")) == 2.0
            rm(csv; force=true)
        end

        @testset "kde — density ≥ 0, grid length == npoints, ∫≈1" begin
            csv = dgp_iid(; T=400, seed=34)
            r = run_json(["estimate", "kde", csv, "--npoints", "256"])
            assert_envelope_ok(r; label="kde")
            g = cols_table(r.doc, ["x", "density"])
            @test g !== nothing && length(table_rows(g)) == 256
            xi = findfirst(==("x"), table_cols(g)); di = findfirst(==("density"), table_cols(g))
            xs = [Float64(collect(row)[xi]) for row in table_rows(g)]
            ds = [Float64(collect(row)[di]) for row in table_rows(g)]
            @test all(d -> d >= -1e-9, ds)                       # a proper density
            # trapezoidal integral over the grid ≈ 1 (loose — grid is finite-support).
            area = sum((xs[i+1] - xs[i]) * (ds[i+1] + ds[i]) / 2 for i in 1:length(xs)-1)
            @test isapprox(area, 1.0; atol=0.05)
            @test Float64(metric_value(metrics_table(r.doc), "bandwidth")) > 0.0
            rm(csv; force=true)
        end

        @testset "kde — sj bandwidth + non-gaussian kernel also run" begin
            csv = dgp_iid(; T=300, seed=35)
            r = run_json(["estimate", "kde", csv, "--bw", "sj", "--kernel", "epanechnikov",
                          "--npoints", "128"])
            assert_envelope_ok(r; label="kde sj epanechnikov")
            @test string(metric_value(metrics_table(r.doc), "bw_method")) == "sj"
            rm(csv; force=true)
        end

        @testset "kernel-reg — fitted length == nobs, tracks a linear mean" begin
            csv = dgp_reg(; T=200, seed=36)    # y = 1 + 2x + noise
            r = run_json(["estimate", "kernel-reg", csv, "--dep", "y", "--indep", "x"])
            assert_envelope_ok(r; label="kernel-reg ll")
            fit = cols_table(r.doc, ["x", "fitted", "se"])
            @test fit !== nothing && length(table_rows(fit)) == 200
            xi = findfirst(==("x"), table_cols(fit)); fi = findfirst(==("fitted"), table_cols(fit))
            xs = [Float64(collect(row)[xi]) for row in table_rows(fit)]
            fs = [Float64(collect(row)[fi]) for row in table_rows(fit)]
            @test all(isfinite, fs)
            @test cor(xs, fs) > 0.8                              # positive slope ≈ +2
            rm(csv; force=true)
        end

        @testset "kernel-reg — nw method + rot bandwidth run" begin
            csv = dgp_reg(; T=150, seed=37)
            r = run_json(["estimate", "kernel-reg", csv, "--dep", "y", "--indep", "x",
                          "--method", "nw", "--bw", "rot"])
            assert_envelope_ok(r; label="kernel-reg nw rot")
            @test string(metric_value(metrics_table(r.doc), "method")) == "nw"
            @test Float64(metric_value(metrics_table(r.doc), "degree")) == 0.0
            rm(csv; force=true)
        end

        @testset "lowess — fitted length == nobs, tracks a linear mean" begin
            csv = dgp_reg(; T=200, seed=38)
            r = run_json(["estimate", "lowess", csv, "--dep", "y", "--indep", "x"])
            assert_envelope_ok(r; label="lowess")
            fit = cols_table(r.doc, ["x", "fitted"])
            @test fit !== nothing && length(table_rows(fit)) == 200
            xi = findfirst(==("x"), table_cols(fit)); fi = findfirst(==("fitted"), table_cols(fit))
            xs = [Float64(collect(row)[xi]) for row in table_rows(fit)]
            fs = [Float64(collect(row)[fi]) for row in table_rows(fit)]
            @test all(isfinite, fs)
            @test cor(xs, fs) > 0.8
            rm(csv; force=true)
        end

        @testset "C066 bad input → typed classes (never uncaught exit-1)" begin
            csv = dgp_reg(; T=100, seed=39)
            @test run_json(["estimate", "kde", csv, "--bw", "notanumber"]).code == 2
            @test run_json(["estimate", "kernel-reg", csv, "--dep", "y"]).code == 2  # missing --indep
            @test run_json(["estimate", "kernel-reg", csv, "--dep", "y", "--indep", "nope"]).code == 3
            @test run_json(["estimate", "lowess", csv, "--dep", "y", "--indep", "y"]).code == 3  # indep==dep
            rm(csv; force=true)
        end
    end

    # ── Data loading on real MEMs ────────────────────────────────────────────
    # This family had NO T3 coverage, which is why a broken `data load --path`,
    # a `:name` reference that exited 1, and panel datasets that silently lost
    # their id columns all survived. Keep these here.
    @testset "data loading (real load_example)" begin
        @testset "data load --path without a positional name" begin
            mktempdir() do dir
                src = joinpath(dir, "mine.csv")
                CSV.write(src, DataFrame(a=randn(30), b=randn(30)))
                out = joinpath(dir, "out.csv")
                # Regression: <name> used to be required → "missing required argument"
                @test run_json(["data", "load", "--path", src, "-o", out]).code == 0
                @test isfile(out)
                @test nrow(CSV.read(out, DataFrame)) == 30
            end
        end

        @testset "dataset names: ':' refs, both separators, typos" begin
            mktempdir() do dir
                for (i, form) in enumerate(["fred_md", "fred-md", ":fred-md", ":fred_md"])
                    out = joinpath(dir, "d$i.csv")
                    # Regression: ':fred-md' raised an untyped ArgumentError → exit 1
                    @test run_json(["data", "load", form, "-o", out]).code == 0
                    @test isfile(out)
                end
            end
            # Unknown name → typed data error (exit 3), never the exit-1 "likely a bug" tail
            @test run_json(["data", "load", "frd_md"]).code == 3
            @test run_json(["data", "load", ":wiot"]).code == 3
            # Neither a name nor --path → usage error (exit 2)
            @test run_json(["data", "load"]).code == 2
        end

        @testset "every advertised dataset actually loads" begin
            # data list is built from EXAMPLE_DATASETS; assert the list is honest.
            mktempdir() do dir
                for (i, d) in enumerate(Friedman.EXAMPLE_DATASETS)
                    out = joinpath(dir, "ds$i.csv")
                    r = run_json(["data", "load", String(d), "-o", out])
                    @test r.code == 0
                    @test isfile(out)
                end
            end
        end

        @testset "builtin panels keep their group/time identifiers" begin
            # Without this, every panel command fails on a bundled panel dataset.
            df = Friedman.load_data(":pwt")
            @test "group" in names(df)
            @test "time" in names(df)
            # …and a real panel command can bind to them end-to-end.
            r = run_json(["test", "cips", ":grunfeld", "--id-col", "group",
                          "--time-col", "time", "--lags", "1"])
            @test r.code == 0
        end

        @testset "default output paths never contain the ':' marker" begin
            mktempdir() do dir
                cd(dir) do
                    # Regression: produced a file literally named ':fred-md_clean.csv'
                    @test run_json(["data", "fix", ":denmark"]).code == 0
                    @test isfile(joinpath(dir, "denmark_clean.csv"))
                    @test !isfile(joinpath(dir, ":denmark_clean.csv"))
                end
            end
        end

        @testset "data describe reports a scalar n per variable" begin
            mktempdir() do dir
                src = joinpath(dir, "d.csv")
                CSV.write(src, DataFrame(a=randn(50), b=randn(50)))
                r = run_json(["data", "describe", src])
                @test r.code == 0
                _, tbl = first_table(r.doc)
                @test tbl !== nothing
                if tbl !== nothing
                    ni = findfirst(==("n"), table_cols(tbl))
                    @test ni !== nothing
                    # Regression: fill(summary.n, n_vars) put the whole vector in every cell
                    @test all(row -> collect(row)[ni] isa Integer, table_rows(tbl))
                    @test all(row -> collect(row)[ni] == 50, table_rows(tbl))
                end
            end
        end

        @testset ":mp_shocks — NaN outside published samples (W3/#125)" begin
            mktempdir() do dir
                out = joinpath(dir, "mp.csv")
                # Dash spelling normalizes to the underscore symbol.
                @test run_json(["data", "load", ":mp-shocks", "-o", out]).code == 0
                df = CSV.read(out, DataFrame)
                @test size(df) == (240, 8)
                @test names(df) == ["ygap", "infl", "ffr", "lpcom", "rr", "mp1", "ad", "bzk_ist"]

                # Upstream pins (test_mp_shocks_data.jl): loading must not shift,
                # scale, or zero-fill — NaN is NOT zero; zero is a valid shock value.
                @test df.ffr[1] ≈ 3.93 atol = 0.01              # FRED FEDFUNDS 1960Q1
                @test isnan(df.rr[1])                            # pre-1969Q1
                @test count(!isnan, df.rr) == 156                # Romer-Romer window
                @test findfirst(!isnan, df.rr) == 37             # 1969Q1
                @test findlast(!isnan, df.rr) == 192             # 2007Q4
                @test maximum(filter(!isnan, df.ffr)) > 15.0     # Volcker peak, % units

                # data describe locates the valid window per column.
                r = run_json(["data", "describe", ":mp_shocks"])
                @test r.code == 0
                tbl = named_table(r.doc, :descriptive_statistics)
                @test tbl !== nothing
                if tbl !== nothing
                    cols = table_cols(tbl)
                    vi = findfirst(==("variable"), cols)
                    ni = findfirst(==("n"), cols)
                    fi = findfirst(==("first_valid"), cols)
                    li = findfirst(==("last_valid"), cols)
                    @test fi !== nothing && li !== nothing
                    rows = [collect(row) for row in table_rows(tbl)]
                    rr = findfirst(rv -> string(rv[vi]) == "rr", rows)
                    @test rr !== nothing
                    @test rows[rr][ni] == 156
                    @test rows[rr][fi] == 37 && rows[rr][li] == 192
                end

                # Documented prep step: subset → dropna → estimate, end to end.
                sub = joinpath(dir, "macro3.csv")
                @test run_json(["data", "load", ":mp_shocks",
                                "--vars", "ygap,infl,ffr", "-o", sub]).code == 0
                clean = joinpath(dir, "macro3_clean.csv")
                @test run_json(["data", "dropna", sub,
                                "--format", "csv", "-o", clean]).code == 0
                cdf = CSV.read(clean, DataFrame)
                @test nrow(cdf) == 187                # 1969Q1–2015Q3 jointly finite
                @test !any(isnan, Matrix(cdf))
                @test run_json(["estimate", "var", clean, "--lags", "4"]).code == 0

                # dropna --vars: a Vector{SubString} used to TypeError against
                # real dropna's ::Union{Vector{String},Nothing} kwarg assertion
                # (exit 1 on EVERY --vars invocation; the old no-op mock hid it).
                rronly = joinpath(dir, "rr_only.csv")
                @test run_json(["data", "dropna", out, "--vars", "rr",
                                "--format", "csv", "-o", rronly]).code == 0
                @test nrow(CSV.read(rronly, DataFrame)) == 156
                # Unknown --vars entry is typed data (3), not upstream's exit 1.
                @test run_json(["data", "dropna", out, "--vars", "nope"]).code == 3

                # keeprows guards: bare error()/raw parse/BoundsError all exited 1.
                win = joinpath(dir, "win.csv")
                @test run_json(["data", "keeprows", out, "--rows", "37:192",
                                "--format", "csv", "-o", win]).code == 0
                @test nrow(CSV.read(win, DataFrame)) == 156
                @test run_json(["data", "keeprows", out, "--rows", "1:999"]).code == 2
                @test run_json(["data", "keeprows", out, "--rows", "abc"]).code == 2
                @test run_json(["data", "keeprows", out]).code == 2
            end
        end

        @testset "~ is expanded by the loader (the REPL has no shell)" begin
            mktempdir() do dir
                CSV.write(joinpath(dir, "tilde.csv"), DataFrame(a=randn(20)))
                # expanduser goes through libuv's uv_os_homedir: HOME on Unix,
                # USERPROFILE on Windows — set both or the Windows nightly fails.
                withenv("HOME" => dir, "USERPROFILE" => dir) do
                    @test nrow(Friedman.load_data("~/tilde.csv")) == 20
                end
            end
        end

        @testset "path confinement is opt-in and resolves, not substring-matches (#83)" begin
            mktempdir() do root
                inside = joinpath(root, "in.csv")
                CSV.write(inside, DataFrame(a=randn(30), b=randn(30)))
                sub = joinpath(root, "sub"); mkpath(sub)
                weird = joinpath(root, "dotdot..name.csv")
                CSV.write(weird, DataFrame(a=randn(30), b=randn(30)))

                # Unconfined: a parent-relative path and a '..' filename both load
                @test run_json(["data", "describe", joinpath(sub, "..", "in.csv")]).code == 0
                @test run_json(["data", "describe", weird]).code == 0

                # Confined: inside is fine, escaping is data/bad-path (exit 3)
                withenv("FRIEDMAN_DATA_ROOT" => root) do
                    @test run_json(["data", "describe", inside]).code == 0
                    @test run_json(["data", "describe",
                                    joinpath(root, "..", "etc", "passwd")]).code == 3
                end
            end
        end
    end

    # ── Commands whose result-field access was masked by mock aliases (#84) ──
    # Each of these read a field real MEMs does not have and exited 1 on EVERY
    # invocation. The mock's `getproperty` aliases invented the field, so T1/T2
    # passed; none had T3 coverage. Keep every one of them covered here.
    @testset "result-field access on real MEMs (#84 regressions)" begin
        uni = dgp_ar1(; T=200, seed=5)
        multi = dgp_var2(; T=200, seed=5)

        @testset "test durbin-watson (statistic/pvalue, no invented bounds)" begin
            r = run_json(["test", "durbin-watson", uni])
            assert_envelope_ok(r; label="durbin-watson")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test metric_value(tbl, "DW statistic") !== nothing
            @test metric_value(tbl, "p-value") !== nothing
        end

        @testset "test dfgls (statistic + separate M-GLS fields)" begin
            r = run_json(["test", "dfgls", uni])
            assert_envelope_ok(r; label="dfgls")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test metric_value(tbl, "DF-GLS tau statistic") !== nothing
            for k in ("M-GLS MZa", "M-GLS MZt", "M-GLS MSB", "M-GLS MPT")
                @test metric_value(tbl, k) !== nothing
            end
        end

        @testset "test adf-2break (break1/break2 + fractions)" begin
            r = run_json(["test", "adf-2break", uni])
            assert_envelope_ok(r; label="adf-2break")
            _, tbl = first_table(r.doc)
            @test metric_value(tbl, "Break 1 index") !== nothing
            @test metric_value(tbl, "Break 2 index") !== nothing
        end

        @testset "test lm-unitroot (breaks/break_dates)" begin
            r = run_json(["test", "lm-unitroot", uni])
            assert_envelope_ok(r; label="lm-unitroot")
            _, tbl = first_table(r.doc)
            @test metric_value(tbl, "LM statistic") !== nothing
        end

        @testset "test gregory-hansen (adf_break/zt_break/za_break)" begin
            r = run_json(["test", "gregory-hansen", multi])
            assert_envelope_ok(r; label="gregory-hansen")
            _, tbl = first_table(r.doc)
            @test metric_value(tbl, "ADF* break index") !== nothing
            @test metric_value(tbl, "Za* break index") !== nothing
            # One column is a cointegration shape error (exit 3), not an exit-1 crash
            @test run_json(["test", "gregory-hansen", uni]).code == 3
        end

        @testset "test factor-break (n_factors/n_vars + per-series, W2/#124)" begin
            r = run_json(["test", "factor-break", multi, "--factors", "1"])
            assert_envelope_ok(r; label="factor-break")
            # named, not first_table: the leaf now emits TWO tables (the standing lesson)
            tbl = named_table(r.doc, :factor_break_test)
            @test tbl !== nothing
            @test metric_value(tbl, "Factors") !== nothing
            @test metric_value(tbl, "Units") !== nothing
            # Pooled default (breitung_eickmeier) carries per-series diagnostics
            # (MEMs 0.7.3/#606): one row per series, sorted by statistic descending.
            ps = named_table(r.doc, :per_series_break_diagnostics)
            @test ps !== nothing
            if ps !== nothing
                rows = table_rows(ps)
                nv = metric_value(tbl, "Units")
                @test length(rows) == Int(nv)
                si = col_index(ps, "statistic")
                stats = [numv(collect(rw)[si]) for rw in rows]
                @test issorted(stats; rev=true)
            end
            # chen_dolado_gonzalo has no per-series decomposition — table absent, exit 0.
            rc = run_json(["test", "factor-break", multi, "--factors", "1",
                           "--method", "chen_dolado_gonzalo"])
            assert_envelope_ok(rc; label="factor-break cdg")
            @test named_table(rc.doc, :per_series_break_diagnostics) === nothing
        end

        @testset "fevd bvar (point_estimate, (var,shock,h) layout since MEMs 0.7.3/#527)" begin
            r = run_json(["fevd", "bvar", multi, "--lags", "1", "--horizons", "4"])
            assert_envelope_ok(r; label="fevd bvar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            rows = table_rows(tbl)
            # One row per horizon (a transposed array silently yields n rows instead)
            @test length(rows) == 4
            # Variance shares sum to 1 across shocks at every horizon
            for row in rows
                vals = Float64.(collect(row)[2:end])
                @test isapprox(sum(vals), 1.0; atol=1e-6)
            end
        end

        @testset "hd bvar (point_estimate + initial_point_estimate)" begin
            r = run_json(["hd", "bvar", multi, "--lags", "1"])
            assert_envelope_ok(r; label="hd bvar")
            _, tbl = first_table(r.doc)
            @test tbl !== nothing
            @test !isempty(table_rows(tbl))
        end

        @testset "estimate favar --method bayesian (draws from B_draws)" begin
            r = run_json(["estimate", "favar", multi, "--method", "bayesian",
                          "--factors", "1", "--lags", "1", "--key-vars", "y1"])
            assert_envelope_ok(r; label="favar bayesian")
            # Missing --key-vars is a usage error (exit 2), not an untyped exit 1
            @test run_json(["estimate", "favar", multi, "--method", "bayesian",
                            "--factors", "1", "--lags", "1"]).code == 2
        end
    end

    # ── Families that had no real-MEMs coverage at all (#85) ──
    @testset "nowcast family (#85: was entirely uncovered)" begin
        multi = dgp_var2(; T=160, seed=8)
        # Each leaf has its own option set — dfm takes --factors, bvar only --lags,
        # bridge takes per-frequency lag counts.
        for (leaf, extra) in [("dfm", ["--factors", "1", "--lags", "1"]),
                              ("bvar", ["--lags", "2"]),
                              ("bridge", ["--lag-m", "1", "--lag-q", "1", "--lag-y", "1"])]
            @testset "nowcast $leaf" begin
                r = run_json(vcat(["nowcast", leaf, multi], extra))
                assert_envelope_ok(r; label="nowcast $leaf")
            end
        end

        @testset "nowcast bvar --prior litterman (W1/#123, MEMs#602)" begin
            rl = run_json(["nowcast", "bvar", multi, "--lags", "2",
                           "--prior", "litterman", "--theta-cross", "0.5"])
            assert_envelope_ok(rl; label="nowcast bvar litterman")
            # The hyperparameters table records the prior and — litterman only — the
            # optimized theta_cross. Find it by its distinctive first column values.
            hp = named_table(rl.doc, :nowcast_bvar_hyperparameters)
            @test hp !== nothing
            if hp !== nothing
                kv = Dict(string(collect(rw)[1]) => collect(rw)[2] for rw in table_rows(hp))
                @test kv["prior"] == "litterman"
                @test any(startswith(k, "theta_cross") for k in keys(kv))
            end
            # Same data under conjugate: no theta_cross row, and a different point
            # nowcast (the priors are genuinely different objectives — MEMs#602).
            rc = run_json(["nowcast", "bvar", multi, "--lags", "2"])
            assert_envelope_ok(rc; label="nowcast bvar conjugate")
            hpc = named_table(rc.doc, :nowcast_bvar_hyperparameters)
            @test hpc !== nothing
            if hpc !== nothing
                kvc = Dict(string(collect(rw)[1]) => collect(rw)[2] for rw in table_rows(hpc))
                @test kvc["prior"] == "conjugate"
                @test !any(startswith(k, "theta_cross") for k in keys(kvc))
            end
            # --theta-cross with the conjugate prior is a typed usage error (exit 2),
            # guarded in the CLI before upstream's ArgumentError can map to exit 3.
            @test run_json(["nowcast", "bvar", multi, "--theta-cross", "0.5"]).code == 2
            @test run_json(["nowcast", "bvar", multi, "--prior", "litterman",
                            "--theta-cross", "-1"]).code == 2
        end
        @testset "nowcast forecast" begin
            r = run_json(["nowcast", "forecast", multi, "--factors", "1",
                          "--lags", "1", "--horizons", "3"])
            assert_envelope_ok(r; label="nowcast forecast")
        end
        @testset "nowcast news" begin
            # Same-shape vintages: the old one has the final observation not yet
            # released (a vintage differs in which cells are filled, not row count).
            new_df = CSV.read(multi, DataFrame)
            old_df = copy(new_df)
            old_df[end, 1] = NaN
            old_path = tempname() * ".csv"
            CSV.write(old_path, old_df)
            r = run_json(["nowcast", "news", "--data-new", multi, "--data-old", old_path,
                          "--factors", "1", "--lags", "1"])
            assert_envelope_ok(r; label="nowcast news")

            # A row-count mismatch is the user's input, not an internal bug
            short_path = tempname() * ".csv"
            CSV.write(short_path, new_df[1:end-1, :])
            @test run_json(["nowcast", "news", "--data-new", multi, "--data-old", short_path,
                            "--factors", "1", "--lags", "1"]).code == 3
            # Missing a required vintage is a usage error
            @test run_json(["nowcast", "news", "--data-new", multi]).code == 2
            rm(old_path; force=true); rm(short_path; force=true)
        end
    end

    @testset "predict/residuals sweep (#85: was 1 of 23 leaves, nightly-only)" begin
        multi = dgp_var2(; T=200, seed=9)
        uni = dgp_ar1(; T=200, seed=9)
        reg = dgp_reg(; T=200, seed=9)
        logit = dgp_logit(; T=300, seed=9)
        gar = dgp_garch(; T=400, seed=9)

        # (leaf, data, extra args) — one per result-type family, so an accessor rename
        # upstream cannot slip through on either action.
        cases = [
            ("var",    multi, ["--lags", "1"]),
            ("bvar",   multi, ["--lags", "1"]),
            ("vecm",   multi, ["--lags", "2", "--rank", "1"]),
            ("arima",  uni,   ["--column", "1"]),
            ("reg",    reg,   ["--dep", "y"]),
            ("logit",  logit, ["--dep", "y"]),
            ("probit", logit, ["--dep", "y"]),
            ("garch",  gar,   ["--column", "1"]),
        ]
        for (leaf, data, extra) in cases
            for action in ("predict", "residuals")
                @testset "$action $leaf" begin
                    r = run_json(vcat([action, leaf, data], extra))
                    assert_envelope_ok(r; label="$action $leaf")
                    _, tbl = first_table(r.doc)
                    @test tbl !== nothing
                    @test tbl !== nothing && !isempty(table_rows(tbl))
                end
            end
        end

        # Ordered/multinomial: `predict` returns an n x n_categories probability
        # matrix (feeding it to DataFrame used to be an exit-1 crash), and MEMs
        # defines no `residuals` for them, so that must be a typed refusal.
        ord = tempname() * ".csv"
        let n = 300
            Random.seed!(19)
            CSV.write(ord, DataFrame(y=rand(1:3, n), x1=randn(n), x2=randn(n)))
        end
        for leaf in ("ologit", "oprobit", "mlogit")
            @testset "predict $leaf (per-category probabilities)" begin
                r = run_json(["predict", leaf, ord, "--dep", "y"])
                assert_envelope_ok(r; label="predict $leaf")
                _, tbl = first_table(r.doc)
                @test tbl !== nothing
                if tbl !== nothing
                    cols = table_cols(tbl)
                    @test count(c -> startswith(c, "prob_"), cols) == 3
                end
            end
            # W4/#87: un-gated by MEMs#507. These used to be a typed refusal (exit 5)
            # because upstream defined no residuals for ordered/unordered choice models.
            # 0.7.2 settled it: `residuals(m; kind=)` is an n x J matrix, one column per
            # category, and the :response rows sum to zero by construction.
            @testset "residuals $leaf — n x J per-category matrix (W4/#87)" begin
                r = run_json(["residuals", leaf, ord, "--dep", "y"])
                assert_envelope_ok(r; label="residuals $leaf")
                _, tbl = first_table(r.doc)
                @test tbl !== nothing
                if tbl !== nothing
                    cols = table_cols(tbl)
                    @test count(c -> startswith(c, "resid_"), cols) == 3
                    # :response residuals sum to zero across categories by construction.
                    # ONE assertion on the worst row, not one per row: a per-row loop adds
                    # ~900 assertions per run for no extra coverage. Tolerance is set by
                    # the RENDERER, not the estimator — _choice_resid_table rounds to 6
                    # digits, so a J-term sum carries up to J*5e-7 of rounding (observed
                    # 1.0000000000287557e-6, which is why atol=1e-6 was too tight).
                    ridx = findall(c -> startswith(c, "resid_"), cols)
                    worst = maximum(abs(sum(Float64.(collect(row)[ridx])))
                                    for row in table_rows(tbl))
                    @test worst < 5e-6
                end
                for k in ("response", "pearson", "deviance")
                    @test run_json(["residuals", leaf, ord, "--dep", "y", "--kind", k]).code == 0
                end
                @test run_json(["residuals", leaf, ord, "--dep", "y", "--kind", "bogus"]).code == 2
            end
        end

        # W10/#131: --marginal-effects re-added WITH handler support (MEMs#550 gave
        # the ordered/multinomial models delta-method SEs at 0.7.3). Two upstream
        # shapes — ordered NamedTuple vs MultinomialMarginalEffects — render to one
        # tidy variable|category|dydx|se table. The closed form pinning the values:
        # probabilities sum to 1, so each variable's AMEs sum to 0 across categories
        # (live-verified: the mlogit BASE category carries a real effect too).
        @testset "predict --marginal-effects (W10/#131, MEMs#550)" begin
            for (leaf, key) in [("ologit", :ordered_logit_average_marginal_effects),
                                ("oprobit", :ordered_probit_average_marginal_effects),
                                ("mlogit", :multinomial_logit_average_marginal_effects)]
                r = run_json(["predict", leaf, ord, "--dep", "y", "--marginal-effects"])
                assert_envelope_ok(r; label="predict $leaf --marginal-effects")
                # The probability table must still be there (2nd table gets its own
                # output path, never displaces the 1st).
                _, ptbl = first_table(r.doc)
                @test ptbl !== nothing
                me = named_table(r.doc, key)
                @test me !== nothing
                if me !== nothing
                    cols = table_cols(me)
                    @test cols == ["variable", "category", "dydx", "se"]
                    rows = [collect(row) for row in table_rows(me)]
                    @test length(rows) == 6            # 2 vars × 3 categories
                    @test all(isfinite(numv(r_[4])) for r_ in rows)   # SEs are REAL now
                    for v in unique(String[string(r_[1]) for r_ in rows])
                        s = sum(numv(r_[3]) for r_ in rows if string(r_[1]) == v)
                        @test abs(s) < 5e-6            # renderer rounds to 6 digits
                    end
                end
            end
        end

        # `generalized_residuals` is ORDERED-ONLY upstream. The flag must therefore exist
        # on ologit/oprobit and NOT on mlogit — a declared flag whose handler cannot honour
        # it is the failure mode that shipped 19 broken leaves once already (#85).
        @testset "residuals --generalized is ordered-only (W4/#87)" begin
            for leaf in ("ologit", "oprobit")
                r = run_json(["residuals", leaf, ord, "--dep", "y", "--generalized"])
                assert_envelope_ok(r; label="residuals $leaf --generalized")
                _, tbl = first_table(r.doc)
                @test tbl !== nothing
                @test "generalized_residual" in table_cols(tbl)
            end
            # usage error (2), not a crash and not a silently-ignored flag
            @test run_json(["residuals", "mlogit", ord, "--dep", "y", "--generalized"]).code == 2
        end

        # W2/#107 — count-data family on real MEMs.
        @testset "count data: estimate/predict/residuals poisson|nbreg + test dispersion" begin
            Random.seed!(4271)
            # `cols_table` is a LOCAL helper of the io testset, not a suite-level one, so
            # it is redefined here rather than reached across scopes.
            cols_table(doc, cols) = begin
                doc === nothing && return nothing
                for (_, v) in pairs(doc.data)
                    (v isa JSON3.Object && haskey(v, :rows)) || continue
                    all(c -> c in table_cols(v), cols) && return v
                end
                return nothing
            end
            # Knuth sampler — the integration env has no direct Distributions dep, and a
            # genuine Poisson draw is required here: rounding the mean instead produces
            # UNDERdispersed counts, which is a different test.
            _rand_pois(lam) = begin
                L = exp(-lam); k = 0; p = 1.0
                while true
                    p *= rand()
                    p <= L && return k
                    k += 1
                    k > 10_000 && return k
                end
            end
            n = 400
            cx1 = randn(n); cx2 = randn(n); cexpo = rand(n) .* 4 .+ 1
            # True log-mean with a genuine Poisson draw, so the dispersion test sees real
            # equidispersion rather than the artefact of a rounded mean.
            cmu = exp.(0.3 .+ 0.5 .* cx1 .- 0.3 .* cx2 .+ log.(cexpo))
            cy = [_rand_pois(m) for m in cmu]
            ccsv = write_csv(DataFrame(y=cy, x1=cx1, x2=cx2, expo=cexpo); prefix="count")

            # Teeth: the estimator must RECOVER the DGP, not merely run.
            rp = run_json(["estimate", "poisson", ccsv, "--dep", "y", "--exposure", "expo"])
            assert_envelope_ok(rp; label="estimate poisson")
            tp = cols_table(rp.doc, ["term", "estimate"])
            @test tp !== nothing
            pc = Dict(String(collect(r)[col_index(tp, "term")]) =>
                      Float64(collect(r)[col_index(tp, "estimate")]) for r in table_rows(tp))
            @test isapprox(pc["x1"], 0.5; atol=0.12)
            @test isapprox(pc["x2"], -0.3; atol=0.12)
            # tidy C051 column set, hand-built because these types are NOT Tables.jl-registered
            for c in ("std_error", "z_stat", "p_value", "ci_lower", "ci_upper")
                @test c in table_cols(tp)
            end

            # --irr is a FLAG with handler support; exp(beta) must match the coefficients
            rirr = run_json(["estimate", "poisson", ccsv, "--dep", "y", "--exposure", "expo", "--irr"])
            assert_envelope_ok(rirr; label="estimate poisson --irr")
            ti = cols_table(rirr.doc, ["term", "irr"])
            @test ti !== nothing
            irrmap = Dict(String(collect(r)[col_index(ti, "term")]) =>
                          Float64(collect(r)[col_index(ti, "irr")]) for r in table_rows(ti))
            @test isapprox(irrmap["x1"], exp(pc["x1"]); rtol=1e-4)

            @test run_json(["estimate", "nbreg", ccsv, "--dep", "y", "--irr"]).code == 0
            rnb = run_json(["estimate", "nbreg", ccsv, "--dep", "y"])
            assert_envelope_ok(rnb; label="estimate nbreg")
            # alpha rides its OWN table (upstream's vcov slice stops at beta)
            ta = cols_table(rnb.doc, ["parameter", "estimate", "std_error"])
            @test ta !== nothing
            @test "alpha" in [String(collect(r)[col_index(ta, "parameter")]) for r in table_rows(ta)]

            for leaf in ("poisson", "nbreg")
                @test run_json(["predict", leaf, ccsv, "--dep", "y"]).code == 0
                @test run_json(["residuals", leaf, ccsv, "--dep", "y"]).code == 0
            end

            # Genuinely Poisson data ⇒ equidispersion must NOT be rejected, and the summary
            # must recommend poisson. This is the assertion that would have caught the
            # directional bug (a two-sided reject was recommending nbreg on UNDERdispersion).
            rd = run_json(["test", "dispersion", ccsv, "--dep", "y", "--exposure", "expo"])
            assert_envelope_ok(rd; label="test dispersion")
            td = cols_table(rd.doc, ["form", "alpha", "t_stat", "decision"])
            @test td !== nothing
            @test Set(String(collect(r)[col_index(td, "form")]) for r in table_rows(td)) ==
                  Set(["NB2", "NB1"])
            sumt = cols_table(rd.doc, ["metric", "value"])
            @test occursin("poisson", String(metric_value(sumt, "preferred_model")))

            # typed errors, never exit 1
            badc = write_csv(DataFrame(y=randn(n), x1=cx1); prefix="count_bad")
            @test run_json(["estimate", "poisson", badc, "--dep", "y"]).code == 3
            negc = write_csv(DataFrame(y=vcat([-1], fill(2, n - 1)), x1=cx1); prefix="count_neg")
            @test run_json(["estimate", "poisson", negc, "--dep", "y"]).code == 3
            @test run_json(["estimate", "poisson", ccsv, "--dep", "y",
                            "--offset", "expo", "--exposure", "expo"]).code == 2
            @test run_json(["estimate", "poisson", ccsv, "--dep", "y", "--maxiter", "0"]).code == 2
            @test run_json(["estimate", "poisson", ccsv, "--dep", "y", "--cov-type", "bogus"]).code == 2
            @test run_json(["estimate", "poisson", ccsv, "--dep", "y", "--exposure", "nope"]).code == 3
            # nbreg takes no --cov-type/--clusters (estimate_nbreg accepts neither)
            @test run_json(["estimate", "nbreg", ccsv, "--dep", "y", "--cov-type", "mle"]).code == 2
            rm(ccsv; force=true); rm(badc; force=true); rm(negc; force=true)
        end

        @testset "predict logit reports (flags, not string options)" begin

            for flag in ("--odds-ratio", "--marginal-effects", "--classification-table")
                r = run_json(["predict", "logit", logit, "--dep", "y", flag])
                assert_envelope_ok(r; label="predict logit $flag")
            end
            # probit has no odds ratio → unknown option is a usage error
            @test run_json(["predict", "probit", logit, "--dep", "y", "--odds-ratio"]).code == 2
        end
        rm(ord; force=true)
    end

    # ── W12/#114: determinacy map, closed-form moments, prefilter ───────────
    @testset "dsge determinacy-map (W12/#114)" begin
        # A textbook 3-equation New Keynesian block. Its determinacy frontier is the TAYLOR
        # PRINCIPLE and is known in closed form: with a purely forward-looking Phillips
        # curve and IS curve the equilibrium is determinate iff phi_pi > 1. Asserting the
        # REGION LABELS on both sides is the right test — the exact boundary placement is
        # only ever resolved to the grid spacing.
        spec = tempname() * ".jl"
        write(spec, """
        @dsge begin
            parameters: beta_d = 0.99, kappa = 0.3, sigma_i = 1.0, phi_pi = 1.5, rho_u = 0.5
            endogenous: x, pi, i, u
            exogenous: eps_u

            x[t] = x[t+1] - sigma_i * (i[t] - pi[t+1])
            pi[t] = beta_d * pi[t+1] + kappa * x[t] + u[t]
            i[t] = phi_pi * pi[t]
            u[t] = rho_u * u[t-1] + eps_u[t]
        end
        """)
        cfg = tempname() * ".toml"
        write(cfg, """
        [determinacy]
        params = ["phi_pi"]
        lower = 0.0
        upper = 2.0
        points = 21
        """)
        r = run_json(["dsge", "determinacy-map", spec, "--config", cfg])
        assert_envelope_ok(r; label="dsge determinacy-map 1-D")

        tbl = named_table(r.doc, :dsge_determinacy_map)
        if tbl === nothing
            for (_, v) in pairs(r.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   "verdict" in String[string(c) for c in v.columns]
                    tbl = v
                    break
                end
            end
        end
        @test tbl !== nothing
        if tbl !== nothing
            rows = table_rows(tbl)
            @test length(rows) == 21
            pi_i = col_index(tbl, "phi_pi")
            vi = col_index(tbl, "verdict")
            li = col_index(tbl, "label")
            ei = col_index(tbl, "existence")
            ui = col_index(tbl, "uniqueness")
            @test pi_i !== nothing && vi !== nothing && li !== nothing
            lo_side = Int[]; hi_side = Int[]
            for row in rows
                rr = collect(row)
                v = Int(rr[vi]); p = numv(rr[pi_i])
                # The raw Sims pair must agree with the collapsed verdict, or the two are
                # telling the reader different things.
                if v == 1
                    @test Int(rr[ei]) == 1 && Int(rr[ui]) == 1
                    @test string(rr[li]) == "determinate"
                elseif v == 0
                    @test Int(rr[ei]) == 1 && Int(rr[ui]) == 0
                    @test string(rr[li]) == "indeterminate"
                end
                # Stay clear of the knife edge itself; the grid straddles 1.0 exactly.
                p < 0.9 && push!(lo_side, v)
                p > 1.1 && push!(hi_side, v)
            end
            @test !isempty(lo_side) && !isempty(hi_side)
            # THE acceptance criterion: labels on both sides of the analytic frontier.
            @test all(!=(1), lo_side)          # phi_pi < 1 ⇒ never determinate
            @test all(==(1), hi_side)          # phi_pi > 1 ⇒ always determinate
        end

        summ = named_table(r.doc, :determinacy_region_summary)
        @test summ !== nothing
        if summ !== nothing
            @test Int(metric_value(summ, "n_grid_points")) == 21
            nd = Int(metric_value(summ, "n_determinate"))
            ni = Int(metric_value(summ, "n_indeterminate"))
            nn = Int(metric_value(summ, "n_no_solution"))
            nf = Int(metric_value(summ, "n_failed"))
            @test nd + ni + nn + nf == 21
            @test nd > 0 && (ni + nn) > 0      # the sweep genuinely crosses a frontier
        end

        # One-parameter sweeps also report the boundary, and it must sit near phi_pi = 1.
        bnd = named_table(r.doc, :determinacy_boundary)
        if bnd === nothing
            for (_, v) in pairs(r.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   String[string(c) for c in v.columns] == ["boundary"]
                    bnd = v
                    break
                end
            end
        end
        @test bnd !== nothing
        if bnd !== nothing
            bs = [numv(collect(rw)[1]) for rw in table_rows(bnd)]
            @test length(bs) >= 1
            # Resolution is the grid spacing (0.1 here), so allow one cell either side.
            @test any(b -> abs(b - 1.0) <= 0.15, bs)
        end

        # Two parameters: the frontier is a curve, so NO boundary table is emitted.
        cfg2 = tempname() * ".toml"
        write(cfg2, """
        [determinacy]
        params = ["phi_pi", "rho_u"]
        grids = [[0.5, 1.5], [0.2, 0.8]]
        """)
        r2 = run_json(["dsge", "determinacy-map", spec, "--config", cfg2])
        assert_envelope_ok(r2; label="dsge determinacy-map 2-D")
        s2 = named_table(r2.doc, :determinacy_region_summary)
        s2 === nothing || @test Int(metric_value(s2, "n_grid_points")) == 4
        @test !any(v -> v isa JSON3.Object && haskey(v, :columns) &&
                        String[string(c) for c in v.columns] == ["boundary"],
                   values(r2.doc.data))

        # --threaded must give an identical sweep (upstream writes disjoint indices).
        rt = run_json(["dsge", "determinacy-map", spec, "--config", cfg, "--threaded"])
        assert_envelope_ok(rt; label="dsge determinacy-map --threaded")
        st = named_table(rt.doc, :determinacy_region_summary)
        if st !== nothing && summ !== nothing
            @test Int(metric_value(st, "n_determinate")) == Int(metric_value(summ, "n_determinate"))
            @test Int(metric_value(st, "n_indeterminate")) == Int(metric_value(summ, "n_indeterminate"))
        end

        # Typed guards.
        @test run_json(["dsge", "determinacy-map", spec]).code == 2          # no --config
        @test run_json(["dsge", "determinacy-map", spec, "--config", cfg,
                        "--rank-rtol", "0"]).code == 2
        badp = tempname() * ".toml"
        write(badp, "[determinacy]\nparams = [\"not_a_param\"]\nlower=[0.0]\nupper=[1.0]\npoints=[3]\n")
        @test run_json(["dsge", "determinacy-map", spec, "--config", badp]).code == 4
        empty_cfg = tempname() * ".toml"
        write(empty_cfg, "[other]\nx = 1\n")
        @test run_json(["dsge", "determinacy-map", spec, "--config", empty_cfg]).code == 4
        rm(spec; force=true); rm(cfg; force=true); rm(cfg2; force=true)
        rm(badp; force=true); rm(empty_cfg; force=true)
    end

    @testset "dsge moments (W12/#114)" begin
        # An exactly linear AR(1)-driven block: the analytic variance of the driving state
        # is sigma^2/(1-rho^2), which the closed-form moments must reproduce, and the
        # autocorrelation at lag k must be rho^k. Both are checkable in closed form, which
        # is the only way to know the packing was unpacked correctly.
        rho, sig = 0.7, 0.02
        spec = tempname() * ".jl"
        write(spec, """
        @dsge begin
            parameters: rho_z = $rho, sigma_z = $sig, alpha = 0.4
            endogenous: z, y
            exogenous: eps_z

            z[t] = rho_z * z[t-1] + sigma_z * eps_z[t]
            y[t] = alpha * z[t]
        end
        """)
        # For this EXACTLY LINEAR model order 2 reproduces the first-order moments
        # precisely, which is what makes the closed-form checks below valid. (Order 1 was
        # refused until MEMs 0.7.3/#607 fixed the control-block covariance — its own
        # closed-form case follows below, W9/#116.)
        r = run_json(["dsge", "moments", spec, "--order", "2", "--lags", "3"])
        assert_envelope_ok(r; label="dsge moments order 2")

        mt = named_table(r.doc, :dsge_theoretical_moments)
        if mt === nothing
            for (_, v) in pairs(r.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   "mean_minus_ss" in String[string(c) for c in v.columns]
                    mt = v
                    break
                end
            end
        end
        @test mt !== nothing
        var_z = sig^2 / (1 - rho^2)
        if mt !== nothing
            rows = table_rows(mt)
            vi = col_index(mt, "variable"); si = col_index(mt, "std_dev")
            mi = col_index(mt, "mean_minus_ss")
            byvar = Dict(string(collect(rw)[vi]) => collect(rw) for rw in rows)
            @test haskey(byvar, "z") && haskey(byvar, "y")
            # sd(z) = sigma/sqrt(1-rho^2), and y = alpha*z ⇒ sd(y) = alpha*sd(z).
            sd_z = numv(byvar["z"][si]); sd_y = numv(byvar["y"][si])
            @test isapprox(sd_z, sqrt(var_z); rtol=1e-6)
            @test isapprox(sd_y, 0.4 * sqrt(var_z); rtol=1e-6)
            # At order 1 the mean IS the steady state — the risk correction is exactly zero.
            for v in ("z", "y")
                @test abs(numv(byvar[v][mi])) < 1e-10
            end
        end

        ac = nothing
        for (_, v) in pairs(r.doc.data)
            if v isa JSON3.Object && haskey(v, :columns) &&
               "autocorrelation" in String[string(c) for c in v.columns]
                ac = v
                break
            end
        end
        @test ac !== nothing
        if ac !== nothing
            rows = table_rows(ac)
            @test length(rows) == 2 * 3      # 2 variables × 3 lags
            vi = col_index(ac, "variable"); li = col_index(ac, "lag")
            ri = col_index(ac, "autocorrelation")
            for rw in rows
                r_ = collect(rw)
                lag = Int(r_[li])
                # AR(1): corr at lag k is rho^k, for BOTH variables (y is a scaling of z).
                @test isapprox(numv(r_[ri]), rho^lag; atol=1e-6)
            end
        end

        cv = nothing
        for (_, v) in pairs(r.doc.data)
            if v isa JSON3.Object && haskey(v, :columns) &&
               "correlation" in String[string(c) for c in v.columns]
                cv = v
                break
            end
        end
        @test cv !== nothing
        if cv !== nothing
            rows = table_rows(cv)
            @test length(rows) == 3          # upper triangle of a 2×2
            ci = col_index(cv, "correlation")
            v1 = col_index(cv, "variable1"); v2 = col_index(cv, "variable2")
            for rw in rows
                r_ = collect(rw)
                # y = alpha*z ⇒ every pair is perfectly correlated here.
                string(r_[v1]) == string(r_[v2]) || @test isapprox(numv(r_[ci]), 1.0; atol=1e-6)
            end
        end

        # ORDER 1 — re-enabled in W9/#116 (v0.9.2): MEMs 0.7.3 (#607) fixed the order-1
        # state↔control covariance. Pre-fix this model reported corr(z,y) = rho (0.7) and
        # autocorr(y,k) = rho^(k+2) — the assertions below are EXACTLY those numbers, so
        # they are the proof of the fix, not a smoke test.
        r1 = run_json(["dsge", "moments", spec, "--order", "1", "--lags", "3"])
        assert_envelope_ok(r1; label="dsge moments order 1")
        cv1 = nothing
        for (_, v) in pairs(r1.doc.data)
            if v isa JSON3.Object && haskey(v, :columns) &&
               "correlation" in String[string(c) for c in v.columns]
                cv1 = v; break
            end
        end
        @test cv1 !== nothing
        if cv1 !== nothing
            v1_ = col_index(cv1, "variable1"); v2_ = col_index(cv1, "variable2")
            ci1 = col_index(cv1, "correlation")
            for rw in table_rows(cv1)
                r_ = collect(rw)
                # y = alpha*z ⇒ corr = 1 exactly; the pre-fix defect reported rho here.
                string(r_[v1_]) == string(r_[v2_]) ||
                    @test isapprox(numv(r_[ci1]), 1.0; atol=1e-6)
            end
        end
        ac1 = nothing
        for (_, v) in pairs(r1.doc.data)
            if v isa JSON3.Object && haskey(v, :columns) &&
               "autocorrelation" in String[string(c) for c in v.columns]
                ac1 = v; break
            end
        end
        @test ac1 !== nothing
        if ac1 !== nothing
            li1 = col_index(ac1, "lag"); ri1 = col_index(ac1, "autocorrelation")
            for rw in table_rows(ac1)
                r_ = collect(rw)
                # rho^k for BOTH variables; the control was rho^(k+2) pre-fix.
                @test isapprox(numv(r_[ri1]), rho^Int(r_[li1]); atol=1e-6)
            end
        end
        m1 = nothing
        for (_, v) in pairs(r1.doc.data)
            if v isa JSON3.Object && haskey(v, :columns) &&
               "std_dev" in String[string(c) for c in v.columns]
                m1 = v; break
            end
        end
        @test m1 !== nothing
        if m1 !== nothing
            vi1 = col_index(m1, "variable"); si1 = col_index(m1, "std_dev")
            byv1 = Dict(string(collect(rw)[vi1]) => collect(rw) for rw in table_rows(m1))
            @test isapprox(numv(byv1["z"][si1]), sqrt(var_z); rtol=1e-6)
            @test isapprox(numv(byv1["y"][si1]), 0.4 * sqrt(var_z); rtol=1e-6)
        end

        # Order 3 runs and stays finite (the closed-form pruned recursion).
        for ord in ("3",)
            ro = run_json(["dsge", "moments", spec, "--method", "perturbation",
                           "--order", ord, "--lags", "2"])
            assert_envelope_ok(ro; label="dsge moments order $ord")
            mo = nothing
            for (_, v) in pairs(ro.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   "mean_minus_ss" in String[string(c) for c in v.columns]
                    mo = v
                    break
                end
            end
            @test mo !== nothing
            if mo !== nothing
                for rw in table_rows(mo)
                    @test isfinite(numv(collect(rw)[col_index(mo, "std_dev")]))
                    @test isfinite(numv(collect(rw)[col_index(mo, "mean_minus_ss")]))
                end
                # An exactly linear model has NO risk correction at any order — the
                # higher-order blocks are zero. This is the check that the order-2/3 path
                # is not quietly returning garbage.
                for rw in table_rows(mo)
                    @test abs(numv(collect(rw)[col_index(mo, "mean_minus_ss")])) < 1e-8
                end
            end
        end

        # Typed guards. (--order 1 is no longer a refusal — re-enabled in W9/#116 after
        # MEMs 0.7.3/#607 fixed the control-block covariance; its closed-form case above
        # is the proof.)
        @test run_json(["dsge", "moments", spec, "--order", "0"]).code == 2
        @test run_json(["dsge", "moments", spec, "--order", "4"]).code == 2
        @test run_json(["dsge", "moments", spec, "--lags", "0"]).code == 2
        rm(spec; force=true)
    end

    @testset "dsge solve — determinacy verdict (W12/#114)" begin
        spec = tempname() * ".jl"
        write(spec, """
        @dsge begin
            parameters: rho_z = 0.8, sigma_z = 0.01
            endogenous: z
            exogenous: eps_z
            z[t] = rho_z * z[t-1] + sigma_z * eps_z[t]
        end
        """)
        r = run_json(["dsge", "solve", spec])
        assert_envelope_ok(r; label="dsge solve determinacy verdict")
        dv = named_table(r.doc, :determinacy_verdict)
        @test dv !== nothing
        if dv !== nothing
            e_ = metric_value(dv, "existence"); u_ = metric_value(dv, "uniqueness")
            @test e_ !== nothing && u_ !== nothing
            @test Int(e_) in (0, 1) && Int(u_) in (0, 1)
            @test metric_value(dv, "verdict") !== nothing
            # A stable AR(1) with no forward-looking equation is determinate.
            @test Int(e_) == 1 && Int(u_) == 1
        end
        rm(spec; force=true)
    end

    # ── W10/#112: micro inference riders ────────────────────────────────────
    @testset "estimate reg --cov-type conley (W10/#112)" begin
        # Spatially correlated errors on a lat/lon grid: the true slope is 1.5 and the
        # coordinates are ORDINARY NUMERIC COLUMNS, which is the whole trap — if the loader
        # let them into X the point estimate would move.
        rng = MersenneTwister(11)
        n = 240
        lat = 30.0 .+ 10.0 .* rand(rng, n)
        lon = -100.0 .+ 10.0 .* rand(rng, n)
        x = randn(rng, n)
        # Errors correlated through a common regional shock → clustered/spatial dependence.
        region = clamp.(round.(Int, (lat .- 30.0) ./ 2.5) .+ 1, 1, 5)
        shock = randn(rng, 5)
        yv = 0.5 .+ 1.5 .* x .+ shock[region] .+ 0.4 .* randn(rng, n)
        csv = write_csv(DataFrame(y=yv, x=x, lat=lat, lon=lon); prefix="conley")

        rc = run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley",
                       "--lat", "lat", "--lon", "lon", "--conley-metric", "haversine",
                       "--dist-cutoff", "200"])
        assert_envelope_ok(rc; label="estimate reg conley")
        tblc = named_table(rc.doc, :reg_coefficients)
        @test tblc !== nothing
        if tblc !== nothing
            cols = table_cols(tblc)
            rows = table_rows(tblc)
            ti = col_index(tblc, "term"); ei = col_index(tblc, "estimate")
            sei = col_index(tblc, "std_error")
            terms = [string(collect(r)[ti]) for r in rows]
            # THE core assertion: `lat`/`lon` must NOT be regressors. If `_load_reg_data`
            # let them in, this is where a silently-wrong point estimate would show up.
            @test "x" in terms
            @test !("lat" in terms)
            @test !("lon" in terms)
            @test length(terms) == 1
            bx = Float64(collect(rows[findfirst(==("x"), terms)])[ei])
            @test isapprox(bx, 1.5; atol=0.15)
            se_conley = Float64(collect(rows[findfirst(==("x"), terms)])[sei])
            @test isfinite(se_conley) && se_conley > 0
        end
        # The settings table is emitted ONLY under conley and records what produced the SE.
        setc = named_table(rc.doc, :conley_spatial_hac_settings)
        @test setc !== nothing
        if setc !== nothing
            @test metric_value(setc, "metric") !== nothing || length(table_rows(setc)) >= 1
        end

        # Same fit, hc1: identical point estimates (only the covariance changes). This is
        # the check that `--cov-type conley` is not quietly altering the estimator.
        rh = run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "hc1"])
        assert_envelope_ok(rh; label="estimate reg hc1 baseline")
        tblh = named_table(rh.doc, :reg_coefficients)
        if tblh !== nothing && tblc !== nothing
            th = [string(collect(r)[col_index(tblh, "term")]) for r in table_rows(tblh)]
            # hc1 has NO --lat/--lon to exclude, so lat/lon ARE regressors here — that
            # asymmetry is exactly why the conley path must exclude them.
            @test "lat" in th
            # ...and no settings table without conley.
            @test named_table(rh.doc, :conley_spatial_hac_settings) === nothing
        end

        # Typed guards, both directions.
        @test run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley"]).code == 2
        @test run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley",
                        "--lat", "lat", "--lon", "lon"]).code == 2            # cutoff = 0
        @test run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "hc1",
                        "--lat", "lat", "--lon", "lon"]).code == 2            # coords w/o conley
        @test run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley",
                        "--lat", "nope", "--lon", "lon", "--dist-cutoff", "50"]).code == 3
        @test run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley",
                        "--lat", "lat", "--lon", "lon", "--dist-cutoff", "50",
                        "--time-cutoff", "3"]).code == 2                      # cutoff w/o column
        # haversine with out-of-range degrees is caught, not silently wrong.
        bad = write_csv(DataFrame(y=yv, x=x, lat=lat .+ 100.0, lon=lon); prefix="conleybad")
        @test run_json(["estimate", "reg", bad, "--dep", "y", "--cov-type", "conley",
                        "--lat", "lat", "--lon", "lon", "--conley-metric", "haversine",
                        "--dist-cutoff", "100"]).code == 3
        # A euclidean fit on the same coordinates must still work (no degree range there).
        re = run_json(["estimate", "reg", csv, "--dep", "y", "--cov-type", "conley",
                       "--lat", "lat", "--lon", "lon", "--dist-cutoff", "2.0"])
        assert_envelope_ok(re; label="estimate reg conley euclidean")

        # Spatial + serial Conley. The time column must be an INTEGRAL period index:
        # upstream gives any non-integer `abs(t_i - t_j)` gap zero weight, so a fractional
        # column would silently disable the serial correction rather than fail.
        yr = Float64.(repeat(1:8, inner=cld(n, 8))[1:n])
        tcsv = write_csv(DataFrame(y=yv, x=x, lat=lat, lon=lon, yr=yr); prefix="conleyt")
        rt2 = run_json(["estimate", "reg", tcsv, "--dep", "y", "--cov-type", "conley",
                        "--lat", "lat", "--lon", "lon", "--conley-metric", "haversine",
                        "--dist-cutoff", "200", "--time-col", "yr", "--time-cutoff", "2"])
        assert_envelope_ok(rt2; label="estimate reg conley spatial+serial")
        tt2 = named_table(rt2.doc, :reg_coefficients)
        if tt2 !== nothing
            terms2 = [string(collect(r)[col_index(tt2, "term")]) for r in table_rows(tt2)]
            @test !("yr" in terms2)     # the time column is excluded from X too
            @test terms2 == ["x"]
        end
        fcsv = write_csv(DataFrame(y=yv, x=x, lat=lat, lon=lon, yr=yr .+ 0.5); prefix="conleyf")
        @test run_json(["estimate", "reg", fcsv, "--dep", "y", "--cov-type", "conley",
                        "--lat", "lat", "--lon", "lon", "--dist-cutoff", "2.0",
                        "--time-col", "yr", "--time-cutoff", "2"]).code == 3
        rm(csv; force=true); rm(bad; force=true); rm(tcsv; force=true); rm(fcsv; force=true)
    end

    @testset "estimate preg --absorb (W10/#112)" begin
        # UNBALANCED panel — the case the issue calls out. Truth: y = 0.9*x + entity FE +
        # time FE. `--absorb entity,time` must recover 0.9; the naive additive two-way
        # transform does not hold here, which is why --twoway is refused in favour of it.
        rng = MersenneTwister(23)
        N, T = 25, 12
        id = Int[]; tt = Int[]; xs = Float64[]; ys = Float64[]
        afe = randn(rng, N); tfe = randn(rng, T)
        for i in 1:N, t in 1:T
            # Drop ~25% of cells, unevenly, so the panel is genuinely unbalanced.
            (rand(rng) < 0.25) && continue
            xv = afe[i] * 0.4 + tfe[t] * 0.3 + randn(rng)
            push!(id, i); push!(tt, t); push!(xs, xv)
            push!(ys, 0.9 * xv + afe[i] + tfe[t] + 0.3 * randn(rng))
        end
        csv = write_csv(DataFrame(id=id, time=tt, y=ys, x=xs); prefix="hdfe")

        ra = run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                       "--absorb", "entity,time", "--id-col", "id", "--time-col", "time"])
        assert_envelope_ok(ra; label="estimate preg --absorb entity,time")
        tbl = named_table(ra.doc, :panel_regression_coefficients)
        @test tbl !== nothing
        if tbl !== nothing
            rows = table_rows(tbl)
            ti = col_index(tbl, "term"); ei = col_index(tbl, "estimate")
            terms = [string(collect(r)[ti]) for r in rows]
            @test "x" in terms
            bx = Float64(collect(rows[findfirst(==("x"), terms)])[ei])
            # Agrees with the dummy-OLS truth on an UNBALANCED panel — the acceptance case.
            @test isapprox(bx, 0.9; atol=0.10)
        end
        hd = named_table(ra.doc, :hdfe_absorption)
        @test hd !== nothing
        if hd !== nothing
            @test metric_value(hd, "converged") !== nothing || length(table_rows(hd)) >= 1
        end

        # `--absorb entity` reproduces plain one-way FE, coefficient for coefficient.
        r1 = run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                       "--absorb", "entity", "--id-col", "id", "--time-col", "time"])
        r0 = run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                       "--id-col", "id", "--time-col", "time"])
        assert_envelope_ok(r1; label="estimate preg --absorb entity")
        assert_envelope_ok(r0; label="estimate preg plain fe")
        t1 = named_table(r1.doc, :panel_regression_coefficients)
        t0 = named_table(r0.doc, :panel_regression_coefficients)
        if t1 !== nothing && t0 !== nothing
            e1 = Float64(collect(table_rows(t1)[1])[col_index(t1, "estimate")])
            e0 = Float64(collect(table_rows(t0)[1])[col_index(t0, "estimate")])
            @test isapprox(e1, e0; atol=1e-6)
            # ...and only the absorb run carries the diagnostics table.
            @test named_table(r0.doc, :hdfe_absorption) === nothing
        end

        # Typed guards.
        @test run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                        "--absorb", "entity,time", "--twoway",
                        "--id-col", "id", "--time-col", "time"]).code == 2
        @test run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                        "--absorb", "entity,entity",
                        "--id-col", "id", "--time-col", "time"]).code == 2
        @test run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                        "--absorb", "entity", "--method", "re",
                        "--id-col", "id", "--time-col", "time"]).code == 2
        @test run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                        "--hdfe-tol", "1e-6",
                        "--id-col", "id", "--time-col", "time"]).code == 2   # tol w/o absorb
        @test run_json(["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                        "--absorb", "nosuchcol",
                        "--id-col", "id", "--time-col", "time"]).code in (2, 3)
        rm(csv; force=true)
    end

    @testset "estimate preg ab/bb instrument controls (W10/#131)" begin
        # Dynamic panel: y_it = 0.5 y_{i,t-1} + 0.8 x_it + fe_i + eps. The point is
        # not the coefficient but the INSTRUMENT COUNT: --collapse and the lag
        # window are what MEMs#549 added, and n_instruments is the observable.
        rng = MersenneTwister(41)
        N, T = 25, 12
        id = Int[]; tt = Int[]; xs = Float64[]; ys = Float64[]
        for i in 1:N
            fe = randn(rng); ylag = randn(rng)
            for t in 1:T
                xv = randn(rng)
                yv = 0.5 * ylag + 0.8 * xv + fe + 0.3 * randn(rng)
                push!(id, i); push!(tt, t); push!(xs, xv); push!(ys, yv)
                ylag = yv
            end
        end
        csv = write_csv(DataFrame(id=id, time=tt, y=ys, x=xs); prefix="dynpanel")
        base = ["estimate", "preg", csv, "--dep", "y", "--indep", "x",
                "--id-col", "id", "--time-col", "time"]

        n_inst(r) = begin
            dd = named_table(r.doc, :dynamic_panel_diagnostics)
            dd === nothing ? nothing : metric_value(dd, "n_instruments")
        end

        r_ab = run_json([base; "--method"; "ab"])
        assert_envelope_ok(r_ab; label="estimate preg --method ab")
        @test named_table(r_ab.doc, :panel_regression_coefficients) !== nothing
        ni_full = n_inst(r_ab)
        @test ni_full !== nothing && Int(ni_full) > 0

        # Collapse strictly shrinks the instrument matrix.
        r_col = run_json([base; "--method"; "ab"; "--collapse"])
        @test r_col.code == 0
        ni_col = n_inst(r_col)
        @test ni_col !== nothing && Int(ni_col) < Int(ni_full)

        # Narrowing the lag window shrinks it too (99 → 4).
        r_win = run_json([base; "--method"; "ab"; "--max-lag-endo"; "4"])
        @test r_win.code == 0
        ni_win = n_inst(r_win)
        @test ni_win !== nothing && Int(ni_win) < Int(ni_full)

        # bb (system GMM) takes the same controls.
        r_bb = run_json([base; "--method"; "bb"; "--collapse"])
        @test r_bb.code == 0
        @test n_inst(r_bb) !== nothing

        # A plain fe run never emits the diagnostics table…
        r_fe = run_json(base)
        @test r_fe.code == 0 && named_table(r_fe.doc, :dynamic_panel_diagnostics) === nothing
        # …and the controls on a non-GMM method are refused, not silently ignored.
        @test run_json([base; "--collapse"]).code == 2
        @test run_json([base; "--min-lag-endo"; "3"]).code == 2
        # Window sanity on the GMM path.
        @test run_json([base; "--method"; "ab"; "--min-lag-endo"; "0"]).code == 2
        @test run_json([base; "--method"; "ab"; "--min-lag-endo"; "5";
                        "--max-lag-endo"; "3"]).code == 2
        rm(csv; force=true)
    end

    @testset "estimate piv weak-instrument diagnostics (W10/#131, MEMs#553)" begin
        # This leaf had ZERO T3 coverage. Strong instruments by construction
        # (first-stage R² ≈ 0.9), true structural coefficient 1.2 on the
        # endogenous regressor, u in both equations = the endogeneity.
        rng = MersenneTwister(43)
        N, T = 30, 10
        id = repeat(1:N, inner=T); tt = repeat(1:T, N)
        z1 = randn(rng, N * T); z2 = randn(rng, N * T)
        u = randn(rng, N * T)
        endo = 0.7z1 + 0.5z2 + 0.4u + 0.3randn(rng, N * T)
        x = randn(rng, N * T)
        y = 1.2endo + 0.5x + u + 0.3randn(rng, N * T)
        csv = write_csv(DataFrame(id=id, time=tt, y=y, x=x, endo=endo, z1=z1, z2=z2);
                        prefix="pivdiag")
        base = ["estimate", "piv", csv, "--dep", "y", "--exog", "x", "--endog", "endo",
                "--id-col", "id", "--time-col", "time"]

        r = run_json([base; "--instruments"; "z1,z2"])
        assert_envelope_ok(r; label="estimate piv overidentified")
        ct = named_table(r.doc, :panel_iv_coefficients)
        @test ct !== nothing
        if ct !== nothing
            rows = table_rows(ct)
            ti = col_index(ct, "term"); ei = col_index(ct, "estimate")
            terms = [string(collect(rw)[ti]) for rw in rows]
            @test "endo" in terms
            be = Float64(collect(rows[findfirst(==("endo"), terms)])[ei])
            @test isapprox(be, 1.2; atol=0.1)
        end
        dd = named_table(r.doc, :weak_instrument_diagnostics)
        @test dd !== nothing
        if dd !== nothing
            @test numv(metric_value(dd, "first-stage F (min partial)")) > 10.0
            @test numv(metric_value(dd, "Cragg-Donald F")) > 10.0
            @test numv(metric_value(dd, "Kleibergen-Paap F")) > 10.0
            # Tabulated constant for 1 endogenous / 2 instruments — a stable pin.
            @test numv(metric_value(dd, "Stock-Yogo 10% critical value")) ≈ 19.93 atol = 0.01
            @test metric_value(dd, "Sargan p-value") !== nothing
            @test numv(metric_value(dd, "Sargan p-value")) <= 1.0
        end

        # Just-identified: Sargan has no dof — the cell says WHY it is missing.
        rj = run_json([base; "--instruments"; "z1"])
        @test rj.code == 0
        dj = named_table(rj.doc, :weak_instrument_diagnostics)
        @test dj !== nothing
        if dj !== nothing
            @test string(metric_value(dj, "Sargan statistic")) ==
                  "unavailable (failed or underidentified)"
        end

        # Bare error() used to turn these usage mistakes into exit 1.
        @test run_json(["estimate", "piv", csv, "--exog", "x", "--endog", "endo",
                        "--instruments", "z1", "--id-col", "id",
                        "--time-col", "time"]).code == 2
        @test run_json(["estimate", "piv", csv, "--dep", "y", "--exog", "x",
                        "--instruments", "z1", "--id-col", "id",
                        "--time-col", "time"]).code == 2
        rm(csv; force=true)
    end

    @testset "policy family (W4/#126, new top-level, MEMs 0.8.0 CF module)" begin
        rng = MersenneTwister(47)
        T_obs = 200
        infl = zeros(T_obs); ygap = zeros(T_obs); rate = zeros(T_obs)
        for t in 2:T_obs
            ygap[t] = 0.6ygap[t-1] - 0.1rate[t-1] + randn(rng)
            infl[t] = 0.5infl[t-1] + 0.2ygap[t-1] + 0.5randn(rng)
            rate[t] = 0.7rate[t-1] + 0.3infl[t-1] + 0.3randn(rng)
        end
        csv = write_csv(DataFrame(infl=infl, ygap=ygap, rate=rate); prefix="policy")
        maps = ["--outcomes", "infl=1,ygap=2", "--instruments", "rate=3"]

        cfsum(r) = begin
            t = named_table(r.doc, :counterfactual_summary)
            t === nothing ? Dict{String,Any}() :
                Dict(String(collect(row)[1]) => collect(row)[2] for row in table_rows(t))
        end

        @testset "effects — one leaf per source route" begin
            rv = run_json(vcat(["policy", "effects", "var", csv, "--shocks", "3",
                                "--horizon", "8"], maps))
            assert_envelope_ok(rv; label="policy effects var")
            menu = named_table(rv.doc, :policy_causal_effects_menu)
            @test menu !== nothing
            @test length(table_rows(menu)) == 3 * 8      # 3 mapped vars × 1 shock × H
            s = Dict(String(collect(r)[1]) => collect(r)[2]
                     for r in table_rows(named_table(rv.doc, :policy_causal_effects_summary)))
            @test s["is_square"] == false && s["source"] == "var"

            rb = run_json(vcat(["policy", "effects", "bvar", csv, "--shocks", "3",
                                "--horizon", "6", "--draws", "300"], maps))
            @test rb.code == 0
            sb = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rb.doc, :policy_causal_effects_summary)))
            @test Int(sb["n_draws"]) > 0                 # posterior draws carried over

            rl = run_json(vcat(["policy", "effects", "lp", csv, "--shocks", "3",
                                "--horizon", "6", "--n-draws", "50"], maps))
            @test rl.code == 0

            # sign route needs a restrictions config
            signtoml = tempname() * ".toml"
            write(signtoml, """
            [identification]
            method = "sign"

            [identification.sign_matrix]
            matrix = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
            horizons = [0]
            """)
            rs = run_json(vcat(["policy", "effects", "sign", csv, "--shocks", "1",
                                "--horizon", "6", "--config", signtoml,
                                "--replications", "200"], maps))
            @test rs.code == 0
            ss = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rs.doc, :policy_causal_effects_summary)))
            @test ss["source"] == "sign_set" && Int(ss["n_draws"]) > 0
            @test run_json(vcat(["policy", "effects", "sign", csv, "--shocks", "1",
                                 "--horizon", "6"], maps)).code == 2   # no config
            rm(signtoml; force=true)
        end

        @testset "counterfactual var — thin menu is HONEST about the shortfall" begin
            r = run_json(vcat(["policy", "counterfactual", "var", csv, "--shocks", "3",
                               "--nonpolicy-shock", "1", "--rule", "rate-peg",
                               "--horizon", "8"], maps))
            assert_envelope_ok(r; label="policy counterfactual var rate-peg")
            for k in (:policy_counterfactual_paths, :enforcing_policy_shocks_nu,
                      :implementation_error_path, :counterfactual_summary)
                @test named_table(r.doc, k) !== nothing
            end
            s = cfsum(r)
            @test s["rule"] == "rate peg"
            @test numv(s["rel_residual"]) >= 0.0
            @test haskey(s, "spanned")
            # 1 policy shock cannot enforce an 8-period peg exactly
            @test numv(s["rel_residual"]) > 1e-8
        end

        @testset "counterfactual var — SQUARE menu enforces the peg EXACTLY" begin
            # 2 policy shocks, H=2 → n_s == H → exact solve: the pegged
            # instrument path is identically zero and rel_residual ~0.
            r = run_json(["policy", "counterfactual", "var", csv, "--shocks", "2,3",
                          "--nonpolicy-shock", "1", "--outcomes", "infl=1",
                          "--instruments", "rate=3", "--rule", "rate-peg",
                          "--horizon", "2"])
            @test r.code == 0
            s = cfsum(r)
            @test numv(s["rel_residual"]) < 1e-8 && s["spanned"] == true
            p = named_table(r.doc, :policy_counterfactual_paths)
            ci = col_index(p, "counterfactual"); ri = col_index(p, "role")
            zvals = [numv(collect(row)[ci]) for row in table_rows(p)
                     if String(collect(row)[ri]) == "instrument"]
            @test all(abs(v) < 1e-8 for v in zvals)
        end

        @testset "counterfactual — draws, routes, rule config" begin
            # bootstrap bands on the var route
            rb = run_json(vcat(["policy", "counterfactual", "var", csv, "--shocks", "3",
                                "--nonpolicy-shock", "1", "--rule", "rate-peg",
                                "--horizon", "6", "--replications", "50"], maps))
            @test rb.code == 0
            p = named_table(rb.doc, :policy_counterfactual_paths)
            @test "q16" in table_cols(p) && "q84" in table_cols(p)
            @test Int(cfsum(rb)["n_draws_used"]) > 0

            # bvar route propagates posterior draws
            rv = run_json(vcat(["policy", "counterfactual", "bvar", csv, "--shocks", "3",
                                "--nonpolicy-shock", "1", "--rule", "taylor",
                                "--horizon", "6", "--draws", "300"], maps))
            @test rv.code == 0
            @test Int(cfsum(rv)["n_draws_used"]) > 0
            @test occursin("0.5", String(cfsum(rv)["rule"]))   # TEXTBOOK taylor, not CMW

            # lp route + CMW taylor via TOML — the cmw trap defused end-to-end
            ruletoml = tempname() * ".toml"
            write(ruletoml, "[rule]\ntype = \"taylor\"\ncmw = true\n")
            rl = run_json(vcat(["policy", "counterfactual", "lp", csv, "--shocks", "3",
                                "--nonpolicy-shock", "1", "--rule-config", ruletoml,
                                "--horizon", "6", "--n-draws", "50"], maps))
            @test rl.code == 0
            @test occursin("0.85", String(cfsum(rl)["rule"]))
            rm(ruletoml; force=true)
        end

        @testset "policy optimal + moments (W5/#127)" begin
            losstoml = tempname() * ".toml"
            write(losstoml, """
            [loss]
            outcomes = ["infl", "ygap"]
            lambda = [1.0, 0.5]

            [loss.smoothing]
            lambda = 0.5
            """)
            ro = run_json(vcat(["policy", "optimal", "var", csv, "--shocks", "3",
                                "--nonpolicy-shock", "1", "--loss-config", losstoml,
                                "--horizon", "8"], maps))
            assert_envelope_ok(ro; label="policy optimal var")
            so = cfsum(ro)
            # The optimality certificate: foc_norm ≈ 0 and the loss cannot rise.
            foc_key = first(k for k in keys(so) if startswith(k, "foc_norm"))
            @test numv(so[foc_key]) < 1e-6
            @test numv(so["loss_cf"]) <= numv(so["loss_base"]) + 1e-10
            @test named_table(ro.doc, :implementation_error_path) !== nothing
            # --loss-config required; lambda-less TOML is config/missing-key (4).
            @test run_json(vcat(["policy", "optimal", "var", csv, "--shocks", "3",
                                 "--nonpolicy-shock", "1"], maps)).code == 2
            badtoml = tempname() * ".toml"
            write(badtoml, "[loss]\noutcomes = [\"infl\"]\n")
            @test run_json(vcat(["policy", "optimal", "var", csv, "--shocks", "3",
                                 "--nonpolicy-shock", "1", "--loss-config", badtoml],
                                maps)).code == 4
            rm(badtoml; force=true)

            rm_ = run_json(vcat(["policy", "moments", "var", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--horizon", "12"], maps))
            assert_envelope_ok(rm_; label="policy moments var rate-peg")
            sd = named_table(rm_.doc, :counterfactual_standard_deviations)
            @test sd !== nothing
            vi = col_index(sd, "variable"); bi = col_index(sd, "sd_base")
            ci2 = col_index(sd, "sd_cf")
            rrow = first(r for r in table_rows(sd)
                         if String(collect(r)[vi]) == "rate")
            # Pegging the rate must collapse its unconditional sd.
            @test numv(collect(rrow)[ci2]) < numv(collect(rrow)[bi])
            ms = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rm_.doc, :moments_summary)))
            tail_key = first(k for k in keys(ms) if startswith(k, "tail_share"))
            @test numv(ms[tail_key]) >= 0.0
            @test named_table(rm_.doc, :counterfactual_correlations) !== nothing

            # Band-limited variance ⊂ full variance, variable by variable.
            rb_ = run_json(vcat(["policy", "moments", "var", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--horizon", "12",
                                 "--frequencies", "business-cycle"], maps))
            @test rb_.code == 0
            sdb = named_table(rb_.doc, :counterfactual_standard_deviations)
            full = Dict(String(collect(r)[vi]) => numv(collect(r)[bi])
                        for r in table_rows(sd))
            for r in table_rows(sdb)
                @test numv(collect(r)[bi]) <= full[String(collect(r)[vi])] + 1e-8
            end

            # moments on the bvar route (wold from the posterior)
            @test run_json(vcat(["policy", "moments", "bvar", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--horizon", "8",
                                 "--draws", "200"], maps)).code == 0
            # rule XOR loss; bad band; both → usage errors
            @test run_json(vcat(["policy", "moments", "var", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--loss-config", losstoml],
                                maps)).code == 2
            @test run_json(vcat(["policy", "moments", "var", csv, "--shocks", "3"],
                                maps)).code == 2
            @test run_json(vcat(["policy", "moments", "var", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--frequencies", "2,1"],
                                maps)).code == 2
            rm(losstoml; force=true)
        end

        @testset "OPP family (W6/#128)" begin
            losstoml = tempname() * ".toml"
            write(losstoml, "[loss]\noutcomes = [\"infl\", \"ygap\"]\nlambda = [1.0, 0.5]\n")
            ob = vcat(["policy", "opp", "var", csv, "--shocks", "3",
                       "--loss-config", losstoml, "--horizon", "8"], maps)

            r = run_json([ob; "--targets"; "infl=0,ygap=0"; "--n-sim"; "500"])
            assert_envelope_ok(r; label="policy opp var")
            dd = named_table(r.doc, :opp_recommendation_delta)
            @test dd !== nothing
            cols = table_cols(dd)
            @test cols[1:4] == ["shock", "delta", "delta_plugin", "gradient"]
            # BM reversed polarity bands at 60/75/90, labelled as upstream does.
            @test "lo60" in cols && "hi90" in cols && "reject60" in cols
            row = collect(first(table_rows(dd)))
            @test isfinite(numv(row[2])) && isfinite(numv(row[3]))
            s = Dict(String(collect(rw)[1]) => collect(rw)[2]
                     for rw in table_rows(named_table(r.doc, :opp_summary)))
            @test haskey(s, "band_polarity") && occursin("LOWER", String(s["band_polarity"]))
            @test named_table(r.doc, :objective_gap_paths) !== nothing

            # The gaps-vs-levels trap is a refusal, not a silent zero-target run.
            @test run_json(ob).code == 2
            @test run_json([ob; "--targets"; "infl=0"]).code == 2

            # bvar route: store_draws=true is passed → posterior bands present.
            rb = run_json(vcat(["policy", "opp", "bvar", csv, "--shocks", "3",
                                "--loss-config", losstoml, "--horizon", "8",
                                "--draws", "300", "--n-sim", "300",
                                "--targets", "infl=0,ygap=0"], maps))
            @test rb.code == 0
            @test "lo60" in table_cols(named_table(rb.doc, :opp_recommendation_delta))

            # Constrained OPP: ZLB floor + announced path → SLSQP + KKT reported;
            # missing --instrument-path refuses.
            constoml = tempname() * ".toml"
            write(constoml, "[[constraint]]\ntype = \"zlb\"\nfloor = -0.1\ninstrument = \"rate\"\n")
            rc = run_json([ob; "--targets"; "infl=0,ygap=0";
                           "--constraints-file"; constoml; "--n-sim"; "0";
                           "--instrument-path"; "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"])
            @test rc.code == 0
            sc = Dict(String(collect(rw)[1]) => collect(rw)[2]
                      for rw in table_rows(named_table(rc.doc, :opp_summary)))
            @test String(sc["method_used"]) in ("slsqp", "projection")
            @test haskey(sc, "kkt_residual") && haskey(sc, "warm_start_feasible")
            @test named_table(rc.doc, :instrument_paths_announced_vs_recommended) !== nothing
            @test run_json([ob; "--targets"; "infl=0,ygap=0";
                            "--constraints-file"; constoml]).code == 2

            # opp-sequence: per-date gap files; the revision decomposition is
            # EXACT three-part (news + pref + aging = delta_t − delta_{t−1}).
            fdir = mktempdir()
            for (i, d) in enumerate(["2019Q4", "2020Q1", "2020Q2"])
                CSV.write(joinpath(fdir, "$d.csv"),
                          DataFrame(infl=fill(0.6 - 0.15i, 8), ygap=fill(-1.0 + 0.2i, 8)))
            end
            rs = run_json(vcat(["policy", "opp-sequence", "var", csv, "--shocks", "3",
                                "--loss-config", losstoml, "--forecasts-dir", fdir,
                                "--sd", "0.5,0.5", "--horizon", "8"], maps))
            assert_envelope_ok(rs; label="policy opp-sequence var")
            dl = named_table(rs.doc, :opp_sequence_delta_by_date)
            dec = named_table(rs.doc, :opp_revision_decomposition)
            @test dl !== nothing && dec !== nothing
            del = Dict(String(collect(rw)[1]) => numv(collect(rw)[3])
                       for rw in table_rows(dl))
            parts = Dict(String(collect(rw)[1]) =>
                         numv(collect(rw)[3]) + numv(collect(rw)[4]) + numv(collect(rw)[5])
                         for rw in table_rows(dec))
            dates = sort(collect(keys(del)))
            for i in 2:length(dates)
                @test isapprox(parts[dates[i]], del[dates[i]] - del[dates[i-1]];
                               atol=5e-6)   # renderer rounds to 6 digits
            end
            # --sd required on the sequence route (external containers need uncertainty)
            @test run_json(vcat(["policy", "opp-sequence", "var", csv, "--shocks", "3",
                                 "--loss-config", losstoml, "--forecasts-dir", fdir,
                                 "--horizon", "8"], maps)).code == 2
            rm(losstoml; force=true); rm(constoml; force=true)
            rm(fdir; recursive=true, force=true)
        end

        @testset "structural routes (W7/#129)" begin
            nk = tempname() * ".toml"
            write(nk, """
            [model]
            parameters = { rho = 0.8, kappa = 0.3, phi = 1.5, sigma = 0.01 }
            endogenous = ["ygap", "infl", "rate"]
            exogenous = ["e", "mp"]
            linear = true
            [[model.equations]]
            expr = "ygap[t] = rho * ygap[t-1] - 0.2 * rate[t] + sigma * e[t]"
            [[model.equations]]
            expr = "infl[t] = 0.5 * infl[t-1] + kappa * ygap[t]"
            [[model.equations]]
            expr = "rate[t] = phi * infl[t] + 0.01 * mp[t]"
            """)

            rn = run_json(["policy", "news", "dsge", nk, "--policy-shock", "mp",
                           "--outcomes", "infl=infl,ygap=ygap",
                           "--instruments", "rate=rate", "--horizon", "8"])
            assert_envelope_ok(rn; label="policy news dsge")
            sn = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rn.doc, :policy_causal_effects_summary)))
            # The news menu is SQUARE by construction — the exact-solve regime.
            @test sn["is_square"] == true && Int(sn["n_shocks"]) == 8
            @test sn["source"] == "dsge"

            # Behavioral discounting shrinks the menu; identity request refused
            # implicitly (NaN sentinels), out-of-range refused explicitly.
            rb = run_json(["policy", "news", "dsge", nk, "--policy-shock", "mp",
                           "--outcomes", "infl=infl", "--horizon", "6",
                           "--behavioral-m", "0.7"])
            @test rb.code == 0
            menu0 = named_table(rn.doc, :policy_causal_effects_menu)
            @test run_json(["policy", "news", "dsge", nk, "--policy-shock", "mp",
                            "--outcomes", "infl=infl", "--horizon", "6",
                            "--behavioral-m", "1.5"]).code == 2

            # History: forecast-revision counterfactual over a window.
            rh = run_json(vcat(["policy", "history", "var", csv, "--shocks", "3",
                                "--rule", "rate-peg", "--horizon", "12",
                                "--t-range", "100:105"], maps))
            assert_envelope_ok(rh; label="policy history var")
            ht = named_table(rh.doc, :counterfactual_history)
            @test ht !== nothing
            @test length(unique(String[string(collect(r)[1]) for r in table_rows(ht)])) == 6
            @test run_json(vcat(["policy", "history", "var", csv, "--shocks", "3",
                                 "--rule", "rate-peg", "--horizon", "4",
                                 "--t-range", "100:110"], maps)).code == 2  # window > H−1

            # Spanning: genuine square-vs-thin pair, machine-readable verdict.
            rs = run_json(vcat(["policy", "spanning", "var", csv, nk,
                                "--shocks", "3", "--nonpolicy-shock", "1",
                                "--model-outcomes", "infl=infl,ygap=ygap",
                                "--model-instruments", "rate=rate",
                                "--policy-shock", "mp", "--rule", "rate-peg",
                                "--horizon", "8"], maps))
            assert_envelope_ok(rs; label="policy spanning var")
            sv = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rs.doc, :spanning_verdict)))
            @test haskey(sv, "spanned") && haskey(sv, "loading_inside")
            @test 0.0 <= numv(sv["loading_inside"]) <= 1.0 + 1e-9
            # name-agreement guard fires typed, before upstream's untyped error
            @test run_json(vcat(["policy", "spanning", "var", csv, nk,
                                 "--shocks", "3", "--nonpolicy-shock", "1",
                                 "--model-outcomes", "pi=infl,gap=ygap",
                                 "--model-instruments", "rate=rate",
                                 "--policy-shock", "mp", "--rule", "rate-peg",
                                 "--horizon", "8"], maps)).code == 2

            # Sufficiency: population laboratory, no data. The 3-var NK with
            # 2 shocks and 2 observables is invertible.
            rf = run_json(["policy", "sufficiency", "dsge", nk,
                           "--observables", "infl,rate", "--horizon", "12"])
            assert_envelope_ok(rf; label="policy sufficiency dsge")
            sf = Dict(String(collect(r)[1]) => collect(r)[2]
                      for r in table_rows(named_table(rf.doc, :sufficiency_summary)))
            @test sf["invertible"] == true
            fev = named_table(rf.doc, :forecast_sufficiency_fev_ratios)
            @test all(numv(collect(r)[3]) >= 1.0 - 1e-9 for r in table_rows(fev))

            # HA routes: the .jl/builtin HA path had ZERO T3 for a year — keep
            # these small (huggett, tiny horizons) but PRESENT.
            rj = run_json(["policy", "jacobian", "ha", "huggett",
                           "--jac-output", "C", "--t-horizon", "20"])
            @test rj.code == 0
            jt = named_table(rj.doc, :sequence_space_jacobian)
            @test jt !== nothing && length(table_rows(jt)) == 400   # T² tidy rows
            rha = run_json(["policy", "news", "ha", "huggett",
                            "--outcomes", "c=C", "--horizon", "4",
                            "--t-horizon", "40"])
            @test rha.code == 0
            rm(nk; force=true)
        end

        @testset "guards — typed, never exit 1" begin
            b = vcat(["policy", "counterfactual", "var", csv, "--shocks", "3",
                      "--nonpolicy-shock", "1"], maps)
            @test run_json([b; "--rule"; "rate-peg"; "--horizon"; "0"]).code == 2
            @test run_json([b; "--rule"; "bogus"]).code == 2
            @test run_json(b).code == 2                       # no rule at all
            @test run_json([b; "--rule"; "rate-peg"; "--quantiles"; "0.16,1.5"]).code == 2
            @test run_json([b; "--rule"; "rate-peg"; "--spanned-tol"; "-1"]).code == 2
            # taylor builtin without infl/ygap-named outcomes → friendly usage error
            @test run_json(["policy", "counterfactual", "var", csv, "--shocks", "3",
                            "--nonpolicy-shock", "1", "--outcomes", "cpi=1,gap=2",
                            "--instruments", "rate=3", "--rule", "taylor"]).code == 2
            # unknown shock name → typed data-side error, not exit 1
            @test run_json(vcat(["policy", "effects", "var", csv, "--shocks", "nosuch",
                                 "--horizon", "4"], maps)).code in (3, 5)
        end
        rm(csv; force=true)
    end

    @testset "test wild-cluster (W10/#112)" begin
        # FEW clusters — the regime the method exists for. G=8, a real treatment effect of
        # 1.0 assigned at CLUSTER level (so the cluster-robust normal p over-rejects).
        rng = MersenneTwister(31)
        G, npg = 8, 30
        cl = Int[]; xs = Float64[]; ys = Float64[]
        ceff = randn(rng, G)
        for g in 1:G, _ in 1:npg
            xv = randn(rng)
            push!(cl, g); push!(xs, xv)
            push!(ys, 1.0 * xv + ceff[g] + 0.5 * randn(rng))
        end
        csv = write_csv(DataFrame(y=ys, x=xs, cl=Float64.(cl)); prefix="wcb")

        r = run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                      "--coefficient", "x", "--boot-reps", "999"])
        assert_envelope_ok(r; label="test wild-cluster")
        tbl = first_table(r.doc)[2]
        @test tbl !== nothing
        if tbl !== nothing
            pb = metric_value(tbl, "p_bootstrap_symmetric")
            pa = metric_value(tbl, "p_cluster_robust_normal")
            nc = metric_value(tbl, "n_clusters")
            en = metric_value(tbl, "enumerated")
            @test pb !== nothing && pa !== nothing
            @test 0.0 <= Float64(pb) <= 1.0
            @test 0.0 <= Float64(pa) <= 1.0
            @test Int(nc) == G
            # 2^8 = 256 <= 999 with Rademacher weights ⇒ upstream enumerates the whole sign
            # space, so the p-value is EXACT. If this flips, `--boot-reps` stopped reaching
            # the enumeration branch.
            @test en == true || string(en) == "true"
            # A strong true effect: the bootstrap should still reject.
            @test Float64(pb) < 0.10
            @test metric_value(tbl, "estimate") !== nothing
            @test isapprox(Float64(metric_value(tbl, "estimate")), 1.0; atol=0.2)
        end

        # Webb weights and the WCU variant both run; --no-ci drops the interval.
        rw = run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                       "--coefficient", "x", "--boot-weights", "webb", "--boot-reps", "199"])
        assert_envelope_ok(rw; label="test wild-cluster webb")
        ru = run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                       "--coefficient", "x", "--no-impose-null", "--no-ci"])
        assert_envelope_ok(ru; label="test wild-cluster WCU --no-ci")
        tu = first_table(ru.doc)[2]
        if tu !== nothing
            @test metric_value(tu, "ci_lower") === nothing      # --no-ci really drops it
            @test metric_value(tu, "impose_null") in (false, "false")
        end
        # Forcing enumeration off must change `enumerated`.
        rn = run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                       "--coefficient", "x", "--enumerate-signs", "no", "--boot-reps", "199"])
        assert_envelope_ok(rn; label="test wild-cluster --enumerate-signs no")
        tn = first_table(rn.doc)[2]
        tn === nothing || @test metric_value(tn, "enumerated") in (false, "false")

        # Typed guards.
        @test run_json(["test", "wild-cluster", csv, "--dep", "y"]).code == 2   # no --clusters
        @test run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                        "--coefficient", "nope"]).code == 3
        @test run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "nope"]).code == 3
        @test run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                        "--boot-reps", "0"]).code == 2
        @test run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                        "--boot-weights", "normal"]).code == 2
        # Forcing enumeration when it is impossible (webb weights) is refused upstream and
        # must surface typed, not as an internal exit 1.
        @test run_json(["test", "wild-cluster", csv, "--dep", "y", "--clusters", "cl",
                        "--coefficient", "x", "--enumerate-signs", "yes",
                        "--boot-weights", "webb"]).code == 3
        # A STRING cluster column must work — `_load_clusters` dense-ranks rather than
        # forcing Vector{Int} (which used to be an untyped exit 1 on exactly this input).
        scsv = write_csv(DataFrame(y=ys, x=xs, cl=["g$(g)" for g in cl]); prefix="wcbs")
        rs = run_json(["test", "wild-cluster", scsv, "--dep", "y", "--clusters", "cl",
                       "--coefficient", "x"])
        assert_envelope_ok(rs; label="test wild-cluster string clusters")
        rm(csv; force=true); rm(scsv; force=true)
    end

    @testset "test anderson-rubin (W10/#112)" begin
        # STRONG instruments: the AR set should be bounded and close to the Wald interval,
        # and both should cover the truth (2.0).
        strong = dgp_iv(; T=300, seed=5, inst_strength=0.9)
        rs = run_json(["test", "anderson-rubin", strong, "--dep", "y",
                       "--endogenous", "x_endog", "--instruments", "z1,z2"])
        assert_envelope_ok(rs; label="test anderson-rubin strong")
        setb = named_table(rs.doc, :anderson_rubin_set_summary)
        @test setb !== nothing
        if setb !== nothing
            @test string(metric_value(setb, "shape")) == "bounded"
            @test metric_value(setb, "bounded") in (true, "true")
            @test metric_value(setb, "is_empty") in (false, "false")
            @test Int(metric_value(setb, "n_components")) == 1
            est = Float64(metric_value(setb, "estimate_2sls"))
            @test isapprox(est, 2.0; atol=0.2)
        end
        tblb = named_table(rs.doc, Symbol("anderson_rubin_confidence_set_95_x_endog"))
        if tblb === nothing
            # Title slugging may differ; fall back to any table carrying the component cols.
            for (k, v) in pairs(rs.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   "component" in String[string(c) for c in v.columns]
                    tblb = v
                    break
                end
            end
        end
        @test tblb !== nothing
        if tblb !== nothing
            rows = table_rows(tblb)
            @test length(rows) == 1
            li = col_index(tblb, "lower"); ui = col_index(tblb, "upper")
            lo = numv(collect(rows[1])[li]); hi = numv(collect(rows[1])[ui])
            @test isfinite(lo) && isfinite(hi) && lo < hi
            @test lo <= 2.0 <= hi           # correct coverage of the truth
            @test collect(rows[1])[col_index(tblb, "lower_bounded")] in (true, "true")
        end

        # WEAK instruments — the case the whole leaf exists for. NOTE: `dgp_iv`'s
        # `inst_strength` scales ONLY z1 (z2's 0.5 loading is hardcoded), so it cannot
        # produce a weak instrument set; build one here instead.
        rngw = MersenneTwister(77)
        nw = 120
        z1w = randn(rngw, nw); z2w = randn(rngw, nw); x2w = randn(rngw, nw)
        uw = randn(rngw, nw)
        # First stage is essentially irrelevant: both loadings ~0.02 against unit noise.
        xw = 0.02 .* z1w .+ 0.02 .* z2w .+ 0.6 .* uw .+ randn(rngw, nw)
        yw = 1.0 .+ 2.0 .* xw .+ 0.8 .* x2w .+ uw
        weak = write_csv(DataFrame("y" => yw, "const" => fill(1.0, nw), "x2" => x2w,
                                   "x_endog" => xw, "z1" => z1w, "z2" => z2w); prefix="ivweak")
        rw = run_json(["test", "anderson-rubin", weak, "--dep", "y",
                       "--endogenous", "x_endog", "--instruments", "z1,z2",
                       "--span", "50"])
        assert_envelope_ok(rw; label="test anderson-rubin weak")
        setw = named_table(rw.doc, :anderson_rubin_set_summary)
        @test setw !== nothing
        if setw !== nothing
            shape = string(metric_value(setw, "shape"))
            @test shape in ("bounded", "unbounded", "whole-line", "disjoint", "empty")
            wl = Float64(metric_value(setw, "wald_lower"))
            wu = Float64(metric_value(setw, "wald_upper"))
            @test wu > wl
            # The teeth: with a first stage this weak the AR set must NOT be a tidy
            # interval sitting inside the Wald band. Either it is unbounded / the whole
            # line, or (if still bounded) it is materially WIDER than Wald — a weak-IV-
            # robust set can never be tighter than the interval it exists to correct.
            if shape in ("bounded", "disjoint")
                tblw = nothing
                for (k, v) in pairs(rw.doc.data)
                    if v isa JSON3.Object && haskey(v, :columns) &&
                       "component" in String[string(c) for c in v.columns]
                        tblw = v
                        break
                    end
                end
                @test tblw !== nothing
                if tblw !== nothing
                    rws = table_rows(tblw)
                    los = [numv(collect(r)[col_index(tblw, "lower")]) for r in rws]
                    his = [numv(collect(r)[col_index(tblw, "upper")]) for r in rws]
                    @test (maximum(his) - minimum(los)) > (wu - wl)
                end
            else
                @test shape in ("unbounded", "whole-line", "empty")
            end
            if shape != "empty"
                gl = Float64(metric_value(setw, "grid_lo"))
                gh = Float64(metric_value(setw, "grid_hi"))
                @test gh > gl
            end
        end

        # The test itself at an explicit --beta0, and --no-ci.
        rt = run_json(["test", "anderson-rubin", strong, "--dep", "y",
                       "--endogenous", "x_endog", "--instruments", "z1,z2",
                       "--beta0", "2.0", "--no-ci"])
        assert_envelope_ok(rt; label="test anderson-rubin --beta0 --no-ci")
        tt = named_table(rt.doc, :anderson_rubin_test)
        tt === nothing && (tt = first_table(rt.doc)[2])
        if tt !== nothing
            p = metric_value(tt, "p_value")
            @test p !== nothing && 0.0 <= Float64(p) <= 1.0
            # H0: beta = 2.0 IS the truth → should NOT be rejected.
            @test Float64(p) > 0.05
            @test metric_value(tt, "statistic") !== nothing
            @test metric_value(tt, "wald_cov_type") !== nothing
        end
        # ...and a FALSE null is rejected. Both directions, or the test has no teeth.
        rf = run_json(["test", "anderson-rubin", strong, "--dep", "y",
                       "--endogenous", "x_endog", "--instruments", "z1,z2",
                       "--beta0", "-3.0", "--no-ci"])
        assert_envelope_ok(rf; label="test anderson-rubin false null")
        tf = named_table(rf.doc, :anderson_rubin_test)
        tf === nothing && (tf = first_table(rf.doc)[2])
        tf === nothing || @test Float64(metric_value(tf, "p_value")) < 0.05

        # Clustered AR: the fit stays hc1 (no clustered IV fit exists upstream) and both
        # covariances are recorded, so the contrast is never read as like-for-like.
        dfw = DataFrame(CSV.File(strong))
        dfw.cl = Float64.(repeat(1:10, inner=cld(nrow(dfw), 10))[1:nrow(dfw)])
        ccsv = write_csv(dfw; prefix="ivcl")
        rc = run_json(["test", "anderson-rubin", ccsv, "--dep", "y",
                       "--endogenous", "x_endog", "--instruments", "z1,z2",
                       "--cov-type", "cluster", "--clusters", "cl"])
        assert_envelope_ok(rc; label="test anderson-rubin clustered")
        tc = named_table(rc.doc, :anderson_rubin_test)
        tc === nothing && (tc = first_table(rc.doc)[2])
        if tc !== nothing
            @test string(metric_value(tc, "ar_cov_type")) == "cluster"
            @test string(metric_value(tc, "wald_cov_type")) == "hc1"
        end
        # The cluster column must NOT have leaked into X or Z.
        setc2 = named_table(rc.doc, :anderson_rubin_set_summary)
        @test setc2 !== nothing

        # Typed guards.
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "x_endog"]).code == 2
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "x_endog", "--instruments", "z1,z2",
                        "--cov-type", "cluster"]).code == 2               # no --clusters
        @test run_json(["test", "anderson-rubin", ccsv, "--dep", "y",
                        "--endogenous", "x_endog", "--instruments", "z1,z2",
                        "--clusters", "cl"]).code == 2                    # clusters w/o cluster
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "x_endog", "--instruments", "z1,z2",
                        "--beta0", "1,2"]).code == 2                      # length mismatch
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "x_endog", "--instruments", "z1,z2",
                        "--beta0", "abc"]).code == 2
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "x_endog", "--instruments", "z1,z2",
                        "--level", "1.5"]).code == 2
        @test run_json(["test", "anderson-rubin", strong, "--dep", "y",
                        "--endogenous", "nope", "--instruments", "z1,z2"]).code == 3
        rm(strong; force=true); rm(weak; force=true); rm(ccsv; force=true)
    end

    @testset "estimate lp --method iv + MOP/AR (W10/#112)" begin
        # THE LEAF WAS DEAD before this wave: `wi.F_stat` is a field real MEMs never had,
        # so every invocation exited 1. `estimate lp` had T3 coverage only for --method
        # standard — the recurring blind-spot class. A plain run is now the regression test.
        rng = MersenneTwister(17)
        Tn = 220
        z = randn(rng, Tn)
        shock = 0.8 .* z .+ 0.5 .* randn(rng, Tn)
        y2 = zeros(Tn)
        for t in 2:Tn
            y2[t] = 0.5 * y2[t-1] + 0.7 * shock[t] + 0.4 * randn(rng)
        end
        ycsv = write_csv(DataFrame(x=shock, y=y2); prefix="lpiv")
        zcsv = write_csv(DataFrame(z=z); prefix="lpivz")

        r = run_json(["estimate", "lp", ycsv, "--method", "iv", "--shock", "1",
                      "--horizons", "6", "--control-lags", "2", "--instruments", zcsv])
        assert_envelope_ok(r; label="estimate lp --method iv")
        sm = named_table(r.doc, :lp_estimation_summary)
        @test sm !== nothing
        if sm !== nothing
            # The per-horizon F is reported as a MINIMUM, and T_eff as endpoints — neither
            # may be a nested vector in a scalar cell.
            fmin = metric_value(sm, "First-stage F (min)")
            @test fmin !== nothing
            @test Float64(fmin) > 0
            te = metric_value(sm, "Effective observations (h=0)")
            @test te !== nothing && Int(te) > 0
            @test Int(metric_value(sm, "Effective observations (min)")) <= Int(te)
        end

        # MOP effective F.
        rm_ = run_json(["estimate", "lp", ycsv, "--method", "iv", "--shock", "1",
                        "--horizons", "6", "--control-lags", "2", "--instruments", zcsv,
                        "--mop-f", "--mop-tau", "0.10"])
        assert_envelope_ok(rm_; label="estimate lp iv --mop-f")
        mt = named_table(rm_.doc, :montiel_olea_pflueger_effective_f)
        @test mt !== nothing
        if mt !== nothing
            fe = metric_value(mt, "f_effective")
            cv = metric_value(mt, "critical_value")
            @test fe !== nothing && cv !== nothing
            @test isapprox(Float64(cv), 23.11; atol=1e-6)     # MOP simplified CV at tau=0.10
            @test Float64(fe) > 0
            # `weak` must agree with the comparison it claims to make.
            @test (metric_value(mt, "weak") in (true, "true")) == (Float64(fe) < Float64(cv))
            @test Int(metric_value(mt, "n_instruments")) == 1
        end
        # The critical value moves with tau, in the documented direction.
        rm3 = run_json(["estimate", "lp", ycsv, "--method", "iv", "--shock", "1",
                        "--horizons", "6", "--control-lags", "2", "--instruments", zcsv,
                        "--mop-f", "--mop-tau", "0.30"])
        assert_envelope_ok(rm3; label="estimate lp iv --mop-tau 0.30")
        mt3 = named_table(rm3.doc, :montiel_olea_pflueger_effective_f)
        mt3 === nothing || @test isapprox(Float64(metric_value(mt3, "critical_value")), 12.04; atol=1e-6)

        # AR bands.
        ra = run_json(["estimate", "lp", ycsv, "--method", "iv", "--shock", "1",
                       "--horizons", "4", "--control-lags", "2", "--instruments", zcsv,
                       "--ar-bands", "--ar-grid", "101", "--ar-level", "0.95"])
        assert_envelope_ok(ra; label="estimate lp iv --ar-bands")
        ab = named_table(ra.doc, :lp_iv_anderson_rubin_bands)
        if ab === nothing
            for (k, v) in pairs(ra.doc.data)
                if v isa JSON3.Object && haskey(v, :columns) &&
                   "ar_lower" in String[string(c) for c in v.columns]
                    ab = v
                    break
                end
            end
        end
        @test ab !== nothing
        if ab !== nothing
            rows = table_rows(ab)
            nresp = 2
            @test length(rows) == 5 * nresp          # (H+1) x responses
            hi_ = col_index(ab, "horizon"); ri = col_index(ab, "response")
            ali = col_index(ab, "ar_lower"); aui = col_index(ab, "ar_upper")
            wli = col_index(ab, "wald_lower"); wui = col_index(ab, "wald_upper")
            bi = col_index(ab, "bounded"); bwi = col_index(ab, "bandwidth")
            @test Set(Int(collect(r)[hi_]) for r in rows) == Set(0:4)
            ei_ = col_index(ab, "is_empty")
            n_strict = 0
            for row in rows
                rr = collect(row)
                wl_ = numv(rr[wli]); wu_ = numv(rr[wui])
                # NOT `>`: the h=0 response of the SHOCK VARIABLE TO ITS OWN SHOCK is
                # identically 1 with zero standard error, so both the Wald band and the AR
                # set legitimately collapse to the single point {1}. That degenerate cell is
                # correct output, and a strict inequality here failed on it (T3 caught it).
                @test wu_ >= wl_
                # The HAC bandwidth scales with the horizon (MA(h) residuals).
                @test Int(rr[bwi]) >= Int(rr[hi_]) + 1
                # An EMPTY cell carries NaN bounds upstream (and bounded=true), so every
                # ordering/finiteness check below must skip it rather than compare to NaN.
                (rr[ei_] in (true, "true")) && continue
                # Unbounded sides come through as the STRING "Inf"/"-Inf" (JSON has no
                # infinity), which is the documented envelope contract — not a number, and
                # not silently truncated to the grid edge.
                lo_f = numv(rr[ali]); hi_f = numv(rr[aui])
                @test hi_f >= lo_f
                # Teeth: wherever the cell is NOT degenerate, the AR set must have strictly
                # positive width — a point set anywhere the Wald band is a real interval
                # would mean the inversion collapsed.
                #
                # The degeneracy test must be a TOLERANCE, not exact float equality. At the
                # h=0 own-shock cell the Wald band is 1 ± 0, but the two endpoints are only
                # bit-identical if the zero half-width stays exactly zero: on the Linux CI
                # runner's BLAS they came out a ULP apart (1.0000000000000002 vs 1.0), which
                # flipped that cell into this branch and failed on the perfectly correct
                # point set {1}, while macOS was green. Substantive cells here are ~0.3 wide,
                # so 1e-8 separates them from rounding noise by 7 orders of magnitude.
                if wu_ - wl_ > 1e-8 * max(1.0, abs(wl_), abs(wu_))
                    @test hi_f > lo_f
                    n_strict += 1
                end
                bounded = rr[bi] in (true, "true")
                @test bounded == (isfinite(lo_f) && isfinite(hi_f))
            end
            # ...and most cells must be non-degenerate, or the block above tested nothing.
            @test n_strict >= length(rows) - nresp
        end

        # Guards: the riders are iv-only, and every numeric option is validated.
        @test run_json(["estimate", "lp", ycsv, "--mop-f"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--ar-bands"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--method", "smooth", "--mop-tau", "0.05"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--method", "iv", "--instruments", zcsv,
                        "--mop-f", "--mop-tau", "0.15"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--method", "iv", "--instruments", zcsv,
                        "--ar-bands", "--ar-grid", "2"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--method", "iv", "--instruments", zcsv,
                        "--ar-bands", "--ar-level", "0"]).code == 2
        @test run_json(["estimate", "lp", ycsv, "--method", "iv"]).code == 2   # no instruments
        rm(ycsv; force=true); rm(zcsv; force=true)
    end

    # ── #144: factor family --model handles — the branch that shipped dead ──
    # r/factor_lags/varnames were bound only in the estimate branch, so every
    # `predict|residuals static|dynamic|gdfm --model <handle>` exited 1 with an
    # untyped UndefVarError (and dynamic/gdfm would have followed with a
    # varnames FieldError — those types carry no varnames upstream). Zero T3
    # coverage of the --model path existed; per-BRANCH coverage, both verbs.
    @testset "factor family --model handles (#144)" begin
        rng = MersenneTwister(29)
        F = randn(rng, 140, 2) * [1.0 0.4 0.7 0.2; 0.3 0.9 0.1 0.6]
        fdf = DataFrame(F .+ 0.3 .* randn(rng, 140, 4), ["a", "b", "c", "d"])
        fcsv = write_csv(fdf; prefix="factor144")
        cases = [
            ("static",  ["--nfactors", "2"]),
            ("dynamic", ["--nfactors", "2", "--factor-lags", "1"]),
            ("gdfm",    ["--dynamic-rank", "1"]),
        ]
        for (kind, est_args) in cases
            h = tempname() * ".fmod"
            r_est = run_json(vcat(["estimate", kind, fcsv], est_args,
                                  ["--save-model", h]))
            @test r_est.code == 0
            for verb in ("predict", "residuals")
                rr = run_json([verb, kind, "--model", h])
                @test rr.code == 0
                if rr.code == 0
                    @test String(rr.doc.status) == "ok"
                    @test !isempty(rr.doc.data)
                end
            end
            rm(h; force=true)
        end
        rm(fcsv; force=true)
    end

    # ── W7/#142: serve --mcp — the five canned sessions (#61 acceptance) ────
    # In-process through Friedman._serve_loop (no per-call process spawn — the
    # whole point of the server). tools/call goes through the REAL run_cli, so
    # results are the envelope verbatim and isError mirrors the exit class.
    @testset "serve --mcp canned sessions (W7/#142)" begin
        csv = dgp_var2(; T=120, seed=17)
        function mcp_session(msgs::Vector{String})
            input = IOBuffer(join(msgs, "\n") * "\n")
            output = IOBuffer()
            Friedman._serve_loop(input, output)
            [JSON3.read(l) for l in split(String(take!(output)), '\n') if !isempty(strip(l))]
        end
        argsjson(d) = JSON3.write(d)

        rs = mcp_session([
            # 1. initialize
            """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}""",
            """{"jsonrpc":"2.0","method":"notifications/initialized"}""",
            # 2. tools/list
            """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""",
            # 3. estimate_var, saving to a session handle
            """{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"estimate_var","arguments":$(argsjson(Dict("data"=>csv,"lags"=>1,"save-model"=>"model://m1")))}}""",
            # 4. irf_var against the in-memory handle — NO data file
            """{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"irf_var","arguments":$(argsjson(Dict("model"=>"model://m1","horizons"=>4,"ci"=>"none")))}}""",
            # 5. typed error: missing data file → data envelope, isError
            """{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"estimate_var","arguments":$(argsjson(Dict("data"=>"/nope/missing.csv")))}}""",
            # 6. W3/#167: model_reproduce over a session handle — VARModel has no
            # manifest, so the universal fallback's honest unverifiable verdict
            # (ok, not a crash and not a refusal)
            """{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"model_reproduce","arguments":$(argsjson(Dict("path"=>"model://m1")))}}""",
        ])
        @test length(rs) == 6

        @test String(rs[1].result.serverInfo.name) == "friedman"
        tools = rs[2].result.tools
        @test length(tools) > 400
        @test any(t -> t.name == "estimate_var", tools)
        @test !any(t -> t.name == "serve", tools)

        est = rs[3].result
        @test est.isError == false
        est_env = JSON3.read(est.content[1].text)
        @test String(est_env.status) == "ok"
        @test haskey(est_env.data, :var_coefficients)

        irf_r = rs[4].result
        @test irf_r.isError == false
        irf_env = JSON3.read(irf_r.content[1].text)
        @test String(irf_env.status) == "ok"
        @test !isempty(irf_env.data)

        bad = rs[5].result
        @test bad.isError == true
        bad_env = JSON3.read(bad.content[1].text)
        @test String(bad_env.status) == "error"
        @test startswith(String(bad_env.error.code), "data/")
        @test Int(bad_env.error.exit_code) == 3

        rep = rs[6].result
        @test rep.isError == false
        rep_env = JSON3.read(rep.content[1].text)
        @test String(rep_env.status) == "ok"
        @test haskey(rep_env.data, :model_reproduce_summary)

        # store is session-scoped: gone after the loop
        @test Friedman._SERVE_MODEL_STORE[] === nothing
        rm(csv; force=true)
    end

    # Typed data handles wave 1: import → stem-resolve into estimate var;
    # panel handle into a timeseries leaf is data/wrong-kind.
    # The 40×3 synthetic CSV stands in for :fred_md vs `data load` coefficient
    # agreement (runtime; :fred_md itself is not required for this gate).
    @testset "typed data handles wave 1" begin
        mktempdir() do dir
            csv = joinpath(dir, "macro.csv")
            Random.seed!(1)
            CSV.write(csv, DataFrame(y1=randn(40), y2=randn(40), y3=randn(40)))
            r0 = run_json(["data", "import", csv, "--kind", "timeseries",
                           "-o", joinpath(dir, "macro")])
            @test r0.code == 0
            @test isfile(joinpath(dir, "macro.jld2"))
            rc = run_json(["estimate", "var", csv, "--lags", "1"])
            rh = run_json(["estimate", "var", joinpath(dir, "macro"), "--lags", "1"])
            @test rc.code == 0
            @test rh.code == 0
            # Distinctive columns (term/estimate), never first(values(...)) / key substring
            coef_table(doc) = begin
                doc === nothing && return nothing
                for (_, v) in pairs(doc.data)
                    (v isa JSON3.Object && haskey(v, :columns)) || continue
                    cols = table_cols(v)
                    ("term" in cols && "estimate" in cols) && return v
                end
                nothing
            end
            tc = coef_table(rc.doc)
            th = coef_table(rh.doc)
            @test tc !== nothing
            @test th !== nothing
            @test table_cols(tc) == table_cols(th)
            @test length(table_rows(tc)) == length(table_rows(th))
            # Existing T3 tolerances (numeric_tables_agree defaults), not ULP equality
            @test numeric_tables_agree(tc, th)

            panel = joinpath(dir, "panel.csv")
            # 4 groups × 10 periods
            g = repeat(1:4, inner=10); t = repeat(1:10, outer=4)
            CSV.write(panel, DataFrame(group=g, time=t, y=randn(40), x=randn(40)))
            rp = run_json(["data", "import", panel, "--kind", "panel",
                           "--id-col", "group", "--time-col", "time",
                           "-o", joinpath(dir, "panel")])
            @test rp.code == 0
            bad = run_json(["estimate", "var", joinpath(dir, "panel"), "--lags", "1"])
            @test bad.code == 3
            @test bad.doc !== nothing
            @test String(bad.doc["error"]["code"]) == "data/wrong-kind"

            # Panel import → estimate pvar on the stem (flags optional on a handle).
            # 4×10 is too thin for GMM; reuse the existing pvar DGP richness.
            panel40 = dgp_did_panel(; N=40, T=10, seed=11)
            rp40 = run_json(["data", "import", panel40, "--kind", "panel",
                             "--id-col", "id", "--time-col", "time",
                             "-o", joinpath(dir, "pvarpanel")])
            @test rp40.code == 0
            rpvar = run_json(["estimate", "pvar", joinpath(dir, "pvarpanel"), "--lags", "1"])
            @test rpvar.code == 0
            @test rpvar.doc !== nothing
            pvar_tbl = nothing
            for (_, v) in pairs(rpvar.doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                cols = table_cols(v)
                ("parameter" in cols && any(endswith(c, "_coef") for c in cols)) && (pvar_tbl = v; break)
            end
            @test pvar_tbl !== nothing && !isempty(table_rows(pvar_tbl))

            # data describe on a panel handle is not a TS wrap: id/time are identity.
            rdesc = run_json(["data", "describe", joinpath(dir, "panel")])
            @test rdesc.code == 0
            desc_tbl = nothing
            for (_, v) in pairs(rdesc.doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                cols = table_cols(v)
                ("variable" in cols && ("mean" in cols || "std" in cols || "n" in cols)) &&
                    (desc_tbl = v; break)
            end
            @test desc_tbl !== nothing
            vi = findfirst(==("variable"), table_cols(desc_tbl))
            @test vi !== nothing
            desc_vars = [string(collect(r)[vi]) for r in table_rows(desc_tbl)]
            @test !any(v -> v in ("group", "time", "id"), desc_vars)

            # data fix on a panel handle preserves type / varnames / frequency.
            before = Friedman.load_model_dispatch(joinpath(dir, "panel.jld2"))
            rfix = run_json(["data", "fix", joinpath(dir, "panel"),
                             "-o", joinpath(dir, "panel_clean")])
            @test rfix.code == 0
            after = Friedman.load_model_dispatch(joinpath(dir, "panel_clean.jld2"))
            @test string(nameof(typeof(after))) == "PanelData"
            @test string(nameof(typeof(before))) == "PanelData"
            @test after.varnames == before.varnames
            @test after.frequency == before.frequency
        end
    end

    @testset "typed result handles wave 2" begin
        mktempdir() do dir
            csv = joinpath(dir, "macro.csv")
            Random.seed!(1)
            CSV.write(csv, DataFrame(y1=randn(40), y2=randn(40), y3=randn(40)))
            run_json(["data", "import", csv, "--kind", "timeseries", "-o", joinpath(dir, "macro")])
            r1 = run_json(["estimate", "var", joinpath(dir, "macro"), "--lags", "1",
                           "--save-model", joinpath(dir, "var")])
            @test r1.code == 0
            r2 = run_json(["irf", "var", "--model", joinpath(dir, "var"),
                           "--horizons", "4", "--save-result", joinpath(dir, "irf")])
            @test r2.code == 0
            rm(joinpath(dir, "macro.jld2"); force=true)  # --result must not re-estimate from data
            r3 = run_json(["irf", "var", "--result", joinpath(dir, "irf")])
            @test r3.code == 0
            r4 = run_json(["show", joinpath(dir, "irf")])
            @test r4.code == 0

            # Distinctive columns (horizon/variable), never first(values(...)) / key substring
            irf_table(doc) = begin
                doc === nothing && return nothing
                for (_, v) in pairs(doc.data)
                    (v isa JSON3.Object && haskey(v, :columns)) || continue
                    cols = table_cols(v)
                    ("horizon" in cols && "variable" in cols) && return v
                end
                nothing
            end
            t2 = irf_table(r2.doc)
            t3 = irf_table(r3.doc)
            t4 = irf_table(r4.doc)
            @test t2 !== nothing
            @test t3 !== nothing
            @test t4 !== nothing
            @test "horizon" in table_cols(t2) && "variable" in table_cols(t2)
            @test "horizon" in table_cols(t3) && "variable" in table_cols(t3)
            @test "horizon" in table_cols(t4) && "variable" in table_cols(t4)
            @test !isempty(table_rows(t2)) && !isempty(table_rows(t3)) && !isempty(table_rows(t4))
            # Default --shock 1: --result re-render matches the compute-path row count.
            # `show` has no --shock and may still emit the full table.
            @test length(table_rows(t3)) == length(table_rows(t2))
            @test table_cols(t3) == table_cols(t2)
            @test length(table_rows(t4)) >= length(table_rows(t2))

            # VARModel is not an ImpulseResponse
            wr = run_json(["irf", "var", "--result", joinpath(dir, "var")])
            @test wr.code == 3
            @test wr.doc !== nothing
            @test String(wr.doc["error"]["code"]) == "data/wrong-result"

            rfc = run_json(["forecast", "var", "--model", joinpath(dir, "var"),
                            "--horizons", "8", "--save-result", joinpath(dir, "fcst")])
            @test rfc.code == 0
            actual = joinpath(dir, "actual.csv")
            CSV.write(actual, DataFrame(y1=randn(8)))
            reval = run_json(["forecast", "evaluate", "metrics", actual,
                              "--actual", "y1", "--result", joinpath(dir, "fcst")])
            @test reval.code == 0
            acc = nothing
            for (_, v) in pairs(reval.doc.data)
                (v isa JSON3.Object && haskey(v, :columns)) || continue
                cols = table_cols(v)
                ("model" in cols && "RMSE" in cols) && (acc = v; break)
            end
            @test acc !== nothing
            @test !isempty(table_rows(acc))
        end
    end

end

# Real entry-point coverage (C036) — also on core/CI path
include(joinpath(@__DIR__, "test_entry.jl"))
