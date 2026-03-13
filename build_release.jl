# build_release.jl — Cross-platform build for CI releases
# Run: julia build_release.jl
#
# Produces: build/friedman/ with platform-appropriate sysimage and launcher
#
# This script:
# 1. Creates a temporary build environment (weak deps excluded for license compat)
# 2. Builds a sysimage via PackageCompiler.create_sysimage()
# 3. Creates a self-contained app directory with sysimage + launcher
# 4. Does NOT modify the source Project.toml
#
# Note: JuMP/Ipopt/PATHSolver are excluded from the sysimage because Ipopt's
# EPL-2.0 license is incompatible with GPL-3.0. Users who need DSGE constrained
# optimization can install them separately: Pkg.add(["JuMP", "Ipopt"])

using Pkg

project_dir = @__DIR__
build_project_dir = joinpath(project_dir, "build_env")
app_dir = joinpath(project_dir, "build", "friedman")

# --- Platform detection ---
sysimage_ext = Sys.iswindows() ? ".dll" : Sys.isapple() ? ".dylib" : ".so"
sysimage_name = "friedman$(sysimage_ext)"

# --- Step 1: Create build environment with all deps ---
println("Setting up build environment...")
rm(build_project_dir; force=true, recursive=true)
mkpath(build_project_dir)

# Copy source files
cp(joinpath(project_dir, "src"), joinpath(build_project_dir, "src"))
cp(joinpath(project_dir, "bin"), joinpath(build_project_dir, "bin"))

# Read original Project.toml, drop weakdeps and extensions (EPL-incompatible)
original_toml = Pkg.TOML.parsefile(joinpath(project_dir, "Project.toml"))
delete!(original_toml, "weakdeps")
delete!(original_toml, "extensions")

# Write Project.toml without weakdeps
open(joinpath(build_project_dir, "Project.toml"), "w") do io
    Pkg.TOML.print(io, original_toml)
end

# Activate build env and install deps (weak deps excluded)
Pkg.activate(build_project_dir)
Pkg.instantiate()

# Add REPL stdlib so interactive mode can load it at runtime
println("Adding REPL stdlib for interactive mode...")
Pkg.add("REPL")

# --- Step 2: Install PackageCompiler ---
println("Loading PackageCompiler...")
Pkg.activate(; temp=true)
Pkg.add("PackageCompiler")
using PackageCompiler
Pkg.activate(build_project_dir)

# --- Step 3: Generate precompile script ---
precompile_script = joinpath(build_project_dir, "precompile_app.jl")
open(precompile_script, "w") do io
    write(io, """
    using Friedman
    app = Friedman.build_app()
    Friedman.dispatch(app, ["--help"])
    Friedman.dispatch(app, ["estimate", "--help"])
    Friedman.dispatch(app, ["test", "--help"])
    Friedman.dispatch(app, ["irf", "--help"])
    Friedman.dispatch(app, ["forecast", "--help"])
    Friedman.dispatch(app, ["filter", "--help"])
    Friedman.dispatch(app, ["data", "--help"])
    Friedman.dispatch(app, ["dsge", "--help"])
    Friedman.dispatch(app, ["did", "--help"])
    Friedman.dispatch(app, ["spectral", "--help"])
    Friedman.dispatch(app, ["nowcast", "--help"])
    Friedman.dispatch(app, ["--version"])
    """)
end

# --- Step 4: Build sysimage ---
sysimage_path = joinpath(build_project_dir, sysimage_name)
println("Building sysimage ($(sysimage_name))...")
println("This will take several minutes.")

create_sysimage(
    [:Friedman];
    sysimage_path=sysimage_path,
    precompile_execution_file=precompile_script,
    project=build_project_dir,
)

# --- Step 5: Bundle into app directory ---
println("Bundling app...")
rm(app_dir; force=true, recursive=true)
mkpath(joinpath(app_dir, "bin"))
mkpath(joinpath(app_dir, "lib"))

# Copy sysimage
cp(sysimage_path, joinpath(app_dir, "lib", sysimage_name))

# Copy project files for LOAD_PATH
cp(joinpath(build_project_dir, "Project.toml"), joinpath(app_dir, "Project.toml"))
if isfile(joinpath(build_project_dir, "Manifest.toml"))
    cp(joinpath(build_project_dir, "Manifest.toml"), joinpath(app_dir, "Manifest.toml"))
end
cp(joinpath(build_project_dir, "src"), joinpath(app_dir, "src"))

# --- Step 5a: Create platform-appropriate launcher ---
if Sys.iswindows()
    # Windows batch launcher
    launcher = joinpath(app_dir, "bin", "friedman.cmd")
    open(launcher, "w") do io
        write(io, """@echo off
rem Friedman-cli — compiled launcher
rem Uses precompiled sysimage for instant startup

set "SCRIPT_DIR=%~dp0.."
set "SYSIMAGE=%SCRIPT_DIR%\\lib\\$(sysimage_name)"

set "JULIA_LOAD_PATH=%SCRIPT_DIR%;@stdlib"

rem Find Julia: prefer juliaup, fallback to julia on PATH
where juliaup >nul 2>&1
if %errorlevel% equ 0 (
    juliaup run +1.12 julia -- --project="%SCRIPT_DIR%" --sysimage="%SYSIMAGE%" --startup-file=no -e "using Friedman; Friedman.main(ARGS)" -- %*
    exit /b %errorlevel%
)

where julia >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%v in ('julia --version 2^>^&1') do set "JULIA_VER=%%v"
    julia --project="%SCRIPT_DIR%" --sysimage="%SYSIMAGE%" --startup-file=no -e "using Friedman; Friedman.main(ARGS)" -- %*
    exit /b %errorlevel%
)

echo Error: Julia 1.12+ is required but not found.
echo Install via: winget install --id Julialang.Juliaup
echo Then run: juliaup add 1.12
exit /b 1
""")
    end
else
    # macOS/Linux bash launcher
    launcher = joinpath(app_dir, "bin", "friedman")
    open(launcher, "w") do io
        write(io, """#!/bin/bash
# Friedman-cli — compiled launcher
# Uses precompiled sysimage for instant startup

# Resolve symlinks (macOS compatible)
SOURCE="\$0"
while [ -L "\$SOURCE" ]; do
    DIR="\$(cd "\$(dirname "\$SOURCE")" && pwd)"
    SOURCE="\$(readlink "\$SOURCE")"
    [[ "\$SOURCE" != /* ]] && SOURCE="\$DIR/\$SOURCE"
done
SCRIPT_DIR="\$(cd "\$(dirname "\$SOURCE")/.." && pwd)"
SYSIMAGE="\$SCRIPT_DIR/lib/$(sysimage_name)"

export JULIA_LOAD_PATH="\$SCRIPT_DIR:@stdlib"

# Find Julia: prefer juliaup run +1.12, fallback to julia on PATH
if command -v juliaup >/dev/null 2>&1; then
    exec juliaup run +1.12 julia -- \\
        --project="\$SCRIPT_DIR" \\
        --sysimage="\$SYSIMAGE" \\
        --startup-file=no \\
        -e 'using Friedman; Friedman.main(ARGS)' \\
        -- "\$@"
elif command -v julia >/dev/null 2>&1; then
    JULIA_VER=\$(julia --version 2>&1 | grep -oE '[0-9]+\\.[0-9]+' | head -1)
    JULIA_MAJOR=\$(echo "\$JULIA_VER" | cut -d. -f1)
    JULIA_MINOR=\$(echo "\$JULIA_VER" | cut -d. -f2)
    if [ "\$JULIA_MAJOR" -ge 1 ] && [ "\$JULIA_MINOR" -ge 12 ]; then
        exec julia \\
            --project="\$SCRIPT_DIR" \\
            --sysimage="\$SYSIMAGE" \\
            --startup-file=no \\
            -e 'using Friedman; Friedman.main(ARGS)' \\
            -- "\$@"
    fi
fi

echo "Error: Julia 1.12+ is required but not found." >&2
echo "Install via: curl -fsSL https://install.julialang.org | sh -s -- --yes" >&2
echo "Then run: juliaup add 1.12" >&2
exit 1
""")
    end
    chmod(launcher, 0o755)
end

# --- Step 6: Clean up build env ---
rm(build_project_dir; force=true, recursive=true)

println()
println("Done! Compiled app: $(app_dir)/bin/friedman$(Sys.iswindows() ? ".cmd" : "")")
println("Sysimage: $(app_dir)/lib/$(sysimage_name)")
