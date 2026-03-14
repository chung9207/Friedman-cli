# Installation

## Quick Install

### macOS and Linux

```bash
curl -fsSL https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.sh | bash
```

### Windows (PowerShell)

```powershell
irm https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.ps1 | iex
```

## What the Installer Does

1. **Checks for Julia 1.12** — if not found, installs [juliaup](https://github.com/JuliaLang/juliaup) (the official Julia version manager) and adds Julia 1.12. Your default Julia version is never changed.
2. **Downloads a precompiled sysimage** — platform-specific binary from GitHub Releases (~670 MB)
3. **Installs to `~/.friedman-cli/`** — self-contained directory with sysimage, source, and launcher
4. **Adds to PATH** — creates a symlink in `~/.local/bin/` (macOS/Linux) or adds to user PATH (Windows)

## Install a Specific Version

### macOS/Linux

```bash
curl -fsSL https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.sh | bash -s -- --version 0.4.0
```

### Windows

```powershell
$env:FRIEDMAN_VERSION = "0.4.0"; irm https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.ps1 | iex
```

## Manual Install from GitHub Releases

1. Go to [Releases](https://github.com/FriedmanJP/Friedman-cli/releases)
2. Download the archive for your platform:
   - `friedman-vX.Y.Z-darwin-arm64.tar.gz` (macOS Apple Silicon)
   - `friedman-vX.Y.Z-linux-x86_64.tar.gz` (Linux x64)
   - `friedman-vX.Y.Z-windows-x86_64.zip` (Windows x64)
3. Extract to `~/.friedman-cli/`
4. Add `~/.friedman-cli/bin` to your PATH

**Requires:** Julia 1.12+ installed via [juliaup](https://github.com/JuliaLang/juliaup) or manually.

## Upgrade

Re-run the install command. The installer replaces the existing installation.

## Uninstall

### macOS/Linux

```bash
rm -rf ~/.friedman-cli ~/.local/bin/friedman
```

### Windows

```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.friedman-cli"
```

Then remove `%USERPROFILE%\.friedman-cli\bin` from your user PATH in System Settings.

## Build from Source

For development or if you prefer to build locally:

```bash
git clone https://github.com/FriedmanJP/Friedman-cli.git
cd Friedman-cli
julia --project -e '
  using Pkg
  Pkg.rm("MacroEconometricModels")
  Pkg.add(url="https://github.com/FriedmanJP/MacroEconometricModels.jl.git")
'
```

Run directly:

```bash
julia --project bin/friedman [command] [subcommand] [args] [options]
```

Or build a local sysimage:

```bash
julia build_release.jl
~/.friedman-cli/bin/friedman --version
```

## Optional Dependencies

For DSGE constrained optimization, install JuMP and solver packages:

```julia
using Pkg
Pkg.add(["JuMP", "Ipopt"])
```

These are **not** included in precompiled release builds due to license incompatibility (Ipopt uses EPL-2.0, which conflicts with GPL-3.0).

## Testing

```bash
# Run all tests (no MacroEconometricModels dependency needed)
julia --project test/runtests.jl
```
