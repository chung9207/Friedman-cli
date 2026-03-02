# Installation

## Requirements

- **Julia 1.12+**
- Git

## Install from Source

```bash
git clone https://github.com/FriedmanJP/Friedman-cli.git
cd Friedman-cli
julia --project -e '
using Pkg
Pkg.rm("MacroEconometricModels")
Pkg.add(url="https://github.com/FriedmanJP/MacroEconometricModels.jl.git")
'
```

### Optional Dependencies

For DSGE constrained solvers (OccBin), install JuMP and Ipopt:

```bash
julia --project -e 'using Pkg; Pkg.add(["JuMP", "Ipopt"])'
```

For GDFM spectral methods, install FFTW:

```bash
julia --project -e 'using Pkg; Pkg.add("FFTW")'
```

This installs `MacroEconometricModels.jl` directly from GitHub along with all other dependencies (`CSV`, `DataFrames`, `PrettyTables`, `JSON3`).

## Running

```bash
julia --project bin/friedman [command] [subcommand] [args...] [options...]
```

For convenience, you can create a shell alias:

```bash
# bash/zsh
alias friedman='julia --project=/path/to/Friedman-cli /path/to/Friedman-cli/bin/friedman'

# fish
alias friedman 'julia --project=/path/to/Friedman-cli /path/to/Friedman-cli/bin/friedman'
```

## Sysimage (Optional)

For faster startup, build a sysimage with PackageCompiler:

```bash
julia --project -e '
using PackageCompiler
create_sysimage(["Friedman"]; sysimage_path="friedman.so",
    precompile_execution_file="bin/friedman")
'

# Run with sysimage
julia --project --sysimage=friedman.so bin/friedman estimate var data.csv
```

## Testing

```bash
# Run all tests (no MacroEconometricModels dependency needed)
julia --project test/runtests.jl
```
