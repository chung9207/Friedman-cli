# Friedman-cli installer for Windows
# Usage:
#   irm https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.ps1 | iex
#
# Specific version (set env var before piping):
#   $env:FRIEDMAN_VERSION = "0.4.0"; irm https://...install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "FriedmanJP/Friedman-cli"
$InstallDir = Join-Path $env:USERPROFILE ".friedman-cli"
$BinDir = Join-Path $InstallDir "bin"

# --- Parse version ---
$Version = $env:FRIEDMAN_VERSION
if (-not $Version) {
    Write-Host "Fetching latest release..."
    try {
        $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest"
        $Version = $Release.tag_name -replace '^v', ''
    } catch {
        Write-Host "Error: Failed to fetch latest release from GitHub API." -ForegroundColor Red
        Write-Host 'You may be rate-limited. Try setting $env:FRIEDMAN_VERSION = "0.4.0" before running.' -ForegroundColor Yellow
        exit 1
    }
}

if (-not $Version) {
    Write-Host "Error: Could not determine version." -ForegroundColor Red
    exit 1
}

Write-Host "Installing Friedman-cli v$Version..."

# --- Construct download URL ---
$ArchiveName = "friedman-v$Version-windows-x86_64.zip"
$DownloadUrl = "https://github.com/$Repo/releases/download/v$Version/$ArchiveName"

# --- Ensure Julia 1.12 is available ---
function Ensure-Julia {
    # Check if juliaup is available
    if (Get-Command juliaup -ErrorAction SilentlyContinue) {
        Write-Host "Found juliaup. Ensuring Julia 1.12 is installed..."
        & juliaup add 1.12 2>$null
        return
    }

    # Check if julia >= 1.12 is on PATH
    if (Get-Command julia -ErrorAction SilentlyContinue) {
        $JuliaVer = & julia --version 2>&1
        if ($JuliaVer -match '(\d+)\.(\d+)') {
            $Major = [int]$Matches[1]
            $Minor = [int]$Matches[2]
            if ($Major -ge 1 -and $Minor -ge 12) {
                Write-Host "Found $JuliaVer"
                return
            }
        }
    }

    # Install juliaup via winget
    Write-Host "Julia 1.12+ not found. Installing juliaup..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id Julialang.Juliaup --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install juliaup via winget." -ForegroundColor Red
            Write-Host "Install Julia manually: https://julialang.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "Machine")
        Write-Host "Installing Julia 1.12..."
        & juliaup add 1.12
    } else {
        Write-Host "Error: winget is not available." -ForegroundColor Red
        Write-Host "Install juliaup manually: https://julialang.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
}

Ensure-Julia

# --- Download archive ---
$TmpDir = Join-Path ([System.IO.Path]::GetTempPath()) "friedman-install-$(Get-Random)"
New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null

try {
    $ArchivePath = Join-Path $TmpDir $ArchiveName
    Write-Host "Downloading $ArchiveName..."
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath -ErrorAction Stop

    # --- Extract to temp, then safe-replace install dir ---
    Write-Host "Installing to $InstallDir..."
    $ExtractDir = Join-Path $TmpDir "extract"
    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force

    # The archive contains a top-level friedman/ directory
    $Extracted = Join-Path $ExtractDir "friedman"
    if (-not (Test-Path $Extracted)) {
        $Extracted = $ExtractDir
    }

    # Safe replacement
    if (Test-Path $InstallDir) {
        Remove-Item -Path $InstallDir -Recurse -Force
    }
    Move-Item -Path $Extracted -Destination $InstallDir

    # --- Add to PATH ---
    $CurrentPath = [System.Environment]::GetEnvironmentVariable("PATH", "User")
    if ($CurrentPath -notlike "*$BinDir*") {
        [System.Environment]::SetEnvironmentVariable("PATH", "$BinDir;$CurrentPath", "User")
        $env:PATH = "$BinDir;$env:PATH"
        Write-Host "Added $BinDir to user PATH."
    }

    # --- Verify ---
    $FriedmanCmd = Join-Path $BinDir "friedman.cmd"
    if (Test-Path $FriedmanCmd) {
        Write-Host ""
        & $FriedmanCmd --version
        Write-Host "Friedman-cli installed successfully!" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "Friedman-cli installed to $InstallDir" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "To uninstall:"
    Write-Host "  Remove-Item -Recurse -Force '$InstallDir'"
    Write-Host "  Then remove '$BinDir' from your PATH in System Settings > Environment Variables"

} catch {
    Write-Host "Error: Failed to download or install." -ForegroundColor Red
    Write-Host "Check that version v$Version exists at:" -ForegroundColor Yellow
    Write-Host "  https://github.com/$Repo/releases" -ForegroundColor Yellow
    exit 1
} finally {
    # Clean up temp
    Remove-Item -Path $TmpDir -Recurse -Force -ErrorAction SilentlyContinue
}
