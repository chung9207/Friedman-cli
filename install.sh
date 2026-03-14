#!/bin/bash
set -euo pipefail

# Friedman-cli installer for macOS and Linux
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/FriedmanJP/Friedman-cli/master/install.sh | bash
#   curl -fsSL https://...install.sh | bash -s -- --version 0.4.0

REPO="FriedmanJP/Friedman-cli"
INSTALL_DIR="$HOME/.friedman-cli"
BIN_DIR="$HOME/.local/bin"

# --- Parse arguments ---
VERSION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            VERSION="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# --- Detect platform ---
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin) PLATFORM="darwin" ;;
    Linux)  PLATFORM="linux" ;;
    *)
        echo "Error: Unsupported OS: $OS" >&2
        echo "Supported platforms: macOS (ARM), Linux (x86_64)" >&2
        exit 1
        ;;
esac

case "$ARCH" in
    arm64|aarch64) ARCH_NAME="arm64" ;;
    x86_64|amd64)  ARCH_NAME="x86_64" ;;
    *)
        echo "Error: Unsupported architecture: $ARCH" >&2
        echo "Supported: arm64, x86_64" >&2
        exit 1
        ;;
esac

echo "Detected platform: ${PLATFORM}-${ARCH_NAME}"

# --- Check for curl ---
if ! command -v curl >/dev/null 2>&1; then
    echo "Error: curl is required but not found." >&2
    echo "Install it via your package manager:" >&2
    echo "  Ubuntu/Debian: sudo apt install curl" >&2
    echo "  RHEL/Fedora:   sudo yum install curl" >&2
    exit 1
fi

# --- Fetch version ---
if [ -z "$VERSION" ]; then
    echo "Fetching latest release..."
    RELEASE_JSON=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" 2>/dev/null) || {
        echo "Error: Failed to fetch latest release from GitHub API." >&2
        echo "You may be rate-limited. Try specifying a version:" >&2
        echo "  curl -fsSL https://raw.githubusercontent.com/${REPO}/master/install.sh | bash -s -- --version 0.4.0" >&2
        exit 1
    }
    VERSION=$(echo "$RELEASE_JSON" | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/')
    if [ -z "$VERSION" ]; then
        echo "Error: Could not parse version from GitHub API response." >&2
        exit 1
    fi
fi

echo "Installing Friedman-cli v${VERSION}..."

# --- Construct download URL ---
ARCHIVE_NAME="friedman-v${VERSION}-${PLATFORM}-${ARCH_NAME}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/v${VERSION}/${ARCHIVE_NAME}"

# --- Ensure Julia 1.12 is available ---
ensure_julia() {
    # Check if juliaup is available
    if command -v juliaup >/dev/null 2>&1; then
        echo "Found juliaup. Ensuring Julia 1.12 is installed..."
        juliaup add 1.12 2>/dev/null || true
        return 0
    fi

    # Check if julia >= 1.12 is on PATH
    if command -v julia >/dev/null 2>&1; then
        JULIA_VER=$(julia --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        JULIA_MAJOR=$(echo "$JULIA_VER" | cut -d. -f1)
        JULIA_MINOR=$(echo "$JULIA_VER" | cut -d. -f2)
        if [ "$JULIA_MAJOR" -ge 1 ] && [ "$JULIA_MINOR" -ge 12 ]; then
            echo "Found Julia $(julia --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
            return 0
        fi
    fi

    # Install juliaup
    echo "Julia 1.12+ not found. Installing juliaup..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes || {
        echo "Error: Failed to install juliaup." >&2
        echo "Install Julia manually: https://julialang.org/downloads/" >&2
        exit 1
    }

    # Source juliaup into current shell
    export PATH="$HOME/.juliaup/bin:$PATH"

    echo "Installing Julia 1.12..."
    juliaup add 1.12
}

ensure_julia

# --- Download archive ---
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${ARCHIVE_NAME}..."
curl -fSL "$DOWNLOAD_URL" -o "$TMPDIR/$ARCHIVE_NAME" || {
    echo "Error: Failed to download ${DOWNLOAD_URL}" >&2
    echo "Check that version v${VERSION} exists at:" >&2
    echo "  https://github.com/${REPO}/releases" >&2
    exit 1
}

# --- Verify checksum ---
CHECKSUM_URL="https://github.com/${REPO}/releases/download/v${VERSION}/checksums.sha256"
echo "Verifying checksum..."
if curl -fsSL "$CHECKSUM_URL" -o "$TMPDIR/checksums.sha256" 2>/dev/null; then
    cd "$TMPDIR"
    # Extract only the line for our archive
    grep "$ARCHIVE_NAME" checksums.sha256 > check.sha256
    if [ -s check.sha256 ]; then
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum -c check.sha256 || {
                echo "Error: Checksum verification failed! The download may be corrupted." >&2
                exit 1
            }
        elif command -v shasum >/dev/null 2>&1; then
            # macOS uses shasum instead of sha256sum
            shasum -a 256 -c check.sha256 || {
                echo "Error: Checksum verification failed! The download may be corrupted." >&2
                exit 1
            }
        else
            echo "Warning: No sha256sum or shasum found, skipping checksum verification."
        fi
    else
        echo "Warning: Archive not found in checksums file, skipping verification."
    fi
    cd - >/dev/null
else
    echo "Warning: Could not download checksums file, skipping verification."
fi

# --- Extract to temp, then safe-replace install dir ---
echo "Installing to ${INSTALL_DIR}..."
mkdir -p "$TMPDIR/extract"
tar -xzf "$TMPDIR/$ARCHIVE_NAME" -C "$TMPDIR/extract"

# The archive contains a top-level friedman/ directory
if [ -d "$TMPDIR/extract/friedman" ]; then
    EXTRACTED="$TMPDIR/extract/friedman"
else
    EXTRACTED="$TMPDIR/extract"
fi

# Safe replacement: only remove old install after new one is fully extracted
rm -rf "$INSTALL_DIR"
mv "$EXTRACTED" "$INSTALL_DIR"

# --- Create PATH shim ---
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/bin/friedman" "$BIN_DIR/friedman"

# --- PATH guidance ---
if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    echo ""
    echo "Add ${BIN_DIR} to your PATH by adding this line to your shell profile:"
    SHELL_NAME="$(basename "$SHELL")"
    case "$SHELL_NAME" in
        zsh)  echo "  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.zshrc" ;;
        bash) echo "  echo 'export PATH=\"${BIN_DIR}:\$PATH\"' >> ~/.bashrc" ;;
        fish) echo "  fish_add_path ${BIN_DIR}" ;;
        *)    echo "  export PATH=\"${BIN_DIR}:\$PATH\"" ;;
    esac
    echo ""
    echo "Then restart your shell or run: export PATH=\"${BIN_DIR}:\$PATH\""
fi

# --- Verify ---
if command -v friedman >/dev/null 2>&1; then
    echo ""
    friedman --version
    echo "Friedman-cli installed successfully!"
else
    echo ""
    echo "Friedman-cli installed to ${INSTALL_DIR}"
    echo "Run: export PATH=\"${BIN_DIR}:\$PATH\" && friedman --version"
fi

echo ""
echo "To uninstall: rm -rf ${INSTALL_DIR} ${BIN_DIR}/friedman"
