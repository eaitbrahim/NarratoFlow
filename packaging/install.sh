#!/bin/sh
# NarratoFlow installer for macOS and Linux: installs every extra, ffmpeg with libass and the
# CAMeL dialect model. Run from the unpacked bundle:  sh install.sh
# Safe to run again: each step is skipped when it is already done.
set -eu

UV_VERSION="0.11.31"
CAMEL_MODEL="dialectid-model26"
HERE=$(cd "$(dirname "$0")" && pwd)

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

set -- "$HERE"/narratoflow-*-py3-none-any.whl
[ "$#" -eq 1 ] && [ -f "$1" ] || fail "expected exactly one narratoflow wheel in $HERE"
WHEEL=$1
CONSTRAINTS="$HERE/constraints.txt"
[ -f "$CONSTRAINTS" ] || fail "constraints.txt is missing from $HERE"

as_root() {
    if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi
}

has_libass() {
    "$1" -hide_banner -filters 2>/dev/null | awk '$2=="ass"{f=1} END{exit !f}'
}

case "$(uname -s)" in
    Darwin)
        [ "$(uname -m)" = "arm64" ] || fail "an Apple Silicon Mac (M1 or later) is required: PyTorch ships no Intel macOS build"
        if ! command -v brew >/dev/null 2>&1; then
            for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
                [ -x "$b" ] && eval "$("$b" shellenv)" && break
            done
        fi
        if ! command -v brew >/dev/null 2>&1; then
            say "Installing Homebrew (it asks for your password and installs the Xcode command line tools)"
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        xcode-select -p >/dev/null 2>&1 || fail "the Xcode command line tools are missing: run 'xcode-select --install', then run this installer again"
        say "Installing ffmpeg with libass (Homebrew ffmpeg-full)"
        brew list ffmpeg-full >/dev/null 2>&1 || brew install ffmpeg-full
        FFMPEG="$(brew --prefix ffmpeg-full)/bin/ffmpeg"
        has_libass "$FFMPEG" || fail "$FFMPEG has no libass (ass filter)"
        line="export NARRATOFLOW_FFMPEG=\"$FFMPEG\""
        for profile in "$HOME/.zprofile" "$HOME/.bash_profile"; do
            if ! grep -qsF "$line" "$profile"; then
                printf '\n# NarratoFlow: ffmpeg with libass\n%s\n' "$line" >>"$profile"
            fi
        done
        export NARRATOFLOW_FFMPEG="$FFMPEG"
        ;;
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            say "Installing ffmpeg, a C++ compiler and curl (apt)"
            as_root apt-get update
            as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                ffmpeg build-essential curl ca-certificates
        elif command -v dnf >/dev/null 2>&1; then
            say "Installing ffmpeg, a C++ compiler and curl (dnf; ffmpeg needs RPM Fusion on Fedora)"
            as_root dnf install -y ffmpeg gcc-c++ make curl
        else
            if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
                fail "install ffmpeg (with libass) and a C++ compiler with your package manager, then run this installer again"
            fi
        fi
        FFMPEG=$(command -v ffmpeg) || fail "ffmpeg is not on PATH"
        has_libass "$FFMPEG" || fail "$FFMPEG has no libass (ass filter); install a build that has it"
        ;;
    *)
        fail "unsupported system $(uname -s); on Windows run install.ps1"
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    say "Installing uv $UV_VERSION"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
fi
export PATH="$HOME/.local/bin:$PATH"

say "Installing NarratoFlow with every extra (several GB on first install)"
uv tool install --force --python 3.12 --constraints "$CONSTRAINTS" "narratoflow[all] @ $WHEEL"
uv tool update-shell >/dev/null 2>&1 || true

ENV_BIN="$(uv tool dir)/narratoflow/bin"
say "Downloading the CAMeL dialect model ($CAMEL_MODEL, 274 MB)"
"$ENV_BIN/camel_data" -i "$CAMEL_MODEL"

say "Checking the installation"
"$ENV_BIN/python" -c "import camel_tools, ctc_forced_aligner, diffusers, edge_tts, gradio, mishkal, piper, torch, transformers"
"$(uv tool dir --bin)/narratoflow" --version

say "Done. Open a new terminal window, then run: narratoflow --help"
