#!/bin/sh
# Build the three install bundles into OUTDIR (default dist/bundles):
#   narratoflow-X.Y.Z-macos.zip, narratoflow-X.Y.Z-linux.tar.gz, narratoflow-X.Y.Z-windows.zip
# Each holds the wheel, constraints.txt (the uv.lock pins for every extra), the OS installer
# and INSTALL.md. Run from the repository root.
set -eu

OUT=${1:-dist/bundles}
VERSION=$(uv version --short)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

uv build --wheel --out-dir "$WORK/wheel"
uv export --frozen --all-extras --no-dev --no-emit-project --no-hashes --no-header \
    --output-file "$WORK/constraints.txt" >/dev/null

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
for os in macos linux windows; do
    name="narratoflow-$VERSION-$os"
    dir="$WORK/$os/$name"
    mkdir -p "$dir"
    cp "$WORK/wheel/narratoflow-$VERSION-py3-none-any.whl" "$WORK/constraints.txt" \
        packaging/INSTALL.md "$dir/"
    if [ "$os" = windows ]; then
        cp packaging/install.ps1 "$dir/"
        (cd "$WORK/$os" && zip -qr "$OUT/$name.zip" "$name")
    else
        cp packaging/install.sh "$dir/"
        chmod 755 "$dir/install.sh"
        if [ "$os" = macos ]; then
            (cd "$WORK/$os" && zip -qr "$OUT/$name.zip" "$name")
        else
            tar -C "$WORK/$os" -czf "$OUT/$name.tar.gz" "$name"
        fi
    fi
done
ls -1 "$OUT"
