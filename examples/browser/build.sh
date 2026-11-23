#!/usr/bin/env bash
# Build candle-core for wasm32 and generate the browser + Node bindings.
#
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.130
#   bash examples/browser/build.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

if ! rustup target list --installed | grep -q '^wasm32-unknown-unknown$'; then
  echo "missing target; run: rustup target add wasm32-unknown-unknown" >&2
  exit 1
fi
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "missing wasm-bindgen; run: cargo install wasm-bindgen-cli --version 0.2.130" >&2
  exit 1
fi

CARGO_TARGET_DIR="$PWD/target" cargo build \
  --release \
  --target wasm32-unknown-unknown \
  --manifest-path wasm/Cargo.toml

WASM="target/wasm32-unknown-unknown/release/candle_wasm.wasm"
wasm-bindgen --target web --out-dir examples/browser/pkg --out-name candle_wasm "$WASM"
wasm-bindgen --target nodejs --out-dir examples/browser/pkg-node --out-name candle_wasm "$WASM"

echo "built examples/browser/pkg and examples/browser/pkg-node"
echo "serve with: python3 -m http.server -d examples/browser 8080"
