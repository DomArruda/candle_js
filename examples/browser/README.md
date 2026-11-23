# Browser demo (WASM + Pyodide)

Runs `candle-core` compiled to WebAssembly in the browser. The page calls the
same module from JavaScript and from Python via Pyodide.

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130
bash examples/browser/build.sh
```

Serve the directory (ES modules need HTTP):

```bash
python3 -m http.server -d examples/browser 8080
```

Then open <http://localhost:8080>. The JavaScript panel works offline; the
Pyodide panel loads Pyodide from jsDelivr on first click.

`bun run wasm` builds and smoke-tests the web + Node bindings, including the
Pyodide bridge. `node examples/browser/pyodide-smoke.mjs` checks the
Python-to-WASM path on its own.

## What this shows

- `matmul` and a 300-step XOR training run in WebAssembly on the CPU.
- Pyodide can call the WASM exports directly, with `Float32Array` values
  created via `to_js` and `js.Float32Array.new`.
- Node can test the same module: `node examples/browser/smoke.mjs` after the
  build (uses `pkg-node/`).

## WebGPU

candle does not currently ship a WebGPU backend; `candle-core` supports CPU,
CUDA, Metal and WASM. WebGPU is an open upstream request
([candle#344](https://github.com/huggingface/candle/issues/344)). This demo is
WASM CPU. The page reports whether the browser exposes `navigator.gpu`, but
does not use it.
