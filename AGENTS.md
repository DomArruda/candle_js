# AGENTS.md

## What this is

`candle_js` exposes [candle](https://github.com/huggingface/candle) to
JavaScript. Three layers:

1. `src/lib.rs` — the Neon native addon. Every export is a thin wrapper over a
   candle operation.
2. `lib/` — the TypeScript library: tensors, autograd, `nn.Module`, optimizers,
   functional ops, `expr`, `VarMap`, and the high-level `fit`/`evaluate` API.
3. `wasm/` — a separate crate that compiles candle-core to
   `wasm32-unknown-unknown` for the browser demo. It is not part of the npm
   package.

## Layout

- `lib/native.ts` — loads `index.node`, then
  `prebuilds/<platform>-<arch>/candle_js.node`.
- `lib/index.ts` — default `candle` export and named exports. Keep the default
  object and the ESM wrapper (`index.mjs`) in sync.
- `lib/highlevel.ts` — `fit`, `evaluate`, `predict`, `predictClasses`,
  `trainTestSplit`, metrics and history.
- `lib/nn.ts` — `Module` plus layers; `Module.fit` delegates to `highlevel.ts`.
- `examples/` — runnable scripts (16). `examples/browser/` and
  `examples/pyodide/` are demos, not part of the npm package.
- `tests/` — Bun tests. `tests/recipes.test.ts` backs the README recipes.
- `scripts/` — prebuild staging, smoke tests, example runner, TF.js benchmark.

## Commands

```bash
bun install
bun run build       # cargo build --release + stage index.node/prebuilds
bun test            # full suite
bun run typecheck   # tsc --noEmit (lib/ only)
bun run examples    # run every examples/*.ts
bun run bench       # candle benchmarks
bun run bench:tfjs  # TensorFlow.js comparison
bun run build:js    # dist/ (ESM wrapper consumes it)
bun run smoke       # Bun + CJS/ESM entry smoke
bun run pyodide     # Pyodide on Node calling the native addon
bun run wasm        # build wasm + smoke JS and Pyodide
```

WASM prerequisites: `rustup target add wasm32-unknown-unknown` and
`cargo install wasm-bindgen-cli --version 0.2.130`.

## Conventions

- API style mirrors PyTorch at the low level and Keras at the high level.
  `model.fit` is additive; do not remove or rename low-level exports.
- Every public symbol gets JSDoc with a runnable example. Keep TypeScript
  types exact: no `any` in `lib/` except the native boundary.
- New training/example behavior needs a test. A README recipe needs a matching
  test in `tests/recipes.test.ts` or an example under `examples/`.
- No comments unless they explain a non-obvious "why".
- Keep the npm `files` whitelist (`dist`, `index.mjs`, `prebuilds`, `LICENSE`,
  `README.md`). Never publish `examples/`, `wasm/`, `target/`, or test files.

## Gotchas

- The native addon is platform-specific. Build and test with Bun on the same
  OS (WSL/Linux `.so` will not load in Windows Bun).
- candle's CPU RNG is not seedable. Tests seed their own data; model init still
  varies. Keep assertions tolerant.
- `AdamW`/`SGD` state is not stored in safetensors; save weights with
  `model.save`, restart the optimizer.
- WebGPU is not a candle backend. The browser demo is WASM/CPU; upstream
  tracking issue: candle#344.
- npm provenance in `release.yml` requires the `repository` URL in
  `package.json` to match the GitHub repo exactly.
