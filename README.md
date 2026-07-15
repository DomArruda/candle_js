# candle_js

![status: WIP](https://img.shields.io/badge/status-WIP-orange)

Neon bindings exposing [candle](https://github.com/huggingface/candle) —
Hugging Face's Rust ML framework — to JavaScript. Tensors, autograd, layers,
and optimizers, callable from Bun or Node.

Train a small network in TypeScript, save it as safetensors, load it back.
Everything documented below works and is covered by tests.

**Status: WIP.** The API is still moving and may break at any time, so pin a commit if you build on it.

## Install

No prebuilt binaries yet — build from source (two commands, see
[Build](#build)). Requires a Rust toolchain.

## Why

To learn the Neon FFI boundary, and because candle's autograd is worth having
in JS.

The existing options solve different problems:
[Transformers.js](https://github.com/huggingface/transformers.js) does
transformer inference via ONNX Runtime, and
[TensorFlow.js](https://github.com/tensorflow/tfjs) is a full framework
(last npm release 4.22.0, ~2 years ago). This one sits lower: raw tensors,
autograd, and training, with candle doing the math in Rust.

## Build


Currently only build from source is supported (for now)

```bash
cargo build --release
cp target/release/libcandle_js.so index.node   # .dylib on macOS
```

The `.node` file is gitignored — build it locally.

## Usage

```ts
const candle = require("./index.node");

const a = candle.tensorFromF32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = candle.tensorFromF32(new Float32Array([5, 6, 7, 8]), [2, 2]);
const c = candle.tensorMatmul(a, b);

const { shape, data } = candle.tensorToF32(c);
console.log(shape, Array.from(new Float32Array(data)));
// [2, 2] [19, 22, 43, 50]
```

### Autograd

```ts
const x = candle.varFromF32(new Float32Array([3]), [1]);
const y = candle.tensorMul(x, x);          // y = x²
const g = candle.gradOf(candle.backward(y), x);
// dy/dx = 6
```

### Training

```ts
const vm = candle.varmapNew();
const fc1 = candle.linearNew(vm, "fc1", 2, 8, true);
const fc2 = candle.linearNew(vm, "fc2", 8, 1, true);

// build the optimizer after the layers — see gotchas
const opt = candle.adamwNew(vm, 0.05, 0.9, 0.999, 1e-8, 0.0);

const forward = (x) =>
  candle.linearForward(fc2, candle.tensorTanh(candle.linearForward(fc1, x)));

for (let i = 0; i < 400; i++) {
  const loss = candle.mseLoss(forward(X), Y);
  candle.adamwStep(opt, candle.backward(loss));
}

candle.varmapSave(vm, "model.safetensors");
```

XOR converges in ~400 AdamW steps, ~3000 with plain SGD.

## API

**Tensors** — `tensorFromF32`, `tensorToF32`, `tensorToScalar`, `tensorShape`

**Vars (trainable)** — `varFromF32`, `varRandn`, `varZeros`

**Ops** — `tensorMatmul`, `tensorAdd`, `tensorBroadcastAdd`, `tensorMul`,
`tensorSub`, `tensorAffine`, `tensorRelu`, `tensorTanh`, `tensorSigmoid`,
`tensorSoftmax`, `tensorTranspose`, `tensorReshape`, `tensorSumAll`,
`tensorMeanAll`, `dot`

Ops accept either a Tensor or a Var and always return a Tensor.

**Layers** — `varmapNew`, `linearNew`, `linearForward`, `embeddingNew`,
`embeddingForward`, `varmapSave`, `varmapLoad`

**Losses** — `mseLoss`, `crossEntropyLoss`

**Autograd** — `backward`, `gradOf`, `sgdStep`, `varmapSgdStep`

**Optimizers** — `adamwNew`, `adamwStep`, `adamwSetLr`, `sgdNew`,
`sgdOptStep`, `sgdSetLr`

**Files** — `safetensorsInspect`

Weights round-trip as safetensors, so files written here work with anything
else that reads the format.

## Scope

Works today: tensor ops, full autograd, Linear and Embedding layers, MSE and
cross-entropy, SGD and AdamW, safetensors save/load and inspection. CPU only.

Not here yet: transformer architectures, tokenizers, GPU. See
[Roadmap](#roadmap).


## Performance

Every call crosses the FFI boundary, so op granularity matters. The XOR loop
above is boundary-dominated rather than compute-dominated — batching helps
(`varmapSgdStep` does one crossing for all params instead of one each), and
fusing hot paths into single exports would help more. Plenty of headroom here.

## Roadmap

- [ ] TypeScript declarations
- [ ] Prebuilt binaries / npm package (currently source-only)
- [ ] Transformer loading via `candle-transformers` — needs a tokenizer and
      some config handling to be worth claiming
- [ ] Tokenizers (`tokenizers` crate)
- [ ] More layers: LayerNorm, Conv
- [ ] `narrow` / `cat` for batching
- [ ] Async ops via `neon::task`
- [ ] macOS build

## Tests

```bash
bun test
```

23 tests across 4 files. Gradients are checked numerically (d/dx x² at x=3 is
6; ∂/∂w Σ(w·x) is x), not just for shape. Save/load asserts identical model
output after a round-trip.

