# candle_js

![status: WIP](https://img.shields.io/badge/status-WIP-orange)

Neon bindings exposing [candle](https://github.com/huggingface/candle) —
Hugging Face's Rust ML framework — to JavaScript. Tensors, autograd, layers,
and optimizers, callable from Bun or Node.

It comes in two layers: a documented, PyTorch-like TypeScript library in `lib/`,
and the thin native addon (`index.node`) underneath it. Train a small network
in TypeScript, save it as safetensors, load it back. Everything documented
below works and is covered by tests.

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

The package has two layers:

- **`lib/` — the library (recommended).** Import the default export as
  `candle`. Tensors have methods, layers are `nn.Module`s, optimizers take
  `model.parameters()`, and every public symbol carries a JSDoc string plus
  TypeScript types.
- **`index.node` — the native addon.** The thin Neon FFI over candle that the
  library is built on. Use it directly for zero-abstraction access.

### Library

```ts
import candle from "candle_js";

const a = candle.tensor([[1, 2], [3, 4]]);
const b = candle.ones([2, 2]);

a.matmul(b).tolist();        // [[3, 3], [7, 7]]
a.add(10).relu().sum(1);     // tensor([13, 17])
candle.randn([2, 3], { std: 0.02 });
```

Creation helpers: `tensor`, `variable`, `zeros`, `ones`, `full`, `randn`,
`rand`, `arange`, `eye`, `tril`, `triu`, `fromU32`, `cat`, `stack`, plus
`zerosVar` / `onesVar` / `fullVar` / `randnVar`.

#### Autograd

```ts
const w = candle.variable([3], [1]);
w.mul(w).backward().get(w).item(); // 6
```

#### Modules and training

Subclass `candle.nn.Module`, register layers with `addModule`, implement
`_forward`. Parameter names are generated from the module tree, so
`save`/`load` round-trip a whole model.

```ts
import candle, { Tensor } from "candle_js";

class MLP extends candle.nn.Module {
  fc1 = this.addModule(new candle.nn.Linear(2, 8));
  fc2 = this.addModule(new candle.nn.Linear(8, 1));

  _forward(x: Tensor): Tensor {
    return this.fc2.forward(this.fc1.forward(x).tanh());
  }
}

const model = new MLP();
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });

for (let i = 0; i < 400; i++) {
  const loss = candle.nn.functional.mseLoss(model.forward(X), Y);
  opt.step(loss.backward());
}
model.save("model.safetensors");
```

`nn.Sequential` composes without a subclass, and layers take options objects
rather than positional flags:

```ts
const model = new candle.nn.Sequential(
  new candle.nn.Linear(2, 8),
  new candle.nn.Tanh(),
  new candle.nn.Linear(8, 1),
);
new candle.nn.Linear(128, 512, { bias: false });
new candle.nn.LayerNorm(512, { eps: 1e-5 });
```

Available: `Linear`, `Embedding`, `LSTM`, `LayerNorm`, `RMSNorm`, `Dropout`,
`Sequential`, `Module`, and the activation modules `ReLU`, `GELU`, `SiLU`,
`Mish`, `Sigmoid`, `Tanh`, `LeakyReLU`, `ELU`, plus `Lambda` for arbitrary
functions. Optimizers: `candle.optim.SGD`, `candle.optim.AdamW`.

Functional ops and losses live on `candle.nn.functional` (aliased as `candle.F`):
`relu`, `gelu`, `geluErf`, `silu`, `swiGLU`, `mish`, `sigmoid`, `tanh`, `erf`,
`hardSigmoid`, `hardSwish`, `relu2`, `relu6`, `selu`, `elu`, `leakyRelu`,
`softmax`, `logSoftmax`, `where`, `linear`, `embedding`, `mseLoss`,
`crossEntropy`, `nll`, `bceWithLogit`, `huber`.

The full library API is documented with JSDoc in `lib/`; your editor shows it
on hover.

#### Operator-style expressions

JavaScript can't overload `+` or `@` on objects (no `Symbol` hook can return a
tensor), so methods are the primary API. If you want infix syntax, `expr` is an
opt-in tagged-template builder supporting `+`, `-`, `*`, `/`, `@` (matmul) and
`^` / `**` (power):

```ts
import candle, { expr } from "candle_js";

const z = expr`${a} @ ${b} + ${c}`;    // a.matmul(b).add(c)
const y = expr`(${a} + ${b}) @ ${c}`;  // a.add(b).matmul(c)
const p = expr`2 * ${x} ^ 3`;          // x.powf(3).mul(2)
```

#### Examples

Runnable, tested examples live in `tests/`:

| File | Shows |
| --- | --- |
| `examples_mlp.test.ts` | regression (fit `sin`), classification, activation zoo, custom `Module` |
| `examples_lstm.test.ts` | `LSTM` running-sum regression and sequence classification |
| `examples_transformer.test.ts` | stacked bidirectional transformer blocks for classification |
| `examples_autoencoder.test.ts` | linear autoencoder + embedding-bag text classifier |
| `library.test.ts` | tensor/autograd tour and a one-block GPT overfitting a sequence |
| `expr.test.ts` | the `expr` operator DSL |
| `benchmark.test.ts` | matmul GFLOP/s, op latency, model throughput |

### Low-level native API

Prefer the library above. The native exports are still public for direct use:

```ts
const candle = require("./index.node");

const a = candle.tensorFromF32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = candle.tensorFromF32(new Float32Array([5, 6, 7, 8]), [2, 2]);
const c = candle.tensorMatmul(a, b);

const { shape, data } = candle.tensorToF32(c);
console.log(shape, Array.from(new Float32Array(data)));
// [2, 2] [19, 22, 43, 50]
```

#### Autograd

```ts
const x = candle.varFromF32(new Float32Array([3]), [1]);
const y = candle.tensorMul(x, x);          // y = x²
const g = candle.gradOf(candle.backward(y), x);
// dy/dx = 6
```

#### Training

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

#### Transformer

Everything needed for a transformer is exposed as primitives, so the
architecture lives in JS. A one-block, two-head GPT that overfits a short
sequence — causal masking, scaled dot-product attention, LayerNorm, GELU and
cross-entropy:

```ts
const D = 16, H = 2, HD = D / H, T = 6, V = 8, B = 2;

const vm = candle.varmapNew();
const tok = candle.embeddingNew(vm, "tok", V, D);
const pos = candle.embeddingNew(vm, "pos", T, D);
const ln1 = candle.layerNormNew(vm, "ln1", D, 1e-5, true);
const ln2 = candle.layerNormNew(vm, "ln2", D, 1e-5, true);
const q = candle.linearNew(vm, "q", D, D, true);
const k = candle.linearNew(vm, "k", D, D, true);
const v = candle.linearNew(vm, "v", D, D, true);
const proj = candle.linearNew(vm, "proj", D, D, true);
const fc1 = candle.linearNew(vm, "fc1", D, 4 * D, true);
const fc2 = candle.linearNew(vm, "fc2", 4 * D, D, true);
const head = candle.linearNew(vm, "head", D, V, true);

const mask = candle.tensorSub(candle.tensorOnes([T, T]), candle.tensorTril(T));
const posIds = new Uint32Array(Array.from({ length: T }, (_, i) => i));
const toHeads = (t) =>
  candle.tensorPermute(candle.tensorReshape(t, [B, T, H, HD]), [0, 2, 1, 3]);

const forward = (ids) => {
  let h = candle.tensorAdd(
    candle.embeddingForward(tok, ids, [B, T]),
    candle.embeddingForward(pos, posIds, [T])
  );
  const n = candle.layerNormForward(ln1, h);
  const qh = toHeads(candle.linearForward(q, n));
  const kh = toHeads(candle.linearForward(k, n));
  const vh = toHeads(candle.linearForward(v, n));

  const scores = candle.tensorAffine(
    candle.tensorMatmul(qh, candle.tensorTranspose(kh, 2, 3)),
    1 / Math.sqrt(HD), 0
  );
  const attn = candle.tensorSoftmax(candle.tensorMaskedFill(scores, mask, -1e9), 3);
  const ctx = candle.tensorMatmul(attn, vh);
  const merged = candle.tensorReshape(
    candle.tensorPermute(ctx, [0, 2, 1, 3]), [B, T, D]
  );
  h = candle.tensorAdd(h, candle.linearForward(proj, merged));

  h = candle.tensorAdd(h, candle.linearForward(fc2,
    candle.tensorGelu(candle.linearForward(fc1, candle.layerNormForward(ln2, h)))));

  return candle.tensorReshape(candle.linearForward(head, h), [B * T, V]);
};

const opt = candle.adamwNew(vm, 0.01, 0.9, 0.999, 1e-8, 0.0);
for (let i = 0; i < 500; i++) {
  const loss = candle.crossEntropyLoss(forward(X), Y); // Y: Uint32Array of targets
  candle.adamwStep(opt, candle.backward(loss));
}
```

`tests/candle_transformer.test.ts` is the full, runnable version.

## Native API

The `lib/` wrapper is the recommended interface; this section lists the raw
exports it builds on.

**Tensors (creation)** — `tensorFromF32`, `tensorFromU32`, `tensorZeros`,
`tensorOnes`, `tensorFull`, `tensorRandn`, `tensorRand`, `tensorArange`,
`tensorEye`, `tensorTril`, `tensorTriu`

**Vars (trainable)** — `varFromF32`, `varRandn`, `varZeros`, `varOnes`,
`varFull`

**Elementwise** — `tensorAdd`, `tensorSub`, `tensorMul`, `tensorDiv` (all
broadcast and accept a scalar), `tensorMaximum`, `tensorMinimum`, `tensorNeg`,
`tensorAbs`, `tensorSqr`, `tensorSqrt`, `tensorExp`, `tensorLog`,
`tensorRecip`, `tensorSin`, `tensorCos`, `tensorCeil`, `tensorFloor`,
`tensorRound`, `tensorSign`, `tensorAffine`, `tensorPow`, `tensorPowf`,
`tensorClamp`, `tensorWhere`, `tensorMaskedFill`, `tensorEq`, `tensorNe`,
`tensorLt`, `tensorLe`, `tensorGt`, `tensorGe`

**Activations** — `tensorRelu`, `tensorTanh`, `tensorSigmoid`, `tensorGelu`,
`tensorGeluErf`, `tensorSilu`, `tensorSwiGLU`, `tensorMish`, `tensorErf`,
`tensorHardSigmoid`, `tensorHardSwish`, `tensorRelu2`, `tensorRelu6`,
`tensorSelu`, `tensorElu`, `tensorLeakyRelu`, `tensorSoftmax`,
`tensorLogSoftmax`, `tensorSoftmaxLastDim`

**Matmul** — `tensorMatmul`, `tensorBroadcastMatmul`

**Reductions** — `tensorSumAll`, `tensorMeanAll`, `tensorMaxAll`,
`tensorMinAll`, `tensorNorm`, `tensorSum`, `tensorSumKeepdim`, `tensorMean`,
`tensorMeanKeepdim`, `tensorMax`, `tensorMaxKeepdim`, `tensorMin`,
`tensorMinKeepdim`, `tensorArgmax`, `tensorArgmaxKeepdim`, `tensorArgmin`,
`tensorArgminKeepdim`, `tensorCumsum`, `tensorLogSumExp`, `dot`

**Shape** — `tensorReshape`, `tensorFlattenAll`, `tensorFlatten`,
`tensorTranspose`, `tensorT`, `tensorPermute`, `tensorSqueeze`,
`tensorUnsqueeze`, `tensorNarrow`, `tensorBroadcastAs`, `tensorExpand`,
`tensorRepeat`, `tensorContiguous`, `tensorDetach`, `tensorToDtype`,
`tensorGet`, `tensorGetOnDim`, `tensorPadWithZeros`, `tensorFlip`,
`tensorRoll`, `tensorChunk`, `tensorCat`, `tensorStack`

**Indexing** — `tensorIndexSelect`, `tensorGather`, `tensorEmbedding`

**Readback** — `tensorToF32`, `tensorToU32`, `tensorToScalar`, `tensorShape`,
`tensorRank`, `tensorElemCount`, `tensorDtype`

Ops accept either a Tensor or a Var and always return a Tensor.

**Layers** — `varmapNew`, `varmapNumVars`, `varmapGet`, `varmapSave`,
`varmapLoad`, `linearNew`, `linearForward`, `embeddingNew`, `embeddingForward`,
`layerNormNew`, `layerNormForward`, `rmsNormNew`, `rmsNormForward`,
`dropoutNew`, `dropoutForward`, `lstmNew`, `lstmSeq`

**Losses** — `mseLoss`, `crossEntropyLoss`, `nllLoss`, `bceWithLogitLoss`,
`huberLoss`

**Autograd** — `backward`, `gradOf`, `sgdStep`, `varmapSgdStep`

**Optimizers** — `adamwNew`, `adamwNewFromVars`, `adamwStep`, `adamwSetLr`,
`sgdNew`, `sgdNewFromVars`, `sgdOptStep`, `sgdSetLr`

**Files** — `safetensorsInspect`

Weights round-trip as safetensors, so files written here work with anything
else that reads the format.

## Scope

Works today: tensor ops, full autograd, Linear, Embedding, LSTM, LayerNorm,
RMSNorm and Dropout layers, a full activation zoo, MSE, cross-entropy, NLL, BCE
and Huber losses, SGD and AdamW, safetensors save/load and inspection, and
enough primitives to build transformers in JS. CPU only.

Not here yet: a bundled transformer architecture, tokenizers, GPU, and a
seedable CPU RNG (candle only supports `set_seed` on CUDA/Metal). See
[Roadmap](#roadmap).


## Performance

Every call crosses the FFI boundary, so op granularity matters. The XOR loop
above is boundary-dominated rather than compute-dominated — batching helps
(`varmapSgdStep` does one crossing for all params instead of one each), and
fusing hot paths into single exports would help more. Plenty of headroom here.

`tests/benchmark.test.ts` measures the hot paths on every `bun test` run. On a
modern x86 CPU, representative numbers:

```
matmul 256x256x256      1.5 ms   (~23 GFLOP/s)
matmul 512x512x512      1.9 ms   (~143 GFLOP/s, multi-threaded gemm)
add 65k                 0.18 ms
softmax [256,256]       0.36 ms
mlp forward [64,64]     1.9 ms
mlp fwd+bwd+adamw       6.6 ms
transformer block [8,32,64]  11.4 ms
lstm seq [16,32,32]->64      15.7 ms
end-to-end 16->32->1, batch 128: ~1.1 ms/step (~900 steps/s)
```

Small sizes are overhead-dominated; large matmuls reach candle's threaded
`gemm`. To see them, run `bun run bench` (or `bun test tests/benchmark.test.ts`).

## Roadmap

- [x] TypeScript declarations / JSDoc (via the `lib/` wrapper)
- [x] Example suite: MLP (regression/classification), LSTM, transformer,
      autoencoder, text classifier, benchmarks
- [ ] Prebuilt binaries / npm package (currently source-only)
- [ ] Browser / WASM build (and a CDN-friendly bundle)
- [ ] Transformer loading via `candle-transformers` — needs a tokenizer and
      some config handling to be worth claiming
- [ ] Tokenizers (`tokenizers` crate)
- [ ] More layers: Conv, BatchNorm, GRU
- [ ] Async ops via `neon::task`
- [ ] macOS build

## Tests

```bash
bun test
```

93 tests across 13 files. The `candle_*` files exercise the native exports;
`library.test.ts` and `examples_*` exercise the `lib/` wrapper end to end —
regression, classification, LSTM, transformer and autoencoder models, all
trained to convergence. `benchmark.test.ts` reports timings. Gradients are
checked numerically (d/dx x² at x=3 is 6; ∂/∂w Σ(w·x) is x), not just for
shape. Save/load asserts identical model output after a round-trip.

