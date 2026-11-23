# candle_js

[![CI](https://github.com/DomArruda/candle_js/actions/workflows/ci.yml/badge.svg)](https://github.com/DomArruda/candle_js/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![status: WIP](https://img.shields.io/badge/status-WIP-orange)

Neon bindings exposing [candle](https://github.com/huggingface/candle) —
Hugging Face's Rust ML framework — to JavaScript. Tensors, autograd, layers,
and optimizers, callable from Bun or Node.

It comes in three layers: the thin native addon (`index.node`), a documented,
PyTorch-like TypeScript library in `lib/`, and a high-level `model.fit` API on
top. Train a small network in TypeScript, save it as safetensors, load it back.
Everything documented below works and is covered by tests.

**Status: WIP.** The API is still moving and may break at any time, so pin a commit if you build on it.

Source, issues and releases: [github.com/DomArruda/candle_js](https://github.com/DomArruda/candle_js).

## Install

```bash
npm install candle_js     # or: bun add candle_js
```

```ts
import candle from "candle_js";

const a = candle.tensor([[1, 2], [3, 4]]);
a.matmul(candle.ones([2, 2])).tolist(); // [[3, 3], [7, 7]]
```

Both module systems work: `import` resolves to the ESM entry (`index.mjs`) and
`require` to the CommonJS build (`dist/index.js`). Both expose the same API,
including `candle.nn.functional`.

The published tarball ships prebuilt native addons for Linux, macOS and
Windows, so no Rust toolchain is required. It is not on npm yet — see
[Publishing](#publishing).

## Why

To learn the Neon FFI boundary, and because candle's autograd is worth having
in JS.

The existing options solve different problems:
[Transformers.js](https://github.com/huggingface/transformers.js) does
transformer inference via ONNX Runtime, and
[TensorFlow.js](https://github.com/tensorflow/tfjs) is a full framework
(last npm release 4.22.0, ~2 years ago). This one sits lower: raw tensors,
autograd, and training, with candle doing the math in Rust. [Performance](#performance)
includes a same-runtime comparison against TensorFlow.js.

## Build from source

Requires a Rust toolchain and [Bun](https://bun.sh) (for the test suite):

```bash
bun install
bun run build     # cargo build --release + stage prebuilds/ and index.node
bun test
bun run examples    # run the 16 example scripts
bun run bench:tfjs  # optional: TensorFlow.js comparison
bun run pyodide     # optional: Python calling candle_js
```

`bun run build` runs `scripts/copy-native.mjs`, which copies the cdylib to
`prebuilds/<platform>-<arch>/candle_js.node` and to `index.node` for local dev.
Both are gitignored.

## Usage

The package has three layers:

- **High-level API.** `model.fit`, `model.evaluate`, `model.predict` and
  `candle.trainTestSplit` cover the common supervised workflow.
- **`lib/` — the tensor library.** Import the default export as `candle`.
  Tensors have methods, layers are `nn.Module`s, optimizers take
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

#### High-level training

When you don't need a custom loop, `model.fit` handles batching, shuffling,
validation, metrics, callbacks and history. Loss and metrics are inferred from
the target: class ids (`Uint32Array` / `number[]`) mean cross-entropy and
accuracy; tensor targets mean MSE.

```ts
import candle from "candle_js";

const model = new candle.nn.Sequential(
  new candle.nn.Linear(2, 16),
  new candle.nn.GELU(),
  new candle.nn.Linear(16, 2),
);
const { train, test } = candle.trainTestSplit(
  { x: X, y: labels },
  { testSize: 0.2, seed: 1 },
);

const history = model.fit(train, {
  epochs: 20,
  batchSize: 32,
  validationSplit: 0.2,
  optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.05 }),
  onEpochEnd: (epoch, logs) =>
    console.log(epoch, logs.loss, logs.valAccuracy),
});

model.evaluate(test, { metrics: ["accuracy"] }); // { loss, accuracy }
model.predict(X);                                // detached Tensor
candle.predictClasses(model, X);                 // number[]
model.numParameters;
```

Also available as `candle.fit(model, data, config)` and `candle.hl.*`.
Custom losses use `loss: (pred, target) => ...`; metrics can be `"accuracy"`,
`"mse"`, `"mae"` or a mix. See `tests/highlevel.test.ts` and
`examples/05-highlevel.ts`.

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

Runnable scripts live in [`examples/`](examples/) — `bun run examples` runs all
16 and reports failures:

| File | Shows |
| --- | --- |
| `examples/01-tensors.ts` | creation, shape, matmul, reductions, readback |
| `examples/02-autograd.ts` | gradients with `backward()` and `GradStore` |
| `examples/03-regression.ts` | manual loop fitting `sin` |
| `examples/04-classification.ts` | two-blob classification |
| `examples/05-highlevel.ts` | `fit`, validation, `evaluate`, `predict` |
| `examples/06-xor.ts` | XOR with the high-level API |
| `examples/07-lstm.ts` | LSTM running-sum regression |
| `examples/08-transformer.ts` | causal self-attention overfitting a sequence |
| `examples/09-autoencoder.ts` | linear autoencoder on low-rank data |
| `examples/10-text.ts` | embedding-bag text classification |
| `examples/11-rl-bandit.ts` | REINFORCE contextual bandit |
| `examples/12-rl-gridworld.ts` | REINFORCE tabular gridworld |
| `examples/13-expr.ts` | the `expr` operator DSL |
| `examples/14-save-load.ts` | safetensors round-trip |
| `examples/15-benchmark.ts` | matmul and elementwise timings |
| `examples/16-raw-native.ts` | the low-level native addon |
| `examples/browser/` | WASM build + JS/Pyodide browser demo |
| `examples/pyodide/` | Python calling candle_js through Pyodide |

The test suite doubles as example code:

| File | Shows |
| --- | --- |
| `examples_mlp.test.ts` | regression (fit `sin`), classification, activation zoo, custom `Module` |
| `examples_lstm.test.ts` | `LSTM` running-sum regression and sequence classification |
| `examples_transformer.test.ts` | stacked bidirectional transformer blocks for classification |
| `examples_autoencoder.test.ts` | linear autoencoder + embedding-bag text classifier |
| `examples_rl.test.ts` | REINFORCE policy gradient on a contextual bandit |
| `library.test.ts` | tensor/autograd tour and a one-block GPT overfitting a sequence |
| `expr.test.ts` | the `expr` operator DSL |
| `benchmark.test.ts` | matmul GFLOP/s, op latency, model throughput |

### Recipes

Short, copy-paste patterns. Every recipe below is exercised by a test under
`tests/`; run `bun test` after adapting them.

#### Train with `model.fit`

```ts
const { train, test } = candle.trainTestSplit(
  { x: X, y: labels },
  { testSize: 0.2, seed: 1 },
);
const history = model.fit(train, {
  epochs: 20,
  batchSize: 32,
  validationSplit: 0.2,
});
const logs = model.evaluate(test, { metrics: ["accuracy"] });
console.log(history.valLoss.at(-1), logs.accuracy);
```

Tested in `tests/highlevel.test.ts`.

#### Fit a regression model

```ts
const X = candle.tensor(xs, [n, 1]);
const Y = candle.tensor(ys, [n, 1]);

const model = new candle.nn.Sequential(
  new candle.nn.Linear(1, 32),
  new candle.nn.Tanh(),
  new candle.nn.Linear(32, 1),
);
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });

for (let step = 0; step < 600; step++) {
  opt.step(candle.F.mseLoss(model.forward(X), Y).backward());
}
```

Tested in `tests/examples_mlp.test.ts` (fits `sin` to MSE < 0.01).

#### Work with raw parameters

`Module` is optional. Create `Var`s, build the loss with functional ops, inspect
gradients if you want, then hand them to an optimizer:

```ts
const w = candle.variable([0], [1, 1]);
const b = candle.variable([0], [1]);
const opt = new candle.optim.AdamW([w, b], { lr: 0.1 });

const grads = candle.F.mseLoss(candle.F.linear(X, w, b), Y).backward();
grads.get(w); // dL/dw, shape [1, 1]
opt.step(grads);

// ... repeat to convergence: y = 2x + 1 recovers w ~ 2, b ~ 1.
```

Tested in `tests/recipes.test.ts`.

#### Share a trunk between two heads

One encoder, two task heads, a single backward pass:

```ts
class MultiTask extends candle.nn.Module {
  trunk = this.addModule(
    new candle.nn.Sequential(
      new candle.nn.Linear(2, 16),
      new candle.nn.Tanh(),
    ),
  );
  sum = this.addModule(new candle.nn.Linear(16, 1));
  diff = this.addModule(new candle.nn.Linear(16, 1));

  protected _forward(x: Tensor): Tensor {
    const h = this.trunk.forward(x);
    return candle.cat([this.sum.forward(h), this.diff.forward(h)], 1);
  }
}

const model = new MultiTask();
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.02 });
opt.step(candle.F.mseLoss(model.forward(X), Y).backward()); // Y: [n, 2]
```

Tested in `tests/recipes.test.ts`.

#### Save and resume training

```ts
model.save("model.safetensors");

const restored = new MLP();
restored.load("model.safetensors");
restored.eval();
restored.forward(X); // same outputs as model

// Optimizer state is not stored in safetensors; start a fresh optimizer.
const opt = new candle.optim.AdamW(restored.parameters(), { lr: 0.02 });
```

Files are plain safetensors, so other runtimes can read the weights too.
Tested in `tests/recipes.test.ts`.

#### Policy gradient (RL)

There is no separate RL API: REINFORCE is ordinary autograd over
`-log π(a|x) * reward`. A policy is a normal `Module`; `logSoftmax` gives the
action log-probabilities; `backward` does the rest.

```ts
import candle, { Tensor } from "candle_js";

// A contextual bandit: four one-hot contexts, three arms, a hidden best arm.
const BEST = [2, 0, 1, 2];
const contexts = Array.from({ length: 64 }, (_, i) => Math.floor(i / 16));
const best = contexts.map((c) => BEST[c]);
const X = candle.tensor(
  contexts.flatMap((c) => [0, 1, 2, 3].map((k) => (k === c ? 1 : 0))),
  [64, 4],
);

class Policy extends candle.nn.Module {
  l1 = this.addModule(new candle.nn.Linear(4, 16));
  l2 = this.addModule(new candle.nn.Linear(16, 3));

  protected _forward(x: Tensor): Tensor {
    return this.l2.forward(this.l1.forward(x).tanh());
  }
}

const model = new Policy();
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
const sample = (p: number[]): number => {
  let r = Math.random();
  for (let a = 0; a < p.length; a++) if ((r -= p[a]) <= 0) return a;
  return p.length - 1;
};

for (let step = 0; step < 300; step++) {
  const logp = model.forward(X).logSoftmax(1); // [batch, arms]
  const probs = logp.toArray().map(Math.exp);
  const actions = Array.from({ length: 64 }, (_, i) =>
    sample(probs.slice(i * 3, i * 3 + 3)),
  );
  const rewards = actions.map((a, i) => (a === best[i] ? 1 : -1));

  const mask = new Float32Array(64 * 3);
  actions.forEach((a, i) => (mask[i * 3 + a] = 1));

  const loss = logp
    .mul(candle.tensor(mask, [64, 3]))
    .sum(1)
    .mul(candle.tensor(rewards))
    .meanAll()
    .neg();
  opt.step(loss.backward());
}

model.forward(X).argmax(1).to("f32").toArray(); // [2, 0, 1, 2]
```

Tested in `tests/examples_rl.test.ts`: 300 steps (~0.3 s) reach 100% greedy
accuracy on all four contexts. Run it with
`bun test tests/examples_rl.test.ts`.

#### Infix math with `expr`

```ts
import { expr } from "candle_js";

const z = expr`${a} @ ${b} + ${c}`;       // a.matmul(b).add(c)
const y = expr`2 * ${x} ^ 3`;             // x.powf(3).mul(2)
```

Tested in `tests/expr.test.ts`.

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

## Browser, WASM and Pyodide

`candle-core` compiles to `wasm32-unknown-unknown`, so the same math can run in a
browser. [`examples/browser/`](examples/browser/) builds a small module exposing
`matmul` and a 300-step XOR training run, then calls it from JavaScript and from
Python in Pyodide:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.130
bash examples/browser/build.sh
python3 -m http.server -d examples/browser 8080
```

`bun run wasm` builds the web + Node bindings and smoke-tests both, including
the Pyodide bridge. The generated `examples/browser/pkg*` directories are
gitignored. `.github/workflows/pages.yml` builds and deploys `examples/browser`
to GitHub Pages on `main`.

Pyodide also runs on Node against the real native addon:

```bash
bun run build:js
bun run pyodide
```

See [`examples/pyodide/`](examples/pyodide/) for the Python side.

**WebGPU:** candle does not currently ship a WebGPU backend. Its devices are
CPU, CUDA, Metal and WASM/CPU. WebGPU is an open upstream request
([candle#344](https://github.com/huggingface/candle/issues/344)); the browser
demo detects `navigator.gpu` but runs on WASM CPU. A real WebGPU backend would
be a candle upstream project, not a binding change here.

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
and Huber losses, SGD and AdamW, safetensors save/load and inspection, a
high-level `fit` / `evaluate` / `predict` API, enough primitives to build
transformers and policy-gradient agents in JS, and a WASM/CPU browser build.
CPU only (WASM included).

Not here yet: a WebGPU backend (blocked on candle upstream), a bundled
transformer architecture, tokenizers, CUDA/Metal access from JS, and a
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

### candle_js vs TensorFlow.js

`bun run bench:tfjs` runs the same operations through candle_js and
`@tensorflow/tfjs` 4.22.0 on the same runtime and CPU. TF.js here is the
pure-JS CPU backend, not the native `@tensorflow/tfjs-node` binding. On this
machine (Linux x64, Bun 1.3.14):

| Case | candle_js | TF.js | Speedup |
| --- | ---: | ---: | ---: |
| matmul 128³ | 1.28 ms | 1.90 ms | 1.5x |
| matmul 256³ | 1.54 ms | 14.17 ms | 9.2x |
| matmul 512³ | 1.90 ms | 124.52 ms | 65.6x |
| add 65k | 0.19 ms | 0.33 ms | 1.7x |
| softmax [256, 256] | 0.34 ms | 8.03 ms | 23.6x |
| MLP forward [64, 64] | 1.47 ms | 1.93 ms | 1.3x |
| train step [128, 16] → 1 | 0.91 ms | 3.50 ms | 3.9x |

The gap is largest where candle calls Rust kernels in a loop (matmul, softmax)
and smallest for elementwise ops and the small MLP, where both sides are
dominated by per-op JS overhead. The native `@tensorflow/tfjs-node` backend
would be a closer fight for the large kernels; it is not included here because
it needs platform-specific binaries. The script takes the best of three timed
rounds per case. Re-run on your machine with `bun run bench:tfjs`; install dev
dependencies first (`bun install`). Numbers move with CPU and thread count.

## Roadmap

- [x] TypeScript declarations / JSDoc (via the `lib/` wrapper)
- [x] High-level API: `model.fit`, `evaluate`, `predict`, `trainTestSplit`
- [x] Example suite: 16 runnable scripts in `examples/` plus the test suite
- [x] MLP, LSTM, transformer, autoencoder, text classifier, REINFORCE RL
- [x] Benchmark against TensorFlow.js (`bun run bench:tfjs`)
- [x] npm packaging: `dist/` build, prebuilt-addon loader, CI/release workflows
- [x] README recipes, tested end to end
- [x] Browser / WASM build and demo (`bun run wasm`)
- [x] Pyodide bridge: Node demo + browser demo calling WASM
- [ ] Publish to npm (needs an account + `NPM_TOKEN`)
- [ ] WebGPU backend — blocked on candle upstream
      ([candle#344](https://github.com/huggingface/candle/issues/344)); candle
      currently exposes CPU, CUDA, Metal and WASM/CPU
- [ ] Transformer loading via `candle-transformers` — needs a tokenizer and
      some config handling to be worth claiming
- [ ] Tokenizers (`tokenizers` crate)
- [ ] More layers: Conv, BatchNorm, GRU
- [ ] Async ops via `neon::task`
- [x] macOS build (x64 + arm64 prebuilds)

## Tests

```bash
bun test
```

105 tests across 17 files. The `candle_*` files exercise the native exports;
`library.test.ts` and `examples_*` exercise the `lib/` wrapper end to end —
regression, classification, LSTM, transformer, autoencoder and REINFORCE
models, all trained to convergence. `highlevel.test.ts` covers `fit`,
validation, metrics and `trainTestSplit`; `recipes.test.ts` runs the README
recipes (raw variables, shared trunk, resume), and `package.test.ts` guards the
npm entry points. `benchmark.test.ts` reports timings. Gradients are checked
numerically (d/dx x² at x=3 is 6; ∂/∂w Σ(w·x) is x), not just for shape.
Save/load asserts identical model output after a round-trip. The WASM build is
smoke-tested in CI on its own (`bun run wasm`).

## Publishing

The npm tarball contains the compiled library (`dist/`), the ESM wrapper
(`index.mjs`), prebuilt addons
(`prebuilds/<platform>-<arch>/candle_js.node`), `README.md` and `LICENSE`.
`.github/workflows/ci.yml` tests on Linux/macOS/Windows and smoke-tests the
compiled package on both Bun and Node; `release.yml` builds one prebuild per
platform and publishes on a `v*` tag with npm provenance.

Local dry run:

```bash
bun install
bun run build        # native -> prebuilds/ + index.node
bun run build:js     # TypeScript -> dist/ (JS + .d.ts)
bun run smoke        # load the ESM and CJS entries
bun pm pack --ignore-scripts  # inspect the tarball
```

You need an npm account; CI needs an `NPM_TOKEN` repo secret. Scoped names
(`@you/candle_js`) also need `--access public`. At runtime `lib/native.ts`
tries `index.node` first (local dev), then
`prebuilds/<platform>-<arch>/candle_js.node` (published package).

First publish:

1. Push the repo to GitHub and enable Actions.
2. Add an automation token from npm as the `NPM_TOKEN` repository secret.
3. `git tag v0.1.0 && git push origin v0.1.0` — `release.yml` builds every
   prebuild, runs the tests against them, and publishes with npm provenance.

Repository, homepage and issue URLs are already set in `package.json`, and the
tarball ships `LICENSE` and the ESM entry (`index.mjs`).

When working inside WSL, use Bun's `bun pm pack` or Linux npm. If `npm` resolves
to `/mnt/c/Program Files/nodejs/npm`, that's Windows npm being invoked from WSL;
its UNC path handling can produce an invalid doubled `Ubuntu` path. Install a
Linux Node/npm toolchain in WSL if you specifically need to run `npm pack` there.

