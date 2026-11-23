import candle, { Tensor } from "../lib/index";

// A one-block, two-head causal transformer overfitting a short sequence.
const V = 8;
const T = 8;
const D = 16;
const H = 2;
const HD = D / H;
const FF = 4 * D;
const B = 2;

let seed = 123;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const base = Array.from({ length: T + 1 }, () => Math.floor(rng() * V));
const X = new Uint32Array([...base.slice(0, T), ...base.slice(0, T)]);
const Y = new Uint32Array([...base.slice(1), ...base.slice(1)]);
const POS = new Uint32Array(Array.from({ length: T }, (_, i) => i));
const MASK = candle.sub(candle.ones([T, T]), candle.tril(T));

class MiniGPT extends candle.nn.Module {
  tok = this.addModule(new candle.nn.Embedding(V, D));
  pos = this.addModule(new candle.nn.Embedding(T, D));
  ln1 = this.addModule(new candle.nn.LayerNorm(D));
  ln2 = this.addModule(new candle.nn.LayerNorm(D));
  q = this.addModule(new candle.nn.Linear(D, D));
  k = this.addModule(new candle.nn.Linear(D, D));
  v = this.addModule(new candle.nn.Linear(D, D));
  proj = this.addModule(new candle.nn.Linear(D, D));
  fc1 = this.addModule(new candle.nn.Linear(D, FF));
  fc2 = this.addModule(new candle.nn.Linear(FF, D));
  head = this.addModule(new candle.nn.Linear(D, V));

  protected _forward(ids: Uint32Array): Tensor {
    let h = this.tok.forward(ids, [B, T]).add(this.pos.forward(POS, [T]));
    const toHeads = (t: Tensor) => t.reshape([B, T, H, HD]).permute([0, 2, 1, 3]);

    const n = this.ln1.forward(h);
    const q = toHeads(this.q.forward(n));
    const k = toHeads(this.k.forward(n));
    const v = toHeads(this.v.forward(n));
    const scores = q.matmul(k.transpose(2, 3)).affine(1 / Math.sqrt(HD), 0);
    const attn = scores.maskedFill(MASK, -1e9).softmax(3);
    const ctx = attn.matmul(v).permute([0, 2, 1, 3]).reshape([B, T, D]);
    h = h.add(this.proj.forward(ctx));

    const f = this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).gelu());
    h = h.add(f);
    return this.head.forward(h).reshape([B * T, V]);
  }
}

const model = new MiniGPT();
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });
let first = 0;
let loss = 0;
for (let step = 0; step < 300; step++) {
  const l = candle.F.crossEntropy(model.forward(X), Y);
  loss = l.item();
  if (step === 0) first = loss;
  opt.step(l.backward());
}

const preds = model.forward(X).argmax(1).to("f32").toArray();
const matches = preds.filter((p, i) => p === Y[i]).length / Y.length;
console.log(`transformer loss: ${first.toFixed(3)} -> ${loss.toFixed(4)}`);
console.log(`next-token accuracy: ${matches} (${model.numParameters} parameters)`);
