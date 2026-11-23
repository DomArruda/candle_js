import candle from "../lib/index";

// Two Gaussian blobs with a small MLP.
let seed = 7;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const gaussian = () => {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

const perClass = 60;
const xs: number[] = [];
const labels: number[] = [];
for (const [cx, cy, label] of [
  [2, 2, 0],
  [-2, -2, 1],
] as const) {
  for (let i = 0; i < perClass; i++) {
    xs.push(cx + gaussian(), cy + gaussian());
    labels.push(label);
  }
}
const X = candle.tensor(xs, [perClass * 2, 2]);
const y = new Uint32Array(labels);

const model = new candle.nn.Sequential(
  new candle.nn.Linear(2, 16),
  new candle.nn.GELU(),
  new candle.nn.Linear(16, 2)
);
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });

let loss = 0;
for (let step = 0; step < 200; step++) {
  const l = candle.F.crossEntropy(model.forward(X), y);
  loss = l.item();
  opt.step(l.backward());
}

const preds = model.forward(X).argmax(1).to("f32").toArray();
const acc = preds.filter((p, i) => p === labels[i]).length / labels.length;
console.log(`loss=${loss.toFixed(4)} accuracy=${acc}`);
