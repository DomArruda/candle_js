import candle from "../lib/index";

// The same two-blob task, but with the high-level API: fit/evaluate/predict.
let seed = 11;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const gaussian = () => {
  const u = Math.max(rng(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
};

const xs: number[] = [];
const labels: number[] = [];
for (const [cx, cy, label] of [
  [2, 2, 0],
  [-2, -2, 1],
] as const) {
  for (let i = 0; i < 80; i++) {
    xs.push(cx + gaussian(), cy + gaussian());
    labels.push(label);
  }
}
const data = {
  x: candle.tensor(xs, [160, 2]),
  y: new Uint32Array(labels),
};
const { train, test } = candle.trainTestSplit(data, { testSize: 0.25, seed: 1 });

const model = new candle.nn.Sequential(
  new candle.nn.Linear(2, 16),
  new candle.nn.GELU(),
  new candle.nn.Linear(16, 2)
);

const history = model.fit(train, {
  epochs: 15,
  batchSize: 16,
  validationSplit: 0.2,
  optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.05 }),
  verbose: true,
  seed: 2,
});

const logs = model.evaluate(test, { metrics: ["accuracy"] });
const preds = candle.predictClasses(model, test.x);
console.log("history length:", history.loss.length);
console.log("test:", logs);
console.log("first predictions:", preds.slice(0, 8));
console.log("parameters:", model.numParameters);
