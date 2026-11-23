import candle from "../lib/index";

// XOR with the high-level training loop.
const X = candle.tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const y = new Uint32Array([0, 1, 1, 0]);

const model = new candle.nn.Sequential(
  new candle.nn.Linear(2, 8),
  new candle.nn.Tanh(),
  new candle.nn.Linear(8, 2)
);

model.fit(
  { x: X, y },
  {
    epochs: 300,
    batchSize: 4,
    optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.05 }),
    seed: 7,
  }
);

const preds = candle.predictClasses(model, X);
console.log("predictions:", preds, "wanted:", Array.from(y));
console.log("accuracy:", preds.filter((p, i) => p === y[i]).length / preds.length);
console.log("parameters:", model.numParameters);
