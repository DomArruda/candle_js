import candle from "../lib/index";

// Train XOR briefly, save the weights, reload them, and compare outputs.
const X = candle.tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const y = new Uint32Array([0, 1, 1, 0]);
const path = "/tmp/candle_example_model.safetensors";

const make = () =>
  new candle.nn.Sequential(
    new candle.nn.Linear(2, 8),
    new candle.nn.Tanh(),
    new candle.nn.Linear(8, 2)
  );

const a = make();
a.fit(
  { x: X, y },
  {
    epochs: 150,
    batchSize: 4,
    optimizer: new candle.optim.AdamW(a.parameters(), { lr: 0.05 }),
    seed: 1,
  }
);
a.save(path);

const b = make();
b.load(path);
const before = a.predict(X).toArray();
const after = b.predict(X).toArray();
console.log("saved to:", path);
console.log("same outputs:", after.every((v, i) => v === before[i]));
console.log("predictions:", candle.predictClasses(b, X), "wanted:", Array.from(y));
