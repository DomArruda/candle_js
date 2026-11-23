import candle from "../lib/index";

// Fit y = sin(x) with a small MLP and a manual training loop.
const n = 64;
const xs: number[] = [];
const ys: number[] = [];
for (let i = 0; i < n; i++) {
  const x = -3 + (6 * i) / (n - 1);
  xs.push(x);
  ys.push(Math.sin(x));
}
const X = candle.tensor(xs, [n, 1]);
const Y = candle.tensor(ys, [n, 1]);

const model = new candle.nn.Sequential(
  new candle.nn.Linear(1, 32),
  new candle.nn.Tanh(),
  new candle.nn.Linear(32, 1)
);
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.01 });

let first = 0;
let loss = 0;
for (let step = 0; step < 600; step++) {
  const l = candle.F.mseLoss(model.forward(X), Y);
  loss = l.item();
  if (step === 0) first = loss;
  opt.step(l.backward());
}

console.log(`sin fit MSE: ${first.toFixed(4)} -> ${loss.toFixed(5)}`);
console.log(
  "sample predictions:",
  [-2, 0, 2].map((x) => model.forward(candle.tensor([x], [1, 1])).item().toFixed(3))
);
