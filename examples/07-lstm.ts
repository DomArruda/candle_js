import candle, { Tensor } from "../lib/index";

// Learn the running sum of a sequence with an LSTM.
const B = 32;
const T = 10;
let seed = 21;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const xs: number[] = [];
const ys: number[] = [];
for (let b = 0; b < B; b++) {
  let acc = 0;
  for (let t = 0; t < T; t++) {
    const v = rng();
    xs.push(v);
    acc += v;
    ys.push(acc);
  }
}
const X = candle.tensor(xs, [B, T, 1]);
const Y = candle.tensor(ys, [B, T, 1]);

class SumNet extends candle.nn.Module {
  lstm = this.addModule(new candle.nn.LSTM(1, 16));
  head = this.addModule(new candle.nn.Linear(16, 1));

  protected _forward(x: Tensor): Tensor {
    return this.head.forward(this.lstm.forward(x));
  }
}

const model = new SumNet();
const history = model.fit(
  { x: X, y: Y },
  {
    epochs: 40,
    batchSize: 16,
    optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.01 }),
    metrics: ["mse"],
    seed: 3,
  }
);

const pred = model.predict(X);
console.log("output shape:", pred.shape);
console.log(
  `running-sum MSE: ${history.loss[0].toFixed(4)} -> ${history.loss.at(-1)!.toFixed(5)}`
);
