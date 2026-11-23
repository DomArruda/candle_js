import candle, { Tensor } from "../lib/index";

// A linear autoencoder on data that lies near a 2-D subspace of R^8.
const DIM = 8;
const LATENT = 2;
const N = 128;
let seed = 41;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const gaussian = () => {
  const u = Math.max(rng(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
};

const basis: number[][] = [];
for (let i = 0; i < LATENT; i++) basis.push(Array.from({ length: DIM }, gaussian));
const xs: number[] = [];
for (let i = 0; i < N; i++) {
  const z = Array.from({ length: LATENT }, gaussian);
  for (let j = 0; j < DIM; j++) {
    let v = 0;
    for (let k = 0; k < LATENT; k++) v += z[k] * basis[k][j];
    xs.push(v + 0.02 * gaussian());
  }
}
const X = candle.tensor(xs, [N, DIM]);

class Autoencoder extends candle.nn.Module {
  enc = this.addModule(new candle.nn.Linear(DIM, LATENT));
  dec = this.addModule(new candle.nn.Linear(LATENT, DIM));

  protected _forward(x: Tensor): Tensor {
    return this.dec.forward(this.enc.forward(x));
  }
}

const model = new Autoencoder();
const history = model.fit(
  { x: X, y: X },
  {
    epochs: 300,
    batchSize: 32,
    optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.05 }),
    metrics: ["mse"],
    seed: 9,
  }
);

const latent = model.enc.forward(X);
console.log("latent shape:", latent.shape);
console.log(
  `autoencoder MSE: ${history.loss[0].toFixed(4)} -> ${history.loss.at(-1)!.toFixed(5)}`
);
