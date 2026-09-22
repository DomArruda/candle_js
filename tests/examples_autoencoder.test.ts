import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { gaussian, makeRng } from "./_util";

const DIM = 8;
const LATENT = 2;
const N = 128;

/** Data lying near a 2-D linear subspace of R^8, so a 2-D bottleneck can fit it. */
function makeData(): Tensor {
  const rng = makeRng(41);
  const w: number[][] = [];
  for (let i = 0; i < LATENT; i++) {
    w.push(Array.from({ length: DIM }, () => gaussian(rng)));
  }
  const xs: number[] = [];
  for (let i = 0; i < N; i++) {
    const z = Array.from({ length: LATENT }, () => gaussian(rng));
    for (let j = 0; j < DIM; j++) {
      let v = 0;
      for (let k = 0; k < LATENT; k++) v += z[k] * w[k][j];
      xs.push(v + 0.02 * gaussian(rng));
    }
  }
  return candle.tensor(xs, [N, DIM]);
}

class Autoencoder extends candle.nn.Module {
  enc = this.addModule(new candle.nn.Linear(DIM, LATENT));
  dec = this.addModule(new candle.nn.Linear(LATENT, DIM));

  protected _forward(x: Tensor): Tensor {
    // Linear bottleneck: this recovers the 2-D subspace the data lives in.
    return this.dec.forward(this.enc.forward(x));
  }
}

describe("example: autoencoder", () => {
  test("compresses and reconstructs a low-rank dataset", () => {
    const X = makeData();
    const model = new Autoencoder();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });

    let first = 0;
    let loss = 0;
    for (let step = 0; step < 1200; step++) {
      const l = candle.F.mseLoss(model.forward(X), X);
      loss = l.item();
      if (step === 0) first = loss;
      opt.step(l.backward());
    }

    console.log(`  autoencoder MSE: ${first.toFixed(4)} -> ${loss.toFixed(5)}`);
    expect(loss).toBeLessThan(0.02);
    expect(loss).toBeLessThan(first);
  }, 30000);

  test("round-trips through safetensors", () => {
    const X = makeData();
    const path = "/tmp/candle_autoencoder.safetensors";

    const a = new Autoencoder();
    a.save(path);
    const b = new Autoencoder();
    b.load(path);

    expect(b.forward(X).toArray()).toEqual(a.forward(X).toArray());
  }, 30000);
});

describe("example: embedding bag text classifier", () => {
  test("classifies short token sequences", () => {
    const rng = makeRng(43);
    const vocab = 20;
    const seq = 6;
    const batch = 64;
    const KEYWORD = 3;

    const ids: number[] = [];
    const labels: number[] = [];
    for (let b = 0; b < batch; b++) {
      let seen = 0;
      for (let t = 0; t < seq; t++) {
        const id = Math.floor(rng() * vocab);
        ids.push(id);
        if (id === KEYWORD) seen = 1;
      }
      labels.push(seen);
    }
    const X = new Uint32Array(ids);
    const Y = new Uint32Array(labels);

    class TextClassifier extends candle.nn.Module {
      emb = this.addModule(new candle.nn.Embedding(vocab, 8));
      head = this.addModule(new candle.nn.Linear(8, 2));

      protected _forward(input: Uint32Array): Tensor {
        const h = this.emb.forward(input, [batch, seq]); // [B, T, 8]
        return this.head.forward(h.mean(1)); // mean-pool -> [B, 2]
      }
    }

    const model = new TextClassifier();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.02 });
    for (let step = 0; step < 300; step++) {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      opt.step(l.backward());
    }

    const preds = model.forward(X).argmax(1).to("f32").toArray();
    let correct = 0;
    for (let i = 0; i < labels.length; i++) if (preds[i] === labels[i]) correct++;
    const acc = correct / labels.length;
    console.log(`  embedding-bag classification acc=${acc}`);
    expect(acc).toBeGreaterThan(0.9);
  }, 30000);
});
