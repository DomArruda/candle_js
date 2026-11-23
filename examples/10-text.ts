import candle, { Tensor } from "../lib/index";

// An embedding-bag text classifier: mean-pool token embeddings, then classify.
const VOCAB = 20;
const SEQ = 6;
const BATCH = 64;
const KEYWORD = 3;

let seed = 43;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const ids: number[] = [];
const labels: number[] = [];
for (let b = 0; b < BATCH; b++) {
  let seen = 0;
  for (let t = 0; t < SEQ; t++) {
    const id = Math.floor(rng() * VOCAB);
    ids.push(id);
    if (id === KEYWORD) seen = 1;
  }
  labels.push(seen);
}
const X = candle.fromU32(ids, [BATCH, SEQ]);
const y = new Uint32Array(labels);

class TextClassifier extends candle.nn.Module {
  emb = this.addModule(new candle.nn.Embedding(VOCAB, 8));
  head = this.addModule(new candle.nn.Linear(8, 2));

  protected _forward(input: Tensor): Tensor {
    const h = this.emb.forward(input); // [B, T, 8]
    return this.head.forward(h.mean(1)); // mean-pool -> [B, 2]
  }
}

const model = new TextClassifier();
model.fit(
  { x: X, y },
  {
    epochs: 25,
    batchSize: 16,
    optimizer: new candle.optim.AdamW(model.parameters(), { lr: 0.02 }),
    seed: 4,
  }
);

const preds = candle.predictClasses(model, X);
const acc = preds.filter((p, i) => p === labels[i]).length / labels.length;
console.log("bag-of-tokens accuracy:", acc);
console.log("parameters:", model.numParameters);
