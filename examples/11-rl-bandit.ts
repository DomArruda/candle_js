import candle from "../lib/index";

// REINFORCE on a contextual bandit: four contexts, three arms, hidden best arm.
const BEST = [2, 0, 1, 2];
const CONTEXTS = 4;
const ARMS = 3;
const PER_CONTEXT = 16;
const BATCH = CONTEXTS * PER_CONTEXT;

const contexts = Array.from({ length: BATCH }, (_, i) => Math.floor(i / PER_CONTEXT));
const best = contexts.map((c) => BEST[c]);
const X = candle.tensor(
  contexts.flatMap((c) => [0, 1, 2, 3].map((k) => (k === c ? 1 : 0))),
  [BATCH, CONTEXTS]
);

class Policy extends candle.nn.Module {
  l1 = this.addModule(new candle.nn.Linear(CONTEXTS, 16));
  l2 = this.addModule(new candle.nn.Linear(16, ARMS));

  protected _forward(x: candle.Tensor): candle.Tensor {
    return this.l2.forward(this.l1.forward(x).tanh());
  }
}

const model = new Policy();
const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
let seed = 42;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const sample = (p: number[]): number => {
  let r = rng();
  for (let a = 0; a < p.length; a++) if ((r -= p[a]) <= 0) return a;
  return p.length - 1;
};

for (let step = 0; step < 300; step++) {
  const logp = model.forward(X).logSoftmax(1);
  const probs = logp.toArray().map(Math.exp);
  const actions = Array.from({ length: BATCH }, (_, i) =>
    sample(probs.slice(i * ARMS, i * ARMS + ARMS))
  );
  const rewards = actions.map((a, i) => (a === best[i] ? 1 : -1));

  const mask = new Float32Array(BATCH * ARMS);
  actions.forEach((a, i) => (mask[i * ARMS + a] = 1));
  const loss = logp
    .mul(candle.tensor(mask, [BATCH, ARMS]))
    .sum(1)
    .mul(candle.tensor(rewards))
    .meanAll()
    .neg();
  opt.step(loss.backward());

  if ((step + 1) % 100 === 0) {
    const avg = rewards.reduce((s, r) => s + r, 0) / BATCH;
    console.log(`step ${step + 1}: reward=${avg.toFixed(2)}`);
  }
}

const preds = candle.predictClasses(model, X);
const perContext = contexts.map((_, i) => preds[i * PER_CONTEXT]);
const acc = preds.filter((p, i) => p === best[i]).length / BATCH;
console.log("learned arms:", perContext, "wanted:", BEST);
console.log("greedy accuracy:", acc);
