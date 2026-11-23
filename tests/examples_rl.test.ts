import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";
import { makeRng } from "./_util";

const CONTEXTS = 4;
const ARMS = 3;
const BEST = [2, 0, 1, 2];
const PER_CONTEXT = 16;
const BATCH = CONTEXTS * PER_CONTEXT;

describe("example: reinforcement learning (REINFORCE)", () => {
  test("learns the best arm in each context", () => {
    // A contextual bandit: one of four one-hot contexts, three arms, and a
    // hidden best arm per context. The agent never sees BEST directly.
    const xs: number[] = [];
    const best: number[] = [];
    for (let c = 0; c < CONTEXTS; c++) {
      for (let i = 0; i < PER_CONTEXT; i++) {
        const row = new Array(CONTEXTS).fill(0);
        row[c] = 1;
        xs.push(...row);
        best.push(BEST[c]);
      }
    }
    const X = candle.tensor(xs, [BATCH, CONTEXTS]);

    class Policy extends candle.nn.Module {
      l1 = this.addModule(new candle.nn.Linear(CONTEXTS, 16));
      l2 = this.addModule(new candle.nn.Linear(16, ARMS));

      protected _forward(x: Tensor): Tensor {
        return this.l2.forward(this.l1.forward(x).tanh());
      }
    }

    const model = new Policy();
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
    const rng = makeRng(42);

    let reward = 0;
    for (let step = 0; step < 300; step++) {
      // log π(a|x) for every context; keep the graph for the update below.
      const logp = model.forward(X).logSoftmax(1);
      const probs = logp.toArray().map(Math.exp);

      const actions: number[] = [];
      const rewards: number[] = [];
      for (let i = 0; i < BATCH; i++) {
        let r = rng();
        let action = ARMS - 1;
        for (let a = 0; a < ARMS; a++) {
          r -= probs[i * ARMS + a];
          if (r <= 0) {
            action = a;
            break;
          }
        }
        actions.push(action);
        rewards.push(action === best[i] ? 1 : -1);
      }
      reward = rewards.reduce((s, r) => s + r, 0) / BATCH;

      // REINFORCE: -E[log π(a|x) * reward]. Plain autograd, no RL API.
      const mask = new Float32Array(BATCH * ARMS);
      actions.forEach((a, i) => (mask[i * ARMS + a] = 1));
      const loss = logp
        .mul(candle.tensor(mask, [BATCH, ARMS]))
        .sum(1)
        .mul(candle.tensor(rewards))
        .meanAll()
        .neg();

      opt.step(loss.backward());
    }

    model.eval();
    const preds = model.forward(X).argmax(1).to("f32").toArray();
    const acc = preds.filter((p, i) => p === best[i]).length / BATCH;
    console.log(
      `  REINFORCE bandit: last-batch reward=${reward.toFixed(2)} greedy acc=${acc}`
    );

    expect(acc).toBeGreaterThan(0.95);
    expect(reward).toBeGreaterThan(0.5);
  }, 30000);
});
