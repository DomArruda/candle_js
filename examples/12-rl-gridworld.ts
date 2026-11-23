import candle, { Tensor } from "../lib/index";

// REINFORCE on a tiny 1-D gridworld: states 0..4, actions left/right, goal 4.
// The policy is a trainable [states, actions] logit table.
const STATES = 5;
const GOAL = 4;
const ACTIONS = 2;
const EPISODES_PER_UPDATE = 16;
const UPDATES = 400;

let seed = 99;
const rng = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};
const sample = (p: number[]): number => {
  let r = rng();
  return r < p[0] ? 0 : 1;
};

const logits = candle.variable(new Array(STATES * ACTIONS).fill(0), [STATES, ACTIONS]);
const opt = new candle.optim.AdamW([logits], { lr: 0.1 });

for (let update = 0; update < UPDATES; update++) {
  const logp = logits.logSoftmax(1);
  const all = logp.toArray();
  const probs = (s: number) => [all[s * ACTIONS], all[s * ACTIONS + 1]];

  let loss: Tensor | null = null;
  for (let ep = 0; ep < EPISODES_PER_UPDATE; ep++) {
    let state = 0;
    const chosen: [number, number][] = [];
    for (let t = 0; t < 6; t++) {
      const action = sample(probs(state).map(Math.exp));
      chosen.push([state, action]);
      state = Math.max(0, Math.min(GOAL, state + (action === 0 ? -1 : 1)));
      if (state === GOAL) break;
    }
    if (state !== GOAL) continue;
    for (const [s, a] of chosen) {
      const stepLoss = logp.get(s).get(a).neg();
      loss = loss ? loss.add(stepLoss) : stepLoss;
    }
  }
  if (loss) opt.step(loss.backward());
}

const greedy = logits.argmax(1).to("f32").toArray();
console.log("policy logits:", logits.tolist());
console.log("greedy actions (0=left, 1=right):", greedy);
console.log("expected rights for states 0..3:", [1, 1, 1, 1]);
