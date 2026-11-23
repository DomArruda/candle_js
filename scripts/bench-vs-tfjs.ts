// candle_js vs TensorFlow.js on the same runtime and CPU.
//
//   bun run bench:tfjs
//
// TensorFlow.js is the pure-JS `@tensorflow/tfjs` CPU backend, not the native
// `@tensorflow/tfjs-node` binding, so this compares candle's Rust kernels with
// TF.js's JavaScript kernels. Both sides run on Bun.
import candle from "../lib/index";
import * as tf from "@tensorflow/tfjs";

interface Case {
  name: string;
  iters: number;
  candle: () => void;
  tfjs: () => void | Promise<unknown>;
}

function warmup(fn: () => void, times: number): void {
  for (let i = 0; i < times; i++) fn();
}

function timeCandle(fn: () => void, iters: number, rounds = 3): number {
  let best = Infinity;
  for (let round = 0; round < rounds; round++) {
    const start = performance.now();
    for (let i = 0; i < iters; i++) fn();
    best = Math.min(best, (performance.now() - start) / iters);
  }
  return best;
}

async function timeTfjs(fn: () => void | Promise<unknown>, iters: number, rounds = 3): Promise<number> {
  let best = Infinity;
  for (let round = 0; round < rounds; round++) {
    const start = performance.now();
    for (let i = 0; i < iters; i++) await fn();
    best = Math.min(best, (performance.now() - start) / iters);
  }
  return best;
}

function addCases(cases: Case[]): void {
  for (const n of [128, 256, 512]) {
    const a = candle.randn([n, n]);
    const b = candle.randn([n, n]);
    const ta = tf.randomNormal([n, n]);
    const tb = tf.randomNormal([n, n]);
    cases.push({
      name: `matmul ${n}x${n}x${n}`,
      iters: n === 512 ? 2 : 10,
      candle: () => {
        a.matmul(b);
      },
      tfjs: () => {
        const c = ta.matMul(tb);
        c.dispose();
      },
    });
  }

  const x = candle.randn([256, 256]);
  const y = candle.randn([256, 256]);
  const tx = tf.randomNormal([256, 256]);
  const ty = tf.randomNormal([256, 256]);

  cases.push(
    {
      name: "add 65k",
      iters: 100,
      candle: () => {
        x.add(y);
      },
      tfjs: () => {
        const z = tx.add(ty);
        z.dispose();
      },
    },
    {
      name: "softmax [256,256]",
      iters: 50,
      candle: () => {
        x.softmax(1);
      },
      tfjs: () => {
        const z = tf.softmax(tx, 1);
        z.dispose();
      },
    }
  );
}

function mlpCases(cases: Case[]): void {
  const candleModel = new candle.nn.Sequential(
    new candle.nn.Linear(64, 128),
    new candle.nn.ReLU(),
    new candle.nn.Linear(128, 128),
    new candle.nn.ReLU(),
    new candle.nn.Linear(128, 10)
  );
  const X = candle.randn([64, 64]);

  const tfModel = tf.sequential();
  tfModel.add(tf.layers.dense({ units: 128, activation: "relu", inputShape: [64] }));
  tfModel.add(tf.layers.dense({ units: 128, activation: "relu" }));
  tfModel.add(tf.layers.dense({ units: 10 }));
  const tX = tf.randomNormal([64, 64]);

  cases.push({
    name: "mlp forward [64,64]",
    iters: 50,
    candle: () => {
      candleModel.forward(X);
    },
    tfjs: () => {
      const out = tfModel.predict(tX) as tf.Tensor;
      out.dispose();
    },
  });
}

function trainCases(cases: Case[]): void {
  const candleModel = new candle.nn.Sequential(
    new candle.nn.Linear(16, 32),
    new candle.nn.Tanh(),
    new candle.nn.Linear(32, 1)
  );
  const X = candle.randn([128, 16]);
  const Y = candle.randn([128, 1]);
  const opt = new candle.optim.AdamW(candleModel.parameters(), { lr: 0.001 });

  const tfModel = tf.sequential();
  tfModel.add(tf.layers.dense({ units: 32, activation: "tanh", inputShape: [16] }));
  tfModel.add(tf.layers.dense({ units: 1 }));
  tfModel.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  const tX = tf.randomNormal([128, 16]);
  const tY = tf.randomNormal([128, 1]);

  cases.push({
    name: "train step [128,16]->1",
    iters: 20,
    candle: () => {
      const loss = candle.F.mseLoss(candleModel.forward(X), Y);
      opt.step(loss.backward());
    },
    tfjs: () => tfModel.trainOnBatch(tX, tY),
  });
}

async function main(): Promise<void> {
  await tf.setBackend("cpu");
  await tf.ready();

  const cases: Case[] = [];
  addCases(cases);
  mlpCases(cases);
  trainCases(cases);

  const bun = process.versions.bun ?? "?";
  console.log(
    `\ncandle_js vs @tensorflow/tfjs ${tf.version.tfjs} ` +
      `(CPU, Bun ${bun}, ${process.platform}-${process.arch})`
  );
  console.log(
    `${"case".padEnd(24)} ${"candle".padStart(12)} ${"tfjs".padStart(12)} ${"speedup".padStart(9)}`
  );
  console.log("-".repeat(61));

  for (const c of cases) {
    warmup(c.candle, 2);
    await timeTfjs(c.tfjs, 2);
    const candleMs = timeCandle(c.candle, c.iters);
    const tfjsMs = await timeTfjs(c.tfjs, Math.min(c.iters, 5));
    console.log(
      `${c.name.padEnd(24)} ` +
        `${(candleMs.toFixed(3) + " ms").padStart(12)} ` +
        `${(tfjsMs.toFixed(3) + " ms").padStart(12)} ` +
        `${(tfjsMs / candleMs).toFixed(2).padStart(8)}x`
    );
  }
  console.log(
    "\n  speedup > 1 means candle_js is faster. TF.js here is the pure-JS backend.\n"
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
