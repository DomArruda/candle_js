import { expect, test, describe } from "bun:test";
import candle, { Tensor } from "../lib/index";

interface BenchResult {
  name: string;
  iters: number;
  ms: number;
  detail: string;
}

function bench(
  name: string,
  iters: number,
  warmup: number,
  fn: () => unknown
): BenchResult {
  for (let i = 0; i < warmup; i++) fn();
  const start = performance.now();
  for (let i = 0; i < iters; i++) fn();
  const ms = performance.now() - start;
  return { name, iters, ms, detail: `${(ms / iters).toFixed(3)} ms/iter` };
}

function report(results: BenchResult[]): void {
  console.log("\n  benchmarks (CPU):");
  for (const r of results) {
    console.log(`    ${r.name.padEnd(30)} ${r.detail.padStart(22)}  x${r.iters}`);
  }
}

describe("benchmarks", () => {
  test("matmul throughput", () => {
    const results: BenchResult[] = [];
    for (const n of [128, 256, 512]) {
      const a = candle.randn([n, n]);
      const b = candle.randn([n, n]);
      const iters = n <= 256 ? 20 : 5;
      const r = bench(`matmul ${n}x${n}x${n}`, iters, 2, () => a.matmul(b));
      const gflops = (2 * n ** 3 * iters) / (r.ms * 1e6);
      r.detail = `${(r.ms / iters).toFixed(2)} ms (${gflops.toFixed(2)} GFLOP/s)`;
      results.push(r);
    }
    report(results);
    expect(results.length).toBe(3);
  });

  test("elementwise and reductions", () => {
    const x = candle.randn([256, 256]);
    const y = candle.randn([256, 256]);
    const results = [
      bench("add 65k", 200, 5, () => x.add(y)),
      bench("mul scalar 65k", 200, 5, () => x.mul(2)),
      bench("sum 65k", 200, 5, () => x.sumAll()),
      bench("softmax [256,256]", 100, 5, () => x.softmax(1)),
    ];
    report(results);
    expect(results.every((r) => r.ms > 0)).toBe(true);
  });

  test("MLP forward and training step", () => {
    const model = new candle.nn.Sequential(
      new candle.nn.Linear(64, 128),
      new candle.nn.GELU(),
      new candle.nn.Linear(128, 128),
      new candle.nn.GELU(),
      new candle.nn.Linear(128, 10)
    );
    const X = candle.randn([64, 64]);
    const Y = new Uint32Array(64);
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.001 });

    const fwd = bench("mlp forward [64,64]", 200, 5, () => model.forward(X));
    const step = bench("mlp fwd+bwd+adamw", 100, 5, () => {
      const l = candle.F.crossEntropy(model.forward(X), Y);
      opt.step(l.backward());
    });
    report([fwd, step]);
    expect(step.ms).toBeGreaterThan(0);
  });

  test("transformer block forward", () => {
    const D = 64;
    const H = 4;
    const HD = D / H;
    const FF = 4 * D;
    const B = 8;
    const T = 32;

    class Block extends candle.nn.Module {
      ln1 = this.addModule(new candle.nn.LayerNorm(D));
      q = this.addModule(new candle.nn.Linear(D, D));
      k = this.addModule(new candle.nn.Linear(D, D));
      v = this.addModule(new candle.nn.Linear(D, D));
      proj = this.addModule(new candle.nn.Linear(D, D));
      ln2 = this.addModule(new candle.nn.LayerNorm(D));
      fc1 = this.addModule(new candle.nn.Linear(D, FF));
      fc2 = this.addModule(new candle.nn.Linear(FF, D));

      protected _forward(x: Tensor): Tensor {
        const [batch, seq] = x.shape;
        const toHeads = (t: Tensor) =>
          t.reshape([batch, seq, H, HD]).permute([0, 2, 1, 3]);
        const n = this.ln1.forward(x);
        const qh = toHeads(this.q.forward(n));
        const kh = toHeads(this.k.forward(n));
        const vh = toHeads(this.v.forward(n));
        const scores = qh.matmul(kh.transpose(2, 3)).affine(1 / Math.sqrt(HD), 0);
        const ctx = scores
          .softmax(3)
          .matmul(vh)
          .permute([0, 2, 1, 3])
          .reshape([batch, seq, D]);
        const h = x.add(this.proj.forward(ctx));
        return h.add(this.fc2.forward(this.fc1.forward(this.ln2.forward(h)).gelu()));
      }
    }

    const block = new Block();
    const X = candle.randn([B, T, D]);
    const r = bench(`transformer block [${B},${T},${D}]`, 50, 3, () => block.forward(X));
    report([r]);
    expect(r.ms).toBeGreaterThan(0);
  });

  test("LSTM sequence forward", () => {
    const B = 16;
    const T = 32;
    const F = 32;
    const H = 64;
    const lstm = new candle.nn.LSTM(F, H);
    const X = candle.randn([B, T, F]);
    const r = bench(`lstm seq [${B},${T},${F}]->${H}`, 30, 2, () => lstm.forward(X));
    report([r]);
    expect(r.ms).toBeGreaterThan(0);
  });

  test("end-to-end training throughput", () => {
    const model = new candle.nn.Sequential(
      new candle.nn.Linear(16, 32),
      new candle.nn.Tanh(),
      new candle.nn.Linear(32, 1)
    );
    const X = candle.randn([128, 16]);
    const Y = candle.randn([128, 1]);
    const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.001 });

    const iters = 300;
    const start = performance.now();
    for (let i = 0; i < iters; i++) {
      const l = candle.F.mseLoss(model.forward(X), Y);
      opt.step(l.backward());
    }
    const ms = performance.now() - start;
    const stepsPerSec = (iters / ms) * 1000;
    console.log(
      `\n  end-to-end: ${(ms / iters).toFixed(3)} ms/step, ${stepsPerSec.toFixed(0)} steps/s (batch 128, 16->32->1)`
    );
    expect(stepsPerSec).toBeGreaterThan(0);
  });
});
