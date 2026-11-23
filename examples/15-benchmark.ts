import candle from "../lib/index";

function time(fn: () => void, iters: number): number {
  for (let i = 0; i < 3; i++) fn();
  let best = Infinity;
  for (let round = 0; round < 3; round++) {
    const start = performance.now();
    for (let i = 0; i < iters; i++) fn();
    best = Math.min(best, (performance.now() - start) / iters);
  }
  return best;
}

for (const n of [128, 256, 512]) {
  const a = candle.randn([n, n]);
  const b = candle.randn([n, n]);
  const iters = n === 512 ? 3 : 10;
  const ms = time(() => a.matmul(b), iters);
  console.log(
    `matmul ${n}^3: ${ms.toFixed(2)} ms  (${((2 * n ** 3) / (ms * 1e6)).toFixed(2)} GFLOP/s)`
  );
}

const x = candle.randn([256, 256]);
const y = candle.randn([256, 256]);
console.log(`add 65k: ${time(() => x.add(y), 100).toFixed(3)} ms`);
console.log(`softmax [256,256]: ${time(() => x.softmax(1), 50).toFixed(3)} ms`);
