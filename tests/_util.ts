/** Deterministic RNG helpers shared by the example tests. */

/** A small LCG so examples are reproducible. */
export function makeRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

/** Standard normal sample via Box-Muller. */
export function gaussian(rng: () => number): number {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Mean of a numeric array. */
export function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

/** Flatten nested arrays into a flat number[]. */
export function flatten(xs: unknown): number[] {
  const out: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) x.forEach(walk);
    else out.push(Number(x));
  };
  walk(xs);
  return out;
}

/** Count how many predictions match the targets. */
export function accuracy(preds: number[], targets: number[]): number {
  let correct = 0;
  for (let i = 0; i < preds.length; i++) if (preds[i] === targets[i]) correct++;
  return correct / preds.length;
}
