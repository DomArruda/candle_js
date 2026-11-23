import * as F from "./ops";
import { AdamW, type Optimizer } from "./optim";
import { Tensor } from "./tensor";
import type { Module } from "./nn";

declare const console: { log(...args: unknown[]): void };

/** A classification target: integer class ids, one per sample. */
export type ClassIds = Uint32Array | number[];

/** A training target: class ids, or a tensor for regression / multi-output. */
export type Target = Tensor | ClassIds;

/** A supervised dataset. `x` is laid out along the first dimension. */
export interface FitData {
  x: Tensor;
  y: Target;
}

/** Metrics supported by {@link fit} and {@link evaluate}. */
export type Metric = "accuracy" | "mse" | "mae";

/** How to compute the loss; `"auto"` picks cross-entropy for class ids, MSE otherwise. */
export type LossSpec = "auto" | "mse" | "crossEntropy" | ((pred: Tensor, target: Target) => Tensor);

/** Epoch-level logs. Metric keys are filled in when requested. */
export interface EpochLogs {
  loss: number;
  [key: string]: number;
}

/** What {@link fit} returns. */
export interface History {
  loss: number[];
  valLoss: number[];
  metrics: Record<string, number[]>;
  valMetrics: Record<string, number[]>;
  epochs: number;
}

/** Options for {@link fit}. */
export interface FitConfig {
  /** Number of passes over the training data (default `10`). */
  epochs?: number;
  /** Samples per update (default `32`). */
  batchSize?: number;
  /** Fraction held out for validation, e.g. `0.2` (default `0`). */
  validationSplit?: number;
  /** Optimizer; defaults to AdamW with `lr: 0.01`. */
  optimizer?: Optimizer;
  /** Loss function or `"auto"` (default `"auto"`). */
  loss?: LossSpec;
  /** Extra metrics. Defaults to accuracy for class ids, MSE for tensors. */
  metrics?: Metric[];
  /** Shuffle each epoch (default `true`). */
  shuffle?: boolean;
  /** Print one line per epoch (default `false`). */
  verbose?: boolean;
  /** Seed for shuffling and the train/validation split (default `42`). */
  seed?: number;
  /** Called after every epoch with the same logs that go into the history. */
  onEpochEnd?: (epoch: number, logs: EpochLogs) => void;
}

/** Options for {@link evaluate}. */
export interface EvaluateConfig {
  batchSize?: number;
  loss?: LossSpec;
  metrics?: Metric[];
}

/** @internal Small deterministic RNG so splits are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** @internal */
function shuffled(n: number, rng: () => number): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }
  return idx;
}

/** @internal */
function isClassIds(target: Target): target is ClassIds {
  return target instanceof Uint32Array || Array.isArray(target);
}

/** @internal */
function takeX(x: Tensor, idx: number[]): Tensor {
  return idx.length === x.shape[0] && idx.every((v, i) => v === i)
    ? x
    : x.indexSelect(new Uint32Array(idx), 0);
}

/** @internal */
function takeY(y: Target, idx: number[]): Target {
  if (y instanceof Tensor) return takeX(y, idx);
  const picked = idx.map((i) => (y as ArrayLike<number>)[i]);
  return y instanceof Uint32Array ? new Uint32Array(picked) : picked;
}

/** @internal */
function takeData(data: FitData, idx: number[]): FitData {
  return { x: takeX(data.x, idx), y: takeY(data.y, idx) };
}

/** @internal */
function resolveLoss(spec: LossSpec | undefined, target: Target): (pred: Tensor, target: Target) => Tensor {
  if (typeof spec === "function") return spec;
  if (spec === "crossEntropy") return (pred, t) => F.crossEntropy(pred, t as ClassIds);
  if (spec === "mse") return (pred, t) => F.mseLoss(pred, t as Tensor);
  return isClassIds(target)
    ? (pred, t) => F.crossEntropy(pred, t as ClassIds)
    : (pred, t) => F.mseLoss(pred, t as Tensor);
}

/** @internal */
function resolveMetrics(metrics: Metric[] | undefined, target: Target): Metric[] {
  if (metrics) return metrics;
  return isClassIds(target) ? ["accuracy"] : ["mse"];
}

/** @internal */
function metric(pred: Tensor, target: Target, name: Metric): number {
  if (name === "accuracy") {
    if (!isClassIds(target)) {
      throw new Error("accuracy needs class id targets (Uint32Array or number[])");
    }
    const predicted = pred.argmax(pred.rank - 1).to("f32").toArray();
    const wanted = Array.from(target);
    let correct = 0;
    for (let i = 0; i < wanted.length; i++) if (predicted[i] === wanted[i]) correct++;
    return wanted.length === 0 ? 0 : correct / wanted.length;
  }
  if (!(target instanceof Tensor)) {
    throw new Error(`${name} needs a tensor target`);
  }
  if (name === "mse") return F.mseLoss(pred, target).item();
  return pred.sub(target).abs().meanAll().item();
}

/** @internal Evaluate one data split without training. */
function evaluateData(
  model: Module,
  data: FitData,
  idx: number[],
  lossFn: (pred: Tensor, target: Target) => Tensor,
  metrics: Metric[]
): EpochLogs {
  if (idx.length === 0) return { loss: 0 };
  const { x, y } = takeData(data, idx);
  const pred = model.forward(x);
  const logs: EpochLogs = { loss: lossFn(pred, y).item() };
  for (const m of metrics) logs[m] = metric(pred, y, m);
  return logs;
}

/**
 * Train `model` on `data` with a high-level, Keras-like loop.
 *
 * Supervised only; for custom losses or multi-input models, write the loop
 * yourself (the low-level API is unchanged). Dropout is enabled during
 * training and disabled for validation.
 *
 * @example
 * ```ts
 * const history = model.fit(
 *   { x: X, y: labels },
 *   { epochs: 20, batchSize: 32, validationSplit: 0.2 },
 * );
 * history.valMetrics.accuracy.at(-1);
 * ```
 */
export function fit(model: Module, data: FitData, config: FitConfig = {}): History {
  const n = data.x.shape[0];
  if (n === 0) throw new Error("fit: empty training set");
  const epochs = config.epochs ?? 10;
  const batchSize = config.batchSize ?? 32;
  if (batchSize <= 0) throw new Error("fit: batchSize must be positive");
  const rng = mulberry32(config.seed ?? 42);
  const lossFn = resolveLoss(config.loss, data.y);
  const metrics = resolveMetrics(config.metrics, data.y);
  const opt = config.optimizer ?? new AdamW(model.parameters(), { lr: 0.01 });

  const order = (config.shuffle ?? true) ? shuffled(n, rng) : Array.from({ length: n }, (_, i) => i);
  const valCount = Math.floor(n * (config.validationSplit ?? 0));
  if (valCount >= n) throw new Error("fit: validationSplit leaves no training samples");
  const trainIdx = order.slice(0, n - valCount);
  const valIdx = order.slice(n - valCount);

  const history: History = { loss: [], valLoss: [], metrics: {}, valMetrics: {}, epochs };
  for (const m of metrics) {
    history.metrics[m] = [];
    history.valMetrics[m] = [];
  }

  const wasTraining = model.training;
  model.train(true);
  try {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const epochOrder = (config.shuffle ?? true) ? shuffled(trainIdx.length, rng) : trainIdx;
      let batchLoss = 0;
      for (let start = 0; start < epochOrder.length; start += batchSize) {
        const batch = epochOrder.slice(start, start + batchSize).map((i) => trainIdx[i]);
        const { x, y } = takeData(data, batch);
        const loss = lossFn(model.forward(x), y);
        batchLoss += loss.item() * batch.length;
        opt.step(loss.backward());
      }

      model.eval();
      const trainLogs = evaluateData(model, data, trainIdx, lossFn, metrics);
      const valLogs = evaluateData(model, data, valIdx, lossFn, metrics);
      model.train(true);

      const logs: EpochLogs = { loss: batchLoss / trainIdx.length };
      for (const m of metrics) logs[m] = trainLogs[m];
      if (valIdx.length > 0) {
        logs.valLoss = valLogs.loss;
        for (const m of metrics) logs[`val${m[0].toUpperCase()}${m.slice(1)}`] = valLogs[m];
      }

      history.loss.push(logs.loss);
      if (valIdx.length > 0) {
        history.valLoss.push(valLogs.loss);
        for (const m of metrics) history.valMetrics[m].push(valLogs[m]);
      }
      for (const m of metrics) history.metrics[m].push(logs[m]);
      config.onEpochEnd?.(epoch, logs);
      if (config.verbose) {
        const parts = [`epoch ${epoch + 1}/${epochs}`, `loss ${logs.loss.toFixed(4)}`];
        for (const m of metrics) parts.push(`${m} ${logs[m].toFixed(4)}`);
        if (valIdx.length > 0) {
          parts.push(`valLoss ${logs.valLoss.toFixed(4)}`);
          for (const m of metrics) {
            parts.push(`val${m[0].toUpperCase()}${m.slice(1)} ${logs[`val${m[0].toUpperCase()}${m.slice(1)}`].toFixed(4)}`);
          }
        }
        console.log(`  ${parts.join("  ")}`);
      }
    }
  } finally {
    model.train(wasTraining);
  }
  return history;
}

/**
 * Score `model` on a dataset without updating it. Dropout is disabled.
 *
 * @example `model.evaluate({ x: X, y: labels }, { metrics: ["accuracy"] }).accuracy`
 */
export function evaluate(model: Module, data: FitData, config: EvaluateConfig = {}): EpochLogs {
  const n = data.x.shape[0];
  const lossFn = resolveLoss(config.loss, data.y);
  const metrics = resolveMetrics(config.metrics, data.y);
  const wasTraining = model.training;
  model.eval();
  try {
    return evaluateData(
      model,
      data,
      Array.from({ length: n }, (_, i) => i),
      lossFn,
      metrics
    );
  } finally {
    model.train(wasTraining);
  }
}

/**
 * Run the model and return a detached tensor. Dropout is disabled.
 * @example `model.predict(X).argmax(1).to("f32").toArray()`
 */
export function predict(model: Module, x: Tensor): Tensor {
  const wasTraining = model.training;
  model.eval();
  try {
    return model.forward(x).detach();
  } finally {
    model.train(wasTraining);
  }
}

/** Argmax class ids for each row of a classification model's predictions. */
export function predictClasses(model: Module, x: Tensor): number[] {
  const out = predict(model, x);
  return out.argmax(out.rank - 1).to("f32").toArray();
}

/** Options for {@link trainTestSplit}. */
export interface SplitOptions {
  /** Fraction for the test set (default `0.2`). */
  testSize?: number;
  /** Shuffle before splitting (default `true`). */
  shuffle?: boolean;
  /** Shuffle seed (default `42`). */
  seed?: number;
}

/**
 * Split a dataset into train and test parts along the first dimension.
 *
 * @example
 * ```ts
 * const { train, test } = candle.trainTestSplit({ x: X, y: labels }, { testSize: 0.25 });
 * const model = new candle.nn.Sequential(...);
 * model.fit(train, { epochs: 20 });
 * model.evaluate(test, { metrics: ["accuracy"] });
 * ```
 */
export function trainTestSplit(
  data: FitData,
  options: SplitOptions = {}
): { train: FitData; test: FitData } {
  const n = data.x.shape[0];
  const rng = mulberry32(options.seed ?? 42);
  const order = (options.shuffle ?? true) ? shuffled(n, rng) : Array.from({ length: n }, (_, i) => i);
  const testCount = Math.floor(n * (options.testSize ?? 0.2));
  return {
    train: takeData(data, order.slice(0, n - testCount)),
    test: takeData(data, order.slice(n - testCount)),
  };
}
