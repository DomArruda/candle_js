import { native } from "./native";
import { GradStore, Var } from "./tensor";

/**
 * Base class for parameter update rules.
 *
 * Optimizers are built from a list of {@link Var}s — normally
 * `model.parameters()` — and consume the {@link GradStore} returned by
 * `loss.backward()`:
 *
 * @example
 * ```ts
 * const opt = new candle.optim.AdamW(model.parameters(), { lr: 0.05 });
 * for (let step = 0; step < 400; step++) {
 *   const loss = candle.nn.functional.mseLoss(model.forward(X), Y);
 *   opt.step(loss.backward());
 * }
 * ```
 */
export abstract class Optimizer {
  /** Apply one update using the supplied gradients. */
  abstract step(grads: GradStore): void;

  /** Change the learning rate for subsequent steps. */
  abstract setLearningRate(lr: number): void;
}

/** Options for {@link SGD}. */
export interface SGDConfig {
  /** Learning rate (default `0.01`). */
  lr?: number;
}

/**
 * Stochastic gradient descent: `p <- p - lr * grad(p)`.
 *
 * Note: candle's SGD does not support momentum or weight decay.
 */
export class SGD extends Optimizer {
  private _opt: unknown;

  /**
   * @param params - Trainable variables, usually `model.parameters()`.
   * @param config - Learning rate.
   */
  constructor(params: Var[], config: SGDConfig = {}) {
    super();
    this._opt = native.sgdNewFromVars(
      params.map((p) => p._handle),
      config.lr ?? 0.01
    );
  }

  step(grads: GradStore): void {
    native.sgdOptStep(this._opt, grads._handle);
  }

  setLearningRate(lr: number): void {
    native.sgdSetLr(this._opt, lr);
  }
}

/** Options for {@link AdamW}. */
export interface AdamWConfig {
  /** Learning rate (default `1e-3`). */
  lr?: number;
  /** Exponential decay rates for the moment estimates (default `[0.9, 0.999]`). */
  betas?: [number, number];
  /** Numerical stability epsilon (default `1e-8`). */
  eps?: number;
  /** Decoupled weight decay (default `0`). */
  weightDecay?: number;
}

/**
 * AdamW — Adam with decoupled weight decay. A solid default optimizer.
 */
export class AdamW extends Optimizer {
  private _opt: unknown;

  /**
   * @param params - Trainable variables, usually `model.parameters()`.
   * @param config - Learning rate, betas, epsilon and weight decay.
   */
  constructor(params: Var[], config: AdamWConfig = {}) {
    super();
    const [beta1, beta2] = config.betas ?? [0.9, 0.999];
    this._opt = native.adamwNewFromVars(
      params.map((p) => p._handle),
      config.lr ?? 1e-3,
      beta1,
      beta2,
      config.eps ?? 1e-8,
      config.weightDecay ?? 0
    );
  }

  step(grads: GradStore): void {
    native.adamwStep(this._opt, grads._handle);
  }

  setLearningRate(lr: number): void {
    native.adamwSetLr(this._opt, lr);
  }
}
