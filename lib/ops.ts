import { native } from "./native";
import { Tensor, Var, type ShapeLike, unwrap } from "./tensor";

/**
 * Stateless operations, mirroring PyTorch's `nn.functional`.
 *
 * These take tensors (or vars) and return tensors; they hold no parameters.
 * The matching parameterised layers live in `lib/nn.ts`.
 */

/** Rectified linear unit: `max(x, 0)`. */
export function relu(x: Tensor): Tensor {
  return new Tensor(native.tensorRelu(unwrap(x)));
}

/** Gaussian error linear unit (tanh approximation), used by GPT-style models. */
export function gelu(x: Tensor): Tensor {
  return new Tensor(native.tensorGelu(unwrap(x)));
}

/** Gaussian error linear unit (exact erf formulation). */
export function geluErf(x: Tensor): Tensor {
  return new Tensor(native.tensorGeluErf(unwrap(x)));
}

/** Sigmoid linear unit (SiLU / swish). */
export function silu(x: Tensor): Tensor {
  return new Tensor(native.tensorSilu(unwrap(x)));
}

/** Error function, elementwise. */
export function erf(x: Tensor): Tensor {
  return new Tensor(native.tensorErf(unwrap(x)));
}

/**
 * SwiGLU: split the last dimension in half and return
 * `silu(firstHalf) * secondHalf`.
 */
export function swiGLU(x: Tensor): Tensor {
  return new Tensor(native.tensorSwiGLU(unwrap(x)));
}

/** Hard sigmoid: `clamp(x/6 + 0.5, 0, 1)`. */
export function hardSigmoid(x: Tensor): Tensor {
  return new Tensor(native.tensorHardSigmoid(unwrap(x)));
}

/** Hard swish: `x * hardSigmoid(x)`. */
export function hardSwish(x: Tensor): Tensor {
  return new Tensor(native.tensorHardSwish(unwrap(x)));
}

/** Squared ReLU: `relu(x)^2`. */
export function relu2(x: Tensor): Tensor {
  return new Tensor(native.tensorRelu2(unwrap(x)));
}

/** ReLU capped at 6: `clamp(x, 0, 6)`. */
export function relu6(x: Tensor): Tensor {
  return new Tensor(native.tensorRelu6(unwrap(x)));
}

/** Scaled exponential linear unit. */
export function selu(
  x: Tensor,
  alpha = 1.6732632423543772,
  gamma = 1.0507009873554805
): Tensor {
  return new Tensor(native.tensorSelu(unwrap(x), alpha, gamma));
}

/** Mish activation. */
export function mish(x: Tensor): Tensor {
  return new Tensor(native.tensorMish(unwrap(x)));
}

/** Logistic sigmoid. */
export function sigmoid(x: Tensor): Tensor {
  return new Tensor(native.tensorSigmoid(unwrap(x)));
}

/** Hyperbolic tangent. */
export function tanh(x: Tensor): Tensor {
  return new Tensor(native.tensorTanh(unwrap(x)));
}

/** Exponential linear unit. */
export function elu(x: Tensor, alpha = 1.0): Tensor {
  return new Tensor(native.tensorElu(unwrap(x), alpha));
}

/** Leaky ReLU. */
export function leakyRelu(x: Tensor, negativeSlope = 0.01): Tensor {
  return new Tensor(native.tensorLeakyRelu(unwrap(x), negativeSlope));
}

/** Softmax along `dim`. */
export function softmax(x: Tensor, dim: number): Tensor {
  return new Tensor(native.tensorSoftmax(unwrap(x), dim));
}

/** Log-softmax along `dim`. */
export function logSoftmax(x: Tensor, dim: number): Tensor {
  return new Tensor(native.tensorLogSoftmax(unwrap(x), dim));
}

/** Elementwise select: `cond ? onTrue : onFalse`. All three must share a shape. */
export function where(cond: Tensor, onTrue: Tensor, onFalse: Tensor): Tensor {
  return new Tensor(
    native.tensorWhere(unwrap(cond), unwrap(onTrue), unwrap(onFalse))
  );
}

/**
 * Affine transform `input @ weight.T + bias`, the functional form of
 * {@link Linear}.
 *
 * @param input - Shape `[..., inFeatures]`.
 * @param weight - Shape `[outFeatures, inFeatures]`.
 * @param bias - Optional shape `[outFeatures]`.
 */
export function linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
  const out = input.broadcastMatmul(weight.t());
  return bias ? out.add(bias) : out;
}

/**
 * Embedding lookup: index `weight` (`[vocab, dim]`) by integer ids.
 * @param weight - The embedding matrix.
 * @param ids - Token ids.
 * @param shape - Shape of `ids` (defaults to `[ids.length]`).
 */
export function embedding(
  weight: Tensor | Var,
  ids: Uint32Array | number[],
  shape?: ShapeLike
): Tensor {
  const arr = ids instanceof Uint32Array ? ids : new Uint32Array(ids);
  const s = shape === undefined ? [arr.length] : Array.isArray(shape) ? shape : [shape];
  return new Tensor(native.tensorEmbedding(unwrap(weight), arr, s));
}

// ---- losses ----

/**
 * Mean squared error between `input` and `target`, averaged over all elements.
 */
export function mseLoss(input: Tensor, target: Tensor): Tensor {
  return new Tensor(native.mseLoss(unwrap(input), unwrap(target)));
}

/**
 * Cross-entropy from raw logits.
 *
 * @param input - Logits of shape `[N, C]`.
 * @param target - Class indices of length `N`.
 * @example
 * ```ts
 * const loss = candle.nn.functional.crossEntropy(logits, new Uint32Array([3, 1]));
 * ```
 */
export function crossEntropy(
  input: Tensor,
  target: Uint32Array | number[]
): Tensor {
  const t = target instanceof Uint32Array ? target : new Uint32Array(target);
  return new Tensor(native.crossEntropyLoss(unwrap(input), t));
}

/**
 * Negative log-likelihood. `input` must already contain log-probabilities
 * (e.g. the output of {@link logSoftmax}).
 *
 * @param input - Log-probabilities of shape `[N, C]`.
 * @param target - Class indices of length `N`.
 */
export function nll(input: Tensor, target: Uint32Array | number[]): Tensor {
  const t = target instanceof Uint32Array ? target : new Uint32Array(target);
  return new Tensor(native.nllLoss(unwrap(input), t));
}

/** Binary cross-entropy with logits (numerically stable). */
export function bceWithLogit(input: Tensor, target: Tensor): Tensor {
  return new Tensor(native.bceWithLogitLoss(unwrap(input), unwrap(target)));
}

/** Huber loss (smooth L1). */
export function huber(input: Tensor, target: Tensor, delta = 1.0): Tensor {
  return new Tensor(native.huberLoss(unwrap(input), unwrap(target), delta));
}
