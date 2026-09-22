import { native } from "./native";
import {
  Tensor,
  Var,
  type ShapeLike,
  type TensorOrScalar,
  toShape,
} from "./tensor";

/** Data accepted by {@link tensor} and {@link variable}: nested arrays or a typed array. */
export type TensorData =
  | number
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array
  | number[]
  | number[][]
  | number[][][]
  | number[][][][];

/** @internal Flatten nested arrays / typed arrays into values plus an inferred shape. */
function flatten(data: TensorData): { values: number[]; shape: number[] } {
  if (ArrayBuffer.isView(data)) {
    const values = Array.from(data as unknown as ArrayLike<number>);
    return { values, shape: [values.length] };
  }
  if (typeof data === "number") {
    return { values: [data], shape: [1] };
  }
  const shape: number[] = [];
  let cursor: unknown = data;
  while (Array.isArray(cursor)) {
    shape.push(cursor.length);
    cursor = cursor[0];
  }
  const values: number[] = [];
  const walk = (x: unknown): void => {
    if (Array.isArray(x)) x.forEach(walk);
    else values.push(Number(x));
  };
  walk(data);
  return { values, shape };
}

/**
 * Create a tensor from nested arrays or a typed array.
 *
 * The shape is inferred from the data unless you pass one explicitly.
 * A bare number becomes a 1-element tensor of shape `[1]`.
 *
 * @param data - Nested arrays, a typed array, or a number.
 * @param shape - Optional explicit shape.
 * @example
 * ```ts
 * candle.tensor([[1, 2], [3, 4]]).shape;      // [2, 2]
 * candle.tensor([1, 2, 3, 4], [2, 2]).shape;  // [2, 2]
 * ```
 */
export function tensor(data: TensorData, shape?: ShapeLike): Tensor {
  const flat = flatten(data);
  const s = shape === undefined ? flat.shape : toShape(shape);
  return new Tensor(native.tensorFromF32(new Float32Array(flat.values), s));
}

/**
 * Create a trainable {@link Var} from data. Equivalent to a `Parameter` in
 * PyTorch. Ops applied to it stay differentiable.
 */
export function variable(data: TensorData, shape?: ShapeLike): Var {
  const flat = flatten(data);
  const s = shape === undefined ? flat.shape : toShape(shape);
  return new Var(native.varFromF32(new Float32Array(flat.values), s));
}

/**
 * Create a tensor from `u32` data, e.g. token ids for {@link Embedding}.
 */
export function fromU32(data: Uint32Array | number[], shape?: ShapeLike): Tensor {
  const values = data instanceof Uint32Array ? data : new Uint32Array(data);
  const s = shape === undefined ? [values.length] : toShape(shape);
  return new Tensor(native.tensorFromU32(values, s));
}

/** A tensor of zeros. */
export function zeros(shape: ShapeLike): Tensor {
  return new Tensor(native.tensorZeros(toShape(shape)));
}

/** A tensor of ones. */
export function ones(shape: ShapeLike): Tensor {
  return new Tensor(native.tensorOnes(toShape(shape)));
}

/** A tensor filled with `value`. */
export function full(shape: ShapeLike, value: number): Tensor {
  return new Tensor(native.tensorFull(toShape(shape), value));
}

/** Options for {@link randn} and {@link randnVar}. */
export interface RandnOptions {  /** Standard deviation (default `1`). */
  std?: number;
  /** Mean (default `0`). */
  mean?: number;
}

/**
 * A tensor of standard normal samples.
 * @param shape - Output shape.
 * @param options - Optional `{ std, mean }`.
 * @example `candle.randn([2, 3]); candle.randn([2, 3], { std: 0.02 });`
 */
export function randn(shape: ShapeLike, options: RandnOptions = {}): Tensor {
  return new Tensor(
    native.tensorRandn(options.mean ?? 0, options.std ?? 1, toShape(shape))
  );
}

/** Options for {@link rand}. */
export interface RandOptions {
  /** Lower bound (default `0`). */
  lo?: number;
  /** Upper bound (default `1`). */
  up?: number;
}

/**
 * A tensor of uniform samples in `[lo, up)`.
 * @param shape - Output shape.
 * @param options - Optional `{ lo, up }`.
 */
export function rand(shape: ShapeLike, options: RandOptions = {}): Tensor {
  return new Tensor(
    native.tensorRand(options.lo ?? 0, options.up ?? 1, toShape(shape))
  );
}

/** Evenly spaced values in `[start, end)`. */
export function arange(start: number, end: number, step = 1): Tensor {
  return new Tensor(native.tensorArange(start, end, step));
}

/** The `n x n` identity matrix. */
export function eye(n: number): Tensor {
  return new Tensor(native.tensorEye(n));
}

/** The `n x n` lower-triangular matrix of ones (useful for causal masks). */
export function tril(n: number): Tensor {
  return new Tensor(native.tensorTril(n));
}

/** The `n x n` upper-triangular matrix of ones. */
export function triu(n: number): Tensor {
  return new Tensor(native.tensorTriu(n));
}

/** A trainable tensor of zeros. */
export function zerosVar(shape: ShapeLike): Var {
  return new Var(native.varZeros(toShape(shape)));
}

/** A trainable tensor of ones. */
export function onesVar(shape: ShapeLike): Var {
  return new Var(native.varOnes(toShape(shape)));
}

/** A trainable tensor filled with `value`. */
export function fullVar(shape: ShapeLike, value: number): Var {
  return new Var(native.varFull(toShape(shape), value));
}

/** A trainable tensor of normal samples. */
export function randnVar(shape: ShapeLike, options: RandnOptions = {}): Var {
  return new Var(native.varRandn(options.std ?? 1, toShape(shape)));
}

/**
 * Concatenate tensors along an existing dimension.
 * @example `candle.cat([a, b], 0)`
 */
export function cat(tensors: Tensor[], dim: number): Tensor {
  return new Tensor(native.tensorCat(tensors.map((t) => t._handle), dim));
}

/**
 * Stack tensors along a new dimension.
 * @example `candle.stack([a, b], 0).shape // [2, ...a.shape]`
 */
export function stack(tensors: Tensor[], dim: number): Tensor {
  return new Tensor(native.tensorStack(tensors.map((t) => t._handle), dim));
}

// ---- functional arithmetic (candle.add, candle.matmul, ...) ----

/** Elementwise addition. */
export function add(a: Tensor, b: TensorOrScalar): Tensor {
  return a.add(b);
}

/** Elementwise subtraction. */
export function sub(a: Tensor, b: TensorOrScalar): Tensor {
  return a.sub(b);
}

/** Elementwise multiplication. */
export function mul(a: Tensor, b: TensorOrScalar): Tensor {
  return a.mul(b);
}

/** Elementwise division. */
export function div(a: Tensor, b: TensorOrScalar): Tensor {
  return a.div(b);
}

/** Elementwise maximum. */
export function maximum(a: Tensor, b: TensorOrScalar): Tensor {
  return a.maximum(b);
}

/** Elementwise minimum. */
export function minimum(a: Tensor, b: TensorOrScalar): Tensor {
  return a.minimum(b);
}

/** Matrix multiplication (batched). */
export function matmul(a: Tensor, b: Tensor): Tensor {
  return a.matmul(b);
}

/** Matrix multiplication with batch broadcasting. */
export function broadcastMatmul(a: Tensor, b: Tensor): Tensor {
  return a.broadcastMatmul(b);
}
