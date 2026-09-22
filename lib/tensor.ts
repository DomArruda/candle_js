import { native } from "./native";

/**
 * Anything accepted where a tensor operand is expected: another tensor, or a
 * plain number that will be broadcast against it.
 */
export type TensorOrScalar = Tensor | number;

/** A shape given either as a list of dimensions or as a single dimension. */
export type ShapeLike = number | number[];

/** A dimension selector: one axis or several. */
export type DimsLike = number | number[];

/** @internal Normalize `number | number[]` into a `number[]`. */
export function toShape(shape: ShapeLike): number[] {
  return Array.isArray(shape) ? shape : [shape];
}

/** @internal Extract the native handle from a wrapped value. */
export function unwrap(x: TensorOrScalar): unknown {
  return x instanceof Tensor ? x._handle : x;
}

/** @internal Wrap a native tensor handle. */
export function wrap(handle: unknown): Tensor {
  return new Tensor(handle);
}

/** @internal Read `{ shape, data }` from the addon as a flat `number[]`. */
function readF32(handle: unknown): { shape: number[]; values: number[] } {
  const { shape, data } = native.tensorToF32(handle);
  return { shape, values: Array.from(new Float32Array(data)) };
}

/**
 * An n-dimensional array of `f32` values, backed by candle in Rust.
 *
 * Ops return plain `Tensor`s even when called on a trainable {@link Var}; the
 * autograd graph is built by candle as you compose them, and
 * {@link Tensor.backward} turns a scalar loss into a {@link GradStore}.
 *
 * Values are read back with {@link Tensor.toArray} (flat),
 * {@link Tensor.tolist} (nested) or {@link Tensor.item} (single element).
 *
 * @example
 * ```ts
 * const a = candle.tensor([1, 2, 3, 4], [2, 2]);
 * const b = candle.tensor([5, 6, 7, 8], [2, 2]);
 * a.matmul(b).tolist(); // [[19, 22], [43, 50]]
 * ```
 */
export class Tensor {
  /** @internal Native handle. Not part of the public API. */
  readonly _handle: unknown;

  /** @internal */
  constructor(handle: unknown) {
    this._handle = handle;
  }

  // ---- metadata ----

  /** The dimensions of the tensor, e.g. `[2, 3]`. */
  get shape(): number[] {
    return native.tensorShape(this._handle);
  }

  /** The element type, e.g. `"f32"` or `"u32"`. */
  get dtype(): string {
    return native.tensorDtype(this._handle);
  }

  /** Number of dimensions (the length of {@link Tensor.shape}). */
  get rank(): number {
    return native.tensorRank(this._handle);
  }

  /** Total number of elements. */
  get numel(): number {
    return native.tensorElemCount(this._handle);
  }

  // ---- readback ----

  /** All elements as a flat `number[]`, in row-major order. */
  toArray(): number[] {
    return readF32(this._handle).values;
  }

  /** The data as a `Float32Array` (zero-copy view of a fresh buffer). */
  toFloat32Array(): Float32Array {
    return new Float32Array(native.tensorToF32(this._handle).data);
  }

  /**
   * The data as nested JS arrays matching {@link Tensor.shape}.
   * @example `candle.eye(2).tolist() // [[1, 0], [0, 1]]`
   */
  tolist(): number | number[] | unknown[] {
    const flat = this.toArray();
    const shape = this.shape;
    let i = 0;
    const build = (dims: number[]): unknown => {
      if (dims.length === 0) return flat[i++];
      const [head, ...rest] = dims;
      return Array.from({ length: head }, () => build(rest));
    };
    return build(shape) as number[] | unknown[];
  }

  /**
   * The value of a single-element tensor.
   * @throws If the tensor does not have exactly one element.
   */
  item(): number {
    const values = this.toArray();
    if (values.length !== 1) {
      throw new Error(`item() requires exactly one element, got ${values.length}`);
    }
    return values[0];
  }

  // ---- elementwise arithmetic ----

  /** Elementwise addition; broadcasts and accepts a scalar. */
  add(other: TensorOrScalar): Tensor {
    return wrap(native.tensorAdd(this._handle, unwrap(other)));
  }

  /** Elementwise subtraction; broadcasts and accepts a scalar. */
  sub(other: TensorOrScalar): Tensor {
    return wrap(native.tensorSub(this._handle, unwrap(other)));
  }

  /** Elementwise multiplication; broadcasts and accepts a scalar. */
  mul(other: TensorOrScalar): Tensor {
    return wrap(native.tensorMul(this._handle, unwrap(other)));
  }

  /** Elementwise division; broadcasts and accepts a scalar. */
  div(other: TensorOrScalar): Tensor {
    return wrap(native.tensorDiv(this._handle, unwrap(other)));
  }

  /** Elementwise maximum against another tensor or scalar. */
  maximum(other: TensorOrScalar): Tensor {
    return wrap(native.tensorMaximum(this._handle, unwrap(other)));
  }

  /** Elementwise minimum against another tensor or scalar. */
  minimum(other: TensorOrScalar): Tensor {
    return wrap(native.tensorMinimum(this._handle, unwrap(other)));
  }

  /** `self * mul + add`, elementwise. */
  affine(mul: number, add: number): Tensor {
    return wrap(native.tensorAffine(this._handle, mul, add));
  }

  /** Additive negation. */
  neg(): Tensor {
    return wrap(native.tensorNeg(this._handle));
  }

  /** Absolute value. */
  abs(): Tensor {
    return wrap(native.tensorAbs(this._handle));
  }

  /** Elementwise square. */
  sqr(): Tensor {
    return wrap(native.tensorSqr(this._handle));
  }

  /** Elementwise square root. */
  sqrt(): Tensor {
    return wrap(native.tensorSqrt(this._handle));
  }

  /** Elementwise `e^x`. */
  exp(): Tensor {
    return wrap(native.tensorExp(this._handle));
  }

  /** Natural logarithm. */
  log(): Tensor {
    return wrap(native.tensorLog(this._handle));
  }

  /** Elementwise reciprocal `1/x`. */
  recip(): Tensor {
    return wrap(native.tensorRecip(this._handle));
  }

  /** Elementwise power with a tensor exponent. */
  pow(exponent: Tensor): Tensor {
    return wrap(native.tensorPow(this._handle, exponent._handle));
  }

  /** Elementwise power with a constant exponent. */
  powf(exponent: number): Tensor {
    return wrap(native.tensorPowf(this._handle, exponent));
  }

  /** Clamp every element into `[min, max]`. */
  clamp(min: number, max: number): Tensor {
    return wrap(native.tensorClamp(this._handle, min, max));
  }

  // ---- comparisons (return u8 tensors) ----

  /** Elementwise `self == other`. */
  eq(other: TensorOrScalar): Tensor {
    return wrap(native.tensorEq(this._handle, unwrap(other)));
  }

  /** Elementwise `self != other`. */
  ne(other: TensorOrScalar): Tensor {
    return wrap(native.tensorNe(this._handle, unwrap(other)));
  }

  /** Elementwise `self < other`. */
  lt(other: TensorOrScalar): Tensor {
    return wrap(native.tensorLt(this._handle, unwrap(other)));
  }

  /** Elementwise `self <= other`. */
  le(other: TensorOrScalar): Tensor {
    return wrap(native.tensorLe(this._handle, unwrap(other)));
  }

  /** Elementwise `self > other`. */
  gt(other: TensorOrScalar): Tensor {
    return wrap(native.tensorGt(this._handle, unwrap(other)));
  }

  /** Elementwise `self >= other`. */
  ge(other: TensorOrScalar): Tensor {
    return wrap(native.tensorGe(this._handle, unwrap(other)));
  }

  // ---- activations ----

  /** Rectified linear unit. */
  relu(): Tensor {
    return wrap(native.tensorRelu(this._handle));
  }

  /** Hyperbolic tangent. */
  tanh(): Tensor {
    return wrap(native.tensorTanh(this._handle));
  }

  /** Logistic sigmoid. */
  sigmoid(): Tensor {
    return wrap(native.tensorSigmoid(this._handle));
  }

  /** Gaussian error linear unit (tanh approximation). */
  gelu(): Tensor {
    return wrap(native.tensorGelu(this._handle));
  }

  /** Gaussian error linear unit (exact, erf-based). */
  geluErf(): Tensor {
    return wrap(native.tensorGeluErf(this._handle));
  }

  /** Sigmoid linear unit (SiLU / swish). */
  silu(): Tensor {
    return wrap(native.tensorSilu(this._handle));
  }

  /** Error function, elementwise. */
  erf(): Tensor {
    return wrap(native.tensorErf(this._handle));
  }

  /**
   * SwiGLU: split the last dimension in half and return
   * `silu(firstHalf) * secondHalf`. Used by LLaMA-style MLPs.
   */
  swiGLU(): Tensor {
    return wrap(native.tensorSwiGLU(this._handle));
  }

  /** Hard sigmoid: `clamp(x/6 + 0.5, 0, 1)`. */
  hardSigmoid(): Tensor {
    return wrap(native.tensorHardSigmoid(this._handle));
  }

  /** Hard swish: `x * hardSigmoid(x)`. */
  hardSwish(): Tensor {
    return wrap(native.tensorHardSwish(this._handle));
  }

  /** Squared ReLU: `relu(x)^2`. */
  relu2(): Tensor {
    return wrap(native.tensorRelu2(this._handle));
  }

  /** ReLU capped at 6: `clamp(x, 0, 6)`. */
  relu6(): Tensor {
    return wrap(native.tensorRelu6(this._handle));
  }

  /** Scaled exponential linear unit. */
  selu(alpha = 1.6732632423543772, gamma = 1.0507009873554805): Tensor {
    return wrap(native.tensorSelu(this._handle, alpha, gamma));
  }

  /** Mish activation. */
  mish(): Tensor {
    return wrap(native.tensorMish(this._handle));
  }

  /** Exponential linear unit. */
  elu(alpha = 1.0): Tensor {
    return wrap(native.tensorElu(this._handle, alpha));
  }

  /** Leaky ReLU with the given negative slope. */
  leakyRelu(negativeSlope = 0.01): Tensor {
    return wrap(native.tensorLeakyRelu(this._handle, negativeSlope));
  }

  /** Softmax along `dim`. */
  softmax(dim: number): Tensor {
    return wrap(native.tensorSoftmax(this._handle, dim));
  }

  /** Log-softmax along `dim` (numerically stable, autograd-friendly). */
  logSoftmax(dim: number): Tensor {
    return wrap(native.tensorLogSoftmax(this._handle, dim));
  }

  // ---- matmul ----

  /** Matrix multiplication (batched; both operands must have equal rank). */
  matmul(other: Tensor): Tensor {
    return wrap(native.tensorMatmul(this._handle, other._handle));
  }

  /** Matrix multiplication with broadcasting of batch dimensions. */
  broadcastMatmul(other: Tensor): Tensor {
    return wrap(native.tensorBroadcastMatmul(this._handle, other._handle));
  }

  /** Sum of the elementwise product with `other`, as a JS number. */
  dot(other: Tensor): number {
    return native.dot(this._handle, other._handle);
  }

  // ---- reductions ----

  /** Sum of every element. */
  sumAll(): Tensor {
    return wrap(native.tensorSumAll(this._handle));
  }

  /** Mean of every element. */
  meanAll(): Tensor {
    return wrap(native.tensorMeanAll(this._handle));
  }

  /** Maximum element. */
  maxAll(): Tensor {
    return wrap(native.tensorMaxAll(this._handle));
  }

  /** Minimum element. */
  minAll(): Tensor {
    return wrap(native.tensorMinAll(this._handle));
  }

  /** L2 norm over all elements. */
  norm(): Tensor {
    return wrap(native.tensorNorm(this._handle));
  }

  /** Sum over one or more dimensions. */
  sum(dims?: DimsLike): Tensor {
    return dims === undefined
      ? this.sumAll()
      : wrap(native.tensorSum(this._handle, toShape(dims)));
  }

  /** Mean over one or more dimensions. */
  mean(dims?: DimsLike): Tensor {
    return dims === undefined
      ? this.meanAll()
      : wrap(native.tensorMean(this._handle, toShape(dims)));
  }

  /** Maximum along a single dimension. */
  max(dim?: number): Tensor {
    return dim === undefined
      ? this.maxAll()
      : wrap(native.tensorMax(this._handle, dim));
  }

  /** Minimum along a single dimension. */
  min(dim?: number): Tensor {
    return dim === undefined
      ? this.minAll()
      : wrap(native.tensorMin(this._handle, dim));
  }

  /** Index of the maximum along `dim` (u32 tensor). */
  argmax(dim: number): Tensor {
    return wrap(native.tensorArgmax(this._handle, dim));
  }

  /** Index of the minimum along `dim` (u32 tensor). */
  argmin(dim: number): Tensor {
    return wrap(native.tensorArgmin(this._handle, dim));
  }

  /** Cumulative sum along `dim`. */
  cumsum(dim: number): Tensor {
    return wrap(native.tensorCumsum(this._handle, dim));
  }

  /** Log-sum-exp over one or more dimensions. */
  logSumExp(dims: DimsLike): Tensor {
    return wrap(native.tensorLogSumExp(this._handle, toShape(dims)));
  }

  // ---- shape ----

  /**
   * Reshape to `shape`. Accepts either `reshape(2, 3)` or `reshape([2, 3])`.
   * One dimension may be `-1` to be inferred.
   */
  reshape(...shape: (number | number[])[]): Tensor {
    const dims = shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape;
    return wrap(native.tensorReshape(this._handle, dims));
  }

  /** Flatten every dimension into a 1-D tensor. */
  flattenAll(): Tensor {
    return wrap(native.tensorFlattenAll(this._handle));
  }

  /** Flatten dimensions `start..=end` (defaults to the whole tensor). */
  flatten(start = 0, end = this.rank - 1): Tensor {
    if (start === 0 && end === this.rank - 1) return this.flattenAll();
    return wrap(native.tensorFlatten(this._handle, start, end));
  }

  /** Swap two dimensions. */
  transpose(dim0: number, dim1: number): Tensor {
    return wrap(native.tensorTranspose(this._handle, dim0, dim1));
  }

  /** Transpose a 2-D tensor. */
  t(): Tensor {
    return wrap(native.tensorT(this._handle));
  }

  /** Permute all dimensions. */
  permute(dims: number[]): Tensor {
    return wrap(native.tensorPermute(this._handle, dims));
  }

  /** Remove a size-1 dimension. */
  squeeze(dim: number): Tensor {
    return wrap(native.tensorSqueeze(this._handle, dim));
  }

  /** Insert a size-1 dimension at `dim`. */
  unsqueeze(dim: number): Tensor {
    return wrap(native.tensorUnsqueeze(this._handle, dim));
  }

  /** A view of `len` elements along `dim`, starting at `start`. */
  narrow(dim: number, start: number, len: number): Tensor {
    return wrap(native.tensorNarrow(this._handle, dim, start, len));
  }

  /** Broadcast to `shape` (each existing dim must be 1 or match). */
  broadcastAs(shape: ShapeLike): Tensor {
    return wrap(native.tensorBroadcastAs(this._handle, toShape(shape)));
  }

  /** Alias for {@link Tensor.broadcastAs} using the expand rules. */
  expand(shape: ShapeLike): Tensor {
    return wrap(native.tensorExpand(this._handle, toShape(shape)));
  }

  /** Tile to `shape`; each dimension must be a multiple of the current one. */
  repeat(shape: ShapeLike): Tensor {
    return wrap(native.tensorRepeat(this._handle, toShape(shape)));
  }

  /** Force a contiguous memory layout. */
  contiguous(): Tensor {
    return wrap(native.tensorContiguous(this._handle));
  }

  /** Detach from the autograd graph. */
  detach(): Tensor {
    return wrap(native.tensorDetach(this._handle));
  }

  /** Cast to another dtype (`"f32"`, `"f64"`, `"u8"`, `"u32"`, `"i64"`). */
  to(dtype: string): Tensor {
    return wrap(native.tensorToDtype(this._handle, dtype));
  }

  /** Select a single index along the first dimension. */
  get(index: number): Tensor {
    return wrap(native.tensorGet(this._handle, index));
  }

  /** Select a single index along `dim`. */
  getOnDim(dim: number, index: number): Tensor {
    return wrap(native.tensorGetOnDim(this._handle, dim, index));
  }

  /** Zero-pad `dim` with `left`/`right` elements. */
  padWithZeros(dim: number, left: number, right: number): Tensor {
    return wrap(native.tensorPadWithZeros(this._handle, dim, left, right));
  }

  /** Reverse the given dimensions. */
  flip(dims: number[]): Tensor {
    return wrap(native.tensorFlip(this._handle, dims));
  }

  /** Circular shift along `dim`. */
  roll(shift: number, dim: number): Tensor {
    return wrap(native.tensorRoll(this._handle, shift, dim));
  }

  /** Split into `chunks` equal parts along `dim`. */
  chunk(chunks: number, dim: number): Tensor[] {
    return native.tensorChunk(this._handle, chunks, dim).map((h: unknown) => wrap(h));
  }

  /** Concatenate with `other` along `dim`. */
  cat(other: Tensor, dim: number): Tensor {
    return wrap(native.tensorCat([this._handle, other._handle], dim));
  }

  /** Stack with `other` along a new dimension `dim`. */
  stack(other: Tensor, dim: number): Tensor {
    return wrap(native.tensorStack([this._handle, other._handle], dim));
  }

  // ---- indexing / masking ----

  /** Select rows by index along `dim` from a `Uint32Array` of ids. */
  indexSelect(ids: Uint32Array, dim = 0): Tensor {
    return wrap(
      native.tensorIndexSelect(this._handle, ids, [ids.length], dim)
    );
  }

  /** Gather along `dim` using an index tensor of the same rank. */
  gather(index: Tensor, dim: number): Tensor {
    return wrap(native.tensorGather(this._handle, index._handle, dim));
  }

  /** Replace elements where `mask` is non-zero with `value`. */
  maskedFill(mask: Tensor, value: number): Tensor {
    return wrap(native.tensorMaskedFill(this._handle, mask._handle, value));
  }

  // ---- autograd ----

  /**
   * Backpropagate from this (scalar) tensor and return the gradients of every
   * trainable variable in its graph.
   */
  backward(): GradStore {
    return new GradStore(native.backward(this._handle));
  }
}

/**
 * A trainable tensor (a PyTorch `nn.Parameter` equivalent). Ops applied to a
 * `Var` return ordinary {@link Tensor}s, but the variable stays connected to
 * the autograd graph.
 *
 * @example
 * ```ts
 * const w = candle.variable([3], [1]);
 * const g = w.mul(w).backward().get(w); // 2 * 3 = 6
 * ```
 */
export class Var extends Tensor {
  /** A detached copy of this variable's current values. */
  get data(): Tensor {
    return new Tensor(native.tensorDetach(this._handle));
  }
}

/**
 * Gradients produced by {@link Tensor.backward}, keyed by the variables they
 * belong to.
 */
export class GradStore {
  /** @internal Native handle. */
  readonly _handle: unknown;

  /** @internal */
  constructor(handle: unknown) {
    this._handle = handle;
  }

  /**
   * The gradient for `param`.
   * @throws If `param` was not part of the graph that produced this store.
   */
  get(param: Var): Tensor {
    return new Tensor(native.gradOf(this._handle, param._handle));
  }
}
