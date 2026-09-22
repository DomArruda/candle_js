import { expect, test, describe } from "bun:test";
const candle = require("../index.node");

const toArr = (t: any) => {
  const { shape, data } = candle.tensorToF32(t);
  return { shape, values: Array.from(new Float32Array(data)) };
};

const toU32 = (t: any) => {
  const { shape, data } = candle.tensorToU32(t);
  return { shape, values: Array.from(new Uint32Array(data)) };
};

const t = (values: number[], shape: number[]) =>
  candle.tensorFromF32(new Float32Array(values), shape);

const u32 = (values: number[], shape: number[]) =>
  candle.tensorFromU32(new Uint32Array(values), shape);

describe("construction", () => {
  test("zeros / ones / full", () => {
    expect(toArr(candle.tensorZeros([2, 2]))).toEqual({
      shape: [2, 2],
      values: [0, 0, 0, 0],
    });
    expect(toArr(candle.tensorOnes([3])).values).toEqual([1, 1, 1]);
    expect(toArr(candle.tensorFull([2], 7)).values).toEqual([7, 7]);
  });

  test("arange", () => {
    expect(toArr(candle.tensorArange(0, 5)).values).toEqual([0, 1, 2, 3, 4]);
    expect(toArr(candle.tensorArange(0, 1, 0.25)).values).toEqual([
      0, 0.25, 0.5, 0.75,
    ]);
  });

  test("eye / tril / triu", () => {
    expect(toArr(candle.tensorEye(3)).values).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    expect(toArr(candle.tensorTril(3)).values).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1]);
    expect(toArr(candle.tensorTriu(3)).values).toEqual([1, 1, 1, 0, 1, 1, 0, 0, 1]);
  });

  test("randn / rand have the right shape and range", () => {
    expect(candle.tensorShape(candle.tensorRandn(0, 1, [4, 5]))).toEqual([4, 5]);
    const r = toArr(candle.tensorRand(0, 1, [100])).values;
    expect(Math.min(...r)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...r)).toBeLessThanOrEqual(1);
  });
});

describe("elementwise", () => {
  test("arithmetic and scalar operands", () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    expect(toArr(candle.tensorAdd(a, 10)).values).toEqual([11, 12, 13, 14]);
    expect(toArr(candle.tensorSub(a, 1)).values).toEqual([0, 1, 2, 3]);
    expect(toArr(candle.tensorMul(a, 2)).values).toEqual([2, 4, 6, 8]);
    expect(toArr(candle.tensorDiv(a, 2)).values).toEqual([0.5, 1, 1.5, 2]);
  });

  test("broadcasting", () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = t([10, 20, 30], [3]);
    expect(toArr(candle.tensorAdd(a, b)).values).toEqual([11, 22, 33, 14, 25, 36]);
  });

  test("unary math", () => {
    expect(toArr(candle.tensorNeg(t([1, -2], [2]))).values).toEqual([-1, 2]);
    expect(toArr(candle.tensorAbs(t([1, -2], [2]))).values).toEqual([1, 2]);
    expect(toArr(candle.tensorSqr(t([2, 3], [2]))).values).toEqual([4, 9]);
    expect(toArr(candle.tensorSqrt(t([4, 9], [2]))).values).toEqual([2, 3]);
    expect(toArr(candle.tensorExp(t([0], [1]))).values[0]).toBeCloseTo(1, 5);
    expect(toArr(candle.tensorLog(t([1], [1]))).values[0]).toBeCloseTo(0, 5);
    expect(toArr(candle.tensorRecip(t([2], [1]))).values[0]).toBeCloseTo(0.5, 5);
    expect(toArr(candle.tensorPowf(t([2], [1]), 3)).values[0]).toBeCloseTo(8, 5);
    expect(toArr(candle.tensorClamp(t([-5, 0, 5], [3]), 0, 1)).values).toEqual([
      0, 0, 1,
    ]);
  });

  test("maximum / minimum / comparisons", () => {
    const a = t([1, 5], [2]);
    const b = t([3, 2], [2]);
    expect(toArr(candle.tensorMaximum(a, b)).values).toEqual([3, 5]);
    expect(toArr(candle.tensorMinimum(a, b)).values).toEqual([1, 2]);
    const gt = candle.tensorToDtype(candle.tensorGt(a, 2), "f32");
    expect(toArr(gt).values).toEqual([0, 1]);
  });

  test("where / maskedFill", () => {
    const cond = candle.tensorToDtype(candle.tensorGt(t([1, 5], [2]), 2), "u8");
    const out = candle.tensorWhere(cond, t([10, 10], [2]), t([0, 0], [2]));
    expect(toArr(out).values).toEqual([0, 10]);

    const m = t([1, 0, 0, 1], [2, 2]);
    const filled = candle.tensorMaskedFill(t([1, 2, 3, 4], [2, 2]), m, -1);
    expect(toArr(filled).values).toEqual([-1, 2, 3, -1]);
  });

  test("activations", () => {
    expect(toArr(candle.tensorRelu(t([-1, 2], [2]))).values).toEqual([0, 2]);
    expect(toArr(candle.tensorSigmoid(t([0], [1]))).values[0]).toBeCloseTo(0.5, 5);
    expect(toArr(candle.tensorSilu(t([0], [1]))).values[0]).toBeCloseTo(0, 5);
    expect(toArr(candle.tensorGelu(t([0], [1]))).values[0]).toBeCloseTo(0, 5);
    expect(toArr(candle.tensorTanh(t([0], [1]))).values[0]).toBeCloseTo(0, 5);
  });

  test("softmax sums to one", () => {
    const s = candle.tensorSoftmax(t([1, 2, 3], [1, 3]), 1);
    expect(toArr(s).values.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    const ls = candle.tensorLogSoftmax(t([1, 2, 3], [1, 3]), 1);
    expect(Math.exp(toArr(ls).values[2])).toBeCloseTo(
      toArr(s).values[2],
      5
    );
  });
});

describe("reductions", () => {
  const a = t([1, 2, 3, 4, 5, 6], [2, 3]);

  test("all", () => {
    expect(candle.tensorToScalar(candle.tensorSumAll(a))).toBeCloseTo(21, 5);
    expect(candle.tensorToScalar(candle.tensorMeanAll(a))).toBeCloseTo(3.5, 5);
    expect(candle.tensorToScalar(candle.tensorMaxAll(a))).toBeCloseTo(6, 5);
    expect(candle.tensorToScalar(candle.tensorMinAll(a))).toBeCloseTo(1, 5);
  });

  test("sum / mean over dims", () => {
    expect(toArr(candle.tensorSum(a, [0])).values).toEqual([5, 7, 9]);
    expect(toArr(candle.tensorMean(a, 1)).values).toEqual([2, 5]);
    expect(toArr(candle.tensorSum(a, [0, 1])).values).toEqual([21]);
    expect(candle.tensorShape(candle.tensorSumKeepdim(a, 1))).toEqual([2, 1]);
  });

  test("max / argmax over a dim", () => {
    expect(toArr(candle.tensorMax(a, 1)).values).toEqual([3, 6]);
    expect(toU32(candle.tensorArgmax(a, 1)).values).toEqual([2, 2]);
    expect(toU32(candle.tensorArgmin(a, 0)).values).toEqual([0, 0, 0]);
    expect(candle.tensorShape(candle.tensorMaxKeepdim(a, 1))).toEqual([2, 1]);
  });

  test("cumsum / logSumExp", () => {
    expect(toArr(candle.tensorCumsum(t([1, 2, 3], [3]), 0)).values).toEqual([1, 3, 6]);
    expect(candle.tensorToScalar(candle.tensorLogSumExp(t([0, 0], [2]), [0])))
      .toBeCloseTo(Math.log(2), 5);
  });
});

describe("shape ops", () => {
  const a = t([1, 2, 3, 4, 5, 6], [2, 3]);

  test("reshape / flatten / transpose / permute", () => {
    expect(candle.tensorShape(candle.tensorReshape(a, [3, 2]))).toEqual([3, 2]);
    expect(candle.tensorShape(candle.tensorFlattenAll(a))).toEqual([6]);
    expect(toArr(candle.tensorT(t([1, 2, 3, 4], [2, 2]))).values).toEqual([
      1, 3, 2, 4,
    ]);
    expect(candle.tensorShape(candle.tensorTranspose(a, 0, 1))).toEqual([3, 2]);
    const b = t([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
    expect(candle.tensorShape(candle.tensorPermute(b, [2, 0, 1]))).toEqual([2, 2, 2]);
  });

  test("squeeze / unsqueeze / narrow", () => {
    const x = t([1, 2, 3], [1, 3]);
    expect(candle.tensorShape(candle.tensorSqueeze(x, 0))).toEqual([3]);
    expect(candle.tensorShape(candle.tensorUnsqueeze(x, 0))).toEqual([1, 1, 3]);
    expect(toArr(candle.tensorNarrow(a, 1, 1, 2)).values).toEqual([2, 3, 5, 6]);
  });

  test("cat / stack / chunk", () => {
    const x = t([1, 2], [2]);
    const y = t([3, 4], [2]);
    expect(toArr(candle.tensorCat([x, y], 0)).values).toEqual([1, 2, 3, 4]);
    expect(candle.tensorShape(candle.tensorStack([x, y], 0))).toEqual([2, 2]);
    const parts = candle.tensorChunk(t([1, 2, 3, 4], [4]), 2, 0);
    expect(parts.length).toBe(2);
    expect(toArr(parts[0]).values).toEqual([1, 2]);
    expect(toArr(parts[1]).values).toEqual([3, 4]);
  });

  test("broadcastAs / repeat / flip / roll / pad", () => {
    const x = t([1, 2], [1, 2]);
    expect(candle.tensorShape(candle.tensorBroadcastAs(x, [3, 2]))).toEqual([3, 2]);
    expect(toArr(candle.tensorRepeat(x, [2, 2])).values).toEqual([
      1, 2, 1, 2, 1, 2, 1, 2,
    ]);
    expect(toArr(candle.tensorFlip(t([1, 2, 3], [3]), [0])).values).toEqual([3, 2, 1]);
    expect(toArr(candle.tensorRoll(t([1, 2, 3], [3]), 1, 0)).values).toEqual([3, 1, 2]);
    expect(toArr(candle.tensorPadWithZeros(t([1, 2], [2]), 0, 1, 1)).values).toEqual([
      0, 1, 2, 0,
    ]);
  });
});

describe("indexing", () => {
  test("indexSelect / gather", () => {
    const a = t([10, 20, 30, 40], [4, 1]);
    expect(toArr(candle.tensorIndexSelect(a, new Uint32Array([3, 0]), [2], 0)).values)
      .toEqual([40, 10]);

    const src = t([1, 2, 3, 4], [2, 2]);
    const idx = u32([1, 0, 0, 1], [2, 2]);
    expect(toArr(candle.tensorGather(src, idx, 1)).values).toEqual([2, 1, 3, 4]);
  });

  test("embedding from a raw weight tensor", () => {
    const w = t([1, 2, 3, 4, 5, 6], [3, 2]);
    const out = candle.tensorEmbedding(w, new Uint32Array([2, 0]), [2]);
    expect(toArr(out)).toEqual({ shape: [2, 2], values: [5, 6, 1, 2] });
  });
});

describe("readback", () => {
  test("dtype / rank / elemCount", () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    expect(candle.tensorDtype(a)).toBe("f32");
    expect(candle.tensorRank(a)).toBe(2);
    expect(candle.tensorElemCount(a)).toBe(4);
    expect(candle.tensorDtype(candle.tensorFromU32(new Uint32Array([1]), [1])))
      .toBe("u32");
  });
});

describe("autograd through new ops", () => {
  test("grad of a / b wrt a", () => {
    const a = candle.varFromF32(new Float32Array([4, 8]), [2]);
    const b = t([2, 4], [2]);
    const y = candle.tensorSumAll(candle.tensorDiv(a, b));
    const g = candle.gradOf(candle.backward(y), a);
    expect(toArr(g).values).toEqual([0.5, 0.25]);
  });

  test("grad of sum over dim", () => {
    const w = candle.varFromF32(new Float32Array([1, 1, 1, 1]), [2, 2]);
    const loss = candle.tensorSumAll(candle.tensorSum(w, [1]));
    const g = candle.gradOf(candle.backward(loss), w);
    expect(toArr(g).values).toEqual([1, 1, 1, 1]);
  });

  test("grad through softmax is finite", () => {
    const x = candle.varFromF32(new Float32Array([1, 2, 3]), [1, 3]);
    const loss = candle.tensorSumAll(candle.tensorSoftmax(x, 1));
    const g = candle.gradOf(candle.backward(loss), x);
    expect(toArr(g).values.every((v) => Number.isFinite(v))).toBe(true);
  });
});

describe("layers", () => {
  test("layerNorm normalizes to mean 0 var 1", () => {
    const vm = candle.varmapNew();
    const ln = candle.layerNormNew(vm, "ln", 4, 1e-5, true);
    const y = toArr(candle.layerNormForward(ln, t([1, 2, 3, 4], [1, 4]))).values;
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    const variance = y.reduce((a, b) => a + (b - mean) ** 2, 0) / y.length;
    expect(mean).toBeCloseTo(0, 4);
    expect(variance).toBeCloseTo(1, 2);
  });

  test("rmsNorm divides by the rms", () => {
    const vm = candle.varmapNew();
    const rn = candle.rmsNormNew(vm, "rn", 2, 0);
    const y = toArr(candle.rmsNormForward(rn, t([3, 4], [1, 2]))).values;
    const rms = Math.sqrt((9 + 16) / 2);
    expect(y[0]).toBeCloseTo(3 / rms, 4);
    expect(y[1]).toBeCloseTo(4 / rms, 4);
  });

  test("dropout is identity in eval mode", () => {
    const d = candle.dropoutNew(0.5);
    const x = t([1, 2, 3, 4], [4]);
    expect(toArr(candle.dropoutForward(d, x, false)).values).toEqual([1, 2, 3, 4]);
    expect(() => candle.dropoutForward(d, x, true)).not.toThrow();
  });

  test("layerNorm is trainable", () => {
    const vm = candle.varmapNew();
    const ln = candle.layerNormNew(vm, "ln", 3, 1e-5, true);
    const x = t([1, 2, 3], [1, 3]);
    const before = toArr(candle.layerNormForward(ln, x)).values;
    const loss = candle.tensorSumAll(candle.tensorMul(
      candle.layerNormForward(ln, x),
      candle.layerNormForward(ln, x)
    ));
    candle.varmapSgdStep(vm, candle.backward(loss), 0.1);
    expect(toArr(candle.layerNormForward(ln, x)).values).not.toEqual(before);
  });
});

describe("losses", () => {
  test("nll matches crossEntropy on log-softmax inputs", () => {
    const logits = t([2, 1, 0.5, 0.1, 3, 0.2], [2, 3]);
    const logprobs = candle.tensorLogSoftmax(logits, 1);
    const ce = candle.tensorToScalar(candle.crossEntropyLoss(logits, new Uint32Array([0, 1])));
    const nll = candle.tensorToScalar(candle.nllLoss(logprobs, new Uint32Array([0, 1])));
    expect(nll).toBeCloseTo(ce, 5);
  });

  test("bceWithLogit and huber", () => {
    const pred = t([0, 2], [2]);
    const target = t([0, 1], [2]);
    expect(candle.tensorToScalar(candle.bceWithLogitLoss(pred, target)))
      .toBeGreaterThan(0);
    expect(candle.tensorToScalar(candle.huberLoss(t([0], [1]), t([1], [1]), 1)))
      .toBeCloseTo(0.5, 5);
  });
});

describe("files", () => {
  test("safetensorsInspect reports names, shapes and dtypes", () => {
    const path = "/tmp/candle_inspect.safetensors";
    const vm = candle.varmapNew();
    candle.linearNew(vm, "fc", 4, 3, true);
    candle.varmapSave(vm, path);

    const info = candle.safetensorsInspect(path);
    expect(info["fc.weight"].shape).toEqual([3, 4]);
    expect(info["fc.bias"].shape).toEqual([3]);
    expect(info["fc.weight"].dtype).toBe("f32");
  });
});
