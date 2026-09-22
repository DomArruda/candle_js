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

const VOCAB = 8;
const SEQ = 6;
const BATCH = 2;
const D_MODEL = 16;
const HEADS = 2;
const HEAD_DIM = D_MODEL / HEADS;
const FFN = 4 * D_MODEL;

let seed = 12345;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const base = Array.from({ length: SEQ + 1 }, () => Math.floor(rnd() * VOCAB));
const X = new Uint32Array([...base.slice(0, SEQ), ...base.slice(0, SEQ)]);
const Y = new Uint32Array([...base.slice(1), ...base.slice(1)]);

const POS_IDS = new Uint32Array(Array.from({ length: SEQ }, (_, i) => i));

// 1 marks the positions to fill (the strict upper triangle).
const causalMask = (n: number) =>
  candle.tensorSub(candle.tensorOnes([n, n]), candle.tensorTril(n));

function buildTransformer() {
  const vm = candle.varmapNew();

  const tok = candle.embeddingNew(vm, "tok", VOCAB, D_MODEL);
  const pos = candle.embeddingNew(vm, "pos", SEQ, D_MODEL);

  const ln1 = candle.layerNormNew(vm, "ln1", D_MODEL, 1e-5, true);
  const ln2 = candle.layerNormNew(vm, "ln2", D_MODEL, 1e-5, true);
  const lnf = candle.layerNormNew(vm, "lnf", D_MODEL, 1e-5, true);

  const q = candle.linearNew(vm, "q", D_MODEL, D_MODEL, true);
  const k = candle.linearNew(vm, "k", D_MODEL, D_MODEL, true);
  const v = candle.linearNew(vm, "v", D_MODEL, D_MODEL, true);
  const proj = candle.linearNew(vm, "proj", D_MODEL, D_MODEL, true);

  const fc1 = candle.linearNew(vm, "fc1", D_MODEL, FFN, true);
  const fc2 = candle.linearNew(vm, "fc2", FFN, D_MODEL, true);
  const head = candle.linearNew(vm, "head", D_MODEL, VOCAB, true);

  const mask = causalMask(SEQ);

  const splitHeads = (t: any) =>
    candle.tensorPermute(candle.tensorReshape(t, [BATCH, SEQ, HEADS, HEAD_DIM]), [
      0, 2, 1, 3,
    ]);

  const forward = (ids: Uint32Array) => {
    let h = candle.tensorAdd(
      candle.embeddingForward(tok, ids, [BATCH, SEQ]),
      candle.embeddingForward(pos, POS_IDS, [SEQ])
    );

    const n = candle.layerNormForward(ln1, h);
    const qh = splitHeads(candle.linearForward(q, n));
    const kh = splitHeads(candle.linearForward(k, n));
    const vh = splitHeads(candle.linearForward(v, n));

    const scores = candle.tensorAffine(
      candle.tensorMatmul(qh, candle.tensorTranspose(kh, 2, 3)),
      1 / Math.sqrt(HEAD_DIM),
      0
    );
    const attn = candle.tensorSoftmax(
      candle.tensorMaskedFill(scores, mask, -1e9),
      3
    );
    const ctx = candle.tensorMatmul(attn, vh);
    const merged = candle.tensorReshape(
      candle.tensorPermute(ctx, [0, 2, 1, 3]),
      [BATCH, SEQ, D_MODEL]
    );
    h = candle.tensorAdd(h, candle.linearForward(proj, merged));

    const n2 = candle.layerNormForward(ln2, h);
    const f = candle.linearForward(
      fc2,
      candle.tensorGelu(candle.linearForward(fc1, n2))
    );
    h = candle.tensorAdd(h, f);

    const logits = candle.linearForward(head, candle.layerNormForward(lnf, h));
    return candle.tensorReshape(logits, [BATCH * SEQ, VOCAB]);
  };

  return { vm, forward };
}

describe("attention primitives", () => {
  test("causal mask makes softmax lower triangular", () => {
    const n = 4;
    const scores = candle.tensorZeros([1, 1, n, n]);
    const attn = toArr(
      candle.tensorSoftmax(
        candle.tensorMaskedFill(scores, causalMask(n), -1e9),
        3
      )
    ).values;

    expect(attn.slice(0, 4)).toEqual([1, 0, 0, 0]);
    const row2 = attn.slice(8, 12);
    expect(row2[0]).toBeCloseTo(1 / 3, 5);
    expect(row2[1]).toBeCloseTo(1 / 3, 5);
    expect(row2[2]).toBeCloseTo(1 / 3, 5);
    expect(row2[3]).toBe(0);
  });

  test("multi-head attention forward keeps shape", () => {
    const x = candle.tensorRandn(0, 1, [BATCH, SEQ, D_MODEL]);
    const qh = candle.tensorPermute(
      candle.tensorReshape(x, [BATCH, SEQ, HEADS, HEAD_DIM]),
      [0, 2, 1, 3]
    );
    const scores = candle.tensorMatmul(qh, candle.tensorTranspose(qh, 2, 3));
    expect(candle.tensorShape(scores)).toEqual([BATCH, HEADS, SEQ, SEQ]);
    const ctx = candle.tensorMatmul(candle.tensorSoftmax(scores, 3), qh);
    expect(candle.tensorShape(ctx)).toEqual([BATCH, HEADS, SEQ, HEAD_DIM]);
  });
});

describe("transformer", () => {
  test("forward produces [batch*seq, vocab] logits", () => {
    const { forward } = buildTransformer();
    expect(candle.tensorShape(forward(X))).toEqual([BATCH * SEQ, VOCAB]);
  });

  test("a one-block GPT overfits a sequence", () => {
    const { vm, forward } = buildTransformer();
    const opt = candle.adamwNew(vm, 0.01, 0.9, 0.999, 1e-8, 0.0);

    const first = candle.tensorToScalar(candle.crossEntropyLoss(forward(X), Y));
    let loss = first;
    for (let i = 0; i < 500; i++) {
      const l = candle.crossEntropyLoss(forward(X), Y);
      loss = candle.tensorToScalar(l);
      candle.adamwStep(opt, candle.backward(l));
    }

    console.log(`transformer loss: ${first.toFixed(3)} -> ${loss.toFixed(4)}`);

    expect(loss).toBeLessThan(0.05);
    expect(loss).toBeLessThan(first);

    const preds = toU32(candle.tensorArgmax(forward(X), 1)).values;
    expect(preds).toEqual(Array.from(Y));
  }, 30000);

  test("transformer round-trips through safetensors", () => {
    const path = "/tmp/candle_transformer.safetensors";
    const a = buildTransformer();
    candle.varmapSave(a.vm, path);

    const b = buildTransformer();
    candle.varmapLoad(b.vm, path);

    expect(toArr(b.forward(X)).values).toEqual(toArr(a.forward(X)).values);
  });
});
