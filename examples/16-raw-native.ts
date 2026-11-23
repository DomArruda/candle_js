// Use the raw Neon exports directly, without the friendly lib/ wrapper.
const native = require("../index.node");

const a = native.tensorFromF32(new Float32Array([1, 2, 3, 4]), [2, 2]);
const b = native.tensorFromF32(new Float32Array([5, 6, 7, 8]), [2, 2]);
const c = native.tensorMatmul(a, b);
const { shape, data } = native.tensorToF32(c);
console.log("native matmul:", shape, Array.from(new Float32Array(data)));

const x = native.varFromF32(new Float32Array([3]), [1]);
const y = native.tensorMul(x, x);
const g = native.gradOf(native.backward(y), x);
console.log("native grad of x^2 at x=3:", native.tensorToScalar(g));

// One SGD step on w, minimizing w^2.
const w = native.varFromF32(new Float32Array([3]), [1]);
const grads = native.backward(native.tensorMul(w, w));
native.sgdStep(grads, w, 0.1);
console.log("after sgd step:", native.tensorToScalar(w));
