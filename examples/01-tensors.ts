import candle from "../lib/index";

const a = candle.tensor([
  [1, 2],
  [3, 4],
]);
const b = candle.ones([2, 2]);

console.log("a:", a.tolist());
console.log("a @ b:", a.matmul(b).tolist());
console.log("a + 10, relu, row sums:", a.add(10).relu().sum(1).toArray());
console.log("softmax row sums:", a.softmax(1).sum(1).toArray());
console.log("shape:", a.shape, "dtype:", a.dtype, "numel:", a.numel);

const stats = candle.randn([1000], { mean: 0, std: 1 });
console.log(
  `randn[1000] mean=${stats.meanAll().item().toFixed(3)} ` +
    `std~=${stats.sub(stats.meanAll()).sqr().meanAll().sqrt().item().toFixed(3)}`
);
