import { Tensor } from "./tensor";

/** An operand in an {@link expr} template: a tensor or a scalar. */
export type ExprOperand = Tensor | number;

type Token =
  | { kind: "operand"; value: ExprOperand }
  | { kind: "op"; op: string }
  | { kind: "lparen" }
  | { kind: "rparen" };

const PRECEDENCE: Record<string, number> = {
  "**": 4,
  "^": 4,
  "@": 3,
  "*": 3,
  "/": 3,
  "+": 2,
  "-": 2,
};

const RIGHT_ASSOC = new Set(["**", "^"]);

/** @internal Scan the literal segments of a template for operators/parens. */
function scanOperators(segment: string, tokens: Token[]): void {
  let j = 0;
  while (j < segment.length) {
    const ch = segment[j];
    if (/\s/.test(ch)) {
      j++;
      continue;
    }
    if (ch === "(") {
      tokens.push({ kind: "lparen" });
      j++;
      continue;
    }
    if (ch === ")") {
      tokens.push({ kind: "rparen" });
      j++;
      continue;
    }
    if (segment.startsWith("**", j)) {
      tokens.push({ kind: "op", op: "**" });
      j += 2;
      continue;
    }
    if (/[0-9.]/.test(ch)) {
      let k = j;
      while (k < segment.length && /[0-9.]/.test(segment[k])) k++;
      const value = Number(segment.slice(j, k));
      if (Number.isNaN(value)) {
        throw new Error(`expr: invalid number '${segment.slice(j, k)}'`);
      }
      tokens.push({ kind: "operand", value });
      j = k;
      continue;
    }
    if ("@*/+-^".includes(ch)) {
      tokens.push({ kind: "op", op: ch });
      j++;
      continue;
    }
    throw new Error(`expr: unexpected character '${ch}'`);
  }
}

/** @internal Interleave operands and operators into a flat token list. */
function tokenize(strings: readonly string[], values: ExprOperand[]): Token[] {
  const tokens: Token[] = [];
  for (let i = 0; i < strings.length; i++) {
    scanOperators(strings[i], tokens);
    if (i < values.length) tokens.push({ kind: "operand", value: values[i] });
  }
  return tokens;
}

/** @internal Shunting-yard: infix tokens to reverse Polish notation. */
function toRpn(tokens: Token[]): Token[] {
  const out: Token[] = [];
  const ops: Token[] = [];
  for (const token of tokens) {
    if (token.kind === "operand") {
      out.push(token);
    } else if (token.kind === "op") {
      while (ops.length) {
        const top = ops[ops.length - 1];
        if (top.kind !== "op") break;
        const topPrec = PRECEDENCE[top.op];
        const curPrec = PRECEDENCE[token.op];
        if (topPrec > curPrec || (topPrec === curPrec && !RIGHT_ASSOC.has(token.op))) {
          out.push(ops.pop()!);
        } else {
          break;
        }
      }
      ops.push(token);
    } else if (token.kind === "lparen") {
      ops.push(token);
    } else {
      while (ops.length && ops[ops.length - 1].kind !== "lparen") {
        out.push(ops.pop()!);
      }
      if (!ops.length) throw new Error("expr: unbalanced parentheses");
      ops.pop();
    }
  }
  while (ops.length) {
    const top = ops.pop()!;
    if (top.kind === "lparen") throw new Error("expr: unbalanced parentheses");
    out.push(top);
  }
  return out;
}

/** @internal Require a tensor (used where scalars make no sense). */
function asTensor(x: ExprOperand): Tensor {
  if (x instanceof Tensor) return x;
  throw new Error("expr: expected a tensor operand");
}

/** @internal Apply one binary operator to two operands. */
function applyOp(op: string, a: ExprOperand, b: ExprOperand): ExprOperand {
  switch (op) {
    case "+":
      return a instanceof Tensor ? a.add(b) : asTensor(b).add(a);
    case "*":
      return a instanceof Tensor ? a.mul(b) : asTensor(b).mul(a);
    case "-":
      return a instanceof Tensor ? a.sub(b) : asTensor(b).neg().add(a);
    case "/":
      return a instanceof Tensor ? a.div(b) : asTensor(b).recip().mul(a);
    case "@":
      return asTensor(a).matmul(asTensor(b));
    case "^":
    case "**":
      return b instanceof Tensor ? asTensor(a).pow(b) : asTensor(a).powf(b);
    default:
      throw new Error(`expr: unknown operator '${op}'`);
  }
}

/**
 * A tagged-template expression builder, the closest thing to operator
 * overloading in JavaScript. It supports `+`, `-`, `*`, `/`, `@` (matmul),
 * `^` / `**` (power) and parentheses, with the usual precedence.
 *
 * This is opt-in sugar on top of the method API — the underlying language has
 * no way to overload `+` or `@` on objects, so a plain `a + b` will always
 * stringify instead. Use methods (`.add`, `.matmul`) when you prefer clarity.
 *
 * @example
 * ```ts
 * import { expr } from "candle_js";
 *
 * const z = expr`${a} @ ${b} + ${c}`;          // a.matmul(b).add(c)
 * const y = expr`(${a} + ${b}) @ ${c}`;        // a.add(b).matmul(c)
 * const p = expr`2 * ${x} ^ 3`;                // x.powf(3).mul(2)
 * ```
 */
export function expr(
  strings: TemplateStringsArray,
  ...values: ExprOperand[]
): Tensor {
  const rpn = toRpn(tokenize(strings, values));
  const stack: ExprOperand[] = [];
  for (const token of rpn) {
    if (token.kind === "operand") {
      stack.push(token.value);
    } else if (token.kind === "op") {
      const b = stack.pop();
      const a = stack.pop();
      if (a === undefined || b === undefined) {
        throw new Error("expr: malformed expression");
      }
      stack.push(applyOp(token.op, a, b));
    }
  }
  if (stack.length !== 1 || !(stack[0] instanceof Tensor)) {
    throw new Error("expr: expression did not evaluate to a tensor");
  }
  return stack[0];
}
