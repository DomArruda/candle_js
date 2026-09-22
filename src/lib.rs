use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{
    ops, AdamW, Dropout, Embedding, LSTMConfig, LayerNorm, LayerNormConfig, Linear, Module,
    Optimizer, ParamsAdamW, RmsNorm, VarBuilder, VarMap, LSTM, RNN, SGD,
};
use neon::prelude::*;
use neon::types::buffer::TypedArray;
use std::cell::RefCell;

pub struct JsTensor(pub Tensor);
impl Finalize for JsTensor {}

pub struct JsVar(pub Var);
impl Finalize for JsVar {}

pub struct JsGrads(pub RefCell<GradStore>);
impl Finalize for JsGrads {}

pub struct JsVarMap(pub RefCell<VarMap>);
impl Finalize for JsVarMap {}

pub struct JsLinear(pub Linear);
impl Finalize for JsLinear {}

pub struct JsEmbedding(pub Embedding);
impl Finalize for JsEmbedding {}

pub struct JsLayerNorm(pub LayerNorm);
impl Finalize for JsLayerNorm {}

pub struct JsRmsNorm(pub RmsNorm);
impl Finalize for JsRmsNorm {}

pub struct JsDropout(pub Dropout);
impl Finalize for JsDropout {}

pub struct JsLstm(pub LSTM);
impl Finalize for JsLstm {}

pub struct JsAdamW(pub RefCell<AdamW>);
impl Finalize for JsAdamW {}

pub struct JsSgd(pub RefCell<SGD>);
impl Finalize for JsSgd {}

// ---- helpers ----

enum TOrS {
    T(Tensor),
    S(f64),
}

fn read_shape(cx: &mut FunctionContext, idx: usize) -> NeonResult<Vec<usize>> {
    let arr = cx.argument::<JsArray>(idx)?;
    let len = arr.len(cx);
    let mut shape = Vec::with_capacity(len as usize);
    for i in 0..len {
        let val: Handle<JsValue> = arr.get(cx, i)?;
        let n = val.downcast::<JsNumber, _>(cx).or_throw(cx)?;
        shape.push(n.value(cx) as usize);
    }
    Ok(shape)
}

/// Like `read_shape` but accepts a bare number as a single dimension.
fn read_dims(cx: &mut FunctionContext, idx: usize) -> NeonResult<Vec<usize>> {
    let v: Handle<JsValue> = cx.argument(idx)?;
    if let Ok(n) = v.downcast::<JsNumber, _>(cx) {
        return Ok(vec![n.value(cx) as usize]);
    }
    read_shape(cx, idx)
}

/// Accepts either a JsBox<JsTensor> or a JsBox<JsVar> at `idx`.
fn arg_tensor(cx: &mut FunctionContext, idx: usize) -> NeonResult<Tensor> {
    let v: Handle<JsValue> = cx.argument(idx)?;
    if let Ok(t) = v.downcast::<JsBox<JsTensor>, _>(cx) {
        return Ok(t.0.clone());
    }
    if let Ok(t) = v.downcast::<JsBox<JsVar>, _>(cx) {
        return Ok(t.0.as_tensor().clone());
    }
    cx.throw_error("expected a Tensor or Var")
}

fn arg_tensor_or_scalar(cx: &mut FunctionContext, idx: usize) -> NeonResult<TOrS> {
    let v: Handle<JsValue> = cx.argument(idx)?;
    if let Ok(n) = v.downcast::<JsNumber, _>(cx) {
        return Ok(TOrS::S(n.value(cx)));
    }
    Ok(TOrS::T(arg_tensor(cx, idx)?))
}

fn arg_tensor_list(cx: &mut FunctionContext, idx: usize) -> NeonResult<Vec<Tensor>> {
    let arr = cx.argument::<JsArray>(idx)?;
    let len = arr.len(cx);
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let v: Handle<JsValue> = arr.get(cx, i)?;
        if let Ok(t) = v.downcast::<JsBox<JsTensor>, _>(cx) {
            out.push(t.0.clone());
        } else if let Ok(t) = v.downcast::<JsBox<JsVar>, _>(cx) {
            out.push(t.0.as_tensor().clone());
        } else {
            return cx.throw_error("expected an array of Tensor or Var");
        }
    }
    Ok(out)
}

fn arg_var_list(cx: &mut FunctionContext, idx: usize) -> NeonResult<Vec<Var>> {
    let arr = cx.argument::<JsArray>(idx)?;
    let len = arr.len(cx);
    let mut out = Vec::with_capacity(len as usize);
    for i in 0..len {
        let v: Handle<JsValue> = arr.get(cx, i)?;
        let var = v.downcast::<JsBox<JsVar>, _>(cx).or_throw(cx)?;
        out.push(var.0.clone());
    }
    Ok(out)
}

// ---- macros ----

macro_rules! unary_op {
    ($fn_name:ident, $method:ident) => {
        fn $fn_name(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
            let a = arg_tensor(&mut cx, 0)?;
            let c = a.$method().or_else(|e| cx.throw_error(e.to_string()))?;
            Ok(cx.boxed(JsTensor(c)))
        }
    };
}

macro_rules! unary_fn_op {
    ($fn_name:ident, $func:path) => {
        fn $fn_name(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
            let a = arg_tensor(&mut cx, 0)?;
            let c = $func(&a).or_else(|e| cx.throw_error(e.to_string()))?;
            Ok(cx.boxed(JsTensor(c)))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $method:ident) => {
        fn $fn_name(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
            let a = arg_tensor(&mut cx, 0)?;
            let b = arg_tensor_or_scalar(&mut cx, 1)?;
            let rhs = match b {
                TOrS::T(t) => t,
                TOrS::S(v) => Tensor::full(v as f32, a.shape().clone(), a.device())
                    .or_else(|e| cx.throw_error(e.to_string()))?,
            };
            let c = a.$method(&rhs).or_else(|e| cx.throw_error(e.to_string()))?;
            Ok(cx.boxed(JsTensor(c)))
        }
    };
}

// ---- basics ----

fn hello(mut cx: FunctionContext) -> JsResult<JsString> {
    Ok(cx.string("hello from candle"))
}

fn add(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let a = cx.argument::<JsNumber>(0)?.value(&mut cx);
    let b = cx.argument::<JsNumber>(1)?.value(&mut cx);
    Ok(cx.number(a + b))
}

// ---- construction ----

fn tensor_from_f32(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let view = cx.argument::<JsTypedArray<f32>>(0)?;
    let data: Vec<f32> = view.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 1)?;

    let t = Tensor::from_slice(&data, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_from_u32(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let view = cx.argument::<JsTypedArray<u32>>(0)?;
    let data: Vec<u32> = view.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 1)?;

    let t = Tensor::from_slice(&data, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_zeros(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let shape = read_shape(&mut cx, 0)?;
    let t = Tensor::zeros(shape, DType::F32, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_ones(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let shape = read_shape(&mut cx, 0)?;
    let t =
        Tensor::ones(shape, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_full(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let shape = read_shape(&mut cx, 0)?;
    let value = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let t = Tensor::full(value, shape, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_randn(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let mean = cx.argument::<JsNumber>(0)?.value(&mut cx) as f32;
    let std = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let shape = read_shape(&mut cx, 2)?;
    let t = Tensor::randn::<_, f32>(mean, std, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_rand(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let lo = cx.argument::<JsNumber>(0)?.value(&mut cx) as f32;
    let up = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let shape = read_shape(&mut cx, 2)?;
    let t = Tensor::rand::<_, f32>(lo, up, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_arange(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let start = cx.argument::<JsNumber>(0)?.value(&mut cx) as f32;
    let end = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let step = if cx.len() > 2 {
        cx.argument::<JsNumber>(2)?.value(&mut cx) as f32
    } else {
        1.0
    };
    let t = Tensor::arange_step(start, end, step, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_eye(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let n = cx.argument::<JsNumber>(0)?.value(&mut cx) as usize;
    let t = Tensor::eye(n, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_tril(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let n = cx.argument::<JsNumber>(0)?.value(&mut cx) as usize;
    let t =
        Tensor::tril2(n, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn tensor_triu(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let n = cx.argument::<JsNumber>(0)?.value(&mut cx) as usize;
    let t =
        Tensor::triu2(n, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(t)))
}

fn var_from_f32(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let view = cx.argument::<JsTypedArray<f32>>(0)?;
    let data: Vec<f32> = view.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 1)?;

    let t = Tensor::from_slice(&data, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let v = Var::from_tensor(&t).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

fn var_randn(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let std = cx.argument::<JsNumber>(0)?.value(&mut cx);
    let shape = read_shape(&mut cx, 1)?;
    let v = Var::randn(0f32, std as f32, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

fn var_zeros(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let shape = read_shape(&mut cx, 0)?;
    let v =
        Var::zeros(shape, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

fn var_ones(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let shape = read_shape(&mut cx, 0)?;
    let v =
        Var::ones(shape, DType::F32, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

fn var_full(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let shape = read_shape(&mut cx, 0)?;
    let value = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let t = Tensor::full(value, shape, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    let v = Var::from_tensor(&t).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

// ---- elementwise ops (broadcasting, PyTorch-like) ----

binary_op!(tensor_add, broadcast_add);
binary_op!(tensor_sub, broadcast_sub);
binary_op!(tensor_mul, broadcast_mul);
binary_op!(tensor_div, broadcast_div);
binary_op!(tensor_maximum, broadcast_maximum);
binary_op!(tensor_minimum, broadcast_minimum);
binary_op!(tensor_eq, broadcast_eq);
binary_op!(tensor_ne, broadcast_ne);
binary_op!(tensor_lt, broadcast_lt);
binary_op!(tensor_le, broadcast_le);
binary_op!(tensor_gt, broadcast_gt);
binary_op!(tensor_ge, broadcast_ge);

unary_op!(tensor_neg, neg);
unary_op!(tensor_abs, abs);
unary_op!(tensor_sqr, sqr);
unary_op!(tensor_sqrt, sqrt);
unary_op!(tensor_exp, exp);
unary_op!(tensor_log, log);
unary_op!(tensor_recip, recip);
unary_op!(tensor_sin, sin);
unary_op!(tensor_cos, cos);
unary_op!(tensor_ceil, ceil);
unary_op!(tensor_floor, floor);
unary_op!(tensor_round, round);
unary_op!(tensor_sign, sign);
unary_op!(tensor_relu, relu);
unary_op!(tensor_tanh, tanh);
unary_op!(tensor_gelu, gelu);
unary_op!(tensor_gelu_erf, gelu_erf);
unary_op!(tensor_silu, silu);
unary_op!(tensor_erf, erf);

unary_fn_op!(tensor_sigmoid, ops::sigmoid);
unary_fn_op!(tensor_mish, ops::mish);
unary_fn_op!(tensor_swiglu, ops::swiglu);
unary_fn_op!(tensor_hard_sigmoid, ops::hard_sigmoid);
unary_fn_op!(tensor_softmax_last_dim, ops::softmax_last_dim);

fn tensor_relu2(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a
        .relu()
        .and_then(|x| x.sqr())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_relu6(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a
        .clamp(0f32, 6f32)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_hard_swish(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = ops::hard_sigmoid(&a)
        .and_then(|s| &a * s)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_selu(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let alpha = cx.argument::<JsNumber>(1)?.value(&mut cx) as f32;
    let gamma = cx.argument::<JsNumber>(2)?.value(&mut cx) as f32;
    let c = ops::selu(&a, alpha, gamma).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_affine(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let mul = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let add = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let c = a
        .affine(mul, add)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_pow(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = a.pow(&b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_powf(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let e = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let c = a.powf(e).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_elu(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let alpha = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let c = a.elu(alpha).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_leaky_relu(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let slope = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let c = ops::leaky_relu(&a, slope).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_clamp(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let min = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let max = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let c = a
        .clamp(min, max)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_softmax(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = ops::softmax(&a, dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_log_softmax(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = ops::log_softmax(&a, dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_where(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let cond = arg_tensor(&mut cx, 0)?;
    let on_true = arg_tensor(&mut cx, 1)?;
    let on_false = arg_tensor(&mut cx, 2)?;
    let c = cond
        .where_cond(&on_true, &on_false)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

/// Replace positions where `mask` is nonzero with `value`.
fn tensor_masked_fill(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let mask = arg_tensor(&mut cx, 1)?;
    let value = cx.argument::<JsNumber>(2)?.value(&mut cx) as f32;
    let mask = mask
        .to_dtype(DType::U8)
        .and_then(|m| m.broadcast_as(a.shape().clone()))
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let filled = Tensor::full(value, a.shape().clone(), a.device())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let c = mask
        .where_cond(&filled, &a)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

// ---- matmul ----

fn tensor_matmul(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = a.matmul(&b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_broadcast_matmul(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = a
        .broadcast_matmul(&b)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

// ---- reductions ----

fn tensor_sum_all(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.sum_all().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_mean_all(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.mean_all().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_max_all(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.max_all().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_min_all(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.min_all().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_norm(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.norm().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_sum(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_dims(&mut cx, 1)?;
    let c = a.sum(dims).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_sum_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_dims(&mut cx, 1)?;
    let c = a
        .sum_keepdim(dims)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_mean(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_dims(&mut cx, 1)?;
    let c = a.mean(dims).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_mean_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_dims(&mut cx, 1)?;
    let c = a
        .mean_keepdim(dims)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_max(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.max(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_max_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a
        .max_keepdim(dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_min(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.min(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_min_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a
        .min_keepdim(dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_argmax(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.argmax(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_argmax_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a
        .argmax_keepdim(dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_argmin(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.argmin(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_argmin_keepdim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a
        .argmin_keepdim(dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_cumsum(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.cumsum(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_log_sum_exp(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_dims(&mut cx, 1)?;
    let c = a
        .log_sum_exp(dims)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn dot(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let scalar: f32 = (&a * &b)
        .and_then(|p| p.sum_all())
        .and_then(|s| s.to_scalar::<f32>())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.number(scalar as f64))
}

// ---- shape manipulation ----

fn tensor_reshape(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shape = read_shape(&mut cx, 1)?;
    let c = a
        .reshape(shape)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_flatten_all(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.flatten_all().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_flatten(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let start = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let end = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a
        .flatten(start, end)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_transpose(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let d0 = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let d1 = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a
        .transpose(d0, d1)
        .and_then(|t| t.contiguous())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_t(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.t().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_permute(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_shape(&mut cx, 1)?;
    let c = a
        .permute(dims)
        .and_then(|t| t.contiguous())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_squeeze(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.squeeze(dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_unsqueeze(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a
        .unsqueeze(dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_narrow(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let start = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let len = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;
    let c = a
        .narrow(dim, start, len)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_broadcast_as(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shape = read_shape(&mut cx, 1)?;
    let c = a
        .broadcast_as(shape)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_expand(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shape = read_shape(&mut cx, 1)?;
    let c = a.expand(shape).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_repeat(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shape = read_shape(&mut cx, 1)?;
    let c = a.repeat(shape).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_contiguous(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.contiguous().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_detach(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    Ok(cx.boxed(JsTensor(a.detach())))
}

fn tensor_to_dtype(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let name = cx.argument::<JsString>(1)?.value(&mut cx);
    let dtype = match name.as_str() {
        "f32" => DType::F32,
        "f64" => DType::F64,
        "u8" => DType::U8,
        "u32" => DType::U32,
        "i64" => DType::I64,
        other => return cx.throw_error(format!("unsupported dtype: {other}")),
    };
    let c = a
        .to_dtype(dtype)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_get(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let i = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = a.get(i).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_get_on_dim(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let i = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a
        .get_on_dim(dim, i)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_pad_with_zeros(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let left = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let right = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;
    let c = a
        .pad_with_zeros(dim, left, right)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_flip(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dims = read_shape(&mut cx, 1)?;
    let c = a.flip(&dims).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_roll(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shift = cx.argument::<JsNumber>(1)?.value(&mut cx) as i32;
    let dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a
        .roll(shift, dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_chunk(mut cx: FunctionContext) -> JsResult<JsArray> {
    let a = arg_tensor(&mut cx, 0)?;
    let chunks = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let parts = a
        .chunk(chunks, dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let arr = JsArray::new(&mut cx, parts.len());
    for (i, t) in parts.into_iter().enumerate() {
        let boxed = cx.boxed(JsTensor(t));
        arr.set(&mut cx, i as u32, boxed)?;
    }
    Ok(arr)
}

fn tensor_cat(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let tensors = arg_tensor_list(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = Tensor::cat(&tensors, dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_stack(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let tensors = arg_tensor_list(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = Tensor::stack(&tensors, dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

// ---- indexing ----

fn tensor_index_select(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let ids: Vec<u32> = cx.argument::<JsTypedArray<u32>>(1)?.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 2)?;
    let dim = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;
    let idx =
        Tensor::from_slice(&ids, shape, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    let c = a
        .index_select(&idx, dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_gather(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let index = arg_tensor(&mut cx, 1)?;
    let dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a
        .gather(&index, dim)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_embedding(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let weight = arg_tensor(&mut cx, 0)?;
    let ids: Vec<u32> = cx.argument::<JsTypedArray<u32>>(1)?.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 2)?;
    let idx =
        Tensor::from_slice(&ids, shape, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    let c = weight
        .embedding(&idx)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

// ---- readback ----

fn tensor_to_f32(mut cx: FunctionContext) -> JsResult<JsObject> {
    let t = arg_tensor(&mut cx, 0)?;
    let shape = t.dims().to_vec();
    let data: Vec<f32> = t
        .flatten_all()
        .and_then(|f| f.to_vec1::<f32>())
        .or_else(|e| cx.throw_error(e.to_string()))?;

    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut buf = JsArrayBuffer::new(&mut cx, bytes.len())?;
    buf.as_mut_slice(&mut cx).copy_from_slice(&bytes);

    let shape_arr = JsArray::new(&mut cx, shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let num = cx.number(dim as f64);
        shape_arr.set(&mut cx, i as u32, num)?;
    }

    let out = cx.empty_object();
    out.set(&mut cx, "shape", shape_arr)?;
    out.set(&mut cx, "data", buf)?;
    Ok(out)
}

fn tensor_to_u32(mut cx: FunctionContext) -> JsResult<JsObject> {
    let t = arg_tensor(&mut cx, 0)?;
    let shape = t.dims().to_vec();
    let data: Vec<u32> = t
        .flatten_all()
        .and_then(|f| f.to_vec1::<u32>())
        .or_else(|e| cx.throw_error(e.to_string()))?;

    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut buf = JsArrayBuffer::new(&mut cx, bytes.len())?;
    buf.as_mut_slice(&mut cx).copy_from_slice(&bytes);

    let shape_arr = JsArray::new(&mut cx, shape.len());
    for (i, &dim) in shape.iter().enumerate() {
        let num = cx.number(dim as f64);
        shape_arr.set(&mut cx, i as u32, num)?;
    }

    let out = cx.empty_object();
    out.set(&mut cx, "shape", shape_arr)?;
    out.set(&mut cx, "data", buf)?;
    Ok(out)
}

fn tensor_to_scalar(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let t = arg_tensor(&mut cx, 0)?;
    let s: f32 = t
        .flatten_all()
        .and_then(|f| f.sum_all())
        .and_then(|f| f.to_scalar::<f32>())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.number(s as f64))
}

fn tensor_shape(mut cx: FunctionContext) -> JsResult<JsArray> {
    let dims = { arg_tensor(&mut cx, 0)?.dims().to_vec() };
    let arr = JsArray::new(&mut cx, dims.len());
    for (i, &dim) in dims.iter().enumerate() {
        let num = cx.number(dim as f64);
        arr.set(&mut cx, i as u32, num)?;
    }
    Ok(arr)
}

fn tensor_rank(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let r = { arg_tensor(&mut cx, 0)?.rank() };
    Ok(cx.number(r as f64))
}

fn tensor_elem_count(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let n = { arg_tensor(&mut cx, 0)?.elem_count() };
    Ok(cx.number(n as f64))
}

fn tensor_dtype(mut cx: FunctionContext) -> JsResult<JsString> {
    let d = { arg_tensor(&mut cx, 0)?.dtype() };
    Ok(cx.string(format!("{d:?}").to_lowercase()))
}

// ---- layers ----

fn varmap_new(mut cx: FunctionContext) -> JsResult<JsBox<JsVarMap>> {
    Ok(cx.boxed(JsVarMap(RefCell::new(VarMap::new()))))
}

fn varmap_num_vars(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let n = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let guard = vm.0.borrow();
        guard.all_vars().len()
    };
    Ok(cx.number(n as f64))
}

fn varmap_get(mut cx: FunctionContext) -> JsResult<JsBox<JsVar>> {
    let name = cx.argument::<JsString>(1)?.value(&mut cx);
    let var = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let guard = vm.0.borrow();
        let data = guard.data().lock().unwrap();
        data.get(&name).cloned()
    };
    match var {
        Some(v) => Ok(cx.boxed(JsVar(v))),
        None => cx.throw_error(format!("no var named {name}")),
    }
}

/// linearNew(varmap, name, inDim, outDim, bias)
fn linear_new(mut cx: FunctionContext) -> JsResult<JsBox<JsLinear>> {
    let l = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let name = cx.argument::<JsString>(1)?.value(&mut cx);
        let in_dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
        let out_dim = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;
        let bias = cx.argument::<JsBoolean>(4)?.value(&mut cx);

        let guard = vm.0.borrow();
        let vb = VarBuilder::from_varmap(&guard, DType::F32, &Device::Cpu);
        if bias {
            candle_nn::linear(in_dim, out_dim, vb.pp(&name))
        } else {
            candle_nn::linear_no_bias(in_dim, out_dim, vb.pp(&name))
        }
    };
    let l = l.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsLinear(l)))
}

fn linear_forward(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let x = arg_tensor(&mut cx, 1)?;
    let y = {
        let l = cx.argument::<JsBox<JsLinear>>(0)?;
        l.0.forward(&x)
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

/// embeddingNew(varmap, name, vocabSize, dim)
fn embedding_new(mut cx: FunctionContext) -> JsResult<JsBox<JsEmbedding>> {
    let e = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let name = cx.argument::<JsString>(1)?.value(&mut cx);
        let vocab = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
        let dim = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;

        let guard = vm.0.borrow();
        let vb = VarBuilder::from_varmap(&guard, DType::F32, &Device::Cpu);
        candle_nn::embedding(vocab, dim, vb.pp(&name))
    };
    let e = e.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsEmbedding(e)))
}

/// embeddingForward(emb, Uint32Array ids, shape) -> [...shape, dim]
fn embedding_forward(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let ids: Vec<u32> = cx.argument::<JsTypedArray<u32>>(1)?.as_slice(&cx).to_vec();
    let shape = read_shape(&mut cx, 2)?;

    let idx =
        Tensor::from_slice(&ids, shape, &Device::Cpu).or_else(|e| cx.throw_error(e.to_string()))?;
    let y = {
        let e = cx.argument::<JsBox<JsEmbedding>>(0)?;
        e.0.forward(&idx)
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

/// layerNormNew(varmap, name, dim, eps, bias)
fn layer_norm_new(mut cx: FunctionContext) -> JsResult<JsBox<JsLayerNorm>> {
    let l = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let name = cx.argument::<JsString>(1)?.value(&mut cx);
        let dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
        let eps = cx.argument::<JsNumber>(3)?.value(&mut cx);
        let affine = cx.argument::<JsBoolean>(4)?.value(&mut cx);

        let config = LayerNormConfig {
            eps,
            remove_mean: true,
            affine,
        };
        let guard = vm.0.borrow();
        let vb = VarBuilder::from_varmap(&guard, DType::F32, &Device::Cpu);
        candle_nn::layer_norm(dim, config, vb.pp(&name))
    };
    let l = l.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsLayerNorm(l)))
}

fn layer_norm_forward(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let x = arg_tensor(&mut cx, 1)?;
    let y = {
        let l = cx.argument::<JsBox<JsLayerNorm>>(0)?;
        let last = x.rank().saturating_sub(1);
        let eps = l.0.eps();
        let weight = l.0.weight();
        x.mean_keepdim(last)
            .and_then(|mean| x.broadcast_sub(&mean))
            .and_then(|x| {
                let var = x.sqr()?.mean_keepdim(last)?;
                x.broadcast_div(&(var + eps)?.sqrt()?)
            })
            .and_then(|x| x.broadcast_mul(weight))
            .and_then(|x| match l.0.bias() {
                Some(bias) => x.broadcast_add(bias),
                None => Ok(x),
            })
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

/// rmsNormNew(varmap, name, dim, eps)
fn rms_norm_new(mut cx: FunctionContext) -> JsResult<JsBox<JsRmsNorm>> {
    let l = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let name = cx.argument::<JsString>(1)?.value(&mut cx);
        let dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
        let eps = cx.argument::<JsNumber>(3)?.value(&mut cx);

        let guard = vm.0.borrow();
        let vb = VarBuilder::from_varmap(&guard, DType::F32, &Device::Cpu);
        candle_nn::rms_norm(dim, eps, vb.pp(&name))
    };
    let l = l.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsRmsNorm(l)))
}

fn rms_norm_forward(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let x = arg_tensor(&mut cx, 1)?;
    let y = {
        let l = cx.argument::<JsBox<JsRmsNorm>>(0)?;
        let last = x.rank().saturating_sub(1);
        let eps = l.0.eps();
        let weight = l.0.weight();
        let var = x.sqr().and_then(|x| x.mean_keepdim(last));
        var.and_then(|var| x.broadcast_div(&(var + eps)?.sqrt()?))
            .and_then(|x| x.broadcast_mul(weight))
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

fn dropout_new(mut cx: FunctionContext) -> JsResult<JsBox<JsDropout>> {
    let p = cx.argument::<JsNumber>(0)?.value(&mut cx) as f32;
    Ok(cx.boxed(JsDropout(Dropout::new(p))))
}

fn dropout_forward(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let x = arg_tensor(&mut cx, 1)?;
    let train = cx.argument::<JsBoolean>(2)?.value(&mut cx);
    let y = {
        let l = cx.argument::<JsBox<JsDropout>>(0)?;
        l.0.forward(&x, train)
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

/// lstmNew(varmap, name, inDim, hiddenDim, bias)
fn lstm_new(mut cx: FunctionContext) -> JsResult<JsBox<JsLstm>> {
    let l = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let name = cx.argument::<JsString>(1)?.value(&mut cx);
        let in_dim = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
        let hidden_dim = cx.argument::<JsNumber>(3)?.value(&mut cx) as usize;
        let bias = cx.argument::<JsBoolean>(4)?.value(&mut cx);

        let config = if bias {
            LSTMConfig::default()
        } else {
            LSTMConfig::default_no_bias()
        };
        let guard = vm.0.borrow();
        let vb = VarBuilder::from_varmap(&guard, DType::F32, &Device::Cpu);
        candle_nn::lstm(in_dim, hidden_dim, config, vb.pp(&name))
    };
    let l = l.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsLstm(l)))
}

/// lstmSeq(lstm, input[B, T, F]) -> [B, T, hidden]
fn lstm_seq(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let input = arg_tensor(&mut cx, 1)?;
    let y = {
        let l = cx.argument::<JsBox<JsLstm>>(0)?;
        l.0.seq(&input)
            .and_then(|states| l.0.states_to_tensor(&states))
    };
    let y = y.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(y)))
}

/// One SGD step over every var in the map. Returns count of updated params.
fn varmap_sgd_step(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let lr = cx.argument::<JsNumber>(2)?.value(&mut cx);

    let result = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let grads = cx.argument::<JsBox<JsGrads>>(1)?;
        let store = grads.0.borrow();
        let guard = vm.0.borrow();

        let mut n = 0;
        let mut err = None;
        for var in guard.all_vars() {
            if let Some(g) = store.get(&var) {
                match (g * lr)
                    .and_then(|s| var.as_tensor().sub(&s))
                    .and_then(|u| var.set(&u))
                {
                    Ok(()) => n += 1,
                    Err(e) => {
                        err = Some(e);
                        break;
                    }
                }
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok(n),
        }
    };

    let n = result.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.number(n as f64))
}

fn varmap_save(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let path = cx.argument::<JsString>(1)?.value(&mut cx);
    let r = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let guard = vm.0.borrow();
        guard.save(&path)
    };
    r.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.undefined())
}

fn varmap_load(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let path = cx.argument::<JsString>(1)?.value(&mut cx);
    let r = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let mut guard = vm.0.borrow_mut();
        guard.load(&path)
    };
    r.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.undefined())
}

/// Inspect a safetensors file: returns `{ name: { shape, dtype } }`.
fn safetensors_inspect(mut cx: FunctionContext) -> JsResult<JsObject> {
    let path = cx.argument::<JsString>(0)?.value(&mut cx);
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }
        .or_else(|e| cx.throw_error(e.to_string()))?;

    let out = cx.empty_object();
    for (name, view) in st.tensors() {
        let shape_arr = JsArray::new(&mut cx, view.shape().len());
        for (i, &d) in view.shape().iter().enumerate() {
            let n = cx.number(d as f64);
            shape_arr.set(&mut cx, i as u32, n)?;
        }
        let entry = cx.empty_object();
        entry.set(&mut cx, "shape", shape_arr)?;
        let dtype = cx.string(format!("{:?}", view.dtype()).to_lowercase());
        entry.set(&mut cx, "dtype", dtype)?;
        out.set(&mut cx, name.as_str(), entry)?;
    }
    Ok(out)
}

// ---- losses ----

fn mse_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let target = arg_tensor(&mut cx, 1)?;
    let l = candle_nn::loss::mse(&pred, &target).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(l)))
}

/// pred: [B, C] logits, target: [B] u32 class indices
fn cross_entropy_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let view = cx.argument::<JsTypedArray<u32>>(1)?;
    let idx: Vec<u32> = view.as_slice(&cx).to_vec();
    let target = Tensor::from_slice(&idx, idx.len(), &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let l = candle_nn::loss::cross_entropy(&pred, &target)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(l)))
}

/// inp: [B, C] log-probabilities, target: [B] u32 class indices
fn nll_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let view = cx.argument::<JsTypedArray<u32>>(1)?;
    let idx: Vec<u32> = view.as_slice(&cx).to_vec();
    let target = Tensor::from_slice(&idx, idx.len(), &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let l = candle_nn::loss::nll(&pred, &target).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(l)))
}

fn bce_with_logit_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let target = arg_tensor(&mut cx, 1)?;
    let l = candle_nn::loss::binary_cross_entropy_with_logit(&pred, &target)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(l)))
}

fn huber_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let target = arg_tensor(&mut cx, 1)?;
    let delta = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let l =
        candle_nn::loss::huber(&pred, &target, delta).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(l)))
}

// ---- autograd ----

fn backward(mut cx: FunctionContext) -> JsResult<JsBox<JsGrads>> {
    let loss = cx.argument::<JsBox<JsTensor>>(0)?;
    let grads = loss
        .0
        .backward()
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsGrads(RefCell::new(grads))))
}

fn grad_of(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let g = {
        let grads = cx.argument::<JsBox<JsGrads>>(0)?;
        let var = cx.argument::<JsBox<JsVar>>(1)?;
        let store = grads.0.borrow();
        store.get(&var.0).cloned()
    };
    match g {
        Some(t) => Ok(cx.boxed(JsTensor(t))),
        None => cx.throw_error("no gradient for that var (not in the graph?)"),
    }
}

/// p <- p - lr * grad(p). Returns true if a grad was found.
fn sgd_step(mut cx: FunctionContext) -> JsResult<JsBoolean> {
    let grads = cx.argument::<JsBox<JsGrads>>(0)?;
    let var = cx.argument::<JsBox<JsVar>>(1)?;
    let lr = cx.argument::<JsNumber>(2)?.value(&mut cx);

    let g = {
        let store = grads.0.borrow();
        store.get(&var.0).cloned()
    };

    let Some(g) = g else {
        return Ok(cx.boolean(false));
    };

    let updated = var
        .0
        .as_tensor()
        .sub(&(g * lr).or_else(|e| cx.throw_error(e.to_string()))?)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    var.0
        .set(&updated)
        .or_else(|e| cx.throw_error(e.to_string()))?;

    Ok(cx.boolean(true))
}

// ---- optimizers ----

/// adamwNew(varmap, lr, beta1, beta2, eps, weightDecay)
fn adamw_new(mut cx: FunctionContext) -> JsResult<JsBox<JsAdamW>> {
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let beta1 = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let beta2 = cx.argument::<JsNumber>(3)?.value(&mut cx);
    let eps = cx.argument::<JsNumber>(4)?.value(&mut cx);
    let wd = cx.argument::<JsNumber>(5)?.value(&mut cx);

    let opt = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let guard = vm.0.borrow();
        let params = ParamsAdamW {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay: wd,
        };
        AdamW::new(guard.all_vars(), params)
    };
    let opt = opt.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsAdamW(RefCell::new(opt))))
}

/// adamwNewFromVars(vars[], lr, beta1, beta2, eps, weightDecay)
fn adamw_new_from_vars(mut cx: FunctionContext) -> JsResult<JsBox<JsAdamW>> {
    let vars = arg_var_list(&mut cx, 0)?;
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let beta1 = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let beta2 = cx.argument::<JsNumber>(3)?.value(&mut cx);
    let eps = cx.argument::<JsNumber>(4)?.value(&mut cx);
    let wd = cx.argument::<JsNumber>(5)?.value(&mut cx);

    let params = ParamsAdamW {
        lr,
        beta1,
        beta2,
        eps,
        weight_decay: wd,
    };
    let opt = AdamW::new(vars, params).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsAdamW(RefCell::new(opt))))
}

fn adamw_step(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let r = {
        let opt = cx.argument::<JsBox<JsAdamW>>(0)?;
        let grads = cx.argument::<JsBox<JsGrads>>(1)?;
        let store = grads.0.borrow();
        let mut o = opt.0.borrow_mut();
        o.step(&store)
    };
    r.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.undefined())
}

fn adamw_set_lr(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    {
        let opt = cx.argument::<JsBox<JsAdamW>>(0)?;
        opt.0.borrow_mut().set_learning_rate(lr);
    }
    Ok(cx.undefined())
}

/// sgdNew(varmap, lr)
fn sgd_new(mut cx: FunctionContext) -> JsResult<JsBox<JsSgd>> {
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let opt = {
        let vm = cx.argument::<JsBox<JsVarMap>>(0)?;
        let guard = vm.0.borrow();
        SGD::new(guard.all_vars(), lr)
    };
    let opt = opt.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsSgd(RefCell::new(opt))))
}

/// sgdNewFromVars(vars[], lr)
fn sgd_new_from_vars(mut cx: FunctionContext) -> JsResult<JsBox<JsSgd>> {
    let vars = arg_var_list(&mut cx, 0)?;
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let opt = SGD::new(vars, lr).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsSgd(RefCell::new(opt))))
}

fn sgd_opt_step(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let r = {
        let opt = cx.argument::<JsBox<JsSgd>>(0)?;
        let grads = cx.argument::<JsBox<JsGrads>>(1)?;
        let store = grads.0.borrow();
        let mut o = opt.0.borrow_mut();
        o.step(&store)
    };
    r.or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.undefined())
}

fn sgd_set_lr(mut cx: FunctionContext) -> JsResult<JsUndefined> {
    let lr = cx.argument::<JsNumber>(1)?.value(&mut cx);
    {
        let opt = cx.argument::<JsBox<JsSgd>>(0)?;
        opt.0.borrow_mut().set_learning_rate(lr);
    }
    Ok(cx.undefined())
}

#[neon::main]
fn main(mut cx: ModuleContext) -> NeonResult<()> {
    cx.export_function("hello", hello)?;
    cx.export_function("add", add)?;

    cx.export_function("tensorFromF32", tensor_from_f32)?;
    cx.export_function("tensorFromU32", tensor_from_u32)?;
    cx.export_function("tensorZeros", tensor_zeros)?;
    cx.export_function("tensorOnes", tensor_ones)?;
    cx.export_function("tensorFull", tensor_full)?;
    cx.export_function("tensorRandn", tensor_randn)?;
    cx.export_function("tensorRand", tensor_rand)?;
    cx.export_function("tensorArange", tensor_arange)?;
    cx.export_function("tensorEye", tensor_eye)?;
    cx.export_function("tensorTril", tensor_tril)?;
    cx.export_function("tensorTriu", tensor_triu)?;

    cx.export_function("varFromF32", var_from_f32)?;
    cx.export_function("varRandn", var_randn)?;
    cx.export_function("varZeros", var_zeros)?;
    cx.export_function("varOnes", var_ones)?;
    cx.export_function("varFull", var_full)?;

    cx.export_function("tensorAdd", tensor_add)?;
    cx.export_function("tensorBroadcastAdd", tensor_add)?;
    cx.export_function("tensorBroadcastSub", tensor_sub)?;
    cx.export_function("tensorBroadcastMul", tensor_mul)?;
    cx.export_function("tensorBroadcastDiv", tensor_div)?;
    cx.export_function("tensorSub", tensor_sub)?;
    cx.export_function("tensorMul", tensor_mul)?;
    cx.export_function("tensorDiv", tensor_div)?;
    cx.export_function("tensorMaximum", tensor_maximum)?;
    cx.export_function("tensorMinimum", tensor_minimum)?;
    cx.export_function("tensorEq", tensor_eq)?;
    cx.export_function("tensorNe", tensor_ne)?;
    cx.export_function("tensorLt", tensor_lt)?;
    cx.export_function("tensorLe", tensor_le)?;
    cx.export_function("tensorGt", tensor_gt)?;
    cx.export_function("tensorGe", tensor_ge)?;
    cx.export_function("tensorNeg", tensor_neg)?;
    cx.export_function("tensorAbs", tensor_abs)?;
    cx.export_function("tensorSqr", tensor_sqr)?;
    cx.export_function("tensorSqrt", tensor_sqrt)?;
    cx.export_function("tensorExp", tensor_exp)?;
    cx.export_function("tensorLog", tensor_log)?;
    cx.export_function("tensorRecip", tensor_recip)?;
    cx.export_function("tensorSin", tensor_sin)?;
    cx.export_function("tensorCos", tensor_cos)?;
    cx.export_function("tensorCeil", tensor_ceil)?;
    cx.export_function("tensorFloor", tensor_floor)?;
    cx.export_function("tensorRound", tensor_round)?;
    cx.export_function("tensorSign", tensor_sign)?;
    cx.export_function("tensorAffine", tensor_affine)?;
    cx.export_function("tensorPow", tensor_pow)?;
    cx.export_function("tensorPowf", tensor_powf)?;
    cx.export_function("tensorClamp", tensor_clamp)?;
    cx.export_function("tensorWhere", tensor_where)?;
    cx.export_function("tensorMaskedFill", tensor_masked_fill)?;

    cx.export_function("tensorRelu", tensor_relu)?;
    cx.export_function("tensorTanh", tensor_tanh)?;
    cx.export_function("tensorSigmoid", tensor_sigmoid)?;
    cx.export_function("tensorGelu", tensor_gelu)?;
    cx.export_function("tensorGeluErf", tensor_gelu_erf)?;
    cx.export_function("tensorSilu", tensor_silu)?;
    cx.export_function("tensorErf", tensor_erf)?;
    cx.export_function("tensorSwiGLU", tensor_swiglu)?;
    cx.export_function("tensorHardSigmoid", tensor_hard_sigmoid)?;
    cx.export_function("tensorHardSwish", tensor_hard_swish)?;
    cx.export_function("tensorRelu2", tensor_relu2)?;
    cx.export_function("tensorRelu6", tensor_relu6)?;
    cx.export_function("tensorSelu", tensor_selu)?;
    cx.export_function("tensorMish", tensor_mish)?;
    cx.export_function("tensorElu", tensor_elu)?;
    cx.export_function("tensorLeakyRelu", tensor_leaky_relu)?;
    cx.export_function("tensorSoftmax", tensor_softmax)?;
    cx.export_function("tensorLogSoftmax", tensor_log_softmax)?;
    cx.export_function("tensorSoftmaxLastDim", tensor_softmax_last_dim)?;

    cx.export_function("tensorMatmul", tensor_matmul)?;
    cx.export_function("tensorBroadcastMatmul", tensor_broadcast_matmul)?;

    cx.export_function("tensorSumAll", tensor_sum_all)?;
    cx.export_function("tensorMeanAll", tensor_mean_all)?;
    cx.export_function("tensorMaxAll", tensor_max_all)?;
    cx.export_function("tensorMinAll", tensor_min_all)?;
    cx.export_function("tensorNorm", tensor_norm)?;
    cx.export_function("tensorSum", tensor_sum)?;
    cx.export_function("tensorSumKeepdim", tensor_sum_keepdim)?;
    cx.export_function("tensorMean", tensor_mean)?;
    cx.export_function("tensorMeanKeepdim", tensor_mean_keepdim)?;
    cx.export_function("tensorMax", tensor_max)?;
    cx.export_function("tensorMaxKeepdim", tensor_max_keepdim)?;
    cx.export_function("tensorMin", tensor_min)?;
    cx.export_function("tensorMinKeepdim", tensor_min_keepdim)?;
    cx.export_function("tensorArgmax", tensor_argmax)?;
    cx.export_function("tensorArgmaxKeepdim", tensor_argmax_keepdim)?;
    cx.export_function("tensorArgmin", tensor_argmin)?;
    cx.export_function("tensorArgminKeepdim", tensor_argmin_keepdim)?;
    cx.export_function("tensorCumsum", tensor_cumsum)?;
    cx.export_function("tensorLogSumExp", tensor_log_sum_exp)?;
    cx.export_function("dot", dot)?;

    cx.export_function("tensorReshape", tensor_reshape)?;
    cx.export_function("tensorFlattenAll", tensor_flatten_all)?;
    cx.export_function("tensorFlatten", tensor_flatten)?;
    cx.export_function("tensorTranspose", tensor_transpose)?;
    cx.export_function("tensorT", tensor_t)?;
    cx.export_function("tensorPermute", tensor_permute)?;
    cx.export_function("tensorSqueeze", tensor_squeeze)?;
    cx.export_function("tensorUnsqueeze", tensor_unsqueeze)?;
    cx.export_function("tensorNarrow", tensor_narrow)?;
    cx.export_function("tensorBroadcastAs", tensor_broadcast_as)?;
    cx.export_function("tensorExpand", tensor_expand)?;
    cx.export_function("tensorRepeat", tensor_repeat)?;
    cx.export_function("tensorContiguous", tensor_contiguous)?;
    cx.export_function("tensorDetach", tensor_detach)?;
    cx.export_function("tensorToDtype", tensor_to_dtype)?;
    cx.export_function("tensorGet", tensor_get)?;
    cx.export_function("tensorGetOnDim", tensor_get_on_dim)?;
    cx.export_function("tensorPadWithZeros", tensor_pad_with_zeros)?;
    cx.export_function("tensorFlip", tensor_flip)?;
    cx.export_function("tensorRoll", tensor_roll)?;
    cx.export_function("tensorChunk", tensor_chunk)?;
    cx.export_function("tensorCat", tensor_cat)?;
    cx.export_function("tensorStack", tensor_stack)?;

    cx.export_function("tensorIndexSelect", tensor_index_select)?;
    cx.export_function("tensorGather", tensor_gather)?;
    cx.export_function("tensorEmbedding", tensor_embedding)?;

    cx.export_function("tensorToF32", tensor_to_f32)?;
    cx.export_function("tensorToU32", tensor_to_u32)?;
    cx.export_function("tensorToScalar", tensor_to_scalar)?;
    cx.export_function("tensorShape", tensor_shape)?;
    cx.export_function("tensorRank", tensor_rank)?;
    cx.export_function("tensorElemCount", tensor_elem_count)?;
    cx.export_function("tensorDtype", tensor_dtype)?;

    cx.export_function("varmapNew", varmap_new)?;
    cx.export_function("varmapNumVars", varmap_num_vars)?;
    cx.export_function("varmapGet", varmap_get)?;
    cx.export_function("varmapSgdStep", varmap_sgd_step)?;
    cx.export_function("varmapSave", varmap_save)?;
    cx.export_function("varmapLoad", varmap_load)?;
    cx.export_function("safetensorsInspect", safetensors_inspect)?;
    cx.export_function("linearNew", linear_new)?;
    cx.export_function("linearForward", linear_forward)?;
    cx.export_function("embeddingNew", embedding_new)?;
    cx.export_function("embeddingForward", embedding_forward)?;
    cx.export_function("layerNormNew", layer_norm_new)?;
    cx.export_function("layerNormForward", layer_norm_forward)?;
    cx.export_function("rmsNormNew", rms_norm_new)?;
    cx.export_function("rmsNormForward", rms_norm_forward)?;
    cx.export_function("dropoutNew", dropout_new)?;
    cx.export_function("dropoutForward", dropout_forward)?;
    cx.export_function("lstmNew", lstm_new)?;
    cx.export_function("lstmSeq", lstm_seq)?;

    cx.export_function("mseLoss", mse_loss)?;
    cx.export_function("crossEntropyLoss", cross_entropy_loss)?;
    cx.export_function("nllLoss", nll_loss)?;
    cx.export_function("bceWithLogitLoss", bce_with_logit_loss)?;
    cx.export_function("huberLoss", huber_loss)?;

    cx.export_function("backward", backward)?;
    cx.export_function("gradOf", grad_of)?;
    cx.export_function("sgdStep", sgd_step)?;

    cx.export_function("adamwNew", adamw_new)?;
    cx.export_function("adamwNewFromVars", adamw_new_from_vars)?;
    cx.export_function("adamwStep", adamw_step)?;
    cx.export_function("adamwSetLr", adamw_set_lr)?;
    cx.export_function("sgdNew", sgd_new)?;
    cx.export_function("sgdNewFromVars", sgd_new_from_vars)?;
    cx.export_function("sgdOptStep", sgd_opt_step)?;
    cx.export_function("sgdSetLr", sgd_set_lr)?;

    Ok(())
}
