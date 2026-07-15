use neon::prelude::*;
use neon::types::buffer::TypedArray;
use candle_core::{Device, Tensor, Var, backprop::GradStore, DType};
use candle_nn::{ops, Linear, Embedding, Module, VarBuilder, VarMap};
use candle_nn::{AdamW, ParamsAdamW, SGD, Optimizer};

pub struct JsAdamW(pub RefCell<AdamW>);
impl Finalize for JsAdamW {}

pub struct JsSgd(pub RefCell<SGD>);
impl Finalize for JsSgd {}
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

// ---- helpers ----

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
    let v = Var::zeros(shape, DType::F32, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsVar(v)))
}

// ---- ops (accept Tensor or Var, always return Tensor) ----

fn tensor_matmul(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = a.matmul(&b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_add(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = (&a + &b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_broadcast_add(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = a.broadcast_add(&b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_mul(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = (&a * &b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_sub(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let c = (&a - &b).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_affine(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let mul = cx.argument::<JsNumber>(1)?.value(&mut cx);
    let add = cx.argument::<JsNumber>(2)?.value(&mut cx);
    let c = a.affine(mul, add).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_relu(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.relu().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_tanh(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = a.tanh().or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_sigmoid(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let c = ops::sigmoid(&a).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_softmax(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let dim = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let c = ops::softmax(&a, dim).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_transpose(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let d0 = cx.argument::<JsNumber>(1)?.value(&mut cx) as usize;
    let d1 = cx.argument::<JsNumber>(2)?.value(&mut cx) as usize;
    let c = a.transpose(d0, d1)
        .and_then(|t| t.contiguous())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

fn tensor_reshape(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let a = arg_tensor(&mut cx, 0)?;
    let shape = read_shape(&mut cx, 1)?;
    let c = a.reshape(shape).or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.boxed(JsTensor(c)))
}

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

fn dot(mut cx: FunctionContext) -> JsResult<JsNumber> {
    let a = arg_tensor(&mut cx, 0)?;
    let b = arg_tensor(&mut cx, 1)?;
    let scalar: f32 = (&a * &b)
        .and_then(|p| p.sum_all())
        .and_then(|s| s.to_scalar::<f32>())
        .or_else(|e| cx.throw_error(e.to_string()))?;
    Ok(cx.number(scalar as f64))
}

// ---- layers ----

fn varmap_new(mut cx: FunctionContext) -> JsResult<JsBox<JsVarMap>> {
    Ok(cx.boxed(JsVarMap(RefCell::new(VarMap::new()))))
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

    let idx = Tensor::from_slice(&ids, shape, &Device::Cpu)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    let y = {
        let e = cx.argument::<JsBox<JsEmbedding>>(0)?;
        e.0.forward(&idx)
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

// ---- losses ----

fn mse_loss(mut cx: FunctionContext) -> JsResult<JsBox<JsTensor>> {
    let pred = arg_tensor(&mut cx, 0)?;
    let target = arg_tensor(&mut cx, 1)?;
    let l = candle_nn::loss::mse(&pred, &target)
        .or_else(|e| cx.throw_error(e.to_string()))?;
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

// ---- autograd ----

fn backward(mut cx: FunctionContext) -> JsResult<JsBox<JsGrads>> {
    let loss = cx.argument::<JsBox<JsTensor>>(0)?;
    let grads = loss.0.backward().or_else(|e| cx.throw_error(e.to_string()))?;
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

    let Some(g) = g else { return Ok(cx.boolean(false)) };

    let updated = var
        .0
        .as_tensor()
        .sub(&(g * lr).or_else(|e| cx.throw_error(e.to_string()))?)
        .or_else(|e| cx.throw_error(e.to_string()))?;
    var.0.set(&updated).or_else(|e| cx.throw_error(e.to_string()))?;

    Ok(cx.boolean(true))
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
    cx.export_function("varFromF32", var_from_f32)?;
    cx.export_function("varRandn", var_randn)?;
    cx.export_function("varZeros", var_zeros)?;

    cx.export_function("tensorMatmul", tensor_matmul)?;
    cx.export_function("tensorAdd", tensor_add)?;
    cx.export_function("tensorBroadcastAdd", tensor_broadcast_add)?;
    cx.export_function("tensorMul", tensor_mul)?;
    cx.export_function("tensorSub", tensor_sub)?;
    cx.export_function("tensorAffine", tensor_affine)?;
    cx.export_function("tensorRelu", tensor_relu)?;
    cx.export_function("tensorTanh", tensor_tanh)?;
    cx.export_function("tensorSigmoid", tensor_sigmoid)?;
    cx.export_function("tensorSoftmax", tensor_softmax)?;
    cx.export_function("tensorTranspose", tensor_transpose)?;
    cx.export_function("tensorReshape", tensor_reshape)?;
    cx.export_function("tensorSumAll", tensor_sum_all)?;
    cx.export_function("tensorMeanAll", tensor_mean_all)?;
    cx.export_function("dot", dot)?;

    cx.export_function("varmapNew", varmap_new)?;
    cx.export_function("varmapSgdStep", varmap_sgd_step)?;
    cx.export_function("varmapSave", varmap_save)?;
    cx.export_function("varmapLoad", varmap_load)?;
    cx.export_function("linearNew", linear_new)?;
    cx.export_function("linearForward", linear_forward)?;
    cx.export_function("embeddingNew", embedding_new)?;
    cx.export_function("embeddingForward", embedding_forward)?;

    cx.export_function("mseLoss", mse_loss)?;
    cx.export_function("crossEntropyLoss", cross_entropy_loss)?;

    cx.export_function("backward", backward)?;
    cx.export_function("gradOf", grad_of)?;
    cx.export_function("sgdStep", sgd_step)?;

    cx.export_function("tensorToF32", tensor_to_f32)?;
    cx.export_function("tensorToScalar", tensor_to_scalar)?;
    cx.export_function("tensorShape", tensor_shape)?;
    cx.export_function("adamwNew", adamw_new)?;
    cx.export_function("adamwStep", adamw_step)?;
    cx.export_function("adamwSetLr", adamw_set_lr)?;
    cx.export_function("sgdNew", sgd_new)?;
    cx.export_function("sgdOptStep", sgd_opt_step)?;
    cx.export_function("sgdSetLr", sgd_set_lr)?;

    Ok(())
}