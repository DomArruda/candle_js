use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn version() -> String {
    format!("candle-wasm {}", env!("CARGO_PKG_VERSION"))
}

/// Matrix multiply two f32 matrices given as flat row-major buffers.
#[wasm_bindgen]
pub fn matmul(
    a: &[f32],
    a_rows: u32,
    a_cols: u32,
    b: &[f32],
    b_rows: u32,
    b_cols: u32,
) -> Result<Vec<f32>, JsError> {
    let dev = Device::Cpu;
    let a = Tensor::from_vec(a.to_vec(), (a_rows as usize, a_cols as usize), &dev)?;
    let b = Tensor::from_vec(b.to_vec(), (b_rows as usize, b_cols as usize), &dev)?;
    Ok(a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()?)
}

/// Train a tiny MLP on XOR and return the four predictions.
#[wasm_bindgen]
pub fn train_xor(steps: u32) -> Result<Vec<f32>, JsError> {
    let dev = Device::Cpu;
    let xs = Tensor::new(
        &[[0f32, 0.], [0., 1.], [1., 0.], [1., 1.]],
        &dev,
    )?;
    let ys = Tensor::new(&[[0f32], [1.], [1.], [0.]], &dev)?;

    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let fc1 = candle_nn::linear(2, 8, vb.pp("fc1"))?;
    let fc2 = candle_nn::linear(8, 1, vb.pp("fc2"))?;
    let mut opt = AdamW::new(
        vm.all_vars(),
        ParamsAdamW {
            lr: 0.05,
            ..Default::default()
        },
    )?;

    for _ in 0..steps {
        let out = fc2.forward(&fc1.forward(&xs)?.tanh()?)?;
        let loss = loss::mse(&out, &ys)?;
        opt.backward_step(&loss)?;
    }

    let out = fc2.forward(&fc1.forward(&xs)?.tanh()?)?;
    Ok(out.flatten_all()?.to_vec1::<f32>()?)
}
