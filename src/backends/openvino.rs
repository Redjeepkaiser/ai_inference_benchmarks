use crate::{Backend, Model};
use ndarray::prelude::*;
use openvino::{Blob, CNNNetwork, Core, ExecutableNetwork, Layout, Precision, TensorDesc};
use std::default::Default;
use std::marker::PhantomData;
use std::mem;
use std::path::Path;

pub struct OpenVINO<T: Model> {
    phantom: PhantomData<T>,
    model: Option<(CNNNetwork, ExecutableNetwork)>,
}

impl<M: Model> Backend for OpenVINO<M> {
    type Model = M;
    type Input = Vec<u8>;

    fn new() -> Self {
        Self {
            phantom: Default::default(),
            model: Default::default(),
        }
    }

    fn load_model(&mut self) {
        let path_to_weights: &Path = Self::Model::get_onnx_weights();

        // Instantiate core
        let mut core = Core::new(None).unwrap();

        // Read network
        let m = core
            .read_network_from_file(path_to_weights.to_str().unwrap(), "AUTO")
            .unwrap();
        let em = core.load_network(&m, "CPU").unwrap();

        self.model = Some((m, em));
    }

    fn generate_input(&self) -> Self::Input {
        let shape = Self::Model::get_input_shape();
        let input = Array::zeros(shape);
        let mut input = input.mapv(|elem: u8| elem as f32);

        unsafe {
            let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
            let length = input.len() * ratio;
            let ptr = input.as_mut_ptr() as *mut u8;
            mem::forget(input);
            Vec::from_raw_parts(ptr, length, length)
        }
    }

    fn predict(&mut self, input: Self::Input) {
        let (m, em) = self.model.as_mut().unwrap();
        let shape = Self::Model::get_input_shape();

        let mut infer_request = em.create_infer_request().unwrap();
        let blob = Blob::new(
            &TensorDesc::new(Layout::NCHW, &shape, Precision::FP32),
            &input,
        )
        .unwrap();
        let input_name = m.get_input_name(0).unwrap();
        infer_request.set_blob(&input_name, &blob).unwrap();

        // Inference
        infer_request.infer().unwrap();

        // Temp
        let output_name = m.get_output_name(0).unwrap();

        let mut output = infer_request.get_blob(&output_name).unwrap();
        let mut output = output.buffer_mut().unwrap().to_vec();

        unsafe {
            let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
            let capacity = output.capacity() / ratio;
            let length = output.len() / ratio;
            let ptr = output.as_mut_ptr() as *mut f32;
            mem::forget(output);
            Vec::from_raw_parts(ptr, length, capacity)
        };
    }
}
