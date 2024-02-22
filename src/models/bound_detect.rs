use crate::{InputShape, Model};
use std::path::Path;

pub struct BoundDetect;

impl Model for BoundDetect {
    fn new() -> Self {
        Self
    }

    fn get_onnx_weights() -> &'static Path {
        Path::new("./models/bound_detect/fieldboundary.onnx")
    }

    fn get_safetensors_weights() -> &'static Path {
        panic!("Model does not support safetensors weights!")
    }

    fn get_pytorch_weights() -> &'static Path {
        panic!("Model does not support pytorch weights!")
    }

    fn get_input_shape() -> InputShape {
        [1, 3, 640, 480]
    }
}
