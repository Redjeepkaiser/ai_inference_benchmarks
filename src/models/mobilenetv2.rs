use crate::{InputShape, Model};
use std::path::Path;

pub struct MobileNetV2;

impl Model for MobileNetV2 {
    fn new() -> Self {
        Self
    }

    fn get_onnx_weights() -> &'static Path {
        Path::new("./models/mobilenetv2/mobilenetv2.onnx")
    }

    fn get_safetensors_weights() -> &'static Path {
        Path::new("./models/mobilenetv2/mobilenetv2.safetensors")
    }

    fn get_pytorch_weights() -> &'static Path {
        Path::new("./models/mobilenetv2/mobilenetv2.pt")
    }

    fn get_input_shape() -> InputShape {
        [1, 3, 224, 224]
    }
}
