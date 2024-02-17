use crate::{InputShape, Model};
use std::path::Path;

pub struct ResNet18;

impl Model for ResNet18 {
    fn new() -> Self {
        Self
    }

    fn get_onnx_weights() -> &'static Path {
        Path::new("./models/resnet18/resnet18.onnx")
    }

    fn get_safetensors_weights() -> &'static Path {
        Path::new("./models/resnet18/resnet18.safetensors")
    }

    fn get_pytorch_weights() -> &'static Path {
        Path::new("./models/resnet18/resnet18.pt")
    }

    fn get_input_shape() -> InputShape {
        [1, 3, 224, 224]
    }
}
