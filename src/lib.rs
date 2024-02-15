// What we want to abstract
// Backend
// Model

// What we need to take into account
// Cores
// Cache
// Memory
// Openvino verions
// Openvino representation
use std::path::Path;

pub mod models;

pub fn load_input() {}

pub enum ModelTypes {
    ResNet18,
}

pub enum Input {}

impl ModelTypes {
    fn weights(&self) -> &'static Path {
        match self {
            ModelTypes::ResNet18 => Path::new("./etc/models/resnet.onnx"),
        }
    }
}

pub trait BenchmarkNetwork {
    type Input;

    fn new(model_type: ModelTypes) -> Self;
    fn generate_sample() -> Input;
    fn predict(&self, input: Input);
}
