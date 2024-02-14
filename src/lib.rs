// What we want to abstract
// Backend
// Model
// Input
use std::path::Path;

pub mod models;

pub fn load_input() {}

pub enum ModelTypes {
    ResNet18,
}

// impl ModelTypes {
//     fn weights(&self) -> &'static Path {
//         match self {
//             ModelTypes::ResNet18 => Path::new("./etc/models/resnet.onnx"),
//         }
//     }
// }

pub trait BenchmarkNetwork {
    fn new() -> Self;
    fn predict(&self, input: Vec<u8>);
}
