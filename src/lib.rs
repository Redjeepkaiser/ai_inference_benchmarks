/// Current benchmarks do not take into account memory usage and multiple cores.
/// This lib abstracts away the backend and model to use for benchmarking to
/// make it easy to compare different backends and models.
use std::path::Path;

pub mod backends;
pub mod models;

/// Stores the model inputs shapes.
type InputShape = [usize; 4];

/// Abstraction of different models.
pub trait Model {
    /// Returns a new model.
    fn new() -> Self;

    /// Returns the shape model inputs should have.
    fn get_input_shape() -> InputShape;

    /// Returns path to the onnx weights of this model.
    fn get_onnx_weights() -> &'static Path;

    /// Returns path to the safetensors of this model.
    fn get_safetensors_weights() -> &'static Path;

    /// Returns path to the pytorch weights of this model.
    fn get_pytorch_weights() -> &'static Path;
}

/// Abstraction of backends for ML inference.
pub trait Backend {
    /// Type of the model that provides path to weights and input shape.
    type Model;

    /// Type of the input the model takes.
    type Input;

    /// Initializes a new backend.
    fn new() -> Self;

    /// Loads the model into memmory.
    fn load_model(&mut self);

    /// Generates test input for the benchmark with the correct shape and data
    /// type.
    fn generate_input(&self) -> Self::Input;

    /// Performs inference.
    fn predict(&mut self, input: Self::Input);
}
