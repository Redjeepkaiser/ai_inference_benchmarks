use crate::{Backend, Model};
use std::default::Default;
use std::marker::PhantomData;
use std::path::Path;

use tract_ndarray::Array;
use tract_onnx::{prelude::*, tract_core::internal::tract_smallvec::SmallVec};

type TractModel = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

pub struct Tract<T: Model> {
    phantom: PhantomData<T>,
    model: Option<TractModel>,
}

fn load_model(path_to_weights: &str, input_shape: [usize; 4]) -> TractResult<TractModel> {
    tract_onnx::onnx()
        .model_for_path(path_to_weights)?
        .with_input_fact(0, f32::fact(input_shape).into())?
        .eliminate_dead_branches()?
        .with_output_fact(0, InferenceFact::default())?
        .into_typed()?
        .into_compact()?
        .into_optimized()?
        .into_runnable()
}

impl<M: Model> Backend for Tract<M> {
    type Model = M;
    type Input = SmallVec<[TValue; 4]>;

    fn new() -> Self {
        Self {
            phantom: Default::default(),
            model: Default::default(),
        }
    }

    fn load_model(&mut self) {
        let path_to_weights: &Path = Self::Model::get_onnx_weights();
        let input_shape = Self::Model::get_input_shape();
        self.model = Some(load_model(path_to_weights.to_str().unwrap(), input_shape).unwrap());
    }

    fn generate_input(&self) -> Self::Input {
        let shape = Self::Model::get_input_shape();
        let input: Tensor = Array::zeros(shape).mapv(|elem: u8| elem as f32).into();
        tvec!(input.into())
    }

    fn predict(&mut self, input: Self::Input) {
        self.model.as_ref().unwrap().run(input).unwrap();
    }
}
