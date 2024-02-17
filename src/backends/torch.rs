use crate::{Backend, Model};
use std::default::Default;
use std::marker::PhantomData;

use tch::{nn::VarStore, CModule, Device, Kind, Tensor};

pub struct Torch<T: Model> {
    phantom: PhantomData<T>,
    variable_storage: VarStore,
    model: Option<CModule>,
}

impl<M: Model> Backend for Torch<M> {
    type Model = M;
    type Input = Tensor;

    fn new() -> Self {
        Self {
            phantom: Default::default(),
            variable_storage: VarStore::new(Device::Cpu),
            model: Default::default(),
        }
    }

    fn load_model(&mut self) {
        let model = tch::CModule::load(Self::Model::get_pytorch_weights()).unwrap();
        self.model = Some(model);
    }

    fn generate_input(&self) -> Self::Input {
        let shape = Self::Model::get_input_shape();
        Tensor::zeros(
            &[
                shape[0] as i64,
                shape[1] as i64,
                shape[2] as i64,
                shape[3] as i64,
            ],
            (Kind::Float, self.variable_storage.device()),
        )
    }

    fn predict(&mut self, input: Self::Input) {
        let _ = self
            .model
            .as_ref()
            .unwrap()
            .forward_ts(&[input])
            .unwrap()
            .softmax(-1, None);
    }
}
