#[cfg(feature = "openvino")]
pub mod openvino;
#[cfg(feature = "openvino")]
pub use openvino::OpenVINO;

#[cfg(feature = "tract")]
pub mod tract;
#[cfg(feature = "tract")]
pub use tract::Tract;

#[cfg(feature = "torch")]
pub mod torch;
#[cfg(feature = "torch")]
pub use torch::Torch;
