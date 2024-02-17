use ai_benchmarks::{backends::openvino::OpenVINO, models::resnet18::ResNet18, Backend};

fn main() {
    let mut backend = OpenVINO::<ResNet18>::new();
    backend.load_model();
    let input = backend.generate_input();
    backend.predict(input);
}

// use tch::Tensor;
// use tch::{
//     nn::VarStore,
//     vision::{imagenet, resnet::resnet18},
//     Device, Kind,
// };
//
// const IMG_PATH: &str = "./data/inputs/elephants.jpg";
// const MODEL_PATH: &str = "./data/models/resnet18.safetensors";
// const IMG_DIMS: [usize; 4] = [1, 3, 224, 224];
//
// fn main() {
//     let mut vs = VarStore::new(Device::Cpu);
//     let resnet18 = resnet18(&vs.root(), imagenet::CLASS_COUNT);
//     vs.load(MODEL_PATH).unwrap();
//
//     let image: Vec<f32> = imagenet::load_image_and_resize224(IMG_PATH)
//         .unwrap()
//         .to_device(vs.device());
//
//     println!("{:?}", image);
//
//     // Apply the forward pass of the model to get the logits
//     let output = image
//         .unsqueeze(0)
//         .apply_t(&resnet18, false)
//         .softmax(-1, Kind::Float);
//
//     // Print the top 5 categories for this image.
//     for (probability, class) in imagenet::top(&output, 5).iter() {
//         println!("{:50} {:5.2}%", class, 100.0 * probability)
//     }
// }
