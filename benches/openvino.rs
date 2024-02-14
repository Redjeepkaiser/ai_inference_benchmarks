use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nshare::ToNdarray3;
use openvino::{Blob, CNNNetwork, Core, ExecutableNetwork, Layout, Precision, TensorDesc};
use rand::prelude::*;
use std::{io::Write, mem, time::Duration};

const ONNX_PATH: &str = "res/resnet.onnx";
// const ONNX_PATH: &str = "./best.onnx";
// const ONNX_PATH: &str = "./idk.onnx";
// const ONNX_PATH: &str = "./fm-classifier.onnx";
// const BIN_PATH: &str = "res/hresnet.bin";
// const XML_PATH: &str = "res/hresnet.xml";
const IMG_PATH: &str = "res/elephants.jpg";
// const IMG_PATH: &str = "./fm-sample.png";

const IMG_DIMS: [usize; 4] = [1, 3, 224, 224];
// const IMG_DIMS: [usize; 4] = [1, 1, 32, 32];
// const IMG_DIMS: [usize; 4] = [1, 3, 640, 640];
// const IMG_DIMS: [usize; 4] = [1, 3, 640, 480];
// const IMAGENET_AVG: [f32; 3] = [0.485, 0.456, 0.406];
// const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn load_img() -> Vec<u8> {
    let img = image::open(IMG_PATH).unwrap().to_rgb8();
    let img = image::imageops::resize(
        &img,
        IMG_DIMS[2] as u32,
        IMG_DIMS[3] as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.into_ndarray3().mapv(|elem| elem as f32);

    // let imagenet_avg = ndarray::arr3(&[[IMAGENET_AVG]])
    //     .into_shape([3, 1, 1])
    //     .unwrap()
    //     .mapv(|elem| elem as f32);
    //
    // let imagenet_std = ndarray::arr3(&[[IMAGENET_STD]])
    //     .into_shape([3, 1, 1])
    //     .unwrap()
    //     .mapv(|elem| elem as f32);

    // let mut img = (img - imagenet_avg) / imagenet_std;
    let mut rng = rand::thread_rng();
    let mut img = img * rng.gen::<f32>();

    unsafe {
        let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
        let length = img.len() * ratio;
        let ptr = img.as_mut_ptr() as *mut u8;
        mem::forget(img);
        Vec::from_raw_parts(ptr, length, length)
    }
}

fn predict(m: &CNNNetwork, model: &mut ExecutableNetwork, img: &Vec<u8>) {
    // Make inference request and set input
    let mut infer_request = model.create_infer_request().unwrap();
    let blob = Blob::new(
        &TensorDesc::new(Layout::NCHW, &IMG_DIMS, Precision::FP32),
        &img,
    )
    .unwrap();
    let input_name = m.get_input_name(0).unwrap();
    infer_request.set_blob(&input_name, &blob).unwrap();

    // Inference
    infer_request.infer().unwrap();

    // Temp
    let output_name = m.get_output_name(0).unwrap();

    let mut output = infer_request.get_blob(&output_name).unwrap();
    let mut output = output.buffer_mut().unwrap().to_vec();

    let b: Vec<f32> = unsafe {
        let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
        let capacity = output.capacity() / ratio;
        let length = output.len() / ratio;
        let ptr = output.as_mut_ptr() as *mut f32;
        mem::forget(output);
        Vec::from_raw_parts(ptr, length, capacity)
    };

    // let mut file = std::fs::File::create("output.txt").unwrap();
    // file.write(format!("{:?}", b).to_string().to_owned().as_bytes())
    //     .unwrap();
}

fn criterion_benchmark(c: &mut Criterion) {
    // Instantiate core
    let mut core = Core::new(None).unwrap();

    // Read network
    // TODO: Look into advantages of using IR vs ONNX
    // let m = core.read_network_from_file(XML_PATH, BIN_PATH).unwrap();
    let m = core.read_network_from_file(ONNX_PATH, "AUTO").unwrap();
    let mut model = core.load_network(&m, "CPU").unwrap();

    // Read image
    let img: Vec<u8> = load_img();

    let mut group = c.benchmark_group("openvino");
    group.significance_level(0.1).sample_size(10);
    group.warm_up_time(Duration::from_nanos(1));
    group.bench_function("model predict", |b| {
        b.iter(|| predict(black_box(&m), black_box(&mut model), black_box(&img)))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

// // TODO: do we want to measure the time this takes as well?
// fn load_output() {
//     // Load output
//     // Converts output back to Vec<f32>
//     let output_name = m.get_output_name(0).unwrap();
//
//     let mut output = infer_request.get_blob(&output_name).unwrap();
//     let mut output = output.buffer_mut().unwrap().to_vec();
//
//     let b: Vec<f32> = unsafe {
//         let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
//         let capacity = output.capacity() / ratio;
//         let length = output.len() / ratio;
//         let ptr = output.as_mut_ptr() as *mut f32;
//         mem::forget(output);
//         Vec::from_raw_parts(ptr, length, capacity)
//     };
//
//     println!("{:?}", b.len());
// }
