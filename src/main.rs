use nshare::ToNdarray3;
use openvino::{Blob, Core, Layout, Precision, TensorDesc};
use std::mem;

const ONNX_PATH: &str = "res/resnet.onnx";
// const BIN_PATH: &str = "res/hresnet.bin";
// const XML_PATH: &str = "res/hresnet.xml";
const IMG_PATH: &str = "res/elephants.jpg";

const IMG_DIMS: [usize; 4] = [1, 3, 224, 224];
const IMAGENET_AVG: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

fn load_img() -> Vec<u8> {
    let img = image::open(IMG_PATH).unwrap().to_rgb8();
    let img = image::imageops::resize(
        &img,
        IMG_DIMS[2] as u32,
        IMG_DIMS[3] as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.into_ndarray3().mapv(|elem| elem as f32);

    let imagenet_avg = ndarray::arr3(&[[IMAGENET_AVG]])
        .into_shape([3, 1, 1])
        .unwrap()
        .mapv(|elem| elem as f32);

    let imagenet_std = ndarray::arr3(&[[IMAGENET_STD]])
        .into_shape([3, 1, 1])
        .unwrap()
        .mapv(|elem| elem as f32);

    let mut img = (img - imagenet_avg) / imagenet_std;
    println!("Image shape: {:?}", img.shape());

    unsafe {
        let ratio = mem::size_of::<u32>() / mem::size_of::<u8>();
        let length = img.len() * ratio;
        let ptr = img.as_mut_ptr() as *mut u8;
        mem::forget(img);
        Vec::from_raw_parts(ptr, length, length)
    }
}

fn main() {
    // Instantiate core
    let mut core = Core::new(None).unwrap();

    // Read network
    // TODO: Look into advantages of using IR vs ONNX
    // let m = core.read_network_from_file(XML_PATH, BIN_PATH).unwrap();
    let m = core.read_network_from_file(ONNX_PATH, "AUTO").unwrap();
    let mut model = core.load_network(&m, "CPU").unwrap();

    // Read image
    let img: Vec<u8> = load_img();

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

    // Load output
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

    println!("{:?}", b.len());
}
