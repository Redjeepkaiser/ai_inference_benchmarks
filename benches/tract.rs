use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tract_ndarray::Array;
use tract_onnx::prelude::*;

const MODEL_PATH: &str = "res/resnet.onnx";
const IMG_PATH: &str = "res/elephants.jpg";

const IMG_DIMS: [usize; 4] = [1, 3, 224, 224];
const IMAGENET_AVG: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

type TractModel = SimplePlan<
    TypedFact,
    Box<dyn TypedOp>,
    tract_onnx::prelude::Graph<TypedFact, Box<dyn TypedOp>>,
>;

fn load_model() -> TractResult<TractModel> {
    tract_onnx::onnx()
        .model_for_path(MODEL_PATH)?
        .with_input_fact(0, f32::fact(IMG_DIMS).into())?
        .into_optimized()?
        .into_runnable()
}

fn predict(model: &TractModel, img: Tensor) {
    let _res = model.run(tvec!(img.into()));
}

fn criterion_benchmark(c: &mut Criterion) {
    let model = load_model().expect("Could not load model!");

    // ** load input image **
    // Imagenet mean and standard deviation
    let avg = Array::from_shape_vec((1, 3, 1, 1), IMAGENET_AVG.to_vec()).unwrap();
    let std = Array::from_shape_vec((1, 3, 1, 1), IMAGENET_STD.to_vec()).unwrap();

    let img = image::open(IMG_PATH).unwrap().to_rgb8();
    let img = image::imageops::resize(
        &img,
        IMG_DIMS[2] as u32,
        IMG_DIMS[3] as u32,
        image::imageops::FilterType::Triangle,
    );
    let img: Tensor = ((tract_ndarray::Array4::from_shape_fn(IMG_DIMS, |(_, c, y, x)| {
        img[(x as _, y as _)][c] as f32 / 255.0
    }) - avg)
        / std)
        .into();
    // ****

    let mut group = c.benchmark_group("tract");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.significance_level(0.1).sample_size(10);
    // TODO: not use deep clone, or bench deep clone separately
    group.bench_function("model predict", |b| {
        b.iter(|| predict(black_box(&model), black_box(img.deep_clone())))
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
