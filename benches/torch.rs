use ai_benchmarks::{
    backends::Torch,
    models::{MobileNetV2, ResNet18},
    Backend,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("torch");

    // Resnet18
    let mut backend = Torch::<ResNet18>::new();
    backend.load_model();
    let input = backend.generate_input();

    group.significance_level(0.1).sample_size(10);
    group.bench_function("ResNet18 performance", |b| {
        b.iter(|| backend.predict(black_box(input.clone(&input))))
    });

    // MobileNetV2
    let mut backend = Torch::<MobileNetV2>::new();
    backend.load_model();
    let input = backend.generate_input();

    group.significance_level(0.1).sample_size(10);
    group.bench_function("MobileNetV2 performance", |b| {
        b.iter(|| backend.predict(black_box(input.clone(&input))))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
