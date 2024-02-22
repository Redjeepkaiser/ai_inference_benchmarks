use ai_benchmarks::{
    backends::Tract,
    models::bound_detect::BoundDetect,
    Backend,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tract");

    // ResNet18
    let mut backend = Tract::<BoundDetect>::new();
    backend.load_model();
    let input = backend.generate_input();

    group.significance_level(0.1).sample_size(10);
    group.bench_function("ResNet18 performance", |b| {
        b.iter(|| backend.predict(black_box(input.clone())))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
