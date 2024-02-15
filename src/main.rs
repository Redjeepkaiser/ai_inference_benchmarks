use ai_benchmarks::BenchmarkNetwork;
use ai_benchmarks::ModelTypes;

fn main() {
    let model_type = ModelTypes::ResNet18;
    let t = BenchmarkNetwork::new(model_type);
}
