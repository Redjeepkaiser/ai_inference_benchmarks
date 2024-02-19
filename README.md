## Get started
You can run all benchmarks with:
```
cargo bench --all-feature
```
Add the *--no-run* flag to compile binaries.

Supported features flags are `tract`, `openvino` and `torch`.

To run only one supported benchmark, for example torch, run the following command:
```
cargo bench -F torch
```

## Results on NAOV6
### Average execution time for a single forward pass utilizing all cores (after warm-up)
| Backend/Model     | ResNet18 | MobileNetv2 |
|-------------------|----------|-------------|
| OpenVINO          | 158ms    | 33ms        |
| tch-rs (torch)    | 250ms    | 143ms       |
| tract             | 9050ms   | 1805ms      |
