First, run *dummy_model.py* to build the model.

Then you can run 
```
cargo bench
```
 to run the benches, add the *--no-run* flag to compile binaries.

### Execution time for 10 forward passes
| Backend/Model     | ResNet18 | MobileNetv2 |
|-------------------|----------|-------------|
| OpenVINO          | 158ms    | 33ms        |
| tch-rs (torch)    | 250ms    | 143ms       |
| tract             | 9050ms   | 1805ms      |