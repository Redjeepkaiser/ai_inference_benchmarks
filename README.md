Then you can run to run all benchmarks
```
cargo bench --all-feature
```
, add the *--no-run* flag to compile binaries.

Supported features flags are `tract`, `openvino` and `torch`.

To run only one supported benchmark, for example torch, run the following command:
```
cargo bench -F torch
```
