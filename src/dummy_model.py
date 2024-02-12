from pathlib import Path

import openvino
import torch
import torchvision
from openvino.tools.ovc import convert_model

if __name__ == "__main__":
    onnx_path = Path("res/hresnet.onnx")
    ir_path = onnx_path.with_suffix(".xml")

    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.resnet18(pretrained=True)
    torch.onnx.export(model, dummy_input, str(onnx_path))

    if not ir_path.exists():
        print("Exporting ONNX model to IR... This may take a few minutes.")
        ov_model = convert_model(onnx_path)
        openvino.runtime.save_model(ov_model, ir_path)
    else:
        print(f"IR model {ir_path} already exists.")
