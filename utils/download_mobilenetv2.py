from pathlib import Path

# import openvino as ov
import torch
import torchvision

# from openvino.tools.ovc import convert_model
# from safetensors import torch as stt
from torchvision.models import MobileNet_V2_Weights

if __name__ == "__main__":
    onnx_path = Path("./models/mobilenetv2/mobilenetv2.onnx")
    pt_path = onnx_path.with_suffix(".pt")
    safetensors_path = Path("./models/mobilenetv2/mobilenetv2.safetensors")
    ir_path = onnx_path.with_suffix(".xml")

    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    torch.onnx.export(model, dummy_input, str(onnx_path))

    # stt.save_file(model.state_dict(), str(safetensors_path))

    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(pt_path)

    # ov_model = convert_model(onnx_path)
    # ov.runtime.save_model(ov_model, ir_path)
