import torch
import torchvision

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.resnet18(pretrained=True)
    torch.onnx.export(model, dummy_input, "res/resnet.onnx")