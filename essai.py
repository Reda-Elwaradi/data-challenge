from torchvision.models import convnext_small, ConvNeXt_Small_Weights
model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
print(model)