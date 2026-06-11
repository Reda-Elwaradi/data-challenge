import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights, resnet18, ResNet18_Weights, convnext_small, ConvNeXt_Small_Weights

class IdemiaLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6 

    def forward(self, preds, targets, genders):
        preds = preds.view(-1)
        targets = targets.view(-1)
        genders = genders.view(-1)
        weights = (1.0 / 30.0) + targets

        mask_f = (genders == 0.0).float()
        mask_m = (genders == 1.0).float()

        sum_weights_f = torch.sum(weights * mask_f)
        err_f = torch.sum(mask_f * weights * (preds - targets)**2) / (sum_weights_f + self.eps)

        sum_weights_m = torch.sum(weights * mask_m)
        err_m = torch.sum(mask_m * weights * (preds - targets)**2) / (sum_weights_m + self.eps)

        score = (err_f + err_m) / 2.0 + torch.abs(err_f - err_m)
        return score

class EfficientNetModel(nn.Module):
    def __init__(self, pretrained, do, freeze):
        super().__init__()

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained is None else None)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.max_blocks = len(self.backbone.features)
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        out = torch.sigmoid(logits)
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)
        return out
    
    def unfreeze_blocks(self, num_blocks):
        num_blocks = min(num_blocks, self.max_blocks)
        start_idx = self.max_blocks - num_blocks

        for idx, block in enumerate(self.backbone.features):
            if idx >= start_idx:
                for param in block.parameters():
                    param.requires_grad = True

class ResNet18Model(nn.Module):
    def __init__(self, pretrained, do, freeze):
        super().__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained is None else None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.max_blocks = 4

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        out = torch.sigmoid(logits)
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)
        return out
    
    def unfreeze_blocks(self, num_blocks):
        num_blocks = min(num_blocks, self.max_blocks)
        start_idx = self.max_blocks - num_blocks
        blocks_to_unfreeze = [f"layer{i}" for i in range(start_idx, self.max_blocks + 1)]
        for name, param in self.named_parameters():
            if any(block_name in name for block_name in blocks_to_unfreeze):
                param.requires_grad = True

class ResNet50Model(nn.Module):
    def __init__(self, pretrained, do, freeze):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained is None else None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.max_blocks = 16

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        out = torch.sigmoid(logits)
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)
        return out
    
    def unfreeze_blocks(self, num_blocks):
        num_blocks = min(num_blocks, self.max_blocks)
        start_idx = self.max_blocks - num_blocks
        blocks_to_unfreeze = [f"layer{i}" for i in range(start_idx, self.max_blocks + 1)]
        for name, param in self.named_parameters():
            if any(block_name in name for block_name in blocks_to_unfreeze):
                param.requires_grad = True

class ConvNext(nn.Module):
    def __init__(self, pretrained, do, freeze):
        super().__init__()

        self.backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT if pretrained is None else None)
        self.feature_dim = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity()
        self.max_blocks = 8

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.fc = nn.Sequential(
            nn.Dropout(do),
            nn.Linear(self.feature_dim, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        out = torch.sigmoid(logits)
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)
        return out
    
    def unfreeze_blocks(self, num_blocks):
        num_blocks = min(num_blocks, self.max_blocks)
        start_idx = self.max_blocks - num_blocks
        for idx in range(start_idx, self.max_blocks):
            for param in self.backbone.features[idx].parameters():
                param.requires_grad = True

def get_model(model_name, pretrained, do, freeze):
    if model_name == 'efficientnet_b0':
        model = EfficientNetModel(pretrained, do, freeze)
    elif model_name == 'resnet18':
        model = ResNet18Model(pretrained, do, freeze)
    elif model_name == 'resnet50':
        model = ResNet50Model(pretrained, do, freeze)
    elif model_name == 'convnext':
        model = ConvNext(pretrained, do, freeze)
    else:
        raise ValueError(f"Model {model_name} non reconnu. Choisissez 'efficientnet_b0', 'resnet18', 'resnet50' ou 'convnext'.")
    if pretrained is not None:
        checkpoint = torch.load(pretrained)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    return model