import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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

class Modele(nn.Module):
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
            # ⚠️ On a retiré nn.Sigmoid() d'ici !
        )
        
        if pretrained is not None:
            checkpoint = torch.load(pretrained)
            if 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        # On applique la Sigmoid manuellement
        out = torch.sigmoid(logits)
        # On empêche la sortie de toucher exactement 0.0 ou 1.0
        out = torch.clamp(out, min=1e-6, max=1.0 - 1e-6)
        return out
    
    def unfreeze_blocks(self, num_blocks):
        total_blocks = len(self.backbone.features)
        num_blocks = min(num_blocks, total_blocks)
        start_idx = total_blocks - num_blocks

        for idx, block in enumerate(self.backbone.features):
            if idx >= start_idx:
                for param in block.parameters():
                    param.requires_grad = True