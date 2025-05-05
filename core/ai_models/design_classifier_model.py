import torch.nn as nn
import torchvision.models as models

class DesignClassifierModel:
    @staticmethod
    def build_model(num_classes, device):
        model = models.convnext_small(weights='DEFAULT')

        feature_dim = 768
        model.features.append(AttentionBlock(feature_dim))

        model.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )

        model = model.to(device)
        return model

class AttentionBlock(nn.Module):
    def __init__(self, in_features, reduction_factor=16):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, in_features//reduction_factor, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_features//reduction_factor, in_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights