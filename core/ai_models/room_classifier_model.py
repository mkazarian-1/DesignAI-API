import torch.nn as nn
import torchvision.models as models

class RoomClassifierModel:
    @staticmethod
    def build_model(num_classes, device):
        model = models.efficientnet_b1(weights='DEFAULT')

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )

        # Move model to device
        model = model.to(device)
        return model