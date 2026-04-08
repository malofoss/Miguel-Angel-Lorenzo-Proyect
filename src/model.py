import torch
import torch.nn as nn
from torchvision import models

class AgeEstimatorCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(AgeEstimatorCNN, self).__init__()

        # ResNet18 preentrenada como base
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Sustituimos la última capa por una cabeza de regresión
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Salida: un solo número (la edad)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def build_model(dropout=0.5):
    model = AgeEstimatorCNN(dropout=dropout)
    return model


if __name__ == "__main__":
    model = build_model()
    print(model)

    # Prueba rápida con un batch falso
    dummy_input = torch.randn(4, 3, 128, 128)  # 4 imágenes, 3 canales, 128x128
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # Esperado: torch.Size([4])
    print(f"Predicciones de prueba: {output}")