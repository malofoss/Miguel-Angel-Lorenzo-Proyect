import torch
import torch.nn as nn
from torchvision import models

class AgeEstimatorCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(AgeEstimatorCNN, self).__init__()

        # ResNet18 preentrenada como base
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Extraer features antes de la capa final
        in_features = self.backbone.fc.in_features

        # Cabeza compartida para extraer features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # Cabeza de clasificación binaria (mayor/menor de edad)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Cabeza de regresión para edad aproximada
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        features = self.backbone(x)  # (batch, 64)
        class_prob = self.classifier(features).squeeze(1)  # (batch,) probabilidad mayor de edad
        age_pred = self.regressor(features).squeeze(1)  # (batch,) edad aproximada
        return class_prob, age_pred


def build_model(dropout=0.5):
    model = AgeEstimatorCNN(dropout=dropout)
    return model


if __name__ == "__main__":
    model = build_model()
    print(model)

    # Prueba rápida con un batch falso
    dummy_input = torch.randn(4, 3, 224, 224)  # 4 imágenes, 3 canales, 224x224 (ResNet standard)
    class_prob, age_pred = model(dummy_input)
    print(f"Input shape:       {dummy_input.shape}")
    print(f"Class prob shape:  {class_prob.shape}")  # Esperado: torch.Size([4])
    print(f"Age pred shape:    {age_pred.shape}")    # Esperado: torch.Size([4])
    print(f"Probabilidades mayor de edad: {class_prob}")
    print(f"Edades aproximadas: {age_pred}")