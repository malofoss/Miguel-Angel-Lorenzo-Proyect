import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import UTKFaceDataset, get_transforms
from src.model import build_model


def train(config=None):
    # Hiperparámetros (serán optimizados por el AG más adelante)
    if config is None:
        config = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "dropout": 0.5,
            "epochs": 20
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar dataset
    train_transform, val_transform = get_transforms()
    dataset = UTKFaceDataset(
        root_dir="data/raw/UTKFace",
        transform=train_transform
    )

    # Dividir en train (80%) y validación (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Modelo, función de pérdida y optimizador
    model = build_model(dropout=config["dropout"]).to(device)
    criterion = nn.L1Loss()  # MAE: mide el error en años
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        # --- Entrenamiento ---
        model.train()
        train_loss = 0.0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, ages).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train MAE: {train_loss:.2f} años | Val MAE: {val_loss:.2f} años")

        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_model.pth")
            print(f"  ✓ Nuevo mejor modelo guardado (Val MAE: {val_loss:.2f})")

    print(f"\nEntrenamiento completado. Mejor Val MAE: {best_val_loss:.2f} años")
    return best_val_loss


if __name__ == "__main__":
    train()