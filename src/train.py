import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.dataset import UTKFaceDataset, get_transforms
from src.model import build_model

# Archivo de log para tracking del progreso
LOG_FILE = "training_log.txt"

def log_progress(msg):
    """Escribe progreso a archivo y consola."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)


def compute_metrics(preds, labels, threshold=0.5):
    """Calcula métricas de clasificación binaria."""
    preds_binary = (preds >= threshold).float()
    labels = labels.float()

    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((preds_binary == 1) & (labels == 1)).sum().item()
    fp = ((preds_binary == 1) & (labels == 0)).sum().item()
    tn = ((preds_binary == 0) & (labels == 0)).sum().item()
    fn = ((preds_binary == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1


def train(config=None):
    # Hiperparámetros (serán optimizados por el AG más adelante)
    if config is None:
        config = {
            "learning_rate": 0.0001,
            "batch_size": 32,
            "dropout": 0.2,
            "epochs": 20
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")

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

    # Modelo, funciones de pérdida y optimizador
    model = build_model(dropout=config["dropout"]).to(device)
    criterion_class = nn.BCELoss()  # Clasificación binaria
    criterion_age = nn.L1Loss()     # Regresión de edad (auxiliar)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Peso para combinar ambas pérdidas (priorizar clasificación)
    alpha = 0.7  # 70% clasificación, 30% regresión

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(config["epochs"]):
        # --- Entrenamiento ---
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for images, age_binary, age_real in train_loader:
            images = images.to(device)
            age_binary = age_binary.to(device)
            age_real = age_real.to(device)

            optimizer.zero_grad()
            class_pred, age_pred = model(images)

            # Pérdida combinada
            loss_class = criterion_class(class_pred, age_binary)
            loss_age = criterion_age(age_pred, age_real)
            loss = alpha * loss_class + (1 - alpha) * loss_age

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(class_pred.detach())
            train_labels.append(age_binary.detach())

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_acc, train_prec, train_rec, train_f1 = compute_metrics(train_preds, train_labels)

        # --- Validación ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, age_binary, age_real in val_loader:
                images = images.to(device)
                age_binary = age_binary.to(device)
                age_real = age_real.to(device)

                class_pred, age_pred = model(images)

                loss_class = criterion_class(class_pred, age_binary)
                loss_age = criterion_age(age_pred, age_real)
                loss = alpha * loss_class + (1 - alpha) * loss_age

                val_loss += loss.item()
                val_preds.append(class_pred)
                val_labels.append(age_binary)

        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_acc, val_prec, val_rec, val_f1 = compute_metrics(val_preds, val_labels)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        log_progress(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.2%}")
        log_progress(f"              Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.2%}")

        # Guardar el mejor modelo (basado en accuracy de validación)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_model.pth")
            log_progress(f"  [OK] New best model saved (Val Acc: {val_acc:.2%})")

    log_progress(f"\nTraining completed. Best Val Acc: {best_val_acc:.2%}")
    return best_val_loss


if __name__ == "__main__":
    train()