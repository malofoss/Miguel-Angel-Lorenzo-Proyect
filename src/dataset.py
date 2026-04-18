import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

THRESHOLD = 18  # Edad mínima para ser considerado mayor de edad


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, threshold=THRESHOLD):
        self.root_dir = root_dir
        self.transform = transform
        self.threshold = threshold
        self.images = []
        self.ages = []

        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                parts = filename.split("_")
                if len(parts) >= 3:
                    try:
                        age = int(parts[0])
                        if 0 <= age <= 116:
                            self.images.append(os.path.join(root_dir, filename))
                            self.ages.append(age)
                    except ValueError:
                        continue

        print(f"Dataset cargado: {len(self.images)} imágenes")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        age_real = torch.tensor(self.ages[idx], dtype=torch.float32)
        # Clasificación binaria: 0 = menor de edad, 1 = mayor de edad
        age_binary = torch.tensor(1.0 if self.ages[idx] >= self.threshold else 0.0, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, age_binary, age_real


# Transformaciones estándar para la CNN (con Aumento Extremo para arreglar sesgos de fondo)
def get_transforms():
    train_transform = transforms.Compose([
        # 1. Resolución nativa: Evita distorsionar las texturas (arrugas)
        transforms.Resize(256),
        # 2. Recortes aleatorios y mini-zooms: Hace que el fondo cambie constantemente cada iteración
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # 3. Simetría: Las caras son iguales en espejo
        transforms.RandomHorizontalFlip(p=0.5),
        # 4. Giros: 15 grados compensa las cabezas torcidas
        transforms.RandomRotation(15),
        # 5. Colores: Variar la iluminación ayuda a no clasificar basándose en el color general de la foto
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # 6. Random Erasing: Tapa aleatoriamente con cuadrados negros pedazos de 
        # la foto. Fuerza a la red a no depender de UN SOLO rastro (ej. el pelo)
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # Evaluación de forma clásica y estricta
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
