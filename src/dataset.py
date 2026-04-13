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


# Transformaciones estándar para la CNN
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
