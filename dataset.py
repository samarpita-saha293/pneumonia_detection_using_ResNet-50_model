import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Load the data from the .npz file (adjust path if needed)
data = np.load("pneumoniamnist.npz")

train_images = data["train_images"]
train_labels = data["train_labels"].flatten()

val_images = data["val_images"]
val_labels = data["val_labels"].flatten()

test_images = data["test_images"]
test_labels = data["test_labels"].flatten()

to_3channel = transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert 1-channel → 3-channel

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    to_3channel,
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize all 3 channels
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    to_3channel,
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ✅ Custom dataset
class PneumoniaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])
        img = np.uint8(img * 255)  # Rescale [0,1] to [0,255] for PIL
        if self.transform:
            img = self.transform(img)
        return img, label

# ✅ Dataloader setup function
def get_dataloaders(batch_size=8):
    train_dataset = PneumoniaDataset(train_images, train_labels, transform=train_transform)
    val_dataset   = PneumoniaDataset(val_images, val_labels, transform=test_transform)
    test_dataset  = PneumoniaDataset(test_images, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 6)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers = 6)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 6)

    return train_loader, val_loader, test_loader

