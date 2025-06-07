import torch
import torch.nn as nn
import os
import torch.optim as optim
from dataset import get_dataloaders
from model import get_model
from utils import evaluate_model

device = torch.device("cpu")
print("Training on CPU...")

# Load data
train_loader, val_loader, _ = get_dataloaders()

# Handle class imbalance
pos = sum([label for _, label in train_loader.dataset])
neg = len(train_loader.dataset) - pos
class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float32).to(device)

# Model, Loss, Optimizer
model = get_model().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #if (i + 1) % 10 == 0:
            #print(f"  Batch {i+1}/{len(train_loader)} - Loss: {running_loss / (i+1):.4f}")

    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/resnet50_cpu.pth")

