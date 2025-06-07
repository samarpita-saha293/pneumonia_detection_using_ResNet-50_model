import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from model import get_model  # Make sure this returns a ResNet50 model
from dataset import get_dataloaders  # Loads train/val/test DataLoaders

# Load test data only
_, _, test_loader = get_dataloaders(batch_size=8)

# Load model and weights
model = get_model()
model.load_state_dict(torch.load("saved_model/resnet50_cpu.pth", map_location='cpu'))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

# Compute metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\nTest Set Evaluation:")
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1 Score : {f1:.4f}")

# Confusion matrix with dynamic text color
cm = confusion_matrix(y_true, y_pred)
class_names = ['Normal', 'Pneumonia']

plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm,
                 annot=False,
                 fmt='d',
                 cmap='Blues',
                 xticklabels=class_names,
                 yticklabels=class_names,
                 cbar=False,
                 linewidths=0.5,
                 linecolor='gray')

# Dynamic annotation for better contrast
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        value = cm[i, j]
        text_color = 'white' if value > cm.max() / 2 else 'black'
        ax.text(j + 0.5, i + 0.5, str(value),
                ha='center', va='center', color=text_color, fontsize=16)

plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout()
plt.savefig("plots/confusion.png")
plt.show()

