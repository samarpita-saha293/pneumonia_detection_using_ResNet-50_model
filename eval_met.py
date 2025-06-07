import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import get_model
from dataset import get_dataloaders

_, _, test_loader = get_dataloaders(batch_size=8)

model = get_model()
model.load_state_dict(torch.load("saved_model/resnet50_cpu.pth", map_location='cpu'))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.tolist())

print("\nTest Set Evaluation Metrics:")
print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_true, y_pred):.4f}")
print(f"  F1 Score : {f1_score(y_true, y_pred):.4f}")

