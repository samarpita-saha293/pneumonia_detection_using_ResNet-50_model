import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from model import get_model
from dataset import get_dataloaders
import torch.nn.functional as F

_, _, test_loader = get_dataloaders(batch_size=8)

model = get_model()
model.load_state_dict(torch.load("saved_model/resnet50_cpu.pth", map_location='cpu'))
model.eval()

y_true, y_probs = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)[:, 1]
        y_probs.extend(probs.tolist())
        y_true.extend(labels.tolist())

fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.show()

