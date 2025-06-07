import torch
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import get_model
from dataset import get_dataloaders

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

precision, recall, _ = precision_recall_curve(y_true, y_probs)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, lw=2, color='blue', label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig("plots/pr_curve.png")
plt.show()

