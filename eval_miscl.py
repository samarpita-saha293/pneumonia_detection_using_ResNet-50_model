import torch
import matplotlib.pyplot as plt
from model import get_model
from dataset import get_dataloaders
import torchvision

_, _, test_loader = get_dataloaders(batch_size=1)

model = get_model()
model.load_state_dict(torch.load("saved_model/resnet50_cpu.pth", map_location='cpu'))
model.eval()

misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        if preds != labels:
            misclassified.append((images[0], labels.item(), preds.item()))
        if len(misclassified) >= 16:
            break

plt.figure(figsize=(12, 8))
for i, (img, true_lbl, pred_lbl) in enumerate(misclassified):
    plt.subplot(4, 4, i+1)
    img = img.permute(1, 2, 0).numpy()
    plt.imshow(img[:, :, 0], cmap='gray')  # Display 1 channel
    plt.title(f"T: {true_lbl} | P: {pred_lbl}", fontsize=10)
    plt.axis('off')
plt.suptitle("Misclassified Images (T=True, P=Pred)", fontsize=16)
plt.tight_layout()
plt.savefig("plots/misclassified.png")
plt.show()

