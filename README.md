# __Pneumonia_detection_using_ResNet-50_model__

This project implements a deep learning pipeline to classify chest X-ray images as either Normal or characterised by Pneumonia. The pipeline uses a fine-tuned ResNet-50 model on the [PneumoniaMNIST dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data) and is designed to work efficiently even on CPU setups.

### Tutorials and Reproducibility section
We use the PneumoniaMNIST dataset in .npz format with NumPy arrays for training, validation, and testing. To Obtain the .npz File go to the  [PneumoniaMNIST dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data) and place it in the root directory of this project.

__Please compile in order :__
1. [dataset.py](dataset.py) - Custom dataset loader and preprocessing
2. [model.py](model.py) - Loads and modifies pretrained ResNet-50
3. [utils.py](utils.py) - Utility functions for evaluation
4. [train.py](train.py) - Trains, fine-tunes the model and saves it
5. [eval.py](eval.py) - Loads the trained model and reports metrics
6. [eval_met.py](eval_met.py) - Prints out the model metrics
7. [eval_miscl.py](eval_miscl.py) - Generates misclassified images
8. [pr_curve.py](pr_curve.py) - Generates the Precision-Recall curve (Highlights performance under class imbalance by focusing on positive class predictions)
9. [eval_roc.py](eval_roc.py) - Generates the ROC curve

Lastly, __saved_model__ stores the trained model checkpoint.

## Code Description

### STEP 1: Data Loading & Preprocessing ([dataset.py](dataset.py))

First, load train/val/test images and labels from a NumPy zipped file.

```
data = np.load("pneumoniamnist.npz")
```

The dataset includes:

- train_images, train_labels
- val_images, val_labels
- test_images, test_labels

Next, Converts grayscale images to 3 channels using Grayscale(num_output_channels=3). The images are then Resized images to 128×128 (smaller than 224×224 to reduce CPU load) and we Convert NumPy arrays to tensors for model input.
```
class PneumoniaMNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
```

Wrapping datasets into PyTorch DataLoader objects for batched loading
```
def get_dataloaders(batch_size=8):
    ...
    return train_loader, val_loader, test_loader
```

### STEP 2: Model Setup (model.py)

Load Pretrained ResNet-50 pretrained on ImageNet. This allows transfer learning on X-ray images.
```
model = models.resnet50(pretrained=True)
```

Modify Output Layer for Binary Classification (classes: Normal, Pneumonia)
```
model.fc = nn.Linear(model.fc.in_features, 2)
``` 

### STEP 3: Utility Functions (utils.py)

The code snippet below returns a dictionary of metrics given prediction and ground truth arrays
```
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }
```

Encapsulate test logic for reusability in both training and evaluation
```
def evaluate_model(model, dataloader, device):
    ...
    return compute_metrics(all_labels, all_preds)
```

### STEP 4: Model Training (train.py)

Compute weights to penalize the majority class less, and balance the loss
```
pos = sum([label for _, label in train_loader.dataset])
neg = len(train_loader.dataset) - pos
class_weights = torch.tensor([1.0, neg / pos], dtype=torch.float32).to(device)
```

Define Loss Function and Optimizer
```
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
```

- Uses weighted cross-entropy loss
- Adam optimizer with learning rate 1e-4 for fine-tuning

For each epoch, the model is trained on mini-batches of data, evaluated on the validation set, and performance metrics such as accuracy and F1-score are reported.
```
for epoch in range(epochs):
    ...
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Epoch {epoch+1}/{epochs} - Val Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
```

Model is saved to saved_model/ for future inference.
```
torch.save(model.state_dict(), "saved_model/resnet50_cpu.pth")
```

We applied data augmentation to the training set to improve generalization and reduce overfitting. This included random flips, rotations, and resized crops to simulate variability in X-ray images. It was Implemented in dataset.py file.
```
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
    transforms.Normalize([0.5]*3, [0.5]*3)
])
```

### STEP 5: Evaluation (eval.py)

Load the trained model and test data.
```
model.load_state_dict(torch.load("saved_model/resnet50_cpu.pth", map_location=device))
_, _, test_loader = get_dataloaders(batch_size=8)
```

Run the model in evaluation mode with gradient tracking turned off
```
with torch.no_grad():
    for x, y in test_loader:
        ...
        preds = torch.argmax(out, dim=1)
```

Finally, use scikit-learn to compute evaluation metrics
```
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
```

➤ Sample Output

Test Set Evaluation:
  Accuracy : 0.9345
  Precision: 0.9123
  Recall   : 0.9501
  F1 Score : 0.9308

### Hyperparameter Choices

| __Parameter__ |	__Value__	| __Rationale__ |
|-----------|-------|-----------|
| Learning Rate |	1e-4 | Suitable for fine-tuning ResNet |
| Batch Size	| 8 |	Memory-efficient for CPU |
| Epochs	| 50	| Adequate for convergence |
| Optimizer |	Adam	| Adaptive and effective |
| Loss Function |	CrossEntropy + class weights |	To handle class imbalance |
| Image Size	| 128x128	| Reduces memory load while preserving structure |

