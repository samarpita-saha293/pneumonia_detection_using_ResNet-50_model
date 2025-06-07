# Pneumonia_detection_using_ResNet-50_model

This project implements a deep learning pipeline to classify chest X-ray images as either Normal or characterised by Pneumonia. The pipeline uses a fine-tuned ResNet-50 model on the [PneumoniaMNIST dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data) and is designed to work efficiently even on CPU setups.

### Dataset
1. __dataset.py__ - Custom dataset loader and preprocessing
2. __model.py__ - Loads and modifies pretrained ResNet-50
3. __utils.py__ - Utility functions for evaluation
4. __train.py__ - Fine-tunes the model and saves it
5. __eval.py__ - Loads the trained model and reports metrics
6. __pneumoniamnist.npz__ - Dataset file (to be downloaded separately)
7. __saved_model__ - Stores the trained model checkpoint

We use the PneumoniaMNIST dataset in .npz format with NumPy arrays for training, validation, and testing.

## STEP 1: Data Loading & Preprocessing (dataset.py)

```
data = np.load("pneumoniamnist.npz")
```
The code above is used to load train/val/test images and labels from a NumPy zipped file.

The dataset includes:

- train_images, train_labels
- val_images, val_labels
- test_images, test_labels
- 
### Hyperparameter Choices

| __Parameter__ |	__Value__	| __Rationale__ |
|-----------|-------|-----------|
| Learning Rate |	1e-4 | Suitable for fine-tuning ResNet |
| Batch Size	| 8 |	Memory-efficient for CPU |
| Epochs	| 50	| Adequate for convergence |
| Optimizer |	Adam	| Adaptive and effective |
| Loss Function |	CrossEntropy + class weights |	To handle class imbalance |
| Image Size	| 128x128	| Reduces memory load while preserving structure |

