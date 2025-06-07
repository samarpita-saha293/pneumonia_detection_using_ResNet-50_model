# Pneumonia_detection_using_ResNet-50_model

This project implements a deep learning pipeline to classify chest X-ray images as either Normal or Pneumonia. The pipeline uses a fine-tuned ResNet-50 model on the [PneumoniaMNIST dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist/data) and is designed to work efficiently even on CPU setups.

For running the codes
### Prerequisites
Python version recommended: 3.8+
- Pytorch
- bcftools
- 

### Dataset
1. __dataset.py__ - Custom dataset loader and preprocessing
2. __model.py__ - Loads and modifies pretrained ResNet-50
3. __utils.py__ - Utility functions for evaluation
4. __train.py__ - Fine-tunes the model and saves it
5. __eval.py__ - Loads the trained model and reports metrics
6. __pneumoniamnist.npz__ - Dataset file (to be downloaded separately)
7. __saved_model__ - Stores the trained model checkpoint

We use the PneumoniaMNIST dataset in .npz format with NumPy arrays for training, validation, and testing.

Before working on the project you need to login to your project account using your DNAnexus account credentials
```
dx login
```
Your credentials will be acquired from https://auth.dnanexus.com. While logging in you will be asked to choose from the list of available projects to work on.

Use dx login --timeout to control the expiration date, or dx logout to end this session.

## STEP 1: SAS sample extraction
