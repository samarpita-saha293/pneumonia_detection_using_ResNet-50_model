# Pneumonia_detection_using_ResNet-50_model

This project fine-tunes a pretrained ResNet-50 model to classify chest X-ray images into two classes: Normal and Pneumonia. It leverages transfer learning to achieve high accuracy using a publicly available dataset.

### Directory Structure
### Prerequisites
Python version recommended: 3.8+
- Python 3.8 or higher
- bcftools

The DNAnexus dxpy Python library offers Python bindings for interacting with the DNAnexus Platform through its API. The dxpy package, part of the DNAnexus platform SDK, requires Python 3.8 or higher. We used python3.11.4 module and bcftools-1.18 on the Institute's HPC system.

Before working on the project you need to login to your project account using your DNAnexus account credentials
```
dx login
```
Your credentials will be acquired from https://auth.dnanexus.com. While logging in you will be asked to choose from the list of available projects to work on.

Use dx login --timeout to control the expiration date, or dx logout to end this session.

## STEP 1: SAS sample extraction
