# Pneumonia Detection from Chest X Rays with Grad CAM Interpretability

A deep learning pipeline that classifies chest X rays as NORMAL or PNEUMONIA using a fine tuned ResNet18, with Grad CAM used to visually interpret and critically evaluate the model's predictions rather than relying on accuracy alone.

## Overview

This project fine tunes a pretrained ResNet18 on the Kaggle Chest X Ray Images (Pneumonia) dataset, then uses Grad CAM to inspect whether the model's predictions are grounded in clinically relevant lung regions. The analysis surfaces a meaningful gap between validation and test performance, and a possible shortcut learning pattern in the model's attention, findings that are more informative than the raw accuracy number alone.

## Dataset

Source: Kaggle Chest X Ray Images (Pneumonia), https://www.kaggle.com/datasets/paultimothymooney/chest xray pneumonia

Classes: NORMAL, PNEUMONIA

Splits used:
Train: 5,216 images split 85/15 into train (4,433) and validation (783), since the dataset's official validation folder contained only 16 images
Test: 624 images, untouched, used only for final evaluation

## Method

| Component | Choice |
|---|---|
| Backbone | ResNet18 (ImageNet pretrained) |
| Fine tuning | Frozen all layers except layer4 and the final classifier head |
| Loss | Cross Entropy Loss |
| Optimizer | Adam, lr = 1e 4 |
| Epochs | 10 |
| Augmentation | Random rotation, color jitter (train split only) |
| Interpretability | Grad CAM on layer4 |

## Results

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 98.8% | 82% |
| Pneumonia Recall | not applicable | 99% (387/390) |
| Normal Recall | not applicable | 53% (124/234) |

The model rarely misses true pneumonia cases, but frequently misclassifies normal X rays as pneumonia on the test set, a gap not visible in validation, pointing to class imbalance effects and distribution differences between splits.

## Grad CAM Findings

Grad CAM was applied to correctly classified pneumonia cases to check whether the model's attention aligned with actual lung pathology. In several cases, activation concentrated near the upper chest and shoulder region rather than the central and lower lung fields typically associated with pneumonia consolidation, a consistent, position based pattern that suggests the model may be partly relying on shortcut features rather than fully localizing disease. This is a documented risk with this specific dataset.

## Project Structure

pneumonia_gradcam_notebook.ipynb, full pipeline covering data, training, evaluation, and Grad CAM
pneumonia_gradcam_writeup.md, short written summary of the project
best_model.pth, saved best model checkpoint by validation accuracy
README.md, this file

## How to Run

1. Clone the repo and open pneumonia_gradcam_notebook.ipynb in Google Colab, GPU runtime recommended.
2. Download the dataset via the Kaggle API, requires a kaggle.json API token.
3. Run all cells in order: data loading, dataloaders, model setup, training, evaluation, Grad CAM.

## Limitations and Future Work

Address class imbalance directly through weighted loss or resampling rather than relying on augmentation alone.
Evaluate on an external dataset from a different institution to test true generalization beyond this dataset's known distribution quirks.
Use quantitative localization metrics, such as comparing Grad CAM regions against radiologist annotated bounding boxes, instead of visual inspection alone.

## Acknowledgements

Dataset by Paul Mooney on Kaggle, https://www.kaggle.com/datasets/paultimothymooney/chest xray pneumonia, originally sourced from Kermany et al.
