# Task 3: Image Classification Using a Pretrained Model

Uses a pretrained MobileNetV2 model with transfer learning to train on image data (CIFAR-10), classify sample images, and display predicted labels with confidence scores.

## Features

- Loads MobileNetV2 pretrained on ImageNet (1.4 million images, 1000 classes)
- Fine-tunes (trains) on CIFAR-10 image dataset (10 classes)
- Classifies sample images and displays predicted labels clearly
- Shows top-3 predictions with confidence bars for each sample
- Prints a brief explanation of how the pretrained model works
- Saves the trained model to disk

## How to Run

```bash
pip install -r requirements.txt
python image_classifier.py
```

## Requirements

- Python 3.8+
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

## CIFAR-10 Classes (10 Categories)

Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck

## Sample Output: Classifying Images & Displaying Predicted Labels

```
  Classifying 10 random test images:

  #    True Label      Predicted Label   Confidence   Result
  -----------------------------------------------------------------
  [+] 1   Cat             Cat               87.3%      CORRECT
  [+] 2   Ship            Ship              92.1%      CORRECT
  [x] 3   Bird            Airplane          45.6%      WRONG
  [+] 4   Truck           Truck             78.9%      CORRECT
  [+] 5   Dog             Dog               65.2%      CORRECT
  [+] 6   Frog            Frog              91.4%      CORRECT
  [x] 7   Deer            Horse             38.7%      WRONG
  [+] 8   Airplane        Airplane          88.0%      CORRECT
  [+] 9   Automobile      Automobile        94.3%      CORRECT
  [+] 10  Horse           Horse             72.6%      CORRECT

  Sample accuracy: 8/10 (80%)

  -------------------------------------------------------
  Top-3 Predictions for first 5 samples:
  -------------------------------------------------------

  Image 1 (True: Cat):
    1. Cat          87.3%  *****************
    2. Dog           8.1%  *
    3. Deer          2.4%

  Image 2 (True: Ship):
    1. Ship         92.1%  ******************
    2. Airplane      4.5%
    3. Automobile    1.8%
```

## How the Pretrained Model Works (In My Own Words)

### What is a Pretrained Model?

A pretrained model is a neural network that has already been trained on a massive dataset. MobileNetV2 was trained on ImageNet (1.4 million images, 1000 categories). During that training, it learned to recognize visual patterns at different levels:

- **Early layers:** Detect simple features like edges, corners, and colors
- **Middle layers:** Detect textures, shapes, and parts of objects
- **Deep layers:** Detect complex structures like faces, wheels, and wings

### What is Transfer Learning?

Instead of training from scratch (which needs millions of images and days of computation), we reuse the pretrained model's knowledge:

1. Take MobileNetV2's learned layers and **freeze** them (keep weights fixed)
2. Add a small custom classifier on top
3. Train **only** the new classifier on our CIFAR-10 images

### Why Does This Work?

Visual features are universal. Edges, textures, and shapes look the same whether you're classifying cats vs dogs or airplanes vs ships. The pretrained model already knows **how to see** — we just teach it **what to look for** in our specific task.

### In Simple Terms

Imagine hiring an expert photographer who already knows how to identify objects in photos. You don't teach them photography from scratch — you just show them your 10 specific categories and say "sort these into the right bins." That's transfer learning.

### Benefits

- Much faster training (minutes instead of hours/days)
- Works well with small datasets (we used only 5,000 images)
- Achieves good accuracy without expensive hardware
- Leverages knowledge from millions of images we don't have access to

## Model Architecture

```
Input (96x96x3 image)
      ↓
MobileNetV2 (frozen, pretrained on ImageNet) → Feature Extraction
      ↓
GlobalAveragePooling2D → Compress features to single vector
      ↓
Dropout (0.3) → Prevent overfitting
      ↓
Dense (128, ReLU) → Learn feature combinations
      ↓
Dropout (0.2) → Prevent overfitting
      ↓
Dense (10, Softmax) → Output probabilities for 10 classes
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Image Size | 96x96 |
| Training Samples | 5,000 |
| Test Samples | 1,000 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

## Technologies

- TensorFlow / Keras
- MobileNetV2 (pretrained on ImageNet)
- CIFAR-10 dataset (auto-downloaded)
- Transfer Learning / Fine-tuning
