import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 96
NUM_CLASSES = 10
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000


def load_and_preprocess_data():
    print("=" * 60)
    print("  STEP 1: Loading and Preprocessing CIFAR-10 Dataset")
    print("=" * 60)
    
    print("\nDownloading/Loading CIFAR-10 dataset...")
    (x_train_full, y_train_full), (x_test_full, y_test_full) = keras.datasets.cifar10.load_data()
    
    print(f"  Full training set: {x_train_full.shape[0]} images")
    print(f"  Full test set: {x_test_full.shape[0]} images")
    print(f"  Image size: {x_train_full.shape[1]}x{x_train_full.shape[2]} pixels, {x_train_full.shape[3]} channels (RGB)")
    
    x_train = x_train_full[:TRAIN_SAMPLES]
    y_train = y_train_full[:TRAIN_SAMPLES]
    x_test = x_test_full[:TEST_SAMPLES]
    y_test = y_test_full[:TEST_SAMPLES]
    
    print(f"\n  Using subset for training: {TRAIN_SAMPLES} images")
    print(f"  Using subset for testing: {TEST_SAMPLES} images")
    
    print(f"\n  Resizing images from 32x32 to {IMG_SIZE}x{IMG_SIZE}...")
    x_train = tf.image.resize(x_train, (IMG_SIZE, IMG_SIZE)).numpy()
    x_test = tf.image.resize(x_test, (IMG_SIZE, IMG_SIZE)).numpy()
    
    x_train = keras.applications.mobilenet_v2.preprocess_input(x_train)
    x_test = keras.applications.mobilenet_v2.preprocess_input(x_test)
    
    print(f"  Pixel value range after preprocessing: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"  Final shape: {x_train.shape}")
    print("\n  Dataset ready!")
    
    return x_train, y_train, x_test, y_test


def build_model():
    print("\n" + "=" * 60)
    print("  STEP 2: Building the Model (Transfer Learning)")
    print("=" * 60)
    
    print("\n  Loading MobileNetV2 pretrained on ImageNet...")
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    print(f"  Base model loaded: {base_model.name}")
    print(f"  Base model parameters: {base_model.count_params():,} (all frozen)")
    
    print("\n  Adding custom classification layers...")
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n  Model Architecture:")
    print("  " + "-" * 50)
    print(f"  {'Layer':<35} {'Trainable'}")
    print("  " + "-" * 50)
    print(f"  {'MobileNetV2 (feature extractor)':<35} {'No (frozen)'}")
    print(f"  {'GlobalAveragePooling2D':<35} {'Yes'}")
    print(f"  {'Dropout (0.3)':<35} {'Yes'}")
    print(f"  {'Dense (128, relu)':<35} {'Yes'}")
    print(f"  {'Dropout (0.2)':<35} {'Yes'}")
    print(f"  {'Dense (10, softmax)':<35} {'Yes'}")
    print("  " + "-" * 50)
    
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable (frozen): {total_params - trainable_params:,}")
    print("\n  Model built successfully!")
    
    return model


def train_model(model, x_train, y_train, x_test, y_test):
    print("\n" + "=" * 60)
    print("  STEP 3: Training the Model")
    print("=" * 60)
    print(f"\n  Training on {len(x_train)} images for {EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Validation on {len(x_test)} images")
    print()
    
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("\n  Training complete!")
    print(f"  Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    
    return history


def evaluate_model(model, x_test, y_test):
    print("\n" + "=" * 60)
    print("  STEP 4: Evaluating the Model")
    print("=" * 60)
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    return accuracy


def classify_sample_images(model, x_test, y_test, num_samples=10):
    print("\n" + "=" * 60)
    print("  STEP 5: Classifying Sample Images & Displaying Predictions")
    print("=" * 60)
    
    np.random.seed(42)
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    sample_images = x_test[indices]
    sample_labels = y_test[indices].flatten()
    
    predictions = model.predict(sample_images, verbose=0)
    
    print(f"\n  Classifying {num_samples} random test images:\n")
    print(f"  {'#':<4} {'True Label':<15} {'Predicted Label':<17} {'Confidence':<12} {'Result'}")
    print(f"  {'-'*65}")
    
    correct = 0
    for i, idx in enumerate(indices):
        true_label = CLASS_NAMES[sample_labels[i]]
        predicted_class = np.argmax(predictions[i])
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = predictions[i][predicted_class] * 100
        
        is_correct = true_label == predicted_label
        result = "CORRECT" if is_correct else "WRONG"
        marker = "+" if is_correct else "x"
        
        if is_correct:
            correct += 1
        
        print(f"  [{marker}] {i+1:<3} {true_label:<15} {predicted_label:<17} {confidence:>6.1f}%      {result}")
    
    print(f"\n  Sample accuracy: {correct}/{num_samples} ({correct/num_samples*100:.0f}%)")
    
    print("\n  " + "-" * 55)
    print("  Top-3 Predictions for first 5 samples:")
    print("  " + "-" * 55)
    for i in range(min(5, num_samples)):
        true_label = CLASS_NAMES[sample_labels[i]]
        top3_indices = np.argsort(predictions[i])[::-1][:3]
        print(f"\n  Image {i+1} (True: {true_label}):")
        for rank, cls_idx in enumerate(top3_indices):
            label = CLASS_NAMES[cls_idx]
            prob = predictions[i][cls_idx] * 100
            bar = "*" * int(prob / 5)
            print(f"    {rank+1}. {label:<12} {prob:>5.1f}%  {bar}")


def display_training_summary(history):
    print("\n" + "=" * 60)
    print("  Training History Summary")
    print("=" * 60)
    print(f"\n  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<14} {'Val Loss':<14} {'Val Acc'}")
    print(f"  {'-'*62}")
    
    for i in range(len(history.history['loss'])):
        print(f"  {i+1:<8} {history.history['loss'][i]:<14.4f} "
              f"{history.history['accuracy'][i]:<14.4f} "
              f"{history.history['val_loss'][i]:<14.4f} "
              f"{history.history['val_accuracy'][i]:.4f}")


def save_model(model, filepath="trained_model.keras"):
    model.save(filepath)
    print(f"\n  Model saved to: {filepath}")
    print("  You can load it later with: keras.models.load_model(filepath)")


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#   IMAGE CLASSIFICATION WITH PRETRAINED MODEL (TRAINING)  #")
    print("#   Using MobileNetV2 + Transfer Learning on CIFAR-10      #")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    model = build_model()
    
    history = train_model(model, x_train, y_train, x_test, y_test)
    
    display_training_summary(history)
    
    evaluate_model(model, x_test, y_test)
    
    classify_sample_images(model, x_test, y_test, num_samples=10)
    
    save_model(model)
    
    print_model_explanation()
    
    print("\n" + "=" * 60)
    print("  Done! The model has been trained and evaluated.")
    print("=" * 60)


def print_model_explanation():
    print("\n" + "=" * 60)
    print("  HOW THE PRETRAINED MODEL WORKS (In My Own Words)")
    print("=" * 60)
    print("""
  What is a Pretrained Model?
  ---------------------------
  A pretrained model is a neural network that has already been trained
  on a large dataset. In our case, MobileNetV2 was trained on ImageNet,
  which contains 1.4 million images across 1000 categories.

  During that original training, the model learned to recognize visual
  patterns at different levels:
    - Early layers detect simple features: edges, corners, colors
    - Middle layers detect textures, shapes, and parts of objects
    - Deeper layers detect complex structures: faces, wheels, wings

  What is Transfer Learning?
  --------------------------
  Instead of training a model from scratch (which needs millions of
  images and days of computation), we REUSE the pretrained model's
  knowledge. This is called Transfer Learning.

  We take MobileNetV2's learned feature-extraction layers and FREEZE
  them (keep their weights fixed). Then we add our own small classifier
  on top and train ONLY that part on our dataset (CIFAR-10).

  Why Does This Work?
  -------------------
  Visual features are universal. Edges, textures, and shapes look the
  same whether you're classifying cats vs dogs or airplanes vs ships.
  The pretrained model already knows HOW to see - we just teach it
  WHAT to look for in our specific task.

  Benefits:
    - Much faster training (minutes instead of hours/days)
    - Works well with small datasets (we used only 5000 images)
    - Achieves good accuracy without expensive hardware
    - Leverages knowledge from millions of images we don't have

  In Simple Terms:
  ----------------
  Imagine hiring an expert photographer who already knows how to
  identify objects in photos. You don't teach them photography from
  scratch - you just show them your 10 specific categories and say
  "sort these into the right bins." That's transfer learning.
  """)


if __name__ == "__main__":
    main()
