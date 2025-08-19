import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from preprocess import get_datasets

# --- Define Paths ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
models_dir = os.path.join(project_root, 'models')
assets_dir = os.path.join(project_root, 'assets')
os.makedirs(assets_dir, exist_ok=True) # Ensure assets directory exists

# --- Load Datasets ---
_, _, test_ds = get_datasets()

# --- Load Models ---
baseline_model_path = os.path.join(models_dir, 'baseline_model.keras')
transfer_model_path = os.path.join(models_dir, 'transfer_model.keras')
optimized_model_path = os.path.join(models_dir, 'optimized_galaxy_model.keras')

baseline_model = tf.keras.models.load_model(baseline_model_path)
transfer_model = tf.keras.models.load_model(transfer_model_path)
optimized_model = tf.keras.models.load_model(optimized_model_path)


# --- Helper function to get predictions and labels ---
def get_preds_and_labels(model, dataset):
    y_true = []
    y_pred_probs = []
    for images, labels in dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_probs.extend(model.predict(images))
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_true, y_pred

# --- Get Predictions ---
y_true_baseline, y_pred_baseline = get_preds_and_labels(baseline_model, test_ds)
y_true_transfer, y_pred_transfer = get_preds_and_labels(transfer_model, test_ds)
y_true_optimized, y_pred_optimized = get_preds_and_labels(optimized_model, test_ds)

# --- Define Class Names ---
class_names = ['Elliptical', 'Spiral']

# --- Generate Reports ---
print("--- Baseline Model Classification Report ---")
print(classification_report(y_true_baseline, y_pred_baseline, target_names=class_names))

print("\n--- Transfer Learning Model Classification Report ---")
print(classification_report(y_true_transfer, y_pred_transfer, target_names=class_names))

print("\n--- Optimized Model Classification Report ---")
print(classification_report(y_true_optimized, y_pred_optimized, target_names=class_names))

# --- Generate Confusion Matrices ---
def plot_confusion_matrix(y_true, y_pred, title, filename_suffix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Save the plot
    filename = os.path.join(assets_dir, f"cm_{filename_suffix}.png")
    plt.savefig(filename)
    print(f"Saved confusion matrix to {filename}")
    
    plt.show()

plot_confusion_matrix(y_true_baseline, y_pred_baseline, 'Baseline Model Confusion Matrix', 'baseline')
plot_confusion_matrix(y_true_transfer, y_pred_transfer, 'Transfer Learning Model Confusion Matrix', 'transfer')
plot_confusion_matrix(y_true_optimized, y_pred_optimized, 'Optimized Model Confusion Matrix', 'optimized')
