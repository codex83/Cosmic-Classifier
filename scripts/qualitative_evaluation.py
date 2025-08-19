import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

# --- Define Paths ---
script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
models_dir = os.path.join(project_root, 'models')

# --- Load Best Model ---
# Our final results showed the baseline model was the best performer.
model_path = os.path.join(models_dir, 'baseline_model.keras')
model = tf.keras.models.load_model(model_path)

# --- Load Test Dataset Filepaths and Labels ---
# We need to get the filepaths and labels before they are batched and processed.
# This logic mirrors the split in preprocess.py to ensure we use the same test set.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

csv_path = os.path.join(project_root, 'data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')
df = pd.read_csv(csv_path)
df['class'] = ''
df.loc[df['Class1.1'] > 0.8, 'class'] = 'Elliptical'
df.loc[df['Class1.2'] > 0.8, 'class'] = 'Spiral'
df_filtered = df[df['class'] != ''].copy()
df_filtered = df_filtered[df_filtered['class'] != 'Artifact']

image_path_base = os.path.join(project_root, 'data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/')
df_filtered['filepath'] = df_filtered['GalaxyID'].apply(lambda x: os.path.join(image_path_base, f"{x}.jpg"))

X = df_filtered['filepath']
y = df_filtered['class']

_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# --- Qualitative Evaluation ---
class_names = ['Elliptical', 'Spiral']

plt.figure(figsize=(15, 15))
for i in range(9):
    # Get a random index
    idx = np.random.randint(0, len(X_test))
    
    # Get the image, path, and true label
    image_path = X_test.iloc[idx]
    true_label = y_test.iloc[idx]
    
    # Preprocess the image for the model
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0
    img_array = tf.expand_dims(img, 0) # Create a batch

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    # Display the image and results
    ax = plt.subplot(3, 3, i + 1)
    # We use PIL to open the image for display, not for processing
    plt.imshow(Image.open(image_path))
    plt.title(f"True: {true_label}\\nPred: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")

plt.tight_layout()
plt.show()
