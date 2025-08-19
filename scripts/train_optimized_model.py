import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Reusing the data functions from the tuning script ---
# (Note: In a larger project, these would be in a shared utility file)

import albumentations as A

def get_data_splits():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    csv_path = os.path.join(project_root, 'data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')
    image_path_base = os.path.join(project_root, 'data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/')

    df = pd.read_csv(csv_path)
    df['class'] = ''
    df.loc[df['Class1.1'] > 0.8, 'class'] = 'Elliptical'
    df.loc[df['Class1.2'] > 0.8, 'class'] = 'Spiral'
    df_filtered = df[df['class'] != ''].copy()
    df_filtered = df_filtered[df_filtered['class'] != 'Artifact']
    
    df_filtered['filepath'] = df_filtered['GalaxyID'].apply(lambda x: os.path.join(image_path_base, f"{x}.jpg"))

    X = df_filtered['filepath']
    y = df_filtered['class']
    
    # We create a final train/val split. No test set needed until the very end.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val_encoded)
    
    return X_train.values, X_val.values, y_train_one_hot, y_val_one_hot

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
])

def augment_image(image):
    data = {"image": image}
    augmented_data = transform(**data)
    return augmented_data["image"]

def create_dataset(X, y, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    def process_path(filepath, label):
        img_path_str = filepath.numpy().decode('utf-8')
        img = cv2.imread(img_path_str)
        if img is None:
            return np.zeros((128, 128, 3), dtype=np.float32), label.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if augment:
            img = augment_image(img)
        img_resized = cv2.resize(img, (128, 128))
        img_normalized = img_resized / 255.0
        return img_normalized.astype(np.float32), label.numpy().astype(np.float32)

    def tf_process_path(filepath, label):
        [img, lbl] = tf.py_function(process_path, [filepath, label], [tf.float32, tf.float32])
        img.set_shape([128, 128, 3])
        lbl.set_shape([2])
        return img, lbl

    dataset = dataset.map(tf_process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

# --- 2. Build the Optimized Model ---
def build_optimized_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(192, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history, filename):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    
    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.savefig(filename)
    print(f"Saved training history plot to {filename}")
    plt.show()

# --- 3. Main Training Block ---
if __name__ == '__main__':
    X_train, X_val, y_train, y_val = get_data_splits()
    
    train_dataset = create_dataset(X_train, y_train, augment=True)
    val_dataset = create_dataset(X_val, y_val, augment=False)
    
    model = build_optimized_model()
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath='../models/optimized_galaxy_model.keras',
        save_best_only=True,
        monitor='val_loss'
    )
    
    print("\n--- Training Optimized Model ---")
    history = model.fit(
        train_dataset,
        epochs=100, # Train for longer, EarlyStopping will find the best epoch
        validation_data=val_dataset,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    print("\n--- Training Complete ---")
    
    # Construct absolute path for saving the plot
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    plot_path = os.path.join(project_root, 'assets/optimized_model_history.png')
    
    plot_history(history, plot_path)
    
    # You can now evaluate this model using the evaluate_models.py script
    # after updating it to load 'optimized_galaxy_model.keras'.
