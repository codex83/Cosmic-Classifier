import tensorflow as tf
import keras_tuner as kt
import albumentations as A
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import os

# --- 1. Data Loading ---
def get_data_splits():
    # Use an absolute path relative to the script's location
    # This makes the script runnable from any directory
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val_encoded)
    
    return X_train.values, X_val.values, y_train_one_hot, y_val_one_hot

# --- 2. Advanced Augmentation ---
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

# --- 3. tf.data Pipeline ---
def create_dataset(X, y, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    def process_path(filepath, label):
        img_path_str = filepath.numpy().decode('utf-8')
        img = cv2.imread(img_path_str)
        if img is None:
            # Handle cases where the image might not be found
            print(f"Warning: Could not read image at {img_path_str}. Returning zeros.")
            return np.zeros((128, 128, 3), dtype=np.float32), label.numpy()
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if augment:
            img = augment_image(img)
            
        # Resize and normalize
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

# --- 4. Hypermodel Definition ---
class GalaxyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(128, 128, 3)))
        
        hp_filters_1 = hp.Int('filters_1', min_value=32, max_value=64, step=32)
        hp_filters_2 = hp.Int('filters_2', min_value=64, max_value=128, step=32)
        
        model.add(tf.keras.layers.Conv2D(hp_filters_1, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Conv2D(hp_filters_2, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        
        model.add(tf.keras.layers.Flatten())
        
        hp_units = hp.Int('units', min_value=128, max_value=256, step=64)
        model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
        
        hp_dropout = hp.Float('dropout', min_value=0.3, max_value=0.6, step=0.1)
        model.add(tf.keras.layers.Dropout(hp_dropout))
        
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    X_train, X_val, y_train, y_val = get_data_splits()
    
    train_dataset = create_dataset(X_train, y_train, augment=True)
    val_dataset = create_dataset(X_val, y_val, augment=False)
    
    tuner = kt.RandomSearch(
        GalaxyHyperModel(),
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='advanced_tuner_results', # Create results in a new folder
        project_name='galaxy_classification'
    )
    
    tuner.search_space_summary()
    
    print("\n--- Starting Hyperparameter Search ---")
    tuner.search(train_dataset, epochs=10, validation_data=val_dataset)
    
    print("\n--- Search Complete ---")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"""
    The hyperparameter search is complete.
    Optimal Filters 1: {best_hps.get('filters_1')}
    Optimal Filters 2: {best_hps.get('filters_2')}
    Optimal Dense Units: {best_hps.get('units')}
    Optimal Dropout Rate: {best_hps.get('dropout')}
    Optimal Learning Rate: {best_hps.get('learning_rate')}
    """)
