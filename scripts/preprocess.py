import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_datasets():
    # Load the dataset
    df = pd.read_csv('data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')

    # --- Define Target Variable ---
    df['class'] = ''
    df.loc[df['Class1.1'] > 0.8, 'class'] = 'Elliptical'
    df.loc[df['Class1.2'] > 0.8, 'class'] = 'Spiral'
    df.loc[df['Class1.3'] > 0.8, 'class'] = 'Artifact'

    # Filter the DataFrame
    df_filtered = df[df['class'] != ''].copy()
    df_filtered = df_filtered[df_filtered['class'] != 'Artifact']

    # --- Split the Data ---
    IMAGE_PATH = 'data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/'
    df_filtered['filepath'] = df_filtered['GalaxyID'].apply(lambda x: f"{IMAGE_PATH}{x}.jpg")

    X = df_filtered['filepath']
    y = df_filtered['class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Label Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)

    # Convert to one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train_encoded)
    y_val_one_hot = tf.keras.utils.to_categorical(y_val_encoded)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test_encoded)
    
    # Create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_one_hot))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_one_hot))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_one_hot))

    # Define image processing function
    IMG_SIZE = 128
    def process_image(filepath, label):
        img = tf.io.read_file(filepath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        return img, label

    # Data Augmentation
    def augment_image(image, label):
        image = tf.image.random_flip_up_down(image)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return image, label

    # Configure datasets
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset
        .map(process_image, num_parallel_calls=AUTOTUNE)
        .map(augment_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_dataset = (
        val_dataset
        .map(process_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .map(process_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    # This block will run when the script is executed directly
    # You can keep your original print statements here for verification
    # Load the dataset
    df = pd.read_csv('data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')

    # --- Define Target Variable ---

    # Create a new 'class' column
    df['class'] = ''

    # Assign classes based on high-confidence votes
    df.loc[df['Class1.1'] > 0.8, 'class'] = 'Elliptical'
    df.loc[df['Class1.2'] > 0.8, 'class'] = 'Spiral'
    df.loc[df['Class1.3'] > 0.8, 'class'] = 'Artifact'

    # Filter the DataFrame
    df_filtered = df[df['class'] != ''].copy()

    # Remove the 'Artifact' class due to severe imbalance
    df_filtered = df_filtered[df_filtered['class'] != 'Artifact']

    print("--- Filtered DataFrame Head ---")
    print(df_filtered.head())
    print(f"\nNumber of samples after filtering: {len(df_filtered)}")

    # --- Analyze Class Distribution ---
    plt.figure(figsize=(8, 6))
    sns.countplot(x='class', data=df_filtered)
    plt.title('Class Distribution of Filtered Dataset')
    plt.xlabel('Galaxy Type')
    plt.ylabel('Count')
    plt.show()

    print("\n--- Class Distribution ---")
    print(df_filtered['class'].value_counts())


    # --- Split the Data ---
    
    # Create a column for image file paths
    IMAGE_PATH = 'data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/'
    df_filtered['filepath'] = df_filtered['GalaxyID'].apply(lambda x: f"{IMAGE_PATH}{x}.jpg")

    # Split the data
    X = df_filtered['filepath']
    y = df_filtered['class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print("\n--- Data Split ---")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    train_ds, val_ds, test_ds = get_datasets()
    
    print("\n--- tf.data.Dataset shapes ---")
    for image_batch, label_batch in train_ds.take(1):
        print(f"Image batch shape: {image_batch.shape}")
        print(f"Label batch shape: {label_batch.shape}")
