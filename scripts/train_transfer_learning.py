import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from preprocess import get_datasets
from train_baseline import plot_history

def create_transfer_model(input_shape=(128, 128, 3), num_classes=2):
    # Load the pre-trained base
    base_model = MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add a custom head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model


if __name__ == '__main__':
    # Get datasets
    train_ds, val_ds, test_ds = get_datasets()

    model, base_model = create_transfer_model()

    # Compile for feature extraction
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("--- Training Head ---")
    history_feature_extraction = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,  # Train for a few epochs
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    )

    # --- Fine-Tuning ---
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Re-compile with a low learning rate
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("\n--- Fine-Tuning ---")
    history_fine_tuning = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,  # Continue training
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                   ModelCheckpoint('../models/transfer_model.keras', save_best_only=True)]
    )

    # Plot combined history
    history = {
        'accuracy': history_feature_extraction.history['accuracy'] + history_fine_tuning.history['accuracy'],
        'val_accuracy': history_feature_extraction.history['val_accuracy'] + history_fine_tuning.history['val_accuracy'],
        'loss': history_feature_extraction.history['loss'] + history_fine_tuning.history['loss'],
        'val_loss': history_feature_extraction.history['val_loss'] + history_fine_tuning.history['val_loss'],
    }
    
    # Create a dummy History object to pass to plot_history
    class DummyHistory:
        def __init__(self, history_dict):
            self.history = history_dict
            
    plot_history(DummyHistory(history))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'\nTest accuracy (transfer learning): {test_acc:.4f}')
