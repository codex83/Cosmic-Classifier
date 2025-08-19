import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image

def predict_on_image(model, image_path):
    """
    Loads an image, preprocesses it, and returns the model's prediction and confidence.
    """
    # 1. Load and preprocess the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img = img / 255.0  # Normalize to [0,1]
    img_array = tf.expand_dims(img, 0)  # Create a batch

    # 2. Make a prediction
    predictions = model.predict(img_array)
    
    # 3. Interpret the prediction
    class_names = ['Spiral', 'Elliptical'] 
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    return predicted_class, confidence

if __name__ == '__main__':
    # --- 1. Setup Argument Parser ---
    parser = argparse.ArgumentParser(description='Classify a single galaxy image.')
    parser.add_argument('--image', type=str, required=True, help='Path to the galaxy image file.')
    args = parser.parse_args()

    # --- 2. Define Paths & Load Model ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    model_path = os.path.join(project_root, 'models', 'baseline_model.keras')
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        exit()
        
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # --- 3. Predict and Display ---
    predicted_class, confidence = predict_on_image(model, args.image)
    
    # Display the image and the prediction
    plt.figure(figsize=(8, 8))
    image = Image.open(args.image)
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%", fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # Save the figure instead of showing it
    base, ext = os.path.splitext(args.image)
    output_path = f"{base}_prediction.png"
    plt.savefig(output_path)
    plt.close() # Close the plot to prevent it from displaying interactively

    print(f"Prediction saved to: {output_path}")
