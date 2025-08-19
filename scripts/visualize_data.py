import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the dataset
df = pd.read_csv('data/galaxy-zoo-the-galaxy-challenge/training_solutions_rev1.csv')
IMAGE_PATH = 'data/galaxy-zoo-the-galaxy-challenge/images_training_rev1/'

# Display 5 random images
for i in range(5):
    # Get a random sample
    sample = df.sample(1)
    galaxy_id = sample['GalaxyID'].iloc[0]
    
    # Load and display the image
    image_file = os.path.join(IMAGE_PATH, f'{galaxy_id}.jpg')
    img = Image.open(image_file)
    
    plt.imshow(img)
    plt.title(f'GalaxyID: {galaxy_id}')
    plt.axis('off')
    plt.show()
    
    # Print the full vote data
    print(f"--- Vote Data for GalaxyID: {galaxy_id} ---")
    print(sample.iloc[0])
    print("\\n" + "="*50 + "\\n")
