# load one image from data/processed/flickr30k/1_images.pkl
from model import UnifiedAutoregressiveDecoder

from torchvision.transforms import ToTensor

import pickle
import pprint
import random
import torch

from src.config import PROCESSED_DATA_DIR

SIZES = {
    1: "1",
    50: "50",
    100: "100",
    500: "500",
    1000: "1k",
    5000: "5k",
    10000: "10k",
}

def main():
    with open(PROCESSED_DATA_DIR / "flickr30k/100_images.pkl", "rb") as f:
        images = pickle.load(f)

    pprint.pprint(images[0])

    with open(PROCESSED_DATA_DIR / "flickr30k/100_captions.pkl", "rb") as f:
        captions = pickle.load(f)
    
    # Convert dictionary to list for proper splitting
    image_ids = list(images.keys())
    random.shuffle(image_ids)
    
    # Calculate split indices
    train_size = int(len(image_ids) * 0.8)
    test_size = int(len(image_ids) * 0.1)
    
    # Split the image IDs
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:train_size+test_size]
    val_ids = image_ids[train_size+test_size:]
    
    # Create dictionaries for each split
    train_images = {id: images[id] for id in train_ids}
    test_images = {id: images[id] for id in test_ids}
    val_images = {id: images[id] for id in val_ids}
    
    print(f"Train images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Validation images: {len(val_images)}")

    model = UnifiedAutoregressiveDecoder()
    
    # Process the first image
    first_id = list(images.keys())[0]
    image_tensor = ToTensor()(images[first_id]["image"]).unsqueeze(0)
    
    # Get first caption ID and convert to input_ids that the model expects
    caption_id = images[first_id]['caption_ids'][0]
    caption_text = captions[caption_id]['caption']
    
    # Use the model's tokenizer to convert text to input_ids
    input_ids = model.tokenizer(caption_text, return_tensors="pt").input_ids
    
    # Forward pass through the model
    result = model(image_tensor, input_ids)
    
    print(f"{result}")

def train(model, train_images, train_captions):
    # Implement your training loop here
    pass

def evaluate(model, test_images, test_captions):
    # Implement your evaluation loop here
    pass

def validate(model, val_images, val_captions):
    # Implement your validation loop here
    pass


if __name__ == "__main__":
    main()