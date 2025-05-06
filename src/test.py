# load one image from data/processed/flickr30k/1_images.pkl
import torch

from torchvision.transforms import ToTensor
from PIL import Image
import io

import pickle
import pprint

import pprint

import pickle

from loguru import logger

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

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

    import transformers

    with open(PROCESSED_DATA_DIR / "flickr30k/100_images.pkl", "rb") as f:
        images = pickle.load(f)

    pprint.pprint(images[0])

    with open(PROCESSED_DATA_DIR / "flickr30k/100_captions.pkl", "rb") as f:
        captions = pickle.load(f)

    pprint.pprint(captions[0])

    # Convert image bytes to PIL Image
    pil_image = Image.open(io.BytesIO(images[0]["image_bytes"]))
    image = ToTensor()(pil_image).unsqueeze(0)
    print(image.shape)

    # pass the image through the CLIP Model Vision Encoder
    model_name = "openai/clip-vit-base-patch32"
    
    # Load the CLIP model
    logger.info(f"Loading CLIP model: {model_name}")
    processor = transformers.CLIPProcessor.from_pretrained(model_name)
    model = transformers.CLIPModel.from_pretrained(model_name)
    
    # Process the image with the CLIP processor
    inputs = processor(images=pil_image, return_tensors="pt", padding=True)
    
    # Get the image features from the vision encoder
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    logger.info(f"Image features shape: {image_features.shape}")
    
    # Process the captions with the CLIP processor
    text = [captions[caption_id]["caption"] for caption_id in captions.keys()]
    
    text_inputs = processor(text=text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        
    logger.info(f"Text features shape: {text_features.shape}")
    
    # Compute similarity between image and text features
    similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
    logger.info(f"Similarity scores: {similarity}")
    
    # Get the top 10 captions with highest similarity scores
    top_k = 10
    top_indices = torch.topk(similarity, top_k).indices
    logger.info("Top 10 most similar captions:")
    for idx in top_indices:
        logger.info(f"Caption: {text[idx]} | Similarity score: {similarity[idx].item()}")

if __name__ == "__main__":
    main()
