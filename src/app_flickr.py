import os
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pickle
import io
from PIL import Image
import torch
import random

# --- CONFIGURABLE PATHS ---
DATA_PATH = 'data/raw/flickr30k/500.pkl'  # Update if your file is elsewhere
MODEL_PATH = 'models/model.pth'  # Update to your model checkpoint

# --- LOAD DATA ---
@st.cache_data

def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data

def load_model():
    from transformers import CLIPProcessor
    # Load the whole model (architecture + weights)
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    # Re-initialize the tokenizer and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    model.processor = processor
    model.tokenizer = processor.tokenizer
    model.eval()
    return model

def get_image_and_captions(data, idx):
    sample = data[idx]
    image_bytes = sample['image_bytes']
    gt_captions = sample['caption'][:5]  # Assumes a list of 5
    return image_bytes, gt_captions

# --- STREAMLIT APP ---
st.title('Image Captioning Demo')

try:
    data = load_data()
    st.success('Data loaded!')
except Exception as e:
    st.error(f'Could not load data: {e}')
    st.stop()

# Set a random initial index only on first load
if 'initial_idx' not in st.session_state:
    st.session_state['initial_idx'] = random.randint(0, len(data)-1)
num_images = len(data)
idx = st.slider('Select image index', 0, num_images-1, st.session_state['initial_idx'])

image_bytes, gt_captions = get_image_and_captions(data, idx)
image = Image.open(io.BytesIO(image_bytes))
st.image(image, caption='Selected Image', use_container_width=True)

# Always load the model fresh on each run
try:
    model = load_model()
    st.success('Model loaded!')
except Exception as e:
    st.error(f'Could not load model: {e}')
    st.stop()

# Generate caption
def generate_caption(model, image):
    import logging
    logger = logging.getLogger("captioning")
    logger.info("Processing image for caption generation...")
    image_tensor = model.process_images(image)["pixel_values"]
    logger.info(f"Extracted pixel_values, type: {type(image_tensor)}, shape: {getattr(image_tensor, 'shape', None)}")
    if isinstance(image_tensor, list):
        logger.info("Converting pixel_values list to tensor via torch.stack...")
        image_tensor = torch.stack(image_tensor)
    if image_tensor.ndim == 3:
        logger.info("Adding batch dimension to image_tensor...")
        image_tensor = image_tensor.unsqueeze(0)
    logger.info(f"Final image_tensor shape: {image_tensor.shape}")
    caption = model.generate_caption(image_tensor, max_new_tokens=25)
    logger.info(f"Raw model caption output: {caption}")
    if isinstance(caption, list):
        logger.info("Caption is a list, returning first element.")
        return caption[0]
    logger.info("Caption is a string, returning as is.")
    return caption

# Always show the button
if st.button('Generate Caption'):
    caption = generate_caption(model, image)
    st.markdown(f'**Model Caption:** {caption}')
    # Optionally, re-show ground truth captions after generating
    st.markdown('**Ground Truth Captions:**')
    for i, cap in enumerate(gt_captions):
        st.markdown(f'{i+1}. {cap}')