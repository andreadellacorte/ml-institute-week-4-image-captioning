import streamlit as st
import io
from PIL import Image
import torch

def load_model():
    from transformers import CLIPProcessor
    MODEL_PATH = 'models/model.pth'
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
    model.processor = processor
    model.tokenizer = processor.tokenizer
    model.eval()
    return model

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

st.title('Image Captioning Demo (Upload Only)')

uploaded_file = st.file_uploader("Upload an image to caption", type=["png", "jpg", "jpeg", "bmp", "gif", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    try:
        model = load_model()
        st.success('Model loaded!')
    except Exception as e:
        st.error(f'Could not load model: {e}')
        st.stop()
    if st.button('Generate Caption'):
        caption = generate_caption(model, image)
        st.markdown(f'**Model Caption:** {caption}')
else:
    st.info('Please upload an image to get a caption.')
