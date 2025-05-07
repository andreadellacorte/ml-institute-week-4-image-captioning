import streamlit as st
import pickle
import io
from PIL import Image
import torch
import os

# --- CONFIGURABLE PATHS ---
DATA_PATH = 'data/raw/flickr30k/500.pkl'  # Update if your file is elsewhere
MODEL_PATH = 'data/checkpoints/model.pt'  # Update to your model checkpoint

# --- LOAD DATA ---
@st.cache_data

def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
    return data

def load_model():
    # Replace with your actual model class and loading logic
    from src.model import UnifiedAutoregressiveDecoder
    model = UnifiedAutoregressiveDecoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
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

num_images = len(data)
idx = st.slider('Select image index', 0, num_images-1, 0)

image_bytes, gt_captions = get_image_and_captions(data, idx)
image = Image.open(io.BytesIO(image_bytes))
st.image(image, caption='Selected Image', use_column_width=True)

# Load model only if needed
if 'model' not in st.session_state:
    try:
        st.session_state['model'] = load_model()
        st.success('Model loaded!')
    except Exception as e:
        st.error(f'Could not load model: {e}')
        st.stop()
model = st.session_state['model']

# Generate caption
def generate_caption(model, image):
    # You may need to adapt this to your model's API
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        caption = model.generate_caption(image_tensor, max_new_tokens=25)
    if isinstance(caption, list):
        return caption[0]
    return caption

if st.button('Generate Caption'):
    caption = generate_caption(model, image)
    st.markdown(f'**Model Caption:** {caption}')

st.markdown('**Ground Truth Captions:**')
for i, cap in enumerate(gt_captions):
    st.markdown(f'{i+1}. {cap}')
