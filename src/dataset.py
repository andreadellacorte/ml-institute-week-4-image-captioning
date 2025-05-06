import torch
from torch.utils.data import Dataset
from PIL import Image
from loguru import logger
import re

from PIL import Image
import io

from torchvision import transforms

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, model, resize_size=224, max_len=25):
        self.images = images
        self.captions = captions

        self.tokenizer = model.tokenizer
        self.bos_token = model.tokenizer.bos_token
        self.eos_token = model.tokenizer.eos_token
        self.pad_token = model.tokenizer.pad_token
        self.max_len = max_len
        self.model = model

        self.image_transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.data = []

        for img_id, img_data in self.images.items():
            for caption_id in img_data["caption_ids"]:
                self.data.append({
                    "img_id": img_id,
                    "caption_id": caption_id
                })
    
    def __len__(self):
        return len(self.data)
    
    def clean_text(self, text):
        words = [word.lower() for word in re.findall(r'\b[a-zA-Z]+\b', text)]
        return ' '.join(words)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        img_id = item["img_id"]
        caption_id = item["caption_id"]

        image = Image.open(io.BytesIO(self.images[img_id]["image_bytes"]))
        image = self.image_transform(image)

        caption = self.captions[caption_id]["caption"]

        caption = self.clean_text(caption)

        input_text = f"{self.bos_token} {caption}"
        label = f"{caption} {self.eos_token}"

        # Use max_length parameter and truncation to ensure correct sequence length
        tokenized_input = self.tokenizer(
            input_text, 
            padding="max_length", 
            max_length=self.max_len + 1,  # +1 to account for later slicing
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = tokenized_input.input_ids[:, :self.max_len]  # Ensure exact length by slicing
        
        tokenized_label = self.tokenizer(
            label, 
            padding="max_length", 
            max_length=self.max_len + 1,  # +1 to account for later slicing
            truncation=True, 
            return_tensors="pt"
        )
        label_ids = tokenized_label.input_ids[:, 1:self.max_len]  # Remove the first element and slice to max_len
        label_ids = torch.cat(
            [label_ids, torch.tensor([[self.tokenizer.eos_token_id]])], dim=1
        )  # Add eos_token_id at the end

        return {
            "image_bytes": image,
            "input_ids": input_ids,
            "label_ids": label_ids
        }