import io

import torch

from torch.utils.data import Dataset
from PIL import Image

import string
from tqdm import tqdm

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, model, max_captions_per_image, clean_captions):
        self.images = images # This is the original images dict with bytes and caption_ids
        self.captions = captions
        self.clean_captions = clean_captions
        self.max_len = model.max_len

        self.tokenizer = model.tokenizer
        self.processor = model.processor

        # Prepare dataset with a progress bar, allowing configurable number of captions per image
        self.data = []

        assert max_captions_per_image > 0, "max_captions_per_image must be greater than 0"
        assert max_captions_per_image <= 5, "max_captions_per_image must be less than or equal to 5"
        
        for img_id, img_data in tqdm(self.images.items(), desc="Preparing dataset"):
            for caption_id in img_data["caption_ids"][:max_captions_per_image]:
                self.data.append({
                "img_id": img_id,
                "caption_id": caption_id
                })
    
    def __len__(self):
        return len(self.data)
    
    def clean_text(self, text):
        # keep only alphanum
        allowed = set(string.ascii_letters + string.digits + " .,-!?-")
        return ''.join([c.lower() for c in text if c in allowed]).strip()
    
    def __getitem__(self, idx):
        item = self.data[idx] # item now only contains img_id and caption_id
        img_id = item["img_id"]
        caption_id = item["caption_id"]

        image_pil = Image.open(io.BytesIO(self.images[img_id]["image_bytes"])).convert("RGB")
        pixel_values = self.processor(images=image_pil, return_tensors="pt")["pixel_values"][0]  # (C, H, W)

        image_bytes = self.images[img_id]["image_bytes"]
        
        caption = self.captions[caption_id]["caption"]
        if self.clean_captions:  # <--- Only clean if enabled
            caption = self.clean_text(caption)

        # Prepare input_ids: [BOS, content_tokens, PAD...]
        input_tokenization = self.tokenizer(
            caption, # clean_text already applied
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        _input_ids = input_tokenization.input_ids[0]
        _attention_mask = input_tokenization.attention_mask[0]

        # Prepare label_ids: [token1, ..., tokenN, EOS, PAD...]
        _label_ids = torch.full((self.max_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        effective_len = torch.sum(_attention_mask).item()

        if effective_len > 1: # if there's more than just BOS
            len_to_copy = effective_len - 1
            _label_ids[:len_to_copy] = _input_ids[1:effective_len]
            if _input_ids[effective_len-1] == self.tokenizer.eos_token_id:
                 _label_ids[len_to_copy-1] = _input_ids[effective_len-1] # This should be EOS
            elif len_to_copy < self.max_len :
                 _label_ids[len_to_copy] = self.tokenizer.eos_token_id

        return {
            "image_tensor": pixel_values, # Only return image bytes for the model
            "image_bytes": image_bytes, # Original image bytes for visualisation
            "input_ids": _input_ids,
            "attention_mask": _attention_mask, # Original attention mask for input_ids
            "label_ids": _label_ids
        }