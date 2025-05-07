import io

import torch

from torch.utils.data import Dataset
from PIL import Image

import string
from tqdm import tqdm

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, model, processed_image_tensors):
        self.images = images # This is the original images dict with bytes and caption_ids
        self.captions = captions
        self.tokenizer = model.tokenizer
        self.bos_token = model.tokenizer.bos_token
        self.eos_token = model.tokenizer.eos_token
        self.pad_token = model.tokenizer.pad_token
        self.pad_token_id = model.tokenizer.pad_token_id

        self.max_len = model.max_len
        self.processor = model.processor  # Use CLIPProcessor for image normalization

        # Pre-process and store normalized images with a progress bar
        self.processed_image_tensors = processed_image_tensors

        # Prepare dataset with a progress bar
        self.data = []
        for img_id, img_data in tqdm(self.images.items(), desc="Preparing dataset"):
            for caption_id in img_data["caption_ids"]:
                self.data.append({
                    "img_id": img_id,
                    "caption_id": caption_id
                })
    
    def __len__(self):
        return len(self.data)
    
    def clean_text(self, text):
        # keep only alphanum
        allowed = set(string.ascii_letters + string.digits + " ")
        return ''.join([c.lower() for c in text if c in allowed]).strip()
    
    def __getitem__(self, idx):
        item = self.data[idx] # item now only contains img_id and caption_id
        img_id = item["img_id"]
        caption_id = item["caption_id"]

        if img_id not in self.processed_image_tensors:
            image_pil = Image.open(io.BytesIO(self.images[img_id]["image_bytes"]))
            with torch.no_grad():
                self.processed_image_tensors[img_id] = self.processor(images=image_pil, return_tensors="pt")["pixel_values"][0]
        
        image_tensor = self.processed_image_tensors[img_id]

        # Retrieve pre-processed image tensor
        image_tensor = self.processed_image_tensors[img_id]
        
        # Retrieve original image bytes from self.images (the original structure)
        image_bytes = self.images[img_id]["image_bytes"]
        
        caption = self.captions[caption_id]["caption"]
        caption = self.clean_text(caption)

        # Prepare input_ids: [BOS, content_tokens, PAD...]
        # The tokenizer should add BOS automatically if configured, 
        # but let's be explicit for clarity during debugging.
        # However, CLIPTokenizer usually handles special tokens based on its configuration.
        # Forcing BOS here might lead to double BOS if tokenizer also adds it.
        # Let's rely on the tokenizer to add BOS and EOS as needed for input_ids
        # and then construct label_ids by shifting.

        # Tokenize for input_ids (let tokenizer add BOS, EOS if it's configured to do so)
        # Standard practice: input_ids = [BOS, token1, ..., tokenN, EOS, PAD...]
        # For CLIP, input_ids are typically `[BOS, text_tokens, EOS, PAD...]`
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
        # Shift input_ids to the left and add pad_token_id at the end.
        # All tokens in label_ids that correspond to PAD in input_ids should be ignore_index (pad_token_id)
        _label_ids = torch.full((self.max_len,), self.pad_token_id, dtype=torch.long)
        
        # Find the actual tokens (not BOS, not PAD)
        # CLIPTokenizer adds BOS at the beginning and EOS at the end if text fits.
        # Example: BOS t1 t2 EOS PAD PAD -> t1 t2 EOS PAD PAD PAD
        
        # Effective length of the tokenized sequence (including BOS and EOS if present, excluding PAD)
        effective_len = torch.sum(_attention_mask).item()

        if effective_len > 1: # if there's more than just BOS
            # Copy from _input_ids[1] (after BOS) to _label_ids[0]
            # Copy up to effective_len - 1 tokens from input_ids
            # e.g. if input is [BOS, t1, t2, EOS], effective_len = 4
            # copy _input_ids[1:3] (t1, t2) to _label_ids[0:2]
            # then add EOS to _label_ids[2]
            len_to_copy = effective_len - 1
            _label_ids[:len_to_copy] = _input_ids[1:effective_len]
            
            # If the original sequence (before truncation for label) had EOS, 
            # and there's space, ensure EOS is the next token in labels.
            # _input_ids already has EOS if it fit. So _input_ids[effective_len-1] is EOS.
            # So _label_ids[len_to_copy-1] would be the token before EOS.
            # And _label_ids[len_to_copy] should be EOS.
            if _input_ids[effective_len-1] == self.tokenizer.eos_token_id: # Corrected: use self.tokenizer.eos_token_id
                 _label_ids[len_to_copy-1] = _input_ids[effective_len-1] # This should be EOS
            elif len_to_copy < self.max_len : # if there is space and last token wasn't EOS (due to truncation)
                 _label_ids[len_to_copy] = self.tokenizer.eos_token_id # Corrected: use self.tokenizer.eos_token_id

        # All other positions in _label_ids are already self.pad_token_id.
        # The loss function will ignore self.pad_token_id.

        return {
            "image_tensor": image_tensor,
            "image_bytes": image_bytes, # Sourced from self.images via img_id
            "input_ids": _input_ids,
            "label_ids": _label_ids,
            "attention_mask": _attention_mask # Original attention mask for input_ids
        }