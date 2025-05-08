import io

import torch

from torch.utils.data import Dataset
from PIL import Image

import string

class VQADataset(Dataset):
    def __init__(self, images, index, model, clean_text):
        self.images = images # This is the original images dict with bytes and caption_ids
        self.index = index
        self.clean_captions = clean_text
        self.max_len = model.max_len

        self.tokenizer = model.tokenizer
        self.processor = model.processor
    
    def __len__(self):
        return len(self.captions)
    
    def clean_text(self, text):
        # keep only alphanum
        allowed = set(string.ascii_letters + string.digits + " .,-!?-")
        return ''.join([c.lower() for c in text if c in allowed]).strip()
    
    def __getitem__(self, idx):
        item = self.index[idx]
        image_id = item["image_id"]
        _image_bytes = self.images[image_id]["image_bytes"]

        image_pil = Image.open(io.BytesIO(_image_bytes)).convert("RGB")
        _pixel_values = self.processor(images=image_pil, return_tensors="pt")["pixel_values"][0]  # (C, H, W)
        
        question = item["question"]
        if self.clean_text:  # <--- Only clean if enabled
            question = self.clean_text(question)

        # Prepare input_ids: [BOS, content_tokens, PAD...]
        input_tokenization = self.tokenizer(
            question, # clean_text already applied
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        _input_ids = input_tokenization.input_ids[0]
        _attention_mask = input_tokenization.attention_mask[0]

        # Label is the tokenized multiple choice answer
        answer = item["multiple_choice_answer"]

        assert len(answer.split()) == 1, f"Answer must be a single word, got: {answer}"

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
            "image_tensor": _pixel_values, # Only return image bytes for the model
            "image_bytes": _image_bytes, # Original image bytes for visualisation
            "input_ids": _input_ids,
            "attention_mask": _attention_mask, # Original attention mask for input_ids
            "label_id": _label_ids
        }