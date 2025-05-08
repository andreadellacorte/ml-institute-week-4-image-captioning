import io

import torch

from torch.utils.data import Dataset
from PIL import Image

import string

class VQADataset(Dataset):
    def __init__(self, images, index, model, clean_questions):
        self.images = images
        self.index = index
        self.clean_questions = clean_questions
        self.max_len = model.max_len

        self.tokenizer = model.tokenizer
        self.processor = model.processor
    
    def __len__(self):
        return len(self.index)
    
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
        if self.clean_questions:  # <--- Only clean if enabled
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

        # Prepare input_ids: [BOS, content_tokens, PAD...]
        output_tokenization = self.tokenizer(
            answer,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )

        _label_ids = output_tokenization.input_ids

        # Remove the first token (BOS) and add an extra PAD token at the end
        _label_ids = torch.cat([_label_ids[0][1:], torch.tensor([self.tokenizer.pad_token_id])])

        return {
            "image_tensor": _pixel_values, # Only return image bytes for the model
            "image_bytes": _image_bytes, # Original image bytes for visualisation
            "input_ids": _input_ids,
            "attention_mask": _attention_mask, # Original attention mask for input_ids
            "label_ids": _label_ids
        }