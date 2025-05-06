import io

import torch

from torch.utils.data import Dataset
from PIL import Image

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, model):
        self.images = images # This is the original images dict with bytes and caption_ids
        self.captions = captions
        self.tokenizer = model.tokenizer
        self.bos_token = model.tokenizer.bos_token
        self.eos_token = model.tokenizer.eos_token
        self.pad_token = model.tokenizer.pad_token
        self.pad_token_id = model.tokenizer.pad_token_id

        self.max_len = model.max_len
        self.processor = model.processor  # Use CLIPProcessor for image normalization

        # Pre-process and store normalized images
        self.processed_image_tensors = {}
        for img_id, img_content in self.images.items():
            image_pil = Image.open(io.BytesIO(img_content["image_bytes"]))
            self.processed_image_tensors[img_id] = self.processor(images=image_pil, return_tensors="pt")["pixel_values"][0]

        self.data = []
        for img_id, img_data in self.images.items(): # Iterate original self.images to get caption_ids
            for caption_id in img_data["caption_ids"]:
                self.data.append({
                    "img_id": img_id,
                    "caption_id": caption_id
                    # "image_bytes" field is removed from items in self.data,
                    # as it will be fetched from self.images[img_id]["image_bytes"] in __getitem__
                })
    
    def __len__(self):
        return len(self.data)
    
    def clean_text(self, text):
        # Less aggressive cleaning: keep alphanum and basic punctuation
        import string
        allowed = set(string.ascii_letters + string.digits + " ")
        return ''.join([c for c in text if c in allowed]).strip()
    
    def __getitem__(self, idx):
        item = self.data[idx] # item now only contains img_id and caption_id
        img_id = item["img_id"]
        caption_id = item["caption_id"]

        # Retrieve pre-processed image tensor
        image_tensor = self.processed_image_tensors[img_id]
        
        # Retrieve original image bytes from self.images (the original structure)
        image_bytes = self.images[img_id]["image_bytes"]
        
        caption = self.captions[caption_id]["caption"]
        caption = self.clean_text(caption)

        # Tokenize the core caption content
        tokenized_content = self.tokenizer(
            caption,
            add_special_tokens=False, # Do not add BOS/EOS here
            return_tensors="pt"
        ).input_ids[0]

        # Truncate tokenized_content if it's too long to fit BOS and EOS
        if tokenized_content.size(0) > self.max_len - 2:
            tokenized_content = tokenized_content[:self.max_len - 2]
        
        content_len = tokenized_content.size(0)

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
        
        # Find the first PAD token in input_ids, or end of sequence
        first_pad_idx = self.max_len
        eos_idx = self.max_len
        
        # Search for EOS and PAD in _input_ids
        input_ids_list = _input_ids.tolist()
        try:
            eos_idx = input_ids_list.index(self.tokenizer.eos_token_id) # Corrected: use self.tokenizer.eos_token_id
        except ValueError:
            # EOS not found, sequence might be truncated before EOS or completely filled
            pass

        try:
            first_pad_idx = input_ids_list.index(self.pad_token_id) # This was already correct (self.pad_token_id is assigned in __init__)
        except ValueError:
            # No PAD token, sequence is full
            pass

        # The actual content for labels starts after BOS and ends before the first PAD or at EOS
        # If BOS is token 0, content starts at 1.
        # If EOS is present, content for labels ends at EOS.
        # If no EOS but PAD is present, content ends before PAD.
        # If neither, content is full length - 1 (excluding BOS for label start).

        # Content for labels: tokens from _input_ids[1:] up to EOS or first PAD
        # Target length for labels is up to where EOS is, or where padding begins in input.
        # Example _input_ids: [BOS, t1, t2, t3, EOS, PAD, PAD]
        # We want _label_ids: [t1, t2, t3, EOS, PAD, PAD, PAD] (ignore_index for PADs)
        
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