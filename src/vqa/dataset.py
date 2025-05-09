import io

import torch

from torch.utils.data import Dataset
from PIL import Image

import string

class VQADataset(Dataset):
    def __init__(self, images, index, model, clean_questions, max_answers=3000):
        self.images = images
        self.index = index
        self.clean_questions = clean_questions
        self.max_len = getattr(model, 'max_len', 20)  # Default to 20 if model doesn't have max_len
        self.tokenizer = model.tokenizer
        self.processor = model.processor
        
        # Create answer dictionary from the most frequent answers in the dataset
        # This will map answers to a much smaller set of indices
        all_answers = [item["multiple_choice_answer"] for item in index]
        answer_counts = {}
        for answer in all_answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
        # Take top N most common answers
        top_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)[:max_answers]
        self.answer_to_idx = {answer: idx for idx, (answer, _) in enumerate(top_answers)}
        self.idx_to_answer = {idx: answer for answer, idx in self.answer_to_idx.items()}
        self.num_answers = len(self.answer_to_idx)
        
        # Add unknown answer token for answers not in the top N
        self.unk_answer_idx = self.num_answers
        self.num_classes = self.num_answers + 1
        
        # Log some statistics about the answer dictionary
        print(f"Created answer dictionary with {self.num_answers} answers")
        print(f"Most common answers: {[a for a, _ in top_answers[:10]]}")
        print(f"Least common answers in top {max_answers}: {[a for a, _ in top_answers[-10:]]}")
        
        # Count stats on answer frequencies
        total_answers = sum(answer_counts.values())
        covered_answers = sum(count for answer, count in top_answers)
        print(f"Answer coverage: {covered_answers}/{total_answers} ({covered_answers/total_answers:.1%})")
    
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

        # Label is the answer mapped to our smaller classification space
        answer = item["multiple_choice_answer"]
        assert len(answer.split()) == 1, f"Answer must be a single word, got: {answer}"

        # Map the answer to our dictionary instead of using raw token IDs
        label_id = self.answer_to_idx.get(answer, self.unk_answer_idx)

        return {
            "image_tensor": _pixel_values, # Only return image bytes for the model
            "image_bytes": _image_bytes, # Original image bytes for visualisation
            "input_ids": _input_ids,
            "attention_mask": _attention_mask, # Original attention mask for input_ids
            "label_id": label_id,  # Class index from our answer dictionary
            "answer_text": answer  # Original answer text for debugging
        }