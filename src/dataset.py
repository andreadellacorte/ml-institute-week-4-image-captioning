import io

from torch.utils.data import Dataset
from PIL import Image

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, model, resize_size, max_len, normalize_image):
        self.images = images
        self.captions = captions
        self.tokenizer = model.tokenizer
        self.bos_token = model.tokenizer.bos_token
        self.eos_token = model.tokenizer.eos_token
        self.pad_token = model.tokenizer.pad_token
        self.pad_token_id = model.tokenizer.pad_token_id
        self.max_len = max_len
        self.model = model
        self.processor = model.processor  # Use CLIPProcessor for image normalization
        self.data = []
        for img_id, img_data in self.images.items():
            for caption_id in img_data["caption_ids"]:
                self.data.append({
                    "img_id": img_id,
                    "caption_id": caption_id,
                    "image_bytes": img_data["image_bytes"]  # Store original bytes for visualization
                })
    
    def __len__(self):
        return len(self.data)
    
    def clean_text(self, text):
        # Less aggressive cleaning: keep alphanum and basic punctuation
        import string
        allowed = set(string.ascii_letters + string.digits + " .,'-?!")
        return ''.join([c for c in text if c in allowed]).strip()
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item["img_id"]
        caption_id = item["caption_id"]
        image = Image.open(io.BytesIO(self.images[img_id]["image_bytes"]))
        # Use CLIPProcessor for normalization
        image_tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        caption = self.captions[caption_id]["caption"]
        caption = self.clean_text(caption)
        input_text = f"{self.bos_token} {caption}"
        label = f"{caption} {self.eos_token}"
        tokenized_input = self.tokenizer(
            input_text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized_input.input_ids[0]
        attention_mask = tokenized_input.attention_mask[0]
        tokenized_label = self.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        label_ids = tokenized_label.input_ids[0]
        # Set PAD tokens in label_ids to pad_token_id for loss masking
        label_ids[tokenized_label.attention_mask[0] == 0] = self.pad_token_id
        return {
            "image_tensor": image_tensor,        # for model
            "image_bytes": item["image_bytes"],  # return original bytes for visualization
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask
        }