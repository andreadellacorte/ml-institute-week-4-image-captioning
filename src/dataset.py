from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer
from pathlib import Path
from loguru import logger

class ImageCaptioningDataset(Dataset):
    def __init__(self, images, captions, bos_token, eos_token, max_length):
        self.max_length = max_length
        
        self.images = images
            
        self.captions = captions

        self.bos_token = bos_token
        self.eos_token = eos_token
        
        self.samples = []
        for img_id, img_data in self.images.items():
            for caption_id in img_data["caption_ids"]:
                if caption_id in self.captions:
                    self.samples.append({
                        "img_id": img_id,
                        "caption_id": caption_id
                    })
        
        logger.info(f"Created dataset with {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]

        img_id = item["img_id"]
        caption_id = item["caption_id"]

        image = self.images[img_id]["image"]
        caption = self.captions[caption_id]["caption"]

        input_text = f"{self.bos_token} {caption}"
        label = f"{caption} {self.bos_token}"
        
        return {