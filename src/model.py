import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from PIL import Image
import io
from src.config import PROCESSED_DATA_DIR

import pprint
import pickle

from torchvision import transforms

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        H = self.n_heads
        q = self.q_proj(x).view(B, T, H, -1).transpose(1, 2)  # (B, H, T, D/H)
        k = self.k_proj(x).view(B, T, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, -1).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, D/H)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class UnifiedAutoregressiveDecoder(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        max_len=77,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.max_len = max_len
        self.d_model = d_model

        # Unfreeze text embeddings for adaptation
        for p in self.clip.vision_model.parameters():
            p.requires_grad = False
        for p in self.clip.text_model.embeddings.parameters():
            p.requires_grad = True  # Unfreeze text embeddings

        # Add a learnable position embedding for the image token
        self.img_pos_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.img_pos_embedding, std=0.02)

        self.image_proj = nn.Linear(self.clip.vision_model.config.hidden_size, d_model)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, self.tokenizer.vocab_size)

    def get_image_embedding(self, pixel_values):
        with torch.no_grad():
            vis_out = self.clip.vision_model(pixel_values=pixel_values)
        # Use pooled_output (CLS token) instead of mean pooling
        pooled = vis_out.pooler_output  # (B, D)
        return self.image_proj(pooled).unsqueeze(1)  # (B, 1, D)

    def get_text_input_embeddings(self, input_ids):
        # Unfrozen text embeddings
        token_emb = self.clip.text_model.embeddings.token_embedding(input_ids)
        pos_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_emb = self.clip.text_model.embeddings.position_embedding(pos_ids + 1)  # shift by 1 for image token
        return token_emb + pos_emb  # (B, T, D)

    def causal_mask(self, sz, device):
        return torch.tril(torch.ones((sz, sz), device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, images, input_ids, attention_mask=None):
        # Use CLIPProcessor for image normalization
        with torch.no_grad():
            processed = self.processor(images=[img for img in images], return_tensors="pt")
            pixel_values = processed["pixel_values"].to(images.device)
        image_emb = self.get_image_embedding(pixel_values)  # (B, 1, D)
        text_emb = self.get_text_input_embeddings(input_ids)  # (B, T, D)
        # Add image position embedding
        image_emb = image_emb + self.img_pos_embedding
        x = torch.cat([image_emb, text_emb], dim=1)  # (B, 1+T, D)
        seq_len = x.size(1)
        # Causal mask
        mask = self.causal_mask(seq_len, x.device)
        # Padding mask
        if attention_mask is not None:
            pad_mask = F.pad(attention_mask, (1, 0), value=1)  # Add 1 for image token
            mask = mask * pad_mask[:, None, None, :]
        for block in self.decoder_blocks:
            x = block(x, mask)
        logits = self.lm_head(x[:, 1:, :])  # Predict only text tokens, exclude image token
        return logits

    def generate_caption(self, images, max_new_tokens=50, decoding="greedy", num_beams=3, top_k=0, top_p=1.0):
        caption_tokens = self.generate_caption_token(images, max_new_tokens, decoding, num_beams, top_k, top_p)
        return self.decode_tokens(caption_tokens)

    def generate_caption_token(self, images, max_new_tokens=50, decoding="greedy", num_beams=3, top_k=0, top_p=1.0):
        # Only greedy and top-k sampling for now
        start_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        device = next(self.parameters()).device
        B = images.size(0)
        tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = self.forward(images, tokens)
            next_token_logits = logits[:, -1, :]
            if decoding == "greedy":
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            elif decoding == "topk":
                probs = F.softmax(next_token_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
                next_token = topk_indices[torch.arange(B), torch.multinomial(topk_probs, 1).squeeze(-1)].unsqueeze(-1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
            if tokens.size(1) >= self.max_len:
                break
        return tokens

    def decode_tokens(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

if __name__ == "__main__":
    with open(PROCESSED_DATA_DIR / "flickr30k/100_images.pkl", "rb") as f:
        images = pickle.load(f)

    pprint.pprint({
        "First image ID": list(images.keys())[0],
        "First image metadata": {
            "image_size": images[list(images.keys())[0]]["image_size"],
            "image_format": images[list(images.keys())[0]]["image_format"],
            "image_mode": images[list(images.keys())[0]]["image_mode"],
            "bytes_length": len(images[list(images.keys())[0]]["image_bytes"]),
            "caption_ids": images[list(images.keys())[0]]["caption_ids"],
        }
    })

    with open(PROCESSED_DATA_DIR / "flickr30k/100_captions.pkl", "rb") as f:
        captions = pickle.load(f)
    
    # Convert dictionary to list for proper splitting
    image_ids = list(images.keys())
    # random.shuffle(image_ids)
    
    # Calculate split indices
    train_size = int(len(image_ids) * 0.8)
    test_size = int(len(image_ids) * 0.1)
    
    # Split the image IDs
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:train_size+test_size]
    val_ids = image_ids[train_size+test_size:]
    
    # Create dictionaries for each split
    train_images = {id: images[id] for id in train_ids}
    test_images = {id: images[id] for id in test_ids}
    val_images = {id: images[id] for id in val_ids}
    
    print(f"Train images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Validation images: {len(val_images)}")

    model = UnifiedAutoregressiveDecoder()
    
    # Process the first image
    first_id = list(images.keys())[0]
    first_image_data = images[first_id]
    
    # Convert bytes back to PIL Image for processing
    pil_image = Image.open(io.BytesIO(first_image_data["image_bytes"]))

    image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    pil_image = image_transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    # Get first caption ID and convert to input_ids that the model expects
    caption_id = first_image_data['caption_ids'][0]
    caption_text = captions[caption_id]['caption']
    
    # Use the model's tokenizer to convert text to input_ids
    input_ids = model.tokenizer(caption_text, return_tensors="pt").input_ids
    
    # Forward pass through the model
    result = model(pil_image, input_ids)
    
    print(f"{result}")

    # a generate example

    # Direct pass of the image dictionary to test our new preprocess_images method
    generated_text = model.generate_caption(pil_image, max_new_tokens=50)

    print(f"Generated text: {generated_text}")
    print(f"Generated text shape: {len(generated_text)}")