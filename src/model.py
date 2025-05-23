from loguru import logger

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPProcessor, CLIPTokenizer

from src.config import PROCESSED_DATA_DIR

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

        attn = torch.nn.functional.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, D/H)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)

class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout_prob=0.1):  # Added dropout_prob
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout_prob),  # Use dropout_prob
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout_prob),  # Use dropout_prob
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
        dropout_prob=0.1,  # Added dropout_prob
    ):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.max_len = max_len
        self.d_model = d_model

        # Freeze the entire CLIP model
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.image_proj = nn.Linear(self.vision_model.config.hidden_size, d_model, bias=False)  # Added image projection
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, d_model, bias= False)  # Added text projection

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout_prob=dropout_prob) for _ in range(n_layers)  # Pass dropout_prob
        ])
        self.lm_head = nn.Linear(d_model, self.tokenizer.vocab_size)

        # Explicitly set dropout to 0 if dropout_prob is 0.0 after model initialization
        if dropout_prob == 0.0:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0

        clip_params = \
            sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad is False) \
            + sum(p.numel() for p in self.text_model.parameters() if p.requires_grad is False)
        decoder_params = sum(p.numel() for p in self.parameters()) - clip_params
        total_params = clip_params + decoder_params

        logger.info(f"Total parameters: {total_params}")

        # assert clip has 0 trainable parameters
        assert not any(p.requires_grad for p in self.vision_model.parameters()), "CLIP model should have no trainable parameters"
        assert not any(p.requires_grad for p in self.text_model.parameters()), "CLIP model should have no trainable parameters"

        # print minus the number parameters in clip
        
        logger.info(f"CLIP parameters (all frozen): {clip_params}")
        logger.info(f"Decoder parameters (all trainable): {decoder_params}")

        # print the ratio of decoder parameters to total parameters
        logger.info(f"Decoder parameters ratio: {decoder_params / total_params:.2%}%")
    
    def process_images(self, images):
        return self.processor(images=images, return_tensors="pt")

    def get_image_embedding(self, pixel_values):
        with torch.no_grad():
            return self.vision_model(pixel_values=pixel_values).last_hidden_state  # (B, P, D)

    def get_text_input_embeddings(self, input_ids):
        with torch.no_grad():
            return self.text_model(input_ids).last_hidden_state  # (B, T, D)

    def causal_mask(self, sz, device):
        return torch.tril(torch.ones((sz, sz), device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, pixel_values, input_ids, attention_mask=None):
        image_emb = self.get_image_embedding(pixel_values)  # (B, P, D)
        text_emb = self.get_text_input_embeddings(input_ids)  # (B, T, D)

        image_proj = self.image_proj(image_emb)  # (B, P, D)
        text_proj = self.text_proj(text_emb)  # (B, T, D)
        
        x = torch.cat([image_proj, text_proj], dim=1)  # (B, P+T, D)

        seq_len = x.size(1)
        n_patches = image_proj.shape[1]  # Use actual number of patches at runtime

        # Causal mask
        mask = self.causal_mask(seq_len, x.device)

        # Padding mask
        if attention_mask is not None:
            pad_mask = torch.nn.functional.pad(
                attention_mask,
                (n_patches, 0),
                value=1)  # Pad for all image patches
            mask = mask * pad_mask[:, None, None, :]

        for block in self.decoder_blocks:
            x = block(x, mask)

        logits = self.lm_head(x[:, n_patches:, :])  # Predict only text tokens, exclude image patches
        
        return logits

    def generate_caption(self, images, max_new_tokens=50, decoding="greedy", top_k=0):
        caption_tokens = self.generate_answer(
            images,
            text=None,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            top_k=top_k)

        return self.decode_tokens(caption_tokens)

    def generate_answer(self, images, text=None, max_new_tokens=50, decoding="greedy", top_k=0):
        # Only greedy and top-k sampling for now
        device = next(self.parameters()).device
        B = images.size(0)
        
        if text is None:
            # Start with the BOS token if no text is provided
            start_token_id = self.tokenizer.bos_token_id
            tokens = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)
        else:
            # Tokenize the provided text
            tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).input_ids.to(device)
        
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
            if (next_token == self.tokenizer.eos_token_id).all():
                break
            if tokens.size(1) >= self.max_len:
                break
        return tokens

    def decode_tokens(self, tokens):
        # Accepts either a 1D or 2D tensor/array
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        if len(tokens.shape) == 1:
            # Single sequence
            tokens = tokens.tolist()
            # Remove padding tokens
            tokens = [t for t in tokens if t != self.tokenizer.pad_token_id]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            # Batch
            return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

class VQAClassifier(nn.Module):
    def __init__(self, caption_model, num_classes):
        super().__init__()
        self.caption_model = caption_model  # Use the pretrained captioning model
        self.image_encoder = caption_model.vision_model  # CLIP vision model
        self.d_model = caption_model.d_model
        
        # Project image embeddings to our model dimensions
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, self.d_model)
        
        # A simple classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, num_classes)
        )
        
        # Freeze the CLIP model for efficiency
        for p in self.image_encoder.parameters():
            p.requires_grad = False
    
    def forward(self, images):
        # Extract image features
        with torch.no_grad():
            image_embeddings = self.caption_model.get_image_embedding(images)  # (B, P, D_clip)
        
        # Project to model dimensions
        image_features = self.image_proj(image_embeddings)  # (B, P, D)
        
        # Global average pooling over patches
        pooled_features = image_features.mean(dim=1)  # (B, D)
        
        # Classification
        logits = self.classifier(pooled_features)  # (B, num_classes)
        return logits
    
    def predict_answer(self, images, idx_to_answer):
        """Predict answers using our classifier and map back to text"""
        logits = self.forward(images)
        pred_ids = logits.argmax(dim=-1).tolist()  # Convert to list of indices
        return [idx_to_answer.get(idx, "unknown") for idx in pred_ids]
        
    @property
    def tokenizer(self):
        # For compatibility with existing code
        return self.caption_model.tokenizer

class MultimodalAttention(nn.Module):
    """Multi-head cross-attention between image and text features"""
    def __init__(self, d_model, n_heads=8, dropout_prob=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Projections for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value, attention_mask=None):
        """
        query: [batch_size, query_len, d_model]
        key_value: [batch_size, kv_len, d_model]
        attention_mask: [batch_size, kv_len] or None
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        kv_len = key_value.shape[1]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, query_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # q: [batch_size, n_heads, query_len, head_dim]
        # k: [batch_size, n_heads, kv_len, head_dim]
        # scores: [batch_size, n_heads, query_len, kv_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Reshape mask for broadcasting: [batch_size, 1, 1, kv_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # attn_weights: [batch_size, n_heads, query_len, kv_len]
        # v: [batch_size, n_heads, kv_len, head_dim]
        # context: [batch_size, n_heads, query_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        # context: [batch_size, query_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        return output

class StandaloneVQAClassifier(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=3000, d_model=512, n_heads=8, n_fusion_layers=2, dropout_prob=0.1):
        super().__init__()
        # Load CLIP vision and text models directly
        self.vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Add max_len attribute for dataset compatibility
        self.max_len = 20
        
        # Get model dimensions
        vision_dim = self.vision_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        
        # Freeze both vision and text models
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Vision and text projections to align dimensions
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Multi-head fusion transformer layers
        self.fusion_layers = nn.ModuleList()
        for _ in range(n_fusion_layers):
            layer = nn.ModuleDict({
                # Image attends to text
                'image_text_attention': MultimodalAttention(d_model, n_heads, dropout_prob),
                'image_norm1': nn.LayerNorm(d_model),
                
                # Text attends to image
                'text_image_attention': MultimodalAttention(d_model, n_heads, dropout_prob),
                'text_norm1': nn.LayerNorm(d_model),
                
                # Self-attention for fused features
                'self_attention': MultiHeadAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                
                # Feed-forward layer
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout_prob)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            self.fusion_layers.append(layer)
        
        # New attention-based feature extraction mechanism
        self.img_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        self.txt_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Feature integrator without losing information
        self.feature_integrator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(d_model, num_classes)
        )
        
        # Count and log trainable parameters
        vision_params = sum(p.numel() for p in self.vision_model.parameters())
        text_params = sum(p.numel() for p in self.text_model.parameters()) 
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = vision_params + text_params + trainable_params
        
        logger.info(f"Vision model parameters (frozen): {vision_params:,}")
        logger.info(f"Text model parameters (frozen): {text_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage trainable: {trainable_params / total_params * 100:.2f}%")
    
    def process_images(self, images):
        """Process raw images into pixel values using CLIP processor"""
        return self.processor(images=images, return_tensors="pt")
    
    def forward(self, pixel_values, input_ids, attention_mask=None):
        """Forward pass through the model with both image and text inputs"""
        batch_size = pixel_values.shape[0]
        
        # Process image through vision model (no gradients)
        with torch.no_grad():
            vision_outputs = self.vision_model(pixel_values)
            image_embeddings = vision_outputs.last_hidden_state  # [batch_size, num_patches, vision_dim]
            
            # Process question through text model
            text_outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeddings = text_outputs.last_hidden_state  # [batch_size, seq_len, text_dim]
        
        # Project to common dimension
        image_features = self.vision_projection(image_embeddings)  # [batch_size, num_patches, d_model]
        text_features = self.text_projection(text_embeddings)  # [batch_size, seq_len, d_model]
        
        # Create attention mask for text-to-image attention (if needed)
        if attention_mask is not None:
            # Expand attention mask for cross attention
            # All image tokens can be attended to (1s)
            image_attention = torch.ones(batch_size, image_features.size(1), device=image_features.device)
            # Keep original text attention mask
            cross_attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        else:
            cross_attention_mask = None
        
        # Process through fusion layers
        img_feats = image_features
        txt_feats = text_features
        
        for layer in self.fusion_layers:
            # Image attends to text (cross-attention)
            img_txt = layer['image_text_attention'](img_feats, txt_feats, attention_mask)
            img_feats = img_feats + img_txt
            img_feats = layer['image_norm1'](img_feats)
            
            # Text attends to image (cross-attention)
            txt_img = layer['text_image_attention'](txt_feats, img_feats, None)  # No mask for attending to images
            txt_feats = txt_feats + txt_img
            txt_feats = layer['text_norm1'](txt_feats)
            
            # Concatenate for self-attention
            fused_feats = torch.cat([img_feats, txt_feats], dim=1)
            
            # Self-attention
            fused_feats = fused_feats + layer['self_attention'](layer['norm1'](fused_feats))
            
            # Feed-forward
            fused_feats = fused_feats + layer['ff'](layer['norm2'](fused_feats))
            
            # Split back
            img_feats, txt_feats = fused_feats.split([img_feats.size(1), txt_feats.size(1)], dim=1)
        
        # Extract key information from features using attention
        # For image features - use learned attention to focus on important regions
        img_query = torch.mean(img_feats, dim=1, keepdim=True)  # [batch_size, 1, d_model]
        attended_img_feats, _ = self.img_attention(img_query, img_feats, img_feats)
        attended_img_feats = attended_img_feats.squeeze(1)  # [batch_size, d_model]
        
        # For text features - use learned attention to focus on important words
        txt_query = torch.mean(txt_feats, dim=1, keepdim=True)  # [batch_size, 1, d_model]
        attended_txt_feats, _ = self.txt_attention(txt_query, txt_feats, txt_feats)
        attended_txt_feats = attended_txt_feats.squeeze(1)  # [batch_size, d_model]
        
        # Concatenate attended features and integrate
        multimodal_features = torch.cat([attended_img_feats, attended_txt_feats], dim=1)  # [batch_size, d_model*2]
        fused_features = self.feature_integrator(multimodal_features)  # [batch_size, d_model]
        
        # Classify
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        return logits
    
    def predict_answer(self, pixel_values, input_ids, attention_mask, idx_to_answer):
        """Predict answers using both image and question inputs"""
        logits = self.forward(pixel_values, input_ids, attention_mask)
        pred_indices = logits.argmax(dim=-1).tolist()
        return [idx_to_answer.get(idx, "unknown") for idx in pred_indices]