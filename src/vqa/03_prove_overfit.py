from src.model import UnifiedAutoregressiveDecoder

from src.vqa.dataset import VQADataset

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR

from loguru import logger

import pickle
import random
import torch
import os

from tqdm import tqdm

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

def main():
    # Set wandb mode
    os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for overfit test

    set_seed(42)

    # Always load 10 samples for overfit test
    with open(PROCESSED_DATA_DIR / "VQAv2/10_images.pkl", "rb") as f:
        images = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "VQAv2/1_index.pkl", "rb") as f:
        index = pickle.load(f)

    # Select 1 image for trivial overfit
    index = index[:10]

    # Use same data for train and test
    train_index = index
    test_index = index

    logger.info(f"Overfit mode: using {len(train_index)} questions for both train and test")

    # Model config (fixed for overfit, dropout=0)
    model = UnifiedAutoregressiveDecoder(
        clip_model_name="openai/clip-vit-base-patch16",
        max_len=25,
        d_model=256,
        n_layers=2,
        n_heads=8,
        d_ff=256,
        dropout_prob=0.05,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use VQADataset for train and validation
    dataset = VQADataset(
        images,
        train_index,
        model,
        clean_questions=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Use same for validation
    val_dataset = VQADataset(
        images,
        test_index,
        model,
        clean_questions=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # Switched to AdamW optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decrease LR every 10 epochs
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)

    # Print detailed debug info for the first sample
    sample = dataset[0]

    # Get caption_id from the dataset's internal data list
    logger.info(f"Label token ids: {sample['label_ids']}")
    logger.info(f"Decoded text: {model.decode_tokens(sample['label_ids'])}")
    logger.info(f"Input token ids: {sample['input_ids']}")
    logger.info(f"Input text: {model.decode_tokens(sample['input_ids'])}")

    # Overfit loop
    for epoch in range(100):  # Increased from 20 to 100
        loss = train_one_epoch(model, dataloader, optimizer, criterion, full_sentence_step)
        scheduler.step()  # Step the scheduler after each epoch
        logger.info(f"[Overfit] Epoch {epoch+1}/100, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]}")
        # Print generated captions every epoch
        model.eval()
        with torch.no_grad():
            sample = val_dataset[0]
            image = sample["image_tensor"].unsqueeze(0).to(device)
            label_ids = sample["label_ids"].unsqueeze(0).to(device)
            # Force generation to match label length (excluding padding)
            label_len = (label_ids != model.tokenizer.pad_token_id).sum().item()
            generated = model.generate_caption(image, max_new_tokens=label_len)
            if isinstance(generated, list):
                generated_text = generated[0]
            else:
                generated_text = generated
            generated_ids = model.tokenizer(generated_text, return_tensors="pt").input_ids[0]
            logger.info(f"Sample 1:\n  Generated: {generated_text}\n  Ground truth: {model.decode_tokens(label_ids.squeeze(0))}")
            logger.info(f"Generated token ids: {generated_ids}")
            logger.info(f"Ground truth token ids: {label_ids.squeeze(0)}")
        model.train()

def train_one_epoch(model, dataloader, optimizer, criterion, step_function):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    for batch in progress_bar:
        images = batch["image_tensor"].to(device)
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Track total loss
        loss = step_function(model, images, input_ids, label_ids, attention_mask, optimizer, criterion)
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def full_sentence_step(model, images, input_ids, label_ids, attention_mask, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(images, input_ids, attention_mask=attention_mask)
    loss = criterion(outputs.view(-1, outputs.size(-1)), label_ids.view(-1))
    loss.backward()
    optimizer.step()
    return loss

if __name__ == "__main__":
    main()