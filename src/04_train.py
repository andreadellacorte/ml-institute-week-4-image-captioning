from PIL import Image
import io
import time

from datetime import datetime

from src.model import UnifiedAutoregressiveDecoder

from src.dataset import ImageCaptioningDataset

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR

from loguru import logger

import pickle
import random
import torch
import wandb
import os

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast  # Added for AMP

# WandB Configuration Settings
WANDB_CONFIG = {
    # Set to 'online', 'offline', or 'disabled'
    "mode": "online",
    "project": "mlx7-week-4-image-captioning",
    "entity": None,  # Set to your wandb username/team or None
}

# Sweep Configuration - Single (for quick targeted sweeps)
SWEEP_CONFIG_SINGLE = {
    "method": "random",
    "metric": {
        "name": "train_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dataset_size": {
            "values": [500, 1000]
        },
        "batch_size": {
            "values": [8, 16, 32]
        },
        "learning_rate": {
            "values": [1e-4, 5e-4]
        },
        "num_epochs": {
            "value": 2  # Fewer epochs for quick sweep
        },
        "optimizer": {
            "value": "adam"
        },
        "max_len": {
            "value": 22
        },
        "step_function": {
            "value": "full_sentence"
        },
        "resize_size": {
            "value": 224
        },
        "normalize": {
            "value": True
        },
        # Adding model architecture parameters
        "d_model": {
            "value": 512
        },
        "n_layers": {
            "value": 6
        },
        "n_heads": {
            "value": 8
        },
        "d_ff": {
            "value": 2048
        },
        "seed": {
            "value": 3047
        },
        "dropout_prob": {  # Added dropout_prob
            "value": 0.1
        }
    }
}

# Sweep Configuration - Full (for comprehensive exploration)
SWEEP_CONFIG_FULL = {
    "method": "bayes",
    "metric": {
        "name": "train_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dataset_size": {
            "values": [10]
        },
        "batch_size": {
            "values": [8]
        },
        "learning_rate": {
            "values": [1e-4]
        },
        "num_epochs": {
            "values": [20]
        },
        "optimizer": {
            "values": ["adam"]
        },
        "max_len": {
            "values": [25]
        },
        "step_function": {
            "values": ["full_sentence"]
        },
        "resize_size": {
            "values": [224]
        },
        "normalize": {
            "values": [False]
        },
        "d_model": {
            "values": [512]
        },
        "n_layers": {
            "values": [6]
        },
        "n_heads": {
            "values": [8]
        },
        "d_ff": {
            "values": [2048]
        },
        "seed": {
            "values": [3047]
        },
        "dropout_prob": {  # Added dropout_prob
            "values": [0.0, 0.1, 0.2]
        },
    }
}

# Sweep Configuration for Debugging Slowdown
SWEEP_CONFIG_DEBUG_SLOWDOWN = {
    "method": "grid",  # Grid search for a single specific configuration
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dataset_size": {
            "value": "5000"
        },
        "batch_size": {
            "value": 256
        },
        "learning_rate": {
            "value": 1e-4  # Fixed learning rate
        },
        "num_epochs": {
            "value": 5  # Reduced epochs for faster debugging turn-around
        },
        "optimizer": {
            "value": "adam"
        },
        "max_len": {
            "value": 25
        },
        "step_function": {
            "value": "full_sentence"
        },
        # Fixed small model architecture
        "d_model": {
            "value": 512
        },
        "n_layers": {
            "value": 4
        },
        "n_heads": {
            "value": 16
        },
        "d_ff": {
            "value": 2048
        },
        "seed": {
            "value": 42
        },
        "dropout_prob": {
            "value": 0.0
        },
        "length_penalty_weight": {
            "value": 0.5
        }
    }
}

os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for overfit test

sweep_config = SWEEP_CONFIG_DEBUG_SLOWDOWN  # Use the debug config

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main():
    """Main function that serves as both regular training entry point and sweep agent."""
    # Set wandb mode
    os.environ["WANDB_MODE"] = WANDB_CONFIG["mode"]

    # Create the sweep - remove the name parameter
    sweep_id = wandb.sweep(
        sweep_config, 
        project=f"{WANDB_CONFIG['project']}", 
        entity=WANDB_CONFIG["entity"]
    )
    logger.info(f"Created sweep with ID: {sweep_id}")
    try:
        # Construct the sweep path string, handling None entity
        entity = WANDB_CONFIG.get("entity")
        if entity is None:
            # Attempt to get default entity if not specified
            # This might require wandb.Api() to be initialized or user to be logged in
            try:
                entity = wandb.Api().default_entity
            except Exception as api_err:
                logger.warning(f"Could not determine default wandb entity: {api_err}. Sweep URL might be incomplete if entity is required.")
                entity = "YOUR_ENTITY" # Placeholder if default entity fetch fails

        project_name = WANDB_CONFIG['project']
        
        # Ensure entity and project_name are not None before forming the path
        if entity and project_name and sweep_id:
            sweep_path_str = f"{entity}/{project_name}/{sweep_id}"
            sweep_url = wandb.Api().sweep(sweep_path_str).url
            logger.info(f"Sweep URL: {sweep_url}")
        else:
            logger.warning("Could not form sweep URL due to missing entity, project, or sweep_id.")
            
    except Exception as e:
        logger.warning(f"Could not retrieve sweep URL: {e}")

    # Start the sweep agent
    wandb.agent(sweep_id, train_model)

def train_model():
    """Train the model with the given configuration."""
    # Initialize wandb run
    run = wandb.init(project=WANDB_CONFIG["project"], entity=WANDB_CONFIG["entity"])
    
    config = run.config
    logger.info(f"Using configuration: {config}")  # Log the whole config
    logger.info(f"Actual batch_size from config: {config.batch_size}")
    
    # Set random seed
    set_seed(config.seed)

    # Load the data
    with open(PROCESSED_DATA_DIR / f"flickr30k/{config.dataset_size}_images.pkl", "rb") as f:
        images = pickle.load(f)

    with open(PROCESSED_DATA_DIR / f"flickr30k/{config.dataset_size}_captions.pkl", "rb") as f:
        captions = pickle.load(f)
    
    # Convert dictionary to list for proper splitting
    image_ids = list(images.keys())
    
    # Calculate split indices
    train_size = int(len(image_ids) * 0.9)
    test_size = int(len(image_ids) * 0.1)
    
    # Split the image IDs
    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:train_size+test_size]
    
    # Create dictionaries for each split
    train_images = {id: images[id] for id in train_ids}
    test_images = {id: images[id] for id in test_ids}
    
    logger.info(f"Train images: {len(train_images)}")
    logger.info(f"Test images: {len(test_images)}")

    model = UnifiedAutoregressiveDecoder(
        clip_model_name="openai/clip-vit-base-patch32",
        max_len=config.max_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout_prob=config.dropout_prob,  # Pass dropout_prob from config
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Model loaded")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"Initial GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    dataset = ImageCaptioningDataset(
        train_images,
        captions,
        model)

    test_dataset = ImageCaptioningDataset(
        test_images,
        captions,
        model)

    logger.info("Dataset loaded")

    # DataLoader optimizations
    num_workers = 4 if device.type == 'cuda' else 0 # Use workers if on GPU
    pin_memory = True if device.type == 'cuda' else False

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info("Dataloader loaded")
    logger.info(f"Length of dataloader: {len(dataloader)}")

    # Configure optimizer based on config
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    # Use ignore_index for PAD tokens
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    
    # Initialize GradScaler for AMP if on CUDA
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Select step function based on config
    if config.step_function == "logit_by_logit":
        step_function_impl = logit_by_logit_step
    else:  # Default to full_sentence
        step_function_impl = full_sentence_step
    
    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start_time = datetime.now()
        # Pass both the implementation and the name of the step function
        # Also pass length_penalty_weight and pad_token_id
        loss = train_one_epoch(
            model, 
            dataloader, 
            optimizer, 
            criterion, 
            step_function_impl, 
            config.step_function,
            config.length_penalty_weight, # Pass new config param
            model.tokenizer.pad_token_id, # Pass pad_token_id
            scaler # Pass GradScaler
        )
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        logger.info(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss:.4f}, Duration: {epoch_duration:.2f}s")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss,
            "epoch_duration": epoch_duration,
        })
        
        # Evaluate model every other epoch to save time
        if (epoch + 1) % 2 == 0 or epoch == config.num_epochs - 1:
            evaluate(model, test_dataset)
            val_loss = validate(
                model, 
                test_dataset, 
                criterion, 
                config.length_penalty_weight, 
                model.tokenizer.pad_token_id 
            )
            wandb.log({"validation_loss": val_loss}) # Log validation loss
    
    # Final logging of metrics
    wandb.log({
        "final_train_loss": loss,
    })
    
    # Close wandb run
    wandb.finish()

def train_one_epoch(model, dataloader, optimizer, criterion, step_function_impl, step_function_name, length_penalty_weight, pad_token_id, scaler): # Added scaler
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda') # Determine if AMP should be used

    if device.type == 'cuda':
        logger.info(f"Start of Epoch GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        images = batch["image_tensor"].to(device)
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        data_to_device_time = time.time() - batch_start_time
        
        loss_val = 0.0
        timings = { 
            'total_step_time': 0.0, 'forward_time': 0.0, 
            'loss_time': 0.0, 'backward_time': 0.0, 'optim_time': 0.0
        }
        
        # Autocast for the step function if on CUDA
        with autocast(enabled=use_amp):
            if step_function_name == "logit_by_logit":
                # logit_by_logit_step does not take attention_mask and returns a float
                # It handles its own AMP scaling internally if needed
                current_loss = step_function_impl(
                    model, images, input_ids, label_ids, optimizer, criterion,
                    length_penalty_weight, pad_token_id, device, scaler, use_amp # Pass scaler and use_amp
                )
                loss_val = current_loss  # Already a float
                # Timings will use the initialized zero values as logit_by_logit doesn't break them down here.
            else:  # full_sentence or other step functions following this pattern
                # full_sentence_step takes attention_mask and returns a tensor and detailed timings
                # AMP is handled by the autocast context here, scaler is passed for backward/step
                loss_tensor, returned_timings = step_function_impl(
                    model, images, input_ids, label_ids, attention_mask, optimizer, criterion,
                    length_penalty_weight, pad_token_id, device, scaler, use_amp # Pass scaler and use_amp
                )
                loss_val = loss_tensor.item() # loss_tensor is already after potential scaling
                timings = returned_timings # Update with actual timings from the step function
            
        total_loss += loss_val # loss_val is item(), so it's detached from graph
        batch_end_time = time.time()
        progress_bar.set_postfix(loss=f"{loss_val:.4f}")
        if batch_idx % 100 == 0: # Log timings every 100 batches
            logger.info(f"Batch {batch_idx}: Data to device = {data_to_device_time:.4f}s, Step func = {timings.get('total_step_time', 0.0):.4f}s (Forward: {timings.get('forward_time', 0.0):.4f}s, Loss: {timings.get('loss_time', 0.0):.4f}s, Backward: {timings.get('backward_time', 0.0):.4f}s, Optim: {timings.get('optim_time', 0.0):.4f}s)")
            if device.type == 'cuda':
                logger.info(f"Batch {batch_idx} GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    return total_loss / len(dataloader)

def full_sentence_step(model, images, input_ids, label_ids, attention_mask, optimizer, criterion, length_penalty_weight, pad_token_id, device, scaler, use_amp): # Added scaler, use_amp
    step_start_time = time.time()
    timings = {}

    optimizer.zero_grad()
    
    # Forward pass is already within autocast in train_one_epoch
    forward_start_time = time.time()
    # No need for explicit autocast here if train_one_epoch's loop is already under autocast
    outputs = model(images, input_ids, attention_mask=attention_mask)
    timings['forward_time'] = time.time() - forward_start_time
    
    loss_start_time = time.time()
    loss = criterion(outputs.view(-1, outputs.size(-1)), label_ids.view(-1))
    
    if length_penalty_weight > 0:
        num_non_pad_tokens = (label_ids != pad_token_id).sum(dim=1).float()
        avg_caption_length = num_non_pad_tokens.mean()
        penalty = length_penalty_weight * avg_caption_length
        loss = loss + penalty
    timings['loss_time'] = time.time() - loss_start_time
        
    backward_start_time = time.time()
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    timings['backward_time'] = time.time() - backward_start_time
    
    optim_start_time = time.time()
    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    timings['optim_time'] = time.time() - optim_start_time

    timings['total_step_time'] = time.time() - step_start_time
    return loss, timings # Return tensor loss for .item() in the caller

def logit_by_logit_step(model, images, input_ids, label_ids, optimizer, criterion, length_penalty_weight, pad_token_id, device, scaler, use_amp): # Added scaler, use_amp
    seq_length = input_ids.shape[1]
    batch_total_loss = 0
    sequence_positions = 0
    
    for i in range(1, seq_length):
        curr_input = input_ids[:, :i]
        optimizer.zero_grad()
        
        # Forward pass with autocast
        # No explicit autocast here if train_one_epoch's loop is already under autocast
        outputs = model(images, curr_input)
        
        target_idx = i
        if target_idx < seq_length:
            curr_target = label_ids[:, target_idx]
            last_token_logits = outputs[:, -1, :]
            step_loss = criterion(last_token_logits, curr_target)
            
            if length_penalty_weight > 0:
                penalty = length_penalty_weight * i 
                step_loss = step_loss + penalty

            # Scale loss and backward pass
            scaler.scale(step_loss).backward()
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update() # Call update after every step for logit_by_logit
            
            batch_total_loss += step_loss.item()
            sequence_positions += 1
        
        if (curr_input == model.tokenizer.eos_token_id).any(dim=1).any():
            break
        if i >= model.max_len - 1:
            break
            
    if sequence_positions > 0:
        avg_batch_loss = batch_total_loss / sequence_positions
    else:
        avg_batch_loss = 0.0
        
    return avg_batch_loss

def evaluate(model, test_dataset):
    model.eval()
    logger.info("Evaluation:")
    
    # Select random samples from the dataset
    num_samples = min(6, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    
    # Lists to store results
    original_images = []
    generated_captions = []
    ground_truth_captions = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for idx in sample_indices:
            sample = test_dataset[idx]
            image = sample["image_tensor"].unsqueeze(0).to(device)  # Add batch dimension
            label_ids = sample["label_ids"].unsqueeze(0).to(device)
            # Use the original image bytes for visualization
            original_images.append(sample["image_bytes"])
            generated_caption = model.generate_caption(image, max_new_tokens=15)[0].split()
            ground_truth = model.decode_tokens(label_ids.squeeze(0).cpu().numpy())
            generated_captions.append(generated_caption)
            ground_truth_captions.append(ground_truth)
    
    # Print results
    logger.info("\nGenerated captions:")
    for i in range(num_samples):
        logger.info(f"Image {i+1}:")
        logger.info(f"Generated: {' '.join(generated_captions[i])}")
        logger.info(f"Ground truth: {ground_truth_captions[i]}")
        logger.info("---")
    
    # Import the plotting function
    from src.plots import plot_images_with_captions
    
    # Create a list of captions with both generated and ground truth
    combined_captions = [f"Generated: {' '.join(gen)}\nGround truth: {gt}" 
                         for gen, gt in zip(generated_captions, ground_truth_captions)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot images with captions
    plot_images_with_captions(
        images=original_images,
        captions=combined_captions,
        title="Image Captioning Results",
        save_path=FIGURES_DIR / f"{timestamp}_caption_results.png",
        show=False
    )
    
    # Log results to wandb
    if wandb.run is not None:
        # Create a wandb table for the results
        caption_table = wandb.Table(columns=["Image", "Generated Caption", "Ground Truth"])
        for img, gen, gt in zip(original_images, generated_captions, ground_truth_captions):
            pil_img = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
            caption_table.add_data(wandb.Image(pil_img), " ".join(gen), gt)
        
        # Log the table
        wandb.log({"caption_examples": caption_table})

def validate(model, val_dataset, criterion, length_penalty_weight, pad_token_id):
    model.eval()
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda') # Determine if AMP should be used for validation
    
    # Optimized DataLoader for validation
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=16, # Consider making this configurable or larger for validation
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image_tensor"].to(device)
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            with autocast(enabled=use_amp): # Use autocast for validation forward pass
                outputs = model(images, input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), label_ids.view(-1))

            # Add length penalty consistent with training
            if length_penalty_weight > 0:
                num_non_pad_tokens = (label_ids != pad_token_id).sum(dim=1).float()
                avg_caption_length = num_non_pad_tokens.mean()
                penalty = length_penalty_weight * avg_caption_length
                loss = loss + penalty
                
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == "__main__":
   main()