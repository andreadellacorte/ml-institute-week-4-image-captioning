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

# WandB Configuration Settings
WANDB_CONFIG = {
    # Set to 'online', 'offline', or 'disabled'
    "mode": "online",
    "project": "mlx7-week-4-image-captioning",
    "entity": None,  # Set to your wandb username/team or None
}

# Default training configuration
DEFAULT_CONFIG = {
    "seed": 3047,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 5,
    "optimizer": "adam",
    "max_len": 22,
    "step_function": "full_sentence",
    "dataset_size": 1000,
    "resize_size": 224,
    "normalize": True,
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
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
            "values": [500]
        },
        "batch_size": {
            "values": [8]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3
        },
        "num_epochs": {
            "values": [3]
        },
        "optimizer": {
            "values": ["adam"]
        },
        "max_len": {
            "values": [16, 22, 32]
        },
        "step_function": {
            "values": ["full_sentence"]
        },
        "resize_size": {
            "values": [224]
        },
        "normalize": {
            "values": [True]
        },
        # Adding model architecture parameters
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
    }
}

sweep_config = SWEEP_CONFIG_FULL

# Number of runs to perform in the sweep
SWEEP_RUNS = 1

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        
    # Start the sweep agent
    wandb.agent(sweep_id, train_model, count=SWEEP_RUNS)

def train_model():
    """Train the model with the given configuration."""
    # Initialize wandb run
    run = wandb.init(project=WANDB_CONFIG["project"], entity=WANDB_CONFIG["entity"])
    
    config = run.config
    
    # Set random seed
    set_seed(config.seed)

    # Load the data
    with open(PROCESSED_DATA_DIR / "flickr30k/1000_images.pkl", "rb") as f:
        images = pickle.load(f)

    with open(PROCESSED_DATA_DIR / "flickr30k/1000_captions.pkl", "rb") as f:
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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Model loaded")

    dataset = ImageCaptioningDataset(
        train_images,
        captions,
        model,
        max_len=config.max_len,
        resize_size=config.resize_size,
        normalize_image=config.normalize)

    logger.info("Dataset loaded")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    logger.info("Dataloader loaded")

    # Configure optimizer based on config
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    criterion = torch.nn.CrossEntropyLoss()
    
    # Select step function based on config
    if config.step_function == "logit_by_logit":
        step_function = logit_by_logit_step
    else:  # Default to full_sentence
        step_function = full_sentence_step
    
    # Training loop
    batch_times = []
    
    for epoch in range(config.num_epochs):
        epoch_start_time = datetime.now()
        loss, epoch_batch_times = train_one_epoch(model, dataloader, optimizer, criterion, step_function)
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
        
        # Track batch times
        batch_times.extend(epoch_batch_times)
        
        logger.info(f"Epoch [{epoch+1}/{config.num_epochs}], Loss: {loss:.4f}, Duration: {epoch_duration:.2f}s")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss,
            "epoch_duration": epoch_duration,
        })
        
        # Evaluate model every other epoch to save time
        if (epoch + 1) % 2 == 0 or epoch == config.num_epochs - 1:
            evaluate(model, test_images, captions)
    
    # Final logging of metrics
    wandb.log({
        "final_train_loss": loss,
    })
    
    # Close wandb run
    wandb.finish()

def train_one_epoch(model, dataloader, optimizer, criterion, step_function):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    batch_times = []
    
    for batch in progress_bar:
        batch_start = datetime.now()
        
        images = batch["image_bytes"].to(device)
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        
        # Track total loss
        loss = step_function(model, images, input_ids, label_ids, optimizer, criterion)

        total_loss += loss.item()
        
        # Update progress bar with non-zero loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Track batch processing time
        batch_end = datetime.now()
        batch_time = (batch_end - batch_start).total_seconds()
        batch_times.append(batch_time)
    
    # Return average loss over all batches and batch times
    return total_loss / len(dataloader), batch_times

def full_sentence_step(model, images, input_ids, label_ids, optimizer, criterion):
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(images, input_ids)
    
    # Calculate loss
    loss = criterion(outputs.view(-1, outputs.size(-1)), label_ids.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()

    return loss

def logit_by_logit_step(model, images, input_ids, label_ids, optimizer, criterion):
    seq_length = input_ids.shape[2]
    
    # Initialize running loss scalar
    batch_total_loss = 0
    sequence_positions = 0
    
    # For each position in the sequence
    for i in range(1, seq_length):
        # Get progressively longer chunks of input
        curr_input = input_ids[:, :, :i]
        
        # Zero gradients for each step
        optimizer.zero_grad()
        
        # Use the forward method to get predictions
        outputs = model(images, curr_input)
        
        # Get corresponding target (next token predictions)
        # We want to predict the next token at each position
        target_idx = i  # The next token after our current input
        if target_idx < seq_length:
            curr_target = label_ids[:, 0, target_idx]  # Access correct dimension
            
            # Calculate loss for this step (predict next token)
            # outputs shape: [batch_size, seq_len, vocab_size]
            # We want the prediction for the last position in the sequence
            last_token_logits = outputs[:, -1, :]  # Get logits for the last token
            step_loss = criterion(last_token_logits, curr_target)
            
            # Backward pass for this step's loss
            step_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Add to running loss
            batch_total_loss += step_loss.item()
            sequence_positions += 1
        
        # Check if any sequence has reached EOS token
        if (curr_input == model.tokenizer.eos_token_id).any(dim=1).any():
            break
            
        # Check if we've reached model's maximum length
        if i >= model.max_len - 1:
            break
    
    # Calculate average loss for this batch
    if sequence_positions > 0:
        avg_batch_loss = batch_total_loss / sequence_positions
    else:
        avg_batch_loss = 0
        
    return avg_batch_loss

def evaluate(model, test_images, test_captions):
    model.eval()
    logger.info("Evaluation:")
    
    # Create the test dataset
    test_dataset = ImageCaptioningDataset(test_images, test_captions, model)
    
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
            # Get a single sample
            sample = test_dataset[idx]

            # Process inputs
            image = sample["image_bytes"].unsqueeze(0).to(device)  # Add batch dimension
            label_ids = sample["label_ids"].unsqueeze(0).to(device) # Add batch dimension

            # Store original image for plotting
            original_images.append(sample["image_bytes"])

            # Use the generate_caption method to get the caption
            generated_caption = model.generate_caption(image, max_new_tokens=15)[0].split()

            # For ground truth, get the text from the label_ids
            ground_truth = model.decode_tokens(label_ids.squeeze(0).cpu().numpy())

            # Store results
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
            caption_table.add_data(wandb.Image(img), " ".join(gen), gt)
        
        # Log the table
        wandb.log({"caption_examples": caption_table})

def validate(model, val_images, val_captions):
    # Implement your validation loop here
    pass

if __name__ == "__main__":
    main()