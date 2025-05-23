from PIL import Image
import io
import time

from datetime import datetime

from src.model import StandaloneVQAClassifier

from src.vqa.dataset import VQADataset

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR, CHECKPOINTS_DATA_DIR

from loguru import logger

import pickle
import random
import torch
import wandb
import os

from tqdm import tqdm
from torch.amp import GradScaler, autocast

# WandB Configuration Settings
WANDB_CONFIG = {
    # Set to 'online', 'offline'
    "mode": "online",
    "project": "mlx7-week-4-vqa",
    "entity": None,  # Set to your wandb username/team or None
    "save_model": False, # Set to True if you want to save the model
}

# Comprehensive Sweep Configuration for optimal results
SWEEP_CONFIG = {
    "method": "bayes",  # Bayesian optimization for more efficient hyperparameter search
    "metric": {
        "name": "validation_accuracy",  # Optimize for accuracy instead of loss
        "goal": "maximize"
    },
    "parameters": {
        # Data parameters
        "dataset_size": {
            "values": ["30000"]  # Keep dataset size fixed for consistent comparison
        },
        "clip_patch_size": {
            "values": [32]  # Try both patch sizes from CLIP
        },
        "classifier_vocab_size": {
            "values": [1000]  # Different vocabulary sizes for answers
        },

        # Training parameters
        "batch_size": {
            "values": [32, 64, 128]  # Try different batch sizes
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-3
        },
        "optimizer": {
            "values": ["adam", "adamw"]  # Compare optimizers
        },
        "weight_decay": {
            "values": [0.0, 0.01, 0.001]  # Try different L2 regularization strengths
        },
        
        # Learning rate schedule
        "scheduler_type": {
            "values": ["step", "cosine", "linear_warmup"]  # Different scheduler types
        },
        "scheduler_step": {
            "values": [5, 10, 20]  # For step scheduler
        },
        "scheduler_gamma": {
            "values": [0.1, 0.5, 0.7]  # For step scheduler
        },
        "warmup_steps": {
            "values": [0, 100, 500]  # For warmup schedulers
        },
        
        # Training duration
        "num_epochs": {
            "values": [15]  # Try training longer
        },
        
        # Regularization
        "label_smoothing": {
            "values": [0.0, 0.05, 0.1]  # Try different label smoothing values
        },
        "dropout_prob": {
            "values": [0.1, 0.2, 0.3]  # Try higher dropout for better generalization
        },
        
        # Model architecture
        "d_model": {
            "values": [256, 512, 768]  # Try different model dimensions
        },
        "n_heads": {
            "values": [4, 8, 12]  # Different attention head counts
        },
        "n_fusion_layers": {
            "values": [1, 2, 3]  # Try different numbers of fusion layers
        },
        
        # Early stopping
        "patience": {
            "values": [5, 10]
        },
        "patience_min_delta_percent": {
            "values": [0.005, 0.01]
        },
        
        # Other
        "clean_questions": {
            "values": [True]  # Keep this fixed for now
        },
    }
}

sweep_config = SWEEP_CONFIG  # Use the comprehensive config

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
    wandb.agent(sweep_id, train_model, count=None)

def train_model():
    """Train the model with the given configuration."""
    # Initialize wandb run
    run = wandb.init(project=WANDB_CONFIG["project"], entity=WANDB_CONFIG["entity"])
    
    config = run.config
    logger.info(f"Using configuration: {config}")  # Log the whole config
    logger.info(f"Actual batch_size from config: {config.batch_size}")
    
    # Set random seed
    set_seed(42)

    # Load the data
    with open(PROCESSED_DATA_DIR / f"VQAv2/{config.dataset_size}_images.pkl", "rb") as f:
        images = pickle.load(f)

    with open(PROCESSED_DATA_DIR / f"VQAv2/{config.dataset_size}_index.pkl", "rb") as f:
        index = pickle.load(f)
    
    # Calculate split indices
    train_size = int(len(index) * 0.9)
    test_size = int(len(index) * 0.1)
    
    # Split the image IDs
    train_index = index[:train_size]
    random.shuffle(train_index)
    test_index = index[train_size:train_size + test_size]
    random.shuffle(test_index)
    
    logger.info(f"Train images: {len(train_index)}")
    logger.info(f"Test images: {len(test_index)}")

    # Create temporary model just to provide processor and tokenizer to datasets
    temp_model = StandaloneVQAClassifier(
        clip_model_name=f"openai/clip-vit-base-patch{config.clip_patch_size}",
        num_classes=config.classifier_vocab_size,  # Temporary value
        d_model=config.d_model,
        dropout_prob=config.dropout_prob
    )
    
    # Create datasets first to get the number of answer classes
    train_dataset = VQADataset(
        images,
        train_index,
        temp_model,
        clean_questions=config.clean_questions)
    
    test_dataset = VQADataset(
        images,
        test_index,
        temp_model,
        clean_questions=config.clean_questions)
    
    # Now create the real model with the correct number of classes
    model = StandaloneVQAClassifier(
        clip_model_name=f"openai/clip-vit-base-patch{config.clip_patch_size}",
        num_classes=train_dataset.num_classes,
        d_model=config.d_model,
        dropout_prob=config.dropout_prob
    )
    
    # Log statistics about the answer space
    logger.info(f"Answer classes: {train_dataset.num_classes} (including unknown class)")
    
    # Sample a few examples to see what we're working with
    sample_answers = list(train_dataset.answer_to_idx.keys())[:10]
    logger.info(f"Sample answers: {sample_answers}")
    
    # log the number of parameters to wandb
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params / 1e6:.2f}M")
    wandb.log({"num_params": num_params})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info("Model loaded")
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"Initial GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    logger.info("Dataset loaded")

    # DataLoader optimizations
    num_workers = 4 if device.type == 'cuda' else 0 # Use workers if on GPU
    pin_memory = True if device.type == 'cuda' else False

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info("Dataloader loaded")
    logger.info(f"Length of dataloader: {len(train_dataloader)}")

    # Configure optimizer based on config
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    # Use appropriate ignore_index for CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing
    )
    
    # Initialize GradScaler for AMP if on CUDA
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    step_function_impl = single_word_step
    
    best_val_loss = float('inf')
    patience = config.patience  # Number of epochs to wait for improvement
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start_time = datetime.now()
        loss = train_one_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            criterion, 
            step_function_impl, 
            scaler # Pass GradScaler
        )
        epoch_end_time = datetime.now()
        epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

        scheduler.step()  # Step the scheduler after each epoch
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]}, Duration: {epoch_duration:.2f}s")
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": loss,
            "epoch_duration": epoch_duration,
        })
        
        # Evaluate model every epoch
        evaluate(run, model, test_dataset, epoch)

        val_loss = validate(
            model, 
            val_dataloader, 
            criterion
        )

        wandb.log({"validation_loss": val_loss}) # Log validation loss

        # Early stopping logic with minimum expected improvement (percentage-based)
        if best_val_loss == float('inf'):
            # Always accept the first validation loss as best
            logger.success(f"Validation loss initial best: {val_loss:.4f}. Caching model.")
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()  # Save best model weights
        else:
            logger.info(f"Validation loss this epoch: {val_loss:.4f} - Prior best loss: {best_val_loss:.4f}.")
            logger.info(f"Validation loss delta: {val_loss - best_val_loss:.4f} (negative = improved).")

            min_improvement = best_val_loss * config.patience_min_delta_percent

            if val_loss < best_val_loss - min_improvement:
                logger.success(f"Validation loss improved enough. Resetting patience.")
                epochs_without_improvement = 0
            else:
                logger.warning(f"Validation loss did not improve enough (at least {min_improvement}).")
                epochs_without_improvement += 1
                logger.warning(f"Epochs since improved: {epochs_without_improvement}/{patience}")

            if val_loss < best_val_loss:
                logger.success(f"Validation loss has new best. Caching model.")
                best_val_loss = val_loss
                best_model_state = model.state_dict()

            if epochs_without_improvement >= patience:
                logger.warning("Early stopping triggered.")
                break

    # Optionally restore best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("Loaded best model weights from early stopping.")

    save_model(run, model, "best", best_val_loss)

    logger.success("Training completed successfully.")

    # Final logging of metrics
    wandb.log({
        "final_epoch": epoch + 1,
    })
    
    # Close wandb run
    wandb.finish()

def save_model(run, model, epoch, best_val_loss):

    # make a new folder in CHECKPOINTS_DATA_DIR for all the sweeps
    if not CHECKPOINTS_DATA_DIR.exists():
        CHECKPOINTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Create a sweep_id directory if it doesn't exist
    sweep_id = os.getenv("WANDB_SWEEP_ID", "local_run")
    sweep_dir = CHECKPOINTS_DATA_DIR / f"sweep_{sweep_id}"
    if not sweep_dir.exists():
        sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save the model use the wandb run name for the filename
    model_save_path = sweep_dir / f"{best_val_loss:.3f}_{run.name}_{epoch}.pt"
    torch.save(model, model_save_path)
    
    if WANDB_CONFIG["save_model"]:
        wandb.save(model_save_path)

    logger.success(f"Model saved to {model_save_path} and uploaded to wandb.")

def train_one_epoch(model, dataloader, optimizer, criterion, step_function_impl, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda') # Determine if AMP should be used

    if device.type == 'cuda':
        logger.info(f"Start of Epoch GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        image_tensor = batch["image_tensor"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_id = batch["label_id"].to(device)
        data_to_device_time = time.time() - batch_start_time
        
        loss_val = 0.0
        timings = { 
            'total_step_time': 0.0, 'forward_time': 0.0, 
            'loss_time': 0.0, 'backward_time': 0.0, 'optim_time': 0.0
        }
        
        # Autocast for the step function if on CUDA
        with autocast(device_type=device.type, enabled=use_amp):
                # Modified step function to include input_ids and attention_mask
                loss_tensor, returned_timings = step_function_impl(
                    model, image_tensor, input_ids, attention_mask, label_id, optimizer, criterion, scaler
                )
                loss_val = loss_tensor.item() # loss_tensor is already after potential scaling
                timings = returned_timings # Update with actual timings from the step function
            
        total_loss += loss_val # loss_val is item(), so it's detached from graph

        progress_bar.set_postfix(loss=f"{loss_val:.4f}")
        if batch_idx % 100 == 0: # Log timings every 100 batches
            logger.info(f"Batch {batch_idx}: Data to device = {data_to_device_time:.4f}s, Step func = {timings.get('total_step_time', 0.0):.4f}s (Forward: {timings.get('forward_time', 0.0):.4f}s, Loss: {timings.get('loss_time', 0.0):.4f}s, Backward: {timings.get('backward_time', 0.0):.4f}s, Optim: {timings.get('optim_time', 0.0):.4f}s)")
            if device.type == 'cuda':
                logger.info(f"Batch {batch_idx} GPU Memory: Allocated = {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB, Reserved = {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB")

    return total_loss / len(dataloader)

def single_word_step(model, images, input_ids, attention_mask, label_id, optimizer, criterion, scaler):
    step_start_time = time.time()
    timings = {}

    optimizer.zero_grad()
    forward_start_time = time.time()
    logits = model(images, input_ids, attention_mask)  # Pass text inputs to the model
    timings['forward_time'] = time.time() - forward_start_time
    loss_start_time = time.time()
    loss = criterion(logits, label_id)
    timings['loss_time'] = time.time() - loss_start_time
    backward_start_time = time.time()
    scaler.scale(loss).backward()
    timings['backward_time'] = time.time() - backward_start_time
    optim_start_time = time.time()
    scaler.step(optimizer)
    scaler.update()
    timings['optim_time'] = time.time() - optim_start_time
    timings['total_step_time'] = time.time() - step_start_time
    return loss, timings

def evaluate(run, model, test_dataset, epoch):
    model.eval()
    logger.info("Evaluation:")
    
    # Select random samples from the dataset
    num_samples = min(6, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)
    
    # Lists to store results
    original_images = []
    questions = []
    generated_answers = []
    ground_truth_answers = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for idx in sample_indices:
            sample = test_dataset[idx]
            image_tensor = sample["image_tensor"].unsqueeze(0).to(device)  # Add batch dimension
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            
            # Use the original image bytes for visualization
            original_images.append(sample["image_bytes"])
            
            # Get the question text from the input_ids
            question = model.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            questions.append(question)
            
            # Generate answer using the classifier and convert back to text
            generated_answer = model.predict_answer(image_tensor, input_ids, attention_mask, test_dataset.idx_to_answer)[0]
            generated_answers.append(generated_answer)
            
            # Get ground truth answer directly from the sample
            ground_truth = sample["answer_text"]
            ground_truth_answers.append(ground_truth)
    
    # Print results
    for i in range(num_samples):
        logger.info("---")
        logger.info(f"Image {i+1}:")
        logger.info(f"Question: {questions[i]}")
        logger.info(f"Generated: {generated_answers[i]}")
        logger.info(f"Ground truth: {ground_truth_answers[i]}")
        logger.info("---")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log results to wandb
    if wandb.run is not None:
        # Create a wandb table for the results
        caption_table = wandb.Table(columns=["Image", "Question", "Generated Answer", "Ground Truth"])
        for img, ques, gen, gt in zip(original_images, questions, generated_answers, ground_truth_answers):
            pil_img = Image.open(io.BytesIO(img)) if isinstance(img, bytes) else img
            caption_table.add_data(wandb.Image(pil_img), ques, gen, gt)
        
        # Log the table
        wandb.log({"caption_examples": caption_table})

def validate(model, val_loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    use_amp = (device.type == 'cuda') # Determine if AMP should be used for validation
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            image_tensor = batch["image_tensor"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_id = batch["label_id"].to(device)
            
            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(image_tensor, input_ids, attention_mask)  # Pass all inputs
                loss = criterion(outputs, label_id)
            
            # Calculate accuracy
            pred = outputs.argmax(dim=1)
            correct += (pred == label_id).sum().item()
            total += label_id.size(0)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    logger.info(f"Validation loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    wandb.log({"validation_accuracy": accuracy})
    
    return avg_loss

if __name__ == "__main__":
   main()