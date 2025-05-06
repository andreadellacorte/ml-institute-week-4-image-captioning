from datetime import datetime

from src.model import UnifiedAutoregressiveDecoder

from src.dataset import ImageCaptioningDataset

from src.config import PROCESSED_DATA_DIR, FIGURES_DIR

import pickle
import random
import torch

from tqdm import tqdm

SIZES = {
    1: "1",
    50: "50",
    100: "100",
    500: "500",
    1000: "1k",
    5000: "5k",
    10000: "10k",
}

def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(3047)

    with open(PROCESSED_DATA_DIR / "flickr30k/100_images.pkl", "rb") as f:
        images = pickle.load(f)

    # pprint.pprint(images[0])

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

    print("Model loaded")

    dataset = ImageCaptioningDataset(train_images, captions, model, max_len=22)

    print("Dataset loaded")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Dataloader loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 5

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")
        evaluate(model, test_images, captions)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    device = next(model.parameters()).device
    
    for batch in progress_bar:
        images = batch["image_bytes"].to(device)
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        
        seq_length = input_ids.shape[2]

        # Print shapes for debugging    
        print(f"images shape: {images.shape}")
        print(f"input_ids shape: {input_ids.shape}")
        print(f"label_ids shape: {label_ids.shape}")
        print(f"seq_length: {seq_length}")
        
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
        
        # Track total loss
        total_loss += avg_batch_loss
        
        # Update progress bar with non-zero loss
        progress_bar.set_postfix(loss=f"{avg_batch_loss:.4f}")
    
    # Return average loss over all batches
    return total_loss / len(dataloader)

def evaluate(model, test_images, test_captions):
    model.eval()
    print("Evaluation:")
    
    # Create the test dataset
    test_dataset = ImageCaptioningDataset(test_images, test_captions, model)
    
    # Select 5 random samples from the dataset
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
    print("\nGenerated captions:")
    for i in range(num_samples):
        print(f"Image {i+1}:")
        print(f"Generated: {generated_captions[i]}")
        print(f"Ground truth: {ground_truth_captions[i]}")
        print("---")
    
    # Import the plotting function
    from src.plots import plot_images_with_captions
    
    # Create a list of captions with both generated and ground truth
    combined_captions = [f"Generated: {gen}\nGround truth: {gt}" 
                         for gen, gt in zip(generated_captions, ground_truth_captions)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot images with captions
    plot_images_with_captions(
        images=original_images,
        captions=combined_captions,
        title="Image Captioning Results",
        save_path=FIGURES_DIR / f"{timestamp}_caption_results.png"
    )

def validate(model, val_images, val_captions):
    # Implement your validation loop here
    pass

if __name__ == "__main__":
    main()