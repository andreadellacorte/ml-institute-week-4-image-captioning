# load one image from data/processed/flickr30k/1_images.pkl
from model import UnifiedAutoregressiveDecoder

from dataset import ImageCaptioningDataset

import pickle
import random
import torch

from src.config import PROCESSED_DATA_DIR
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

    dataset = ImageCaptioningDataset(train_images, captions, model)

    print("Dataset loaded")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    print("Dataloader loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = 5

    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, optimizer, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    evaluate(model, test_images, captions)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        images_bytes = batch["image_bytes"]
        input_ids = batch["input_ids"]
        label_ids = batch["label_ids"]

        optimizer.zero_grad()

        outputs = model(images_bytes, input_ids)
        loss = 0

        # Compare logits one at a time
        for i in range(outputs.size(1)):  # Iterate over sequence length
        progress_bar.set_postfix(loss=loss.item())

    return loss

def evaluate(model, test_images, test_captions):
    # Implement your evaluation loop here
    pass

def validate(model, val_images, val_captions):
    # Implement your validation loop here
    pass

if __name__ == "__main__":
    main()