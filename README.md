# ml-institute-week-4-image-captioning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Objective
This project aims to build a multimodal Transformer-based model that generates natural language captions for images. The goal is to explore and implement state-of-the-art techniques for image captioning using deep learning, focusing on combining visual and textual modalities.

## Dataset
- **Flickr30k**: The dataset used is [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k), which contains 31,000 images, each annotated with five captions. Data is loaded and processed from HuggingFace Datasets (`nlphuji/flickr30k`).

## Model Architecture
- **Vision Encoder**: Frozen CLIP Vision Transformer (ViT) from `openai/clip-vit-base-patch32` or `patch16`.
- **Text Encoder**: Frozen CLIP text encoder.
- **Transformer Decoder**: A custom Transformer decoder (UnifiedAutoregressiveDecoder) with configurable depth, width, and attention heads. The decoder receives image embeddings and autoregressively generates captions.
- **Training**: Only the decoder is trained; CLIP encoders are kept frozen. Training uses cross-entropy loss and supports mixed-precision (AMP) and early stopping.

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── requirements-cpu.txt      <- Requirements for CPU training/inference
├── requirements-gpu.txt      <- Requirements for GPU training/inference
├── pyproject.toml
├── setup.cfg
├── setup.py
├── setup.sh
├── output.log
├── script.pid
│
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   │   └── flickr30k
│   └── raw
│
├── docs
│
├── models
│
├── notebooks
│   └── flickr30k_eda.ipynb
│
├── references
│   └── *.pdf
│
├── reports
│   └── figures
│
├── src
│   ├── __init__.py
│   ├── 01_get_data.py        <- Download and serialize Flickr30k
│   ├── 02_process_data.py    <- Process and clean data
│   ├── 03_prove_overfit.py   <- Overfit test on a single image
│   ├── 04_train.py           <- Main training script
│   ├── app_flickr.py         <- Streamlit demo app for Flickr30k
│   ├── app_upload_only.py    <- Streamlit app for user-uploaded images
│   ├── config.py
│   ├── dataset.py            <- PyTorch Dataset for image-caption pairs
│   ├── model.py              <- Model architecture (CLIP + Transformer decoder)
│   ├── plots.py
│   ├── test.py
│   └── modeling/
│       ├── __init__.py
│       ├── predict.py        <- Inference utilities
│       └── train.py          <- (Optional) Training utilities
│
├── tests
│   └── test_data.py
│
├── wandb/                    <- Weights & Biases experiment logs
│
```

