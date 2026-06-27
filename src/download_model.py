"""
download_model.py — Download and save model weights locally.
Run this ONCE before the competition. Model is then used offline.

Usage:
  python src/download_model.py
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SAVE_PATH = "models/minilm"


def main():
    save_path = Path(SAVE_PATH)
    if save_path.exists():
        print(f"Model already exists at {save_path}. Skipping download.")
        # Quick test
        model = SentenceTransformer(str(save_path))
        test = model.encode(["test sentence"], normalize_embeddings=True)
        print(f"Model loaded OK. Embedding dim: {test.shape[1]}")
        return

    print(f"Downloading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    # Test
    test = model.encode(["test sentence"], normalize_embeddings=True)
    print(f"Model OK. Embedding dim: {test.shape[1]}")


if __name__ == "__main__":
    main()
