from pathlib import Path
import numpy as np

from torchvision import datasets
from torchvision.transforms import ToTensor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "fashion_mnist"

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    print("Downloading Fashion-MNIST...")

    train = datasets.FashionMNIST(
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test = datasets.FashionMNIST(
        root=DATASET_DIR,
        train=False,
        download=True,
        transform=ToTensor()
    )

    # Convert to numpy
    xb = train.data.numpy().reshape(-1, 28*28).astype(np.float32)
    xq = test.data.numpy().reshape(-1, 28*28).astype(np.float32)

    # Normalize (important!)
    xb /= 255.0
    xq /= 255.0

    np.save(DATASET_DIR / "base.npy", xb)
    np.save(DATASET_DIR / "query.npy", xq)

    print(f"Saved base: {xb.shape}")
    print(f"Saved query: {xq.shape}")

def main():
    print("=== Fashion-MNIST Setup ===")
    create_dataset_dir()
    download_dataset()

if __name__ == "__main__":
    main()