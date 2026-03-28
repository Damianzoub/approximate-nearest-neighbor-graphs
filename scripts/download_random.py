from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "random"

N = 100_000
D = 128
Q = 10_000
SEED = 42

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def generate_dataset():
    base_path = DATASET_DIR / "base.npy"
    query_path = DATASET_DIR / "query.npy"

    if base_path.exists() and query_path.exists():
        print("Random dataset already exists")
        return

    print("Generating random dataset...")
    rng = np.random.default_rng(SEED)
    xb = rng.random((N, D), dtype=np.float32)
    xq = rng.random((Q, D), dtype=np.float32)

    np.save(base_path, xb)
    np.save(query_path, xq)

    print(f"Saved to: {base_path}")
    print(f"Saved to: {query_path}")

def main():
    print("=== Random Dataset Setup ===")
    create_dataset_dir()
    generate_dataset()

if __name__ == "__main__":
    main()