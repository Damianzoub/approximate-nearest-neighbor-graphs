from pathlib import Path
from urllib.request import urlretrieve

URL = "https://ann-benchmarks.com/gist-960-euclidean.hdf5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "gist960ann"
FILE_PATH = DATASET_DIR / "gist-960-euclidean.hdf5"

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_file():
    if FILE_PATH.exists():
        print("Dataset already exists")
        return
    print("Downloading GIST960 ANN...")
    urlretrieve(URL, FILE_PATH)
    print(f"Downloaded to: {FILE_PATH}")

def main():
    print("=== GIST960 ANN Dataset Setup ===")
    create_dataset_dir()
    download_file()

if __name__ == "__main__":
    main()