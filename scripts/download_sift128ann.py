from pathlib import Path
from _download_utils import download

URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "sift128ann"
FILE_PATH = DATASET_DIR / "sift-128-euclidean.hdf5"


def create_dataset_dir():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def download_file():
    if FILE_PATH.exists():
        print("Dataset already exists")
        return
    print("Downloading SIFT128 ANN...")
    download(URL, FILE_PATH)
    print(f"Downloaded to: {FILE_PATH}")


def main():
    print("=== SIFT128 ANN Dataset Setup ===")
    create_dataset_dir()
    download_file()


if __name__ == "__main__":
    main()
