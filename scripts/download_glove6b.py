import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://nlp.stanford.edu/data/glove.6B.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "glove6b"
ARCHIVE_PATH = DATASET_DIR / "glove.6B.zip"

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_file():
    if ARCHIVE_PATH.exists():
        print("Archive already exists")
        return
    print("Downloading GloVe 6B...")
    urlretrieve(URL, ARCHIVE_PATH)
    print(f"Downloaded to: {ARCHIVE_PATH}")

def extract_archive():
    print("Extracting archive...")
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print(f"Extracted into: {DATASET_DIR}")

def cleanup():
    print("Removing archive...")
    ARCHIVE_PATH.unlink(missing_ok=True)

def main():
    print("=== GloVe6B Dataset Setup ===")
    create_dataset_dir()
    download_file()
    extract_archive()
    cleanup()

if __name__ == "__main__":
    main()