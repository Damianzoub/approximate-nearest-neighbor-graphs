import os 
import tarfile
from pathlib import Path 
from urllib.request import urlretrieve

URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
ARCHIVE_PATH = DATASET_DIR / "sift.tar.gz"

def create_dataset_dir():
    print("Created Dataset Directory...")
    DATASET_DIR.mkdir(parents=True,exist_ok=True)

def download_file(): #download from URL
    if ARCHIVE_PATH.exists():
        print("Archive already exists")
        return 
    print("Downloading...")
    urlretrieve(URL,ARCHIVE_PATH)
    print(f"Download to: {ARCHIVE_PATH}")

def extract_archive(): #extract zip file
    print("Extracting archive...")
    with tarfile.open(ARCHIVE_PATH,'r:gz') as tar:
        tar.extractall(path=DATASET_DIR)
    print(f"Extracted into: {DATASET_DIR}")

def cleanup():
    # if later want to remove archive from directory
    print("Removing archive")
    ARCHIVE_PATH.unlink(missing_ok=True)


def main():
    print("=== SIFT Dataset Setup ===")
    create_dataset_dir()
    download_file()
    extract_archive()
    cleanup()

if __name__ == "__main__":
    main()