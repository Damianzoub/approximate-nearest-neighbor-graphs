from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "siftsmall_base.fvecs": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall_base.fvecs",
    "siftsmall_query.fvecs": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall_query.fvecs",
    "siftsmall_groundtruth.ivecs": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall_groundtruth.ivecs",
    "siftsmall_learn.fvecs": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall_learn.fvecs",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "siftsmall"

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_files():
    for filename, url in FILES.items():
        out_path = DATASET_DIR / filename
        if out_path.exists():
            print(f"[skip] {filename} already exists")
            continue
        print(f"Downloading {filename}...")
        urlretrieve(url, out_path)
        print(f"Saved to: {out_path}")

def main():
    print("=== SIFTSMALL Dataset Setup ===")
    create_dataset_dir()
    download_files()

if __name__ == "__main__":
    main()