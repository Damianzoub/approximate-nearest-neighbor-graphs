import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset" / "msmarco_text"
JSONL_PATH = DATASET_DIR / "passages.jsonl"
TXT_PATH = DATASET_DIR / "passages.txt"

LIMIT = 50000

def create_dataset_dir():
    print("Creating Dataset directory...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset():
    if JSONL_PATH.exists() and TXT_PATH.exists():
        print("MS MARCO files already exist")
        return

    print("Downloading MS MARCO from Hugging Face...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split=f"train[:{LIMIT}]")

    with open(JSONL_PATH, "w", encoding="utf-8") as f_jsonl, open(TXT_PATH, "w", encoding="utf-8") as f_txt:
        for row in ds:
            passages = row.get("passages", {})
            passage_texts = passages.get("passage_text", []) if isinstance(passages, dict) else []

            for passage in passage_texts:
                f_jsonl.write(json.dumps({"passage": passage}, ensure_ascii=False) + "\n")
                f_txt.write(passage.replace("\n", " ") + "\n")

    print(f"Saved to: {JSONL_PATH}")
    print(f"Saved to: {TXT_PATH}")

def main():
    print("=== MS MARCO Text Dataset Setup ===")
    create_dataset_dir()
    download_dataset()

if __name__ == "__main__":
    main()