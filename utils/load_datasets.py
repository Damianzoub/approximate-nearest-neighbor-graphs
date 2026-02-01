from utils.read_files import read_fvecs, read_ivecs
from pathlib import Path

DATASET_ROOT = Path("/Users/Damian/approximate-nearest-neighbor-graphs/Datasets")

def load_siftsmall(base_dir):
    xb = read_fvecs(f"{base_dir}/siftsmall_base.fvecs")
    xq = read_fvecs(f"{base_dir}/siftsmall_query.fvecs")
    xt = read_ivecs(f"{base_dir}/siftsmall_groundtruth.ivecs")
    return xb,xq,xt


def list_datasets(root:Path = DATASET_ROOT)-> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Datasets root not found: {root}")
    
    #dataset -> directory containing at least one file:
    ds = []
    for p in sorted(root.iterdir):
        if p.is_dir():
            if any(x.is_file() for x in p.rglob("*")):
                ds.append(p)
    return ds

def pick_dataset_interactive(root:Path = DATASET_ROOT)->Path:
    datasets = list_datasets(root)
    if not datasets:
        raise RuntimeError(f"No datasets found in: {root}")

    print("\nAvailable Datasets:\n")
    for i ,d in enumerate(datasets,start=1):
        nfiles = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f" [{i}] {d.name} ({nfiles} files)")
    
    while True:
        raw  = input("\n Choose dataset by number (or 'q' to quit): ")
        if raw in {"q","quit","exit"}:
            raise SystemExit("Bye.")

        if raw.isdigit():
            idx = int(raw)
            if 1<= idx <= len(datasets):
                chosen = datasets[idx-1]
                print(f"\nSelected: {chosen.name}")
                return chosen
            
        print("Invalid choice. Try again.")
    pass 

def show_dataset_contents(ds_path:Path, max_lines:int=40)->None:
    files = sorted([p for p in ds_path.rglob("*") if p.is_file()])
    print("\nFiles found: \n")
    for p in files[:max_lines]:
        rel = p.relative_to(ds_path)
        print(f" - {rel}")
    if len(files) > max_lines:
        print(f" ... ({len(files)-max_lines} more)")
    pass

def auto_detect_ann_files(ds_path):
    files = [p for p in ds_path.rglob("*") if p.is_file()]
    lower = [(p,p.name.lower()) for p in files ]

    def pick_first(pred):
        for p, name in lower:
            if pred(name):
                return p 
        return None 
    
    base = pick_first(lambda n: ("base" in n and (n.endswith(".fvecs") or n.endswith(".npy"))))
    query = pick_first(lambda n: (("query" in n or "queries" in n) and (n.endswith(".fvecs") or n.endswith(".npy"))))
    gt = pick_first(lambda n: (("groundtruth" in n or "ground_truth" in n or "gt" in n) and n.endswith(".ivecs")))

    if gt is None:
        gt = pick_first(lambda n: n.endswith(".ivecs"))
    return {"base":base, "query":query, "groundtruth":gt}
    


def get_dataset()->dict:
    ds_path = pick_dataset_interactive(DATASET_ROOT)
    show_dataset_contents(ds_path)
    detected = auto_detect_ann_files(ds_path)

    print("\nAuto-detected files")
    for k,v in detected.items():
        if v is None:
            print(f" {k:>12}: (not found)")
        else:
            print(f"{k:>12}: {v.relative_to(ds_path)}")
    
    return {"dataset_dir": ds_path, **detected}

