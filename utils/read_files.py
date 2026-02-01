import numpy as np
import os 
def read_ivecs(fname):
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"ivecs file not found: {fname}")
    a = np.fromfile(fname, dtype='int32')
    if a.size == 0:
        raise ValueError(f"ivecs file is empty: {fname}")
    d = a[0]
    
    return a.reshape(-1, d + 1)[:, 1:].copy()


def read_fvecs(fname):
    return read_ivecs(fname).view('float32')