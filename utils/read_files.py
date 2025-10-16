import numpy as np

def read_fvecs(path):
    data = np.fromfile(path,dtype=np.int32)
    if not data:
        raise ValueError("File is empty or could not be read.")
    dim = data[0]
    assert data.shape%(dim+1) == 0, "File size is not consistent with fvecs format."
    n = data.shape[0]//(dim+1)
    data = data.reshape(n,dim+1)[:,1:].astype('float32')

    return data 

def read_ivecs(path):
    data = np.fromfile(path,dtype=np.int32)
    if not data:
        raise ValueError("File is empty or wrong file path")
    dim = data[0]
    assert data.shape%(dim+1) == 0, "File size is not consistent with ivecs format."
    n = data.shape[0]//(dim+1)
    data = data.reshape(n,dim+1)[:,1:].astype('int32')
    return data 