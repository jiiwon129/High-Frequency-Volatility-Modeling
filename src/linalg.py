import numpy as np

def project(a, floor=0):
    """Project a symmetric matrix onto the PSD cone."""
    v, q = np.linalg.eigh((a + a.T) / 2)
    return q * np.maximum(v, floor) @ q.T
