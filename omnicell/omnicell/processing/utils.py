import scanpy as sc

#Importing union type
from typing import List, Tuple
from scipy import sparse

from scipy.sparse import issparse
import numpy as np

def to_dense(X):
    if issparse(X):
        return X.toarray()
    elif isinstance(X, sparse.coo_matrix):
        return X.toarray()
    else:
        return X
    

def to_coo(X):
    if not isinstance(X, sparse.coo_matrix):
        return sparse.coo_matrix(X)
    else:
        return X
