# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport round
from numpy.random import Generator, PCG64

# Define C types for better performance
ctypedef np.float32_t F32_t
ctypedef np.float64_t F64_t
ctypedef np.int64_t I64_t  # Changed to int64


def get_proportional_weighted_dist(X):
    # Convert input to float32
    X = np.asarray(X, dtype=np.float32)
    
    cdef np.ndarray[F32_t, ndim=2] X_arr = X
    cdef int n_rows = X.shape[0]
    cdef int n_cols = X.shape[1]
    cdef np.ndarray[F32_t, ndim=2] weighted_dist = np.zeros((n_rows, n_cols), dtype=np.float32)
    cdef float col_sum
    cdef int i, j
    
    for j in range(n_cols):
        col_sum = 0.0
        for i in range(n_rows):
            col_sum += X_arr[i, j]
            
        if col_sum > 0:
            for i in range(n_rows):
                weighted_dist[i, j] = X_arr[i, j] / col_sum
                
    return weighted_dist

def sample_multinomial_batch(np.ndarray[F32_t, ndim=2] probs,
                           np.ndarray[F32_t, ndim=1] counts,
                           np.ndarray[F32_t, ndim=2] max_count=None,
                           int max_rejections=100):
    cdef int n_rows = probs.shape[0]
    cdef int n_cols = probs.shape[1]
    cdef np.ndarray[F32_t, ndim=2] result = np.zeros((n_rows, n_cols), dtype=np.float32)
    cdef np.ndarray[F64_t, ndim=1] p_tmp = np.zeros(n_rows, dtype=np.float64)
    cdef np.ndarray[I64_t, ndim=1] sample = np.zeros(n_rows, dtype=np.int64)  # Changed to int64
    
    # Create random number generator
    rng = Generator(PCG64())
    
    cdef int i, j, k
    cdef F64_t p_sum
    cdef float count
    cdef bint has_max_count = max_count is not None
    
    for j in range(n_cols):
        count = counts[j]
        if count == 0:
            continue
            
        # Copy probabilities for this column and convert to float64 for sampling
        p_sum = 0.0
        for i in range(n_rows):
            p_tmp[i] = float(probs[i, j])
            p_sum += p_tmp[i]
            
        if p_sum == 0:
            continue
            
        # Normalize probabilities
        for i in range(n_rows):
            p_tmp[i] /= p_sum
            
        # Sample from multinomial using numpy's generator
        sample = rng.multinomial(n=int(count), pvals=p_tmp)
        
        # Handle max_count constraints if present
        if has_max_count:
            for k in range(max_rejections):
                over_max = False
                for i in range(n_rows):
                    if sample[i] > max_count[i, j]:
                        over_max = True
                        break
                        
                if not over_max:
                    break
                    
                # Resample if needed
                p_sum = 0.0
                for i in range(n_rows):
                    if sample[i] > max_count[i, j]:
                        sample[i] = int(max_count[i, j])
                    p_tmp[i] = float(probs[i, j]) if sample[i] < max_count[i, j] else 0
                    p_sum += p_tmp[i]
                    
                if p_sum > 0:
                    for i in range(n_rows):
                        p_tmp[i] /= p_sum
                    
                    remaining_count = int(count - np.sum(sample))
                    if remaining_count > 0:
                        sample += rng.multinomial(n=remaining_count, pvals=p_tmp)
        
        # Copy results back to float32
        for i in range(n_rows):
            result[i, j] = float(sample[i])
    
    return result

def sample_pert(np.ndarray[F32_t, ndim=2] ctrl,
                np.ndarray[F32_t, ndim=2] weighted_dist,
                np.ndarray[F32_t, ndim=1] mean_shift,
                int max_rejections=100):
    cdef int n_rows = ctrl.shape[0]
    cdef int n_cols = ctrl.shape[1]
    cdef np.ndarray[F32_t, ndim=1] count_shift = np.round(mean_shift * n_rows).astype(np.float32)
    cdef np.ndarray[F32_t, ndim=2] max_count = np.copy(ctrl)
    cdef np.ndarray[F32_t, ndim=2] samples
    cdef np.ndarray[F32_t, ndim=2] sampled_pert
    cdef int i, j
    cdef float sign
    
    # Set max_count to large value where count_shift > 0
    for j in range(n_cols):
        if count_shift[j] > 0:
            for i in range(n_rows):
                max_count[i, j] = 1e10
    
    samples = sample_multinomial_batch(
        weighted_dist, 
        np.abs(count_shift), 
        max_count=max_count, 
        max_rejections=max_rejections
    )
    
    sampled_pert = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    # Apply shifts with proper sign
    for j in range(n_cols):
        sign = 1.0 if count_shift[j] > 0 else -1.0
        for i in range(n_rows):
            sampled_pert[i, j] = max(0.0, ctrl[i, j] + sign * samples[i, j])
    
    return sampled_pert