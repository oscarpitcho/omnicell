import numpy as np
import scanpy as sc
import warnings
import torch

def sliced_wasserstein_distance(X1, X2, n_projections=100, p=2):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two batches using POT.

    Args:
        X1: Tensor of shape (N, d) - First batch of points.
        X2: Tensor of shape (M, d) - Second batch of points.
        n_projections: Number of random projections (default: 100).
        p: Power of distance metric (default: 2).

    Returns:
        SWD (scalar tensor).
    """
    device = X1.device
    d = X1.shape[1]  # Feature dimension

    # Generate random projection vectors
    projections = torch.randn((n_projections, d), device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)  # Normalize

    # Project both distributions onto 1D subspaces
    X1_proj = X1 @ projections.T  # Shape: (N, n_projections)
    X2_proj = X2 @ projections.T  # Shape: (M, n_projections)

    # Sort projections along each 1D slice
    X1_proj_sorted, _ = torch.sort(X1_proj, dim=0)
    X2_proj_sorted, _ = torch.sort(X2_proj, dim=0)

    # Compute 1D Wasserstein distance per projection (L_p norm)
    SW_dist = torch.mean(torch.abs(X1_proj_sorted - X2_proj_sorted) ** p) ** (1/p)

    return SW_dist

def sinkhorn(
    X,
    Y,
    reg = 10,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    p = 2,
    **kwargs
):
    """
    X: (n, d) tensor of source samples
    Y: (m, d) tensor of target samples
    reg: regularization parameter
    Returns: Sinkhorn loss between empirical distributions of X and Y
    """
    # Device and dtype setup
    device = X.device
    dtype = X.dtype
    
    # Create uniform distributions
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.ones(n, 1, device=device, dtype=dtype) / n
    b = torch.ones(m, 1, device=device, dtype=dtype) / m

    # Compute pairwise cost matrix (squared Euclidean)
    M = torch.cdist(X, Y, p=p)**p
    reg = 0.1 * torch.median(M)
    print(reg)
    #reg = 0.1 * torch.median(M)

    # Initialize dual vectors
    u = torch.ones(n, 1, device=device, dtype=dtype) / n
    v = torch.ones(m, 1, device=device, dtype=dtype) / m

    # Compute kernel matrix with numerical stability
    K = torch.exp(-M / (reg + 1e-16))  # (n, m)
    # Scaling vector precomputation
    Kp = (1 / a) * K  # (n, 1) * (n, m) = (n, m)
    torch.abs(X[0] - Y[0])
    # Sinkhorn iterations
    for ii in range(numItermax):
        uprev = u.clone()
        vprev = v.clone()

        # Update v then u
        Ktu = torch.mm(K.t(), u)  # (m, 1)
        v = b / (Ktu + 1e-16)  # (m, 1)
        u = 1.0 / (torch.mm(Kp, v) + 1e-16)  # (n, 1)

        # Check for numerical issues
        if (torch.any(Ktu.abs() < 1e-9) or 
            torch.any(torch.isnan(u)) or 
            torch.any(torch.isnan(v)) or 
            torch.any(torch.isinf(u)) or 
            torch.any(torch.isinf(v))):
            if warn:
                warnings.warn(f"Numerical errors at iteration {ii}")
            u = uprev
            v = vprev
            break

    # Compute transport plan and loss
    P = u * K * v.t()  # (n, m)
    loss = torch.sum(P * M)
    print(P)
    print("----------")

    return loss.squeeze()


def mmd(X: torch.Tensor, Y: torch.Tensor, gamma: float, p =2) -> torch.Tensor:
    """
    Biased MMD² estimator with RBF kernel (includes diagonal terms)
    Compatible with PyTorch gradients

    Args:
        X: (n, d) tensor - samples from distribution P
        Y: (m, d) tensor - samples from distribution Q
        gamma: RBF kernel bandwidth parameter (1/(2σ²))

    Returns:
        Scalar tensor containing MMD² (biased)
    """
    # Compute pairwise squared distances
    XX = torch.cdist(X, X, p=p).pow(2)  # (n, n)
    YY = torch.cdist(Y, Y, p=p).pow(2)  # (m, m)
    XY = torch.cdist(X, Y, p=p).pow(2)  # (n, m)

    # Compute RBF kernels
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)

    # Compute biased MMD² (includes diagonal terms)
    mmd_squared = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    
    return mmd_squared

def compute_gamma(X: torch.Tensor, Y: torch.Tensor, p = 2) -> float:
    """
    Compute gamma using the median heuristic.
    Args:
        X: (n, d) tensor - samples from distribution P
        Y: (m, d) tensor - samples from distribution Q
    Returns:
        gamma: RBF kernel bandwidth parameter
    """
    # Combine samples
    Z = torch.cat([X, Y], dim=0)
    
    # Compute pairwise squared distances
    D = torch.cdist(Z, Z, p=p).pow(2)  # (n+m, n+m)
    
    # Extract upper triangular (excluding diagonal)
    upper_tri = torch.triu(D, diagonal=1)
    
    # Get non-zero distances
    distances = upper_tri[upper_tri > 0].sqrt()  # Convert to L2 distances
    
    # Compute median
    sigma = torch.median(distances).item()
    
    # Avoid division by zero
    sigma = max(sigma, 1e-8)
    
    gamma = 1.0 / (2 * sigma ** 2)
    return gamma

def sinkhorn_divergence(
    X,
    Y,
    reg=0.0001,
    numItermax=1000,
    stopThr=1e-9,
    verbose=False,
    log=False,
    warn=True,
    p=2,
    **kwargs
):
    """
    Computes the Sinkhorn Divergence between empirical distributions of X and Y.
    
    X: (n, d) tensor of source samples
    Y: (m, d) tensor of target samples
    reg: regularization parameter (must be fixed across all terms)
    Returns: Sinkhorn divergence = S(X,Y) - 0.5*S(X,X) - 0.5*S(Y,Y)
    """
    # Compute all three terms using the original Sinkhorn function
    S_XY = sinkhorn(X, Y, reg, numItermax, stopThr, verbose, log, warn, p, **kwargs)
    S_XX = sinkhorn(X, X, reg, numItermax, stopThr, verbose, log, warn, p, **kwargs)
    S_YY = sinkhorn(Y, Y, reg, numItermax, stopThr, verbose, log, warn, p, **kwargs)
    
    divergence = S_XY - 0.5 * S_XX - 0.5 * S_YY
    return divergence

def sinkhorn_log(X, Y, reg = 100, numItermax=1000, stopThr=1e-9, p =2):
    """
    PyTorch Sinkhorn loss function with entropic regularization.
    
    Args:
        a: (dim_a,) tensor representing the source distribution.
        b: (dim_b,) tensor representing the target distribution.
        M: (dim_a, dim_b) tensor representing the cost matrix.
        reg: Regularization strength (lambda).
        numItermax: Maximum number of Sinkhorn iterations.
        stopThr: Stop threshold on the marginal difference.
        
    Returns:
        loss: The computed regularized Sinkhorn loss.
    """

    device = X.device
    dtype = X.dtype
    
    # Create uniform distributions
    n = X.shape[0]
    m = Y.shape[0]
    a = torch.ones(n, device=device, dtype=dtype) / n
    b = torch.ones(m, device=device, dtype=dtype) / m
    
    # Compute pairwise cost matrix (squared Euclidean)
    M = torch.cdist(X, Y, p=p)**p
    print(M)

    assert a.dim() == 1 and b.dim() == 1 and M.dim() == 2, "Input dimensions incorrect"
    assert M.size(0) == a.size(0) and M.size(1) == b.size(0), "Size mismatch in cost matrix"

    # Convert to log domain and normalize
    reg = 0.1 * torch.median(M)
    log_a = torch.log(a)
    log_b = torch.log(b)
    Mr = -M / reg
    print(Mr)

    # Initialize dual potentials
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for i in range(numItermax):
        # Update v and u in log space
        v = log_b - torch.logsumexp(Mr + u.unsqueeze(1), dim=0)
        u = log_a - torch.logsumexp(Mr + v.unsqueeze(0), dim=1)

        # Check convergence every 10 iterations
        if i % 10 == 0:
            with torch.no_grad():
                gamma = torch.exp(Mr + u.unsqueeze(1) + v.unsqueeze(0))
                marginal = gamma.sum(dim=0)
                err = torch.norm(marginal - b)
                if err < stopThr:
                    break
    
    # Compute final transport matrix and loss
    gamma = torch.exp(Mr + u.unsqueeze(1) + v.unsqueeze(0))
    loss = (gamma * M).sum() + reg * (gamma * (gamma.log())).sum()
    print(gamma)
    return loss