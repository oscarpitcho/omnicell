import numpy as np
import torch
def get_proportional_weighted_dist(X):
    weighted_dist = (X / X.sum(axis=0))
    weighted_dist[:, X.sum(axis=0) == 0] = 0
    weighted_dist = weighted_dist.astype(np.float64)
    weighted_dist /= weighted_dist.sum(axis=0)
    return weighted_dist


def sample_multinomial_batch(probs, counts, max_count=None, max_attempts=50):
    """
    GPU-optimized multinomial sampling with unrolled attempts and vectorized validation
    Args:
        probs: (B, C) probability matrix
        counts: (C,) total counts per category
        max_count: (B, C) maximum allowed counts
        max_attempts: number of precomputed attempts
    Returns:
        (B, C) tensor of sampled counts meeting constraints

    """

    device = "cuda"

    probs = torch.from_numpy(probs).to(device)
    counts = torch.from_numpy(counts).to(device)
    B, C = probs.shape
    results = torch.zeros((max_attempts, B, C), dtype=torch.long, device=device)
    valid_masks = torch.zeros((max_attempts, B, C), dtype=torch.bool, device=device)

    for c in range(C):
        n = counts[c].item()
        if n == 0:
            continue

        # Initial sampling
        p = probs[:, c].clone()
        if p.sum() < 1e-12:
            continue

        # Generate initial samples for all attempts
        samples = torch.multinomial(p, n * max_attempts, replacement=True)
        samples = samples.view(max_attempts, n)
        
        # Create attempt tensor (max_attempts, B)
        attempt_counts = torch.zeros(max_attempts, B, dtype=torch.long, device=device)
        for a in range(max_attempts):
            attempt_counts[a] = torch.bincount(samples[a], minlength=B)
        
        if max_count is not None:
            mc = max_count[:, c]

            # Vectorized constraint application
            excess = (attempt_counts - mc[None,:]).clamp_min(0)
            redist = excess * (mc[None,:] > 0).long()
            
            # Vectorized redistribution (max_attempts, B)
            for a in range(1, max_attempts):
                prev_excess = redist[a-1]
                valid = prev_excess == 0
                
                # Carry forward valid solutions
                attempt_counts[a] = torch.where(valid, 
                                              attempt_counts[a-1],
                                              attempt_counts[a])
                
                # Redistribute excess using matrix operations
                adj_p = p.repeat(max_attempts, 1)
                adj_p[attempt_counts[a-1] > mc[None,:]] = 0
                adj_p = adj_p / adj_p.sum(dim=1, keepdim=True).clamp_min(1e-12)
                
                # Vectorized multinomial redistribution
                redist_samples = torch.multinomial(
                    adj_p[a-1], prev_excess.sum().item(), replacement=True
                )
                redist_counts = torch.bincount(redist_samples, minlength=B)
                
                attempt_counts[a] = torch.minimum(attempt_counts[a], mc) + redist_counts

            # Create validity mask across attempts
            valid_masks[:, :, c] = (attempt_counts <= mc[None,:])

        # Store all attempts
        results[:, :, c] = attempt_counts

    # Find first valid attempt for each (B, C)
    valid_idx = valid_masks.long().argmax(dim=0)
    valid_idx = torch.clamp_max(valid_idx, max_attempts-1)
    
    # Gather results using vectorized indexing
    final = torch.gather(
        results.permute(1, 2, 0),  # (B, C, A)
        2, 
        valid_idx.unsqueeze(-1)    # (B, C, 1)
    ).squeeze(-1)

    return final

def sample_pert(ctrl, weighted_dist, mean_shift, max_rejections=100):
    count_shift = np.round(mean_shift * ctrl.shape[0])
    max_count = 1. * ctrl
    max_count[:, count_shift > 0] = np.inf
    samples = sample_multinomial_batch(
        weighted_dist, np.abs(count_shift), max_count=max_count, max_attempts=max_rejections
    )
    sampled_pert = ctrl + np.sign(count_shift) * samples
    sampled_pert = np.clip(sampled_pert, 0, None)
    return sampled_pert

def hard_sample_pert(ctrl, weighted_dist, mean_shift, round_pert=False):
    # this can probably be better too. really we should like solve a linear program to find the closest integer solution
    count_shift = np.round(mean_shift * ctrl.shape[0])
    sampled_pert = ctrl + np.round(count_shift * weighted_dist) if round_pert else ctrl + (count_shift * weighted_dist)
    sampled_pert = np.clip(sampled_pert, 0, None)
    return sampled_pert