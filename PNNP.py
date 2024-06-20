import numpy as np
from scipy.spatial.distance import cdist

def PNNP(perturbed_GT, perturbed_predicted):
    # Concatenate the arrays along the first axis
    concatenated = np.vstack((perturbed_GT, perturbed_predicted))
    
    # Compute the pairwise distances
    distances = cdist(concatenated, concatenated)
    
    # Set self-distances to a very high value to avoid self-matching
    np.fill_diagonal(distances, np.inf)
    
    # Find the index of the nearest neighbor for each observation
    nearest_neighbors = np.argmin(distances, axis=1)
    
    # Total number of observations in perturbed_GT and perturbed_predicted
    num_GT = perturbed_GT.shape[0]
    num_predicted = perturbed_predicted.shape[0]
    
    # Identify the group of each observation
    labels = np.array(['GT'] * num_GT + ['Predicted'] * num_predicted)
    
    # Count how many perturbed_predicted have nearest neighbors also in perturbed_predicted
    predicted_indices = range(num_GT, num_GT + num_predicted)
    neighbor_counts = sum(labels[nearest_neighbors[i]] == 'Predicted' for i in predicted_indices)
    
    # Calculate percentage
    percentage_mixed = 100 * neighbor_counts / num_predicted
    
    return percentage_mixed