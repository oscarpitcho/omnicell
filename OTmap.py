import math
from tqdm import tqdm
import torch
import numpy as np
import os
from typing import Optional, Union, Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import scanpy as sc
from torch.utils.data import RandomSampler
import json
import sys
import hashlib
from sc_etl_utils import *
from functools import partial
import ot as pot
from torch.utils.data import BatchSampler, SequentialSampler, Sampler
from torch import nn
import anndata as ad
import scipy
from scipy.sparse import issparse
import anndata
import pandas as pd  
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
import random
import scipy.stats as stats


        
def compute_ot_mapping(adata, cost_threshold=0.01):
    mappings = {}
    valid_perturbed_indices = []

    for cell_type in adata.obs['cell_type'].unique():
        for gene in adata.obs['gene'].unique():
            if gene == 'NT':
                continue
            
            # Filter data for the specific cell type and gene
            perturbed_cells = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['gene'] == gene)]
            control_cells = adata[(adata.obs['cell_type'] == cell_type) & (adata.obs['gene'] == 'NT')]
            
            # Extract total counts for both sets of cells
            perturbed_counts = perturbed_cells.obs['nCount_RNA'].values
            control_counts = control_cells.obs['nCount_RNA'].values
            
            # Compute the cost matrix based on the absolute difference in total counts
            cost_matrix = np.abs(perturbed_counts[:, None] - control_counts[None, :])
            
            # Solve the linear sum assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Discard mappings that differ by more than 1 percent in their counts (or whatever you set cost_threshold to)
            valid_mapping = cost_matrix[row_ind, col_ind] / perturbed_counts[row_ind] <= cost_threshold
            
            # Save the valid perturbed indices
            valid_perturbed_indices.extend(perturbed_cells.obs.index[row_ind[valid_mapping]].tolist())
            
            # Save the mappings
            mappings[(cell_type, gene)] = {
                "perturbed_indices": perturbed_cells.obs.index[row_ind[valid_mapping]].tolist(),
                "control_indices": control_cells.obs.index[col_ind[valid_mapping]].tolist()
            }
    
    return mappings, valid_perturbed_indices

class OTMappingDataset(Dataset):
    def __init__(self, adata, mappings, valpertind):
        self.adata = adata
        self.mappings = mappings
        self.valpertind = valpertind

    def __len__(self):
        return len(self.valpertind)

    def __getitem__(self, idx):
        
        curpert = self.adata[self.valpertind[idx]]

        gene = curpert.obs['gene'].item()
        cell_type = curpert.obs['cell_type'].item()

        
        pert_mapping = self.mappings[(cell_type, gene)]
        ridex = pert_mapping['perturbed_indices'].index(curpert.obs_names[0])
        perturbed_index = pert_mapping['perturbed_indices'][ridex]
        control_index = pert_mapping['control_indices'][ridex]
        

        # Get the perturbed cell and the corresponding NT cell
        perturbed_cell = self.adata[perturbed_index]
        nt_cell = self.adata[control_index]
        
        # Find the index of the perturbed gene in var['gene']
        gene_index = np.where(self.adata.var['gene'] == gene)[0][0]
        
        # Get the expression value for the perturbed cell at that gene
        expression_value = perturbed_cell.X[0, gene_index]

        # Convert everything to torch tensors and return
        # nt_cell_tensor = torch.tensor(nt_cell.X.toarray(), dtype=torch.float32).cuda()
        # perturbed_cell_tensor = torch.tensor(perturbed_cell.X.toarray(), dtype=torch.float32).cuda()
        # gene_index_tensor = torch.tensor(gene_index, dtype=torch.long).cuda()
        # expression_value_tensor = torch.tensor(expression_value, dtype=torch.float32).cuda()

        return nt_cell_tensor, perturbed_cell_tensor, gene_index_tensor.unsqueeze(-1), expression_value_tensor.unsqueeze(-1)




