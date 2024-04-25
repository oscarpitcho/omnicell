import torch
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
)

import numpy as np


class FMDataset(torch.utils.data.Dataset):
    def __init__(self, source, target, size=int(1e4)):
        self.source = source
        self.target = target
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


class CFMDataset(torch.utils.data.Dataset):
    def __init__(self, source, target, pert_ids, pert_mat, size=int(1e4)):
        self.source = np.array(source)
        self.target = np.array(target)
        self.pert_ids = np.array(pert_ids)
        self.pert_mat = np.array(pert_mat)
        self.size = size

        assert len(self.target) == len(self.pert_ids)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (
            self.source[idx % len(self.source)],
            self.target[idx % len(self.target)],
            self.pert_mat[self.pert_ids[idx % len(self.pert_ids)]],
        )
    
class SCFMDataset(torch.utils.data.Dataset):
    def __init__(
        self, source, target, pert_ids, pert_mat, source_strata, target_strata, 
        probs=None, size=int(1e4), batch_size=32
    ):
        assert len(target) == len(pert_ids)
        assert len(source) == len(source_strata)
        assert len(target) == len(target_strata)
        
        source, target = np.array(source), np.array(target)
        pert_ids, pert_mat = np.array(pert_ids), np.array(pert_mat)
        
        self.size = size
        self.source_strata = source_strata
        self.target_strata = target_strata
        self.strata = np.unique(source_strata)
        self.num_strata = len(self.strata)
        
        self.source = [source[source_strata == stratum] for stratum in self.strata]
        self.target = [target[target_strata == stratum] for stratum in self.strata]
        self.pert_ids = [pert_ids[target_strata == stratum] for stratum in self.strata] 
        self.pert_mat = pert_mat
        if probs is None:
            probs = np.array([
                self.source[stratum].shape[0] + self.target[stratum].shape[0] 
                for stratum in range(self.num_strata)
            ]).astype(float)
            probs /= probs.sum()
            print(probs)
        self.probs = probs
        self.stratum = 0
        self.batch_pos = 0
        self.batch_size = batch_size

        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.batch_pos % self.batch_size == 0:
            self.stratum = np.random.choice(self.num_strata, p=self.probs)
        self.batch_pos += 1
        sidx = idx % len(self.source[self.stratum])
        tidx = idx % len(self.target[self.stratum])
        return (
            self.source[self.stratum][sidx],
            self.target[self.stratum][tidx],
            self.pert_mat[self.pert_ids[self.stratum][tidx]],
        )



class torch_wrapper(torch.nn.Module):
    def __init__(self, model, perturbation=None):
        super().__init__()
        self.model = model
        self.perturbation = perturbation

    def forward(self, t, x, args=None):
        if self.perturbation is not None:
            return self.model(
                torch.cat(
                    [
                        x,
                        self.perturbation.repeat(x.shape[0], 1).to(x.device),
                        t.repeat(x.shape[0])[:, None],
                    ],
                    1,
                )
            )
        else:
            return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def fm_collate(
    batch,
    return_noise=False,
    FM=ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),
):
    noises = []

    control, pert = zip(*batch)
    control = torch.tensor(np.array(control))
    pert = torch.tensor(np.array(pert))

    if return_noise:
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(
            control,
            pert,
            return_noise=return_noise,
        )
        noises.append(eps)
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(
            control,
            pert,
            return_noise=return_noise,
        )
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut


def cfm_collate(
    batch,
    return_noise=False,
    FM=ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),
):
    noises = []

    control, target, perturb = zip(*batch)
    control = torch.tensor(np.array(control))
    target = torch.tensor(np.array(target))
    perturb = torch.tensor(np.array(perturb))

    if return_noise:
        t, xt, ut, eps = FM.sample_location_and_conditional_flow(
            control,
            target,
            return_noise=return_noise,
        )
        noises.append(eps)
    else:
        t, xt, ut = FM.sample_location_and_conditional_flow(
            control,
            target,
            return_noise=return_noise,
        )
    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut, perturb
