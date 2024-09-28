import torch
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,OTPlanSampler
)

from torch.utils.data import Sampler
from typing import Iterator, List

import numpy as np

class StratifiedBatchSampler(Sampler[List[int]]):
    def __init__(
        self, ns, batch_size: int
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        
        self.num_strata = np.prod(ns.shape)
        self.ns = ns
        self.probs = ns.flatten() / np.sum(ns)
        print("Strata probs", np.sort(self.probs))
        self.batch_size = batch_size
        self.batch_sizes = np.minimum(ns, batch_size)
    
    def get_random_stratum(self):
        linear_idx = np.random.choice(self.num_strata, p=self.probs)
        stratum = np.unravel_index(linear_idx, self.ns.shape)
        return stratum

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        while True:
            stratum = self.get_random_stratum()
            try:
                batch_stratum = np.repeat(np.array(stratum)[None, :], self.batch_sizes[stratum], axis=0)
                batch = np.random.choice(self.ns[stratum], self.batch_sizes[stratum], replace=False)
                yield zip(batch_stratum, batch)
            except StopIteration:
                break

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return np.sum(self.ns) // self.batch_size

class SCFMDataset(torch.utils.data.Dataset):
    def __init__(
        self, source, target, pert_ids, pert_mat, source_strata, target_strata, size=int(1e4)
    ):
        source, target = np.array(source), np.array(target)
        pert_ids, pert_mat = np.array(pert_ids), np.array(pert_mat)
        
        assert target.shape[0] == pert_ids.shape[0]
        assert source.shape[0] == source_strata.shape[0]
        assert target.shape[0] == target_strata.shape[0]
        
        self.size = size
        self.source_strata = source_strata
        self.target_strata = target_strata
        self.strata = np.unique(source_strata)
        self.num_strata = len(self.strata)
        
        self.pert_ids = np.unique(pert_ids)
        
        self.source = [source[source_strata == stratum] for stratum in self.strata]
        self.target = [
            [
                target[target_strata == stratum][pert_ids[target_strata == stratum] == pert_id] 
                for pert_id in self.pert_ids
            ] for stratum in self.strata
        ]
        self.pert_ids = [
            [
                pert_ids[target_strata == stratum][pert_ids[target_strata == stratum] == pert_id] 
                for pert_id in self.pert_ids
            ] for stratum in self.strata
        ]
        self.pert_mat = pert_mat
        

    def __len__(self):
        return self.size

    def __getitem__(self, strata_idx):
        stratum, idx = strata_idx
        sidx = np.random.choice(self.source[stratum[0]].shape[0])
        return (
            self.source[stratum[0]][sidx],
            self.target[stratum[0]][stratum[1]][idx],
            self.pert_mat[self.pert_ids[stratum[0]][stratum[1]][idx]],
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


def ot_collate(
    batch,
    return_noise=False,
    ot_sampler = OTPlanSampler(method="exact")
):

    batch = list(zip(*batch))
    batch = [torch.tensor(np.array(x)) for x in batch]
    control, target = batch[:2]
    
    x0, x1 = ot_sampler.sample_plan(control, target)

    return x0, x1, *batch[2:]
