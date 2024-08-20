import torch
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,OTPlanSampler
)

from torch.utils.data import BatchSampler, SequentialSampler, Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

import numpy as np

class StratifiedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(
        self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, probs, num_strata
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.probs = probs
        self.num_strata = num_strata

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                stratum = np.repeat(np.random.choice(self.num_strata, p=self.probs), self.batch_size)
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield zip(stratum, batch)
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            stratum = np.repeat(np.random.choice(self.num_strata, p=self.probs), self.batch_size)
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield zip(stratum, batch)
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
                    stratum = np.repeat(np.random.choice(self.num_strata, p=self.probs), self.batch_size)
                if idx_in_batch > 0:
                    yield zip(stratum, batch[:idx_in_batch])

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


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
        
        self.source = [source[source_strata == stratum] for stratum in self.strata]
        self.target = [target[target_strata == stratum] for stratum in self.strata]
        self.pert_ids = [pert_ids[target_strata == stratum] for stratum in self.strata] 
        self.pert_mat = pert_mat
        if probs is None:
            probs = np.array([
                (self.source[stratum].shape[0] + self.target[stratum].shape[0]) \
                * (self.source[stratum].shape[0] != 0) * (self.target[stratum].shape[0] != 0)
                for stratum in range(self.num_strata)
            ]).astype(float)
            probs /= probs.sum()
            print(probs)
        self.probs = probs
        

    def __len__(self):
        return self.size

    def __getitem__(self, strata_idx):
        stratum, idx = strata_idx
        sidx = idx % len(self.source[stratum])
        tidx = idx % len(self.target[stratum])
        return (
            self.source[stratum][sidx],
            self.target[stratum][tidx],
            self.pert_mat[self.pert_ids[stratum][tidx]],
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
