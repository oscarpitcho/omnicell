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
