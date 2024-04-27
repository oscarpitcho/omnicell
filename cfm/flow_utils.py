import math
import torch
import numpy as np
from torchdyn.core import NeuralODE
from datamodules import torch_wrapper

def compute_conditional_flow(model, control, pert_ids, pert_mat, batch_size=1_000, num_steps=400, n_batches=1e8):
    device = "cuda:0"
    node = NeuralODE(
        torch_wrapper(model).to(device), solver="dopri5", sensitivity="adjoint"
    )
    n_batches = min(math.ceil(control.shape[0] / batch_size), n_batches)
    traj = np.zeros((num_steps, min(n_batches * batch_size, control.shape[0]), control.shape[1]))
    with torch.no_grad():
        for i in range(n_batches):
            control_batch = control[batch_size*i:batch_size*(i+1)]
            pert_batch = pert_mat[pert_ids][:control_batch.shape[0]]
            model.cond = torch.from_numpy(pert_batch).to(device)
            inp = torch.from_numpy(control_batch).float()
            inp = inp.to_device()
            outp = node.trajectory(
                inp.to(device),
                t_span=torch.linspace(0, 1, num_steps)
            )
            outp = outp.cpu()
            traj[:, batch_size*i:batch_size*(i+1), :] = outp
            
    return traj