import math
import torch
import numpy as np
from torchdyn.core import NeuralODE

def compute_conditional_flow(model, control, pert_ids, pert_mat, batch_size=1_000, num_steps=400):
    device = "cuda:0"
    node = NeuralODE(
        torch_wrapper(model).to(device), solver="dopri5", sensitivity="adjoint"
    )
    traj = np.zeros((num_steps, control.shape[0], control.shape[1]))
    with torch.no_grad():
        for i in range(math.ceil(control.shape[0] / batch_size)):
            control_batch = control[batch_size*i:batch_size*(i+1)]
            pert_batch = pert_mat[pert_ids][:control_batch.shape[0]]
            model.cond = torch.from_numpy(pert_batch).to(device)
            traj[:, batch_size*i:batch_size*(i+1), :] = node.trajectory(
                torch.from_numpy(control_batch).float().to(device),
                t_span=torch.linspace(0, 1, num_steps)
            ).cpu()
            
    return traj