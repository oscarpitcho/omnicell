There are three things we need to run eval, these should be in a directory for each run, so for each model/dataset/architecture/replicate we need a seperate directory which should include:
  1. A json config file which including architecture details/training dataset, and to lists of heldout cells/perts. These lists are paired, so holdout_cells[0], holdout_perts[0] indicates one cell of the matrix which was held out during training. For example:
```
{
  "model_name": "e2e",
  "batch_size": 32,
  "max_epochs": 100,
  "model_kwargs": {
    "latent_space": 64,
  },
  "dataset": "Satija_TGFB_HVG",
  "control_pert": "NT",
  "holdout_cells": [
    "A549","A549","A549","A549","A549",
  ],
  "holdout_perts": [
    "RUNX2","RPS6KB1","PPP2CA","RUNX1","EP300"
  ],
}
```
  2. A set of predictions saved as pred_{pert_type}_{cell_type}.npz for each held out pert_type x cell type, containing at least pred_pert, true_pert, and control. For viz/other evals it would be nice to also include the embeddings, but no current evals use these. If there is no embedding I've just been writing the raw expression twice. The loop for generating this looks like this:
```
for cell_type, pert_type in zip(holdout_cells, holdout_perts):
    ...
    print(f"Saving {pert_type} predictions for {cell_type}")
    np.savez(
        f"{save_path}/pred_{pert_type}_{cell_type}.npz", 
        pred_pert=..., 
        true_pert=..., 
        control=...,
        true_pert_embedding=..., 
        control_embedding=...
    )
```
  3. Some of our evals involve running UMAP. If we want to do that we should additionally save npz files containing the training embeddings, control and perturbations with the perturbation ids:
```
np.savez(
    f"{save_path}/train_embeddings.npz", 
    control_embedding=...,
    pert_embedding=...,
    pert_type=...
)
```
