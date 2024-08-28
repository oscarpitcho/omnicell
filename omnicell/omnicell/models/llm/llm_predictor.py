import scanpy as sc
import torch
import yaml
import numpy as np
import math 
from omnicell.models.llm import MAE

class LLMPredictor():


    def __init__(self, config, input_size: int, device: str):
        
        
        self.model_config = config['model']
        self.trainig_config = config['training']
        self.device = device

        base_learning_rate = 5e-5
        
        self.model = model = MAE(
            input_size, 
            self.model_config['emb_dim'], 
            self.model_config['decoder_layer'],
             self.model_config['encoder_layer'], 
             self.model_config['encoder_head'], 
             self.model_config['decoder_head'],
             self.model_config
             ff_dim=128,
        true_sparsity=False, expr_activation="sigmoid")

        self.optim = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
        self.lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-5), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=self.lr_func, verbose=True)
        # model = torch.load(f"...")
        # device = 'cpu'
        self.model = model.to(device)
        

    #Should take care of saving the model under some results/model/checkpoints in 
    #BTW I think hidden dirs fuck with with the cluster, so don't call it .checkpoint
    def train(self, dl, save_path):


        ### CODE FROM AE TRAINING LOOP ###

        step_count = 0
        self.optim.zero_grad()
        pert_task = 0
        for e in range(2):
            self.model.train()
            losses = {'control': [], 'pert': []}
            for (bcontrol, bpert, bpert_index) in (pbar := tqdm(iter(dl))):
                curr_batch_size = bcontrol.shape[0]
                for i in range(curr_batch_size // minibatch_size):
                    control = bcontrol[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert = bpert[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = bpert_index[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = pert_index.squeeze()
                    active_weights = 10 * (control > 0).float().to(device) + 1 if use_active_weights else 1
                    control = control.float().to(self.device)
                    step_count += 1

                    control_results = self.model(control, mask=use_mask_task)
                    control_recon = control_results[0]
                    
                    control_loss = torch.sum(active_weights * torch.abs(control_recon - control)) / minibatch_size
                    if self.config['use_sparsity_loss'] and len(control_results) == 3:
                        control_sparsity = control_results[1]
                        control_loss += torch.sum(active_weights * torch.abs(control_sparsity - (control > 0).float())) / minibatch_size

                    loss = control_loss
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()
                    losses['control'].append(control_loss.item())
                    if step_count % self.trainig_config['lr_step'] == 0:
                        self.lr_scheduler.step()

                    pbar.set_description(
                        f"tv: {np.array(losses['control'])[-self.training_config['lr_step']:].mean():.3f}"
                    )
            
            avg_loss = sum(losses['control']) / len(losses['control'])
            torch.save(self.model, f"{save_path}{e}")
            # writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')


        
        ## CODE FROM PERT RECONSTRUCTION LOOP ##

        optim.zero_grad()
        pert_task = 0
        for e in range(2):
            model.train()
            losses = {'control': [], 'pert': []}
            for (bcontrol, bpert, bpert_index) in (pbar := tqdm(iter(dl))):
                curr_batch_size = bcontrol.shape[0]
                for i in range(curr_batch_size // minibatch_size):
                    control = bcontrol[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert = bpert[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = bpert_index[(i * minibatch_size):((i + 1) * minibatch_size)]
                    pert_index = pert_index.squeeze()
                    active_weights = 10 * (control > 0).float().to(device) + 1 if use_active_weights else 1
                    pert_active_weights = 10 * (pert > 0).float().to(device) + 1 if use_active_weights else 1
                    control = control.float().to(device)
                    pert = pert.float().to(device)
                    step_count += 1

                    pert_expr = pert[torch.arange(pert.size(0)), pert_index, None]
                    control_results, pert_results = model(
                        control, pert_expr=pert_expr, pert_index=pert_index, mask=use_mask_task, recon_and_pert=True
                    )
                    
                    control_recon, pert_recon = control_results[0], pert_results[0]
                    control_loss = torch.sum(active_weights * torch.abs(control_recon - control)) / minibatch_size
                    pert_loss = torch.sum(pert_active_weights * torch.abs(pert_recon - pert)) / minibatch_size
                    
                    if use_sparsity_loss and len(control_results) == 3:
                        control_sparsity, pert_sparsity = control_results[1], pert_results[1]
                        control_loss += torch.sum(active_weights * torch.abs(control_sparsity - (control > 0).float())) / minibatch_size
                        pert_loss += torch.sum(pert_active_weights * torch.abs(pert_sparsity - (pert > 0).float())) / minibatch_size

                    loss = (pert_loss + control_loss)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                    losses['control'].append(control_loss.item())
                    losses['pert'].append(pert_loss.item())
                    if step_count % lr_step == 0:
                        lr_scheduler.step()
                    pbar.set_description(
                        f"tv: {np.array(losses['control'])[-lr_step:].mean():.3f}, ptv: {np.array(losses['pert'])[-lr_step:].mean():.3f}"
                    )
            
            avg_loss = sum(losses['control']) / len(losses['control'])
            torch.save(model, f"{save_dir}{e}")
            # writer.add_scalar('mae_loss', avg_loss, global_step=e)
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

        
        return preds
    

    #I mean to we need to evaluate anything? 
    def make_predict(self, adata: sc.AnnData | torch.DataLoader, pert_id: str, cell_type: str) -> np.ndarray:
        pass 
