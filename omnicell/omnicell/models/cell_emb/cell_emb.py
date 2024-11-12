import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

# from omnicell.models.flows.flow_utils import ExactOptimalTransportConditionalFlowMatcher

from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,OTPlanSampler
)

class GeneEmbedding(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=128):
        super().__init__()
        self.pos = torch.nn.Parameter(torch.zeros(1, input_dim, emb_dim))
        nn.init.normal_(self.pos)
        
        
class PertEmbedder(torch.nn.Module):
    def __init__(self, gene_embedding):
        super().__init__()
        _, input_dim, emb_dim = gene_embedding.pos.shape
        self.gene_embedding = gene_embedding
        self.pert_token = torch.nn.Parameter(torch.zeros(emb_dim))
        nn.init.normal_(self.pert_token)
        
    def forward(self, pert_index, pert_expression):
        pert_pos = self.gene_embedding.pos[:, pert_index][0]

        pert_embed_and_expr = torch.cat(
            (
                pert_pos + self.pert_token, 
                pert_expression.unsqueeze(-1)
            ), dim=-1
        )
        return pert_embed_and_expr.unsqueeze(1)
    
class CellEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim  
        self.latent_dim = latent_dim  # Dimension of the latent space
        self.dropout_rate = dropout_rate  
        

        # Encoder network definitions
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim, bias=False) 
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dim)  
        self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.encoder_bn2 = nn.BatchNorm1d(hidden_dim) 
        self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True) 
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True) 

    # Encoder function
    def forward(self, x):
        h = F.leaky_relu(self.encoder_bn1(self.encoder_fc1(x)))  
        h = F.dropout(h, p=self.dropout_rate, training=self.training)  
        h = F.leaky_relu(self.encoder_bn2(self.encoder_fc2(h)))  
        return nn.ELU()(self.fc_mu(h))# , self.fc_logvar(h)  


class TransformerCellEncoder(torch.nn.Module):
    def __init__(self,
                 gene_embedding,
                 num_layer=6,
                 num_head=3,
                 ff_dim=128,
                 ) -> None:
        super().__init__()
        
        _, input_dim, emb_dim = gene_embedding.pos.shape
        self.gene_embedding = gene_embedding

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                emb_dim, num_head, dim_feedforward=ff_dim, batch_first=True, norm_first=True
            ) 
            for _ in range(num_layer)
        ])
        
        self.initial_embed = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + 1, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, emb_dim),
        )

    def forward(self, expr, idx, pert_embed_and_expr=None):
        embed_and_expr = torch.cat(
            (
                torch.tile(self.gene_embedding.pos[:, idx], (expr.shape[0], 1, 1)), 
                expr.unsqueeze(-1)
            ), dim=-1
        )
        if pert_embed_and_expr is not None:
            embed_and_expr = torch.cat((embed_and_expr, pert_embed_and_expr), dim=1)
        features_init = features = self.initial_embed(embed_and_expr)
        
        for layer in self.transformer_layers:
            features = layer(features + features_init)

        return features.mean(axis=-1)

class ExprPred(torch.nn.Module):
    def __init__(self,
                 gene_embedding,
                 ff_dim=128,
                 ) -> None:
        super().__init__()
        
        _, _, emb_dim = gene_embedding.pos.shape
        self.gene_embedding = gene_embedding

        self.pred_expr = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + emb_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, 1),
            torch.nn.ELU()
        ) 
        
        self.pred_bin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim + emb_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, ff_dim),
            torch.nn.SELU(),
            torch.nn.Linear(ff_dim, 1),
            torch.nn.Sigmoid()
        ) 

    def forward(self, cell_embedding, pred_idx):
        embed_and_cell_embed = torch.cat(
            (
            torch.tile(cell_embedding.unsqueeze(1), (1, pred_idx.shape[0], 1)),
            torch.tile(self.gene_embedding.pos[:, pred_idx], (cell_embedding.shape[0], 1, 1))
            ), dim=-1
        )
        pred_expr = self.pred_expr(embed_and_cell_embed) + 1
        pred_bin = self.pred_bin(embed_and_cell_embed)
        
        return pred_expr.squeeze(), pred_bin.squeeze()

class CMLP(pl.LightningModule):
    def __init__(self, feat_dim, cond_dim, out_dim=None, w1=128, w2=128, n_combo_layer=4, n_cond_layer=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + cond_dim, w1), torch.nn.SELU(),
            *([torch.nn.Linear(w1, w1), torch.nn.SELU()] * n_combo_layer),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond = None
        
    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        # cond = self.cond_net(cond)
        return self.combo_net(torch.cat([x, cond], dim=-1))

    
class BernoulliSampleLayer(nn.Module):
    def __init__(self):
        super(BernoulliSampleLayer, self).__init__()

    def forward(self, probs):
        sample = torch.bernoulli(probs)
        return sample + probs - probs.detach()    

    
class MAE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 ff_dim=128,
                 emb_dim=128,
                 encoder_layer=6,
                 ) -> None:
        super().__init__()
        
        self.gene_embedding = GeneEmbedding(input_dim=input_dim, emb_dim=emb_dim)
        self.pert_embedding = PertEmbedder(self.gene_embedding)
        self.encoder = CellEncoder(
            input_dim, emb_dim, hidden_dim=ff_dim
        )
        self.recon = ExprPred(self.gene_embedding)
        self.sparse_sampler = BernoulliSampleLayer()
        
        self.flow = CMLP(feat_dim=emb_dim, cond_dim=emb_dim, time_varying=True, w1=ff_dim)
        self.cfm_sampler = ExactOptimalTransportConditionalFlowMatcher(sigma=0.1)

    def forward(self, expr):
        cell_embedding = self.encoder(expr)
        return cell_embedding 
    
    def sparsify(self, pred_expr, pred_bin):
        sparsity = self.sparse_sampler(pred_bin)
        pred_expr *= sparsity
        return pred_expr
    
    def ae_loss(self, batch_emb, batch, gene_ids, lambd=0.5, return_recon=False):
        batch_bin = (batch > 0).float()
        batch_recon, batch_bin_recon = self.recon(batch_emb, gene_ids)
        recon_loss = torch.mean(batch_bin * (batch_recon - batch[:, gene_ids])**2) # / minibatch_size
        bin_loss = F.binary_cross_entropy(batch_bin_recon, batch_bin[:, gene_ids])
        loss = lambd * recon_loss + (1 - lambd) * bin_loss
        if return_recon:
            return loss, batch_recon, batch_bin_recon
        return loss
    
    def flow_loss(self, source_emb, target_emb, cond):
        t, xt, ut = self.cfm_sampler.sample_location_and_conditional_flow(
            source_emb, target_emb
        )

        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self.flow(inp, cond)
        return torch.nn.functional.mse_loss(vt, ut) 
