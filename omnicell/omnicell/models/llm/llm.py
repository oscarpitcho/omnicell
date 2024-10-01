import torch
from torch import nn
import numpy as np

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return sequences[torch.arange(sequences.size(0)).unsqueeze(1), indexes]

class GeneShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, genes : torch.Tensor, mask=True):
        B, T, C = genes.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=0), dtype=torch.long).to(genes.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=0), dtype=torch.long).to(genes.device)

        genes = take_indexes(genes, forward_indexes)
        if mask:
            genes = genes[:, :remain_T]

        return genes, forward_indexes, backward_indexes
    

class BernoulliSampleLayer(nn.Module):
    def __init__(self):
        super(BernoulliSampleLayer, self).__init__()

    def forward(self, probs):
        sample = torch.bernoulli(probs)
        return sample + probs - probs.detach()
    
class PosEmbedding(torch.nn.Module):
    def __init__(self, input_dim, emb_dim=12):
        super().__init__()
        self.pos = torch.nn.Parameter(torch.zeros(1, input_dim, emb_dim))
        nn.init.normal_(self.pos)

class PertEmbedder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        _, input_dim, emb_dim = encoder.pos_embedding.pos.shape
        self.encoder = encoder
        self.pert_token = torch.nn.Parameter(torch.zeros(1, emb_dim))
        nn.init.normal_(self.pert_token)
        
    def forward(self, pert_index, pert_expression):
        pert_features = self.encoder.expression_embed(pert_expression.unsqueeze(-1))
        # pert_features = pert_features + self.encoder.pos_embedding.pos[:, pert_index, :] + self.pert_token
        pert_pos = self.encoder.pos_embedding.pos[:, pert_index, :][0].unsqueeze(1)
        pert_features = pert_features + pert_pos + self.pert_token
        return pert_features

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 pos_embedding,
                 num_layer=6,
                 num_head=3,
                 mask_ratio=0.75,
                 ff_dim=128,
                 ) -> None:
        super().__init__()
        
        _, input_dim, emb_dim = pos_embedding.pos.shape
        self.pos_embedding = pos_embedding
        
        self.shuffle = GeneShuffle(mask_ratio)

        self.expression_embed = torch.nn.Linear(1, emb_dim)

        self.transformer = torch.nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    emb_dim, num_head, dim_feedforward=ff_dim, batch_first=True, dropout=0.
                ) 
                for _ in range(num_layer)
            ]        
        )

        self.layer_norm = torch.nn.LayerNorm(emb_dim)


    def forward(self, batch, mask=True):
        genes = self.expression_embed(batch.unsqueeze(-1))
        genes = genes + self.pos_embedding.pos

        features, forward_indexes, backward_indexes = self.shuffle(genes, mask=mask)
        features = self.layer_norm(self.transformer(features))

        return features, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 pos_embedding,
                 num_layer=6,
                 num_head=3,
                 ff_dim=128,
                 true_sparsity=True,
                 expr_activation='sigmoid'
                 ) -> None:
        super().__init__()
        
        _, input_dim, emb_dim = pos_embedding.pos.shape
        self.pos_embedding = pos_embedding

        self.mask_token = torch.nn.Parameter(torch.zeros(1, emb_dim))

        self.transformer = torch.nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    emb_dim, num_head, dim_feedforward=ff_dim, batch_first=True, dropout=0.
                ) 
                for _ in range(num_layer)
            ]
        )

        self.head = torch.nn.Linear(emb_dim, 2)
        self.sigmoid = torch.nn.Sigmoid()
        self.sparsity = BernoulliSampleLayer()
        self.true_sparsity = true_sparsity
        if expr_activation == 'sigmoid':
            self.expr_activation = torch.nn.Sigmoid()
        elif expr_activation == 'identity':
            self.expr_activation = lambda x: x
        else:
            print("Error: expr_activation must be sigmoid or identity")
        
        nn.init.normal_(self.mask_token, std=.02)


    def forward(self, features, backward_indexes, pert_features=None):
        T = features.shape[1]
        features = torch.cat(
            [features, self.mask_token.expand(features.shape[0], backward_indexes.shape[1] - features.shape[1], -1)], dim=1
        )
        # we have to add this or the masked objective will not work!
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding.pos
        
        if pert_features is not None:
            _, N, _ = pert_features.shape
            features = torch.cat([features, pert_features], dim=1)

        features = self.transformer(features)
        
        if pert_features is not None:
            features = features[:, :-N, :]
        expr_logits, sparsity_logits = self.head(features).unbind(-1)
        
        mask = torch.zeros_like(expr_logits)
        mask[:, T:] = 1
        mask = take_indexes(mask, backward_indexes)
        
        expr = self.sigmoid(expr_logits)
        
        if self.true_sparsity:
            sparsity_probs = self.sigmoid(sparsity_logits)
            sparsity = self.sparsity(sparsity_probs)
            expr = expr * sparsity
            return expr, sparsity_probs, mask

        return expr, mask


class MAE(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 ff_dim=128,
                 emb_dim=256,
                 encoder_layer=6,
                 encoder_head=4,
                 decoder_layer=6,
                 decoder_head=4,
                 mask_ratio=0.75,
                 true_sparsity=True,
                 expr_activation='sigmoid'
                 ) -> None:
        super().__init__()
        
        self.pos_embedding = PosEmbedding(input_dim=input_dim, emb_dim=emb_dim)
        self.encoder = MAE_Encoder(
            self.pos_embedding, encoder_layer, encoder_head, mask_ratio, ff_dim=ff_dim)
        self.pert_embedding = PertEmbedder(self.encoder)
        self.decoder = MAE_Decoder(
            self.pos_embedding, decoder_layer, decoder_head, ff_dim=ff_dim,
            true_sparsity=true_sparsity, expr_activation=expr_activation
        )
        self.pert_decoder = MAE_Decoder(self.pos_embedding, decoder_layer, decoder_head, ff_dim=ff_dim,true_sparsity=true_sparsity, expr_activation=expr_activation)
        # self.pert_decoder = self.decoder

    def forward(self, batch, mask=True, pert_index=None, pert_expr=None, recon_and_pert=False):
        features, backward_indexes = self.encoder(batch, mask=mask)
        if pert_index is not None and pert_expr is not None:
            pert_features = self.pert_embedding(pert_index, pert_expr)
            if recon_and_pert:
                recon_out = self.decoder(features, backward_indexes)
                pert_out = self.pert_decoder(
                    features, backward_indexes, pert_features=pert_features
                )
                return recon_out, pert_out
            else:
                return self.pert_decoder(
                    features, backward_indexes, pert_features=pert_features
                )
        else:
            return self.decoder(features, backward_indexes)
