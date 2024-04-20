import torch
import pytorch_lightning as pl

class MLP(pl.LightningModule):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

    def _shared_step(self, batch, batch_idx):
        t, xt, ut = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class CMLP(pl.LightningModule):
    def __init__(self, feat_dim, cond_dim, out_dim=None, w1=64, w2=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + w2, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, w1)
        )
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2)
        )
        self.cond = None
        

    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        cond = self.cond_net(cond)
        return self.combo_net(torch.cat([x, cond], dim=-1))

    def _shared_step(self, batch, batch_idx):
        t, xt, ut, pt = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp, pt)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
class CMHA(pl.LightningModule):
    def __init__(self, feat_dim, cond_dim, out_dim=None, w1=64, w2=64, num_heads=3, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = feat_dim
        self.feat_net = torch.nn.Sequential(
            torch.nn.Linear(feat_dim + (1 if time_varying else 0) + w2, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, w1 * num_heads)
        )
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2),
            torch.nn.SELU(),
            torch.nn.Linear(w2, w2)
        )
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=w1 * num_heads, num_heads=num_heads)
        self.mha2 = torch.nn.MultiheadAttention(embed_dim=w1 * num_heads, num_heads=num_heads)
        self.combo_net = torch.nn.Sequential(
            torch.nn.Linear(w1 * num_heads, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, w1),
            torch.nn.SELU(),
            torch.nn.Linear(w1, out_dim)
        )
        self.cond = None
        

    def forward(self, x, cond=None):
        if cond is None:
            cond = self.cond
        cond = self.cond_net(cond)
        combo_rep = torch.cat([x, cond], dim=-1)
        x = self.feat_net(torch.cat([x, cond], dim=-1))
        z, w1 = self.mha1(x, x, x)
        z, w2 = self.mha2(z + x, z + x, z + x)
        return self.combo_net(z)

    def _shared_step(self, batch, batch_idx):
        t, xt, ut, pt = batch
        inp = torch.cat([xt, t[:, None]], dim=-1)
        vt = self(inp, pt)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)