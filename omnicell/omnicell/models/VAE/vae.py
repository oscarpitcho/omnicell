import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from numpy import genfromtxt
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Net(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, alpha, dropout_rate):
            super().__init__()
            self.input_dim = input_dim  
            self.latent_dim = latent_dim  # Dimension of the latent space
            self.dropout_rate = dropout_rate  
            self.alpha = alpha  # Weight for the KL divergence term in loss

            # Encoder network definitions
            self.encoder_fc1 = nn.Linear(input_dim, hidden_dim, bias=False) 
            self.encoder_bn1 = nn.BatchNorm1d(hidden_dim)  
            self.encoder_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False) 
            self.encoder_bn2 = nn.BatchNorm1d(hidden_dim) 
            self.fc_mu = nn.Linear(hidden_dim, latent_dim, bias=True) 
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim, bias=True) 
            
            # Decoder network definitions
            self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim, bias=False) 
            self.decoder_bn1 = nn.BatchNorm1d(hidden_dim)  
            self.decoder_fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  
            self.decoder_bn2 = nn.BatchNorm1d(hidden_dim)  
            self.decoder_fc3 = nn.Linear(hidden_dim, input_dim, bias=True) 
            

        # Encoder function
        def encode(self, x):
            h = F.leaky_relu(self.encoder_bn1(self.encoder_fc1(x)))  
            h = F.dropout(h, p=self.dropout_rate, training=self.training)  
            h = F.leaky_relu(self.encoder_bn2(self.encoder_fc2(h)))  
            return self.fc_mu(h), self.fc_logvar(h)  

        # Reparameterization trick to sample from Q(z|X)
        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)  # Standard deviation
            eps = torch.randn_like(std)  # Random noise
            return mu + eps * std  # Sampled latent variable

        # Decoder function
        def decode(self, z):
            h = self.decoder_fc1(z)
            h = self.decoder_bn1(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.decoder_fc2(h)
            h = self.decoder_bn2(h)
            h = F.leaky_relu(h)
            h = F.dropout(h, p=self.dropout_rate, training=self.training)
            h = self.decoder_fc3(h)
            h = torch.relu(h)  
            return h


        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))  # Encode input
            z = self.reparameterize(mu, logvar)  # Reparameterize
            return self.decode(z), mu, logvar  # Decode and return reconstruction and latent variables


        def loss_function(self, recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x.view(-1, self.input_dim), reduction='sum')  # Reconstruction loss
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
            return recon_loss + self.alpha * KLD  # Total loss
