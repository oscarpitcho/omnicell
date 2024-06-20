import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pickle

# Decode function to reconstruct data from latent space variables using a pre-trained VAE.
# The input to the function includes latent means 'inp', latent log variances 'inpvar', and paths to the model.
def Decode(inp, inpvar, model_path='Models/', model_name='VAE'):
    # Configure the device (GPU or CPU) based on availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model parameters from a stored file
    with open(f'{model_path}{model_name}.pkl', 'rb') as fp:
        param = pickle.load(fp)
    epochs, batsize, latent_dim, hidden_dim, dropout_rate, learning_rate, alpha, num_genes = param
    
    # Define the VAE model architecture
    class Net(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.input_dim = input_dim  
            self.latent_dim = latent_dim  # Dimension of the latent space
            self.dropout_rate = dropout_rate  
            self.alpha = alpha  # Weight for the KL divergence term in loss
            self.model_path = model_path  # Directory to save model
            os.makedirs(self.model_path, exist_ok=True)  # Ensure model directory exists
            
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
            
            # Optimizer for the model
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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


    # Load the model, set to evaluation mode and load weights
    net = Net(num_genes)
    net.to(device)
    net.load_state_dict(torch.load(f'{model_path}{model_name}.pth'))
    net.eval()
    
    # Prepare input data and output storage
    inp = torch.from_numpy(inp.astype(np.float32)).to(device)
    inpvar = torch.from_numpy(inpvar.astype(np.float32)).to(device)
    out = torch.zeros((inp.shape[0], num_genes), device=device)

    # Process each batch
    for lower in range(0, inp.shape[0], batsize):
        upper = min(lower + batsize, inp.shape[0])
        lower = min(inp.shape[0] - batsize, lower)
        batch = inp[lower:upper, :]
        batchvar = inpvar[lower:upper, :]
        reparam = net.reparameterize(batch, batchvar)  # Reparameterize to get latent variables
        decod = net.decode(reparam)  # Decode latent variables to reconstruct data
        out[lower:upper, :] = decod  # Store reconstructed data

    # Convert tensor back to numpy array and return
    out = out.cpu().detach().numpy()
    return out

   