import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pickle

# Function to encode input data using a pre-trained VAE model.
# The input data should be a numpy array where rows represent samples (e.g., cells)
# and columns represent features (e.g., gene expressions).
def Encode(inp, model_path='Models/', model_name='VAE'):
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model parameters from a pickle file
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

        # Encoder function
        def encode(self, x):
            h = F.leaky_relu(self.encoder_bn1(self.encoder_fc1(x)))  
            h = F.dropout(h, p=self.dropout_rate, training=self.training)  
            h = F.leaky_relu(self.encoder_bn2(self.encoder_fc2(h)))  
            return self.fc_mu(h), self.fc_logvar(h)  



    # Initialize the VAE model, load the pre-trained weights, and set it to evaluation mode
    net = Net(num_genes)
    net.to(device)
    net.load_state_dict(torch.load(f'{model_path}{model_name}.pth'))
    net.eval()
    
    # Prepare input data
    inplen = inp.shape[0]
    inp = torch.from_numpy(inp.astype(np.float32)).to(device)
    
    # Output tensors for storing encoded means and log variances
    out = torch.zeros((inplen, latent_dim), device=device)
    outvar = torch.zeros((inplen, latent_dim), device=device)

    # Process each batch and store the encoded outputs
    for lower in range(0, inplen, batsize):
        upper = min(lower + batsize, inplen)
        lower = min(inplen - batsize, lower)
        batch = inp[lower:upper, :]
        mu, logvar = net.encode(batch)
        out[lower:upper, :] = mu
        outvar[lower:upper, :] = logvar

    # Convert outputs back to numpy arrays
    out = out.cpu().detach().numpy()
    outvar = outvar.cpu().detach().numpy()
    
    return out, outvar

   