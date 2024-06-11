import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import pickle

# Train a Variational Autoencoder (VAE) based on a numpy array where rows correspond to samples (cells)
# and columns to features (genes). This function includes parameter definitions and training logic.
def Train_VAE(inp, model_path='Models/', model_name='VAE', epochs=300, batsize=32, latent_dim=100, 
              hidden_dim=800, dropout_rate=0.2, learning_rate=0.001, alpha=0.001):
    # Set device to GPU if available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
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

    # Prepare data for training
    datalen = inp.shape[0]
    indices = np.random.permutation(datalen)
    train = inp[indices[:np.int32(datalen*0.9)]]  # 90% training data
    valid = inp[indices[np.int32(datalen*0.9):]]  # 10% validation data
    
    trainlen = train.shape[0]
    validlen = valid.shape[0]
    num_genes = train.shape[1]
    net = Net(num_genes)  
    net.to(device)  # Model for training
    neteval = Net(num_genes)  # Model for evaluation
    neteval.to(device)
    neteval.eval()  # Set evaluation mode

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    running_loss = 0

    train = torch.from_numpy(train.astype(np.float32)).to(device)  # Convert train data to torch tensor and move to device
    valid = torch.from_numpy(valid.astype(np.float32)).to(device)  # Convert validation data to torch tensor and move to device
    
    # Training loop
    for e in range(epochs):
        running_loss = 0
    
        for lower in range(0, trainlen, batsize):
            upper = min(lower + batsize, trainlen)
            lower = min(trainlen - batsize, lower)
            batch = train[lower:upper, :]
            optimizer.zero_grad()  
            out, mu, logvar = net(batch)  # Forward pass
            loss = net.loss_function(out, batch, mu, logvar) 
            loss.backward() 
            running_loss += loss.item()
            optimizer.step() 
        print(f'Epoch {e+1}/{epochs}')
        print(f'Train loss: {running_loss/1000000}')
    
        running_loss = 0
        state_dict = net.state_dict()
        neteval.load_state_dict(state_dict)  # Update evaluation model state
        
        # Validation loop
        with torch.no_grad():
            for lower in range(0, validlen, batsize):
                upper = min(lower + batsize, validlen)
                lower = min(validlen - batsize, lower)
                batch = valid[lower:upper, :]
                out, mu, logvar = neteval(batch)
                loss = neteval.loss_function(out, batch, mu, logvar)
                running_loss += loss.item()
            print(f'Valid loss: {running_loss/1000000}')
    

    print(f'Saving model to {model_path}{model_name}.pth')
    torch.save(net.state_dict(), f'{model_path}{model_name}.pth')  # Save model weights
    params = [epochs, batsize, latent_dim, hidden_dim, dropout_rate, learning_rate, alpha, num_genes]

    with open(f'{model_path}{model_name}.pkl', 'wb') as fp:
        pickle.dump(params, fp)  # Save model parameters

