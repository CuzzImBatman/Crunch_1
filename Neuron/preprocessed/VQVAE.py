import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Vector Quantizer
from torch.utils.tensorboard import SummaryWriter
import csv
import os

# TensorBoard Logger
log_dir = 'encoder_tensorboard_logs'
os.makedirs(log_dir, exist_ok=True)
tensorboard_writer = SummaryWriter(log_dir=log_dir)

# Model Checkpointing Function
def save_model_checkpoint(model, optimizer, epoch, loss, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f'encoder-epoch-{epoch:04d}-loss-{loss:.4f}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    print(f"Model checkpoint saved: {checkpoint_path}")

# CSV Logger
csv_log_path = 'encoder_logs.csv'
if not os.path.exists(csv_log_path):
    # Create a new CSV file with headers
    with open(csv_log_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'loss'])

def log_to_csv(epoch, loss, log_path=csv_log_path):
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss])
    print(f"Logged to CSV: epoch={epoch}, loss={loss}")
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Embedding table
        self.embeddings = nn.Parameter(torch.randn(embedding_dim, num_embeddings))

    def forward(self, x):
        # Flatten input
        flat_x = x.view(-1, self.embedding_dim)

        # Compute L2 distances
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=0)
            - 2 * torch.matmul(flat_x, self.embeddings)
        )

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()
        quantized = torch.matmul(encodings, self.embeddings.t()).view(*x.shape)

        # Loss calculation
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        loss = self.beta * e_latent_loss + q_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss


# Encoder
class Encoder(nn.Module):
    def __init__(self, image_size=20, latent_dim=16):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 1),
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=16):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


# VQ-VAE Model
class VQVAE(nn.Module):
    def __init__(self, image_size=28, latent_dim=16, num_embeddings=1024):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(image_size, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq_layer(z_e)
        recon_x = self.decoder(z_q)
        return recon_x, vq_loss


# Data Generator
class VQVAEDataGenerator(Dataset):
    def __init__(self, tensor_folder, augmentation=True, random_seed=1234, k=5, m_list=[0, 1, 2, 3, 4]):
        self.augmentation = augmentation
        # loc_list = sorted([os.path.basename(name) for name in glob.glob(os.path.join(image_folder, '*'))])
        # random.seed(random_seed)
        # random.shuffle(loc_list)

        # partitions = self.partition(loc_list, k)
        # selected_loc_list = [i for m in m_list for i in partitions[m]]

        # self.image_list = [
        #     img for loc in selected_loc_list for img in glob.glob(os.path.join(image_folder, loc, '*.png'))
        # ]
        NAMES = ['DC1','DC5', 'UC1_I', 'UC1_NI', 'UC6_I', 'UC6_NI', 'UC7_I', 'UC9_I']
        # NAMES= NAMES[:1]
        tensors_list=[            
            torch.load(f'{tensor_folder}/{name}.pt') for name in NAMES
        ]

        self.tensors_list = torch.cat(tensors_list, dim=0)

    def __len__(self):
        return len(self.tensors_list)

    def __getitem__(self, index):
        tensor =self.tensors_list[index]

        if self.augmentation:
            # Flip upside-down (vertically)
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[1])  # Flip along the height dimension

            # Flip left-to-right (horizontally)
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])

        # img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
        return tensor

    @staticmethod
    def partition(lst, n):
        division = len(lst) / float(n)
        return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


# Training Function
def train_vqvae(vqvae, dataloader, optimizer, num_epochs, device):
    vqvae = vqvae.to(device)
    for epoch in range(num_epochs):
        print(epoch)
        total_loss, recon_loss, vq_loss = 0, 0, 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, vqvae_loss = vqvae(batch)
            reconstruction_loss = torch.mean((batch - recon_batch) ** 2)
            loss = reconstruction_loss + vqvae_loss
            
            tensorboard_writer.add_scalar('Loss/train', loss, epoch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += reconstruction_loss.item()
            vq_loss += vqvae_loss.item()
        save_model_checkpoint(vqvae, optimizer, epoch, total_loss, save_dir='./checkpoint')

        print(f"Epoch {epoch + 1}: Total Loss: {total_loss:.4f}, Recon Loss: {recon_loss:.4f}, VQ Loss: {vq_loss:.4f}")


    # Example Usage
def main():
    tensor_folder = f'E:/Projects/Gene_expression/Crunch/patches/20'
    dataset = VQVAEDataGenerator(tensor_folder)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    vqvae = VQVAE(image_size=20, latent_dim=16, num_embeddings=1024)
    optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_vqvae(vqvae, dataloader, optimizer, num_epochs=10, device=device)


    # Example Usage in Training Loop



    # Close TensorBoard writer after training
    tensorboard_writer.close()
if __name__ == '__main__':
    main()