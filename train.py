import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import Generator
from models.discriminator import Discriminator
from data.data_loader import create_dataloader, save_level_with_tileset
import os
import numpy as np

def train_gan(generator, discriminator, dataloader, tileset_dir, epochs=200, z_dim=100, batch_size=64, device="cuda"):
    """
    GAN modelini eğitir
    """
    # GPU kontrolü
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Eğitim cihazı: {device}")
    
    # Modelleri device'a taşı
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Optimizerleri tanımla
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # BCE loss yerine BCELoss kullan çünkü discriminator artık sigmoid içeriyor
    criterion = nn.BCELoss()

    # Eğitim döngüsü
    for epoch in range(epochs):
        for i, real_levels in enumerate(dataloader):
            # Batch boyutunu al
            current_batch_size = real_levels.size(0)
            
            # Gerçek ve sahte etiketler
            real_labels = torch.ones(current_batch_size).to(device)
            fake_labels = torch.zeros(current_batch_size).to(device)

            # ---------------------
            #  Discriminator Eğitimi
            # ---------------------
            optimizer_D.zero_grad()
            
            # Gerçek örnekler üzerinde eğitim
            real_levels = real_levels.to(device)
            d_output_real = discriminator(real_levels).view(-1)
            d_loss_real = criterion(d_output_real, real_labels)

            # Sahte örnekler üzerinde eğitim
            z = torch.randn(current_batch_size, z_dim).to(device)
            fake_levels = generator(z)
            d_output_fake = discriminator(fake_levels.detach()).view(-1)
            d_loss_fake = criterion(d_output_fake, fake_labels)

            # Toplam discriminator kaybı
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Generator Eğitimi
            # -----------------
            optimizer_G.zero_grad()
            
            # Generator çıktılarını discriminator'dan geçir
            g_output = discriminator(fake_levels).view(-1)
            g_loss = criterion(g_output, real_labels)
            
            g_loss.backward()
            optimizer_G.step()

            # Her 100 adımda bir durumu yazdır
            if i % 100 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}] "
                    f"[D(x): {d_output_real.mean().item():.4f}] [D(G(z)): {g_output.mean().item():.4f}]"
                )

            # Her 500 adımda bir örnek üret
            batches_done = epoch * len(dataloader) + i
            if batches_done % 500 == 0:
                with torch.no_grad():
                    fake_levels = generator(torch.randn(1, z_dim).to(device))
                    # [-1, 1] aralığından [0, 1] aralığına dönüştür
                    fake_levels = (fake_levels + 1) / 2.0
                    fake_level = fake_levels[0].cpu().numpy()
                    fake_level = np.transpose(fake_level, (1, 2, 0))
                    
                    # Örnek leveli kaydet
                    output_path = f'outputs/samples/level_{batches_done}.png'
                    os.makedirs('outputs/samples', exist_ok=True)
                    save_level_with_tileset(fake_level, tileset_dir, output_path)

    return generator, discriminator

if __name__ == "__main__":
    # Parametreleri ayarla
    z_dim = 100
    batch_size = 64  # Batch size'ı artır
    tileset_dir = 'tileset_dir'
    device = "cuda"  # Cihazı belirle
    
    # Dataloader'ı oluştur
    dataloader = create_dataloader(tileset_dir, batch_size=batch_size)
    
    # Modelleri başlat
    generator = Generator(z_dim)
    discriminator = Discriminator()
    
    # Eğitimi başlat
    train_gan(generator, discriminator, dataloader, tileset_dir, device=device)