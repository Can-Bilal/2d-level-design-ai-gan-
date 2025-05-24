import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

print("CUDA kullanılabilir mi:", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Transform tanımla - resim boyutunu ayarla
IMG_HEIGHT = 550
IMG_WIDTH = 720
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), antialias=True),
    transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 3 else x)
])

from train import train_gan
from models.generator import Generator
from models.discriminator import Discriminator
from data.data_loader import LevelDataset, save_level_with_tileset
import os

def main():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA kullanılıyor, cudnn benchmark aktif")

    # Veri yolu ayarları
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tileset_dir = os.path.join(base_dir, 'tileset_dir')  # tileset_d klasörünü kullan
    
    # Veri seti oluştur
    dataset = LevelDataset(tileset_dir)
    print(f"Toplam {len(dataset)} görüntü yüklendi")
    
    # İlk görüntünün boyutunu yazdır
    first_image = dataset[0]
    if isinstance(first_image, Image.Image):
        print(f"İlk görüntü boyutu: {first_image.size}")
    
    # Modelleri oluştur ve GPU'ya taşı
    generator = Generator(latent_dim=100, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    generator = generator.to(device)
    discriminator = Discriminator(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    
    # Optimizer ve loss fonksiyonu
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    print("Eğitim başlıyor...")
    
    # Eğitim
    for epoch in range(1):  
        for i in range(len(dataset)): 
            real_level = dataset[i]
            
            # Transform uygula ve GPU'ya taşı
            if isinstance(real_level, Image.Image):
                real_level = transform(real_level)
            elif isinstance(real_level, torch.Tensor) and real_level.dim() == 3:
                real_level = real_level.unsqueeze(0)
            
            real_level = real_level.to(device)
            
            # Print shape for debugging (only first batch of first epoch)
            if i == 0 and epoch == 0:
                print(f"Input tensor shape: {real_level.shape}")
            
            # Discriminator eğitim
            optimizer_d.zero_grad()
            outputs = discriminator(real_level)
            real_loss = criterion(outputs, torch.ones_like(outputs))
            
            # Generator için fake data
            z = torch.randn(1, 100).to(device)
            fake_level = generator(z)
            outputs = discriminator(fake_level.detach())
            fake_loss = criterion(outputs, torch.zeros_like(outputs))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Generator eğitim
            optimizer_g.zero_grad()
            outputs = discriminator(fake_level)
            g_loss = criterion(outputs, torch.ones_like(outputs))
            g_loss.backward()
            optimizer_g.step()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{1}], Step [{i}/{len(dataset)}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}")

    print("Eğitim tamamlandı!")
    # Örnek leveller üret
    generate_sample_levels(generator, tileset_dir, num_samples=5, device=device)

def generate_sample_levels(generator, tileset_dir, num_samples=5, device='cpu'):
    os.makedirs('outputs/final_samples', exist_ok=True)
    
    z = torch.randn(num_samples, generator.latent_dim).to(device)
    
    generator.eval()
    
    with torch.no_grad():
        fake_levels = generator(z)
        fake_levels = (fake_levels + 1) / 2.0
        
        for i in range(num_samples):
            level = fake_levels[i].detach().cpu().numpy()
            
            print(f"Level shape: {level.shape}")
            print(f"Level min value: {level.min():.4f}, max value: {level.max():.4f}")
            
            level = np.transpose(level, (1, 2, 0))
            
            output_path = f'outputs/final_samples/generated_level_{i+1}.png'
            save_level_with_tileset(level, tileset_dir, output_path)

if __name__ == "__main__":
    main()
