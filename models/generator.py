import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_height=550, img_width=720):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width
        
        # Calculate initial sizes
        self.init_height = img_height // 8
        self.init_width = img_width // 8
        
        # Debug bilgisi
        print(f"Generator initial size: {self.init_height}x{self.init_width}")
        
        # Level elementlerinin özellikleri
        self.level_elements = {
            'walls': {'count': 4, 'required': True},  # 4 duvar
            'floor': {'required': True},  # Zemin
            'door': {'count': 1, 'required': True},  # 1 çıkış kapısı
            'key': {'count': 1, 'required': True},  # 1 anahtar
            'character': {'count': 1, 'required': True},  # 1 karakter
            'enemies': {'min': 1, 'max': 5},  # 1-5 arası düşman
            'barriers': {'min': 2, 'max': 8}  # 2-8 arası bariyer
        }
        
        # Initial dense layer
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_height * self.init_width),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            nn.Upsample(scale_factor=2),  # First upscale
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # Second upscale
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),  # Third upscale
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

        print("\nGenerator özellikleri:")
        print("Level elementleri:")
        print("- 4 duvar (zorunlu)")
        print("- Zemin (zorunlu)")
        print("- 1 çıkış kapısı (zorunlu)")
        print("- 1 anahtar (zorunlu)")
        print("- 1 karakter (zorunlu)")
        print(f"- {self.level_elements['enemies']['min']}-{self.level_elements['enemies']['max']} düşman")
        print(f"- {self.level_elements['barriers']['min']}-{self.level_elements['barriers']['max']} bariyer")

    def forward(self, z):
        # Initial dense layer
        out = self.l1(z)
        # Reshape to match convolutional layers
        out = out.view(out.shape[0], 256, self.init_height, self.init_width)
        # Debug bilgisi
        print(f"Generator reshape size: {out.shape}")
        # Apply convolutional blocks
        img = self.conv_blocks(out)
        # Debug bilgisi
        print(f"Generator output size: {img.shape}")
        return img
    
    def generate_level(self, batch_size=1, device='cpu'):
        # Rastgele gürültü oluştur
        z = torch.randn(batch_size, self.latent_dim).to(device)
        # Level üret
        generated_levels = self.forward(z)
        
        # Çıktıyı görüntü formatına çevir
        generated_levels = generated_levels * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        generated_levels = generated_levels.clamp(0, 1)
        
        return generated_levels
