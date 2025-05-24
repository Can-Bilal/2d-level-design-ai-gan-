import torch
import os
from models.generator import Generator
from data.data_loader import save_level_with_tileset

def generate_levels(num_levels=10, output_dir='outputs/generated_levels'):
    # Cihazı belirle
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Generator'ı yükle
    z_dim = 100
    generator = Generator(z_dim).to(device)
    
    # Çıktı klasörünü oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Level üret ve kaydet
    for i in range(num_levels):
        # Level oluştur
        with torch.no_grad():
            generated_level = generator.generate_level(batch_size=1, device=device)
            
            # Tensörü numpy dizisine çevir
            level_image = generated_level[0].permute(1, 2, 0).cpu().numpy() * 255
            level_image = level_image.astype('uint8')
            
            # Kaydet
            output_path = os.path.join(output_dir, f'generated_level_{i+1}.jpg')
            from PIL import Image
            Image.fromarray(level_image).save(output_path)
        
        print(f"Generated level {i+1} saved to {output_path}")

if __name__ == "__main__":
    generate_levels(num_levels=20)  # 20 adet level üret
