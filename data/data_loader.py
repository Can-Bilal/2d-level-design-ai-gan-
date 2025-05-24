import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LevelDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Level elementlerinin özellikleri
        self.required_elements = {
            'walls': (4, True),           # (sayı, zorunlu mu)
            'floor': (1, True),           # Zemin
            'door': (1, True),            # Çıkış kapısı
            'key': (1, True),             # Anahtar
            'character': (1, True),        # Karakter
            'enemies': (1, 5),            # Min-Max düşman sayısı
            'barriers': (2, 8)            # Min-Max bariyer sayısı
        }
        
        print("\nLevel özellikleri:")
        print("- 4 duvar (zorunlu)")
        print("- Zemin (zorunlu)")
        print("- 1 çıkış kapısı (zorunlu)")
        print("- 1 anahtar (zorunlu)")
        print("- 1 karakter (zorunlu)")
        print(f"- {self.required_elements['enemies'][0]}-{self.required_elements['enemies'][1]} düşman")
        print(f"- {self.required_elements['barriers'][0]}-{self.required_elements['barriers'][1]} bariyer")
        
        print(f"\nToplam {len(self.image_files)} görüntü yüklendi")
        if len(self.image_files) > 0:
            img_path = os.path.join(self.dataset_path, self.image_files[0])
            img = Image.open(img_path).convert('RGB')
            print(f"İlk görüntü boyutu: {img.size}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Dönüşüm işlemini uygula
        if self.transform:
            image = self.transform(image)
            print(f"Dönüştürülen görüntü boyutu: {image.size()}")  # Dönüştürülen tensör boyutunu yazdır
        
        return image

def create_dataloader(dataset_path, batch_size=64, image_size=(128, 128)):
    """
    Veri yükleyici oluşturur
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = LevelDataset(dataset_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader

def save_level_with_tileset(level_array, tileset_dir, output_path):
    """
    Üretilen leveli tileset kullanarak kaydeder
    """
    # [-1, 1] aralığından [0, 255] aralığına dönüştür
    level_array = ((level_array + 1) * 127.5).astype(np.uint8)
    
    # Görüntüyü kaydet
    image = Image.fromarray(level_array)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Level kaydedildi: {output_path}")
