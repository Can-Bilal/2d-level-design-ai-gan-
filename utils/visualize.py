import torch
import matplotlib.pyplot as plt
import numpy as np

# Level elementleri için renkler
COLORS = {
    0: 'white',     # Boş alan
    1: 'gray',      # Platform
    2: 'yellow',    # Coin/Ödül
    3: 'red',       # Engel
    4: 'green',     # Başlangıç
    5: 'blue'       # Bitiş
}

def save_level_image(level_array, save_path):
    """
    Level array'ini görselleştirip kaydeder
    
    Args:
        level_array: Shape (32, 16) olan numpy array
        save_path: Görüntünün kaydedileceği dosya yolu
    """
    plt.figure(figsize=(8, 4))
    
    # Her hücre için uygun rengi kullan
    colored_array = np.zeros((level_array.shape[0], level_array.shape[1], 3))
    for i in range(level_array.shape[0]):
        for j in range(level_array.shape[1]):
            cell_type = level_array[i, j]
            color = COLORS[cell_type]
            if color == 'white':
                colored_array[i, j] = [1, 1, 1]
            elif color == 'gray':
                colored_array[i, j] = [0.5, 0.5, 0.5]
            elif color == 'yellow':
                colored_array[i, j] = [1, 1, 0]
            elif color == 'red':
                colored_array[i, j] = [1, 0, 0]
            elif color == 'green':
                colored_array[i, j] = [0, 1, 0]
            elif color == 'blue':
                colored_array[i, j] = [0, 0, 1]
    
    plt.imshow(colored_array)
    plt.grid(True)
    plt.title('Generated Level Layout')
    plt.savefig(save_path)
    plt.close()

def display_level(level_array):
    """
    Level array'ini ekranda gösterir
    
    Args:
        level_array: Shape (32, 16) olan numpy array
    """
    save_level_image(level_array, 'temp.png')
    img = plt.imread('temp.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def visualize(generator, z_dim=100):
    # Yeni bir rastgele gürültü vektörü oluştur
    z = torch.randn(1, z_dim)
    
    # Generator ile yeni bir görüntü oluştur
    generated_img = generator(z)
    
    # Görüntüyü görselleştir
    plt.imshow(generated_img[0].detach().numpy().transpose(1, 2, 0))
    plt.axis('off')  # Eksenleri gizle
    plt.show()
