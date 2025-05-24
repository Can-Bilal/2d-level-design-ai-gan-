import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_height=550, img_width=720):
        super(Discriminator, self).__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        
        def discriminator_block(in_filters, out_filters, bn=True, kernel_size=4, stride=2, padding=1):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block
        
        # Daha az katman kullanarak boyutu azalt
        self.model = nn.Sequential(
            *discriminator_block(3, 32, bn=False),      # 720x550 -> 360x275
            *discriminator_block(32, 64),               # 360x275 -> 180x138
            *discriminator_block(64, 128),              # 180x138 -> 90x69
            *discriminator_block(128, 256),             # 90x69 -> 45x35
        )

        # Calculate the size of the output features
        ds_height = img_height // (2**4)  # 4 discriminator blocks
        ds_width = img_width // (2**4)
        
        # Debug bilgisi
        print(f"Discriminator output size: {ds_height}x{ds_width}")
        
        self.adv_layer = nn.Sequential(
            nn.Flatten(),  # Düzleştirme katmanı ekle
            nn.Linear(256 * ds_height * ds_width, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        # Debug bilgisi
        print(f"Features shape before flatten: {features.shape}")
        validity = self.adv_layer(features)
        return validity
