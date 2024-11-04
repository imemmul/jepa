import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead3D(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1, hidden_dims=[512, 256, 128], upsample_temporal=True):
        """
        Segmenter with 3D transposed convolutions for upsampling.
        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels for segmentation (e.g., number of classes).
            hidden_dims (list of int): Number of channels for each hidden 3D transposed convolution layer.
            upsample_temporal (bool): Whether to upsample the temporal dimension to 16 if needed.
        """
        super(SegmentationHead3D, self).__init__()
        
        layers = []
        current_channels = in_channels
        
        if upsample_temporal:
            layers.append(
                nn.ConvTranspose3d(
                    current_channels, current_channels, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )
            )
        
        for h_dim in hidden_dims:
            layers.append(
                nn.ConvTranspose3d(
                    current_channels, h_dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
                )
            )
            layers.append(nn.BatchNorm3d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            current_channels = h_dim

        # Final layer to reach the desired output resolution
        layers.append(
            nn.ConvTranspose3d(
                current_channels, out_channels, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
            )
        )

        self.upsampler = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsampler(x)
        return x
# class SegmentationHead3D(nn.Module):
#     def __init__(self, feature_dim, num_classes=1):
#         super(SegmentationHead3D, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv3d(feature_dim, 512, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm3d(512)
        
#         self.conv2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm3d(256)
        
#         self.conv3 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm3d(128)
        
#         self.conv4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm3d(64)
        
#         self.conv_final = nn.Conv3d(64, num_classes, kernel_size=1)        
#         self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         self.upconv2 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         self.upconv3 = nn.ConvTranspose3d(num_classes, num_classes, kernel_size=(1, 4, 4), stride=(1, 4, 4))  # Upsample H, W by 4
#     def forward(self, x):
#         # x: [batch_size, feature_dim, num_frames, height_patches, width_patches] => [B, C, T, H, W]
        
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.upconv1(x)
        
#         x = F.relu(self.bn4(self.conv4(x)))
        
#         x = self.upconv2(x)
        
#         x = self.conv_final(x)
        
#         x = self.upconv3(x)
        
#         x = x.squeeze(1)
        
#         return x