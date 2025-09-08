import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Basic Conv Block ----
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---- U-Net for 1D ----
class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, features=[64, 128, 256, 512]):
        super(UNet1D, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(DoubleConv1D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv1D(features[-1], features[-1]*2)

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv1D(feature*2, feature))

        # Final Convolution
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Upsampling path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Transposed convolution
            skip_connection = skip_connections[idx//2]

            # If necessary, pad to match dimension
            if x.shape[-1] != skip_connection.shape[-1]:
                x = F.pad(x, (0, skip_connection.shape[-1] - x.shape[-1]))

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)


# ---- Example Usage ----
if __name__ == "__main__":
    model = UNet1D(in_channels=1, out_channels=4)  # 4 = background, class1, class2, class3
    x = torch.randn(4, 1, 512)  # batch=4, channel=1, length=512
    preds = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", preds.shape)  # (batch, out_channels, 512)
    # assert preds.shape == (4, 2, 512), "Output shape is incorrect"
