
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes=None, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        self.ratio = ratio
        
        
        if in_planes is not None:
            self.create_mlp(in_planes)
        else:
            
            self.shared_MLP = None
            
    def create_mlp(self, in_planes):
        
        reduced_planes = max(1, in_planes // self.ratio)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        )
            
    def forward(self, x):
        
        if self.shared_MLP is None:
            _, c, _, _ = x.shape
            self.create_mlp(c)
            
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels=None):
        super(CBAM, self).__init__()
        
        self.ca = ChannelAttention(channels, ratio=8 if channels and channels < 128 else 16)
        self.sa = SpatialAttention(kernel_size=7)
        
    def forward(self, x):
        
        ca_output = self.ca(x)
        x = x * ca_output
        
        
        sa_output = self.sa(x)
        x = x * sa_output
        
        return x
# Example usage
if __name__ == "__main__":
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 32, 32)  # Example input
    y = cbam(x)
    print("Output shape:", y.shape)