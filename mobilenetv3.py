import torch
import torch.nn as nn
import torch.nn.functional as F

class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel),
            hsigmoid()
        )

    def forward(self, x):
        b, c = x.size()[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride, expansion, se=False, nl='RE'):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        # Activation selection
        act = hswish if nl == 'HS' else nn.ReLU
        
        # Expansion phase
        expanded_channels = expansion * in_channels
        self.conv1 = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.act1 = act()
        
        # Depthwise convolution
        self.conv2 = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride, padding=(kernel_size-1)//2,
            groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(expanded_channels)
        self.se = SEBlock(expanded_channels) if se else nn.Identity()
        self.act2 = act()
        
        # Output phase
        self.conv3 = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = act()

    def forward(self, x):
        residual = x
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.use_res_connect:
            out += residual
        return self.act3(out)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes=3755):
        super().__init__()
        # Adjusted for 32x32 input
        self.features = nn.Sequential(
            # Initial conv (stride=1 for small input)
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            hswish(),
            
            # Bottleneck config (expansion, out_channels, kernel, stride, se, nl)
            Bottleneck(16, 16, 3, 1, 1, se=True, nl='RE'),
            Bottleneck(16, 24, 3, 2, 4, se=False, nl='RE'),
            Bottleneck(24, 24, 3, 1, 3, se=False, nl='RE'),
            Bottleneck(24, 40, 5, 2, 4, se=True, nl='HS'),
            Bottleneck(40, 40, 5, 1, 6, se=True, nl='HS'),
            Bottleneck(40, 40, 5, 1, 6, se=True, nl='HS'),
            Bottleneck(40, 48, 5, 1, 3, se=True, nl='HS'),
            Bottleneck(48, 48, 5, 1, 3, se=True, nl='HS'),
            Bottleneck(48, 96, 5, 2, 6, se=True, nl='HS'),
            Bottleneck(96, 96, 5, 1, 6, se=True, nl='HS'),
            Bottleneck(96, 96, 5, 1, 6, se=True, nl='HS'),
        )
        
        # Final layers
        self.conv = nn.Conv2d(96, 576, 1, bias=False)
        self.bn = nn.BatchNorm2d(576)
        self.act = hswish()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            hswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = self.act(self.bn(self.conv(x)))
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)

# def test():
#     net = MobileNetV3(num_classes=3755)
#     x = torch.randn(2, 3, 32, 32)
#     y = net(x)
#     print(y.size())  # Should be [2, 3755]


# if __name__ == "__main__":
#     test()

