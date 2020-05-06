"""
U-Net Model.
"""
import torch
import torch.nn as nn
from torch.nn.functional import softmax


def weights_init_normal(m):
    """initial weights
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.01)
        torch.nn.init.constant_(m.bias.data, 0.0)


def conv3x3(in_channels, out_channels):
    """3 x 3 2d convolution
    Args:
        in_channels: Number of channels in the input image;
        out_channels: Number of channels produced by the convolution;
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=drop_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate):
        super(UpSample, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Dropout2d(p=drop_rate),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class uNet(nn.Module):
    def __init__(self, in_channels=1, class_num=2, drop_rate=0, num_residual_blocks=5):
        super(uNet, self).__init__()

        self.resblock1 = []
        self.resblock2 = []
        self.concats = []
        self.upsample = []
        self.num_residual_blocks = num_residual_blocks
        self.max_pool = nn.MaxPool2d(2)
        self.deconv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, class_num, kernel_size=3, stride=1, padding=0),
            nn.Dropout2d(p=drop_rate),
            nn.ReLU()
        )
        # down sampling
        out_channels = 64
        for i in range(self.num_residual_blocks):
            self.resblock1 += [ResidualBlock(in_channels, out_channels, drop_rate)]
            in_channels = out_channels
            out_channels *= 2

        # up sampling
        out_channels = in_channels // 2
        for i in range(self.num_residual_blocks-1):
            self.resblock2 += [ResidualBlock(in_channels, out_channels, drop_rate)]
            self.upsample += [UpSample(in_channels, out_channels, drop_rate)]
            in_channels = out_channels
            out_channels //= 2

    def forward(self, x):

        concat = self.resblock1[0](x)
        self.concats.append(concat)

        for i in range(1, self.num_residual_blocks):
            x = self.max_pool(concat)
            concat = self.resblock1[i](x)
            self.concats.append(concat)

        self.concats.pop()
        self.concats.reverse()

        x = concat
        for i in range(self.num_residual_blocks-1):
            y = self.upsample[i](x)
            result = torch.cat((self.concats[i], y), 1)
            x = self.resblock2[i](result)
        
        output = self.deconv(x)
        return softmax(output, dim=1)


def uNet_test():
    """for test
    """
    unet = uNet(class_num=2)
    unet.apply(weights_init_normal)
    input = torch.ones(3, 1, 240, 240)
    # input = torch.randn(3, 1, 240, 240)
    output = unet(input)
    print(output.size())
    print(output)


if __name__ == '__main__':
    uNet_test()