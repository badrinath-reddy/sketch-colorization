from torch import nn
import torch


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpSampleBlock, self).__init__()
        layers = [nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):  # Add skip after the layer
        x = self.model(x)
        x = torch.cat((x, skip), dim=1)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(DownSampleBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.down1 = DownSampleBlock(3, 64, normalize=False)  # 128, 128, 64
        self.down2 = DownSampleBlock(64, 128)  # 64, 64, 128
        self.down3 = DownSampleBlock(128, 256)  # 32, 32, 256
        self.down4 = DownSampleBlock(256, 512)  # 16, 16, 512
        self.down5 = DownSampleBlock(512, 512)  # 8, 8, 512
        self.down6 = DownSampleBlock(512, 512)  # 4, 4, 512
        self.down7 = DownSampleBlock(512, 512)  # 2, 2, 512
        self.down8 = DownSampleBlock(512, 512)  # 1, 1, 512

        self.up1 = UpSampleBlock(512, 512, dropout=0.5)  # 2, 2, 1024
        self.up2 = UpSampleBlock(1024, 512, dropout=0.5)  # 4, 4, 1024
        self.up3 = UpSampleBlock(1024, 512, dropout=0.5)  # 8, 8, 1024
        self.up4 = UpSampleBlock(1024, 512)  # 16, 16, 1024
        self.up5 = UpSampleBlock(1024, 256)  # 32, 32, 512
        self.up6 = UpSampleBlock(512, 128)  # 64, 64, 256
        self.up7 = UpSampleBlock(256, 64)  # 128, 128, 128
        self.up8 = nn.Sequential(nn.ConvTranspose2d(
            128, 3, kernel_size=4, stride=2, padding=1), nn.Tanh())  # 256, 256, 3

    def forward(self, x):
        skip1 = self.down1(x)  # 128, 128, 64
        skip2 = self.down2(skip1)  # 64, 64, 128
        skip3 = self.down3(skip2)  # 32, 32, 256
        skip4 = self.down4(skip3)  # 16, 16, 512
        skip5 = self.down5(skip4)  # 8, 8, 512
        skip6 = self.down6(skip5)  # 4, 4, 512
        skip7 = self.down7(skip6)  # 2, 2, 512
        skip8 = self.down8(skip7)  # 1, 1, 512

        x = self.up1(skip8, skip7)  # 2, 2, 1024
        x = self.up2(x, skip6)  # 4, 4, 1024
        x = self.up3(x, skip5)  # 8, 8, 1024
        x = self.up4(x, skip4)  # 16, 16, 1024
        x = self.up5(x, skip3)  # 32, 32, 512
        x = self.up6(x, skip2)  # 64, 64, 256
        x = self.up7(x, skip1)  # 128, 128, 128
        x = self.up8(x)  # 256, 256, 3

        return x


class Discriminator(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.down1 = DiscriminatorBlock(6, 64, normalize=False)  # 128, 128, 64
        self.down2 = DiscriminatorBlock(64, 128)  # 64, 64, 128
        self.down3 = DiscriminatorBlock(128, 256)  # 32, 32, 256
        self.down4 = DiscriminatorBlock(256, 512)  # 16, 16, 512
        self.zero = nn.ZeroPad2d((1, 0, 1, 0))  # 18, 18, 512
        self.down5 = nn.Conv2d(512, 1, kernel_size=4,
                               stride=1, padding=1)  # 15, 15, 1

    def forward(self, img1, img2):
        x = torch.cat((img1, img2), dim=1)
        x = self.down1(x)  # 128, 128, 64
        x = self.down2(x)  # 64, 64, 128
        x = self.down3(x)  # 32, 32, 256
        x = self.down4(x)  # 16, 16, 512
        x = self.zero(x)
        x = self.down5(x)

        return x
