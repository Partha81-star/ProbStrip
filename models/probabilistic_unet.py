import torch
import torch.nn as nn
from models.strip_conv import StripConvBlock


class ProbabilisticUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256],
        strip_kernel_size=7,
        dropout_prob=0.2,
    ):
        super(ProbabilisticUNet, self).__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        curr_channels = in_channels

        for feature in features:
            self.encoders.append(
                StripConvBlock(
                    curr_channels,
                    feature,
                    kernel_size=strip_kernel_size,
                    dropout_prob=dropout_prob,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_channels = feature

        bottleneck_channels = features[-1] * 2
        self.bottleneck = StripConvBlock(
            features[-1],
            bottleneck_channels,
            kernel_size=strip_kernel_size,
            dropout_prob=dropout_prob,
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        curr_channels = bottleneck_channels

        for feature in reversed(features):
            self.upconvs.append(
                nn.ConvTranspose2d(
                    curr_channels, feature, kernel_size=2, stride=2
                )
            )
            self.decoders.append(
                StripConvBlock(
                    feature * 2,
                    feature,
                    kernel_size=strip_kernel_size,
                    dropout_prob=dropout_prob,
                )
            )
            curr_channels = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    def forward(self, x):
        skip_connections = []
        out = x

        for encoder, pool in zip(self.encoders, self.pools):
            out = encoder(out)
            skip_connections.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.upconvs)):
            out = self.upconvs[i](out)
            skip = skip_connections[i]
            if out.shape != skip.shape:
                out = torch.nn.functional.interpolate(
                    out, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            out = torch.cat([skip, out], dim=1)
            out = self.decoders[i](out)

        return self.final_conv(out)
