import torch
import torch.nn as nn


class StripConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(StripConv2d, self).__init__()
        padding = kernel_size // 2
        self.conv_h = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            bias=False,
        )
        self.conv_v = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            bias=False,
        )
        self.fusion = nn.Conv2d(
            out_channels * 2, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h = self.conv_h(x)
        x_v = self.conv_v(x)
        x_cat = torch.cat([x_h, x_v], dim=1)
        out = self.fusion(x_cat)
        out = self.bn(out)
        return self.act(out)


class MultiScaleStripConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super(MultiScaleStripConv2d, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.h_convs = nn.ModuleList()
        self.v_convs = nn.ModuleList()

        for k in kernel_sizes:
            pad_h = (0, k // 2)
            pad_v = (k // 2, 0)
            self.h_convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, k),
                    padding=pad_h,
                    bias=False,
                )
            )
            self.v_convs.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(k, 1),
                    padding=pad_v,
                    bias=False,
                )
            )

        num_branches = len(kernel_sizes) * 2
        self.fusion = nn.Conv2d(
            out_channels * num_branches, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        outputs = []
        for h_conv, v_conv in zip(self.h_convs, self.v_convs):
            outputs.append(h_conv(x))
            outputs.append(v_conv(x))
        cat_out = torch.cat(outputs, dim=1)
        fused = self.fusion(cat_out)
        out = self.bn(fused)
        return self.act(out)


class StripConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        dropout_prob=0.2,
        use_residual=True,
    ):
        super(StripConvBlock, self).__init__()
        self.use_residual = use_residual
        self.strip_conv1 = StripConv2d(
            in_channels, out_channels, kernel_size=kernel_size
        )
        self.strip_conv2 = StripConv2d(
            out_channels, out_channels, kernel_size=kernel_size
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)
        if in_channels != out_channels and use_residual:
            self.res_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.res_conv = None

    def forward(self, x):
        res = x
        out = self.strip_conv1(x)
        out = self.dropout(out)
        out = self.strip_conv2(out)
        if self.use_residual:
            if self.res_conv is not None:
                res = self.res_conv(res)
            out = out + res
        return out
