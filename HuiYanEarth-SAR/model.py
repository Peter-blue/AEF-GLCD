import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HuiYanSARGenerator(nn.Module):
    """
    Optical/SAR translation generator used by MTCDN replacement flow.
    Input:  [B, 3, H, W]
    Output: [B, 3, H, W], range [-1, 1]
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 3, n_residual_blocks: int = 9, base_channels: int = 64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, base_channels, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]

        in_ch = base_channels
        out_ch = in_ch * 2
        for _ in range(2):
            model.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)

        for _ in range(n_residual_blocks):
            model.append(ResidualBlock(in_ch))

        for _ in range(2):
            out_ch = in_ch // 2
            model.extend(
                [
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch

        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, output_nc, kernel_size=7, stride=1, padding=0),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
