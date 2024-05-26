import torch
from dataclasses import dataclass


def conv3(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv1d:
    """3x3 convolution with padding"""
    return torch.nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv1d:
    """1x1 convolution"""
    return torch.nn.Conv1d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


class Bottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1(inplanes, planes)
        self.bn1 = torch.nn.BatchNorm1d(planes)
        self.conv2 = conv3(planes, planes, stride)
        self.bn2 = torch.nn.BatchNorm1d(planes)
        self.conv3 = conv1(planes, planes * self.expansion)
        self.bn3 = torch.nn.BatchNorm1d(planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = torch.nn.Sequential(
                conv1(inplanes, planes * self.expansion, stride),
                torch.nn.BatchNorm1d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@dataclass
class ResNetArgs:
    planes: [int]
    blocks: [int]


class ResNet(torch.nn.Module):
    def __init__(self, args: ResNetArgs) -> None:
        super().__init__()

        self.inplanes = args.planes[0]
        self.conv1 = torch.nn.Conv1d(
            12, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = torch.nn.BatchNorm1d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_bottleneck(
            planes=args.planes[0], blocks=args.blocks[0]
        )
        self.layer2 = self._make_bottleneck(
            planes=args.planes[1], blocks=args.blocks[1], stride=2
        )
        self.layer3 = self._make_bottleneck(
            planes=args.planes[2], blocks=args.blocks[2], stride=2
        )
        self.layer4 = self._make_bottleneck(
            planes=args.planes[3], blocks=args.blocks[3], stride=2
        )
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_bottleneck(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> torch.nn.Sequential:
        layers = []
        layers.append(
            Bottleneck(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride
            )
        )

        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    inplanes=self.inplanes,
                    planes=planes,
                )
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.squeeze(2)

        return x
