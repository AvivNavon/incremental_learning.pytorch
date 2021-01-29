import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math

# class ResNet18Pytorch(nn.Module):
#     """Linear weights, e.g. \sum_j \alpha_j * l_j
#     """
#     def __init__(self, dims=(512, 100), args=None, clf=True, pretrained=True):
#         super().__init__()
#
#         self.args = args
#
#         # pretrained feature extractor
#         self.FE = models.resnet18(pretrained=pretrained)
#         self.FE.fc = nn.Linear(512, dims[0])
#
#         self.layers = []
#         for i in range(0, len(dims) - 2):
#             self.layers.append(nn.Linear(dims[i], dims[i + 1]))
#             if self.args.activation[i].lower() == 'relu':
#                 self.layers.append(nn.ReLU())
#             elif self.args.activation[i].lower() == 'leakyrelu':
#                 self.layers.append(nn.LeakyReLU(negative_slope=0.01))
#             elif self.args.activation[i].lower() == 'linear':
#                 continue
#
#             if self.args.use_BN == True:
#                 self.layers.append(nn.BatchNorm1d(dims[i+1]))
#                 # dropout create stochasisety in the output and may cause large differences between Zs
#                 # self.layers.append(nn.Dropout(dropout))
#
#         self.fast_adapt_layers = nn.Sequential(*self.layers) if len(self.layers) > 0 else nn.Identity()
#         if clf:
#             self.clf_layer = nn.Linear(dims[-2], dims[-1])
#
#     def forward(self, x, classify=True):
#         features = self.fast_adapt_layers(self.FE(x))
#         if classify:
#             return self.clf_layer(features)
#         return features
#
#     def forward_basic_features(self, x):
#         return self.FE(x)  # z's
#
#     def forward_fast_features(self, features, skip=True):
#         fast_features = self.fast_adapt_layers(features) + features if skip \
#             else self.fast_adapt_layers(features)
#         return fast_features
#
#     def forward_basic_fast_features(self, x, skip=True):
#         features = self.FE(x)  # z's
#         fast_features = self.fast_adapt_layers(features) + features if skip \
#             else self.fast_adapt_layers(features)
#         return features, fast_features
#
#     def forward_classification(self, x, skip=True):
#         features = self.FE(x)  # z's
#         fast_features = self.fast_adapt_layers(features) + features if skip \
#             else self.fast_adapt_layers(features)
#         return self.clf_layer(fast_features)
#
#
# class ResNet18Pytorch_V2(ResNet18Pytorch):
#     def __init__(self, dims=(512, 100), args=None):
#         super().__init__(dims, args, clf=False)
#         # fast adapt layers learnable weight
#         self.alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
#         self.clf_layer = nn.Linear(dims[0], dims[-1])
#
#     def forward(self, x, classify=True):
#         features = self.FE(x)
#         if classify:
#             return self.clf_layer(features)
#         return features
#
#     def forward_skip(self, features):
#         return self.fast_adapt_layers(features) * self.alpha + features
#
#     def forward_fast_features(self, features, skip=True):
#         fast_features = self.forward_skip(features) if skip else self.fast_adapt_layers(features)
#         return fast_features
#
#     def forward_basic_fast_features(self, x, skip=True):
#         features = self.FE(x)  # z's
#         fast_features = self.forward_skip(features) if skip else self.fast_adapt_layers(features)
#         return features, fast_features
#
#     def forward_classification(self, x, skip=True):
#         features = self.FE(x)  # z's
#         fast_features = self.forward_skip(features) if skip else self.fast_adapt_layers(features)
#         return self.clf_layer(fast_features)


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, final_spatial_size=11):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.out_dim = 512

        # trunk = [self.conv1, self.bn1, nn.ReLU(), self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AvgPool2d(final_spatial_size)
        # trunk.append(avgpool)
        # self.flatten = Flatten()

        # self.trunk = nn.Sequential(*trunk)

        # self.linear = nn.Linear(512*block.expansion)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x, classify=True):
    #     # out = F.relu(self.bn1(self.conv1(x)))
    #     # out = self.layer1(out)
    #     # out = self.layer2(out)
    #     # out = self.layer3(out)
    #     # out = self.layer4(out)
    #     # out = F.avg_pool2d(out, out.shape[-1])
    #     # out = out.view(out.size(0), -1)
    #     out = self.trunk(x)
    #     if classify:
    #         out = self.linear(out)
    #     return out
    #

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    @property
    def last_block(self):
        return self.layer4

    @property
    def last_conv(self):
        return self.layer4[-1].conv2

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        raw_features = self.end_features(x_4)
        features = self.end_features(F.relu(x_4, inplace=False))

        return {
            "raw_features": raw_features,
            "features": features,
            "attention": [x_1, x_2, x_3, x_4]
        }

    def end_features(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    # def end_relu(self, x):
    #     if hasattr(self, "last_relu") and self.last_relu:
    #         return F.relu(x)
    #     return x


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    from torchsummary import summary
    import torch

    model = ResNet18()
    summary(model, input_size=(3, 84, 84), device='cpu')
    x = torch.empty((1, 3, 84, 84))
    out = model(x)
    print(out['features'].shape)
