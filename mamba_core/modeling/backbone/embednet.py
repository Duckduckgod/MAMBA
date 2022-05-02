import torch.nn.functional as F
from torch import nn


class EmbedNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_conv1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.embed_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.embed_conv3 = nn.Conv2d(512, 2048, kernel_size=1, stride=1)

        for _l in [
            self.embed_conv1,
            self.embed_conv2,
            self.embed_conv3,
        ]:
            nn.init.kaiming_uniform_(_l.weight, a=1)
            nn.init.zeros_(_l.bias)

    def forward(self, x):
        x = F.relu(self.embed_conv1(x))
        x = F.relu(self.embed_conv2(x))
        x = self.embed_conv3(x)

        return x


def build_embednet(cfg):
    return EmbedNet(cfg)
