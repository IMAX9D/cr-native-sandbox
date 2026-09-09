"""Full-resolution spatial correction to the existing conditional position logits."""
import math
import torch
from torch import nn


class SpatialPositionHead(nn.Module):
    def __init__(self, input_channels, width, channels):
        super().__init__()
        # Coordinate planes preserve absolute arena position without downsampling.
        y, x = torch.meshgrid(torch.linspace(-1, 1, 32), torch.linspace(-1, 1, 18), indexing='ij')
        self.register_buffer('coordinates', torch.stack((x, y))[None], persistent=False)
        self.spatial = nn.Sequential(nn.Conv2d(input_channels+2, channels, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.query = nn.Sequential(nn.Linear(2*width, channels), nn.ReLU(), nn.Linear(channels, channels))
        # Start as exactly the original position head; learn a spatial correction.
        nn.init.zeros_(self.query[-1].weight)
        nn.init.zeros_(self.query[-1].bias)
        self.channels = channels

    def forward(self, grid, context, hand):
        spatial = self.spatial(torch.cat((grid, self.coordinates.to(grid).expand(len(grid), -1, -1, -1)), 1))
        query = self.query(torch.cat((context[:, None].expand_as(hand), hand), -1))
        # Shared map is computed once per observation, not once per hand slot.
        return torch.einsum('nsc,ncp->nsp', query, spatial.flatten(2))/math.sqrt(self.channels)
