"""Attribute encoder used by both entity types."""
from torch import nn


def mlp(a, h, b, dropout):
    return nn.Sequential(nn.Linear(a, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, b))
