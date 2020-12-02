# Copyright (C) 2020, Jiameng Fan <jmfan@bu.edu>
#
# This program is licenced under the MIT License,
# contained in the LICENCE file in this directory.


import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
