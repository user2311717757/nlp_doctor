#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Zodiac
#Version  : 1.0
#Filename : model.py
from __future__ import print_function

import torch
from torch import nn

class Attacker(nn.Module):
    def __init__(self, repr_dim, hiddens=256, n_classes=2):
        super().__init__()

        self.proj = nn.Linear(repr_dim, hiddens)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(hiddens, n_classes)

    def forward(self, inputs):
        return self.out_proj(self.relu(self.proj(inputs)))


class Attackers(nn.Module):
    def __init__(self, repr_dim, n_attack, hiddens=256, n_classes=2):
        super().__init__()

        self.attackers = nn.ModuleList([Attacker(repr_dim, hiddens, n_classes) for _ in range(n_attack)])

        self.xentropy = nn.CrossEntropyLoss()

    def forward(self, input, outputs):
        loss = 0.
        preds = []

        for i, output in enumerate(outputs):
            logits = self.attackers[i](input)
            preds.append(torch.argmax(logits, dim=-1))
            loss += self.xentropy(logits, output)

        return loss, preds
