# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Config.TextDCNN import *

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, config.embed, (k, config.embed),stride=1,padding=(k // 2,0)) for k in config.filter_sizes])
        self.convs2 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed),padding=(k // 2,0)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        inn_fea=config.num_filters * len(config.filter_sizes)
        self.linear1=nn.Linear(inn_fea,inn_fea//2)
        self.linear2 = nn.Linear(inn_fea//2, config.num_classes)


    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        l=[]
        for conv in self.convs1:
            l.append(torch.transpose(F.relu(conv(out)).squeeze(3),1,2))
        out = l
        l=[]
        for conv ,output in zip(self.convs2,out):
            l.append(F.relu(conv(output.unsqueeze(1))).squeeze(3))
        out = l
        l=[]
        for i in out:
            l.append(F.max_pool1d(i,kernel_size=i.size(2)).squeeze(2))
        out = torch.cat(l, 1)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.linear2(F.relu(out))
        return out
