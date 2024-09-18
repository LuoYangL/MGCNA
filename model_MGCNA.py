import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, windowgraphconv_h1,windowgraphconv_h2,windowgraphconv_h3,windowgraphconv,windowgraphconv_h1_only
import torch
import math
import numpy as np
from torch_geometric.nn import GCNConv,global_mean_pool, BatchNorm
from CHBcreateA import generatePearsonA
class GCNcovmBMABC(nn.Module): #加入注意力机制SeNet的部分
    def __init__(self, nfeat, nhid1,nhid2, num_layers, nclass,nodenumber, num_heads,embed_dim):
        super(GCNcovmBMABC, self).__init__()
        self.temp_conv1 = nn.Sequential(
            nn.Conv1d(22, 22, kernel_size=2, stride=2,groups=22),

            nn.Conv1d(22, 22, kernel_size=2, stride=2,groups=22)
        )
        self.temp_conv2 = nn.Sequential(

            nn.Conv2d(22, 22, kernel_size=(1, 3), stride=(1, 1), padding=(2, 0)),
            nn.BatchNorm2d(22),
            nn.GELU(),
        )
        self.mlp=MLP(nfeat,64)

        self.windowgraph=windowgraphconv(64, nhid1,nhid2, num_layers, nclass,nodenumber, num_heads)

        self.attention=Attention(embed_dim, num_heads=num_heads)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.classifier=Classifier(feature_size=8)

        self.mu_proj1 = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, 2, padding=1),#,groups=embed_dim
            nn.ReLU(inplace=True),)
        self.mu_proj2 = nn.Sequential(
            #nn.Conv2d(32, 16, 1, 1, 0, bias=False),  # don't need bias
            #nn.BatchNorm2d(16),
            nn.Conv2d(32, 8, kernel_size=(5, 3), stride=(1, 1), padding=(2, 0)),
            nn.BatchNorm2d(8),
            nn.GELU(),)
        self.mu_proj3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten()
        )

    def forward(self, x, indices,values,ratio_corr,window,PearsonMatrix):
        size=int(x.size(2)/window)

        x1 = self.temp_conv1(x)

        h1=self.windowgraph(x1, indices,values,ratio_corr,window,PearsonMatrix)
        h=self.attention(h1)+h1

        mu = self.mu_proj1(h)  # (128,32,1,1)
        mu = self.mu_proj2(mu)
        mu = self.mu_proj3(mu)
        self.mu = mu
        logits = self.classifier(mu)
        probs = torch.sigmoid(logits)
        return probs