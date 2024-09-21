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
class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):  # 128 2 12 0
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qkv = nn.Conv2d(embed_dim, 2 * embed_dim + num_heads, 1, 1, bias=bias)
        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  #
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        b, c, t, n = x.shape  # (b,c,t,n)
        qkv = self.to_qkv(x).reshape(b, self.num_heads, -1, t, n)
        i, k, v = torch.split(
            qkv, split_size_or_sections=[1, self.head_dim, self.head_dim], dim=2
        )
        scores = self.attend(i)  # (b, h, 1, t, n)
        attn_vector = k * scores  # (b, h, c//h, t, n)
        attn_vector = torch.sum(attn_vector, dim=-1, keepdim=True)  # (b, h, c//h, t, 1)
        out = (F.relu(v) * attn_vector.expand_as(v)).reshape(b, -1, t, n)  # (b, c, t, n)
        out = self.to_out(out)
        return out

class windowgraphconv(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, num_layers, nclass, nodenumber, num_heads):
        super(windowgraphconv, self).__init__()

        self.num_layers = num_layers
        self.conv1_0 = GraphConvolution(nfeat, nhid1, nodenumber)
        self.batchnorm1_0 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1_0 = nn.Dropout(p=0.5)
        self.conv1_1 = GraphConvolution(nhid1, nhid2, nodenumber)
        self.batchnorm1_1 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1_1 = nn.Dropout(p=0.5)
        self.conv1_2 = GraphConvolution(nhid2, int(nhid2 / 2), nodenumber)
        self.batchnorm1_2 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1_2 = nn.Dropout(p=0.5)
        self.conv1_3 = GraphConvolution(int(nhid2 / 2), nclass, nodenumber)
        self.batchnorm1_3 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout1_3 = nn.Dropout(p=0.5)

        self.conv2_0 = GCNConv(nfeat, nhid1, improved=True, cached=True, normalize=False)
        self.batchnorm2_0 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2_0 = nn.Dropout(p=0.5)
        self.conv2_1 = GCNConv(nhid1, nhid2, improved=True, cached=True, normalize=False)
        self.batchnorm2_1 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2_1 = nn.Dropout(p=0.5)
        self.conv2_2 = GCNConv(nhid2, int(nhid2 / 2), improved=True, cached=True, normalize=False)
        self.batchnorm2_2 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2_2 = nn.Dropout(p=0.5)
        self.conv2_3 = GCNConv(int(nhid2 / 2), nclass, improved=True, cached=True, normalize=False)
        self.batchnorm2_3 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout2_3 = nn.Dropout(p=0.5)

        self.conv3_0 = GraphConv_MultiA(nfeat, nhid1)
        self.batchnorm3_0 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)  # nn.Dropout(p=0.5)
        self.dropout3_0 = nn.Dropout(p=0.5)
        self.conv3_1 = GraphConv_MultiA(nhid1, nhid2)
        self.batchnorm3_1 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)  # nn.Dropout(p=0.5)
        self.dropout3_1 = nn.Dropout(p=0.5)
        self.conv3_2 = GraphConv_MultiA(nhid2, int(nhid2 / 2))
        self.batchnorm3_2 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)  # nn.Dropout(p=0.5)
        self.dropout3_2 = nn.Dropout(p=0.5)
        self.conv3_3 = GraphConv_MultiA(int(nhid2 / 2), nclass)
        self.batchnorm3_3 = BatchNorm(22, eps=1e-05, momentum=0.1, affine=True,
                                      track_running_stats=True)  # nn.Dropout(p=0.5)
        self.dropout3_3 = nn.Dropout(p=0.5)

    def forward(self, x, indices, values, ratio_corr, window, PearsonMatrix):

        '''
        x1 = self.dropout1_0(F.leaky_relu(self.conv1_0(x)))
        layer_out = x1
        for i in range(1, self.num_layers):
            conv = getattr(self, 'conv1_{}'.format(i))
            dropout = getattr(self, 'dropout1_{}'.format(i))
            x1 = dropout(F.leaky_relu(conv(x1)))
        h1 =x1
        '''
        x1 = F.leaky_relu(self.batchnorm1_0(self.conv1_0(x)))
        layer_out = x1
        for i in range(1, self.num_layers):
            conv = getattr(self, 'conv1_{}'.format(i))
            batchnorm = getattr(self, 'batchnorm1_{}'.format(i))
            dropout = getattr(self, 'dropout1_{}'.format(i))

            x1 = F.leaky_relu(batchnorm(conv(x1)))
            #layer_out = torch.cat((x1, layer_out), dim=-1)
        #h1 = x1
        h1 = torch.cat((layer_out, x1), -1)
        '''
            # x1 = x[:, :, :,np.newaxis]
            # layer_out.append(x)
            ##layer_out = torch.cat((layer_out, x1), -1)

        x1 = self.dropout1_0(F.leaky_relu(self.conv1_0(x)))
        layer_out = x1
        # layer_out = layer_out[:,:, :,np.newaxis]
        for i in range(1, self.num_layers):
            conv = getattr(self, 'conv1_{}'.format(i))
            dropout = getattr(self, 'dropout1_{}'.format(i))
            x1 = dropout(F.leaky_relu(conv(x1)))
            # x1 = x[:, :, :,np.newaxis]
            # layer_out.append(x)
            layer_out = torch.cat((layer_out, x1), -1)
            # layer_out[:,:,:,i:i+1]=x1[:,:,:,0]

        h1 = layer_out
        '''
        # h1 = h1 * self.channelattention1(h1)
        # h1 = h1 * self.spatialattention1(h1)
        # h1=self.mlp1(h1)
        # h1= h1.view(h1.size(0), -1)
        # h1 = self.mlp1_1(h1)
        # h1 = self.mlp1_2(h1)
        #########对第二个分支进行计算
        '''
        x2 = self.conv2_0(x, indices, values)
        # x2 = self.batchnorm2_0(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.01)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        layer_out = x2
        # layer_out = layer_out[:,:, :,np.newaxis]
        for i in range(1, self.num_layers):
            conv2 = getattr(self, 'conv2_{}'.format(i))
            # dropout2 = getattr(self, 'batchnorm2_{}'.format(i))
            x2 = conv2(x2, indices, values)
            # x2=dropout2(x2)
            x2 = F.leaky_relu(x2, negative_slope=0.01)
            x2 = F.dropout(x2, p=0.5, training=self.training)
        h2=x2
        '''
        x2 = F.leaky_relu(self.batchnorm2_0(self.conv2_0(x, indices, values)))
        layer_out = x2
        for i in range(1, self.num_layers):
            conv = getattr(self, 'conv2_{}'.format(i))
            batchnorm = getattr(self, 'batchnorm2_{}'.format(i))
            dropout = getattr(self, 'dropout2_{}'.format(i))
            x2 = F.leaky_relu(batchnorm(conv(x2, indices, values)))
            #layer_out = torch.cat((x2, layer_out), dim=-1)
        #h2 = x2
        # x1 = x[:, :, :,np.newaxis]
        # layer_out.append(x)

        # layer_out = torch.cat((layer_out, x2), -1)
        # layer_out[:,:,:,i:i+1]=x1[:,:,:,0]

        h2 =torch.cat((layer_out, x2), -1)
        '''
        x2 = self.conv2_0(x, indices, values)
        # x2 = self.batchnorm2_0(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.01)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        layer_out = x2
        for i in range(1, self.num_layers):
            conv2 = getattr(self, 'conv2_{}'.format(i))
            # dropout2 = getattr(self, 'batchnorm2_{}'.format(i))
            x2 = conv2(x2, indices, values)
            # x2=dropout2(x2)
            x2 = F.leaky_relu(x2, negative_slope=0.01)
            x2 = F.dropout(x2, p=0.5, training=self.training)
            # x1 = x[:, :, :,np.newaxis]
            # layer_out.append(x)

            layer_out = torch.cat((layer_out, x2), -1)
            # layer_out[:,:,:,i:i+1]=x1[:,:,:,0]

        h2 = layer_out

        # h2 = h2 * self.channelattention2(h2)
        # h2 = h2 * self.spatialattention2(h2)
        '''
        #########第三种构图方式
        # 计算相关构图
        '''
        #PearsonMatrix = generatePearsonA(x, ratio_corr).cuda()
        x3 = self.dropout3_0(self.conv3_0(x, PearsonMatrix))
        #print('1')
        layer_out = x3
        for i in range(1, self.num_layers):
            conv2 = getattr(self, 'conv3_{}'.format(i))
            # dropout2 = getattr(self, 'batchnorm2_{}'.format(i))
            x3 = conv2(x3, PearsonMatrix)
            # x2=dropout2(x2)
            x3 = F.leaky_relu(x3, negative_slope=0.01)
            x3 = F.dropout(x3, p=0.5, training=self.training)
            # x1 = x[:, :, :,np.newaxis]
            # layer_out.append(x)

            #layer_out = torch.cat((layer_out, x3), -1)
        h3=x3
        '''
        x3 = F.leaky_relu(self.batchnorm3_0(self.conv3_0(x, PearsonMatrix)))
        layer_out = x3
        for i in range(1, self.num_layers):
            conv = getattr(self, 'conv3_{}'.format(i))
            batchnorm = getattr(self, 'batchnorm3_{}'.format(i))
            dropout = getattr(self, 'dropout3_{}'.format(i))
            x3 = F.leaky_relu(batchnorm(conv(x3, PearsonMatrix)))
            #layer_out = x3
            #layer_out = torch.cat((x3, layer_out), dim=-1)
        #h3 = x3
        h3 = torch.cat((layer_out, x3), -1)
        '''
        x3 = self.dropout1_0(self.conv3_0(x, PearsonMatrix))
        layer_out = x3
        # layer_out = layer_out[:,:, :,np.newaxis]
        for i in range(1, self.num_layers):
            conv2 = getattr(self, 'conv2_{}'.format(i))
            # dropout2 = getattr(self, 'batchnorm2_{}'.format(i))
            x3 = conv2(x3, indices, values)
            # x2=dropout2(x2)
            x3 = F.leaky_relu(x3, negative_slope=0.01)
            x3 = F.dropout(x3, p=0.5, training=self.training)
            # x1 = x[:, :, :,np.newaxis]
            # layer_out.append(x)

            layer_out = torch.cat((layer_out, x3), -1)

        h3 = layer_out
        '''
        ########multihead attention
        h1 = h1.unsqueeze(1)
        h2 = h2.unsqueeze(1)
        h3 = h3.unsqueeze(1)
        Cat = torch.cat((h1, h2), dim=1)
        Cat = torch.cat((Cat, h3), dim=1)
        return Cat
