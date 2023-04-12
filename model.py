from models.base_models import *
from layers.hyp_layers import *
from geoopt import ManifoldParameter
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.modules.module import Module
from torch import FloatTensor
from torch.nn.parameter import Parameter
import manifolds

class HypClassifier(nn.Module):
    """
    Hyperbolic Classifier
    """

    def __init__(self, args):
        super(HypClassifier, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim * 2
        self.output_dim = args.num_classes
        self.use_bias = args.bias
        self.cls = ManifoldParameter(self.manifold.random_normal((args.num_classes, self.input_dim), std=1./math.sqrt(self.input_dim)), manifold=self.manifold)
        if args.bias:
            self.bias = nn.Parameter(torch.zeros(args.num_classes))

    def forward(self, x):
        return (2 + 2 * self.manifold.cinner(x, self.cls)) + self.bias


class DistanceAdj(Module):

    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen, args):
        # To support batch operations
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        self.dist = torch.from_numpy(squareform(dist)).to(args.device)
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to(args.device)
        return self.dist



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.manifold = getattr(manifolds, args.manifold)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1

        self.disAdj = DistanceAdj()

        self.conv1d1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)

        self.HFSGCN = FHyperGCN(args)
        self.HTRGCN = FHyperGCN(args)

        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.HyperCLS = HypClassifier(args)
        self.args = args


    def forward(self, inputs, seq_len):

        xv = inputs[:,:,:1024]
        xa = inputs[:,:,1024:]
        xv = xv.permute(0, 2, 1)  # for conv1d
        xv = self.relu(self.conv1d1(xv))
        xv = self.dropout(xv)
        xv = self.relu(self.conv1d2(xv))
        xv = self.dropout(xv)
        xv = xv.permute(0, 2, 1)  # b*t*c

        x = torch.cat((xv, xa), -1)

        disadj = self.disAdj(x.shape[0], x.shape[1], self.args).to(x.device)
        proj_x = self.expm(x)
        adj = self.adj(proj_x, seq_len)

        x1 = self.relu(self.HFSGCN.encode(proj_x, adj))
        x1 = self.dropout(x1)
        x2 = self.relu(self.HTRGCN.encode(proj_x, disadj))
        x2 = self.dropout(x2)


        out_x = torch.cat((x1, x2), 2)
        frame_prob = self.HyperCLS(out_x)
        mil_logits = self.clas(frame_prob, seq_len)

        return mil_logits, frame_prob

    def expm(self, x):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, :, 0:1], x], dim=-1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
            return x
        else:
            return x

    def adj(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        x2 = torch.exp(-x2)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.8, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def clas(self, logits, seq_len):
        logits = logits.squeeze()
        instance_logits = torch.zeros(0).to(logits.device)  # tensor([])
        for i in range(logits.shape[0]):
            if seq_len is None:
                tmp = torch.mean(logits[i]).view(1)
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                                largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric
        '''
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        temp = x @ diag
        xy_inner = -(temp @ y.transpose(-1, -2))
        xy_inner_ = F.threshold(xy_inner, 1, 1)
        sqrt_k = k**0.5
        dist = sqrt_k * self.arccosh(xy_inner_ / k)
        dist = torch.clamp(dist, min=self.eps[x.dtype], max=200)
        return dist

    def arccosh(self, x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))

