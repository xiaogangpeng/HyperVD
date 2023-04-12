"""Base model class."""

import torch
import torch.nn as nn
import manifolds
import models.encoders as encoders


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))

        self.manifold = getattr(manifolds, self.manifold_name)()

        self.encoder = getattr(encoders, args.model)(self.c, args)


    def encode(self, x, adj):

        h = self.encoder.encode(x, adj)
        return h


class FHyperGCN(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(FHyperGCN, self).__init__(args)

    def decode(self, h, adj):
        raise

