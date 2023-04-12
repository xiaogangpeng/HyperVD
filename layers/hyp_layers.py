"""Hyperbolic layers."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

import geoopt as gt
import torch.nn
import torch.nn.functional
import geoopt.manifolds.stereographic.math as pmath

MIN_NORM = 1e-15


def get_dim_act_curv(args):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))

    # if args.task in ['lp', 'rec']:
    #     dims += [args.dim]
    #     acts += [act]
    #     n_curvatures = args.num_layers
    # else:
    #     n_curvatures = args.num_layers - 1
    n_curvatures = args.num_layers - 1


    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.device) for curv in curvatures]
    return dims, acts, curvatures


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
        output = h, adj
        return output


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            # print(f"adj: {adj.shape}   x_tangent :{x_tangent.shape}")
            # support_t = torch.spmm(adj, x_tangent)
            support_t = torch.matmul(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )



class MobiusLinear(nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = gt.PoincareBall(c=c)
        if self.bias is not None:
            if hyperbolic_bias:
                self.bias = gt.ManifoldParameter(self.bias, manifold=self.ball)
                with torch.no_grad():
                    self.bias.set_(pmath.expmap0(self.bias.normal_() * 1e-3, k=self.ball.k))
        with torch.no_grad():
            fin, fout = self.weight.size()
            k = (6 / (fin + fout)) ** 0.5  # xavier uniform
            self.weight.uniform_(-k, k)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.ball.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += ", hyperbolic_input={}".format(self.hyperbolic_input)
        if self.bias is not None:
            info += ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


# class MobiusConcat(nn.Module):
#     def __init__(self, output_dim, input_dims, second_input_dim=None, third_input_dim=None, nonlin=None):
#         super(MobiusConcat, self).__init__()
#         b_input_dims = second_input_dim if second_input_dim is not None else input_dims
#
#         self.lin_a = MobiusLinear(input_dims, output_dim, bias=False, nonlin=nonlin)
#         self.lin_b = MobiusLinear(b_input_dims, output_dim, bias=False, nonlin=nonlin)
#
#         if third_input_dim:
#             self.lin_c = MobiusLinear(third_input_dim, output_dim, bias=False, nonlin=nonlin)
#
#         self.ball = gt.PoincareBall()
#         b = torch.randn(output_dim) * 1e-5
#         self.bias = gt.ManifoldParameter(pmath.expmap0(b, k=self.ball.k), manifold=self.ball)
#
#     def forward(self, input_a, input_b, third_input=None):
#         """
#         :param input_a: batch x * x input_dim_a
#         :param input_b: batch x * x input_dim_b
#         :return: batch x output_dim
#         """
#         out_a = self.lin_a(input_a)
#         out_b = self.lin_b(input_b)
#         out_total = self.add(out_a, out_b)
#
#         if third_input is not None:
#             out_c = self.lin_c(third_input)
#             out_total = self.add(out_total, out_c)
#
#         out_total = self.add(out_total, self.bias)
#         return out_total
#
#     def add(self, a, b):
#         out = pmath.mobius_add(a, b, k=self.ball.k)
#         return pmath.project(out, k=self.ball.k)





class MobiusMLR(nn.Module):
    """
    Multinomial logistic regression in the Poincare Ball
    It is based on formulating logits as distances to margin hyperplanes.
    In Euclidean space, hyperplanes can be specified with a point of origin
    and a normal vector. The analogous notion in hyperbolic space for a
    point $p \in \mathbb{D}^n$ and
    $a \in T_{p} \mathbb{D}^n \backslash \{0\}$ would be the union of all
    geodesics passing through $p$ and orthogonal to $a$. Given $K$ classes
    and $k \in \{1,...,K\}$, $p_k \in \mathbb{D}^n$,
    $a_k \in T_{p_k} \mathbb{D}^n \backslash \{0\}$, the formula for the
    hyperbolic MLR is:
    \begin{equation}
        p(y=k|x) f\left(\lambda_{p_k} \|a_k\| \operatorname{sinh}^{-1} \left(\frac{2 \langle -p_k \oplus x, a_k\rangle}
                {(1 - \| -p_k \oplus x \|^2)\|a_k\|} \right) \right)
    \end{equation}
    """

    def __init__(self, in_features, out_features, c=1.0):
        """
        :param in_features: number of dimensions of the input
        :param out_features: number of classes
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = gt.PoincareBall(c=c)
        points = torch.randn(out_features, in_features) * 1e-5
        points = pmath.expmap0(points, k=self.ball.k)
        self.p_k = gt.ManifoldParameter(points, manifold=self.ball)

        tangent = torch.Tensor(out_features, in_features)
        stdv = (6 / (out_features + in_features)) ** 0.5  # xavier uniform
        torch.nn.init.uniform_(tangent, -stdv, stdv)
        self.a_k = torch.nn.Parameter(tangent)

    def forward(self, input):
        """
        :param input: batch x space_dim: points (features) in the Poincaré ball
        :return: batch x classes: logit of probabilities for 'out_features' classes
        """
        input = input.unsqueeze(-2)     # batch x aux x space_dim
        distance, a_norm = self._dist2plane(x=input, p=self.p_k, a=self.a_k, c=self.ball.c, k=self.ball.k, signed=True)
        result = 2 * a_norm * distance
        return result

    def _dist2plane(self, x, a, p, c, k, keepdim: bool = False, signed: bool = False, dim: int = -1):
        """
        Taken from geoopt and corrected so it returns a_norm and this value does not have to be calculated twice
        """
        sqrt_c = c ** 0.5
        minus_p_plus_x = pmath.mobius_add(-p, x, k=k, dim=dim)
        mpx_sqnorm = minus_p_plus_x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM)
        mpx_dot_a = (minus_p_plus_x * a).sum(dim=dim, keepdim=keepdim)
        if not signed:
            mpx_dot_a = mpx_dot_a.abs()
        a_norm = a.norm(dim=dim, keepdim=keepdim, p=2).clamp_min(MIN_NORM)
        num = 2 * sqrt_c * mpx_dot_a
        denom = (1 - c * mpx_sqnorm) * a_norm
        return pmath.arsinh(num / denom.clamp_min(MIN_NORM)) / sqrt_c, a_norm

    def extra_repr(self):
        return "in_features={in_features}, out_features={out_features}".format(**self.__dict__) + f" k={self.ball.k}"



class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, use_bias, dropout, act, use_att, local_agg, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)
        # self.lorentz_act = LorentzAct(manifold, c_in, c_out, act)
        # self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, input):
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        # h = self.lorentz_act.forward(h)
        # h = self.hyp_act.forward(h)
        output = h, adj
        return output



class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att, local_agg):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            # self.att = DenseAtt(in_features, dropout)
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        # x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                # x_local_tangent = []
                # # for i in range(x.size(0)):
                # #     x_local_tangent.append(self.manifold.logmap(x[i], x))
                # # x_local_tangent = torch.stack(x_local_tangent, dim=0)
                # x_local_tangent = self.manifold.clogmap(x, x)
                # # import pdb; pdb.set_trace()
                # adj_att = self.att(x, adj)
                # # att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                # support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                # output = self.manifold.expmap(x, support_t)
                # return output
                query = self.query_linear(x)
                key = self.key_linear(x)
                att_adj = 2 + 2 * self.manifold.cinner(query, key)
                att_adj = att_adj / self.scale + self.bias
                att_adj = torch.sigmoid(att_adj)
                att_adj = torch.mul(adj.to_dense(), att_adj)
                support_t = torch.matmul(att_adj, x)
            else:
                adj_att = self.att(x, adj)
                support_t = torch.matmul(adj_att, x)
        else:
            # support_t = torch.spmm(adj, x)
            support_t = torch.matmul(adj, x)
        # output = self.manifold.expmap0(support_t, c=self.c)
        denom = (-self.manifold.inner(None, support_t, keepdim=True))
        denom = denom.abs().clamp_min(1e-8).sqrt()
        output = support_t / denom
        return output

    def attention(self, x, adj):
        pass


class LorentzAct(Module):
    """
    Lorentz activation layer
    """
    def __init__(self, manifold, c_in, c_out, act):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.log_map_zero(x, c=self.c_in))
        xt = self.manifold.normalize_tangent_zero(xt, self.c_in)
        return self.manifold.exp_map_zero(xt, c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
                self.c_in, self.c_out
        )
