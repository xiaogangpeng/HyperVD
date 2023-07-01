import argparse

parser = argparse.ArgumentParser(description='HyperVD')
parser.add_argument('--rgb-list', default='list/rgb.list', help='list of rgb features ')
parser.add_argument('--flow-list', default='list/flow.list', help='list of flow features')
parser.add_argument('--audio-list', default='list/audio.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='list/rgb_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='list/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='list/audio_test.list', help='list of test audio features')
parser.add_argument('--dataset-name', default='XD-Violence', help='dataset to train on XD-Violence')
parser.add_argument('--gt', default='list/gt.npy', help='file of ground truth ')


parser.add_argument('--cuda', default=0, help='which cuda device to use (-1 for cpu training)')
parser.add_argument('--modality', default='MIX2', help='the type of the input, AUDIO,RGB,FLOW, MIX1, MIX2, '
                                                          'or MIX3, MIX_ALL')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate (default: 0.0005)')
parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='pxg', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--dropout', default=0.6, help='x x')


parser.add_argument('--model', default='HyboNet', help='which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN, HyboNet]')
parser.add_argument('--manifold', default='Lorentz', help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]')
parser.add_argument('--c', default=None, help='hyperbolic radius, set to None for trainable curvature')
parser.add_argument('--num-layers', default=2, help='layers of hgcn')
parser.add_argument('--act', default='leaky_relu', help='which activation function to use (or None for no activation)')
parser.add_argument('--dim', default=32, help='embedding dimension')
parser.add_argument('--bias', default=1, help='whether to use bias (1) or not (0)')
parser.add_argument('--use-att', default=0, help='whether to use hyperbolic attention or not')
parser.add_argument('--local-agg', default=0, help='whether to local tangent space aggregation or not')
parser.add_argument('--tie_weight', default=True, help='whether to tie transformation matrices')
parser.add_argument('--feat-dim', type=int, default=256, help='input size of feature for HGCN (default: 2048)')

parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--max-seqlen', type=int, default=200, help='maximum sequence length during training')
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train (default: 50)')
parser.add_argument('--seed', type=int, default=9, help='Random Initiation (default: 9)')
