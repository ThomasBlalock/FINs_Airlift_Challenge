import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn.functional as F

# TODO: make a sorted list of keys to ensure the order of the input set

class FlexibleInputNetwork(nn.Module):

    def __init__(self, in_set):
        super(SelfProjection, self).__init__()
        """
        in_set: {
            'x1': embed dim (int),
            'x2': embed dim (int),
            ...
        }
        """
        self.in_set = in_set
        self.n = len(in_set)
        assert self.n>0, "Input set must have at least one element"


class SelfProjection(FlexibleInputNetwork):

    def __init__(self, in_set, config):
        super(SelfProjection, self).__init__(in_set)
        """
        config: {
            'non_linearity': relu default (nn.Module),
            'x1': {
                'W_Z': sqrt(aggregation dim) (int),
                d_ia: augmentation dim (int),
                'MLP_hidden_dim': 2*d_ia default (int),
                'FF_expansion_factor': 2 default (int),
                'ouput': True default (bool)
            },
            ...
        }
        """
        self.config = config
        if 'non_linearity' not in config.keys():
            config['non_linearity'] = nn.ReLU()
        for input_key in config.keys():
            if not isinstance(config[input_key], dict):
                continue
            if 'MLP_hidden_dim' not in config[input_key]:
                config[input_key]['MLP_hidden_dim'] = 2*config[input_key]['d_ia']
            if 'FF_expansion_factor' not in config[input_key]:
                config[input_key]['FF_expansion_factor'] = 2
            if 'output' not in config[input_key]:
                config[input_key]['output'] = True

        # Z = vec(||_i=1->N(X_i*W_i1).T * (X_i*W_i)2)
        self.W_Z = {}
        for input_key in in_set.keys():
            self.W_Z[input_key] = (nn.Parameter(torch.randn(in_set[input_key], config[input_key]['W_Z'])),
                                   nn.Parameter(torch.randn(in_set[input_key], config[input_key]['W_Z'])),)
        
        # Z el-of R^(SUM_i=1->n(d_i^2))
        self.Z_dim = sum([d_i**2 for d_i in in_set.values()])

        # A_i = mat(MLP(Z), d_i, d_ia)
        self.MLP_A = {}
        for input_key in in_set.keys():
            if not config[input_key]['output']:
                continue
            self.MLP_A[input_key] = nn.Sequential(
                nn.Linear(self.Z_dim, config['MLP_hidden_dim']),
                config['non_linearity'],
                nn.Linear(config['MLP_hidden_dim'], in_set[input_key]*config[input_key]['d_ia'])
            )

        # SP_i = FF_i(X_i||x_i*A_i)
        self.FF = {}
        for input_key in in_set.keys():
            if not config[input_key]['output']:
                continue
            FF_dim = in_set[input_key]+config[input_key]['d_ia']
            self.FF[input_key] = nn.Sequential(
                nn.Linear(FF_dim, FF_dim*config[input_key]['FF_expansion_factor']),
                config['non_linearity'],
                nn.Linear(FF_dim*config[input_key]['FF_expansion_factor'], in_set[input_key])
            )

    def forward(self, X):

        # Z = vec(||_i=1->N(X_i*W_i1).T * (X_i*W_i)2)
        Z = torch.cat([torch.matmul(X[input_key], W[0]).T * torch.matmul(X[input_key], W[1])
                       for input_key, W in self.W_Z.items()], dim=1)
        Z = torch.reshape(Z, (-1, self.Z_dim))
        Z = self.config['non_linearity'](Z)

        # A_i = mat(MLP(Z), d_i, d_ia)
        A = {}
        for input_key in self.MLP_A.keys():
            if not self.config[input_key]['output']:
                continue
            A[input_key] = torch.reshape(
                self.MLP_A[input_key](Z), (-1, self.in_set[input_key], self.config[input_key]['d_ia']))
        
        # out_set_i = FF_i(X_i||x_i*A_i)
        Y = {}
        for input_key in self.FF.keys():
            if not self.config[input_key]['output']:
                continue
            Y[input_key] = self.FF[input_key](
                self.config['non_linearity'](
                    torch.cat([X[input_key], torch.matmul(X[input_key], A[input_key])], dim=2)))
        
        return Y
    


class MixingAttention(FlexibleInputNetwork):

    def __init__(self, in_set, config):
        super(MixingAttention, self).__init__(in_set)
        """
        config: {
            'non_linearity': relu default (nn.Module),
            'x1': {
                'num_heads': 4 default (int),
                'ouput': True default (bool),
                'FF_expansion_factor': 2 default (int),
            },
            ...
        }
        """
        self.config = config
        if 'non_linearity' not in config.keys():
            config['non_linearity'] = nn.ReLU()
        for input_key in config.keys():
            if not isinstance(config[input_key], dict):
                continue
            if 'num_heads' not in config[input_key]:
                config[input_key]['num_heads'] = 4
            if 'output' not in config[input_key]:
                config[input_key]['output'] = True
            if 'FF_expansion_factor' not in config[input_key]:
                config[input_key]['FF_expansion_factor'] = 2

        # Z_i = LN(||_j=1->n(MHA(X_i, X_j, X_j)))
        self.MHA = {}
        self.LN_Z = {}
        self.Z_dim = sum([d_i for d_i in in_set.values() if config[input_key]['output']])
        for input_key in in_set.keys():
            if not config[input_key]['output']:
                continue
            self.MHA[input_key] = nn.MultiheadAttention(in_set[input_key], config[input_key]['num_heads'])
            self.LN_Z[input_key] = nn.LayerNorm(self.Z_dim)
        
        # A_i = (FF(Z_i)+Z_i)W0_i
        self.FF_A = {}
        self.W0_A = {}
        for input_key in in_set.keys():
            if not config[input_key]['output']:
                continue
            self.FF_A[input_key] = nn.Sequential(
                nn.Linear(self.Z_dim, self.Z_dim*config[input_key]['FF_expansion_factor']),
                config['non_linearity'],
                nn.Linear(self.Z_dim*config[input_key]['FF_expansion_factor'], self.Z_dim)
            )
            self.W0_A[input_key] = nn.Parameter(torch.randn(self.Z_dim, in_set[input_key]))
        
        # MA_i = LN(X_i + A_i)
        self.LN_out = {}
        for input_key in in_set.keys():
            if not config[input_key]['output']:
                continue
            self.LN_out[input_key] = nn.LayerNorm(in_set[input_key])

    def forward(self, X):

        # Z_i = LN(||_j=1->n(MHA(X_i, X_j, X_j)))
        Z = {}
        for i in X.keys():
            if not self.config[i]['output']:
                continue
            Z.append(self.LN_Z[i](torch.cat([self.MHA[i](X[i], X[j], X[j])[0] for j in X.keys()], dim=2)))
        
        # A_i = (FF(Z_i)+Z_i)W0_i
        A = {}
        for i in X.keys():
            if not self.config[i]['output']:
                continue
            A[i] = torch.matmul(self.FF_A[i](Z[i])+Z[i], self.W0_A[i])
        
        # MA_i = LN(X_i + A_i)
        Y = {}
        for i in X.keys():
            if not self.config[i]['output']:
                continue
            Y[i] = self.LN_out[i](X[i]+A[i])
        
        return Y