import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

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
        self.input_keys = list(in_set.keys()).sort()
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
        self.W_Z = nn.ParameterDict()
        for input_key in self.input_keys:
            self.W_Z[input_key] = (nn.Parameter(torch.randn(in_set[input_key], config[input_key]['W_Z'])),
                                   nn.Parameter(torch.randn(in_set[input_key], config[input_key]['W_Z'])))
        
        # Z el-of R^(SUM_i=1->n(d_i^2))
        self.Z_dim = sum([d_i**2 for d_i in in_set.values()])

        # A_i = mat(MLP(Z), d_i, d_ia)
        self.MLP_A = nn.ModuleDict()
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.MLP_A[input_key] = nn.Sequential(
                nn.Linear(self.Z_dim, config['MLP_hidden_dim'], bias=True),
                config['non_linearity'],
                nn.Linear(config['MLP_hidden_dim'], in_set[input_key]*config[input_key]['d_ia'], bias=True)
            )

        # SP_i = LN(FF_i(X_i||x_i*A_i))
        self.FF = nn.ModuleDict()
        self.LN = nn.ModuleDict()
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            FF_dim = in_set[input_key]+config[input_key]['d_ia']
            self.FF[input_key] = nn.Sequential(
                nn.Linear(FF_dim, FF_dim*config[input_key]['FF_expansion_factor'], bias=True),
                config['non_linearity'],
                nn.Linear(FF_dim*config[input_key]['FF_expansion_factor'], in_set[input_key])
            )
            self.LN[input_key] = nn.LayerNorm(in_set[input_key])

    def forward(self, X):

        # Z = vec(||_i=1->N(X_i*W_i1).T * (X_i*W_i)2)
        Z = torch.cat([torch.matmul(torch.matmul(X[input_key], self.W_Z[input_key][0]).T,\
                        torch.matmul(X[input_key], self.W_Z[input_key][1]))
                       for input_key in self.input_keys], dim=1)
        Z = torch.reshape(Z, (-1, self.Z_dim))
        Z = self.config['non_linearity'](Z)

        # A_i = softmax(mat(MLP(Z), d_i, d_ia), across columns)
        A = {}
        for input_key in self.input_keys:
            if not self.config[input_key]['output']:
                continue
            A[input_key] = F.softmax(torch.reshape(
                self.MLP_A[input_key](Z), (-1, self.in_set[input_key], self.config[input_key]['d_ia'])), dim=-2)
        
        # out_set_i = relu(FF_i(X_i||X_i*A_i))
        Y = {}
        for input_key in self.input_keys:
            if not self.config[input_key]['output']:
                continue
            Y[input_key] = self.LN[input_key](self.FF[input_key](
                    torch.cat([X[input_key], self.config['non_linearity'](
                        torch.matmul(X[input_key], A[input_key]))], dim=2)))
        
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

        self.input_keys = list(in_set.keys()).sort()

        # Z_i = LN(||_j=1->n(MHA(X_i, X_j, X_j)))
        self.MHA = nn.ModuleDict()
        self.LN_Z = nn.ModuleDict()
        self.Z_dim = sum([d_i for d_i in in_set.values() if config[input_key]['output']])
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.MHA[input_key] = nn.MultiheadAttention(in_set[input_key], config[input_key]['num_heads'])
            self.LN_Z[input_key] = nn.LayerNorm(self.Z_dim)
        
        # A_i = (FF(Z_i)+Z_i)W0_i
        self.FF_A = nn.ModuleDict()
        self.W0_A = nn.ParameterDict()
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.FF_A[input_key] = nn.Sequential(
                nn.Linear(self.Z_dim, self.Z_dim*config[input_key]['FF_expansion_factor']),
                config['non_linearity'],
                nn.Linear(self.Z_dim*config[input_key]['FF_expansion_factor'], self.Z_dim)
            )
            self.W0_A[input_key] = nn.Parameter(torch.randn(self.Z_dim, in_set[input_key]))
        
        # MA_i = LN(X_i + A_i)
        self.LN_out = nn.ModuleDict()
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.LN_out[input_key] = nn.LayerNorm(in_set[input_key])

    def forward(self, X):

        # Z_i = LN(||_j=1->n(MHA(X_i, X_j, X_j)))
        Z = {}
        for i in self.input_keys:
            if not self.config[i]['output']:
                continue
            Z[i] = self.LN_Z[i](torch.cat([self.MHA[i](X[i], X[j], X[j])[0] for j in self.input_keys], dim=2))
        
        # A_i = relu(FF(Z_i)+Z_i)W0_i
        A = {}
        for i in self.input_keys:
            if not self.config[i]['output']:
                continue
            A[i] = torch.matmul(self.config['non_linearity'](self.FF_A[i](Z[i])+Z[i]), self.W0_A[i])
        
        # MA_i = relu(LN(X_i + A_i))
        Y = {}
        for i in self.input_keys:
            if not self.config[i]['output']:
                continue
            Y[i] = self.config['non_linearity'](self.LN_out[i](X[i]+A[i]))
        
        return Y



class Ptr(nn.Module):

    def __init__(self, config):
        super(Ptr, self).__init__()
        """
        config: {
            'embed_dim': int, dimension of the embeddings
            'hidden_dim': int (default=embed_dim), dimension of the hidden layer
            'softmax': bool (default=True), whether to apply softmax
        }
        """
        if 'softmax' not in config.keys():
            config['softmax'] = True
        if 'hidden_dim' not in config.keys():
            try:
                config['hidden_dim'] = config['embed_dim']
            except:
                raise ValueError("embed_dim not included in Ptr config")
        self.config = config

        # xc = Wx0X0 + Wx1X1 + ... + Bx
        self.Wx = nn.Linear(config['embed_dim'],
                            config['hidden_dim'], bias=True)

        # yc = Wy0Y0 + Wy1Y1 + ... + By
        self.Wy = nn.Linear(config['embed_dim'],
                            config['hidden_dim'], bias=True)

    def forward(self, x, y, mask=None, add_choice=False):
        """
        Args:
            x (list): list of tensors for the first entity
            y (list): list of tensors for the second entity
            mask (tensor): mask for the attention (additive mask)

        Returns:
            ptr_mtx: tensor (#x x #y) attention weights
        """

        xc = self.Wx(x)
        yc = self.Wy(y)

        # ptr = softmax((xc * ycT) / sqrt(d_k))
        ptr_mtx = torch.matmul(xc, yc.T) / xc.size(-1)**0.5

        # Add a NOOP choice if needed
        if add_choice:
            ptr_mtx = torch.cat(
                (ptr_mtx, torch.zeros(ptr_mtx.size(0), 1)), dim=1)

        if mask is not None:
            ptr_mtx = ptr_mtx + mask

        # softmax
        if self.config['softmax']:
            ptr_mtx = F.softmax(ptr_mtx, dim=-1)

        return ptr_mtx



class TransformerLayer(nn.Module):

    def __init__(self, config):
        """
        config: {
            'embed_dim': int, dimension of the embeddings
            'num_heads': int, number of heads
            'ff_expansion_factor': int, expansion of the feedforward layer
            'non_linearity': nn.Module, non-linearity function
        }
        """
        super(TransformerLayer, self).__init__()
        self.config = config

        # Define the layers of the model
        self.attention = nn.MultiheadAttention(config['embed_dim'], config['num_heads'])
        self.ff = nn.Sequential(
            nn.Linear(config['embed_dim'], 
                      config['ff_expansion_facto']*config['embed_dim'], bias=True),
            config['non_linearity'],
            nn.Linear(config['ff_expansion_facto']*config['embed_dim'],
                      config['embed_dim'])
        )
        self.norm1 = nn.LayerNorm(config['embed_dim'])
        self.norm2 = nn.LayerNorm(config['embed_dim'])

    def forward(self, q, k, v):

        v, _ = self.attention(q, k, v)
        v = self.norm1(v + q)
        v = self.norm2(v + self.ff(v))

        return v
    


class GATTransformerLayer(nn.Module):

    def __init__(self, config):
        """
        config: {
            'embed_dim': int, dimension of the embeddings
            'num_heads': int, number of heads
            'ff_expansion_factor': int, expansion of the feedforward layer
            'non_linearity': nn.Module, non-linearity function
        }
        """
        super(GATTransformerLayer, self).__init__()
        self.config = config

        # Define the layers of the model
        self.GAT = GATConv(self.config['embed_dim'], self.config['embed_dim'],
                           heads=self.config['num_heads'])
        self.ff = nn.Sequential(
            nn.Linear(config['embed_dim'],
                      config['ff_expansion_factor']*config['embed_dim'], bias=True),
            config['non_linearity'],
            nn.Linear(config['ff_expansion_factor']*config['embed_dim'],
                      config['embed_dim'])
        )
        self.norm1 = nn.LayerNorm(config['embed_dim'])
        self.norm2 = nn.LayerNorm(config['embed_dim'])

    def forward(self, x, edge_index, edge_attr):

        v = self.GAT(x, edge_index, edge_attr=edge_attr)
        v = self.norm1(v + x)
        v = self.norm2(v + self.ff(v))

        return v