import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn.functional as F

class LogicModelMA(nn.Module):

    def __init__(self, in_set, embed_dim, num_FIN_layers=4):
        super(LogicModelMA, self).__init__()
        """
        in_set: {
            'prop': embed dim (int),
            'op': embed dim (int),
            'goal': embed dim (int)
        }
        embed_dim: int, dimension of the internal embeddings
        """
        
        self.in_set = in_set
        self.embed_in_set = {
            'prop': embed_dim,
            'op': embed_dim,
            'goal': embed_dim
        }
        self.embed_dim = embed_dim
        self.num_FIN_layers = num_FIN_layers

        # Up-projection to embed dim
        self.projection_prop = nn.Sequential(
            nn.Linear(in_set['prop'], embed_dim),
            nn.ReLU()
        )
        self.projection_op = nn.Sequential(
            nn.Linear(in_set['op'], embed_dim),
            nn.ReLU()
        )
        self.projection_goal = nn.Sequential(
            nn.Linear(in_set['goal'], embed_dim),
            nn.ReLU()
        )

        # MA Layers
        MA_config={
            'non_linearity': nn.ReLU(),
            'prop': {
                'num_heads': 4,
                'output': True,
                'FF_expansion_factor': 2
            },
            'op': {
                'num_heads': 4,
                'output': True,
                'FF_expansion_factor': 2
            },
            'goal': {
                'num_heads': 4,
                'output': True,
                'FF_expansion_factor': 2
            }
        }
        self.MA_list = nn.ModuleList([])
        for _ in range(self.num_FIN_layers):
            self.MA_list.append(MixingAttention(
                in_set=self.embed_in_set,
                config=MA_config
            ))
        # MA_config['prop']['output'] = False
        # MA_config['op']['output'] = False
        # self.MA_list.append(MixingAttention(
        #     in_set=self.embed_in_set,
        #     config=MA_config
        # ))

        # Down-project to 3
        self.dp = nn.Linear(embed_dim, 3)


    def forward(self, X):
        """
        Input:
            X: {
                'prop': tensor (batch_size, num_props, prop_dim),
                'op': tensor (batch_size, num_ops, op_dim),
                'goal': tensor (batch_size, num_goals, goal_dim)
            }
        Return:
            Y: {
                'goal': tensor (batch_size, num_goals, 3)
            } // [True, False, Unknowable]
        """
        # Up-projection to embed dim
        X = {
            'prop': self.projection_prop(X['prop']),
            'op': self.projection_op(X['op']),
            'goal': self.projection_goal(X['goal'])
        }

        # MA Layers
        for MA in self.MA_list:
            # print keys in X
            print(X.keys())
            X = MA(X)

        # down-project to 3
        Y = self.dp(X['goal'])
        Y = F.softmax(Y, dim=-1)

        return Y            


class FlexibleInputNetwork(nn.Module):

    def __init__(self, in_set):
        super(FlexibleInputNetwork, self).__init__()
        """
        in_set: {
            'x1': embed dim (int),
            'x2': embed dim (int),
            ...
        }
        """
        self.in_set = in_set
        self.n = len(in_set)
        self.input_keys = [k for k in in_set.keys()]
        self.input_keys.sort()
        assert self.n>0, "Input set must have at least one element"
        

class SelfProjection(FlexibleInputNetwork):

    def __init__(self, in_set, config):
        super(SelfProjection, self).__init__(in_set)
        """
        config: {
            'non_linearity': relu default (nn.Module),
            'x1': {
                'W_Z': sqrt(aggregation dim) (int),
                'd_ia': augmentation dim (int),
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
        self.Z_dim = sum([k['W_Z']**2 for k in config.values() if isinstance(k, dict)])

        # A_i = mat(MLP(Z), d_i, d_ia)
        self.MLP_A = nn.ModuleDict()
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.MLP_A[input_key] = nn.Sequential(
                nn.Linear(self.Z_dim, config[input_key]['MLP_hidden_dim'], bias=True),
                config['non_linearity'],
                nn.Linear(config[input_key]['MLP_hidden_dim'], in_set[input_key]*config[input_key]['d_ia'], bias=True)
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
        Z = torch.cat([torch.bmm(torch.matmul(X[input_key], self.W_Z[input_key][0]).permute(0, 2, 1),\
                        torch.matmul(X[input_key], self.W_Z[input_key][1]))
                       for input_key in self.input_keys], dim=-1)
        Z = torch.reshape(Z, (Z.shape[0], self.Z_dim))
        Z = self.config['non_linearity'](Z)

        # A_i = softmax(mat(MLP(Z), d_i, d_ia), across columns)
        A = {}
        for input_key in self.input_keys:
            if not self.config[input_key]['output']:
                continue
            A[input_key] = F.softmax(torch.reshape(
                self.MLP_A[input_key](Z), (Z.shape[0], self.in_set[input_key], self.config[input_key]['d_ia'])), dim=-1)
        
        # out_set_i = relu(FF_i(X_i||X_i*A_i))
        Y = {}
        for input_key in self.input_keys:
            if not self.config[input_key]['output']:
                continue
            Y[input_key] = self.LN[input_key](self.FF[input_key](
                    torch.cat([X[input_key],
                        torch.matmul(X[input_key], A[input_key])], dim=-1)))
        
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
        self.MHA = nn.ModuleDict()
        self.LN_Z = nn.ModuleDict()
        self.Z_dim = sum([d_i for d_i in in_set.values()])
        for input_key in self.input_keys:
            if not config[input_key]['output']:
                continue
            self.MHA[input_key] = nn.ModuleDict()
            for j in self.input_keys:
                self.MHA[input_key][j] = nn.MultiheadAttention(in_set[input_key], config[input_key]['num_heads'])
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
        print(self.input_keys)

        # Z_i = LN(||_j=1->n(MHA(X_i, X_j, X_j)))
        Z = {}

        for i in self.input_keys:
            if not self.config[i]['output']:
                continue
            
            Z[i] = self.LN_Z[i](torch.cat([
                self.MHA[i][j](
                    X[i].permute(1, 0, 2), 
                    X[j].permute(1, 0, 2),
                    X[j].permute(1, 0, 2))[0]\
                    .permute(1, 0, 2) for j in self.input_keys], dim=-1))
        
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
