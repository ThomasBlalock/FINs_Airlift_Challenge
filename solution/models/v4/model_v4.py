import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from neural_modules import MixingAttention, SelfProjection,\
Ptr, TransformerLayer, GATTransformerLayer



"""
NOTE: Ideas
- To cut the param count, make the output of the MLPs embed_size x small_fixed_size.. would cut params in half
"""

DESTINATION_ACTION = 0
LOAD_UNLOAD_ACTION = 1
NOOP_ACTION = 2


def count_parameters(model):
    param_count = (sum(p.numel() for p in model.parameters()), {})
    for new_name, module in model.named_children():
        param_count[1][new_name] = count_parameters(module)
    return param_count


class Encoder(nn.Module):

    def __init__(self, config):
        """
        config: { 
            nodes: {
                in_dim: int input dimension
                out_dim: int output dimension
            },
            agents: {
                in_dim: int input dimension
                out_dim: int output dimension
            },
            cargo: {
                in_dim: int input dimension
                out_dim: int output dimension
            }
        }
        """
        super(Encoder, self).__init__()
        self.config = config
        self.nl = nn.LeakyReLU(0.2) #nl = non-linearity

        # Up-Projections
        self.up_projection_n = nn.Sequential(
            nn.Linear(config['nodes']['in_dim'], config['nodes']['out_dim'],
                      bias=True),
            self.nl
        )
        self.up_projection_a = nn.Sequential(
            nn.Linear(config['agents']['in_dim'], config['agents']['out_dim'],
                      bias=True),
            self.nl
        )
        self.up_projection_c = nn.Sequential(
            nn.Linear(config['cargo']['in_dim'], config['cargo']['out_dim'],
                      bias=True),
            self.nl
        )

        # GAT
        self.GAT = nn.ModuleList(
            [GATTransformerLayer({
                'embed_dim': config['nodes']['out_dim'],
                'num_heads': 4,
                'ff_expansion_factor': 2,
                'non_linearity': self.nl,
            }) for _ in range(3)]
        )

        # Plane Transformer
        self.plane_transformer = nn.ModuleList(
            [TransformerLayer({
                'embed_dim': config['agents']['out_dim'],
                'num_heads': 4,
                'ff_expansion_factor': 2,
                'non_linearity': self.nl,
            }) for _ in range(3)]
        )

        # Cargo Transformer
        self.cargo_transformer = nn.ModuleList(
            [TransformerLayer({
                'embed_dim': config['cargo']['out_dim'],
                'num_heads': 4,
                'ff_expansion_factor': 2,
                'non_linearity': self.nl,
            }) for _ in range(3)]
        )


    def forward(self, x):
        """
        Args:
            x (dict): formatted observation

        Returns: {
            cargo_embeddings: tensor (c x f_c) embeddings for cargo
            plane_embeddings: tensor (p x f_p) embeddings for planes
            node_embeddings: tensor (n x f_n) embeddings for nodes
        }
        """

        # Validate the input
        assert 'nodes' in x, 'nodes not found in the input'
        assert 'agents' in x, 'agents not found in the input'
        assert 'cargo' in x, 'cargo not found in the input'
        assert 'PyGeom' in x['nodes'], 'PyGeom object not found in the input'
        assert 'tensor' in x['agents'], 'agents tensor not found in the input'
        assert 'tensor' in x['cargo'], 'cargo tensor not found in the input'
        assert x['agents']['tensor'].shape[1] == self.plane_in_dim, 'agents tensor has incorrect shape'
        assert x['cargo']['tensor'].shape[1] == self.cargo_in_dim, 'cargo tensor has incorrect shape'

        # Unpack the input
        node_embeddings = x['nodes']['PyGeom']
        plane_embeddings = x['agents']['tensor']
        cargo_embeddings = x['cargo']['tensor']

        # Embed Nodes
        node_embeddings, edge_index, edge_attr =\
            node_embeddings.x, node_embeddings.edge_index, node_embeddings.edge_attr
        node_embeddings = self.up_projection_n(node_embeddings)
        for layer in self.GAT:
            node_embeddings = layer(node_embeddings, edge_index, edge_attr)
        
        # Embed Planes
        for layer in self.plane_transformer:
            plane_embeddings = layer(plane_embeddings)

        # Embed Cargo
        for layer in self.cargo_transformer:
            cargo_embeddings = layer(cargo_embeddings)

        return {
            'cargo_embeddings': cargo_embeddings,
            'plane_embeddings': plane_embeddings,
            'node_embeddings': node_embeddings
        }


class ValueHead(nn.Module):

    def __init__(self, config):
        """
        config: { 
            nodes_in_dim: in_dim,
            agents_in_dim: in_dim,
            cargo_in_dim: in_dim,
        }
        """
        super(ValueHead, self).__init__()
        self.config = config
        self.nl = nn.LeakyReLU(0.2)

        # Base FIN Blocks
        self.FIN_base = nn.ModuleList(
            [MixingAttention(in_set={
                'nodes': config['nodes_in_dim'],
                'agents': config['agents_in_dim'],
                'cargo': config['cargo_in_dim']
            }, config={
                'non_linearity': self.nl,
                'nodes': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'agents': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'cargo': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                }
            }) for _ in range(3)])
        self.FIN_agent = nn.ModuleList([
            SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    }
                }
            ) for _ in range(2)] + [
                SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    }
                }
            )])
        self.FIN_cargo = nn.ModuleList([
            SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    }
                }
            ) for _ in range(2)] + [
                SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    }
                }
            )])

        # Transformer Layers
        self.transformer_plane = TransformerLayer({
            'embed_dim': config['agents_in_dim'],
            'num_heads': 4,
            'ff_expansion_factor': 2,
            'non_linearity': self.nl
        })
        self.transformer_cargo = TransformerLayer({
            'embed_dim': config['cargo_in_dim'],
            'num_heads': 4,
            'ff_expansion_factor': 2,
            'non_linearity': self.nl
        })

        # Down-Projections
        self.down_projection_plane = nn.Linear(config['agents_in_dim'], 1, bias=True)
        self.down_projection_cargo = nn.Linear(config['cargo_in_dim'], 1, bias=True)

    def forward(self, n, p, c):
        """
        Args:
            n: embeddings for nodes
            p: embeddings for planes
            c: embeddings for cargo

        Returns: {
            plane_value: tensor (p x 1) state value for planes
            cargo_value: tensor (c x 1) state value for cargo
            value: tensor (1) state value for the entire state
        }
        """

        # Base FIN Blocks
        for layer in self.FIN_base:
            X = layer({'nodes': n, 'agents': p, 'cargo': c})
            n, p, c = X['nodes'], X['agents'], X['cargo']

        # Agent Head
        n_a, p_a, c_a = n, p, c
        for layer in self.FIN_agent:
            X = layer({'nodes': n_a, 'agents': p_a, 'cargo': c_a})
            if 'nodes' in X:
                n_a = X['nodes']
            if 'agents' in X:
                p_a = X['agents']
            if 'cargo' in X:
                c_a = X['cargo']
        plane_value = self.transformer_plane(p_a)
        plane_value = self.down_projection_plane(plane_value)
        del n_a, p_a, c_a, X

        # Cargo Head
        n_c, p_c, c_c = n, p, c
        for layer in self.FIN_cargo:
            X = layer({'nodes': n_c, 'agents': p_c, 'cargo': c_c})
            if 'nodes' in X:
                n_c = X['nodes']
            if 'agents' in X:
                p_c = X['agents']
            if 'cargo' in X:
                c_c = X['cargo']
        cargo_value = self.transformer_cargo(c_c)
        cargo_value = self.down_projection_cargo(cargo_value)
        del n_c, p_c, c_c, n, p, c, X

        return {
            'plane_value': plane_value,
            'cargo_value': cargo_value,
            'value': plane_value + cargo_value
        }


class Value(nn.Module):

    def __init__(self):
        super(Value, self).__init__()
        self.encoder = Encoder({
            'nodes': {
                'in_dim': 3,
                'out_dim': 32,
            },
            'agents': {
                'in_dim': 9,
                'out_dim': 32,
            },
            'cargo': {
                'in_dim': 11,
                'out_dim': 32,
            }
        })
        self.head = ValueHead({
            'nodes_in_dim': 32,
            'agents_in_dim': 32,
            'cargo_in_dim': 32
        })

    def forward(self, x):
        """
        Args:
            x (dict): formatted observation

        Returns: {
            plane_value: tensor (p x 1) state value for planes
            cargo_value: tensor (c x 1) state value for cargo
            value: tensor (1) state value for the entire state
        }
        """

        embeddings = self.encoder(x)
        return self.head(embeddings['node_embeddings'], embeddings['plane_embeddings'], embeddings['cargo_embeddings'])


class PolicyHead(nn.Module):

    def __init__(self, config):
        """
        config: {
            plane_dim: int dimension of plane embeddings
            node_dim: int dimension of node embeddings
            cargo_dim: int dimension of cargo embeddings
        }
        """
        super(PolicyHead, self).__init__()
        self.config = config
        self.nl = nn.LeakyReLU(0.2)
        self.aug_dim = 16

        # Action Head
        self.FIN_action = nn.ModuleList([
            MixingAttention(in_set={
                'nodes': config['nodes_in_dim'],
                'agents': config['agents_in_dim'],
                'cargo': config['cargo_in_dim']
            }, config={
                'non_linearity': self.nl,
                'nodes': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'agents': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'cargo': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                }
            }) for _ in range(3)] + [
                SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    }
                }
            )
        ])
        self.transformer_action = TransformerLayer({
            'embed_dim': config['agents_in_dim'],
            'num_heads': 4,
            'ff_expansion_factor': 2,
            'non_linearity': self.nl
        })
        self.down_projection_action = nn.Linear(config['agents_in_dim'], 3, bias=True)

        # Destination Head
        self.down_projection_destination = nn.ModuleDict({
            'nodes': nn.Sequential(
                nn.Linear(config['agents_in_dim']*2, config['agents_in_dim'], bias=True),
                self.nl
            ),
            'agents': nn.Sequential(
                nn.Linear(config['agents_in_dim']*2 + 3, config['agents_in_dim'], bias=True),
                self.nl
            ),
            'cargo': nn.Sequential(
            nn.Linear(config['agents_in_dim']*2, config['agents_in_dim'], bias=True),
            self.nl
        )
        })
        self.FIN_destination = nn.ModuleList([
            MixingAttention(in_set={
                'nodes': config['nodes_in_dim'],
                'agents': config['agents_in_dim'],
                'cargo': config['cargo_in_dim']
            }, config={
                'non_linearity': self.nl,
                'nodes': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'agents': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'cargo': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                }
            }) for _ in range(3)] + [
                SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    }
                }
            )
        ])
        self.ptr_destination = Ptr({
            'embed_dim': config['agents_in_dim'],
            'hidden_dim': config['agents_in_dim'],
            'softmax': True
        })

        # Cargo Head
        self.augmentation = nn.ModuleDict({
            'nodes': nn.Sequential(
                nn.Linear(config['agents_in_dim'], self.aug_dim, bias=True),
                self.nl
            ),
            'agents': nn.Sequential(
                nn.Linear(config['nodes_in_dim'], self.aug_dim, bias=True),
                self.nl
            )
        })
        self.down_projection_cargo = nn.ModuleDict({
            'nodes': nn.Sequential(
                nn.Linear(config['agents_in_dim']*2 + self.aug_dim, config['agents_in_dim'], bias=True),
                self.nl
            ),
            'agents': nn.Sequential(
                nn.Linear(config['agents_in_dim']*2 + self.aug_dim + 3, config['agents_in_dim'], bias=True),
                self.nl
            ),
            'cargo': nn.Sequential(
            nn.Linear(config['agents_in_dim']*2, config['agents_in_dim'], bias=True),
            self.nl
        )
        })
        self.FIN_cargo = nn.ModuleList([
            MixingAttention(in_set={
                'nodes': config['nodes_in_dim'],
                'agents': config['agents_in_dim'],
                'cargo': config['cargo_in_dim']
            }, config={
                'non_linearity': self.nl,
                'nodes': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'agents': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                },
                'cargo': {
                    'num_heads': 4,
                    'output': True,
                    'FF_expansion_factor': 2
                }
            }) for _ in range(3)] + [
                SelfProjection(
                in_set={
                    'nodes': config['nodes_in_dim'],
                    'agents': config['agents_in_dim'],
                    'cargo': config['cargo_in_dim']
                }, config={
                    'nodes': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': False,
                    },
                    'agents': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    },
                    'cargo': {
                        'W_Z': 16,
                        'd_ia': 16,
                        'MLP_hidden_dim': 64,
                        'FF_expansion_factor': 2,
                        'ouput': True,
                    }
                }
            )
        ])
        self.ptr_cargo = Ptr({
            'embed_dim': config['agents_in_dim'],
            'hidden_dim': config['agents_in_dim'],
            'softmax': True
        })


    def sample(self, logits, mask=None):
        """
        Args:
            logits: tensor (p x n) probabilities
            mask: additive mask

        Returns: {
            actions: tensor (p x n) one-hot encoded actions
            action_logits: tensor (p x n) log probabilities
            destinations: tensor (p x n) one-hot encoded destinations
            destination_logits: tensor (p x n) log probabilities
            cargo: tensor (p x n) one-hot encoded cargo
            cargo_logits: tensor (p x n) log probabilities
        }
        """

        if mask is not None:
            logits = logits + mask
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1).detach()

    def update_masks(self, actions, destination_mask, cargo_mask):

        cargo_mask = cargo_mask.T
        for plane, action in enumerate(actions):
            action_id = action.argmax().item()
            if action_id != DESTINATION_ACTION:
                destination_mask[plane] = torch.tensor(
                    [-float('inf')]*destination_mask.shape[1])
            if action_id != LOAD_UNLOAD_ACTION:
                cargo_mask[plane] = torch.tensor(
                    [-float('inf')]*cargo_mask.shape[1])

        return destination_mask, cargo_mask.T

    def forward(self, n, p, c, action_mask=None, destination_mask=None, cargo_mask=None):
        """
        Args:
            n, p, c: embeddings for nodes, planes, and cargo
            action_mask: additive mask for action head
            destination_mask: additive mask for destination head
            cargo_mask: additive mask for cargo head


        """

        # Action Head
        n_a, p_a, c_a = n, p, c
        for layer in self.FIN_action:
            X = layer({'nodes': n_a, 'agents': p_a, 'cargo': c_a})
            if 'nodes' in X:
                n_a = X['nodes']
            if 'agents' in X:
                p_a = X['agents']
            if 'cargo' in X:
                c_a = X['cargo']
        del X
        action_logits = self.transformer_action(p_a)
        action_logits = self.down_projection_action(action_logits)
        actions = self.sample(action_logits, mask=action_mask)

        # Update Masks
        if action_mask is not None:
            destination_mask, cargo_mask = self.update_masks(
                actions, destination_mask, cargo_mask)
            
        # Destination Head
        n_d = self.down_projection_destination['nodes'](torch.cat((n, n_a), dim=-1))
        p_d = self.down_projection_destination['agents'](torch.cat((p, p_a, actions), dim=-1))
        c_d = self.down_projection_destination['cargo'](torch.cat((c, c_a), dim=-1))
        del n_a, p_a, c_a
        for layer in self.FIN_destination:
            X = layer({'nodes': n_d, 'agents': p_d, 'cargo': c_d})
            if 'nodes' in X:
                n_d = X['nodes']
            if 'agents' in X:
                p_d = X['agents']
            if 'cargo' in X:
                c_d = X['cargo']
        destination_logits = self.ptr_destination(p_d, n_d, mask=destination_mask)
        destinations = self.sample(destination_logits)
        del X

        # Cargo Head
        aug_n = torch.matmul(destinations.T, self.augmentation['nodes'](p_d))
        aug_p = torch.matmul(destinations, self.augmentation['agents'](n_d))
        n_c = self.down_projection_cargo['nodes'](torch.cat((n, n_d, aug_n), dim=-1))
        p_c = self.down_projection_cargo['agents'](torch.cat((p, p_d, aug_p, actions), dim=-1))
        c_c = self.down_projection_cargo['cargo'](torch.cat((c, c_d), dim=-1))
        del n_d, p_d, c_d, aug_n, aug_p
        for layer in self.FIN_cargo:
            X = layer({'nodes': n_c, 'agents': p_c, 'cargo': c_c})
            if 'nodes' in X:
                n_c = X['nodes']
            if 'agents' in X:
                p_c = X['agents']
            if 'cargo' in X:
                c_c = X['cargo']
        del X
        cargo_logits = self.ptr_cargo(c_c, p_c, mask=cargo_mask)
        cargo = self.sample(cargo_logits)

        return {
            'actions': actions,
            'action_logits': action_logits,
            'destinations': destinations,
            'destination_logits': destination_logits,
            'cargo': cargo,
            'cargo_logits': cargo_logits
        }


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.encoder = Encoder({
            'nodes': {
                'in_dim': 3,
                'out_dim': 32,
            },
            'agents': {
                'in_dim': 9,
                'out_dim': 32,
            },
            'cargo': {
                'in_dim': 11,
                'out_dim': 32,
            }
        })
        self.head = PolicyHead({
            'nodes_in_dim': 32,
            'agents_in_dim': 32,
            'cargo_in_dim': 32
        })

    def forward(self, x):
        """
        Args:
            x (dict): formatted observation

        Returns: {
            actions: tensor (p x n) one-hot encoded actions
            action_logits: tensor (p x n) log probabilities
            destinations: tensor (p x n) one-hot encoded destinations
            destination_logits: tensor (p x n) log probabilities
            cargo: tensor (p x n) one-hot encoded cargo
            cargo_logits: tensor (p x n) log probabilities
        }
        """

        embeddings = self.encoder(x)
        return self.head(embeddings['node_embeddings'], embeddings['plane_embeddings'], embeddings['cargo_embeddings'],
                         action_mask=x['agents']['action_mask'], destination_mask=x['agents']['destination_mask'], cargo_mask=x['cargo']['mask'])