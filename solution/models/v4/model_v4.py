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

    def __init__(self):
        super(ValueHead, self).__init__()
        self.non_linearity = nn.LeakyReLU(0.2)
        config = get_config()
        self.f_n_in = config['encoder']['node_out_dim']
        self.f_p = config['encoder']['plane_out_dim']
        self.f_c = config['encoder']['cargo_out_dim']
        self.f_n = 32
        self.agg_dim = 8
        self.output_head_hidden_dim = 128

        # Define the layers of the model

        # Node up-projection
        self.up_projection_n = nn.Linear(self.f_n_in, self.f_n, bias=True)

        # Mixing Attention Configs
        n_config = {
            'embed_dim': self.f_n,
            'num_heads': 4,
            'ff_dim': 128,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        p_config = {
            'embed_dim': self.f_p,
            'num_heads': 4,
            'ff_dim': 128,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        c_config = {
            'embed_dim': self.f_c,
            'num_heads': 4,
            'ff_dim': 128,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }

        # Mixing Attention layers
        self.MA_NPC_planes = MixingAttentionBlock({
            'num_layers': 3,
            'x': n_config,
            'y': p_config,
            'z': c_config
        })
        self.MA_NPC_cargo = MixingAttentionBlock({
            'num_layers': 3,
            'x': n_config,
            'y': p_config,
            'z': c_config
        })

        # Aggregation Configs
        n_agg_config = {
            'embed_dim': self.f_n,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        p_agg_config = {
            'embed_dim': self.f_p,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        c_agg_config = {
            'embed_dim': self.f_c,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }

        # Aggregator layers
        self.agg_planes_n = Aggregator(n_agg_config)
        self.agg_planes_p = Aggregator(p_agg_config)
        self.agg_planes_c = Aggregator(c_agg_config)
        self.agg_cargo_n = Aggregator(n_agg_config)
        self.agg_cargo_p = Aggregator(p_agg_config)
        self.agg_cargo_c = Aggregator(c_agg_config)

        # MLP Planes
        in_dim = 3 * self.agg_dim**2
        out_dim = self.f_p**2
        self.MLP_planes = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, out_dim),
            self.non_linearity
        )

        # MLP Cargo
        in_dim = 3 * self.agg_dim**2
        out_dim = self.f_c**2
        self.MLP_cargo = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, out_dim),
            self.non_linearity
        )

        # MHA Planes
        self.MHA_planes = MHA(p_config)
        self.ff_out_planes = nn.Sequential(
            nn.Linear(self.f_p, self.output_head_hidden_dim, bias=True),
            self.non_linearity,
            nn.Linear(self.output_head_hidden_dim, 1)
        )

        # MHA Cargo
        self.MHA_cargo = MHA(c_config)
        self.ff_out_cargo = nn.Sequential(
            nn.Linear(self.f_c, self.output_head_hidden_dim, bias=True),
            self.non_linearity,
            nn.Linear(self.output_head_hidden_dim, 1)
        )

    def forward(self, n, p, c):
        """
        Args:
            n, p, c: embeddings for nodes, planes, and cargo

        Returns: {
            plane_value: tensor (p x 1) state value for planes
            cargo_value: tensor (c x 1) state value for cargo
            value: tensor (1) state value for the entire state
        }
        """

        # Node up-projection
        n = self.up_projection_n(n)
        n = self.non_linearity(n)

        # Mixing Attention layers
        pp, pn, pc = self.MA_NPC_planes(p, n, c)
        cp, cn, cc = self.MA_NPC_cargo(p, n, c)

        # Aggregator layers
        pan = self.agg_planes_n(pn)
        pap = self.agg_planes_p(pp)
        pac = self.agg_planes_c(pc)
        can = self.agg_cargo_n(cn)
        cap = self.agg_cargo_p(cp)
        cac = self.agg_cargo_c(cc)

        # MLP Planes
        ap = torch.cat((pan, pap, pac), dim=1)
        ap = self.MLP_planes(ap).view(self.f_p, self.f_p)

        # MLP Cargo
        ac = torch.cat((can, cap, cac), dim=1)
        ac = self.MLP_cargo(ac).view(self.f_c, self.f_c)

        # MHA Planes
        p = self.MHA_planes(pp, ap, ap)
        p = self.ff_out_planes(p)

        # MHA Cargo
        c = self.MHA_cargo(cc, ac, ac)
        c = self.ff_out_cargo(c)

        return {
            'plane_value': p,
            'cargo_value': c,
            'value': p.sum() + c.sum()
        }


class Value(nn.Module):

    def __init__(self):
        super(Value, self).__init__()
        self.encoder = Encoder()
        self.head = ValueHead()

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

    def __init__(self):
        super(PolicyHead, self).__init__()
        self.non_linearity = nn.LeakyReLU(0.2)
        config = get_config()

        # Define the layers of the model

        # Rs
        self.R1 = R1()
        self.R2 = R2()
        self.R3 = R3()

        # Heads
        self.action_head = ActionHead()
        self.destination_head = DestinationHead()
        self.cargo_head = CargoHead()

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

        # Select Actions
        r1 = self.R1(n=n, p=p, c=c)
        action_logits = self.action_head(
            p=r1['p'], a=r1['a'], mask=action_mask)
        actions = self.sample(action_logits)

        # Update masks with the selected actions
        destination_mask, cargo_mask = self.update_masks(
            actions, destination_mask, cargo_mask)

        # Select Destinations
        r2 = self.R2(n=n, p=p, c=p, a_prev=r1['a'], y_a=actions)
        destination_logits = self.destination_head(
            p=r2['p'], n=r2['n'], ap=r2['ap'], an=r2['an'], mask=destination_mask)
        destinations = self.sample(destination_logits)

        # Select Cargo
        r2a = torch.concat((r2['ap'], r2['an']), dim=1)
        r3 = self.R3(n=n, p=p, c=c, a_prev=r2a, y_a=actions, y_d=destinations)
        cargo_logits = self.cargo_head(
            c=r3['c'], p=r3['p'], ac=r3['ac'], ap=r3['ac'], mask=cargo_mask)
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
        self.encoder = Encoder()
        self.head = PolicyHead()

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


class ActionHead(nn.Module):

    def __init__(self):
        super(ActionHead, self).__init__()
        self.non_linearity = nn.LeakyReLU(0.2)
        config = get_config()

        # Define the layers of the model

        self.plane_dim = config['encoder']['plane_out_dim']
        self.MHA = MHA({
            'embed_dim': self.plane_dim,
            'num_heads': 4,
            'ff_dim': self.plane_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity
        })
        self.FF = nn.Sequential(
            nn.Linear(self.plane_dim, self.plane_dim*2, bias=True),
            self.non_linearity,
            nn.Linear(self.plane_dim*2, 3)
        )

    def forward(self, p, a, mask=None):
        """
        Args:
            p: embeddings for planes
            a: flattened aggregation tensor
            mask: additive mask

        Returns: tensor (p x 3) action probabilities
            0: destination
            1: load/unload
            2: wait (NOOP)
        """

        a = a.view(self.plane_dim, self.plane_dim)
        p = self.MHA(p, a, a)
        p = self.FF(p)

        if mask is not None:
            p = p + mask
        return F.log_softmax(p, dim=-1)


class DestinationHead(nn.Module):

    def __init__(self):
        super(DestinationHead, self).__init__()
        self.non_linearity = nn.LeakyReLU(0.2)
        config = get_config()
        self.plane_dim = config['R2']['plane_dim']
        self.node_dim = config['R2']['node_dim']
        self.supp_dim = self.plane_dim
        self.ptr_dim = 64
        self.mlp_out_dim = config['R2']['mlp_out_dim']

        # Define the layers of the model

        # CEs
        self.CE_n = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim*2, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_out_dim*2, self.supp_dim),
            self.non_linearity
        )
        self.CE_p = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim*2, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_out_dim*2, self.supp_dim),
            self.non_linearity
        )

        # Projections
        self.projection_p = nn.Sequential(
            nn.Linear(self.plane_dim+self.supp_dim, self.ptr_dim, bias=True),
            self.non_linearity
        )
        self.projection_n = nn.Sequential(
            nn.Linear(self.node_dim+self.supp_dim, self.ptr_dim, bias=True),
            self.non_linearity
        )

        # PtrNet
        self.ptr = Ptr({
            'embed_dim': self.ptr_dim,
            'hidden_dim': self.ptr_dim*2,
            'softmax': True
        })

    def forward(self, p, n, ap, an, mask=None):
        """
        Args:
            p: embeddings for planes
            n: embeddings for nodes
            a: flattened aggregation tensor (p_f x n_f)
            mask: additive mask

        Returns: tensor (p x n) destination probabilities
        """

        # CE
        ap = ap.view(self.plane_dim, self.mlp_out_dim)
        an = an.view(self.node_dim, self.mlp_out_dim)
        p_supp = torch.matmul(p, self.CE_p(ap))
        n_supp = torch.matmul(n, self.CE_n(an))

        # Projection
        p = torch.cat((p, p_supp), dim=1)
        n = torch.cat((n, n_supp), dim=1)
        p = self.projection_p(p)
        n = self.projection_n(n)

        # PtrNet
        return self.ptr(p, n, mask)


class CargoHead(nn.Module):

    def __init__(self):
        super(CargoHead, self).__init__()
        self.non_linearity = nn.LeakyReLU(0.2)
        config = get_config()
        self.plane_dim = config['R3']['plane_dim']
        self.cargo_dim = config['R3']['cargo_dim']
        self.supp_dim = 16
        self.ptr_dim = 64
        self.mlp_out_dim = config['R3']['mlp_out_dim']

        # Define the layers of the model

        # CEs
        self.CE_c = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim*2, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_out_dim*2, self.supp_dim),
            self.non_linearity
        )
        self.CE_p = nn.Sequential(
            nn.Linear(self.mlp_out_dim, self.mlp_out_dim*2, bias=True),
            self.non_linearity,
            nn.Linear(self.mlp_out_dim*2, self.supp_dim),
            self.non_linearity
        )

        # Projections
        self.projection_c = nn.Sequential(
            nn.Linear(self.cargo_dim+self.supp_dim, self.ptr_dim, bias=True),
            self.non_linearity
        )
        self.projection_p = nn.Sequential(
            nn.Linear(self.plane_dim+self.supp_dim, self.ptr_dim, bias=True),
            self.non_linearity
        )

        # PtrNet
        self.ptr = Ptr({
            'embed_dim': self.ptr_dim,
            'hidden_dim': self.ptr_dim*2,
            'softmax': True
        })

    def forward(self, c, p, ac, ap, mask=None):
        """
        Args:
            c: embeddings for cargo
            p: embeddings for planes
            a: flattened aggregation tensor (c_f x p_f)
            mask: additive mask

        Returns: tensor (c x p) cargo probabilities
        """

        # CE
        ac = ac.view(self.cargo_dim, self.mlp_out_dim)
        ap = ap.view(self.plane_dim, self.mlp_out_dim)
        c_supp = torch.matmul(c, self.CE_c(ac))
        p_supp = torch.matmul(p, self.CE_p(ap))

        # Projection
        c = torch.cat((c, c_supp), dim=1)
        p = torch.cat((p, p_supp), dim=1)
        c = self.projection_c(c)
        p = self.projection_p(p)

        # PtrNet
        return self.ptr(c, p, mask, add_choice=True)


class R1(nn.Module):

    def __init__(self):
        super(R1, self).__init__()
        config = get_config()
        self.non_linearity = nn.LeakyReLU(0.2)
        self.node_in_dim = config['encoder']['node_out_dim']
        self.node_dim = config['R1']['node_dim']
        self.plane_dim = config['encoder']['plane_out_dim']
        self.cargo_dim = config['encoder']['cargo_out_dim']
        self.agg_dim = 8

        # Define the layers of the model

        # Node up-projection
        self.up_projection_n = nn.Linear(
            self.node_in_dim, self.node_dim, bias=True)

        # Mixing Attention Configs
        n_config = {
            'embed_dim': self.node_dim,
            'num_heads': 4,
            'ff_dim': self.node_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        p_config = {
            'embed_dim': self.plane_dim,
            'num_heads': 4,
            'ff_dim': self.plane_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        c_config = {
            'embed_dim': self.cargo_dim,
            'num_heads': 4,
            'ff_dim': self.cargo_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }

        # Mixing Attention layer
        self.MA = MixingAttentionBlock({
            'num_layers': 3,
            'x': n_config,
            'y': p_config,
            'z': c_config
        })

        # Aggregation Configs
        n_agg_config = {
            'embed_dim': self.node_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        p_agg_config = {
            'embed_dim': self.plane_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        c_agg_config = {
            'embed_dim': self.cargo_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }

        # Aggregator layers
        self.agg_n = Aggregator(n_agg_config)
        self.agg_p = Aggregator(p_agg_config)
        self.agg_c = Aggregator(c_agg_config)

        # MLP
        in_dim = 3 * self.agg_dim**2
        out_dim = self.plane_dim**2
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, in_dim*2),
            self.non_linearity,
            nn.Linear(in_dim*2, out_dim),
            self.non_linearity
        )

    def forward(self, n, p, c):
        """
        Args:
            n, p, c: embeddings for nodes, planes, and cargo

        Returns: {
            n: tensor (n x f_n) embeddings for nodes after mixing attention
            p: tensor (p x f_p) embeddings for planes after mixing attention
            c: tensor (c x f_c) embeddings for cargo after mixing attention
            a: tensor (1 x f_p^2) flattened aggregation after MLP
        }
        """

        # Node up-projection
        n = self.up_projection_n(n)
        n = self.non_linearity(n)

        # Mixing Attention layer
        n, p, c = self.MA(n, p, c)

        # Aggregator layers
        n_ = self.agg_n(n)
        p_ = self.agg_p(p)
        c_ = self.agg_c(c)

        # MLP
        a = torch.cat((n_, p_, c_), dim=1)
        a = self.MLP(a)

        return {
            'n': n,
            'p': p,
            'c': c,
            'a': a
        }


class R2(nn.Module):
    # TODO: Impliment skip connections os p, n, and c from R1

    def __init__(self):
        super(R2, self).__init__()
        config = get_config()
        self.non_linearity = nn.LeakyReLU(0.2)
        self.node_in_dim = config['encoder']['node_out_dim']
        self.node_dim = config['R2']['node_dim']
        self.plane_in_dim = config['encoder']['plane_out_dim'] + 3
        self.plane_dim = config['R2']['plane_dim']
        self.cargo_dim = config['encoder']['cargo_out_dim']
        self.agg_dim = 16
        self.mlp_hidden_dim = 1024
        self.mlp_out_dim = config['R2']['mlp_out_dim']

        # Define the layers of the model

        # Projections
        self.up_projection_n = nn.Linear(
            self.node_in_dim, self.node_dim, bias=True)
        self.down_projection_p = nn.Linear(
            self.plane_in_dim, self.plane_dim, bias=True)

        # Mixing Attention Configs
        n_config = {
            'embed_dim': self.node_dim,
            'num_heads': 4,
            'ff_dim': self.node_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        p_config = {
            'embed_dim': self.plane_dim,
            'num_heads': 4,
            'ff_dim': self.plane_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        c_config = {
            'embed_dim': self.cargo_dim,
            'num_heads': 4,
            'ff_dim': self.cargo_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }

        # Mixing Attention layer
        self.MA = MixingAttentionBlock({
            'num_layers': 3,
            'x': n_config,
            'y': p_config,
            'z': c_config
        })

        # Aggregation Configs
        n_agg_config = {
            'embed_dim': self.node_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        p_agg_config = {
            'embed_dim': self.plane_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        c_agg_config = {
            'embed_dim': self.cargo_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }

        # Aggregator layers
        self.agg_n = Aggregator(n_agg_config)
        self.agg_p = Aggregator(p_agg_config)
        self.agg_c = Aggregator(c_agg_config)

        # MLP
        in_dim = 3 * self.agg_dim**2 + self.plane_dim**2
        out_dim_plane = self.plane_dim*self.mlp_out_dim
        out_dim_cargo = self.node_dim*self.mlp_out_dim
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, self.mlp_hidden_dim),
            self.non_linearity,
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            self.non_linearity,
        )
        self.MLP_plane = nn.Sequential(
            nn.Linear(self.mlp_hidden_dim, out_dim_plane),
            self.non_linearity
        )
        self.MLP_nodes = nn.Sequential(
            nn.Linear(self.mlp_hidden_dim, out_dim_cargo),
            self.non_linearity
        )

    def forward(self, n, p, c, a_prev, y_a):
        """
        Args:
            n, p, c: embeddings for nodes, planes, and cargo
            a_prev: previous flattened aggregation tensor
            y_a: select action head's output (p x 3)

        Returns: {
            n: tensor (n x f_n) embeddings for nodes after mixing attention
            p: tensor (p x f_p) embeddings for planes after mixing attention
            c: tensor (c x f_c) embeddings for cargo after mixing attention
            a: tensor (1 x f_p*f_n) flattened aggregation after MLP
        }
        """

        # Concat additional features
        p = torch.cat((p, y_a), dim=1)

        # Projection
        n = self.up_projection_n(n)
        n = self.non_linearity(n)
        p = self.down_projection_p(p)
        p = self.non_linearity(p)

        # Mixing Attention layer
        n, p, c = self.MA(n, p, c)

        # Aggregator layers
        n_ = self.agg_n(n)
        p_ = self.agg_p(p)
        c_ = self.agg_c(c)

        # MLP
        a = torch.cat((n_, p_, c_, a_prev), dim=1)
        a = self.MLP(a)
        ap = self.MLP_plane(a)
        an = self.MLP_nodes(a)

        return {
            'n': n,
            'p': p,
            'c': c,
            'ap': ap,
            'an': an
        }


class R3(nn.Module):
    # TODO: Impliment skip connections os p, n, and c from R1

    def __init__(self):
        super(R3, self).__init__()
        config = get_config()
        self.plane_dim_prev = config['R2']['plane_dim']
        self.node_dim_prev = config['R2']['node_dim']

        self.non_linearity = nn.LeakyReLU(0.2)
        self.node_in_dim = config['encoder']['node_out_dim']
        self.node_dim = 64
        self.plane_in_dim = config['encoder']['plane_out_dim'] + 3
        self.plane_dim = config['R3']['plane_dim']
        self.cargo_in_dim = config['encoder']['cargo_out_dim']
        self.supp_dim = 16
        self.cargo_dim = config['R3']['cargo_dim']
        self.agg_dim = 8
        self.CE_hidden_dim = 128
        self.mlp_hidden_dim = 1024
        self.mlp_out_dim = config['R3']['mlp_out_dim']

        # Define the layers of the model

        # Create Embeddings
        self.CE_P = nn.Sequential(
            nn.Linear(self.plane_in_dim, self.CE_hidden_dim, bias=True),
            self.non_linearity,
            nn.Linear(self.CE_hidden_dim, self.supp_dim),
            self.non_linearity
        )
        self.CE_N = nn.Sequential(
            nn.Linear(self.node_in_dim, self.CE_hidden_dim, bias=True),
            self.non_linearity,
            nn.Linear(self.CE_hidden_dim, self.supp_dim),
            self.non_linearity
        )

        # Projections
        self.projection_n = nn.Linear(
            self.node_in_dim+self.supp_dim, self.node_dim, bias=True)
        self.projection_p = nn.Linear(
            self.supp_dim+self.plane_in_dim, self.plane_dim, bias=True)
        self.projection_c = nn.Linear(
            self.cargo_in_dim, self.cargo_dim, bias=True)

        # Mixing Attention Configs
        n_config = {
            'embed_dim': self.node_dim,
            'num_heads': 4,
            'ff_dim': self.node_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        p_config = {
            'embed_dim': self.plane_dim,
            'num_heads': 4,
            'ff_dim': self.plane_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }
        c_config = {
            'embed_dim': self.cargo_dim,
            'num_heads': 4,
            'ff_dim': self.cargo_dim*2,
            'bias': False,
            'non_linearity': self.non_linearity,
            'self_attention': False
        }

        # Mixing Attention layer
        self.MA = MixingAttentionBlock({
            'num_layers': 3,
            'x': n_config,
            'y': p_config,
            'z': c_config
        })

        # Aggregation Configs
        n_agg_config = {
            'embed_dim': self.node_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        p_agg_config = {
            'embed_dim': self.plane_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }
        c_agg_config = {
            'embed_dim': self.cargo_dim,
            'agg_dim': self.agg_dim,
            'num_layers': 3,
            'ff_dim': 256,
            'bias': True,
            'non_linearity': self.non_linearity
        }

        # Aggregator layers
        self.agg_n = Aggregator(n_agg_config)
        self.agg_p = Aggregator(p_agg_config)
        self.agg_c = Aggregator(c_agg_config)

        # MLP
        in_dim = 3 * self.agg_dim**2 + self.plane_dim_prev*config['R2']['mlp_out_dim'] +\
            self.node_dim_prev*config['R2']['mlp_out_dim']
        cargo_out_dim = self.plane_dim*self.mlp_out_dim
        plane_out_dim = self.cargo_dim*self.mlp_out_dim
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, self.mlp_hidden_dim),
            self.non_linearity,
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            self.non_linearity
        )
        self.MLP_cargo = nn.Sequential(
            nn.Linear(self.mlp_hidden_dim, cargo_out_dim),
            self.non_linearity
        )
        self.MLP_plane = nn.Sequential(
            nn.Linear(self.mlp_hidden_dim, plane_out_dim),
            self.non_linearity
        )

    def forward(self, n, p, c, a_prev, y_a, y_d):
        """
        Args:
            n, p, c: embeddings for nodes, planes, and cargo
            a_prev: previous flattened aggregation tensor
            y_a: select action head's output (p x 3)
            y_d: select destination head's output (p x n)

        Returns: {
            n: tensor (n x f_n) embeddings for nodes after mixing attention
            p: tensor (p x f_p) embeddings for planes after mixing attention
            c: tensor (c x f_c) embeddings for cargo after mixing attention
            a: tensor (1 x f_p*f_n) flattened aggregation after MLP
        }
        """

        # Concat additional features
        p = torch.cat((p, y_a), dim=1)

        # Create Embeddings
        n_supp = torch.matmul(y_d.T, self.CE_P(p))
        p_supp = torch.matmul(y_d, self.CE_N(n))

        # Concat additional features
        p = torch.cat((p, p_supp), dim=1)
        n = torch.cat((n, n_supp), dim=1)

        # Projection
        n = self.non_linearity(self.projection_n(n))
        p = self.non_linearity(self.projection_p(p))
        c = self.non_linearity(self.projection_c(c))

        # Mixing Attention layer
        n, p, c = self.MA(n, p, c)

        # Aggregator layers
        n_ = self.agg_n(n)
        p_ = self.agg_p(p)
        c_ = self.agg_c(c)

        # MLP
        a = torch.cat((n_, p_, c_, a_prev), dim=1)
        a = self.MLP(a)
        ac = self.MLP_cargo(a)
        ap = self.MLP_plane(a)

        return {
            'n': n,
            'p': p,
            'c': c,
            'ac': ac,
            'ap': ap
        }

class GATBlock(nn.Module):
    """
    GCN layer for the Airlift solution.
    """

    def __init__(self, config):
        """
        Initialize the GCN layer. Dynamically creates the layers based on the number of layers param.

        Args:
            in_channels: int, number of input features
            hidden_channels: int, number of hidden features
            out_channels: int, number of output features
            num_layers: int, number of layers in the GCN
        """
        super(GATBlock, self).__init__()
        self.config = config

        in_channels = config['in_features']
        hidden_channels = config['hidden_features']
        out_channels = config['out_features']
        num_layers = config['num_layers']
        edge_dim = config['edge_features']

        # Define the layers of the model
        self.layers = []
        self.norms = []
        if num_layers == 1:
            self.layers.append(
                GATConv(in_channels, out_channels, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(out_channels))
        else:
            for i in range(num_layers-1):
                in_channels = in_channels if i == 0 else hidden_channels
                self.layers.append(
                    GATConv(in_channels, hidden_channels, edge_dim=edge_dim))
                self.norms.append(nn.LayerNorm(hidden_channels))
            self.layers.append(
                GATConv(hidden_channels, out_channels, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(out_channels))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, data):
        """
        Forward pass of the GCN layer.

        Args:
            data: torch_geometric.data.Data object

        Returns:
            y: tensor (#nodes x #features) embeddings for nodes
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.config['non_linearity'](x)
            x = norm(x)

        return x
