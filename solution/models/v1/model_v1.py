import torch.nn as nn
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


"""
Ideas to try if the model is not learning:
- Place GATs between each MHA nd Mixing MHA in the encoder and wherever else a dense layer would be used
- Add skip connections from directly after the the up-projection, Self MHAs, and Mixing MHAs in the encoder
- Rethink the attention mechanism in the beginning of the actor, or maybe add more attention layers
- Mask each step of the model to guide its logic after an appropriate layer
- Add multiple heads to the CPtrs & Contexts
- Intermix the Self-Attention layers with the Mixing Attention layers
- include the adj_mtx as input (nxn matrix, should be easy to find an appropriate machanism and placement)
- Separate the plane assignment and cargo assignment into two separate models with distinct encoders
"""


class Encoder(nn.Module):
    """
    Encoder for the Airlift solution.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        # Define the layers of the model

        # GAT for nodes
        self.GAT = GATBlock(config['GAT'])
        
        # Up-projection layers
        self.node_up_projection = nn.Linear(config['node_features'],
                                            config['embed_size'])
        self.plane_up_projection = nn.Linear(config['node_features']+\
                                             config['plane_key_size']+config['agent_features'],
                                             config['embed_size'])
        self.cargo_up_projection = nn.Linear(config['node_features']*2+\
                                             config['plane_key_size']+config['cargo_features'],
                                             config['embed_size'])
        self.node_up_projection_norm = nn.LayerNorm(config['embed_size'])
        self.plane_up_projection_norm = nn.LayerNorm(config['embed_size'])
        self.cargo_up_projection_norm = nn.LayerNorm(config['embed_size'])

        # Self-attention layers
        self.MHA_nodes = MHABlock(config['MHA_nodes'])
        self.MHA_planes = MHABlock(config['MHA_planes'])
        self.MHA_cargo = MHABlock(config['MHA_cargo'])

        # Mixing Attention layers
        self.mixing_attention = MixingAttentionBlock(config['MixingAttention'])

    def forward(self, x):
        """
        Forward pass of the encoder.
        Takes in the semi-structured graph, cargo dict, and plane dict.
        Generates embeddings for each node and then uses those embeddings to complete
        the embeddings for each plane and cargo.

        Args:
            x (dict): The semi-structred graph, cargo dict, and plane dict

        Returns: {
            cargo_embeddings: tensor (c x f) embeddings for cargo
            plane_embeddings: tensor (p x f) embeddings for planes
            node_embeddings: tensor (n x f) embeddings for nodes
        }
        """

        # NOTE: I think nodes are 1-indexed in the location/destination fields of the agents and cargo

        nodes_geom_obj = x['nodes']
        agents_dict_list = x['agents']['agents']
        cargo_dict_list = x['cargo']['cargo']

        # Get node embeddings from GAT
        node_embeddings = self.GAT(nodes_geom_obj) # (n x config[GAT][node_embed_size])
        # node_embeddings = x['nodes'].x


        # Stitch together plane embeddings
        plane_embeddings = []
        cargo_onboard = {}
        for agent in agents_dict_list:
            # Generate a random plane key for the network to use to identify plane-cargo pairs
            plane_key = torch.rand(self.config['plane_key_size'])
            location_embedding = node_embeddings[agent['location']-1]
            agent_tensor = torch.cat([plane_key, location_embedding, agent['tensor']], dim=0)
            for cargo_id in agent['cargo_onboard']:
                cargo_onboard[cargo_id] = plane_key
            plane_embeddings.append(agent_tensor)
        plane_embeddings = torch.stack(plane_embeddings)
        # (p x config[GAT][node_embed_size]+config[plane_key_size]+config[agent_features])
        # plane_embeddings(p x f) = [[plane_key, location_embedding, state_one_hot, current_weight, max_weight]...]


        # Stitch together cargo embeddings
        cargo_embeddings = []
        for cargo in cargo_dict_list:
            cargo_tensor = cargo['tensor']
            location_embedding = node_embeddings[cargo['location']-1]
            destination_embedding = node_embeddings[cargo['destination']-1]
            if cargo['id'] in cargo_onboard:
                plane_key = cargo_onboard[cargo['id']]
            else:
                plane_key = torch.ones(self.config['plane_key_size'])
            cargo_tensor = torch.cat([plane_key, location_embedding, destination_embedding, cargo_tensor], dim=0)
            cargo_embeddings.append(cargo_tensor)
        cargo_embeddings = torch.stack(cargo_embeddings)
        # (c x 2*config[GAT][node_embed_size]+config[plane_key_size]+config[cargo_features])
        # cargo_embeddings(c x f) = [[plane_key, location_embedding, destination_embedding, weight, 
        #                             earliest_pickup_time, soft_deadline, hard_deadline]...]


        # Up-project the embeddings
        node_embeddings = self.node_up_projection(node_embeddings)
        plane_embeddings = self.plane_up_projection(plane_embeddings)
        cargo_embeddings = self.cargo_up_projection(cargo_embeddings)
        node_embeddings = self.node_up_projection_norm(node_embeddings)
        plane_embeddings = self.plane_up_projection_norm(plane_embeddings)
        cargo_embeddings = self.cargo_up_projection_norm(cargo_embeddings)


        # Self-attention layers
        node_embeddings = self.MHA_nodes(node_embeddings, node_embeddings, node_embeddings)
        plane_embeddings = self.MHA_planes(plane_embeddings, plane_embeddings, plane_embeddings)
        cargo_embeddings = self.MHA_cargo(cargo_embeddings, cargo_embeddings, cargo_embeddings)


        # Mixing Attention layers
        node_embeddings, plane_embeddings, cargo_embeddings = \
            self.mixing_attention(node_embeddings, plane_embeddings, cargo_embeddings)

        return {
            'cargo_embeddings': cargo_embeddings,
            'plane_embeddings': plane_embeddings,
            'node_embeddings': node_embeddings
        }



class Actor(nn.Module):
    """
    Actor for the Airlift solution.
    """

    def __init__(self, config):
        super(Actor, self).__init__()
        self.config = config

        # Define the layers of the model

        # Encoder
        self.encoder = Encoder(config['encoder'])

        # Attention: planes to nodes
        self.MHA_NPP1 = []
        for _ in range(config['MHA']['num_layers']):
            self.MHA_NPP1.append(MHA(config['MHA']))
        self.MHA_NPP1 = nn.ModuleList(self.MHA_NPP1)
        # Attention: cargo to planes
        self.MHA_PCC1 = []
        for _ in range(config['MHA']['num_layers']):
            self.MHA_PCC1.append(MHA(config['MHA']))
        self.MHA_PCC1 = nn.ModuleList(self.MHA_PCC1)
        # Attention: nodes to cargo
        self.MHA_CNN1 = []
        for _ in range(config['MHA']['num_layers']):
            self.MHA_CNN1.append(MHA(config['MHA']))
        self.MHA_CNN1 = nn.ModuleList(self.MHA_CNN1)


        # CPtr: planes to nodes
        self.CPtr_PN = CPtr(config['CPtr'])
        # Context: planes to nodes
        self.context_PN = Context(config['MHA'])

        # Attention: cargo to planes
        self.MHA_PCC2 = []
        for _ in range(config['MHA']['num_layers']):
            self.MHA_PCC2.append(MHA(config['MHA']))
        self.MHA_PCC2 = nn.ModuleList(self.MHA_PCC2)
        # Attention planes to cargo
        self.MHA_CPP2 = []
        for _ in range(config['MHA']['num_layers']):
            self.MHA_CPP2.append(MHA(config['MHA']))
        self.MHA_CPP2 = nn.ModuleList(self.MHA_CPP2)

        # CPtr: cargo to planes
        self.CPtr_CP = CPtr(config['CPtr'])

    def process_logits(self, logits):
        """
        Process the logits into actions.

        Args:
            logits: tensor logits

        Returns:
            logits: tensor logits wighed by the choice selected
        """
        argmax = torch.argmax(logits, dim=1)
        ptr_mtx = torch.ones_like(logits)
        ptr_mtx[torch.arange(logits.size(0)), argmax] = logits.size(0)*100
        ptr_mtx = ptr_mtx / ptr_mtx.sum(dim=1, keepdim=True)
        logits = logits * ptr_mtx
        return logits

    def forward(self, obs):
        """
        Forward pass of the actor.

        Args:
            x: dict of semi-structured graph, cargo dict, and plane dict

        Returns: {
            plane_assignments_mtx: tensor (p x n) routing assignments for planes
            cargo_assignments_mtx: tensor (c x p) assignment of cargo to planes
            n1: tensor (n x f) for skip connection to critic
            p1: tensor (p x f) for skip connection to critic
            p2: tensor (p x f) for skip connection to critic
            c2: tensor (c x f) for skip connection to critic
        }
        """

        # Encoder
        x = self.encoder(obs)
        ci, pi, ni = x['cargo_embeddings'], x['plane_embeddings'], x['node_embeddings']


        # First set of MHA layers
        pp, cp, np = pi, ci, ni
        for NPP, PCC, CNN in zip(self.MHA_NPP1, self.MHA_PCC1, self.MHA_CNN1):
            n1 = NPP(np, pp, pp)
            p1 = PCC(pp, cp, cp)
            c1 = CNN(cp, np, np)
            np, pp, cp = n1, p1, c1
        del np, pp, cp

        # CPtr: planes to nodes
        ptr_mtx_pn = self.CPtr_PN([p1], [n1], mask=obs['agents']['mask'])
        # Assignments: planes to nodes
        ptr_mtx_pn = self.process_logits(ptr_mtx_pn)

        # Context: planes to nodes
        p2 = self.context_PN(ptr_mtx_pn, p1, n1)

        # Second set of MHA layers
        pp, cp = p2, c1
        for PCC, CPP in zip(self.MHA_PCC2, self.MHA_CPP2):
            p2 = PCC(pp, cp, cp)
            c2 = CPP(cp, pp, pp)
            pp, cp = p2, c2
        del pp, cp

        # CPtr: cargo to planes
        # Append a row of 1s to represent the choice of not choosing a plane to p2
        p2_m = torch.cat((p2, torch.ones(1, p2.size(1))), dim=0)
        # Mask
        cargo_mask = obs['cargo']['mask']
        cargo_mask = torch.cat((cargo_mask, torch.zeros(cargo_mask.size(0), 1)), dim=1)
        ptr_mtx_cp = self.CPtr_CP([c2], [p2_m], mask=cargo_mask)
        # Assignments: cargo to planes
        ptr_mtx_cp = self.process_logits(ptr_mtx_cp)

        return {
            'plane_assignments_mtx': ptr_mtx_pn,
            'cargo_assignments_mtx': ptr_mtx_cp,
            'n1': n1,
            'p1': p1,
            'p2': p2,
            'c2': c2
        }
    


class Critic(nn.Module):
    """
    Critic for the Airlift solution.
    """

    def __init__(self, config):
        super(Critic, self).__init__()
        self.config = config

        # Define the layers of the model

        # Encoder
        self.encoder = Encoder(config['encoder'])


        # CPtr: planes to nodes
        self.CPtr_PN = CPtr(config['CPtr_2'])


        # --- Cargo Reward Block
        # Context:
        self.context_PN_cargo = Context(config['MHA'])
        self.context_NP_cargo = Context(config['MHA'])

        # Ptr: planes to nodes
        self.CPtr_PN_cargo = CPtr(config['CPtr'])


        # CPtr: Planes to nodes 2
        self.CPtr_PN_cargo_2 = CPtr(config['CPtr'])
        
        # Context:
        self.context_CP_cargo = Context(config['MHA'])
        self.context_PC_cargo = Context(config['MHA'])

        # CPtr: cargo to planes
        self.CPtr_CP_cargo21 = CPtr(config['CPtr_2'])
        self.CPtr_CP_cargo22 = CPtr(config['CPtr'])

        # FF layers
        self.context_CP_cargo_out = Context(config['MHA'])
        self.ff_cargo = nn.Linear(config['MHA']['embed_dim'], 1)


        # --- Plane Reward Block
        # Context: planes to nodes
        self.context_PN_plane = Context(config['MHA'])
        self.context_NP_plane = Context(config['MHA'])

        # Attention: planes to nodes
        self.MHA_PNN_plane = MHA(config['MHA_out'])

        # Feed forward output layers
        self.ff_planes = nn.Linear(config['MHA']['embed_dim'], 1)

    def forward(self, obs, actor_output):
        """
        Forward pass of the critic.

        Args:
            obs: dict of semi-structured graph, cargo dict, and plane dict
            actor_output: {
                node_assignments: tensor (#planes x #nodes) routing assignments for planes
                cargo_assignments: tensor (#cargo x #planes) assignment of cargo to planes
                n1: tensor (n x f) for skip connection to critic
                p1: tensor (p x f) for skip connection to critic
                p2: tensor (p x f) for skip connection to critic
                c2: tensor (c x f) for skip connection to critic
            }

        Returns: {
            planes_rwd: tensor (#planes x 1) value of the state
            cargo_rwd: tensor (#cargo x 1) value of the state
        }
        """

        # Unpack the actor output
        node_assignments = actor_output['plane_assignments_mtx']
        cargo_assignments = actor_output['cargo_assignments_mtx']
        an = actor_output['n1']
        ap1 = actor_output['p1']
        ap2 = actor_output['p2']
        ac = actor_output['c2']

        # Fix cargo assignment dimention by deleting the last col
        cargo_assignments = cargo_assignments[:, :-1]

        # Encoder
        x = self.encoder(obs)
        c, p, n = x['cargo_embeddings'], x['plane_embeddings'], x['node_embeddings']


        # CPtr: planes to nodes
        pn1 = self.CPtr_PN([p, ap1], [n, an])
        del an, ap1

        # Mask with the assignments
        pn1 = pn1 * node_assignments
        del node_assignments


        # --- Cargo Reward Block

        # Context & CPtr: planes to nodes
        pf11 = self.context_PN_cargo(pn1, p, n)
        nf11 = self.context_NP_cargo(pn1.T, n, p)
        pn1 = self.CPtr_PN_cargo([pf11], [nf11])
        del pf11

        # CPtr: cargo to planes
        cp2 = self.CPtr_CP_cargo21([c, ac], [p, ap2])
        del ap2, ac

        # Mask with the assignments
        cp2 = cp2 * cargo_assignments
        del cargo_assignments

        # Context & CPtr
        cf2 = self.context_CP_cargo(cp2, c, p)
        pf2 = self.context_PC_cargo(cp2.T, p, c)
        cp2 = self.CPtr_CP_cargo22([cf2], [pf2])
        del c, pf2

        # MatMul pn, cp -> cn
        cn = torch.matmul(cp2, pn1) / cp2.size(-1)**0.5
        del cp2

        # FF layers
        cargo_rwd = self.context_CP_cargo_out(cn, cf2, nf11)
        cargo_rwd = self.ff_cargo(cargo_rwd)
        del cn, cf2, nf11


        # --- Plane Reward Block

        # Context & Ptr: planes to nodes
        pf12 = self.context_PN_plane(pn1, p, n)
        nf12 = self.context_NP_plane(pn1.T, n, p)
        pn1 = self.MHA_PNN_plane(pf12, nf12, nf12)
        del p, n, pf12, nf12

        # FF layers
        plane_rwd = self.ff_planes(pn1)

        return {
            'planes_rwd': plane_rwd,
            'cargo_rwd': cargo_rwd
        }
    

class MHA(nn.Module):
    """
    Attention layer for the Airlift solution.
    """

    def __init__(self, config):
        super(MHA, self).__init__()
        self.config = config

        # Define the layers of the model
        self.attention = nn.MultiheadAttention(config['embed_dim'], config['num_heads'],
                                                     bias=config['bias'])
        self.ff = nn.Sequential(
            nn.Linear(config['embed_dim'], config['ff_dim']),
            config['non_linearity'],
            nn.Linear(config['ff_dim'], config['embed_dim'])
        )
        self.norm1 = nn.LayerNorm(config['embed_dim'])
        self.norm2 = nn.LayerNorm(config['embed_dim'])


    def forward(self, q, k, v):
        """
        Forward pass of the attention layer.

        Args:
            k: tensor (#k x #f) keys
            q: tensor (#q x #f) queries
            v: tensor (#v x #f) values

        Returns:
            r: tensor (#q x #f) attended values
        """

        v, _ = self.attention(q, k, v)
        v = self.norm1(v + q)
        v = self.norm2(v + self.ff(v))

        return v



class Context(nn.Module):
    """
    Context layer
    The weight and distribution multiplications inside attention
    """

    def __init__(self, config):
        super(Context, self).__init__()
        self.config = config

        # Define the layers of the model

        self.W = nn.Linear(config['embed_dim'], config['embed_dim'])
        self.ff = nn.Sequential(
            nn.Linear(config['embed_dim'], config['ff_dim']),
            config['non_linearity'],
            nn.Linear(config['ff_dim'], config['embed_dim'])
        )
        self.norm1 = nn.LayerNorm(config['embed_dim'])
        self.norm2 = nn.LayerNorm(config['embed_dim'])

    def forward(self, att_mtx, q, v):
        """
        Forward pass of the context layer.

        Args:
            att_mtx: tensor - softmax distribution from ptr
            v: tensor - embeddings of the entity

        Returns:
            v: tensor - attended embeddings
        """

        v = torch.matmul(att_mtx, self.W(v))
        v = self.norm1(v + q)
        v = self.norm2(v + self.ff(v))

        return v



class MHABlock(nn.Module):
    """
    Multiple layers of MHA for the Airlift solution.
    """

    def __init__(self, config):
        super(MHABlock, self).__init__()
        self.config = config

        # Define the layers of the model

        # MHA layers x4
        self.layers = []
        for _ in range(config['num_layers']):
            self.layers.append(MHA(config))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, q, k, v):
        """
        Forward pass of the MHABlock.

        Args:
            k: tensor (#k x #f) keys
            q: tensor (#q x #f) queries
            v: tensor (#v x #f) values

        Returns:
            r: tensor (#q x #f) attended values
        """

        for layer in self.layers:
            if self.config['self_attention']:
                v = layer(v, v, v)
            else:
                v = layer(q, k, v)

        return v



class CPtr(nn.Module):
    """
    Pointer Network With Context
    """

    def __init__(self, config):
        super(CPtr, self).__init__()
        self.config = config

        # xc = Wx0X0 + Wx1X1 + ... + Bx
        self.Wx_layers = []
        for _ in range(config['num_contexts_x']):
            self.Wx_layers.append(nn.Linear(config['embed_dim'], config['embed_dim']))
        self.Bx = nn.Parameter(torch.rand(1, config['embed_dim']))

        # yc = Wy0Y0 + Wy1Y1 + ... + By
        self.Wy_layers = []
        for _ in range(config['num_contexts_y']):
            self.Wy_layers.append(nn.Linear(config['embed_dim'], config['embed_dim']))
        self.By = nn.Parameter(torch.rand(1, config['embed_dim']))

    def forward(self, x, y, mask=None):
        """
        Forward pass of the PNWC layer.

        Args:
            x (list): list of tensors for the first entity
            y (list): list of tensors for the second entity
            mask (tensor): mask for the attention (additive mask)

        Returns:
            ptr_mtx: tensor (#x x #y) attention weights
        """

        # Assert that the number of contexts = input length
        assert len(self.Wx_layers) == len(x)
        assert len(self.Wy_layers) == len(y)

        # xc = Wx0X0 + Wx1X1 + ... + Bx
        xc = sum(
            layer(x[i]) for i, layer in enumerate(self.Wx_layers)
            ) + self.Bx

        # yc = Wy0Y0 + Wy1Y1 + ... + By
        yc = sum(
            layer(y[i]) for i, layer in enumerate(self.Wy_layers)
            ) + self.By

        # ptr = softmax((xc * ycT) / sqrt(d_k))
        ptr_mtx = torch.matmul(xc, yc.T) / xc.size(-1)**0.5

        if mask is not None:
            ptr_mtx = ptr_mtx + mask

        # softmax
        if self.config['softmax']:
            ptr_mtx = F.softmax(ptr_mtx, dim=-1)

        return ptr_mtx
    


class MixingAttention(nn.Module):
    """
    Mixing Attention layer for the Airlift solution.
    """

    def __init__(self, config):
        super(MixingAttention, self).__init__()
        self.config = config

        # Define the layers of the model

        # attn(X, Y, Y)
        self.MHA_XYY = MHA(config)
        # attn(X, Z, Z)
        self.MHA_XZZ = MHA(config)
        # WX
        self.WX = nn.Linear(config['embed_dim']*2, config['embed_dim'])
        # LayerNorm
        self.norm_x = nn.LayerNorm(config['embed_dim'])

        # attn(Y, X, X)
        self.MHA_YXX = MHA(config)
        # attn(Y, Z, Z)
        self.MHA_YZZ = MHA(config)
        # WY
        self.WY = nn.Linear(config['embed_dim']*2, config['embed_dim'])
        # LayerNorm
        self.norm_y = nn.LayerNorm(config['embed_dim'])

        # attn(Z, X, X)
        self.MHA_ZXX = MHA(config)
        # attn(Z, Y, Y)
        self.MHA_ZYY = MHA(config)
        # WZ
        self.WZ = nn.Linear(config['embed_dim']*2, config['embed_dim'])
        # LayerNorm
        self.norm_z = nn.LayerNorm(config['embed_dim'])

    def forward(self, x, y, z):
        """
        Forward pass of the mixing attention layer.

        Args:
            x, y, z: tensor embeddings of three related entities

        Returns:
            X, Y, Z: tensors attended to by the other two
        """

        # X = concat(attn(X, Y, Y), attn(X, Z, Z))WX
        X = torch.cat((self.MHA_XYY(x, y, y), self.MHA_XZZ(x, z, z)), dim=1)
        X = self.WX(X)
        X = self.norm_x(self.config['non_linearity'](X))

        # Y = concat(attn(Y, X, X), attn(Y, Z, Z))WY
        Y = torch.cat((self.MHA_YXX(y, x, x), self.MHA_YZZ(y, z, z)), dim=1)
        Y = self.WY(Y)
        Y = self.norm_y(self.config['non_linearity'](Y))
        
        # Z = concat(attn(Z, X, X), attn(Z, Y, Y))WZ
        Z = torch.cat((self.MHA_ZXX(z, x, x), self.MHA_ZYY(z, y, y)), dim=1)
        Z = self.WZ(Z)
        Z = self.norm_z(self.config['non_linearity'](Z))

        return X, Y, Z


class MixingAttentionBlock(nn.Module):
    """
    Layers of MixingAttention for the Airlift solution.
    """

    def __init__(self, config):
        super(MixingAttentionBlock, self).__init__()
        self.config = config

        # Define the layers of the model

        # Mixing Attention layers x4
        self.layers = []
        for _ in range(config['num_layers']):
            self.layers.append(MixingAttention(config))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x, y, z):
        """
        Forward pass of the MixingAttentionBlock.

        Args:
            x, y, z: tensor embeddings of three related entities

        Returns:
            X, Y, Z: tensors attended to by the other two
        """

        for layer in self.layers:
            x, y, z = layer(x, y, z)

        return x, y, z



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

        in_channels = config['node_features']
        hidden_channels = config['hidden_features']
        out_channels = config['node_embed_size']
        num_layers = config['num_layers']
        edge_dim = config['edge_features']

        # Define the layers of the model
        self.layers = []
        self.norms = []
        if num_layers == 1:
            self.layers.append(GATConv(in_channels, out_channels, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(out_channels))
        else:
            for i in range(num_layers-1):
                in_channels = in_channels if i == 0 else hidden_channels
                self.layers.append(GATConv(in_channels, hidden_channels, edge_dim=edge_dim))
                self.norms.append(nn.LayerNorm(hidden_channels))
            self.layers.append(GATConv(hidden_channels, out_channels, edge_dim=edge_dim))
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




def get_config(version):
    """
    Get the configuration for the model.

    Args:
        version: str, version of the model

    Returns:
        config: dict, configuration for the model
    """

    # VERSION 1---------------------------------------------------------------
    if version == 'v1':
        # Define the model configuration
        embed_dim = 32
        node_embed = 8
        non_linearity = torch.nn.LeakyReLU(0.2)

        config_encoder_MHA = {
            'non_linearity': non_linearity,
            'num_heads': 4,
            'num_layers': 4,
            'embed_dim': embed_dim,
            'ff_dim': 256,
            'bias': False,
            'self_attention': True,
        }

        config_encoder_MixingAttention = config_encoder_MHA

        config_encoder = {
            'non_linearity': non_linearity,
            'GAT': {
                'non_linearity': non_linearity,
                'node_features': 3,
                'hidden_features': 64,
                'node_embed_size': node_embed,
                'num_layers': 2,
                'edge_features': 5
            },
            'plane_key_size': 16,
            'agent_features': 6, # one hot encoded state (4), weight, max weight
            'cargo_features': 4,
            'node_features': node_embed,
            'embed_size': embed_dim,
            'MHA_nodes': config_encoder_MHA,
            'MHA_planes': config_encoder_MHA,
            'MHA_cargo': config_encoder_MHA,
            'MixingAttention': config_encoder_MixingAttention
        }

        return {
            'actor': {
                'encoder': config_encoder,
                'MHA': config_encoder_MHA,
                'CPtr': {
                    'embed_dim': embed_dim,
                    'num_contexts_x': 1,
                    'num_contexts_y': 1,
                    'softmax': True,
                }
            },
            'critic': {
                'encoder': config_encoder,
                'CPtr': {
                    'embed_dim': embed_dim,
                    'num_contexts_x': 1,
                    'num_contexts_y': 1,
                    'softmax': False,
                },
                'CPtr_2': {
                    'embed_dim': embed_dim,
                    'num_contexts_x': 2,
                    'num_contexts_y': 2,
                    'softmax': False,
                },
                'MHA': config_encoder_MHA,
                'MHA_out': config_encoder_MHA,
                'out_ff': {
                    'embed_dim': embed_dim,
                }
            }
        }