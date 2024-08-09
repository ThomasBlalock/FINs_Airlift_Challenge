import torch

DESTINATION_ACTION = 0
LOAD_UNLOAD_ACTION = 1
NOOP_ACTION = 2

def post_process(x, y):
    """
    Post-process the action logits into valid actions.

    Args:
        x: { # Output from the obs preprocessor
            'nodes': {
                'PyGeom': torch_geometric.data.Data, PyTorch Geometric Data object
                'nodes': dict of Airports
            }
            'agents': {
                'map': list, map[i] = agent_id for the i-th agent in the tensor
                'agents': dict of Airplane objects,
                'tensor': torch.Tensor, tensor[i] = agent_i features
                'mask': torch.Tensor, mask[i, j] = 0 if agent_i can go to airport_j+1, -inf otherwise
            },
            'cargo': {
                'map': list, map[i] = cargo_id for the i-th cargo in the tensor
                'cargo': dict of Cargo objects, 
                'tensor': torch.Tensor, tensor[i] = cargo_i features
                'mask': torch.Tensor, mask[i, j] = 0 if cargo_i can be loaded onto agent_j, -inf otherwise
        }
        y: dict of tensors { # Output from the model
            'actions': actions,
            'action_logits': action_logits,
            'destinations': destinations,
            'destination_logits': destination_logits,
            'cargo': cargo,
            'cargo_logits': cargo_logits
        }

    returns: actions: (dict) {agent: action, ...},
    """

    y['actions'] = y['actions'][0]
    y['destinations'] = y['destinations'][0]
    y['cargo'] = y['cargo'][0]

    y['cargo'] = y['cargo'].T
    actions = {}
    num_planes = x['agents']['tensor'].shape[0]
    
    for p in range(x['agents']['tensor'].shape[1]):
        agent_id = x['agents']['map'][p]
        action_type = torch.argmax(y['actions'][p]).item()

        if action_type == NOOP_ACTION:
            actions[agent_id] = {
                'priority': 0,
                'cargo_to_load': [],
                'cargo_to_unload': [],
                'destination': 0
            }
        elif action_type == DESTINATION_ACTION:
            actions[agent_id] = {
                'priority': 1,
                'cargo_to_load': [],
                'cargo_to_unload': [],
                'destination': torch.argmax(y['destinations'][p]).item()+1 # Convert to 1-indexed
            }
        elif action_type == LOAD_UNLOAD_ACTION:
            cargo_onboard = x['agents']['agents'][agent_id].cargo
            actions[agent_id] = {
                'priority': 1,
                'cargo_to_load': [x['cargo']['map'][c] for c in range(y['cargo'][p].shape[0]) if y['cargo'][p, c].item() > 0],
                'cargo_to_unload': [c.id for c in cargo_onboard if y['cargo'][c.idx, num_planes].item() > 0 ],
                'destination': 0
            }

    return actions