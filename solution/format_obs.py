import networkx as nx
import torch
from torch_geometric.utils import from_networkx

WAITING = 0
PROCESSING = 1
MOVING = 2
READY_FOR_TAKEOFF = 3

DESTINATION_ACTION = 0
LOAD_UNLOAD_ACTION = 1
NOOP_ACTION = 2

def format_obs(obs, t, max_airport_capacity=5e+09, max_weight=10000, max_time=10000, prev_x=None):
    """
    Format the observation into a format that can be used by the model.
    Needs to have the previous timestep's cargo and airplanes to determine cargo's location
    and airplane's ETA since the env makes loading cargo disappear.

    Args:
        obs (dict): observation from the environment
        t (int): current timestep
        max_airport_capacity (float): maximum airport capacity for normalization
        max_weight (float): maximum weight for normalization
        max_time (float): maximum time for normalization
        prev_x (dict): previous timestep's formatted observation
    
    Return: {
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
    """

    if prev_x is not None:
        prev_cargo = prev_x['cargo']['cargo']
        prev_airplanes = prev_x['agents']['agents']
    else:
        prev_cargo = {}
        prev_airplanes = {}

    # Extract the global state
    gs = obs['a_0']['globalstate']


    """
    globalstate(dict: 6 items)
      route_map(dict: 1 items)
        0(<class 'networkx.classes.digraph.DiGraph'>)
      plane_types(list: 1 items)(<class 'airlift.envs.airlift_env.PlaneTypeObservation'>)
      agents(dict: 2 items)
        ...
      active_cargo(list: 40 items)(<class 'airlift.envs.airlift_env.CargoObservation'>)
      event_new_cargo(list: 0 items)(empty)
      scenario_info(list: 1 items)(<class 'airlift.envs.airlift_env.ScenarioObservation'>)
    """

    route_map = gs['route_map'][0]
    plane_types = gs['plane_types']
    agents_dict = gs['agents']
    active_cargo = gs['active_cargo']
    event_new_cargo = gs['event_new_cargo']
    scenario_info = gs['scenario_info']


    # Unpack agents and plane types
    """
    agents_dict(dict: #planes items)
        a_0(dict: 10 items)
          cargo_onboard(list: 0 items)(empty)
          state(<enum 'PlaneState'>)
          plane_type(<class 'int'>)
          current_weight(<class 'int'>)
          max_weight(<class 'int'>)
          available_routes(list: 5 items)(<class 'int'>)
          next_action(dict: 4 items)
            priority(<class 'int'>)
            cargo_to_load(list: 0 items)(empty)
            cargo_to_unload(list: 0 items)(empty)
            destination(<class 'int'>)
          current_airport(<class 'int'>)
          cargo_at_current_airport(list: 0 items)(empty)
          destination(<class 'int'>)
    """
    
    # Process graph into torch_geometric.data.Data format for GCN
    for node, data in route_map.nodes(data=True):
        route_map.nodes[node]['x'] = torch.tensor([data['pos'][0], data['pos'][1],
                                                   data['working_capacity']], dtype=torch.float32)
    for u, v, data in route_map.edges(data=True):
        ems = data['expected_mal_steps'] if 'expected_mal_steps' in data else 0
        data['expected_mal_steps'] = data['expected_mal_steps'] if 'expected_mal_steps' in data else 0
        ra = 1 if data['route_available'] else 0
        edge_attr = torch.tensor([data['cost'], data['time'], data['mal'], ems, ra], dtype=torch.float32)
        route_map.edges[u, v]['edge_attr'] = edge_attr
    torch_gr = from_networkx(route_map)
    keep = ['x', 'edge_attr', 'edge_index']
    for key in torch_gr.keys():
        if key not in keep:
            del torch_gr[key]
    # torch_gr = Data(x=[8, 3], edge_index=[2, 32], edge_attr=[32, 5])
    torch_gr.x[:, 2] /= max_airport_capacity



    cargo_list = CargoList(prev_cargo=prev_cargo)
    for c in active_cargo:
        cargo_list.add_cargo(c)
    
    # # If there is nothing to do, return None
    # if cargo_list.cargo.keys()==[]:
    #     return None
    
    airplane_list = AirplaneList()
    for k, a in obs.items():
        airplane_list.add_airplane(a, k, cargo_list, prev_airplanes=prev_airplanes)

    airport_list = {}
    for node, data in route_map.nodes(data=True):
        airport = Airport()
        airport.id = node
        airport.capacity = data['working_capacity']
        airport.coords = data['pos']
        airport_list[node] = airport

    # Agents
    agents_map = []
    agents_tensor = []
    agents_mask = []
    action_mask = []
    for _, airplane in airplane_list.airplanes.items():
        agents_map.append(airplane.id)

        # Agents Tensor
        state_one_hot = lambda n, size: [1 if i == n else 0 for i in range(size)]

        agents_tensor.append([
            airport_list[airplane.location].coords[0],
            airport_list[airplane.location].coords[1]
        ] + state_one_hot(airplane.state, 4) + [
            airplane.current_weight/max_weight,
            airplane.max_weight/max_weight,
            airplane.eta/max_time
        ])

        # Action Mask
        if airplane.state != READY_FOR_TAKEOFF:
            action_mask.append([-float('inf'), -float('inf'), 0])
        elif airplane.cargo_at_current_airport == [] and airplane.cargo == {}:
            # If there is no cargo at the airport and no cargo on the plane, mask unload/load option
            action_mask.append([0, -float('inf'), 0])
        else:
            action_mask.append([0, 0, 0])
            

        # Agents Mask
        if airplane.state == READY_FOR_TAKEOFF: # 1-based indexing
            agents_mask.append([0 if (i+1) in airplane.available_routes else -float('inf') for i in range(len(airport_list))])
        else:
            agents_mask.append([0 if (i+1) == airplane.location else -float('inf') for i in range(len(airport_list))])

    agents_tensor = torch.tensor(agents_tensor, dtype=torch.float32)
    agents_mask = torch.tensor(agents_mask, dtype=torch.float32)
    action_mask = torch.tensor(action_mask, dtype=torch.float32)

    # Cargo
    cargo_map = []
    cargo_tensor = []
    cargo_mask = []
    cargo_idx = 0
    for _, cargo in cargo_list.cargo.items():
        cargo_map.append(cargo.id)
        cargo_list.cargo[cargo.id].idx = cargo_idx
        cargo_idx += 1

        # Cargo Tensor
        location = cargo.location if cargo.on_a_plane == 0 else cargo.location.location

        cargo_tensor.append([
            airport_list[location].coords[0],
            airport_list[location].coords[1],
            airport_list[cargo.destination].coords[0],
            airport_list[cargo.destination].coords[1],
            cargo.on_a_plane,
            cargo.processing,
            cargo.weight/max_weight,
            (cargo.earliest_pickup_time - t)/max_time,
            (cargo.soft_deadline - t)/max_time,
            (cargo.hard_deadline - t)/max_time,
            cargo.location.eta/max_time if cargo.on_a_plane == 1 else 0
        ])

        # Cargo Mask
        if t < cargo.earliest_pickup_time:
            if cargo.on_a_plane == 1:
                cargo_mask.append([1 if airplane_list.airplanes[a]==cargo.location else -float('inf') for a in agents_map] + [1])
            else:
                cargo_mask.append([1 if airplane_list.airplanes[a].location==cargo.location else -float('inf') for a in agents_map] + [1])
        else:
            cargo_mask.append([-float('inf') for _ in range(len(agents_map))] + [1])
    
    cargo_tensor = torch.tensor(cargo_tensor, dtype=torch.float32)
    cargo_mask = torch.tensor(cargo_mask, dtype=torch.float32)


    return {
        'nodes': {
            'PyGeom': [torch_gr],
            'nodes': airport_list,
        },
        'agents': {
            'map': agents_map,
            'agents': airplane_list.airplanes,
            'tensor': agents_tensor.unsqueeze(0),
            'destination_mask': agents_mask.unsqueeze(0),
            'action_mask': action_mask.unsqueeze(0)
        },
        'cargo': {
            'map': cargo_map,
            'cargo': cargo_list.cargo,
            'tensor': cargo_tensor.unsqueeze(0),
            'mask': cargo_mask.unsqueeze(0)
        }
    }


class Airport:

    def __init__(self):
        self.id = None
        self.capacity = None
        self.coords = None



class Airplane:

    def __init__(self):
        self.state = None
        self.id = None
        self.cargo = {}
        self.max_weight = None
        self.current_weight = None
        self.location = None
        self.available_routes = []
        self.cargo_at_current_airport = None
        self.eta = 0


class AirplaneList:

    def __init__(self):
        self.airplanes = {}

    def add_airplane(self, a, id, cargo_list, prev_airplanes={}):
        if id not in self.airplanes.keys():
            self.airplanes[id] = Airplane()
            self.airplanes[id].state = a['state']
            self.airplanes[id].id = id
            self.airplanes[id].max_weight = a['max_weight']
            self.airplanes[id].current_weight = a['current_weight']
            # if flying, use destination as location
            self.airplanes[id].location = a['destination'] if a['state'] == MOVING else a['current_airport']
            self.airplanes[id].available_routes = a['available_routes'] + [self.airplanes[id].location]
            cargo_list.add_airplane(self.airplanes[id], a['cargo_onboard'])
            self.airplanes[id].cargo_at_current_airport = a['cargo_at_current_airport']
            if id in prev_airplanes.keys() and a['state'] == MOVING:
                self.airplanes[id].eta = prev_airplanes[id].eta - 1


class Cargo:

    def __init__(self):
        self.id = None
        self.idx = None
        self.destination = None
        self.soft_deadline = None
        self.hard_deadline = None
        self.weight = None
        self.earliest_pickup_time = None
        self.location = None
        self.on_a_plane = 0
        self.processing = 0


class CargoList:

    def __init__(self, prev_cargo):
        self.cargo = {}
        self.prev_cargo = prev_cargo

    def add_airplane(self, a, cargo_onboard):
        for c in cargo_onboard:
            self.cargo[c].location = a
            self.cargo[c].on_a_plane = 1
            a.cargo[c] = self.cargo[c]
    
    def add_cargo(self, c,):
        if c.id not in self.cargo.keys():
            self.cargo[c.id] = Cargo()
            self.cargo[c.id].id = c.id
            self.cargo[c.id].destination = c.destination
            self.cargo[c.id].soft_deadline = c.soft_deadline
            self.cargo[c.id].hard_deadline = c.hard_deadline
            self.cargo[c.id].weight = c.weight
            self.cargo[c.id].earliest_pickup_time = c.earliest_pickup_time
            if c.location!=0:
                self.cargo[c.id].location = c.location
            elif c.id in self.prev_cargo.keys():
                self.cargo[c.id].location = self.prev_cargo[c.id].location
                self.cargo[c.id].processing = 1
            else:
                raise ValueError('Cargo {c.id} spawned off-map.')

