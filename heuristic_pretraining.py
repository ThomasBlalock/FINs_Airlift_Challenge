# Imports

# Environment
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs import PlaneType
from airlift.envs.generators.map_generators import PlainMapGenerator

# Generators
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator

# Dynamic events
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator

# Solutiona
from solution.mysolution import MySolution
from solution.heuristic_model.mysolution import MySolution as HeuristicSolution

# Helper methods
from airlift.solutions import doepisode
from eval_solution import write_results
from solution.models.v4.model_v4 import Policy
from solution.format_obs import format_obs
from solution.post_process import post_process
import torch
import torch.nn as nn
import random
from torch.distributions import Categorical
from tqdm import tqdm
import networkx as nx
import pickle
from torch_geometric.data import Data

DESTINATION_ACTION = 0
LOAD_UNLOAD_ACTION = 1
NOOP_ACTION = 2

class Scheduler:
    """
    This class is responsible for creating the environment that the agent will interact with.
    The plan is to maybe make the environments progressively larger as training progresses (hence the stage param).
    Right now, it just returns a fixed environment.
    """
    def __init__(self, seed=0, num_epochs=10):
        self.seed = seed


    def small_env_with_purturbations(self, max_cycles=5000):
        """
        Returns a preset environment
        - 4 nodes
        - 2 airplane
        - purturbations present
        """

        # Decide plane types
        plane_types = [PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)]

        # Decide airport generator
        airport_generator = RandomAirportGenerator(
            max_airports=5,
            make_drop_off_area=True,
            make_pick_up_area=True,
            num_drop_off_airports=1,
            num_pick_up_airports=1,
            mapgen=PlainMapGenerator(),
        )

        # Decide route generator
        route_generator = RouteByDistanceGenerator(
            route_ratio=2,
            poisson_lambda=1/2,
            malfunction_generator=EventIntervalGenerator(
                                        min_duration=10,
                                        max_duration=30),
        )

        # Decide cargo generator
        cargo_generator = DynamicCargoGenerator(
            cargo_creation_rate=1/100,
            max_cargo_to_create=5,
            num_initial_tasks=10,
            max_weight=3,
            max_stagger_steps=max_cycles/2,
            soft_deadline_multiplier=10,
            hard_deadline_multiplier=20,
        )

        # Decide the number of agents
        num_agents = 2

        # Create the environment
        env = AirliftEnv(
                world_generator=AirliftWorldGenerator(
                plane_types=plane_types,
                airport_generator=airport_generator,
                route_generator=route_generator,
                cargo_generator=cargo_generator,
                airplane_generator=AirplaneGenerator(num_of_agents=num_agents),
                max_cycles=max_cycles
                ),
            )
        
        return env
    

    def hardest_env(self, max_cycles=5000):
        """
        Returns a preset environment
        - 8 nodes
        - 2 airplane
        - purturbations present
        """

        num_agents = 244
        plane_types = [
            PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=6),
            PlaneType(id=1, max_range=1.0, speed=0.15, max_weight=2),
        ]
        num_airports = 122
        num_initial_cargo = 732
        max_cargo = 852
        soft_deadline_multiplier = 5
        hard_deadline_multiplier = 6
        malfunction_min_duration = 12
        malfunction_max_duration = 48
        malfunction_rate = 0.18
        

        # Decide airport generator
        airport_generator = RandomAirportGenerator(
            max_airports=num_airports,
            make_drop_off_area=True,
            make_pick_up_area=True,
            num_drop_off_airports=2,
            num_pick_up_airports=2,
            mapgen=PlainMapGenerator(),
        )

        # Decide route generator
        route_generator = RouteByDistanceGenerator(
            route_ratio=2,
            poisson_lambda=malfunction_rate,
            malfunction_generator=EventIntervalGenerator(
                                        min_duration=malfunction_min_duration,
                                        max_duration=malfunction_max_duration),
        )

        # Decide cargo generator
        cargo_generator = DynamicCargoGenerator(
            cargo_creation_rate=1/100,
            max_cargo_to_create=max_cargo-num_initial_cargo,
            num_initial_tasks=num_initial_cargo,
            max_weight=3,
            max_stagger_steps=max_cycles/2,
            soft_deadline_multiplier=soft_deadline_multiplier,
            hard_deadline_multiplier=hard_deadline_multiplier,
        )

        # Create the environment
        env = AirliftEnv(
                world_generator=AirliftWorldGenerator(
                plane_types=plane_types,
                airport_generator=airport_generator,
                route_generator=route_generator,
                cargo_generator=cargo_generator,
                airplane_generator=AirplaneGenerator(num_of_agents=num_agents),
                max_cycles=max_cycles
                ),
            )
        
        return env


    def next(self, stage):
        """
        Returns the environment and epsilon for the next episode
        """
        
        # Decide environment type
        env = self.small_env_with_purturbations()

        return env


def heuristic_action_to_tensor(actions, x):
    """
    The heuristic model outputs actions. This function converts those actions into tensors.
    """
    action_type = [[0, 0, 0] for _ in range(len(x['agents']['map']))]
    plane_dest = [[0 for _ in range(len(x['nodes']['nodes']))]\
                  for _ in range(len(x['agents']['map']))]
    cargo_assign = [[0 for _ in range(len(x['agents']['map'])+1)]\
                  for _ in range(len(x['cargo']['map']))]
    
    for agent_id, action in actions.items():
        cargo_to_load = action['cargo_to_load']
        cargo_to_unload = action['cargo_to_unload']
        destination = action['destination']
        agent_idx = x['agents']['map'].index(agent_id)

        if destination != 0:
            action_type[agent_idx][0] = 1
            plane_dest[agent_idx][destination-1] = 1
        
        elif cargo_to_load != [] or cargo_to_unload != []:
            action_type[agent_idx][1] = 1
            for cargo_id in cargo_to_load:
                cargo_idx = x['cargo']['map'].index(cargo_id)
                cargo_assign[cargo_idx][agent_idx] = 1
            for cargo_id in cargo_to_unload:
                cargo_idx = x['cargo']['map'].index(cargo_id)
                cargo_assign[cargo_idx][len(x['agents']['map'])] = 1
        
        else:
            action_type[agent_idx][2] = 1
        
    return {
        'actions': torch.tensor(action_type, dtype=torch.float32),
        'destinations': torch.tensor(plane_dest, dtype=torch.float32),
        'cargo': torch.tensor(cargo_assign, dtype=torch.float32)
    }


def stack_nested_dicts(dict_list):
    if not dict_list:
        return {}
    
    result = {}
    result['nodes'] = {}
    # result['nodes']['PyGeom'] = []
    for key in dict_list[0].keys():
        if isinstance(dict_list[0][key], dict):
            result[key] = stack_nested_dicts([d[key] for d in dict_list])
        elif isinstance(dict_list[0][key], torch.Tensor):
            result[key] = torch.stack([d[key] for d in dict_list])
            if result[key].dim() == 4:
                result[key] = torch.reshape(result[key], (result[key].shape[0], result[key].shape[2], result[key].shape[3]))
        elif isinstance(dict_list[0][key], list) and isinstance(dict_list[0][key][0], Data):
            result[key] = [d[key][0] for d in dict_list]
        else:
            result[key] = [d[key] for d in dict_list]
    
    return result


def calculate_batch_loss(pi, criterion, x_batch, target_batch):
    y_batch = pi(x_batch)
    
    # Action Loss
    action_loss = criterion(y_batch['action_logits'].view(-1, 3), target_batch['actions'].view(-1, 3))

    # Destination Loss
    dest_mask = (target_batch['actions'][:, :, 0] == 1).float().unsqueeze(-1)\
        .expand(-1, -1, x_batch['nodes']['PyGeom'][0].x.shape[0])
    destination_loss = criterion(
        torch.nan_to_num(y_batch['destination_logits'], nan=0)*dest_mask,
        target_batch['destinations']*dest_mask
    )
    
    # Cargo Loss
    cargo_mask = (target_batch['actions'][:, :, 1] == 1).float()
    cargo_mask = torch.tensor([y+[1] for y in cargo_mask.tolist()])
    cargo_mask = cargo_mask.unsqueeze(-1).expand(-1, -1, y_batch['cargo'].shape[1])
    cargo_mask = cargo_mask.permute(0, 2, 1)

    cargo_loss = criterion(
        torch.nan_to_num(y_batch['cargo_logits'], nan=0)*cargo_mask,
        target_batch['cargo']*cargo_mask
    )

    total_loss = action_loss + destination_loss + cargo_loss
    return total_loss



def heuristic_pretraining(pi, h, optim, epochs=10, num_batches=32, timesteps_per_minibatch=32, num_envs=4, seed=0):
    """
    This function is responsible for pretraining the agent using a heuristic solution.
    It adds a scaled loss each timestep and backpropagates at regular intervals.
    The loss for the action head is always calculated.
    The loss for the destination head and cargo head is calculated only if the action is a destination or cargo action respectively.
    """

    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    scheduler = Scheduler(seed=seed, num_epochs=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0

        envs = [scheduler.next(epoch) for _ in range(num_envs)]
        xs = [None for _ in range(num_envs)]
        obs = [env.reset(seed=seed) for env in envs]
        for env_id in range(num_envs):
            h[env_id].reset(obs[env_id])
        
        for batch in tqdm(range(num_batches)):
            losses = [0 for _ in range(num_envs)]
            x_batch = []
            y_batch = []

            # Env Loop
            for env_id in range(num_envs):
                env = envs[env_id]
                ob = obs[env_id]
                
                # Episode Loop

                for _ in range(timesteps_per_minibatch):
                    t = h[env_id].current_time
                    
                    # Forward Pass & Get Labels
                    x = format_obs(ob, t, prev_x=xs[env_id])
                    xs[env_id] = x
                    y = pi(x)
                    for p in range(x['agents']['tensor'].shape[0]):
                        if torch.isnan(y['actions'][p][0][0]).item():
                            raise ValueError('NaN in action logits')
                    actions = post_process(x, y)
                    target_actions = h[env_id].policies(ob, None, None)
                    targets = heuristic_action_to_tensor(target_actions, x)

                    x_batch.append(x)
                    y_batch.append(targets)

                    # Step Env
                    ob, _, dones, _ = env.step(actions=actions)

                    # If done, get new environment
                    if all(dones.values()):
                        envs[env_id] = scheduler.next(epoch)
                        ob = envs[env_id].reset(seed=seed)
                        h[env_id].reset(ob)
                
                obs[env_id] = ob
            x_batch = stack_nested_dicts(x_batch)
            y_batch = stack_nested_dicts(y_batch)
            total_loss = calculate_batch_loss(pi, criterion, x_batch, y_batch)

            # Backward Pass
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1.0)
            optim.step()
        
            epoch_loss += total_loss.item()
            print(f"Batch Loss: {total_loss.item()}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")
        scheduler.step(avg_epoch_loss)
        
        # Save the model for each epoch
        pth = 'solution/models/v4/v4_epoch-'+str(epoch)+'_pi.pth'
        torch.save(pi.state_dict(), pth)
                

def main():
    """
    Main function
    """

    seed = 0
    lr = 0.00000000000000001
    num_envs = 4

    h = [HeuristicSolution() for _ in range(num_envs)]

    # Load the policy model
    pi = Policy()
    optim = torch.optim.Adam(pi.parameters(), lr=lr)

    # Pretrain the policy model
    heuristic_pretraining(pi, h, optim, epochs=10, num_batches=32, timesteps_per_minibatch=32, num_envs=num_envs, seed=seed)


main()