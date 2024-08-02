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
            max_airports=3,
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


    def next(self):
        """
        Returns the environment and epsilon for the next episode
        """
        
        # Decide environment type
        env = self.small_env_with_purturbations()

        self.episode += 1
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

        envs = [scheduler.next(epoch) for _ in range(num_envs)]
        xs = [None for _ in range(num_envs)]
        obs = [env.reset(seed=seed) for env in envs]
        for env_id in range(num_envs):
            h[env_id].reset(obs[env_id])
        
        for batch in tqdm(range(num_batches)):
            losses = [0 for _ in range(num_envs)]

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
                        if torch.isnan(y['actions'][p][0]).item():
                            print("1111111111111111111111111111111111111111111111111")
                            print(x)
                            print("11111111111111111111111111111111111111111111111111111111111111")
                            break
                    actions = post_process(x, y)
                    target_actions = h[env_id].policies(ob, None, None)
                    targets = heuristic_action_to_tensor(target_actions, x)

                    # Calculate Loss
                    h_cargo_selections = [False for _ in range(x['cargo']['tensor'].shape[0])]
                    for plane_idx in range(x['agents']['tensor'].shape[0]):
                        loss = criterion(y['action_logits'][plane_idx], targets['actions'][plane_idx])\
                            / (timesteps_per_minibatch * num_envs)
                        losses[env_id] += loss

                        action_type = torch.argmax(y['actions'][plane_idx]).item()
                        h_action_type = torch.argmax(targets['actions'][plane_idx]).item()

                        # Loss for Destination only if both actions are destination actions
                        if action_type == DESTINATION_ACTION and h_action_type == DESTINATION_ACTION:
                            loss = criterion(y['destination_logits'][plane_idx], targets['destinations'][plane_idx])\
                                / (timesteps_per_minibatch * num_envs)
                            # check is loss is NaN
                            if torch.isnan(loss).item():
                                print(t, plane_idx, y['destination_logits'], targets['destinations'], y['actions'], targets['actions'])
                                with open('ob.pkl', 'wb') as f:
                                    pickle.dump(ob, f)
                                raise ValueError('Loss is NaN on Destinations')

                            losses[env_id] += loss
                        
                        # Loss for Cargo only if both actions are load/unload actions
                        elif action_type == LOAD_UNLOAD_ACTION and h_action_type == LOAD_UNLOAD_ACTION:
                            for cargo_idx in range(x['cargo']['tensor'].shape[0]):
                                h_cargo_selections[cargo_idx] = True
                    
                    # Loss for Cargo only if both actions are load/unload actions
                    for cargo_idx in range(x['cargo']['tensor'].shape[0]):
                        if h_cargo_selections[cargo_idx]:
                            loss = criterion(y['cargo_logits'][cargo_idx], targets['cargo'][cargo_idx])\
                                / (timesteps_per_minibatch * num_envs)
                            if torch.isnan(loss).item():
                                #pickle x
                                print(t)
                                print(y['cargo_logits'][cargo_idx], targets['cargo'][cargo_idx])
                                with open('ob.pkl', 'wb') as f:
                                    pickle.dump(ob, f)
                                raise ValueError('Loss is NaN on Destinations')
                            losses[env_id] += loss

                    # Step Env
                    ob, rews, dones, infos = env.step(actions=actions)

                    # If done, get new environment
                    done = True
                    for _, d in dones.items():
                        if not d:
                            done = False
                            break
                    if done:
                        envs[env_id] = scheduler.next(epoch)
                        ob = envs[env_id].reset(seed=seed)
                        h[env_id].reset(ob)
                
                obs[env_id] = ob
            
            # Backward Pass
            total_loss = torch.stack(losses).sum()
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pi.parameters(), max_norm=1.0)
            optim.step()
        
            print(f"Loss: {total_loss.item()}")
        
        # Save the model for each epoch
        pth = 'solution/models/v4/v4_epoch-'+str(epoch)+'_pi.pth'
        torch.save(pi.state_dict(), pth)
                

def main():
    """
    Main function
    """

    seed = 0
    lr = 0.0001
    num_envs = 4

    h = [HeuristicSolution() for _ in range(num_envs)]

    # Load the policy model
    pi = Policy()
    optim = torch.optim.Adam(pi.parameters(), lr=lr)

    # Pretrain the policy model
    heuristic_pretraining(pi, h, optim, epochs=10, num_batches=32, timesteps_per_minibatch=32, num_envs=num_envs, seed=seed)


main()