from airlift.solutions import Solution
from airlift.envs import ActionHelper
from solution.format_obs import format_obs
from solution.post_process import post_process
import random
import torch


class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self, pi, seed=None):
        super().__init__()

        self.seed = seed
        self.model = pi
        # self.model.load_state_dict(torch.load(policy_path))
        self.t = 0
        self.prev_cargo = {}
        self.prev_airplanes = {}
    
    def reset(self):
        self.prev_cargo = {}
        self.prev_airplanes = {}
        self.t = 0


    def policies(self, obs, dones, infos):

        # Print warnings
        if infos is not None:
            for k, w in infos.items():
                if w['warnings'] != []:
                    print(w['warnings'])

        # Format obs into tensors, incriment timestep
        x = format_obs(obs, self.t, prev_cargo=self.prev_cargo, prev_airplanes=self.prev_airplanes)
        self.t += 1

        # Obtain the actions from the model
        y = self.model.forward(x)

        # Post-process the actions
        actions = post_process(x, y)

        return actions