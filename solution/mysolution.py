from airlift.solutions import Solution
from airlift.envs import ActionHelper
from solution.format_obs import format_obs
from solution.models.v2.model_v2 import Policy, get_config
from solution.post_process import post_process
import random
import torch

""" TODO Block

"""

class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    def __init__(self, policy_path='solution/models/v2/v2_episode_30_pi.pth', seed=None):
        super().__init__()

        self.seed = seed
        self.model = Policy(get_config('v1')['policy'])
        self.model.load_state_dict(torch.load(policy_path))
        self.t = 0


    def policies(self, obs, dones, infos):

        # Print warnings
        if infos is not None:
            for k, w in infos.items():
                if w['warnings'] != []:
                    print(w['warnings'])

        # Format obs into tensors, incriment timestep
        x = format_obs(obs, self.t)
        self.t += 1

        # Obtain the actions from the model
        y = self.model.forward(x)

        # Post-process the actions
        actions = post_process(x, y)

        return actions