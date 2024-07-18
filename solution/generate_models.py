from models.v3.model_v3 import Policy, Value, get_config
import torch

"""
Param Count (Policy): 13,208,443
Param Count (Value): 3,806,298
"""

version = 'v3'

# Create the models
policy = Policy()
value = Value()

# Print the number of parameters in the models
print(f'Param Count (Policy): {sum(p.numel() for p in policy.parameters())}')
print(f'Param Count (Value): {sum(p.numel() for p in value.parameters())}')

# Export the models
policy_path = 'solution/models/'+version+'/policy_'+version+'_base.pth'
value_path = 'solution/models/'+version+'/value_'+version+'_base.pth'
torch.save(policy.state_dict(), policy_path)
torch.save(value.state_dict(), value_path)