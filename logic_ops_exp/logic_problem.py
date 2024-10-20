from gen_data import gen_data
from model import LogicModelMA
import torch

class LogicProblem:

    def __init__(self, key_size, loss_fn=None, embed_dim=None, num_FIN_layers=None):
        self.key_size, self.embed_size, self.num_FIN_layers =\
              key_size, None, None
        self.model = None
        self.loss = None
        if loss_fn is not None:
            self.init_loss(loss_fn)
        if embed_dim is not None and num_FIN_layers is not None:
            self.init_model(embed_dim, num_FIN_layers)

    def init_loss(self, loss_fn):
        self.loss = loss_fn

    def init_model(self, embed_dim, num_FIN_layers):
        self.embed_size = embed_dim
        self.num_FIN_layers = num_FIN_layers

        x = gen_data(1, self.key_size)

        # Convert data to tensors
        x['prop_mtx'] = torch.tensor(x['prop_mtx'], dtype=torch.float32)
        x['op_mtx'] = torch.tensor(x['op_mtx'], dtype=torch.float32)
        x['goal_mtx'] = torch.tensor(x['goal_mtx'], dtype=torch.float32)
        
        # Get dimensions
        prop_dim = x['prop_mtx'].shape[1]
        op_dim = x['op_mtx'].shape[1]
        goal_dim = x['goal_mtx'].shape[1]

        # Create model
        self.model = LogicModelMA(
            in_set = {
                'prop': prop_dim,
                'op': op_dim,
                'goal': goal_dim
            },
            embed_dim=64,
            num_FIN_layers=1
        )

    def __call__(self, n):
        # run the model

        x = gen_data(n, self.key_size)

        # Convert data to tensors
        x['prop_mtx'] = torch.tensor(x['prop_mtx'], dtype=torch.float32)
        x['op_mtx'] = torch.tensor(x['op_mtx'], dtype=torch.float32)
        x['goal_mtx'] = torch.tensor(x['goal_mtx'], dtype=torch.float32)
        x['labels_mtx'] = torch.tensor(x['labels_mtx'], dtype=torch.float32)

        # Expand dimension so theres a batch of 1
        x['prop_mtx'] = x['prop_mtx'].unsqueeze(0)
        x['op_mtx'] = x['op_mtx'].unsqueeze(0)
        x['goal_mtx'] = x['goal_mtx'].unsqueeze(0)
        x['labels_mtx'] = x['labels_mtx'].unsqueeze(0)

        x_tensors = {
            'prop': x['prop_mtx'],
            'op': x['op_mtx'],
            'goal': x['goal_mtx']
        }
        return { 
            'y': self.model(x_tensors),
            'labels': x['labels_mtx']
        }
    
    def run_batch(self, batch_size):
        outputs = self(batch_size)
        loss = self.loss(outputs['y'], outputs['labels'])
        loss.backward()
        return loss
    
    def train(self, epochs, batch_size, lr=0.001):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.run_batch(batch_size)
            optimizer.step()
            print(f'Epoch {epoch} loss: {loss.item()}')
        return self.model
