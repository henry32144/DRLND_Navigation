import torch
import torch.nn as nn
import torch.nn.functional as F
    
class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(DuelingQNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc_state1 = nn.Linear(64, 64)
        self.fc_state2 = nn.Linear(64, 1)
        self.fc_advantange1 = nn.Linear(64, 64)
        self.fc_advantange2 = nn.Linear(64, action_size)
        
        self.seed = torch.manual_seed(seed)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = F.relu(self.fc1(state))
        
        state_value = F.relu(self.fc_state1(state))
        state_value = self.fc_state2(state_value).expand(64, self.action_size)
        adv_value = F.relu(self.fc_advantange1(state))
        adv_value = self.fc_advantange2(adv_value)
        
        # sum up state value and action value minus mean of action value
        total_Q = state_value + adv_value - adv_value.mean(1).unsqueeze(1).expand(64, self.action_size)
        return total_Q
