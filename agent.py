import random
import torch
import torch.optim as optim
import numpy as np
from model import DuelingQNetwork
from replaybuffers import PrioritizedReplayBuffer

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, batch_size, buffer_size, gamma, alpha, beta, beta_increment_per_step, epsilon, tau, lr, update_very):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): seed for generate random number
            batch_size (int): how much data will be used in one learning step
            buffer_size (int): size of replay buffer
            gamma (float): discount factor
            alpha (float): how much prioritization is used
            beta (float): To what degree to use importance weights
            beta_increment_per_step (float): increase beta in every sample step until beta reach one
            epsilon (float): float A constant number add up with prioirty when updating
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_very (int): how often to update the network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_very

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Used device for training: ', self.device)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha, beta, beta_increment_per_step, epsilon, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Store experience into memory and learn every UPDATE_EVERY time steps.
        
        Params
        ======
            state (array_like): current state
            action (int): performed action
            reward (float): got reward
            next_state (array_like): next state
            done (bool): whether is finish or not
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences, idxs, is_weights  = self.memory.sample(self.batch_size)
                self.learn(experiences, self.gamma, idxs, is_weights)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, idxs, is_weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Double Q Learning
        # Get the action indices which can produce maximum Q value in local Q network.
        next_action_indice = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        # Perform these actions in Q target network.
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_action_indice)
        # Compute Q targets value
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())
        # Compute TD-error
        errors = torch.abs(Q_expected - Q_targets).data.cpu().numpy()

        # update memory priority
        self.memory.update_priorities(idxs, errors)
        
        self.optimizer.zero_grad()

        # Weighted MSE Loss function
        loss = (torch.from_numpy(is_weights).cpu().float() * ((Q_expected-Q_targets).cpu() ** 2)).mean()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
