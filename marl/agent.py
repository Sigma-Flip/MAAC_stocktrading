from policy import DiscretePolicy
from utils import hard_update, soft_update
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam

class AttentionAgent:
    def __init__(self, pol_inp_dim, pol_out_dim, hidden_dim=64, lr=0.01, onehot_dim=0):
        """
        Initialize the AttentionAgent with a behavior and target policy.

        Args:
            pol_inp_dim (int): Input dimension for the policy
            pol_out_dim (int): Output dimension for the policy (number of actions)
            hidden_dim (int): Number of hidden dimensions
            lr (float): Learning rate for the optimizer
            onehot_dim (int): One-hot encoding dimension (optional)
        """
        self.policy = DiscretePolicy(input_dim=pol_inp_dim, out_dim=pol_out_dim, hidden_dim=hidden_dim, onehot_dim=onehot_dim)
        self.target_policy = DiscretePolicy(input_dim=pol_inp_dim, out_dim=pol_out_dim, hidden_dim=hidden_dim, onehot_dim=onehot_dim)

        # Initialize target policy with the current policy parameters
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step in the environment using the policy to choose an action.

        Args:
            obs (Tensor): Observations of shape (n_agent, n_envs, obs_dim + onehot_dim)
                          or (n_agent, batch_size, obs_dim + action_dim)
            explore (bool): Whether to explore (True) or exploit (False).

        Returns:
            Tensor: Selected actions of shape (n_agent, n_envs, action_dim)
                    or (n_agent, batch_size, action_dim)
        """
        return self.policy(obs=obs, sample=explore)

    def get_params(self):
        """
        Get the current parameters of the policy, target policy, and optimizer.

        Returns:
            dict: A dictionary containing the state_dict of policy, target_policy, and policy_optimizer.
        """
        return {
            'policy': self.policy.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }

    def load_params(self, params):
        """
        Load parameters into the policy, target policy, and optimizer.

        Args:
            params (dict): A dictionary containing the state_dicts to load.
        """
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])


if __name__ == "__main__":
    # Example configuration
    n_agents = 3          # Number of agents
    obs_dim = 4           # Observation dimension
    action_dim = 3        # Number of possible actions
    hidden_dim = 64       # Hidden layer size
    learning_rate = 0.01  # Learning rate

    # Create an instance of the AttentionAgent
    agent = AttentionAgent(pol_inp_dim=obs_dim, pol_out_dim=action_dim, hidden_dim=hidden_dim, lr=learning_rate)

    # Generate random observations for agents (n_agents x obs_dim)
    observations = torch.randn(n_agents, obs_dim)

    # Get actions using the agent (with exploration enabled)
    actions = agent.step(observations, explore=True)

    # Print the selected actions
    print("Selected actions:", actions)

    # Retrieve the agent parameters
    params = agent.get_params()
    print("Agent parameters:", params)
