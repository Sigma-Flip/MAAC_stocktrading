import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import onehot_from_logits, categorical_sample

class BasePolicy(nn.Module):
    """
    Base policy network for both continuous and discrete action spaces.
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
                 norm_in=True, onehot_dim=0):
        """
        Initialize the base policy network.

        Args:
            input_dim (int): Input feature dimension
            out_dim (int): Output feature dimension (number of actions)
            hidden_dim (int): Hidden layer dimension
            nonlin (function): Nonlinearity function (default: Leaky ReLU)
            norm_in (bool): Whether to normalize input with BatchNorm
            onehot_dim (int): One-hot encoding dimension (optional)
        """
        super(BasePolicy, self).__init__()

        # Normalize inputs if needed
        self.in_fn = nn.BatchNorm1d(input_dim, affine=False) if norm_in else lambda x: x

        # Network layers
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, X):
        """
        Forward pass through the network.

        Args:
            X (Tensor or tuple): Observations or (observations, onehot vector)

        Returns:
            Tensor: Output actions
        """
        onehot = None
        if isinstance(X, tuple):
            X, onehot = X

        inp = self.in_fn(X)
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=1)

        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out


class DiscretePolicy(BasePolicy):
    """
    Discrete action policy network (for action spaces with discrete choices).
    """
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False,
                return_log_pi=False, regularize=False, return_entropy=False):
        """
        Forward pass with sampling and optional entropy, log probability returns.

        Args:
            obs (Tensor): Observations
            sample (bool): Whether to sample from action distribution
            return_all_probs (bool): If True, returns full action probabilities
            return_log_pi (bool): If True, returns log probability of action
            regularize (bool): If True, applies regularization to output
            return_entropy (bool): If True, returns entropy of the action distribution

        Returns:
            Tensor or list: Action or actions with additional outputs if requested
        """
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda

        # Sample or take the highest probability action
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)

        rets = [act]

        # Optionally return additional information
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())

        return rets[0] if len(rets) == 1 else rets


if __name__ == "__main__":
    # Configuration
    n_agents = 3          # Number of agents
    obs_dim = 4           # Observation dimensions
    hidden_dim = 64       # Hidden layer dimensions
    output_dim = 3        # Number of actions (e.g., 3 discrete choices)

    # Create an instance of the DiscretePolicy
    policy = DiscretePolicy(input_dim=obs_dim, out_dim=output_dim, hidden_dim=hidden_dim)

    # Generate random observations (n_agents x obs_dim)
    observations = torch.randn(n_agents, obs_dim)

    # Determine actions through the policy
    actions = policy.forward(observations)

    # Output the actions
    print("Actions (policy outputs):", actions)
