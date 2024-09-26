import numpy as np
import torch


class ReplayBuffer:
    """
    Replay Buffer for multi-agent Reinforcement Learning with parallel rollouts.
    """

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Initializes the replay buffer.

        Args:
            max_steps (int): Maximum number of timepoints to store in the buffer.
            num_agents (int): Number of agents in the environment.
            obs_dims (list of int): Number of observation dimensions for each agent.
            ac_dims (list of int): Number of action dimensions for each agent.
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = [torch.zeros((max_steps, odim), dtype=torch.float32) for odim in obs_dims]
        self.ac_buffs = [torch.zeros((max_steps, adim), dtype=torch.float32) for adim in ac_dims]
        self.rew_buffs = [torch.zeros(max_steps, dtype=torch.float32) for _ in range(num_agents)]
        self.next_obs_buffs = [torch.zeros((max_steps, odim), dtype=torch.float32) for odim in obs_dims]
        self.done_buffs = [torch.zeros(max_steps, dtype=torch.uint8) for _ in range(num_agents)]

        self.filled_i = 0  # Index of first empty location in buffer
        self.curr_i = 0  # Current index to write to

    def __len__(self):
        """Returns the number of filled entries in the buffer."""
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        Pushes new experience into the replay buffer.

        Args:
            observations (ndarray): Observations from the current time step.
            actions (ndarray): Actions taken by agents.
            rewards (ndarray): Rewards received by agents.
            next_observations (ndarray): Observations from the next time step.
            dones (ndarray): Boolean flags indicating whether episodes have ended.
        """
        nentries = observations.shape[0]  # Number of entries to add

        if self.curr_i + nentries > self.max_steps:
            self._rollover(nentries)

        for agent_i in range(self.num_agents):
            self._store_experience(agent_i, observations, actions, rewards, next_observations, dones, nentries)

        self.curr_i += nentries
        self.filled_i = min(self.max_steps, self.filled_i + nentries)

    def _rollover(self, nentries):
        """Handles rollover when the buffer is full."""
        rollover = self.max_steps - self.curr_i  # Number of indices to roll over
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i] = torch.roll(self.obs_buffs[agent_i], rollover, dims=0)
            self.ac_buffs[agent_i] = torch.roll(self.ac_buffs[agent_i], rollover, dims=0)
            self.rew_buffs[agent_i] = torch.roll(self.rew_buffs[agent_i], rollover, dims=0)
            self.next_obs_buffs[agent_i] = torch.roll(self.next_obs_buffs[agent_i], rollover, dims=0)
            self.done_buffs[agent_i] = torch.roll(self.done_buffs[agent_i], rollover, dims=0)
        self.curr_i = 0
        self.filled_i = self.max_steps

    def _store_experience(self, agent_i, observations, actions, rewards, next_observations, dones, nentries):
        """Stores experience for a specific agent in the buffer."""
        self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = torch.tensor(observations[:, agent_i])
        self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = torch.tensor(actions[agent_i])
        self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = torch.tensor(rewards[:, agent_i])
        self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = torch.tensor(next_observations[:, agent_i])
        self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = torch.tensor(dones[:, agent_i])

    def sample(self, N, device='cpu', norm_rews=True):
        """
        Samples a batch of experiences from the replay buffer.

        Args:
            N (int): Number of samples to retrieve.
            device (str): Device to store the tensors ('cpu' or 'gpu').
            norm_rews (bool): Whether to normalize the rewards.

        Returns:
            tuple: Sampled observations, actions, rewards, next observations, and done flags.
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)

        def to_tensor(arr):
            return torch.tensor(arr).to(device)

        ret_rews = []
        for i in range(self.num_agents):
            if norm_rews:
                mean_rew = self.rew_buffs[i][:self.filled_i].mean()
                std_rew = self.rew_buffs[i][:self.filled_i].std() + 1e-5
                normalized_rew = (self.rew_buffs[i][inds] - mean_rew) / std_rew
                ret_rews.append(to_tensor(normalized_rew))
            else:
                ret_rews.append(to_tensor(self.rew_buffs[i][inds]))

        return (
            [to_tensor(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
            [to_tensor(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
            ret_rews,
            [to_tensor(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [to_tensor(self.done_buffs[i][inds]) for i in range(self.num_agents)]
        )

    def get_average_rewards(self, N):
        """
        Computes average rewards over the last N steps.

        Args:
            N (int): Number of most recent steps to consider.

        Returns:
            list: Average rewards for each agent.
        """
        inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean().item() for i in range(self.num_agents)]
