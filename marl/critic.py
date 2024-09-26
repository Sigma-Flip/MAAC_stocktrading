import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from itertools import chain


class AttentionCritic(nn.Module):
    def __init__(self, sasr_sizes, attend_heads=1, norm_in=True, hidden_dim=32):
        """
        Initialize the AttentionCritic with separate encoders and decoders for each agent.

        Args:
            sasr_sizes (list of tuple): Each tuple contains (state_dim, action_dim, sub_reward_dim) for agents.
            attend_heads (int): Number of attention heads.
            norm_in (bool): Whether to apply normalization to the input layers.
            hidden_dim (int): Size of the hidden layers.
        """
        super(AttentionCritic, self).__init__()
        assert hidden_dim % attend_heads == 0, "hidden_dim must be divisible by attend_heads"

        self.attend_heads = attend_heads
        self.sasr_sizes = sasr_sizes
        self.hidden_dim = hidden_dim
        self.nagents = len(sasr_sizes)

        # Initialize encoders and decoders for each agent
        self.critic_encoders = nn.ModuleList()
        self.critic_decoders = nn.ModuleList()
        self.state_encoders = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.queries = nn.ModuleList()
        self.values = nn.ModuleList()

        for sdim, adim, srdim in self.sasr_sizes:
            sasrdim = sdim + adim + srdim

            critic_encoder = nn.Sequential()
            if norm_in:
                critic_encoder.add_module('critic_bn', nn.BatchNorm1d(sasrdim, affine=False))
            critic_encoder.add_module('critic_encoder_fc1', nn.Linear(sasrdim, hidden_dim))
            critic_encoder.add_module('critic_encoder_nl', nn.LeakyReLU())

            critic_decoder = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, adim)
            )

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('state_encoder_bn', nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module('state_encoder_fc1', nn.Linear(sdim, hidden_dim))
            state_encoder.add_module('state_encoder_nl', nn.LeakyReLU())

            self.critic_encoders.append(critic_encoder)
            self.critic_decoders.append(critic_decoder)
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads

        for _ in range(attend_heads):
            self.keys.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.queries.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.values.append(nn.Sequential(
                nn.Linear(hidden_dim, attend_dim),
                nn.LeakyReLU()
            ))

        self.shared_modules = [self.keys, self.queries, self.values, self.critic_encoders]

    def shared_parameters(self):
        """Retrieve all shared parameters across the model."""
        return chain(*[p.parameters() for p in self.shared_modules])

    def scale_shared_grads(self):
        """Scale gradients for shared parameters."""
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, inputs, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
        """
        Forward pass through the AttentionCritic.

        Args:
            inputs (list): List of inputs for each agent containing (state, action, sub_reward).
            agents (iterable): Specific agents to consider, defaults to all agents.
            return_q (bool): Whether to return Q values.
            return_all_q (bool): Whether to return all Q values.
            regularize (bool): Whether to apply regularization.
            return_attend (bool): Whether to return attention weights.
            logger (object): Logger for tracking metrics.
            niter (int): Current iteration for logging.

        Returns:
            list: Output from the forward pass, can include Q values and regularization terms.
        """
        if agents is None:
            agents = range(self.nagents)

        # Unpack inputs
        states = [s for s, _, _ in inputs]
        actions = [a for _, a, _ in inputs]
        sub_rewards = [sr for _, _, sr in inputs]
        inps = torch.stack([torch.cat((s, a, sr), dim=1) for s, a, sr in inputs], dim=0)

        # Encode inputs
        sasr_encodings = [ce(sasr) for ce, sasr in zip(self.critic_encoders, inps)]
        state_encodings = [se(s) for se, s in zip(self.state_encoders, states)]

        # Compute attention values
        all_head_keys = [[key_layer(sa) for sa in sasr_encodings] for key_layer in self.keys]
        all_head_values = [[value_layer(sa) for sa in sasr_encodings] for value_layer in self.values]
        all_head_queries = [[query_layer(sa) for i, sa in enumerate(sasr_encodings) if i in agents] for query_layer in
                            self.queries]

        # Store attention results
        all_attention_logits = []
        all_attention_weights = []
        all_attention_values = []

        for keys, values, queries in zip(all_head_keys, all_head_values, all_head_queries):
            queries = torch.stack(queries).permute(1, 0, 2)  # (batch, n, attend_dim)
            keys = torch.stack(keys).permute(1, 2, 0)  # (batch, attend_dim, n)
            values = torch.stack(values).permute(1, 0, 2)  # (batch, n, attend_dim)

            attend_logits = torch.matmul(queries, keys)  # (batch, n, n)
            attend_weights = F.softmax(attend_logits, dim=2)  # (batch, n, n)
            attention_values = torch.matmul(attend_weights, values)  # (batch, n, attend_dim)

            all_attention_logits.append(attend_logits)
            all_attention_weights.append(attend_weights)
            all_attention_values.append(attention_values)

        # Concatenate attention results
        multihead_attention_logits = torch.cat(all_attention_logits, dim=2)  # (batch, n, n * attend_heads)
        multihead_attention_values = torch.cat(all_attention_values, dim=2)  # (batch, n, hidden_dim)

        log_attention_probs = torch.log(multihead_attention_weights + 1e-10)  # (batch, n, n * attend_heads)
        hard_entropies = -torch.sum(multihead_attention_weights * log_attention_probs, dim=-1)  # (batch, n)

        # Prepare inputs for critic decoders
        state_encodings = torch.stack(state_encodings).permute(1, 0, 2)  # (batch, n, hidden_dim)
        critic_input = torch.cat((state_encodings, multihead_attention_values), dim=-1)  # (batch, n, 2 * hidden_dim)
        critic_inputs = torch.split(critic_input, split_size_or_sections=1, dim=1)  # n * (batch, 1, 2 * hidden_dim)
        critic_inputs = [x.squeeze(1) for x in critic_inputs]  # n * (batch, 2 * hidden_dim)

        agent_rets = []
        all_rets = []

        # Regularization
        if regularize:
            reg = 1e-3 * sum([(logit ** 2).mean() for logit in all_attention_logits])  # Scalar value
            agent_rets.append((reg,))  # Convert to tuple

        if return_attend:
            agent_rets.append(all_attention_weights)

        for a_i, critic_in, critic_decoder in zip(range(len(critic_inputs)), critic_inputs, self.critic_decoders):
            all_q = critic_decoder(critic_in)  # (batch, action_dim)
            int_actions = actions[a_i].max(dim=1, keepdim=True)[1]  # (batch, 1)
            q = all_q.gather(1, int_actions)  # (batch, 1)

            if return_q:
                agent_rets.append(q)

            if return_all_q:
                agent_rets.append(all_q)

            # Log metrics if logger is provided
            if logger is not None:
                logger.add_scalars(f'agent{a_i}/attention',
                                   {f'head{h_i}_entropy': ent for h_i, ent in
                                    enumerate(hard_entropies[:, a_i].mean(dim=0))},
                                   niter)

        # Return results
        return agent_rets if len(agent_rets) > 1 else agent_rets[0]


# Example usage of the AttentionCritic
if __name__ == "__main__":
    # Example dimensions
    state_dim = 3
    action_dim = 5
    sub_reward_dim = 1
    n_agents = 2
    batch_size = 10
    hidden_dim = 32
    attend_heads = 2

    # Define sasr_sizes for each agent
    sasr_sizes = [(state_dim, action_dim, sub_reward_dim) for _ in range(n_agents)]

    # Initialize AttentionCritic
    attention_critic = AttentionCritic(sasr_sizes, attend_heads=attend_heads, hidden_dim=hidden_dim)

    # Generate random input data
    states = [torch.randn(batch_size, state_dim) for _ in range(n_agents)]
    actions = [torch.randn(batch_size, action_dim) for _ in range(n_agents)]
    sub_rewards = [torch.randn(batch_size, sub_reward_dim) for _ in range(n_agents)]

    inputs = list(zip(states, actions, sub_rewards))

    # Forward pass
    output = attention_critic(inputs, regularize=True, return_q=False, return_all_q=True, return_attend=False)

    # Print the output to check if it runs correctly
    print("Output:", output)
    print('Output length:', len(output))
