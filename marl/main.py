import numpy as np
import torch
from scenario import Stockenv
from sac import AttentionSAC
from utils import *
from buffer import ReplayBuffer
from config import config
# Environment and model settings


config = config.modelConfig

# Initialize the stock environment
env = Stockenv()
obs, obs_dim = env.reset()
# Generate random data
state_dim = 5
action_dim = 10
sub_reward_dim = 2
sasr_sizes = [(state_dim, action_dim, sub_reward_dim) for _ in range(config.n_agents)]

# Initialize the AttentionSAC model
model = AttentionSAC(
    env,
    sasr_size=sasr_sizes,
    pol_input_dim=obs_dim,
    pol_out_dim=action_dim,
    tau=config.tau,
    pi_lr=config.pi_lr,
    q_lr=config.q_lr,
    gamma=config.gamma,
    pol_hidden_dim=config.pol_hidden_dim,
    critic_hidden_dim=config.critic_hidden_dim,
    attend_heads=config.attend_heads,
    reward_scale=config.reward_scale
)

# Initialize replay buffer
replay_buffer = ReplayBuffer(config.buffer_length, config.n_agents, [obs_dim], [action_dim])

# Training loop
for ep_i in range(config.n_episodes):
    print(f"Starting Episode {ep_i + 1}")

    # Reset environment and get initial observation
    obs = env.reset()

    for et_i in range(config.episode_length):
        torch_obs = torch.Tensor(np.vstack(obs))  # Convert observations to tensor

        # Compute actions using the AttentionSAC model
        torch_agent_actions = model.step(torch_obs)

        # Step the environment and collect results
        next_obs, rewards, dones, infos = env.step(torch_agent_actions)

        # Store experience in replay buffer
        replay_buffer.push(obs, torch_agent_actions, rewards, next_obs, dones)

        # Update current observations
        obs = next_obs

        # Update time
        t = ep_i * config.episode_length + et_i

        # Perform model updates if enough data is in the buffer
        if len(replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads:
            device = 'gpu' if config.use_gpu else 'cpu'
            model.prep_training(device=device)

            for u_i in range(config.num_updates):
                # Sample from replay buffer
                sample = replay_buffer.sample(config.batch_size, to_tensor=config.use_gpu)

                # Update critic and policy
                model.update_critic(sample)
                model.update_policies(sample)
                model.update_all_targets()

            model.prep_rollouts(device='cpu')

    # Log average rewards for this episode
    ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
    for a_i, a_ep_rew in enumerate(ep_rews):
        if logger:
            logger.add_scalar(f'agent{a_i}/mean_episode_rewards', a_ep_rew * config.episode_length, ep_i)

# Save model and close logger if it exists
model.save('model.pt')
if logger:
    logger.export_scalars_to_json('summary.json')
    logger.close()
