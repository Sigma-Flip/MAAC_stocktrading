class EnvConfig:
    def __init__(self):
        self.nenv = 10  # Number of environments


class StockConfig:
    def __init__(self):
        self.stock_codes = ['', '' ]
class BrokerConfig:
    def __init__(self):
        self.key = "PSCYfKelpDuvkEuCqt3CxrX5enEpPpAhFQ96"  # API key
        self.acc_no = "501119-20"  # Account number (to be set)
        self.secret = "JsMxWAJnVHbhhq6Rm/FNusLvEbXXX/JI/ChxpFoD/jPuFDlXDELJhJu2wbcpNEnsY0sFMm4xl2a/UWKa9XMQmUKk1tZLEa/+FYxxvgTsHIAVEan3OtldvP8nlWGMBG6DgRQDxcVWW2CDLqt1smF+Af9fmrpnEgGirgUcJU+HxoFFpvghfJw="  # API secret
        self.moc = True
class ModelConfig:
    def __init__(self):
        self.tau = 0.01                # Soft update parameter for target networks
        self.pi_lr = 0.001             # Learning rate for policy (actor) network
        self.q_lr = 0.001              # Learning rate for critic (Q-value) network
        self.gamma = 0.99              # Discount factor for future rewards
        self.pol_hidden_dim = 128       # Hidden layer dimension for the policy network
        self.critic_hidden_dim = 128    # Hidden layer dimension for the critic network
        self.attend_heads = 4           # Number of attention heads in the AttentionCritic
        self.reward_scale = 10.0        # Scaling factor for rewards
        self.batch_size = 10            # Batch size for training
        self.buffer_length = 1000       # Replay buffer length
        self.n_episodes = 10            # Number of episodes to train
        self.episode_length = 1000      # Length of each episode
        self.n_agents = 4               # Number of agents
        self.n_rollout_threads = 1      # Number of rollout threads
        self.steps_per_update = 5       # Steps per model update
        self.use_gpu = False             # Flag to use GPU
        self.num_updates = 5             # Number of updates per training step

class Config:
    def __init__(self):
        self.envConfig = EnvConfig()
        self.brokerConfig = BrokerConfig()
        self.modelConfig = ModelConfig()
        self.stockConfig = StockConfig()

config = Config()
