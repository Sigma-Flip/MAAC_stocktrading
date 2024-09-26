import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update, enable_gradients, disable_gradients
from agent import AttentionAgent
from critic import AttentionCritic

MSELoss = torch.nn.MSELoss()


class AttentionSAC(object):
    """
    Wrapper class for Soft Actor-Critic (SAC) agents with a central attention critic
    in a multi-agent task.
    """

    def __init__(self, sasr_size, pol_input_dim, pol_out_dim,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10., pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4, **kwargs):
        """
        Initializes the Attention SAC.

        Args:
            sasr_size (list of (int, int, int)): Size of state and action space for each agent.
            pol_input_dim (int): Input dimensions to policy.
            pol_out_dim (int): Output dimensions to policy.
            gamma (float): Discount factor.
            tau (float): Target update rate.
            pi_lr (float): Learning rate for policy.
            q_lr (float): Learning rate for critic.
            reward_scale (float): Scaling for reward (affects optimal policy entropy).
            pol_hidden_dim (int): Number of hidden dimensions for networks.
            critic_hidden_dim (int): Number of hidden dimensions for critic networks.
            attend_heads (int): Number of attention heads.
            **kwargs: Additional arguments for further customization.
        """
        self.nagents = len(sasr_size)
        self.pol_input_dim = pol_input_dim
        self.agents = [AttentionAgent(pol_input_dim=pol_input_dim, pol_out_dim=pol_out_dim) for _ in
                       range(self.nagents)]
        self.critic = AttentionCritic(sasr_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sasr_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)

        hard_update(self.target_critic, self.critic)  # Initialize target critic
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)

        # SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale

        # Device configuration
        self.pol_dev = 'cpu'
        self.critic_dev = 'cpu'
        self.trgt_pol_dev = 'cpu'
        self.trgt_critic_dev = 'cpu'
        self.niter = 0

    @property
    def policies(self):
        return [agent.policy for agent in self.agents]

    @property
    def target_policies(self):
        return [agent.target_policy for agent in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in the environment with all agents.

        Args:
            observations (list): List of observations for each agent.
            explore (bool): Whether to explore or exploit.

        Returns:
            list: Actions taken by each agent.
        """
        return [agent.step(obs, explore=explore) for agent, obs in zip(self.agents, observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update the central critic for all agents.

        Args:
            sample (tuple): Contains observations, actions, rewards, next observations, and dones.
            soft (bool): Whether to use soft updates.
            logger (optional): Logger for recording metrics.
            **kwargs: Additional arguments.
        """
        obs, acs, rews, next_obs, dones = sample

        # Prepare next actions and log probabilities
        next_acs, next_log_pis = [], []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)

        # Compute target Q values
        target_critic_input = list(zip(next_obs, next_acs))
        next_qs = self.target_critic(target_critic_input)

        # Compute current Q values
        critic_input = list(zip(obs, acs))
        critic_rets = self.critic(critic_input, regularize=True, logger=logger, niter=self.niter)

        # Calculate Q loss
        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg  # Regularizing attention

        # Backpropagation
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        # Log metrics
        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)

        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        """
        Update policies for all agents.

        Args:
            sample (tuple): Contains observations, actions, rewards, next observations, and dones.
            soft (bool): Whether to use soft updates.
            logger (optional): Logger for recording metrics.
            **kwargs: Additional arguments.
        """
        obs, acs, rews, next_obs, dones = sample
        samp_acs, all_probs, all_log_pis, all_pol_regs = [], [], [], []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True,
                                                       return_entropy=True)
            logger.add_scalar(f'agent{a_i}/policy_entropy', ent, self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        # Compute current Q values for policies
        critic_input = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_input, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.nagents), all_probs, all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v

            # Compute policy loss
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()

            # Add regularization
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # Policy regularization

            # Backpropagation (with gradients from critic disabled)
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            # Log metrics
            if logger is not None:
                logger.add_scalar(f'agent{a_i}/losses/pol_loss', pol_loss, self.niter)
                logger.add_scalar(f'agent{a_i}/grad_norms/pi', grad_norm, self.niter)

    def update_all_targets(self):
        """
        Update all target networks after normal updates have been performed for each agent.
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for agent in self.agents:
            soft_update(agent.target_policy, agent.policy, self.tau)

    def prep_training(self, device='gpu'):
        """Prepare networks for training on the specified device."""
        self.critic.train()
        self.target_critic.train()
        for agent in self.agents:
            agent.policy.train()
            agent.target_policy.train()

        fn = lambda x: x.cuda() if device == 'gpu' else x.cpu()

        # Move agents' policies to the specified device
        if self.pol_dev != device:
            for agent in self.agents:
                agent.policy = fn(agent.policy)
            self.pol_dev = device

        # Move critics to the specified device
        if self.critic_dev != device:
            self.critic = fn(self.critic)
            self.critic_dev = device

        # Move target policies to the specified device
        if self.trgt_pol_dev != device:
            for agent in self.agents:
                agent.target_policy = fn(agent.target_policy)
            self.trgt_pol_dev = device

        # Move target critic to the specified device
        if self.trgt_critic_dev != device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        """Prepare agents for rollouts on the specified device."""
        for agent in self.agents:
            agent.policy.eval()

        fn = lambda x: x.cuda() if device == 'gpu' else x.cpu()

        # Move main policies to the specified device
        if self.pol_dev != device:
            for agent in self.agents:
                agent.policy = fn(agent.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file.

        Args:
            filename (str): Path to the file where the model will be saved.
        """
        self.prep_training(device='cpu')  # Move parameters to CPU before saving
        save_dict = {
            'init_dict': self.init_dict,
            'agent_params': [agent.get_params() for agent in self.agents],
            'critic_params': {
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()
            }
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate an instance of this class from a file created by the 'save' method.

        Args:
            filename (str): Path to the saved model file.
            load_critic (bool): Whether to load critic parameters.

        Returns:
            AttentionSAC: An instance of the AttentionSAC class.
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']

        for agent, params in zip(instance.agents, save_dict['agent_params']):
            agent.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])

        return instance


# Example Usage
if __name__ == "__main__":
    # Define parameters
    sasr_size = [(10, 2, 2)] * 3  # Example state-action-size for 3 agents
    pol_input_dim = 10
    pol_out_dim = 2

    # Initialize Attention SAC
    sac = AttentionSAC(sasr_size, pol_input_dim, pol_out_dim)

    # Simulated observations for agents
    observations = [torch.randn(10) for _ in range(3)]  # Random observations

    # Step through the environment
    actions = sac.step(observations, explore=True)
    print("Actions taken by agents:", actions)

    # Update critic and policies (simulate training)
    sample = (observations, actions, [torch.tensor([1.0])] * 3, observations, [0] * 3)
    sac.update_critic(sample)
    sac.update_policies(sample)

    # Save the model
    sac.save("sac_model.pth")

    # Load the model
    new_sac = AttentionSAC.init_from_save("sac_model.pth", load_critic=True)
