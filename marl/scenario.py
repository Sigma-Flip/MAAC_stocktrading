import time
import numpy as np
import mojito
from config import config
from agent import MasterAgent


class Stockenv:
    def __init__(self):
        """Initializes the stock environment with a master agent and broker connection."""
        self.master_agent = MasterAgent()
        self.agents = self.master_agent.agents
        self.timestep = 0
        self.broker = self._initialize_broker()

    def _initialize_broker(self):
        """Initializes the broker with credentials from the config."""
        bc = config.brokerConfig
        return mojito.KoreaInvestment(
            api_key=bc.key,
            api_secret=bc.secret,
            acc_no=bc.acc_no,
            mock = bc.moc
        )

    def step(self, actions):
        """
        Takes a step in the environment with the provided actions.

        Args:
            actions (list): List of actions for each agent.

        Returns:
            tuple: Next observations, rewards, done flags, and additional info.
        """
        next_obs, rewards, dones, infos = [], [], [], []
        self.update_info()  # Update account and stock info

        for agent, action in zip(self.agents, actions):
            self._execute_action(agent, action)
            agent.update_past_state()

            next_ob = agent.get_curr_state()
            rewards.append(agent.reward)
            dones.append(agent.done)
            infos.append(agent.history)

            next_obs.append(next_ob)

        self.timestep += 1
        return np.array(next_obs), np.array(rewards), np.array(dones), infos

    def _execute_action(self, agent, action):
        """Executes the action for a given agent based on the action value."""
        agent.action.action_p = action[0][0].item()

        if agent.action.action_p <= 9:  # Selling
            self._handle_sell(agent, action)
        elif agent.action.action_p <= 20:  # Buying
            self._handle_buy(agent, action)

    def _handle_sell(self, agent, action):
        """Handles the sell action for an agent."""
        sell_percentage = (action + 1) * 10
        agent.sell_quantity = agent.curr_state.quantity * (sell_percentage / 100.0)
        agent.sell_quantity = max(agent.sell_quantity, 0)

        if agent.sell_quantity > 0:
            self.broker.create_market_sell_order(
                symbol=agent.stock_code,
                quantity=agent.sell_quantity
            )
        time.sleep(3)

    def _handle_buy(self, agent, action):
        """Handles the buy action for an agent."""
        buy_percentage = (action - 9) * 10
        max_buyable_quantity = self.master_agent._dcna_tot_amt // agent.curr_state.price
        agent.buy_quantity = max_buyable_quantity * (buy_percentage / 100.0)
        agent.buy_quantity = max(agent.buy_quantity, 0)

        if agent.buy_quantity > 0:
            self.broker.create_market_buy_order(
                symbol=agent.stock_code,
                quantity=agent.buy_quantity
            )
        time.sleep(3)

    def reset(self):
        """Resets the environment to the initial state."""
        self.master_agent.reset()
        self.update_info()

        obs = [agent.get_curr_state()[0] for agent in self.agents]
        return np.array(obs)

    def update_info(self):
        """Updates the agent and account information from the broker."""
        resp = self.broker.fetch_balance()
        self.master_agent._tot_evlu_amt = resp['output2'][0]['tot_evlu_amt']
        self.master_agent._dcna_tot_amt = resp['output2'][0]['dnca_tot_amt']

        for agent in self.agents:
            self._update_agent_info(agent, resp)

    def _update_agent_info(self, agent, resp):
        """Updates the current state of an agent based on the API data."""
        agent.curr_state.info = self.broker.fetch_price(agent.stock_code)

        for comp in resp['output1']:
            if agent.stock_code == comp['pdno']:
                agent.curr_state.pchs_amt = comp['pchs_amt']
                agent.curr_state.quantity = comp['hldg_qty']
                agent.curr_state.evlu_amt = comp['evlu_amt']
                agent.curr_state.price = agent.curr_state.info['output']['stck_prpr']

    def check_done(self):
        """Checks if the episode has ended based on the timestep."""
        return self.timestep > 1000


# Example of using the Stockenv class
if __name__ == "__main__":
    env = Stockenv()
    obs, ob_dim = env.reset()

    for episode in range(10):  # Run for 10 episodes
        actions = [agent.get_action() for agent in env.agents]  # Replace with your action fetching logic
        next_obs, rewards, dones, infos = env.step(actions)
        print(f"Episode: {episode}, Rewards: {rewards}")
