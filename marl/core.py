import numpy as np
from config import config

config = config.stockConfig


class AgentAction:
    def __init__(self):
        self.action_p = None  # Real action (e.g., 10%, 20%, ...)
        self.action_c = None  # Communication action (e.g., 1, 2, 3, ...)

class AgentState:
    def __init__(self):
        self.price = 0
        self.quantity = 0
        self.info = {}
        self.pchs_amt = 0  # Purchase amount
        self.evlu_amt = 0  # Evaluation amount

    def reset(self):
        """Reset the state attributes."""
        self.price = 0
        self.quantity = 0
        self.info = {}
        self.pchs_amt = 0
        self.evlu_amt = 0

    def to_numpy(self):
        """Convert state attributes to a numpy array."""
        info_values = list(self.info.values())
        data = np.array([self.price, self.quantity, self.pchs_amt, self.evlu_amt] + info_values)
        return data, len(data)

class Agent:
    def __init__(self):
        self.stock_name = ''
        self.stock_code = ''
        self.reward = 0
        self.done = None
        self.info = None
        self.ob_dim = 0
        self.past_state = AgentState()
        self.curr_state = AgentState()
        self.action = AgentAction()
        self.history = {}

    def update_past_state(self):
        """Update past state to current state."""
        self.past_state.__dict__ = self.curr_state.__dict__.copy()

    def get_curr_state(self):
        """Get the current state as a numpy array."""
        return self.curr_state.to_numpy()

    def reset(self):
        """Reset the agent's attributes and states."""
        self.reward = 0
        self.done = None
        self.info = None
        self.action = AgentAction()  # Reset action
        self.history = {}  # Clear history
        self.past_state.reset()  # Reset past state
        self.curr_state.reset()  # Reset current state

class MasterAgent:
    def __init__(self):
        self._tot_evlu_amt = 0 # Total evaluation amount
        self._dcna_tot_amt = 100000000            # Cash amount
        self.nagents = len(config.stock_codes)
        self.agents = [Agent(stock_code) for _, stock_code in zip(range(self.nagents), config.stock_codes)]  # Create agents

    def reset(self):
        """Reset the master agent and all its agents."""
        self._tot_evlu_amt = 100000000
        self._dcna_tot_amt = 0
        for agent in self.agents:
            agent.reset()
