from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, initial_state=None, action_space_dim=None, state_space_dim=None):
        self.initial_state = initial_state
        self.action_dim = action_space_dim
        self.state_dim = state_space_dim

    @abstractmethod
    def model_logic(self, state, action, options=None):
        pass
    
    
    
    @abstractmethod
    def valid_actions(self, state):
        pass
    
    @abstractmethod
    def reset_model(self):
        pass