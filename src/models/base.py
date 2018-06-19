from abc import ABC, abstractmethod, abstractproperty

class Model(ABC):
    @abstractproperty
    def stateful(self):
        pass

    @abstractmethod
    def start_state(self, batch_size):
        pass

    @abstractmethod
    def step(self, observations, states):
        pass

class TFQNetwork(Model):
    def __init__(self, session, num_actions, obs_vectorizer, name):
        self.session = session
        self.num_actions = num_actions
        self.obs_vectorizer = obs_vectorizer
        self.name = name
        self.variables = []

    @abstractmethod
    def transition_loss(self, target_net, obses, actions, rews, new_obses, terminals, discounts):
        pass

    @abstractproperty
    def input_dtype(self):
        pass

