class Agent():
    def __init__(self, env):
        self.env = env
    
    def act(self, obs):
        pass

    def store(self, obs, action, new_obs, reward):
        pass

class AgentRandom(Agent):
    def __init__(self, env):
        super.__init__(env)

    def act(self, obs):
        return self.env.action_space.sample()

    def store(self, obs, action, new_obs, reward):
        pass

class AgentPolicy(Agent):
    def __init__(self, env, pi):
        super().__init__(env)
        self.pi = pi

    def act(self, obs):
        return self.pi[obs]

    def store(self, obs, action, new_obs, reward):
        pass