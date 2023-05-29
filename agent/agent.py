import os
import sys
import copy as cp

sys.path.append(os.path.abspath("../Network"))
from DQN import *

class Agent():
    def __init__(self, env):
        self.env = env
    
    def act(self, obs):
        pass

    def store(self, obs, action, new_obs, reward):
        pass

class AgentRandom(Agent):
    def __init__(self, env):
        super().__init__(env)

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

class AgentDQN(Agent):
    def __init__(self,env, gamma=0.99, batch=32):
        super().__init__(env)
        self.taille_state = self.env.observation_space.shape[0] #Box
        self.taille_action = self.env.action_space.n
        self.batch_size = batch
        self.buffer = Buffer(5*self.batch_size)
        # NN
        self.dqn = DQN(self.taille_state, self.taille_action)
        # Hyper-param
        self.explore = 1.0
        self.explore_min = 0.01
        self.explore_decay = 0.995
        self.gamma = gamma
        
    def act(self, state):
        if torch.rand(1).item() < self.explore: # solf greedy
            return torch.randint(0, self.taille_action, (1,)).item()
        return torch.argmax(self.dqn(state)).item()
    
    def act_opt(self, state):
        return torch.argmax(self.dqn(state)).item()
    
    def setDQN(self, path):
        self.dqn = torch.load(path)

    def store(self, state, reward, action, done,  state_suivant):
        self.buffer.add([state, reward, action, done, state_suivant])

    def replay(self, batch_seuil, decay):
        if self.buffer.getLen() < batch_seuil:
            return
        mini_batch = self.buffer.sampleState(self.batch_size)
        for state, reward, action, done, state_suivant in mini_batch:
            y_action = reward + self.gamma*torch.max(self.dqn.forward(state_suivant)).detach().item()
            if done:
                y_action = reward # bah, si done -> perdu donc faut savoir pour eviter
            y = self.dqn.forward(state)
            y[0][action] = y_action # tel action amene a tel score
            self.dqn.fit(state, y, epoch=1) 
        if decay and self.explore > self.explore_min:
            self.explore *= self.explore_decay

class AgentDQN_TargetNetwork(AgentDQN):
    def __init__(self, env, gamma=0.99, batch=32, K=32):
        super().__init__(env, gamma, batch)
        self.K = K # nombre de pas pour maj Target Network
        self.counterK = 0 
        self.dqnTarget = cp.deepcopy(self.dqn)

    def replay(self, batch_seuil, decay):
        if self.buffer.getLen() < batch_seuil:
            return
        mini_batch = self.buffer.sampleState(self.batch_size)
        for state, reward, action, done, state_suivant in mini_batch:
            # target est calculé par Target Network
            y_action = reward + self.gamma*torch.max(self.dqnTarget.forward(state_suivant)).detach().item()
            if done:
                y_action = reward # bah, si done -> perdu donc faut savoir pour eviter
            y = self.dqnTarget.forward(state)
            y[0][action] = y_action # tel action amene a tel score, 1 batch, tensor donc [0] ..
            # MAJ Q-network
            self.dqn.fit(state, y, epoch=1)
            self.counterK += 1
            # Update target network every K steps ...
            if self.counterK == self.K:
                self.dqnTarget.updateParam(self.dqn)
                self.counterK = 0 # reset counter, ugly code ...
        if decay and self.explore > self.explore_min:
            self.explore *= self.explore_decay