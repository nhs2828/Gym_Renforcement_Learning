import os
import sys
import copy as cp
from torch.distributions.uniform import Uniform
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
                y_action = reward # bah, si done -> no more futur
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



class AgentDDPG(Agent):
    def __init__(self, env, config):
        super().__init__(env)
        # continuous environment
        assert env.continuous == True
        # param env
        if env.unwrapped.spec.id == "LunarLander-v2":
            self.taille_state = self.env.observation_space.shape[0] #Box
            self.taille_action = self.env.action_space.shape[0] #Box
        # Neural networks
        self.Critic = Critic(self.taille_state + self.taille_action, 1, config["hidden_layer"]["critic"])
        self.CriticTarget = cp.deepcopy(self.Critic)
        self.Actor = Actor(self.taille_state , self.taille_action, config["hidden_layer"]["actor"]) # I make only 1 actor here
        self.ActorTarget = cp.deepcopy(self.Actor)
        # Optim
        self.optCritic = torch.optim.Adam(self.Critic.parameters(), config["lr"]["critic"]) # optim Critic
        self.optActor = torch.optim.Adam(self.Actor.parameters(), config["lr"]["actor"]) # optim Actor
   
        # Loss
        self.f_loss = torch.nn.MSELoss()
        # Hyper-param
        self.batch_size = config["batch"]
        self.buffer = Buffer(config["len_buffer"])
        self.explore = config["explore"]
        self.explore_min = config["explore_min"]
        self.explore_decay = config["explore_decay"]
        self.gamma = config["gamma"]
        self.tau = config["tau"] # Tau to update Target Network
        self.sigma = config["noise"] # noise
        self.decay_sigma = config["noise_decay_step"]
        self.step_count = 0
        self.start_learning = config["start_learning_step"]

    def act(self, state):
        self.step_count += 1
        if torch.rand(1).item() < self.explore: # solf greedy
            if self.env.unwrapped.spec.id == "LunarLander-v2":
                return np.random.uniform(-1,1,(2,)) # np.array([main, lateral])
        if self.step_count>0 and self.step_count%self.decay_sigma==0: #decay noise every 5000 steps
            self.sigma *= 0.95
        a = self.Actor(state)[0].cpu().detach().numpy()
        a = addGaussianNoise(a, self.sigma)
        return a
    
    def act_opt(self, state): # 1 actor
        return self.Actor(state)[0].detach().numpy()
    
    def saveActor(self, path):
        torch.save(self.Actor, path)
    
    def setActor(self, path): 
        self.Actor = torch.load(path)

    def addReward(self, reward):
        """
            Tracking last N-reward
        """
        self.buffer.addReward(reward)
        return self.buffer.getLastMean()

    def store(self, state, reward, action, done,  state_suivant):
        self.buffer.add([state, reward, action, done, state_suivant])

    def updateTargetDDPG(self):
        # critic
        for param, target_param in zip(self.Critic.parameters(), self.CriticTarget.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # actor
        for param, target_param in zip(self.Actor.parameters(), self.ActorTarget.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def updateNetworks(self, state, reward, action, done, state_suivant):
        assert list(state.shape) == [1, self.taille_state]
        assert list(action.shape) == [1, self.taille_action]
        state_suivant.requires_grad = True # to update Actor
        # prepare data actor
        outActor_suivant = self.ActorTarget.forward(state_suivant) # pi(state_suivant)
        # prepare data critic
        input_Q = torch.hstack((state, action))
        input_Q_suivant = torch.hstack((state_suivant, outActor_suivant)) 
        #
        Q = self.Critic.forward(input_Q) # y_hat
        Q_suivant = self.CriticTarget.forward(input_Q_suivant)
        y = reward + self.gamma*Q_suivant
        if done:
            y = torch.tensor(reward, dtype=torch.float32).view(1,-1)
        # Critic, 1 epoch
        loss = self.f_loss(Q, y)
        self.optCritic.zero_grad()
        loss.backward()
        self.optCritic.step()

    def replay(self, decay):
        if self.buffer.getLen() < self.start_learning:
            return
        mini_batch = self.buffer.sampleState(self.batch_size)
        batch_state = []
        for state, reward, action, done, state_suivant in mini_batch:
            batch_state.append(state.squeeze(0))
            self.updateNetworks(state, reward, action, done, state_suivant)
        batch_state = torch.stack(batch_state)
        # Actor
        outActor = self.Actor.forward(batch_state)
        intputQ_actor_update = torch.hstack((batch_state, outActor))
        lossActor = -self.Critic(intputQ_actor_update).mean() # if Q is bad -> loss is positif, Q is good -> loss is neg (good thing)
        self.optActor.zero_grad()
        lossActor.backward()
        self.optActor.step()
        # Target
        self.updateTargetDDPG()
        if decay and self.explore > self.explore_min:
            self.explore *= self.explore_decay




