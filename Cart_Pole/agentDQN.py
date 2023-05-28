import gymnasium as gym
import torch
import math
import time
from random import sample


class Buffer:
    def __init__(self, taille_max):
        self.taille = taille_max
        self.memoire = []
        
    def getLen(self):
        return len(self.memoire)
    
    def add(self, element):
        if self.getLen() >= self.taille:
            del self.memoire[0]
        self.memoire.append(element)

    def sampleState(self, taille_sample):
        return sample(self.memoire, taille_sample)

class DQN(torch.nn.Module):
    def __init__(self, taille_state, taille_action, lr = 3e-4):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.lr = lr
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, 24),
                        torch.nn.ReLU(),
                        torch.nn.Linear(24, 24),
                        torch.nn.ReLU(),
                        torch.nn.Linear(24, self.taille_action)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.f_loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.net(x)
    
    def fit(self, x, y, epoch=10):
        for _ in range(epoch):
            y_hat = self.forward(x)
            loss = self.f_loss(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

class AgentDQN():
    def __init__(self, taille_state, taille_action, gamma=0.99, batch=32):
        self.taille_state = taille_state
        self.taille_action = taille_action
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


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    TAILE_STATE = env.observation_space.shape[0] #Box
    TAILLE_ACTION = env.action_space.n
    TAILLE_BATCH = 32
    gamma = 0.95
    nb_episode = 400

    agent = AgentDQN(TAILE_STATE, TAILLE_ACTION, gamma, TAILLE_BATCH)
    score_max = -math.inf
    path_best = 'bestModel/best.pt'
    for i in range(nb_episode):
        #state, _ = env.reset(seed=0)
        state, _ = env.reset()
        state = torch.as_tensor(state).view(1,-1) # to tensor et reshape (Batch, blah blah)
        cum_reward = 0
        for frame in range(5000):
            action = agent.act(state)
            state_suivant, reward, done, _, info = env.step(action)
            state_suivant = torch.as_tensor(state_suivant).view(1,-1)
            agent.store(state, reward, action, done, state_suivant)
            agent.replay(TAILLE_BATCH, i%30==0) # sample batch scenario de 1.5*batch
            cum_reward += reward
            if done:
                score_max = max(score_max, cum_reward)
                print(f"Episode {i}/{nb_episode}, fini à {frame} frame, explore {agent.explore},score: {cum_reward}")
                break
            state = torch.tensor(state_suivant)
    torch.save(agent.dqn, path_best)
    print("max:", score_max)
    # avec image sur model final
    fps = 60
    env = gym.make('CartPole-v1', render_mode = 'human')
    r = 0
    state, _ = env.reset()
    state = torch.as_tensor(state).view(1,-1)
    for _ in range(5000):
        action = agent.act_opt(state)
        state_suivant, reward, done, _, info = env.step(action)
        state_suivant = torch.as_tensor(state_suivant).view(1,-1) # un peu moche, faut reorganiser un peu ...
        r += reward
        if fps > 0:
            env.render()
            time.sleep(1/fps)
        if done:
            break
        state = torch.tensor(state_suivant)
    print(f"Fini, score = {r}")





