import gymnasium as gym
import torch

env = gym.make('CartPole-v1')

taille_state = env.observation_space.shape[0] #Box
taille_action = env.action_space.n
TAILLE_LATENT = 24

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

class DQN(torch.nn.Module):
    def __init__(self, taille_state, taille_action, lr = 1e-3, epoch=25):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.lr = lr
        self.epoch = epoch
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, TAILLE_LATENT),
                        torch.nn.ReLU(),
                        torch.nn.Linear(TAILLE_LATENT, self.taille_action)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.f_loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.net(x)
    
    def fit(self, x, y):
        for _ in range(self.epoch):
            y_hat = self.forward(x)
            loss = self.f_loss(y_hat, y)
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

class AgentDQN(torch.nn.Module):
    def __init__(self, taille_state, taille_action):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, TAILLE_LATENT),
                        torch.nn.ReLU(),
                        torch.nn.Linear(TAILLE_LATENT, self.taille_action)
        )
        
    def act(self, state):
        return torch.argmax(self.net(state))
