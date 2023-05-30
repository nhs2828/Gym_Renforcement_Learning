import torch
from random import sample
from torch.distributions import Normal

class DQN(torch.nn.Module):
    def __init__(self, taille_state, taille_action, lr = 3e-4, hidden = 24):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.lr = lr
        self.hidden = hidden
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, self.hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden, self.hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden, self.taille_action)
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

    def updateParam(self, dqn):
        """
            Pour Target Network, copier les parametres d'autre DQN
        """
        self.load_state_dict(dqn.state_dict())

    def getNet(self):
        return self.net

class Critic(torch.nn.Module):
    def __init__(self, taille_state, taille_action, hidden):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.hidden = hidden
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, self.hidden[0]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden[0], self.hidden[1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden[1], self.taille_action)
        )

    def forward(self, x):
        return self.net(x)
    
    def getNet(self):
        return self.net


class Actor(torch.nn.Module):
    def __init__(self, taille_state, taille_action, hidden):
        super().__init__()
        self.taille_state = taille_state
        self.taille_action = taille_action
        self.hidden = hidden
        self.net = torch.nn.Sequential(
                        torch.nn.Linear(self.taille_state, self.hidden[0]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden[0], self.hidden[1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(self.hidden[1], self.taille_action),
                        torch.nn.Tanh() # lunar action [-1,1]
        )

    def forward(self, x):
        return self.net(x)

    def getNet(self):
        return self.net

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
    
def addGaussianNoise(action, sigma=0.1):
    a = torch.tensor(action)
    dist = Normal(a, sigma)
    return dist.sample().detach().numpy()


