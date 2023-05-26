import gymnasium as gym
import copy as cp
import numpy as np
from agent import *
from taxi import *

def value_iteration(env, gamma, thresh):
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    MDP = env.env.P
    V = np.random.randint(0,5, nb_state) # aleatoirement
    V_suivant = np.random.randint(0, 5, nb_state) # aleatoirement, peu important

    def recompense(s,a): # MDP deterministe
        return MDP[s][a][0][2]

    def state_suivant(s,a): # MDP deterministe
        return MDP[s][a][0][1]

    for state in range(nb_state):
        reward = np.zeros(nb_action)
        for a in range(nb_action):
            reward[a] = 1*recompense(state, a) + gamma*V[state_suivant(state,a)]
        V_suivant[state] = np.max(reward)

    # Calculer V*
    while np.linalg.norm(V_suivant-V) > thresh:
        V = cp.deepcopy(V_suivant)
        for state in range(nb_state):
            reward = np.zeros(nb_action)
            for a in range(nb_action):
                reward[a] = 1*recompense(state, a) + gamma*V[state_suivant(state,a)]
            V_suivant[state] = np.max(reward)

    # Calculer pi*
    pi = ini_pi(env)
    for state in range(nb_state):
        reward = np.zeros(nb_action)
        for a in range(nb_action):
            reward[a] = 1*recompense(state,a) + gamma*V[state_suivant(state, a)]
        pi[state] = np.argmax(reward)
    
    return pi


if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode = "human")
    gamma = 0.99
    thresh = 0.005
    pi = value_iteration(env,gamma, thresh)
    play(AgentPolicy(env,pi))


