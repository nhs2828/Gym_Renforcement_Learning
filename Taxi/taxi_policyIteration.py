import gymnasium as gym
import copy as cp
import os
import sys
sys.path.append(os.path.abspath("../agent/"))
from agent import *
from taxi import *

def eval_pol(pi, env, gamma, thresh):
    nb_state = env.observation_space.n
    MDP = env.env.P
    
    def recompense(s,a): # probleme taxi MDP deterministe, donc y a pas s1
        return MDP[s][a][0][2] # reward inx 2
    
    def state_suivant(s,a):# probleme taxi MDP deterministe, donc y a pas s1
        return MDP[s][a][0][1] # state suivant inx 1
    
    # Ini
    V_courant = np.zeros(nb_state)
    V_suivant = np.zeros(nb_state)
    
    for state in range(nb_state):
        V_suivant[state] = 1*recompense(state, pi[state])+gamma*V_courant[state_suivant(state, pi[state])]
    while np.linalg.norm(V_suivant-V_courant) > thresh:
        V_courant = cp.deepcopy(V_suivant)
        for state in range(nb_state):
            V_suivant[state] = 1*recompense(state, pi[state])+gamma*V_courant[state_suivant(state, pi[state])]
    return V_suivant

def get_pol(V, env, gamma):
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    MDP = env.env.P
    
    def recompense(s,a): # probleme taxi MDP deterministe, donc y a pas s1
        return MDP[s][a][0][2] # reward inx 2
    
    def state_suivant(s,a):# probleme taxi MDP deterministe, donc y a pas s1
        return MDP[s][a][0][1] # state suivant inx 1
    
    pi = {}
    for state in range(nb_state):
        r = np.zeros(nb_action)
        for a in range(nb_action):
            r[a] = recompense(state,a) + gamma*V[state_suivant(state,a)]
        pi[state] = np.argmax(r)
    return pi

def policy_iteration(env, gamma, thresh):
    # Ini
    pi = ini_pi(env)    
    while True: # Algo commence
        V = eval_pol(pi, env, gamma, thresh)
        pi_suivant = get_pol(V, env, gamma)
        if pi == pi_suivant:
            break
        pi = cp.deepcopy(pi_suivant)
    return pi


if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode = 'human')
    gamma = 0.99
    thresh = 0.005
    pi = policy_iteration(env,gamma, thresh)
    play(AgentPolicy(env,pi))
