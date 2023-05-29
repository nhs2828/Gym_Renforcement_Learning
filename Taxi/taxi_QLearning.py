import gymnasium as gym
import os
import sys
sys.path.append(os.path.abspath("../agent/"))
from agent import *
from taxi import *
import copy as cp

def Q_learning(env, gamma, learning_rate, episode):
    """
        Q_t+1(s,a) = Q_t(s,a) + lr*(reward(s,a,s') + gamma max_a'(Q(s',a')) - Q_t(s,a)
    """
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    # Initialisaion de Q-scores
    Q = np.zeros((nb_state, nb_action))
    for _ in range(episode):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            new_state,reward, done, _, _ = env.step(int(action))
            Q[state][action] = Q[state][action] + learning_rate*(reward + gamma*np.max(Q[new_state]) - Q[state][action])
            state = new_state
    # Greedy
    pi = ini_pi(env)
    for state in range(nb_state):
        pi[state] = np.argmax(Q[state])
    return pi

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode = 'human')
    gamma = 0.95
    learning_rate = 0.1
    nb_episode = 10000
    print("Begin training")
    pi = Q_learning(gym.make('Taxi-v3'), gamma, learning_rate, nb_episode)
    print("Traning done")
    play(AgentPolicy(env, pi))