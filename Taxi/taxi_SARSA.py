import gymnasium as gym
from taxi import *
import os
import sys
sys.path.append(os.path.abspath("../agent/"))
from agent import *

def SARSA(env, gamma, lr, episode):
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    Q = np.zeros((nb_state, nb_action))

    for _ in range(episode):
        state, _ = env.reset()
        done = False
        action  = np.argmax(Q[state])
        while not done:
            new_state, reward, done, _, info = env.step(int(action))
            new_action = np.argmax(Q[new_state])
            Q[state][action] = Q[state][action] + lr*(reward + gamma*Q[new_state][new_action] - Q[state][action])
            state = new_state
            action = new_action
    
    pi = ini_pi(env)
    for state in range(nb_state):
        pi[state] = np.argmax(Q[state])

    return pi

if __name__ == '__main__':
    env = gym.make('Taxi-v3', render_mode = 'human')
    gamma = 0.95
    lr = 0.1
    nb_episode = 10000
    pi = SARSA(gym.make('Taxi-v3'), gamma, lr, nb_episode)
    play(AgentPolicy(env, pi))