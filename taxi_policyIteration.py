import gymnasium as gym
import time
import copy as cp
import numpy as np

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


def ini_pi(env):
    pi = {}
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    for state in range(nb_state):
        pi[state] = np.random.randint(nb_action)
    return pi

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

def play(agent, fps = 60, verbose = False):
    obs, _ = agent.env.reset()
    eps = 500
    cum_r = 0
    for i in range(eps):
        last_obs = obs
        action = agent.act(obs)
        obs, reward, done, _, info = agent.env.step(int(action))
        agent.store(last_obs, action, obs, reward)
        cum_r += reward
        if fps > 0:
            agent.env.render()
            if verbose:
                print(f"Iter{i}", info)
            time.sleep(1/fps)
        if done:
            break
    print("Reward", cum_r)

if __name__ == '__main__':
    env = gym.make("Taxi-v3", render_mode = 'human')
    gamma = 0.99
    thresh = 0.005
    pi = policy_iteration(env,gamma, thresh)
    play(AgentPolicy(env,pi))
