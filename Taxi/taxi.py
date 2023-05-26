import time
import numpy as np

def ini_pi(env):
    pi = {}
    nb_state = env.observation_space.n
    nb_action = env.action_space.n
    for state in range(nb_state):
        pi[state] = np.random.randint(nb_action)
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