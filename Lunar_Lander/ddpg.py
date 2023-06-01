import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath("../agent/"))
from agent import *
sys.path.append(os.path.abspath("../Network/"))
from config import cfgDDGP
from gym.wrappers import RecordVideo

path_model = 'bestModel/net400.pt'
path_video = 'video'

if __name__ == '__main__':
    #env
    env = gym.make("LunarLander-v2",continuous = True, render_mode = 'rgb_array')
    env = RecordVideo(env, path_video)
    #agent
    agent = AgentDDPG(env, cfgDDGP)
    agent.setActor(path_model)

    r = 0
    state, _ = env.reset()
    state = torch.as_tensor(state).view(1,-1)
    for _ in range(5000):
        action = agent.act_opt(state)
        state_suivant, reward, done, _, info = env.step(action)
        state_suivant = torch.as_tensor(state_suivant).view(1,-1) # un peu moche, faut reorganiser un peu ...
        #transform action 
        action = torch.as_tensor(action).view(1,-1)
        r += reward
        if done:
            break
        state = torch.tensor(state_suivant)
    print(f"Fini, score = {r}")