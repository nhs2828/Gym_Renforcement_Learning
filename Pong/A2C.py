import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath("../agent/"))
from agent import *
sys.path.append(os.path.abspath("../Network/"))
from config import cfgA2C1step
from gym.wrappers import RecordVideo
import math

# path_model = 'bestModel/net400.pt'
# path_video = 'video'

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    nb_episode = 500
    #f = open("score/reward", "w")
    agent = AgentA2C_1step(env, cfgA2C1step)
    score_max = -math.inf
    path_best = 'bestModel/best.pt'
    for i in range(nb_episode):
        #state, _ = env.reset(seed=0)
        state, _ = env.reset()
        state = torch.as_tensor(state).view(1,-1) # to tensor et reshape (Batch, blah blah)
        cum_reward = 0
        for frame in range(5000):
            action = agent.act(state)
            state_suivant, reward, done, _, info = env.step(action.detach().numpy())
            state_suivant = torch.as_tensor(state_suivant).view(1,-1)
            #transform action 
            action = torch.as_tensor(action).view(1,-1)

            agent.updateNetworks(state, reward, action, done, state_suivant)

            cum_reward += reward
            if done:
                score_max = max(score_max, cum_reward)
                print(f"Episode {i}/{nb_episode}, fini à {frame} frame, explore {agent.explore},score: {cum_reward}")
                #f.write(f"{cum_reward}\n")
                break
            state = torch.tensor(state_suivant)
    #f.close()