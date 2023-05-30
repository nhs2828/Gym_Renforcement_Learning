import os
import sys
import gymnasium as gym
import time
import math
sys.path.append(os.path.abspath("../agent/"))
from agent import *
sys.path.append(os.path.abspath("../Network/"))
from config import cfgDDGP




if __name__ == '__main__':
    env = gym.make("LunarLander-v2",continuous = True)
    nb_episode = 500
    f = open("score/reward", "w")
    agent = AgentDDPG(env, cfgDDGP)
    score_max = -math.inf
    path_best = 'bestModel/best.pt'
    for i in range(nb_episode):
        #state, _ = env.reset(seed=0)
        state, _ = env.reset()
        state = torch.as_tensor(state).view(1,-1) # to tensor et reshape (Batch, blah blah)
        cum_reward = 0
        for frame in range(5000):
            action = agent.act(state)
            action = addGaussianNoise(action, sigma = 0.1) # add Gaussian Noise ...
            state_suivant, reward, done, _, info = env.step(action)
            state_suivant = torch.as_tensor(state_suivant).view(1,-1)
            #transform action 
            action = torch.as_tensor(action).view(1,-1)
    
            agent.store(state, reward, action, done, state_suivant)
            agent.replay(cfgDDGP["batch"], i>=70) # sample batch scenarios after 69th episode
            cum_reward += reward
            if done:
                score_max = max(score_max, cum_reward)
                print(f"Episode {i}/{nb_episode}, fini à {frame} frame, explore {agent.explore},score: {cum_reward}")
                f.write(f"{cum_reward}\n")
                break
            state = torch.tensor(state_suivant)
    f.close()
    # torch.save(agent.actor, path_best)
    # print("max:", score_max)
    # # avec image sur model final
    # fps = 60
    # env = gym.make("LunarLander-v2",continuous = True, render_mode='human')
    # r = 0
    # state, _ = env.reset()
    # state = torch.as_tensor(state).view(1,-1)
    # for _ in range(5000):
    #     action = agent.act_opt(state)
    #     state_suivant, reward, done, _, info = env.step(action)
    #     state_suivant = torch.as_tensor(state_suivant).view(1,-1) # un peu moche, faut reorganiser un peu ...
    #     r += reward
    #     if fps > 0:
    #         env.render()
    #         time.sleep(1/fps)
    #     if done:
    #         break
    #     state = torch.tensor(state_suivant)
    # print(f"Fini, score = {r}")