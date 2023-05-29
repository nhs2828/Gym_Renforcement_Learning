import gymnasium as gym
import math
import os
import sys

sys.path.append(os.path.abspath("../agent/"))
from agent import *


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    TAILLE_BATCH = 32
    gamma = 0.95
    nb_episode = 400

    agent = AgentDQN(env, gamma, TAILLE_BATCH)
    score_max = -math.inf
    #path_best = 'bestModel/best.pt'
    for i in range(nb_episode):
        #state, _ = env.reset(seed=0)
        state, _ = env.reset()
        state = torch.as_tensor(state).view(1,-1) # to tensor et reshape (Batch, blah blah)
        cum_reward = 0
        for frame in range(5000):
            action = agent.act(state)
            state_suivant, reward, done, _, info = env.step(action)
            state_suivant = torch.as_tensor(state_suivant).view(1,-1)
            agent.store(state, reward, action, done, state_suivant)
            agent.replay(TAILLE_BATCH, i%30==0) # sample batch scenario de 1.5*batch
            cum_reward += reward
            if done:
                score_max = max(score_max, cum_reward)
                print(f"Episode {i}/{nb_episode}, fini à {frame} frame, explore {agent.explore},score: {cum_reward}")
                break
            state = torch.tensor(state_suivant)
    #torch.save(agent.dqn, path_best)
    print("max:", score_max)
    # avec image sur model final
    fps = 60
    env = gym.make('CartPole-v1', render_mode = 'human')
    r = 0
    state, _ = env.reset()
    state = torch.as_tensor(state).view(1,-1)
    for _ in range(5000):
        action = agent.act_opt(state)
        state_suivant, reward, done, _, info = env.step(action)
        state_suivant = torch.as_tensor(state_suivant).view(1,-1) # un peu moche, faut reorganiser un peu ...
        r += reward
        if fps > 0:
            env.render()
            time.sleep(1/fps)
        if done:
            break
        state = torch.tensor(state_suivant)
    print(f"Fini, score = {r}")
