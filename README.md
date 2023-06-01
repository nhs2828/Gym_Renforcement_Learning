# Gym_Renforcement_Learning
## Taxi-v3
For this game Taxi, 4 algorithmes were applied, which are Policy Iteration, Value Iteration, Q-Learning and SARSA.
![taxi](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/e8fa41f7-0f42-44ea-8bc0-3a92eb86d5b4)

## Cart Pole
<p>For this mini-game, I've used 2 versions, Deep-Q Network and Deep-Q Network with Target Network, both with replay buffer to train the agent.</p>
<p>Objective: Minimise Temporal Difference Loss (TD Loss) using MSE Loss, which is:</p>

```math
\delta_i = r_i + \gamma*\text{max}_a Q(s_{i+1}, a | \theta) - Q(s_i, a_i |\theta)
```
where the target is: 
```math 
r_i + \gamma*\text{max}_a Q(s_{i+1}, a | \theta)
```
The result after 400 episodes </br>
![cart_Pole](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/8370611c-54cb-4da7-9065-712243486937)
<p>Comparison between DQN and DQN with Target Network, DQN with Target Network is more stable, since we fixed target value, the performance is much more better.</p>


![fig](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/18d44506-3bd4-4ef4-9ae4-6d04ed2ba686)

## LunarLander-v2
For the game where actions are continuous, the goal is to reach 200 points for average point of last 100 episodes, I use the algorithm DDGP for my model [[ref]](https://arxiv.org/pdf/1509.02971.pdf).

Things I've learnt over this implementation
- The algorithm is very expensive, choosing the parameters and hyper-parameters is crucial
- 256 neurones for all hidden layers is enough to reach the goal
- Gaussian noise for action works well, with decay to reduce the noise overtime to collect more accuracy data [[ref]](https://web.stanford.edu/class/aa228/reports/2019/final162.pdf)
- Exploration without learning for first K-steps

**The result after 400 episodes**

![rl-video-episode-0](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/e7df11bc-97f4-4347-9be8-3dd8de4bf2ed)

The result of training over 500 episodes, X-axis is number of episode, Y-axis is the reward, I've reached the goal (200 on avg of last 100 episodes) at 500th episode. The training time on colab is 10 hours

![ddpg_plot](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/235a83b9-678c-4e47-b83c-57fed958bfc9)
