# Gym_Renforcement_Learning
<b>Taxi-v3</b></br>
For this game Taxi, 4 algorithmes were applied, which are Policy Iteration, Value Iteration, Q-Learning and SARSA.
![taxi](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/e8fa41f7-0f42-44ea-8bc0-3a92eb86d5b4)

<b>Cart Pole</b></br>
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


![rl-video-episode-0](https://github.com/nhs2828/Gym_Renforcement_Learning/assets/78078713/e7df11bc-97f4-4347-9be8-3dd8de4bf2ed)
