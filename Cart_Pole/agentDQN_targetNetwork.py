from agentDQN import *
import copy as cp


class AgentDQN_TargetNetwork(AgentDQN):
    def __init__(self, taille_state, taille_action, gamma=0.99, batch=32, K=32):
        super().__init__(taille_state, taille_action, gamma, batch)
        self.K = K # nombre de pas pour maj Target Network
        self.counterK = 0 
        self.dqnTarget = cp.deepcopy(self.dqn)

    def replay(self, batch_seuil, decay):
        if self.buffer.getLen() < batch_seuil:
            return
        mini_batch = self.buffer.sampleState(self.batch_size)
        for state, reward, action, done, state_suivant in mini_batch:
            # target est calculé par Target Network
            y_action = reward + self.gamma*torch.max(self.dqnTarget.forward(state_suivant)).detach().item()
            if done:
                y_action = reward # bah, si done -> perdu donc faut savoir pour eviter
            y = self.dqnTarget.forward(state)
            y[0][action] = y_action # tel action amene a tel score, 1 batch, tensor donc [0] ..
            # MAJ Q-network
            self.dqn.fit(state, y, epoch=1)
            self.counterK += 1
            # Update target network every K steps ...
            if self.counterK == self.K:
                self.dqnTarget.updateParam(self.dqn)
                self.counterK = 0 # reset counter, ugly code ...
        if decay and self.explore > self.explore_min:
            self.explore *= self.explore_decay


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    TAILE_STATE = env.observation_space.shape[0] #Box
    TAILLE_ACTION = env.action_space.n
    TAILLE_BATCH = 32
    K = 32
    gamma = 0.95
    nb_episode = 400

    agent = AgentDQN_TargetNetwork(TAILE_STATE, TAILLE_ACTION, gamma, TAILLE_BATCH, K)
    score_max = -math.inf
    #path_best = 'bestModel/TNbest.pt'
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
    #print("max:", score_max)

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