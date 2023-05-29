from agentDQN import *
from gym.wrappers import RecordVideo

path_model = 'bestModel/best.pt'
path_video = 'video'

if __name__ == '__main__':
    #env
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    env = RecordVideo(env, path_video)
    TAILE_STATE = env.observation_space.shape[0] #Box
    TAILLE_ACTION = env.action_space.n
    #path DQN
    path_model = 'bestModel/best.pt'
    #agent
    agent = AgentDQN(TAILE_STATE, TAILLE_ACTION)
    agent.setDQN(path_model)

    r = 0
    state, _ = env.reset()
    state = torch.as_tensor(state).view(1,-1)
    for _ in range(5000):
        action = agent.act_opt(state)
        state_suivant, reward, done, _, info = env.step(action)
        state_suivant = torch.as_tensor(state_suivant).view(1,-1) # un peu moche, faut reorganiser un peu ...
        r += reward
        if done:
            break
        state = torch.tensor(state_suivant)
    print(f"Fini, score = {r}")