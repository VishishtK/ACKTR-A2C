import torch
import argparse
from torch.autograd import Variable
from envs import make_env
import matplotlib.pyplot as plt

def runEnv(filePath):
    actor_critic = torch.load(filePath)
    actor_critic.eval()
    return actor_critic

env_name = "StarGunnerNoFrameskip-v4"
filePath_ACKTR = "trained_models/acktr/StarGunnerNoFrameskip-v4.pt"
filePath_A2C = "trained_models/a2c/StarGunnerNoFrameskip-v4.pt"
seeds = 1
num_stack = 4
filePath = [filePath_ACKTR]
Algos = ["ACKTR"]

RewardsACKTR = []
RewardsA2C = []
numEpisodes = 100
for z in range(len(filePath)):
    env = make_env(env_name, 1, 0, 'logs')()
    agent = runEnv(filePath[z])
    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    current_state = torch.zeros(1, *obs_shape)


    def GoToNextState(state):
        shape_dim0 = env.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    totalReward = 0
    env.render('human')
    state = env.reset()
    GoToNextState(state)
    Episodes = []
    for i in range(numEpisodes):
        Episodes.append(i)
    Episode = 0
    print("Running"+Algos[z])
    while True:
        value, action = agent.act(Variable(current_state, volatile=True),deterministic=True)
        cpu_actions = action.data.cpu().numpy()
        state, _1, done, _2 = env.step(cpu_actions[0])
        env.render('human')
        totalReward += _1
        if done:
            state = env.reset()
            agent = runEnv(filePath[z])
            if _2.get('ale.lives')==0:
                print("Episode Ended:")
                Reward = 240*totalReward
                print("Reward=", Reward)
                if z==0:
                    RewardsACKTR.append(Reward)
                else:
                    RewardsA2C.append(Reward)
                Episode+=1
                if Episode==numEpisodes:
                    break
                totalReward = 0
        GoToNextState(state)
plt.plot(Episodes,RewardsACKTR,"r",Episodes,RewardsA2C,'b')
plt.ylim(0, max(max(RewardsACKTR),max(RewardsA2C)))
plt.show()