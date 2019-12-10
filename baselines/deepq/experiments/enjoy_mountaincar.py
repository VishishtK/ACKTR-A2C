import gym

from baselines import deepq


def main():
    env = gym.make("MountainCar-v0")
    act = deepq.load("mountaincar_model.pkl")
    episode_reward_list = []
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        episode_reward_list.append(episode_rew)
        print("Episode reward", episode_rew)
        print("Avg Reward Till Now", sum(episode_reward_list)/len(episode_reward_list))

if __name__ == '__main__':
    main()
