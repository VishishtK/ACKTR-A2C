import gym

from baselines import deepq


def main():
    env = gym.make("Enduro-v0")
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([64], layer_norm=True)
    act = deepq.learn(
        env,
        q_func=model,
        lr=2.33e-4,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=10,
        param_noise=True,
        gamma=0.99
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("mountaincar_model.pkl")


if __name__ == '__main__':
    main()
