import copy
import glob
import os

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from arguments import get_args
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from kfac import KFACOptimizer
from model import CNNPolicy
from storage import RolloutStorage

args = get_args()
args.algo == 'a2c'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.json'))
    for f in files:
        os.remove(f)





os.environ['OMP_NUM_THREADS'] = '1'

envs = SubprocVecEnv([
    make_env(args.env_name, args.seed, i, args.log_dir)
    for i in range(args.num_processes)
])

obs_shape = envs.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

actor_critic = CNNPolicy(obs_shape[0], envs.action_space)

if envs.action_space.__class__.__name__ == "Discrete":
    action_shape = 1
else:
    action_shape = envs.action_space.shape[0]

if args.cuda:
    actor_critic.cuda()

optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)

rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space)
current_state = torch.zeros(args.num_processes, *obs_shape)

def update_current_state(state):
    shape_dim0 = envs.observation_space.shape[0]
    state = torch.from_numpy(state).float()
    if args.num_stack > 1:
        current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
    current_state[:, -shape_dim0:] = state

state = envs.reset()
update_current_state(state)

rollouts.states[0].copy_(current_state)

episode_rewards = torch.zeros([args.num_processes, 1])
final_rewards = torch.zeros([args.num_processes, 1])

if args.cuda:
    current_state = current_state.cuda()
    rollouts.cuda()

for j in range(num_updates):
    for step in range(args.num_steps):
        value, action = actor_critic.act(Variable(rollouts.states[step], volatile=True))
        cpu_actions = action.data.squeeze(1).cpu().numpy()

        state, reward, done, info = envs.step(cpu_actions)
        reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
        episode_rewards += reward

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        final_rewards *= masks
        final_rewards += (1 - masks) * episode_rewards
        episode_rewards *= masks

        if args.cuda:
            masks = masks.cuda()

        if current_state.dim() == 4:
            current_state *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_state *= masks

        update_current_state(state)
        rollouts.insert(step, current_state, action.data, value.data, reward, masks)

    next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

    if hasattr(actor_critic, 'obs_filter'):
        actor_critic.obs_filter.update(rollouts.states[:-1].view(-1, *obs_shape))

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

    values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(Variable(rollouts.states[:-1].view(-1, *obs_shape)), Variable(rollouts.actions.view(-1, action_shape)))

    values = values.view(args.num_steps, args.num_processes, 1)
    action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

    advantages = Variable(rollouts.returns[:-1]) - values
    value_loss = advantages.pow(2).mean()

    action_loss = -(Variable(advantages.data) * action_log_probs).mean()

    optimizer.zero_grad()
    (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

    nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

    optimizer.step()

    rollouts.states[0].copy_(rollouts.states[-1])

    if j % args.save_interval == 0 and args.save_dir != "":
        save_path = os.path.join(args.save_dir, args.algo)
        try:
            os.makedirs(save_path)
        except OSError:
            pass

        save_model = actor_critic
        if args.cuda:
            save_model = copy.deepcopy(actor_critic).cpu()
        torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

    if j % args.log_interval == 0:
        print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
            format(j, (j + 1) * args.num_processes * args.num_steps,
                   final_rewards.mean(),
                   final_rewards.median(),
                   final_rewards.min(),
                   final_rewards.max(), -dist_entropy.data[0],
                   value_loss.data[0], action_loss.data[0]))


