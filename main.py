import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import itertools
from copy import deepcopy
import math
import argparse
from tqdm import tqdm
import time
from codecarbon import EmissionsTracker
import torch.distributed as dist
import pickle

from spikerl.agent import SpikingAC_Network
from spikerl.buffer import ReplayBuffer

def spikeRL_train(env_fn, actor_critic=SpikingAC_Network, ac_kwargs=dict(), seed=0,
                  steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
                  polyak=0.995, learning_rate=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
                  update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
                  noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
                  save_freq=5, norm_clip_limit=3, norm_update=50, use_cuda=True, rank=0, run=0,
                  device=None, local_rank=0, backend='nccl', args=None):

    if rank == 0:
        print(f"Using device: {device}")
        print(f"Batch Size: {batch_size}")
        print(f"Hidden Sizes: {ac_kwargs['hidden_sizes']}")

    ac_kwargs['device'] = device
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    ac = nn.parallel.DistributedDataParallel(ac, device_ids=[local_rank], output_device=local_rank)
    ac_targ = nn.parallel.DistributedDataParallel(ac_targ, device_ids=[local_rank], output_device=local_rank)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.module.q1.parameters(), ac.module.q2.parameters())

    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update)

    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = ac.module.q1(o, a)
        q2 = ac.module.q2(o, a)
        with torch.no_grad():
            spikeRL_targ = ac_targ.module.spikeRL(o2, batch_size)
            epsilon = torch.randn_like(spikeRL_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = spikeRL_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)
            q1_spikeRL_targ = ac_targ.module.q1(o2, a2)
            q2_spikeRL_targ = ac_targ.module.q2(o2, a2)
            q_spikeRL_targ = torch.min(q1_spikeRL_targ, q2_spikeRL_targ)
            backup = r + gamma * (1 - d) * q_spikeRL_targ
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2
        loss_info = dict(
            Q1Vals=q1.float().cpu().detach().numpy(),
            Q2Vals=q2.float().cpu().detach().numpy()
        )
        return loss_q, loss_info

    def compute_loss_spikeRL(data):
        o = data['obs']
        action = ac.module.spikeRL(o, batch_size)
        q1_spikeRL = ac.module.q1(o, action)
        return -q1_spikeRL.mean()

    spikeRL_optimizer = optim.Adam(ac.module.spikeRL.parameters(), lr=learning_rate)
    q_optimizer = optim.Adam(q_params, lr=q_lr)
    scaler = torch.cuda.amp.GradScaler()

    def update(data, timer):
        q_optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss_q, loss_info = compute_loss_q(data)
        scaler.scale(loss_q).backward()
        scaler.step(q_optimizer)
        scaler.update()
        if timer % policy_delay == 0:
            for p in q_params:
                p.requires_grad = False
            spikeRL_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_spikeRL = compute_loss_spikeRL(data)
            scaler.scale(loss_spikeRL).backward()
            scaler.step(spikeRL_optimizer)
            scaler.update()
            for p in q_params:
                p.requires_grad = True
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.module.act(torch.as_tensor(o, dtype=torch.float32, device=device), 1)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        test_reward_sum = 0
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                action = get_action(replay_buffer.normalize_obs(o), 0)[0]
                o, r, d, _, _ = test_env.step(action)
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum

    save_test_reward = []
    save_test_reward_steps = []
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    model_dir = "output"
    os.makedirs(model_dir, exist_ok=True)
    for t in tqdm(range(total_steps)):
        if t > start_steps:
            a = get_action(replay_buffer.normalize_obs(o), act_noise)[0]
        else:
            a = env.action_space.sample()
        o2, r, d, _, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len == max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch, timer=j)
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if rank == 0:
                if (epoch % save_freq == 0) or (epoch == epochs):
                    ac.module.spikeRL.to('cpu')
                    torch.save(ac.module.spikeRL.state_dict(),
                               f"{model_dir}/{args.env}_b{batch_size}_e{epochs}_run{run}_{backend}.pt")
                    ac.module.spikeRL.to(device)
                    pickle.dump([replay_buffer.mean, replay_buffer.var],
                                open(f"{model_dir}/{args.env}_b{batch_size}_e{epochs}_run{run}_{backend}_mean_var.p", "wb+"))
            test_reward_sum = test_agent()
            test_reward_sum_tensor = torch.tensor(test_reward_sum, device=device)
            dist.all_reduce(test_reward_sum_tensor, op=dist.ReduceOp.SUM)
            total_test_episodes = num_test_episodes * dist.get_world_size()
            test_mean_reward = test_reward_sum_tensor.item() / total_test_episodes
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            if rank == 0:
                print(f"Epoch: {epoch}, Step: {t+1}, Test Reward: {test_mean_reward:.3f}")
    if rank == 0:
        pickle.dump([save_test_reward, save_test_reward_steps],
                    open(f"{model_dir}/{args.env}_b{batch_size}_e{epochs}_run{run}_{backend}_test_rewards.p", "wb+"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Walker2d-v4')
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--run', type=int, default=0)

    args = parser.parse_args()
    run = args.run
    if args.backend == 'nccl':
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if rank == 0:
                print(f"Number of GPUs: {num_gpus}")
                print(f"Using device: {device}")
        else:
            device = torch.device("cpu")
            num_gpus = 0
            if rank == 0:
                print("Using CPU")
    elif args.backend == 'mpi':
        dist.init_process_group(backend='mpi')
        rank = dist.get_rank()
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            local_rank = rank % num_gpus
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if rank == 0:
                print(f"Number of GPUs: {num_gpus}")
                print(f"Using device: {device}")
        else:
            device = torch.device("cpu")
            num_gpus = 0
            if rank == 0:
                print("Using CPU")
    else:
        raise ValueError(f"Invalid backend: {args.backend}")

    USE_POISSON = False
    AC_KWARGS = dict(hidden_sizes=args.hidden_sizes,
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     device=device,
                     use_poisson=USE_POISSON)

    if rank == 0:
        tracker = EmissionsTracker()
        tracker.start()
    start_time = time.time()
    spikeRL_train(
        env_fn=lambda: gym.make(args.env),
        actor_critic=SpikingAC_Network,
        ac_kwargs=AC_KWARGS,
        learning_rate=1e-4,
        q_lr=1e-3,
        gamma=0.99,
        batch_size=args.batch_size,
        seed=10 * run,
        epochs=args.epochs,
        norm_clip_limit=3.0,
        rank=rank,
        run=run,
        device=device,
        local_rank=local_rank,
        backend=args.backend,
        args=args
    )
    end_time = time.time()
    training_time = end_time - start_time
    if rank == 0:
        emissions: float = tracker.stop()
        total_energy = tracker.final_emissions_data.energy_consumed
        print(f"SpikeRL Distributed ({args.backend}) Training Results on {args.env}")
        print("---------------------------------------------------")
        print(f"SpikeRL_Execution_Time_(s) = {training_time:.3f}")
        print(f"SpikeRL_Carbon_Emissions_(kgCO2) = {emissions:.3f}")
        print(f"SpikeRL_Total_Energy_Consumed_(kWh) = {total_energy:.3f}")

    dist.destroy_process_group()
