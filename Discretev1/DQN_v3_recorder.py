from DQN_v3 import DQN
from gym.wrappers.monitoring import video_recorder
import numpy as np
from collections import deque
import gym
from torch.utils.tensorboard import SummaryWriter

def run_record(env_id, num_episodes = 100, folder="recordings", video_name="LunarLander", eval_mode=False, agent=None, checkpoint_file = 'checkpoint.pth', seed = 0, random=False):
    # Check provided parameters are ok
    if num_episodes <= 0:
        raise f'Cannot run for {num_episodes} episodes - must be positive'

    # Check if folder exists (create one if not)
    if not os.path.isdir(folder):
        os.mkdir(folder)

    # Create agent or use param one
    if agent is None and not random:
        agent = DQN(env_id, seed)

    env = gym.make(env_id)
    env.seed(seed)
    vid = video_recorder.VideoRecorder(env, base_path=folder+'/'+video_name)

    if random:
        run_random(env, vid, num_episodes, video_name)
        return

    # Either run training mode or eval mode
    if eval_mode:
        run_eval(env, vid, num_episodes, agent, checkpoint_file)
    else:
        run_train(env, vid, num_episodes, agent)

def run_train(env, recorder, num_episodes, agent):
    rewards = []
    last_10 = deque(maxlen=10)
    solve_episode = -1
    agent.eps = agent.eps_max
    total_loss = 0

    for i in range(1,num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(agent.max_steps):
            if i % 10 == 0:
                env.render(mode='rgb_array')
                recorder.capture_frame()
            action = agent.act(state, agent.eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        agent.add_to_tensorboard(i, score, agent.curr_episode_loss, t)
        total_loss += agent.curr_episode_loss
        rewards.append(score)
        last_10.append(score)
        print(f'Episode {i} completed in {t} steps | score = {score} avg = {np.mean(last_10)}')

        agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)

        if np.mean(last_10) >= 200.0:
            agent.save_network()
            print("Environment solved! Model saved and exiting")
            solve_episode = i
            break

    # Closing writer
    agent.save_hyperparams_tensorboard(np.mean(last_10), total_loss, solve_episode)
    agent.writer.close()
    
def run_eval(env, recorder, num_episodes, agent, checkpoint_file = 'checkpoint.pth'):
    rewards = []
    last_10 = deque(maxlen=10)
    solve_episode = -1

    agent.load_network(checkpoint_file)

    agent.curr_episode_loss = 0
    agent.eps = 0 # No exploration during evaluation

    total_loss = 0
    for i in range(1, num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(agent.max_steps):
            if i % 10 == 0:
                env.render(mode='rgb_array')
                recorder.capture_frame()
            action = agent.act(state, agent.eps)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break
        agent.add_to_tensorboard(i, score, agent.curr_episode_loss, t)
        total_loss += agent.curr_episode_loss
        rewards.append(score)
        last_10.append(score)
        print(
            f'Episode {i} completed in {t} steps | score = {score} avg = {np.mean(last_10)}')

    # Closing writer
    agent.save_hyperparams_tensorboard(np.mean(last_10), total_loss, solve_episode)
    agent.writer.close()

def run_random(env, recorder, num_episodes, log_name):
    rewards = []
    last_10 = deque(maxlen=10)

    writer = SummaryWriter(
            log_dir=f'Discretev1/Logs/{log_name}',
            comment=f'RANDOM_AGENT_DISCRETE'
        )

    for i in range(1,num_episodes+1):
        state = env.reset()
        score = 0
        for t in range(1000):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break

        add_random_tensorlog(i, score, t, writer)
        rewards.append(score)
        last_10.append(score)
        print(f'Episode {i} completed in {t} steps | score = {score} avg = {np.mean(last_10)}')

    writer.close()

def add_random_tensorlog(episode, total_reward, episode_duration, writer):
    writer.add_scalar('Episode Reward', total_reward, episode)
    writer.add_scalar('Episode Duration', episode_duration, episode)

    writer.flush()

import os

if __name__ == '__main__':
    seed = 0
    agent = DQN("LunarLander-v2", 0, clipping=False, 
        gamma=0.99,
        hidden_1_size=32,
        hidden_2_size=32,
        tau=0.001, 
        lr=0.00005,
        update_every=4,
        max_steps=600)
    run_record('LunarLander-v2', 100, 'default_videos', f'DQNEvaluation_{seed}', False, seed=seed, agent=agent)
    