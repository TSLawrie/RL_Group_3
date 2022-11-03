import random
from collections import deque
from datetime import datetime
from typing import NamedTuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class DQN():
    def __init__(self, env_id, seed, buffer_size=int(1e6), batch_size=128, eps_decay=0.995, eps_min=0.01, gamma=0.99, tau=1e-3, lr=5e-4, update_every=4, learn_every=4, clipping=False, reward_range=(-1, 1), grad_range=(-1, 1), hidden_1_size=128, hidden_2_size=128, max_steps=1000, n_eps_solved=10, log_name=None):
        """ Initialize an Agent object.
        """
        self.env_id = env_id
        _env = gym.make(env_id)
        self.state_size = _env.observation_space.shape[0]
        self.action_size = _env.action_space.n
        _env.close()
        self.seed = random.seed(seed)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(
            self.state_size, self.action_size, seed, hidden_1_size, hidden_2_size).to(self.device)
        self.target_net = QNetwork(
            self.state_size, self.action_size, seed, hidden_1_size, hidden_2_size).to(self.device)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(
            self.action_size, buffer_size, batch_size, self.device, seed)
        self.timestep = 0

        self.eps = 0.15
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.learn_every = learn_every
        self.max_steps = max_steps

        self.eps_max = 1.0
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.max_steps = max_steps
        self.n_eps_solved = n_eps_solved

        # Tensorboard setup
        if log_name is None:
            NOW = datetime.now()
            log_name = NOW.strftime("%Y%m%d-%H%M%S")

        self.writer = SummaryWriter(
            log_dir=f'Discretev1/Logs/{log_name}',
            comment=f'batch_size={self.batch_size} buffer_size={buffer_size} tau={self.tau} gamma={self.gamma} lr={self.lr} update_every={self.update_every} hidden_1={hidden_1_size} hidden_2={hidden_2_size})'
        )

        self.hyperparam_dict = {
            'batch_size': self.batch_size,
            'memory_size': buffer_size,
            'tau': self.tau,
            'gamma': self.gamma,
            'lr': self.lr,
            'update_freq': self.update_every,
            'max_steps': self.max_steps,
            'hidden_1': hidden_1_size,
            'hidden_2': hidden_2_size,
            'reward_clipping': clipping,
            'n_eps_solved': self.n_eps_solved,
            'seed': seed
        }

        self.curr_episode_loss = 0

        # Reward clipping setup

        if clipping:
            self.max_reward = reward_range[1]
            self.min_reward = reward_range[0]
        else:
            self.max_reward = None
            self.min_reward = None

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.timestep = self.timestep % self.update_every
        if self.timestep == 0:
            if len(self.memory) > self.batch_size:
                if self.min_reward:
                    batch = self.memory.sample(
                        min_reward=self.min_reward, max_reward=self.max_reward)
                else:
                    batch = self.memory.sample(
                        min_reward=self.min_reward, max_reward=self.max_reward)
                self.learn(batch, self.gamma)

    def act(self, state, eps=0.):
        if random.random() > eps:
            state = torch.from_numpy(
                state).float().unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                action_values = self.policy_net(state)
            self.policy_net.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, batch, gamma):
        states, actions, rewards, next_states, dones = batch

        next_state_targets = self.target_net(
            next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_state_targets * (1 - dones))

        q_values = self.policy_net(states).gather(1, actions)

        loss = F.huber_loss(q_values, q_targets)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.curr_episode_loss += loss.item()

        self.soft_update(self.policy_net, self.target_net, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                (1.0-tau)*target_param.data + tau*local_param.data)

    def train(self, num_episodes=1000):
        rewards = []
        total_loss = 0
        last_10 = deque(maxlen=self.n_eps_solved)
        solve_episode = -1

        self.eps = self.eps_max

        env = gym.make(self.env_id)
        for i in range(1, num_episodes+1):
            state = env.reset()
            score = 0
            self.curr_episode_loss = 0
            for t in range(self.max_steps):
                if i % 10 == 0:
                    env.render()
                action = self.act(state, self.eps)
                next_state, reward, done, _ = env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    self.add_to_tensorboard(
                        i, score, self.curr_episode_loss, t)
                    total_loss += self.curr_episode_loss
                    break

            rewards.append(score)
            last_10.append(score)
            print(
                f'Episode {i} completed in {t} steps | score = {score} avg = {np.mean(last_10)}')

            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            if np.mean(last_10) >= 200.0:
                self.save_network()
                print("Environment solved! Model saved and exiting")
                solve_episode = i
                break
        # Closing writer
        self.save_hyperparams_tensorboard(np.mean(last_10), total_loss, solve_episode)
        self.writer.close()
        return rewards

    def eval(self, num_episodes):
        rewards = []
        total_loss = 0
        last_10 = deque(maxlen=self.n_eps_solved)
        solve_episode = -1

        self.load_network()

        self.curr_episode_loss = 0
        self.eps = 0 # No exploration during evaluation

        env = gym.make(self.env_id)
        for i in range(1, num_episodes+1):
            state = env.reset()
            score = 0
            for t in range(self.max_steps):
                if i % 10 == 0:
                    env.render()
                action = self.act(state, self.eps)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                score += reward
                if done:
                    self.add_to_tensorboard(
                        i, score, self.curr_episode_loss, t)
                    total_loss += self.curr_episode_loss
                    break

            rewards.append(score)
            last_10.append(score)
            print(
                f'Episode {i} completed in {t} steps | score = {score} avg = {np.mean(last_10)}')

        # Closing writer
        self.save_hyperparams_tensorboard(np.mean(last_10), total_loss, solve_episode)
        self.writer.close()
        return rewards

    def save_network(self):
        torch.save(self.policy_net.state_dict(), 'checkpoint.pth')

    def load_network(self, file_location = 'checkpoint.pth'):
        self.policy_net.load_state_dict(torch.load(file_location))
        self.policy_net.eval()
        self.target_net.load_state_dict(torch.load(file_location))
        self.target_net.eval()

    def add_to_tensorboard(self, episode, total_reward, total_loss, episode_duration):
        self.writer.add_scalar('Episode Reward', total_reward, episode)
        self.writer.add_scalar('Loss', total_loss, episode)
        self.writer.add_scalar('Episode Duration', episode_duration, episode)
        self.writer.add_scalar('Epsilon', self.eps, episode)

        self.writer.add_histogram(
            "q_network.linear1.bias", self.policy_net.fc_input.bias, episode)
        self.writer.add_histogram(
            "q_network.linear1.weight", self.policy_net.fc_input.weight, episode)
        self.writer.add_histogram(
            "q_network.linear2.bias", self.policy_net.fc_hidden.bias, episode)
        self.writer.add_histogram(
            "q_network.linear2.weight", self.policy_net.fc_hidden.weight, episode)
        self.writer.add_histogram(
            "q_network.linear3.bias", self.policy_net.fc_output.bias, episode)
        self.writer.add_histogram(
            "q_network.linear3.weight", self.policy_net.fc_output.weight, episode)

        self.writer.add_histogram(
            "q_target_network.linear1.bias", self.target_net.fc_input.bias, episode)
        self.writer.add_histogram(
            "q_target_network.linear1.weight", self.target_net.fc_input.weight, episode)
        self.writer.add_histogram(
            "q_target_network.linear2.bias", self.target_net.fc_hidden.bias, episode)
        self.writer.add_histogram(
            "q_target_network.linear2.weight", self.target_net.fc_hidden.weight, episode)
        self.writer.add_histogram(
            "q_target_network.linear3.bias", self.target_net.fc_output.bias, episode)
        self.writer.add_histogram(
            "q_target_network.linear3.weight", self.target_net.fc_output.weight, episode)

        self.writer.flush()

    def save_hyperparams_tensorboard(self, total_reward, total_loss, solved_episode = -1):
        result_dict = {'Episode Reward': total_reward,
                       'Loss': total_loss,
                       'Solved Episode': solved_episode
                       }
        self.writer.add_hparams(self.hyperparam_dict, result_dict)


class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: int


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state=state, action=action,
                           reward=reward, next_state=next_state, done=int(done)))

    def sample(self, min_reward=None, max_reward=None):
        batch = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in batch])
        actions = np.vstack([e.action for e in batch])
        rewards = np.vstack([e.reward for e in batch])
        next_states = np.vstack([e.next_state for e in batch])
        dones = np.vstack([e.done for e in batch]).astype(np.uint8)

        if min_reward is not None and max_reward is not None:
            rewards = np.clip(rewards, a_min=min_reward, a_max=max_reward)

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_1_units=128, hidden_2_units=128):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_input = nn.Linear(state_size, hidden_1_units)
        self.fc_hidden = nn.Linear(hidden_1_units, hidden_2_units)
        self.fc_output = nn.Linear(hidden_2_units, action_size)

    def forward(self, state):
        x = self.fc_input(state)
        x = F.relu(x)
        x = self.fc_hidden(x)
        x = F.relu(x)
        return self.fc_output(x)


if __name__ == '__main__':
    # DQN with optimum hyperparameters
    ag = DQN("LunarLander-v2", 0, clipping=False, 
        gamma=0.99,
        hidden_1_size=32,
        hidden_2_size=32,
        tau=0.001, 
        lr=0.00005,
        update_every=4,
        max_steps=600)
    rewards = ag.train(500)
