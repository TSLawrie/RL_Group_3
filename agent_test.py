from audioop import avg
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime


class CriticNetwork(nn.Module):
    """Q-Network with one hidden layer"""
    def __init__(self, input_size, hidden_size, output_size, device):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size[0]).to(self.device)
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1]).to(self.device)
        self.linear3 = nn.Linear(hidden_size[1], output_size).to(self.device)

    def forward(self, state, action):
        """
        Return the value function (output layer) resulting from feeding forward a state and action (input layer) through the Critic Network
        """
        x = torch.cat([state, action], 1).to(self.device)
        x = F.relu(self.linear1(x)).to(self.device)
        x = F.relu(self.linear2(x)).to(self.device)
        x = self.linear3(x).to(self.device)

        return x

class ActorNetwork(nn.Module):
    """Policy-Network with one hidden layer"""
    def __init__(self, input_size, hidden_size, output_size, device):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, hidden_size[0]).to(self.device)
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1]).to(self.device)
        self.linear3 = nn.Linear(hidden_size[1], output_size).to(self.device)
        
    def forward(self, state):
        """
        Return the optimal action (output layer) resulting from feeding forward a state (input layer) through the Actor Network
        """
        x = F.relu(self.linear1(state)).to(self.device)
        x = F.relu(self.linear2(x)).to(self.device)
        x = torch.tanh(self.linear3(x)).to(self.device)

        return x


# Use same parameters as used for relevant model's training
device = "cuda:0"
gym_env = "LunarLanderContinuous-v2"
episodes = 100
n_eps_solved = 100
state_size = 8
hidden_size = [32,64]
action_size = 2
run = "20220505-173108"
best_or_last_model = "last"     # 'best' for testing best performing model during training, 'last' for model at end of training
NOW = datetime.now()

env = gym.make(gym_env)

critic = CriticNetwork(state_size+action_size, hidden_size, 1, device).to(device)
actor = ActorNetwork(state_size, hidden_size, action_size, device).to(device)

critic.load_state_dict(torch.load(f'./ContinuousV1/Models/critic_{run}_{best_or_last_model}.pth'))
critic.eval()
actor.load_state_dict(torch.load(f'./ContinuousV1/Models/actor_{run}_{best_or_last_model}.pth'))
actor.eval()

# Average total reward of past 10 episodes
avg_rewards = deque(maxlen=n_eps_solved)

writer = SummaryWriter('./logs/ddpg/' + NOW.strftime("%Y%m%d-%H%M%S") + '/')

for episode in range(episodes):
    state = env.reset()

    has_positive_reward = False
    total_reward = 0 
    done = False

    if episode % 1 == 0:
        render = True
    else:
        render = False
    
    step = 0
    while not done:
        # Rendering environment
        if render:
            env.render()
            ##time.sleep(0.01)
        # Select action
        action = actor.forward(torch.from_numpy(state).to(device)).cpu().detach().numpy()
        # Execute action and observe reward and next state
        next_state, reward, done, info = env.step(action)
        
        # Update state
        state = next_state

        # Update total reward
        total_reward += reward
    
        if done:
            print(f"Episode: {episode}, Reward: {total_reward}")
            avg_rewards.append(total_reward)
            writer.add_scalar('Episode Reward', total_reward, episode)
            break

        step += 1

    solved = False
    if len(avg_rewards) == n_eps_solved and sum(avg_rewards) / len(avg_rewards) >= 200:
        solved = True
        break

writer.close()
    
if solved:
    print(f"{gym_env} solved! Average reward of last {n_eps_solved} episodes: {sum(avg_rewards) / len(avg_rewards)}")
else:
    print(f"{gym_env} not solved... Average reward of last {n_eps_solved} episodes: {sum(avg_rewards) / len(avg_rewards)}")
