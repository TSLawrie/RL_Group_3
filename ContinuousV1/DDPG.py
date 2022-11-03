from logging import exception
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import deque
from torch.utils.tensorboard import SummaryWriter
#import torch.profiler
from statistics import mean
#import matplotlib.pyplot as plt
import pandas as pd
import sys, getopt
import logging
from datetime import date, datetime
import math


class Experience():
    """Simple experience class for accessing data easily"""
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ReplayMemory():
    """Replay memory buffer for storing the agent's experiences. A deque of instances of the Experience class. The most recent experience is located at the right end of the deque (biggest index)."""
    def __init__(self, size):
        self.N = size
        self.buffer = deque(maxlen=self.N)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, size):
        """Sample a batch of experiences of size N from the buffer. Each experience can only be selected once."""
        batch = np.random.choice(np.array(self.buffer), size=size)

        states, actions, rewards, next_states = [], [], [], []

        for experience in batch:
            states.append(experience.state)
            actions.append(experience.action)
            rewards.append(experience.reward)
            next_states.append(experience.next_state)

        return states, actions, rewards, next_states

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

    #def __repr__(self):
    #    writer = SummaryWriter('runs/experiment_1')
    #    writer.add_graph(self)
    #    writer.close()
    #    return super().__repr__()

class Tuner():
    def __init__(self, agent):
        self.agent = copy.deepcopy(agent)

    def opti_sigma(self):
        """Returns the sigma value optimized for getting rewards greater than the specified minimum optimal reward"""
        successes = {}
        for sigma in np.arange(0, 0.2, 0.01):
        # Make the environment
            env = gym.make(self.agent.gym_env)
            state = env.reset()
            
            total_successes = 0
            # Pre-fill Replay Memory Buffer
            for step in range(10000):
                #if step>95:
                #    env.render()
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                if reward >= self.agent.optim_min_reward:
                    total_successes += 1

                # Reset environment if agent crashes
                if done:
                    env.reset()

            successes[sigma] = total_successes / 10000

        return max(successes, key=successes.get)


class DDPG_Lunarlander(): 
    """
    DDPG Agent whose task is to learn to land the lunar lander safely as fast as possible.
    Tips:
        - A random agent will usually go through 90-120 steps before crashing and terminating the episode.
    """
    def __init__(self, max_eps, sigma, sigma_ini, sigma_decay, sigma_min, batch_size, memory_size, prefill_size, beta, gamma, actor_lr,  critic_lr, update_freq, hidden_size, gpu, n_eps_solved, seed):
        self.gym_env = "LunarLanderContinuous-v2"
        self.mu = 0     # mean for normal distribution of noise
        self.sigma = sigma    # standard deviation for normal distribution of noise
        self.sigma_ini = sigma_ini
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)
        self.prefill_size = prefill_size
        self.episodes = max_eps
        self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        self.beta = beta    # learning rate for target networks
        self.gamma = gamma      # discount factor
        self.loss = nn.HuberLoss(reduction='mean', delta=1)      # critic MSE loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_freq = update_freq       # Number of steps between each network update
        self.optim_min_reward = 0      # Minimum total episode reward to optimize hyperparameters for (using random agent)
        self.ep_rewards = []       # Database of rewards per episode
        self.hidden_size = hidden_size
        self.n_eps_solved = n_eps_solved
        self.seed = seed
        self.writer = SummaryWriter('./logs/ddpg/' + NOW.strftime("%Y%m%d-%H%M%S") + '/', comment=f'sigma={sigma} sigma_ini={sigma_ini} sigma_decay={sigma_decay} sigma_min={sigma_min} batch_size={batch_size} memory_size={memory_size} prefill_size={prefill_size} beta={beta} gamma={gamma} actor_lr={actor_lr} critic_lr={critic_lr} update_freq={update_freq} hidden_size={hidden_size} n_eps_solved={n_eps_solved} seed={seed}')
    
    def train(self):
        """Train the agent."""
        # Profiler Logging
        #prof = torch.profiler.profile(
        #        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/ddpg'),
        #        record_shapes=True,
        #        with_stack=True)
        #prof.start()

        # Make the environment
        env = gym.make(self.gym_env)
        env.seed(self.seed)
        state = env.reset()
        
        # Pre-fill Replay Memory Buffer
        for step in range(self.prefill_size):
            #if step>95:
            #    env.render()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            self.memory.push(state, action, reward, next_state, done)
            state = next_state

            # Reset environment if agent crashes
            if done:
                state = env.reset()

        #self.sigma = Tuner(self).opti_sigma()
        #print(self.sigma)

        # Initialise actor and critic networks with random weights
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        actor = ActorNetwork(state_size, self.hidden_size, action_size, self.device).to(self.device)
        critic = CriticNetwork(state_size+action_size, self.hidden_size, 1, self.device).to(self.device)

        # Make networks run on multiple GPUs
        #if torch.cuda.device_count() > 1:
        #    actor = nn.DataParallel(actor)
        #    critic = nn.DataParallel(critic)

        #Initialise target actor and target critic networks with same weights
        target_actor = copy.deepcopy(actor).to(self.device)
        target_critic = copy.deepcopy(critic).to(self.device)

        #writer = SummaryWriter('logs/net')
        #writer.add_graph(actor, torch.rand((8,), device=self.device), verbose=True)

        # Initialise target networks' learning rate
        beta = self.beta

        # Average total reward of past 10 episodes
        avg_rewards = deque(maxlen=self.n_eps_solved)

        # Highest episodic reward achieved
        best_reward = 0

        for episode in range(self.episodes):
            # Decay sigma
            #self.sigma = (self.sigma_ini - self.sigma_min) * pow(sigma_decay, episode) + self.sigma_min
            # Initialise state S1
            state = env.reset()

            has_positive_reward = False
            total_reward = 0 
            critic_total_loss = 0
            actor_total_loss = 0
            done = False

            if episode % 1 == 0:
                render = True
            else:
                render = False

            # Enable anomaly detection
            #torch.autograd.set_detect_anomaly(True)
            
            step = 0
            while not done:
                # Rendering environment
                if render:
                    env.render()
                # Select action
                action = actor.forward(torch.from_numpy(state).to(self.device))
                # Initialise random process N for exploration
                noise = torch.from_numpy(np.random.normal(self.mu,self.sigma,size=2)).to(self.device)
                # Add noise
                action = torch.add(action, noise).cpu().detach().numpy()
                # Execute action and observe reward and next state
                next_state, reward, done, info = env.step(action)
                # Store transition in replay memory
                self.memory.push(state, action, reward, next_state, done)
                # Sample random minibatch of transitions from replay memory
                states, actions, rewards, next_states = self.memory.sample_batch(self.batch_size)
                states = torch.FloatTensor(np.array(states)).to(self.device)
                actions = torch.FloatTensor(np.array(actions)).to(self.device)
                rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
                next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

                # Instanciate optimizers
                actor_optimizer = optim.Adam(actor.parameters(), lr=self.actor_lr)
                critic_optimizer = optim.Adam(critic.parameters(), lr=self.critic_lr)

                # Calculate critic loss
                target_critic_vals = target_critic.forward(next_states, target_actor.forward(next_states))
                target_critic_updated_vals = rewards + self.gamma * target_critic_vals
                critic_vals = critic.forward(states, actions)
                critic_loss = self.loss(target_critic_updated_vals, critic_vals).to(self.device)
                critic_optimizer.zero_grad()

                # Perform gradient descent on critic
                critic_loss.backward()
                critic_optimizer.step()

                # Calculate actor loss
                actor_loss = - critic.forward(states, actor.forward(states)).mean()
                actor_optimizer.zero_grad()

                # Perform gradient descent on actor
                actor_loss.backward()
                actor_optimizer.step()            

                # Update target networks
                if step % self.update_freq == 0:
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(self.beta * param + (1 - self.beta) * target_param)
                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(self.beta * param + (1 - self.beta) * target_param)
                
                # Update state
                state = next_state

                # Update total reward
                total_reward += reward

                # Update total losses
                critic_total_loss += critic_loss.item()
                actor_total_loss += actor_loss.item()

                # Profiler Logging
                #prof.step()
            
                if done:
                    print(f"Episode: {episode}, Reward: {total_reward}, Sigma: {self.sigma}")
                    self.ep_rewards.append([episode, total_reward])

                    self.writer.add_scalar('Episode Reward', total_reward, episode)
                    self.writer.add_scalar('Critic Loss', critic_total_loss, episode)
                    self.writer.add_scalar('Actor Loss', actor_total_loss, episode)

                    self.writer.add_histogram("critic_linear1.bias", critic.linear1.bias, episode)
                    self.writer.add_histogram("critic_linear1.weight", critic.linear1.weight, episode)
                    self.writer.add_histogram("critic_linear2.bias", critic.linear2.bias, episode)
                    self.writer.add_histogram("critic_linear2.weight", critic.linear2.weight, episode)
                    self.writer.add_histogram("critic_linear3.bias", critic.linear3.bias, episode)
                    self.writer.add_histogram("critic_linear3.weight", critic.linear3.weight, episode)

                    self.writer.add_histogram("actor_linear1.bias", actor.linear1.bias, episode)
                    self.writer.add_histogram("actor_linear1.weight", actor.linear1.weight, episode)
                    self.writer.add_histogram("actor_linear2.bias", actor.linear2.bias, episode)
                    self.writer.add_histogram("actor_linear2.weight", actor.linear2.weight, episode)
                    self.writer.add_histogram("actor_linear3.bias", actor.linear3.bias, episode)
                    self.writer.add_histogram("actor_linear3.weight", actor.linear3.weight, episode)

                    avg_rewards.append(total_reward)

                    # Save reward if above 200 and better than previous best reward
                    if total_reward >= 200 and total_reward >= best_reward:
                        best_reward = total_reward
                        # Save best neural networks
                        torch.save(critic.state_dict(), f'./ContinuousV1/Models/critic_' + NOW.strftime("%Y%m%d-%H%M%S") + '_best.pth')
                        torch.save(actor.state_dict(), f'./ContinuousV1/Models/actor_' + NOW.strftime("%Y%m%d-%H%M%S") + '_best.pth')

                    break
            
                #avg_rewards.append(np.mean(total_rewards))

                step += 1

            # If average reward of past 10 episodes is at least 200, problem is solved
            if sum(avg_rewards) / len(avg_rewards) >= 200:
                break

        # Tensorboard logging
        #self.writer.add_graph(actor, states)
        #self.writer.add_graph(critic, (states, actions))
        self.writer.add_hparams({'sigma': self.sigma, 'sigma_ini': self.sigma_ini, 'sigma_decay': self.sigma_decay, 'sigma_min': self.sigma_min,'batch_size': self.batch_size, 'memory_size': self.memory_size, 'prefill_size': self.prefill_size, 'beta': self.beta, 'gamma': self.gamma, 'actor_lr': self.actor_lr, 'critic_lr': self.critic_lr, 'update_freq': self.update_freq, 'hidden_size': f'[{self.hidden_size[0]},{self.hidden_size[1]}]', 'n_eps_solved': self.n_eps_solved, 'seed': self.seed}, {'Episode Reward': total_reward, 'Critic Loss': critic_total_loss, 'Actor Loss': actor_total_loss})

        # Tensorboard logging
        self.writer.close()

        # Save neural networks
        torch.save(critic.state_dict(), f'./ContinuousV1/Models/critic_' + NOW.strftime("%Y%m%d-%H%M%S") + '_last.pth')
        torch.save(actor.state_dict(), f'./ContinuousV1/Models/actor_' + NOW.strftime("%Y%m%d-%H%M%S") + '_last.pth')

        # Profiler logging
        #prof.stop()

    #def plot_rewards(self):
    #    """Plots the episodic rewards during training"""
    #    ep_rewards = self.ep_rewards
    #    graph = plt.plot(ep_rewards[0], ep_rewards[1])
    #    plt.grid(axis='y', linewidth=0.25)
    #    plt.show()

    def store_rewards(self, agent_n="0"):
        """Store the list of episodic rewards in a csv file. THIS FUNCTION NEEDS TO BE MODIFIED WHEN TESTING DIFFERENT HYPERPARAMETERS."""
        ep_rewards = self.ep_rewards
        ep_rewards = pd.DataFrame(ep_rewards, columns=['Episodes', 'Rewards'])
        ep_rewards.to_csv(f'ContinuousV1/Results/agent{self.hidden_size}_{self.beta}_{agent_n}', index=False)


options = ["max_eps=", "sigma=", "sigma_ini=", "sigma_decay=", "sigma_min=", "batch_size=", "memory_size=", "prefill_size=", "beta=", "gamma=", "actor_lr=", "critic_lr=", "update_freq=", "agent_n=", "hidden_size=", "gpu=", "n_eps_solved=", "seed="]
explanations = {"max_eps": "Maximum number of episodes. If problem is not solved, program will exit upon reaching the maximum. Default=1000.",
                "sigma": "NaN",
                "batch_size": "NaN",
                "memory_size": "NaN",
                "prefill_size": "NaN",
                "beta": "NaN",
                "gamma": "NaN",
                "actor_lr": "NaN",
                "critic_lr": "NaN",
                "update_freq": "NaN",
                "agent": "NaN",
                "hidden_size": "Format as [i,i] where i is hidden layer size. NO SPACES.",
                "gpu": "GPU to use. Format 'cuda:i' where i is gpu number."
                ""
                }

def help(options, explanations):
    print("Usage:\n")
    print("ddpg.py --hyperparameter <value> --hyperparameter <value> ...\n")
    print("Hyperparameters:")
    for option in options:
        print(f"{option[:-1]}: {explanations[option[:-1]]}")

# Default values
max_eps = 1000
sigma = 0.1
sigma_ini = 0
sigma_decay = 0
sigma_min = 0
batch_size = 64
memory_size = 20000
prefill_size = 1000
beta = 0.001
gamma = 0.99
actor_lr = 5e-5
critic_lr = 5e-4
update_freq = 3
hidden_size = [32,64]
agent_n = '3'
gpu = 'cuda:0'
n_eps_solved = 10
seed = 0

try:
    opts, args = getopt.getopt(sys.argv[1:], shortopts=["h"], longopts=options)
except Exception as e:
    print(e)
    help(options, explanations)

if "-h" in args:
    help(options, explanations)
else:
    for opt, arg in opts:
        if opt == "--max_eps":
            max_eps = int(arg)
        if opt == "--sigma":
            sigma = float(arg)
        if opt == "--sigma_ini":
            sigma_ini = float(arg)
        if opt == "--sigma_decay":
            sigma_decay = float(arg)
        if opt == "--sigma_min":
            sigma_min = float(arg)
        if opt == "--batch_size":
            batch_size = int(arg)
        if opt == "--memory_size":
            memory_size = int(arg)
        if opt == "--prefill_size":
            prefill_size = int(arg)
        if opt == "--beta":
            beta = float(arg)
        if opt == "--gamma":
            gamma = float(arg)
        if opt == "--actor_lr":
            actor_lr = float(arg)
        if opt == "--critic_lr":
            critic_lr = float(arg)
        if opt == "--update_freq":
            update_freq = int(arg)
        if opt == "--agent_n":
            agent_n = arg
        if opt == "--hidden_size":
            hidden_size = [int(arg[arg.find('[')+1:arg.find(',')]), int(arg[arg.find(',')+1:arg.find(']')])]
        if opt == "--gpu":
            gpu = arg
        if opt == "n_eps_solved":
            n_eps_solved = int(arg)
        if opt == "--seed":
            seed = int(arg)

    try:
        for seed in range(0,4):
            NOW = datetime.now()
            agent = DDPG_Lunarlander(max_eps, sigma, sigma_ini, sigma_decay, sigma_min, batch_size, memory_size, prefill_size, beta, gamma, actor_lr,  critic_lr, update_freq, hidden_size, gpu, n_eps_solved, seed)
            agent.train()
    except Exception as e:
        # Create a logging instance
        logger = logging.getLogger('ddpg')
        logger.setLevel(logging.INFO) # you can set this to be DEBUG, INFO, ERROR

        # Assign a file-handler to that instance
        fh = logging.FileHandler("error.txt")
        fh.setLevel(logging.INFO) # again, you can set this differently

        # Format your logs (optional)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter) # This will set the format to the file handler

        # Add the handler to your logging instance
        logger.addHandler(fh)

        logger.exception(e)
    except KeyboardInterrupt:
        agent.writer.close()

