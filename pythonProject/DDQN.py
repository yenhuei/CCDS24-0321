import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import copy
from gym.envs.classic_control.EnvironmentGen import DroneEnv
from itertools import count
import time

#Saving run data
energyCostList = []
dataSizeList = []
runName = 'DDQN History 1e-2'

# hyper-parameters
NUM_EPISODES = 600
DISCOUNT = GAMMA = 0.001
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-2
UPDATE_RATE = 1e-3 #Update rate of the target network -> copying params from actor Target = (1-rate)Target + rate(Actor)
UPDATE_INTERVAL = Q_NETWORK_ITERATION = 100
MEMORY_CAPACITY = 1024*5
BATCH_SIZE = 128
n_actions = 100
n_obs = 10

env = gym.make("DroneEnv-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space
NUM_STATES = env.observation_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_actions, n_observations):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        # self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        return self.layer4(x)


state, info = env.reset()
policy_net = DQN(n_actions, n_obs).to(device)
target_net = DQN(n_actions, n_obs).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=False)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0

episode_durations = []
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result of '+ runName)
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Energy/kBits')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= BATCH_SIZE:
        means = durations_t.unfold(0, BATCH_SIZE, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(BATCH_SIZE-1), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #Input state and return max reward step returned by model
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    criterion = nn.MSELoss() #nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = NUM_EPISODES
else:
    num_episodes = 256

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    # print("Episode ", i_episode)
    totalReward = 0
    totalData = 0
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        reward, observation, info, currentTask, data = env.step(action)
        totalReward += reward[0]
        reward = torch.tensor(reward, device=device)
        totalData += data
        done = terminated = currentTask==10

        if terminated:
            next_state = None
            energyCostList.append(round(totalReward))
            dataSizeList.append(totalData)
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*UPDATE_RATE + target_net_state_dict[key]*(1-UPDATE_RATE)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(reward)
            plot_durations()
            break


f = open(runName + ' History.txt', 'w')
myString = str(dataSizeList) + '\n\n' + str(energyCostList)
f.write(myString)
f.close()

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()