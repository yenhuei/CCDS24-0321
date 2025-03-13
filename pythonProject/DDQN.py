import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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
from gym.envs.classic_control.EnvironmentGen import DroneEnv, NUM_DEVICES
from itertools import count
import time
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')
assert(writer != None)
runName = 'DDQN with Different LR'
DQN = False
result1 = []
result2 = []
result3 = []
loss1 = []
loss2 = []
loss3 = []
time_list05 = []
time_list10 = []
time_list20 = []
time_list_no = []
time_list_full = []
energy_list05 = []
energy_list10 = []
energy_list20 = []
energy_list_no = []
energy_list_full = []
runNumber = 0
i_episode = 0

# hyper-parameters
NUM_EPISODES = 600
DISCOUNT = GAMMA = 0.001
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000
LR = 5e-2
UPDATE_RATE = 1e-3 #Update rate of the target network -> copying params from actor Target = (1-rate)Target + rate(Actor)
UPDATE_INTERVAL = Q_NETWORK_ITERATION = 100
MEMORY_CAPACITY = 1024*8
BATCH_SIZE = 256
n_actions = 100
n_obs = NUM_DEVICES

env = gym.make("DroneEnv-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space
NUM_STATES = env.observation_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "cost"))
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
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer4(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            #Input state and return max cost step returned by model
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_costs(show_result=False):
    global costs_t, costs_t2, costs_t3, runNumber
    plt.figure(1)
    if show_result:
        plt.title('Result of '+ runName)
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Energy/kBits')

    if runNumber==0:
        costs_t = torch.tensor(episode_costs, dtype=torch.float)
        plt.plot(-costs_t.numpy(), label='LR=0.05')
    elif runNumber == 1:
        costs_t2 = torch.tensor(episode_costs, dtype=torch.float)
        plt.plot(-costs_t.numpy(), label='LR=0.05')
        plt.plot(-costs_t2.numpy(), label='LR=0.10')
    elif runNumber == 2:
        costs_t3 = torch.tensor(episode_costs, dtype=torch.float)
        plt.plot(-costs_t.numpy(), label='LR=0.05')
        plt.plot(-costs_t2.numpy(), label='LR=0.10')
        plt.plot(-costs_t3.numpy(), label='LR=0.20')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            # pass
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    global steps_done, writer, i_episode, runNumber
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
    cost_batch = torch.cat(batch.cost)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        policy_q_values = policy_net(non_final_next_states)
        target_q_values = target_net(non_final_next_states)
        next_state_values[non_final_mask] = target_q_values.gather(1, torch.max(policy_q_values, 1)[1].unsqueeze(1)).squeeze(1)

    expected_state_action_values = (next_state_values * GAMMA) + cost_batch
    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    if runNumber == 0:
        # writer.add_scalar("Loss/Training Loss for DDQN LR=0.05", loss, global_step=steps_done%6001)
        loss1.append(loss)
    elif runNumber == 1:
        # writer.add_scalar("Loss/Training Loss for DDQN LR=0.10", loss, global_step=steps_done%6001)
        loss2.append(loss)
    elif runNumber == 2:
        # writer.add_scalar("Loss/Training Loss for DDQN LR=0.20", loss, global_step=steps_done%6001)
        loss3.append(loss)

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


for runNumber in range(3):
    state, info = env.reset()
    policy_net = DQN(n_actions, n_obs).to(device)
    target_net = DQN(n_actions, n_obs).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    memory = ReplayMemory(MEMORY_CAPACITY)
    steps_done = 0
    episode_costs = []
    if runNumber == 0:
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    elif runNumber == 1:
        optimizer = optim.AdamW(policy_net.parameters(), lr=10e-2, amsgrad=True)
    elif runNumber == 2:
        optimizer = optim.AdamW(policy_net.parameters(), lr=20e-2, amsgrad=True)

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            cost, observation, info, currentTask, energy_tuple, time_tuple = env.step(action)
            cost = torch.tensor(cost, device=device)
            done = terminated = currentTask==n_obs

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, cost)

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
                episode_costs.append(-info)
                plot_costs()
                if runNumber == 0:
                    result1.append(info)
                    energy_list05.append(energy_tuple[0])
                    time_list05.append(time_tuple[0])
                elif runNumber == 1:
                    result2.append(info)
                    energy_list10.append(energy_tuple[0])
                    time_list10.append(time_tuple[0])
                elif runNumber == 2:
                    result3.append(info)
                    energy_list20.append(energy_tuple[0])
                    energy_list_no.append(energy_tuple[1])
                    energy_list_full.append(energy_tuple[2])
                    time_list20.append(time_tuple[0])
                    time_list_no.append(time_tuple[1])
                    time_list_full.append(time_tuple[2])
                break

print('Complete')

for epoch in range(len(result1)):
    writer.add_scalars(f"Joules-per-kbit for DDQN ",
                       {
                           f'LR=0.05': result1[epoch],
                           f'LR=0.10': result2[epoch],
                           f'LR=0.20': result3[epoch],
                       }, epoch+1)


for epoch in range(len(result1)):
    writer.add_scalars("Energy Comparison ",
                       {
                           'No Offloading': energy_list_no[epoch],
                           'DDQN LR=0.05': energy_list05[epoch],
                           'DDQN LR=0.10': energy_list10[epoch],
                           'DDQN LR=0.20': energy_list20[epoch],
                           'Full Offloading': energy_list_full[epoch],
                       }, epoch+1)

for epoch in range(len(result1)):
    writer.add_scalars("Time Comparison ",
                       {
                           'No Offloading': time_list_no[epoch],
                           'DDQN LR=0.05': time_list05[epoch],
                           'DDQN LR=0.10': time_list10[epoch],
                           'DDQN LR=0.20': time_list20[epoch],
                           'Full Offloading': time_list_full[epoch],
                       }, epoch+1)

plot_costs(show_result=True)

plt.ioff()
plt.plot(-costs_t.numpy(), label="LR = 0.05")
plt.plot(-costs_t2.numpy(), label="LR = 0.10")
plt.plot(-costs_t3.numpy(), label="LR = 0.20")
writer.flush()
writer.close()

plt.legend()
plt.show()
