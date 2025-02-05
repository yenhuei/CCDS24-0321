import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from gym.envs.classic_control.EnvironmentGen import DroneEnv


# hyper-parameters
NUM_EPISODES = 500
DISCOUNT = GAMMA = 0.01
EPISILO = 0.99
LR = 0.05
UPDATE_INTERVAL = Q_NETWORK_ITERATION = 100
MEMORY_CAPACITY = 1000
BATCH_SIZE = 64

env = gym.make("gym_examples/DroneEnv-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space
NUM_STATES = env.observation_space

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
class DQN(nn.Module):
    def __init__(self, n_actions, n_observations):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128,128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)