import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from gym.envs.classic_control.EnvironmentGen import DroneEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("DroneEnv-v0")
env = env.unwrapped
env.reset()
currentTask = 0

while(1):
    userInput = input("Input Step Amount 0-999 ")
    reward, observation, info, currentTask =  env.step(int(userInput))
    if currentTask == 1:
        env.reset()

    print("Reward = ", reward)
    # print("Action = ", action)
    print("Observation = ", observation)
    print("Info = ", info)
    print("Current Task = ", currentTask)