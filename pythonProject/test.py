import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import gym
from gym.envs.classic_control.EnvironmentGen import DroneEnv, NUM_DEVICES
import numpy as np

env = gym.make("DroneEnv-v0")
env = env.unwrapped
env.reset()

while True:
    a = input("STEP\n")
    cost, observation, info, currentTask, energy_tuple, time_tuple = env.step(a)
    if currentTask==4:
        env.reset()
    print("Energy: ", energy_tuple)
    print("Time: ", time_tuple)