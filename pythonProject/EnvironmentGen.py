import gym
from gym import spaces
import pygame
import numpy as np


class DroneGridEnv(gym.Env):

    def __init__(self, devices=4):
        self.devices = devices #Number of UAVs in the system


        self.observation_space = spaces.Dict(
            {
                "positions":
            }
        )

        self.action_space = spaces.Box(low=0, high=1.0, dtype=np.float32)
