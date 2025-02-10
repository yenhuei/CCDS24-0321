import SystemModel as sm
import gym
from SystemModel import uplink_rate, transmit_time, channel_gain
from gym import spaces
import numpy as np
from gym.envs.registration import register
import math


class DroneEnv(gym.Env):

    def __init__(self, devices=4):
        self.devices = devices #Number of UAVs in the system
        self.data = list()
        self.cycle = list()
        self.task = list()
        self.total_energy = 0
        self.local_energy = 0
        self.uplink_energy = 0
        self.currentTask = 0

        # One system with four separate device/tasks with unique data size and cycle counts
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1000) #0.0 to 99.9

    #Returns the amount of data size and cycle counts for each device's task
    def _get_obs(self):
        return [self.task[self.currentTask][0]]

    #Returns the energy cost thus far
    def _get_info(self):
        return self.total_energy#=, self.local_energy, self.uplink_energy)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.currentTask = 0
        self.total_energy = 0
        self.local_energy = 0
        self.uplink_energy = 0

        #Reset task data and cycle
        data_percentage = (np.random.rand(1, 4))[0]
        self.data = list()
        self.cycle = list()
        self.task = list()
        for i in range(self.devices):
            data = round(data_percentage[i] * 200 + 300)*10e3 # kbits
            cycle = (data)*10e3  # megacycles
            self.data.append(data)
            self.cycle.append(cycle)
            self.task.append((data,cycle))

        observation = self._get_obs()
        info = self._get_info()
        print("\nReward = ", self.total_energy)
        print("Observation = ", observation)
        print("Info = ", info)
        print("Current Task = ", self.currentTask)

        return observation, info

    def step(self, action):
        offload_percentage = float(action/1000)
        local_cycle_counts = (1-offload_percentage)*(self.cycle[self.currentTask])
        offload_data_size = offload_percentage*(self.data[self.currentTask])
        self.task[self.currentTask] = (0,0)
        self.local_energy = sm.local_compute_energy(local_cycle_counts)
        path_loss = sm.path_loss(offload_data_size)
        gain = sm.channel_gain(path_loss)
        upload_rate = sm.uplink_rate(gain)
        upload_time = sm.transmit_time(offload_data_size, upload_rate)
        self.uplink_energy = sm.uplink_energy(upload_time)
        self.total_energy += (self.local_energy+self.uplink_energy)

        local_compute_time = sm.local_compute_time(local_cycle_counts)
        offload_compute_time = sm.offload_compute_time(offload_data_size*10e3)
        total_time = max(local_compute_time , offload_compute_time + upload_time)

        full_local_time = sm.local_compute_time(self.data[self.currentTask]*10e3)
        full_offload_time = sm.offload_compute_time(self.data[self.currentTask]*10e3) + transmit_time(self.data[self.currentTask], upload_rate)
        min_binary_offloading_time = min(full_local_time, full_offload_time)

        print("Total Time = ", total_time)
        print("Minimum Binary Offload Time = ", min_binary_offloading_time)
        if total_time >= min_binary_offloading_time:
            self.total_energy *= 10
            self.currentTask=3

        if self.currentTask<3:
            self.currentTask += 1
            observation = self._get_obs()
            info = self._get_info()
        else:
            observation = self._get_obs()
            info = self._get_info()
            self.currentTask+=1

        print("\nReward = ", self.total_energy)
        # print("Action = ", action)
        # print("Observation = ", observation)
        # print("Info = ", info)
        # print("Current Task = ", self.currentTask)

        # return self.total_energy, observation, info, self.currentTask
        return [-info], observation, info, self.currentTask