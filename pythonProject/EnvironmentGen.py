import SystemModel as sm
import gym
from SystemModel import uplink_rate, transmit_time, channel_gain
from gym import spaces
import numpy as np
from gym.envs.registration import register
import math

NUM_ACTIONS = 100
NUM_DEVICES = 10
class DroneEnv(gym.Env):

    def __init__(self, devices=NUM_DEVICES):
        self.devices = devices #Number of UAVs in the system
        self.data = list()
        self.cycle = list()
        self.task = list()
        self.total_energy = 0
        self.local_energy = 0
        self.total_time = 0
        self.uplink_energy = 0
        self.currentTask = 0
        self.total_data = 0
        self.power = 0

        # One system with four separate device/tasks with unique data size and cycle counts
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(NUM_ACTIONS) #0.0 to 99.9

    #Returns the amount of data size and cycle counts for each device's task
    def _get_obs(self):
        return self.data

    #Returns the energy cost thus far
    def _get_info(self):
        return self.total_energy

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.currentTask = 0
        self.total_energy = 0
        self.local_energy = 0
        self.uplink_energy = 0
        self.total_data = 0
        self.power = 0
        self.total_time = 0

        #Reset task data and cycle
        data_percentage = (np.random.rand(1, NUM_DEVICES))[0]
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
        # print("\nReward = ", self.total_energy)
        # print("Observation = ", observation)
        # print("Info = ", info)
        # print("Current Task = ", self.currentTask)

        return observation, info

    def step(self, action):
        offload_percentage = round(float(action/NUM_ACTIONS),3)
        local_cycle_counts = (1-offload_percentage)*(self.cycle[self.currentTask])
        offload_data_size = offload_percentage*(self.data[self.currentTask])
        self.total_data += self.data[self.currentTask]
        self.local_energy = sm.local_compute_energy(local_cycle_counts)
        path_loss = sm.path_loss(offload_data_size)
        gain = sm.channel_gain(path_loss)
        upload_rate = sm.uplink_rate(gain)
        upload_time = sm.transmit_time(offload_data_size, upload_rate)
        self.uplink_energy = sm.uplink_energy(upload_time)
        self.total_energy += (self.local_energy+self.uplink_energy)

        # Finding the max time taken to compute locally and to offload and compute
        local_compute_time = sm.local_compute_time(local_cycle_counts)
        # print('Local time = ' ,local_compute_time)
        offload_compute_time = sm.offload_compute_time(offload_data_size * 10e3)
        # print('Offload time = ', offload_compute_time)
        total_time = max(local_compute_time, offload_compute_time + upload_time)
        self.total_time += total_time

        # Finding the minimum energy between full offloading and no offloading
        full_local_energy = sm.local_compute_energy(self.cycle[self.currentTask])
        path_loss = sm.path_loss(self.data[self.currentTask])
        gain = sm.channel_gain(path_loss)
        upload_rate = sm.uplink_rate(gain)
        upload_time = sm.transmit_time(self.data[self.currentTask], upload_rate)
        full_offload_energy = sm.uplink_energy(upload_time)
        min_binary_energy = min(full_local_energy, full_offload_energy)

        # Finding the minimum time between full offloading and no offloading
        full_local_time = sm.local_compute_time(self.data[self.currentTask] * 10e3)
        full_offload_time = sm.offload_compute_time(self.data[self.currentTask] * 10e3) + transmit_time(
            self.data[self.currentTask], upload_rate)
        min_binary_offloading_time = min(full_local_time, full_offload_time)

        energy_difference = self.total_energy - min_binary_energy
        time_difference = total_time - min_binary_offloading_time
        time_ratio = total_time / min_binary_offloading_time

        self.data[self.currentTask] = 0
        observation = self._get_obs()
        info = self._get_info()
        self.currentTask+=1

        # print("\nEnergy = ", self.total_energy)
        # print("Percentage Offloaded = ", offload_percentage)
        # print("Energy Used Local= ", total_time)
        # print("Energy Used Off= ", min_binary_offloading_time)
        # print("Time Ratio = ", time_ratio)
        # print("Observation = ", observation)
        # print("Info = ", info)

        # print("Energy/Bit =  " , round((-info/self.total_data*10e3)/self.total_time, 5)) #Power/kBits
        # return [round((-info/self.total_data*10e3)/self.total_time, 5)], observation, info, self.currentTask, self.data[self.currentTask-1]

        # print("Energy/kBits =  ", (-info / self.total_data * 10e3))
        return [(-info / self.total_data * 10e3)], observation, info, self.currentTask,self.data[self.currentTask - 1]