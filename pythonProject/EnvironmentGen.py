import SystemModel as sm
import gym
from SystemModel import uplink_rate, transmit_time, channel_gain
from gym import spaces
import numpy as np
from gym.envs.registration import register



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
        # return {
        #         "device_one": (self.task[0]),
        #         "device_two": (self.task[1]),
        #         "device_three": (self.task[2]),
        #         "device_four": (self.task[3])
        #     }

    #Returns the energy cost thus far
    def _get_info(self):
        return self.total_energy#, self.local_energy, self.uplink_energy)

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
            data = round(data_percentage[i] * 200 + 300)  # kbits
            cycle = data + 600  # megacycles
            self.data.append(data)
            self.cycle.append(cycle)
            self.task.append((float(data), float(cycle)))

        observation = self._get_obs()
        info = self._get_info()
        print("\nReward = ", self.total_energy)
        print("Observation = ", observation)
        print("Info = ", info)
        print("Current Task = ", self.currentTask)

        return observation, info

    def step(self, action):
        print("Action is:", action)
        offload_percentage = float(action/1000)
        local_cycle_counts = (1-offload_percentage)*(self.cycle[self.currentTask])
        offload_data_size = offload_percentage*(self.data[self.currentTask])
        self.task[self.currentTask] = (0,0)
        self.local_energy = sm.local_compute_energy(local_cycle_counts)
        path_loss = sm.path_loss()
        gain = sm.channel_gain(path_loss)
        upload_rate = sm.uplink_rate(gain)
        upload_time = sm.transmit_time(offload_data_size, upload_rate)
        self.uplink_energy = sm.uplink_energy(upload_time)
        self.total_energy -= self.local_energy-self.uplink_energy
        observation = self._get_obs()
        info = self._get_info()


        print("\nReward = ", self.total_energy)
        print("Observation = ", observation)
        print("Info = ", info)
        print("Current Task = ", self.currentTask)

        if self.currentTask<3:
            self.currentTask += 1
        else:
            self.reset()

        observation = self._get_obs()
        info = self._get_info()
        return self.total_energy, observation, info, self.currentTask