import SystemModel as sm
import gym
from SystemModel import uplink_rate, transmit_time, channel_gain
from gym import spaces
import numpy as np
from gym.envs.registration import register



class DroneEnv(gym.Env):
    register(
        id='gym_examples/DroneEnv',
        entry_point='gym_examples.envs:DroneEnv',
        max_episode_steps=100,
    )
    def __init__(self, devices=4):
        self.devices = devices #Number of UAVs in the system
        data_percentage = (np.random.rand(1,4))[0]
        cycle_percentage = (np.random.rand(1,4))[0]

        self.data = list()
        self.cycle = list()
        self.task = list()
        for i in range(devices):
            data = round(data_percentage[i]*200000+300000)
            self.data.append(data)
            cycle = round(cycle_percentage[i]*200000000+900000000)
            self.cycle.append(cycle)
            self.task.append((data,cycle))

        # One system with four separate tasks with unique data size and cycle counts
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1000) #0.0 to 99.9

    #Returns the amount of data size and cycle counts for each device's task
    def _get_obs(self):
        return {
                "device_one": (self.task[0]),
                "device_two": (self.task[1]),
                "device_three": (self.task[2]),
                "device_four": (self.task[3])
            }

    #Returns the energy cost thus far
    # def _get_info(self):
    #     return (self.total_energy, self.local_energy, self.uplink_energy)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        #Reset task data and cycle
        self.data = list()
        self.cycle = list()
        self.task = list()
        data_percentage = (np.random.rand(1, 4))[0]
        cycle_percentage = (np.random.rand(1, 4))[0]
        for i in range(self.devices):
            data = round(data_percentage[i] * 200000 + 300000)
            self.data.append(data)
            cycle = round(cycle_percentage[i] * 200000000 + 900000000)
            self.cycle.append(cycle)
            self.task.append((data, cycle))

        observation = self._get_obs()
        # info = self._get_info()

        return observation#, info

    def step(self, action):
        offload_percentage = float(action/100)
        local_cycle_counts = (1-offload_percentage)*(sum(self.cycle))
        offload_data_size = offload_percentage*(sum(self.data))
        # An episode is done iff the agent has reached the target
        #f_uav = 3.2 GHz
        local_energy = sm.local_computing_energy(3.2, local_cycle_counts)
        gain = channel_gain(1.2, 20, offload_data_size)
        upload_rate = uplink_rate(gain)
        upload_time = transmit_time(offload_data_size, upload_rate)
        uplink_energy = sm.uplink_energy(upload_time)
        reward = -local_energy-uplink_energy
        observation = self._get_obs()
        # info = self._get_info()

        return reward