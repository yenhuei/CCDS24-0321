import SystemModel as sm
import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register
import math

NUM_ACTIONS = 100
NUM_DEVICES = 4


class DroneEnv(gym.Env):

    def __init__(self, devices=NUM_DEVICES):
        self.devices = devices  # Number of UAVs in the system
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
        self.no_offload_total_time = 0
        self.full_offload_total_time = 0
        self.no_offload_total_energy = 0
        self.full_offload_total_energy = 0
        self.real_total_energy = [0,0,0]

        # One system with four separate device/tasks with unique data size and cycle counts
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(NUM_ACTIONS)  # 0.0 to 99.9

    # Returns the amount of data size and cycle counts for each device's task
    def _get_obs(self):
        return self.data

    # Returns the energy cost thus far
    def _get_info(self):
        if (self.total_data != 0):
            return self.total_energy/self.total_data*1e3
        else:
            return 0

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
        self.no_offload_total_time = 0
        self.full_offload_total_time = 0
        self.no_offload_total_energy = 0
        self.full_offload_total_energy = 0
        self.real_total_energy = [0,0,0]

        # Reset task data and cycle
        data_percentage = (np.random.rand(1, NUM_DEVICES))[0]
        self.data = list()
        self.cycle = list()
        self.task = list()
        for i in range(self.devices):
            data = round(data_percentage[i] * 200 + 300)  # kbits
            cycle = (data + 600) * 1e6  # megacycles
            data = data * 1e3
            self.data.append(data)
            self.cycle.append(cycle)
            self.task.append((data, cycle))

        observation = self._get_obs()
        info = self._get_info()
        # print("\nReward = ", self.total_energy)
        # print("Observation = ", observation)
        # print("Info = ", info)
        # print("Current Task = ", self.currentTask)

        return observation, info

    def step(self, action):
        data = self.data[self.currentTask]
        cycle = self.cycle[self.currentTask]
        self.data[self.currentTask] = 0
        self.cycle[self.currentTask] = 0

        a = round(float(action) / NUM_ACTIONS, 2)
        local_data = a * data
        local_cycle = a * cycle
        offload_data = (1 - a) * data
        offload_cycle = (1 - a) * cycle

        # Step 1: Upload from node to UAV
        node_upload_time = sm.node_transmit_time(data)
        # Step 2: Process local data and then offload remaining from UAV to GBS
        uav_compute_time = sm.local_compute_time(local_cycle)
        uav_upload_time = sm.uav_transmit_time(offload_data)
        uav_compute_energy = sm.local_compute_energy(uav_compute_time)
        uav_upload_energy = sm.uplink_energy(uav_upload_time)
        # Step 3: Process offloaded data on GBS
        gbs_compute_time = sm.offload_compute_time(offload_cycle)

        # Energy cost and time delay for taking action 'a'
        total_energy = round(uav_compute_energy + uav_upload_energy, 3)
        total_time = round(node_upload_time + max(uav_compute_time, (uav_upload_time + gbs_compute_time)) , 3)

        # Energy cost and time delay for NO offloading
        no_offloading_compute_time = sm.local_compute_time(cycle)
        no_offloading_time = round(node_upload_time + no_offloading_compute_time ,3)
        no_offloading_energy = round(sm.local_compute_energy(no_offloading_compute_time), 3)

        # Energy cost and time delay for FULL offloading
        full_offloading_upload_time = sm.uav_transmit_time(data)
        full_offloading_compute_time = sm.offload_compute_time(cycle)
        full_offloading_time = round(node_upload_time + full_offloading_upload_time + full_offloading_compute_time, 3)
        full_offloading_energy = round(sm.uplink_energy(full_offloading_upload_time), 3)

        # Statistics Tracking
        self.currentTask += 1
        self.real_total_energy[0] += total_energy
        self.real_total_energy[1] += no_offloading_energy
        self.real_total_energy[2] += full_offloading_energy
        real_energy = tuple(self.real_total_energy)
        self.total_time += total_time
        self.no_offload_total_time += no_offloading_time
        self.full_offload_total_time += full_offloading_time
        self.total_data += data
        #energy_tuple = (self.total_energy+total_energy, self.no_offload_total_energy, self.full_offload_total_energy)
        time_tuple = (self.total_time, self.no_offload_total_time, self.full_offload_total_time)
        observation = self._get_obs()


        #Penalties for violation
        MAX_task_delay = min(no_offloading_time, full_offloading_time) - 0.015

        if total_time > MAX_task_delay:
            total_energy = data/1e3/4*(total_time/MAX_task_delay)
        if no_offloading_time > MAX_task_delay:
            no_offloading_energy = data/1e3/4*(no_offloading_time/MAX_task_delay)
        if full_offloading_time > MAX_task_delay:
            full_offloading_energy = data/1e3/4*(full_offloading_time/MAX_task_delay)

        self.total_energy += total_energy
        self.no_offload_total_energy += no_offloading_energy
        self.full_offload_total_energy += full_offloading_energy

        energy_tuple = (self.total_energy, self.no_offload_total_energy, self.full_offload_total_energy)
        total_energy_per_kbits = total_energy/data*1e3
        info = self._get_info()

        return [-total_energy_per_kbits], observation, info, self.currentTask, energy_tuple, time_tuple, real_energy
