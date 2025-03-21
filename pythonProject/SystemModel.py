import math
bandwidth = 10e9 #bandwidth about 50-60 GHz
carrier_frequency = 0.55e12 #At 41m, total usable Bandwidth is 0.06-0.40 THz.
gain_receiver = gain_transmitter = 20 #20 dBi find out about dB, assert
k_abs = 6.7141e-4
node_power = 29.0309 #800 mW
transmit_power =  30 #1 Watt
f_uav = 10e9 #10 GHz
f_mec = 10e9 #10 GHz
noise = -110 #-110 dBm
distance = 41 #for (x,y,z)=(sqrt(300), sqrt(300), 30) then absolute distance is approx. 41m
speed_of_light = 3e8 #m/s
k_compute = 10e-28


def node_transmit_time(data):
    pl = path_loss(data)
    cg = channel_gain(pl)
    gR = gT = db_to_linear(gain_transmitter)
    transmit_linear = db_to_linear(node_power)
    rate = gR * gT * transmit_linear * math.pow(cg, 2)
    noise_linear = db_to_linear(noise)
    rate = rate / noise_linear / bandwidth
    rate = 1 + rate
    rate = bandwidth * math.log(rate, 2)
    time = data/rate
    return time

def uplink_rate(channel_gain):
    gR = gT = db_to_linear(gain_transmitter)
    transmit_linear = db_to_linear(transmit_power)
    rate = gR*gT*transmit_linear*math.pow(channel_gain,2)
    noise_linear = db_to_linear(noise)
    rate = rate/noise_linear/bandwidth
    rate = 1 + rate
    rate = bandwidth*math.log(rate,2)
    return rate

def channel_gain(path_loss):
    gain = math.sqrt((1/path_loss))
    return gain

def path_loss(data):
    absorption_loss= 4*math.pi*carrier_frequency*distance/speed_of_light
    absorption_loss = pow(absorption_loss,2)
    spread_loss = pow(math.e, k_abs*distance)
    PL = absorption_loss*spread_loss
    return PL

def uav_transmit_time(data):
    pl = path_loss(data)
    cg = channel_gain(pl)
    rate = uplink_rate(cg)
    time = data/rate
    return time

def local_compute_energy(time): #This is equivalent to k_compute * f_uav^2 * cycles
    energy = k_compute*math.pow(f_uav,3)*time
    return energy

def uplink_energy(time):
    energy = transmit_power*time
    return energy

def db_to_linear(db:float):
    ratio = db/10
    linear = pow(10, ratio)
    return linear

def local_compute_time(cycle):
    time = (cycle)/f_uav
    return time

def offload_compute_time(cycle):
    time = (cycle)/f_mec
    return time


# for i in range(20):
#     data = (i*10 + 300)
#     cycle = (data + 600) *1e6
#     data *= 1e3
#
#     upload_time = uav_transmit_time(data)
#     full_offload_energy = uplink_energy(upload_time)
#     full_local_compute_time = local_compute_time(cycle)
#     full_local_energy = local_compute_energy(full_local_compute_time)
#     node_time = node_transmit_time(data)
#     print('Full Offloading Energy Cost = ', full_offload_energy)
#     print('Full Offloading Time Cost = ', upload_time+offload_compute_time(cycle)+node_time)
#     print('Full Local Compute Energy Cost = ', full_local_energy)
#     print('Full Local Compute Time Cost = ', full_local_compute_time+node_time)
#     print("Time Differences = ", ((upload_time+offload_compute_time(cycle)) - full_local_compute_time), '\n')

#tensorboard --logdir=C:\Users\TYH\PycharmProjects\pythonProject\.venv\runs
#tensorboard --logdir=C:\Users\iShoo.ADAN\PycharmProjects\pythonProject1\runs
