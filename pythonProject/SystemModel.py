import math

bandwidth = 50e9 #bandwidth about 50-60 GHz
carrier_frequency = 0.55e12 #At 41m, total usable Bandwidth is 0.06-0.40 THz.
gain_receiver = gain_transmitter = 20 #20 dBi find out about dB, assert
k_abs = 6.7141e-4
transmit_power =  30 #1 Watt
f_uav = 10e9 #10 GHz
f_mec = 10e9 #10 GHz
noise = -110 #-110 dBm
distance = 41 #for (x,y,z)=(sqrt(300), sqrt(300), 30) then absolute distance is approx. 41m
speed_of_light = 3e8 #m/s
k_compute = 10e-28


def uplink_rate(channel_gain):
    gR = gT = db_to_linear(gain_transmitter)
    transmit_linear = db_to_linear(transmit_power)
    rate = gR*gT*transmit_linear*math.pow(channel_gain,2)
    noise_linear = db_to_linear(noise)
    rate = rate/noise_linear/bandwidth
    rate = 1 + rate
    rate = bandwidth*math.log(rate,2)

    # print("Uplink Rate = ",rate)
    return rate
def transmit_time(data,rate):
    assert rate != 0
    time = data/rate

    # print("Tansmit Time = ", time)
    return time

def channel_gain(path_loss):
    gain = math.sqrt((1/path_loss))

    # print("Channel Gain = ", gain)
    return gain

def path_loss(data):
    absorption_loss= 4*math.pi*carrier_frequency*distance/speed_of_light
    absorption_loss = pow(absorption_loss,2)
    spread_loss = pow(math.e, k_abs*distance)
    PL = absorption_loss*spread_loss
    # print("Pathloss = ", PL)
    return PL

def local_compute_energy(cycle):
    energy = k_compute*math.pow(f_uav,2)*cycle
    # print("Local Energy = ", energy)
    return energy

def uplink_energy(time):
    energy = transmit_power*time

    # print("Uplink Energy = ", energy)
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

# path_loss = path_loss(3e6)
# gain = channel_gain(path_loss)
# upload_rate = uplink_rate(gain)
# upload_time = transmit_time(3e6, upload_rate)
# full_offload_energy = uplink_energy(upload_time)
# full_local_energy = local_compute_energy(3e9)
#
#
#
# print('Full Offloading Energy Cost = ', full_offload_energy)
# print('Full Offloading Time Cost = ', upload_time+offload_compute_time(3e9), '\n')
# print('Full Local Compute Energy Cost = ', full_local_energy)
# print('Full Local Compute Time Cost = ', local_compute_time(3e9))

#tensorboard --logdir=C:\Users\TYH\PycharmProjects\pythonProject\.venv\runs
#tensorboard --logdir=C:\Users\iShoo.ADAN\PycharmProjects\pythonProject1\runs
