import math

bandwidth = 0.8e12#0.8=4x0.2 or 0.55 THz
gain_receiver = gain_transmitter = 20 #20 dBi find out about dB, assert
k_abs = 6.7141e-4
transmit_power =  5 #500mW in Watts in dBm for transmit power range [0,10] db.
f_uav = 5e9 #5 GHz
f_mec = 8e9 #8 GHz
noise = -110 #Unknown also in dBm
distance = 52 #20m - 30m height, for (x,y,z)=(30,30,30) then absolute distance is approx. 51.962
speed_of_light = 3e8 #m/s
k_compute = 10e-26


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
    absorption_loss= 4*math.pi*bandwidth*distance/speed_of_light
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
#
# print(full_offload_energy)
#
# print(math.sqrt(pow(30,2)*3))