import math

bandwidth = 0.55 #0.55 THz
gain_receiver = gain_transmitter = 20 #20 dBi
k_abs = 6.7141 / 10000
transmit_power =  500 #500mW
f_uav = 5 #5 GHz
noise = 1 #Unknown
distance = 20 #20m
speed_of_light = 299792458 #m/s
k_compute = math.pow(10,-25)


def uplink_rate(channel_gain):
    rate = gain_receiver*gain_transmitter*transmit_power*math.pow(channel_gain,2)
    rate = rate/noise/bandwidth
    rate = 1 + rate
    rate = bandwidth*math.log(rate,2)

    return rate
def transmit_time(data,rate):
    time = data/rate0
    return time

def channel_gain(path_loss):
    gain = math.sqrt((1/path_loss))
    return gain

def path_loss(data):
    PL = 4*math.pi*bandwidth*distance/speed_of_light
    PL = pow(PL,2)
    PL = PL*pow(math.e, k_abs*bandwidth*data)

    return PL

def local_compute_energy(cycle):
    energy = k_compute*math.pow(f_uav,2)*cycle
    return energy

def uplink_energy(time):
    energy = transmit_power*time
    return energy