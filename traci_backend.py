import sys, os
from shutil import which
import time

tools_path = os.path.join('/usr/share/sumo', 'tools')
sys.path.append(tools_path)
import traci

from utils import *
sumo_bin = 'sumo'
if not is_valid_binary(sumo_bin):
    raise ImportError("{} is not an executable binary".format(sumoCommand))

# todo
config_file = "/home/yaoshihu/rl-project/code/project_code/code/mynet/config.sumocfg"

sumo_command = [sumo_bin, "-c", config_file]

class TraciBackend(object):
    CACHE = {}
    def __init__(self):
        self.phase = {}

    def start(self):
        traci.start(sumo_command)

    def close(self):
        traci.close()

    def simulate_step(self):
        traci.simulationStep()

    def get_lane_ids(self):
        return traci.lane.getIDList()

    def get_vehicle_cnt(self, lane_id):
        return traci.lane.getLastStepVehicleNumber(lane_id)

    def get_halt_vehicle_cnt(self, lane_id):
        return traci.lane.getLastStepHaltingNumber(lane_id)

    def get_vehicle_ids(self):
        return traci.vehicle.getIDList()

