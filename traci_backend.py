import sys, os
from shutil import which
import time

if "SUMO_HOME" in os.environ:
    sumo_home = os.environ['SUMO_HOME']
    tools_path = os.path.join(sumo_home, 'tools')
    sys.path.append(tools_path)
    import traci
else:
    raise ImportError("SUMO_HOME is not defined, try to export it first")

from utils import *
sumo_bin = 'sumo'
if not is_valid_binary(sumo_bin):
    raise ImportError("{} is not an executable binary".format(sumoCommand))

# todo
config_file = "./nets_config/single_intersection/cross3ltl.sumocfg"

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
    
    def get_vehicle_cnt_edge(self, edge_id):
        return traci.edge.getLastStepVehicleNumber(edge_id)

    def get_halt_vehicle_cnt(self, lane_id):
        return traci.lane.getLastStepHaltingNumber(lane_id)
    
    def get_halt_vehicle_cnt_edge(self, edge_id):
        return traci.edge.getLastStepHaltingNumber(edge_id)

    def get_vehicle_ids(self):
        return traci.vehicle.getIDList()

    def get_light_ids(self):
        return traci.trafficlight.getIDList();

    def get_light_definition(self, tlsID):
        return traci.trafficlight.getCompleteRedYellowGreenDefinition(tlsID)

    def get_cur_light_state(self, tlsID):
        return traci.trafficlight.getPhase(tlsID)
    
    def is_end(self):
        return traci.simulation.getMinExpectedNumber() <= 0

    # set the traffic light at <tlsID> to follow the <index> rule
    # in the current definition, there will only be two rules. rule 1: red; rule 2: green
    def set_light_phase(self, tlsID, index):
        return traci.trafficlight.setPhase(tlsID, index)
