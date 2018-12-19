## train in batch mode
## change the transition phase
## no target network
## no clear memory
## no update epsilon
## new conv model

from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.append('/usr/share/sumo/tools/')
from sumolib import checkBinary

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#import sys
import optparse
import subprocess
import random
import traci
import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model
from keras import backend as K
from tqdm import tqdm
import tensorflow as tf

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(12, 12, 1))
        ## first block
        conv1_1 = Conv2D(16, (3, 3), strides=(1, 1))(input_1)
        bn1_1 = BatchNormalization(axis=bn_axis, scale=False)(conv1_1)
        act1_1 = Activation('relu')(bn1_1)
        # pooling1_1 = MaxPooling2D(pool_size=2)(act1_1)
        # x1_1 = Dropout(0.1)(act1_1)
        ## second block
        conv1_2 = Conv2D(32, (2, 2), strides=(1, 1))(act1_1)
        bn1_2 = BatchNormalization(axis=bn_axis, scale=False)(conv1_2)
        act1_2 = Activation('relu')(bn1_2)
        # pooling1_2 = MaxPooling2D(pool_size=2)(act1_2)
        # x1_2 = Dropout(0.1)(act1_2)
        output1 = Flatten()(act1_2)
        
        input_2 = Input(shape=(12, 12, 1))
        ## first block
        conv2_1 = Conv2D(16, (3, 3), strides=(1, 1))(input_2)
        bn2_1 = BatchNormalization(axis=bn_axis, scale=False)(conv2_1)
        act2_1 = Activation('relu')(bn2_1)
        # pooling2_1 = MaxPooling2D(pool_size=2)(act2_1)
        # x2_1 = Dropout(0.1)(act2_1)
        ## second block
        conv2_2 = Conv2D(32, (2, 2), strides=(1, 1))(act2_1)
        bn2_2 = BatchNormalization(axis=bn_axis, scale=False)(conv2_2)
        act2_2 = Activation('relu')(bn2_2)
        # pooling2_2 = MaxPooling2D(pool_size=2)(act2_2)
        # x2_2 = Dropout(0.1)(act2_2)
        output2 = Flatten()(act2_2)
        

        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([output1, output2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        # model.compile(optimizer=keras.optimizers.RMSprop(
        #     lr=self.learning_rate), loss='categorical_crossentropy')
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        #print(act_values)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        Y = []
        pos = []
        velo = []
        lgt = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            pos.append(state[0][0])
            velo.append(state[1][0])
            lgt.append(state[2][0])
            Y.append(target_f[0])
        
        pos = np.array(pos)
        velo = np.array(velo)
        lgt = np.array(lgt)
        Y = np.array(Y)
        # print('pos', pos.shape, 'lgt', lgt.shape, 'Y', Y.shape)
        
        self.model.fit([pos, velo, lgt], Y, batch_size = batch_size, epochs=5, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SumoIntersection:
    def __init__(self):
        # we need to import python modules from the $SUMO_HOME/tools directory
        try:
            sys.path.append(os.path.join(os.path.dirname(
                __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
            sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
                os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
            from sumolib import checkBinary  # noqa
        except ImportError:
            sys.exit(
                "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

    def generate_routefile(self):
        random.seed(42)  # make tests reproducible
        N = 3600  # number of time steps
        # demand per second from different directions
        pH = 1. / 7
        pV = 1. / 11
        pAR = 1. / 30
        pAL = 1. / 25
        with open("input_routes.rou.xml", "w") as routes:
            print('''<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="SUMO_DEFAULT_TYPE" accel="0.8" decel="4.5" sigma="0" length="5" minGap="2" maxSpeed="70"/>
    <route id="always_right" edges="1fi 1si 4o 4fi 4si 2o 2fi 2si 3o 3fi 3si 1o 1fi"/>
    <route id="always_left" edges="3fi 3si 2o 2fi 2si 4o 4fi 4si 1o 1fi 1si 3o 3fi"/>
    <route id="horizontal" edges="2fi 2si 1o 1fi 1si 2o 2fi"/>
    <route id="vertical" edges="3fi 3si 4o 4fi 4si 3o 3fi"/>

    ''', file=routes)
            lastVeh = 0
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < pH:
                    print('    <vehicle id="right_%i" type="SUMO_DEFAULT_TYPE" route="horizontal" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pV:
                    print('    <vehicle id="left_%i" type="SUMO_DEFAULT_TYPE" route="vertical" depart="%i" />' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pAL:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_left" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
                if random.uniform(0, 1) < pAR:
                    print('    <vehicle id="down_%i" type="SUMO_DEFAULT_TYPE" route="always_right" depart="%i" color="1,0,0"/>' % (
                        vehNr, i), file=routes)
                    vehNr += 1
                    lastVeh = i
            print("</routes>", file=routes)

    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def getState(self):
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        junctionPosition = traci.junction.getPosition('0')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('1si')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('2si')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('3si')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('4si')
        for i in range(12):
            positionMatrix.append([])
            velocityMatrix.append([])
            for j in range(12):
                positionMatrix[i].append(0)
                velocityMatrix[i].append(0)

        for v in vehicles_road1:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[2 - traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[3 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('0')[1]
        for v in vehicles_road3:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 12):
                positionMatrix[6 + 2 -
                               traci.vehicle.getLaneIndex(v)][11 - ind] = 1
                velocityMatrix[6 + 2 - traci.vehicle.getLaneIndex(
                    v)][11 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road4:
            ind = int(
                abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
            if(ind < 12):
                positionMatrix[9 + traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[9 + traci.vehicle.getLaneIndex(
                    v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        light = []
        if(traci.trafficlight.getPhase('0') == 4):
            light = [1, 0]
        else:
            light = [0, 1]

        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 12, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 12, 1)

        lgts = np.array(light)
        lgts = lgts.reshape(1, 2, 1)

        return [position, velocity, lgts]
    
def restrict_reward(reward,func="unstrict"):
    if func == "linear":
        bound = -100
        reward = 0 if reward < bound else (reward/(-bound) + 1)
    elif func == "neg_log":
        reward = math.log(-reward+1)
    else:
        pass

    return reward


if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    options = sumoInt.get_options()

    #if options.nogui:
    #if True:
    sumoBinary = checkBinary('sumo')
    #else:
    #    sumoBinary = checkBinary('sumo-gui')
    # sumoInt.generate_routefile()
    logfile = 'traffic_light_control_version2_5.txt'
    weightfile = 'traffic_light_control_version2_5/traffic_light_control_version0_'

    # Main logic
    # parameters
    episodes = 100
    batch_size = 32

    tg = 10
    ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/reinf_traf_control1.h5')
    except:
        print('No models found')

    for e in range(episodes):
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        log = open(logfile, 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        
        #reward3 related to fuel consumption
        reward3 = 0
        total_reward = reward1 - reward2
        stepz = 0
        action = 0
        
        # overall fuel consumption
        fuel_consumptions = 0
        # overall vehicles
        num_vehicles = 0

        traci.start([sumoBinary, "-c", "cross3ltl.sumocfg", '--start'])
        traci.trafficlight.setPhase("0", 0)
        traci.trafficlight.setPhaseDuration("0", 200)
        
        
        total_time = 7000
        pbar = tqdm(total = total_time)
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < total_time:
         
            traci.simulationStep()
            stepz += 1
            # print('current_time', traci.simulation.getCurrentTime() / 1000)
            state = sumoInt.getState()
            action= agent.act(state)
            light = state[2]

            if(action == 0 and light[0][0][0] == 0):
                # action = turn on the horizontal green light, current horizontal is red, vertical is green
                # Transition Phase
                for i in range(3):
                    
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 1)
                    traci.simulationStep()
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)
                for i in range(5):
                    
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 2)
                    traci.simulationStep()
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)
                for i in range(3):
                    
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 3)
                    traci.simulationStep()
                    
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

                # Action Execution
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 4)
                    traci.simulationStep()
                    
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

            if(action == 0 and light[0][0][0] == 1):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '1si') + traci.edge.getLastStepVehicleNumber('2si')
                reward2 = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 4)
                    traci.simulationStep()
                    
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '1si') + traci.edge.getLastStepVehicleNumber('2si')
                    reward2 += traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

            if(action == 1 and light[0][0][0] == 0):
                # Action Execution, no state change
                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 0)
                    traci.simulationStep()
                    
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                                       
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

            if(action == 1 and light[0][0][0] == 1):
                for i in range(3):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 5)
                    traci.simulationStep()
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)
                for i in range(5):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 6)
                    traci.simulationStep()
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)
                for i in range(3):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 7)
                    traci.simulationStep()
                    
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

                reward1 = traci.edge.getLastStepVehicleNumber(
                    '4si') + traci.edge.getLastStepVehicleNumber('3si')
                reward2 = traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                for i in range(10):
                    stepz += 1
                    pbar.update(1)
                    
                    traci.trafficlight.setPhase('0', 0)
                    traci.simulationStep()
                    
                    reward1 += traci.edge.getLastStepVehicleNumber(
                        '4si') + traci.edge.getLastStepVehicleNumber('3si')
                    reward2 += traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber('2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si')
                    waiting_time += (traci.edge.getLastStepHaltingNumber('1si') + traci.edge.getLastStepHaltingNumber(
                        '2si') + traci.edge.getLastStepHaltingNumber('3si') + traci.edge.getLastStepHaltingNumber('4si'))
                    
                    # print('current_time', traci.simulation.getCurrentTime() / 1000)

            new_state = sumoInt.getState()
            
            reward = reward1 - reward2
            # print('reward', reward)
            agent.remember(state, action, reward, new_state, False)
            # print('len agent.memory',len(agent.memory))
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
            if(len(agent.memory) > batch_size):
                agent.replay(batch_size)
                
        pbar.close()
        
        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        log.write('episode - ' + str(e) + ', total waiting time - ' +
                 str(waiting_time) + ', static waiting time - 338798 \n')
        log.close()
        print('episode - ' + str(e) + ' total waiting time - ' + str(waiting_time))
        agent.save(weightfile + str(e) + '.h5')
        traci.close(wait=False)

sys.stdout.flush()
