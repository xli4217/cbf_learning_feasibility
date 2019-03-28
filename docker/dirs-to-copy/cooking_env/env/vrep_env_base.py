from cooking_env.utils.configuration import Configuration

import time

try:
    import cooking_env.vrep as vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')

default_config = {}
    
class VrepEnvBase(object):
    # Defines an rllab environment for the V-Rep scene ur5_sim_test.ttt
    def __init__(self, config={}, port_num=19997):
        self.VrepEnvBase_config = Configuration(default_config)
        self.VrepEnvBase_config.update(config)

        print ('Program started')
        vrep.simxFinish(-1)  # just in case, close all opened connections
        self.clientID = vrep.simxStart('127.0.0.1', port_num, True,
                                       True, 5000, 5)  # Connect to V-REP
        if self.clientID != -1:
            print ('Connected to remote API server')
            vrep.simxStartSimulation(
                self.clientID, vrep.simx_opmode_oneshot_wait)
            vrep.simxSynchronous(self.clientID, True)
            # Now try to retrieve data in a blocking fashion (i.e. a service
            # call):
            res, objs = vrep.simxGetObjects(
                self.clientID, vrep.sim_handle_all, vrep.simx_opmode_blocking)
            if res == vrep.simx_return_ok:
                print ('Number of objects in the scene: ', len(objs))
            else:
                print ('Remote API function call returned with error code: ', res)
            print("connected through port number: {}".format(port_num))
                
            # used to connect multiple clients in synchronous mode http://www.coppeliarobotics.com/helpFiles/en/remoteApiModusOperandi.htm
            return_code, iteration = vrep.simxGetIntegerSignal(self.clientID, "iteration", vrep.simx_opmode_streaming)
     
            time.sleep(2)



        