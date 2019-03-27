from vrep_env_base import VrepEnvBase
import cooking_env.vrep as vrep
import numpy as np
from cooking_env.utils.configuration import Configuration
from future.utils import viewitems
import time
from cooking_env.utils.tf import transformations

default_config = {
    # Common to all envs
    "seed": 10,
    "state_space": None,
    "action_space": None,
    "get_state": None,
    "get_reward": None,
    "is_done": None,
    "get_info": None,
    # specific to this env
    "suffix": "",
    "arm": "right",
    "control_mode": "velocity",
}

class CookingEnv(VrepEnvBase):

    '''
    'port_num' determines which simulation this env is connected to
    'suffix' determines which robot this env is connected to
    'reset' is a class that should return a dict with new_joint_angles as an entry
    '''
    def __init__(self, config={}, port_num=19997, suffix="", reset=None, seed=None, logger=None):

        super(VrepBaxterEnv, self).__init__(config, port_num)

        self.VrepBaxterEnv_config = Configuration(default_config)
        self.VrepBaxterEnv_config.update(config)

        self.object_handles = None
        
        if seed:
            self.set_seed(seed)
        else:
            self.set_seed(self.VrepBaxterEnv_config.get('seed'))
        
        self.left_joint_handles = []
        self.right_joint_handles = []

        self.logger = logger
        
        self.suffix = suffix
        if not suffix:
            self.suffix = self.VrepBaxterEnv_config.get('suffix')

    def set_velocity_control_mode(self):
       joint_handles = self.joint_handles[self.VrepBaxterEnv_config.get('arm')]
       for joint_handle in joint_handles:
          vrep.simxSetJointForce(self.clientID, joint_handle, 10000, vrep.simx_opmode_oneshot)
          vrep.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 0, vrep.simx_opmode_oneshot)

    def set_position_control_mode(self):
       joint_handles = self.joint_handles[self.VrepBaxterEnv_config.get('arm')]
       for joint_handle in joint_handles:
          vrep.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 1, vrep.simx_opmode_oneshot)

          
    def get_state(self):
        self.update_all_info()
        if self.VrepBaxterEnv_config.get('get_state'):
            return self.VrepBaxterEnv_config.get('get_state')(self.all_info)
        else:
            return np.array([0])

    def reset(self):
       self.set_position_control_mode()
       if self.VrepBaxterEnv_reset is not None:
           new_joint_angles = self.VrepBaxterEnv_reset.reset(self.all_info)['new_joint_angles']
       else:
           new_joint_angles = [ 0.5564515308054339, -1.1094516048381255, 0.006135923151541655, 0.9990049881103757, 0.15033011721277054, 1.5780827355371194, -0.11888351106111957]
       for _ in range(5):
            for i in range(len(self.joint_handles[self.VrepBaxterEnv_config.get('arm')])):
                return_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_handles[self.VrepBaxterEnv_config.get('arm')][i], new_joint_angles[i], vrep.simx_opmode_oneshot)
            self.synchronous_trigger()
        
       if self.VrepBaxterEnv_config.get('control_mode') == "velocity":
          print("vel control mode enable")
          self.set_velocity_control_mode()
        
       return self.get_state()


    def synchronous_trigger(self):
        return_code, iteration1 = vrep.simxGetIntegerSignal(self.clientID, 'iteration', vrep.simx_opmode_buffer)
        if return_code != vrep.simx_return_ok:
            iteration1 = -1
        vrep.simxSynchronousTrigger(self.clientID)

        # vrep.simxGetPingTime(self.clientID)
        iteration2 = iteration1
        while iteration2 == iteration1: # wait until the iteration counter has changed
            return_code, iteration2 = vrep.simxGetIntegerSignal(self.clientID, 'iteration', vrep.simx_opmode_buffer)
            if return_code != vrep.simx_return_ok:
                iteration2 = -1
        self.update_all_info()
                
    def get_reward(self, state=None, action=None, next_state=None):
        if self.VrepBaxterEnv_config.get('get_reward'):
            return self.VrepBaxterEnv_config.get('get_reward')(state, action, next_state)
        else:
            return 0
            
    def is_done(self, state=None, action=None, next_state=None):
        if self.VrepBaxterEnv_config.get('is_done'):
            return self.VrepBaxterEnv_config.get('is_done')(state, action, next_state)
        else:
            return False
            # if self.all_info['ee_pose'][2] < -0.06: # under the table
            #     return True
            # else:
            #     return False

    def update_all_info(self):
        """Update all info used to extract states, currently includes
           joint angles(rad), object pose relative to end effector 
        """
            
        #### retrive joint angles ####
        joint_positions = []
        for joint_handle in self.joint_handles[self.VrepBaxterEnv_config.get('arm')]:
            return_code, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_streaming)
            joint_positions.append(joint_position)

        #### retrive end-effector pose ####
        return_code, ee_pos = vrep.simxGetObjectPosition(self.clientID, self.baxter_right_gripper_handle, self.baxter_world_frame_handle, vrep.simx_opmode_oneshot)
        return_code, ee_ori = vrep.simxGetObjectQuaternion(self.clientID, self.baxter_right_gripper_handle, self.baxter_world_frame_handle, vrep.simx_opmode_oneshot)
        
        #### retrive object position relative to end-effector ####
        if self.object_handles is not None:
            object_positions = {}
            if self.object_handles:
                for object_name, object_handle in viewitems(self.object_handles):
                    return_code, object_position = vrep.simxGetObjectPosition(self.clientID, object_handle, self.baxter_world_frame_handle, vrep.simx_opmode_streaming)
                    return_code, object_ori = vrep.simxGetObjectQuaternion(self.clientID, object_handle, self.baxter_world_frame_handle, vrep.simx_opmode_streaming)
                    object_positions[object_name] = object_position

            self.all_info['object_pos'] = object_positions
        
        #### populate all_info ####
        self.all_info['joint_angles'] = joint_positions
        self.all_info['ee_pose'] = list(ee_pos) + list(ee_ori)

        
    def step(self, actions, axis=0):
        if actions.ndim == 2:
            actions = actions[0,:]
        # clip actions to limits
        if self.VrepBaxterEnv_config.get('action_space') is not None:
            clipped_actions = np.clip(np.array(actions), self.action_space['lower_bound'], self.action_space['upper_bound'])
        else:
            clipped_actions = actions

        for i in range(len(self.joint_handles[self.VrepBaxterEnv_config.get('arm')])):
            if self.VrepBaxterEnv_config.get('control_mode') == "velocity":
                return_code = vrep.simxSetJointTargetVelocity(self.clientID, self.joint_handles[self.VrepBaxterEnv_config.get('arm')][i], clipped_actions[i], vrep.simx_opmode_oneshot)
            elif self.VrepBaxterEnv_config.get('control_mode') == "position":
                return_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_handles[self.VrepBaxterEnv_config.get('arm')][i], clipped_actions[i], vrep.simx_opmode_oneshot)
            else:
                raise ValueError("control mode not supported")

                
        self.synchronous_trigger()
        self.update_all_info()
        
        
    def get_info(self):
        return self.all_info
        
    def set_seed(self, seed):
        np.random.seed(seed)

    def stop(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)

    def pause(self):
        vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot)
    
    def close(self):
        pass

    @property
    def state_space(self):
        return self.VrepBaxterEnv_config.get('state_space')

    @property
    def action_space(self):
        return self.VrepBaxterEnv_config.get('action_space')

    def save(self, save_dir):
        pass
        
    def restore(self, restore_dir):
        pass

    def run(self):
        i = 1
        while True:
            action = np.zeros(7)
            action[0] = np.sin(0.1*i)
            i += 1
            self.step(action)
