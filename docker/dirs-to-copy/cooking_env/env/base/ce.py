from vrep_env_base import VrepEnvBase
import cooking_env.vrep as vrep
import numpy as np
from cooking_env.utils.configuration import Configuration
from future.utils import viewitems
import time
from cooking_env.utils import transformations
from env_info import robot_handles, object_handles

default_config = {
    # Common to all envs
    "seed": 10,
    # specific to this env
    "suffix": "",
    "particle_test": False,
    "arm": "ur5",
    "control_mode": "velocity",
}

class CookingEnv(VrepEnvBase):

    '''
    'port_num' determines which simulation this env is connected to
    'suffix' determines which robot this env is connected to
    'reset' is a class that should return a dict with new_joint_angles as an entry
    '''
    def __init__(self, config={}, port_num=19997, suffix="", reset=None, seed=None, logger=None):

        super(CookingEnv, self).__init__(config, port_num)

        self.CookingEnv_config = Configuration(default_config)
        self.CookingEnv_config.update(config)

        self.CookingEnv_reset = reset
        self.object_handles = None
        
        self.logger = logger
        
        self.suffix = suffix
        if not suffix:
            self.suffix = self.CookingEnv_config.get('suffix')

        self.all_info = {}
        ####
        if self.CookingEnv_config.get('arm') == 'ur5':
            rh = robot_handles['UR5']
            self.init_angles = [-2.69, -61.47, -3.35, -23.27, 89.80, -2.68]
        elif self.CookingEnv_config.get('arm') == 'jaco':
            rh = robot_handles['Jaco']
            self.init_angles = [-2.69, -61.47, -3.35, -23.27, 89.80, -2.68]
        else:
            raise ValueError('arm not supported')

        #### joint handles ####
        self.joint_handles = []
        for jh in rh['joint_handles']:
            rc, h = vrep.simxGetObjectHandle(self.clientID, jh, vrep.simx_opmode_oneshot_wait) 
            self.joint_handles.append(h)
            
        #### gripper handles ####
        self.gripper_toggle_signal_name = rh['gripper_handles']['toggle_handle']
        
        _, self.gripper_attachpoint_handle = vrep.simxGetObjectHandle(self.clientID,
                                                                      rh['gripper_handles']['attachpoint_handle'],
                                                                      vrep.simx_opmode_oneshot_wait)
       
        rc, self.gripper_prox_sensor_handle = vrep.simxGetObjectHandle(self.clientID,
                                                                       rh['gripper_handles']['prox_sensor_handle'],
                                                                       vrep.simx_opmode_oneshot_wait)
        
        # rc, ds, dp, dh, dn = vrep.simxReadProximitySensor(self.clientID, self.gripper_prox_sensor_handle, vrep.simx_opmode_streaming)
        # while rc != 0:
        #     rc, ds, dp, dh, dn = vrep.simxReadProximitySensor(self.clientID, self.gripper_prox_sensor_handle, vrep.simx_opmode_buffer)
            
            
        
        #### target handle ####
        _, self.target_handle = vrep.simxGetObjectHandle(self.clientID,
                                                         rh['goal_handle'],
                                                         vrep.simx_opmode_oneshot_wait)

        
        #### object handles ####
        self.object_handles = []
        for oh in object_handles:
            _, h = vrep.simxGetObjectHandle(self.clientID, oh['handle'], vrep.simx_opmode_oneshot_wait)
            self.object_handles.append(dict(name=oh['name'], handle=h))
        

        #### test particle ####
        self.test_particle_handle = None
        if self.CookingEnv_config.get('particle_test'):
            _, self.test_particle_handle = vrep.simxGetObjectHandle(self.clientID, rh['particle_handle'], vrep.simx_opmode_oneshot_wait)
        
        
            
    # def set_velocity_control_mode(self):
    #    joint_handles = self.joint_handles[self.CookingEnv_config.get('arm')]
    #    for joint_handle in joint_handles:
    #       vrep.simxSetJointForce(self.clientID, joint_handle, 10000, vrep.simx_opmode_oneshot)
    #       vrep.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 0, vrep.simx_opmode_oneshot)

    def set_position_control_mode(self):
       for jh in self.joint_handles:
          vrep.simxSetObjectIntParameter(self.clientID, jh, 2001, 1, vrep.simx_opmode_oneshot)
          self.synchronous_trigger()

    def robot_open_gripper(self):
        vrep.simxSetIntegerSignal(self.clientID, self.gripper_toggle_signal_name, 0, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()
        
    def robot_close_gripper(self):
        # check which dynamically non-static and respondable object is in-between the fingers. Then attach the object to the gripper
        # for h in self.object_handles:
        #     if h['handle'] == -1:
        #         break
        #     _, static = vrep.simxGetObjectIntParameter(self.clientID, h['handle'], vrep.sim_shapeintparam_static, vrep.simx_opmode_blocking)
        #     _, respondable = vrep.simxGetObjectIntParameter(self.clientID, h['handle'], vrep.sim_shapeintparam_respondable, vrep.simx_opmode_blocking)            
            # if not static and respondable and ds == 1:
            #     break

        # close gripper
        vrep.simxSetIntegerSignal(self.clientID, self.gripper_toggle_signal_name, 1, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()
        
    def set_target_position(self, pos):
        assert pos.shape == (3,)
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle
        vrep.simxSetObjectPosition(self.clientID, handle, -1, pos, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()
        
    def set_target_quaternion(self, quat):
        assert quat.shape == (4,)
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle
      
        vrep.simxSetObjectQuaternion(self.clientID, handle, -1, quat, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()
        
    def set_target_euler_angles(self, rpy):
        assert rpy.shape == (3,)

        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle
      
        vrep.simxSetObjectOrientation(self.clientID, handle, -1, rpy, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()

    def get_target_pose(self):
        self.update_all_info()

        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle

        return self.all_info['target_pose']

    def get_target_velocity(self):
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle

        return_code, linear_vel, angular_vel = vrep.simxGetObjectVelocity(self.clientID, handle, vrep.simx_opmode_blocking)

        return np.array(linear_vel), np.array(angular_vel)
        
        
    def reset(self):
        #### clear signals ####
        vrep.simxClearIntegerSignal(self.clientID, "", vrep.simx_opmode_oneshot)
        vrep.simxClearFloatSignal(self.clientID, "", vrep.simx_opmode_oneshot)
        vrep.simxClearStringSignal(self.clientID, "", vrep.simx_opmode_oneshot)

        #### reset robot and environment ####
        self.set_position_control_mode()
        if self.CookingEnv_reset is not None:
            new_joint_angles = self.CookingEnv_reset.reset(self.all_info)['new_joint_angles']
        else:
            new_joint_angles = self.init_angles
        for _ in range(5):
            for i in range(6):
                return_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_handles[i], new_joint_angles[i], vrep.simx_opmode_oneshot)
            self.synchronous_trigger()
        self.update_all_info()
            
       # if self.CookingEnv_config.get('control_mode') == "velocity":
       #    print("vel control mode enable")
       #    self.set_velocity_control_mode()
        
       # return self.get_state()


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
                

    def update_all_info(self):
        """Update all info used to extract states, currently includes
           joint angles(rad), object pose relative to end effector 
        """

        # TODO: Need to get pose in the right frame after calibration with hardware !!!!
        #### retrive joint angles ####
        joint_positions = []
        for joint_handle in self.joint_handles:
            return_code, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_streaming)
            joint_positions.append(joint_position)
        self.all_info['joint_angles'] = np.array(joint_positions)
            
        #### retrive end-effector pose ####
        # return_code, ee_pos = vrep.simxGetObjectPosition(self.clientID, self.baxter_right_gripper_handle, self.baxter_world_frame_handle, vrep.simx_opmode_oneshot)
        # return_code, ee_ori = vrep.simxGetObjectQuaternion(self.clientID, self.baxter_right_gripper_handle, self.baxter_world_frame_handle, vrep.simx_opmode_oneshot)

        #### target pose ####
        _, target_pos = vrep.simxGetObjectPosition(self.clientID, self.target_handle, -1, vrep.simx_opmode_blocking)
        _, target_quat = vrep.simxGetObjectQuaternion(self.clientID, self.target_handle, -1, vrep.simx_opmode_blocking)
        self.all_info['target_pose'] = np.array(target_pos + target_quat)
        
        #### retrive object position relative to end-effector ####
        if self.object_handles is not None:
            object_pose = {}
            for obj in self.object_handles:
                return_code, object_position = vrep.simxGetObjectPosition(self.clientID, obj['handle'], -1, vrep.simx_opmode_streaming)
                return_code, object_quat = vrep.simxGetObjectQuaternion(self.clientID, obj['handle'], -1, vrep.simx_opmode_streaming)
                object_pose[obj['name']] = np.array(object_position + object_quat)

            self.all_info['object_pose'] = object_pose
            
    def step(self, actions, axis=0):
        if actions.ndim == 2:
            actions = actions[0,:]
        # clip actions to limits
        if self.CookingEnv_config.get('action_space') is not None:
            clipped_actions = np.clip(np.array(actions), self.action_space['lower_bound'], self.action_space['upper_bound'])
        else:
            clipped_actions = actions

        for i in range(len(self.joint_handles[self.CookingEnv_config.get('arm')])):
            if self.CookingEnv_config.get('control_mode') == "velocity":
                return_code = vrep.simxSetJointTargetVelocity(self.clientID, self.joint_handles[self.CookingEnv_config.get('arm')][i], clipped_actions[i], vrep.simx_opmode_oneshot)
            elif self.CookingEnv_config.get('control_mode') == "position":
                return_code = vrep.simxSetJointTargetPosition(self.clientID, self.joint_handles[self.CookingEnv_config.get('arm')][i], clipped_actions[i], vrep.simx_opmode_oneshot)
            elif self.CookingEnv_config.get('control_mode') is None:
                pass
            else:
                raise ValueError("control mode not supported")

                
        self.synchronous_trigger()
        self.update_all_info()

        
        
    def get_info(self):
        return self.all_info
        
  
    def stop(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot)

    def pause(self):
        vrep.simxPauseSimulation(self.clientID, vrep.simx_opmode_oneshot)
    
    def close(self):
        pass

  
if __name__ == "__main__":
    import time
    
    config = {
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
        "arm": "ur5",
        "control_mode": "velocity",
    }

    env = CookingEnv(config)
    env.reset()

    target_pos = env.all_info['target_pose'][:3]
    target_pos[2] -= 0.08
    env.set_target_position(target_pos)
    
    
    env.robot_close_gripper()
    for i in range(20):
        env.synchronous_trigger()

   

    target_pos = env.all_info['target_pose'][:3]
    target_pos[2] += 0.1
    env.set_target_position(target_pos)
    for i in range(30):
        env.synchronous_trigger()

    
    env.robot_open_gripper()
    env.synchronous_trigger()

        
    # env.reset()
    # for i in range(100):
        
        #### test set target position ####
        # target_pos = env.all_info['target_pose'][:3]
        # target_pos[2] += 0.02
        # env.set_target_position(target_pos)

        #### test set target quaternion ####
        # target_quat = env.all_info['target_pose'][3:]
        # rpy = transformations.euler_from_quaternion(target_quat)
        # rpy_new = np.array(rpy)
        # rpy_new[2] += 0.2
        # quat_new = transformations.quaternion_from_euler(rpy_new[0], rpy_new[1], rpy_new[2])
        # quat_new = np.array(quat_new)
        # env.set_target_quaternion(quat_new)


        # env.synchronous_trigger()
        # time.sleep(0.1)