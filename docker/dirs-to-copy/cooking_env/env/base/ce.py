from cooking_env.env.base.vrep_env_base import VrepEnvBase
import cooking_env.vrep as vrep
import numpy as np
from utils.configuration import Configuration
from future.utils import viewitems
import time
from utils import transformations
from cooking_env.env.base.env_info import robot_handles, object_handles, obstacle_handles

default_config = {
    # Common to all envs
    "seed": 10,
    # specific to this env
    "suffix": "",
    "particle_test": False,
    "arm": "jaco",
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
                                                         rh['target_handle'],
                                                         vrep.simx_opmode_oneshot_wait)
        #### goal handle ####
        _, self.goal_handle = vrep.simxGetObjectHandle(self.clientID,
                                                         rh['goal_handle'],
                                                         vrep.simx_opmode_oneshot_wait)
  
        
        #### object handles ####
        self.object_handles = {}
        for oh in object_handles:
            _, h = vrep.simxGetObjectHandle(self.clientID, oh['handle'], vrep.simx_opmode_oneshot_wait)
            self.object_handles[oh['name']] = h
        

        #### obstacle handles ####
        self.obstacle_handles = []
        for obs_h in obstacle_handles:
            _, h = vrep.simxGetObjectHandle(self.clientID, obs_h['handle'], vrep.simx_opmode_oneshot_wait)
            self.obstacle_handles.append(dict(name=obs_h['name'], handle=h))
        
        #### test particle ####
        self.test_particle_handle = None
        if self.CookingEnv_config.get('particle_test'):
            _, self.particle_handle = vrep.simxGetObjectHandle(self.clientID, rh['particle_handle'], vrep.simx_opmode_oneshot_wait)
        
        #### ee sample region ####
        _, self.ee_sample_region_handle = vrep.simxGetObjectHandle(self.clientID, rh['ee_sample_region_handle'], vrep.simx_opmode_oneshot_wait)

        #### ee motion region ####
        _, self.ee_motion_region_handle = vrep.simxGetObjectHandle(self.clientID, rh['ee_motion_region_handle'], vrep.simx_opmode_oneshot_wait)
        
    # def set_velocity_control_mode(self):
    #    joint_handles = self.joint_handles[self.CookingEnv_config.get('arm')]
    #    for joint_handle in joint_handles:
    #       vrep.simxSetJointForce(self.clientID, joint_handle, 10000, vrep.simx_opmode_oneshot)
    #       vrep.simxSetObjectIntParameter(self.clientID, joint_handle, 2001, 0, vrep.simx_opmode_oneshot)

    def get_region_info(self, region='sample_region'):
        if region == 'sample_region':
            handle = self.ee_sample_region_handle
        elif region == 'motion_region':
            handle = self.ee_motion_region_handle
        else:
            raise ValueError('unsupported region')
        
        _, pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_blocking)

        _, bb_min_x = vrep.simxGetObjectFloatParameter(self.clientID, handle, 15, vrep.simx_opmode_blocking)
        _, bb_min_y = vrep.simxGetObjectFloatParameter(self.clientID, handle, 16, vrep.simx_opmode_blocking)
        _, bb_min_z = vrep.simxGetObjectFloatParameter(self.clientID, handle, 17, vrep.simx_opmode_blocking)
        
        
        _, bb_max_x = vrep.simxGetObjectFloatParameter(self.clientID, handle, 18, vrep.simx_opmode_blocking)
        _, bb_max_y = vrep.simxGetObjectFloatParameter(self.clientID, handle, 19, vrep.simx_opmode_blocking)
        _, bb_max_z = vrep.simxGetObjectFloatParameter(self.clientID, handle, 20, vrep.simx_opmode_blocking)

        bb = {
            'x': np.array([bb_min_x, bb_max_x]) + pos[0],
            'y': np.array([bb_min_y, bb_max_y]) + pos[1],
            'z': np.array([bb_min_z, bb_max_z]) + pos[2],
        }

        return bb
        
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
        
    def set_target_pose(self, pt):
        assert pt.shape == (7,)
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle

        pos = pt[:3]
        quat = pt[3:]    
        vrep.simxSetObjectPosition(self.clientID, handle, -1, pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectQuaternion(self.clientID, handle, -1, quat, vrep.simx_opmode_oneshot)
        self.synchronous_trigger()

        
    def set_goal_pose(self, pt):
        pt = np.array(pt)
        assert pt.shape == (7,)
        handle = self.goal_handle

        pos = pt[:3]
        quat = pt[3:]    
        vrep.simxSetObjectPosition(self.clientID, handle, -1, pos, vrep.simx_opmode_oneshot)
        vrep.simxSetObjectQuaternion(self.clientID, handle, -1, quat, vrep.simx_opmode_oneshot)
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
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle

        rc = 1
        while rc != 0:    
            rc, target_pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_streaming)
            rc, target_quat = vrep.simxGetObjectQuaternion(self.clientID, handle, -1, vrep.simx_opmode_streaming)
            
        return np.array(target_pos), np.array(target_quat)

    def get_obstacle_info(self):
        obs_info = []
        for obs in self.obstacle_handles:
            rc = 1
            while rc != 0:
                rc, pos = vrep.simxGetObjectPosition(self.clientID, obs['handle'], -1, vrep.simx_opmode_streaming)
            obs_info.append({'name': obs['name'], 'position': pos, 'radius': 0.175})

        return obs_info
            
    def get_goal_pose(self):
        handle = self.goal_handle

        rc = 1
        while rc != 0:
            rc, goal_pos = vrep.simxGetObjectPosition(self.clientID, handle, -1, vrep.simx_opmode_streaming)
            rc, goal_quat = vrep.simxGetObjectQuaternion(self.clientID, handle, -1, vrep.simx_opmode_streaming)
    
            
        return np.array(goal_pos), np.array(goal_quat)

        
    def get_target_velocity(self):
        if self.CookingEnv_config.get('particle_test'):
            handle = self.particle_handle
        else:
            handle = self.target_handle

        rc = 1
        while rc != 0:    
            rc, linear_vel, angular_vel = vrep.simxGetObjectVelocity(self.clientID, handle, vrep.simx_opmode_streaming)

        
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
        

    def get_joint_angles(self):
        #### retrive joint angles ####
        joint_positions = []
        for joint_handle in self.joint_handles:
            rc = 1
            while rc != 0:
                rc, joint_position = vrep.simxGetJointPosition(self.clientID, joint_handle, vrep.simx_opmode_streaming)
            joint_positions.append(joint_position)
        return np.array(joint_positions)
     
    def get_object_pose(self):
        #### retrive object pose ####
        if self.object_handles is not None:
            object_pose = {}
            for obj_name, obj_handle in viewitems(self.object_handles):
                rc = 1
                while rc != 0:
                    rc, object_position = vrep.simxGetObjectPosition(self.clientID, obj_handle, -1, vrep.simx_opmode_streaming)
                    rc, object_quat = vrep.simxGetObjectQuaternion(self.clientID, obj_handle, -1, vrep.simx_opmode_streaming)
                object_pose[obj_name] = np.array(object_position + object_quat)

            return object_pose
        else:
            return None
       
        
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
        "arm": "jaco",
        "control_mode": "velocity",
    }

    env = CookingEnv(config)
    env.reset()

    print(env.get_sample_region_info())
    
    #### test grasping ####
    # target_pos = env.all_info['target_pose'][:3]
    # target_pos[2] -= 0.08
    # env.set_target_position(target_pos)
    
    
    # env.robot_close_gripper()
    # for i in range(20):
    #     env.synchronous_trigger()

   

    # target_pos = env.all_info['target_pose'][:3]
    # target_pos[2] += 0.1
    # env.set_target_position(target_pos)
    # for i in range(30):
    #     env.synchronous_trigger()

    
    # env.robot_open_gripper()
    # env.synchronous_trigger()

    #### other test ####
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
