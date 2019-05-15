import cloudpickle
import os
import numpy as np
import transformations as tf

def quaternion_log(q):
    u = q[:3]
    v = q[3]

    if np.linalg.norm(u) == 0:
        return np.array([0,0,0])
    else:
        if v > 0.999:
            v = 0.999
        return np.arccos(v) * (u / np.linalg.norm(u))


def quaternion_dist(q1, q2):
    conjugate_product = tf.transformations.quaternion_multiply(q1, tf.transformations.quaternion_conjugate(q2))

    if all(conjugate_product == np.array([0,0,0,-1])):
        return 2*np.pi
    else:
        return 2 * np.linalg.norm(quaternion_log(conjugate_product))


def get_p2(p1, Mp1p2):
    Mp1 = tf.transformations.quaternion_matrix(p1[3:])
    Mp1[:3,3] = p1[:3]

    Mp2 = Mp1.dot(Mp1p2)

    p2_quat = tf.transformations.quaternion_from_matrix(Mp2)

    p2 = np.concatenate([Mp2[:3,3], p2_quat])

    return p2

def pos_distance(p1, p2):
    pos_dist = np.linalg.norm(p1[:3] - p2[:3])
    return pos_dist

def quat_distance(p1, p2):
    quat_dist_arg = 2 * np.inner(p1[3:], p2[3:]) - 1
    quat_dist_arg = np.modf(quat_dist_arg)[0]

    if np.abs(quat_dist_arg) > 0.99 or np.abs(quat_dist_arg) < 0.05:
        quat_distance = 0.
    else:
        quat_distance = np.arccos(quat_dist_arg)

    return quat_distance

    
def pose_distance(p1, p2):
    assert len(p1) == len(p2) == 7

    pos_dist = pos_distance(p1[:3], p2[:3])
    quat_dist = quat_distance(p1[3:], p2[3:])
    
    return pos_dist, quat_dist


def load_policy_and_preprocessor(loading_config):
    policy = None
    state_preprocessor = None

    config_restore_path = loading_config['training_config_restore_path']
        
    if config_restore_path is not None:
        training_config = cloudpickle.loads(open(config_restore_path, 'rb').read())

    #### load policy ####
    policy_restore_path = loading_config['policy_restore_path']
    if policy_restore_path is not None:
        policy_config = training_config.get(['Actor', 'config'])
        policy_config['obs_dim'] = loading_config['state_space']['shape'][0]
        policy_config['action_dim'] = loading_config['action_space']['shape'][0]
        policy_config['action_space'] = loading_config['action_space']
        policy = training_config.get(['Actor', 'type'])(policy_config)
        policy.restore(model_dir=policy_restore_path, model_name='policy')
        print("Loaded policy from {}".format(os.path.join(policy_restore_path, 'policy')))

    #### load state preprocessor ####
    state_preprocessor_restore_path = loading_config['state_preprocessor_restore_path']
    if state_preprocessor_restore_path is not None:    
        state_preprocessor_config = training_config.get(['Preprocessors', 'state_preprocessor', 'config'])
        state_preprocessor_config['dim'] = loading_config['state_space']['shape'][0]
        state_preprocessor_type = training_config.get(['Preprocessors', 'state_preprocessor', 'type'])

        if state_preprocessor_type is not None:
            state_preprocessor = state_preprocessor_type(state_preprocessor_config)
            state_preprocessor.restore_preprocessor(state_preprocessor_restore_path)

    return policy, state_preprocessor
            
        