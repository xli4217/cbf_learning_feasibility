import os
import numpy as np
import glob
from tqdm import tqdm
import copy

from rl_pipeline.replay_buffer.replay_buffer_base import ReplayBufferBase
from rl_pipeline.exploration.epsilon_linear_decay import EpsilonLinearDecay
from rl_pipeline.sampler.batch_sampler import BatchSampler
from rl_pipeline.train.rl.sync_runner import SynchronousRunner
from rl_pipeline.evaluation.evaluation import Evaluation
from rl_pipeline.preprocessor.running_average import RunningAverage
from rl_pipeline.hyperparameter_tuner.random_search import RandomSearch
from rl_pipeline.hyperparameter_tuner.skopt_tuner import SkoptTuner
from rl_pipeline.algo_devel.agent_library import AgentLibrary
from rl_pipeline.logging.tensorboardX_logger import TensorboardXLogger
from rl_pipeline.preprocessor.running_average import RunningAverage


###################################
# Config to rl base runner config #
###################################
def construct_rl_base_runner_config(restore_runner_dir=None,
                                    # environment config
                                    env_name='vrep',
                                    task='makehotdog',
                                    fsa_save_dir="",
                                    fsa_name="fsa",
                                    headless=False,
                                    robot='jaco',
                                    components= {'mdp': True, 'fsa': False, 'translation_gen': 'clf_cbf', 'orientation_gen': 'dmp'},
                                    particle_test=False,
                                    # reset config
                                    reset_config=None,
                                    # critic_config
                                    # agent config
                                    agent='ddpg',
                                    # -- common configs
                                    lr=3e-4,
                                    nb_optimization_steps_per_itr=10,
                                    gamma=0.98,
                                    no_update_until=5000,
                                    save_replay_buffer=False,
                                    log_components={},
                                    # -- ddpg configs
                                    batch_size=32,
                                    target_q_update_interval=10,
                                    soft_target_update=0.005,
                                    update_critic=True,
                                    td3=False,
                                    # -- ppo configs
                                    policy_lr=3e-4,
                                    minibatch_size=32,
                                    nb_training_epoches=1,
                                    value_loss_coef=0.1,
                                    entropy_coef=0.2,
                                    lam=0,
                                    # replay_buffer_config
                                    per_alpha=0,
                                    per_beta0=0.4,
                                    # sampler config
                                    rollout_batch_size=5,
                                    max_episode_timesteps=100,
                                    log_episode=False,
                                    log_batch=False,
                                    use_preprocessors=False,
                                    update_preprocessors=False,
                                    save_preprocessors=False,
                                    # construction config
                                    cmd_log=""):

    
    #### Environment ####
    from experiment_config import ExperimentConfig
    exp_config = ExperimentConfig(config={'env_name': env_name,
                                          'robot': robot,
                                          'components': components,
                                          'task': task,
                                          'fsa_save_dir': fsa_save_dir,
                                          'fsa_name': fsa_name,
                                          'headless':headless,
                                          'particle_test': particle_test})
    
    process_rewards = exp_config.process_rewards    
    Environment = exp_config.Environment
    Reset = exp_config.reset
    
    agent_library = AgentLibrary()
    
    common_agent_config  = {
        'lr': lr, 
        'gamma': gamma,
        'batch_size': batch_size,
        'nb_optimization_steps_per_itr': nb_optimization_steps_per_itr,
        'target_q_update_interval': target_q_update_interval,
        'no_update_until': no_update_until,
        'save_replay_buffer': save_replay_buffer,
        'log_components':log_components
    }
       
    #### Agent ####
    if agent == "ddpg":
        ddpg_agent_config = copy.copy(common_agent_config)
        ddpg_agent_config.update(update_critic=update_critic,
                                 soft_target_update=soft_target_update,
                                 td3=td3)
        Exploration, Actor, Critic, Agent = agent_library.get_agent('ddpg', ddpg_agent_config)
        
        logger_type = TensorboardXLogger
        
    elif agent == 'ppo':
        ppo_agent_config = copy.copy(common_agent_config)
        ppo_agent_config.update(policy_lr=policy_lr,
                                minibatch_size=minibatch_size,
                                nb_training_epoches=nb_training_epoches,
                                value_loss_coef=value_loss_coef,
                                entropy_coef=entropy_coef,
                                lam=lam)
        
        # ppo_agent_config.update()
        Exploration, Actor, Critic, Agent = agent_library.get_agent('ppo', ppo_agent_config)
                
        logger_type = TensorboardXLogger

    else:
        raise ValueError('unsupported agent')
 
        
        
    base_runner_config = {
        'ExpDir': None,
        'seed': 0,
        'cmd_log': cmd_log,
        'restore_runner_dir': restore_runner_dir,
        'base_config': {},
        'Logger': {
            'type': logger_type,
            'config': {
                "csv_data_params": {},
                "log_params": {},
                "model_params": {
                    "interval": 50 # save model every 10 iterations
                },
                "transitions_params":{
                    "interval": 50, # record trajecory batch and the computation graph every this many iterations,
                }
            }
        },
        'Reset': Reset,
        'Environment': Environment,
        'ReplayBuffer': {
            'type': ReplayBufferBase,
            'config': {
                # demo_data = {"States": <states>, "Actions": <actions>}
                'demo_data': None, 
                'capacity': 1e6,
                'seed': 0,
                'shuffle': True,
                'per_alpha': per_alpha, # 0 is no PER 
                'beta0': per_beta0, # To what degree to use importance weights, (0 - no corrections, 1 - full correction)
                'epsilon': 1e-6,
                'prioritized_replay_beta_itrs': 1e5,
                'restore_dir': None
            }
        },
        "Preprocessors":{
            "state_preprocessor": {"type": RunningAverage, 'config':{"dim": None, "shift": True, "scale": True}},
            "reward_preprocessor": {"type": None, 'config':{"dim": 1, "shift": False, "scale": True}}
        },
        'Exploration':Exploration,
        'Actor':Actor,
        'Critic': Critic,
        'Agent':Agent,
        'Sampler':{
            'type': BatchSampler,
            'config': {
                'rollout_batch_size': rollout_batch_size, # in units of episodes (if ray is used, this is for each agent)
                "max_episode_timesteps": max_episode_timesteps,  
                "init_batch_size": 0, # run a few episode of untrained policy to initialize scaler
                'use_preprocessors': use_preprocessors,
                'update_preprocessors': update_preprocessors, # this might be false if multiple runners are used to collect sample, their preprocessors might need to be updated with all experiences
                'save_preprocessors': save_preprocessors,
                'log_info_keys': [],
                'process_rewards': process_rewards,
                'log_episode': log_episode,
                'log_batch': log_batch
            }
            
        }
    }

    return base_runner_config

    
###########################
# Construct rl experiment #
###########################
def construct_rl_experiment_config(experiment_root_dir="",
                                   exp_name='test',
                                   experiment_mode='train',
                                   HyperparameterTuner=None,
                                   # runner_config
                                   max_itr=10000,
                                   # runner_config -> rl_base_runner_config
                                   restore_runner_dir=None,
                                   restore_policy_dir=None,
                                   restore_policy_name=None,
                                   # runner_config -> rl_base_runner_config -> env_config
                                   env_name='vrep',
                                   task='gotogoal',
                                   headless=False,
                                   fsa_save_dir="",
                                   fsa_name="fsa",
                                   robot='jaco',
                                   # translation_gen can be 'clf', 'cbf', 'clf_cbf', 'dmp', orientation_gen can be 'dmp'
                                   components= {'mdp': True, 'fsa': False, 'translation_gen': 'clf_cbf', 'orientation_gen': 'dmp'},
                                   particle_test=False,
                                   # runner_config -> rl_base_runner_config -> agent_config
                                   agent='sac',
                                   # -- agent common
                                   reset_config=None,
                                   lr=3e-4,
                                   nb_optimization_steps_per_itr=10,
                                   target_q_update_interval=10,
                                   gamma=0.98,
                                   no_update_until=5000,
                                   save_replay_buffer=False,
                                   log_components={},
                                   # -- ddpg config
                                   batch_size=32,
                                   soft_target_update=0.005,
                                   update_critic=True,
                                   td3=False,
                                   # -- ppo config
                                   policy_lr=3e-4,
                                   minibatch_size=32,
                                   nb_training_epoches=1,
                                   value_loss_coef=0.1,
                                   entropy_coef=0.2,
                                   lam=0,
                                   # runner_config -> rl_base_runner_config -> replay_buffer_config
                                   per_alpha=0,
                                   per_beta0=0.4,
                                   # runner_config -> rl_base_runner_config -> sampler_config
                                   rollout_batch_size=5,
                                   max_episode_timesteps=100,
                                   log_episode=False,
                                   log_batch=False,
                                   use_preprocessors=False,
                                   update_preprocessors=False,
                                   save_preprocessors=False,
                                   cmd_log=""):

    if agent == 'sac':
        from rl_pipeline.algo_devel.sac.pytorch.sac_runner import SACRunner
        base_runner = SACRunner
    if agent == 'ddpg':
        from rl_pipeline.algo_devel.ddpg.pytorch.ddpg_runner import DDPGRunner
        base_runner = DDPGRunner
    if agent == 'ppo':
        from rl_pipeline.algo_devel.ppo.pytorch.ppo_runner import PPORunner
        base_runner = PPORunner
    experiment_config = {
        # this can be 'train' or 'hyperparam_tuning'
        'mode': experiment_mode,
        'nb_random_seed_exp': 1,
        'ExpDir': None,
        'exp_root_dir': experiment_root_dir,
        'exp_name': exp_name,
        'seed': 0,
        'Runner':{
            'type': SynchronousRunner,
            'config': {
                'BaseRunner':{
                    'type': base_runner,
                    'config': construct_rl_base_runner_config(restore_runner_dir=restore_runner_dir,
                                                              # env config
                                                              env_name=env_name,
                                                              headless=headless,
                                                              task=task,
                                                              fsa_save_dir=fsa_save_dir,
                                                              fsa_name=fsa_name,
                                                              robot=robot,
                                                              components=components,
                                                              particle_test=particle_test,
                                                              # agent_config
                                                              agent=agent,
                                                              # -- common
                                                              reset_config=reset_config,
                                                              lr=lr,
                                                              batch_size=batch_size,
                                                              nb_optimization_steps_per_itr=nb_optimization_steps_per_itr,
                                                              gamma=gamma,
                                                              no_update_until=no_update_until,
                                                              save_replay_buffer=save_replay_buffer,
                                                              log_components=log_components,
                                                              # -- ddpg config
                                                              target_q_update_interval=target_q_update_interval,
                                                              soft_target_update=soft_target_update,
                                                              update_critic=update_critic,
                                                              td3=td3,
                                                              # -- ppo config
                                                              policy_lr=policy_lr,
                                                              minibatch_size=minibatch_size,
                                                              nb_training_epoches=nb_training_epoches,
                                                              value_loss_coef=value_loss_coef,
                                                              entropy_coef=entropy_coef,
                                                              lam=lam,
                                                              # sampler config
                                                              rollout_batch_size=rollout_batch_size,
                                                              max_episode_timesteps=max_episode_timesteps,
                                                              use_preprocessors=use_preprocessors,
                                                              update_preprocessors=update_preprocessors,
                                                              save_preprocessors=save_preprocessors,
                                                              # replay buffer config
                                                              log_episode=log_episode,
                                                              log_batch=log_batch,
                                                              per_alpha=per_alpha,
                                                              per_beta0=per_beta0,
                                                              cmd_log=cmd_log)
                },
                'Evaluator': {
                    'type': Evaluation,
                    'config': {
                        'gamma': 0.99,
                        'log_state_dimensions': [],
                        'log_action_dimensions': [],
                        'log_info_keys': []
                    }
                },
                'nb_remote_runners': 0,
                # this is for vrep
                'ports': [20001, 20002, 20003, 20004],
                'suffix': ["", "#1", "#3", "#5"],
                'max_itr': max_itr,
                'evaluation_logging': False,
                'evaluation_batch_size': 0,
                'evaluation_logging_interval': 0,
            }
        },
        'HyperparameterTuner': HyperparameterTuner
    }

    return experiment_config


#############################
# construct deployer config #  
#############################
def construct_deployer_config(agent='ppo',
                              nb_trial_runs=10,
                              sampler_traj=True,
                              # env_config
                              env_name='',
                              env_from_config=True,
                              task='makehotdog',
                              fsa_save_dir=os.getcwd(),
                              fsa_name='fsa',
                              headless=False,
                              robot='jaco',
                              # translation_gen can be 'clf', 'cbf', 'clf_cbf', 'dmp', orientation_gen can be 'dmp'
                              components= {'mdp': True, 'fsa': False, 'translation_gen': 'clf_cbf', 'orientation_gen': 'dmp'},
                              # policy config
                              exp_name='test',
                              hyperparam_dir='seed0',
                              itr=0,
                              # sampler config
                              max_episode_timesteps=50,
                              log_episode=False,
                              log_batch=False,
                              use_preprocessors=False,
                              update_preprocessors=False,
                              save_preprocessors=False
):

    from rl_pipeline.logging.tensorboardX_logger import TensorboardXLogger

    deployer_config = {
        'agent': agent,
        'env_name': env_name,
        'env_from_config': env_from_config,
        'sampler_traj': sampler_traj,
        'exp_config': {
            'env_name': env_name,
            'task': task,
            'fsa_save_dir': fsa_save_dir,
            'fsa_name': fsa_name,
            "headless":headless,
            'robot': robot,
            'components': components,
        },
        'experiment_name': exp_name,
        'hyperparam_dir': hyperparam_dir,
        'itr': itr,
        'nb_trial_runs': nb_trial_runs,
        # samper config
        'max_episode_timesteps': max_episode_timesteps,
        'log_episode': log_episode,
        'log_batch': log_batch,
        'use_preprocessors': use_preprocessors,
        # logger
        'Logger': {
            'type':TensorboardXLogger,
            'config':{
                "csv_data_params": {},
                "log_params": {},
            }
        }
    }
    
    return deployer_config
    
######################################################
# construct experiment config (highest level config) #
######################################################

default_args = {
    'experiment_root_dir': os.path.join(os.environ['LEARNING_PATH'], 'learning'),
    'exp_name': 'test',
    # this can be 'train' or 'hyperparam_tuning', 'deploy', 'teleop'
    'mode': 'train',
    #### rl_config ####
    'restore_policy_dir': None,
    'restore_policy_name': None,
    #### rl_config -> run_one_experiment_config ####
    'restore_runner_dir': None,
    'max_itr': 1000,
    #### environment config ####
    'env_name': 'vrep',
    'task': 'serve',
    'headless': False,
    'fsa_save_dir': os.path.join(os.environ['LEARNING_PATH'], 'learning', 'figures'),
    'fsa_name': "fsa",
    'robot': 'baxter',
    # translation_gen can be 'clf', 'cbf', 'clf_cbf', 'dmp', 'None', orientation_gen can be 'dmp' or None
    'components': {'mdp': True, 'fsa': True, 'translation_gen': None, 'orientation_gen': None},
    'particle_test': False,
    ##### replay buffer config ####
    'per_alpha': 0.,
    'per_beta0': 0.4,
    #### agent_config ####
    'agent': 'ppo',
    # -- common
    'lr': 3e-4,
    'batch_size':32,
    'nb_optimization_steps_per_itr':10,
    'reset_config':{},
    'gamma':0.98,
    'no_update_until':5000,
    'save_replay_buffer': False,
    'log_components':{
        'state_dist': False,
        'action_dist': True,
        'reward_dist': False,
        'done_dist': False,
        'adv_dist': False,
        'policy_gradient_dist': False,
        'value_gradient_dist':False,
        'policy_weights': False,
        'value_weights': False,
        'other_dist': False
    },
    # -- ddpg configs
    'target_q_update_interval':1,
    'soft_target_update':0.005,
    'update_critic': True,
    'td3': False,
    # -- ppo configs
    'policy_lr': 3e-4,
    'minibatch_size':32,
    'nb_training_epoches': 1,
    'entropy_coef': 0.2,
    'value_loss_coef': 0.1,
    'lam': 0,
    #### sampler config ####
    'rollout_batch_size': 5,
    'max_episode_timesteps':100,
    'use_preprocessors': False,
    'update_preprocessors': False,
    'save_preprocessors': False,
    #### deploy config ####
    'hyperparam_dir': 'seed0',
    'itr': 0,
    'nb_trial_runs': 10,
    'sampler_traj': True,
    'env_from_config': True,
    'log_episode': False,
    'log_batch': False,
    #### HyperparamterTuner ####
    'HyperparameterTuner':{
        'type': SkoptTuner,
        'config':{
            "performance_metric": "average_return",
            "nb_steps": 100,
            "params_dict":{
                #### Common ####
                'lr': {"type": "float", "range": [1e-5, 5e-3], "prior": "log-uniform", "transform": "identity"},                                                                                                                        
                "batch_size": {"type": "integer", "range": [64, 256], "prior": "uniform", "transform": "identity"},                                                                                                                         
                "nb_optimization_steps_per_itr": {"type": "integer", "range": [10, 50], "prior": "uniform", "transform": "identity"},                                                                                                                     
                
                'soft_target_update': {"type": "float", "range": [0.001, 0.005], "prior": "uniform", "transform": "identity"},
                # 'per_alpha': {"type": "float", "range": [0., 0.6], "prior": "uniform", "transform": "identity"},
                #### PPO agent ####
                'policy_lr': {"type": "float", "range": [1e-3, 5e-2], "prior": "log-uniform", "transform": "identity"},                                                                                                                 
                "minibatch_size": {"type": "integer", "range": [64, 256], "prior": "uniform", "transform": "identity"},                                                                                                 

                "nb_training_epoches": {"type": "integer", "range": [5, 10], "prior": "uniform", "transform": "identity"},                                                                                                                     
                'value_loss_coef': {"type": "float", "range": [0.01, 0.1], "prior": "uniform", "transform": "identity"},                                                                                                                       
                'entropy_coef': {"type": "float", "range": [0.1, 0.2], "prior": "uniform", "transform": "identity"},                                                                                                                           
                'lam': {"type": "float", "range": [0.5, 1.], "prior": "uniform", "transform": "identity"}     
                
            },
            # this is optimizer specific arguments
            "optimizer": {
                "base_estimator": "GP",
                "n_initial_points": 1,
                "random_state": 5,
            }
        }
    }
}
    
def construct_experiment_config(experiment_root_dir=default_args['experiment_root_dir'],
                                exp_name=default_args['exp_name'],
                                mode=default_args['mode'],
                                #### rl config ####
                                restore_policy_dir=default_args['restore_policy_dir'],
                                restore_policy_name=default_args['restore_policy_name'],
                                #### run_one_experiment_config #### 
                                restore_runner_dir=default_args['restore_runner_dir'],
                                max_itr=default_args['max_itr'],
                                #### env config ####
                                env_name=default_args['env_name'],
                                headless=default_args['headless'],
                                task=default_args['task'],
                                fsa_save_dir=default_args['fsa_save_dir'],
                                fsa_name=default_args['fsa_name'],
                                robot=default_args['robot'],
                                components=default_args['components'],
                                particle_test=default_args['particle_test'],
                                #### agent config ####
                                agent=default_args['agent'],
                                # -- common configs
                                reset_config=default_args['reset_config'],
                                lr=default_args['lr'],
                                batch_size=default_args['batch_size'],
                                gamma=default_args['gamma'],
                                no_update_until=default_args['no_update_until'],
                                nb_optimization_steps_per_itr=default_args['nb_optimization_steps_per_itr'],
                                save_replay_buffer=default_args['save_replay_buffer'],
                                log_components=default_args['log_components'],
                                # -- ddpg configs
                                target_q_update_interval=default_args['target_q_update_interval'],
                                soft_target_update=default_args['soft_target_update'],
                                update_critic=default_args['update_critic'],
                                td3=default_args['td3'],
                                # -- ppo configs
                                policy_lr=default_args['policy_lr'],
                                minibatch_size=default_args['minibatch_size'],
                                nb_training_epoches=default_args['nb_training_epoches'],
                                value_loss_coef=default_args['value_loss_coef'],
                                entropy_coef=default_args['entropy_coef'],
                                lam=default_args['lam'],
                                #### sampler config ####
                                rollout_batch_size=default_args['rollout_batch_size'],
                                max_episode_timesteps=default_args['max_episode_timesteps'],
                                log_episode=default_args['log_episode'],
                                log_batch=default_args['log_batch'],
                                use_preprocessors=default_args['use_preprocessors'],
                                update_preprocessors=default_args['update_preprocessors'],
                                save_preprocessors=default_args['save_preprocessors'],
                                #### replay buffer config ####
                                per_alpha=default_args['per_alpha'],
                                per_beta0=default_args['per_beta0'],
                                #### hyper parameter tuner ####
                                HyperparameterTuner=default_args['HyperparameterTuner'],
                                #### deploy config ####
                                hyperparam_dir=default_args['hyperparam_dir'],
                                itr=default_args['itr'],
                                nb_trial_runs=default_args['nb_trial_runs'],
                                sampler_traj=default_args['sampler_traj'],
                                env_from_config=default_args['env_from_config'],
                                cmd_log=""):

    #### RL ####
    from rl_pipeline.train.rl.run_one_experiment import RunOneExperiment

    #### task specific ####
    from examples.iclr2019.experiment_config import construct_simple_task

    spec = None

    experiment_config = {
        # experiment_root_dir: root of the experiment folder where training/evaluation results are saved 
        'experiment_root_dir': experiment_root_dir,
        # exp_name="test": current experiment name
        'exp_name': exp_name,
        'mode': mode,
        'ReinforcementLearner': {
            'type': RunOneExperiment,
            'config': {
                'rl_run_one_experiment_config': construct_rl_experiment_config(experiment_root_dir=experiment_root_dir,
                                                                               exp_name=exp_name,
                                                                               experiment_mode=mode,
                                                                               #### runner_config ####
                                                                               max_itr=max_itr,
                                                                               # runner_config -> rl_base_runner_config
                                                                               restore_runner_dir=restore_runner_dir,
                                                                               restore_policy_dir=restore_policy_dir,
                                                                               restore_policy_name=restore_policy_name,
                                                                               #### env config ####
                                                                               env_name=env_name,
                                                                               headless=headless,
                                                                               task=task,
                                                                               fsa_save_dir=fsa_save_dir,
                                                                               fsa_name=fsa_name,
                                                                               robot=robot,
                                                                               components=components,
                                                                               particle_test=particle_test,
                                                                               #### agent_config ####
                                                                               agent=agent,
                                                                               # -- common configs
                                                                               lr=lr,
                                                                               batch_size=batch_size,
                                                                               nb_optimization_steps_per_itr=nb_optimization_steps_per_itr,
                                                                               reset_config=reset_config,
                                                                               gamma=gamma,
                                                                               no_update_until=no_update_until,
                                                                               save_replay_buffer=save_replay_buffer,
                                                                               log_components=log_components,
                                                                               # -- ddpg configs
                                                                               target_q_update_interval=target_q_update_interval,
                                                                               soft_target_update=soft_target_update,
                                                                               update_critic=update_critic,
                                                                               td3=td3,
                                                                               # -- ppo configs
                                                                               policy_lr=policy_lr,
                                                                               minibatch_size=minibatch_size,
                                                                               nb_training_epoches=nb_training_epoches,
                                                                               value_loss_coef=value_loss_coef,
                                                                               entropy_coef=entropy_coef,
                                                                               lam=lam,
                                                                               #### replay buffer config ####
                                                                               per_alpha=per_alpha,
                                                                               per_beta0=per_beta0,
                                                                               #### sampler config ####
                                                                               rollout_batch_size=rollout_batch_size,
                                                                               max_episode_timesteps=max_episode_timesteps,
                                                                               log_episode=log_episode,
                                                                               log_batch=log_batch,
                                                                               use_preprocessors=use_preprocessors,
                                                                               update_preprocessors=update_preprocessors,
                                                                               save_preprocessors=save_preprocessors,
                                                                               #### hyperparamter tuner ####
                                                                               HyperparameterTuner=HyperparameterTuner,
                                                                               cmd_log=cmd_log)
                                                                              
            },
        },
        'Deployer': {
            'type': None,
            'config': construct_deployer_config(agent=agent,
                                                sampler_traj=sampler_traj,
                                                # env_config
                                                env_name=env_name,
                                                env_from_config=env_from_config,
                                                task=task,
                                                fsa_save_dir=fsa_save_dir,
                                                fsa_name=fsa_name,
                                                robot=robot,
                                                components=components,
                                                # policy config
                                                exp_name=exp_name,
                                                hyperparam_dir=hyperparam_dir,
                                                itr=itr,
                                                # sampler config
                                                nb_trial_runs=nb_trial_runs,
                                                max_episode_timesteps=max_episode_timesteps,
                                                log_episode=log_episode,
                                                log_batch=log_batch,
                                                use_preprocessors=use_preprocessors)
            
        }
    }

               
    
    return experiment_config


if __name__ == "__main__":
    print(construct_experiment_config())