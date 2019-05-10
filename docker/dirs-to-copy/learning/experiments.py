import os
import numpy as np
import glob
from tqdm import tqdm
import cloudpickle

from experiment_config import ExperimentConfig
import fire

default_config = {
    # experiment_root_dir: root of the experiment folder where training/evaluation results are saved 
    'experiment_root_dir': os.getcwd(),
    # exp_name="test": current experiment name
    'exp_name': 'test',
    # this can be 'train' or 'deploy_sim_vrep' or 'deploy_real' or 'teleop'
    'mode': 'train',
    'ReinforcementLearner': {
        'type': None,
        'config': None
    },
    'Deployer': {
        'type': None,
        'config': {}
    }
}



class RunExperiment(object):

    def __init__(self, config={}):
        self.config = default_config
        self.config.update(config)    

    def teleop(self):
        rl_config = self.config['ReinforcementLearner']
        rl_experiment_config = rl_config['config']['rl_run_one_experiment_config']
        rl_base_runner_config = rl_experiment_config['Runner']['config']['BaseRunner']['config']

        env = rl_base_runner_config['Environment']['type'](rl_base_runner_config['Environment']['config'])
        env.reset()
        prev_state =  env.get_state()
        R = 0
        while True:
            cmd = input('insert cmd: ')
            env.teleop(cmd)
            curr_state = env.get_state()
            done = env.is_done(state=curr_state)
            r = env.get_reward(state=prev_state, next_state=curr_state)
            print("state: ", curr_state)
            print("reward: ", r)
            print("done: ", done)
            if not isinstance(r, tuple):
                R += r
            if any(list(done)):
                print("return: ", R)
                R = 0
                env.reset()
            prev_state = curr_state
            
    def reinforcement_learning(self, config={}):
        '''
        Starts reinforcement learning
        '''
        rl_config = self.config['ReinforcementLearner']
        rl_config.update(config)
        rl_experiment_config = rl_config['config']['rl_run_one_experiment_config']
        rl_base_runner_config = rl_experiment_config['Runner']['config']['BaseRunner']['config']

        run_one_experiment = rl_config['type'](rl_experiment_config)
        run_one_experiment.run_one_experiment()

    def deploy(self, config={}):
        deploy_config = self.config['Deployer']['config']
        deploy_config.update(config)

        experiment_root_dir = self.config['experiment_root_dir']

        #### Construct depoly logger ####
        from rl_pipeline.logging.tensorboardX_logger import TensorboardXLogger
        from rl_pipeline.configuration.experiment_dirs import ExperimentDirs

        experiment_dir = os.path.join(experiment_root_dir, 'experiments')
        curr_experiment_dir = os.path.join(experiment_dir, 'deploy')
        exp_dir = ExperimentDirs(experiment_dir, curr_experiment_dir)
        exp_dir.create_hyperparam_exp('seed0')

        logger = deploy_config['Logger']['type'](csv_data_dir=exp_dir.csv_data_hyperparam_exp_dir,
                                                 log_dir=exp_dir.log_hyperparam_exp_dir,
                                                 model_dir=exp_dir.model_hyperparam_exp_dir,
                                                 config_dir=exp_dir.config_hyperparam_exp_dir,
                                                 info_dir=exp_dir.info_hyperparam_exp_dir,
                                                 transitions_dir=exp_dir.transitions_hyperparam_exp_dir,
                                                 logging_params=deploy_config['Logger']['config'])
        ##########################
        
        itr_dir = os.path.join(experiment_root_dir,
                               'experiments',
                               deploy_config['experiment_name'],
                               'transitions',
                               deploy_config['hyperparam_dir'],
                               'itr_'+str(deploy_config['itr']))

        config_path = os.path.join(experiment_root_dir,
                                   'experiments',
                                   deploy_config['experiment_name'],
                                   'config',
                                   deploy_config['hyperparam_dir'],
                                   'config.pkl')

        #### load config ####
        config = cloudpickle.loads(open(config_path, 'rb').read())
        
        
        #### construct env from config ####
        if deploy_config['env_from_config']:
            #### construct reset ####
            reset_type = config.get(['Reset', 'type'])
            reset_config = config.get(['Reset', 'config'])
           
            reset = None
            if reset_type is not None:
                reset = reset_type(reset_config)

            #### construct environment ####
            env_config = config.get(['Environment', 'config'])
            env_config['headless'] = False

            env = config.get(['Environment', 'type'])(env_config)
            
        else:
            #### construct fresh environment (useful if want to change param) ####
            from experiment_config import ExperimentConfig
            exp_config = ExperimentConfig(config=deploy_config['exp_config'])
            process_rewards = exp_config.process_rewards    
            Environment = exp_config.Environment
            reset = None
            if deploy_config['env_name'] == 'vrep_baxter':
                Reset = exp_config.reset
                reset = Reset['type'](Reset['config'])
            env = Environment['type'](Environment['config'])
                
            
        #### load exploration ####
        exploration_config = config.get(['Exploration', 'config'])
        explore = None
        if config.get(['Exploration', 'type']) is not None:
            explore = config.get(['Exploration', 'type'])(exploration_config, action_space=env.action_space)
        #### load policy ####
        policy_config = config.get(['Actor', 'config'])
        policy_config['obs_dim'] = env.state_space['shape'][0]
        policy_config['action_dim'] = env.action_space['shape'][0]
        policy_config['action_space'] = env.action_space
        policy = config.get(['Actor', 'type'])(policy_config, exploration_strategy=explore, logger=logger)
        policy.restore(model_dir=itr_dir, model_name='policy')
        print("Loaded policy from {}".format(os.path.join(itr_dir, 'policy')))

        #### load state preprocessor ####
        state_preprocessor_config = config.get(['Preprocessors', 'state_preprocessor', 'config'])
        state_preprocessor_config['dim'] = env.state_space['shape'][0]
        state_preprocessor_type = config.get(['Preprocessors', 'state_preprocessor', 'type'])

        state_preprocessor = None
        if state_preprocessor_type is not None and deploy_config['use_preprocessors']:
            state_preprocessor = state_preprocessor_type(state_preprocessor_config)
            ## restore state preprocessor
            state_preprocessor_restore_path = os.path.join(experiment_root_dir,
                                                           'experiments',
                                                           deploy_config['experiment_name'],
                                                           'info',
                                                           deploy_config['hyperparam_dir'],
                                                           'state_preprocessor_params.pkl')
            state_preprocessor.restore_preprocessor(state_preprocessor_restore_path)
            
        #######
        # Run #
        #######

        #### Finer episodic version ####
        if not deploy_config['sampler_traj']:
            return_list = []
            for _ in range(deploy_config['nb_trial_runs']):
                env.reset(s=None)
                print('reset!')
                R = 0
                obs_prev = None
                action_prev = None
                for i in range(deploy_config['max_episode_timesteps']):
                    obs = env.get_state()
                    if env.is_done(obs):
                        break
                    if state_preprocessor is not None and deploy_config['use_preprocessors']:
                        obs = state_preprocessor.get_scaled_x(obs)
                    # print(obs[0])
                    action = policy.get_action(obs, deterministic=True)
                    env.step(action)
                    if i > 0:
                        R += env.get_reward(obs_prev, action_prev, obs)
                    obs_prev = obs
                    action_prev = action
                print("episode return: {}".format(R))
                return_list.append(R)
            print("average return: {}".format(np.mean(np.array(return_list))))
            print("return std: {}".format(np.std(np.array(return_list))))

        #### Sampler version ####
        if deploy_config['sampler_traj']:
            sampler_config = config.get(['Sampler', 'config'])
            sampler_config['log_episode'] = deploy_config['log_episode']
            sampler_config['log_batch'] = deploy_config['log_batch']
            sampler_config['use_preprocessors'] = deploy_config['use_preprocessors']
            sampler_config['update_preprocessors'] = False
            sampler_config['save_preprocessors'] = False
            sampler_type = config.get(['Sampler', 'type'])
            sampler = sampler_type(sampler_config,
                                   env=env,
                                   policy=policy,
                                   state_preprocessor=state_preprocessor,
                                   logger=logger)

            unscaled_batch, scaled_batch = sampler.get_batch(batch_size=deploy_config['nb_trial_runs'],
                                                             episode_horizon=deploy_config['max_episode_timesteps'],
                                                             deterministic=True)
            for traj in unscaled_batch:
                R = np.sum(traj['Rewards'])
                print("Return:", R)
            
    def run(self):
        '''
        Function to run 
        '''
        if self.config['mode'] == 'train' or self.config['mode'] == 'hyperparam_tuning':
            self.reinforcement_learning()
        elif 'deploy' in self.config['mode']:
            self.deploy()
        elif self.config['mode'] == 'teleop':
            self.teleop()
        else:
            raise ValueError('unsupported mode')
        
        

if __name__ == "__main__":
    from training_config import construct_experiment_config
    experiment_config = fire.Fire(construct_experiment_config)
    cls = RunExperiment(experiment_config)
    cls.run()
