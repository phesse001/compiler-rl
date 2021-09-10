import compiler_gym
import gym
from gym import spaces
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit, CycleOverBenchmarks, RandomOrderBenchmarks, ObservationWrapper
import numpy as np
import random
import itertools
from itertools import cycle, islice 
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import CompilerEnvWrapper, ObservationWrapper, RewardWrapper

# code based off of example RLlib implementation of compiler_gym environment -> https://github.com/facebookresearch/CompilerGym/blob/development/examples/rllib.ipynb

# define a make_env() helper function to make environement.
# [optional] use the compiler_gym.wrappers API to implement custom contraints

# try making wrapper to use done logic
class stepWrapper(CompilerEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.patience = 10
        self.reward_counter = 0
        self.actions_taken = []

    def step(self, action, **kwargs):
        next_state, reward, done, info = self.env.step(action, **kwargs)
        if reward <= 0:
            self.reward_counter += 1
        else:
            self.reward_counter = 0

        if self.reward_counter > self.patience:
            done = True

        return next_state, reward, done, info


class observationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        # modify obs
        return obs

class rewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        # access attributes of parent class
    def reward(self, rew):
        # modify rew
        # make reward penalized for staying the same (i.e choosing action with 0 reward over and over)
        print(f'The reward is {rew}')
        return rew

class actionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        # modify action
        return action

def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make("llvm-v0", observation_space="InstCount", reward_space="IrInstructionCountOz")
    env = stepWrapper(env)
    return env

# create benchmarks to be used
with make_env() as env:
    # grab ~1000 benchmarks for training from csmith dataset
    #csmith = list(islice(env.datasets['generator://csmith-v0'].benchmarks(), 1000))
    cbench = list(env.datasets['benchmark://cbench-v1'].benchmark_uris())
    train_benchmarks = cbench

def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    del args
    return RandomOrderBenchmarks(make_env(), train_benchmarks)

tune.register_env("compiler_gym", make_training_env)

if ray.is_initialized():
    ray.shutdown()
ray.init(include_dashboard=True, ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

# edit default config
config['num_workers'] = 39
# prob with using gpu and torch in rllib...
config['num_gpus'] = 1
# this splits a rollout into an episode fragment of size n - make 10 cuz that is the min ep len
config['rollout_fragment_length'] = 10
# this will combine fragements into a batch to perform sgd
config['train_batch_size'] = 390
# number of points to randomly select for GD
config['sgd_minibatch_size'] = 20
config['lr'] = 0.0001
config['gamma'] = 0.995
# make maximum episode length 60 time steps
config['horizon'] = 60
config['framework'] = 'torch'
config['env'] = 'compiler_gym'
config['model']['fcnet_activation'] = 'relu'
config['model']['fcnet_hiddens'] = [1024, 1024, 1024] 

#train, load, and test functions from https://bleepcoder.com/ray/644594660/rllib-best-workflow-to-train-save-and-test-agent

def train(stop_criteria, save_dir):
    """
    Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
    :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    analysis = ray.tune.run(ppo.PPOTrainer, config=config, local_dir=save_dir, stop=stop_criteria,
                            checkpoint_at_end=True)
    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    trial = analysis.get_best_trial('episode_reward_mean', 'max', 'all', True)
    checkpoints = analysis.get_trial_checkpoints_paths(trial=trial, metric='episode_reward_mean')
    # retrieve the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    return checkpoint_path, analysis

def load(path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    agent = ppo.PPOTrainer(config=config, env="compiler_gym")
    agent.restore(path)
    return agent

def rollout(agent, env):
    """Test trained agent for a single episode. Return the episode reward"""
    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
    return episode_reward

def test(env, agent_path):
    test_agent = load(agent_path)
    env.observation_space = "InstCount"
    # wrap env so episode can terminate after n rewardless steps
    env = stepWrapper(env)
    rollout(test_agent, env)

# start training
if __name__ == "__main__":
    #eval_llvm_instcount_policy(test)
    #test_agent = load(agent_path)
    save_dir = './log_dir'
    agent_path, anaysis_obj = train({"episodes_total":1000000}, save_dir)
    agent = load(agent_path)
    eval_llvm_instcount_policy(test(agent_path=agent_path))

