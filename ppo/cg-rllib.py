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

# code based off of example RLlib implementation of compiler_gym environment -> https://github.com/facebookresearch/CompilerGym/blob/development/examples/rllib.ipynb

# define a make_env() helper function to make environement.
# [optional] use the compiler_gym.wrappers API to implement custom contraints

# try making wrapper to use done logic
class envWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.patience = 10
        self.reward_counter = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if reward <= 0:
            self.reward_counter += 1
        else:
            self.reward_counter = 0

        if self.reward_counter > self.patience:
            done = True
        return next_state, reward, done, info


def make_env() -> compiler_gym.envs.CompilerEnv:
    env = envWrapper(compiler_gym.make("llvm-v0", observation_space="Autophase", reward_space="IrInstructionCountOz"))
    return env

# create benchmarks to be used
with make_env() as env:
    # grab ~100 benchmarks for training from different datasets
    cbench = list(env.datasets['cbench-v1'].benchmark_uris())
    gh = list(env.datasets['github-v0'].benchmark_uris())
    gh = random.sample(gh, 25)
    linux = list(env.datasets['linux-v0'].benchmark_uris())
    linux = random.sample(linux, 25)
    blas = list(env.datasets['blas-v0'].benchmark_uris())
    blas = random.sample(blas, 25)
    # use cbench for testing
    csmith = list(islice(env.datasets['generator://csmith-v0'].benchmark_uris(), 25))
    train_benchmarks = cbench + gh + linux + blas
    test_benchmarks = csmith

def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    del args
    return RandomOrderBenchmarks(make_env(), train_benchmarks)

tune.register_env("compiler_gym", make_training_env)

if ray.is_initialized():
    ray.shutdown()
ray.init(include_dashboard=True, ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

# edit default config
config['num_workers'] = 20
# prob with using gpu and torch in rllib...
#config['num_gpus'] = 1
# this splits a rollout into an episode fragment of size n
config['rollout_fragment_length'] = 8
# this will combine fragements into a batch to perform sgd
config['train_batch_size'] = 160
# number of points to randomly select for GD
config['sgd_minibatch_size'] = 20
config['lr'] = 0.0001
config['gamma'] = 0.995
config['horizon'] = 100
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

def test(agent, n_episodes):
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    rewards = []
    for ep in range(n_episodes):
        with make_training_env() as env:

            # run until episode ends
            episode_reward = 0
            done = False
            obs = env.reset()
            while not done:
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                print(f"Action taken {action}, reward given {reward}")
                episode_reward += reward
                
          rewards.append(episode_reward)

     return rewards

# start training
if __name__ == "__main__":
    test_agent = load(agent_path)
    cumulative_rewards = test(test_agent, 20)
    print(cumulative_rewards)
    #save_dir = './log_dir'
    #agent_path,anaysis_obj = train({"episodes_total":200000}, save_dir)
    #test_agent = load(agent_path)
    #cumulative_reward = test(test_agent)
