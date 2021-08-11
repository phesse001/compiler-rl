import compiler_gym
import gym
from gym import spaces
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import Logger
from ray.rllib.models import ModelCatalog
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit, CycleOverBenchmarks, RandomOrderBenchmarks, ObservationWrapper
import numpy as np
import random
import itertools
from itertools import cycle, islice 

# code based off of example RLlib implementation of compiler_gym environment -> https://github.com/facebookresearch/CompilerGym/blob/development/examples/rllib.ipynb

# define a make_env() helper function to make environement.
# [optional] use the compiler_gym.wrappers API to implement custom contraints

def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make("llvm-v0", observation_space="Autophase", reward_space="IrInstructionCountOz")
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
config['num_workers'] = 8
config['num_gpus'] = 1
# this splits a rollout into an episode fragment of size 10
config['rollout_fragment_length'] = 40
# this will combine fragements into a batch to perform sgd
config['train_batch_size'] = 320
# number of points to randomly select for GD
config['sgd_minibatch_size'] = 10
config['lr'] = 0.0001
config['gamma'] = 0.995
config['horizon'] = 40
config['framework'] = 'torch'
config['env'] = 'compiler_gym'

config['model']['fcnet_activation'] = 'relu'
config['model']['fcnet_hiddens'] = [1024, 1024, 1024]


