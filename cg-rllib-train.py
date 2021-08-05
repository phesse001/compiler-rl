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

# make sure random benchmarks are selected
with make_training_env() as env:
    # todo add observation wrapper
    '''
    env = ObservationWrapper(env)
    env.observation = custom_observation
    '''
    state = env.reset()
    print(state)
    print(env.benchmark)
    env.reset()
    print(env.benchmark)
    env.reset()
    print(env.benchmark)

tune.register_env("compiler_gym", make_training_env)

if ray.is_initialized():
    ray.shutdown()
ray.init(include_dashboard=True, ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

# edit default config
config['num_workers'] = 8
config['num_gpus'] = 1
config['rollout_fragment_length'] = 100
config['train_batch_size'] = 100
config['sgd_minibatch_size'] = 256
config['num_sgd_itr'] = 20
config['lr'] = 0.0001
config['gamma'] = 0.995
config['horizon'] = 40
config['framework'] = 'torch'
config['env'] = 'compiler_gym'

config['model']['fcnet_activation'] = 'relu'
config['model']['fcnet_hiddens'] = [1024, 1024, 1024]

analysis = tune.run(
    PPOTrainer,
    checkpoint_at_end=True,
    stop={
        "episodes_total": 100000,
    },
    config=config
)

# evaluation

agent = PPOTrainer(
    env = "compiler_gym",
    config={
        "num_workers" : 8,
        "explore": False
    }
)

checkpoint = analysis.get_best_checkpoint(
    metric="episode_reward_mean",
    mode="max",
    trial=analysis.trials[0]
)

agent.restore(checkpoint)

# Lets define a helper function to make it easy to evaluate the agent's 
# performance on a set of benchmarks.

def run_agent_on_benchmarks(benchmarks):
  """Run agent on a list of benchmarks and return a list of cumulative rewards."""
  with make_env() as env:
    rewards = []
    for i, benchmark in enumerate(benchmarks, start=1):
        observation, done = env.reset(benchmark=benchmark), False
        while not done:
            action = agent.compute_action(observation)
            observation, _, done, _ = env.step(action)
        rewards.append(env.episode_reward)
        print(f"[{i}/{len(benchmarks)}] {env.state}")

  return rewards

# Evaluate agent performance on the validation set.
test_rewards = run_agent_on_benchmarks(test_benchmarks)
