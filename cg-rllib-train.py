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
from itertools import cycle

# code based off of example RLlib implementation of compiler_gym environment -> https://github.com/facebookresearch/CompilerGym/blob/development/examples/rllib.ipynb

# define a make_env() helper function to make environement.
# [optional] use the compiler_gym.wrappers API to implement custom contraints

def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make("llvm-v0", observation_space="Autophase", reward_space="IrInstructionCountOz")
    return env

# create benchmarks to be used
with make_env() as env:
    cbench = env.datasets['cbench-v1']
    gh = env.datasets['github-v0']
    train_benchmarks = list(cbench.benchmark_uris())
    test_benchmarks = list(cbench.benchmarks())

def custom_observation(observation):
    return 1

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
ray.init(include_dashboard=False, ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

# edit default config
config['num_workers'] = 39
config['rollout_fragment_length'] = 10
config['train_batch_size'] = 390
config['sgd_minibatch_size'] = 390
config['gamma'] = 0.90
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
        "num_workers" : 39,
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