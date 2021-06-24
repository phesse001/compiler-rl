import compiler_gym
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from compiler_gym.wrappers import ConstrainedCommandline, TimeLimit, CycleOverBenchmarks
from itertools import islice

# code based off of example RLlib implementation of compiler_gym environment -> https://github.com/facebookresearch/CompilerGym/blob/development/examples/rllib.ipynb

# define a make_env() helper function to make environement.
# [optional] use the compiler_gym.wrappers API to implement custom contraints


def make_env() -> compiler_gym.envs.CompilerEnv:
    env = compiler_gym.make("llvm-v0", observation_space="Autophase", reward_space="IrInstructionCountOz")
    '''
    env = ConstrainedCommandline(env, flags=[
        "-break-crit-edges",
        "-early-cse-memssa",
        "-gvn-hoist",
        "-gvn",
        "-instcombine",
        "-instsimplify",
        "-jump-threading",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-versioning",
        "-mem2reg",
        "-newgvn",
        "-reg2mem",
        "-simplifycfg",
        "-sroa",
    ])
    env = TimeLimit(env, max_episode_steps=5)
    '''
    return env

# create benchmarks to be used

with make_env() as env:
    cbench = env.datasets["cbench-v1"]
    #anghabench = env.datasets["anghabench-v1"]
    #train_benchmarks = list(islice(anghabnech.benchmarks(), 1000))
    #train_benchmarks, test_benchmarks, val_benchmarks = train_benchmarks[:700], train_benchmarks[700:850], train_benchmarks[850:]
    train_benchmarks = list(cbench)
    test_benchmarks = list(cbench)

# register environment with rllib
def make_training_env(*args) -> compiler_gym.envs.CompilerEnv:
    del args # unused env_config passed by ray
    return CycleOverBenchmarks(make_env(), train_benchmarks)

tune.register_env("compiler_gym", make_training_env)
if ray.is_initialized():
    ray.shutdown()
ray.init(include_dashboard=False, ignore_reinit_error=True)

analysis = tune.run(
    PPOTrainer,
    checkpoint_at_end=True,
    stop={
        "episodes_total": 500,
    },
    config={
        "framework": "torch",
        "seed": None,
        "num_workers": 1,
        # Specify the environment to use, where "compiler_gym" is the name we 
        # passed to tune.register_env().
        "env": "compiler_gym",
        # Reduce the size of the batch/trajectory lengths to match our short 
        # training run.
        "rollout_fragment_length": 5,
        "train_batch_size": 5,
        "sgd_minibatch_size": 5,
    }
)