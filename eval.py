#
"""Evaluate deep q network for leaderboard"""
from absl import app
import gym
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from dqn import rollout,Agent
import torch

# load the model 

env = gym.make("llvm-ic-v0")

env.observation_space = "InstCountNorm"

def run(env: LlvmEnv) -> None:
    
    agent = Agent(n_actions = env.action_space.n, input_dims = [69])

    agent.Q_eval.load_state_dict(torch.load("./dqn.pth"))

    env.observation_space = "InstCountNorm"
    rollout(agent, env)

if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
