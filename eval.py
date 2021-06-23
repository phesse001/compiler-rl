#
"""Evaluate deep q network for leaderboard"""
from absl import app
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from dqn import rollout, train, Agent, train_and_run
import torch

def run(env: LlvmEnv) -> None:

    agent = Agent(n_actions = 15, input_dims = [69])
    env.observation_space = "InstCountNorm"
    agent.Q_eval.load_state_dict(torch.load("./H10-N4000-INSTCOUNTNORM.pth"))
    rollout(agent, env)

if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(run))
