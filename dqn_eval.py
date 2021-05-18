#
"""Evaluate deep q network for leaderboard"""
from absl import app
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from dqn import train,run,Agent

def train_and_run(env: LlvmEnv) -> None:
    agent = Agent(n_actions = env.action_space.n, input_dims = [56])
    env.observation_space = "Autophase"
    training_env = env.fork()
    train(agent, training_env)
    training_env.close()
    run(agent, env)

if __name__ == "__main__":
    app.run(eval_llvm_instcount_policy(train_and_run))
