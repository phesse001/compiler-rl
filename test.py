from dqn import train, Agent
import gym
import compiler_gym
from absl import app
import torch


if __name__ == "__main__":
    env = gym.make("llvm-ic-v0")
    env.observation_space = "InstCountNorm"
    agent = Agent(input_dims = [69], n_actions = env.action_space.n)
    app.run(train(agent, env))
    PATH = './dqn.pth'
    torch.save(agent.Q_eval.state_dict(), PATH)
