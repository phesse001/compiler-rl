from dqn import train, Agent
import gym
import compiler_gym
from absl import app
import torch

if __name__ == "__main__":
    env = gym.make("llvm-ic-v0")
    agent = Agent(input_dims = [69], n_actions = 15)
    train(agent, env)
    PATH = './opencv-mibench_dqn.pth'
    torch.save(agent.Q_eval.state_dict(), PATH)
