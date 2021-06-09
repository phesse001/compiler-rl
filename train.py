from dqn import train, Agent
import gym
import compiler_gym
from absl import app
import torch

if __name__ == "__main__":
    env = gym.make("llvm-ic-v0")
<<<<<<< HEAD
    agent = Agent(input_dims = [69], n_actions = env.action_space.n)
||||||| merged common ancestors
    agent = Agent(input_dims = [69], n_actions = 15)
=======
    agent = Agent(input_dims = [15], n_actions = 15)
>>>>>>> c4f89cdb1fb857e0baef28f6a240ea72e489d225
    train(agent, env)
    PATH = './H15-N5000-ACTIONHISTORY-cbench.pth'
    torch.save(agent.Q_eval.state_dict(), PATH)
