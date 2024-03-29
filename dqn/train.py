from dqn import train, Agent
import gym
import compiler_gym
from absl import app

# try making wrapper to use done logic
class compilerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.patience = 5
        self.reward_counter = 0

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if reward <= 0:
            self.reward_counter += 1
        else:
            self.reward_counter = 0

        if self.reward_counter > self.patience:
            done = True
        return next_state, reward, done, info

if __name__ == "__main__":
    env = compilerWrapper(gym.make("llvm-ic-v0"))
    agent = Agent(input_dims = [56], n_actions = 15)
    train(agent, env)
    
