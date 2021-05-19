import gym
import compiler_gym
from dqn import Agent
import numpy as np
import matplotlib.pyplot as plt
import torch

# envs -> compiler_gym.COMPILER_GYM_ENVS

# Compiler = LLVM | Observation Type = Autophase | Reward Signal = IR Instruction count relative to -Oz

env = gym.make("llvm-ic-v0")
env.observation_space = "InstCount"

# Use existing dqn to make better decisions
agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 32,
            n_actions = env.action_space.n, eps_end = 0.05, input_dims = [70], alpha = 0.005)

# download existing dataset of programs/benchmarks
env.require_datasets(['cBench-v0', 'cBench-v1'])

action_space = env.action_space.names

# action space is described by env.action_space
# the observation space (autophase) is a 56 dimensional vector

# env.reset() must be called to initialize environment and create initial observation of said env
tmp = 0
# collect data for visualization
iterations = []
avg_total = []
for i in range(1,15001):
	#observation is the 56 dimensional static feature vector from autophase
    observation = env.reset()
    total = 0
    actions_taken = 0
    agent.actions_taken = []
    change_count = 0
    while done == False and actions_taken < 100 and change_count < 20:
        action = agent.choose_action(observation)
        new_observation, reward, done, info = env.step(action)
        actions_taken += 1
        #check total to allow for sequence of actions
        total += reward
        if reward == 0:
            change_count += 1
        else:
            change_count = 0
        agent.store_transition(action, observation, reward, new_observation, done)
        agent.learn()
        observation = new_observation
        print("Step: " + str(i) + " Episode Total: " + "{:.4f}".format(total) +
              " Epsilon: " + "{:.4f}".format(agent.epsilon) + " Action: " + str(action_space[action-1]))
    tmp += total
    print("Average Episode Reward: " + str(tmp/i))
    avg_total.append(tmp/i)
    iterations.append(i)
    
plt.scatter(iterations,avg_total)
plt.savefig("dqn_avg_tot.png")
PATH = './cg_dqn.pth'
torch.save(agent.Q_eval.state_dict(), PATH)
env.close()
