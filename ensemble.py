from dqn import Agent
import gym
import compiler_gym
import numpy as np
import torch
from absl import flags
import math

agent1 = Agent(input_dims = [15], n_actions= 15)
agent2 = Agent(input_dims = [69], n_actions = 15)

'''load action history model'''
agent1.Q_eval.load_state_dict(torch.load("H10-N4000-ACTIONHISTORY-CBENCH.pth"))
'''load instcountnorm model'''
agent2.Q_eval.load_state_dict(torch.load("H10-N4000-INSTCOUNTNORM-CBENCH.pth"))

env = gym.make("llvm-ic-v0")

env.observation_space = "InstCountNorm"

actions = [
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
        "-sroa"
	]

global_observation = np.zeros(15)
observation = np.copy(global_observation)
env.reset(benchmark = "benchmark://cbench-v1/stringsearch")
total = 0

# choose 10 actions using action_history
for i in range(10):
	action = agent1.choose_action(observation)
	global_observation[action] += 1
	flag = actions[action]
	observation, reward, done, info = env.step(env.action_space.flags.index(flag))
	total += reward
	if done:
		break
	observation = global_observation/np.linalg.norm(global_observation)
	print("Action = " + actions[action] + ", reward = " + str(reward))

print("Total reward = " + str(total))

##########################################################################################

observation = env.reset(benchmark = "benchmark://cbench-v1/stringsearch")
total = 0

# choose 10 actions using instcount
for i in range(10):
	action = agent2.choose_action(observation)
	flag = actions[action]
	observation, reward, done, infor = env.step(env.action_space.flags.index(flag))
	total += reward
	if done:
		break
	print("Action = " + actions[action] + ", reward = " + str(reward))

print("Total reward = " + str(total))

###########################################################################################

observation = env.reset(benchmark = "benchmark://cbench-v1/stringsearch")
global_observation = np.zeros(15)
observation1 = np.copy(global_observation)
total = 0
actions_taken = []

# choose 10 actions using instcount
for i in range(10):
	action_set1 = agent1.get_actions(observation1)
	action_set2 = agent2.get_actions(observation)

	action_ensemble = (action_set1 + action_set2)/2

	while torch.argmax(action_ensemble).item() in actions_taken:
		action_ensemble[0][torch.argmax(action_ensemble).item()] = 0.0

	action = torch.argmax(action_ensemble).item()
	actions_taken.append(action)

	global_observation[action] += 1

	flag = actions[action]

	observation, reward, done, infor = env.step(env.action_space.flags.index(flag))

	observation1 = global_observation/np.linalg.norm(global_observation)

	total += reward
	if done:
		break
	print("Action = " + actions[action] + ", reward = " + str(reward))

print("Total reward = " + str(total))

env.close()