import torch
from dqn import Agent
import gym
import compiler_gym
import numpy as np
import matplotlib.pyplot as plt

PATH = "../dqn-results/cg_dqn_llvm_InstCount.pth"

env = gym.make("llvm-ic-v0")
env.observation_space = "InstCount"
env.require_datasets(['cBench-v1'])

agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 32,
            n_actions = env.action_space.n, eps_end = 0.05, input_dims = [70], alpha = 0.005)

the_model = agent.Q_eval
the_model.load_state_dict(torch.load(PATH))
the_model.eval()

agent.epsilon = agent.eps_end
print("Agents epsilon is " + str(agent.epsilon))

tmp = 0
iterations = []
avg_total = []
tot = []
for i in range(1,501):
	#observation is the 56 dimensional static feature vector from autophase
    observation = env.reset()
    done = False
    total = 0
    actions_taken = 0
    agent.actions_taken = []
    # collect data for visualization
    change_count = 0
    while done == False and actions_taken < 100 and change_count < 10:
        #only apply finite number of actions to given program
        action = agent.choose_action(observation)
        print("action taken was " + str(action))

        new_observation, reward, done, info = env.step(action)
        actions_taken += 1
        #check total to allow for sequence of actions
        total += reward
        if reward == 0:
            change_count += 1
        else:
            change_count = 0
        #might be more useful to only store memory's of transitions where there was an effect(good or bad)
        observation = new_observation
        print("Step " + str(i) + " Cumulative Total " + str(total) +
              " Epsilon " + str(agent.epsilon) + " Action " + str(action) + 
              " No Effect " + str(info))
    tmp += total
    print("avg is " + str(tmp/i))
    avg_total.append(tmp/i)
    iterations.append(i)
    tot.append(total)

plt.figure(0)
plt.scatter(iterations,avg_total)
plt.savefig("dqn_avg_tot.png")
plt.figure(1)
plt.ylim(-5, 5)
plt.scatter(iterations, tot)
plt.savefig("dqn_test_iterations.png")
env.close()



