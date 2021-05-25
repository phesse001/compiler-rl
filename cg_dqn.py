import gym
import compiler_gym
from dqn import Agent
import numpy as np
import matplotlib.pyplot as plt
import torch

def save_observation(observation, observations):
    n = 69
    tmp = np.copy(observations)
    tmp[3*n:4*n] = tmp[2*n:3*n]
    tmp[2*n:3*n] = tmp[n:2*n]
    tmp[n:2*n] = tmp[0:n]
    tmp[0:n] = observation

    return tmp


# Start implementing ideas from Deep RL Bootcamp series on youtube

'''

- add normalization
- concatentate observation with previous observations (they used 4)
- use huber loss (same from [-1,1], but penalizes less for larger errors)
- use RMSProp instead of grad descent
- add more exploration at the beginning
- could try prioritized experience replay again...
- could also try dueling dqn to see the effect of the advantage property

'''

env = gym.make("llvm-ic-v0")
env.observation_space = "InstCountNorm"

for d in env.datasets:
    if d.name != "benchmark://cbench-v1":
        del env.datasets[d.name]

dataset = []

for benchmark in env.datasets.benchmarks():
    dataset.append(benchmark)


agent = Agent(gamma = 0.95, epsilon = 1.0, batch_size = 32,
            n_actions = env.action_space.n, eps_end = 0.05, input_dims = [276], alpha = 0.005)

action_space = env.action_space.names

total = 0
iterations = []
avg_total = []

for i in range(1,100001):
    observation = env.reset(benchmark = dataset[np.random.choice(len(dataset))])
    print(env.benchmark)

    observations = np.zeros(276)
    observations = save_observation(observation, observations)

    done = False
    episode_total = 0
    actions_taken = 0
    agent.actions_taken = []
    change_count = 0

    while done == False and actions_taken < env.action_space.n and change_count < 10:
        action = agent.choose_action(observations)
        new_observation, reward, done, info = env.step(action)
        new_observations = save_observation(new_observation, observations)

        actions_taken += 1

        #check total to allow for sequence of actions
        episode_total += reward
        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        agent.store_transition(action, observations, reward, new_observations, done)
        agent.learn()
        observations = new_observations
        print("Step: " + str(i) + " Episode Total: " + "{:.4f}".format(episode_total) +
              " Epsilon: " + "{:.4f}".format(agent.epsilon) + " Action: " + str(action_space[action]))
    
    total += episode_total
    print("Average Episode Reward: " + str(total/i))
    avg_total.append(total/i)
    iterations.append(i)
    
plt.scatter(iterations,avg_total)
plt.savefig("motion1_avg_tot.png")
PATH = './motion1_dqn.pth'
torch.save(agent.Q_eval.state_dict(), PATH)
env.close()
