import gym
import compiler_gym
from dqn import Agent
import numpy as np
import matplotlib.pyplot as plt
import torch

def normalize(obv):
    mean = np.mean(obv)
    std = np.std(obv)
    return (obv - mean)/std


def equal(a1, a2):
    diff = []
    for i in range(len(a1)):
        if(a1[i] != a2[i]):
            diff.append(i)

    return diff

observations = np.zeros(280)

def save_observation(observation):
    n = 70
    observations[3*n:4*n] = observations[2*n:3*n]
    observations[2*n:3*n] = observations[n:2*n]
    observations[n:2*n] = observations[0:n]
    observations[0:n] = observation


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
env.observation_space = "InstCount"
benchmarks = [b for b in env.benchmarks if "cBench" in b] # get all cbench benchmarks
benchmarks = np.asarray(benchmarks)

agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 32,
            n_actions = env.action_space.n, eps_end = 0.1, input_dims = [280], alpha = 0.005)

# download existing dataset of programs/benchmarks

action_space = env.action_space.names

tmp = 0
# collect data for visualization
iterations = []
avg_total = []
for i in range(1,15001):
	#observation is the 56 dimensional static feature vector from autophase
    observation = env.reset(benchmark = benchmarks[np.random.choice(len(benchmarks))])
    print(env.benchmark)
    # Let's normalize the input vector...
    observation = normalize(observation)

    save_observation(observation)

    done = False
    total = 0
    actions_taken = 0
    agent.actions_taken = []
    change_count = 0

    while done == False and actions_taken < env.action_space.n and change_count < 20:

        action = agent.choose_action(observations)
        new_observation, reward, done, info = env.step(action)
        new_observation = normalize(new_observation)
        save_observation(new_observation)

        # save old observation to be used for store_transition
        old_observations = observations

        actions_taken += 1

        #check total to allow for sequence of actions
        total += reward
        if reward == 0:
            change_count += 1
        else:
            change_count = 0

        # TODO - fix so new observation is saved: right now it is just a 70d vector but should be 280
        agent.store_transition(action, old_observations, reward, observations, done)
        agent.learn()

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
