import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from absl import flags
import sys
import matplotlib.pyplot as plt
import random

# Start implementing ideas from Deep RL Bootcamp series on youtube

'''
- concatentate observation with previous observations (they used 4)
- use huber loss (same from [-1,1], but penalizes less for larger errors)
- use RMSProp instead of grad descent
- add more exploration at the beginning
- could try prioritized experience replay again...
- could also try dueling dqn to see the effect of the advantage property

'''


# Flags

flags.DEFINE_float("gamma", 0.90, "The percent of how often the actor stays on policy.")
flags.DEFINE_float("epsilon", 1.0, "The starting value for epsilon.")
flags.DEFINE_float("epsilon_end", 0.05, "The ending value for epsilon.")
flags.DEFINE_float("epsilon_dec", 5e-5, "The decrement value for epsilon.")
flags.DEFINE_float("alpha", 0.001, "The learning rate.")
flags.DEFINE_integer("batch_size", 32, "The batch size.")
flags.DEFINE_integer("max_mem_size", 100000, "The maximum memory size.")
flags.DEFINE_integer("replace", 500, "The number of iterations to run before replacing target network")
flags.DEFINE_integer("fc_dim", 512, "The dimension of a fully connected layer")
flags.DEFINE_integer("episodes", 10000, "The number of episodes used to learn")
flags.DEFINE_integer("episode_length", 12, "The (MAX) number of transformation passes per episode")
flags.DEFINE_integer("patience", 5, "The (MAX) number of times to apply a series of transformations without observable change")
flags.DEFINE_integer("learn", 32, "The number of fully exploratory episodes to run before starting learning")
flags.DEFINE_list(
    "actions",
    [
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
        "-sroa",
    ],
    "A list of action names to explore from.",
)


FLAGS = flags.FLAGS
FLAGS(sys.argv)


# The Network

class DQN(nn.Module):
	def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
		super(DQN,self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.fc3_dims = fc3_dims
		self.n_actions = n_actions
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
		self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr = ALPHA)
		self.loss = nn.SmoothL1Loss() # try huber loss
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)


	def forward(self, state):
		# first fully connected layer takes the state in as input, pass that output as input to activation function
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		actions = self.fc4(x)

		return actions

class Agent(nn.Module):
	def __init__(self, input_dims, n_actions):
		super(Agent,self).__init__()
		self.eps_dec = FLAGS.epsilon_dec
		self.eps_end = FLAGS.epsilon_end
		self.max_mem_size = FLAGS.max_mem_size
		self.replace_target_cnt = FLAGS.replace
		self.gamma = FLAGS.gamma
		self.epsilon = FLAGS.epsilon
		self.eps_end = FLAGS.epsilon_end
		self.eps_dec = FLAGS.epsilon_dec
		self.n_actions = n_actions
		self.action_space = [i for i in range(n_actions)]
		self.max_mem_size = FLAGS.max_mem_size
		self.batch_size = FLAGS.batch_size
		# keep track of position of first available memory
		self.mem_cntr = 0
		self.Q_eval = DQN(FLAGS.alpha, input_dims, fc1_dims=FLAGS.fc_dim, fc2_dims=FLAGS.fc_dim, fc3_dims=FLAGS.fc_dim,
						  n_actions=self.n_actions)
		self.Q_next = DQN(FLAGS.alpha, input_dims, fc1_dims=FLAGS.fc_dim, fc2_dims=FLAGS.fc_dim, fc3_dims=FLAGS.fc_dim,
						  n_actions=self.n_actions)
		self.actions_taken = []
		# star unpacks list into positional arguments
		self.state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.new_state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
		self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
		self.terminal_mem = np.zeros(self.max_mem_size, dtype=np.bool)
		self.learn_step_counter = 0
		self.to(self.Q_eval.device)

	def store_transition(self, action, state, reward, new_state, done):
		# what is the position of the first unoccupied memory
		index = self.mem_cntr % self.max_mem_size
		self.state_mem[index] = state
		self.new_state_mem[index] = new_state
		self.action_mem[index] = action
		self.reward_mem[index] = reward
		self.terminal_mem[index] = done

		self.mem_cntr += 1

	def choose_action(self, observation):
		if np.random.random() > self.epsilon:
			# sends observation as tensor to device
			# convert to float - > compiler gyms autophase vector is a long
			observation = observation.astype(np.float32)
			state = torch.tensor([observation]).to(self.Q_eval.device)
			actions = self.Q_eval.forward(state)
			# network seems to choose same action over and over, even with zero reward,
			# trying giving negative reward for choosing same action multiple times
			while torch.argmax(actions).item() in self.actions_taken:
				actions[0][torch.argmax(actions).item()] = 0.0
				
			action = torch.argmax(actions).item()

			self.actions_taken.append(action)

		else:
			# take random action
			action = np.random.choice(self.action_space)

		return action

	def replace_target_network(self):
		if self.learn_step_counter % self.replace_target_cnt == 0:
			self.Q_next.load_state_dict(self.Q_eval.state_dict())


	def learn(self):
		# start learning as soon as batch size of memory is filled
		if self.mem_cntr < FLAGS.learn:
			return
		# set gradients to zero
		self.Q_eval.optimizer.zero_grad()
		self.replace_target_network()
		# select subset of memorys
		max_mem = min(self.mem_cntr, self.max_mem_size)
		# take a selection of the size of the batch size from the current pool of memory's
		# pool of memory's will be full by the time we get here
		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		# have to calculate scalar of importance so that we don't update network in a biased way
		batch_index = np.arange(self.batch_size, dtype = np.int32)
		# sending a batch of states to device
		state_batch = torch.tensor(self.state_mem[batch]).to(self.Q_eval.device)
		new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
		reward_batch = torch.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
		terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)
		action_batch = self.action_mem[batch]
		'''
		calling forward with a batch of states gives us a batch of Q-values.
		The batch_index just selects each group of Q-values and action_batch
		selects the action we took in each group of Q-values.
		We use batch_index here instead of batch because order no longer
		matters after passing through the network. Ex.) a batch of [22,74,3,43]
		would select those respective states from the state memory, and pass them through
		the network, but after they are passed though we are indexing based on the size of
		the batch, not the replay buffer.
		'''
		q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
		q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
		# if and index of the batch is done (True), then set next reward to 0
		q_next[terminal_batch] = 0.0
		q_target = reward_batch + self.gamma * q_next
		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()
		self.learn_step_counter += 1

		if self.epsilon > self.eps_end:
		    self.epsilon -= self.eps_dec
		else:
			self.epsilon = self.eps_end

def save_observation(observation, observations):
    n = 69
    tmp = np.copy(observations)
    tmp[3*n:4*n] = tmp[2*n:3*n]
    tmp[2*n:3*n] = tmp[n:2*n]
    tmp[n:2*n] = tmp[0:n]
    tmp[0:n] = observation

    return tmp

def train(agent, env):
    action_space = env.action_space.names
    env.observation_space = "InstCountNorm"
	# opencv (right), mibench(down)
    #opencv = random.sample(list(env.datasets["benchmark://opencv-v0"].benchmarks()) , 25)
    #mibench = random.sample(list(env.datasets["benchmark://mibench-v0"].benchmarks()), 25)
    #train_benchmarks = np.concatenate([opencv, mibench])
    train_benchmarks = list(env.datasets["benchmark://cbench-v1"].benchmarks())
    history_size = 100
    mem_cntr = 0
    history = np.zeros(history_size)

    for i in range(1, FLAGS.episodes + 1):
	    observation = env.reset(benchmark = train_benchmarks[np.random.choice(len(train_benchmarks))])
	    print(env.benchmark)
	    done = False
	    total = 0
	    actions_taken = 0
	    agent.actions_taken = []
	    change_count = 0
	    while done == False and actions_taken < FLAGS.episode_length and change_count < FLAGS.patience:
	        action = agent.choose_action(observation)
	        flag = FLAGS.actions[action]
	        # translate to global action number via global index of flag
	        new_observation, reward, done, info = env.step(env.action_space.flags.index(flag))
	        actions_taken += 1
	        total += reward

	        if reward == 0:
	            change_count += 1
	        else:
	            change_count = 0

	        agent.store_transition(action, observation, reward, new_observation, done)
	        agent.learn()
	        observation = new_observation

	        print("Step: " + str(i) + " Episode Total: " + "{:.4f}".format(total) +
	              " Epsilon: " + "{:.4f}".format(agent.epsilon) + " Action: " + flag)
	    index = mem_cntr % history_size
	    history[index] = total
	    mem_cntr +=1

	    print("Average sum of rewards is " + str(np.mean(history)))

def rollout(agent, env):
	observation = env.reset()
	action_seq, rewards = [], []
	done = False
	agent.actions_taken = []
	change_count = 0
	for i in range(FLAGS.episode_length):
	    action = agent.choose_action(observation)
	    flag = FLAGS.actions[action]
	    action_seq.append(action)
	    observation, reward, done, info = env.step(env.action_space.flags.index(flag))
	    rewards.append(reward)
	    if reward == 0:
	        change_count += 1
	    else:
	        change_count = 0

	    if done == True or change_count > FLAGS.patience:
	    	break

	return sum(rewards)
