import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from numpy import *
from absl import app, flags

# Add the ability to replay memory -> The memory should store transitions that the agent observes.
# By sammpling from this set of memories randomly, the transitions build up a decorrelated batch
# which stabilizes the training.


# Maps the state,action to the next_state,reward
# Acts sort of like a class -> can call T = Transition(some_state, some_action, ...)

# Flags

flags.DEFINE_float("gamma", 0.99, "The percent of how often the actor stays on policy.")
flags.DEFINE_float("epsilon", 1.0, "The starting value for epsilon.")
flags.DEFINE_float("epsilon_end", 0.01, "The ending value for epsilon.")
flags.DEFINE_float("epsilon_dec", 5e-5, "The decrement value for epsilon.")
flags.DEFINE_float("alpha", .05, "The learning rate.")
flags.DEFINE_integer("batch_size", 32, "The batch size.")
flags.DEFINE_integer("max_mem_size", 32, "The maximum memory size.")
flags.DEFINE_integer("replace", 500, "The number of iterations to run before replacing target network")
flags.DEFINE_integer("fc1_dim", 1024, "The dimension of the first fully connected layer")
flags.DEFINE_integer("fc2_dim", 1024, "The dimension of the second fully connected layer")
flags.DEFINE_integer("fc3_dim", 1024, "The dimension of the third fully connected layer")
flags.DEFINE_integer("episodes", 10000, "The number of episodes used to learn")
flags.DEFINE_integer("episode_length", 100, "The (MAX) number of transformation passes per episode")
flags.DEFINE_integer("stagnant_value", 10, "The (MAX) number of times to apply a series of transformations without observable change")


FLAGS = flags.FLAGS

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
		self.loss = nn.MSELoss()
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
	# gamma is the weighting of furture rewards
	# epsilon is the amount of time the agent explores environment???
	def __init__(self, input_dims, n_actions):
		super(Agent,self).__init__()
		self.replace = FLAGS.replace
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
		self.Q_eval = DQN(FLAGS.alpha, input_dims, fc1_dims=FLAGS.fc1_dim, fc2_dims=FLAGS.fc2_dim, fc3_dims=FLAGS.fc3_dim, n_actions = self.n_actions)
		self.Q_next = DQN(FLAGS.alpha, input_dims, fc1_dims=FLAGS.fc1_dim, fc2_dims=FLAGS.fc2_dim, fc3_dims=FLAGS.fc3_dim, n_actions = self.n_actions)
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
				actions[0][torch.argmax(actions).item()] = 0
			
			# the maximum action that hasen't been taken
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
		if self.mem_cntr < self.batch_size * 100:
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
		# gets the values from the actions taken
		q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
		q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
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

def train(agent, env):
    # try running using cbench dataset
    for i in range(1, FLAGS.episodes + 1):
	    #observation is the 56 dimensional static feature vector from autophase
	    observation = env.reset()
	    #maybe try setting done to true every time code size increases
	    done = False
	    total = 0
	    actions_taken = 0
	    # reset actions_taken vector for each new benchmark
	    agent.actions_taken = []
	    change_count = 0
	    while done == False and actions_taken < FLAGS.episode_length and change_count < FLAGS.stagnant_value:
	    	 #only apply finite number of actions to given program
	        action = agent.choose_action(observation)
	        #add to previous actions
	        new_observation, reward, done, info = env.step(action)

	        actions_taken += 1
	        #check total to allow for sequence of actions
	        total += reward
	        if reward == 0:
	            change_count += 1
	        else:
	            change_count = 0
	        #might be more useful to only store memory's of transitions where there was an effect(good or bad)
	        agent.store_transition(action, observation, reward, new_observation, done)
	        agent.learn()
	        observation = new_observation

def run(agent, env):
	observation = env.reset()
	action_seq, rewards = [], []
	done = False
	agent.actions_taken = []
	change_count = 0
	for i in range(FLAGS.episode_length):
	    action = agent.choose_action(observation)
	    action_seq.append(action)
	    observation, reward, done, info = env.step(action)
	    rewards.append(reward)
	    if reward == 0:
	        change_count += 1
	    else:
	        change_count = 0

	    if done == True or change_count > FLAGS.stagnant_value:
	    	break
	return sum(rewards)
