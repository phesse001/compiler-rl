import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from numpy import *

# Add the ability to replay memory -> The memory should store transitions that the agent observes.
# By sammpling from this set of memories randomly, the transitions build up a decorrelated batch
# which stabilizes the training.


# Maps the state,action to the next_state,reward
# Acts sort of like a class -> can call T = Transition(some_state, some_action, ...)

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

class Agent():
	# gamma is the weighting of furture rewards
	# epsilon is the amount of time the agent explores environment???
	def __init__(self, gamma, epsilon, alpha, input_dims, batch_size,
	         n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-5, replace = 1000):
		self.replace_target_cnt = replace
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_end = eps_end
		self.eps_dec = eps_dec
		self.alpha = alpha
		self.n_actions = n_actions
		self.action_space = [i for i in range(n_actions)]
		self.max_mem_size = max_mem_size
		self.batch_size = batch_size
		# keep track of position of first available memory
		self.mem_cntr = 0
		self.Q_eval = DQN(alpha, input_dims, fc1_dims=1024, fc2_dims=1024, fc3_dims=1024, n_actions = self.n_actions)
		self.Q_next = DQN(alpha, input_dims, fc1_dims=1024, fc2_dims=1024, fc3_dims=1024, n_actions = self.n_actions)
		self.actions_taken = []
		# star unpacks list into positional arguments
		self.state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.new_state_mem = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
		self.action_mem = np.zeros(self.max_mem_size, dtype=np.int32)
		self.reward_mem = np.zeros(self.max_mem_size, dtype=np.float32)
		self.terminal_mem = np.zeros(self.max_mem_size, dtype=np.bool)
		self.priority_mem = np.ones(self.max_mem_size,dtype=np.int32)
		self.learn_step_counter = 0

	def store_transition(self, action, state, reward, new_state, done):
		# what is the position of the first unoccupied memory
		index = self.mem_cntr % self.max_mem_size
		self.state_mem[index] = state
		self.new_state_mem[index] = new_state
		self.action_mem[index] = action
		self.reward_mem[index] = reward
		self.terminal_mem[index] = done
		self.priority_mem[index] = max(self.priority_mem, default=1)

		self.mem_cntr += 1

	def get_probabilities(self, priority_scale):
		scaled_probabilities = self.priority_mem ** priority_scale
		sample_probabilities = scaled_probabilities/sum(scaled_probabilities)
		where_are_NaNs = isnan(sample_probabilities)
		sample_probabilities[where_are_NaNs] = 0
		return sample_probabilities

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

	def get_importance(self,probabilites):
		importance = 1/self.mem_cntr * 1/probabilites
		importance_normalized = importance / max(importance)
		return importance_normalized

	def set_priorties(self,indicies, errors, offset=0.1):
		for i,e in zip(indicies,errors):
			self.priority_mem[i] = abs(e) + offset

	def scale_probs(self, probs):
		total = sum(probs)
		for i in range(len(probs)):
			probs[i] = probs[i]/total
		return probs

	def learn(self, priority_scale):
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
		# use probability distribution calculated from get_probabilities()
		batch_probs = self.get_probabilities(priority_scale)
		# not sure if this breaks some mathematical rule, but trying scaling chosen probabilites
		# for ex, [.1,.2,.1,.1] -> [.2,.4,.2,.2]
		scaled_probs = self.scale_probs(batch_probs[:max_mem])
		batch = np.random.choice(max_mem, self.batch_size, replace=False, p=scaled_probs)
		# have to calculate scalar of importance so that we don't update network in a biased way
		importance = self.get_importance(batch_probs) ** (1-self.epsilon)
		batch_index = np.arange(self.batch_size, dtype = np.int32)
		# sending a batch of states to device
		state_batch = torch.tensor(self.state_mem[batch]).to(self.Q_eval.device)
		new_state_batch = torch.tensor(self.new_state_mem[batch]).to(self.Q_eval.device)
		reward_batch = torch.tensor(self.reward_mem[batch]).to(self.Q_eval.device)
		terminal_batch = torch.tensor(self.terminal_mem[batch]).to(self.Q_eval.device)
		action_batch = self.action_mem[batch]
		importance_batch = torch.tensor(importance[batch]).to(self.Q_eval.device)
		# gets the values from the actions taken
		q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
		q_next = self.Q_next.forward(new_state_batch).max(dim=1)[0]
		q_next[terminal_batch] = 0.0
		q_target = reward_batch + self.gamma * q_next
		error = q_eval - q_target
		self.set_priorties(batch, error)
		q_target *= importance_batch
		q_eval *= importance_batch
		loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
		loss.backward()
		self.Q_eval.optimizer.step()
		self.learn_step_counter += 1

		if self.epsilon > self.eps_end:
		    self.epsilon -= self.eps_dec
		else:
			self.epsilon = self.eps_end
