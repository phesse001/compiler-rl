import gym
import compiler_gym
from dqn import Agent
import numpy as np
import torch

def train(agent, env):
    # try running using cbench dataset
    env.require_datasets(['cBench-v0'])
    for i in range(1,10001):
	    #observation is the 56 dimensional static feature vector from autophase
	    observation = env.reset()
	    #maybe try setting done to true every time code size increases
	    done = False
	    total = 0
	    actions_taken = 0
	    agent.actions_taken = []
	    change_count = 0
	    while done == False and actions_taken < 100 and change_count < 10:
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
	for i in range(1,101):
	    #observation is the 56 dimensional static feature vector from autophase
	    observation = env.reset()
	    #maybe try setting done to true every time code size increases
	    done = False
	    total = 0
	    actions_taken = 0
	    agent.actions_taken = []
	    change_count = 0
	    while done == False and actions_taken < 100 and change_count < 10:
	    	 #only apply finite number of actions to given program
	        action = agent.choose_action(observation)
	        #add to previous actions
	        new_observation, reward, done, info = env.step(action)

	        print("Step : " + str(i) + " Episode Reward: " + str(total))

	        actions_taken += 1
	        #check total to allow for sequence of actions
	        total += reward
	        if reward == 0:
	            change_count += 1
	        else:
	            change_count = 0

	        observation = new_observation
