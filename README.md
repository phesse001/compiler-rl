# compiler-rl
The idea is to make sequential decisions about what compiler transformation should be applied to a codes IR to optimize some objective function

## DQN

### Further Info
The bells and whistles added to this dqn are a target network and the use of experience replay.

The `choose_action` method in `dqn.py` also sets all q-values of actions that have already been selected to 0.0.
Otherwise, since InstCount is the observation, if a pass is applied that does not effect the program state, and
`epsilon == epsilon_end` (i.e. we are acting greedily), then it will just repeatedly apply the same pass. There is 
definitely a better solution, but this was the 'hack' I came up with. Ensembling with a model trained on action
history or concatenating action history could maybe fix this, but I had no luck.

To evalulate a model, just change the hardcoded path of `agent.Q_eval.load_state_dict(torch.load(<hard-coded-path-to-model.pth>))`

## PPO
The PPO implementation uses ray to orchestrate a cluster of on-premise nodes and utilizes rllib to run distributed rollouts which are then gathered to the head node for learning.

### Further Info

