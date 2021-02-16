### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)

	############################
	# YOUR IMPLEMENTATION HERE #

	#initizialize v0(s)=0 for all s
	#for k=1 until convergence:
	#for all s that exist in set of States:
	#vk+1(s)=sum of policy for state s given action a times(R(s,a)+gamma sum(probability of action a going from s to s' times vk(s')))
	
	#deterministic policy system, therefore there is only 1 given state per action

	while(True):
		v_next=np.zeros(nS)

		#if convergence >0.1:
		for s in range(nS):
			action=policy[s]
			for outcome in P[s][action]:
				prob_next_state,next_state,reward,done=outcome
				#print(f"*** Action: {action} Reward: {reward} Prob_Next_state {prob_next_state}, Next State {next_state}")

				v_next[s]+=prob_next_state*(reward+(gamma*value_function[next_state]))

		#print(f"vk: {value_function}\nvk+1: {v_next}")
		
		
		if(max(abs(v_next-value_function)))<tol:
    			break
		value_function=v_next		



	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')

	for s in range(nS):
		state_action_vals=np.zeros(nA)
		for a in range(nA):
			for outcome in P[s][a]:
				prob_next_state,next_state,reward,done=outcome
				state_action_vals[a]+=prob_next_state*(reward+(gamma*value_from_policy[next_state]))
		new_policy[s]=np.argmax(state_action_vals)


	############################

	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #

	while (True):
		value_function=policy_evaluation(P,nS,nA,policy)
		improved_policy=policy_improvement(P,nS,nA,value_function,policy)

		if(max(abs(improved_policy-policy))<tol):
			break
		
		policy=improved_policy	

	############################
	return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	
	while(True):
		v_next=np.zeros(nS)

		for s in range(nS):
			state_action_vals=np.zeros(nA)
			for action in range(nA):
				for outcome in P[s][action]:
					prob_next_state,next_state,reward,done=outcome
					#print(outcome)
					state_action_vals[action]+=prob_next_state*(reward+(gamma*value_function[next_state]))
				
			v_next[s]=max(state_action_vals)

		if(max(abs(v_next-value_function)))<tol:
			break
		value_function=v_next

	#print(value_function)
	new_policy = np.zeros(nS, dtype='int')

	for s in range(nS):
		state_action_vals=np.zeros(nA)
		for a in range(nA):
			for outcome in P[s][a]:
				prob_next_state,next_state,reward,done=outcome
				state_action_vals[a]+=prob_next_state*(reward+(gamma*value_function[next_state]))
		new_policy[s]=np.argmax(state_action_vals)


	############################
	return value_function, new_policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    #time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)
  return episode_reward


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	env_det = gym.make("Deterministic-4x4-FrozenLake-v0")
	env_stoch = gym.make("Stochastic-4x4-FrozenLake-v0")

	envs=[env_det,env_stoch]
	names={env_det:"det",env_stoch:"stoch"}

	vals={}
	for env in envs:
		policy_times=[]
		policy_success=0
		value_times=[]
		value_success=0
		for i in range(200):
			#print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
			time1=time.time()
			
			V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
			policy_success+=render_single(env, p_pi, 100)
			time2=time.time()
			#print("Total runtime: "+str(time2-time1))
			policy_times.append(time2-time1)
			#print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
			time1=time.time()
			V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
			value_success+=render_single(env, p_vi, 100)
			time2=time.time()
			#print("Total runtime: "+str(time2-time1))
			value_times.append(time2-time1)
		print(names[env])
		vals[names[env]]=(np.mean(policy_times),policy_success/200,np.mean(value_times),value_success/200)

	print("-----------------Deterministic-----------------")
	print("Policy Iteration Average Time to Complete: "+str(vals["det"][0]))
	print("Policy Iteration Success Rate: "+str(vals["det"][1]))
	print("Value Iteration Average Time to Complete: "+str(vals["det"][2]))
	print("Value Iteration Success Rate: "+str(vals["det"][3]))
	print("\n------------------Stochastic-------------------")
	print("Policy Iteration Average Time to Complete: "+str(vals["stoch"][0]))
	print("Policy Iteration Success Rate: "+str(vals["stoch"][1]))
	print("Value Iteration Average Time to Complete: "+str(vals["stoch"][2]))
	print("Value Iteration Success Rate: "+str(vals["stoch"][3]))
		# print(str(names[env])+" Average time Policy Iteration: "+str(np.mean(policy_times)))
		# print(str(names[env])+" Success Rate Policy Iteration:"+str(policy_success/50))
		# print(str(names[env])+" Average time Value Iteration: "+str(np.mean(value_times)))
		# print(str(names[env])+" Success Rate Value Iteration:"+str(value_success/50))

