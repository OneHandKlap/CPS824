import numpy as np
episode=[(0,1,1),(4,2,-1),(6,3,0),(8,0,2)]
len_episode=len(episode)


gammas=[.9**i for i in range(0,len_episode)]
rewards=[episode[i][2] for i in range(len_episode)]
returns=[np.dot(gammas[:len(gammas)-i],rewards[i:]) for i in range(len_episode)]
print(rewards)
print(gammas)
print(returns)