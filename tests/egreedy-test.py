#%%
from modrl.bandits import EpsilonGreedy
import numpy as np

N_ARMS = 3
EPSILON = 0.1
SEED = 123

bandit = EpsilonGreedy(n_arms=N_ARMS, epsilon=EPSILON, seed=SEED)

bandit.train(data=np.array([[1,0,0], [1,0,0], [1,0,0], [1,0,0]]))

print("Q values:", bandit.q.flatten())
print("N values:", bandit.n.flatten())
print("Actions experienced:", bandit.actions)
print("Rewards experienced:", bandit.rewards)
print("Optimal rewards:", bandit.optimal_rewards)
print("Epsilon:", bandit.epsilon)
print("Number of arms:", bandit.n_arms)
print("Random seed:", SEED)
# %%
