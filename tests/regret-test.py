#%%
from modrl.bandits import EpsilonGreedy
from modrl.evaluation import cumulative_regret
import numpy as np

N_ARMS = 3
EPSILON = 0.1
SEED = 123

bandit = EpsilonGreedy(n_arms=N_ARMS, epsilon=EPSILON, seed=SEED)

bandit.train(data=np.array([[1,0,0], [1,0,0], [1,0,0], [1,0,0]]))

regret_score = cumulative_regret(bandit.rewards, bandit.optimal_rewards)

print("Actions:", bandit.actions)
print("Rewards:", bandit.rewards)
print("Optimal Rewards:", bandit.optimal_rewards)
print("Regret score:", regret_score)