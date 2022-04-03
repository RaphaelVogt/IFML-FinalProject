import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

MEAN_ELEMS = 5

coins = []

with open("improvement_coins.txt", "r") as f:
    for elem in f:
        coins.append(int(elem))

coins = np.array(coins)

means = np.mean(coins.reshape(-1, MEAN_ELEMS), axis=1)
means[0] = 0

iterations = list(range(len(coins)))
iterations_mean = np.linspace(0, len(coins), len(means))

print(sum(coins)/len(coins))

plt.figure(figsize=(8, 5))
plt.plot(iterations_mean, means, c='g', label="Mean", lw=2)
plt.scatter(iterations, coins, c='r', s=10, label="Coin")
plt.xlabel("Episodes")
plt.ylabel("Coins")
plt.legend(loc="upper left", prop={'size': 15}, facecolor='white', framealpha=1)
plt.show()
