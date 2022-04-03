import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

MEAN_ELEMS = 5

score = []

with open("improvement_reward.txt", "r") as f:
    for elem in f:
        score.append(int(elem))

score = np.array(score)

means = np.mean(score.reshape(-1, MEAN_ELEMS), axis=1)
means[0] = 0

iterations = list(range(len(score)))
iterations_mean = np.linspace(0, len(score), len(means))

print(sum(score)/len(score))

plt.figure(figsize=(8, 5))
plt.plot(iterations_mean, means, c='g', label="Mean", lw=2)
plt.scatter(iterations, score, c='r', s=10, label="Reward")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend(loc="upper left", prop={'size': 15}, facecolor='white', framealpha=1)
plt.show()
