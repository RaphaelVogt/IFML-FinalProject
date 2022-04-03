import numpy as np
import pickle
import matplotlib.pyplot as plt


with open("avocadoman-model.pt", "rb") as file:
    reg = pickle.load(file)


feature_importance = np.zeros(9)
for estimator in reg.estimators_:
    feature_importance += estimator.feature_importances_

sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
print(feature_importance[sorted_idx[8]])
plt.figure(figsize=(8, 5))
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(["attack", "save_x", "save_y", "enemy_x", "enemy_y", "coin_x", "coin_y", "crate_x", "crate_y"])[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.show()