from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

experiment_maps = {"Basic": [0.7481407242026616, 0.748721526, 0.749476923],
                   "Pre-Trained": [0.716062727546937, 0.713773825744106, 0.712525594821692],
                   "Fine-Tuned": [0.733641353296024, 0.730644964750732, 0.729181126395337]}

results = [[name, np.mean(maps), np.std(maps)] for name, maps in experiment_maps.items()]
df = pd.DataFrame(results, columns=["Experiment", "Mean mAP", "Standard Deviation mAP"])

plt.margins(x=0.2)
plt.xlabel("Experiment")
plt.ylabel("Mean mAP")
plt.errorbar(df["Experiment"], df["Mean mAP"], yerr=df["Standard Deviation mAP"], fmt='.')
plt.savefig("Experiment 2 point plot - Basic vs Pre-Trained vs Fine-Tuned.pdf")
