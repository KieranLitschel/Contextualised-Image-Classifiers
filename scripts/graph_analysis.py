from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

experiment_maps = {"Original": [0.7481407242026616, 0.748721526, 0.749476923],
                   "Fine-Tuned": [0.733641353296024, 0.730644964750732, 0.729181126395337],
                   "Most Likely Class": [0.635850179],
                   "Original Floored": [0.647748983982311, 0.645886573764493, 0.646263863153314],
                   "Machine-Generated Labels": [0.878656405867957],
                   "English Original": [0.746953834, 0.743078918, 0.745654112],
                   "English Fine-Tuned": [0.728503208, 0.729714927, 0.72694073],
                   "Other Original": [0.77260902, 0.771070024, 0.773496219],
                   "Other Fine-Tuned": [0.763937961, 0.760464632, 0.764313694]}

graphs = [("Original", "Most Likely Class"), ("Original Floored", "Machine-Generated Labels"),
          ("English Original", "English Fine-Tuned"), ("Other Original", "Other Fine-Tuned")]

for pair in graphs:
    results = [[name, np.mean(maps), np.std(maps)] for name, maps in
               {name: experiment_maps[name] for name in pair}.items()]
    df = pd.DataFrame(results, columns=["Experiment", "Mean mAP", "Standard Deviation mAP"])

    plt.margins(x=0.4)
    plt.xlabel("Experiment")
    plt.ylabel("Mean mAP")
    plt.errorbar(df["Experiment"], df["Mean mAP"], yerr=df["Standard Deviation mAP"], fmt='.')
    plt.savefig("Experiment Analysis point plot - {} vs {}.pdf".format(pair[0], pair[1]))
    plt.clf()
