import pickle

import numpy as np
from matplotlib import pyplot as plt

# from yfcc100m.dataset import count_detected_languages_cld2

dataset_path = "C:\\Honors Project\\YFCC100M\\dataset\\yfcc\\train.tsv"

# lang_counts = count_detected_languages_cld2(dataset_path, keep_numbers=False)

# with open('cld2_lang_counts.pickle', 'wb') as handle:
#    pickle.dump(lang_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(
        'C:\\Users\\kiera\\OneDrive - University of Edinburgh\\Every Day Files\\Documents\\Microsoft Office\\Word\\Homework\\University\\Year 4\\Honours Project\\Contextualised-CNN\\cld2_lang_counts.pickle',
        'rb') as handle:
    lang_counts = pickle.load(handle)

english_labels = ["English", "Other"]

english_values = [lang_counts["english"], sum(lang_counts.values()) - lang_counts["english"] - lang_counts["unknown"]]

top_ten_other_langs = [lang for lang, count in sorted(lang_counts.items(), reverse=True, key=lambda t: t[1])[2:13]]
top_ten_other_values = [lang_counts[lang] for lang in top_ten_other_langs]
top_ten_other_langs = [lang.title() if lang != "chineset" else "Chinese" for lang in top_ten_other_langs]
top_ten_other_langs.append("Other {} languages".format(len(lang_counts.keys()) - 12))
top_ten_other_values.append(
    sum(lang_counts.values()) - sum(top_ten_other_values) - lang_counts["english"] - lang_counts["unknown"])

for name, labels, values in zip(["English vs Others", "Others"], [english_labels, top_ten_other_langs],
                                [english_values, top_ten_other_values]):

    fig, ax = plt.subplots(figsize=(8, 5) if name == "Others" else (6, 3), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.5),
                           startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate("{} ({}%)".format(labels[i], round((values[i] / sum(values)) * 100, 1)), xy=(x, y),
                    xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)
    plt.tight_layout()
    plt.savefig("{}.pdf".format(name))
    plt.clf()
