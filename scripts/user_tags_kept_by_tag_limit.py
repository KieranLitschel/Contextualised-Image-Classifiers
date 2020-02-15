from embeddings.load import count_user_tags
from matplotlib import pyplot as plt
import numpy as np

training_subset = "C:\\Honors Project\\YFCC100M\\dataset\\oiv\\train.tsv"

for tag_limit in range(1, 11):
    user_tag_counts_at_limit = sorted(count_user_tags(training_subset, tag_limit).items(), key=lambda x: x[1],
                                      reverse=True)
    cumulative_user_tag_counts = []
    for tag, count in user_tag_counts_at_limit:
        if not cumulative_user_tag_counts or cumulative_user_tag_counts[-1][0] != count:
            cumulative_freq = 1
            if cumulative_user_tag_counts:
                cumulative_freq += cumulative_user_tag_counts[-1][1]
            cumulative_user_tag_counts.append([count, cumulative_freq])
        else:
            cumulative_user_tag_counts[-1][1] += 1
    x = np.log10([count for count, _ in cumulative_user_tag_counts])
    y = np.log10([cumu_freq for _, cumu_freq in cumulative_user_tag_counts])
    plt.plot(x, y, label=tag_limit)
plt.ylabel("User tags kept (log 10)")
plt.xlabel("User tag threshold (log 10)")
plt.legend()
plt.tight_layout()
plt.savefig("User tag threshold vs tags kept.pdf")
