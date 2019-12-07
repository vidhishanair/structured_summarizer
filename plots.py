#%%
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# %%
def average_counter(counter):
    vals = np.array(list(counter.keys()))
    counts = np.array(list(counter.values()))
    return vals * counts / np.sum(counts)

# %%
def statistics(counter):
    list_counter = []
    for key in counter.keys():
        list_counter += [key] * counter[key]
    array_counter = np.array(list_counter)
    mean = np.average(array_counter)
    stdv = np.std(array_counter)

    return mean, stdv

#%%
def plot_calc_stats(baseline_counter, pointgen_counter, pointgen_cov_counter, strcut_sum_counter):
    max_key = max(list(baseline_counter.keys()) + list(pointgen_counter.keys()) + list(pointgen_cov_counter.keys()) + list(strcut_sum_counter.keys()))
    print(max_key) #16
    # Let's make it 10 not so useful beyond that.
    max_key = 10

    X = range(max_key + 1)
    baseline_y = np.array([baseline_counter[x] for x in X])
    pointgen_y = np.array([pointgen_counter[x] for x in X])
    pointgen_cov_y = np.array([pointgen_cov_counter[x] for x in X])
    struct_sum_y = np.array([strcut_sum_counter[x] for x in X])

    baseline_ynorm = baseline_y / np.sum(baseline_y)
    pointgen_ynorm = pointgen_y / np.sum(pointgen_y)
    pointgen_cov_ynorm = pointgen_cov_y / np.sum(pointgen_cov_y)
    struct_sum_ynorm = struct_sum_y / np.sum(struct_sum_y)


    fig, axs = plt.subplots(1, 4, sharey=True, tight_layout=True)
    axs[0].bar(X, baseline_ynorm)
    axs[0].set_title('Baseline')
    axs[1].bar(X, pointgen_ynorm)
    axs[1].set_title('PointerGen')
    axs[2].bar(X, pointgen_cov_ynorm)
    axs[2].set_title('PointerGen+Cov')
    axs[3].bar(X, struct_sum_ynorm)
    axs[3].set_title('StructSum')
    plt.show()

    baseline_stats = statistics(baseline_counter)
    print('Baseline: mean {}, stdv {}'.format(baseline_stats[0], baseline_stats[1]))

    pointgen_stats = statistics(pointgen_counter)
    print('PointerGen: mean {}, stdv {}'.format(pointgen_stats[0], pointgen_stats[1]))

    pointergen_cov_stats = statistics(pointgen_cov_counter)
    print('PointerGen + Cov: mean {}, stdv {}'.format(pointergen_cov_stats[0], pointergen_cov_stats[1]))

    struct_sum_stats = statistics(strcut_sum_counter)
    print('StructSum: mean {}, stdv {}'.format(struct_sum_stats[0], struct_sum_stats[1]))

#%%
# Baselines.
baseline_counter_3 = Counter({0: 19967, 1: 18228, 2: 11478, 3: 1824, 4: 222, 5: 83, 6: 45, 8: 24, 7: 14, 9: 7, 10: 6, 11: 2, 12: 2})
pointgen_counter_3 = Counter({0: 16180, 1: 14268, 2: 10405, 3: 1728, 4: 308, 5: 92, 6: 56, 7: 19, 8: 10, 10: 2, 9: 2, 11: 2})
pointgen_cov_counter_3 = Counter({0: 15299, 1: 13815, 2: 10389, 3: 1774, 4: 342, 5: 86, 6: 37, 7: 9, 8: 8, 9: 1, 11: 1})
# StructSum
strcut_sum_counter_3 = Counter({0: 21565, 1: 21045, 2: 17975, 3: 11024, 4: 3990, 5: 742, 6: 174, 7: 49, 8: 27, 9: 6, 12: 1, 13: 1, 16: 1, 11: 1, 10: 1})


# Baselines.
baseline_counter_2 = Counter({0: 28373, 1: 25055, 2: 15941, 3: 2508, 4: 316, 5: 120, 6: 65, 8: 32, 7: 26, 9: 12, 10: 6, 11: 2, 12: 2})
pointgen_counter_2 = Counter({0: 18894, 1: 15736, 2: 11305, 3: 1888, 4: 336, 5: 99, 6: 62, 7: 19, 8: 11, 10: 2, 9: 2, 11: 2})
pointgen_cov_counter_2 = Counter({0: 16775, 1: 14611, 2: 10950, 3: 1868, 4: 373, 5: 93, 6: 40, 7: 10, 8: 9, 9: 1, 11: 1})
# StructSum
strcut_sum_counter_2 = Counter({0: 21565, 1: 21045, 2: 17975, 3: 11024, 4: 3990, 5: 742, 6: 174, 7: 49, 8: 27, 9: 6, 12: 1, 13: 1, 16: 1, 11: 1, 10: 1})


#%%
plot_calc_stats(baseline_counter_3, pointgen_counter_3, pointgen_cov_counter_3, strcut_sum_counter_3)

#%%
plot_calc_stats(baseline_counter_2, pointgen_counter_2, pointgen_cov_counter_2, strcut_sum_counter_2)