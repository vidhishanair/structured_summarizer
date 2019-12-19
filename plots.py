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
def statistics(counter, max_key=None):
    list_counter = []
    keys = counter.keys()
    if max_key is not None:
        keys = [key for key in counter.keys() if key < max_key]
    print(keys)
    for key in keys:
        list_counter += [key] * counter[key]
    array_counter = np.array(list_counter)
    mean = np.average(array_counter)
    stdv = np.std(array_counter)

    return mean, stdv

def plot_counter(counter, min_key=0, max_key=10, title=None):
    X = range(min_key, max_key + 1)
    counter_y = np.array([counter[x] for x in X])
    counter_ynorm = counter_y / np.sum(counter_y)
    plt.bar(X, counter_ynorm)
    if title is not None:
        plt.title(title)
    plt.show()


#%%
def plot_calc_stats(baseline_counter, pointgen_counter, pointgen_cov_counter, strcut_sum_counter, reference_counter, max_key=10):
    X = range(max_key + 1)
    baseline_y = np.array([baseline_counter[x] for x in X])
    pointgen_y = np.array([pointgen_counter[x] for x in X])
    pointgen_cov_y = np.array([pointgen_cov_counter[x] for x in X])
    struct_sum_y = np.array([strcut_sum_counter[x] for x in X])
    reference_y = np.array([reference_counter[x] for x in X])

    baseline_ynorm = baseline_y / np.sum(baseline_y)
    pointgen_ynorm = pointgen_y / np.sum(pointgen_y)
    pointgen_cov_ynorm = pointgen_cov_y / np.sum(pointgen_cov_y)
    struct_sum_ynorm = struct_sum_y / np.sum(struct_sum_y)
    reference_ynorm = reference_y / np.sum(reference_y)

    fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)
    axs[0].bar(X, baseline_ynorm)
    axs[0].set_title('S2S')
    axs[1].bar(X, pointgen_ynorm)
    axs[1].set_title('PG')
    axs[2].bar(X, pointgen_cov_ynorm)
    axs[2].set_title('PG+Cov')
    axs[3].bar(X, struct_sum_ynorm)
    axs[3].set_title('StructS')
    axs[4].bar(X, reference_ynorm)
    axs[4].set_title('Ref')
    plt.show()

    baseline_stats = statistics(baseline_counter, max_key=max_key)
    print('Baseline: mean {}, stdv {}'.format(baseline_stats[0], baseline_stats[1]))

    pointgen_stats = statistics(pointgen_counter, max_key=max_key)
    print('PointerGen: mean {}, stdv {}'.format(pointgen_stats[0], pointgen_stats[1]))

    pointergen_cov_stats = statistics(pointgen_cov_counter, max_key=max_key)
    print('PointerGen + Cov: mean {}, stdv {}'.format(pointergen_cov_stats[0], pointergen_cov_stats[1]))

    struct_sum_stats = statistics(strcut_sum_counter, max_key=max_key)
    print('StructSum: mean {}, stdv {}'.format(struct_sum_stats[0], struct_sum_stats[1]))

    struct_sum_stats = statistics(reference_counter, max_key=max_key)
    print('Reference: mean {}, stdv {}'.format(struct_sum_stats[0], struct_sum_stats[1]))


def plot_calc_stats_bin(baseline_counter, pointgen_counter, pointgen_cov_counter, strcut_sum_counter, reference_counter, max_key=10):
    X_bin = range(0, 21, 4)
    X = range(20)
    baseline_y = np.array([[baseline_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
    baseline_y_sum= np.sum(baseline_y, axis=1)
    pointgen_y = np.array([[pointgen_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
    pointgen_y_sum= np.sum(pointgen_y, axis=1)
    pointgen_cov_y = np.array([[pointgen_cov_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
    pointgen_cov_y_sum= np.sum(pointgen_cov_y, axis=1)
    reference_y = np.array([[reference_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
    reference_y_sum= np.sum(reference_y, axis=1)
    struct_sum_y = np.array([[strcut_sum_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
    struct_sum_y_sum= np.sum(struct_sum_y, axis=1)

    baseline_ynorm = baseline_y_sum / np.sum(baseline_y_sum)
    pointgen_ynorm = pointgen_y_sum / np.sum(pointgen_y_sum)
    pointgen_cov_ynorm = pointgen_cov_y_sum / np.sum(pointgen_cov_y_sum)
    struct_sum_ynorm = struct_sum_y_sum / np.sum(struct_sum_y_sum)
    reference_ynorm = reference_y_sum / np.sum(reference_y_sum)

    X_bin = range(5)

    fig, axs = plt.subplots(1, 5, sharey=True, tight_layout=True)
    axs[0].bar(X_bin, baseline_ynorm)
    axs[0].set_title('S2S')
    axs[1].bar(X_bin, pointgen_ynorm)
    axs[1].set_title('PG')
    axs[2].bar(X_bin, pointgen_cov_ynorm)
    axs[2].set_title('PG+Cov')
    axs[3].bar(X_bin, struct_sum_ynorm)
    axs[3].set_title('StructS')
    axs[4].bar(X_bin, reference_ynorm)
    axs[4].set_title('Ref')
    plt.show()

    baseline_stats = statistics(baseline_counter, max_key=max_key)
    print('Baseline: mean {}, stdv {}'.format(baseline_stats[0], baseline_stats[1]))

    pointgen_stats = statistics(pointgen_counter, max_key=max_key)
    print('PointerGen: mean {}, stdv {}'.format(pointgen_stats[0], pointgen_stats[1]))

    pointergen_cov_stats = statistics(pointgen_cov_counter, max_key=max_key)
    print('PointerGen + Cov: mean {}, stdv {}'.format(pointergen_cov_stats[0], pointergen_cov_stats[1]))

    struct_sum_stats = statistics(strcut_sum_counter, max_key=max_key)
    print('StructSum: mean {}, stdv {}'.format(struct_sum_stats[0], struct_sum_stats[1]))

    struct_sum_stats = statistics(reference_counter, max_key=max_key)
    print('Reference: mean {}, stdv {}'.format(struct_sum_stats[0], struct_sum_stats[1]))


#%%
# Baselines.
baseline_counter_3 = Counter({0: 19967, 1: 18228, 2: 11478, 3: 1824, 4: 222, 5: 83, 6: 45, 8: 24, 7: 14, 9: 7, 10: 6, 11: 2, 12: 2})
pointgen_counter_3 = Counter({0: 16180, 1: 14268, 2: 10405, 3: 1728, 4: 308, 5: 92, 6: 56, 7: 19, 8: 10, 10: 2, 9: 2, 11: 2})
pointgen_cov_counter_3 = Counter({0: 15299, 1: 13815, 2: 10389, 3: 1774, 4: 342, 5: 86, 6: 37, 7: 9, 8: 8, 9: 1, 11: 1})
# StructSum
strcut_sum_counter_3 = Counter({0: 21565, 1: 21045, 2: 17975, 3: 11024, 4: 3990, 5: 742, 6: 174, 7: 49, 8: 27, 9: 6, 12: 1, 13: 1, 16: 1, 11: 1, 10: 1})
reference_3 = Counter({0: 13688, 1: 11517, 2: 10327, 3: 6527, 4: 2490, 5: 1085, 6: 456, 7: 198, 8: 85, 9: 35, 10: 17, 11: 7, 13: 6, 14: 4, 16: 4, 17: 2, 18: 2, 22: 2, 30: 2, 35: 2, 19: 1, 21: 1, 23: 1, 26: 1, 28: 1, 27: 1, 29: 1, 12: 1, 25: 1})


# Baselines.
baseline_counter_2 = Counter({0: 28373, 1: 25055, 2: 15941, 3: 2508, 4: 316, 5: 120, 6: 65, 8: 32, 7: 26, 9: 12, 10: 6, 11: 2, 12: 2})
pointgen_counter_2 = Counter({0: 18894, 1: 15736, 2: 11305, 3: 1888, 4: 336, 5: 99, 6: 62, 7: 19, 8: 11, 10: 2, 9: 2, 11: 2})
pointgen_cov_counter_2 = Counter({0: 16775, 1: 14611, 2: 10950, 3: 1868, 4: 373, 5: 93, 6: 40, 7: 10, 8: 9, 9: 1, 11: 1})
# StructSum
strcut_sum_counter_2 = Counter({0: 21565, 1: 21045, 2: 17975, 3: 11024, 4: 3990, 5: 742, 6: 174, 7: 49, 8: 27, 9: 6, 12: 1, 13: 1, 16: 1, 11: 1, 10: 1})


#%%
plot_calc_stats(baseline_counter_3, pointgen_counter_3, pointgen_cov_counter_3, strcut_sum_counter_3, reference_3)

#%%
plot_calc_stats(baseline_counter_2, pointgen_counter_2, pointgen_cov_counter_2, strcut_sum_counter_2, strcut_sum_counter_2)

# %%

tree_height_counter = Counter({3: 5453, 2: 4126, 4: 1648, 5: 238, 6: 23, 1: 1, 7: 1})
leaf_proportion_counter = Counter({9.0: 6576, 8.0: 4380, 7.0: 475, 6.0: 53, 5.0: 5, 10.0: 1})

# %%
plot_counter(tree_height_counter, min_key=1, max_key=7, title='Tree depth')

# %%


# %%
X = range(100, 0, -10)
counter_y = np.array([leaf_proportion_counter[x/10] for x in X])
counter_ynorm = counter_y / np.sum(counter_y)
plt.bar(X[::-1], counter_ynorm[::-1])
plt.title('Proportion of leaf nodes')
plt.show()

#%%
plot_counter(leaf_proportion_counter, min_key=1, max_key=9, title='Proportion of leaf nodes', inverse=-10)


# %%
max_key = max(list(b_article_sent_id_counter.keys()) + list(pg_article_sent_id_counter.keys()) + list(pgc_article_sent_id_counter.keys()) + list(ss_article_sent_id_counter.keys()) + list(ref_article_sent_id_counter.keys()))
print(max_key) #16
# Let's make it 10 not so useful beyond that.
max_key = 170

#%%
############################################
# LS only
############################################

ss_article_sent_id_counter = Counter({1: 15636, 2: 14322, 3: 9745, 4: 9740, 0: 8774, 5: 7033, 6: 4541, 7: 3348, 8: 2982, 9: 2585, 10: 2180, 11: 1860, 12: 1588, 13: 1361, 14: 1153, 15: 987, 16: 814, 17: 726, 19: 563, 18: 546, 20: 181, 21: 162, 22: 139, 24: 125, 23: 116, 25: 109, 26: 98, 27: 83, 28: 74, 29: 73, 30: 63, 33: 51, 32: 51, 31: 50, 36: 37, 34: 35, 41: 34, 37: 33, 40: 32, 35: 31, 38: 27, 46: 24, 44: 22, 39: 21, 48: 19, 45: 17, 43: 16, 42: 15, 57: 11, 49: 10, 56: 9, 51: 8, 50: 8, 58: 7, 55: 6, 64: 6, 47: 5, 62: 5, 53: 4, 59: 3, 54: 3, 52: 3, 63: 2, 71: 2, 86: 2, 70: 2, 67: 2, 66: 1, 69: 1, 60: 1, 74: 1, 61: 1, 107: 1, 96: 1, 65: 1, 68: 1, 95: 1, 84: 1, 91: 1})
b_article_sent_id_counter = Counter({1: 12683, 2: 8864, 0: 7427, 3: 6535, 4: 5981, 5: 4741, 6: 3469, 7: 2515, 8: 2013, 9: 1824, 10: 1486, 11: 1348, 12: 1192, 13: 1025, 14: 929, 15: 834, 16: 623, 17: 587, 18: 473, 19: 370, 20: 296, 21: 255, 22: 216, 23: 193, 24: 178, 25: 171, 26: 141, 27: 137, 29: 134, 32: 108, 28: 107, 30: 98, 36: 94, 31: 86, 33: 71, 34: 62, 43: 61, 39: 59, 37: 59, 38: 57, 44: 55, 35: 54, 40: 50, 42: 50, 41: 42, 45: 37, 53: 36, 52: 33, 47: 33, 54: 32, 46: 30, 48: 28, 51: 26, 57: 25, 50: 23, 55: 23, 58: 20, 61: 20, 63: 19, 49: 16, 59: 16, 72: 15, 65: 15, 56: 15, 73: 14, 62: 13, 78: 13, 64: 12, 60: 12, 66: 11, 74: 11, 70: 10, 67: 10, 69: 9, 71: 9, 83: 8, 68: 7, 82: 6, 105: 5, 84: 5, 93: 5, 87: 5, 80: 5, 85: 5, 75: 5, 77: 5, 98: 4, 79: 4, 100: 4, 90: 4, 115: 4, 76: 3, 81: 3, 96: 3, 89: 3, 94: 3, 108: 2, 91: 2, 104: 2, 86: 2, 92: 1, 103: 1, 113: 1, 114: 1, 121: 1, 97: 1, 130: 1, 116: 1, 120: 1, 131: 1, 144: 1, 101: 1})
pg_article_sent_id_counter = Counter({1: 8922, 2: 8177, 3: 5572, 4: 4724, 0: 4607, 5: 3778, 6: 2652, 7: 1890, 8: 1365, 9: 1113, 10: 851, 11: 685, 12: 631, 13: 461, 14: 320, 15: 317, 16: 288, 17: 201, 18: 175, 20: 136, 19: 133, 21: 84, 22: 71, 24: 58, 23: 58, 26: 45, 25: 44, 28: 39, 31: 38, 34: 38, 29: 38, 27: 36, 30: 35, 32: 31, 36: 30, 33: 27, 38: 24, 37: 22, 35: 20, 39: 16, 41: 14, 40: 13, 50: 12, 51: 10, 60: 10, 52: 10, 47: 10, 43: 10, 49: 10, 45: 10, 44: 9, 62: 8, 48: 8, 63: 8, 71: 7, 53: 7, 42: 7, 54: 7, 46: 6, 61: 5, 67: 5, 65: 4, 55: 4, 58: 4, 57: 4, 69: 4, 109: 3, 74: 3, 76: 3, 59: 2, 64: 2, 66: 2, 70: 2, 78: 2, 99: 2, 91: 2, 56: 1, 68: 1, 106: 1, 75: 1, 82: 1, 73: 1, 72: 1, 84: 1, 86: 1, 102: 1, 85: 1, 103: 1})
pgc_article_sent_id_counter = Counter({1: 8529, 2: 7633, 0: 4935, 3: 4753, 4: 4109, 5: 3337, 6: 2564, 7: 1852, 8: 1402, 9: 1090, 10: 857, 11: 693, 12: 560, 13: 455, 14: 320, 15: 251, 16: 234, 17: 184, 18: 138, 19: 103, 20: 88, 21: 70, 22: 51, 23: 47, 24: 39, 26: 35, 25: 33, 27: 31, 31: 29, 28: 28, 29: 20, 34: 18, 30: 16, 32: 15, 35: 14, 39: 14, 33: 14, 36: 13, 37: 10, 38: 10, 40: 9, 51: 8, 45: 8, 49: 7, 42: 7, 41: 6, 52: 5, 43: 5, 56: 5, 59: 5, 48: 5, 44: 5, 50: 4, 47: 4, 54: 4, 55: 3, 57: 3, 79: 3, 60: 3, 82: 3, 58: 2, 62: 2, 53: 2, 71: 2, 99: 2, 73: 2, 67: 2, 70: 2, 63: 2, 65: 1, 68: 1, 115: 1, 81: 1, 78: 1, 75: 1, 64: 1, 69: 1, 76: 1, 80: 1, 94: 1, 66: 1})
ref_article_sent_id_counter = Counter({1: 8743, 2: 6695, 0: 6180, 4: 5081, 3: 5065, 5: 4207, 6: 3298, 7: 2847, 8: 2425, 9: 2186, 10: 1954, 11: 1927, 12: 1728, 13: 1552, 14: 1494, 15: 1356, 16: 1337, 17: 1254, 18: 1130, 19: 1024, 20: 981, 21: 928, 22: 848, 23: 779, 24: 683, 25: 652, 26: 605, 27: 560, 28: 529, 30: 494, 29: 488, 31: 463, 32: 415, 33: 394, 34: 355, 37: 320, 35: 319, 36: 313, 38: 304, 39: 264, 40: 256, 41: 239, 43: 214, 42: 205, 44: 197, 45: 182, 47: 173, 48: 164, 46: 156, 49: 151, 51: 138, 50: 137, 53: 129, 54: 112, 52: 108, 55: 91, 56: 90, 57: 77, 65: 72, 60: 71, 58: 64, 59: 63, 67: 63, 62: 62, 61: 58, 63: 55, 64: 50, 72: 44, 69: 43, 66: 41, 68: 35, 71: 35, 74: 33, 80: 32, 70: 32, 82: 30, 78: 28, 76: 27, 79: 27, 73: 24, 75: 24, 77: 23, 81: 21, 83: 20, 85: 20, 84: 18, 86: 18, 98: 13, 89: 12, 87: 11, 97: 10, 96: 9, 88: 9, 91: 8, 99: 7, 95: 7, 93: 7, 92: 7, 106: 6, 94: 5, 112: 5, 100: 5, 110: 5, 119: 4, 105: 4, 103: 3, 101: 3, 170: 2, 109: 2, 108: 2, 115: 2, 118: 2, 90: 2, 107: 2, 132: 2, 144: 2, 134: 1, 166: 1, 127: 1, 104: 1, 111: 1, 117: 1, 129: 1, 135: 1, 155: 1, 153: 1, 125: 1, 154: 1, 157: 1, 126: 1, 136: 1, 156: 1})

plot_calc_stats_bin(b_article_sent_id_counter, pg_article_sent_id_counter, pgc_article_sent_id_counter, ss_article_sent_id_counter, ref_article_sent_id_counter, max_key=20)
#%%

###########################################
# ES only
###########################################



###########################################
# ES + LS
###########################################



###########################################
# ES + LS + SA
###########################################



#%%
###########################################
# NGram
###########################################
novel_ngram_counter_ref = Counter({4: 498487, 3: 442825, 2: 313106, 1: 71625, 5: 45174})
repeated_ngram_counter_ref = Counter({1: 448695, 2: 333460, 3: 202244, 4: 136398, 5: 1555})

novel_ngram_counter_b = Counter({4: 199144, 3: 153668, 2: 89741, 5: 31227, 1: 20707})
repeated_ngram_counter_b = Counter({2: 317410, 1: 314378, 3: 262472, 4: 219223, 5: 1894})

novel_ngram_counter_pg = Counter({4: 79889, 3: 51814, 5: 21000, 2: 20532, 1: 704})
repeated_ngram_counter_pg = Counter({2: 500764, 3: 471706, 4: 440100, 1: 432462, 5: 12721})

novel_ngram_counter_pgc = Counter({4: 65538, 3: 41286, 5: 16439, 2: 15568, 1: 390})
repeated_ngram_counter_pgc = Counter({2: 657030, 3: 633041, 4: 600812, 1: 540376, 5: 17610})

novel_ngram_counter_ss = Counter({4: 228670, 3: 154468, 2: 65395, 5: 36677, 1: 3480})
repeated_ngram_counter_ss = Counter({2: 661452, 3: 581499, 1: 576815, 4: 501097, 5: 10200})


#%%
def novel_ngram_prob(novel_counter, repeated_counter):
    sorted_keys = list(novel_counter.keys())
    sorted_keys.sort()
    print(sorted_keys)
    # novel_count = np.array([novel_counter[key] )
    return np.array([novel_counter[key] / (novel_counter[key]+repeated_counter[key]) for key in sorted_keys])

ref_ngram_prob = novel_ngram_prob(novel_ngram_counter_ref, repeated_ngram_counter_ref)
b_ngram_prob = novel_ngram_prob(novel_ngram_counter_b, repeated_ngram_counter_b)
pg_ngram_prob = novel_ngram_prob(novel_ngram_counter_pg, repeated_ngram_counter_pg)
pgc_ngram_prob = novel_ngram_prob(novel_ngram_counter_pgc, repeated_ngram_counter_pgc)
ss_ngram_prob = novel_ngram_prob(novel_ngram_counter_ss, repeated_ngram_counter_ss)


#%%

nb_models=3
labels = ['1-gram', '2-gram', '3-gram', '4-gram', 'sent']

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
# rectspg = ax.bar(x + (-len(labels)/2 + 1) * width, pg_ngram_prob*100, width, label='PG')
rectspgc = ax.bar(x + (-nb_models/2 + 0) * width, pgc_ngram_prob*100, width, label='PG+C', color='gold')
rectsss = ax.bar(x + (-nb_models/2 +1) * width, ss_ngram_prob*100, width, label='SS', color='orangered')
rectsb = ax.bar(x + (-nb_models/2 + 2) * width, b_ngram_prob*100, width, label='S2S', color='wheat')
rectsref = ax.bar(x + (-nb_models/2 + 3) * width, ref_ngram_prob*100, width, label='Ref', edgecolor='salmon', hatch='//',facecolor = 'none')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('% Novel')
ax.set_title('Percent of novel n-grams')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# fig.figure(dpi=1200)
fig.savefig('percent_novel_ngrams.png', dpi=1200)
fig.show()

# %%


# %%


# %%
sum(list(novel_ngram_counter_pgc.values())) / (sum(list(novel_ngram_counter_pgc.values())) + sum(list(repeated_ngram_counter_pgc.values())))


# %%
sum(list(novel_ngram_counter_ss.values())) / (sum(list(novel_ngram_counter_ss.values())) + sum(list(repeated_ngram_counter_ss.values())))


# %%
X_bin = range(0, 21, 4)
X = range(20)
baseline_y = np.array([[b_article_sent_id_counter[x] for x in X if x < b and x >= b-4] for b in X_bin[1:]])
baseline_y_sum= np.sum(baseline_y, axis=1)

# %%
baseline_y_sum

# %%
baseline_list_counter = []
# for key in [key for key in b_article_sent_id_counter.keys() if key < 20]:
for key in [key for key in b_article_sent_id_counter.keys()]:
        baseline_list_counter += [key] * b_article_sent_id_counter[key]

pgc_list_counter = []
# for key in [key for key in pgc_article_sent_id_counter.keys() if key < 20]:
for key in [key for key in pgc_article_sent_id_counter.keys()]:
        pgc_list_counter += [key] * pgc_article_sent_id_counter[key]

ss_list_counter = []
# for key in [key for key in ss_article_sent_id_counter.keys() if key < 20]:
for key in [key for key in ss_article_sent_id_counter.keys()]:
        ss_list_counter += [key] * ss_article_sent_id_counter[key]

ref_list_counter = []
# for key in [key for key in ref_article_sent_id_counter.keys() if key < 20]:
for key in [key for key in ref_article_sent_id_counter.keys()]:
        ref_list_counter += [key] * ref_article_sent_id_counter[key]



data_to_plot = [pgc_list_counter, ss_list_counter, baseline_list_counter, ref_list_counter]


labels = ['PG+Cov', 'StructSum', 'S2S', 'Ref']
x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
ax.boxplot(data_to_plot)
ax.set_ylabel('Article sentence index')
ax.set_title('Distribution of information selection')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 20)
ax.legend()
# fig.figure(dpi=1200)
fig.savefig('information_selection.png', dpi=1200)
fig.show()

#%%
ss_ngram_prob*100

# %%
