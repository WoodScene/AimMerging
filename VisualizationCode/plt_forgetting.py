# 

import matplotlib.pyplot as plt

import numpy as np
import sys
import seaborn as sns

upper_bound_list = [0.88, 0.96, 0.87, 0.82, 0.83]
upper_bound = np.mean(upper_bound_list)


vanilla = [0.8693, 0.858, 0.8093, 0.8087, 0.84, 0.8, 0.6553, 0.6007, 0.6893, 0.6107, 0.672, 0.5747, 0.5493, 0.6207, 0.54]
baseline = [0.8693, 0.862, 0.8573, 0.8487, 0.8533, 0.8247, 0.8107, 0.822, 0.8027, 0.836, 0.8253, 0.8107, 0.7853, 0.7907, 0.7727]
ours = [0.8693, 0.858, 0.866, 0.8547, 0.8627, 0.8547, 0.8427, 0.8613, 0.8513, 0.864, 0.8553, 0.8307, 0.8133, 0.8433, 0.832]

vanilla_var = [0, 0.005726027, 0.008730692, 0.0043445, 0.00910807, 0.016413,
 0.02924104, 0.01725931, 0.0125951, 0.0227025, 0.01338715, 0.03110609,
 0.02052592, 0.0186692, 0.05756977]
baseline_var = [0.006, 0.005, 0.007, 0.006, 0.006, 0.005, 0.010, 0.009, 0.008, 0.009, 0.006, 0.008, 0.01, 0.009, 0.01]
ours_var = [0.005, 0.004, 0.006, 0.007, 0.005, 0.004, 0.009, 0.005, 0.008, 0.004, 0.005, 0.007, 0.006, 0.008, 0.009]


x = [i+1 for i in range(15)]
print(x) 
fig, ax = plt.subplots(figsize=(13, 8))
plt.rcParams['font.family'] = 'Times New Roman'



ax = sns.lineplot(x=x, y=vanilla, marker='o', linestyle="--", color = '#1a73d9', ci=95, label = "Vanilla T5-large")
ax = sns.lineplot(x=x, y=baseline, marker='^', linestyle="--", color = '#e87a25',ci=95, label = "Recurrent-KIF")
ax = sns.lineplot(x=x, y=ours, marker='^', linestyle="--", color = '#68ad39',ci=95, label = "AimMerging (ours)")

plt.fill_between(x, np.array(vanilla) - np.array(vanilla_var), np.array(vanilla) + np.array(vanilla_var), 
                 color='#1a73d9', alpha=0.2)
plt.fill_between(x, np.array(baseline) - np.array(baseline_var), np.array(baseline) + np.array(baseline_var), 
                 color='#FF8C00', alpha=0.2)
plt.fill_between(x, np.array(ours) - np.array(ours_var), np.array(ours) + np.array(ours_var), 
                 color='#32CD32', alpha=0.2)

plt.axhline(y=upper_bound,ls=":",c="black",linewidth=2, label = 'MULTI (upper bound)')
my_x_ticks = np.arange(1, 16, 1)

my_y_ticks = np.arange(0.5, 1, 0.1)
plt.xticks(my_x_ticks,fontname='Times New Roman', fontsize=14)
plt.yticks(my_y_ticks,fontname='Times New Roman', fontsize=14)

font1 = {'size': 19}
plt.grid(alpha=0.5, linestyle='-.')  # 网格线，更好看
plt.legend(prop=font1)

plt.xlabel("Tasks ID", fontdict={'family' : 'Times New Roman','size': 28})
plt.ylabel("Performance on Task 1", fontdict={'family' : 'Times New Roman','size': 26})
plt.tick_params(labelsize=18)
# from matplotlib.ticker import MaxNLocator
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# #plt.show()

plt.savefig("./Fig_paper/" + "Forgetting.png", dpi=600, bbox_inches='tight')
