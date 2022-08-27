import os
from config import *
import numpy as np

from utils_cpu import cpu_min
from utils import *
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['figure.figsize'] = (8.0, 3.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'black' # 设置 颜色 style

fig_name = 'exp_1_'+band_str+'.jpg'
cpu_powers, cpu_types = cpu_min(number)
if cpu_powers==False:
    print('Error! The required amount of cpus is larger than that available.')
    os._exit(0)
cpu_powers=np.array(cpu_powers)
print(cpu_powers)
print(cpu_types)

len_layers = len(layers)
len_methods = len(methods)
results = []
for i in range(len_methods):
    one_line = []
    for j in range(len_layers):
        one_line.append(1/methods[i](layers[j], cpu_powers, band))
        # if i==len_methods-1:
        #     print(111, one_line[j])
    results.append(one_line)

x=list(range(len_layers))
total_width, n = 0.8, len_methods
width = total_width / n

for i in range(len(x)):
    x[i] = x[i] - 1*width
for i in range(len_methods):
    if i==1:
        plt.bar(x, results[i], width=width, label=legend[i], tick_label=name_list, fc=fcs[i])
    elif i==len_methods-1:
        if fcs[i]=='':
            plt.bar(x, results[i], width=width, label=legend[i])
        else:
            plt.bar(x, results[i], width=width, label=legend[i], fc=fcs[i])
    else:
        # print(results[i])
        # print(x)
        # print(width)
        plt.bar(x, results[i], width=width, label=legend[i], fc=fcs[i])
    for j in range(len(x)):
        x[j] = x[j] + width

# print(results)
results=np.array(results)
print(results[0]/results[1])
print(results[0]/results[2])
print(results[0]/results[3])

# plt.ylim((0, 7))

plt.legend(fontsize=fs, loc='upper left')
plt.grid(color='black')
# plt.xlabel('DNN', fontsize=fs)
plt.ylabel('FPS', fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.tight_layout()
plt.savefig(fig_name, dpi=600)
plt.show()

