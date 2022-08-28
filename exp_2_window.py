import os
from config import *
import numpy as np

from utils_cpu import cpu_window
from utils import *
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['figure.figsize'] = (8.0, 12.0) # 设置figure_size尺寸
# plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'black' # 设置 颜色 style

fig_name = 'exp_2_'+band_str+'.jpg'

# ra = range(len_layers)
ra = [0, 1, 2, 3]
for i in ra:
    layer_result = []
    for j in range(len_methods):
        method_result = []
        for idx in range(start, end):
            cpu_powers, cpu_types = cpu_window(idx, idx + 5)
            if cpu_powers == False:
                # print('Error! The required amount of cpus is larger than that available.')
                break
            # print(band)
            method_result.append(1 / methods[j](layers[i], cpu_powers, band))

        layer_result.append(method_result)
    x=list(range(1, len(layer_result[0])+1))
    plt.subplot(len(ra), 1, list(ra).index(i)+1)
    for j in range(len_methods):
        plt.plot(x, layer_result[j], label=legend[j],
                 marker=markers[j], color=fcs_line[j],
                 markevery=markevery)
        # if legend[j]=='DADS':
        #     print('DADS', layer_result[j])
        # elif legend[j]=='NM':
        #     print('NM', layer_result[j])
        # print(legend[j], layer_result[j])
    plt.legend(fontsize=fs, loc='best')
    plt.grid(color='black')
    # plt.xlabel('DNN', fontsize=fs)
    plt.ylabel('FPS\n(' + name_list[i]+')', fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)

plt.xlabel('Index', fontsize=fs)
plt.tight_layout()
plt.savefig(fig_name, dpi=600)
plt.show()

