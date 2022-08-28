
from utils import *

# 54 450 866.7 1730
band_str = input('Bandwidth: ')
band_judge = float(band_str)
band = float(band_str)/8
number = 5

gs = GS()
methods = [recursion_call, middle_cut_call, min_cut, others]
layers = [gs.vgg_layers, gs.nin_layer, gs.ale_layers, gs.res_layers]

len_layers = len(layers)
len_methods = len(methods)

legend = ['T-GRB', 'T-BPA', 'DADS', 'NM']

fs = 18
# fcs=['royalblue', 'orange', 'limegreen', '']
fcs = ['#F47A20', '#d41b23', '#2351A3', '#9370db']
fcs_line = ['#2351a3', '#00babb', '#231f20', '#d1181f']
fcs = fcs_line

markers=['o','*','D','+']

name_list=['VGG-16', 'NiN', 'AlexNet', 'ResNet-18']

if band_judge == 54:
    start = 0
    end = 100
elif band_judge == 450:
    start = 0
    end = 550
else:
    start = 0
    end = 550
internal = int(end/100)
markevery = []
plt_internal = int(end / 10)
for i in range(end):
    if not ((i+1)%plt_internal):
        markevery.append(i)

