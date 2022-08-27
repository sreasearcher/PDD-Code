from torchstat import stat
from torchvision.models import resnet18
import torch
import numpy as np

model = resnet18()
stat(model, (3, 224, 224))
res_f = np.array([np.sum([118013952,1605632,802816,802816]),
                  np.sum([115605504,401408,200704]),
                  np.sum([115605504,401408]),
                  np.sum([115605504,401408,200704]),
                  115605504+401408,
                  57802752+200704+100352,
                  115605504+200704,
                  115605504+200704+100352,
                  115605504+200704,
                  57802752+100352+50176,
                  115605504+100352,
                  115605504+100352+50176,
                  115605504+100352,
                  57802752+50176+25088,
                  115605504+50176,
                  115605504+50176+25088,
                  115605504+50176,
                  512000])
res_f_down = np.array([6422528+200704,
                       6422528+100352,
                       6422528+50176])
res_f_down_idx = [6,10,14]

res_o = np.array([3*224*224,
                  64*56*56,
                  64*56*56,
                  64*56*56,
                  64*56*56,
                  64*56*56,
                  128*28*28,
                  64*56*56,
                  128*28*28,
                  128*28*28,
                  256*14*14,
                  256*14*14,
                  256*14*14,
                  256*14*14,
                  512*7*7,
                  512*7*7,
                  512*7*7,
                  512*7*7])/1024/1024*8
res_o_d = np.array([128*28*28,
                    256*14*14,
                  512*7*7])/1024/1024*8
print(res_o)
print(res_o_d)
