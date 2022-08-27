from torchstat import stat
from torchvision.models import alexnet
import numpy as np

model = alexnet()
stat(model, (3, 224, 224))

ale_f = np.array([
    70470400+193600+193600,
    224088768+139968+139968,
    112205184+64896,
    149563648+43264,
    99723520+43264+43264,
    37748736+4096,
    16777216+4096,
    4096000
])

ale_o = np.array([
    3*224*224,
    64*27*27,
    192*13*13,
    384*13*13,
    256*13*13,
    256*6*6,
    4096,
    4096
])/1024/1024*8

