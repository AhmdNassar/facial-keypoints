import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from torchvision.models import vgg16
class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        # buld model
        self.vgg16 = vgg16()
        self.vgg16.features[0] = nn.Conv2d(1,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg16.classifier[-1] = nn.Linear(4096,136)

    def forward(self,x):
        x = self.vgg16(x)
    
        return x