from torch.nn import Module
from torch import nn

class BottomNet(Module):
    def __init__(self,originalModel,layerNum):
        super(BottomNet,self).__init__()
        self.features=nn.Sequential(*list(originalModel.children())[0][:layerNum])

        for param in self.parameters():
            param.requires_grad=False

    def forward(self,x):
        x=self.features(x)
        return x