from torch.nn import Module
from torch import nn

class BottomNet(Module):
    def __init__(self,originalModel,layerNum=None):
        super(BottomNet,self).__init__()
        if layerNum==None:
            self.features = nn.Sequential(*list(originalModel.children())[0])
        else:
            self.features = nn.Sequential(*list(originalModel.children())[0][:layerNum])

        # for param in self.parameters():
        #     param.requires_grad=False

    def forward(self,x):
        x=self.features(x)
        return x