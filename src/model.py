from globalSetting import *
from utils import *

class ANet(nn.Module):
    def __init__(self,netLst):
        super(ANet, self).__init__()
        self.netLst = netLst
        self.layer = []
        for i in range(len(netLst)-1):
            self.layer.append(nn.Linear(netLst[i],netLst[i+1]))

    # Input.size = [batchSize, layer[0].inputSize]
    def forward(self, input):
        for layer in self.layer:
            output = layer(input)
            output = F.relu(output)
            input = output
        return output
        


class CNet(nn.Module):
    def __init__(self,para):
        super(CNet, self).__init__()


if (__name__=="__main__"):
    print ("Model test")
    benchName = 'bessel_Jnu'
    x,y,_,_ = loadData(benchName)
    print (x[:10])
    print (y[:10])
    os._exit(0)
    netA,netC = getNetStructure(benchName,3)
    A = ANet(netA)
    testInput = [1,2]
    testInput = torch.tensor([[1,2]],dtype=torch.float)
    output = A(testInput)
    print (output)