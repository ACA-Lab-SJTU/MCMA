from globalSetting import *
from utils import *

class ANet(nn.Module):
    def __init__(self,netLst,activate=nn.Sigmoid()):
        super(ANet, self).__init__()
        self.activate = []
        self.layer = []
        for i in range(len(netLst)-1):
            self.layer.append(nn.Linear(netLst[i],netLst[i+1]))
            self.activate.append(activate)

    # Input.size = [batchSize, layer[0].inputSize]
    def forward(self, dataflow):
        for i in range(len(self.layer)):
            dataflow = self.layer[i](dataflow)
            dataflow = self.activate[i](dataflow)
        return dataflow 

class CNet(nn.Module):
    def __init__(self,netLst, activate=nn.Sigmoid()):
        super(CNet, self).__init__()
        self.layer = []
        self.activate = [] 
        for i in range(len(netLst)-1):
            self.layer.append(nn.Linear(netLst[i],netLst[i+1]))
            self.activate.append(activate if (i!=len(netLst)-2) else nn.LogSoftmax(dim=1))
    def forward(self, dataflow):
        for i in range(len(self.layer)):
            dataflow = self.layer[i](dataflow)
            dataflow = self.activate[i](dataflow)
        return dataflow

if (__name__=="__main__"):
    print ("Model test")
    benchName = 'bessel_Jnu'
    x,y,_,_ = loadData(benchName)
    netA,netC = getNetStructure(benchName,3)
    A = ANet(netA)
    C = CNet(netC)
    minix = miniBatch(x,8,1)
    output = A(minix)
    print (output)
    output = C(minix)
    print (output)
