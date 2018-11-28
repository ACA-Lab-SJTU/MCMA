from globalSetting import *
from utils import *
from data import *
from model import *

def parserSetting():
    parser = argparse.ArgumentParser(description="--bench benchName")
    parser.add_argument(
            "--bench",
            type = str,
            nargs = 1,
            default = 'bessel_Jnu',
            help = 'Check ../data/* for all the benchmarks.'
            )
    args = parser.parse_args()
    benchName = args.bench[0]
    return args

def configSetting():
    global c
    print("config loadding")        
    c = json.load(open(configPath,'r'))
    configSavedPath = os.path.join(workDir, 'config.json')
    configSavedFile = open(configSavedPath, 'w')
    json.dump(c, configSavedFile, sort_keys=False, indent=4)

def logSetting():
    logging.basicConfig(
        filename=logPath,
        filemode="a+",
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%y/%m/%d_%H:%M:%S",
        level=logging.DEBUG
    )
    logging.debug("test log function")

def dataReading():
    global trainSrc,trainTgt,testSrc,testTgt,benchName
    print ("data reading")
    trainSrc,trainTgt,testSrc,testTgt = loadData(benchName)

def modelLoading():
    print ("model loading")
    global A,C,netA,netC
    numA = c['model']['numA']
    netA,netC = getNetStructure(benchName,numA)
    A = [ANet(netA) for i in range(numA)]
    C = CNet(netC)

def trainModel(model, src, tgt):
    predict = model(src)

def train():
    print ("training begin")
    epochA = c['train']['epochA']
    epochC = c['train']['epochC']
    batchSizeA = c['train']['batchSizeA']
    iterNum = c['train']['iteration']
    dataType = c['train']['dataUpdataType'] #See this in the paper
    eb = c['train']['errorBound']

    for iterN in range(iterNum):
        minix = miniBatch(trainSrc,batchSizeA,1) 

    
if (__name__=="__main__"):
    print ("Process begins")
    parserSetting()
    configSetting()
    logSetting()
    dataReading()
    modelLoading()
    logging.info("initiation done")
    train()
    
    
