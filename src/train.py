from globalSetting import *
from utils import *

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
    print ("data reading")

def modelLoading():
    print ("model loading")
    
if (__name__=="__main__"):
    print ("Process begins")
    configSetting()
    logSetting()
    dataReading()
    modelLoading()
    logging.info("initiation done")
    
    
