import os
import time
from time import gmtime, strftime

class fileOperations:
    def __init__(self, modelName, dataset):
        super(fileOperations, self).__init__()
        self.weights_to = "./weights/" + modelName + "/" + dataset + "/"
        # self.getSaveDirectory()
    
    def getSaveDirectory(self):
        cvtTime = strftime("%Y-%d-%m %H:%M:%S", gmtime())
        cvtTime = cvtTime[5:-3].replace('-', '_')
        cvtTime = cvtTime.replace(' ', '_')
        cvtTime = cvtTime.replace(':', '_')
        self.weights_to = os.path.join(self.weights_to, 'run' + cvtTime)
    
    def mkdir_if_missing(self):
        self.getSaveDirectory()
        os.makedirs(self.weights_to, exist_ok=True)
        return self.weights_to
    
    def createLogFile(self, log_filename):
        log_file = os.path.join(self.weights_to, log_filename)
        return log_file
    
    def writeLogFile(self, log_file, log_content):
        f_log = open(log_file, "a")
        f_log.write(log_content)
        f_log.close()