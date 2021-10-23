from datetime import datetime
import os
import json
class Files:
    def __init__(self, fileName = datetime.utcnow().strftime("%d_%m_%Y-%H_%M_%S")):
        self.fileName = fileName

    @staticmethod
    def findFile(fileName, path):
        for root, dirs, files in os.walk(path):
            if fileName in files:
                return os.path.join(root, fileName)
    
    @staticmethod
    def jsonFile(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        return loaded_model_json

    def initializeLog(self):
        with open("logs/" + self.fileName + "_log.json", 'w') as logFile:
            json.dump({"logs": []}, logFile)

    def writeLog(self, logValue):
        with open("logs/" + self.fileName + "_log.json", 'r+') as logFile:
            logFileData = json.load(logFile)
            logFileData["logs"].append(logValue)
            logFile.seek(0)
            json.dump(logFileData, logFile, indent = 4)
            