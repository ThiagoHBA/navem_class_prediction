from datetime import datetime
import sys
import os
import json

class Files:
    def __init__(self, fileName = datetime.utcnow().strftime("%d_%m_%Y-%H_%M_%S")):
        self.fileName = fileName

    @staticmethod
    def findFile(fileToSearch, path):
        for root, dirs, files in os.walk(path):
            if fileToSearch in files:
                return os.path.join(root, fileToSearch)
    
    @staticmethod
    def jsonFile(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        return loaded_model_json

    @staticmethod
    def createPathIfNotExist(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def createExperimentFile(self, experimentName):
        Files.createPathIfNotExist('experiments')
        if experimentName != '':
            if os.path.isdir('./experiments/' + experimentName):
                if not str(input("This path already exist, do you want to keep going? [y/n]: ")).lower() == 'y':
                    sys.exit()
                return experimentName
            else:
                os.mkdir('./experiments/' + experimentName)
                return experimentName
        os.mkdir('./experiments/' + self.fileName)
        return self.fileName

    def initializeLog(self):
        if not os.path.isdir('logs'):
            os.mkdir('logs')
        with open("logs/" + self.fileName + "_log.json", 'w') as logFile:
            json.dump({"logs": []}, logFile)

    def writeLog(self, logValue):
        with open("logs/" + self.fileName + "_log.json", 'r+') as logFile:
            logFileData = json.load(logFile)
            logFileData["logs"].append(logValue)
            logFile.seek(0)
            json.dump(logFileData, logFile, indent = 4)
            