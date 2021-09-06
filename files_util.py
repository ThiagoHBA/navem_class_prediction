import os
import sys

class Files:
    def findFile(fileName, path):
        for root, dirs, files in os.walk(path):
            if fileName in files:
                return os.path.join(root, fileName)
            
    def jsonFile(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        return loaded_model_json

class FileModel:
    def __init__(self, path, modelName, weightFile):
        self.modelName = modelName
        self.path = path
        self.weightFile = weightFile
            
    def jsonModel(self):
        file = Files.findFile(self.modelName, self.path)
        return Files.jsonFile(file)

    def weightsModel(self):
        weightFile = Files.findFile(self.weightFile, self.path)
        return weightFile

class FileDataset():
    def __init__(self, path, datasetName, architecture, imageFile):
        self.datasetName = datasetName
        self.path = path
        self.architecture = architecture
        self.imageFile = imageFile

    def getImageSize(self):
        if self.architecture == 'dronet':
            return 200,200

    def getDatasetTrainPathImage(self):
        datasetFile = self.getDatasetFile()
        return os.path.join(datasetFile, "train", self.datasetName, "images", self.imageFile)

    def getDatasetFile(self):
        return os.path.join(self.path, self.datasetName)
        
