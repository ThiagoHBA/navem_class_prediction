from .files_util import Files

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
