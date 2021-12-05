from .files_util import Files
from keras.models import load_model
from keras.models import model_from_json
import sys

class FileModel:
    def __init__(self, path, modelName, weightFile):
        self.modelName = modelName
        self.path = path
        self.weightFile = weightFile
            
    def __jsonModel(self):
        file = Files.findFile(self.modelName, self.path)
        return Files.jsonFile(file)

    def __weightsModel(self):
        weightFile = Files.findFile(self.weightFile, self.path)
        return weightFile

    def compileTensorflowModel(self):
        print("Compiling Model...")
        print("="*15)
        try:
            _model = model_from_json(self.__jsonModel())
            _model.load_weights(self.__weightsModel())
            _model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
            
            return _model
        except:
            print('Impossible to load model, check the file paths')
            sys.exit()

    def compileTensorflowLiteModel(self):
        print("Compiling Model...")
        print("="*15)
        sys.exit()
            
    
