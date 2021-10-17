from common.utils.file_model_util import FileModel
from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)

def main():
    '''Configurations'''
    limitPredictions = 3
    infinity = True
    showMetrics = True

    '''Load Models and Specify Architecture'''
    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()
    datasetArchitecture = DatasetArchitectureUtil('dronet')

    '''Start Predict Process'''
    ClassificationUtil(cam, limitPredictions, datasetArchitecture, modelX, modelY, infinity, showMetrics).predictLoopProcess()
   
if __name__ == "__main__":
    main()

        
    
