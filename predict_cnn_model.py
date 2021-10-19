from common.utils.file_model_util import FileModel
from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.classification_util import ClassificationUtil
import cv2

from common.utils.image_util import ImageUtil

cam = cv2.VideoCapture(0)

def main():
    '''Configurations'''
    limitPredictions = 3
    infinity = True
    showMetrics = False
    path = None
    datasetArchitecture = DatasetArchitectureUtil('dronet')

    '''Load Models'''
    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()

    '''Start Predict Process'''
    ClassificationUtil(limitPredictions, datasetArchitecture, modelX, modelY, cam, path, infinity, showMetrics).realTimeLoopProcess()
       
if __name__ == "__main__":
    main()

        
    
