from common.utils.file_model_util import FileModel
from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)

def main():
    '''Configurations'''
    limitPredictions  = 3
    infinity = True
    showMetrics = False
    path = 'D:/Documentos/datasets/2020_06_25-16_03_09/'
    datasetArchitecture = DatasetArchitectureUtil('dronet')

    '''Load Models'''
    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()

    '''Start Predict Process'''
    ClassificationUtil(limitPredictions, datasetArchitecture, modelX, modelY, cam, path, infinity, showMetrics).filePredictProcess(750)
       
if __name__ == "__main__":
    main()