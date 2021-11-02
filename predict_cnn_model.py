from common.utils.file_model_util import FileModel
from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.classification_util import ClassificationUtil
from common.utils.files_util import Files
import cv2

cam = cv2.VideoCapture(0)

def main():
    experimentName = Files().createExperimentFile()
    path = None
    datasetArchitecture = DatasetArchitectureUtil('dronet')
    limitPredictions  = 3
    infinity = True
    showMetrics = False
    
    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()

    ClassificationUtil(limitPredictions, datasetArchitecture, modelX, modelY, cam, path, experimentName, infinity, showMetrics).realTimeLoopProcess()
       
if __name__ == "__main__":
    main()