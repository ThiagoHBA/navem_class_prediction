from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)

def main():
    configurations = ConfigurationUtil()

    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()
    
    ClassificationUtil(modelX, modelY, configurations.limitPredictions, configurations.datasetArchitecture, cam, configurations.infinity, configurations.showMetrics, configurations.loops, configurations.showPreview, logOnImage = configurations.logOnImage).realTimeLoopProcess()
       
if __name__ == "__main__":
    main()