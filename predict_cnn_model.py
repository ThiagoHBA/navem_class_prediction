from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.classification_util import ClassificationUtil
from common.utils.files_util import Files
import cv2

cam = cv2.VideoCapture(0)

def main():
    configurations = ConfigurationUtil() 
    if not str(input("Continue with default configurations? [y/n]: ")).lower() == 'y':
        configurations.updateConfigurations()
    experimentName = Files().createExperimentFile(str(input("Enter the experiment name: ")))
    modelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5').compileModel()
    modelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5').compileModel()
    ClassificationUtil(modelX, modelY, configurations.limitPredictions, configurations.datasetArchitecture, cam, experimentName, configurations.infinity, configurations.showMetrics, configurations.loops).realTimeLoopProcess()
       
if __name__ == "__main__":
    main()