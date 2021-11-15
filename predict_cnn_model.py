from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)

def main():
    configurations = ConfigurationUtil()

    architectureDetails = configurations.datasetArchitecture.getArchictecureDetails()
    
    modelX = FileModel(architectureDetails['path'][0], architectureDetails['model_struct'], architectureDetails['weight_file']).compileModel()
    modelY = FileModel(architectureDetails['path'][1], architectureDetails['model_struct'], architectureDetails['weight_file']).compileModel()
    
    ClassificationUtil(modelX, modelY, configurations.limitPredictions, configurations.datasetArchitecture, cam, configurations.infinity, configurations.showMetrics, configurations.loops, configurations.showPreview, framerate = configurations.fps, logOnImage = configurations.logOnImage).realTimeLoopProcess()
       
if __name__ == "__main__":
    main()