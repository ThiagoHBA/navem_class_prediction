from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)
configurations = ConfigurationUtil()


def main():
    architectureDetails = configurations.datasetArchitecture.getArchictecureDetails(
        useTensorflowLite=configurations.tensorflowLite
    )

    if not configurations.tensorflowLite:
        modelX = FileModel(
            path=architectureDetails['path'][0],
            modelName=architectureDetails['model_struct'],
            weightFile=architectureDetails['weight_file']
        ).compileTensorflowModel()

        modelY = FileModel(
            path=architectureDetails['path'][1],
            modelName=architectureDetails['model_struct'],
            weightFile=architectureDetails['weight_file']
        ).compileTensorflowModel()
    else:
        modelX = FileModel(
            path=architectureDetails['path'][0],
            modelName=architectureDetails['model_struct'],
            weightFile=architectureDetails['weight_file'],
        ).compileTensorflowLiteModel()

        modelY = FileModel(
            path=architectureDetails['path'][1],
            modelName=architectureDetails['model_struct'],
            weightFile=architectureDetails['weight_file']
        ).compileTensorflowLiteModel()


    ClassificationUtil(
        kerasModelX=modelX,
        kerasModelY=modelY,
        cam=cam,
        configurations=configurations,
    ).realTimeLoopProcess()


if __name__ == "__main__":
    main()
