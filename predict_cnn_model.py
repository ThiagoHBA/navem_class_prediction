from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)
configurations = ConfigurationUtil()
architectureDetails = configurations.datasetArchitecture.getArchictecureDetails(
    useTensorflowLite = configurations.tensorflowLite
)

def main():
    tensorflowModels = obtainTensorflowModels()

    ClassificationUtil(
        tensorflowModelX = tensorflowModels[0],
        tensorflowModelY = tensorflowModels[1],
        cam = cam,
        configurations = configurations,
    ).realTimeLoopProcess()


def obtainTensorflowModels():
    modelX = FileModel(
                path=architectureDetails['path'][0],
                modelName=architectureDetails['model_struct'],
                weightFile=architectureDetails['weight_file']
            )
    modelY = FileModel(
                path=architectureDetails['path'][1],
                modelName=architectureDetails['model_struct'],
                weightFile=architectureDetails['weight_file']
            )

    if configurations.tensorflowLite:
        return (modelX.compileTensorflowLiteModel(), modelY.compileTensorflowLiteModel())

    return (modelX.compileTensorflowModel(), modelY.compileTensorflowModel())


if __name__ == "__main__":
    main()