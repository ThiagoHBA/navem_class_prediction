from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
from common.utils.classification_util import ClassificationUtil
import cv2

cam = cv2.VideoCapture(0)
configurations = ConfigurationUtil()


def main():
    architectureDetails = configurations.datasetArchitecture.getArchictecureDetails(useTensorflowLite = True)
    configurations.tensorflowLite = True
    tensorflowLiteModels = obtainTensorflowModels(architectureDetails)
    architectureDetails = configurations.datasetArchitecture.getArchictecureDetails(useTensorflowLite = False)
    configurations.tensorflowLite = False
    tensorflowNormalModels = obtainTensorflowModels(architectureDetails)

    ClassificationUtil(
        tensorflowModelX=tensorflowNormalModels[0],
        tensorflowModelY=tensorflowNormalModels[1],
        cam=cam,
        configurations=configurations,
    ).compareTensorflowLiteVersusNormal(
        pathImages = 'D:/Documentos/datasets/sidewalk_x_images/',
        tensorflowLiteModel=tensorflowLiteModels,
        tensorflowNormalModel=tensorflowNormalModels,
    )


def obtainTensorflowModels(architectureDetails):
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
