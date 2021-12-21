from common.utils.configuration_util import ConfigurationUtil
from common.utils.file_model_util import FileModel
# from common.utils.classification_util import ClassificationUtil
import cv2

import os
import pandas as pd
import numpy as np
import glob

configurations = ConfigurationUtil()
architectureDetails = configurations.datasetArchitecture.getArchictecureDetails(
    useTensorflowLite = configurations.tensorflowLite
)

def save(path, fileName, dataFrame):
    file = open(os.path.join(path, fileName), "w")
    for sample in range(dataFrame.shape[0]):
        # print(dataFrame.iloc[sample]["pred"])
        file.write(str(dataFrame.iloc[sample]["predLite"]) + " " + str(dataFrame.iloc[sample]["real"]) + "\n")
    file.close()
    print("File saved")

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

def main():
    datasetName = "market_accy_all_datasets_classes_362_00"
    type = "test"

    basePath = os.path.join("./../datasets/vgg16/", datasetName, datasetName, type, datasetName)
    pathImages = os.path.join(basePath, "images")
    # path_labels = os.path.join(base_path, "gyro.txt")

    pathBaseExperiment = os.path.join("./../experiments")
    experimentName="exp_417"
    fileName="predict_truth_test_model_weights_299.h5_0_.txt"

    df = pd.read_csv(os.path.join(pathBaseExperiment, experimentName, fileName), sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])

    tensorflowModels = obtainTensorflowModels()

    predictLite=[]
    for file in glob.glob(os.path.join(pathImages, "*.jpg")):
        print(file)
        predictLite.append()

    df["predLite"] = predictLite
    save(os.path.join(pathBaseExperiment, experimentName), datasetName + "_lite.txt", df)

if __name__ == "__main__":
    main()
