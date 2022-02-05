from common.utils.dataset_architecture_util import DatasetArchitectureUtil
from common.utils.file_model_util import FileModel
import os
import tensorflow as tf
import sys

from common.utils.files_util import Files

def main():
    evaluate = isEvaluate()
    architectureName = getArchitectureName()
    architectureDetails = DatasetArchitectureUtil(architectureName).getArchictecureDetails(evaluate = evaluate)
    axis = getAxis()

    tensorflowLiteModel = generateTensorflowLiteFromTensorflowModel(
        tensorflowModel =  FileModel(
            path = architectureDetails['path'][0 if axis == 'x' else 1],
            modelName = architectureDetails['model_struct'],
            weightFile = architectureDetails['weight_file']
        ).compileTensorflowModel()
    )

    createTensorflowLiteFile(model = tensorflowLiteModel, name = architectureName, axis = axis, evaluate = evaluate)

def generateTensorflowLiteFromTensorflowModel(tensorflowModel):
    print(("=" * 15) + " Generating Model " + ("=" * 15))
    converter = tf.lite.TFLiteConverter.from_keras_model(tensorflowModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

def createTensorflowLiteFile(model, name, axis, evaluate):
    tensorflowLitePath = f'./models/{name}/tensorflow_lite' + ('_evaluate' if evaluate else '')
    Files.createPathIfNotExist(f'{tensorflowLitePath}/')

    tensorflowLiteFileName = f'{name}_model_{axis}.tflite'
    completePath = f'{tensorflowLitePath}/{tensorflowLiteFileName}'

    open(completePath, 'wb').write(model)

def getArchitectureName():
    architecturesPath = os.listdir('./models')
    showArchitectureOptions()
    return architecturesPath[getSelectedArchitecture()]

def showArchitectureOptions():
    options = os.listdir('./models')
    print("\n" + ("=" * 30))
    for i in range(len(options)):
        print(f"{i + 1}) {options[i]}")

def getSelectedArchitecture():
    try:
        return int(input("\nArchitecture Number: ")) - 1
    except:
        print('\nNot possible to find the architecture')
        sys.exit()

def getAxis():
    axis = str(input('Axis [x/y]: ')).lower()
    if(axis != 'x' and axis != 'y'):
        print("Invalid Axis")
        sys.exit()
    return axis

def isEvaluate():
    return str(input("Evaluate? [y/n]: ")).lower() == 'y'

if __name__ == "__main__":
    main()
