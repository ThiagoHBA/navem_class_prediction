import cv2

class DatasetArchitectureUtil:
    def __init__(self, architecture):
        self.architecture = architecture

    def getImageSize(self):
        imageSizeDict = {
            'dronet': (200,200),
            'resnet': (224,224),
            'vgg16': (224,224),
        }
        architectureSize = imageSizeDict[self.architecture]
        return architectureSize if architectureSize != None else print("Architecture not found")
            
    def getImageColorScale(self):
        colorScaleDict = {
            'dronet': cv2.COLOR_BGR2GRAY,
            'resnet': cv2.COLOR_RGB2BGR,
            'vgg16': cv2.COLOR_RGB2BGR,
        }
        architectureColorScale = colorScaleDict[self.architecture]
        return architectureColorScale if architectureColorScale != None else print("Architecture not found")

    def getArchictecureDetails(self, useTensorflowLite = False, evaluate = False):
        tensorflowPath = 'tensorflow' if not useTensorflowLite else 'tensorflow_lite'

        dronetPath = ('dronet_x/' if not useTensorflowLite else '/dronet_model_x.tflite', 'dronet_y/' if not useTensorflowLite else '/dronet_model_y.tflite',)
        resnetPath = ('resnet_x/' if not useTensorflowLite else '/resnet_model_x.tflite', 'resnet_y/' if not useTensorflowLite else '/resnet_model_y.tflite')
        vgg16Path = ('vgg16_x/' if not useTensorflowLite else '/vgg16_model_x.tflite', 'vgg16_y/' if not useTensorflowLite else '/vgg16_model_y.tflite')

        if(evaluate):
            tensorflowPath = tensorflowPath + '_evaluate'

        architectureDetailsDict = {
            'dronet': {
                'path': (
                    f'models/dronet/{tensorflowPath}/{dronetPath[0]}',
                    f'models/dronet/{tensorflowPath}/{dronetPath[1]}',
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_299.h5'
            },
            'resnet': {
                'path': (
                    f'models/resnet/{tensorflowPath}/{resnetPath[0]}',
                    f'models/resnet/{tensorflowPath}/{resnetPath[1]}',
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            },
            'vgg16': {
                'path': (
                    f'models/vgg16/{tensorflowPath}/{vgg16Path[0]}',
                    f'models/vgg16/{tensorflowPath}/{vgg16Path[1]}',
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            }
        }
        architectureDetails = architectureDetailsDict[self.architecture]
        return architectureDetails if architectureDetails != None else print("Architecture not found")