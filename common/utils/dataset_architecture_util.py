import cv2

class DatasetArchitectureUtil:
    def __init__(self, architecture):
        self.architecture = architecture

    def getImageSize(self):
        if self.architecture == 'dronet':
            return 200,200
        elif self.architecture == 'vgg16':
            return 224,224
        elif self.architecture == 'resnet':
            return 224,224
        else:
            print("Architecture not found")
            
    def getImageColorScale(self):
        if self.architecture == 'dronet':
            return cv2.COLOR_BGR2GRAY
        elif self.architecture == 'vgg16':
            return cv2.COLOR_RGB2BGR
        elif self.architecture == 'resnet':
            return cv2.COLOR_RGB2BGR
        else:
            print("Architecture not found")

    def getArchictecureDetails(self, useTensorflowLite = False):
        modelPath = 'models'
        tensorflowPath = 'tensorflow' if not useTensorflowLite else 'tensorflowLite'
        if self.architecture == 'dronet':
            return {
                'path': (
                    modelPath + '/dronet/' + tensorflowPath + '/exp_349_x', 
                    modelPath + '/dronet/' + tensorflowPath + '/exp_335_y'
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_299.h5'
            }
        elif self.architecture == 'vgg16':
            return {
                'path': (
                    'models/vgg16/exp_313_x', 
                    'models/vgg16/exp_340_y'
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            }
        elif self.architecture == 'resnet':
            return {
                'path': (
                    'models/resnet/exp_366_x', 
                    'models/resnet/exp_320_y'
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            }
        else:
            print("Architecture not found")
            
