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
        tensorflowPath = 'tensorflow' if not useTensorflowLite else 'tensorflow_lite'
        if self.architecture == 'dronet':
            return {
                'path': (
                    modelPath + '/dronet/' + tensorflowPath + ('/exp_349_x' if not useTensorflowLite else '/dronet_model_x.tflite'), 
                    modelPath + '/dronet/' + tensorflowPath + ('/exp_335_y'  if not useTensorflowLite else '/dronet_model_y.tflite')
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_299.h5'
            }
        elif self.architecture == 'resnet':
            return {
                'path': (
                    modelPath + '/resnet/' + tensorflowPath + ('/exp_366_x' if not useTensorflowLite else '/resnet_model_x.tflite'), 
                    modelPath + '/resnet/' + tensorflowPath + ('/exp_320_y'  if not useTensorflowLite else '/resnet_model_y.tflite')
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            }
        elif self.architecture == 'vgg16':
            return {
                'path': (
                    modelPath + '/vgg16/' + tensorflowPath + ('/exp_340_x' if not useTensorflowLite else '/vgg16_model_x.tflite'), 
                    modelPath + '/vgg16/' + tensorflowPath + ('/exp_313_y'  if not useTensorflowLite else '/vgg16_model_y.tflite')
                ),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_99.h5'
            }
        else:
            print("Architecture not found")
            
