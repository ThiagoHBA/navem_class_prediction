import cv2

class DatasetArchitectureUtil:
    def __init__(self, architecture):
        self.architecture = architecture

    def getImageSize(self):
        if self.architecture == 'dronet':
            return 200,200
        elif self.architecture == 'vgg16':
            return 224,224
        else:
            print("Architecture not found")
            
    def getImageColorScale(self):
        if self.architecture == 'dronet':
            return cv2.COLOR_BGR2GRAY
        elif self.architecture == 'vgg16':
            return cv2.COLOR_BAYER_BG2BGR
        else:
            print("Architecture not found")

    def getArchictecureDetails(self):
        if self.architecture == 'dronet':
            return {
                'path': ('models/dronet/exp_349_x', 'models/dronet/exp_335_y'),
                'model_struct': 'model_struct.json',
                'weight_file': 'model_weights_299.h5'
            }
        else:
            print("Architecture not found")
            
