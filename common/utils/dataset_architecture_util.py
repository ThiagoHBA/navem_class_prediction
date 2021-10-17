import cv2

class DatasetArchitectureUtil:
    def __init__(self, architecture):
        self.architecture = architecture

    def getImageSize(self):
        if self.architecture == 'dronet':
            return 200,200
        else:
            print("Architecture not found")
            
    def getImageColorScale(self):
        if self.architecture == 'dronet':
            return cv2.COLOR_BGR2GRAY
        else:
            print("Architecture not found")
