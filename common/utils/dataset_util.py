import os
import sys
import cv2

class FileDataset:
    def __init__(self, path, datasetName, architecture, imageFile):
        self.datasetName = datasetName
        self.path = path
        self.architecture = architecture
        self.imageFile = imageFile

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

    def getDatasetTrainPathImage(self):
        datasetFile = self.getDatasetFile()
        return os.path.join(datasetFile, "train", self.datasetName, "images", self.imageFile)

    def getDatasetFile(self):
        return os.path.join(self.path, self.datasetName)
