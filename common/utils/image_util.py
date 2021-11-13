from keras.preprocessing import image
from datetime import date, datetime
from os import path, times
from time import sleep
import common.utils.classification_util as classificationModule
import numpy as np
import cv2

class ImageUtil:
    @staticmethod
    def predictImage(image, kerasModel):       
        np.set_printoptions(suppress=True)
        image = np.vstack([image])
        result = kerasModel.predict(image, batch_size=64)
        
        return result

    @staticmethod
    def captureImage(cam, metrics = False, showPreview = False, framerate = 10):
        timeStart = datetime.now()

        camImage = cam.read()
        if(showPreview):
            ImageUtil.__showImage(camImage[1], 60)

        loadImageTime = (datetime.now() - timeStart).microseconds / 1000000
        ImageUtil.__waitFrameTime(framerate, loadImageTime)

        timeEnd = datetime.now()

        if(metrics):
            classificationModule.ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd, "Capture Image")

        return camImage[1]

    @staticmethod
    def resizeImage(image, imageSize, colorScale):
        return ImageUtil.__resizeImage(image, imageSize, colorScale)

    @staticmethod
    def saveImage(image, fileName: str, savePath = None):
        experimentPath = './experiments/' + savePath
        if(savePath != None and path.isdir(experimentPath)):
            cv2.imwrite(experimentPath + '/' + fileName + ".jpg", image)

    @staticmethod
    def openAndResizedImage(path, imageSize, colorScale, metrics = False, showPreview = False):
        timeStart = datetime.now()
        image = cv2.imread(path)
        if(showPreview):
            ImageUtil.__showImage(image, 1)
        resizedImage = ImageUtil.__resizeImage(image, imageSize, colorScale)
        timeEnd = datetime.now()
        if(metrics):
            classificationModule.ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd, "Open Image")
        return resizedImage

    def __waitFrameTime(framerate, imageCaptureTime):
        total = (1/framerate) - imageCaptureTime
        if(total > 0):
            sleep(total) 

    def __resizeImage(imageToResize, imageSize, colorScale):
        imageToResize = cv2.resize(imageToResize, imageSize)
        imageToResize = cv2.cvtColor(imageToResize, colorScale)
        x = image.img_to_array(imageToResize)
        x = np.expand_dims(x, axis=0)

        return x

    def __showImage(imageToShow, waitTime):
        imageToShow = cv2.resize(imageToShow, (480,720))
        cv2.imshow('Imagem', imageToShow)
        cv2.waitKey(waitTime)
