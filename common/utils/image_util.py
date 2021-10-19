from keras.preprocessing import image
from datetime import date, datetime
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
    def captureAndResizedImage(cam, imageSize, colorScale, metrics = False):
        timeStart = datetime.now()
        camImage = cam.read()
        ImageUtil.__showImage(camImage[1], 60)
        resizedImage = ImageUtil.__resizeImage(camImage[1], imageSize, colorScale)
        timeEnd = datetime.now()

        if(metrics):
            classificationModule.ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd, "Capture Image")
        return resizedImage

    @staticmethod
    def openAndResizedImage(path, imageSize, colorScale, metrics = False):
        timeStart = datetime.now()
        image = cv2.imread(path)
        ImageUtil.__showImage(image, 1)
        resizedImage = ImageUtil.__resizeImage(image, imageSize, colorScale)
        timeEnd = datetime.now()

        if(metrics):
            classificationModule.ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd, "Open Image")
        return resizedImage

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
