from keras.preprocessing import image
from datetime import datetime
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
        ret, camImage = cam.read()
        cv2.imshow('Imagem', camImage)
        cv2.waitKey(60) # ~6 frames per second
        camImage = cv2.resize(camImage, imageSize)
        camImage = cv2.cvtColor(camImage, colorScale)
        x = image.img_to_array(camImage)
        x = np.expand_dims(x, axis=0)
        timeEnd = datetime.now()

        if(metrics):
            classificationModule.ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd, "Capture Image")
        return x