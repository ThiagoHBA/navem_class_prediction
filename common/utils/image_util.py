from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from datetime import date, datetime
from os import path, times, system
from time import sleep
#import common.utils.classification_util as classificationModule

import numpy as np
import cv2
import importlib

foundPicameraModule = False
spam_loader = importlib.util.find_spec('picamera')

if spam_loader is not None:
    from picamera import PiCamera #type: ignore
    foundPicameraModule = True

class ImageUtil:
    @staticmethod
    def predictImageTensorflow(image, tensorflowModel):
        np.set_printoptions(suppress=True)
        image = np.vstack([image])
        result = tensorflowModel.predict(image, batch_size=64)

        return result

    @staticmethod
    def predictImageTensorflowLite(image, tensorflowLiteModel):
        inputDetails = tensorflowLiteModel.get_input_details()
        outputDetails = tensorflowLiteModel.get_output_details()

        tensorflowLiteModel.set_tensor(inputDetails[0]['index'], image)
        tensorflowLiteModel.invoke()

        return tensorflowLiteModel.get_tensor(outputDetails[0]['index'])

    @staticmethod
    def captureImage(cam, showPreview = False, framerate = 10):
        if(foundPicameraModule):
            return ImageUtil.__callPicameraCapture(showPreview, framerate)
        return ImageUtil.__callOpenCVCapture(cam, showPreview, framerate)

    @staticmethod
    def resizeImage(image, imageSize, colorScale):
        return ImageUtil.__resizeImage(image, imageSize, colorScale)

    @staticmethod
    def normalizeImage(image):
        return ImageUtil.__normalizeImageAsKerasMath(image)

    @staticmethod
    def saveImage(image, fileName: str, savePath = None):
        experimentPath = './experiments/' + savePath
        if(savePath != None and path.isdir(experimentPath)):
            cv2.imwrite(experimentPath + '/' + fileName + ".jpg", image)

    @staticmethod
    def writeTextInImage(image, org, imageText: str, fontColor = (0,0,0)):
        if(imageText != None):
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            thickness              = 2
            lineType               = 2

            cv2.putText(image, imageText, org, font, fontScale, fontColor, thickness, lineType)

        return image

    @staticmethod
    def openImage(path, showPreview = False):
        image = cv2.imread(path)
        if(showPreview):
            ImageUtil.__showImage(image, 1)
        return image

    def __callPicameraCapture(showPreview = False, framerate = 10):
        startCaptureTime = datetime.now()

        camera = PiCamera()
        camera.resolution = (640 , 480)
        camera.framerate = 10

        #Capture opencv object
        output = np.empty((480 * 640 * 3,), dtype=np.uint8)
        camera.capture(output, 'bgr')
        output = output.reshape((480, 640, 3))

        if(showPreview):
            ImageUtil.__showImage(output, 60)

        loadImageTime = (datetime.now() - startCaptureTime).microseconds / 1000000
        ImageUtil.__waitFrameTime(framerate, loadImageTime)

        camera.close()

        return output

    def __callOpenCVCapture(cam, showPreview = False, framerate = 10):
        startCaptureTimer = datetime.now()

        camImage = cam.read()
        if(showPreview):
            ImageUtil.__showImage(camImage[1], 60)

        loadImageTime = (datetime.now() - startCaptureTimer).microseconds / 1000000
        ImageUtil.__waitFrameTime(framerate, loadImageTime)

        return camImage[1]

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

    '''
        Using Numpy lib, but only work if image is gray scale
    '''
    def __normalizeImageNumpy(image):
        return (image - np.min(image))/np.ptp(image)

    def __normalizeImageKeras(image, file):
        #print(len(image))
        #print(image[0][0])
        print(file[:-10] + '\\')
        #img = load_img(file)
        #img_arr = np.expand_dims(img_to_array(image), axis=0)
        datagen = ImageDataGenerator(rescale=1./255)
        #cv2.imshow('asf', datagen.flow(img_arr, batch_size=1)[0])
        #cv2.waitKey()
        #system('pause')
        pathImages = r'D:\\Mestrado\\datasets\\dronet\\sidewalk_accy_all_datasets_classes_new_1630_07\\sidewalk_accy_all_datasets_classes_new_1630_07\\test\\sidewalk_accy_all_datasets_classes_new_1630_07\\images',
        val_generator = datagen.flow_from_directory(pathImages, batch_size=1, color_mode='grayscale')
        system('pause')
        x,y = val_generator.next()
        print(x)
        return x[0]
        #return datagen.flow_from_directory(file[:-11], batch_size=1, color_mode='grayscale')[0]

    def __normalizeImageAsKerasMath(image):
        return image*1./255

    def __normalizeImage(image):
        return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    def __showImage(imageToShow, waitTime):
        imageToShow = cv2.resize(imageToShow, (480,720))
        cv2.imshow('Imagem', imageToShow)
        cv2.waitKey(waitTime)
