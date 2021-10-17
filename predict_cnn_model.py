from common.utils.file_model_util import FileModel
from common.utils.dataset_util import FileDataset
from common.utils.classification_util import ClassificationUtil

from keras.preprocessing import image
from datetime import datetime
import numpy as np
import cv2

cam = cv2.VideoCapture(0)

def main():
    limitPredictions = 3
    
    fileModelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5')
    fileModelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5')
    fileDataset = FileDataset('datasets','sidewalk_accy_all_datasets_classes_new_1630_00', 'dronet', '000001.jpg')

    modelX = fileModelX.compileModel()
    modelY = fileModelY.compileModel()
    
    infinity = True
    showMetrics = True
    
    predictLoopProcess(limitPredictions, fileDataset, modelX, modelY, infinity, showMetrics)

def predictLoopProcess(limit, fileDataset, kerasModelX, kerasModelY, infinity = False, metrics = False, loops = 1):
    classificationMethods = ClassificationUtil()

    for i in range(loops):
        classPredictions = []
        timeStart = datetime.now()
        
        for j in range(limit):
            resizedImage = captureAndResizedImage(fileDataset.getImageSize(), fileDataset.getImageColorScale())
            
            classX = predictImage(resizedImage, kerasModelX)
            classY = predictImage(resizedImage, kerasModelY)

            classPredictions.append((np.argmax(classX), np.argmax(classY)))

        classificationMethods.selectClassificationClass('x', classPredictions)
        classificationMethods.selectClassificationClass('y', classPredictions)

        timeEnd = datetime.now()
        
        if(metrics):
            calculeElapsedTime(timeStart, timeEnd, "Class Result")

    if(infinity):
        predictLoopProcess(limit, fileDataset, kerasModelX, kerasModelY, True, metrics ,loops)


def calculeElapsedTime(timeStart, timeEnd, label = ''):
    print("=" * 15)
    print(label + "ElapsedTime in ms: {}".format((timeEnd-timeStart).microseconds / 1000))
    print("=" * 15 + "\n")
        
def predictImage(image, kerasModel):       
    np.set_printoptions(suppress=True)
    images = np.vstack([image])
    result = kerasModel.predict(image, batch_size=64)
    
    return result

def captureAndResizedImage(imageSize, colorScale, metrics = False):
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
        calculeElapsedTime(timeStart, timeEnd, "Capture Image")
    return x

   
if __name__ == "__main__":
    main()

        
    
