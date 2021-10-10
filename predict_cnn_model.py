from common.utils.file_model_util import FileModel
from common.utils.dataset_util import FileDataset
from common.utils.classification_util import ClassificationUtil
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from datetime import datetime
import numpy as np
import cv2
import time

cam = cv2.VideoCapture(0)

def main():
    limitPredictions = 3
    
    fileModelX = FileModel('models/exp_349_x', 'model_struct.json', 'model_weights_299.h5')
    fileModelY = FileModel('models/exp_335_y', 'model_struct.json', 'model_weights_299.h5')
    fileDataset = FileDataset('datasets','sidewalk_accy_all_datasets_classes_new_1630_00', 'dronet', '000001.jpg')

    modelX = compileModel(fileModelX)
    modelY = compileModel(fileModelY)
    infinity = True
    showMetrics = True
    
    predictLoopProcess(limitPredictions, fileDataset, modelX, modelY, infinity, showMetrics)

def predictLoopProcess(limit, fileDataset, kerasModelX, kerasModelY, infinity = False, metrics = False, loops = 1):
    for i in range(loops):
        classPredictions = []
        timeStart = datetime.now()
        
        for j in range(limit):
            resizedImage = captureAndResizedImage(fileDataset.getImageSize(), fileDataset.getImageColorScale())
            
            classX = predictImage(resizedImage, kerasModelX)
            classY = predictImage(resizedImage, kerasModelY)

            classPredictions.append((np.argmax(classX), np.argmax(classY)))

        ClassificationUtil.selectClassificationClass('x', classVote(classPredictions, 'x'))
        ClassificationUtil.selectClassificationClass('y', classVote(classPredictions, 'y'))

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

def compileModel(fileModel):
    print("Compiling Model...")
    print("="*15)
    try:
        _model = model_from_json(fileModel.jsonModel())
        _model.load_weights(fileModel.weightsModel())
        _model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    except:
        print('Impossible to load model, check the file paths')
        
    return _model

def captureAndResizedImage(imageSize, colorScale, metrics = False):
    timeStart = datetime.now()
    ret, camImage = cam.read()
    cv2.imshow('Imagem', camImage)
    #cv2.waitKey(166) # ~6 frames per second
    camImage = cv2.resize(camImage, imageSize)
    camImage = cv2.cvtColor(camImage, colorScale)
    x = image.img_to_array(camImage)
    x = np.expand_dims(x, axis=0)
    timeEnd = datetime.now()

    if(metrics):
        calculeElapsedTime(timeStart, timeEnd, "Capture Image")
    return x

def classVote(classValues, direction):
  directionIndex = 1 if direction.lower() == 'y' else 0
  values = []
  for i in range(len(classValues)):
    values.append(classValues[i][directionIndex])

  highestOccurrency = 0
  selectedClass = values[-1]

  for occurrency in range(len(values)-1, -1 , -1):
    counterOccurrency = values.count(values[occurrency])
    if(counterOccurrency > highestOccurrency):
      highestOccurrency = counterOccurrency
      selectedClass = values[occurrency]

  return selectedClass


   
if __name__ == "__main__":
    main()

        
    
