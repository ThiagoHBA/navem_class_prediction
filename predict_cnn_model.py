from files_util import FileModel
from files_util import FileDataset
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
    
    fileModelX = FileModel('exp_349_x', 'model_struct.json', 'model_weights_299.h5')
    fileModelY = FileModel('exp_335_y', 'model_struct.json', 'model_weights_299.h5')
    fileDataset = FileDataset('datasets','sidewalk_accy_all_datasets_classes_new_1630_00', 'dronet', '000001.jpg')

    modelX = compileModel(fileModelX)
    modelY = compileModel(fileModelY)
    
    predictLoopProcess(limitPredictions, fileDataset, modelX, modelY, False)

def predictLoopProcess(limit, fileDataset, kerasModelX, kerasModelY, infinity = False, loops = 1):
    for i in range(loops):
        classPredictions = []
        
        for j in range(limit):
            resizedImage = captureAndResizedImage(fileDataset.getImageSize(), fileDataset.getImageColorScale())
            
            classX = predictImage(resizedImage, kerasModelX)
            classY = predictImage(resizedImage, kerasModelY)

            classPredictions.append((np.argmax(classX), np.argmax(classY)))

        print("Image in class X: {}".format(classVote(classPredictions, 'x')))
        print("Imagem in class Y: {}".format(classVote(classPredictions, 'y')))

    if(infinity):
        predictLoopProcess(limit, fileDataset, kerasModelX, kerasModelY, True, loops)
        
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

def captureAndResizedImage(imageSize, colorScale):
    ret, camImage = cam.read()
    cv2.imshow('Imagem', camImage)
    cv2.waitKey(166) # ~6 frames per second
    camImage = cv2.resize(camImage, imageSize)
    camImage = cv2.cvtColor(camImage, colorScale)
    x = image.img_to_array(camImage)
    x = np.expand_dims(x, axis=0)

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

        
    
