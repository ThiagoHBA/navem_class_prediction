from files_util import FileModel
from files_util import FileDataset
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import cv2
import time

cam = cv2.VideoCapture(0)


def main():
    fileModel = FileModel('exp_335', 'model_struct.json', 'model_weights_299.h5')
    fileDataset = FileDataset('datasets','sidewalk_accy_all_datasets_classes_new_1630_00', 'dronet', '000001.jpg')

    try:
        model = model_from_json(fileModel.jsonModel())
        model.load_weights(fileModel.weightsModel())
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    except:
        print('Impossible to load model, check the file paths')

##    imagePath = fileDataset.getDatasetTrainPathImage()
##    loadedImage = image.load_img(imagePath, target_size = fileDataset.getImageSize(), color_mode='grayscale')

    while True:
        resizedImage = captureAndResizedImage(fileDataset.getImageSize(), cv2.COLOR_BGR2GRAY)
        
        np.set_printoptions(suppress=True)
        images = np.vstack([resizedImage])
        classes = model.predict(resizedImage, batch_size=64)

        print("Image in class: {}".format(np.argmax(classes)))


def captureAndResizedImage(imageSize, colorScale):   
    ret, camImage = cam.read()
    cv2.imshow('Imagem', camImage)

    cv2.waitKey(100)
    camImage = cv2.resize(camImage, imageSize)
    camImage = cv2.cvtColor(camImage, colorScale)

    x = image.img_to_array(camImage)
    x = np.expand_dims(x, axis=0)

    return x
   
if __name__ == "__main__":
    main()

        
    
