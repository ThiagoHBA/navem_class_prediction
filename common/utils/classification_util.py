from time import sleep
from datetime import datetime
from common.utils.image_util import ImageUtil
import numpy as np
import common.enum.classification_enum as classificationEnum
class ClassificationUtil:
    def __init__(self, cam, limit, datasetArchitecture, kerasModelX, kerasModelY, infinity = False, metrics = False, loops = 1):
        self.cam = cam
        self.limit = limit
        self.datasetArchitecture = datasetArchitecture
        self.kerasModelX = kerasModelX
        self.kerasModelY = kerasModelY
        self.infinity = infinity
        self.metrics = metrics
        self.loops = loops

    def predictLoopProcess(self):
        for i in range(self.loops):
            classPredictions = []
            timeStart = datetime.now()
            
            for j in range(self.limit):
                resizedImage = ImageUtil.captureAndResizedImage(self.cam, self.datasetArchitecture.getImageSize(), self.datasetArchitecture.getImageColorScale(), self.metrics)

                classX = ImageUtil.predictImage(resizedImage, self.kerasModelX)
                classY = ImageUtil.predictImage(resizedImage, self.kerasModelY)

                classPredictions.append((np.argmax(classX), np.argmax(classY)))

            self.__selectClassificationClass('x', classPredictions)
            self.__selectClassificationClass('y', classPredictions)

            timeEnd = datetime.now()
            
            if(self.metrics):
                self.calculeClassificationElapsedTime(timeStart, timeEnd, "Class Result")

        if(self.infinity):
            self.predictLoopProcess()

    @staticmethod
    def calculeClassificationElapsedTime(timeStart, timeEnd, label = ''):
        print("=" * 15)
        print(label + "ElapsedTime in ms: {}".format((timeEnd-timeStart).microseconds / 1000))
        print("=" * 15 + "\n")

    def __selectClassificationClass(self,axis, value):
        if(axis.lower() == 'x'):
            self.__handleClassificationValue(classificationEnum.LinearClassificationClass, self.__classVote(value, 'x'))
        else:
            self.__handleClassificationValue(classificationEnum.SidesClassificationClass, self.__classVote(value, 'y'))

    def __handleClassificationValue(self,enum ,value):
        try:
            result = enum(value)
            print("=" * 30)
            print(result.name)
            print("=" * 30)
            sleep(0.15 * result.value + 1)
        except:
            print("Class value has not been mapped yet")

    def __classVote(self, classValues, direction):
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
