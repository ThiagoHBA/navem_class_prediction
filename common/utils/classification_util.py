from datetime import datetime
from common.utils.files_util import Files
from common.utils.image_util import ImageUtil
import numpy as np
import common.enum.classification_enum as classificationEnum
import os
class ClassificationUtil:
    def __init__(self, limit, datasetArchitecture, kerasModelX, kerasModelY, cam=None, path=None, infinity=False, metrics=False, loops=1):
        self.cam = cam
        self.limit = limit
        self.datasetArchitecture = datasetArchitecture
        self.kerasModelX = kerasModelX
        self.kerasModelY = kerasModelY
        self.infinity = infinity
        self.metrics = metrics
        self.loops = loops
        self.path = path

    def realTimeLoopProcess(self):
        for i in range(self.loops):
            print("\nReal Time Classification")
            classPredictions = []
            timeStart = datetime.now()

            for j in range(self.limit):
                if(self.cam != None):
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
            self.realTimeLoopProcess()

    def filePredictProcess(self, start=0):
        if(self.path != None):
            logs = Files()
            numberOfItens = self.__countFilesInDir(self.path)
            imageIndex = 0

            logs.initializeLog()
            for i in range(start, numberOfItens, self.limit):
                print("\nFile Classification")
                classPredictions = []
                timeStart = datetime.now()
                imageIndex = i

                for j in range(self.limit):
                    resizedImage = ImageUtil.openAndResizedImage(self.path + str(i + j) + '.jpg', self.datasetArchitecture.getImageSize(), self.datasetArchitecture.getImageColorScale(), self.metrics)
                    classX = ImageUtil.predictImage(resizedImage, self.kerasModelX)
                    classY = ImageUtil.predictImage(resizedImage, self.kerasModelY)
                    classPredictions.append((np.argmax(classX), np.argmax(classY)))

                logs.writeLog(self.classificationToMap(imageIndex, self.__selectClassificationClass('x', classPredictions), self.__selectClassificationClass('y', classPredictions)))
                
                timeEnd = datetime.now()
                if(self.metrics):
                    self.calculeClassificationElapsedTime(timeStart, timeEnd, "File Predict Process ")

    @staticmethod
    def calculeClassificationElapsedTime(timeStart, timeEnd, label=''):
        print("=" * 15)
        print(label + "ElapsedTime in ms: {}".format((timeEnd -
              timeStart).microseconds / 1000))
        print("=" * 15 + "\n")

    @staticmethod
    def classificationToMap(index, linearValue, sidesValue):
        return {
            "index": index,
            "data": {
                "linear": linearValue,
                "sides": sidesValue,
            },
        }

    def __countFilesInDir(self, path: str) -> int:
        file_entries = [entry for entry in os.scandir(path) if entry.is_file()]

        return len(file_entries)

    def __selectClassificationClass(self, axis, value):
        if(axis.lower() == 'x'):
            return self.__handleClassificationValue("Linear: ", classificationEnum.LinearClassificationClass, self.__classVote(value, 'x'))
        else:
            return self.__handleClassificationValue("Lateral: ", classificationEnum.SidesClassificationClass, self.__classVote(value, 'y'))

    def __handleClassificationValue(self, type, enum, value):
        try:
            result = enum(value)
            print("~" * 30)
            print(type + result.name)
            print("~" * 30)
            return result.name
            #sleep(0.15 * result.value + 1)
        except:
            print("Class value has not been mapped yet")

    def __classVote(self, classValues, direction):
        directionIndex = 1 if direction.lower() == 'y' else 0
        values = []
        for i in range(len(classValues)):
            values.append(classValues[i][directionIndex])

        highestOccurrency = 0
        selectedClass = values[-1]

        for occurrency in range(len(values)-1, -1, -1):
            counterOccurrency = values.count(values[occurrency])
            if(counterOccurrency > highestOccurrency):
                highestOccurrency = counterOccurrency
                selectedClass = values[occurrency]

        return selectedClass
