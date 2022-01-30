from datetime import date, datetime
import glob
import pandas as pd
from common.utils.files_util import Files
from common.utils.image_util import ImageUtil
import numpy as np
import common.enum.classification_enum as classificationEnum
import os
class ClassificationUtil:
    def __init__(self, tensorflowModelX, tensorflowModelY, configurations, experimentName = None, cam=None):
        self.cam = cam
        self.tensorflowModelX = tensorflowModelX
        self.tensorflowModelY = tensorflowModelY
        self.configurations = configurations
        self.experimentName = experimentName if experimentName != None else str(input("Enter the experiment name: "))
        self.logs = self.__generateLogFile()

    def realTimeLoopProcess(self):
        index = 0
        imageIndex = 0
        classPredictions = []
        self.experimentName = Files().createExperimentFile(self.experimentName)

        while index < self.configurations.loops:
            self.__processLog('Real Time Classification')
            while len(classPredictions) < self.configurations.limitPredictions:
                if(self.cam != None):
                    startProcessImageTime = datetime.now()

                    capturedImage = ImageUtil.captureImage(self.cam, self.configurations.showPreview, self.configurations.fps)
                    resizedImage = ImageUtil.resizeImage(capturedImage, self.configurations.datasetArchitecture.getImageSize(),  self.configurations.datasetArchitecture.getImageColorScale())
                    normalizedImage = ImageUtil.normalizeImage(resizedImage)

                    finishProcessImageTime = datetime.now()

                    if(self.configurations.tensorflowLite):
                        classX = ImageUtil.predictImageTensorflowLite(normalizedImage, self.tensorflowModelX)
                        classY = ImageUtil.predictImageTensorflowLite(normalizedImage, self.tensorflowModelY)
                    else:
                        classX = ImageUtil.predictImageTensorflow(normalizedImage, self.tensorflowModelX)
                        classY = ImageUtil.predictImageTensorflow(normalizedImage, self.tensorflowModelY)

                    classPredictions.append((np.argmax(classX), np.argmax(classY)))
                    imageIndex += 1

            imageResult = self.classificationToMap(imageIndex, self.selectClassificationClass('x', classPredictions), self.selectClassificationClass('y', classPredictions))
            self.logs.writeLog(imageResult)

            if self.configurations.logOnImage:
                linearText = imageResult['data']['linear']
                sidesText = imageResult['data']['sides']
                capturedImage = ImageUtil.writeTextInImage(capturedImage, (10 , 150), linearText, fontColor = (15, 150, 0)) #BGR
                capturedImage = ImageUtil.writeTextInImage(capturedImage, (10 , 200), sidesText, fontColor = (0, 0, 255))

            ImageUtil.saveImage(capturedImage, str(imageIndex).zfill(5), self.experimentName)
            classPredictions.pop(0)

            finishPredictionTime = datetime.now()

            if(self.configurations.showMetrics):
                print("\n" + "-" * 15 + "\tMetrics\t" + "-" * 15)
                self.printElapsedTime(startProcessImageTime, finishProcessImageTime, "Process Image")
                self.printElapsedTime(finishProcessImageTime, finishPredictionTime, "Predict Result")
                self.printElapsedTime(startProcessImageTime, finishPredictionTime, "Total ")

            index = 0 if(self.configurations.infinity) else index + 1

    def filePredictProcess(self, path = None, start=0):
        if(path != None):
            numberOfItens = self.__countFilesInDir(path)
            imageIndex = 0
            for i in range(start, numberOfItens, self.configurations.limitPredictions):
                self.__processLog('File Classification')
                classPredictions = []
                imageIndex = i

                for j in range(self.configurations.limitPredictions):
                    openedImageTimeStart = datetime.now()

                    openedImage = ImageUtil.openImage(path + str(i + j) + '.jpg', self.configurations.showPreview)
                    resizedImage = ImageUtil.resizeImage(openedImage, self.configurations.datasetArchitecture.getImageSize(),  self.configurations.datasetArchitecture.getImageColorScale())

                    openedImageTimeFinish = datetime.now()

                    if(self.configurations.tensorflowLite):
                        classX = ImageUtil.predictImageTensorflowLite(resizedImage, self.tensorflowModelX)
                        classY = ImageUtil.predictImageTensorflowLite(resizedImage, self.tensorflowModelY)
                    else:
                        classX = ImageUtil.predictImageTensorflow(resizedImage, self.tensorflowModelX)
                        classY = ImageUtil.predictImageTensorflow(resizedImage, self.tensorflowModelY)

                    classPredictions.append((np.argmax(classX), np.argmax(classY)))

                self.logs.writeLog(self.classificationToMap(imageIndex, self.selectClassificationClass('x', classPredictions), self.selectClassificationClass('y', classPredictions)))

                timeEnd = datetime.now()

                if(self.configurations.showMetrics):
                    print("\n" + "-" * 15 + "\tMetrics\t" + "-" * 15)
                    self.printElapsedTime(openedImageTimeStart, openedImageTimeFinish, "Opened Image")
                    self.printElapsedTime(openedImageTimeFinish, timeEnd, "File Predict Process ")
                    self.printElapsedTime(openedImageTimeStart, timeEnd, "Total ")


    def evaluateDataset(self, generatedFileName: str , axis: str, pathTxt = None, pathImages = None,):
        if(pathImages != None):
            imageIndex = 0
            evaluatePredictions = []
            df = pd.read_csv(pathTxt, sep=" ", engine="python", encoding="ISO-8859-1", names=['pred', 'real'])
            for file in glob.glob(os.path.join(pathImages, "*.jpg")):
                self.__processLog('Evaluate Dataset')

                openedImageTimeStart = datetime.now()

                openedImage = ImageUtil.openImage(file, self.configurations.showPreview)
                resizedImage = ImageUtil.resizeImage(openedImage, self.configurations.datasetArchitecture.getImageSize(),  self.configurations.datasetArchitecture.getImageColorScale())
                normalizedImage = ImageUtil.normalizeImage(resizedImage)

                openedImageTimeFinish = datetime.now()

                if(self.configurations.tensorflowLite):
                    classX = ImageUtil.predictImageTensorflowLite(normalizedImage, self.tensorflowModelX)
                    classY = ImageUtil.predictImageTensorflowLite(normalizedImage, self.tensorflowModelY)
                else:
                    classX = ImageUtil.predictImageTensorflow(normalizedImage, self.tensorflowModelX)
                    classY = ImageUtil.predictImageTensorflow(normalizedImage, self.tensorflowModelY)

                evaluatePredictions.append(np.argmax(classX) if axis.lower() == 'x' else np.argmax(classY))

                imageIndex += 1
                timeEnd = datetime.now()

                if(self.configurations.showMetrics):
                    print("\n" + "-" * 15 + "\tMetrics\t" + "-" * 15)
                    self.printElapsedTime(openedImageTimeStart, openedImageTimeFinish, "Opened Image")
                    self.printElapsedTime(openedImageTimeFinish, timeEnd, "File Predict Process ")
                    self.printElapsedTime(openedImageTimeStart, timeEnd, "Total ")
                                
            df['predLite'] = evaluatePredictions
            self.__save('./', generatedFileName + '_' + axis  + ".txt", df)

    def __save(self, path, fileName, dataFrame):
        file = open(os.path.join(path, fileName), "w")
        for sample in range(dataFrame.shape[0]):
            file.write(str(dataFrame.iloc[sample]["predLite"]) + " " + str(dataFrame.iloc[sample]["real"]) + "\n")
        file.close()
        print("File saved")

    @staticmethod
    def calculeClassificationElapsedTime(timeStart, timeEnd):
        return (timeEnd - timeStart).total_seconds()

    @staticmethod
    def printElapsedTime(timeStart, timeEnd, label=''):
        print("\n" + "=" * 15)
        print(label + "ElapsedTime in seconds: ", ClassificationUtil.calculeClassificationElapsedTime(timeStart, timeEnd))
        print("=" * 15)

    @staticmethod
    def classificationToMap(index, linearValue, sidesValue):
        return {
            "index": index,
            "data": {
                "linear": linearValue,
                "sides": sidesValue,
            },
        }

    def classVote(self, classValues, direction):
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

    def selectClassificationClass(self, axis, value):
        if(axis.lower() == 'x'):
            return self.__handleClassificationValue("Linear: ", classificationEnum.LinearClassificationClass, self.classVote(value, 'x'))
        else:
            return self.__handleClassificationValue("Lateral: ", classificationEnum.SidesClassificationClass, self.classVote(value, 'y'))

    def __generateLogFile(self):
        logs = Files(self.experimentName) if self.experimentName != None else Files()
        logs.initializeLog()
        return logs

    def __countFilesInDir(self, path: str) -> int:
        file_entries = [entry for entry in os.scandir(path) if entry.is_file()]

        return len(file_entries)

    def __handleClassificationValue(self, type, enum, value):
        try:
            result = enum(value)
            print("~" * 30)
            print(type + result.name)
            print("~" * 30)
            return result.name
        except:
            print("Class value has not been mapped yet")


    def __processLog(self, process):
        print("\n" + "=" * 15 + '\t' + process + " {}\t".format(self.configurations.datasetArchitecture.architecture) + "=" * 15)
