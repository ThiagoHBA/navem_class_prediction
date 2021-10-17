from time import sleep
import common.enum.classification_enum as classificationEnum

class ClassificationUtil:
    def selectClassificationClass(self,axis, value):
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