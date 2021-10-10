from time import sleep
import common.enum.classification_enum as classificationEnum


class ClassificationUtil:
    def selectClassificationClass(axis, value):
        if(axis.lower() == 'x'):
            ClassificationUtil.handleClassificationValue(classificationEnum.LinearClassificationClass, value)
        else:
            ClassificationUtil.handleClassificationValue(classificationEnum.SidesClassificationClass, value)

    def handleClassificationValue(enum ,value):
        try:
            result = enum(value)
            print("=" * 30)
            print(result.name)
            print("=" * 30)
            sleep(0.15 * result.value + 1)
        except:
            print("Class value has not been mapped yet")