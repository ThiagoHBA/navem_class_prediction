import unittest
from common.enum.classification_enum import LinearClassificationClass, SidesClassificationClass
from common.utils.classification_util import ClassificationUtil

class ClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.classificationObject = ClassificationUtil(None, None, None, None, experimentName = 'test')
        return super().setUp()

    def test_ClassVote(self):
        classValues = [(0, 1), (1, 0), (0, 1)]
        self.assertEqual(self.classificationObject.classVote(classValues, 'x'), 0, 'Class Vote X test')
        self.assertEqual(self.classificationObject.classVote(classValues, 'y'), 1, 'Class Vote Y test')
        classValues = [(1, 0), (1, 0), (1, 0)]
        self.assertEqual(self.classificationObject.classVote(classValues, 'X'), 1, 'Class Vote X test')
        self.assertEqual(self.classificationObject.classVote(classValues, 'Y'), 0, 'Class Vote Y test')

    def test_LinearClassificationValues(self):
        classValues = [(0, 1), (1, 0), (0, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('x', classValues), LinearClassificationClass.URGENT_STOP.name)
        classValues = [(1, 1), (1, 0), (0, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('X', classValues), LinearClassificationClass.REDUCE_A_LOT.name)
        classValues = [(2, 1), (1, 0), (2, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('x', classValues), LinearClassificationClass.SLIGHTLY_REDUCE.name)
        classValues = [(3, 1), (3, 0), (3, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('X', classValues), LinearClassificationClass.KEEP_SPEED.name)
        classValues = [(4, 1), (4, 0), (3, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('x', classValues), LinearClassificationClass.SPEED_UP.name)

    def test_SidesClassificationValues(self):
        classValues = [(0, 1), (1, 0), (0, 0)]
        self.assertEqual(self.classificationObject.selectClassificationClass('y', classValues), SidesClassificationClass.STRONG_SIDESTEP_LEFT.name)
        classValues = [(1, 1), (1, 0), (0, 1)]
        self.assertEqual(self.classificationObject.selectClassificationClass('Y', classValues), SidesClassificationClass.SLIGHTLY_SIDESTEP_LEFT.name)
        classValues = [(2, 2), (1, 0), (2, 2)]
        self.assertEqual(self.classificationObject.selectClassificationClass('y', classValues), SidesClassificationClass.NO_SIDESTEP.name)
        classValues = [(3, 3), (3, 0), (3, 3)]
        self.assertEqual(self.classificationObject.selectClassificationClass('Y', classValues), SidesClassificationClass.SLIGHTLY_SIDESTEP_RIGHT.name)
        classValues = [(4, 4), (4, 0), (3, 4)]
        self.assertEqual(self.classificationObject.selectClassificationClass('y', classValues), SidesClassificationClass.STRONG_SIDESTEP_RIGHT.name)
        
if __name__ == '__main__':
    unittest.main()