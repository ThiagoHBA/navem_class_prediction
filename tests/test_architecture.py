import unittest
import cv2
from common.utils.dataset_architecture_util import DatasetArchitectureUtil

class ArchitectureTest(unittest.TestCase):
    def test_DronetArchitectureValues(self):
        dronetValues = DatasetArchitectureUtil('dronet')
        self.assertEqual(dronetValues.getImageSize(), (200,200))
        self.assertEqual(dronetValues.getImageColorScale(), cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':
    unittest.main()