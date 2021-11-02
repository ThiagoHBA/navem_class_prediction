from common.utils.classification_util import ClassificationUtil
from common.utils.files_util import Files
from io import StringIO
import os
import unittest
import sys

class FilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filesObject = Files("UnitTestFile")
        return super().setUp()

    def test_Logs(self):
        self.filesObject.initializeLog()
        self.assertEqual(os.path.isdir('logs'), True)
        self.filesObject.writeLog(ClassificationUtil.classificationToMap(0, 1 ,2))
        self.assertEqual(os.path.exists('logs/UnitTestFile_log.json'), True)

    def test_Experiment(self):
        self.filesObject.createExperimentFile()
        sys.stdin(StringIO("UnitTestExperiment"))
        

if __name__ == '__main__':
    unittest.main()