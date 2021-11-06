from common.utils.dataset_architecture_util import DatasetArchitectureUtil


class ConfigurationUtil:
    def __init__(self, limitPredictions = 3, infinity = True, loops = 1, showMetrics = False, datasetArchitecture = DatasetArchitectureUtil('dronet')):
        self.limitPredictions = limitPredictions
        self.infinity = infinity
        self.loops = loops
        self.showMetrics = showMetrics
        self.datasetArchitecture = datasetArchitecture
        self.checkIfUpdateConfigurations();

    @staticmethod
    def configurationUtilToMap(limitPredictions, infinity, loops, showMetrics, datasetArchitecture):
        return {
            "limitPredictions": limitPredictions,
            "infinity": infinity,
            "loops": loops,
            "showMetrics": showMetrics,
            "datasetArchitecture": datasetArchitecture,
        }

    def showConfigurations(self):
        print("\nCurrent values: \n")
        print("limitPredictions: {}".format(self.limitPredictions))
        print("Infinity Classification: {}".format(self.infinity))
        print("Loops quantity: {}".format(self.loops))
        print("Show metrics: {}".format(self.showMetrics))
        print("Dataset architecture: {}".format(self.datasetArchitecture.architecture))
        
    def updateConfigurations(self):
        self.showConfigurations()
        self.limitPredictions = self._updateInteger('limit prediction', self.limitPredictions)
        self.infinity = self._updateBoolean('infinity classification', self.infinity)
        self.loops = self._updateInteger('loops quantity', self.loops)
        self.showMetrics = self._updateBoolean('show metrics', self.showMetrics)
        self.datasetArchitecture = DatasetArchitectureUtil(self._updateString('dataset architecture', self.datasetArchitecture.architecture))

    def checkIfUpdateConfigurations(self):        
        if not str(input("Continue with default configurations? [y/n]: ")).lower() == 'y':
            self.updateConfigurations();

    def _updateInteger(self, title, currentValue):
        if str(input("\nUptade the " + title + "? [y/n]: ")).lower() == 'y':
            newValue = int(input("Enter the new value of " + title + ": "))
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue
        return currentValue

    def _updateBoolean(self, title, currentValue):
        if str(input("\nChange the value of " + title + "? [y/n]: ")).lower() == 'y':
            newValue = str(input("Enter the new value of " + title+ " [y/n]: ")).lower() == 'y'
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue  
        return currentValue   

    def _updateString(self, title, currentValue):
        if str(input("\nChange the value of " + title + "? [y/n]: ")).lower() == 'y':
            newValue = str(input("Enter the new value of " + title + ": "))
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue  
        return currentValue