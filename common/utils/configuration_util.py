from common.utils.dataset_architecture_util import DatasetArchitectureUtil
import os
import json

class ConfigurationUtil:
    def __init__(self):
        self.__readConfigurationFile();
        self.checkIfUpdateConfigurations();

    @staticmethod
    def configurationUtilToMap(limitPredictions, infinity, loops, showMetrics, datasetArchitecture, showPreview):
        return {
            "limitPredictions": limitPredictions,
            "infinity": infinity,
            "loops": loops,
            "showMetrics": showMetrics,
            "datasetArchitecture": datasetArchitecture,
            "showPreview": showPreview
        }
        
    @staticmethod
    def configurationUtilFromMap(map):
        return ConfigurationUtil(map['limitPredictions'], map['infinity'], map['loops'], map['showMetrics'], DatasetArchitectureUtil(map['datasetArchitecture'], map['showPreview']))

    def showConfigurations(self):
        print("\nCurrent values: \n")
        print("limitPredictions: {}".format(self.limitPredictions))
        print("Infinity Classification: {}".format(self.infinity))
        print("Loops quantity: {}".format(self.loops))
        print("Show metrics: {}".format(self.showMetrics))
        print("Dataset architecture: {}".format(self.datasetArchitecture.architecture))
        print("Show preview: {}".format(self.showPreview))
        
    def updateConfigurations(self):
        self.showConfigurations()
        self.limitPredictions = self.__updateInteger('limit prediction', self.limitPredictions)
        self.infinity = self.__updateBoolean('infinity classification', self.infinity)
        self.loops = self.__updateInteger('loops quantity', self.loops)
        self.showMetrics = self.__updateBoolean('show metrics', self.showMetrics)
        self.datasetArchitecture = DatasetArchitectureUtil(self.__updateString('dataset architecture', self.datasetArchitecture.architecture))
        self.showPreview = self.__updateBoolean('show preview', self.showPreview)
        self.checkIfSaveConfigurations()

    def saveConfigurations(self):
        if self.__checkIfConfigurationExist():
            with open("configurations.json", 'w') as configurationFile:
                json.dump(self.configurationUtilToMap(self.limitPredictions, self.infinity, self.loops, self.showMetrics, self.datasetArchitecture.architecture, self.showPreview), configurationFile, indent = 4)
                print("\nSuccessfully save...")
                
    def checkIfUpdateConfigurations(self):        
        if not str(input("Continue with default configurations? [y/n]: ")).lower() == 'y':
            self.updateConfigurations();
    
    def checkIfSaveConfigurations(self):
        if str(input("\nDo you want save this configurations? [y/n]: ")).lower() == 'y':
            self.saveConfigurations();

    def __updateConfigurationValues(self, map):
        self.limitPredictions = map['limitPredictions']
        self.infinity = map['infinity']
        self.loops = map['loops']
        self.showMetrics = map['showMetrics']
        self.datasetArchitecture = DatasetArchitectureUtil(map['datasetArchitecture'])
        self.showPreview = map['showPreview']

    def __updateInteger(self, title, currentValue):
        if str(input("\nUptade the " + title + "? [y/n]: ")).lower() == 'y':
            newValue = int(input("Enter the new value of " + title + ": "))
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue
        return currentValue

    def __updateBoolean(self, title, currentValue):
        if str(input("\nChange the value of " + title + "? [y/n]: ")).lower() == 'y':
            newValue = str(input("Enter the new value of " + title+ " [y/n]: ")).lower() == 'y'
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue  
        return currentValue   

    def __updateString(self, title, currentValue):
        if str(input("\nChange the value of " + title + "? [y/n]: ")).lower() == 'y':
            newValue = str(input("Enter the new value of " + title + ": "))
            if newValue != '':
                print("\nSuccessfully updated...")
                return newValue  
        return currentValue

    def __readConfigurationFile(self):
        if not self.__checkIfConfigurationExist():
            with open("configurations.json", 'w') as configurationFile:
                self.limitPredictions = 3
                self.infinity = True
                self.loops = 1
                self.showMetrics = False
                self.datasetArchitecture = DatasetArchitectureUtil('dronet')
                self.showPreview = False
                json.dump(self.configurationUtilToMap(self.limitPredictions, self.infinity, self.loops, self.showMetrics, self.datasetArchitecture.architecture, self.showPreview), configurationFile, indent = 4)
            return configurationFile
        with open("configurations.json", 'r+') as configurationFile:
            self.__updateConfigurationValues(json.load(configurationFile))

    def __checkIfConfigurationExist(self):
        return os.path.exists('configurations.json')
