import os

class Files:
    def findFile(fileName, path):
        for root, dirs, files in os.walk(path):
            if fileName in files:
                return os.path.join(root, fileName)
            
    def jsonFile(json_model_path):
        with open(json_model_path, 'r') as json_file:
            loaded_model_json = json_file.read()

        return loaded_model_json
