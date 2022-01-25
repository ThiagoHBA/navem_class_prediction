from common.utils.file_model_util import FileModel
import tensorflow as tf

def main():
    tensorflowModel = FileModel('./models/dronet/tensorflow_evaluate/exp_335_y/', 'model_struct.json','model_weights_299.h5').compileTensorflowModel()
    
    converter = tf.lite.TFLiteConverter.from_keras_model(tensorflowModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    open('./models/dronet/tensorflow_lite/dronet_evaluate_model_y.tflite', 'wb').write(tflite_model)

if __name__ == "__main__":
    main()
