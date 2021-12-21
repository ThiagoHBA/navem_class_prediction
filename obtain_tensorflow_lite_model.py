from common.utils.file_model_util import FileModel
import tensorflow as tf

def main():
    tensorflowModel = FileModel('./models/vgg16/tensorflow/exp_340_x/', 'model_struct.json','model_weights_99.h5').compileTensorflowModel()
    
    converter = tf.lite.TFLiteConverter.from_keras_model(tensorflowModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    open('./models/vgg16/tensorflow_lite/vgg16_model_x.tflite', 'wb').write(tflite_model)

if __name__ == "__main__":
    main()
