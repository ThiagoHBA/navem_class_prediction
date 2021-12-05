from common.utils.file_model_util import FileModel
import tensorflow as tf

def main():
    tensorflowModel = FileModel('./models/resnet/tensorflow/exp_366_x/', 'model_struct.json','model_weights_99.h5').compileTensorflowModel()
    
    converter = tf.lite.TFLiteConverter.from_keras_model(tensorflowModel)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()

    open('./models/resnet/tensorflow_lite/resnet_model_x.tflite', 'wb').write(tflite_model)

if __name__ == "__main__":
    main()
