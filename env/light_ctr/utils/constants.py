import tensorflow as tf

# 定义配置文本到 TensorFlow 数据类型的映射
TF_DTYPE_MAPPING = {
    "float16": tf.float16,
    "float32": tf.float32,
    "float64": tf.float64,
    "int8": tf.int8,
    "int16": tf.int16,
    "int32": tf.int32,
    "int64": tf.int64,
    "uint8": tf.uint8,
    "uint16": tf.uint16,
    "uint32": tf.uint32,
    "uint64": tf.uint64,
    "bool": tf.bool,
    "string": tf.string,
    "complex64": tf.complex64,
    "complex128": tf.complex128,
    "qint8": tf.qint8,
    "qint16": tf.qint16,
    "qint32": tf.qint32,
    "quint8": tf.quint8,
    "quint16": tf.quint16,
}
