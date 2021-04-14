


import torch
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# Imports TF
import tensorflow as tf
from tensorflow.python.layers.base import Layer
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

class eeg2vec_tf(tf.keras.models.Model):
    def __init__(self):
        super(eeg2vec_tf, self).__init__()
