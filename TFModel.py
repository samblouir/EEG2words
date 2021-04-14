import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa


class tf_model(tf.keras.Model):
    def __init__(self, channel_indxs=105, ):
        self.optimizer = tfa.optimizers.LAMB()
        # self.compile(self.optimizer)

        # Build the encoder
        self.inputs = tf.keras.Input(shape=(channel_indxs, 28))
        x = self.inputs
        x = tf.keras.layers.Conv1D(filters=28, kernel_size=5, strides=5, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.activations.lisht(x)
        x = tf.keras.layers.Conv1D(filters=28, kernel_size=5, strides=5, padding='valid')(x)
        encoder_output = x
        self.encoder = tf.keras.Model(self.inputs, encoder_output)

        # Build the decoder
        x = encoder_output
        x = tf.keras.layers.Conv1DTranspose(filters=28, kernel_size=5, strides=5, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tfa.activations.lisht(x)
        x = tf.keras.layers.Conv1DTranspose(filters=28, kernel_size=5, strides=5, padding='valid')(x)
        decoder_output = x
        self.decoder = tf.keras.Model(encoder_output, decoder_output)

        super(tf_model, self).__init__(self.inputs, decoder_output)
        depth = 3
        self.convs = [tf.keras.layers.Conv1D(filters=28, kernel_size=5, strides=5, padding='valid') for _ in range(depth)]
        self.deconvs = [tf.keras.layers.Conv1DTranspose(filters=28, kernel_size=5, strides=5, padding='valid') for _ in range(depth)]
        self.activations = [tfa.activations.lisht for _ in range(depth * 2)]
        self.BNs = [tf.keras.layers.BatchNormalization() for _ in range(depth * 2)]

    @tf.function(experimental_compile=True)
    def call(self, inputs):
        # if inputs:
        # x = tf.concat(inputs, axis=0)
        # print("\n\n CALL!!")
        # for each in inputs:
        #     print(each)
        # print(inputs)
        # print(f"\n concat: {x}")
        x = self(inputs)
        return x
        # else:
        #     print(f"\n Something is fishy!, len(inputs): {len(inputs)}")
        #     for each in inputs:
        #         print(each)

    @tf.function(experimental_compile=True)
    def train_step(self, data):
        anchor, positive, negative = data[:2], data[2:4], data[4:]
        # print(f"  anchor: {anchor}")
        # print(f"  positive: {positive}")
        # print(f"  negative: {negative}")
        with tf.GradientTape() as tape:
            a = self.call(anchor)
            p = self.call(positive)
            n = self.call(negative)

            dist_pos = tf.norm(a - p)
            dist_neg = tf.norm(a - n)
            x = dist_pos - dist_neg + 1.0

            # print(f"  dist_pos: {dist_pos}")
            # print(f" dist_neg: {dist_neg}")
            # print(f"  x: {x}")
            x = tf.keras.activations.relu(x)
            loss = tf.reduce_mean(x)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {"loss": loss}


class validation_callback(tf.keras.callbacks.Callback):
    def __init__(self, ):
        super(validation_callback, self).__init__()

        testfile = "data/WD/test_dict.dict"
        # with open(testfile, 'rb') as datafile:
        # testdata = pickle.load(datafile)

    def on_epoch_end(self, epoch, logs=None):
        pass


def loss_func():
    pass
