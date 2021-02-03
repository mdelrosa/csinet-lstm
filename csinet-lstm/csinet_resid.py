import tensorflow as tf
from tensorflow.keras.layers import concatenate, Dense, BatchNormalization, Reshape, add, LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import numpy as np

Conv2D = tf.keras.layers.Conv2D

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=200, epochs=600, max_lr=1e-3, min_lr=1e-4):
    super(CosineSchedule, self).__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    self.epochs = tf.cast(epochs, tf.float32)
    self.max_lr = tf.cast(max_lr, tf.float32)
    self.min_lr = tf.cast(min_lr, tf.float32)
    self.diff_lr = max_lr - min_lr

  def warmup_rate(self):
    return self.diff_lr * self.step / self.warmup_steps + self.min_lr

  def cosine_rate(self):
    return self.diff_lr * ((tf.math.cos(self.step-self.warmup_steps*np.pi / self.epochs - self.warmup_steps) + 1) / 2) + self.min_lr
            
  def get_config(self):
    config = {
#       'epochs': self.epochs,
#       'warmup_steps': self.warmup_steps,
#       'max_lr': self.max_lr,
#       'min_lr': self.min_lr,
#       'diff_lr': self.diff_lr,
#       'rate': self.rate
    }
    return config


  def __call__(self, step):
    self.step = step
    rate = tf.cond(step < self.warmup_steps, self.warmup_rate, self.cosine_rate)
    self.rate = rate
    return rate

def CsiNet(img_channels, img_height, img_width, encoded_dim, encoder_in=None, residual_num=2, aux=None, encoded_in=None, data_format="channels_last",name=None,out_activation='sigmoid'):
        img_total = img_channels*img_height*encoded_dim 

        # Bulid the autoencoder model of CsiNet
        def residual_network(x, residual_num, encoded_dim, aux):
                def add_common_layers(y,enc_bool=False):
                        if enc_bool:
                                y = BatchNormalization(name='CR2_batch_normalization')(y)
                                y = LeakyReLU(name='CR2_leaky_re_lu')(y)
                        else:
                                y = BatchNormalization()(y)
                                y = LeakyReLU()(y)
                        return y
                def residual_block_decoded(y):
                        shortcut = y

                        # according to CsiNet-LSTM paper Fig. 1, residual network has 2-filter conv2D layers before other conv2D layers
                        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
                        y = add_common_layers(y)

                        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
                        y = add_common_layers(y)
                        
                        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
                        y = add_common_layers(y)
                        
                        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format=data_format)(y)
                        y = BatchNormalization()(y)

                        y = add([shortcut, y])
                        y = LeakyReLU()(y)

                        return y
                
                # if encoder_in:
                x = Conv2D(2, (3, 3), padding='same', data_format=data_format, name='CR2_conv2d')(x)
                x = add_common_layers(x,enc_bool=True)
                
                x = Reshape((img_total,), name='CR2_reshape')(x)
                encoded = Dense(encoded_dim, activation='linear', name='CR2_dense')(x)

                print("Aux check: {}".format(aux))
                tens_type = type(x)
                if type(aux) == tens_type:
                        x = Dense(img_total, activation='linear')(concatenate([aux,encoded]))
                else:
                        x = Dense(img_total, activation='linear')(encoded)
                # reshape based on data_format
                if(data_format == "channels_first"):
                        x = Reshape((img_channels, img_height, img_width,))(x)
                elif(data_format == "channels_last"):
                        x = Reshape((img_height, img_width, img_channels,))(x)

                for i in range(residual_num):
                        x = residual_block_decoded(x)
                x = Conv2D(2, (3, 3), activation=out_activation, padding='same', data_format=data_format)(x)

                return [x, encoded]

        if(data_format == "channels_last"):
                image_tensor = Input((img_height, img_width, img_channels))
        elif(data_format == "channels_first"):
                image_tensor = Input((img_channels, img_height, img_width))
        else:
                print("Unexpected tensor_shape param in CsiNet input.")
                # raise Exception
        [network_output, encoded] = residual_network(image_tensor, residual_num, encoded_dim, aux)
        print('network_output: {} - encoded: {} -  aux: {}'.format(network_output, encoded, aux))
        tens_type = type(image_tensor)
        print('image_tensor.dtype: {}'.format(tens_type))
        print('type(aux): {}'.format(type(aux)))
        if type(aux) == tens_type:
                autoencoder = Model(inputs=[aux,image_tensor], outputs=[network_output,encoded])
        else:
                autoencoder = Model(inputs=[image_tensor], outputs=[network_output, encoded])
        if encoder_in:
                autoencoder.load_weights(by_name=True)
        return [autoencoder, encoded]