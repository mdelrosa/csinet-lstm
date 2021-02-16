from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, Lambda, Reshape, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import regularizers, initializers, activations
import os

class CsiNet_quant():
    """
    Wrapper that breaks out encoder/decoder. Allows for insertion of quantization layer.
    """
    def __init__(self, encoded_dim, dynamic_range=32, img_channels=2, img_height=32, img_width=32, encoder_in=None, residual_num=2, aux_shape=512, encoded_in=None, data_format="channels_first",name=None,out_activation='tanh', code_min = -1, code_max = 1, side_min = -1, side_max = 1):
        self.dynamic_range = dynamic_range
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width
        self.img_total = img_channels*img_height*img_width
        self.encoded_dim = encoded_dim
        self.encoder_in = encoder_in
        self.residual_num = residual_num
        self.aux_shape = aux_shape
        self.aux = tf.keras.Input(aux_shape)
        self.encoded_in = encoded_in
        self.data_format = data_format
        self.name = name
        self.out_activation = out_activation
        self.use_bias = True # not sure on this one
        self.code_min = code_min
        self.code_max = code_max 
        self.side_min = side_min
        self.side_max = side_max

    def add_common_layers(self, y, enc_bool=False):
        if enc_bool:
            y = BatchNormalization(name='CR2_batch_normalization')(y)
            y = LeakyReLU(name='CR2_leaky_re_lu')(y)
        else:
            y = BatchNormalization(axis=1)(y)
            y = LeakyReLU()(y)
        return y

    def residual_block_decoder(self, y):

        y = Conv2D(128, kernel_size=(1, 1), padding='same',data_format=self.data_format,name="deconv1", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(64, kernel_size=(1, 1), padding='same',data_format=self.data_format,name="deconv2", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(32, kernel_size=(3, 3), padding='same',data_format=self.data_format,name="deconv3", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(32, kernel_size=(3, 3), padding='same',data_format=self.data_format,name="deconv4", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(16, kernel_size=(3, 3), padding='same',data_format=self.data_format,name="deconv5", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(16, kernel_size=(3, 3), padding='same',data_format=self.data_format,name="deconv6", use_bias=self.use_bias)(y)
        y = self.add_common_layers(y)
        y = Conv2D(2, (3, 3), activation=self.out_activation, padding='same',data_format=self.data_format,name="predict", use_bias=self.use_bias)(y)
        return y

    # Bulid the autoencoder model of CsiNet
    def encoder_network(self, x):
                
        x = Conv2D(8, (3, 3), padding='same', data_format=self.data_format, name='CR2_conv2d_1', use_bias=self.use_bias)(x)
        x = self.add_common_layers(x)
        x = Conv2D(16, (3, 3), padding='same', data_format=self.data_format, name='CR2_conv2d_2', use_bias=self.use_bias)(x)
        x = self.add_common_layers(x)
        x = Conv2D(2, (3, 3), padding='same', data_format=self.data_format, name='CR2_conv2d_3', use_bias=self.use_bias)(x)
        x = self.add_common_layers(x)
        
        x = Reshape((self.img_total,), name='CR2_reshape')(x)
        print(f"--- In encoder_network: self.encoded_dim={self.encoded_dim} ---")
        encoded = Dense(self.encoded_dim, activation='linear', name='CR2_dense')(x)

        return encoded

    def encoder_quantizer(self, encoded, x_min, x_max):
        return None

    def decoder_network(self, encoded):
        x = Dense(self.img_total, activation='linear')(encoded)
        if(self.data_format == "channels_first"):
            x = Reshape((self.img_channels, self.img_height, self.img_width,))(x)
        elif(self.data_format == "channels_last"):
            x = Reshape((self.img_height, self.img_width, self.img_channels,))(x)

        x = self.residual_block_decoder(x)

        return x

    # --- quantizer layers ---
    # --> these require extrema from pre-trained network <--

    # helper function -- mu-law encoding
    def encoder_mu(self, x, encoded_dim, img_total = 2048, dynamic_range_i=32, mu = 255., side_bool = False):
        code_max = self.side_max if side_bool else self.code_max
        code_min = self.side_min if side_bool else self.code_min
        code_abs_max = np.amax(np.absolute([code_max, code_min]))
        encoded_quan_norm = Lambda(lambda x: x / code_abs_max)(x)
        encoded_quan_norm_u = Lambda(lambda x: K.sign(x) * K.log(1 + mu * K.abs(x)) / np.log(1 + mu))(encoded_quan_norm)
        encoded = Lambda(lambda x: x*(dynamic_range_i-1))(encoded_quan_norm_u)
        return encoded

    def decoder_mu(self, encoded, side_encoded = None, dynamic_range_i=32, mu = 255.):
        x = Lambda(lambda x: x / (dynamic_range_i-1))(encoded)
        x = Lambda(lambda x: K.sign(x) * (1 / mu) *(K.exp(K.abs(x)*np.log(1 + mu)) - 1))(x)
        code_abs_max = np.amax(np.absolute([self.code_max, self.code_min]))
        x = Lambda(lambda x: x * code_abs_max)(x)

        if type(side_encoded) != type(None):
            # decode t1 codeword if present
            y = Lambda(lambda x: x / (dynamic_range_i-1))(side_encoded)
            y = Lambda(lambda x: K.sign(x) * (1 / mu) *(K.exp(K.abs(x)*np.log(1 + mu)) - 1))(y)
            code_abs_max = np.amax(np.absolute([self.side_max, self.side_min]))
            y = Lambda(lambda x: x * code_abs_max)(y)
            x = concatenate([x,y])

        return x

    def quantizer_o(self, encoded, encoded_dim, dynamic_range_i=32, kernel_regularizer = None, name_list=["bequantization", "dequantization", "quantization"]):
        hada1 = Hadamard(encoded_dim=encoded_dim, kernel_regularizer=kernel_regularizer, name=name_list[0])  # kernel_initializer=keras.initializers.Constant(dynamic_range),
        # hada1 = Hadamard(encoded_dim=encoded_dim, kernel_regularizer=quan_reg(quan_lam))

        hada2 = Hadamard_div(encoded_dim=encoded_dim, name=name_list[1])

        self.create_inversed_weights(hada1, hada2, (None,) + (encoded_dim,))
        encoded_bp = hada1(encoded)
        encoded_quan = Roundings(dynamic_range_i=dynamic_range_i, name=name_list[2])(encoded_bp)
        encoded_dequan = hada2(encoded_quan)

        return encoded_dequan

    def create_inversed_weights(self, hada1, hada2, input_shape):
        with K.name_scope(hada1.name):
            hada1.build(input_shape)
        with K.name_scope(hada2.name):
            hada2.build(input_shape)
        # hada2.kernel = K.variable(1)/hada1.kernel
        hada2.kernel = hada1.kernel
        hada2._trainable_weights = []
        hada2._trainable_weights.append(hada2.kernel)
    
    def build_full_network(self):
        if(self.data_format == "channels_last"):
            image_tensor = Input((self.img_height, self.img_width, self.img_channels))
        elif(self.data_format == "channels_first"):
            image_tensor = Input((self.img_channels, self.img_height, self.img_width))
        else:
            print("Unexpected tensor_shape param in CsiNet input.")
            # raise Exception
        encoded = self.encoder_network(image_tensor)

        # mu-law companding (no trainable scalars)
        companded = self.mu_law_pipeline(encoded, side_bool=False, dynamic_range_i=self.dynamic_range)
        side_companded = self.mu_law_pipeline(self.aux, side_bool=True, dynamic_range_i=self.dynamic_range, compander_name="side_companded")
        x = concatenate([side_companded,companded])

        # mu-law companding (trainable layers)
        # x = self.encoder_mu(encoded, self.encoded_dim, side_bool = False)
        # mu_aux = self.encoder_mu(self.aux, self.aux_shape, side_bool = True)
        # x = self.quantizer_o(x, self.encoded_dim, dynamic_range_i=self.dynamic_range)
        # y = self.quantizer_o(mu_aux, self.aux_shape, dynamic_range_i=self.dynamic_range, name_list=["side_bequantization", "side_dequantization", "side_quantization"])
        # x = self.decoder_mu(x, side_encoded=y) # decoder handles quantization of encoded tensor and aux encoded tensor

        # x = concatenate([self.aux, encoded])
        network_output = self.decoder_network(x)

        tens_type = type(image_tensor)
        if type(self.aux) == tens_type:
            autoencoder = Model(inputs=[self.aux,image_tensor], outputs=[companded, encoded, network_output])
        else:
            autoencoder = Model(inputs=[image_tensor], outputs=[companded, encoded, network_output])
        # if self.encoder_in:
        #     autoencoder.load_weights(by_name=True)
        self.autoencoder = autoencoder
        self.encoded = encoded

    def mu_law_pipeline(self, x, img_total = 2048, dynamic_range_i=32, mu = 255., side_bool = False, compander_name="companded"):
        code_max = self.side_max if side_bool else self.code_max
        code_min = self.side_min if side_bool else self.code_min
        code_abs_max = np.amax(np.absolute([code_max, code_min]))
        encoded_quan_norm = Lambda(lambda x: x / code_abs_max)(x)
        encoded_quan_norm_u = Lambda(lambda x: K.sign(x) * K.log(1 + mu * K.abs(x)) / np.log(1 + mu))(encoded_quan_norm)
        encoded = Lambda(lambda x: tf.math.round(x*(dynamic_range_i-1)))(encoded_quan_norm_u)
        print(encoded)
        x = Lambda(lambda x: x / (dynamic_range_i-1))(encoded)
        x = Lambda(lambda x: K.sign(x) * (1 / mu) *(K.exp(K.abs(x)*np.log(1 + mu)) - 1))(x)
        x = Lambda(lambda x: x * code_abs_max, name = compander_name)(x)
        return x

    def load_template_weights(self, template_model):
        """
        load weights from pretrained unquantized model into model with quantizer
        see here: https://stackoverflow.com/a/43702449
        """

        # ideally something like this would work...
        # temp_name = 'temp.h5'
        # template_model.save_weights(temp_name)
        # self.autoencoder.load_weights(temp_name, by_name=True)

        # but this might have to be done :facepalm:
        # magic numbers - models are the same up to enc_limit
        enc_limit = 12
        # dec_offset = 11 # this worked for single mu law quantizer/dequantizer
        dec_offset = 18
        temp_idx = 0
        quant_idx = 0
        template_weights = template_model.get_weights()
        print("--- Loading {} weights into quantized model ---".format(len(template_weights)))
        p_bar = tqdm(total = len(template_weights))
        while (temp_idx < len(template_weights)):
            quant_layer_weights = self.autoencoder.layers[quant_idx].weights
            quant_num = len(quant_layer_weights)
            layer_weights = []
            if (quant_num == 0):
                quant_idx += 1
            else:
                for i in range(quant_num):
                    layer_weights.append(template_weights[temp_idx+i])
                self.autoencoder.layers[quant_idx].set_weights(layer_weights)
                temp_idx += quant_num
                p_bar.update(quant_num)
                quant_idx += 1
                if quant_idx == enc_limit:
                    quant_idx += dec_offset-1 # skip ahead by num layers in quant layers
                layer_weights = []
        p_bar.close()
        print("--- Finished loading ---")

    def load_template_weights_v2(self, template_model):
        """
        load weights from pretrained unquantized model into model with quantizer
        see here: https://stackoverflow.com/a/43702449
        """

        enc_limit = 20
        dec_offset = 0
        template_weights = template_model.get_weights()
        quant_slice = self.autoencoder.get_weights()[enc_limit:enc_limit+dec_offset]
        template_weights_new = template_weights[:enc_limit] + quant_slice + template_weights[enc_limit:]
        self.autoencoder.set_weights(np.array(template_weights_new))
        for i, temp_weights in tqdm(enumerate(template_weights_new)):
            for quant, temp in zip(self.autoencoder.get_weights()[i], temp_weights):
                foo = quant == temp
                assert(foo.all())

    def load_template_weights_v3(self, template_model):
        """
        load weights from pretrained unquantized model into model with quantizer
        see here: https://stackoverflow.com/a/43702449
        """
        enc_limit = 12
        # dec_offset = 11 # this worked for single mu law quantizer/dequantizer
        dec_offset = 18
        temp_idx = 0
        quant_idx = 0
        template_weights = template_model.get_weights()
        print("--- Loading {} weights into quantized model ---".format(len(template_weights)))
        p_bar = tqdm(total = len(template_weights))
        while (temp_idx < len(template_weights)):
            quant_layer_weights = self.autoencoder.layers[quant_idx].weights
            quant_num = len(quant_layer_weights)
            layer_weights = []
            if (quant_num == 0):
                quant_idx += 1
            else:
                for i in range(quant_num):
                    layer_weights.append(template_weights[temp_idx+i])
                self.autoencoder.layers[quant_idx].set_weights(layer_weights)
                temp_idx += quant_num
                p_bar.update(quant_num)
                quant_idx += 1
                if quant_idx == enc_limit:
                    quant_idx += dec_offset-1 # skip ahead by num layers in quant layers
                layer_weights = []
        p_bar.close()
        print("--- Finished loading ---")

class Roundings(Layer):

    def __init__(self, dynamic_range_i=16, **kwargs):
        super(Roundings, self).__init__(**kwargs)
        self.supports_masking = True
        self.dynamic_range = dynamic_range_i

    def sum_sigmoid(self, x):
        dynamic_range = self.dynamic_range
        r = 100.
        i = tf.constant(-dynamic_range + 0.5, dtype=tf.float32)
        j = tf.zeros(shape=tf.shape(x), dtype=tf.float32)
        [_, approx_round] = tf.while_loop(lambda i, j: tf.less(i, dynamic_range - 0.5),
                                          lambda i, j: [tf.add(i, 1), tf.add(j, K.sigmoid(r * (x - i)))], [i, j])
        return tf.add(approx_round, -dynamic_range)

    def call(self, inputs, **kwargs):
        return self.sum_sigmoid(inputs)

    def get_config(self):
        config = {'dynamic_range': int(self.dynamic_range)}
        base_config = super(Roundings, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Hadamard_div(Layer):

    def __init__(self, kernel_initializer='ones', \
                 kernel_regularizer=None, activation='linear', encoded_dim = 32, trainable = True, **kwargs):
        super(Hadamard_div, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.encoded_dim = encoded_dim

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.encoded_dim,),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(Hadamard_div, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape, self.kernel.shape)
        outputs = x / self.kernel
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        # print(input_shape)
        return input_shape

class Hadamard(Layer):

    def __init__(self, kernel_initializer='ones', \
                 kernel_regularizer=None, encoded_dim = 32, trainable = True, **kwargs):
        super(Hadamard, self).__init__(**kwargs)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.encoded_dim = encoded_dim

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.encoded_dim,),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        super(Hadamard, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print(x.shape, self.kernel.shape)
        outputs = x * self.kernel
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape