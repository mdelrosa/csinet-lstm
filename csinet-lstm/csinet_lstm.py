# CsiNet_LSTM.py

import tensorflow as tf
# try:
from tensorflow.keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM, CuDNNLSTM, ConvLSTM2D
from tensorflow.keras import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.callbacks import TensorBoard, Callback
from tensorflow.keras.utils import plot_model
from tensorflow.keras import initializers 
from tensorflow.keras.optimizers import Adam
# except:
# 	from keras.layers import concatenate, Lambda, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU, LSTM, CuDNNLSTM, ConvLSTM2D
# 	from keras import Input
# 	from keras.models import Model, model_from_json
# 	from keras.callbacks import TensorBoard, Callback
# 	from keras.utils import plot_model
# 	from keras import initializers 
# 	from keras.optimizers import Adam
import scipy.io as sio 
import numpy as np
import math
import time
# from CsiNet import *
from csinet import *
# from unpack_json import *

# tf.reset_default_graph()
# tf.enable_eager_execution()

# image params
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels
# network params
residual_num = 2
# encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

def get_file(envir,encoded_dim,train_date):
        file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_'+train_date
        return "result/model_%s.h5"%file

def make_CsiNet(aux_bool,M_1, img_channels, img_height, img_width, encoded_dim, data_format, lo_bool=False):
        if aux_bool:
            aux = Input((M_1,))
        else:
            aux = None
        # build CsiNet
        out_activation = 'tanh'
        autoencoder, encoded = CsiNet(img_channels, img_height, img_width, encoded_dim, aux=aux, data_format=data_format, out_activation=out_activation) # CSINet with M_1 dimensional latent space
        # autoencoder = Model(inputs=autoencoder.inputs,outputs=autoencoder.outputs[0])
        prediction = autoencoder.outputs[0]
        encoded = autoencoder.outputs[1]
        if (lo_bool):
            model = Model(inputs=autoencoder.inputs,outputs=prediction)
        else:
            model = Model(inputs=autoencoder.inputs,outputs=[encoded,prediction])
        # return [autoencoder, encoded]
        # optimizer = Adam()
        # model.compile(optimizer=optimizer, loss='mse') 
        return model

def CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, envir="indoor", LSTM_depth=3,data_format='channels_first',t1_trainable=False,t2_trainable=True,pre_t1_bool=True,pre_t2_bool=True,aux_bool=True, share_bool=True, pass_through_bool=False, lstm_latent_bool=False, pre_lstm_bool=True, conv_lstm_bool=False, subnetwork_path=".", pretrained_bool=True, LSTM_only_bool=False):
        # base CSINet models
        aux=Input((M_1,))
        if(data_format == "channels_last"):
                x = Input((T, img_height, img_width, img_channels))
        elif(data_format == "channels_first"):
                x = Input((T, img_channels, img_height, img_width))
        else:
                print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint
        if (not LSTM_only_bool):
            CsiNet_hi = make_CsiNet(aux_bool, M_1, img_channels, img_height, img_width, M_1, data_format)
            if (pretrained_bool):
                #     if envir == "indoor": 
                #         config_hi = 'config/indoor0001/v2/angular/csinet_cr512.json'
                #     elif envir == "outdoor":
                #         config_hi = 'config/outdoor300/v2/csinet_cr512.json'
                #     else:
                #         print("Invalid environment variable.")
                #     dim, date, model_dir = unpack_compact_json(config_hi)
                #     network_name = get_network_name(config_hi)
                #     CsiNet_hi = load_weights_into_model(network_name,model_dir,CsiNet_hi)
                weights_file = f"{subnetwork_path}_cr{M_1}.h5"
                CsiNet_hi.load_weights(weights_file)
            CsiNet_hi._name = "CsiNet_hi"
            CsiNet_hi.trainable = t1_trainable
            print("--- High Dimensional (M_1) Latent Space CsiNet ---")
            CsiNet_hi.summary()
            print('CsiNet_hi.inputs: {}'.format(CsiNet_hi.inputs))
            print('CsiNet_hi.outputs: {}'.format(CsiNet_hi.outputs))
            # TO-DO: split large input tensor to use as inputs to 1:T CSINets
            CsiOut = []
            CsiOut_temp = []
            for i in range(T):
                    CsiIn = Lambda( lambda x: x[:,i,:,:,:])(x)
                    print('#{}: CsiIn: {}'.format(i,CsiIn))
                    if i == 0:
                            # use CsiNet_hi for t=1
                            EncodedLayer, OutLayer = CsiNet_hi([aux,CsiIn])
                            print('EncodedLayer: {}'.format(EncodedLayer))
                    else:
                            # choose whether or not to share parameters between low-dimensional timeslots
                            if (i==1 or not share_bool):
                                    CsiNet_lo = make_CsiNet(aux_bool, M_1, img_channels, img_height, img_width, M_2, data_format, lo_bool=True)
                                    if (pretrained_bool):
                                        #     if envir == "indoor": 
                                        #         config_lo = 'config/indoor0001/v2/angular/csinet_cr{}.json'.format(M_2)
                                        #     elif envir == "outdoor":
                                        #         config_lo = 'config/outdoor300/v2/csinet_cr{}.json'.format(M_2)
                                        #     else:
                                        #         print("Invalid environment variable.")
                                        weights_file = f"{subnetwork_path}_cr{M_2}.h5" 
                                        CsiNet_lo.load_weights(weights_file)
                                        #     dim, date, model_dir = unpack_compact_json(config_lo)
                                        #     network_name = get_network_name(config_lo)
                                        #     CsiNet_lo = load_weights_into_model(network_name,model_dir,CsiNet_lo)
                                    CsiNet_lo.trainable = t2_trainable
                                    CsiNet_lo._name = "CsiNet_lo_{}".format(i)
                                    print('CsiNet_lo.inputs: {}'.format(CsiNet_lo.inputs))
                                    print('CsiNet_lo.outputs: {}'.format(CsiNet_lo.outputs))
                                    if i==1:
                                            print("--- Low Dimensional (M_2) Latent Space CsiNet ---")
                                            CsiNet_lo.summary()
                            if aux_bool:
                                    OutLayer = CsiNet_lo([EncodedLayer,CsiIn])
                            else:
                                    # use CsiNet_lo for t in [2:T]
                                    OutLayer = CsiNet_lo(CsiIn)
                    print('#{} - OutLayer: {}'.format(i, OutLayer))
                    if data_format == "channels_last":
                            CsiOut.append(Reshape((1,img_height,img_width,img_channels))(OutLayer)) 
                    if data_format == "channels_first":
                            CsiOut.append(Reshape((1,img_channels,img_height,img_width))(OutLayer)) 
                    # for the moment, we don't handle separate case of loading convLSTM
            LSTM_in = concatenate(CsiOut,axis=1)
        else:
            LSTM_in = x # skip CsiNets
        # lstm_config = 'config/indoor0001/lstm_depth3_opt.json'
        print('--- Non-convolutional recurrent activations ---')
        LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=LSTM_depth,data_format=data_format)
        # comment back in to load in weights
        # if (pretrained_bool):
        #     lstm_config = 'config/outdoor300/lstm_depth3_opt.json'
        #     dim, date, model_dir = unpack_compact_json(lstm_config)
        #     network_name = get_network_name(lstm_config)
        #     LSTM_model = load_weights_into_model(network_name,model_dir,LSTM_model) # option to load weights; try random initialization for the network
        print(LSTM_model.summary())

        print('LSTM_in.shape: {}'.format(LSTM_in.shape))
        LSTM_out = LSTM_model(LSTM_in)

        # compile full model with large 4D tensor as input and LSTM 4D tensor as output
        if LSTM_only_bool:
            full_model = Model(inputs=[LSTM_in], outputs=[LSTM_out])
        else:
            if pass_through_bool:
                full_model = Model(inputs=[aux,x], outputs=[LSTM_in])
            else:
                full_model = Model(inputs=[aux,x], outputs=[LSTM_out])
        full_model.compile(optimizer='adam', loss='mse')
        full_model.summary()
        return full_model

def split_CsiNet(model, CR):
        # split model into encoder and decoder
        layers = []
        for layer in model.layers:
                # print("layer.name: {} - type(layer): {}".format(layer.name, type(layer)))
                # layers.append(layer)
                # if 'dense' in layer.name:
                #     print('Dense layer "{}" has output shape {}'.format(layer.name,layer.output_shape))
                if layer.output_shape == (None,CR):
                    print('Feedback layer "{}"'.format(layer.name))
                    feedback_layer_output = layer.output # take feedback layer as output of decoder
                elif 'dense' in layer.name:
                    enc_input = layer.input # get concatenate layer's dimension to generate new inp for encoder
        dec_input = model.input
        # enc_input = Input((enc_in_dim))
        dec_model = Model(inputs=[dec_input],outputs=[feedback_layer_output])
        enc_model = Model(inputs=[enc_input],outputs=[model.output])  

def stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=3, plot_bool=False, data_format="channels_first",kernel_initializer=initializers.glorot_uniform(seed=100), recurrent_initializer=initializers.orthogonal(seed=100)):
        # assume entire time-series of CSI from 1:T is concatenated
        LSTM_dim = img_channels*img_height*img_width
        if(data_format == "channels_last"):
                orig_shape = (T, img_height, img_width, img_channels)
        elif(data_format == "channels_first"):
                orig_shape = (T, img_channels, img_height, img_width)
        x = Input(shape=orig_shape)
        LSTM_tup = (T,LSTM_dim)
        recurrent_out = Reshape(LSTM_tup)(x)
        for i in range(LSTM_depth):
            # By default, orthogonal/glorot_uniform initializers for recurrent/kernel
                # recurrent_out = LSTM(LSTM_dim, return_sequences=True, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, stateful=False)(recurrent_out)
                recurrent_out = CuDNNLSTM(LSTM_dim, return_sequences=True, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, stateful=False)(recurrent_out) # CuDNNLSTM does not support recurrent activations; switch back to vanilla LSTM in meantime
                # print("Dim of LSTM #{} - {}".format(i+1,recurrent_out.shape))
        out = Reshape(orig_shape)(recurrent_out)
        LSTM_model = Model(inputs=[x], outputs=[out])
        return LSTM_model

def add_common_layers(y):
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y
