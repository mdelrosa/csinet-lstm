
if __name__ == "__main__":
    import argparse
    import os
    import copy
    import sys
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import calc_NMSE, get_NMSE, denorm_H3, renorm_H4, denorm_H4, denorm_sphH4
    from utils.data_tools import dataset_pipeline_col, subsample_batches
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
    parser.add_argument("-b", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-t", "--train_argv", type=str2bool, default=True, help="flag for toggling training")
    parser.add_argument("-g", "--n_gpu", type=int, default=1, help="index of gpu for training")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-de", "--depth", type=int, default=3, help="depth of lstm")
    parser.add_argument("-p", "--pretrained_bool", type=str2bool, default=True, help="bool for using pretrained CsiNet for each timeslot")
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=False, help="bool for loading weights into CsiNet-LSTM network")
    parser.add_argument("-a", "--aux_bool", type=str2bool, default=True, help="bool for building CsiNet with auxiliary input")
    parser.add_argument("-m", "--aux_size", type=int, default=512, help="integer for auxiliary input's latent rate")
    parser.add_argument("-sr", "--stride", type=int, default=1, help="space between timeslots for each step (default 1); controls feedback interval")
    parser.add_argument("-q", "--quantization_bits", type=int, default=5, help="quantization bits per value")
    parser.add_argument("-i", "--t1_bits", type=int, default=8, help="quantization bits for first timeslot")
    parser.add_argument("-ql", "--quan_lam", type=float, default=1e-9, help="quantization regularizer")
    opt = parser.parse_args()

    quan_lam=opt.quan_lam
    dynamic_range_i = 2**(opt.quantization_bits - 1) 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(opt.n_gpu);  # Do other imports now...
    print("debug_flag: {} -- train_argv: {}".format(opt.debug_flag, opt.train_argv))

    if opt.env == "indoor":
        # if opt.rate != 512:
        # json_config = 'config/indoor0001/T10/replication/angular/csinet_lstm_v2_CR{}.json'.format(CR_arg) # 500 epochs
        json_config = '../config/csinet_lstm_indoor_cost2100.json' 
        temp_config = lstm_config = None
        # else:
        #     # json_config = 'config/indoor0001/T10/replication/angular/csinet_lstm_v2_CR{}_best.json'.format(CR_arg) # 500 epochs
        #     json_config = '../config/csinet_lstm_outdoor_cost2100.json' # 0 epochs 
        #     temp_config, lstm_config = get_keys_from_json(json_config, keys=['temp_config', 'lstm_config'])
    elif opt.env == "outdoor":
        json_config = '../config/csinet_lstm_outdoor_cost2100.json' 
        # json_config = 'config/outdoor300/T10/csinet_lstm_v2_CR{}.json'.format(CR_arg) # Depth 3 
        temp_config = lstm_config = None

    M_1, data_format, network_name, subnetwork_name, model_dir, norm_range, minmax_file, share_bool, T, dataset_spec, diff_spec, batch_num, lr, batch_size, subsample_prop, thresh_idx_path = get_keys_from_json(json_config, keys=['M_1', 'df', 'network_name', 'subnetwork_name', 'model_dir', 'norm_range', 'minmax_file', 'share_bool', 'T', 'dataset_spec', 'diff_spec', 'batch_num', 'lr', 'batch_size', 'subsample_prop', 'thresh_idx_path'])
    aux_bool, quant_bool, LSTM_only_bool, pass_through_bool, t1_train, t2_train, lstm_latent_bool = get_keys_from_json(json_config, keys=['aux_bool', 'quant_bool', 'LSTM_only_bool', 'pass_through_bool', 't1_train', 't2_train', 'lstm_latent_bool'],is_bool=True) # import these as booleans rather than int, str

    import scipy.io as sio 
    import numpy as np
    import math
    import time
    import sys
    # import os
    try:
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, EarlyStopping
    except:
        import keras
        from keras.optimizers import Adam
        from keras.callbacks import TensorBoard, Callback, ModelCheckpoint
    from tensorflow.core.protobuf import rewriter_config_pb2
    from csinet_lstm import *
    from csinet_quant import CsiNet_quant
    # from QuantizeData import quantize, get_minmax

    def reset_keras():
        sess = tf.keras.backend.get_session()
        tf.keras.backend.clear_session()
        sess.close()
        # limit gpu resource allocation
        try:
    	    config = tf.compat.v1.ConfigProto()
        except:
    	    config = tf.ConfigProto()
        # config.gpu_options.visible_device_list = '1'
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
    
        # disable arithmetic optimizer
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off
   
        try: 
    	    session = tf.compat.v1.Session(config=config)
    	    tf.compat.v1.keras.backend.set_session(session)
        except:
    	    session = tf.Session(config=config)
    	    keras.backend.set_session(session)
        # tf.global_variables_initializer()

    reset_keras()

    # fit params
    # image params
    img_height = 32
    img_width = 32
    img_channels = 2 
    img_total = img_height*img_width*img_channels

    # Data loading
    batch_num = 1 if opt.debug_flag else batch_num # we'll use batch_num-1 for training and 1 for validation
    epochs = 1 if opt.debug_flag else opt.epochs

    # T_dummy = 1 # while debugging, we'll repeat the first timeslot
    pow_diff, data_train, data_val = dataset_pipeline_col(opt.debug_flag, opt.aux_bool, dataset_spec, diff_spec, opt.aux_size, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, data_format = data_format, train_argv = opt.train_argv, subsample_prop=subsample_prop, thresh_idx_path=thresh_idx_path, stride=opt.stride)

    print(f"pow_diff.shape: {pow_diff.shape}")

    # loading directly from unnormalized data; normalize data
    aux_val, x_val = data_val
    x_val = renorm_H4(x_val,minmax_file)
    data_val = aux_val, x_val 
    print(f"-> aux_val.shape: {aux_val.shape} - x_val.shape: {x_val.shape}")
    print('-> post-renorm: x_val range is from {} to {}'.format(np.min(x_val),np.max(x_val)))

    aux_train, x_train = data_train
    x_train = renorm_H4(x_train,minmax_file)
    data_train = [aux_train, x_train]
    print(f"-> aux_train.shape: {aux_train.shape} - x_train.shape: {x_train.shape}")
    print('-> post-renorm: x_train range is from {} to {}'.format(np.min(x_train),np.max(x_train)))

    outpath_base = f"{model_dir}/{opt.env}"
    if opt.dir != None:
        outpath_base += "/" + opt.dir 

    rates = [512, 256, 128, 64, 32]
    for rate in rates:
        outfile_base = f"{outpath_base}/cr{rate}/{network_name}"
        subnetwork_spec = [outpath_base, subnetwork_name]

        # load model
        if temp_config == None:
            # outfile = f"{model_dir}/model_CsiNet_LSTM_{envir}_dim{M_2}_{dates[0]}"
            outfile_base = f"{outpath_base}/cr{rate}/{network_name}"
            CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, rate, envir=opt.env, LSTM_depth=opt.depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool,LSTM_only_bool=LSTM_only_bool, subnetwork_spec=subnetwork_spec)
            # tf.keras.models.model_from_json("{}.json".format(outfile))
            # template_model.load_weights("{}.h5".format(outfile))
            template_model = tf.keras.models.load_model("{}.h5".format(outfile_base))
            n_layers = len(template_model.layers)
            if n_layers == 2:
                # handle the combined model
                print("Combined model -- load each layer by name.")
                CsiNet_T10_model = template_model[0]
                LSTM_model = template_model[1]
                print("CsiNet_T10_model.summary()")
                CsiNet_T10_model.summary()
                print("LSTM_model.summary()")
                LSTM_model.summary()
            else:
                CsiNet_LSTM_model.load_weights("{}.h5".format(outfile_base))
        else:
            LSTM_model, CsiNet_LSTM_model = combine_model(temp_config, lstm_config, json_config, data_train, data_val, data_test, debug_flag=opt.debug_flag)

        # preloaded performance
        if (opt.debug_flag):
            print ("--- Pre-loaded network performance is... ---")
            x_hat = CsiNet_LSTM_model.predict(data_val)

            print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
            print("x_hat.dtype: {}".format(x_hat.dtype)) # sanity check on output datatype
            if norm_range == "norm_H3":
                x_hat_denorm = denorm_H3(x_hat,minmax_file)
                x_val_denorm = denorm_H3(data_val[1],minmax_file)
            if norm_range == "norm_H4":
                x_hat_denorm = denorm_H4(x_hat,minmax_file)
                x_val_denorm = denorm_H4(data_val[1],minmax_file)
            print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
            print('-> x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))
            calc_NMSE(x_hat_denorm,x_val_denorm,T=T)

        CsiNet_models = []
        CsiNet_names = ["CsiNet_hi"] + [f"CsiNet_lo_{i}" for i in range(1,10)]

        # autoencoder_models = []
        CsiOut = []

        aux = tf.keras.Input((M_1))
        if(data_format == "channels_last"):
            x = tf.keras.Input((T, img_height, img_width, img_channels))
        elif(data_format == "channels_first"):
            x = tf.keras.Input((T, img_channels, img_height, img_width))
        else:
            print("Unexpected data_format param in CsiNet input.") # raise an exception eventually. For now, print a complaint

        print("aux (full network aux input): {}".format(aux))
        print("x (full network input): {}".format(x))

        side_max = 1
        side_min = -1

        # iterate through CsiNet models, insert quantizer in between encoder and decoder
        for i, model_name in enumerate(CsiNet_names):
            CsiIn = Lambda( lambda x: x[:,i,:,:,:])(x)
            # CsiIn = CsiNet_LSTM_model.layers[lambda_idx[i]
            x_all = np.squeeze(np.vstack((data_train[1][:,i,:,:,:], data_val[1][:,i,:,:,:]))).astype('float32')
            aux_all = np.vstack((data_train[0], data_val[0])).astype('float32') if i == 0 else aux_t1
            data_all = [aux_all, x_all]
            model_name_adj = model_name if temp_config == None else f"{model_name}_new"
            CsiNet_model = CsiNet_LSTM_model.get_layer(f"{model_name_adj}")
    
            # use CsiNet_aux_quant to load in template model weights
            encoded_dim = M_1 if i == 0 else rate
            encoded_in = aux if i == 0 else t1_encoded
            if i == 0:
                enc_hat, x_hat = CsiNet_model.predict(data_all)
                dynamic_range = 2 ** (opt.t1_bits - 1)
            else:
                CsiNet_enc_model = Model(inputs=CsiNet_model.input,
                                        outputs=CsiNet_model.get_layer('CR2_dense').output)
                enc_hat = CsiNet_enc_model.predict(data_all)
                dynamic_range = dynamic_range_i
            print(f"#{i} x_hat.shape: {x_hat.shape}, enc_hat.shape: {enc_hat.shape}")
            enc_min = np.min(enc_hat)
            enc_max = np.max(enc_hat)
            print(f"enc_min: {enc_min}, enc_max: {enc_max}")
            print(f"side_min: {side_min}, side_max: {side_max}")
            CsiNet_quant_model = CsiNet_quant(encoded_dim,
                                            code_min = enc_min,
                                            code_max = enc_max,
                                            side_min = side_min,
                                            side_max = side_max,
                                            dynamic_range = dynamic_range)
            CsiNet_quant_model.build_full_network()
            CsiNet_quant_model.load_template_weights_v2(CsiNet_model)
            print(f"#{i} encoded_in.shape: {encoded_in.shape}")
            companded, encoded, autoencoder_out = CsiNet_quant_model.autoencoder([encoded_in,CsiIn])
            print(f"#{i} encoded.shape: {encoded.shape}")
            if i == 0:
                t1_encoded = encoded
                aux_t1 = enc_hat # actual values to be fed in as side info at all t_i
                side_min = enc_min
                side_max = enc_max

            if data_format == "channels_last":
                    CsiOut.append(Reshape((1,img_height,img_width,img_channels))(autoencoder_out)) 
            if data_format == "channels_first":
                    CsiOut.append(Reshape((1,img_channels,img_height,img_width))(autoencoder_out)) 

        LSTM_in = concatenate(CsiOut,axis=1)

        # get pretrained LSTM model
        # TODO: Get rid of magic string -- rename LSTM model appropriately

        # option 1
        # LSTM_model = CsiNet_LSTM_model.get_layer("model_20")

        # option 2
        # LSTM_name_new = "model_22" if temp_config == None else "model_20"
        # LSTM_template_model = CsiNet_LSTM_model.get_layer(LSTM_name_new)

        # option 3
        LSTM_template_model = CsiNet_LSTM_model.layers[-1] # last layer is LSTM model

        LSTM_model = stacked_LSTM(img_channels, img_height, img_width, T, lstm_latent_bool, LSTM_depth=opt.depth,data_format=data_format)
        LSTM_model.set_weights(LSTM_template_model.get_weights())
        # for i, temp_weights in tqdm(enumerate(LSTM_template_model.get_weights())):
        dummy_i = 0
        for target, temp in zip(LSTM_model.get_weights(), LSTM_template_model.get_weights()):
            print(f"Assertion #{dummy_i}")
            foo = target == temp
            assert(foo.all())
            dummy_i += 1
        LSTM_out = LSTM_model(LSTM_in) 

        # LSTM_out = LSTM_template_model(LSTM_in)

        # CsiNet_LSTM_code_quant_model = Model(inputs = [aux,x], outputs = [LSTM_in])
        CsiNet_LSTM_code_quant_model = Model(inputs = [aux,x], outputs = [LSTM_out])
        CsiNet_LSTM_code_quant_model.summary()
        CsiNet_LSTM_code_quant_model.compile(optimizer = "adam", loss = "mse")

        # evaluate model
        print (f"--- {opt.env} {rate} with quantized codewords ({opt.quantization_bits} bits) is... ---")
        x_hat = CsiNet_LSTM_code_quant_model.predict(data_val)

        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_val_denorm = denorm_H3(data_val[1],minmax_file)
        if norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_val_denorm = denorm_H4(data_val[1],minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))
        calc_NMSE(x_hat_denorm,x_val_denorm,T=T,pow_diff=pow_diff)