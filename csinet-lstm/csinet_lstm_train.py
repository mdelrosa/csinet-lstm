# CsiNet_LSTM_train.py

if __name__ == "__main__":
    import argparse
    import os
    import copy
    import sys
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import calc_NMSE, get_NMSE, denorm_H3, denorm_H4, denorm_sphH4
    from utils.data_tools import dataset_pipeline, subsample_batches
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
    parser.add_argument("-b", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-t", "--train_argv", type=int, default=1, help="flag for toggling training")
    parser.add_argument("-g", "--n_gpu", type=int, default=1, help="index of gpu for training")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-de", "--depth", type=int, default=3, help="depth of lstm")
    parser.add_argument("-p", "--pretrained_bool", type=str2bool, default=True, help="bool for using pretrained CsiNet for each timeslot")
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=False, help="bool for loading weights into CsiNet-LSTM network")
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(opt.n_gpu);  # Do other imports now...
    print("debug_flag: {} -- train_argv: {}".format(opt.debug_flag, opt.train_argv))

    json_config = '../config/csinet_lstm_outdoor_cost2100.json' # 0 epochs 
    # quant_config = "../config/quant/10bits.json"

    M_1, data_format, network_name, subnetwork_name, model_dir, norm_range, minmax_file, share_bool, T, dataset_spec, batch_num, lrs, batch_sizes = get_keys_from_json(json_config, keys=['M_1', 'df', 'network_name', 'subnetwork_name', 'model_dir', 'norm_range', 'minmax_file', 'share_bool', 'T', 'dataset_spec', 'batch_num', 'lrs', 'batch_sizes'])
    aux_bool, quant_bool, LSTM_only_bool, pass_through_bool, t1_train, t2_train, lstm_latent_bool = get_keys_from_json(json_config, keys=['aux_bool', 'quant_bool', 'LSTM_only_bool', 'pass_through_bool', 't1_train', 't2_train', 'lstm_latent_bool'],is_bool=True) # import these as booleans rather than int
    lr = lrs[0]
    batch_size = batch_sizes[0]

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

    data_train, data_val, data_test = dataset_pipeline(batch_num, opt.debug_flag, aux_bool, dataset_spec, M_1, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, data_format = data_format, train_argv = opt.train_argv, merge_val_test = True)
    aux_train, x_train = data_train
    aux_val, x_val = data_val
    aux_test, x_test = data_test

    # tf Dataset object
    # SHUFFLE_BUFFER_SIZE = batch_size*5

    # train_gen = tf.data.Dataset.from_tensor_slices((data_train, x_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).repeat()
    # val_gen = tf.data.Dataset.from_tensor_slices((data_val, x_val)).shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).repeat()

    # if (quant_bool):
    #     if (val_min == "range"): # hacky -- gets minmax from our file
    #         print("--- Get quantization range from minmax file ---")
    #         val_min, val_max = get_minmax([x_train,x_val,x_test]) # gets downlink by default
    #         print("val_min: {} -- val_max: {}".format(val_min,val_max))
    #     x_quant_test = quantize(x_test,val_min=val_min,val_max=val_max,bits=bits)
    #     print("NMSE for quantization error...")
    #     if norm_range == "norm_H3":
    #         x_quant_test_denorm = denorm_H3(x_quant_test,minmax_file)
    #         x_test_denorm = denorm_H3(x_test,minmax_file)
    #     if norm_range == "norm_H4":
    #         x_quant_test_denorm = denorm_H4(x_quant_test,minmax_file)
    #         x_test_denorm = denorm_H4(x_test,minmax_file)
    #     print('-> x_quant range is from {} to {}'.format(np.min(x_quant_test_denorm),np.max(x_quant_test_denorm)))
    #     print('-> x_test range is from {} to {} '.format(np.min(x_test_denorm),np.max(x_test_denorm)))
    #     print('test: {}'.format(np.mean(np.sum(x_quant_test_denorm-x_test_denorm))))
    #     calc_NMSE(x_quant_test_denorm,x_test_denorm,T=T)

    # CRs = [128,64,32] # sweep compression ratios for latent space
    # for i in range(len(encoded_dims)):
    #     M_2 = encoded_dims[i]
        # date = dates[i]

    M_2 = opt.rate
    reset_keras()
    try:
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    except:
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
    print('-------------------------------------')
    print("Build CsiNet-LSTM for CR2={}".format(M_2))
    print('-------------------------------------')
    
    # def callbacks
    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []
    
        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))
            
        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('val_loss'))
    
    outpath_base = f"{model_dir}/{opt.env}"
    if opt.dir != None:
        outpath_base += "/" + opt.dir 
    outfile_base = f"{outpath_base}/{network_name}"
    subnetwork_path = f"{outpath_base}/{subnetwork_name}"
    if opt.load_bool:
        # if (LSTM_only_bool):
        #     file = f"{file_base}_{opt.env}_D{opt.depth}"
        # elif (network_name != 'model_weights_test'):
        #     file = file_base+(opt.env)+'_dim'+str(M_2)+"_{}".format(date)
        # else:
        #     file = "weights_test" 
        CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, envir=opt.env, LSTM_depth=opt.depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, LSTM_only_bool=LSTM_only_bool, subnetwork_path=subnetwork_path, pretrained_bool=opt.pretrained_bool)
        # outfile = "{}/model_{}.h5".format(model_dir,file)
        CsiNet_LSTM_model.load_weights(f"{outfile_base}.h5")
        # CsiNet_LSTM_model = tf.keras.models.load_model(outfile)
        print ("--- Pre-loaded network performance is... ---")
        x_hat = CsiNet_LSTM_model.predict(data_test)

        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        print("x_hat.dtype: {}".format(x_hat.dtype)) # sanity check on output datatype
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_test_denorm = denorm_H3(x_test,minmax_file)
        if norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_test_denorm = denorm_H4(x_test,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_test range is from {} to {} '.format(np.min(x_test_denorm),np.max(x_test_denorm)))
        calc_NMSE(x_hat_denorm,x_test_denorm,T=T)
    else:
        CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=opt.depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, LSTM_only_bool=LSTM_only_bool, subnetwork_path=subnetwork_path, pretrained_bool=opt.pretrained_bool)
        CsiNet_LSTM_model.compile(optimizer=optimizer, loss='mse')

    if (opt.train_argv):
        # save+serialize model to JSON
        model_json = CsiNet_LSTM_model.to_json()
        # outfile = f"{model_dir}/{opt.dir}/{network_name}_{opt.env}.json"
        with open(f"{outfile_base}.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        # outfile = f"{model_dir}/model_{file}.h5"
        checkpoint = ModelCheckpoint(f"{outfile_base}.h5", monitor="val_loss",verbose=1,save_best_only=True,mode="min")
        early = EarlyStopping(monitor="val_loss", patience=50,verbose=1)

        history = LossHistory()
             
        CsiNet_LSTM_model.fit(data_train, x_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(data_val, x_val),
                        callbacks=[checkpoint,
                                    early,
                                    history])
                                    # TensorBoard(log_dir = path),
            
        filename = f'{model_dir}/{opt.env}/{opt.dir}/{network_name}_trainloss.csv'
        loss_history = np.array(history.losses_train)
        np.savetxt(filename, loss_history, delimiter=",")
            
        filename = f'{model_dir}/{opt.env}/{opt.dir}/{network_name}_valloss.csv'
        loss_history = np.array(history.losses_val)
        np.savetxt(filename, loss_history, delimiter=",")
            
        #Testing data
        tStart = time.time()
        x_hat = CsiNet_LSTM_model.predict(data_test)
        tEnd = time.time()
        print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/x_test.shape[0],x_test.shape[0]))
            
        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_test_denorm = denorm_H3(x_test,minmax_file)
        elif norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_test_denorm = denorm_H4(x_test,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_test range is from {} to {} '.format(np.min(x_test_denorm),np.max(x_test_denorm)))
        calc_NMSE(x_hat_denorm,x_test_denorm,T=T)
    else:
        print("--- train_argv == False -> Skipping training ---")
