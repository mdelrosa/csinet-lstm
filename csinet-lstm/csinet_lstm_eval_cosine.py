# CsiNet_LSTM_eval_cosine.py
# script for evaluating cosine similarity of CsiNet-LSTM

if __name__ == "__main__":
    import argparse
    import os
    import copy
    import sys
    import pickle
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import calc_NMSE, get_NMSE, denorm_H3, renorm_H4, denorm_H4, denorm_sphH4
    from utils.cosine_sim_performance import cosine_similarity, cosine_similarity_mat
    from utils.data_tools import dataset_pipeline_full, subsample_batches, split_complex
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
    parser.add_argument("-v", "--viz_batch", type=int, default=-1, help="index of element to save for visualization")
    parser.add_argument("-nc", "--n_carriers", type=int, default=128, help="num carriers to test cosine similarity against")
    opt = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(opt.n_gpu);  # Do other imports now...
    print("debug_flag: {} -- train_argv: {}".format(opt.debug_flag, opt.train_argv))

    if opt.env == "indoor":
        json_config = '../config/csinet_lstm_indoor_cost2100_full.json' # 0 epochs 
    elif opt.env == "outdoor":
        json_config = '../config/csinet_lstm_outdoor_cost2100_full.json' # 0 epochs 
    # quant_config = "../config/quant/10bits.json

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

    pow_diff, data_train, data_val = dataset_pipeline_full(batch_num, opt.debug_flag, aux_bool, dataset_spec, diff_spec, M_1, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, train_argv = True, mode = "truncate")

    print(f"--- Load full CSI matrices truncated to {opt.n_carriers} carriers ---")
    pow_diff, data_train_full, data_val_full = dataset_pipeline_full(batch_num, opt.debug_flag, aux_bool, dataset_spec, diff_spec, M_1, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, train_argv = True, mode = "full", n_truncate=opt.n_carriers)
    # make freq-spatial domain from angular-spatial doman
    print(f"--- Convert angular-spatial to freq-spatial {opt.n_carriers} carriers ---")
    x_val_full_freq = np.fft.fft(data_val_full.view("complex"), axis=2) # delay-spatial -> freq-spatial

    # loading directly from unnormalized data; normalize data
    aux_val, x_val = data_val
    x_val = split_complex(x_val.view("complex"),T=T)
    print(f"--- x_val min={np.min(x_val)}, max={np.max(x_val)} ---")
    x_val = renorm_H4(x_val,minmax_file)
    data_val = aux_val, x_val 
    print('-> post-renorm: x_val range is from {} to {}'.format(np.min(x_val),np.max(x_val)))

    aux_train, x_train = data_train
    if opt.train_argv:
        x_train = split_complex(x_train.view("complex"),T=T)
        x_train = renorm_H4(x_train,minmax_file)
        print(f"--- x_train min={np.min(x_train)}, max={np.max(x_train)} ---")
        data_train = [aux_train, x_train]
        print('-> post-renorm: x_train range is from {} to {}'.format(np.min(x_train),np.max(x_train)))

    CR_list = [128]
    # CR_list = [512, 256, 128, 64, 32]
    for M_2 in CR_list:
        # M_1, M_2 = opt.aux_size, opt.rate 
        M_1 = opt.aux_size
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
        outfile_base = f"{outpath_base}/cr{M_2}/{network_name}"
        subnetwork_spec = [outpath_base, subnetwork_name]

        # if opt.load_bool:
        #     CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, envir=opt.env, LSTM_depth=opt.depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, LSTM_only_bool=LSTM_only_bool, subnetwork_spec=subnetwork_spec, pretrained_bool=opt.pretrained_bool)
        #     CsiNet_LSTM_model.load_weights(f"{outfile_base}.h5")
        #     print ("--- Pre-loaded network performance is... ---")
        #     x_hat = CsiNet_LSTM_model.predict(data_val)

        #     print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        #     print("x_hat.dtype: {}".format(x_hat.dtype)) # sanity check on output datatype
        #     if norm_range == "norm_H3":
        #         x_hat_denorm = denorm_H3(x_hat,minmax_file)
        #         x_val_denorm = denorm_H3(x_val,minmax_file)
        #     if norm_range == "norm_H4":
        #         x_hat_denorm = denorm_H4(x_hat,minmax_file)
        #         x_val_denorm = denorm_H4(x_val,minmax_file)
        #     print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        #     print('-> x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))
        #     calc_NMSE(x_hat_denorm,x_val_denorm,T=T)
        # else:
        CsiNet_LSTM_model = CsiNet_LSTM(img_channels, img_height, img_width, T, M_1, M_2, LSTM_depth=opt.depth, data_format=data_format, t1_trainable=t1_train, t2_trainable=t2_train, share_bool=share_bool, pass_through_bool=pass_through_bool, LSTM_only_bool=LSTM_only_bool, subnetwork_spec=subnetwork_spec, pretrained_bool=opt.pretrained_bool)
        if opt.load_bool:
            print(f"--- Loading weights from {outfile_base}.h5 ---")
            CsiNet_LSTM_model.load_weights(f"{outfile_base}.h5")
        CsiNet_LSTM_model.compile(optimizer=optimizer, loss='mse')

        if (opt.train_argv):
            # save+serialize model to JSON
            model_json = CsiNet_LSTM_model.to_json()
            # outfile = f"{model_dir}/{opt.dir}/{network_name}_{opt.env}.json"
            with open(f"{outfile_base}.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            # outfile = f"{model_dir}/model_{file}.h5"
            checkpoint = ModelCheckpoint(f"{outfile_base}_full.h5", monitor="val_loss",verbose=1,save_best_only=True,mode="min")
            early = EarlyStopping(monitor="val_loss", patience=50,verbose=1)

            history = LossHistory()
             
            CsiNet_LSTM_model.fit(data_train, x_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(data_val, x_val),
                            callbacks=[checkpoint,
                                        # early,
                                        history])
                                        # TensorBoard(log_dir = path),
            
            filename = f'{outfile_base}_trainloss.csv'
            loss_history = np.array(history.losses_train)
            np.savetxt(filename, loss_history, delimiter=",")
            
            filename = f'{outfile_base}_valloss.csv'
            loss_history = np.array(history.losses_val)
            np.savetxt(filename, loss_history, delimiter=",")
        else:
            CsiNet_LSTM_model.load_weights(f"{outfile_base}_full.h5")

        #Testing data
        tStart = time.time()
        x_hat = CsiNet_LSTM_model.predict(data_val)
        tEnd = time.time()
        print ("It cost %f sec per sample (%f samples)" % ((tEnd - tStart)/x_val.shape[0],x_val.shape[0]))
            
        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_val_denorm = denorm_H3(x_val,minmax_file)
        elif norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_val_denorm = denorm_H4(x_val,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))

        #TODO: validate this pow_diff behavior
        pow_val = pow_diff[x_train.shape[0]:,:,:]
        calc_NMSE(x_hat_denorm,x_val_denorm,T=T,pow_diff=pow_val)

        #TODO: change this to load gt freq domain data
        #TODO: change this to append zeros to estimate delay domain data

        x_zeros = np.zeros((x_hat.shape[0], T, 2, opt.n_carriers-img_height, img_width))
        x_hat_denorm = np.concatenate((x_hat_denorm, x_zeros), axis=3)
        x_val_denorm = np.concatenate((x_val_denorm, x_zeros), axis=3)
        x_hat_denorm = x_hat_denorm[:,:,0,:,:] + 1j*x_hat_denorm[:,:,1,:,:]
        x_val_denorm = x_val_denorm[:,:,0,:,:] + 1j*x_val_denorm[:,:,1,:,:]
        x_hat_freq = np.fft.fft(np.fft.fft(x_hat_denorm, axis=2), axis=3)
        x_val_freq = np.fft.fft(np.fft.fft(x_val_denorm, axis=2), axis=3)
        # rho_truncate, rho_all = cosine_similarity(x_hat_freq, x_val_freq, pow_diff_T=pow_val)
        # print(f"--- rho_truncate = {rho_truncate:6.5f}, rho_all = {rho_all:6.5f} ---") 
        rho_truncate = cosine_similarity_mat(x_hat_freq, x_val_freq)
        print(f"--- rho_truncate = {rho_truncate:6.5f} ---") 

        # now, report cosine similarity with gt 
        rho_truncate_full = cosine_similarity_mat(x_hat_freq, x_val_full_freq)
        print(f"--- rho_truncate_full = {rho_truncate_full:6.5f} ---") 
    
        if opt.viz_batch > -1 and not opt.train_argv:
            print(f"=== Saving input/output batch {opt.viz_batch} from validation set ===")
            # save input/output of validation batch for visualization
            viz_dict = {
                        "input": x_val_denorm[opt.viz_batch, :, :, :, :],
                        "output": x_hat_denorm[opt.viz_batch, :, :, :, :]
                    }

            with open(f"{outfile_base}_batch{opt.viz_batch}.pkl", "wb") as f:
                pickle.dump(viz_dict, f)
                f.close()