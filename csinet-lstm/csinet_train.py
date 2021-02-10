
if __name__ == "__main__":
    import argparse
    import pickle
    import os
    import copy
    import sys
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import calc_NMSE, get_NMSE, denorm_H3, renorm_H4, denorm_H4, denorm_sphH4
    from utils.data_tools import dataset_pipeline_col, dataset_pipeline_complex, subsample_batches, load_pow_diff
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-tr", "--train_argv", type=str2bool, default=True, help="flag for toggling training")
    parser.add_argument("-g", "--n_gpu", type=int, default=1, help="index of gpu for training")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-lo", "--load_bool", type=str2bool, default=False, help="bool for loading weights into CsiNet")
    parser.add_argument("-a", "--aux_bool", type=str2bool, default=True, help="bool for building CsiNet with auxiliary input")
    parser.add_argument("-m", "--aux_size", type=int, default=512, help="integer for auxiliary input's latent rate")
    opt = parser.parse_args()

    if opt.env == "outdoor":
        json_config = '../config/csinet_outdoor_cost2100_pow.json'
        # json_config = '../config/csinet_outdoor_cost2100_pow_subsample.json' 
    elif opt.env == "indoor":
        json_config = '../config/csinet_indoor_cost2100_pow.json' 
        # json_config = '../config/csinet_indoor_cost2100_pow_subsample.json' 
        # json_config = '../config/csinet_indoor_cost2100_old.json' # requires dataset_pipeline_complex

    model_dir, norm_range, minmax_file, dataset_spec, diff_spec, batch_num, lr, batch_size, network_name, T, data_format, subsample_prop, thresh_idx_path = get_keys_from_json(json_config, keys=['model_dir','norm_range','minmax_file','dataset_spec', 'diff_spec', 'batch_num', 'learning_rate', 'batch_size', 'network_name', 'T', 'df', 'subsample_prop', 'thresh_idx_path'])
    # lr = lrs[0]
    # batch_size = batch_sizes[0]

    # encoded_dims, dates, result_dir, aux_bool, opt.rate, data_format, epochs, t1_train, t2_train, gpu_num, lstm_latent_bool, conv_lstm_bool = unpack_json(json_config)

    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(opt.n_gpu);  # Do other imports now...

    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, EarlyStopping
    import scipy.io as sio 
    import numpy as np
    import math
    import time
    import sys

    from csinet import *
    from tensorflow.core.protobuf import rewriter_config_pb2
    # from NMSE_performance import calc_NMSE, denorm_H3, denorm_H4

    # norm_range = get_norm_range(json_config)

    def reset_keras():
        sess = tf.keras.backend.get_session()
        tf.keras.backend.clear_session()
        sess.close()
        # limit gpu resource allocation
        config = tf.compat.v1.ConfigProto()
        # config.gpu_options.visible_device_list = '1'
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        # physical_devices = tf.config.list_physical_devices('GPU')
        # try:
        #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # except:
        #     # Invalid device or cannot modify virtual devices once initialized.
        #     print("Cannot access 'set_memory_growth' - skipping.")
        #     pass 

        # disable arithmetic optimizer
        off = rewriter_config_pb2.RewriterConfig.OFF
        config.graph_options.rewrite_options.arithmetic_optimization = off
    
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

    reset_keras()

    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1.0
    # session = tf.Session(config=config)

    # image params
    img_height = 32
    img_width = 32
    img_channels = 2 
    img_total = img_height*img_width*img_channels
    # network params
    residual_num = 2
    # encoded_dim = 512  #compress rate=1/4->dim.=512, compress rate=1/16->dim.=128, compress rate=1/32->dim.=64, compress rate=1/64->dim.=32

    epochs = 1 if opt.debug_flag else opt.epochs
    batch_num = 1 if opt.debug_flag else batch_num 

    pow_diff, data_train, data_val = dataset_pipeline_col(opt.debug_flag, opt.aux_bool, dataset_spec, diff_spec, opt.aux_size, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, data_format = data_format, train_argv = opt.train_argv, subsample_prop=subsample_prop, thresh_idx_path=thresh_idx_path)
    # data_train, data_val = dataset_pipeline_complex(opt.debug_flag, opt.aux_bool, dataset_spec, diff_spec, opt.aux_size, T = T, img_channels = img_channels, img_height = img_height, img_width = img_width, data_format = data_format, train_argv = opt.train_argv, subsample_prop=subsample_prop)

    # print(f"-> pre reshape: pow_diff.shape: {pow_diff.shape}")
    # pow_diff = np.reshape(np.real(pow_diff), (pow_diff.shape[0]*pow_diff.shape[1], -1))
    # print(f"-> post reshape: pow_diff.shape: {pow_diff.shape}")

    # SHUFFLE_BUFFER_SIZE = batch_size*5

    # loading directly from unnormalized data; normalize data
    aux_val, x_val = data_val
    print('-> pre-renorm: x_val range is from {} to {}'.format(np.min(x_val),np.max(x_val)))
    x_val = renorm_H4(x_val,minmax_file)
    data_val = aux_val, x_val 
    print(f"-> pre reshape: x_val.shape: {x_val.shape}")
    # x_val = np.reshape(x_val, (x_val.shape[0]*x_val.shape[1], x_val.shape[2], x_val.shape[3], x_val.shape[4]))
    # print(f"-> post reshape: x_val.shape: {x_val.shape}")
    # aux_val = np.tile(aux_val, (T,1))
    print(f"-> aux_val.shape: {aux_val.shape} - x_val.shape: {x_val.shape}")
    print('-> post-renorm: x_val range is from {} to {}'.format(np.min(x_val),np.max(x_val)))
    # val_gen = tf.data.Dataset.from_tensor_slices(({"input_1": aux_val, "input_2": x_val}, x_val)).batch(batch_size).repeat()

    if opt.train_argv:
        aux_train, x_train = data_train
        x_train = renorm_H4(x_train,minmax_file)
        data_train = aux_train, x_train 
        # print(f"pre reshape: x_train.shape: {x_train.shape}")
        # x_train = np.reshape(x_train, (x_train.shape[0]*x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4]))
        # print(f"post reshape: x_train.shape: {x_train.shape}")
        # aux_train = np.tile(aux_train, (T,1))
        print(f"-> aux_train.shape: {aux_train.shape} - x_train.shape: {x_train.shape}")
        print('-> post-renorm: x_train range is from {} to {}'.format(np.min(x_train),np.max(x_train)))
        # train_gen = tf.data.Dataset.from_tensor_slices(({"input_1": aux_train, "input_2": x_train}, x_train)).shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size).repeat()

    # opt.rates = [512, 128, 64, 32]
    print('Build and train CsiNet for rate={}'.format(opt.rate))
    # reset_keras()
    optimizer = Adam(learning_rate=lr)
    if opt.aux_bool:
        aux = Input((opt.aux_size,))
    else:
        aux = None

    # build CsiNet
    outpath_base = f"{model_dir}/{opt.env}"
    if opt.dir != None:
        outpath_base += "/" + opt.dir 
    outfile_base = f"{outpath_base}/cr{opt.rate}/{network_name}"
    # file = 'CsiNet_'+(envir)+'_dim'+str(opt.opt.rate)+'_{}'.format(date)

    out_activation = 'tanh'
    autoencoder, encoded = CsiNet(img_channels, img_height, img_width, opt.rate, aux=aux, data_format=data_format, out_activation=out_activation) # CSINet with opt.rate dimensional latent space
    autoencoder = Model(inputs=autoencoder.inputs,outputs=autoencoder.outputs[0])

    if opt.load_bool:
        # outfile = "{}/model_{}.h5".format(model_dir,file)
        autoencoder.load_weights(f"{outfile_base}.h5")
        print ("--- Pre-loaded network performance is... ---")
        x_hat = autoencoder.predict(data_val)

        print("For Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(lr,batch_size,norm_range))
        if norm_range == "norm_H3":
            x_hat_denorm = denorm_H3(x_hat,minmax_file)
            x_val_denorm = denorm_H3(x_val,minmax_file)
        if norm_range == "norm_H4":
            x_hat_denorm = denorm_H4(x_hat,minmax_file)
            x_val_denorm = denorm_H4(x_val,minmax_file)
        print('-> x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
        print('-> x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))
        calc_NMSE(x_hat_denorm,x_val_denorm,T=T)
    else:
        model_json = autoencoder.to_json()
        # outfile = "{}/model_{}.json".format(result_dir,file)
        with open(f"{outfile_base}.json", "w") as json_file:
                json_file.write(model_json)

    autoencoder.compile(optimizer=optimizer, loss='mse')
    print(autoencoder.summary())

    if opt.train_argv:
        class LossHistory(Callback):
                def on_train_begin(self, logs={}):
                        self.losses_train = []
                        self.losses_val = []

                def on_batch_end(self, batch, logs={}):
                        self.losses_train.append(logs.get('loss'))
                            
                def on_epoch_end(self, epoch, logs={}):
                        self.losses_val.append(logs.get('val_loss'))

        history = LossHistory()

        # early stopping callback
        es = EarlyStopping(monitor='val_loss',mode='min',patience=20,verbose=1)

        # path = f'{outfile_base}_tensorboard'

        # save+serialize model to JSON
        # model_json = autoencoder.to_json()
        # outfile = "{}/model_{}.json".format(result_dir,file)
        # with open(outfile, "w") as json_file:
        #         json_file.write(model_json)
        # serialize weights to HDF5
        # outfile = "{}/model_{}.h5".format(result_dir,file)
        # autoencoder.save_weights(outfile)

        outfile = f"{outfile_base}.h5"
        checkpoint = ModelCheckpoint(outfile, monitor="val_loss",verbose=1,save_best_only=True,mode="min")

        steps_per_epoch = x_train.shape[0] // batch_size
        val_steps = x_val.shape[0] // batch_size

        autoencoder.fit(
                        # train_gen,
                        data_train,
                        x_train,
                        epochs=epochs,
                        # steps_per_epoch=steps_per_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(data_val, x_val),
                        # validation_data=val_gen,
                        # validation_steps=val_steps,
                        callbacks=[history, checkpoint]
                        )
                                # TensorBoard(log_dir = path)])

    # filename = f'{model_dir}/{opt.env}/{opt.dir}/{network_name}_trainloss.csv'
    # loss_history = np.array(history.losses_train)
    # np.savetxt(filename, loss_history, delimiter=",")
            
    # filename = f'{model_dir}/{opt.env}/{opt.dir}/{network_name}_valloss.csv'
    # loss_history = np.array(history.losses_val)
    # np.savetxt(filename, loss_history, delimiter=",")

    #Testing data
    weights_file = f"{outfile_base}.h5"
    print(f"--- Loading weights from {weights_file} ---")
    autoencoder.load_weights(weights_file)
    autoencoder.trainable = False
    # tStart = time.time()
    if opt.aux_bool == 1:
        x_hat = autoencoder.predict([aux_val, x_val])
    else:
        x_hat = autoencoder.predict(x_val)
    # tEnd = time.time()
    # print ("It cost %f sec" % ((tEnd - tStart)/x_val.shape[0]))
    print(64*'=')
    print("For CR2={} // Adam with lr={:1.1e} // batch_size={} // norm_range={}".format(opt.rate,lr,batch_size,norm_range))
    print('-> pre-denorm: x_hat range is from {} to {}'.format(np.min(x_hat),np.max(x_hat)))
    print('-> pre-denorm: x_val range is from {} to {} '.format(np.min(x_val),np.max(x_val)))
    if norm_range == "norm_H3":
        x_hat_denorm = denorm_H3(x_hat,minmax_file)
        x_val_denorm = denorm_H3(x_val,minmax_file)
    if norm_range == "norm_H4":
        x_hat_denorm = denorm_H4(x_hat,minmax_file)
        x_val_denorm = denorm_H4(x_val,minmax_file)
    print('-> post-denorm: x_hat range is from {} to {}'.format(np.min(x_hat_denorm),np.max(x_hat_denorm)))
    print('-> post-denorm: x_val range is from {} to {} '.format(np.min(x_val_denorm),np.max(x_val_denorm)))

    # new method (borrowed from PyTorch impl, trace-based)
    x_hat_denorm = x_hat_denorm[:,0,:,:] + 1j*x_hat_denorm[:,1,:,:]
    x_val_denorm = x_val_denorm[:,0,:,:] + 1j*x_val_denorm[:,1,:,:]
    x_shape = x_val_denorm.shape
    mse, nmse = get_NMSE(x_hat_denorm, x_val_denorm, return_mse=True, n_ang=x_shape[1], n_del=x_shape[2]) 
    results = {}
    print(f"-> Truncated NMSE = {nmse:5.3f} | MSE = {mse:.4E}")
    results["best_nmse"] = nmse
    results["best_mse"] = mse
    if len(diff_spec) != 0: 
        # pow_diff = load_pow_diff(diff_spec)
        mse, nmse = get_NMSE(x_hat_denorm, x_val_denorm, return_mse=True, n_ang=x_shape[1], n_del=x_shape[2], pow_diff_timeslot=pow_diff) 
        print(f"-> Full NMSE = {nmse:5.3f} | MSE = {mse:.4E}")
        results["best_nmse_full"] = nmse
        results["best_mse_full"] = mse

    # original method (tensorflow, CsiNet-LSTM, sum of squared errors)
    # if len(diff_spec) != 0: 
    #     pow_diff = load_pow_diff(diff_spec)
    # results = calc_NMSE(x_hat_denorm, x_val_denorm, T=1, diff_test=pow_diff)

    print(64*'=')

    # dump nmse results to pickle file
    if opt.train_argv:
        with open(f"{outfile_base}_results.pkl", "wb") as f:
            pickle.dump(results, f) 
            f.close()
