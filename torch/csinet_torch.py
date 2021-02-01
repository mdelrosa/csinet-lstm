import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

class Encoder(torch.nn.Module):
    """ encoder for CsiNet """
    def __init__(self, n_chan, H, W, latent_dim):
        super(Encoder, self).__init__()
        self.img_total = H*W
        self.n_chan = n_chan
        self.latent_dim = latent_dim
        self.enc_conv1 = nn.Conv2d(2, 8, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(8)
        self.enc_conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(16)
        self.enc_conv3 = nn.Conv2d(16, 2, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(2)
        self.enc_dense = nn.Linear(H*W*n_chan, latent_dim)

        # TODO: try different activation functions here (i.e., swish)
        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope

    def forward(self, x):
        x = self.activ(self.bn_1(self.enc_conv1(x)))
        x = self.activ(self.bn_2(self.enc_conv2(x)))
        x = self.activ(self.bn_3(self.enc_conv3(x)))
        x = torch.reshape(x, (x.size(0), -1,)) # TODO: verify -- does this return num samples in both channels?
        x = self.enc_dense(x)
        return x

class Decoder(torch.nn.Module):
    """ decoder for CsiNet """
    def __init__(self, n_chan, H, W, latent_dim, aux_dim=512):
        super(Decoder, self).__init__()
        self.H = H
        self.W = W
        self.img_total = H*W
        self.n_chan = n_chan
        self.dec_dense = nn.Linear(latent_dim+aux_dim, self.img_total*self.n_chan)
        self.dec_conv1 = nn.Conv2d(2, 128, 1)
        self.bn_1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, 64, 1)
        self.bn_2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn_3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_4 = nn.BatchNorm2d(32)
        self.dec_conv5 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn_5 = nn.BatchNorm2d(16)
        self.dec_conv6 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn_6 = nn.BatchNorm2d(16)
        self.dec_conv7 = nn.Conv2d(16, 2, 3, padding=1)

        self.activ = nn.LeakyReLU(0.1) # TODO: make sure slope matches TF slope
        self.out_activ = nn.Tanh()

    def forward(self, x):
        """ x = aux, input """
        aux, H_in = x
        x = self.dec_dense(torch.cat((aux, H_in), 1))
        x = torch.reshape(x, (x.size(0), self.n_chan, self.H, self.W))
        x = self.activ(self.bn_1(self.dec_conv1(x)))
        x = self.activ(self.bn_2(self.dec_conv2(x)))
        x = self.activ(self.bn_3(self.dec_conv3(x)))
        x = self.activ(self.bn_4(self.dec_conv4(x)))
        x = self.activ(self.bn_5(self.dec_conv5(x)))
        x = self.activ(self.bn_6(self.dec_conv6(x)))
        x = self.out_activ(self.dec_conv7(x))
        return x

class CsiNet(nn.Module):
    """ CsiNet for csi estimation """
    def __init__(self, encoder, decoder, latent_dim, device=None):
        super(CsiNet, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.device = device
        self.training = True

    def forward(self, x):
        """forward call for CsiNet"""
        aux, H_in = x
        h_enc = self.encoder(H_in)
        return self.decoder((aux, h_enc))

    def latent_loss(self, z_mean, z_stddev):
        """ if we want to do semi-supervised learning, then we could define the loss here """
        pass

if __name__ == "__main__":
    import argparse
    import pickle
    import copy
    import sys
    sys.path.append("/home/mdelrosa/git/brat")
    from utils.NMSE_performance import get_NMSE, denorm_H3, denorm_sphH4, denorm_H4, renorm_H4
    from utils.data_tools import dataset_pipeline_col, subsample_batches
    from utils.parsing import str2bool
    from utils.timing import Timer
    from utils.unpack_json import get_keys_from_json
    from utils.trainer import fit, score, save_predictions, save_checkpoint_history, load_checkpoint_history

    # set up timers
    timers = {
             "fit_timer": Timer("Fit"),              
             "predict_timer": Timer("Predict"),
             "score_timer": Timer("Score")
             }

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_flag", type=int, default=0, help="flag for toggling debugging mode")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="number for torch device (cuda:gpu_num)")
    parser.add_argument("-b", "--n_batch", type=int, default=20, help="number of batches to fit on (ignored during debug mode)")
    parser.add_argument("-l", "--dir", type=str, default=None, help="subdirectory for saving model, checkpoint, history")
    parser.add_argument("-e", "--env", type=str, default="indoor", help="environment (either indoor or outdoor)")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="number of epochs to train for")
    parser.add_argument("-tr", "--train_argv", type=str2bool, default=True, help="flag for toggling training")
    parser.add_argument("-t", "--n_truncate", type=int, default=32, help="value to truncate to along delay axis.")
    parser.add_argument("-ts", "--timeslot", type=int, default=0, help="timeslot which we are training (0-indexed).")
    parser.add_argument("-r", "--rate", type=int, default=512, help="number of elements in latent code (i.e., encoding rate)")
    parser.add_argument("-dt", "--data_type", type=str, default="norm_H4", help="type of dataset to train on (norm_H4, norm_sphH4)")
    parser.add_argument("-a", "--aux_bool", type=str2bool, default=True, help="bool for building CsiNet with auxiliary input")
    parser.add_argument("-m", "--aux_size", type=int, default=512, help="integer for auxiliary input's latent rate")
    opt = parser.parse_args()

    device = torch.device(f'cuda:{opt.gpu_num}' if torch.cuda.is_available() else 'cpu')
    print(f"--- Device is {device} ---")

    if opt.env == "outdoor":
        json_config = '../config/csinet_outdoor_cost2100_pow.json'
    elif opt.env == "indoor":
        json_config = '../config/csinet_indoor_cost2100_pow.json' 

#     elif opt.data_type == "norm_sphH4":
#         # json_config = "../config/csinet-pro-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-outdoor300-sph.json"
#         json_config = "../config/csinet-pro-quadriga-indoor0001-sph.json" if opt.env == "indoor" else "../config/csinet-pro-quadriga-outdoor300-sph.json"

#     model_dir, norm_range, minmax_file, dataset_spec, diff_spec, batch_num, lrs, batch_sizes, network_name, T, data_format = get_keys_from_json(json_config, keys=['model_dir','norm_range','minmax_file','dataset_spec', 'diff_spec', 'batch_num', 'lrs', 'batch_sizes', 'network_name', 'T', 'df'])
    dataset_spec, minmax_file, img_channels, data_format, norm_range, T, network_name, model_dir, n_delay, lr, batch_size, diff_spec = get_keys_from_json(json_config, keys=["dataset_spec", "minmax_file", "img_channels", "df", "norm_range", "T", "network_name", "model_dir", "n_delay", "learning_rate", "batch_size", "diff_spec"])
#     lr = lrs[0]
#     batch_size = batch_sizes[0]
    aux_bool_list = get_keys_from_json(json_config, keys=["aux_bool"], is_bool=True)
    aux_bool = aux_bool_list[0] # dumb, but get_keys_from_json returns list

    input_dim = (2,32,n_delay)
    epochs = 10 if opt.debug_flag else opt.epochs

    batch_num = 1 if opt.debug_flag else opt.n_batch # dataset batches
    M_1 = None # legacy holdover from CsiNet-LSTM

    # load all data splits

    data_train, data_val = dataset_pipeline_col(opt.debug_flag, opt.aux_bool, dataset_spec, opt.aux_size, T = T, img_channels = input_dim[0], img_height = input_dim[1], img_width = input_dim[2], data_format = data_format, train_argv = opt.train_argv)

    aux_val, x_val = data_val
    x_val = renorm_H4(x_val,minmax_file)
    print('-> post-renorm: x_val range is from {} to {}'.format(np.min(x_val),np.max(x_val)))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(aux_val).float().to(device), torch.from_numpy(x_val).to(device))

    if opt.train_argv:
        aux_train, x_train = data_train
        x_train = renorm_H4(x_train,minmax_file)
        print('-> post-renorm: x_train range is from {} to {}'.format(np.min(x_train),np.max(x_train)))
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(aux_train).float().to(device), torch.from_numpy(x_train).to(device))

    model_dir += "/" + opt.env
    if opt.dir != None:
        model_dir += "/" + opt.dir

    cr_list = [512, 256, 128, 64, 32] if opt.rate == 0 else [opt.rate]# rates for different compression ratios
    for cr in cr_list:

        valid_ldr = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
        if opt.train_argv:
            train_ldr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        encoder = Encoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        decoder = Decoder(input_dim[0], input_dim[1], opt.n_truncate, cr)
        csinet_pro = CsiNet(encoder, decoder, cr, device=device).to(device)

        pickle_dir = f"{model_dir}/cr{cr}/t1"
        print(f"--- pickle_dir is {pickle_dir} ---")

        if opt.train_argv:
            print(f"--- Fitting on training set ({x_train.shape[0]} batches) ---")
            model, checkpoint, history, optimizer, timers = fit(csinet_pro,
                                                                train_ldr,
                                                                valid_ldr,
                                                                batch_num,
                                                                epochs=epochs,
                                                                timers=timers,
                                                                json_config=json_config,
                                                                debug_flag=opt.debug_flag,
                                                                pickle_dir=pickle_dir)
        else:                                                            
            print(f"--- Loading model, checkpoint, history, optimizer from {model_dir} ---")
            model, checkpoint, history, optimizer = load_checkpoint_history(pickle_dir, csinet_pro, network_name=network_name)

        checkpoint = score(csinet_pro,
                        valid_ldr,
                        x_val,
                        batch_num,
                        checkpoint,
                        history,
                        optimizer,
                        timers=timers,
                        json_config=json_config,
                        debug_flag=opt.debug_flag,
                        str_mod=f"CsiNet CR={cr} - {opt.env} -",
                        diff_spec=diff_spec
                        )

        if not opt.debug_flag:                

            # del train_ldr
            # train_ldr = torch.utils.data.DataLoader((torch.from_numpy(aux_train).to(device), torch.from_numpy(x_train).to(device)), batch_size=batch_size)
            # train_ldr = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
            # save_predictions(csinet_pro, train_ldr, data_train, optimizer, timers, json_config=json_config, dir=pickle_dir, split="train")
            # save_predictions(csinet_pro, valid_ldr, data_test, optimizer, timers, json_config=json_config, dir=pickle_dir, split="valid")
            save_checkpoint_history(checkpoint, history, optimizer, dir=pickle_dir, network_name=network_name)
            # del train_ldr, valid_ldr